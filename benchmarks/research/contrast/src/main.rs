// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `warm-contrast`: runs the two kill tests and writes JSON reports.
//!
//! * `nfcorpus` (Part A): own BM25 top-`k` over `corpus.jsonl`, reranked by
//!   `minmax(BM25) + alpha * minmax(feature)` for each structural feature;
//!   `alpha` (and `lambda` for mean-foil) are tuned on a disjoint set (SciFact
//!   by default) and reported on the untouched NFCorpus queries as nDCG@10.
//! * `synth` (Part B): the deterministic agent-memory corpus; per method the
//!   accept statistic of a query is its top candidate's score (present queries
//!   count as rejected when the top candidate is not the source document).
//!   Reports FAR at TAR 0.80 with a threshold calibrated on the first half of
//!   each query class and evaluated on the second half, ROC AUC, and the
//!   single-thread median latency per query over the 20-candidate set.
//! * `all`: both with defaults.
//!
//! Output paths default to `<crate>/out/part_a.json` and `<crate>/out/part_b.json`
//! regardless of the working directory. Schemas are documented in RUN.md.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use warm_research_contrast::bm25::{self, Bm25Index};
use warm_research_contrast::data;
use warm_research_contrast::eval;
use warm_research_contrast::hv::{self, DIM};
use warm_research_contrast::sketch::{self, DocSketch, PairTable, GENERIC_FOILS};
use warm_research_contrast::synth::{self, SynthConfig, Triple};

const OUT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/out");
const OUT_A: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/out/part_a.json");
const OUT_B: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/out/part_b.json");

/// Fusion weights swept for every structural feature.
const ALPHAS: [f64; 10] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0];
/// Foil weights swept for mean-foil contrast (Part A).
const LAMBDAS: [f64; 6] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
/// Foil weights in quarters swept for mean-foil contrast (Part B, exact integer scorer).
const LAMBDA_QUARTERS: [u32; 6] = [1, 2, 3, 4, 6, 8];
/// Part A pass thresholds.
const PASS_A_NDCG: f64 = 0.312;
const PASS_A_GAIN: f64 = 0.008;
/// Part B pass thresholds.
const PASS_B_AUC_GAIN: f64 = 0.10;
const PASS_B_LATENCY_US: f64 = 100.0;
const TAR_TARGET: f64 = 0.80;

#[derive(Parser)]
#[command(name = "warm-contrast", about = "Counterfactual structural contrast kill tests")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum TuneMode {
    /// Tune alpha/lambda on SciFact, report on every NFCorpus query.
    Scifact,
    /// Tune on the first `--head` NFCorpus queries, report on the rest.
    NfcorpusHead,
}

#[derive(Subcommand)]
enum Cmd {
    /// Part A: NFCorpus BM25 top-k rerank.
    Nfcorpus {
        /// NFCorpus directory (default `~/.cache/warm-bench/nfcorpus`).
        #[arg(long)]
        data: Option<PathBuf>,
        /// Tuning directory (default `~/.cache/warm-bench/scifact`).
        #[arg(long)]
        tune_data: Option<PathBuf>,
        /// Where alpha/lambda are tuned.
        #[arg(long, value_enum, default_value = "scifact")]
        tune: TuneMode,
        /// BM25 candidates per query.
        #[arg(long, default_value_t = 200)]
        candidates: usize,
        /// Tuning queries for `--tune nfcorpus-head`.
        #[arg(long, default_value_t = 100)]
        head: usize,
        /// JSON report path.
        #[arg(long, default_value = OUT_A)]
        out: PathBuf,
    },
    /// Part B: synthetic agent memory.
    Synth {
        /// Generator seed.
        #[arg(long, default_value_t = 0x5EED_C0FF_EE00_0002u64)]
        seed: u64,
        /// Documents (base + twin).
        #[arg(long, default_value_t = 2000)]
        docs: usize,
        /// Queries per class (present / absent).
        #[arg(long, default_value_t = 500)]
        queries: usize,
        /// Candidates per query.
        #[arg(long, default_value_t = 20)]
        candidates: usize,
        /// Timed repeats (median reported).
        #[arg(long, default_value_t = 7)]
        repeats: usize,
        /// Unmeasured warmup passes.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// JSON report path.
        #[arg(long, default_value = OUT_B)]
        out: PathBuf,
        /// Skip writing `synth_corpus.jsonl` / `synth_queries.jsonl` next to the report.
        #[arg(long)]
        no_dump: bool,
    },
    /// Both parts with defaults.
    All,
}

fn home_cache(sub: &str) -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("warm-bench").join(sub)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(value).context("serialising report")?;
    fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Part A
// ---------------------------------------------------------------------------

struct Collection {
    idx: HashMap<String, u32>,
    index: Bm25Index,
    sketches: Vec<DocSketch>,
    pairs: Vec<PairTable>,
    sketch_us_per_doc: f64,
    mean_pairs_per_doc: f64,
}

fn load_collection(dir: &Path) -> Result<Collection> {
    let rows = data::load_rows(&dir.join("corpus.jsonl"))?;
    if rows.is_empty() {
        bail!("{}: empty corpus", dir.display());
    }
    let seeds: Vec<Vec<u64>> = rows.iter().map(|r| bm25::seeds(&r.text)).collect();
    let index = Bm25Index::build(&seeds);
    let t0 = Instant::now();
    let sketches: Vec<DocSketch> = seeds.iter().map(|s| sketch::doc_sketch(s)).collect();
    let sketch_us_per_doc = t0.elapsed().as_secs_f64() * 1e6 / rows.len() as f64;
    let mean_pairs_per_doc =
        sketches.iter().map(|s| s.pairs as f64).sum::<f64>() / sketches.len() as f64;
    let pairs: Vec<PairTable> = seeds.iter().map(|s| PairTable::build(s)).collect();
    let idx = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id.clone(), i as u32))
        .collect();
    eprintln!(
        "{}: {} docs, sketch {:.1} us/doc, {:.0} pairs/doc",
        dir.display(),
        rows.len(),
        sketch_us_per_doc,
        mean_pairs_per_doc
    );
    Ok(Collection {
        idx,
        index,
        sketches,
        pairs,
        sketch_us_per_doc,
        mean_pairs_per_doc,
    })
}

#[derive(Clone)]
struct Cand {
    doc: u32,
    bm25: f64,
    pos: f64,
    foils: [f64; GENERIC_FOILS],
    energy: f64,
    pairs: f64,
}

struct QueryCands {
    cands: Vec<Cand>,
    rels: HashMap<u32, f64>,
    hit: bool,
}

fn build_candidates(
    col: &Collection,
    queries: &[data::Row],
    qrels: &HashMap<String, HashMap<String, f64>>,
    k: usize,
) -> Vec<QueryCands> {
    let mut out = Vec::with_capacity(queries.len());
    let mut phantom = u32::MAX;
    for q in queries {
        let Some(rel) = qrels.get(&q.id) else {
            continue;
        };
        let seeds = bm25::seeds(&q.text);
        let counts = bm25::counts(&seeds);
        let top = col.index.top_k(&counts, k);
        let qs = sketch::query_sketch(&seeds);
        let qp = PairTable::build(&seeds);
        let cands: Vec<Cand> = top
            .iter()
            .map(|&(d, s)| {
                let ds = &col.sketches[d as usize];
                let dots = qs.foil_scores(ds);
                let mut foils = [0.0f64; GENERIC_FOILS];
                for (slot, &v) in foils.iter_mut().zip(dots.iter()) {
                    *slot = f64::from(v);
                }
                Cand {
                    doc: d,
                    bm25: s,
                    pos: f64::from(qs.positive_score(ds)),
                    foils,
                    energy: qs.cleanup_energy(ds),
                    pairs: qp.coverage(&col.pairs[d as usize]),
                }
            })
            .collect();
        let mut rels: HashMap<u32, f64> = HashMap::with_capacity(rel.len());
        for (doc_id, &gain) in rel {
            match col.idx.get(doc_id) {
                Some(&d) => {
                    rels.insert(d, gain);
                }
                None => {
                    // Keeps the ideal DCG honest for qrel docs missing from the corpus.
                    rels.insert(phantom, gain);
                    phantom -= 1;
                }
            }
        }
        let hit = cands
            .iter()
            .any(|c| rels.get(&c.doc).is_some_and(|&g| g > 0.0));
        out.push(QueryCands { cands, rels, hit });
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Variant {
    Bm25,
    Positive,
    PositiveInt8,
    MeanFoil,
    MaxFoil,
    HashedPairs,
}

const VARIANTS: [Variant; 6] = [
    Variant::Bm25,
    Variant::Positive,
    Variant::PositiveInt8,
    Variant::MeanFoil,
    Variant::MaxFoil,
    Variant::HashedPairs,
];

impl Variant {
    fn uses_alpha(self) -> bool {
        self != Variant::Bm25
    }

    fn uses_lambda(self) -> bool {
        self == Variant::MeanFoil
    }
}

fn feature(v: Variant, c: &Cand, lambda: f64) -> f64 {
    match v {
        Variant::Bm25 => 0.0,
        Variant::Positive => c.pos,
        Variant::PositiveInt8 => c.energy,
        Variant::MeanFoil => c.pos - lambda * eval::mean(&c.foils),
        Variant::MaxFoil => c.pos - c.foils.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        Variant::HashedPairs => c.pairs,
    }
}

fn rerank_ndcg(q: &QueryCands, v: Variant, alpha: f64, lambda: f64) -> f64 {
    let n = q.cands.len();
    if n == 0 {
        return 0.0;
    }
    let mut b: Vec<f64> = q.cands.iter().map(|c| c.bm25).collect();
    eval::minmax(&mut b);
    let mut f: Vec<f64> = q.cands.iter().map(|c| feature(v, c, lambda)).collect();
    eval::minmax(&mut f);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let si = b[i] + alpha * f[i];
        let sj = b[j] + alpha * f[j];
        sj.total_cmp(&si)
            .then_with(|| q.cands[j].bm25.total_cmp(&q.cands[i].bm25))
            .then_with(|| q.cands[i].doc.cmp(&q.cands[j].doc))
    });
    let ranked: Vec<u32> = order.iter().take(10).map(|&i| q.cands[i].doc).collect();
    eval::ndcg_at_k(&ranked, &q.rels, 10)
}

fn mean_ndcg(set: &[QueryCands], v: Variant, alpha: f64, lambda: f64) -> f64 {
    if set.is_empty() {
        return 0.0;
    }
    set.iter().map(|q| rerank_ndcg(q, v, alpha, lambda)).sum::<f64>() / set.len() as f64
}

#[derive(Serialize)]
struct GridCell {
    alpha: f64,
    lambda: f64,
    ndcg10: f64,
}

#[derive(Serialize)]
struct VariantReport {
    name: Variant,
    alpha: f64,
    lambda: f64,
    ndcg10_tune: f64,
    ndcg10_eval: f64,
    grid: Vec<GridCell>,
}

fn tune_variant(v: Variant, tune: &[QueryCands]) -> (f64, f64, f64, Vec<GridCell>) {
    let alphas: &[f64] = if v.uses_alpha() { &ALPHAS } else { &[0.0] };
    let lambdas: &[f64] = if v.uses_lambda() { &LAMBDAS } else { &[0.0] };
    let mut best = (0.0f64, 0.0f64, f64::NEG_INFINITY);
    let mut grid = Vec::with_capacity(alphas.len() * lambdas.len());
    for &alpha in alphas {
        for &lambda in lambdas {
            let ndcg = mean_ndcg(tune, v, alpha, lambda);
            grid.push(GridCell {
                alpha,
                lambda,
                ndcg10: ndcg,
            });
            if ndcg > best.2 {
                best = (alpha, lambda, ndcg);
            }
        }
    }
    (best.0, best.1, best.2, grid)
}

#[derive(Serialize)]
struct SetReport {
    name: String,
    dir: String,
    n_queries: usize,
    candidates_k: usize,
    recall_at_k: f64,
    n_docs: usize,
    sketch_build_us_per_doc: f64,
    mean_pairs_per_doc: f64,
}

#[derive(Serialize)]
struct VerdictA {
    pass: bool,
    best_contrast: Variant,
    ndcg10_best_contrast: f64,
    ndcg10_positive_only: f64,
    ndcg10_bm25: f64,
    threshold_abs: f64,
    threshold_gain_over_positive: f64,
}

#[derive(Serialize)]
struct PartAReport {
    part: &'static str,
    tune_mode: String,
    eval_set: SetReport,
    tune_set: SetReport,
    variants: Vec<VariantReport>,
    verdict: VerdictA,
}

fn set_report(name: &str, dir: &Path, set: &[QueryCands], col: &Collection, k: usize) -> SetReport {
    let hits = set.iter().filter(|q| q.hit).count();
    SetReport {
        name: name.to_string(),
        dir: dir.display().to_string(),
        n_queries: set.len(),
        candidates_k: k,
        recall_at_k: if set.is_empty() {
            0.0
        } else {
            hits as f64 / set.len() as f64
        },
        n_docs: col.sketches.len(),
        sketch_build_us_per_doc: col.sketch_us_per_doc,
        mean_pairs_per_doc: col.mean_pairs_per_doc,
    }
}

fn run_part_a(
    data_dir: &Path,
    tune_dir: &Path,
    mode: TuneMode,
    k: usize,
    head: usize,
    out: &Path,
) -> Result<()> {
    let t_start = Instant::now();
    let col = load_collection(data_dir)?;
    let queries = data::load_rows(&data_dir.join("queries.jsonl"))?;
    let qrels = data::load_qrels(&data_dir.join("qrels.tsv"))?;
    let all = build_candidates(&col, &queries, &qrels, k);
    eprintln!("nfcorpus: {} queries with qrels, candidates built in {:.1}s", all.len(), t_start.elapsed().as_secs_f64());

    let (tune_set, eval_set, tune_report, eval_name) = match mode {
        TuneMode::Scifact => {
            let tcol = load_collection(tune_dir)?;
            let tq = data::load_rows(&tune_dir.join("queries.jsonl"))?;
            let tqr = data::load_qrels(&tune_dir.join("qrels.tsv"))?;
            let tune = build_candidates(&tcol, &tq, &tqr, k);
            if tune.is_empty() {
                bail!("{}: no tuning queries with qrels", tune_dir.display());
            }
            let report = set_report("scifact (all queries with qrels)", tune_dir, &tune, &tcol, k);
            (tune, all, report, "nfcorpus (all queries with qrels)")
        }
        TuneMode::NfcorpusHead => {
            let mut tune = all;
            let n = head.min(tune.len());
            let eval = tune.split_off(n);
            if tune.is_empty() || eval.is_empty() {
                bail!("--head {head} leaves an empty tuning or evaluation split");
            }
            let report = set_report(
                &format!("nfcorpus first {n} queries (file order)"),
                data_dir,
                &tune,
                &col,
                k,
            );
            (tune, eval, report, "nfcorpus remaining queries")
        }
    };
    let eval_report = set_report(eval_name, data_dir, &eval_set, &col, k);

    let mut variants = Vec::with_capacity(VARIANTS.len());
    for v in VARIANTS {
        let (alpha, lambda, ndcg_tune, grid) = tune_variant(v, &tune_set);
        let ndcg_eval = mean_ndcg(&eval_set, v, alpha, lambda);
        eprintln!(
            "{:?}: alpha={alpha} lambda={lambda} tune nDCG@10={ndcg_tune:.4} eval nDCG@10={ndcg_eval:.4}",
            v
        );
        variants.push(VariantReport {
            name: v,
            alpha,
            lambda,
            ndcg10_tune: ndcg_tune,
            ndcg10_eval: ndcg_eval,
            grid,
        });
    }
    let eval_of = |v: Variant| -> f64 {
        variants
            .iter()
            .find(|r| r.name == v)
            .map_or(0.0, |r| r.ndcg10_eval)
    };
    let (best_contrast, ndcg_best) = [Variant::MeanFoil, Variant::MaxFoil]
        .into_iter()
        .map(|v| (v, eval_of(v)))
        .fold((Variant::MeanFoil, f64::NEG_INFINITY), |acc, x| if x.1 > acc.1 { x } else { acc });
    let ndcg_positive = eval_of(Variant::Positive);
    let ndcg_bm25 = eval_of(Variant::Bm25);
    let pass = ndcg_best >= PASS_A_NDCG && ndcg_best >= ndcg_positive + PASS_A_GAIN;
    println!(
        "PART A: bm25={ndcg_bm25:.4} positive={ndcg_positive:.4} {best_contrast:?}={ndcg_best:.4} \
         mean_foil={:.4} max_foil={:.4} hashed_pairs={:.4} int8={:.4} -> {}",
        eval_of(Variant::MeanFoil),
        eval_of(Variant::MaxFoil),
        eval_of(Variant::HashedPairs),
        eval_of(Variant::PositiveInt8),
        if pass { "PASS" } else { "FAIL" }
    );
    let report = PartAReport {
        part: "A",
        tune_mode: format!("{mode:?}"),
        eval_set: eval_report,
        tune_set: tune_report,
        variants,
        verdict: VerdictA {
            pass,
            best_contrast,
            ndcg10_best_contrast: ndcg_best,
            ndcg10_positive_only: ndcg_positive,
            ndcg10_bm25: ndcg_bm25,
            threshold_abs: PASS_A_NDCG,
            threshold_gain_over_positive: PASS_A_GAIN,
        },
    };
    write_json(out, &report)
}

// ---------------------------------------------------------------------------
// Part B
// ---------------------------------------------------------------------------

struct Corpus {
    index: Bm25Index,
    sketches: Vec<DocSketch>,
    pairs: Vec<PairTable>,
    triples: Vec<HashSet<Triple>>,
}

struct QIn {
    seeds: Vec<u64>,
    swapped: Vec<u64>,
    triple: Triple,
    candidates: Vec<u32>,
    doc: u32,
    present: bool,
    voice_match: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
enum Method {
    Bm25,
    RawOverlap,
    CleanupEnergy,
    Positive,
    MeanFoil,
    MaxFoil,
    HashedPairs,
    ExactTuple,
}

const METHODS: [Method; 8] = [
    Method::Bm25,
    Method::RawOverlap,
    Method::CleanupEnergy,
    Method::Positive,
    Method::MeanFoil,
    Method::MaxFoil,
    Method::HashedPairs,
    Method::ExactTuple,
];

const BASELINES: [Method; 3] = [Method::Bm25, Method::RawOverlap, Method::CleanupEnergy];
const CONTRASTS: [Method; 2] = [Method::MeanFoil, Method::MaxFoil];
const DEMOTERS: [Method; 2] = [Method::HashedPairs, Method::ExactTuple];

/// Scores every candidate of `q` into `out`; builds the query-side representation inside.
fn score_candidates(m: Method, corpus: &Corpus, q: &QIn, lambda_quarters: u32, out: &mut Vec<f64>) {
    out.clear();
    match m {
        Method::Bm25 => {
            let c = bm25::counts(&q.seeds);
            for &d in &q.candidates {
                out.push(corpus.index.score_doc(&c, d));
            }
        }
        Method::RawOverlap => {
            let c = bm25::counts(&q.seeds);
            let n = q.seeds.len() as u32;
            for &d in &q.candidates {
                out.push(bm25::unigram_overlap(&c, n, corpus.index.doc_terms(d)));
            }
        }
        Method::CleanupEnergy => {
            let qs = sketch::doc_sketch(&q.seeds);
            let scale = (DIM as f64).sqrt();
            for &d in &q.candidates {
                let ds = &corpus.sketches[d as usize];
                let e = if ds.int8_norm == 0.0 {
                    0.0
                } else {
                    f64::from(hv::dot_bits_i8(&qs.bits, &ds.int8)) / (ds.int8_norm * scale)
                };
                out.push(e);
            }
        }
        Method::Positive => {
            let qs = sketch::doc_sketch(&q.seeds);
            for &d in &q.candidates {
                out.push(f64::from(hv::dot_bits(&qs.bits, &corpus.sketches[d as usize].bits)));
            }
        }
        Method::MeanFoil => {
            let mut qs = sketch::query_sketch(&q.seeds);
            qs.push_foil(&q.swapped);
            let cv = qs.contrast_vector(lambda_quarters);
            for &d in &q.candidates {
                out.push(cv.score(&corpus.sketches[d as usize]));
            }
        }
        Method::MaxFoil => {
            let mut qs = sketch::query_sketch(&q.seeds);
            qs.push_foil(&q.swapped);
            for &d in &q.candidates {
                out.push(f64::from(qs.max_foil_score(&corpus.sketches[d as usize])));
            }
        }
        Method::HashedPairs => {
            let pt = PairTable::build(&q.seeds);
            for &d in &q.candidates {
                out.push(pt.coverage(&corpus.pairs[d as usize]));
            }
        }
        Method::ExactTuple => {
            for &d in &q.candidates {
                out.push(if corpus.triples[d as usize].contains(&q.triple) {
                    1.0
                } else {
                    0.0
                });
            }
        }
    }
}

/// Top candidate's score; a present query whose top candidate is not its source document is a reject.
fn accept_stat(q: &QIn, scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut best = 0usize;
    for i in 1..scores.len() {
        let better = scores[i] > scores[best]
            || (scores[i] == scores[best] && q.candidates[i] < q.candidates[best]);
        if better {
            best = i;
        }
    }
    if q.present && q.candidates[best] != q.doc {
        f64::NEG_INFINITY
    } else {
        scores[best]
    }
}

fn all_stats(m: Method, corpus: &Corpus, qins: &[QIn], lq: u32) -> Vec<f64> {
    let mut buf = Vec::with_capacity(32);
    qins.iter()
        .map(|q| {
            score_candidates(m, corpus, q, lq, &mut buf);
            accept_stat(q, &buf)
        })
        .collect()
}

#[derive(Serialize)]
struct Subset {
    n_present: usize,
    n_absent: usize,
    far: f64,
    tar: f64,
    auc: f64,
}

#[derive(Serialize)]
struct MethodReport {
    name: Method,
    lambda: Option<f64>,
    threshold: f64,
    far_at_tar80_calibrated: f64,
    tar_realized: f64,
    far_at_tar80_oracle: f64,
    auc_eval: f64,
    auc_all: f64,
    latency_us_per_query: f64,
    added_latency_us: f64,
    voice_match: Subset,
    voice_mismatch: Subset,
}

#[derive(Serialize)]
struct DemoteNote {
    method: Method,
    far: f64,
    latency_us_per_query: f64,
}

#[derive(Serialize)]
struct VerdictB {
    pass: bool,
    pass_far: bool,
    pass_auc: bool,
    pass_latency: bool,
    contrast_method: Method,
    contrast_far: f64,
    contrast_auc: f64,
    contrast_added_latency_us: f64,
    baseline_best_far: f64,
    baseline_best_auc: f64,
    demote: Vec<DemoteNote>,
}

#[derive(Serialize)]
struct PartBReport {
    part: &'static str,
    config: SynthConfig,
    n_docs: usize,
    n_queries: usize,
    calibration: &'static str,
    repeats: usize,
    warmup: usize,
    sketch_build_us_per_doc: f64,
    methods: Vec<MethodReport>,
    verdict: VerdictB,
}

fn find_report(reports: &[MethodReport], m: Method) -> Option<&MethodReport> {
    reports.iter().find(|r| r.name == m)
}

/// Indices of the calibration half (first half of each class) and the evaluation half.
fn halves(qins: &[QIn]) -> (Vec<usize>, Vec<usize>) {
    let mut by_class: HashMap<bool, Vec<usize>> = HashMap::new();
    for (i, q) in qins.iter().enumerate() {
        by_class.entry(q.present).or_default().push(i);
    }
    let (mut cal, mut ev) = (Vec::new(), Vec::new());
    for present in [true, false] {
        let idx = by_class.remove(&present).unwrap_or_default();
        let half = idx.len() / 2;
        cal.extend_from_slice(&idx[..half]);
        ev.extend_from_slice(&idx[half..]);
    }
    (cal, ev)
}

fn split_scores<'a>(
    stats: &[f64],
    qins: &[QIn],
    idx: impl Iterator<Item = &'a usize>,
) -> (Vec<f64>, Vec<f64>) {
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    for &i in idx {
        if qins[i].present {
            pos.push(stats[i]);
        } else {
            neg.push(stats[i]);
        }
    }
    (pos, neg)
}

fn subset(stats: &[f64], qins: &[QIn], ev: &[usize], threshold: f64, voice_match: bool) -> Subset {
    let (pos, neg) = split_scores(
        stats,
        qins,
        ev.iter().filter(|&&i| qins[i].voice_match == voice_match),
    );
    Subset {
        n_present: pos.len(),
        n_absent: neg.len(),
        far: eval::rate_at_or_above(&neg, threshold),
        tar: eval::rate_at_or_above(&pos, threshold),
        auc: eval::roc_auc(&pos, &neg),
    }
}

fn run_part_b(
    cfg: &SynthConfig,
    repeats: usize,
    warmup: usize,
    out: &Path,
    dump: bool,
) -> Result<()> {
    let t_start = Instant::now();
    let set = synth::generate(cfg);
    eprintln!(
        "synth: {} docs, {} queries, generated in {:.2}s",
        set.docs.len(),
        set.queries.len(),
        t_start.elapsed().as_secs_f64()
    );
    if dump {
        let dir = out.parent().unwrap_or_else(|| Path::new(OUT_DIR));
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let mut corpus_lines = String::new();
        for d in &set.docs {
            corpus_lines.push_str(&serde_json::to_string(d).context("serialising synth doc")?);
            corpus_lines.push('\n');
        }
        let corpus_path = dir.join("synth_corpus.jsonl");
        fs::write(&corpus_path, corpus_lines)
            .with_context(|| format!("writing {}", corpus_path.display()))?;
        let mut query_lines = String::new();
        for q in &set.queries {
            query_lines.push_str(&serde_json::to_string(q).context("serialising synth query")?);
            query_lines.push('\n');
        }
        let query_path = dir.join("synth_queries.jsonl");
        fs::write(&query_path, query_lines)
            .with_context(|| format!("writing {}", query_path.display()))?;
        eprintln!("wrote {} and {}", corpus_path.display(), query_path.display());
    }

    let seeds: Vec<Vec<u64>> = set.docs.iter().map(|d| bm25::seeds(&d.text)).collect();
    let index = Bm25Index::build(&seeds);
    let t0 = Instant::now();
    let sketches: Vec<DocSketch> = seeds.iter().map(|s| sketch::doc_sketch(s)).collect();
    let sketch_us_per_doc = t0.elapsed().as_secs_f64() * 1e6 / seeds.len().max(1) as f64;
    let pairs: Vec<PairTable> = seeds.iter().map(|s| PairTable::build(s)).collect();
    let triples: Vec<HashSet<Triple>> = set.docs.iter().map(|d| d.triples.clone()).collect();
    let corpus = Corpus {
        index,
        sketches,
        pairs,
        triples,
    };
    let qins: Vec<QIn> = set
        .queries
        .iter()
        .map(|q| QIn {
            seeds: bm25::seeds(&q.text),
            swapped: bm25::seeds(&q.swapped_text),
            triple: q.triple,
            candidates: q.candidates.clone(),
            doc: q.doc,
            present: q.present,
            voice_match: q.voice_match,
        })
        .collect();
    let (cal, ev) = halves(&qins);
    if cal.is_empty() || ev.is_empty() {
        bail!("need at least two queries per class");
    }

    // Lambda for mean-foil: best calibration-half AUC (smallest lambda at ties).
    let mut best_lq = (LAMBDA_QUARTERS[0], f64::NEG_INFINITY);
    for &lq in &LAMBDA_QUARTERS {
        let stats = all_stats(Method::MeanFoil, &corpus, &qins, lq);
        let (pos, neg) = split_scores(&stats, &qins, cal.iter());
        let auc = eval::roc_auc(&pos, &neg);
        eprintln!("mean_foil lambda={} calibration AUC={auc:.4}", lq as f64 / 4.0);
        if auc > best_lq.1 {
            best_lq = (lq, auc);
        }
    }
    let lq = best_lq.0;

    let mut reports: Vec<MethodReport> = Vec::with_capacity(METHODS.len());
    let mut latencies: HashMap<Method, f64> = HashMap::new();
    for m in METHODS {
        let stats = all_stats(m, &corpus, &qins, lq);
        let (cal_pos, _) = split_scores(&stats, &qins, cal.iter());
        let (ev_pos, ev_neg) = split_scores(&stats, &qins, ev.iter());
        let (all_pos, all_neg) = split_scores(&stats, &qins, (0..qins.len()).collect::<Vec<_>>().iter());
        let threshold = eval::threshold_at_tar(&cal_pos, TAR_TARGET);
        let oracle = eval::threshold_at_tar(&ev_pos, TAR_TARGET);

        let mut buf = Vec::with_capacity(32);
        let mut sink = 0.0f64;
        let secs = eval::time_median_secs(warmup, repeats, || {
            let mut acc = 0.0f64;
            for q in &qins {
                score_candidates(m, &corpus, q, lq, &mut buf);
                let s = accept_stat(q, &buf);
                if s.is_finite() {
                    acc += s;
                }
            }
            sink += black_box(acc);
        });
        black_box(sink);
        let latency_us = secs * 1e6 / qins.len() as f64;
        latencies.insert(m, latency_us);

        let report = MethodReport {
            name: m,
            lambda: if m == Method::MeanFoil { Some(lq as f64 / 4.0) } else { None },
            threshold,
            far_at_tar80_calibrated: eval::rate_at_or_above(&ev_neg, threshold),
            tar_realized: eval::rate_at_or_above(&ev_pos, threshold),
            far_at_tar80_oracle: eval::rate_at_or_above(&ev_neg, oracle),
            auc_eval: eval::roc_auc(&ev_pos, &ev_neg),
            auc_all: eval::roc_auc(&all_pos, &all_neg),
            latency_us_per_query: latency_us,
            added_latency_us: 0.0,
            voice_match: subset(&stats, &qins, &ev, threshold, true),
            voice_mismatch: subset(&stats, &qins, &ev, threshold, false),
        };
        eprintln!(
            "{:?}: FAR@TAR80={:.4} (tar {:.3}) oracle FAR={:.4} AUC={:.4} latency={:.2} us",
            m,
            report.far_at_tar80_calibrated,
            report.tar_realized,
            report.far_at_tar80_oracle,
            report.auc_eval,
            latency_us
        );
        reports.push(report);
    }
    let bm25_latency = latencies.get(&Method::Bm25).copied().unwrap_or(0.0);
    for r in reports.iter_mut() {
        r.added_latency_us = r.latency_us_per_query - bm25_latency;
    }

    let find = |m: Method| find_report(&reports, m);
    let contrast = CONTRASTS
        .iter()
        .filter_map(|&m| find(m))
        .min_by(|a, b| {
            a.far_at_tar80_calibrated
                .total_cmp(&b.far_at_tar80_calibrated)
                .then_with(|| b.auc_eval.total_cmp(&a.auc_eval))
        })
        .context("contrast methods missing from the report")?;
    let baseline_best_far = BASELINES
        .iter()
        .filter_map(|&m| find(m))
        .map(|r| r.far_at_tar80_calibrated)
        .fold(f64::INFINITY, f64::min);
    let baseline_best_auc = BASELINES
        .iter()
        .filter_map(|&m| find(m))
        .map(|r| r.auc_eval)
        .fold(f64::NEG_INFINITY, f64::max);
    let pass_far = contrast.far_at_tar80_calibrated <= 0.5 * baseline_best_far;
    let pass_auc = contrast.auc_eval >= baseline_best_auc + PASS_B_AUC_GAIN;
    let pass_latency = contrast.added_latency_us < PASS_B_LATENCY_US;
    let pass = pass_far && pass_auc && pass_latency;
    let demote: Vec<DemoteNote> = DEMOTERS
        .iter()
        .filter_map(|&m| find(m))
        .filter(|r| {
            r.far_at_tar80_calibrated <= contrast.far_at_tar80_calibrated
                && r.latency_us_per_query < contrast.latency_us_per_query
        })
        .map(|r| DemoteNote {
            method: r.name,
            far: r.far_at_tar80_calibrated,
            latency_us_per_query: r.latency_us_per_query,
        })
        .collect();

    println!(
        "PART B: contrast={:?} FAR={:.4} (baseline best {:.4}, half {:.4}) AUC={:.4} (baseline best {:.4}) \
         added={:.2} us -> far:{} auc:{} latency:{} -> {}",
        contrast.name,
        contrast.far_at_tar80_calibrated,
        baseline_best_far,
        0.5 * baseline_best_far,
        contrast.auc_eval,
        baseline_best_auc,
        contrast.added_latency_us,
        pass_far,
        pass_auc,
        pass_latency,
        if pass { "PASS" } else { "FAIL" }
    );
    for d in &demote {
        println!(
            "DEMOTE: {:?} FAR={:.4} at {:.2} us/query matches or beats {:?} FAR={:.4} at {:.2} us/query",
            d.method,
            d.far,
            d.latency_us_per_query,
            contrast.name,
            contrast.far_at_tar80_calibrated,
            contrast.latency_us_per_query
        );
    }

    let verdict = VerdictB {
        pass,
        pass_far,
        pass_auc,
        pass_latency,
        contrast_method: contrast.name,
        contrast_far: contrast.far_at_tar80_calibrated,
        contrast_auc: contrast.auc_eval,
        contrast_added_latency_us: contrast.added_latency_us,
        baseline_best_far,
        baseline_best_auc,
        demote,
    };
    let report = PartBReport {
        part: "B",
        config: cfg.clone(),
        n_docs: set.docs.len(),
        n_queries: set.queries.len(),
        calibration: "threshold and lambda from the first half of each query class; FAR/TAR/AUC on the second half",
        repeats,
        warmup,
        sketch_build_us_per_doc: sketch_us_per_doc,
        methods: reports,
        verdict,
    };
    write_json(out, &report)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Nfcorpus {
            data,
            tune_data,
            tune,
            candidates,
            head,
            out,
        } => {
            let data = data.unwrap_or_else(|| home_cache("nfcorpus"));
            let tune_data = tune_data.unwrap_or_else(|| home_cache("scifact"));
            run_part_a(&data, &tune_data, tune, candidates, head, &out)
        }
        Cmd::Synth {
            seed,
            docs,
            queries,
            candidates,
            repeats,
            warmup,
            out,
            no_dump,
        } => {
            let cfg = SynthConfig {
                seed,
                n_docs: docs,
                n_queries_per_class: queries,
                candidates,
                ..SynthConfig::default()
            };
            run_part_b(&cfg, repeats, warmup, &out, !no_dump)
        }
        Cmd::All => {
            run_part_a(
                &home_cache("nfcorpus"),
                &home_cache("scifact"),
                TuneMode::Scifact,
                200,
                100,
                Path::new(OUT_A),
            )?;
            run_part_b(&SynthConfig::default(), 7, 1, Path::new(OUT_B), true)
        }
    }
}
