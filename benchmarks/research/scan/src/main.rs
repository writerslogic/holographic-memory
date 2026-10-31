// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `warm-research-scan`: build a dataset, freeze the public basis, encode the
//! split-manifold / int8 / exact representations, and measure recall@10,
//! single-thread QPS (median of `--runs` passes over all queries after a
//! warmup), encode time, bytes per document, and delete/replace bytes
//! touched. Prints PASS/FAIL per panel threshold and writes one JSON report
//! (schema in `RUN.md`).
//!
//! `--sweep` evaluates head dims x residual mults on one dataset against the
//! shared int8/exact baselines and prints a compact table (no mutation
//! experiment).
//!
//! Everything runs on one thread; all randomness derives from `--seed`.

#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;

use warm_research_scan::codes::{
    fit_basis, BasisKind, BasisParams, Int8Index, Primary, ResidualMode, Scratch, SplitIndex,
};
use warm_research_scan::data::{self, Dataset, Rng, SyntheticParams};
use warm_research_scan::median;
use warm_research_scan::scan::{Baseline, Searcher};

const SCHEMA: &str = "warm-research-scan/v2";
const SWEEP_SCHEMA: &str = "warm-research-scan/sweep-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum DatasetKind {
    /// Fixed-seed anisotropic Gaussian clusters.
    Synthetic,
    /// BEIR NFCorpus MiniLM-L6 384-d rows padded with on-manifold distractors.
    Nfcorpus,
    /// Seeded subset of ann-benchmarks glove-100-angular.
    Glove,
}

#[derive(Parser, Debug)]
#[command(
    name = "warm-research-scan",
    about = "Kill test: graph-free split-manifold scan vs int8 and exact f32 scans"
)]
struct Cli {
    /// Corpus source.
    #[arg(long, value_enum, default_value_t = DatasetKind::Synthetic)]
    dataset: DatasetKind,
    /// Corpus size (synthetic: generated; nfcorpus: real rows + distractors; glove: subset).
    #[arg(long, default_value_t = 50_000)]
    n: usize,
    /// Dimension (synthetic only; real datasets read it from meta.json).
    #[arg(long, default_value_t = 768)]
    dim: usize,
    /// Synthetic cluster count.
    #[arg(long, default_value_t = 200)]
    clusters: usize,
    /// Synthetic within-cluster perturbation norm.
    #[arg(long, default_value_t = 0.75)]
    spread: f32,
    /// Synthetic spectrum decay exponent (0 = isotropic).
    #[arg(long, default_value_t = 0.5)]
    spectrum_alpha: f32,
    /// Query count (capped by the real query file for nfcorpus/glove).
    #[arg(long, default_value_t = 500)]
    queries: usize,
    /// Master seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Head subspace: PCA of the calibration sample or a fixed random basis.
    #[arg(long, value_enum, default_value_t = BasisKind::Pca)]
    basis: BasisKind,
    /// Head dimension H: 16, 32 or 64 int8 lanes.
    #[arg(long, alias = "head-dims", default_value_t = 16)]
    head_dim: usize,
    /// Independent residual rotations M (bits default to M x padded dim when M >= 2).
    #[arg(long, default_value_t = 1)]
    residual_mult: usize,
    /// Residual evaluation: 1-bit query (XOR/popcount) or 4-bit query planes.
    #[arg(long, value_enum, default_value_t = ResidualMode::Sym1)]
    residual_mode: ResidualMode,
    /// Residual bits B (multiple of 64, <= M x padded dim; default: dim rounded up to 64, or M x padded dim when M >= 2).
    #[arg(long)]
    residual_bits: Option<usize>,
    /// Calibration sample size for mu / PCA.
    #[arg(long, default_value_t = 1000)]
    calib: usize,
    /// PCA subspace-iteration sweeps.
    #[arg(long, default_value_t = 32)]
    pca_iters: usize,
    /// Candidate counts K for the approximate methods.
    #[arg(long, value_delimiter = ',', default_values_t = [64usize, 256, 1024])]
    ks: Vec<usize>,
    /// Timed passes over the query set per (method, K); QPS uses the median.
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Warmup queries before timing.
    #[arg(long, default_value_t = 50)]
    warmup: usize,
    /// Results returned per query (recall@top).
    #[arg(long, default_value_t = 10)]
    top: usize,
    /// Delete/replace operations in the mutation experiment.
    #[arg(long, default_value_t = 1000)]
    deletes: usize,
    /// NFCorpus directory (default ~/.cache/warm-bench/nfcorpus).
    #[arg(long)]
    nfcorpus_dir: Option<PathBuf>,
    /// glove-100-angular directory (default ~/.cache/warm-bench/glove-100-angular).
    #[arg(long)]
    glove_dir: Option<PathBuf>,
    /// NFCorpus distractor perturbation norm.
    #[arg(long, default_value_t = 0.6)]
    distractor_noise: f32,
    /// Replacement-vector perturbation norm in the mutation experiment.
    #[arg(long, default_value_t = 0.3)]
    replace_noise: f32,
    /// Output JSON path (default benchmarks/research/scan/out/<dataset>_<n>_<basis>_<mode>[_sweep].json).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Threshold: split recall@top at some K <= threshold-max-k.
    #[arg(long, default_value_t = 0.95)]
    target_recall: f64,
    /// Threshold: split single-thread QPS at the matched K.
    #[arg(long, default_value_t = 2500.0)]
    min_qps: f64,
    /// Threshold: split/int8 QPS ratio at matched recall.
    #[arg(long, default_value_t = 1.5)]
    min_ratio: f64,
    /// Threshold: split encode time for 50k docs (scaled linearly to --n).
    #[arg(long, default_value_t = 250.0)]
    max_encode_ms_50k: f64,
    /// Largest K the recall/QPS thresholds may use.
    #[arg(long, default_value_t = 256)]
    threshold_max_k: usize,
    /// Sweep head dims x residual mults on this dataset (table + JSON, no mutation experiment).
    #[arg(long, default_value_t = false)]
    sweep: bool,
    /// Head dims for --sweep.
    #[arg(long, value_delimiter = ',', default_values_t = [16usize, 32, 64])]
    sweep_heads: Vec<usize>,
    /// Residual mults for --sweep.
    #[arg(long, value_delimiter = ',', default_values_t = [1usize, 2])]
    sweep_mults: Vec<usize>,
}

#[derive(Serialize)]
struct DatasetInfo {
    kind: DatasetKind,
    name: String,
    source: String,
    n_docs: usize,
    n_real_docs: usize,
    dim: usize,
    n_queries: usize,
    seed: u64,
    load_ms: f64,
    ground_truth_ms: f64,
}

#[derive(Serialize)]
struct SharedConfig {
    basis: BasisKind,
    residual_mode: ResidualMode,
    ks: Vec<usize>,
    runs: usize,
    warmup_queries: usize,
    rerank_top: usize,
    calib_vectors: usize,
    pca_iters: usize,
    threads: usize,
    target: String,
}

#[derive(Serialize)]
struct SplitConfig {
    head_dim: usize,
    head_words: usize,
    residual_mult: usize,
    residual_bits: usize,
    residual_words: usize,
    padded_dim: usize,
    padded_total: usize,
    head_variance_fraction: f32,
    basis_fit_ms: f64,
}

#[derive(Serialize, Clone)]
struct KReport {
    k: usize,
    recall_at_top: f64,
    qps: f64,
    us_per_query: f64,
    median_run_ms: f64,
    run_ms: Vec<f64>,
}

#[derive(Serialize)]
struct MethodReport {
    bytes_per_doc: usize,
    encode_ms: f64,
    encode_us_per_doc: f64,
    per_k: Vec<KReport>,
}

#[derive(Serialize)]
struct Methods {
    split: MethodReport,
    int8: MethodReport,
    exact: MethodReport,
}

#[derive(Serialize)]
struct EqualK {
    k: usize,
    split_recall: f64,
    int8_recall: f64,
    split_qps: f64,
    int8_qps: f64,
    ratio_split_vs_int8: f64,
}

#[derive(Serialize)]
struct Ablation {
    target_recall: f64,
    threshold_max_k: usize,
    split_k_star: Option<usize>,
    int8_k_star: Option<usize>,
    split_qps_at_k_star: Option<f64>,
    int8_qps_at_k_star: Option<f64>,
    exact_qps: f64,
    ratio_split_vs_int8_matched: Option<f64>,
    ratio_split_vs_exact_matched: Option<f64>,
    ratio_at_equal_k: Vec<EqualK>,
}

#[derive(Serialize)]
struct MutReport {
    delete_bytes_touched: usize,
    delete_ns_per_op: f64,
    replace_bytes_touched: usize,
    replace_us_per_op: f64,
}

#[derive(Serialize)]
struct Mutation {
    ops: usize,
    leak_check_k: usize,
    deleted_leaks: usize,
    replace_self_hits: usize,
    replace_self_checks: usize,
    split: MutReport,
    int8: MutReport,
    exact: MutReport,
}

#[derive(Serialize, Clone)]
struct Threshold {
    name: String,
    value: f64,
    limit: f64,
    pass: bool,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    generated_unix_s: u64,
    dataset: DatasetInfo,
    config: SharedConfig,
    split_config: SplitConfig,
    methods: Methods,
    ablation: Ablation,
    mutation: Mutation,
    thresholds: Vec<Threshold>,
    pass: bool,
}

#[derive(Serialize)]
struct SweepRow {
    split_config: SplitConfig,
    bytes_per_doc: usize,
    encode_ms: f64,
    per_k: Vec<KReport>,
    k_star: Option<usize>,
    qps_at_k_star: Option<f64>,
    ratio_vs_int8_matched: Option<f64>,
    thresholds: Vec<Threshold>,
    pass: bool,
}

#[derive(Serialize)]
struct SweepReport {
    schema: &'static str,
    generated_unix_s: u64,
    dataset: DatasetInfo,
    config: SharedConfig,
    int8: MethodReport,
    exact: MethodReport,
    int8_k_star: Option<usize>,
    rows: Vec<SweepRow>,
    any_pass: bool,
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn home_cache(sub: &str) -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("warm-bench")
        .join(sub)
}

fn tag(basis: BasisKind, mode: ResidualMode) -> (&'static str, &'static str) {
    let b = match basis {
        BasisKind::Pca => "pca",
        BasisKind::Random => "random",
    };
    let m = match mode {
        ResidualMode::Sym1 => "sym1",
        ResidualMode::Asym4 => "asym4",
    };
    (b, m)
}

fn target_description() -> String {
    let mut v = vec![std::env::consts::ARCH.to_string()];
    if cfg!(target_feature = "neon") {
        v.push("neon".to_string());
    }
    if cfg!(target_feature = "dotprod") {
        v.push("dotprod".to_string());
    }
    if cfg!(target_feature = "avx2") {
        v.push("avx2".to_string());
    }
    if cfg!(target_feature = "avx512f") {
        v.push("avx512f".to_string());
    }
    if cfg!(target_feature = "popcnt") {
        v.push("popcnt".to_string());
    }
    v.join(",")
}

fn build_dataset(cli: &Cli) -> Result<Dataset> {
    match cli.dataset {
        DatasetKind::Synthetic => Ok(data::synthetic(&SyntheticParams {
            n: cli.n,
            dim: cli.dim,
            clusters: cli.clusters.max(1),
            n_queries: cli.queries,
            seed: cli.seed,
            spread: cli.spread,
            spectrum_alpha: cli.spectrum_alpha,
        })),
        DatasetKind::Nfcorpus => {
            let dir = cli
                .nfcorpus_dir
                .clone()
                .unwrap_or_else(|| home_cache("nfcorpus"));
            data::nfcorpus(&dir, cli.n, cli.queries, cli.distractor_noise, cli.seed)
        }
        DatasetKind::Glove => {
            let dir = cli
                .glove_dir
                .clone()
                .unwrap_or_else(|| home_cache("glove-100-angular"));
            data::glove(&dir, cli.n, cli.queries, cli.seed)
        }
    }
}

fn recall(results: &[u32], truth: &[u32]) -> f64 {
    if truth.is_empty() {
        return 0.0;
    }
    let hit = results.iter().filter(|r| truth.contains(r)).count();
    hit as f64 / truth.len() as f64
}

/// Recall pass, warmup, then `runs` timed passes; returns (recall, run_ms).
fn measure(
    nq: usize,
    warmup: usize,
    runs: usize,
    gt: &[Vec<u32>],
    f: &mut dyn FnMut(usize, &mut Vec<u32>),
) -> (f64, Vec<f64>) {
    let mut buf: Vec<u32> = Vec::with_capacity(16);
    let mut hits = 0.0f64;
    for qi in 0..nq {
        f(qi, &mut buf);
        hits += recall(&buf, &gt[qi]);
    }
    let rec = hits / nq as f64;
    for qi in 0..warmup.min(nq) {
        f(qi, &mut buf);
        black_box(buf.len());
    }
    let mut run_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        for qi in 0..nq {
            f(qi, &mut buf);
            black_box(buf.len());
        }
        run_ms.push(ms_since(t));
    }
    (rec, run_ms)
}

fn k_report(k: usize, nq: usize, rec: f64, run_ms: Vec<f64>) -> KReport {
    let med = median(&run_ms);
    let qps = if med > 0.0 { nq as f64 / (med / 1e3) } else { 0.0 };
    KReport {
        k,
        recall_at_top: rec,
        qps,
        us_per_query: if nq > 0 { med * 1e3 / nq as f64 } else { 0.0 },
        median_run_ms: med,
        run_ms,
    }
}

/// Smallest K (<= max_k) whose recall reaches `target`.
fn k_star(per_k: &[KReport], target: f64, max_k: usize) -> Option<&KReport> {
    per_k
        .iter()
        .filter(|r| r.k <= max_k && r.recall_at_top >= target)
        .min_by_key(|r| r.k)
}

/// Default residual bits: all `M x P` coordinates for `M >= 2`, else `D`
/// rounded up to 64 (capped at `P`).
fn default_bits(dim: usize, mult: usize) -> usize {
    let padded = dim.next_power_of_two().max(64);
    if mult >= 2 {
        padded * mult
    } else {
        (dim.div_ceil(64).max(1) * 64).min(padded)
    }
}

struct SplitEval {
    config: SplitConfig,
    encode_ms: f64,
    bytes_per_doc: usize,
    per_k: Vec<KReport>,
    index: SplitIndex,
    scratch: Scratch,
    searcher: Searcher,
}

#[allow(clippy::too_many_arguments)]
fn eval_split(
    cli: &Cli,
    head_dim: usize,
    residual_mult: usize,
    residual_bits: Option<usize>,
    primary: &Primary,
    queries: &[f32],
    gt: &[Vec<u32>],
    ks: &[usize],
    calib: &[f32],
) -> Result<SplitEval> {
    let dim = primary.dim;
    let n = primary.n;
    let nq = gt.len();
    let bits = residual_bits.unwrap_or_else(|| default_bits(dim, residual_mult));
    let t = Instant::now();
    let basis = fit_basis(
        &BasisParams {
            kind: cli.basis,
            dim,
            head_dim,
            residual_mult,
            residual_bits: bits,
            seed: cli.seed,
            pca_iters: cli.pca_iters,
        },
        calib,
    )?;
    let basis_fit_ms = ms_since(t);
    let max_k = ks.iter().copied().max().unwrap_or(cli.top).max(cli.top);
    let mut searcher = Searcher::new(&basis, cli.residual_mode, max_k, cli.top);
    eprintln!(
        "[basis] head={} mult={} bits={} words={} padded={} head_var_frac={:.3} fit={:.1} ms",
        head_dim, residual_mult, basis.bits, basis.words, basis.padded, basis.head_variance_fraction, basis_fit_ms
    );
    let config = SplitConfig {
        head_dim,
        head_words: basis.head_words,
        residual_mult,
        residual_bits: basis.bits,
        residual_words: basis.words,
        padded_dim: basis.padded,
        padded_total: basis.padded_total,
        head_variance_fraction: basis.head_variance_fraction,
        basis_fit_ms,
    };
    let mut index = SplitIndex::with_capacity(basis, n, cli.residual_mode);
    let mut scratch = Scratch::new(&index.basis);
    let t = Instant::now();
    for i in 0..n {
        index.encode(i, primary.row(i), &mut scratch);
    }
    let encode_ms = ms_since(t);
    let bytes_per_doc = index.bytes_per_doc();
    eprintln!(
        "[encode] split head={} mult={} {:.1} ms ({} B/doc)",
        head_dim, residual_mult, encode_ms, bytes_per_doc
    );
    let mut per_k = Vec::with_capacity(ks.len());
    for &k in ks {
        searcher.set_k(k);
        let (rec, run_ms) = measure(nq, cli.warmup, cli.runs, gt, &mut |qi, out| {
            let r = searcher.split(&index, primary, &queries[qi * dim..(qi + 1) * dim]);
            out.clear();
            out.extend_from_slice(r);
        });
        let rep = k_report(k, nq, rec, run_ms);
        eprintln!(
            "[split] head={} mult={} K={:<5} recall@{}={:.4} qps={:.0} ({:.1} us/q)",
            head_dim, residual_mult, k, cli.top, rep.recall_at_top, rep.qps, rep.us_per_query
        );
        per_k.push(rep);
    }
    Ok(SplitEval {
        config,
        encode_ms,
        bytes_per_doc,
        per_k,
        index,
        scratch,
        searcher,
    })
}

fn eval_int8(
    cli: &Cli,
    base: &mut Baseline,
    primary: &Primary,
    queries: &[f32],
    gt: &[Vec<u32>],
    ks: &[usize],
) -> (Int8Index, MethodReport) {
    let dim = primary.dim;
    let n = primary.n;
    let nq = gt.len();
    let mut int8 = Int8Index::with_capacity(n, dim);
    let t = Instant::now();
    for i in 0..n {
        int8.encode(i, primary.row(i));
    }
    let encode_ms = ms_since(t);
    eprintln!("[encode] int8 {:.1} ms ({} B/doc)", encode_ms, int8.bytes_per_doc());
    let mut per_k = Vec::with_capacity(ks.len());
    for &k in ks {
        base.set_k(k);
        let (rec, run_ms) = measure(nq, cli.warmup, cli.runs, gt, &mut |qi, out| {
            let r = base.int8(&int8, primary, &queries[qi * dim..(qi + 1) * dim]);
            out.clear();
            out.extend_from_slice(r);
        });
        let rep = k_report(k, nq, rec, run_ms);
        eprintln!(
            "[int8]  K={:<5} recall@{}={:.4} qps={:.0} ({:.1} us/q)",
            k, cli.top, rep.recall_at_top, rep.qps, rep.us_per_query
        );
        per_k.push(rep);
    }
    let report = MethodReport {
        bytes_per_doc: int8.bytes_per_doc(),
        encode_ms,
        encode_us_per_doc: encode_ms * 1e3 / n as f64,
        per_k,
    };
    (int8, report)
}

fn eval_exact(
    cli: &Cli,
    base: &mut Baseline,
    primary: &Primary,
    queries: &[f32],
    gt: &[Vec<u32>],
) -> MethodReport {
    let dim = primary.dim;
    let n = primary.n;
    let nq = gt.len();
    let t = Instant::now();
    let copy = primary.rows.clone();
    let encode_ms = ms_since(t);
    black_box(copy.len());
    drop(copy);
    let (rec, run_ms) = measure(nq, cli.warmup, cli.runs, gt, &mut |qi, out| {
        let r = base.exact(primary, &queries[qi * dim..(qi + 1) * dim]);
        out.clear();
        out.extend_from_slice(r);
    });
    let rep = k_report(cli.top, nq, rec, run_ms);
    eprintln!(
        "[exact] recall@{}={:.4} qps={:.0} ({:.1} us/q)",
        cli.top, rep.recall_at_top, rep.qps, rep.us_per_query
    );
    MethodReport {
        bytes_per_doc: primary.bytes_per_doc(),
        encode_ms,
        encode_us_per_doc: encode_ms * 1e3 / n as f64,
        per_k: vec![rep],
    }
}

struct Verdict {
    thresholds: Vec<Threshold>,
    split_star: Option<KReport>,
    int8_star: Option<KReport>,
    ratio_matched: Option<f64>,
}

/// Panel thresholds 1-4 for one split configuration against the int8 baseline.
fn core_thresholds(
    cli: &Cli,
    n: usize,
    split_k: &[KReport],
    int8_k: &[KReport],
    encode_ms: f64,
) -> Verdict {
    let split_star = k_star(split_k, cli.target_recall, cli.threshold_max_k).cloned();
    let int8_star = k_star(int8_k, cli.target_recall, cli.threshold_max_k).cloned();
    let ratio_matched = match (&split_star, &int8_star) {
        (Some(s), Some(i)) if i.qps > 0.0 => Some(s.qps / i.qps),
        _ => None,
    };
    let encode_limit = cli.max_encode_ms_50k * n as f64 / 50_000.0;
    let best_recall = split_k
        .iter()
        .filter(|r| r.k <= cli.threshold_max_k)
        .map(|r| r.recall_at_top)
        .fold(0.0f64, f64::max);
    let thresholds = vec![
        Threshold {
            name: format!(
                "split recall@{} >= {} at some K <= {}",
                cli.top, cli.target_recall, cli.threshold_max_k
            ),
            value: best_recall,
            limit: cli.target_recall,
            pass: best_recall >= cli.target_recall,
        },
        Threshold {
            name: format!(
                "split single-thread QPS >= {} at K* (smallest K <= {} reaching recall {})",
                cli.min_qps, cli.threshold_max_k, cli.target_recall
            ),
            value: split_star.as_ref().map(|s| s.qps).unwrap_or(0.0),
            limit: cli.min_qps,
            pass: split_star
                .as_ref()
                .map(|s| s.qps >= cli.min_qps)
                .unwrap_or(false),
        },
        Threshold {
            name: format!(
                "split / int8+rerank QPS ratio at matched recall >= {}",
                cli.min_ratio
            ),
            value: ratio_matched.unwrap_or(0.0),
            limit: cli.min_ratio,
            pass: ratio_matched.map(|r| r >= cli.min_ratio).unwrap_or(false),
        },
        Threshold {
            name: format!(
                "split encode time for {} docs < {:.1} ms ({} ms per 50k, scaled)",
                n, encode_limit, cli.max_encode_ms_50k
            ),
            value: encode_ms,
            limit: encode_limit,
            pass: encode_ms < encode_limit,
        },
    ];
    Verdict {
        thresholds,
        split_star,
        int8_star,
        ratio_matched,
    }
}

fn print_thresholds(thresholds: &[Threshold], pass: bool) {
    for t in thresholds {
        println!(
            "[{}] {} (value={:.4}, limit={:.4})",
            if t.pass { "PASS" } else { "FAIL" },
            t.name,
            t.value,
            t.limit
        );
    }
    println!("[{}] overall", if pass { "PASS" } else { "FAIL" });
}

fn write_json<T: Serialize>(out: &Path, report: &T) -> Result<()> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(report).context("serialize report")?;
    fs::write(out, text).with_context(|| format!("write {}", out.display()))?;
    println!("wrote {}", out.display());
    Ok(())
}

/// Delete/replace experiment on the single configuration; mutates all indexes.
#[allow(clippy::too_many_arguments)]
fn mutation_experiment(
    cli: &Cli,
    ev: &mut SplitEval,
    int8: &mut Int8Index,
    primary: &mut Primary,
    queries: &[f32],
    max_k: usize,
) -> Mutation {
    let dim = primary.dim;
    let n = primary.n;
    let nq = queries.len() / dim;
    let ops = cli.deletes.min(n / 2);
    let mut mrng = Rng::new(cli.seed ^ 0x0DE1_E7E0_0000_0001);
    let dead_ids = mrng.sample_indices(n, ops);
    let dead_set: HashSet<u32> = dead_ids.iter().copied().collect();
    let per_op = |t: Instant, scale: f64| -> f64 {
        if ops == 0 {
            0.0
        } else {
            ms_since(t) * scale / ops as f64
        }
    };

    let t = Instant::now();
    let mut split_del_bytes = 0usize;
    for &id in &dead_ids {
        split_del_bytes = ev.index.delete(id as usize) + primary.delete(id as usize);
    }
    let split_del_ns = per_op(t, 1e6);
    let t = Instant::now();
    let mut int8_del_bytes = 0usize;
    for &id in &dead_ids {
        int8_del_bytes = int8.delete(id as usize) + primary.delete(id as usize);
    }
    let int8_del_ns = per_op(t, 1e6);
    let t = Instant::now();
    let mut exact_del_bytes = 0usize;
    for &id in &dead_ids {
        exact_del_bytes = primary.delete(id as usize);
    }
    let exact_del_ns = per_op(t, 1e6);

    let leak_k = max_k.min(n);
    ev.searcher.set_k(leak_k);
    let mut leaks = 0usize;
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        leaks += ev
            .searcher
            .split(&ev.index, primary, q)
            .iter()
            .filter(|id| dead_set.contains(*id))
            .count();
        leaks += ev
            .searcher
            .int8(int8, primary, q)
            .iter()
            .filter(|id| dead_set.contains(*id))
            .count();
        leaks += ev
            .searcher
            .exact(primary, q)
            .iter()
            .filter(|id| dead_set.contains(*id))
            .count();
    }

    let mut fresh = vec![0.0f32; ops * dim];
    let mut src = vec![0.0f32; dim];
    for row in fresh.chunks_exact_mut(dim) {
        let mut j = mrng.below(n);
        while primary.live[j] == 0 {
            j = mrng.below(n);
        }
        src.copy_from_slice(primary.row(j));
        data::perturb(&mut mrng, &src, cli.replace_noise, row);
    }
    let t = Instant::now();
    for (j, &id) in dead_ids.iter().enumerate() {
        let v = &fresh[j * dim..(j + 1) * dim];
        ev.index.encode(id as usize, v, &mut ev.scratch);
        primary.replace(id as usize, v);
    }
    let split_rep_us = per_op(t, 1e3);
    let split_rep_bytes = ev.index.bytes_per_doc() + primary.bytes_per_doc();
    let t = Instant::now();
    for (j, &id) in dead_ids.iter().enumerate() {
        let v = &fresh[j * dim..(j + 1) * dim];
        int8.encode(id as usize, v);
        primary.replace(id as usize, v);
    }
    let int8_rep_us = per_op(t, 1e3);
    let int8_rep_bytes = int8.bytes_per_doc() + primary.bytes_per_doc();
    let t = Instant::now();
    for (j, &id) in dead_ids.iter().enumerate() {
        primary.replace(id as usize, &fresh[j * dim..(j + 1) * dim]);
    }
    let exact_rep_us = per_op(t, 1e3);
    let exact_rep_bytes = primary.bytes_per_doc();

    let self_checks = ops.min(50);
    let mut self_hits = 0usize;
    let small_k = cli.ks.iter().copied().min().unwrap_or(cli.top).max(cli.top);
    ev.searcher.set_k(small_k);
    for j in 0..self_checks {
        let id = dead_ids[j];
        let v = &fresh[j * dim..(j + 1) * dim];
        if ev.searcher.split(&ev.index, primary, v).first() == Some(&id) {
            self_hits += 1;
        }
    }
    eprintln!(
        "[mutation] ops={} leaks={} replace_self_hits={}/{} split del {:.0} ns/{} B, replace {:.1} us/{} B",
        ops, leaks, self_hits, self_checks, split_del_ns, split_del_bytes, split_rep_us, split_rep_bytes
    );
    Mutation {
        ops,
        leak_check_k: leak_k,
        deleted_leaks: leaks,
        replace_self_hits: self_hits,
        replace_self_checks: self_checks,
        split: MutReport {
            delete_bytes_touched: split_del_bytes,
            delete_ns_per_op: split_del_ns,
            replace_bytes_touched: split_rep_bytes,
            replace_us_per_op: split_rep_us,
        },
        int8: MutReport {
            delete_bytes_touched: int8_del_bytes,
            delete_ns_per_op: int8_del_ns,
            replace_bytes_touched: int8_rep_bytes,
            replace_us_per_op: int8_rep_us,
        },
        exact: MutReport {
            delete_bytes_touched: exact_del_bytes,
            delete_ns_per_op: exact_del_ns,
            replace_bytes_touched: exact_rep_bytes,
            replace_us_per_op: exact_rep_us,
        },
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.runs >= 1, "--runs must be >= 1");
    ensure!(!cli.ks.is_empty(), "--ks must list at least one K");
    ensure!(cli.top >= 1, "--top must be >= 1");
    ensure!(cli.n >= 2, "--n must be >= 2");
    ensure!(cli.queries >= 1, "--queries must be >= 1");
    ensure!(cli.calib >= 1, "--calib must be >= 1");
    if cli.sweep {
        ensure!(
            !cli.sweep_heads.is_empty() && !cli.sweep_mults.is_empty(),
            "--sweep-heads and --sweep-mults must be non-empty"
        );
    }
    let (basis_tag, mode_tag) = tag(cli.basis, cli.residual_mode);

    let t_load = Instant::now();
    let ds = build_dataset(&cli)?;
    let load_ms = ms_since(t_load);
    let dim = ds.dim;
    let n = ds.n_docs;
    let nq = ds.n_queries;
    ensure!(n >= 2 && nq >= 1, "dataset is empty");
    let ds_name = ds.name.clone();
    let ds_source = ds.source.clone();
    let n_real = ds.n_real_docs;
    let Dataset { docs, queries, .. } = ds;
    let mut primary = Primary::new(docs, dim);
    eprintln!(
        "[data] {} n={} dim={} queries={} load={:.1} ms",
        ds_name, n, dim, nq, load_ms
    );

    let calib_n = cli.calib.min(n);
    let calib_idx = Rng::new(cli.seed ^ 0xCA11_B000_0000_0001).sample_indices(n, calib_n);
    let mut calib = vec![0.0f32; calib_n * dim];
    for (dst, &i) in calib.chunks_exact_mut(dim).zip(&calib_idx) {
        dst.copy_from_slice(primary.row(i as usize));
    }

    let mut ks_sorted = cli.ks.clone();
    ks_sorted.sort_unstable();
    ks_sorted.dedup();
    let max_k = ks_sorted.iter().copied().max().unwrap_or(cli.top).max(cli.top);
    let mut base = Baseline::new(dim, max_k, cli.top);

    let t_gt = Instant::now();
    let mut gt: Vec<Vec<u32>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        gt.push(base.exact(&primary, &queries[qi * dim..(qi + 1) * dim]).to_vec());
    }
    let ground_truth_ms = ms_since(t_gt);
    eprintln!(
        "[gt] exact top-{} for {} queries in {:.1} ms",
        cli.top, nq, ground_truth_ms
    );

    let (mut int8, int8_report) = eval_int8(&cli, &mut base, &primary, &queries, &gt, &ks_sorted);
    let exact_report = eval_exact(&cli, &mut base, &primary, &queries, &gt);

    let dataset_info = DatasetInfo {
        kind: cli.dataset,
        name: ds_name.clone(),
        source: ds_source,
        n_docs: n,
        n_real_docs: n_real,
        dim,
        n_queries: nq,
        seed: cli.seed,
        load_ms,
        ground_truth_ms,
    };
    let shared = SharedConfig {
        basis: cli.basis,
        residual_mode: cli.residual_mode,
        ks: ks_sorted.clone(),
        runs: cli.runs,
        warmup_queries: cli.warmup.min(nq),
        rerank_top: cli.top,
        calib_vectors: calib_n,
        pca_iters: cli.pca_iters,
        threads: 1,
        target: target_description(),
    };
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");

    if cli.sweep {
        let int8_star = k_star(&int8_report.per_k, cli.target_recall, cli.threshold_max_k)
            .map(|r| r.k);
        let mut rows: Vec<SweepRow> = Vec::new();
        for &head in &cli.sweep_heads {
            for &mult in &cli.sweep_mults {
                let ev = eval_split(
                    &cli, head, mult, None, &primary, &queries, &gt, &ks_sorted, &calib,
                )?;
                let v = core_thresholds(&cli, n, &ev.per_k, &int8_report.per_k, ev.encode_ms);
                let pass = v.thresholds.iter().all(|t| t.pass);
                rows.push(SweepRow {
                    split_config: ev.config,
                    bytes_per_doc: ev.bytes_per_doc,
                    encode_ms: ev.encode_ms,
                    per_k: ev.per_k,
                    k_star: v.split_star.as_ref().map(|s| s.k),
                    qps_at_k_star: v.split_star.as_ref().map(|s| s.qps),
                    ratio_vs_int8_matched: v.ratio_matched,
                    thresholds: v.thresholds,
                    pass,
                });
            }
        }
        println!(
            "== sweep {} n={} dim={} basis={} mode={} target recall@{} >= {} at K <= {} ==",
            ds_name, n, dim, basis_tag, mode_tag, cli.top, cli.target_recall, cli.threshold_max_k
        );
        let int8_star_qps = int8_star
            .and_then(|k| int8_report.per_k.iter().find(|r| r.k == k))
            .map(|r| r.qps);
        println!(
            "int8 baseline: {} B/doc, K*={:?}, qps@K*={:?}; exact: {} B/doc, qps={:.0}",
            int8_report.bytes_per_doc,
            int8_star,
            int8_star_qps.map(|q| q.round()),
            exact_report.bytes_per_doc,
            exact_report.per_k[0].qps
        );
        let mut header = format!(
            "{:>4} {:>4} {:>5} {:>6} {:>5} {:>7}",
            "head", "mult", "bits", "B/doc", "hvar", "enc_ms"
        );
        for k in &ks_sorted {
            header.push_str(&format!(" | K={:<5} rec   qps", k));
        }
        header.push_str(&format!(" | {:>5} {:>7} {:>6} {}", "K*", "qps@K*", "ratio", "verdict"));
        println!("{header}");
        for r in &rows {
            let mut line = format!(
                "{:>4} {:>4} {:>5} {:>6} {:>5.3} {:>7.1}",
                r.split_config.head_dim,
                r.split_config.residual_mult,
                r.split_config.residual_bits,
                r.bytes_per_doc,
                r.split_config.head_variance_fraction,
                r.encode_ms
            );
            for kr in &r.per_k {
                line.push_str(&format!(" | {:<7} {:.4} {:>5.0}", "", kr.recall_at_top, kr.qps));
            }
            let kstar = r.k_star.map(|k| k.to_string()).unwrap_or_else(|| "-".to_string());
            let qstar = r
                .qps_at_k_star
                .map(|q| format!("{q:.0}"))
                .unwrap_or_else(|| "-".to_string());
            let ratio = r
                .ratio_vs_int8_matched
                .map(|x| format!("{x:.2}"))
                .unwrap_or_else(|| "-".to_string());
            line.push_str(&format!(
                " | {:>5} {:>7} {:>6} {}",
                kstar,
                qstar,
                ratio,
                if r.pass { "PASS" } else { "FAIL" }
            ));
            println!("{line}");
        }
        let any_pass = rows.iter().any(|r| r.pass);
        println!(
            "[{}] at least one head x mult configuration passes thresholds 1-4",
            if any_pass { "PASS" } else { "FAIL" }
        );
        let report = SweepReport {
            schema: SWEEP_SCHEMA,
            generated_unix_s: unix_now(),
            dataset: dataset_info,
            config: shared,
            int8: int8_report,
            exact: exact_report,
            int8_k_star: int8_star,
            rows,
            any_pass,
        };
        let out = cli.out.clone().unwrap_or_else(|| {
            out_dir.join(format!(
                "{}_{}_{}_{}_sweep.json",
                ds_name, n, basis_tag, mode_tag
            ))
        });
        return write_json(&out, &report);
    }

    let mut ev = eval_split(
        &cli,
        cli.head_dim,
        cli.residual_mult,
        cli.residual_bits,
        &primary,
        &queries,
        &gt,
        &ks_sorted,
        &calib,
    )?;
    drop(calib);
    let v = core_thresholds(&cli, n, &ev.per_k, &int8_report.per_k, ev.encode_ms);
    let ratio_exact = v
        .split_star
        .as_ref()
        .filter(|_| exact_report.per_k[0].qps > 0.0)
        .map(|s| s.qps / exact_report.per_k[0].qps);
    let ratio_at_equal_k: Vec<EqualK> = ev
        .per_k
        .iter()
        .zip(&int8_report.per_k)
        .map(|(s, i)| EqualK {
            k: s.k,
            split_recall: s.recall_at_top,
            int8_recall: i.recall_at_top,
            split_qps: s.qps,
            int8_qps: i.qps,
            ratio_split_vs_int8: if i.qps > 0.0 { s.qps / i.qps } else { 0.0 },
        })
        .collect();

    let mutation = mutation_experiment(&cli, &mut ev, &mut int8, &mut primary, &queries, max_k);

    let mut thresholds = v.thresholds.clone();
    thresholds.push(Threshold {
        name: "no deleted id returned by any method after delete".to_string(),
        value: mutation.deleted_leaks as f64,
        limit: 0.0,
        pass: mutation.deleted_leaks == 0,
    });
    if mutation.replace_self_checks > 0 {
        thresholds.push(Threshold {
            name: "replaced slots are found by their own new vector (split)".to_string(),
            value: mutation.replace_self_hits as f64,
            limit: mutation.replace_self_checks as f64,
            pass: mutation.replace_self_hits == mutation.replace_self_checks,
        });
    }
    let pass = thresholds.iter().all(|t| t.pass);

    println!(
        "== warm-research-scan {} n={} dim={} basis={} mode={} head={} mult={} bits={} ==",
        ds_name, n, dim, basis_tag, mode_tag, ev.config.head_dim, ev.config.residual_mult, ev.config.residual_bits
    );
    println!(
        "{:<8} {:>6} {:>10} {:>10} {:>10}",
        "method", "K", "recall", "qps", "us/query"
    );
    for r in &ev.per_k {
        println!(
            "{:<8} {:>6} {:>10.4} {:>10.0} {:>10.1}",
            "split", r.k, r.recall_at_top, r.qps, r.us_per_query
        );
    }
    for r in &int8_report.per_k {
        println!(
            "{:<8} {:>6} {:>10.4} {:>10.0} {:>10.1}",
            "int8", r.k, r.recall_at_top, r.qps, r.us_per_query
        );
    }
    let ex = &exact_report.per_k[0];
    let exact_qps = ex.qps;
    println!(
        "{:<8} {:>6} {:>10.4} {:>10.0} {:>10.1}",
        "exact", ex.k, ex.recall_at_top, ex.qps, ex.us_per_query
    );
    println!(
        "bytes/doc split={} int8={} exact={} | encode ms split={:.1} int8={:.1} | K* split={:?} int8={:?} | ratio matched={:?}",
        ev.bytes_per_doc,
        int8_report.bytes_per_doc,
        exact_report.bytes_per_doc,
        ev.encode_ms,
        int8_report.encode_ms,
        v.split_star.as_ref().map(|s| s.k),
        v.int8_star.as_ref().map(|s| s.k),
        v.ratio_matched
    );
    print_thresholds(&thresholds, pass);

    let report = Report {
        schema: SCHEMA,
        generated_unix_s: unix_now(),
        dataset: dataset_info,
        config: shared,
        split_config: ev.config,
        methods: Methods {
            split: MethodReport {
                bytes_per_doc: ev.bytes_per_doc,
                encode_ms: ev.encode_ms,
                encode_us_per_doc: ev.encode_ms * 1e3 / n as f64,
                per_k: ev.per_k,
            },
            int8: int8_report,
            exact: exact_report,
        },
        ablation: Ablation {
            target_recall: cli.target_recall,
            threshold_max_k: cli.threshold_max_k,
            split_k_star: v.split_star.as_ref().map(|s| s.k),
            int8_k_star: v.int8_star.as_ref().map(|s| s.k),
            split_qps_at_k_star: v.split_star.as_ref().map(|s| s.qps),
            int8_qps_at_k_star: v.int8_star.as_ref().map(|s| s.qps),
            exact_qps,
            ratio_split_vs_int8_matched: v.ratio_matched,
            ratio_split_vs_exact_matched: ratio_exact,
            ratio_at_equal_k,
        },
        mutation,
        thresholds,
        pass,
    };
    let out = cli.out.clone().unwrap_or_else(|| {
        out_dir.join(format!("{}_{}_{}_{}.json", ds_name, n, basis_tag, mode_tag))
    });
    write_json(&out, &report)
}
