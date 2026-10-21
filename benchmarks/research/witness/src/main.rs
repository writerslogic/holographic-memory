// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `hms-research-witness`: load an ann-benchmarks dataset, compute exact
//! ground truth for disjoint reporting / selection query sets, then
//!
//! * stage 1 (always): for every `M x B x Q x t` configuration measure
//!   rank-10 survival, candidate recall@10, emitted candidates, posting bytes
//!   streamed and projection+selection time, binned survival by true
//!   similarity, and print PASS/FAIL against the panel gate;
//! * stage 2 (`--timed`): for the configurations that passed stage 1 on the
//!   selection set (or `--config M,B,Q,t`, or `--timed-all`), rebuild the
//!   index for that single budget, time the complete single-thread query
//!   (projection, selection, join, bulk rerank) over the reporting queries
//!   (median of `--runs` passes after warmup) and report recall@10, QPS,
//!   mean / p95 latency, build time and index bytes.
//!
//! Timing is single-threaded; rayon (`--build-threads`) is used only in the
//! build. All randomness derives from `--seed`. JSON schemas: `RUN.md` and
//! `out/SCHEMA.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;

use hms_research_witness::bank::{Bank, SelectScratch, GRID_LEVELS};
use hms_research_witness::data::{self, AnnDataset, Rng};
use hms_research_witness::join::Joiner;
use hms_research_witness::postings::{self, Occupancy, Postings};
use hms_research_witness::{exact_topk, median, overlap, summarize, Index, QueryParams, QueryScratch, Summary};

const SCHEMA_STAGE1: &str = "hms-research-witness/stage1/v1";
const SCHEMA_STAGE2: &str = "hms-research-witness/stage2/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum DatasetKind {
    /// ann-benchmarks glove-100-angular (1,183,514 x 100).
    Glove,
    /// ann-benchmarks nytimes-256-angular (290,000 x 256).
    Nytimes,
}

impl DatasetKind {
    fn dir_name(self) -> &'static str {
        match self {
            DatasetKind::Glove => "glove-100-angular",
            DatasetKind::Nytimes => "nytimes-256-angular",
        }
    }

    fn short(self) -> &'static str {
        match self {
            DatasetKind::Glove => "glove100",
            DatasetKind::Nytimes => "nytimes256",
        }
    }
}

/// One `(M, B, Q, t)` configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct ConfigKey {
    m: usize,
    b: usize,
    q: usize,
    t: u8,
}

fn parse_config(s: &str) -> std::result::Result<ConfigKey, String> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        return Err(format!("expected M,B,Q,t, got `{s}`"));
    }
    let num = |i: usize, name: &str| -> std::result::Result<usize, String> {
        parts[i]
            .parse::<usize>()
            .map_err(|e| format!("{name} in `{s}`: {e}"))
    };
    let t = num(3, "t")?;
    if !(1..=4).contains(&t) {
        return Err(format!("t must be in 1..=4, got {t}"));
    }
    Ok(ConfigKey {
        m: num(0, "M")?,
        b: num(1, "B")?,
        q: num(2, "Q")?,
        t: t as u8,
    })
}

#[derive(Parser, Debug)]
#[command(
    name = "hms-research-witness",
    about = "Kill test: occupancy-balanced rare-witness decoding for large-N ANN"
)]
struct Cli {
    /// Dataset.
    #[arg(long, value_enum, default_value_t = DatasetKind::Glove)]
    dataset: DatasetKind,
    /// Dataset directory (default ~/.cache/hms-bench/<dataset>).
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// Corpus rows to load (prefix of train.f32); 0 = all.
    #[arg(long, default_value_t = 0)]
    n: usize,
    /// Reporting queries (headline numbers).
    #[arg(long, default_value_t = 1000)]
    report_queries: usize,
    /// Parameter-selection queries (disjoint from the reporting set).
    #[arg(long, default_value_t = 500)]
    select_queries: usize,
    /// Corpus rows sampled for the per-channel quantile fit.
    #[arg(long, default_value_t = 50_000)]
    sample: usize,
    /// Master seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Channel counts M (each a multiple of 2 * next_power_of_two(dim)).
    #[arg(long, value_delimiter = ',', default_values_t = [4096usize, 32768])]
    ms: Vec<usize>,
    /// Membership budgets B (expected memberships per vector).
    #[arg(long, value_delimiter = ',', default_values_t = [16usize, 24, 32])]
    bs: Vec<usize>,
    /// Channels selected per query Q.
    #[arg(long, value_delimiter = ',', default_values_t = [16usize, 32, 64])]
    qs: Vec<usize>,
    /// Witness thresholds t (1..=4).
    #[arg(long, value_delimiter = ',', default_values_t = [2u8, 3])]
    ts: Vec<u8>,
    /// Results per query (recall@top).
    #[arg(long, default_value_t = 10)]
    top: usize,
    /// Run stage 2 for the configurations that passed stage 1 on the selection set.
    #[arg(long)]
    timed: bool,
    /// Run stage 2 for every configuration regardless of stage 1.
    #[arg(long)]
    timed_all: bool,
    /// Run stage 2 for exactly this configuration (repeatable): M,B,Q,t.
    #[arg(long = "config", value_parser = parse_config)]
    configs: Vec<ConfigKey>,
    /// Timed passes over the reporting queries in stage 2; QPS uses the median.
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Warmup queries before timing.
    #[arg(long, default_value_t = 100)]
    warmup: usize,
    /// Threads for the build-time projection / quantile passes; 0 = all cores.
    #[arg(long, default_value_t = 0)]
    build_threads: usize,
    /// Rows per build window (bounds build memory).
    #[arg(long, default_value_t = 65_536)]
    window_rows: usize,
    /// Stage-1 JSON path (default benchmarks/research/witness/out/stage1_<dataset>_<n>.json).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Stage-2 JSON directory (default benchmarks/research/witness/out).
    #[arg(long)]
    out_dir: Option<PathBuf>,
    /// Gate 1: candidate recall@top.
    #[arg(long, default_value_t = 0.98)]
    min_candidate_recall: f64,
    /// Gate 1: rank-10 survival.
    #[arg(long, default_value_t = 0.95)]
    min_rank10_survival: f64,
    /// Gate 1: median emitted candidates.
    #[arg(long, default_value_t = 1000.0)]
    max_median_candidates: f64,
    /// Gate 1: p95 emitted candidates.
    #[arg(long, default_value_t = 5000.0)]
    max_p95_candidates: f64,
    /// Gate 1: median posting bytes streamed (KB).
    #[arg(long, default_value_t = 100.0)]
    max_median_posting_kb: f64,
    /// Gate 1: p95 posting bytes streamed (KB).
    #[arg(long, default_value_t = 250.0)]
    max_p95_posting_kb: f64,
    /// Gate 1: mean projection + selection time (us).
    #[arg(long, default_value_t = 20.0)]
    max_projsel_us: f64,
    /// Gate 2: recall@top of the complete query.
    #[arg(long, default_value_t = 0.95)]
    min_recall: f64,
    /// Gate 2: mean complete-query latency (us).
    #[arg(long, default_value_t = 70.0)]
    max_mean_query_us: f64,
}

#[derive(Serialize, Clone)]
struct Gate1 {
    min_candidate_recall: f64,
    min_rank10_survival: f64,
    max_median_candidates: f64,
    max_p95_candidates: f64,
    max_median_posting_kb: f64,
    max_p95_posting_kb: f64,
    max_projsel_us: f64,
}

impl Gate1 {
    fn evaluate(&self, s: &SetStats) -> Vec<String> {
        let mut f = Vec::new();
        if s.candidate_recall_at_top < self.min_candidate_recall {
            f.push(format!("cand_recall {:.4} < {}", s.candidate_recall_at_top, self.min_candidate_recall));
        }
        if s.rank10_survival < self.min_rank10_survival {
            f.push(format!("rank10_survival {:.4} < {}", s.rank10_survival, self.min_rank10_survival));
        }
        if s.candidates.median > self.max_median_candidates {
            f.push(format!("median_cands {:.0} > {}", s.candidates.median, self.max_median_candidates));
        }
        if s.candidates.p95 > self.max_p95_candidates {
            f.push(format!("p95_cands {:.0} > {}", s.candidates.p95, self.max_p95_candidates));
        }
        if s.posting_bytes.median > self.max_median_posting_kb * 1e3 {
            f.push(format!("median_posting_kb {:.1} > {}", s.posting_bytes.median / 1e3, self.max_median_posting_kb));
        }
        if s.posting_bytes.p95 > self.max_p95_posting_kb * 1e3 {
            f.push(format!("p95_posting_kb {:.1} > {}", s.posting_bytes.p95 / 1e3, self.max_p95_posting_kb));
        }
        if s.projection_selection_us.mean > self.max_projsel_us {
            f.push(format!("projsel_us {:.1} > {}", s.projection_selection_us.mean, self.max_projsel_us));
        }
        f
    }
}

#[derive(Serialize, Clone)]
struct Gate2 {
    min_recall: f64,
    max_mean_query_us: f64,
}

#[derive(Serialize, Clone)]
struct DatasetInfo {
    kind: DatasetKind,
    name: String,
    dir: String,
    n_docs: usize,
    n_train_total: usize,
    dim: usize,
    dim_pad: usize,
    n_report_queries: usize,
    n_select_queries: usize,
    sample_rows: usize,
    seed: u64,
    top: usize,
    load_ms: f64,
    ground_truth_ms: f64,
    /// Mean overlap of the recomputed top-`top` with the first `top` columns of
    /// `neighbors.i32` over the reporting queries (null when the file is absent
    /// or the corpus was truncated with `--n`).
    gt_file_agreement: Option<f64>,
    target: String,
}

#[derive(Serialize, Clone)]
struct SimBin {
    lo: f64,
    hi: f64,
    n: usize,
    rank10_survival: f64,
    all10_n: usize,
    all10_survival: f64,
}

#[derive(Serialize, Clone)]
struct SetStats {
    n_queries: usize,
    candidate_recall_at_top: f64,
    rank10_survival: f64,
    candidates: Summary,
    posting_bytes: Summary,
    projection_selection_us: Summary,
    mean_true_sim_rank10: f64,
    survival_by_sim: Vec<SimBin>,
}

#[derive(Serialize, Clone)]
struct ConfigResult {
    m: usize,
    b: usize,
    q: usize,
    t: u8,
    /// `Q * 4 * N * B / M`: the posting bytes a perfectly balanced bank streams.
    expected_posting_bytes: f64,
    report: SetStats,
    select: SetStats,
    fails_report: Vec<String>,
    fails_select: Vec<String>,
    pass_report: bool,
    pass_select: bool,
}

#[derive(Serialize, Clone)]
struct BudgetReport {
    b: usize,
    tail_p: f64,
    sample_rank: usize,
    total_memberships: usize,
    mean_memberships_per_vector: f64,
    posting_bytes: usize,
    occupancy: Occupancy,
}

#[derive(Serialize, Clone)]
struct BankReport {
    m: usize,
    directions: usize,
    blocks: usize,
    grid_levels: usize,
    table_bytes: usize,
    fit_ms: f64,
    build_all_budgets_ms: f64,
    build_threads: usize,
    budgets: Vec<BudgetReport>,
}

#[derive(Serialize, Clone)]
struct CurvePoint {
    m: usize,
    b: usize,
    q: usize,
    t: u8,
    median_posting_bytes: f64,
    p95_posting_bytes: f64,
    median_candidates: f64,
    candidate_recall_at_top: f64,
    rank10_survival: f64,
    pass: bool,
}

#[derive(Serialize)]
struct Stage1Report {
    schema: &'static str,
    generated_unix_s: u64,
    dataset: DatasetInfo,
    sweep: Sweep,
    gate: Gate1,
    banks: Vec<BankReport>,
    configs: Vec<ConfigResult>,
    /// Every configuration as a (posting bytes, candidate recall) point on
    /// the reporting set, sorted by median posting bytes.
    curve: Vec<CurvePoint>,
    /// The Pareto frontier of `curve` (fewer bytes and higher recall).
    pareto: Vec<CurvePoint>,
    n_pass_report: usize,
    n_pass_select: usize,
}

#[derive(Serialize, Clone)]
struct Sweep {
    ms: Vec<usize>,
    bs: Vec<usize>,
    qs: Vec<usize>,
    ts: Vec<u8>,
}

#[derive(Serialize)]
struct Stage2Build {
    fit_ms: f64,
    postings_ms: f64,
    total_ms: f64,
    threads: usize,
    total_memberships: usize,
    mean_memberships_per_vector: f64,
    occupancy: Occupancy,
}

#[derive(Serialize)]
struct Stage2Bytes {
    postings: usize,
    bank_tables: usize,
    thresholds: usize,
    rerank_rows: usize,
    total: usize,
}

#[derive(Serialize)]
struct Stage2Query {
    n_queries: usize,
    runs: usize,
    warmup_queries: usize,
    run_ms: Vec<f64>,
    median_run_ms: f64,
    qps: f64,
    mean_us: f64,
    latency_us: Summary,
    candidates: Summary,
    posting_bytes: Summary,
}

#[derive(Serialize)]
struct Threshold {
    name: String,
    value: f64,
    limit: f64,
    pass: bool,
}

#[derive(Serialize)]
struct Stage2Report {
    schema: &'static str,
    generated_unix_s: u64,
    dataset: DatasetInfo,
    config: ConfigKey,
    top: usize,
    build: Stage2Build,
    index_bytes: Stage2Bytes,
    query: Stage2Query,
    recall_at_top: f64,
    thresholds: Vec<Threshold>,
    pass: bool,
}

const NBINS: usize = 11;

fn sim_bin(s: f32) -> usize {
    if s < 0.0 {
        0
    } else {
        (1 + (s * 10.0).floor() as usize).min(NBINS - 1)
    }
}

fn bin_bounds(i: usize) -> (f64, f64) {
    if i == 0 {
        (-1.0, 0.0)
    } else {
        ((i - 1) as f64 / 10.0, i as f64 / 10.0)
    }
}

#[derive(Default, Clone, Copy)]
struct BinAcc {
    n: usize,
    surv: usize,
    all_n: usize,
    all_surv: usize,
}

struct Acc {
    t: u8,
    cands: Vec<f64>,
    recall_sum: f64,
    rank10: usize,
    bins: [BinAcc; NBINS],
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Row `i` of a row-major matrix.
fn row(m: &[f32], dim: usize, i: usize) -> &[f32] {
    &m[i * dim..(i + 1) * dim]
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
        .join("hms-bench")
        .join(sub)
}

fn target_description() -> String {
    let mut v = vec![std::env::consts::ARCH.to_string()];
    if cfg!(target_feature = "neon") {
        v.push("neon".to_string());
    }
    if cfg!(target_feature = "avx2") {
        v.push("avx2".to_string());
    }
    if cfg!(target_feature = "avx512f") {
        v.push("avx512f".to_string());
    }
    v.join(",")
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create output dir {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(value).context("serialize report")?;
    fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Overlap of recomputed truth with the shipped `neighbors.i32` rows.
fn file_agreement(nb: &[i32], file_k: usize, test_idx: &[u32], gt: &[Vec<(f32, u32)>], top: usize) -> f64 {
    if test_idx.is_empty() || file_k == 0 {
        return 0.0;
    }
    let k = top.min(file_k);
    let mut sum = 0.0f64;
    for (qi, &ti) in test_idx.iter().enumerate() {
        let row = &nb[ti as usize * file_k..ti as usize * file_k + k];
        let hit = gt[qi]
            .iter()
            .filter(|(_, id)| row.iter().any(|&f| f >= 0 && f as u32 == *id))
            .count();
        sum += hit as f64 / gt[qi].len().max(1) as f64;
    }
    sum / test_idx.len() as f64
}

/// Stage-1 statistics of one `(bank, postings, Q)` over a query set, one
/// `SetStats` per witness threshold in `ts`.
fn collect(
    bank: &Bank,
    postings: &Postings,
    joiner: &mut Joiner,
    queries: &[f32],
    dim: usize,
    gt: &[Vec<(f32, u32)>],
    q: usize,
    ts: &[u8],
    proj: &mut [f32],
    sel: &mut SelectScratch,
    chans: &mut Vec<u32>,
) -> Vec<SetStats> {
    let nq = queries.len() / dim;
    let mut accs: Vec<Acc> = ts
        .iter()
        .map(|&t| Acc {
            t,
            cands: Vec::with_capacity(nq),
            recall_sum: 0.0,
            rank10: 0,
            bins: [BinAcc::default(); NBINS],
        })
        .collect();
    let mut bytes: Vec<f64> = Vec::with_capacity(nq);
    let mut us: Vec<f64> = Vec::with_capacity(nq);
    let mut sim10_sum = 0.0f64;
    for qi in 0..nq {
        let qv = row(queries, dim, qi);
        let t0 = Instant::now();
        bank.project(qv, proj);
        bank.select(proj, q, sel, chans);
        us.push(t0.elapsed().as_secs_f64() * 1e6);
        let tally = joiner.join_tally(postings, &chans[..]);
        bytes.push(tally.posting_bytes as f64);
        let truth = &gt[qi];
        let last = truth.last().copied();
        if let Some((s10, _)) = last {
            sim10_sum += s10 as f64;
        }
        for acc in accs.iter_mut() {
            let t = acc.t;
            acc.cands.push(tally.reached[(t - 1) as usize] as f64);
            let mut hit = 0usize;
            for &(s, id) in truth {
                let ok = joiner.count(id) >= t;
                let bi = sim_bin(s);
                acc.bins[bi].all_n += 1;
                if ok {
                    hit += 1;
                    acc.bins[bi].all_surv += 1;
                }
            }
            if !truth.is_empty() {
                acc.recall_sum += hit as f64 / truth.len() as f64;
            }
            if let Some((s10, id10)) = last {
                let bi = sim_bin(s10);
                acc.bins[bi].n += 1;
                if joiner.count(id10) >= t {
                    acc.rank10 += 1;
                    acc.bins[bi].surv += 1;
                }
            }
        }
        joiner.reset(postings, &chans[..]);
    }
    let bytes_s = summarize(&mut bytes);
    let us_s = summarize(&mut us);
    let nqf = nq.max(1) as f64;
    accs.into_iter()
        .map(|mut acc| SetStats {
            n_queries: nq,
            candidate_recall_at_top: acc.recall_sum / nqf,
            rank10_survival: acc.rank10 as f64 / nqf,
            candidates: summarize(&mut acc.cands),
            posting_bytes: bytes_s,
            projection_selection_us: us_s,
            mean_true_sim_rank10: sim10_sum / nqf,
            survival_by_sim: acc
                .bins
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let (lo, hi) = bin_bounds(i);
                    SimBin {
                        lo,
                        hi,
                        n: b.n,
                        rank10_survival: if b.n > 0 { b.surv as f64 / b.n as f64 } else { 0.0 },
                        all10_n: b.all_n,
                        all10_survival: if b.all_n > 0 { b.all_surv as f64 / b.all_n as f64 } else { 0.0 },
                    }
                })
                .collect(),
        })
        .collect()
}

fn survival_line(s: &SetStats) -> String {
    s.survival_by_sim
        .iter()
        .filter(|b| b.n > 0)
        .map(|b| format!("[{:.1},{:.1}):{:.3}(n={})", b.lo, b.hi, b.rank10_survival, b.n))
        .collect::<Vec<_>>()
        .join(" ")
}

fn pareto(points: &[CurvePoint]) -> Vec<CurvePoint> {
    points
        .iter()
        .filter(|p| {
            !points.iter().any(|o| {
                o.median_posting_bytes <= p.median_posting_bytes
                    && o.candidate_recall_at_top >= p.candidate_recall_at_top
                    && (o.median_posting_bytes < p.median_posting_bytes
                        || o.candidate_recall_at_top > p.candidate_recall_at_top)
            })
        })
        .cloned()
        .collect()
}

/// Stage 2 for one configuration: rebuild the single-budget index, time the
/// complete query over the reporting set.
fn stage2(
    cli: &Cli,
    ds: &AnnDataset,
    info: &DatasetInfo,
    sample: &[f32],
    report_q: &[f32],
    gt: &[Vec<(f32, u32)>],
    key: ConfigKey,
    pool: &rayon::ThreadPool,
    gate: &Gate2,
) -> Result<Stage2Report> {
    let dim = ds.dim;
    let n = ds.n_docs;
    let nq = report_q.len() / dim;
    let t_fit = Instant::now();
    let mut bank = Bank::new(key.m, dim, cli.seed)?;
    let mut taus = bank.fit(sample, &[key.b], pool)?;
    let fit_ms = ms_since(t_fit);
    let t_build = Instant::now();
    let mut built = postings::build(&bank, &taus, &ds.docs, n, pool, cli.window_rows)?;
    let postings_ms = ms_since(t_build);
    let postings = built.remove(0);
    let tau = taus.remove(0).tau;
    let occupancy = postings.occupancy();
    let total_memberships = postings.len();
    let mut index = Index {
        bank,
        b: key.b,
        tau,
        postings,
        joiner: Joiner::new(n),
    };
    let params = QueryParams {
        q: key.q,
        t: key.t,
        top: cli.top,
    };
    let mut scratch = QueryScratch::new(&index.bank);
    let mut out: Vec<(f32, u32)> = Vec::with_capacity(cli.top);

    let mut hits = 0.0f64;
    let mut cands: Vec<f64> = Vec::with_capacity(nq);
    let mut bytes: Vec<f64> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let tr = index.query(&ds.docs, row(report_q, dim, qi), &params, &mut scratch, &mut out);
        hits += overlap(&out, &gt[qi]);
        cands.push(tr.candidates as f64);
        bytes.push(tr.posting_bytes as f64);
    }
    let recall = hits / nq.max(1) as f64;
    let warmup = cli.warmup.min(nq);
    for qi in 0..warmup {
        index.query(&ds.docs, row(report_q, dim, qi), &params, &mut scratch, &mut out);
        black_box(out.len());
    }
    let mut run_ms = Vec::with_capacity(cli.runs);
    for _ in 0..cli.runs {
        let t = Instant::now();
        for qi in 0..nq {
            index.query(&ds.docs, row(report_q, dim, qi), &params, &mut scratch, &mut out);
            black_box(out.len());
        }
        run_ms.push(ms_since(t));
    }
    let mut lat: Vec<f64> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let t = Instant::now();
        index.query(&ds.docs, row(report_q, dim, qi), &params, &mut scratch, &mut out);
        black_box(out.len());
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let med = median(&run_ms);
    let qps = if med > 0.0 { nq as f64 / (med / 1e3) } else { 0.0 };
    let mean_us = if nq > 0 { med * 1e3 / nq as f64 } else { 0.0 };
    let latency = summarize(&mut lat);
    let cands_s = summarize(&mut cands);
    let bytes_s = summarize(&mut bytes);
    let rerank_rows = n * dim * 4;
    let index_bytes = Stage2Bytes {
        postings: index.postings.bytes(),
        bank_tables: index.bank.table_bytes(),
        thresholds: index.tau.len() * 4,
        rerank_rows,
        total: index.index_bytes() + rerank_rows,
    };
    let thresholds = vec![
        Threshold {
            name: format!("recall@{} >= {}", cli.top, gate.min_recall),
            value: recall,
            limit: gate.min_recall,
            pass: recall >= gate.min_recall,
        },
        Threshold {
            name: format!("mean complete-query latency <= {} us", gate.max_mean_query_us),
            value: mean_us,
            limit: gate.max_mean_query_us,
            pass: mean_us <= gate.max_mean_query_us,
        },
    ];
    let pass = thresholds.iter().all(|t| t.pass);
    println!(
        "[stage2] M={} B={} Q={} t={} | recall@{}={:.4} qps={:.0} mean={:.1} us p50={:.1} us p95={:.1} us | cands p50={:.0} p95={:.0} | posting KB p50={:.1} p95={:.1} | build fit={:.0} ms postings={:.0} ms ({} threads) | index {:.1} MB (+{:.1} MB rerank rows) | memberships/vec={:.2} | {}",
        key.m,
        key.b,
        key.q,
        key.t,
        cli.top,
        recall,
        qps,
        mean_us,
        latency.median,
        latency.p95,
        cands_s.median,
        cands_s.p95,
        bytes_s.median / 1e3,
        bytes_s.p95 / 1e3,
        fit_ms,
        postings_ms,
        pool.current_num_threads(),
        index.index_bytes() as f64 / 1e6,
        rerank_rows as f64 / 1e6,
        total_memberships as f64 / n as f64,
        if pass { "PASS" } else { "FAIL" }
    );
    for t in &thresholds {
        println!(
            "[stage2] [{}] {} (value={:.4}, limit={:.4})",
            if t.pass { "PASS" } else { "FAIL" },
            t.name,
            t.value,
            t.limit
        );
    }
    Ok(Stage2Report {
        schema: SCHEMA_STAGE2,
        generated_unix_s: unix_now(),
        dataset: info.clone(),
        config: key,
        top: cli.top,
        build: Stage2Build {
            fit_ms,
            postings_ms,
            total_ms: fit_ms + postings_ms,
            threads: pool.current_num_threads(),
            total_memberships,
            mean_memberships_per_vector: total_memberships as f64 / n as f64,
            occupancy,
        },
        index_bytes,
        query: Stage2Query {
            n_queries: nq,
            runs: cli.runs,
            warmup_queries: warmup,
            run_ms,
            median_run_ms: med,
            qps,
            mean_us,
            latency_us: latency,
            candidates: cands_s,
            posting_bytes: bytes_s,
        },
        recall_at_top: recall,
        thresholds,
        pass,
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.top >= 1, "--top must be >= 1");
    ensure!(cli.runs >= 1, "--runs must be >= 1");
    ensure!(!cli.ms.is_empty() && !cli.bs.is_empty() && !cli.qs.is_empty() && !cli.ts.is_empty(), "empty sweep");
    for &t in &cli.ts {
        ensure!((1..=4).contains(&t), "--ts entries must be in 1..=4");
    }
    for &q in &cli.qs {
        ensure!((1..=255).contains(&q), "--qs entries must be in 1..=255");
        for &m in &cli.ms {
            ensure!(q <= m / 2, "Q={} exceeds the {} directions of M={}", q, m / 2, m);
        }
    }
    for &b in &cli.bs {
        for &m in &cli.ms {
            ensure!(b >= 1 && b < m, "B={} must be in [1, M={})", b, m);
        }
    }
    for c in &cli.configs {
        ensure!(c.q >= 1 && c.q <= 255 && c.q <= c.m / 2 && c.b >= 1 && c.b < c.m, "bad --config {:?}", c);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cli.build_threads)
        .build()
        .context("build rayon pool")?;
    let gate1 = Gate1 {
        min_candidate_recall: cli.min_candidate_recall,
        min_rank10_survival: cli.min_rank10_survival,
        max_median_candidates: cli.max_median_candidates,
        max_p95_candidates: cli.max_p95_candidates,
        max_median_posting_kb: cli.max_median_posting_kb,
        max_p95_posting_kb: cli.max_p95_posting_kb,
        max_projsel_us: cli.max_projsel_us,
    };
    let gate2 = Gate2 {
        min_recall: cli.min_recall,
        max_mean_query_us: cli.max_mean_query_us,
    };

    let dir = cli
        .data_dir
        .clone()
        .unwrap_or_else(|| home_cache(cli.dataset.dir_name()));
    let t_load = Instant::now();
    let ds = data::load_ann(&dir, cli.dataset.short(), cli.n)?;
    let load_ms = ms_since(t_load);
    let dim = ds.dim;
    let n = ds.n_docs;
    ensure!(n >= 2, "corpus too small");
    eprintln!(
        "[data] {} dir={} n={} (of {}) dim={} test={} neighbors.i32={} load={:.0} ms",
        ds.name,
        ds.dir,
        n,
        ds.n_train_total,
        dim,
        ds.n_test,
        if ds.file_neighbors.is_some() { format!("{} cols", ds.file_k) } else { "absent".to_string() },
        load_ms
    );
    let split = data::split_queries(ds.n_test, cli.report_queries, cli.select_queries, cli.seed)?;
    let report_q = data::gather_rows(&ds.test, dim, &split.report);
    let select_q = data::gather_rows(&ds.test, dim, &split.select);
    let t_gt = Instant::now();
    let gt_report = exact_topk(&ds.docs, dim, n, &report_q, cli.top);
    let gt_select = exact_topk(&ds.docs, dim, n, &select_q, cli.top);
    let ground_truth_ms = ms_since(t_gt);
    let gt_file_agreement = ds
        .file_neighbors
        .as_ref()
        .filter(|_| n == ds.n_train_total)
        .map(|nb| file_agreement(nb, ds.file_k, &split.report, &gt_report, cli.top));
    eprintln!(
        "[gt] exact top-{} for {} report + {} select queries in {:.0} ms; agreement with neighbors.i32: {}",
        cli.top,
        split.report.len(),
        split.select.len(),
        ground_truth_ms,
        gt_file_agreement.map(|a| format!("{a:.4}")).unwrap_or_else(|| "n/a".to_string())
    );

    let n_sample = cli.sample.min(n);
    let mut picks = Rng::new(cli.seed ^ 0x5A4D_504C_4500_0000).sample_indices(n, n_sample);
    picks.sort_unstable();
    let sample = data::gather_rows(&ds.docs, dim, &picks);
    let info = DatasetInfo {
        kind: cli.dataset,
        name: ds.name.clone(),
        dir: ds.dir.clone(),
        n_docs: n,
        n_train_total: ds.n_train_total,
        dim,
        dim_pad: dim.next_power_of_two().max(2),
        n_report_queries: split.report.len(),
        n_select_queries: split.select.len(),
        sample_rows: n_sample,
        seed: cli.seed,
        top: cli.top,
        load_ms,
        ground_truth_ms,
        gt_file_agreement,
        target: target_description(),
    };

    let mut joiner = Joiner::new(n);
    let mut results: Vec<ConfigResult> = Vec::new();
    let mut banks: Vec<BankReport> = Vec::new();
    println!("== hms-research-witness stage 1: {} n={} dim={} report={} select={} sample={} ==", ds.name, n, dim, split.report.len(), split.select.len(), n_sample);
    println!(
        "{:>6} {:>3} {:>3} {:>2} | {:>8} {:>8} | {:>7} {:>7} | {:>8} {:>8} | {:>8} | {:>7} | verdict (reporting set; [sel] = selection-set verdict)",
        "M", "B", "Q", "t", "cand_rec", "r10_surv", "cand_50", "cand_95", "postKB50", "postKB95", "projselu", "exp_KB"
    );
    for &m in &cli.ms {
        let mut bank = Bank::new(m, dim, cli.seed)?;
        let t_fit = Instant::now();
        let taus = bank.fit(&sample, &cli.bs, &pool)?;
        let fit_ms = ms_since(t_fit);
        let t_build = Instant::now();
        let built = postings::build(&bank, &taus, &ds.docs, n, &pool, cli.window_rows)?;
        let build_ms = ms_since(t_build);
        let mut budgets = Vec::with_capacity(cli.bs.len());
        for (bi, p) in built.iter().enumerate() {
            let occ = p.occupancy();
            if taus[bi].sample_rank < 8 {
                eprintln!(
                    "[warn] M={} B={}: membership quantile rests on only {} sample rows per channel; raise --sample",
                    m, taus[bi].b, taus[bi].sample_rank
                );
            }
            eprintln!(
                "[bank] M={} B={} tail_p={:.2e} rank={} | memberships/vec={:.3} | occupancy min={} p05={} p50={} p95={} max={} empty={} | postings {:.1} MB",
                m,
                taus[bi].b,
                taus[bi].tail_p,
                taus[bi].sample_rank,
                p.len() as f64 / n as f64,
                occ.min,
                occ.p05,
                occ.p50,
                occ.p95,
                occ.max,
                occ.empty,
                p.bytes() as f64 / 1e6
            );
            budgets.push(BudgetReport {
                b: taus[bi].b,
                tail_p: taus[bi].tail_p,
                sample_rank: taus[bi].sample_rank,
                total_memberships: p.len(),
                mean_memberships_per_vector: p.len() as f64 / n as f64,
                posting_bytes: p.bytes(),
                occupancy: occ,
            });
        }
        eprintln!(
            "[bank] M={} dirs={} blocks={} fit={:.0} ms build(all B)={:.0} ms on {} threads",
            m, bank.dirs, bank.blocks, fit_ms, build_ms, pool.current_num_threads()
        );
        banks.push(BankReport {
            m,
            directions: bank.dirs,
            blocks: bank.blocks,
            grid_levels: GRID_LEVELS,
            table_bytes: bank.table_bytes(),
            fit_ms,
            build_all_budgets_ms: build_ms,
            build_threads: pool.current_num_threads(),
            budgets,
        });
        let mut proj = vec![0.0f32; bank.dirs];
        let mut sel = SelectScratch::default();
        let mut chans: Vec<u32> = Vec::with_capacity(256);
        for (bi, &b) in cli.bs.iter().enumerate() {
            for &q in &cli.qs {
                let rep = collect(&bank, &built[bi], &mut joiner, &report_q, dim, &gt_report, q, &cli.ts, &mut proj, &mut sel, &mut chans);
                let selp = collect(&bank, &built[bi], &mut joiner, &select_q, dim, &gt_select, q, &cli.ts, &mut proj, &mut sel, &mut chans);
                for (ti, &t) in cli.ts.iter().enumerate() {
                    let r = &rep[ti];
                    let s = &selp[ti];
                    let fails_report = gate1.evaluate(r);
                    let fails_select = gate1.evaluate(s);
                    let expected = (q * b) as f64 * n as f64 / m as f64 * 4.0;
                    println!(
                        "{:>6} {:>3} {:>3} {:>2} | {:>8.4} {:>8.4} | {:>7.0} {:>7.0} | {:>8.1} {:>8.1} | {:>8.1} | {:>7.1} | {} [sel {}]{}",
                        m,
                        b,
                        q,
                        t,
                        r.candidate_recall_at_top,
                        r.rank10_survival,
                        r.candidates.median,
                        r.candidates.p95,
                        r.posting_bytes.median / 1e3,
                        r.posting_bytes.p95 / 1e3,
                        r.projection_selection_us.mean,
                        expected / 1e3,
                        if fails_report.is_empty() { "PASS" } else { "FAIL" },
                        if fails_select.is_empty() { "PASS" } else { "FAIL" },
                        if fails_report.is_empty() { String::new() } else { format!(" {}", fails_report.join("; ")) }
                    );
                    println!("         rank10 survival by true sim: {}", survival_line(r));
                    results.push(ConfigResult {
                        m,
                        b,
                        q,
                        t,
                        expected_posting_bytes: expected,
                        report: r.clone(),
                        select: s.clone(),
                        pass_report: fails_report.is_empty(),
                        pass_select: fails_select.is_empty(),
                        fails_report,
                        fails_select,
                    });
                }
            }
        }
    }

    let mut curve: Vec<CurvePoint> = results
        .iter()
        .map(|r| CurvePoint {
            m: r.m,
            b: r.b,
            q: r.q,
            t: r.t,
            median_posting_bytes: r.report.posting_bytes.median,
            p95_posting_bytes: r.report.posting_bytes.p95,
            median_candidates: r.report.candidates.median,
            candidate_recall_at_top: r.report.candidate_recall_at_top,
            rank10_survival: r.report.rank10_survival,
            pass: r.pass_report,
        })
        .collect();
    curve.sort_by(|a, b| {
        a.median_posting_bytes
            .total_cmp(&b.median_posting_bytes)
            .then(b.candidate_recall_at_top.total_cmp(&a.candidate_recall_at_top))
    });
    let frontier = pareto(&curve);
    let n_pass_report = results.iter().filter(|r| r.pass_report).count();
    let n_pass_select = results.iter().filter(|r| r.pass_select).count();
    println!(
        "[stage1] {} of {} configurations pass on the reporting set, {} on the selection set",
        n_pass_report,
        results.len(),
        n_pass_select
    );
    println!("[stage1] recall-vs-posting-bytes Pareto frontier (reporting set):");
    for p in &frontier {
        println!(
            "         M={} B={} Q={} t={} | median {:.1} KB | cand_recall {:.4} | r10_surv {:.4} | cands {:.0} | {}",
            p.m, p.b, p.q, p.t, p.median_posting_bytes / 1e3, p.candidate_recall_at_top, p.rank10_survival, p.median_candidates,
            if p.pass { "PASS" } else { "FAIL" }
        );
    }
    let out_dir = cli
        .out_dir
        .clone()
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("out"));
    let stage1_path = cli
        .out
        .clone()
        .unwrap_or_else(|| out_dir.join(format!("stage1_{}_{}.json", ds.name, n)));
    let stage1 = Stage1Report {
        schema: SCHEMA_STAGE1,
        generated_unix_s: unix_now(),
        dataset: info.clone(),
        sweep: Sweep {
            ms: cli.ms.clone(),
            bs: cli.bs.clone(),
            qs: cli.qs.clone(),
            ts: cli.ts.clone(),
        },
        gate: gate1.clone(),
        banks,
        configs: results.clone(),
        curve,
        pareto: frontier,
        n_pass_report,
        n_pass_select,
    };
    write_json(&stage1_path, &stage1)?;

    if !(cli.timed || cli.timed_all || !cli.configs.is_empty()) {
        return Ok(());
    }
    let chosen: Vec<ConfigKey> = if !cli.configs.is_empty() {
        cli.configs.clone()
    } else {
        let mut v: Vec<&ConfigResult> = results
            .iter()
            .filter(|r| cli.timed_all || r.pass_select)
            .collect();
        v.sort_by(|a, b| {
            a.select
                .posting_bytes
                .median
                .total_cmp(&b.select.posting_bytes.median)
                .then(a.select.candidates.median.total_cmp(&b.select.candidates.median))
        });
        v.iter()
            .map(|r| ConfigKey {
                m: r.m,
                b: r.b,
                q: r.q,
                t: r.t,
            })
            .collect()
    };
    if chosen.is_empty() {
        println!("[stage2] skipped: no configuration passed stage 1 on the selection set (use --timed-all or --config M,B,Q,t to force)");
        return Ok(());
    }
    println!("== stage 2: {} configuration(s), {} runs x {} reporting queries, warmup {} ==", chosen.len(), cli.runs, split.report.len(), cli.warmup);
    for key in chosen {
        let rep = stage2(&cli, &ds, &info, &sample, &report_q, &gt_report, key, &pool, &gate2)?;
        let path = out_dir.join(format!(
            "stage2_{}_{}_m{}_b{}_q{}_t{}.json",
            ds.name, n, key.m, key.b, key.q, key.t
        ));
        write_json(&path, &rep)?;
    }
    Ok(())
}
