// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `hms-research-scan`: build a dataset, freeze the public basis, encode the
//! split-manifold / int8 / exact representations, and measure recall@10,
//! single-thread QPS (median of `--runs` passes over all queries after a
//! warmup), encode time, bytes per document, and delete/replace bytes
//! touched. Prints PASS/FAIL per panel threshold and writes one JSON report
//! (schema in `RUN.md`).
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

use hms_research_scan::codes::{
    fit_basis, BasisKind, BasisParams, Int8Index, Primary, ResidualMode, Scratch, SplitIndex,
};
use hms_research_scan::data::{self, Dataset, Rng, SyntheticParams};
use hms_research_scan::scan::Searcher;
use hms_research_scan::{median, HEAD_DIM};

const SCHEMA: &str = "hms-research-scan/v1";

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
    name = "hms-research-scan",
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
    /// Residual evaluation: 1-bit query (XOR/popcount) or 4-bit query planes.
    #[arg(long, value_enum, default_value_t = ResidualMode::Sym1)]
    residual_mode: ResidualMode,
    /// Residual bits B (multiple of 64, <= padded dim; default = dim rounded up to 64).
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
    /// NFCorpus directory (default ~/.cache/hms-bench/nfcorpus).
    #[arg(long)]
    nfcorpus_dir: Option<PathBuf>,
    /// glove-100-angular directory (default ~/.cache/hms-bench/glove-100-angular).
    #[arg(long)]
    glove_dir: Option<PathBuf>,
    /// NFCorpus distractor perturbation norm.
    #[arg(long, default_value_t = 0.6)]
    distractor_noise: f32,
    /// Replacement-vector perturbation norm in the mutation experiment.
    #[arg(long, default_value_t = 0.3)]
    replace_noise: f32,
    /// Output JSON path (default benchmarks/research/scan/out/<dataset>_<n>_<basis>_<mode>.json).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Threshold: split recall@top at some K <= 256.
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
struct ConfigInfo {
    basis: BasisKind,
    head_dim: usize,
    residual_bits: usize,
    residual_words: usize,
    padded_dim: usize,
    residual_mode: ResidualMode,
    ks: Vec<usize>,
    runs: usize,
    warmup_queries: usize,
    rerank_top: usize,
    calib_vectors: usize,
    pca_iters: usize,
    head_variance_fraction: f32,
    threads: usize,
    target: String,
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

#[derive(Serialize)]
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
    config: ConfigInfo,
    basis_fit_ms: f64,
    methods: Methods,
    ablation: Ablation,
    mutation: Mutation,
    thresholds: Vec<Threshold>,
    pass: bool,
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn home_cache(sub: &str) -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("hms-bench")
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.runs >= 1, "--runs must be >= 1");
    ensure!(!cli.ks.is_empty(), "--ks must list at least one K");
    ensure!(cli.top >= 1, "--top must be >= 1");
    ensure!(cli.n >= 2, "--n must be >= 2");
    ensure!(cli.queries >= 1, "--queries must be >= 1");
    ensure!(cli.calib >= 1, "--calib must be >= 1");
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
    let query = |qi: usize| -> &[f32] { &queries[qi * dim..(qi + 1) * dim] };
    let mut primary = Primary::new(docs, dim);
    eprintln!(
        "[data] {} n={} dim={} queries={} load={:.1} ms",
        ds_name, n, dim, nq, load_ms
    );

    let residual_bits = cli.residual_bits.unwrap_or_else(|| {
        let padded = dim.next_power_of_two().max(64);
        (dim.div_ceil(64).max(1) * 64).min(padded)
    });
    let calib_n = cli.calib.min(n);
    let calib_idx = Rng::new(cli.seed ^ 0xCA11_B000_0000_0001).sample_indices(n, calib_n);
    let mut calib = vec![0.0f32; calib_n * dim];
    for (dst, &i) in calib.chunks_exact_mut(dim).zip(&calib_idx) {
        dst.copy_from_slice(primary.row(i as usize));
    }
    let t_fit = Instant::now();
    let basis = fit_basis(
        &BasisParams {
            kind: cli.basis,
            dim,
            residual_bits,
            seed: cli.seed,
            pca_iters: cli.pca_iters,
        },
        &calib,
    )?;
    let basis_fit_ms = ms_since(t_fit);
    drop(calib);
    eprintln!(
        "[basis] kind={} bits={} words={} padded={} head_var_frac={:.3} fit={:.1} ms",
        basis_tag, basis.bits, basis.words, basis.padded, basis.head_variance_fraction, basis_fit_ms
    );

    let max_k = *cli.ks.iter().max().unwrap_or(&cli.top);
    let mut srch = Searcher::new(&basis, cli.residual_mode, max_k.max(cli.top), cli.top);

    let t_gt = Instant::now();
    let mut gt: Vec<Vec<u32>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        gt.push(srch.exact(&primary, query(qi)).to_vec());
    }
    let ground_truth_ms = ms_since(t_gt);
    eprintln!("[gt] exact top-{} for {} queries in {:.1} ms", cli.top, nq, ground_truth_ms);

    let mut split = SplitIndex::with_capacity(basis, n, cli.residual_mode);
    let mut scratch = Scratch::new(&split.basis);
    let t_enc = Instant::now();
    for i in 0..n {
        split.encode(i, primary.row(i), &mut scratch);
    }
    let split_encode_ms = ms_since(t_enc);
    let mut int8 = Int8Index::with_capacity(n, dim);
    let t_enc = Instant::now();
    for i in 0..n {
        int8.encode(i, primary.row(i));
    }
    let int8_encode_ms = ms_since(t_enc);
    let t_enc = Instant::now();
    let copy = primary.rows.clone();
    let exact_encode_ms = ms_since(t_enc);
    black_box(copy.len());
    drop(copy);
    eprintln!(
        "[encode] split={:.1} ms ({} B/doc) int8={:.1} ms ({} B/doc) exact copy={:.1} ms ({} B/doc)",
        split_encode_ms,
        split.bytes_per_doc(),
        int8_encode_ms,
        int8.bytes_per_doc(),
        exact_encode_ms,
        primary.bytes_per_doc()
    );

    let mut split_k: Vec<KReport> = Vec::with_capacity(cli.ks.len());
    let mut int8_k: Vec<KReport> = Vec::with_capacity(cli.ks.len());
    let mut ks_sorted = cli.ks.clone();
    ks_sorted.sort_unstable();
    ks_sorted.dedup();
    for &k in &ks_sorted {
        srch.set_k(k);
        let (rec, run_ms) = measure(nq, cli.warmup, cli.runs, &gt, &mut |qi, out| {
            let r = srch.split(&split, &primary, query(qi));
            out.clear();
            out.extend_from_slice(r);
        });
        let rep = k_report(k, nq, rec, run_ms);
        eprintln!(
            "[split] K={:<5} recall@{}={:.4} qps={:.0} ({:.1} us/q)",
            k, cli.top, rep.recall_at_top, rep.qps, rep.us_per_query
        );
        split_k.push(rep);

        let (rec, run_ms) = measure(nq, cli.warmup, cli.runs, &gt, &mut |qi, out| {
            let r = srch.int8(&int8, &primary, query(qi));
            out.clear();
            out.extend_from_slice(r);
        });
        let rep = k_report(k, nq, rec, run_ms);
        eprintln!(
            "[int8]  K={:<5} recall@{}={:.4} qps={:.0} ({:.1} us/q)",
            k, cli.top, rep.recall_at_top, rep.qps, rep.us_per_query
        );
        int8_k.push(rep);
    }
    let (rec, run_ms) = measure(nq, cli.warmup, cli.runs, &gt, &mut |qi, out| {
        let r = srch.exact(&primary, query(qi));
        out.clear();
        out.extend_from_slice(r);
    });
    let exact_rep = k_report(cli.top, nq, rec, run_ms);
    eprintln!(
        "[exact] recall@{}={:.4} qps={:.0} ({:.1} us/q)",
        cli.top, exact_rep.recall_at_top, exact_rep.qps, exact_rep.us_per_query
    );

    let split_star = k_star(&split_k, cli.target_recall, cli.threshold_max_k).cloned();
    let int8_star = k_star(&int8_k, cli.target_recall, cli.threshold_max_k).cloned();
    let ratio_matched = match (&split_star, &int8_star) {
        (Some(s), Some(i)) if i.qps > 0.0 => Some(s.qps / i.qps),
        _ => None,
    };
    let ratio_exact = split_star
        .as_ref()
        .filter(|_| exact_rep.qps > 0.0)
        .map(|s| s.qps / exact_rep.qps);
    let ratio_at_equal_k: Vec<EqualK> = split_k
        .iter()
        .zip(&int8_k)
        .map(|(s, i)| EqualK {
            k: s.k,
            split_recall: s.recall_at_top,
            int8_recall: i.recall_at_top,
            split_qps: s.qps,
            int8_qps: i.qps,
            ratio_split_vs_int8: if i.qps > 0.0 { s.qps / i.qps } else { 0.0 },
        })
        .collect();

    // Mutation experiment: O(1) slot deletes and replaces, bytes touched per op.
    let ops = cli.deletes.min(n / 2);
    let mut mrng = Rng::new(cli.seed ^ 0x0DE1_E7E0_0000_0001);
    let dead_ids = mrng.sample_indices(n, ops);
    let dead_set: HashSet<u32> = dead_ids.iter().copied().collect();
    let time_ops = |t: Instant, ops: usize, scale: f64| -> f64 {
        if ops == 0 {
            0.0
        } else {
            ms_since(t) * scale / ops as f64
        }
    };
    let t = Instant::now();
    let mut split_del_bytes = 0usize;
    for &id in &dead_ids {
        split_del_bytes = split.delete(id as usize) + primary.delete(id as usize);
    }
    let split_del_ns = time_ops(t, ops, 1e6);
    let t = Instant::now();
    let mut int8_del_bytes = 0usize;
    for &id in &dead_ids {
        int8_del_bytes = int8.delete(id as usize) + primary.delete(id as usize);
    }
    let int8_del_ns = time_ops(t, ops, 1e6);
    let t = Instant::now();
    let mut exact_del_bytes = 0usize;
    for &id in &dead_ids {
        exact_del_bytes = primary.delete(id as usize);
    }
    let exact_del_ns = time_ops(t, ops, 1e6);

    let leak_k = max_k.min(n);
    srch.set_k(leak_k);
    let mut leaks = 0usize;
    for qi in 0..nq {
        let q = query(qi);
        leaks += srch
            .split(&split, &primary, q)
            .iter()
            .filter(|id| dead_set.contains(*id))
            .count();
        leaks += srch
            .int8(&int8, &primary, q)
            .iter()
            .filter(|id| dead_set.contains(*id))
            .count();
        leaks += srch
            .exact(&primary, q)
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
        split.encode(id as usize, v, &mut scratch);
        primary.replace(id as usize, v);
    }
    let split_rep_us = time_ops(t, ops, 1e3);
    let split_rep_bytes = split.bytes_per_doc() + primary.bytes_per_doc();
    let t = Instant::now();
    for (j, &id) in dead_ids.iter().enumerate() {
        let v = &fresh[j * dim..(j + 1) * dim];
        int8.encode(id as usize, v);
        primary.replace(id as usize, v);
    }
    let int8_rep_us = time_ops(t, ops, 1e3);
    let int8_rep_bytes = int8.bytes_per_doc() + primary.bytes_per_doc();
    let t = Instant::now();
    for (j, &id) in dead_ids.iter().enumerate() {
        primary.replace(id as usize, &fresh[j * dim..(j + 1) * dim]);
    }
    let exact_rep_us = time_ops(t, ops, 1e3);
    let exact_rep_bytes = primary.bytes_per_doc();

    let self_checks = ops.min(50);
    let mut self_hits = 0usize;
    srch.set_k(cli.ks.iter().copied().min().unwrap_or(cli.top).max(cli.top));
    for j in 0..self_checks {
        let id = dead_ids[j];
        let v = &fresh[j * dim..(j + 1) * dim];
        if srch.split(&split, &primary, v).first() == Some(&id) {
            self_hits += 1;
        }
    }
    eprintln!(
        "[mutation] ops={} leaks={} replace_self_hits={}/{} split del {:.0} ns/{} B, replace {:.1} us/{} B",
        ops, leaks, self_hits, self_checks, split_del_ns, split_del_bytes, split_rep_us, split_rep_bytes
    );

    let encode_limit = cli.max_encode_ms_50k * n as f64 / 50_000.0;
    let best_recall = split_k
        .iter()
        .filter(|r| r.k <= cli.threshold_max_k)
        .map(|r| r.recall_at_top)
        .fold(0.0f64, f64::max);
    let mut thresholds = vec![
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
            pass: split_star.as_ref().map(|s| s.qps >= cli.min_qps).unwrap_or(false),
        },
        Threshold {
            name: format!("split / int8+rerank QPS ratio at matched recall >= {}", cli.min_ratio),
            value: ratio_matched.unwrap_or(0.0),
            limit: cli.min_ratio,
            pass: ratio_matched.map(|r| r >= cli.min_ratio).unwrap_or(false),
        },
        Threshold {
            name: format!(
                "split encode time for {} docs < {:.1} ms ({} ms per 50k, scaled)",
                n, encode_limit, cli.max_encode_ms_50k
            ),
            value: split_encode_ms,
            limit: encode_limit,
            pass: split_encode_ms < encode_limit,
        },
        Threshold {
            name: "no deleted id returned by any method after delete".to_string(),
            value: leaks as f64,
            limit: 0.0,
            pass: leaks == 0,
        },
    ];
    if self_checks > 0 {
        thresholds.push(Threshold {
            name: "replaced slots are found by their own new vector (split)".to_string(),
            value: self_hits as f64,
            limit: self_checks as f64,
            pass: self_hits == self_checks,
        });
    }
    let pass = thresholds.iter().all(|t| t.pass);

    println!("== hms-research-scan {} n={} dim={} basis={} mode={} ==", ds_name, n, dim, basis_tag, mode_tag);
    println!("{:<8} {:>6} {:>10} {:>10} {:>10}", "method", "K", "recall", "qps", "us/query");
    for r in &split_k {
        println!("{:<8} {:>6} {:>10.4} {:>10.0} {:>10.1}", "split", r.k, r.recall_at_top, r.qps, r.us_per_query);
    }
    for r in &int8_k {
        println!("{:<8} {:>6} {:>10.4} {:>10.0} {:>10.1}", "int8", r.k, r.recall_at_top, r.qps, r.us_per_query);
    }
    println!(
        "{:<8} {:>6} {:>10.4} {:>10.0} {:>10.1}",
        "exact", exact_rep.k, exact_rep.recall_at_top, exact_rep.qps, exact_rep.us_per_query
    );
    println!(
        "bytes/doc split={} int8={} exact={} | encode ms split={:.1} int8={:.1} | K* split={:?} int8={:?} | ratio matched={:?}",
        split.bytes_per_doc(),
        int8.bytes_per_doc(),
        primary.bytes_per_doc(),
        split_encode_ms,
        int8_encode_ms,
        split_star.as_ref().map(|s| s.k),
        int8_star.as_ref().map(|s| s.k),
        ratio_matched
    );
    for t in &thresholds {
        println!(
            "[{}] {} (value={:.4}, limit={:.4})",
            if t.pass { "PASS" } else { "FAIL" },
            t.name,
            t.value,
            t.limit
        );
    }
    println!("[{}] overall", if pass { "PASS" } else { "FAIL" });

    let report = Report {
        schema: SCHEMA,
        generated_unix_s: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        dataset: DatasetInfo {
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
        },
        config: ConfigInfo {
            basis: cli.basis,
            head_dim: HEAD_DIM,
            residual_bits: split.basis.bits,
            residual_words: split.basis.words,
            padded_dim: split.basis.padded,
            residual_mode: cli.residual_mode,
            ks: ks_sorted.clone(),
            runs: cli.runs,
            warmup_queries: cli.warmup.min(nq),
            rerank_top: cli.top,
            calib_vectors: calib_n,
            pca_iters: cli.pca_iters,
            head_variance_fraction: split.basis.head_variance_fraction,
            threads: 1,
            target: target_description(),
        },
        basis_fit_ms,
        methods: Methods {
            split: MethodReport {
                bytes_per_doc: split.bytes_per_doc(),
                encode_ms: split_encode_ms,
                encode_us_per_doc: split_encode_ms * 1e3 / n as f64,
                per_k: split_k,
            },
            int8: MethodReport {
                bytes_per_doc: int8.bytes_per_doc(),
                encode_ms: int8_encode_ms,
                encode_us_per_doc: int8_encode_ms * 1e3 / n as f64,
                per_k: int8_k,
            },
            exact: MethodReport {
                bytes_per_doc: primary.bytes_per_doc(),
                encode_ms: exact_encode_ms,
                encode_us_per_doc: exact_encode_ms * 1e3 / n as f64,
                per_k: vec![exact_rep.clone()],
            },
        },
        ablation: Ablation {
            target_recall: cli.target_recall,
            threshold_max_k: cli.threshold_max_k,
            split_k_star: split_star.as_ref().map(|s| s.k),
            int8_k_star: int8_star.as_ref().map(|s| s.k),
            split_qps_at_k_star: split_star.as_ref().map(|s| s.qps),
            int8_qps_at_k_star: int8_star.as_ref().map(|s| s.qps),
            exact_qps: exact_rep.qps,
            ratio_split_vs_int8_matched: ratio_matched,
            ratio_split_vs_exact_matched: ratio_exact,
            ratio_at_equal_k,
        },
        mutation: Mutation {
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
        },
        thresholds,
        pass,
    };

    let out = cli.out.clone().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("out")
            .join(format!("{}_{}_{}_{}.json", ds_name, n, basis_tag, mode_tag))
    });
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(&report).context("serialize report")?;
    fs::write(&out, text).with_context(|| format!("write {}", out.display()))?;
    println!("wrote {}", out.display());
    Ok(())
}
