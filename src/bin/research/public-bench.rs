// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Public-benchmark driver. Reads data written by benchmarks/public/prepare.py and
//! writes machine-readable results; benchmarks/public/evaluate.py scores them and runs
//! the comparison libraries on the same data.
//!
//! `ann`:  ann-benchmarks sets (dense vectors + shipped ground truth). Measures the raw
//!         sparse-vector path (`from_dense` + inverted index): recall@10, encode and
//!         search latency per query, build time, resident memory.
//! `ann-qgraph`: the same sets through the quantized graph index (`core::qgraph`). Builds on
//!         all cores, then times single-threaded queries one at a time over an `ef` x
//!         `max-exact` sweep, each configuration repeated and gated on the 1-minute load.
//!         `--index vertex` uses the per-vertex 8-bit code index (`VGraph`) instead, swept over
//!         `ef` x `rerank`; `--residual` adds its second code for the re-rank. Angular and
//!         (vertex index only) Euclidean sets.
//! `beir`: BEIR sets with precomputed embeddings and text. Writes ranked runs for the
//!         document API (lexical, dense, hybrid) and the raw sparse-vector path.
//! `longmemeval`: LongMemEval_S with precomputed embeddings. Every question gets a fresh store
//!         holding only its own haystack; writes full ranked runs per granularity (turn,
//!         session) for the document API (lexical, dense, hybrid).
//! `lme-scores`: per-question scoring for benchmarks/public/longmemeval_pipeline.py. One fresh
//!         store per question holds that question's keys; each query variant runs one hybrid
//!         search over the whole store, and the engine's BM25 and exact-cosine scores of every key
//!         are written out for fusion.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use holographic_memory::core::qgraph::{
    BuildParams, Graph, Metric, QGraph, Residency, SearchParams, UpdatableVGraph, VGraph,
    VSearchParams,
};
use holographic_memory::core::HmsConfig;
use holographic_memory::{DocumentInput, EmbeddingSpace, EntangledHVec, HmsCore, SearchOptions};
use rayon::prelude::*;
use serde_json::{json, Value};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
}

// Parsed once; the ann-qgraph variant carries every sweep list, and boxing it buys nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Mode {
    Ann {
        /// Dataset directory written by prepare.py.
        #[arg(long)]
        data: PathBuf,
        /// Sparse hypervector dimension.
        #[arg(long, default_value_t = 16384)]
        dim: u32,
        /// Only the first N test queries (all by default).
        #[arg(long)]
        queries: Option<usize>,
        #[arg(long)]
        out: PathBuf,
    },
    AnnUpdates {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long, value_parser = ["serve", "cold-start", "replace", "delete", "session"])]
        action: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "10,16,24,32,48,64,96,128,192,256"
        )]
        ef: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "32,64")]
        rerank: Vec<usize>,
        #[arg(long, default_value_t = 2)]
        repeats: usize,
        #[arg(long)]
        cold: bool,
    },
    AnnQgraph {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Number of neighbors to return and score (VIBE uses 100).
        #[arg(long, default_value = "10", value_parser = ["10", "100"])]
        count: String,
        /// Distance: `angular` or `euclidean` (vertex index only). Defaults to the dataset's
        /// metric; a value that contradicts it is an error.
        #[arg(long, value_parser = ["angular", "euclidean"])]
        metric: Option<String>,
        /// Candidate pool sizes to sweep.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "10,16,24,32,48,64,96,128,192,256,384,512"
        )]
        ef: Vec<usize>,
        /// Caps on exactly scored vertices per query (0 = no cap).
        #[arg(long, value_delimiter = ',', default_value = "0")]
        max_exact: Vec<usize>,
        #[arg(long, default_value_t = 128)]
        build_ef: usize,
        /// Score the construction search with 8-bit codes (exact prune).
        #[arg(long)]
        quantized_search: bool,
        #[arg(long, default_value_t = 1.0)]
        alpha: f32,
        #[arg(long, default_value_t = 32)]
        degree: usize,
        /// Edge code length in bits (0 = smallest power of two >= dim).
        #[arg(long, default_value_t = 0)]
        code_bits: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Time only when the 1-minute load average is below this.
        #[arg(long, default_value_t = 3.0)]
        max_load: f64,
        /// Total seconds the sweep may spend waiting for the load gate; once spent, runs
        /// proceed and record the load and `load_gate_met: false`.
        #[arg(long, default_value_t = 3600)]
        max_wait_secs: u64,
        #[arg(long)]
        queries: Option<usize>,
        /// Tuning mode: hold out every (n / N)-th train vector as a query (N in total), build
        /// on the rest and score against exact truth in the metric; the test queries are not
        /// read.
        #[arg(long, conflicts_with = "queries")]
        holdout: Option<usize>,
        /// Index layout: `edge` (1-bit code per edge, QGraph) or `vertex` (8-bit code per
        /// vertex, VGraph).
        #[arg(long, default_value = "edge", value_parser = ["edge", "vertex"])]
        index: String,
        /// Vertex index only: also store the residual code used by the re-rank.
        #[arg(long)]
        residual: bool,
        /// Vertex index only: bits per residual coordinate with `--residual` (8 or 4).
        #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u8).range(4..=8))]
        residual_bits: u8,
        /// Vertex index only: bits per traversal code coordinate (8 or 4).
        #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u8).range(4..=8))]
        vertex_bits: u8,
        /// Vertex index only: bytes per stored neighbour id (4, 3 below 2^24 - 1 vertices, or
        /// 0 for sorted delta-coded varint lists).
        #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(0..=4))]
        id_bytes: u8,
        /// Vertex index only: start every code row on a 128-byte cache line.
        #[arg(long)]
        align_rows: bool,
        /// Vertex index only: renumber vertices in breadth-first order for locality.
        #[arg(long)]
        reorder: bool,
        /// Vertex index only: pool candidates re-scored with the float query (0 = none).
        #[arg(long, value_delimiter = ',', default_value = "0")]
        rerank: Vec<usize>,
        /// Vertex index only: stop after this many expansions without a change to the 10 best
        /// estimates (0 = pool rule only); swept with ef and rerank.
        #[arg(long, value_delimiter = ',', default_value = "0")]
        patience: Vec<usize>,
        /// Vertex index only: build the 1-bit screen consulted before every 8-bit estimate.
        #[arg(long)]
        screen: bool,
        /// Vertex index only: screen margins in standard deviations of its error, swept with
        /// ef (a negative value searches without the screen).
        #[arg(long, value_delimiter = ',', default_value = "1.0")]
        screen_sigmas: Vec<f32>,
        /// Vertex index only: bits per coordinate of the neighbour-block codes scanned with a
        /// 16-entry lookup table per nibble (1, 2 or 4; 0 = no blocks, per-neighbour rows).
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=4))]
        block_bits: u8,
        /// Vertex index with blocks only: fuse repeated block estimates of a vertex and
        /// re-admit it when the fused estimate qualifies.
        #[arg(long)]
        fuse: bool,
        /// Vertex index, per-row search: estimate only the first N neighbours of each expanded
        /// vertex (0 = all).
        #[arg(long, default_value_t = 0)]
        neighbour_cap: usize,
        /// Paired timing: after the build write `<SYNC>.ready`, then before timing round `r`
        /// block until `<SYNC>.go.<r>` exists and afterwards write `<SYNC>.done.<r>`, so an
        /// outside coordinator holding the timing lock can alternate two processes. The build
        /// does not wait for the load gate and timed runs only record it.
        #[arg(long)]
        sync: Option<PathBuf>,
        /// Load the graph structure from this file if it exists, else build and save it there
        /// (benchmark support for search experiments; the file must match data and params).
        #[arg(long)]
        graph_cache: Option<PathBuf>,
        /// After the throughput repeats, time every query of every configuration once more
        /// individually and record p50/p99 latency (the throughput runs are unchanged).
        #[arg(long)]
        latency: bool,
        /// Build and encode the vertex index, write it to this file (and the build record to
        /// `<file>.json`), then exit without searching: the build half of a two-process run
        /// whose serving process passes `--load-index`.
        #[arg(long, conflicts_with = "load_index")]
        save_index: Option<PathBuf>,
        #[arg(long, requires = "save_index")]
        save_updates: Option<PathBuf>,
        /// Serve a vertex index written by `--save-index` instead of building one. The train
        /// vectors are not read; the build parameters and dataset must match the saved ones.
        #[arg(long)]
        load_index: Option<PathBuf>,
        /// Where a loaded index lives: `prefaulted` (mapped, every page touched while
        /// opening), `mapped` (pages read on first touch) or `heap` (read into owned memory).
        #[arg(long, default_value = "prefaulted")]
        residency: String,
        /// With `--load-index` (Linux): drop the index file's cached pages before opening it,
        /// so the first queries read it from disk.
        #[arg(long)]
        cold: bool,
    },
    Beir {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value_t = 16384)]
        dim: u32,
        #[arg(long)]
        out: PathBuf,
    },
    Longmemeval {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value_t = 16384)]
        dim: u32,
        /// Only the first N questions (all by default).
        #[arg(long)]
        questions: Option<usize>,
        #[arg(long)]
        out: PathBuf,
    },
    LmeScores {
        /// JSONL keys: {"q": question id, "id": key id, "text", "e": row in --item-emb}.
        #[arg(long)]
        items: PathBuf,
        #[arg(long)]
        item_emb: PathBuf,
        /// JSONL query variants: {"q": question id, "v": variant, "text", "e": row in --query-emb}.
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        query_emb: PathBuf,
        /// Embedding dimension of both .f32 files.
        #[arg(long)]
        emb_dim: usize,
        #[arg(long, default_value_t = 16384)]
        dim: u32,
        #[arg(long)]
        out: PathBuf,
    },
    /// HMS's on-device model stages for the LongMemEval job (feature `local-models`).
    #[cfg(feature = "local-models")]
    LmeModel {
        /// `embed` (JSON list of texts -> little-endian f32 rows), `rerank` (list of
        /// [query, document] -> list of p(yes)), `facts` (list of sessions, each a list of user
        /// turns -> raw LLM outputs), `query` (list of [question date, question] -> raw outputs),
        /// `chat` (list of user messages -> greedy replies). `device` times the document API end
        /// to end: `--input` is the LongMemEval_S cache directory, `--model` the directory
        /// holding the three small models, `--revision` is ignored (pinned revisions are used).
        #[arg(long)]
        stage: String,
        /// Local model directory (safetensors + tokenizer.json); never downloaded here.
        #[arg(long)]
        model: PathBuf,
        /// Commit sha the directory must record.
        #[arg(long)]
        revision: String,
        /// `embed` only: inputs are queries (instruction prefix).
        #[arg(long)]
        query: bool,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Sequences per forward pass (encoders) or decoded together (`facts`, `chat`);
        /// default: the device default for encoders, 1 for decoding.
        #[arg(long)]
        batch: Option<usize>,
        /// Load gate: wait (inside any timing lock) until the 1-minute load average is below
        /// this before loading and timing; the timing line records the loads and whether the
        /// gate was met.
        #[arg(long)]
        max_load: Option<f64>,
        #[arg(long, default_value_t = 3600)]
        max_wait_secs: u64,
    },
}

fn read_f32(path: &Path, dim: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = fs::read(path).with_context(|| path.display().to_string())?;
    ensure!(
        bytes.len() % (4 * dim) == 0,
        "{} is not a multiple of {dim} floats",
        path.display()
    );
    Ok(bytes
        .chunks_exact(4 * dim)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        })
        .collect())
}

fn read_i32(path: &Path, width: usize) -> Result<Vec<Vec<i32>>> {
    let bytes = fs::read(path).with_context(|| path.display().to_string())?;
    ensure!(
        bytes.len() % (4 * width) == 0,
        "{} has a ragged shape",
        path.display()
    );
    Ok(bytes
        .chunks_exact(4 * width)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect()
        })
        .collect())
}

fn read_jsonl(path: &Path) -> Result<Vec<(String, String)>> {
    fs::read_to_string(path)
        .with_context(|| path.display().to_string())?
        .lines()
        .map(|l| {
            let v: Value = serde_json::from_str(l)?;
            Ok((
                v["id"].as_str().unwrap_or_default().to_string(),
                v["text"].as_str().unwrap_or_default().to_string(),
            ))
        })
        .collect()
}

fn meta(dir: &Path) -> Result<Value> {
    Ok(serde_json::from_str(&fs::read_to_string(
        dir.join("meta.json"),
    )?)?)
}

/// A fresh store under the system temp directory, removed when dropped.
struct TempStore(PathBuf);
impl TempStore {
    fn new(tag: &str) -> Result<Self> {
        let p = std::env::temp_dir().join(format!("hms-public-bench-{tag}-{}", std::process::id()));
        if p.exists() {
            fs::remove_dir_all(&p)?;
        }
        Ok(Self(p))
    }
}
impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `path` as one row-major matrix of `dim` little-endian floats per row, read in 1 MiB chunks
/// so the data is resident once (not as the file's bytes and again as rows).
fn read_f32_flat(path: &Path, dim: usize) -> Result<Vec<f32>> {
    use std::io::Read;
    let mut file = fs::File::open(path).with_context(|| path.display().to_string())?;
    let len = usize::try_from(file.metadata()?.len())?;
    ensure!(
        len.is_multiple_of(4 * dim),
        "{} is not a multiple of {dim} floats",
        path.display()
    );
    let mut out = Vec::with_capacity(len / 4);
    let mut buf = vec![0u8; 1 << 20];
    let mut left = len;
    while left > 0 {
        let take = left.min(buf.len());
        file.read_exact(&mut buf[..take])?;
        out.extend(
            buf[..take]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        );
        left -= take;
    }
    Ok(out)
}

/// Memory of this process. Linux: the resident set and its anonymous and file-backed parts
/// with the peak so far (`/proc/self/status`), and `Rss`, `Anonymous` and `Pss_File` from
/// `/proc/self/smaps_rollup`, in bytes. Elsewhere: the resident set from `ps`.
fn memory() -> Value {
    #[cfg(target_os = "linux")]
    {
        let mut out = serde_json::Map::new();
        let sources: [(&str, &str, &[&str]); 2] = [
            (
                "/proc/self/status",
                "",
                &["VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"],
            ),
            (
                "/proc/self/smaps_rollup",
                "smaps_",
                &["Rss", "Anonymous", "Pss_File"],
            ),
        ];
        for (file, prefix, keys) in sources {
            let Ok(text) = fs::read_to_string(file) else {
                continue;
            };
            for line in text.lines() {
                let Some((k, v)) = line.split_once(':') else {
                    continue;
                };
                let kb = v.trim().strip_suffix("kB").map(|n| n.trim().parse::<u64>());
                if let (true, Some(Ok(kb))) = (keys.contains(&k), kb) {
                    out.insert(format!("{prefix}{k}"), json!(kb * 1024));
                }
            }
        }
        Value::Object(out)
    }
    #[cfg(not(target_os = "linux"))]
    json!({ "VmRSS": rss_bytes() })
}

/// Records the memory at the end of phase `name`. Peaks are never reset: the harness reads the
/// process peak from `wait4`, which the same counter feeds.
fn phase(phases: &mut Vec<Value>, name: &str) {
    let m = memory();
    let mib = |k: &str| m[k].as_u64().map_or(-1.0, |b| b as f64 / (1 << 20) as f64);
    eprintln!(
        "phase {name}: rss {:.0} MiB (anon {:.0}, file {:.0}), peak {:.0} MiB",
        mib("VmRSS"),
        mib("RssAnon"),
        mib("RssFile"),
        mib("VmHWM")
    );
    phases.push(json!({"phase": name, "memory": m}));
}

/// Drops the page-cache pages of `path` that no process maps (Linux
/// `posix_fadvise(POSIX_FADV_DONTNEED)`), so the next reads of it come from disk.
fn evict(path: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        extern "C" {
            fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        }
        const POSIX_FADV_DONTNEED: i32 = 4;
        let file = fs::File::open(path).with_context(|| path.display().to_string())?;
        // SAFETY: `file` is open for the duration of the call, which only advises the kernel
        // about its cached pages.
        let rc = unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
        ensure!(rc == 0, "posix_fadvise({}) failed: {rc}", path.display());
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        anyhow::bail!("--cold needs Linux (posix_fadvise)")
    }
}

/// The build record written next to a saved index.
fn index_record(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".json");
    PathBuf::from(s)
}

fn rss_bytes() -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(out.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn latency_summary(mut us: Vec<f64>) -> Value {
    us.sort_by(f64::total_cmp);
    json!({
        "mean_us": us.iter().sum::<f64>() / us.len() as f64,
        "p50_us": percentile(&us, 0.5),
        "p95_us": percentile(&us, 0.95),
        "p99_us": percentile(&us, 0.99),
    })
}

fn environment() -> Value {
    let cmd = |c: &str, a: &[&str]| {
        std::process::Command::new(c)
            .args(a)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    json!({
        "cpu": cmd("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "load_average": cmd("sysctl", &["-n", "vm.loadavg"]),
        "hms_version": env!("CARGO_PKG_VERSION"),
        "rayon_threads": rayon::current_num_threads(),
    })
}

fn ann(data: &Path, dim: u32, queries: Option<usize>, out: &Path) -> Result<()> {
    let m = meta(data)?;
    let d = m["dim"].as_u64().context("meta.dim")? as usize;
    let width = m["n_neighbors"].as_u64().context("meta.n_neighbors")? as usize;
    let train = read_f32(&data.join("train.f32"), d)?;
    let mut test = read_f32(&data.join("test.f32"), d)?;
    let truth = read_i32(&data.join("neighbors.i32"), width)?;
    if let Some(q) = queries {
        test.truncate(q);
    }
    const K: usize = 10;
    const PREFILTER: usize = 100;

    let t = Instant::now();
    let codes: Vec<EntangledHVec> = train
        .par_iter()
        .map(|v| EntangledHVec::from_dense(v, dim as usize))
        .collect();
    let encode_secs = t.elapsed().as_secs_f64();
    drop(train);

    let store = TempStore::new("ann")?;
    let core = HmsCore::new(dim, Some(store.0.display().to_string()), None)?;
    let t = Instant::now();
    for (i, code) in codes.into_iter().enumerate() {
        core.memorize(i.to_string(), code)?;
    }
    let insert_secs = t.elapsed().as_secs_f64();
    let rss = rss_bytes();

    // Single-threaded queries, as in ann-benchmarks.
    let (mut enc_us, mut search_us) = (Vec::new(), Vec::new());
    let (mut hits, mut prefilter_hits) = (0usize, 0usize);
    for (q, t_ids) in test.iter().zip(&truth) {
        let gold: Vec<String> = t_ids[..K].iter().map(|i| i.to_string()).collect();
        let t0 = Instant::now();
        let code = EntangledHVec::from_dense(q, dim as usize);
        let t1 = Instant::now();
        let res = core.query(&code, K as u32);
        let t2 = Instant::now();
        enc_us.push((t1 - t0).as_secs_f64() * 1e6);
        search_us.push((t2 - t1).as_secs_f64() * 1e6);
        hits += res.iter().filter(|r| gold.contains(&r.id)).count();
        let wide = core.query(&code, PREFILTER as u32);
        prefilter_hits += wide.iter().filter(|r| gold.contains(&r.id)).count();
    }
    let n = test.len();
    let total_us: Vec<f64> = enc_us.iter().zip(&search_us).map(|(a, b)| a + b).collect();
    let mean_total = total_us.iter().sum::<f64>() / n as f64;
    let report = json!({
        "dataset": m,
        "environment": environment(),
        "path": "raw sparse vector: EntangledHVec::from_dense + HmsCore::query (inverted index)",
        "sparse_dim": dim,
        "k": K,
        "n_queries": n,
        "recall_at_10": hits as f64 / (n * K) as f64,
        "experimental_prefilter": {
            "note": "fraction of the true top-10 inside the sparse top-100; an exact re-rank of those candidates would return this recall. Not a shipped path.",
            "candidates": PREFILTER,
            "recall_at_10": prefilter_hits as f64 / (n * K) as f64,
        },
        "build": {"encode_secs_parallel": encode_secs, "insert_secs": insert_secs, "rss_bytes_after_build": rss},
        "query_encode": latency_summary(enc_us),
        "query_search": latency_summary(search_us),
        "query_total": latency_summary(total_us),
        "qps_single_thread": 1e6 / mean_total,
    });
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    println!(
        "{}",
        json!({"recall_at_10": report["recall_at_10"], "qps": report["qps_single_thread"], "prefilter": report["experimental_prefilter"]["recall_at_10"]})
    );
    Ok(())
}

/// Truncate to the engine's single-chunk limits (4096 words, 64 KiB) on a char boundary.
fn load_1m() -> Option<f64> {
    if cfg!(target_os = "linux") {
        return fs::read_to_string("/proc/loadavg")
            .ok()?
            .split_whitespace()
            .next()?
            .parse()
            .ok();
    }
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// Waits until the 1-minute load is below `max` or `deadline` passes; returns the load and
/// whether the gate was met.
fn wait_for_idle(max: f64, deadline: Instant) -> (f64, bool) {
    loop {
        let l = load_1m().unwrap_or(f64::NAN);
        if l < max {
            return (l, true);
        }
        if Instant::now() >= deadline {
            return (l, false);
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}

struct QgraphArgs {
    data: PathBuf,
    out: PathBuf,
    count: String,
    metric: Option<String>,
    ef: Vec<usize>,
    max_exact: Vec<usize>,
    build_ef: usize,
    quantized_search: bool,
    alpha: f32,
    code_bits: usize,
    degree: usize,
    repeats: usize,
    max_load: f64,
    max_wait_secs: u64,
    queries: Option<usize>,
    holdout: Option<usize>,
    index: String,
    residual: bool,
    residual_bits: u8,
    vertex_bits: u8,
    id_bytes: u8,
    align_rows: bool,
    reorder: bool,
    rerank: Vec<usize>,
    patience: Vec<usize>,
    screen: bool,
    screen_sigmas: Vec<f32>,
    block_bits: u8,
    fuse: bool,
    neighbour_cap: usize,
    sync: Option<PathBuf>,
    graph_cache: Option<PathBuf>,
    latency: bool,
    save_index: Option<PathBuf>,
    save_updates: Option<PathBuf>,
    load_index: Option<PathBuf>,
    residency: String,
    cold: bool,
}

enum Built {
    Edge(QGraph),
    Vertex(Box<VGraph>),
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Indexed rows, query rows and per-query ground-truth ids.
type Split = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<i32>>);

/// Splits `rows` into a tuning set of `n` queries (every `rows.len() / n`-th row) and the
/// rest, with the exact top-`k` of each query over the rest as ground truth: by cosine
/// (rows normalized) for angular, by Euclidean distance (rows as given) otherwise.
fn holdout_split(mut rows: Vec<Vec<f32>>, n: usize, k: usize, metric: Metric) -> Result<Split> {
    ensure!(n > 0 && n < rows.len(), "holdout must be in 1..n_train");
    let angular = metric == Metric::Angular;
    let step = rows.len() / n;
    let mut queries = Vec::with_capacity(n);
    let mut keep = Vec::with_capacity(rows.len() - n);
    for (i, mut r) in rows.drain(..).enumerate() {
        if angular {
            normalize(&mut r);
        }
        // A zero row has no cosine neighbours, so it is never a query; it stays indexed.
        if i % step == 0 && queries.len() < n && (!angular || r.iter().any(|&v| v != 0.0)) {
            queries.push(r);
        } else {
            keep.push(r);
        }
    }
    let truth: Vec<Vec<i32>> = queries
        .par_iter()
        .map(|q| {
            let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
            for (i, r) in keep.iter().enumerate() {
                // Larger is nearer: the inner product, or the negated squared distance,
                // accumulated in f64 (exact for integer data such as SIFT).
                let s = if angular {
                    q.iter().zip(r).map(|(a, b)| a * b).sum::<f32>()
                } else {
                    -(q.iter()
                        .zip(r)
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                        .sum::<f64>() as f32)
                };
                if top.len() < k || s > top[k - 1].0 {
                    let pos = top.partition_point(|t| t.0 >= s);
                    top.insert(pos, (s, i as u32));
                    top.truncate(k);
                }
            }
            top.iter().map(|t| t.1 as i32).collect()
        })
        .collect();
    Ok((keep, queries, truth))
}

fn ann_qgraph(a: &QgraphArgs) -> Result<()> {
    match a.count.as_str() {
        "10" => ann_qgraph_count::<10>(a),
        "100" => ann_qgraph_count::<100>(a),
        _ => anyhow::bail!("count must be 10 or 100"),
    }
}

fn ann_qgraph_count<const K: usize>(a: &QgraphArgs) -> Result<()> {
    ensure!(a.repeats > 0, "repeats must be positive");
    let m = meta(&a.data)?;
    let data_metric = m["metric"].as_str().context("meta.metric")?;
    let metric_name = a.metric.as_deref().unwrap_or(data_metric);
    ensure!(
        metric_name == data_metric,
        "--metric {metric_name} contradicts the dataset metric {data_metric}"
    );
    let metric = match metric_name {
        "angular" => Metric::Angular,
        "euclidean" => Metric::Euclidean,
        other => anyhow::bail!("qgraph supports angular and euclidean data, not {other}"),
    };
    ensure!(
        metric == Metric::Angular || a.index == "vertex",
        "euclidean data needs --index vertex (the edge index is angular only)"
    );
    let d = m["dim"].as_u64().context("meta.dim")? as usize;
    let width = m["n_neighbors"].as_u64().context("meta.n_neighbors")? as usize;
    ensure!(d > 0, "dimension must be positive");
    ensure!(
        a.holdout.is_some() || width >= K,
        "ground truth has {width} neighbors, count is {K}"
    );
    let load = a.load_index.as_deref();
    ensure!(
        load.is_none() || (a.holdout.is_none() && a.index == "vertex"),
        "--load-index serves a saved vertex index on the dataset's test queries (no --holdout)"
    );
    ensure!(
        a.save_index.is_none() || a.index == "vertex",
        "--save-index writes the vertex index (--index vertex)"
    );
    ensure!(
        !a.cold || load.is_some(),
        "--cold applies to an index opened with --load-index"
    );
    let residency = match a.residency.as_str() {
        "prefaulted" => Residency::Prefaulted,
        "mapped" => Residency::Mapped,
        "heap" => Residency::Heap,
        other => anyhow::bail!("--residency is prefaulted, mapped or heap, not {other}"),
    };
    let mut phases = Vec::new();
    let (mut train, mut test, truth) = match a.holdout {
        Some(n) => {
            let rows = read_f32(&a.data.join("train.f32"), d)?;
            let (keep, q, t) = holdout_split(rows, n, K, metric)?;
            (keep.concat(), q, t)
        }
        None => {
            let mut test = read_f32(&a.data.join("test.f32"), d)?;
            let truth = read_i32(&a.data.join("neighbors.i32"), width)?;
            if let Some(q) = a.queries {
                test.truncate(q);
            }
            let train = match load {
                Some(_) => Vec::new(),
                None => read_f32_flat(&a.data.join("train.f32"), d)?,
            };
            (train, test, truth)
        }
    };
    phase(&mut phases, "input");
    ensure!(
        load.is_some() || train.len() / d >= K,
        "corpus must contain at least {K} indexed rows"
    );
    ensure!(!test.is_empty(), "at least one query is required");
    ensure!(
        truth.len() >= test.len(),
        "ground truth has fewer rows than queries"
    );
    // Angular queries are normalized outside the timer, as evaluate.py does for the other
    // systems; Euclidean ones are used as given.
    if metric == Metric::Angular {
        for q in &mut test {
            normalize(q);
        }
    }
    let params = BuildParams {
        metric,
        build_ef: a.build_ef,
        quantized_search: a.quantized_search,
        alpha: a.alpha,
        code_bits: a.code_bits,
        degree: a.degree,
        ..BuildParams::default()
    };
    let build_params = json!({"index": a.index, "metric": metric_name, "degree": a.degree, "build_ef": a.build_ef, "quantized_search": a.quantized_search,
                              "alpha": a.alpha, "code_bits": a.code_bits, "seed": params.seed,
                              "residual": a.residual, "residual_bits": a.residual_bits,
                              "vertex_bits": a.vertex_bits, "id_bytes": a.id_bytes, "screen": a.screen,
                              "block_bits": a.block_bits, "align_rows": a.align_rows, "reorder": a.reorder});
    let wait = if a.sync.is_some() { 0 } else { a.max_wait_secs };
    let deadline = Instant::now() + std::time::Duration::from_secs(wait);
    let (load_before_build, build_gate) = wait_for_idle(a.max_load, deadline);
    let t = Instant::now();
    let mut graph_hash = None;
    let (mut open_secs, mut index_file_bytes, mut saved_build_secs) = (None, None, None);
    let index = match (a.index.as_str(), load) {
        (_, Some(p)) => {
            let record: Value = serde_json::from_str(
                &fs::read_to_string(index_record(p))
                    .with_context(|| format!("reading the build record of {}", p.display()))?,
            )?;
            ensure!(
                record["build_params"] == build_params,
                "{} was built with {} but {build_params} was requested",
                p.display(),
                record["build_params"]
            );
            ensure!(
                record["dataset"] == m,
                "{} was built for another dataset ({})",
                p.display(),
                record["dataset"]
            );
            graph_hash = record["graph_hash"].as_str().map(String::from);
            saved_build_secs = record["build_secs"].as_f64();
            if a.cold {
                evict(p)?;
            }
            let t = Instant::now();
            let v =
                VGraph::open(p, residency).with_context(|| format!("opening {}", p.display()))?;
            open_secs = Some(t.elapsed().as_secs_f64());
            ensure!(
                v.metric() == metric.into() && v.dim() == d,
                "{} indexes {:?} vectors of dimension {}, the dataset is {metric:?} in {d}",
                p.display(),
                v.metric(),
                v.dim()
            );
            ensure!(
                v.len() >= K,
                "corpus must contain at least {K} indexed rows"
            );
            index_file_bytes = Some(fs::metadata(p)?.len());
            phase(&mut phases, "open");
            Built::Vertex(Box::new(v))
        }
        ("edge", None) => Built::Edge(QGraph::build(&train, d, &params)?),
        _ => {
            // The cache header records n, dim, degree, build_ef, alpha, seed and a data
            // fingerprint; a mismatch is an error, never a silent reuse.
            let graph = match &a.graph_cache {
                Some(p) if p.exists() => Graph::load(&train, d, p, &params)
                    .with_context(|| format!("loading {}", p.display()))?,
                cache => {
                    let g = Graph::build_owned(std::mem::take(&mut train), d, &params)?;
                    if let Some(p) = cache {
                        g.save(p)
                            .with_context(|| format!("saving {}", p.display()))?;
                    }
                    g
                }
            };
            graph_hash = Some(format!("{:016x}", graph.structure_hash()));
            phase(&mut phases, "graph");
            anyhow::ensure!(
                matches!(a.residual_bits, 4 | 8) && matches!(a.vertex_bits, 4 | 8),
                "--residual-bits and --vertex-bits must be 4 or 8"
            );
            let bits = if a.residual { a.residual_bits } else { 0 };
            let mut v = VGraph::from_graph_with(
                &graph,
                a.vertex_bits,
                bits,
                a.align_rows,
                a.reorder,
                a.id_bytes,
                a.screen,
            )?;
            if a.block_bits > 0 {
                v = v.with_blocks(&graph, a.block_bits)?;
            }
            phase(&mut phases, "encode");
            if let Some(path) = &a.save_updates {
                UpdatableVGraph::create_from_graph(path, &graph, &v)?;
            }
            Built::Vertex(Box::new(v))
        }
    };
    let build_secs = saved_build_secs.unwrap_or_else(|| t.elapsed().as_secs_f64());
    let load_after_build = load_1m();
    if load.is_none() {
        phase(&mut phases, "drop_graph");
    }
    drop(train);
    let (index_bytes, residual_bytes) = match &index {
        Built::Edge(g) => (g.index_bytes(), 0),
        Built::Vertex(g) => (g.index_bytes(), g.residual_bytes()),
    };
    eprintln!("built in {build_secs:.1}s, {index_bytes} bytes");
    let code_rows_huge_bytes = match &index {
        Built::Vertex(g) => g.code_rows_huge_bytes(),
        Built::Edge(_) => None,
    };
    if let (Some(p), Built::Vertex(v)) = (&a.save_index, &index) {
        v.save(p)
            .with_context(|| format!("saving {}", p.display()))?;
        let record = json!({"graph_hash": graph_hash, "build_params": build_params, "dataset": m, "build_secs": build_secs});
        fs::write(
            index_record(p),
            serde_json::to_string_pretty(&record)? + "\n",
        )?;
        index_file_bytes = Some(fs::metadata(p)?.len());
    }
    phase(&mut phases, "serve_start");
    let rss_after_build = rss_bytes();

    // (sweep label, search) pairs; the second parameter is max_exact or rerank.
    let seconds: &[usize] = match &index {
        Built::Edge(_) => &a.max_exact,
        Built::Vertex(_) => &a.rerank,
    };
    let (mut edge_s, mut vertex_s) = match &index {
        Built::Edge(g) => (Some(g.searcher()), None),
        Built::Vertex(g) => (None, Some(g.searcher())),
    };
    let second_name = match &index {
        Built::Edge(_) => "max_exact",
        Built::Vertex(_) => "rerank",
    };
    // (second, ef, patience, screen_sigmas); the edge index ignores the last two.
    let (pat, sig): (&[usize], &[f32]) = match &index {
        Built::Edge(_) => (&[0], &[1.0]),
        Built::Vertex(_) => (&a.patience, &a.screen_sigmas),
    };
    // A build-only run (`--save-index`) searches nothing.
    let configs: Vec<(usize, usize, usize, f32)> = seconds
        .iter()
        .filter(|_| a.save_index.is_none())
        .flat_map(|&second| {
            a.ef.iter().flat_map(move |&ef| {
                pat.iter()
                    .flat_map(move |&p| sig.iter().map(move |&s| (second, ef, p, s)))
            })
        })
        .collect();
    // Returns (estimates, neighbours the screen skipped).
    let mut search = |q: &[f32],
                      (second, ef, patience, sigmas): (usize, usize, usize, f32),
                      out: &mut Vec<u32>|
     -> (usize, usize) {
        if let Some(s) = edge_s.as_mut() {
            let e = s.search(
                q,
                K,
                SearchParams {
                    ef,
                    max_exact: second,
                },
                out,
            );
            (e, 0)
        } else {
            let s = vertex_s.as_mut().expect("one searcher exists");
            let e = s
                .search(
                    q,
                    K,
                    VSearchParams {
                        ef,
                        reservoir: 0,
                        rerank: second,
                        patience,
                        screen_sigmas: sigmas.max(0.0),
                        screen: sigmas >= 0.0,
                        fuse: a.fuse,
                        neighbour_cap: a.neighbour_cap,
                    },
                    out,
                )
                .expect("queries are validated by the loader");
            (e, s.last_screened())
        }
    };
    let sync_path = |suffix: String| -> Option<PathBuf> {
        a.sync.as_ref().map(|p| {
            let mut s = p.clone().into_os_string();
            s.push(suffix);
            PathBuf::from(s)
        })
    };
    if let Some(p) = sync_path(".ready".into()) {
        fs::write(&p, "")?;
    }
    // Per configuration: qps runs, loads, gate, recall, evals/query, screened/query.
    type Acc = (Vec<f64>, Vec<f64>, bool, f64, f64, f64);
    let mut acc: Vec<Acc> = vec![(Vec::new(), Vec::new(), true, 0.0, 0.0, 0.0); configs.len()];
    let mut ids: Vec<u32> = Vec::with_capacity(test.len() * K);
    let mut out = Vec::with_capacity(K);
    // The first query alone, before the timed rounds: after `--cold` it pays the first page
    // faults of the index.
    // Also its wall-clock end, from which a harness measures the cold start of the process.
    let (first_query_us, first_query_unix_ns) = match (configs.first(), test.first()) {
        (Some(&cfg), Some(q)) => {
            let t = Instant::now();
            search(q, cfg, &mut out);
            let us = t.elapsed().as_secs_f64() * 1e6;
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
            (Some(us), Some(now.as_nanos() as u64))
        }
        _ => (None, None),
    };
    // Round-major, so one round of every configuration is a unit a coordinator can pair.
    for rep in 0..a.repeats {
        if let Some(go) = sync_path(format!(".go.{rep}")) {
            while !go.exists() {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        for (&cfg, (qps, loads, gate, recall, exact_mean, screen_mean)) in
            configs.iter().zip(&mut acc)
        {
            let (load, met) = wait_for_idle(a.max_load, deadline);
            loads.push(load);
            *gate &= met;
            ids.clear();
            let (mut exact, mut screened) = (0usize, 0usize);
            let t = Instant::now();
            for q in &test {
                let (e, s) = search(q, cfg, &mut out);
                exact += e;
                screened += s;
                ids.extend_from_slice(&out);
                ids.resize(ids.len() + K - out.len(), u32::MAX);
            }
            qps.push(test.len() as f64 / t.elapsed().as_secs_f64());
            if rep == 0 {
                let hits: usize = ids
                    .as_chunks::<K>()
                    .0
                    .iter()
                    .zip(&truth)
                    .map(|(f, t)| {
                        let t = &t[..K];
                        let mut f = f.to_vec();
                        f.sort_unstable();
                        f.dedup();
                        f.iter().filter(|&&x| t.contains(&(x as i32))).count()
                    })
                    .sum();
                *recall = hits as f64 / (K * test.len()) as f64;
                *exact_mean = exact as f64 / test.len() as f64;
                *screen_mean = screened as f64 / test.len() as f64;
            }
        }
        if let Some(done) = sync_path(format!(".done.{rep}")) {
            fs::write(&done, "")?;
        }
    }
    let rss_end = rss_bytes();
    phase(&mut phases, "serve_end");
    let mut latency: Vec<Option<(f64, f64)>> = vec![None; configs.len()];
    if a.latency {
        for (slot, &cfg) in latency.iter_mut().zip(&configs) {
            let mut us = Vec::with_capacity(test.len());
            for q in &test {
                let t = Instant::now();
                search(q, cfg, &mut out);
                us.push(t.elapsed().as_secs_f64() * 1e6);
            }
            us.sort_by(f64::total_cmp);
            *slot = Some((percentile(&us, 0.5), percentile(&us, 0.99)));
        }
    }
    let mut rows = Vec::new();
    for (
        (&(second, ef, patience, sigmas), (qps, loads, gate, recall, exact_mean, screen_mean)),
        lat,
    ) in configs.iter().zip(&acc).zip(&latency)
    {
        let mut sorted = qps.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        eprintln!("ef={ef} {second_name}={second} patience={patience} sigmas={sigmas} recall={recall:.4} qps={median:.0} evals/q={exact_mean:.0} screened/q={screen_mean:.0} load={loads:?}");
        rows.push(json!({
            "params": {"ef": ef, second_name: second, "patience": patience, "screen": a.screen && sigmas >= 0.0, "screen_sigmas": sigmas},
            "screened_per_query": screen_mean,
            (format!("recall_at_{K}")): recall,
            "qps_single_thread": median,
            "qps_runs": qps,
            "latency_us": lat.map(|(p50, p99)| json!({"p50": p50, "p99": p99})),
            "load_1m_before_runs": loads,
            "load_gate_met": gate,
            "mean_exact_evals_per_query": exact_mean,
        }));
    }
    let report = json!({
        "system": match a.index.as_str() {
            "edge" => "hms qgraph",
            _ => "hms qgraph-vertex",
        },
        "dataset": m,
        "environment": environment(),
        "n_queries": test.len(),
        "k": K,
        "build": {"params": {"index": a.index, "metric": metric_name, "degree": a.degree, "build_ef": a.build_ef, "quantized_search": a.quantized_search,
                             "alpha": a.alpha, "code_bits": a.code_bits, "seed": params.seed,
                             "residual": a.residual, "residual_bits": a.residual_bits,
                             "vertex_bits": a.vertex_bits, "id_bytes": a.id_bytes, "screen": a.screen,
                             "block_bits": a.block_bits, "fuse": a.fuse, "neighbour_cap": a.neighbour_cap,
                             "align_rows": a.align_rows, "reorder": a.reorder,
                             "graph_cache": a.graph_cache.as_ref().map(|p| p.display().to_string())},
                  "graph_hash": graph_hash,
                  "build_secs": build_secs, "load_1m_before_build": load_before_build,
                  "load_1m_after_build": load_after_build, "load_gate_met": build_gate,
                  "threads": rayon::current_num_threads()},
        "index_bytes": index_bytes,
        "huge_pages": {"process": holographic_memory::core::qgraph::huge_page_status(),
                       "code_rows_huge_bytes": code_rows_huge_bytes},
        "residual_bytes": residual_bytes,
        "rss_bytes": {"after_build": rss_after_build, "end": rss_end},
        "memory_phases": phases,
        "index_file": {"saved": a.save_index.as_ref().map(|p| p.display().to_string()),
                       "loaded": load.map(|p| p.display().to_string()),
                       "bytes": index_file_bytes, "residency": load.map(|_| a.residency.as_str()),
                       "cold": a.cold, "open_secs": open_secs},
        "first_query_us": first_query_us,
        "first_query_unix_ns": first_query_unix_ns,
        "holdout": a.holdout.map(|n| json!({
            "n_queries": n,
            "note": "Tuning run: queries are train vectors held out of the index (every n/N-th row), truth is exact (cosine or Euclidean, per the metric) over the rest; the test set was not read.",
        })),
        "repeats": a.repeats,
        "max_load": a.max_load,
        "paired_sync": a.sync.is_some(),
        "sweep": rows,
        "notes": [
            "Single-threaded, one query at a time; query rotation and quantization are inside the timer, normalization is outside (as for the other systems).",
            "qps_single_thread is the median of the repeats; recall is from the first repeat (search is deterministic).",
            "Edge index: each expanded vertex is scored exactly; mean_exact_evals_per_query counts those plus the upper-layer descent. Vertex index: it counts every code estimate (descent, traversal) plus the re-rank scores.",
        ],
    });
    fs::write(&a.out, serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(())
}

fn one_chunk(text: &str) -> String {
    let mut end = text.len().min(60 * 1024);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let words: Vec<&str> = text[..end].split_whitespace().take(4000).collect();
    if words.is_empty() {
        "_".into()
    } else {
        words.join(" ")
    }
}

fn beir(data: &Path, dim: u32, out: &Path) -> Result<()> {
    let m = meta(data)?;
    let d = m["dim"].as_u64().context("meta.dim")? as usize;
    let corpus = read_jsonl(&data.join("corpus.jsonl"))?;
    let queries = read_jsonl(&data.join("queries.jsonl"))?;
    let corpus_emb = read_f32(&data.join("corpus.f32"), d)?;
    let query_emb = read_f32(&data.join("queries.f32"), d)?;
    ensure!(
        corpus.len() == corpus_emb.len() && queries.len() == query_emb.len(),
        "row counts differ"
    );
    const K: u32 = 100;

    let doc_store = TempStore::new("beir-doc")?;
    let config = HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: m["model"].as_str().unwrap_or("unknown").into(),
            revision: m["model_revision"].as_str().unwrap_or("unknown").into(),
            dimensions: d,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..HmsConfig::default()
    };
    let docs = HmsCore::new(dim, Some(doc_store.0.display().to_string()), Some(config))?;
    let sparse_store = TempStore::new("beir-sparse")?;
    let sparse = HmsCore::new(dim, Some(sparse_store.0.display().to_string()), None)?;

    let t = Instant::now();
    for ((id, text), emb) in corpus.iter().zip(&corpus_emb) {
        docs.memorize_document(DocumentInput {
            id: id.clone(),
            text: one_chunk(text),
            source_uri: None,
            version: None,
            metadata: None,
            chunk_words: Some(4096),
            overlap_words: Some(0),
            store_text: Some(false),
            embeddings: Some(vec![emb.iter().map(|&x| f64::from(x)).collect()]),
            title_weight: None,
        })?;
    }
    let doc_ingest_secs = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let codes: Vec<EntangledHVec> = corpus_emb
        .par_iter()
        .map(|v| EntangledHVec::from_dense(v, dim as usize))
        .collect();
    for ((id, _), code) in corpus.iter().zip(codes) {
        sparse.memorize(format!("d:{id}"), code)?;
    }
    let sparse_ingest_secs = t.elapsed().as_secs_f64();

    let mut runs: HashMap<&str, HashMap<String, Vec<String>>> = HashMap::new();
    let mut latency: HashMap<&str, Vec<f64>> = HashMap::new();
    for ((qid, text), emb) in queries.iter().zip(&query_emb) {
        let emb64: Vec<f64> = emb.iter().map(|&x| f64::from(x)).collect();
        for (name, query_text, embedding, lexical, semantic) in [
            ("hms_lexical", text.as_str(), None, 1.0, 0.0),
            ("hms_dense_exact", "", Some(emb64.clone()), 0.0, 1.0),
            ("hms_hybrid", text.as_str(), Some(emb64.clone()), 1.0, 1.0),
        ] {
            let options = SearchOptions {
                k: Some(K),
                candidate_limit: Some(K.max(100)),
                embedding,
                lexical_weight: Some(lexical),
                semantic_weight: Some(semantic),
                min_semantic_score: Some(-1.0),
                ..SearchOptions::default()
            };
            let t0 = Instant::now();
            let hits = docs.search_documents(query_text, &options)?;
            latency
                .entry(name)
                .or_default()
                .push(t0.elapsed().as_secs_f64() * 1e6);
            runs.entry(name).or_default().insert(
                qid.clone(),
                hits.into_iter().map(|h| h.document_id).collect(),
            );
        }
        let t0 = Instant::now();
        let code = EntangledHVec::from_dense(emb, dim as usize);
        let hits = sparse.query(&code, K);
        latency
            .entry("hms_sparse")
            .or_default()
            .push(t0.elapsed().as_secs_f64() * 1e6);
        runs.entry("hms_sparse").or_default().insert(
            qid.clone(),
            hits.into_iter()
                .map(|h| h.id.trim_start_matches("d:").to_string())
                .collect(),
        );
    }
    let report = json!({
        "dataset": m,
        "environment": environment(),
        "sparse_dim": dim,
        "k": K,
        "ingest_secs": {"document_api": doc_ingest_secs, "sparse_encode_and_insert": sparse_ingest_secs},
        "latency": latency.into_iter().map(|(k, v)| (k.to_string(), latency_summary(v))).collect::<serde_json::Map<_, _>>(),
        "runs": runs,
    });
    fs::write(out, serde_json::to_string(&report)?)?;
    Ok(())
}

struct LmeItem {
    id: String,
    text: String,
    emb: usize,
}

/// Ranked ids per mode for one question's own haystack, plus per-mode search latency (us).
type LmeQuestion = (
    Vec<(&'static str, Vec<String>)>,
    Vec<(&'static str, f64)>,
    f64,
);

fn lme_question(
    gran: &str,
    qi: usize,
    items: &[LmeItem],
    embeddings: &[Vec<f32>],
    query: (&str, &[f32]),
    dim: u32,
    model: &Value,
) -> Result<LmeQuestion> {
    let store = TempStore::new(&format!("lme-{gran}-{qi}"))?;
    let config = HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: model["model"].as_str().unwrap_or("unknown").into(),
            revision: model["model_revision"].as_str().unwrap_or("unknown").into(),
            dimensions: embeddings.first().map_or(0, Vec::len),
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..HmsConfig::default()
    };
    let core = HmsCore::new(dim, Some(store.0.display().to_string()), Some(config))?;
    let t = Instant::now();
    for item in items {
        core.memorize_document(DocumentInput {
            id: item.id.clone(),
            text: one_chunk(&item.text),
            source_uri: None,
            version: None,
            metadata: None,
            chunk_words: Some(4096),
            overlap_words: Some(0),
            store_text: Some(false),
            embeddings: Some(vec![embeddings[item.emb]
                .iter()
                .map(|&x| f64::from(x))
                .collect()]),
            title_weight: None,
        })?;
    }
    let ingest = t.elapsed().as_secs_f64();
    // Rank the whole haystack so turn-to-session conversion can reach any depth.
    let k = items.len().clamp(1, 1000) as u32;
    let emb64: Vec<f64> = query.1.iter().map(|&x| f64::from(x)).collect();
    let (mut runs, mut lat) = (Vec::new(), Vec::new());
    for (name, query_text, embedding, lexical, semantic) in [
        ("lexical", query.0, None, 1.0, 0.0),
        ("dense_exact", "", Some(emb64.clone()), 0.0, 1.0),
        ("hybrid", query.0, Some(emb64.clone()), 1.0, 1.0),
    ] {
        let options = SearchOptions {
            k: Some(k),
            candidate_limit: Some(k.max(100)),
            embedding,
            lexical_weight: Some(lexical),
            semantic_weight: Some(semantic),
            min_semantic_score: Some(-1.0),
            ..SearchOptions::default()
        };
        let t0 = Instant::now();
        let hits = core.search_documents(query_text, &options)?;
        lat.push((name, t0.elapsed().as_secs_f64() * 1e6));
        runs.push((name, hits.into_iter().map(|h| h.document_id).collect()));
    }
    Ok((runs, lat, ingest))
}

fn longmemeval(data: &Path, dim: u32, questions: Option<usize>, out: &Path) -> Result<()> {
    let m = meta(data)?;
    let d = m["dim"].as_u64().context("meta.dim")? as usize;
    let qs = read_jsonl(&data.join("questions.jsonl"))?;
    let n = questions.unwrap_or(qs.len()).min(qs.len());
    let query_emb = read_f32(&data.join("queries.f32"), d)?;
    ensure!(
        query_emb.len() >= n,
        "queries.f32 has fewer rows than questions"
    );
    let mut report =
        json!({"dataset": m, "environment": environment(), "sparse_dim": dim, "runs": {}});
    for gran in ["turn", "session"] {
        let embeddings = read_f32(&data.join(format!("{gran}.f32")), d)?;
        let mut by_q: Vec<Vec<LmeItem>> = (0..n).map(|_| Vec::new()).collect();
        for line in fs::read_to_string(data.join(format!("{gran}.jsonl")))?.lines() {
            let v: Value = serde_json::from_str(line)?;
            let q = v["q"].as_u64().context("q")? as usize;
            if q < n {
                by_q[q].push(LmeItem {
                    id: v["id"].as_str().context("id")?.to_string(),
                    text: v["text"].as_str().unwrap_or_default().to_string(),
                    emb: v["e"].as_u64().context("e")? as usize,
                });
            }
        }
        let t = Instant::now();
        let per_q: Vec<LmeQuestion> = by_q
            .par_iter()
            .enumerate()
            .map(|(qi, items)| {
                lme_question(
                    gran,
                    qi,
                    items,
                    &embeddings,
                    (&qs[qi].1, &query_emb[qi]),
                    dim,
                    &m,
                )
            })
            .collect::<Result<_>>()?;
        let total = t.elapsed().as_secs_f64();
        let mut runs: HashMap<&str, HashMap<String, Vec<String>>> = HashMap::new();
        let mut latency: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut ingest = 0.0;
        for ((qid, _), (q_runs, q_lat, q_ingest)) in qs.iter().zip(per_q) {
            ingest += q_ingest;
            for (name, ids) in q_runs {
                runs.entry(name).or_default().insert(qid.clone(), ids);
            }
            for (name, us) in q_lat {
                latency.entry(name).or_default().push(us);
            }
        }
        report["runs"][gran] = json!({
            "wall_secs": total,
            "ingest_cpu_secs": ingest,
            "latency": latency.into_iter().map(|(k, v)| (k.to_string(), latency_summary(v))).collect::<serde_json::Map<_, _>>(),
            "rankings": runs,
        });
        println!("{gran}: {n} questions in {total:.1}s");
    }
    fs::write(out, serde_json::to_string(&report)?)?;
    Ok(())
}

struct LmeQuery {
    variant: String,
    text: String,
    emb: usize,
}

/// JSONL rows grouped by their "q" field, in first-seen order.
fn group_jsonl<T>(
    path: &Path,
    mut parse: impl FnMut(&Value) -> Result<T>,
) -> Result<Vec<(String, Vec<T>)>> {
    let mut groups: Vec<(String, Vec<T>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for line in fs::read_to_string(path)
        .with_context(|| path.display().to_string())?
        .lines()
    {
        let v: Value = serde_json::from_str(line)?;
        let q = v["q"].as_str().context("q")?.to_string();
        let slot = *index.entry(q.clone()).or_insert_with(|| {
            groups.push((q, Vec::new()));
            groups.len() - 1
        });
        groups[slot].1.push(parse(&v)?);
    }
    Ok(groups)
}

/// BM25 and exact-cosine scores of every key for every query variant, one store per question.
fn lme_scores(
    items: &Path,
    item_emb: &Path,
    queries: &Path,
    query_emb: &Path,
    emb_dim: usize,
    dim: u32,
    out: &Path,
) -> Result<()> {
    let keys = group_jsonl(items, |v| {
        Ok(LmeItem {
            id: v["id"].as_str().context("id")?.to_string(),
            text: v["text"].as_str().unwrap_or_default().to_string(),
            emb: v["e"].as_u64().context("e")? as usize,
        })
    })?;
    let variants: HashMap<String, Vec<LmeQuery>> = group_jsonl(queries, |v| {
        Ok(LmeQuery {
            variant: v["v"].as_str().context("v")?.to_string(),
            text: v["text"].as_str().unwrap_or_default().to_string(),
            emb: v["e"].as_u64().context("e")? as usize,
        })
    })?
    .into_iter()
    .collect();
    let key_emb = read_f32(item_emb, emb_dim)?;
    let q_emb = read_f32(query_emb, emb_dim)?;
    ensure!(
        keys.iter()
            .flat_map(|(_, k)| k)
            .all(|k| k.emb < key_emb.len()),
        "key embedding row out of range"
    );
    ensure!(
        variants.values().flatten().all(|v| v.emb < q_emb.len()),
        "query embedding row out of range"
    );
    let t = Instant::now();
    let scored: Vec<(String, Value)> = keys
        .par_iter()
        .enumerate()
        .map(|(n, (qid, items))| {
            let store = TempStore::new(&format!("lme-scores-{n}"))?;
            let config = HmsConfig {
                embedding_space: Some(EmbeddingSpace {
                    model: "external".into(),
                    revision: "see benchmarks/public/longmemeval_modal.py".into(),
                    dimensions: emb_dim,
                    normalization: "l2".into(),
                    metric: "cosine".into(),
                }),
                ..HmsConfig::default()
            };
            let core = HmsCore::new(dim, Some(store.0.display().to_string()), Some(config))?;
            for item in items {
                core.memorize_document(DocumentInput {
                    id: item.id.clone(),
                    text: one_chunk(&item.text),
                    source_uri: None,
                    version: None,
                    metadata: None,
                    chunk_words: Some(4096),
                    overlap_words: Some(0),
                    store_text: Some(false),
                    embeddings: Some(vec![key_emb[item.emb]
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect()]),
                    title_weight: None,
                })?;
            }
            // The API returns at most 1000 hits, so each search is restricted to one chunk of
            // ids; BM25 statistics are store-wide and cosine is exact, so scores do not depend
            // on the chunking.
            let ids: Vec<String> = items.iter().map(|i| i.id.clone()).collect();
            let mut per_variant = serde_json::Map::new();
            for query in variants.get(qid).map_or(&[][..], Vec::as_slice) {
                let mut rows: Vec<Value> = Vec::with_capacity(ids.len());
                for chunk in ids.chunks(1000) {
                    let k = u32::try_from(chunk.len())?;
                    let options = SearchOptions {
                        k: Some(k),
                        candidate_limit: Some(k),
                        document_ids: Some(chunk.to_vec()),
                        embedding: Some(q_emb[query.emb].iter().map(|&x| f64::from(x)).collect()),
                        lexical_weight: Some(1.0),
                        semantic_weight: Some(1.0),
                        min_semantic_score: Some(-1.0),
                        ..SearchOptions::default()
                    };
                    rows.extend(
                        core.search_documents(&query.text, &options)?
                            .into_iter()
                            .map(|h| json!([h.document_id, h.lexical_score, h.semantic_score])),
                    );
                }
                per_variant.insert(query.variant.clone(), Value::Array(rows));
            }
            Ok((qid.clone(), Value::Object(per_variant)))
        })
        .collect::<Result<_>>()?;
    let report = json!({
        "environment": environment(),
        "sparse_dim": dim,
        "wall_secs": t.elapsed().as_secs_f64(),
        "scores": scored.into_iter().collect::<serde_json::Map<_, _>>(),
    });
    fs::write(out, serde_json::to_string(&report)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ann_updates(
    index_path: &Path,
    data: Option<&Path>,
    action: &str,
    out: Option<&Path>,
    efs: &[usize],
    reranks: &[usize],
    repeats: usize,
    cold: bool,
) -> Result<()> {
    if action == "session" {
        return ann_update_session(index_path);
    }
    let data = data.context("--data is required outside session mode")?;
    let metadata = meta(data)?;
    let dim = usize::try_from(metadata["dim"].as_u64().context("meta.dim")?)?;
    let mut test = read_f32(&data.join("test.f32"), dim)?;
    ensure!(!test.is_empty(), "empty query set");
    if metadata["metric"] == "angular" {
        for q in &mut test {
            normalize(q);
        }
    }
    if cold {
        for entry in fs::read_dir(index_path)? {
            let path = entry?.path();
            if path.is_file() {
                evict(&path)?;
            }
        }
    }
    let start = Instant::now();
    let index = UpdatableVGraph::open(index_path)?;
    let open_s = start.elapsed().as_secs_f64();
    let params = VSearchParams {
        ef: *efs.first().context("empty ef sweep")?,
        rerank: *reranks.first().context("empty rerank sweep")?,
        ..Default::default()
    };
    let first = Instant::now();
    std::hint::black_box(index.search(&test[0], 10, params)?);
    let first_query_s = first.elapsed().as_secs_f64();
    let cold_start_s = start.elapsed().as_secs_f64();
    let first_query_unix_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let rss_after_open = rss_bytes();
    let report = match action {
        "cold-start" => {
            json!({"open_s": open_s, "first_query_s": first_query_s, "cold_start_s": cold_start_s, "page_cache_dropped": cold, "memory_phases": [{"phase": "first_query", "memory": memory()}]})
        }
        "replace" | "delete" => {
            let train = read_f32_flat(&data.join("train.f32"), dim)?;
            let count = (train.len() / dim / 100).max(1);
            ensure!(
                count * 3 <= train.len() / dim,
                "dataset too small for update batch"
            );
            let begin = Instant::now();
            for offset in 0..count {
                if action == "replace" {
                    index.replace(
                        offset as u32,
                        &train[(offset + 2 * count) * dim..(offset + 2 * count + 1) * dim],
                    )?;
                } else {
                    index.delete((offset + count) as u32)?;
                }
            }
            let apply_s = begin.elapsed().as_secs_f64();
            let flush = Instant::now();
            index.commit()?;
            let flush_s = flush.elapsed().as_secs_f64();
            json!({"operation": action, "count": count, "apply_s": apply_s, "flush_s": flush_s, "per_op_ms_durable": (apply_s+flush_s)*1000./count as f64, "memory_phases": [{"phase": "committed", "memory": memory()}]})
        }
        "serve" => {
            ensure!(repeats > 0, "repeats must be positive");
            let width = usize::try_from(
                metadata["n_neighbors"]
                    .as_u64()
                    .context("meta.n_neighbors")?,
            )?;
            let truth = read_i32(&data.join("neighbors.i32"), width)?;
            ensure!(
                truth.len() >= test.len(),
                "ground truth shorter than query set"
            );
            let mut configs = Vec::new();
            let mut found = Vec::new();
            for &rerank in reranks {
                for &ef in efs {
                    let p = VSearchParams {
                        ef,
                        rerank,
                        ..Default::default()
                    };
                    let mut hits = 0;
                    for (q, truth) in test.iter().zip(&truth) {
                        index.search_into(q, 10, p, &mut found)?;
                        hits += found
                            .iter()
                            .filter(|&&id| truth[..10.min(truth.len())].contains(&(id as i32)))
                            .count();
                    }
                    let mut timings = Vec::new();
                    for _ in 0..repeats {
                        let mut lat = Vec::with_capacity(test.len());
                        for q in &test {
                            let t = Instant::now();
                            index.search_into(q, 10, p, &mut found)?;
                            std::hint::black_box(&found);
                            lat.push(t.elapsed().as_secs_f64() * 1e6);
                        }
                        let qps = test.len() as f64 / (lat.iter().sum::<f64>() * 1e-6);
                        lat.sort_by(f64::total_cmp);
                        timings.push(json!({"qps":qps, "p50_us":percentile(&lat,0.5), "p99_us":percentile(&lat,0.99)}));
                    }
                    configs.push(json!({"params":{"ef":ef,"rerank":rerank}, "recall_at_10":hits as f64/(test.len()*10) as f64, "repeats":timings, "load_1m":load_1m(), "load_gate_met":null}));
                }
            }
            json!({"n":index.len(), "n_queries":test.len(), "configs":configs, "open_s":open_s, "cold_start_s":cold_start_s, "first_query_unix_ns":first_query_unix_ns, "rss_steady":rss_bytes(), "rss_after_open":rss_after_open, "memory_phases":[{"phase":"serve_end","memory":memory()}]})
        }
        _ => anyhow::bail!("unsupported update action {action}"),
    };
    let bytes = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = out {
        fs::write(path, bytes)?;
    } else {
        print!("{bytes}");
    }
    Ok(())
}

fn ann_update_session(path: &Path) -> Result<()> {
    use std::io::{BufRead, Write};
    let index = UpdatableVGraph::open(path)?;
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{{\"ready\":true}}")?;
    stdout.flush()?;
    for line in std::io::stdin().lock().lines() {
        let request: Value = serde_json::from_str(&line?)?;
        let result = (|| -> Result<Value> {
            match request["op"].as_str().context("operation")? {
                "commit" => index.commit()?,
                "replace" | "delete" => {
                    let ids = request["ids"].as_array().context("ids")?;
                    for (at, id) in ids.iter().enumerate() {
                        let id = u32::try_from(id.as_u64().context("id")?)?;
                        if request["op"] == "delete" {
                            index.delete(id)?;
                        } else {
                            let row = request["vectors"][at]
                                .as_array()
                                .context("vector")?
                                .iter()
                                .map(|v| v.as_f64().context("coordinate").map(|v| v as f32))
                                .collect::<Result<Vec<_>>>()?;
                            index.replace(id, &row)?;
                        }
                    }
                }
                _ => anyhow::bail!("unknown operation"),
            }
            Ok(json!({"ok":true}))
        })();
        let response = match result {
            Ok(v) => v,
            Err(e) => json!({"error":e.to_string()}),
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().mode {
        Mode::Ann {
            data,
            dim,
            queries,
            out,
        } => ann(&data, dim, queries, &out),
        Mode::AnnUpdates {
            index,
            data,
            action,
            out,
            ef,
            rerank,
            repeats,
            cold,
        } => ann_updates(
            &index,
            data.as_deref(),
            &action,
            out.as_deref(),
            &ef,
            &rerank,
            repeats,
            cold,
        ),
        Mode::AnnQgraph {
            data,
            out,
            count,
            metric,
            ef,
            max_exact,
            build_ef,
            quantized_search,
            alpha,
            code_bits,
            degree,
            repeats,
            max_load,
            max_wait_secs,
            queries,
            holdout,
            index,
            residual,
            residual_bits,
            vertex_bits,
            id_bytes,
            align_rows,
            reorder,
            rerank,
            patience,
            screen,
            screen_sigmas,
            block_bits,
            fuse,
            neighbour_cap,
            sync,
            graph_cache,
            latency,
            save_index,
            save_updates,
            load_index,
            residency,
            cold,
        } => ann_qgraph(&QgraphArgs {
            data,
            out,
            count,
            metric,
            ef,
            max_exact,
            build_ef,
            quantized_search,
            alpha,
            code_bits,
            degree,
            repeats,
            max_load,
            max_wait_secs,
            queries,
            holdout,
            index,
            residual,
            residual_bits,
            vertex_bits,
            id_bytes,
            align_rows,
            reorder,
            rerank,
            patience,
            screen,
            screen_sigmas,
            block_bits,
            fuse,
            neighbour_cap,
            sync,
            graph_cache,
            latency,
            save_index,
            save_updates,
            load_index,
            residency,
            cold,
        }),
        Mode::Beir { data, dim, out } => beir(&data, dim, &out),
        Mode::Longmemeval {
            data,
            dim,
            questions,
            out,
        } => longmemeval(&data, dim, questions, &out),
        Mode::LmeScores {
            items,
            item_emb,
            queries,
            query_emb,
            emb_dim,
            dim,
            out,
        } => lme_scores(&items, &item_emb, &queries, &query_emb, emb_dim, dim, &out),
        #[cfg(feature = "local-models")]
        Mode::LmeModel {
            stage,
            model,
            revision,
            query,
            input,
            out,
            batch,
            max_load,
            max_wait_secs,
        } => lme_model(
            &stage,
            &model,
            &revision,
            query,
            &input,
            &out,
            batch,
            max_load.map(|m| (m, max_wait_secs)),
        ),
    }
}

/// Runs one model stage over a JSON input file; prints a timing line to stderr.
#[cfg(feature = "local-models")]
#[allow(clippy::too_many_arguments)]
fn lme_model(
    stage: &str,
    model: &Path,
    revision: &str,
    query: bool,
    input: &Path,
    out: &Path,
    batch: Option<usize>,
    gate: Option<(f64, u64)>,
) -> Result<()> {
    use holographic_memory::core::models::{
        default_device, Embedder, Generator, ModelSource, QueryRewriter, Reranker,
    };
    if stage == "device" {
        return lme_device(input, model, out);
    }
    let source = ModelSource::new(model, revision);
    let device = default_device()?;
    let raw = fs::read(input)?;
    let gate_result = gate.map(|(max, wait)| {
        let deadline = Instant::now() + std::time::Duration::from_secs(wait);
        wait_for_idle(max, deadline)
    });
    let load_before = load_avgs();
    let t = Instant::now();
    let load_secs: f64;
    let mut generated = 0usize;
    let n = match stage {
        "embed" => {
            let texts: Vec<String> = serde_json::from_slice(&raw)?;
            let mut embedder = Embedder::load(&source, &device)?;
            if let Some(b) = batch {
                embedder.set_batch(b);
            }
            load_secs = t.elapsed().as_secs_f64();
            let rows = embedder.embed(&texts, query)?;
            let bytes: Vec<u8> = rows
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            fs::write(out, bytes)?;
            texts.len()
        }
        "rerank" => {
            let pairs: Vec<(String, String)> = serde_json::from_slice(&raw)?;
            let mut reranker = Reranker::load(&source, &device)?;
            if let Some(b) = batch {
                reranker.set_batch(b);
            }
            load_secs = t.elapsed().as_secs_f64();
            let scores = reranker.score(&pairs)?;
            fs::write(out, serde_json::to_vec(&scores)?)?;
            pairs.len()
        }
        "chat" | "facts" | "query" => {
            let mut generator = Generator::load(&source, &device)?;
            if let Some(b) = batch {
                generator.set_batch(b);
            }
            load_secs = t.elapsed().as_secs_f64();
            let outs: Vec<String> = if stage == "chat" || stage == "facts" {
                // Both decode as one batch; `facts` is `FactExtractor::extract_batch`'s raw
                // output, generated here directly to count the tokens.
                let prompts: Vec<String> = if stage == "chat" {
                    serde_json::from_slice(&raw)?
                } else {
                    let sessions: Vec<Vec<String>> = serde_json::from_slice(&raw)?;
                    sessions
                        .iter()
                        .map(|s| holographic_memory::core::models::prompts::fact_prompt(s))
                        .collect()
                };
                let done = generator.generate(&prompts)?;
                generated = done.iter().map(|c| c.tokens).sum();
                done.into_iter().map(|c| c.text).collect()
            } else {
                let qs: Vec<(String, String)> = serde_json::from_slice(&raw)?;
                let x = QueryRewriter(&generator);
                qs.iter()
                    .map(|(today, q)| x.rewrite(today, q).map(|r| r.0))
                    .collect::<Result<_>>()?
            };
            fs::write(out, serde_json::to_vec(&outs)?)?;
            outs.len()
        }
        other => anyhow::bail!("unknown stage {other}"),
    };
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "{}",
        json!({"stage": stage, "items": n, "device": format!("{device:?}"),
               "batch": batch, "secs": secs, "load_secs": load_secs,
               "run_secs": secs - load_secs, "generated_tokens": generated,
               "load_avg_before": load_before, "load_avg_after": load_avgs(),
               "max_load": gate.map(|g| g.0),
               "load_at_gate": gate_result.map(|g| g.0),
               "load_gate_met": gate_result.map(|g| g.1)})
    );
    Ok(())
}

/// The 1, 5 and 15-minute load averages.
#[cfg(feature = "local-models")]
fn load_avgs() -> Option<Vec<f64>> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    let v: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .filter_map(|x| x.parse().ok())
        .collect();
    (v.len() == 3).then_some(v)
}

/// On-device cost through the document API with every stage on: each LongMemEval_S session
/// (its user turns) is ingested as one document (fact extraction + embedding), then questions
/// are searched (rewrite + query embeddings + hybrid search + re-rank). Sizes from
/// `HMS_DEVICE_SESSIONS` (default 20) and `HMS_DEVICE_QUERIES` (default 20).
#[cfg(feature = "local-models")]
fn lme_device(data: &Path, models: &Path, out: &Path) -> Result<()> {
    use holographic_memory::core::models::{
        default_device, Embedder, Generator, ModelSource, ModelStages, Reranker, PINNED,
    };
    let env_n = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (n_sessions, n_queries) = (
        env_n("HMS_DEVICE_SESSIONS", 20),
        env_n("HMS_DEVICE_QUERIES", 20),
    );
    let source = |id: &str| {
        let rev = PINNED
            .iter()
            .find(|p| p.0 == id)
            .map(|p| p.1)
            .unwrap_or_default();
        ModelSource::new(models.join(id.rsplit('/').next().unwrap_or(id)), rev)
    };
    let device = default_device()?;
    let t = Instant::now();
    let stages = ModelStages {
        embedder: Some(Embedder::load(
            &source("Qwen/Qwen3-Embedding-0.6B"),
            &device,
        )?),
        reranker: Some(Reranker::load(
            &source("Qwen/Qwen3-Reranker-0.6B"),
            &device,
        )?),
        generator: Some(Generator::load(
            &source("Qwen/Qwen3-4B-Instruct-2507"),
            &device,
        )?),
        extract_facts: true,
        rewrite_queries: true,
        params: Default::default(),
    };
    let load_secs = t.elapsed().as_secs_f64();
    let dims = stages.embedder.as_ref().map_or(0, Embedder::dimensions);
    // Sessions in file order: consecutive user turns sharing the `<session>_<n>` prefix.
    let mut sessions: Vec<(String, Vec<String>)> = Vec::new();
    for line in fs::read_to_string(data.join("turn.jsonl"))?.lines() {
        let v: Value = serde_json::from_str(line)?;
        let id = v["id"].as_str().context("id")?;
        let sid = id.rsplit_once('_').map_or(id, |x| x.0).to_string();
        let text = v["text"].as_str().unwrap_or_default().to_string();
        if let Some((_, turns)) = sessions.last_mut().filter(|(s, _)| *s == sid) {
            turns.push(text);
        } else if sessions.len() == n_sessions {
            break;
        } else {
            sessions.push((sid, vec![text]));
        }
    }
    let questions: Vec<String> = fs::read_to_string(data.join("questions.jsonl"))?
        .lines()
        .take(n_queries)
        .map(|l| {
            Ok(serde_json::from_str::<Value>(l)?["text"]
                .as_str()
                .unwrap_or_default()
                .to_string())
        })
        .collect::<Result<_>>()?;
    let store = TempStore::new("lme-device")?;
    let config = HmsConfig {
        embedding_space: Some(
            stages
                .embedder
                .as_ref()
                .context("missing local embedder")?
                .space(),
        ),
        ..HmsConfig::default()
    };
    let core = HmsCore::new(16384, Some(store.0.display().to_string()), Some(config))?;
    core.set_model_stages(Some(std::sync::Arc::new(stages)))?;
    let mut ingest = Vec::new();
    for (sid, turns) in &sessions {
        let t = Instant::now();
        core.memorize_document(DocumentInput {
            id: sid.clone(),
            text: turns.join("\n\n"),
            source_uri: None,
            version: None,
            metadata: None,
            chunk_words: Some(4096),
            overlap_words: Some(0),
            store_text: Some(true),
            embeddings: None,
            title_weight: None,
        })?;
        ingest.push(t.elapsed().as_secs_f64());
    }
    let mut query = Vec::new();
    for q in &questions {
        let t = Instant::now();
        core.search_documents(
            q,
            &SearchOptions {
                k: Some(10),
                ..SearchOptions::default()
            },
        )?;
        query.push(t.elapsed().as_secs_f64());
    }
    let summary = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let at = |p: f64| {
            v.get(((v.len() as f64 * p) as usize).min(v.len().saturating_sub(1)))
                .copied()
        };
        json!({"n": v.len(), "mean_secs": v.iter().sum::<f64>() / v.len().max(1) as f64,
               "p50_secs": at(0.5), "p90_secs": at(0.9), "max_secs": v.last()})
    };
    let words: usize = sessions
        .iter()
        .flat_map(|s| &s.1)
        .map(|t| t.split_whitespace().count())
        .sum();
    let report = json!({
        "environment": environment(),
        "device": format!("{device:?}"),
        "models": PINNED.iter().take(3).map(|p| format!("{}@{}", p.0, p.1)).collect::<Vec<_>>(),
        "model_load_secs": load_secs,
        "session_user_words_mean": words as f64 / sessions.len().max(1) as f64,
        "ingest_per_session": summary(ingest),
        "query": summary(query),
    });
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(())
}
