// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public-benchmark driver. Reads data written by benchmarks/public/prepare.py and
//! writes machine-readable results; benchmarks/public/evaluate.py scores them and runs
//! the comparison libraries on the same data.
//!
//! `ann`:  ann-benchmarks sets (dense vectors + shipped ground truth). Measures the raw
//!         sparse-vector path (`from_dense` + inverted index): recall@10, encode and
//!         search latency per query, build time, resident memory.
//! `beir`: BEIR sets with precomputed embeddings and text. Writes ranked runs for the
//!         document API (lexical, dense, hybrid) and the raw sparse-vector path.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use holographic_memory::core::HmsConfig;
use holographic_memory::{DocumentInput, EmbeddingSpace, EntangledHVec, HmsCore, SearchOptions};
use rayon::prelude::*;
use serde_json::{json, Value};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
}

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
    Beir {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value_t = 16384)]
        dim: u32,
        #[arg(long)]
        out: PathBuf,
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

fn main() -> Result<()> {
    match Cli::parse().mode {
        Mode::Ann {
            data,
            dim,
            queries,
            out,
        } => ann(&data, dim, queries, &out),
        Mode::Beir { data, dim, out } => beir(&data, dim, &out),
    }
}
