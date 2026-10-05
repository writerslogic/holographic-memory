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

fn main() -> Result<()> {
    match Cli::parse().mode {
        Mode::Ann {
            data,
            dim,
            queries,
            out,
        } => ann(&data, dim, queries, &out),
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
    }
}
