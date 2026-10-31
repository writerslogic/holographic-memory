// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Retrieval gain and cost of the local embedder through the real document API.
//!
//! Reads a task file `{"stores": [{"docs": [{"id", "text"}], "queries": [{"id", "text"}]}]}`,
//! builds one store per entry and answers each query in the requested variants:
//!
//! * `lexical`   the default build's document search (BM25-style postings), no embedder
//! * `semantic`  embedder only (`lexicalWeight` 0)
//! * `hybrid`    both lists blended with the API's default weights
//!
//! `--encoder default` runs only `lexical` on a default-config store (the baseline, whose process
//! never loads an embedder); `--encoder local-embed-v1` also runs the other two on a store whose
//! documents are embedded at ingest. Output: ranked ids per variant plus timings. nDCG is computed
//! by benchmarks/embedding/run_store_eval.py; RSS and cold start by the process wrapper there.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use writerslogic_warm::core::embed::EmbedOptions;
use writerslogic_warm::core::encoding::TextEncoder;
use writerslogic_warm::core::WarmConfig;
use writerslogic_warm::{DocumentInput, SearchOptions, WarmCore};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    task: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// `default` (lexical baseline) or `local-embed-v1`.
    #[arg(long, default_value = "default")]
    encoder: String,
    /// Registry id or pack path.
    #[arg(long, default_value = "default")]
    model: String,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 4096)]
    dimensions: u32,
    #[arg(long, default_value_t = 100)]
    k: u32,
    /// Documents per `memorize_documents` call.
    #[arg(long, default_value_t = 256)]
    ingest_batch: usize,
}

fn input(id: &str, text: &str) -> DocumentInput {
    DocumentInput {
        id: id.into(),
        text: text.into(),
        source_uri: None,
        version: None,
        metadata: None,
        chunk_words: Some(4096),
        overlap_words: Some(0),
        store_text: Some(false),
        embeddings: None,
        title_weight: None,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let task: Value = serde_json::from_slice(&std::fs::read(&args.task).context("read task")?)?;
    let local = args.encoder == "local-embed-v1";
    let t_start = Instant::now();
    let mut stores_out = Vec::new();
    let (mut ingest_secs, mut ingested, mut first_ready) = (0.0f64, 0usize, None);
    let mut query_ms: Vec<(&str, Vec<f64>)> = Vec::new();
    for store in task["stores"].as_array().context("stores")? {
        let dir = tempfile::tempdir()?;
        let config = if local {
            WarmConfig {
                text_encoder: TextEncoder::LocalEmbedV1,
                local_embed: Some(EmbedOptions {
                    model: args.model.clone(),
                    threads: args.threads,
                    batch_size: args.batch_size,
                    ..EmbedOptions::default()
                }),
                ..WarmConfig::default()
            }
        } else {
            WarmConfig::default()
        };
        let core = WarmCore::new(
            args.dimensions,
            Some(dir.path().to_string_lossy().into_owned()),
            Some(config),
        )?;
        let docs = store["docs"].as_array().context("docs")?;
        let t = Instant::now();
        for chunk in docs.chunks(args.ingest_batch) {
            let inputs = chunk
                .iter()
                .map(|d| {
                    Ok(input(
                        d["id"].as_str().context("id")?,
                        d["text"].as_str().context("text")?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            core.memorize_documents(inputs)?;
        }
        ingest_secs += t.elapsed().as_secs_f64();
        ingested += docs.len();
        let variants: &[(&str, f64, f64)] = if local {
            &[
                ("lexical", 1.0, 0.0),
                ("semantic", 0.0, 1.0),
                ("hybrid", 1.0, 1.0),
            ]
        } else {
            &[("lexical", 1.0, 0.0)]
        };
        let mut runs = serde_json::Map::new();
        for q in store["queries"].as_array().context("queries")? {
            let (qid, text) = (
                q["id"].as_str().context("qid")?,
                q["text"].as_str().context("qtext")?,
            );
            let mut per = serde_json::Map::new();
            for (name, lw, sw) in variants {
                let options = SearchOptions {
                    k: Some(args.k),
                    candidate_limit: Some(args.k.max(100)),
                    lexical_weight: Some(*lw),
                    semantic_weight: Some(*sw),
                    min_semantic_score: Some(-1.0),
                    ..SearchOptions::default()
                };
                let t = Instant::now();
                let hits = core.search_documents(text, &options)?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                match query_ms.iter_mut().find(|(n, _)| n == name) {
                    Some((_, v)) => v.push(ms),
                    None => query_ms.push((name, vec![ms])),
                }
                per.insert(
                    (*name).into(),
                    json!(hits.into_iter().map(|h| h.document_id).collect::<Vec<_>>()),
                );
            }
            runs.insert(qid.into(), Value::Object(per));
        }
        if first_ready.is_none() {
            first_ready = Some(t_start.elapsed().as_secs_f64());
        }
        stores_out.push(Value::Object(runs));
    }
    let summary = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        json!({"n": v.len(), "p50_ms": v[v.len() / 2], "p95_ms": v[(v.len() * 95 / 100).min(v.len() - 1)],
               "mean_ms": v.iter().sum::<f64>() / v.len() as f64})
    };
    let latency: serde_json::Map<String, Value> = query_ms
        .iter_mut()
        .map(|(n, v)| ((*n).to_string(), summary(v)))
        .collect();
    std::fs::write(
        &args.out,
        serde_json::to_vec(&json!({
            "encoder": args.encoder,
            "model": args.model,
            "documents": ingested,
            "ingest_seconds": ingest_secs,
            "ingest_docs_per_second": ingested as f64 / ingest_secs,
            "total_seconds": t_start.elapsed().as_secs_f64(),
            "query_latency": latency,
            "runs": stores_out,
        }))?,
    )?;
    Ok(())
}
