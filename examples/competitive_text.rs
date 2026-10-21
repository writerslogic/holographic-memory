// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! HMS side of the competitive text benchmark (benchmarks/competitive). The input file is the
//! `{documents, queries}` JSON written by `hms-bench`; every phase is its own process so the
//! orchestrator can read the child's peak RSS and cold-start time.
//!
//! `build`  bulk-ingests the corpus into `--store` with the batch API.
//! `serve`  opens the store, answers the first query (cold start), then times `--passes`
//!          passes over all queries one at a time.
//! `ops`    replaces `--ops` documents and then deletes `--ops` others, one durable commit each.

use anyhow::{ensure, Result};
use clap::Parser;
use holographic_memory::core::documents::{EncodeMode, IngestOptions};
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde::Deserialize;
use serde_json::json;
use std::{
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
struct Args {
    /// build | serve | ops
    phase: String,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    store: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 3)]
    passes: usize,
    #[arg(long, default_value_t = 100)]
    ops: usize,
    /// Unix epoch nanoseconds taken by the orchestrator just before spawning this process.
    #[arg(long, default_value_t = 0)]
    t0_ns: u128,
    /// `build` only: eager (chunk hypervectors at ingest, the default) | deferred | lexical
    /// (BM25 index only, like-for-like with the lexical comparators).
    #[arg(long, default_value = "eager")]
    encode: String,
    /// `build` only: documents per atomic batch (0 = the whole corpus in one batch).
    #[arg(long, default_value_t = 0)]
    batch: usize,
}

#[derive(Deserialize)]
struct Input {
    documents: Vec<DocumentInput>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Query {
    id: String,
    text: String,
    options: SearchOptions,
}

fn epoch_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input: Input = serde_json::from_slice(&std::fs::read(&args.input)?)?;
    let store = args.store.display().to_string();
    let report = match args.phase.as_str() {
        "build" => {
            ensure!(!args.store.exists(), "store exists; remove it first");
            let mode: EncodeMode = args.encode.parse()?;
            let options = IngestOptions { encode: mode };
            let start = Instant::now();
            let core = HmsCore::new(1024, Some(store), None)?;
            // One atomic batch, as the comparators load the corpus in one transaction.
            let count = input.documents.len();
            if args.batch == 0 {
                core.memorize_documents_with(input.documents, &options)?;
            } else {
                for chunk in input.documents.chunks(args.batch) {
                    core.memorize_documents_with(chunk.to_vec(), &options)?;
                }
            }
            json!({"documents": count, "ingest_s": start.elapsed().as_secs_f64(),
                "encode": mode.name()})
        }
        "serve" => {
            ensure!(
                args.passes > 0 && !input.queries.is_empty(),
                "nothing to time"
            );
            let start = Instant::now();
            let core = HmsCore::new(1024, Some(store), None)?;
            let open_s = start.elapsed().as_secs_f64();
            let first = &input.queries[0];
            core.search_documents(&first.text, &first.options)?;
            let cold_start_s = (epoch_ns().saturating_sub(args.t0_ns)) as f64 / 1e9;
            for query in input.queries.iter().take(16) {
                core.search_documents(&query.text, &query.options)?;
            }
            let mut rows = Vec::new();
            let mut passes = Vec::new();
            for pass in 0..args.passes {
                let mut ns = Vec::with_capacity(input.queries.len());
                for query in &input.queries {
                    let t = Instant::now();
                    let results = core.search_documents(&query.text, &query.options)?;
                    ns.push(t.elapsed().as_nanos() as u64);
                    if pass == 0 {
                        rows.push(json!({"id": query.id, "results": results}));
                    }
                }
                passes.push(ns);
            }
            json!({"open_s": open_s, "cold_start_s": cold_start_s, "rows": rows, "passes_ns": passes})
        }
        "ops" => {
            ensure!(
                input.documents.len() >= 2 * args.ops,
                "corpus too small for the ops count"
            );
            let core = HmsCore::new(1024, Some(store), None)?;
            let t = Instant::now();
            for document in input.documents.iter().take(args.ops) {
                let mut changed = document.clone();
                changed.text.push_str(" benchmarkrevision");
                changed.version = Some("2".into());
                core.memorize_document(changed)?;
            }
            let replace_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for document in input.documents.iter().skip(args.ops).take(args.ops) {
                ensure!(core.delete_document(&document.id)?, "document missing");
            }
            let delete_s = t.elapsed().as_secs_f64();
            json!({"count": args.ops,
                "replace": {"supported": true, "per_op_ms_durable": replace_s / args.ops as f64 * 1e3},
                "delete": {"supported": true, "per_op_ms_durable": delete_s / args.ops as f64 * 1e3}})
        }
        other => anyhow::bail!("unknown phase {other}"),
    };
    std::fs::write(args.out, serde_json::to_vec(&report)?)?;
    Ok(())
}
