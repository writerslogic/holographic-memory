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

use anyhow::{bail, ensure, Context, Result};
use holographic_memory::core::documents::{
    BatchTooLarge, EncodeMode, IngestOptions, MAX_BATCH_ITEMS,
};
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const USAGE: &str =
    "usage: competitive_text <build|serve|ops> --input FILE --store DIR --out FILE \
    [--passes N (3)] [--ops N (100)] [--t0-ns NS (0)] [--encode eager|deferred|lexical (eager)] \
    [--batch N (0)]";

/// Command line, parsed by hand: this binary is what the harness measures as HMS's install
/// footprint, and an argument-parsing framework would be most of it.
struct Args {
    /// build | serve | ops
    phase: String,
    input: PathBuf,
    store: PathBuf,
    out: PathBuf,
    passes: usize,
    ops: usize,
    /// Unix epoch nanoseconds taken by the orchestrator just before spawning this process.
    t0_ns: u128,
    /// `build` only: eager (chunk hypervectors at ingest, the default) | deferred | lexical
    /// (BM25 index only, like-for-like with the lexical comparators).
    encode: String,
    /// `build` only: documents per atomic batch (0 = as many per batch as the engine accepts).
    batch: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut words = std::env::args().skip(1);
        let phase = words.next().context(USAGE)?;
        let (mut input, mut store, mut out) = (None, None, None);
        let mut args = Args {
            phase,
            input: PathBuf::new(),
            store: PathBuf::new(),
            out: PathBuf::new(),
            passes: 3,
            ops: 100,
            t0_ns: 0,
            encode: "eager".into(),
            batch: 0,
        };
        while let Some(flag) = words.next() {
            let value = words
                .next()
                .with_context(|| format!("{flag} needs a value; {USAGE}"))?;
            let number = || format!("{flag} takes a non-negative integer, not {value:?}");
            match flag.as_str() {
                "--input" => input = Some(PathBuf::from(&value)),
                "--store" => store = Some(PathBuf::from(&value)),
                "--out" => out = Some(PathBuf::from(&value)),
                "--passes" => args.passes = value.parse().with_context(number)?,
                "--ops" => args.ops = value.parse().with_context(number)?,
                "--t0-ns" => args.t0_ns = value.parse().with_context(number)?,
                "--encode" => args.encode = value,
                "--batch" => args.batch = value.parse().with_context(number)?,
                _ => bail!("unknown argument {flag}; {USAGE}"),
            }
        }
        args.input = input.context(USAGE)?;
        args.store = store.context(USAGE)?;
        args.out = out.context(USAGE)?;
        Ok(args)
    }
}

#[derive(Deserialize)]
struct Input {
    documents: Vec<DocumentInput>,
    queries: Vec<Query>,
}

/// One hit as the serve report lists it.
#[derive(Serialize)]
struct Hit {
    #[serde(rename = "documentId")]
    document_id: String,
    #[serde(rename = "lexicalScore")]
    lexical_score: f64,
}

#[derive(Serialize)]
struct Row<'a> {
    id: &'a str,
    results: Vec<Hit>,
}

#[derive(Serialize)]
struct ServeReport<'a> {
    open_s: f64,
    cold_start_s: f64,
    rows: Vec<Row<'a>>,
    passes_ns: Vec<Vec<u64>>,
    memory: serde_json::Value,
}

#[derive(Deserialize)]
struct Query {
    id: String,
    text: String,
    options: SearchOptions,
}

/// The serving process's peak and current resident memory in KiB (Linux procfs: `VmHWM` and
/// `RssFile` from `status`, `Rss` and `Anonymous` from `smaps_rollup`); null elsewhere.
fn process_memory() -> serde_json::Value {
    #[cfg(target_os = "linux")]
    {
        let read = |path: &str| std::fs::read_to_string(path).unwrap_or_default();
        let field = |text: &str, name: &str| -> Option<u64> {
            text.lines()
                .find_map(|line| line.strip_prefix(name))
                .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse().ok())
        };
        let (status, rollup) = (read("/proc/self/status"), read("/proc/self/smaps_rollup"));
        json!({"vm_hwm_kib": field(&status, "VmHWM:"), "rss_kib": field(&rollup, "Rss:"),
            "anonymous_kib": field(&rollup, "Anonymous:"), "rss_file_kib": field(&status, "RssFile:")})
    }
    #[cfg(not(target_os = "linux"))]
    serde_json::Value::Null
}

fn epoch_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Ingests `documents` in the fewest atomic batches the engine accepts. The engine bounds a
/// batch by items, chunks, source and metadata bytes, embedding values and transaction bytes, and
/// rejects an oversized batch before writing anything; so start at `MAX_BATCH_ITEMS` documents and
/// halve on a batch-limit error, keeping the size that worked. A fixed size takes `fixed`.
fn ingest(
    core: &HmsCore,
    documents: &[DocumentInput],
    options: &IngestOptions,
    fixed: usize,
) -> Result<()> {
    let mut size = if fixed > 0 { fixed } else { MAX_BATCH_ITEMS };
    let mut at = 0;
    while at < documents.len() {
        let end = (at + size).min(documents.len());
        match core.memorize_documents_with(documents[at..end].to_vec(), options) {
            Ok(_) => at = end,
            Err(error) if fixed == 0 && size > 1 && error.is::<BatchTooLarge>() => size /= 2,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let input: Input = serde_json::from_slice(&std::fs::read(&args.input)?)?;
    let store = args.store.display().to_string();
    let report = match args.phase.as_str() {
        "build" => {
            ensure!(!args.store.exists(), "store exists; remove it first");
            let mode: EncodeMode = args.encode.parse()?;
            let options = IngestOptions {
                encode: mode,
                ..Default::default()
            };
            let count = input.documents.len();
            let start = Instant::now();
            let core = HmsCore::new(1024, Some(store), None)?;
            ingest(&core, &input.documents, &options, args.batch)?;
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
                        // The same fields the Python workers keep: id and score per hit.
                        let results = results
                            .into_iter()
                            .map(|r| Hit {
                                document_id: r.document_id,
                                lexical_score: r.score,
                            })
                            .collect();
                        rows.push(Row {
                            id: &query.id,
                            results,
                        });
                    }
                }
                passes.push(ns);
            }
            // Streamed to the file: the report is never held as a JSON value tree.
            let report = ServeReport {
                open_s,
                cold_start_s,
                rows,
                passes_ns: passes,
                memory: process_memory(),
            };
            let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
            serde_json::to_writer(&mut out, &report)?;
            std::io::Write::flush(&mut out)?;
            return Ok(());
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
