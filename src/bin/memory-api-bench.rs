// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cost of the memory API against plain document search.
//!
//! For each store size the parent runs three isolated child processes so resident memory is
//! comparable: `raw` stores the texts as plain documents, `memory` stores the same texts as
//! facts, `history` stores them as facts of which a fifth are superseded and a fifth carry a
//! validity window (so recall must filter). Each child reports RSS after the build and a
//! warmed query, and per-query latency of the same queries through `search_documents` and
//! through `Memory::recall` on the same store, interleaved so drift hits both alike.
//!
//! `cargo run --release --bin memory-api-bench -- --out benchmarks/results/memory_api_overhead.json`

use anyhow::{ensure, Context, Result};
use clap::Parser;
use holographic_memory::core::memory_api::{
    HistoryMode, Memory, RecallOptions, RememberInput, SupersessionSpec,
};
use holographic_memory::core::HmsConfig;
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// Where the parent writes the report.
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', default_values_t = [1_000usize, 10_000, 100_000])]
    sizes: Vec<usize>,
    #[arg(long, default_value_t = 400)]
    queries: usize,
    /// Interleaved passes over the query set per child.
    #[arg(long, default_value_t = 5)]
    passes: usize,
    #[arg(long, default_value_t = 4096)]
    dimensions: u32,
    /// Internal: run one child of this kind.
    #[arg(long)]
    child: Option<String>,
    #[arg(long, default_value_t = 0)]
    n: usize,
}

const VOCABULARY: u64 = 20_000;
const WORDS: usize = 24;

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

/// A deterministic synthetic text; word frequencies are skewed so lexical scoring has
/// realistic ties and rare terms.
fn text(index: usize) -> String {
    let mut state = (index as u64).wrapping_mul(0x9E37_79B9) ^ 0xA5A5;
    (0..WORDS)
        .map(|_| {
            let a = next(&mut state) % VOCABULARY;
            let b = next(&mut state) % VOCABULARY;
            format!("w{}", a.min(b))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn query(index: usize, total: usize) -> String {
    let mut state = (index as u64 + 7) * 0x2545_F491;
    let doc = (next(&mut state) as usize) % total;
    text(doc)
        .split(' ')
        .step_by(5)
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
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
        .map(|kib| kib * 1024)
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    sorted[(((sorted.len() - 1) as f64) * p).round() as usize]
}

fn summary(mut nanos: Vec<u128>) -> Value {
    nanos.sort_unstable();
    let mean = nanos.iter().sum::<u128>() as f64 / nanos.len() as f64;
    json!({"n": nanos.len(), "mean_ns": mean, "p50_ns": percentile(&nanos, 0.5),
           "p95_ns": percentile(&nanos, 0.95), "p99_ns": percentile(&nanos, 0.99)})
}

fn child(args: &Args, kind: &str) -> Result<Value> {
    let dir = tempfile::tempdir()?;
    let core = Arc::new(HmsCore::new(
        args.dimensions,
        Some(dir.path().display().to_string()),
        Some(HmsConfig::default()),
    )?);
    let n = args.n;
    let build_start = Instant::now();
    let memory = Memory::with_config(
        core.clone(),
        holographic_memory::core::memory_api::MemoryConfig {
            supersession: SupersessionSpec::Explicit,
            ..Default::default()
        },
    )?;
    for start in (0..n).step_by(1000) {
        let end = (start + 1000).min(n);
        match kind {
            "raw" => {
                let docs: Vec<DocumentInput> = (start..end)
                    .map(|i| {
                        serde_json::from_value(json!({"id": format!("f{i}"), "text": text(i),
                            "metadata": {}}))
                        .unwrap()
                    })
                    .collect();
                core.memorize_documents(docs)?;
            }
            _ => {
                let facts: Vec<RememberInput> = (start..end)
                    .map(|i| {
                        let mut fact = RememberInput {
                            id: Some(format!("f{i}")),
                            text: text(i),
                            ..Default::default()
                        };
                        if kind == "history" {
                            if i % 5 == 1 {
                                // Superseded by the previous fact of the batch.
                                fact.supersedes = Some(vec![format!("f{}", i - 1)]);
                            }
                            if i % 5 == 2 {
                                fact.valid_from = Some(1);
                                fact.valid_to = Some(2);
                            }
                        }
                        fact
                    })
                    .collect();
                memory.remember_many(facts)?;
            }
        }
    }
    let build_seconds = build_start.elapsed().as_secs_f64();
    let queries: Vec<String> = (0..args.queries).map(|q| query(q, n)).collect();
    // Warm both paths: builds the index (for `history`) and faults in the postings.
    for q in queries.iter().take(20) {
        core.search_documents(q, &SearchOptions::default())?;
        memory.recall(q, &RecallOptions::default())?;
    }
    let rss_after_build = rss_bytes();
    let mut raw_ns = Vec::new();
    let mut recall_ns = Vec::new();
    let mut explain_ns = Vec::new();
    let mut history_ns = Vec::new();
    let mut same_top = 0usize;
    let mut compared = 0usize;
    for pass in 0..args.passes {
        for (i, q) in queries.iter().enumerate() {
            // Alternate the order so neither path always runs second.
            let raw_first = (pass + i) % 2 == 0;
            let run_raw = || -> Result<(u128, Vec<String>)> {
                let t = Instant::now();
                let r = core.search_documents(q, &SearchOptions::default())?;
                Ok((
                    t.elapsed().as_nanos(),
                    r.into_iter().map(|d| d.document_id).collect(),
                ))
            };
            let run_recall = |options: &RecallOptions| -> Result<(u128, Vec<String>)> {
                let t = Instant::now();
                let r = memory.recall(q, options)?;
                Ok((
                    t.elapsed().as_nanos(),
                    r.into_iter().map(|d| d.id).collect(),
                ))
            };
            let plain = RecallOptions::default();
            let (raw, recalled) = if raw_first {
                let a = run_raw()?;
                (a, run_recall(&plain)?)
            } else {
                let b = run_recall(&plain)?;
                (run_raw()?, b)
            };
            raw_ns.push(raw.0);
            recall_ns.push(recalled.0);
            if kind != "history" {
                compared += 1;
                same_top += usize::from(raw.1 == recalled.1);
            }
            let explained = run_recall(&RecallOptions {
                explain: true,
                ..Default::default()
            })?;
            explain_ns.push(explained.0);
            if kind == "history" {
                let all = run_recall(&RecallOptions {
                    history: Some(HistoryMode::All),
                    ..Default::default()
                })?;
                history_ns.push(all.0);
            }
        }
    }
    let rss_after_queries = rss_bytes();
    let stored_bytes: u64 = std::fs::read_dir(dir.path())?
        .filter_map(Result::ok)
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum();
    Ok(json!({
        "kind": kind, "n": n, "build_seconds": build_seconds,
        "rss_bytes_after_build": rss_after_build, "rss_bytes_after_queries": rss_after_queries,
        "store_dir_bytes": stored_bytes,
        "search_documents": summary(raw_ns), "recall_default": summary(recall_ns),
        "recall_explain": summary(explain_ns),
        "recall_history_all": if history_ns.is_empty() { Value::Null } else { summary(history_ns) },
        "identical_ranking_queries": same_top, "compared_queries": compared,
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(kind) = args.child.clone() {
        ensure!(
            ["raw", "memory", "history"].contains(&kind.as_str()),
            "unknown kind"
        );
        println!("{}", child(&args, &kind)?);
        return Ok(());
    }
    let exe = std::env::current_exe()?;
    let mut runs = Vec::new();
    for &n in &args.sizes {
        for kind in ["raw", "memory", "history"] {
            let output = std::process::Command::new(&exe)
                .args(["--child", kind, "--n", &n.to_string()])
                .args(["--queries", &args.queries.to_string()])
                .args(["--passes", &args.passes.to_string()])
                .args(["--dimensions", &args.dimensions.to_string()])
                .output()
                .context("child failed to start")?;
            ensure!(
                output.status.success(),
                "child {kind} n={n} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let line = String::from_utf8(output.stdout)?;
            runs.push(serde_json::from_str::<Value>(line.trim())?);
            eprintln!("done {kind} n={n}");
        }
    }
    let report = json!({
        "synthetic": true,
        "description": "memory API recall against document search on the same synthetic fact store; see docs/memory-api",
        "words_per_text": WORDS, "vocabulary": VOCABULARY,
        "queries": args.queries, "passes": args.passes, "dimensions": args.dimensions,
        "host": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH,
                 "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)},
        "runs": runs,
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    match args.out {
        Some(path) => std::fs::write(path, rendered)?,
        None => println!("{rendered}"),
    }
    Ok(())
}
