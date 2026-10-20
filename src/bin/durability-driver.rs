// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Driver for `benchmarks/public/durability_matrix.py`: writes deterministic documents in
//! committed batches (announcing each commit on stdout), compacts on request, verifies a
//! store after a crash, and exercises readers during compaction.
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use holographic_memory::{
    core::{EmbeddingSpace, HmsConfig},
    DocumentInput, HmsCore, SearchOptions,
};
use serde_json::json;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

const DIM: usize = 128;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    /// Write `docs` documents in batches, printing `committed <batch>` after each commit and
    /// `compacting <batch>` / `compacted <batch>` around every `compact_every`-th batch.
    Write {
        #[arg(long)]
        store: String,
        #[arg(long, default_value_t = 4000)]
        docs: usize,
        #[arg(long, default_value_t = 20)]
        batch: usize,
        #[arg(long, default_value_t = 0)]
        compact_every: usize,
        /// Revision suffix: a second pass rewrites the same ids with new text.
        #[arg(long, default_value_t = 0)]
        revision: usize,
    },
    /// Open the store, then check that every document of the first `committed` batches is
    /// present with its payload; prints one JSON line.
    Verify {
        #[arg(long)]
        store: String,
        #[arg(long, default_value_t = 4000)]
        docs: usize,
        #[arg(long, default_value_t = 20)]
        batch: usize,
        #[arg(long)]
        committed: usize,
        #[arg(long, default_value_t = 0)]
        revision: usize,
    },
    /// Readers verify documents while a writer rewrites batches and a compaction runs.
    Readers {
        #[arg(long)]
        store: String,
        #[arg(long, default_value_t = 4000)]
        docs: usize,
        #[arg(long, default_value_t = 20)]
        batch: usize,
        #[arg(long, default_value_t = 3)]
        readers: usize,
    },
}

fn config() -> HmsConfig {
    HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "synthetic-one-hot-control".into(),
            revision: "1".into(),
            dimensions: DIM as _,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    }
}

fn embedding(n: usize) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    v[n % DIM] = 1.0;
    v
}

fn payload(n: usize, revision: usize) -> String {
    // A unique marker per document and revision; deterministic across processes.
    let h = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (revision as u64) << 48;
    format!("document {n} revision {revision} marker {h:016x} text body filler words")
}

fn input(n: usize, revision: usize) -> DocumentInput {
    DocumentInput {
        id: format!("d{n:06}"),
        text: payload(n, revision),
        source_uri: Some(format!("durability://{n}")),
        version: Some(revision.to_string()),
        metadata: None,
        chunk_words: None,
        overlap_words: None,
        store_text: Some(true),
        embeddings: Some(vec![embedding(n)]),
    }
}

/// True when document `n` is served with its exact payload of `revision`.
fn present(core: &HmsCore, n: usize, revision: usize) -> Result<bool> {
    let options = SearchOptions {
        k: Some(1),
        candidate_limit: Some(1),
        document_ids: Some(vec![format!("d{n:06}")]),
        embedding: Some(embedding(n)),
        ..Default::default()
    };
    let results = core.search_documents("durability", &options)?;
    let text = serde_json::to_string(&results)?;
    Ok(text.contains(&format!("\"d{n:06}\"")) && text.contains(&payload(n, revision)))
}

fn main() -> Result<()> {
    match Args::parse().mode {
        Mode::Write {
            store,
            docs,
            batch,
            compact_every,
            revision,
        } => {
            let core = HmsCore::new(1024, Some(store), Some(config()))?;
            let mut out = std::io::stdout().lock();
            for (b, first) in (0..docs).step_by(batch).enumerate() {
                let inputs: Vec<DocumentInput> = (first..(first + batch).min(docs))
                    .map(|n| input(n, revision))
                    .collect();
                for input in inputs {
                    core.memorize_document(input)?;
                }
                writeln!(out, "committed {b}")?;
                out.flush()?;
                if compact_every > 0 && (b + 1) % compact_every == 0 {
                    writeln!(out, "compacting {b}")?;
                    out.flush()?;
                    core.compact()?;
                    writeln!(out, "compacted {b}")?;
                    out.flush()?;
                }
            }
            core.flush()?;
            writeln!(out, "done")?;
            Ok(())
        }
        Mode::Verify {
            store,
            docs,
            batch,
            committed,
            revision,
        } => {
            let t = Instant::now();
            let opened = HmsCore::new(1024, Some(store.clone()), Some(config()));
            let recovery_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut out = std::io::stdout().lock();
            let core = match opened {
                Ok(core) => core,
                Err(e) => {
                    writeln!(
                        out,
                        "{}",
                        json!({"open_ok": false, "error": e.to_string(), "recovery_ms": recovery_ms})
                    )?;
                    return Ok(());
                }
            };
            let (mut present_n, mut missing, mut mismatched, mut extra) = (0, 0, 0, 0);
            let limit = (committed * batch).min(docs);
            let mut errors = 0;
            for n in 0..limit {
                match present(&core, n, revision) {
                    Ok(true) => present_n += 1,
                    Ok(false) => {
                        // Distinguish an absent document from one served with other text.
                        let r = if revision > 0 {
                            present(&core, n, revision - 1)
                        } else {
                            Ok(false)
                        };
                        if r? {
                            missing += 1;
                        } else {
                            let options = SearchOptions {
                                k: Some(1),
                                candidate_limit: Some(1),
                                document_ids: Some(vec![format!("d{n:06}")]),
                                embedding: Some(embedding(n)),
                                ..Default::default()
                            };
                            let any = core.search_documents("durability", &options)?;
                            if any.is_empty() {
                                missing += 1;
                            } else {
                                mismatched += 1;
                            }
                        }
                    }
                    Err(_) => errors += 1,
                }
            }
            for n in limit..docs {
                if present(&core, n, revision).unwrap_or(false) {
                    extra += 1;
                }
            }
            let bytes = dir_bytes(&store)?;
            writeln!(
                out,
                "{}",
                json!({"open_ok": true, "recovery_ms": recovery_ms, "checked": limit,
                    "present": present_n, "missing": missing, "mismatched": mismatched,
                    "read_errors": errors, "uncommitted_present": extra, "store_bytes": bytes})
            )?;
            Ok(())
        }
        Mode::Readers {
            store,
            docs,
            batch,
            readers,
        } => {
            let core = HmsCore::new(1024, Some(store), Some(config()))?;
            let stop = AtomicBool::new(false);
            let checks = AtomicUsize::new(0);
            let bad = AtomicUsize::new(0);
            let report = std::thread::scope(|scope| -> Result<serde_json::Value> {
                let core = &core;
                let (stop, checks, bad) = (&stop, &checks, &bad);
                for r in 0..readers {
                    scope.spawn(move || {
                        let mut n = r;
                        while !stop.load(Ordering::Relaxed) {
                            // Either revision is consistent; anything else is a torn read.
                            let ok = present(core, n, 1).unwrap_or(false)
                                || present(core, n, 0).unwrap_or(false);
                            checks.fetch_add(1, Ordering::Relaxed);
                            if !ok {
                                bad.fetch_add(1, Ordering::Relaxed);
                            }
                            n = (n + readers) % docs;
                        }
                    });
                }
                let writer = scope.spawn(move || -> Result<usize> {
                    let mut batches = 0;
                    for first in (0..docs).step_by(batch) {
                        let inputs: Vec<DocumentInput> = (first..(first + batch).min(docs))
                            .map(|n| input(n, 1))
                            .collect();
                        for input in inputs {
                            core.memorize_document(input)?;
                        }
                        batches += 1;
                    }
                    Ok(batches)
                });
                let t = Instant::now();
                core.compact()?;
                let compaction_ms = t.elapsed().as_secs_f64() * 1e3;
                let batches = writer.join().expect("writer")?;
                core.compact()?;
                stop.store(true, Ordering::Relaxed);
                Ok(json!({"batches": batches, "compaction_ms": compaction_ms}))
            })?;
            ensure!(checks.load(Ordering::Relaxed) > 0, "readers made no checks");
            println!(
                "{}",
                json!({"reads": checks.load(Ordering::Relaxed), "torn_reads": bad.load(Ordering::Relaxed),
                    "writer": report})
            );
            Ok(())
        }
    }
}

fn dir_bytes(path: &str) -> Result<u64> {
    let mut total = 0;
    for entry in walkdir(path)? {
        total += entry;
    }
    Ok(total)
}

fn walkdir(path: &str) -> Result<Vec<u64>> {
    let mut sizes = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(path)];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                sizes.push(meta.len());
            }
        }
    }
    Ok(sizes)
}
