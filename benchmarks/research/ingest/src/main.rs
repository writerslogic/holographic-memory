// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Ingest timing of the three `IngestOptions::encode` modes, with search parity against a
//! store that encoded at ingest.
//!
//! Input: the `{documents, queries}` JSON `hms-bench` writes (the file
//! `examples/competitive_text.rs build` reads): `documents` are `DocumentInput`s, `queries`
//! are `{id, text, options}` with `SearchOptions`.
//!
//! Mechanism: after `--warmups` discarded runs (default none), each of `--repeats` runs
//! (default one) opens a fresh store in a temporary directory (`HmsCore::new(dim, path,
//! None)`, the harness configuration) and ingests the corpus in batches of `--batch`
//! documents with `memorize_documents_with`, timing the wall clock from before the open to
//! after the last commit, as the harness does. The encoder's memo tables are per-thread and
//! survive between builds of one process, so only the first timed build without warmups is
//! cold (`cold_ingest_s`, the figure comparable with `examples/competitive_text.rs build` in
//! its own process); later builds are reported as `warm_ingest_s`. In
//! `deferred` mode the catch-up (`encode_pending_documents`) is timed separately. The last
//! store is then compared with an eager reference store built from the same inputs: the
//! lexical results of the first `--parity-queries` queries (JSON of `search_documents` with
//! each query's own options) must be identical in every mode, and the vector results
//! (`query` over `encode_text` of the query, as a set of `(chunk id, similarity)`) must be
//! identical after the deferred catch-up and empty in lexical mode.
//!
//! Determinism: the encoder's memo tables are per-thread pure caches and the batch commit is
//! in input order, so stored vectors do not depend on the thread count; timings do.
//!
//! Output (`--out`): one JSON object
//! `{mode, dim, batch, documents, chunks, store_text, repeats, warmups, ingest_s: [f64],
//!   cold_ingest_s: f64|null, warm_ingest_s: [f64], median_warm_ingest_s: f64|null,
//!   median_ingest_s, catch_up_s: [f64], median_catch_up_s: f64|null, vectors_after_ingest,
//!   vectors_after_catch_up, parity: {queries, lexical_identical, vector_identical,
//!   lexical_mismatches: [query id], vector_mismatches: [query id]}}`.
//! A run passes when both parity flags are true; `median_ingest_s` is the figure to compare
//! with `INGEST-PROFILE.md`.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use holographic_memory::core::documents::{EncodeMode, IngestOptions};
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
struct Args {
    /// The hms-bench `{documents, queries}` JSON.
    #[arg(long)]
    input: PathBuf,
    /// eager | deferred | lexical
    #[arg(long, default_value = "eager")]
    encode: String,
    /// Timed builds in this process. The first one with no warmups is the cold figure (the
    /// encoder's per-thread memo tables are empty, as in a fresh ingest process); later ones
    /// are warm and reported separately.
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    /// Discarded builds before the timed ones; any warmup makes every timed build warm.
    #[arg(long, default_value_t = 0)]
    warmups: usize,
    /// Documents per `memorize_documents_with` call (one transaction each).
    #[arg(long, default_value_t = 512)]
    batch: usize,
    #[arg(long, default_value_t = 1024)]
    dim: usize,
    /// auto | true | false: whether documents store their text. `auto` keeps the input's
    /// setting except in deferred mode, which needs the text and forces it on.
    #[arg(long, default_value = "auto")]
    store_text: String,
    #[arg(long, default_value_t = 16)]
    parity_queries: usize,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Deserialize)]
struct Input {
    documents: Vec<DocumentInput>,
    #[serde(default)]
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Query {
    id: String,
    text: String,
    options: SearchOptions,
}

#[derive(Serialize)]
struct Parity {
    queries: usize,
    lexical_identical: bool,
    vector_identical: bool,
    lexical_mismatches: Vec<String>,
    vector_mismatches: Vec<String>,
}

#[derive(Serialize)]
struct Report {
    mode: String,
    dim: usize,
    batch: usize,
    documents: usize,
    chunks: u64,
    store_text: Option<bool>,
    repeats: usize,
    warmups: usize,
    /// Every timed build, in order.
    ingest_s: Vec<f64>,
    /// The first timed build when nothing warmed the process: the figure comparable with a
    /// single-process ingest (`examples/competitive_text.rs build`). Null after warmups.
    cold_ingest_s: Option<f64>,
    /// Timed builds after the first, in this process, with the encoder memo tables warm.
    warm_ingest_s: Vec<f64>,
    median_warm_ingest_s: Option<f64>,
    /// Median over every timed build (equals `cold_ingest_s` at the default one repeat).
    median_ingest_s: f64,
    catch_up_s: Vec<f64>,
    median_catch_up_s: Option<f64>,
    vectors_after_ingest: u64,
    vectors_after_catch_up: u64,
    parity: Parity,
}

struct Built {
    _dir: tempfile::TempDir,
    core: HmsCore,
    ingest_s: f64,
    chunks: u64,
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    })
}

fn build(documents: &[DocumentInput], dim: usize, batch: usize, options: &IngestOptions) -> Result<Built> {
    let dir = tempfile::tempdir().context("temporary store directory")?;
    let start = Instant::now();
    let core = HmsCore::new(dim as u32, Some(dir.path().display().to_string()), None)?;
    let mut chunks = 0u64;
    for group in documents.chunks(batch) {
        let counts = core.memorize_documents_with(group.to_vec(), options)?;
        chunks += counts.iter().map(|&count| u64::from(count)).sum::<u64>();
    }
    let ingest_s = start.elapsed().as_secs_f64();
    Ok(Built {
        _dir: dir,
        core,
        ingest_s,
        chunks,
    })
}

fn vector_hits(core: &HmsCore, text: &str) -> BTreeSet<(String, u64)> {
    core.query(&core.encode_text(text), 4096)
        .into_iter()
        .map(|hit| (hit.id, hit.similarity.to_bits()))
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repeats >= 1, "repeats must be at least 1");
    ensure!(args.batch >= 1, "batch must be at least 1");
    let mode: EncodeMode = args.encode.parse()?;
    let options = IngestOptions { encode: mode };
    let bytes = std::fs::read(&args.input)
        .with_context(|| format!("read {}", args.input.display()))?;
    let mut input: Input = serde_json::from_slice(&bytes).context("parse input JSON")?;
    let store_text = match args.store_text.as_str() {
        "auto" => (mode == EncodeMode::Deferred).then_some(true),
        "true" => Some(true),
        "false" => Some(false),
        other => anyhow::bail!("store-text must be auto, true or false, not {other:?}"),
    };
    if let Some(store_text) = store_text {
        for document in &mut input.documents {
            document.store_text = Some(store_text);
        }
    }

    for _ in 0..args.warmups {
        build(&input.documents, args.dim, args.batch, &options)?;
    }
    let mut ingest_s = Vec::with_capacity(args.repeats);
    let mut catch_up_s = Vec::new();
    let mut last: Option<Built> = None;
    let mut vectors_after_ingest = 0;
    for _ in 0..args.repeats {
        let built = build(&input.documents, args.dim, args.batch, &options)?;
        ingest_s.push(built.ingest_s);
        vectors_after_ingest = built.core.vector_count();
        if mode == EncodeMode::Deferred {
            let start = Instant::now();
            built.core.encode_pending_documents()?;
            catch_up_s.push(start.elapsed().as_secs_f64());
        }
        last = Some(built);
    }
    let last = last.context("at least one repeat")?;
    let vectors_after_catch_up = last.core.vector_count();

    let reference = build(
        &input.documents,
        args.dim,
        args.batch,
        &IngestOptions::default(),
    )?;
    let mut lexical_mismatches = Vec::new();
    let mut vector_mismatches = Vec::new();
    let queries: Vec<&Query> = input.queries.iter().take(args.parity_queries).collect();
    for query in &queries {
        let expected = serde_json::to_value(reference.core.search_documents(&query.text, &query.options)?)?;
        let actual = serde_json::to_value(last.core.search_documents(&query.text, &query.options)?)?;
        if expected != actual {
            lexical_mismatches.push(query.id.clone());
        }
        let expected_vectors = match mode {
            EncodeMode::Lexical => BTreeSet::new(),
            EncodeMode::Eager | EncodeMode::Deferred => vector_hits(&reference.core, &query.text),
        };
        if vector_hits(&last.core, &query.text) != expected_vectors {
            vector_mismatches.push(query.id.clone());
        }
    }

    let report = Report {
        mode: mode.name().to_string(),
        dim: args.dim,
        batch: args.batch,
        documents: input.documents.len(),
        chunks: last.chunks,
        store_text,
        repeats: args.repeats,
        warmups: args.warmups,
        cold_ingest_s: (args.warmups == 0).then(|| ingest_s[0]),
        median_warm_ingest_s: median(&ingest_s[1..]),
        warm_ingest_s: ingest_s[1..].to_vec(),
        median_ingest_s: median(&ingest_s).context("ingest timings")?,
        ingest_s,
        median_catch_up_s: median(&catch_up_s),
        catch_up_s,
        vectors_after_ingest,
        vectors_after_catch_up,
        parity: Parity {
            queries: queries.len(),
            lexical_identical: lexical_mismatches.is_empty(),
            vector_identical: vector_mismatches.is_empty(),
            lexical_mismatches,
            vector_mismatches,
        },
    };
    std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args.out.display()))?;
    println!(
        "{} median ingest {:.3} s over {} repeats ({} documents, {} chunks); catch-up {:?}; lexical parity {}, vector parity {}",
        report.mode,
        report.median_ingest_s,
        report.repeats,
        report.documents,
        report.chunks,
        report.median_catch_up_s,
        report.parity.lexical_identical,
        report.parity.vector_identical
    );
    Ok(())
}
