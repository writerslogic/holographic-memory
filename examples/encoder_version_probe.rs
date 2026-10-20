// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use clap::Parser;
use holographic_memory::core::{admin::inspect_store, encoding::TextEncoder, HmsConfig};
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, io::Read, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    store: PathBuf,
    #[arg(long)]
    dimensions: u32,
    #[arg(long)]
    encoder: TextEncoder,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    documents: Vec<DocumentInput>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Query {
    id: String,
    text: String,
    relevant_ids: Vec<String>,
}

fn file_sizes(root: &std::path::Path, path: &std::path::Path) -> Result<BTreeMap<String, u64>> {
    let mut sizes = BTreeMap::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        ensure!(!kind.is_symlink(), "unexpected store symlink");
        if kind.is_dir() {
            sizes.extend(file_sizes(root, &entry.path())?);
        } else {
            ensure!(kind.is_file(), "unexpected special file");
            sizes.insert(
                entry.path().strip_prefix(root)?.display().to_string(),
                entry.metadata()?.len(),
            );
        }
    }
    Ok(sizes)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.store.exists() && !args.output.exists(),
        "fresh paths required"
    );
    ensure!((1..=65536).contains(&args.dimensions), "dimension bounds");
    let mut bytes = Vec::new();
    std::fs::File::open(&args.input)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 16 * 1024 * 1024, "input exceeds 16 MiB");
    let input: Input = serde_json::from_slice(&bytes)?;
    ensure!(
        (1..=512).contains(&input.documents.len()) && (1..=128).contains(&input.queries.len()),
        "input count bounds"
    );
    let config = HmsConfig {
        text_encoder: args.encoder,
        ..Default::default()
    };
    let path = args.store.display().to_string();
    let started = Instant::now();
    let hms = HmsCore::new(args.dimensions, Some(path.clone()), Some(config.clone()))?;
    let init_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    for document in &input.documents {
        ensure!(
            holographic_memory::chunk_document(document)?.len() == 1,
            "one chunk per source"
        );
        hms.memorize_document(document.clone())?;
    }
    hms.flush()?;
    let ingest_ns = started.elapsed().as_nanos();
    let mut queries = Vec::new();
    for query in &input.queries {
        ensure!(query.text.len() <= 64 * 1024, "query exceeds 64 KiB");
        let started = Instant::now();
        let vector = hms.encode_text(&query.text);
        let results = hms.query(&vector, 10);
        let query_ns = started.elapsed().as_nanos();
        queries.push(json!({
            "id": query.id, "text": query.text, "relevant_ids": query.relevant_ids,
            "indices": vector.indices(), "results": results, "query_ns": query_ns,
            "lexical": hms.search_documents(&query.text, &SearchOptions::default())?,
        }));
    }
    let vectors: Vec<_> = input
        .documents
        .iter()
        .map(|document| {
            let chunk = holographic_memory::chunk_document(document)?.remove(0);
            Ok(json!({"id": chunk.id, "document_id": document.id,
            "indices": hms.encode_text(&chunk.text).indices()}))
        })
        .collect::<Result<_>>()?;
    let before = inspect_store(&args.store, true)?;
    let before_files = file_sizes(&args.store, &args.store)?;
    let started = Instant::now();
    hms.compact()?;
    let compact_ns = started.elapsed().as_nanos();
    drop(hms);
    let started = Instant::now();
    let reopened = HmsCore::new(args.dimensions, Some(path), Some(config))?;
    let restart_ns = started.elapsed().as_nanos();
    let restart_results: Vec<_> = input
        .queries
        .iter()
        .map(|query| {
            Ok(json!({"id": query.id,
                "results": reopened.query(&reopened.encode_text(&query.text), 10),
                "lexical": reopened.search_documents(&query.text, &SearchOptions::default())?,
            }))
        })
        .collect::<Result<_>>()?;
    ensure!(
        reopened.vector_count() == vectors.len() as u64,
        "restart vector count"
    );
    drop(reopened);
    let after = inspect_store(&args.store, true)?;
    let after_files = file_sizes(&args.store, &args.store)?;
    let output = json!({
        "schema": 1, "encoder": args.encoder.name(), "dimensions": args.dimensions,
        "input_bytes": bytes.len(), "init_ns": init_ns, "ingest_ns": ingest_ns,
        "compact_ns": compact_ns, "restart_ns": restart_ns,
        "vectors": vectors, "queries": queries, "restart_results": restart_results,
        "before": before, "after": after, "before_files": before_files, "after_files": after_files,
    });
    std::fs::write(args.output, serde_json::to_vec_pretty(&output)?)?;
    Ok(())
}
