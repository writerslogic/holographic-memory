// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use clap::Parser;
use serde::Deserialize;
use std::{path::PathBuf, time::Instant};
use writerslogic_warm::{DocumentInput, SearchOptions, WarmCore};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    store: PathBuf,
    #[arg(long)]
    out: PathBuf,
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

fn main() -> Result<()> {
    let args = Args::parse();
    let input: Input = serde_json::from_slice(&std::fs::read(args.input)?)?;
    ensure!(input.documents.len() <= 10_000 && input.queries.len() <= 2_000);
    let ingest = !args.store.exists();
    let start = Instant::now();
    let core = WarmCore::new(1024, Some(args.store.display().to_string()), None)?;
    if ingest {
        for document in &input.documents {
            ensure!(core.memorize_document(document.clone())? == 1);
        }
    }
    let open_ns = start.elapsed().as_nanos();
    for query in input.queries.iter().take(16) {
        core.search_documents(&query.text, &query.options)?;
    }
    let mut rows = Vec::with_capacity(input.queries.len());
    for query in &input.queries {
        let start = Instant::now();
        let results = core.search_documents(&query.text, &query.options)?;
        let elapsed_ns = start.elapsed().as_nanos();
        rows.push(serde_json::json!({
            "id": query.id, "elapsed_ns": elapsed_ns, "results": results,
        }));
    }
    std::fs::write(
        args.out,
        serde_json::to_vec(&serde_json::json!({
            "documents": input.documents.len(), "open_ns": open_ns,
            "ingest": ingest, "rows": rows,
        }))?,
    )?;
    Ok(())
}
