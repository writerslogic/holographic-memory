// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use clap::Parser;
use serde_json::json;
use std::{path::PathBuf, time::Instant};
use writerslogic_warm::{core::WarmConfig, DocumentInput, EmbeddingSpace, SearchOptions, WarmCore};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 10000)]
    documents: usize,
    #[arg(long)]
    out: PathBuf,
}

fn vector(id: usize) -> Vec<f64> {
    (0..384)
        .map(|j| (((id + 1) * 17 + j * 31) as f64).sin())
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (100..=100000).contains(&args.documents),
        "documents outside 100..=100000"
    );
    let directory = tempfile::tempdir()?;
    let config = WarmConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "synthetic-sine".into(),
            revision: "1".into(),
            dimensions: 384,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    let warm = WarmCore::new(
        1024,
        Some(directory.path().to_string_lossy().into_owned()),
        Some(config),
    )?;
    let start = Instant::now();
    for id in 0..args.documents {
        let input: DocumentInput = serde_json::from_value(json!({"id":format!("d{id:08}"),
            "text":format!("common text running topic{} rare{id} alpha beta gamma delta epsilon",id%100),
            "metadata":{"group":id%100}, "version":"1", "embeddings":[vector(id)]}))?;
        warm.memorize_document(input)?;
    }
    let ingest_seconds = start.elapsed().as_secs_f64();
    let mut rows = Vec::new();
    for repetition in 0..3 {
        for query_id in 0..32 {
            for filtered in [false, true] {
                for mode in ["lexical", "semantic", "hybrid"] {
                    let query = if mode == "semantic" {
                        String::new()
                    } else {
                        format!("running topic{} rare{}", query_id % 100, query_id * 13)
                    };
                    let options = SearchOptions {
                        filter: filtered.then(|| json!({"group":query_id%100})),
                        embedding: (mode != "lexical").then(|| vector(query_id * 13)),
                        lexical_weight: Some(if mode == "semantic" { 0.0 } else { 1.0 }),
                        ..Default::default()
                    };
                    let start = Instant::now();
                    let found = warm.search_documents(&query, &options)?;
                    rows.push(json!({"repetition":repetition,"query":query_id,"filtered":filtered,"mode":mode,
                        "elapsed_ns":start.elapsed().as_nanos(),"results":found}));
                }
            }
        }
    }
    std::fs::write(
        args.out,
        serde_json::to_vec(&json!({"synthetic":true,"documents":args.documents,
        "dimensions":384,"ingest_seconds":ingest_seconds,"rows":rows}))?,
    )?;
    Ok(())
}
