// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Context, Result};
use holographic_memory::{core::HmsConfig, DocumentInput, EmbeddingSpace, HmsCore, SearchOptions};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Dataset {
    documents: Vec<DocumentInput>,
    queries: Vec<Query>,
    embedding_space: Option<EmbeddingSpace>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Query {
    text: String,
    relevant_ids: BTreeSet<String>,
    embedding: Option<Vec<f64>>,
}

pub fn run(path: &Path, dimensions: u32, k: usize, minimum: f64) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let dataset: Dataset = serde_json::from_slice(&bytes).context("invalid relevance dataset")?;
    ensure!(
        !dataset.documents.is_empty() && !dataset.queries.is_empty(),
        "dataset must contain documents and held-out queries"
    );
    ensure!((1..=1000).contains(&k), "k must be 1..=1000");
    let ids: BTreeSet<_> = dataset.documents.iter().map(|d| d.id.clone()).collect();
    for query in &dataset.queries {
        ensure!(
            !query.relevant_ids.is_empty() && query.relevant_ids.is_subset(&ids),
            "query relevance labels must reference stored document IDs"
        );
    }
    let directory = tempfile::tempdir()?;
    let config = HmsConfig {
        embedding_space: dataset.embedding_space,
        ..Default::default()
    };
    let hms = HmsCore::new(
        dimensions,
        Some(directory.path().display().to_string()),
        Some(config),
    )?;
    let documents = dataset.documents.len();
    for document in dataset.documents {
        hms.memorize_document(document)?;
    }
    let mut recalls = Vec::new();
    let mut reciprocal_ranks = Vec::new();
    let mut ndcgs = Vec::new();
    let mut latency = Vec::new();
    for query in &dataset.queries {
        let started = Instant::now();
        let results = hms.search_documents(
            &query.text,
            &SearchOptions {
                k: Some(1000),
                candidate_limit: Some(1000),
                embedding: query.embedding.clone(),
                ..Default::default()
            },
        )?;
        latency.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        let mut seen = BTreeSet::new();
        let ranked: Vec<_> = results
            .iter()
            .filter(|r| seen.insert(r.document_id.clone()))
            .take(k)
            .collect();
        let relevant_ranks: Vec<_> = ranked
            .iter()
            .enumerate()
            .filter(|(_, r)| query.relevant_ids.contains(&r.document_id))
            .map(|(i, _)| i + 1)
            .collect();
        recalls.push(relevant_ranks.len() as f64 / query.relevant_ids.len() as f64);
        reciprocal_ranks.push(
            relevant_ranks
                .first()
                .map_or(0.0, |&rank| 1.0 / rank as f64),
        );
        let dcg: f64 = relevant_ranks
            .iter()
            .map(|&rank| 1.0 / (rank as f64 + 1.0).log2())
            .sum();
        let ideal: f64 = (1..=query.relevant_ids.len().min(k))
            .map(|rank| 1.0 / (rank as f64 + 1.0).log2())
            .sum();
        ndcgs.push(dcg / ideal);
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    latency.sort_by(f64::total_cmp);
    let recall = mean(&recalls);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 2, "evaluation": "document_relevance", "dataset": path.display().to_string(),
            "dataset_crc32": format!("{:08x}", crc32fast::hash(&bytes)), "hms_version": env!("CARGO_PKG_VERSION"),
            "architecture": std::env::consts::ARCH, "os": std::env::consts::OS,
            "documents": documents, "queries": dataset.queries.len(), "k": k,
            "recall_at_k": recall, "mrr_at_k": mean(&reciprocal_ranks), "ndcg_at_k": mean(&ndcgs),
            "mean_latency_us": mean(&latency), "p95_latency_us": super::percentile(&latency, 0.95),
        }))?
    );
    ensure!(
        recall >= minimum,
        "recall@{k} {recall:.4} is below required {minimum:.4}"
    );
    Ok(())
}
