// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Reproducible retrieval-quality and latency regression harness.

use anyhow::{bail, Context, Result};
use holographic_memory::{EntangledHVec, HmsCore};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct Evaluation {
    schema_version: u32,
    vectors: usize,
    queries: usize,
    dimensions: usize,
    recall_at_1: f64,
    mean_latency_us: f64,
    p95_latency_us: f64,
    route: String,
}

fn value(args: &[String], flag: &str, default: usize) -> Result<usize> {
    match args.iter().position(|arg| arg == flag) {
        Some(index) => args
            .get(index + 1)
            .with_context(|| format!("missing value for {flag}"))?
            .parse()
            .with_context(|| format!("invalid value for {flag}")),
        None => Ok(default),
    }
}

fn float_value(args: &[String], flag: &str, default: f64) -> Result<f64> {
    match args.iter().position(|arg| arg == flag) {
        Some(index) => args
            .get(index + 1)
            .with_context(|| format!("missing value for {flag}"))?
            .parse()
            .with_context(|| format!("invalid value for {flag}")),
        None => Ok(default),
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len().saturating_sub(1)) as f64 * percentile).round() as usize;
    sorted.get(index).copied().unwrap_or_default()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let vectors = value(&args, "--vectors", 1_500)?;
    let queries = value(&args, "--queries", 100)?.min(vectors);
    let dimensions = value(&args, "--dimensions", 16_384)?;
    let minimum_recall = float_value(&args, "--assert-min-recall", 0.0)?;
    if vectors == 0 || dimensions == 0 {
        bail!("vectors and dimensions must be non-zero");
    }

    let directory = tempfile::tempdir()?;
    let hms = HmsCore::new(
        dimensions.try_into().context("dimensions exceed u32")?,
        Some(directory.path().to_string_lossy().into_owned()),
        None,
    )?;
    for seed in 0..vectors {
        hms.memorize(
            format!("vector-{seed}"),
            EntangledHVec::new_deterministic(dimensions, seed as u64),
        )?;
    }
    hms.train_nsg()?;

    let mut hits = 0usize;
    let mut latencies = Vec::with_capacity(queries);
    let stride = (vectors / queries.max(1)).max(1);
    let mut route = String::new();
    for query_index in 0..queries {
        let seed = (query_index * stride).min(vectors - 1);
        let query = EntangledHVec::new_deterministic(dimensions, seed as u64);
        route = hms.explain_query(&query, 1).route;
        let started = Instant::now();
        let result = hms.query(&query, 1);
        latencies.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        hits += usize::from(
            result
                .first()
                .is_some_and(|r| r.id == format!("vector-{seed}")),
        );
    }
    latencies.sort_by(f64::total_cmp);
    let recall = hits as f64 / queries.max(1) as f64;
    let report = Evaluation {
        schema_version: 1,
        vectors,
        queries,
        dimensions,
        recall_at_1: recall,
        mean_latency_us: latencies.iter().sum::<f64>() / latencies.len().max(1) as f64,
        p95_latency_us: percentile(&latencies, 0.95),
        route,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if recall < minimum_recall {
        bail!("recall@1 {recall:.4} is below required {minimum_recall:.4}");
    }
    Ok(())
}
