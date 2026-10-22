// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use clap::Parser;
use holographic_memory::core::qgraph::{BuildParams, Graph, Metric, VGraph, VSearchParams};
use serde_json::{json, Value};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 4)]
    bits: u8,
    #[arg(long, default_value_t = 1000)]
    rerank: usize,
    #[arg(long, default_value_t = 2000)]
    ef: usize,
    #[arg(long, default_value_t = 0)]
    patience: usize,
    #[arg(long)]
    training: Option<PathBuf>,
    #[arg(long, default_value_t = 8192)]
    training_count: usize,
    #[arg(long, default_value_t = 16)]
    projected_degree: usize,
    #[arg(long)]
    profile: bool,
    #[arg(long)]
    persist_dir: Option<PathBuf>,
}

fn floats(path: PathBuf, expected: usize) -> Result<Vec<f32>> {
    ensure!(expected <= (1 << 29), "matrix exceeds diagnostic bound");
    ensure!(
        fs::metadata(&path)?.len() == (expected as u64) * 4,
        "matrix size mismatch"
    );
    let bytes = fs::read(path)?;
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite matrix");
    Ok(values)
}

fn coverage(ids: &[u32], truth: &[u32]) -> f64 {
    truth.iter().filter(|id| ids.contains(id)).count() as f64 / truth.len() as f64
}

fn exact_rerank(ids: &[u32], base: &[f32], query: &[f32], dim: usize) -> Vec<u32> {
    let mut scored: Vec<_> = ids
        .iter()
        .map(|&id| {
            let product = base[id as usize * dim..(id as usize + 1) * dim]
                .iter()
                .zip(query)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>();
            (product, id)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.iter().take(100).map(|v| v.1).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "output exists");
    ensure!([4, 8].contains(&args.bits), "bits must be 4 or 8");
    ensure!(
        (100..=2000).contains(&args.rerank),
        "rerank must be in 100..=2000"
    );
    ensure!((100..=2000).contains(&args.ef), "ef must be in 100..=2000");
    let meta: Value = serde_json::from_slice(&fs::read(args.data.join("meta.json"))?)?;
    ensure!(
        meta["source_metric"] == "ip" && meta["official_test_read"] == false,
        "requires validation-only original inner-product data"
    );
    let dim = meta["dimension"]
        .as_u64()
        .filter(|&v| (1..=8192).contains(&v))
        .context("dimension")? as usize;
    let n = meta["rows"]
        .as_u64()
        .filter(|&v| (100..=1_000_000).contains(&v))
        .context("rows")? as usize;
    let nq = meta["validation_queries"]
        .as_u64()
        .filter(|&v| (1..=1000).contains(&v))
        .context("queries")? as usize;
    let base = floats(
        args.data.join("base.f32"),
        n.checked_mul(dim).context("shape overflow")?,
    )?;
    let queries = floats(args.data.join("validation.f32"), nq * dim)?;
    let truth_bytes = fs::read(args.data.join("truth.u32"))?;
    ensure!(truth_bytes.len() == nq * 100 * 4, "truth shape mismatch");
    let truth: Vec<_> = truth_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| u32::from_le_bytes(*v))
        .collect();
    ensure!(truth.iter().all(|&v| (v as usize) < n), "invalid truth ID");
    for row in truth.as_chunks::<100>().0 {
        let mut unique = row.to_vec();
        unique.sort_unstable();
        unique.dedup();
        ensure!(unique.len() == 100, "duplicate truth IDs");
    }
    let norms: Vec<f64> = base
        .chunks_exact(dim)
        .map(|row| row.iter().map(|&v| f64::from(v).powi(2)).sum())
        .collect();
    let maximum = norms.iter().copied().fold(0., f64::max);
    let mut augmented = Vec::with_capacity(n * (dim + 1));
    for (row, norm) in base.chunks_exact(dim).zip(norms) {
        augmented.extend_from_slice(row);
        augmented.push((maximum - norm).max(0.).sqrt() as f32);
    }
    let params = BuildParams {
        metric: Metric::Euclidean,
        degree: 64,
        build_ef: 500,
        quantized_search: true,
        ..Default::default()
    };
    let start = Instant::now();
    let cache_hit = args.graph.exists();
    let mut graph = if cache_hit {
        Graph::load(&augmented, dim + 1, &args.graph, &params)?
    } else {
        let graph = Graph::build_owned(augmented, dim + 1, &params)?;
        graph.save(&args.graph)?;
        graph
    };
    let graph_seconds = start.elapsed().as_secs_f64();
    let base_graph_hash = graph.structure_hash();
    let start = Instant::now();
    let mut projection = Value::Null;
    if let Some(training) = &args.training {
        ensure!(
            (1..=32768).contains(&args.training_count),
            "invalid training count"
        );
        ensure!(
            (1..=64).contains(&args.projected_degree),
            "invalid projected degree"
        );
        let training_meta: Value = serde_json::from_slice(&fs::read(training.join("meta.json"))?)?;
        ensure!(
            training_meta["source_sha256"] == meta["source_sha256"]
                && training_meta["dimension"] == meta["dimension"]
                && training_meta["rows"] == meta["rows"]
                && training_meta["neighbors"] == 100
                && training_meta["official_test_read"] == false,
            "training provenance mismatch"
        );
        let available = training_meta["queries"]
            .as_u64()
            .filter(|&v| v <= 32768 && v >= args.training_count as u64)
            .context("training rows")? as usize;
        let values = floats(training.join("queries.f32"), available * dim)?;
        let mut transformed = Vec::with_capacity(args.training_count * (dim + 1));
        for row in values.chunks_exact(dim).take(args.training_count) {
            transformed.extend_from_slice(row);
            transformed.push(0.);
        }
        let bytes = fs::read(training.join("neighbors.u32"))?;
        ensure!(
            bytes.len() == available * 100 * 4,
            "training neighbor shape mismatch"
        );
        let ids: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(args.training_count * 100)
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        if args.profile {
            ensure!(
                args.training_count == 32768 && args.projected_degree == 32,
                "production profile requires frozen training settings"
            );
            graph.train_diagnostic(&transformed)?;
            projection = json!({"input": training_meta, "routine": "production training_neighbors and projection"});
        } else {
            projection = json!({"input": training_meta, "stats": graph.project_diagnostic(
                Some(&transformed), &ids, 100, args.projected_degree)?});
        }
    }
    let projection_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut index = VGraph::from_graph_with(&graph, args.bits, 8, true, true, 3, false)?;
    let encoding_seconds = start.elapsed().as_secs_f64();
    let settings = VSearchParams {
        ef: args.ef,
        rerank: args.rerank,
        patience: args.patience,
        ..Default::default()
    };
    if args.profile {
        let transformed: Vec<f32> = queries
            .chunks_exact(dim)
            .flat_map(|q| q.iter().copied().chain([0.0]))
            .collect();
        let mut reference = Vec::new();
        let mut layouts = Vec::new();
        let mut persisted = Vec::new();
        if let Some(directory) = &args.persist_dir {
            fs::create_dir(directory)?;
        }
        for compact in [false, true] {
            let started = Instant::now();
            if compact {
                index.compact_adjacency()?;
            }
            let compaction_seconds = started.elapsed().as_secs_f64();
            let mut saved = Value::Null;
            if let Some(directory) = &args.persist_dir {
                let file = directory.join(if compact { "compact.hms" } else { "fixed.hms" });
                let started = Instant::now();
                index.save(&file)?;
                let save_seconds = started.elapsed().as_secs_f64();
                let started = Instant::now();
                let reopened = VGraph::open(
                    &file,
                    holographic_memory::core::qgraph::Residency::Prefaulted,
                )?;
                let open_seconds = started.elapsed().as_secs_f64();
                drop(reopened);
                saved = json!({"path": file, "bytes": fs::metadata(&file)?.len(), "save_seconds": save_seconds,
                    "open_prefaulted_seconds": open_seconds});
                persisted.push(file);
            }
            let mut searcher =
                index.searcher_with(holographic_memory::core::qgraph::SearchKnobs::OFF);
            let mut out = Vec::new();
            for _ in 0..4 {
                for query in transformed.chunks_exact(dim + 1) {
                    searcher.search(query, 100, settings, &mut out)?;
                }
            }
            let mut work = Vec::new();
            let mut hits = 0;
            for (i, (query, expected)) in transformed
                .chunks_exact(dim + 1)
                .zip(truth.chunks_exact(100))
                .enumerate()
            {
                let profile = searcher.profile_search(query, 100, settings, &mut out)?;
                hits += out.iter().filter(|id| expected.contains(id)).count();
                if compact {
                    ensure!(out == reference[i], "layout changed query {i}");
                } else {
                    reference.push(out.clone());
                }
                work.push(profile);
            }
            layouts.push(
                json!({"compact": compact, "index_bytes": index.index_bytes(),
                "compaction_seconds": compaction_seconds, "persisted": saved,
                "recall": hits as f64 / (nq * 100) as f64, "queries": work}),
            );
        }
        if persisted.len() == 2 {
            use std::io::Read;
            let mut a = fs::File::open(&persisted[0])?;
            let mut b = fs::File::open(&persisted[1])?;
            let (mut aa, mut bb) = ([0u8; 65536], [0u8; 65536]);
            loop {
                let (na, nb) = (a.read(&mut aa)?, b.read(&mut bb)?);
                ensure!(
                    na == nb && aa[..na] == bb[..nb],
                    "compaction changed persisted bytes"
                );
                if na == 0 {
                    break;
                }
            }
        }
        let report = json!({"schema_version": 1,
            "scope": "validation-only instrumented production traversal; stage timers are not QPS evidence",
            "input": meta, "projection": projection, "graph_hash": graph.structure_hash(),
            "graph_cache_hit": cache_hit, "encoding_seconds": encoding_seconds,
            "settings": {"bits": args.bits, "residual_bits": 8, "ef": args.ef,
                "rerank": args.rerank, "patience": args.patience},
            "persisted_bytes_identical": persisted.len() == 2, "returned_ids_identical": true,
            "unavailable": ["cycles", "instructions", "cache misses", "bandwidth", "allocations",
                "separate distance and queue time", "Python wrapper time"], "layouts": layouts});
        fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
        println!("{}", args.out.display());
        return Ok(());
    }
    let mut rows = Vec::new();
    for (i, (query, expected)) in queries
        .chunks_exact(dim)
        .zip(truth.as_chunks::<100>().0)
        .enumerate()
    {
        let mut transformed = query.to_vec();
        transformed.push(0.);
        for (name, oracle, exact) in [
            ("B", None, false),
            ("O", Some(expected[0]), false),
            ("E", None, true),
            ("OE", Some(expected[0]), true),
        ] {
            let found = index.diagnose(&graph, &transformed, 100, settings, oracle, exact)?;
            let reranked = exact_rerank(&found.rerank_candidates, &base, query, dim);
            let all_retained = exact_rerank(&found.retained, &base, query, dim);
            rows.push(json!({"query": i, "arm": name, "recall": coverage(&found.results, expected),
                "evaluated_coverage": coverage(&found.evaluated, expected),
                "evaluated_unique": found.evaluated.len(), "retained_coverage": coverage(&found.retained, expected),
                "rerank_candidate_coverage": coverage(&found.rerank_candidates, expected),
                "exact_rerank_recall": coverage(&reranked, expected),
                "exact_all_retained_recall": coverage(&all_retained, expected), "ids": found.results}));
        }
    }
    let mut aggregate = serde_json::Map::new();
    for arm in ["B", "O", "E", "OE"] {
        let current: Vec<_> = rows.iter().filter(|row| row["arm"] == arm).collect();
        let mut mean = serde_json::Map::new();
        for field in [
            "recall",
            "evaluated_coverage",
            "evaluated_unique",
            "retained_coverage",
            "rerank_candidate_coverage",
            "exact_rerank_recall",
            "exact_all_retained_recall",
        ] {
            mean.insert(
                field.into(),
                json!(
                    current
                        .iter()
                        .map(|r| r[field].as_f64().expect("number"))
                        .sum::<f64>()
                        / nq as f64
                ),
            );
        }
        aggregate.insert(arm.into(), Value::Object(mean));
    }
    let report = json!({"schema_version": 1, "scope": "untimed validation diagnosis; no official test queries",
        "input": meta, "build": {"degree": 64, "build_ef": 500, "seed": params.seed,
            "quantized_search": true, "metric": "euclidean_asymmetric_mips", "graph_hash": graph.structure_hash(),
            "cache_hit": cache_hit, "graph_seconds": graph_seconds, "encoding_seconds": encoding_seconds,
            "base_graph_hash": base_graph_hash, "projection_seconds": projection_seconds, "projection": projection,
            "index_bytes": index.index_bytes(), "graph_cache_bytes": fs::metadata(&args.graph)?.len()},
        "search": {"ef": args.ef, "rerank": args.rerank, "patience": args.patience, "bits": args.bits, "residual_bits": 8, "rotation": false,
            "exact_f32_rerank": false, "knobs": "OFF", "k": 100},
        "aggregate": aggregate, "queries": rows});
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string(&report["aggregate"])?);
    Ok(())
}
