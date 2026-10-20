// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Context, Result};
use clap::Parser;
use holographic_memory::core::qgraph::{
    diagnostics::{residual_feedback, ExactSearcher},
    BuildParams, Graph, VGraph, VSearchParams,
};
use serde_json::{json, Value};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    dim: usize,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 24301)]
    seed: u64,
    #[arg(long, value_delimiter = ',', default_value = "32,64,128,256")]
    ef: Vec<usize>,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
}

fn read_rows(path: PathBuf, dim: usize) -> Result<Vec<f32>> {
    let width = dim.checked_mul(4).context("dimension overflow")?;
    ensure!(dim > 0 && dim <= 4096, "dimension outside 1..=4096");
    let size = fs::metadata(&path)?.len();
    ensure!(
        size > 0 && size <= 512 * 1024 * 1024 && size.is_multiple_of(width as u64),
        "input must contain complete rows, at most 512 MiB"
    );
    let bytes = fs::read(path)?;
    let rows: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    ensure!(rows.iter().all(|v| v.is_finite()), "nonfinite input");
    Ok(rows)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!((1..=11).contains(&args.repeats), "repeats outside 1..=11");
    ensure!(
        !args.ef.is_empty()
            && args.ef.len() <= 16
            && args.ef.iter().all(|&ef| (16..=4096).contains(&ef)),
        "ef outside 16..=4096 or too many settings"
    );
    let train = read_rows(args.data.join("train.f32"), args.dim)?;
    let queries = read_rows(args.data.join("queries.f32"), args.dim)?;
    let n = train.len() / args.dim;
    ensure!(
        n >= 10 && queries.len() / args.dim <= 10000,
        "require n>=10 and at most 10000 diagnostic queries"
    );
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let start = Instant::now();
    let graph = pool.install(|| {
        Graph::build(
            &train,
            args.dim,
            &BuildParams {
                degree: 32,
                build_ef: 200,
                alpha: 1.0,
                seed: args.seed,
                ..Default::default()
            },
        )
    })?;
    let graph_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let index = pool.install(|| VGraph::from_graph_with(&graph, 8, 8, true, false, 3, false))?;
    let encoding_seconds = start.elapsed().as_secs_f64();
    let mut compact = index.searcher();
    let mut exact = ExactSearcher::new(&graph);
    let mut candidates = Vec::new();
    let mut second = Vec::new();
    let mut found = Vec::new();
    let mut records: Vec<Value> = Vec::new();
    let arms = [
        "residual16",
        "original16",
        "original64",
        "original_pool",
        "exact_traversal",
        "feedback_union",
        "double_ef",
        "flat",
    ];
    for repetition in 0..args.repeats {
        for (qi, query) in queries.chunks_exact(args.dim).enumerate() {
            for &ef in &args.ef {
                for offset in 0..arms.len() {
                    let arm = arms[(offset + repetition + qi) % arms.len()];
                    let params = VSearchParams {
                        ef,
                        rerank: 0,
                        patience: 0,
                        screen: false,
                        ..Default::default()
                    };
                    let start = Instant::now();
                    let (evaluations, coverage_candidates) = match arm {
                        "residual16" => {
                            let count = compact.search(
                                query,
                                10,
                                VSearchParams {
                                    rerank: 16,
                                    ..params
                                },
                                &mut found,
                            )?;
                            (count, None)
                        }
                        "exact_traversal" => (exact.search(query, 10, ef, &mut found)?, None),
                        "flat" => (exact.flat(query, 10, &mut found)?, None),
                        _ => {
                            let beam = if arm == "double_ef" { 2 * ef } else { ef }.min(n);
                            let mut count = compact.search(
                                query,
                                beam,
                                VSearchParams { ef: beam, ..params },
                                &mut candidates,
                            )?;
                            if arm == "feedback_union" {
                                let probe = residual_feedback(query)?;
                                count += compact.search(&probe, beam, params, &mut second)?;
                                candidates.extend_from_slice(&second);
                                candidates.sort_unstable();
                                candidates.dedup();
                            }
                            let take = match arm {
                                "original16" => 16,
                                "original64" => 64,
                                _ => candidates.len(),
                            }
                            .min(candidates.len());
                            exact.rerank(query, &candidates[..take], 10, &mut found)?;
                            count += take;
                            (count, Some(take))
                        }
                    };
                    let elapsed_ns = start.elapsed().as_nanos();
                    let ids = coverage_candidates.map(|take| candidates[..take].to_vec());
                    records.push(
                        json!({"query":qi, "repetition":repetition, "ef":ef, "arm":arm,
                        "ids":found, "pool_ids":ids, "candidate_count":coverage_candidates,
                        "evaluations":evaluations, "elapsed_ns":elapsed_ns}),
                    );
                }
            }
        }
    }
    let report = json!({"schema_version":1, "seed":args.seed, "n":n, "dim":args.dim,
        "queries":queries.len()/args.dim, "graph_hash":format!("{:016x}",graph.structure_hash()),
        "graph_build_seconds":graph_seconds,"encoding_seconds":encoding_seconds,
        "compact_index_bytes":index.index_bytes(),"original_vector_bytes":train.len()*4,
        "diagnostic_norm_bytes":n*8,"diagnostic_seen_bytes":n*4,
        "not_isolated_serving_rss":true,"exact_scoring":"float64 cosine on graph-normalized float32 rows",
        "records":records});
    fs::write(args.out, serde_json::to_vec(&report)?)?;
    Ok(())
}
