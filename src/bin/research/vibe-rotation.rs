// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use clap::Parser;
use holographic_memory::core::qgraph::{
    diagnostics::{RotationControl, RotationSearcher},
    BuildParams, Graph, VGraph, VSearchParams, VSearcher,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    graph_cache: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 24301)]
    rotation_seed: u64,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, value_delimiter = ',', default_value = "128,256,512,1000,2000")]
    ef: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "100,400,1000")]
    rerank: Vec<usize>,
}

fn bytes(path: &Path, width: usize) -> Result<Vec<u8>> {
    let row_bytes = width.checked_mul(4).context("row width overflow")?;
    let length = fs::metadata(path)?.len();
    ensure!(
        row_bytes > 0
            && length > 0
            && length <= (1 << 30)
            && length.is_multiple_of(row_bytes as u64),
        "invalid or oversized matrix: {}",
        path.display()
    );
    fs::read(path).with_context(|| path.display().to_string())
}

fn floats(path: &Path, width: usize) -> Result<Vec<f32>> {
    let rows: Vec<_> = bytes(path, width)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    ensure!(rows.iter().all(|x| x.is_finite()), "nonfinite matrix");
    Ok(rows)
}

enum Index {
    Plain(Box<VGraph>),
    Rotated(Box<RotationControl>),
}

enum Search<'a> {
    Plain(VSearcher<'a>),
    Rotated(RotationSearcher<'a>),
}

impl Index {
    fn bytes(&self) -> usize {
        match self {
            Self::Plain(g) => g.index_bytes(),
            Self::Rotated(g) => g.index_bytes(),
        }
    }
    fn searcher(&self) -> Search<'_> {
        match self {
            Self::Plain(g) => Search::Plain(g.searcher()),
            Self::Rotated(g) => Search::Rotated(g.searcher()),
        }
    }
}

impl Search<'_> {
    fn search(&mut self, query: &[f32], params: VSearchParams, out: &mut Vec<u32>) -> Result<()> {
        match self {
            Self::Plain(s) => {
                s.search(query, 100, params, out)?;
            }
            Self::Rotated(s) => {
                s.search(query, 100, params, out)?;
            }
        }
        Ok(())
    }
}

struct Row {
    arm: usize,
    ef: usize,
    rerank: usize,
    ns: Vec<Vec<u64>>,
    ids: Vec<u32>,
    vibe: f64,
    strict: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.out.exists() && !args.out.with_extension("ids.u32").exists(),
        "output exists"
    );
    ensure!((1..=11).contains(&args.repeats), "repeats outside 1..=11");
    ensure!(
        !args.ef.is_empty()
            && args.ef.len() <= 16
            && args.ef.iter().all(|&v| (100..=4096).contains(&v)),
        "ef outside 100..=4096 or too many settings"
    );
    ensure!(
        !args.rerank.is_empty()
            && args.rerank.len() <= 8
            && args.rerank.iter().all(|&v| (100..=4096).contains(&v)),
        "rerank outside 100..=4096 or too many settings"
    );
    let meta: Value = serde_json::from_slice(&fs::read(args.data.join("meta.json"))?)?;
    ensure!(
        meta["metric"] == "angular" && meta["n_neighbors"] == 100,
        "requires angular data and 100-neighbor truth"
    );
    let dim = meta["dim"]
        .as_u64()
        .filter(|&v| v > 0 && v <= 8192)
        .context("invalid dimension")? as usize;
    let train = floats(&args.data.join("train.f32"), dim)?;
    let queries = floats(&args.data.join("test.f32"), dim)?;
    let truth: Vec<_> = bytes(&args.data.join("neighbors.i32"), 100)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| i32::from_le_bytes(*x))
        .collect();
    let distances = floats(&args.data.join("distances.f32"), 100)?;
    let n = train.len() / dim;
    let nq = queries.len() / dim;
    ensure!(
        (100..=250000).contains(&n)
            && nq <= 1000
            && truth.len() == nq * 100
            && distances.len() == nq * 100,
        "invalid data shape"
    );
    ensure!(
        truth.iter().all(|&v| v >= 0 && (v as usize) < n),
        "truth ID outside corpus"
    );
    let params = BuildParams {
        degree: 64,
        build_ef: 500,
        quantized_search: true,
        ..Default::default()
    };
    let graph = Graph::load(&train, dim, &args.graph_cache, &params)?;
    let mut indexes = Vec::new();
    let mut encoding_seconds = Vec::new();
    for bits in [4, 8] {
        let t = Instant::now();
        indexes.push(Index::Plain(Box::new(VGraph::from_graph_with(
            &graph, bits, 8, true, true, 3, false,
        )?)));
        encoding_seconds.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let rotated = RotationControl::from_graph(&graph, bits, args.rotation_seed)?;
        ensure!(
            rotated.graph_hash() == graph.structure_hash(),
            "rotation changed topology"
        );
        indexes.push(Index::Rotated(Box::new(rotated)));
        encoding_seconds.push(t.elapsed().as_secs_f64());
    }
    let names = ["plain4", "rotated4", "plain8", "rotated8"];
    let normalize = meta["source_metric"] == "cosine";
    let norms: Vec<f64> = train
        .chunks_exact(dim)
        .map(|row| {
            row.iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let query_norms: Vec<f64> = queries
        .chunks_exact(dim)
        .map(|row| {
            row.iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    ensure!(
        norms.iter().all(|&v| v > 0.) && query_norms.iter().all(|&v| v > 0.),
        "zero vector"
    );
    let mut searchers: Vec<_> = indexes.iter().map(Index::searcher).collect();
    let mut rows = Vec::new();
    for &ef in &args.ef {
        for &rerank in &args.rerank {
            for arm in 0..4 {
                rows.push(Row {
                    arm,
                    ef,
                    rerank,
                    ns: Vec::new(),
                    ids: Vec::new(),
                    vibe: 0.,
                    strict: 0.,
                });
            }
        }
    }
    let mut found = Vec::with_capacity(100);
    let mut load = Vec::new();
    for repetition in 0..args.repeats {
        for group in rows.as_chunks_mut::<4>().0 {
            for offset in 0..4 {
                let row = &mut group[(offset + repetition) % 4];
                let search = &mut searchers[row.arm];
                let settings = VSearchParams {
                    ef: row.ef,
                    rerank: row.rerank,
                    ..Default::default()
                };
                search.search(&queries[..dim], settings, &mut found)?;
                let mut times = Vec::with_capacity(nq);
                let mut ids = Vec::with_capacity(nq * 100);
                for query in queries.chunks_exact(dim) {
                    let t = Instant::now();
                    search.search(query, settings, &mut found)?;
                    times.push(t.elapsed().as_nanos() as u64);
                    ensure!(
                        found.len() <= 100 && found.iter().all(|&id| (id as usize) < n),
                        "invalid search results: arm={}, ef={}, rerank={}, query={}, count={}, max_id={:?}",
                        names[row.arm], row.ef, row.rerank, ids.len() / 100, found.len(), found.iter().max()
                    );
                    let mut unique = found.clone();
                    unique.sort_unstable();
                    unique.dedup();
                    ensure!(unique.len() == found.len(), "duplicate search IDs");
                    ids.extend_from_slice(&found);
                    ids.resize(ids.len() + 100 - found.len(), u32::MAX);
                }
                if repetition == 0 {
                    let mut hits = 0;
                    let mut strict = 0;
                    for (qi, (query, ids)) in queries
                        .chunks_exact(dim)
                        .zip(ids.as_chunks::<100>().0)
                        .enumerate()
                    {
                        for &id in ids {
                            if id == u32::MAX {
                                continue;
                            }
                            strict +=
                                usize::from(truth[qi * 100..(qi + 1) * 100].contains(&(id as i32)));
                            let inner: f64 = query
                                .iter()
                                .zip(&train[id as usize * dim..(id as usize + 1) * dim])
                                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                .sum();
                            let similarity = if normalize {
                                inner / (norms[id as usize] * query_norms[qi])
                            } else {
                                inner
                            };
                            hits += usize::from(
                                (1. - similarity).clamp(0., 2.)
                                    <= f64::from(distances[qi * 100 + 99]) + 1e-3,
                            );
                        }
                    }
                    row.vibe = hits as f64 / (nq * 100) as f64;
                    row.strict = strict as f64 / (nq * 100) as f64;
                    row.ids = ids;
                } else {
                    ensure!(row.ids == ids, "results changed across repeats");
                }
                row.ns.push(times);
                load.push(
                    fs::read_to_string("/proc/loadavg")
                        .ok()
                        .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok()),
                );
            }
        }
        eprintln!("completed paired round {}/{}", repetition + 1, args.repeats);
    }
    let mut predictions = Vec::new();
    let records: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            for id in &row.ids {
                predictions.extend_from_slice(&id.to_le_bytes());
            }
            let qps: Vec<_> = row
                .ns
                .iter()
                .map(|ns| nq as f64 * 1e9 / ns.iter().sum::<u64>() as f64)
                .collect();
            let mut sorted = qps.clone();
            sorted.sort_by(f64::total_cmp);
            let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.;
            json!({"arm": names[row.arm], "ef": row.ef, "rerank": row.rerank, "recall": row.vibe,
            "strict_id_recall": row.strict, "qps_runs": qps, "median_qps": median,
            "missing_results": row.ids.iter().filter(|&&id| id == u32::MAX).count(),
            "query_ns": row.ns, "prediction_row_offset": i * nq * 100})
        })
        .collect();
    let report = json!({"schema_version": 1, "scope": "paired encoding-only diagnostic; no production defaults changed",
        "data": meta, "k": 100, "n_queries": nq, "rotation_seed": args.rotation_seed,
        "rotation": "three signed normalized Walsh-Hadamard rounds, includes query transform in timing",
        "graph_hash": format!("{:016x}", graph.structure_hash()), "build_params": {"degree": 64, "build_ef": 500, "quantized_search": true},
        "repeats": args.repeats, "load_1m": load,
        "arms": names.iter().enumerate().map(|(i, &name)| json!({"name": name, "index_bytes": indexes[i].bytes(), "encoding_seconds": encoding_seconds[i]})).collect::<Vec<_>>(),
        "rows": records});
    fs::write(args.out.with_extension("ids.u32"), predictions)?;
    fs::write(args.out, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
