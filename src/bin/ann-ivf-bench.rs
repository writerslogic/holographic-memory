// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use holographic_memory::core::qgraph::ivf::{IvfSearchParams, IvfSearcher, QuantizedIvf};
use serde::Serialize;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    train: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    centroids: PathBuf,
    #[arg(long)]
    assignments: PathBuf,
    #[arg(long)]
    dim: usize,
    #[arg(long, default_value_t = 8)]
    bits: u8,
    #[arg(long, default_value_t = 0)]
    residual_bits: u8,
    #[arg(long, value_delimiter = ',', default_value = "2,4,8,16,32,64,128")]
    nprobe: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "0,64,256")]
    rerank: Vec<usize>,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    index_out: PathBuf,
}

#[derive(Serialize)]
struct ConfigResult {
    nprobe: usize,
    rerank: usize,
    predictions: Vec<Vec<u32>>,
    elapsed_ns: Vec<Vec<u64>>,
    batch_elapsed_ns: Vec<u64>,
    scanned_counts: Vec<usize>,
    evaluated_counts: Vec<usize>,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    dim: usize,
    database_rows: usize,
    query_rows: usize,
    bits: u8,
    residual_bits: u8,
    build_ns: u64,
    index_bytes: usize,
    auxiliary_index_bytes: usize,
    roundtrip_verified: bool,
    timing_boundary: &'static str,
    configs: Vec<ConfigResult>,
}

fn input(path: &Path) -> Result<Vec<u8>> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let length = file.metadata()?.len();
    if length == 0 || length > 1024 * 1024 * 1024 || length % 4 != 0 {
        bail!("input size must be a nonzero multiple of 4, at most 1 GiB");
    }
    use std::io::Read;
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(length + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != length {
        bail!("input changed size while reading");
    }
    Ok(bytes)
}

fn floats(path: &Path, dim: usize) -> Result<Vec<f32>> {
    let bytes = input(path)?;
    if !(1..=4096).contains(&dim) || !(bytes.len() / 4).is_multiple_of(dim) {
        bail!("invalid dimension or incomplete float input row");
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

fn ns(start: Instant) -> Result<u64> {
    u64::try_from(start.elapsed().as_nanos()).context("timer exceeded u64 nanoseconds")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(1..=10).contains(&args.repeats) || args.nprobe.len() > 32 || args.rerank.len() > 32 {
        bail!("repeats must be 1..=10; each parameter grid is limited to 32 values");
    }
    if args.out.exists() || args.index_out.exists() || args.out == args.index_out {
        bail!("output paths must be distinct and must not exist");
    }
    let train = floats(&args.train, args.dim)?;
    let queries = floats(&args.queries, args.dim)?;
    let centers = floats(&args.centroids, args.dim)?;
    let assignments = input(&args.assignments)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    let start = Instant::now();
    let index = QuantizedIvf::build(
        &train,
        args.dim,
        &centers,
        &assignments,
        args.bits,
        args.residual_bits,
    )?;
    let build_ns = ns(start)?;
    if index.len() < 10 || queries.len() / args.dim > 10000 {
        bail!("benchmark requires at least 10 database rows and at most 10000 queries");
    }
    let bytes = index.to_bytes();
    let loaded = QuantizedIvf::from_bytes(&bytes)?;
    if loaded.to_bytes() != bytes {
        bail!("serialized index changed on reload");
    }
    let mut search = IvfSearcher::new(&index);
    let mut restored = IvfSearcher::new(&loaded);
    let mut configs = Vec::new();
    for &nprobe in &args.nprobe {
        for &rerank in &args.rerank {
            let params = IvfSearchParams { nprobe, rerank };
            let mut predictions: Vec<Vec<u32>> = Vec::new();
            let mut scanned_counts = Vec::new();
            let mut evaluated_counts = Vec::new();
            for query in queries.chunks_exact(args.dim) {
                let hits = search.search(query, 10, params)?;
                if hits != restored.search(query, 10, params)? {
                    bail!("reloaded index changed search results");
                }
                predictions.push(hits.iter().map(|h| h.0).collect());
                scanned_counts.push(search.scanned());
                evaluated_counts.push(search.evaluated());
            }
            let mut elapsed_ns = Vec::new();
            let mut batch_elapsed_ns = Vec::new();
            for _ in 0..args.repeats {
                let batch = Instant::now();
                let mut timings = Vec::with_capacity(predictions.len());
                for (i, query) in queries.chunks_exact(args.dim).enumerate() {
                    let start = Instant::now();
                    let hits = search.search(query, 10, params)?;
                    timings.push(ns(start)?);
                    if hits.iter().map(|h| h.0).ne(predictions[i].iter().copied()) {
                        bail!("repeated search changed predictions");
                    }
                }
                batch_elapsed_ns.push(ns(batch)?);
                elapsed_ns.push(timings);
            }
            configs.push(ConfigResult {
                nprobe,
                rerank,
                predictions,
                elapsed_ns,
                batch_elapsed_ns,
                scanned_counts,
                evaluated_counts,
            });
        }
    }
    let report = Report {
        schema_version: 1, dim: args.dim, database_rows: index.len(), query_rows: queries.len() / args.dim,
        bits: args.bits, residual_bits: args.residual_bits, build_ns, index_bytes: bytes.len(),
        auxiliary_index_bytes: index.auxiliary_bytes(),
        roundtrip_verified: true,
        timing_boundary: "one native query including normalization, candidate scan, rerank, result allocation; no file I/O or Python dispatch",
        configs,
    };
    use std::io::Write;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.index_out)?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out)?;
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}
