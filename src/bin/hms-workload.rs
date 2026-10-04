use anyhow::{ensure, Result};
use clap::Parser;
use holographic_memory::{EntangledHVec, HmsCore};
use std::sync::{Arc, Barrier};
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 1200)]
    vectors: usize,
    #[arg(long, default_value_t = 300)]
    operations: usize,
    #[arg(long, default_value_t = 4096)]
    dimensions: u32,
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.vectors > 0 && args.operations > 0,
        "vectors and operations must be positive"
    );
    let directory = tempfile::tempdir()?;
    let hms = Arc::new(HmsCore::new(
        args.dimensions,
        Some(directory.path().display().to_string()),
        None,
    )?);
    for i in 0..args.vectors {
        hms.memorize(
            format!("v{i}"),
            EntangledHVec::new_deterministic(args.dimensions as usize, i as u64),
        )?;
    }
    hms.train_nsg()?;
    let barrier = Arc::new(Barrier::new(2));
    let writer = Arc::clone(&hms);
    let write_barrier = Arc::clone(&barrier);
    let writer_task = std::thread::spawn(move || -> Result<Vec<f64>> {
        let mut latencies = Vec::with_capacity(args.operations);
        write_barrier.wait();
        for i in 0..args.operations {
            let start = Instant::now();
            let id = format!("v{}", i % args.vectors);
            if i % 5 == 0 {
                writer.delete(&id)?;
            }
            writer.memorize(
                id,
                EntangledHVec::new_deterministic(args.dimensions as usize, i as u64 + 1_000_000),
            )?;
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(latencies)
    });
    let mut query_latency = Vec::with_capacity(args.operations);
    barrier.wait();
    let start = Instant::now();
    for i in 0..args.operations {
        let query =
            EntangledHVec::new_deterministic(args.dimensions as usize, (i % args.vectors) as u64);
        let now = Instant::now();
        hms.query(&query, 10);
        query_latency.push(now.elapsed().as_secs_f64() * 1000.0);
    }
    let mut write_latency = writer_task
        .join()
        .map_err(|_| anyhow::anyhow!("writer thread panicked"))??;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "hms_version": env!("CARGO_PKG_VERSION"), "architecture": std::env::consts::ARCH,
            "os": std::env::consts::OS, "vectors": args.vectors, "operations_per_thread": args.operations,
            "dimensions": args.dimensions, "elapsed_seconds": start.elapsed().as_secs_f64(),
            "query_p95_ms": percentile(&mut query_latency, 0.95),
            "query_p99_ms": percentile(&mut query_latency, 0.99),
            "update_p95_ms": percentile(&mut write_latency, 0.95),
            "update_p99_ms": percentile(&mut write_latency, 0.99),
            "live_vectors": hms.vector_count(),
        }))?
    );
    Ok(())
}
