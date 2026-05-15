use clap::{Parser, Subcommand};
use hms_native::core::engine::HmsCore;
use hms_native::core::entangled::EntangledHVec;
use serde_json::json;
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest vectors into the index
    Load {
        #[arg(short, long)]
        path: String,
        #[arg(short, long, default_value_t = 16384)]
        dim: usize,
        #[arg(short, long, default_value_t = 10000)]
        count: usize,
    },
    /// Run queries and return metrics
    Query {
        #[arg(short, long)]
        path: String,
        #[arg(short, long, default_value_t = 16384)]
        dim: usize,
        #[arg(short, long, default_value_t = 10)]
        k: u32,
        #[arg(short, long, default_value_t = 100)]
        queries: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Load { path, dim, count } => {
            let hms = HmsCore::new(*dim as u32, Some(path.clone()), None)?;
            println!("Loading {} synthetic vectors...", count);

            let start = Instant::now();
            for i in 0..*count {
                let v = EntangledHVec::random(*dim, i as u64);
                hms.memorize(format!("v_{}", i), v)?;
            }
            let duration = start.elapsed();

            println!(
                "{}",
                json!({
                    "action": "load",
                    "count": count,
                    "duration_ms": duration.as_millis(),
                    "throughput_ops_sec": (*count as f64 / duration.as_secs_f64()) as usize
                })
            );
        }
        Commands::Query {
            path,
            dim,
            k,
            queries,
        } => {
            let hms = HmsCore::new(*dim as u32, Some(path.clone()), None)?;

            // Train if not trained
            if !hms.nsg_trained() {
                println!("Training NSG index for benchmark...");
                hms.train_nsg()?;
            }

            let mut latencies = Vec::with_capacity(*queries);
            let mut black_box_sum = 0;

            for i in 0..*queries {
                let q_vec = EntangledHVec::random(*dim, (i + 1000000) as u64);
                let start = Instant::now();
                let results = hms.query(&q_vec, *k);
                latencies.push(start.elapsed().as_micros() as u64);
                black_box_sum += results.len();
            }

            if black_box_sum == 0 {
                eprintln!("Warning: No results found in queries.");
            }

            latencies.sort_unstable();
            let p50 = latencies[latencies.len() / 2];
            let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
            let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

            println!(
                "{}",
                json!({
                    "action": "query",
                    "k": k,
                    "queries": queries,
                    "p50_us": p50,
                    "p95_us": p95,
                    "p99_us": p99,
                    "total_vectors": hms.vector_count()
                })
            );
        }
    }

    Ok(())
}
