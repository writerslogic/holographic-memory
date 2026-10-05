// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cost measurement for `core::private_query`. Writes
//! `benchmarks/results/private_query.json`. Run with
//! `cargo run --release --features private-search --bin private-query-bench`.

use holographic_memory::core::entangled::EntangledHVec;
use holographic_memory::core::intersection::sparse_intersection_count;
use holographic_memory::core::private_query::{
    Client, Params, PublicMatrix, Server, ERROR_SIGMA, ERROR_TAIL, LWE_DIM, SEED_LEN,
};
use serde_json::json;
use std::time::Instant;

const DIM: usize = 16384;
const SIZES: [usize; 3] = [1_000, 10_000, 100_000];
const HINT_BUILDS: usize = 3;
const QUERIES: usize = 20;
const OUTPUT: &str = "benchmarks/results/private_query.json";

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn summary(samples: &mut [f64]) -> serde_json::Value {
    samples.sort_by(f64::total_cmp);
    json!({
        "median": samples[samples.len() / 2],
        "min": samples[0],
        "max": samples[samples.len() - 1],
        "samples": samples.len(),
    })
}

fn main() -> anyhow::Result<()> {
    let params = Params::for_dim(DIM)?;
    let start = Instant::now();
    let matrix = PublicMatrix::expand(params, [42; SEED_LEN]);
    let matrix_expand_ms = ms(start);
    let client = Client::new(matrix.clone());

    let mut results = Vec::new();
    for n in SIZES {
        let rows: Vec<EntangledHVec> = (0..n)
            .map(|i| EntangledHVec::new_deterministic(DIM, i as u64))
            .collect();
        let row_weight_max = rows.iter().map(|r| r.indices().len()).max().unwrap_or(0);

        let mut hint_ms = Vec::with_capacity(HINT_BUILDS);
        let mut server = None;
        for _ in 0..HINT_BUILDS {
            let start = Instant::now();
            server = Some(Server::new(&matrix, &rows)?);
            hint_ms.push(ms(start));
        }
        let server = server.expect("HINT_BUILDS > 0");

        let (mut query_ms, mut answer_ms, mut decode_ms) = (Vec::new(), Vec::new(), Vec::new());
        let (mut query_bytes, mut answer_bytes) = (0, 0);
        for i in 0..QUERIES {
            // Half the queries are stored rows, so full-weight scores occur.
            let code = if i % 2 == 0 {
                rows[(i * 7919) % n].clone()
            } else {
                EntangledHVec::new_deterministic(DIM, (1 << 40) + i as u64)
            };

            let start = Instant::now();
            let (query, state) = client.query(&code)?;
            query_ms.push(ms(start));

            let start = Instant::now();
            let answer = server.answer(&query)?;
            answer_ms.push(ms(start));

            let start = Instant::now();
            let scores = state.decode(server.hint(), &answer)?;
            decode_ms.push(ms(start));

            for (row, &score) in rows.iter().zip(&scores) {
                let expected = sparse_intersection_count(row.indices(), code.indices()) as u32;
                anyhow::ensure!(score == expected, "decoded {score}, expected {expected}");
            }
            query_bytes = query.byte_len();
            answer_bytes = answer.byte_len();
        }

        let entry = json!({
            "rows": n,
            "row_weight_max": row_weight_max,
            "hint_build_ms": summary(&mut hint_ms),
            "client_query_ms": summary(&mut query_ms),
            "server_answer_ms": summary(&mut answer_ms),
            "client_decode_ms": summary(&mut decode_ms),
            "query_bytes": query_bytes,
            "answer_bytes": answer_bytes,
            "hint_bytes": server.hint().byte_len(),
        });
        println!("{entry}");
        results.push(entry);
    }

    let report = json!({
        "benchmark": "private_query",
        "note": "Experimental. Wall-clock, single machine, release build. All decoded scores were checked against plaintext intersection counts.",
        "parameters": {
            "dim": DIM,
            "lwe_dim": LWE_DIM,
            "modulus_bits": 32,
            "error_sigma": ERROR_SIGMA,
            "error_tail": ERROR_TAIL,
            "plaintext_modulus": params.plaintext_modulus(),
            "max_row_weight": params.max_row_weight(),
        },
        "environment": {
            "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "rayon_threads": rayon::current_num_threads(),
            "crate_version": env!("CARGO_PKG_VERSION"),
        },
        "matrix_expand_ms": matrix_expand_ms,
        "matrix_bytes": DIM * LWE_DIM * size_of::<u32>(),
        "results": results,
    });
    std::fs::write(OUTPUT, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("wrote {OUTPUT}");
    Ok(())
}
