// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Delete and search latency, for comparing the scrubbing delete with the previous one.
//! `cargo run --release --example purge_bench -- [documents] [deletes] [searches]` prints JSON.

use std::time::Instant;

use holographic_memory::{DocumentInput, HmsCore, SearchOptions};

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn summary(mut micros: Vec<f64>) -> serde_json::Value {
    micros.sort_by(f64::total_cmp);
    serde_json::json!({
        "n": micros.len(),
        "p50_us": percentile(&micros, 0.5),
        "p95_us": percentile(&micros, 0.95),
        "max_us": micros[micros.len() - 1],
    })
}

fn main() -> anyhow::Result<()> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (documents, deletes, searches) = (
        args.first().copied().unwrap_or(5000),
        args.get(1).copied().unwrap_or(500),
        args.get(2).copied().unwrap_or(500),
    );
    let dir = tempfile::tempdir()?;
    let core = HmsCore::new(4096, Some(dir.path().display().to_string()), None)?;
    let words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
    ];
    let text = |i: usize| -> String {
        (0..60)
            .map(|j| format!("{}{}", words[(i * 7 + j * 13) % words.len()], (i + j) % 97))
            .collect::<Vec<_>>()
            .join(" ")
    };
    for chunk in (0..documents).collect::<Vec<_>>().chunks(256) {
        let batch: Vec<DocumentInput> = chunk
            .iter()
            .map(|&i| {
                serde_json::from_value(serde_json::json!({
                    "id": format!("doc-{i}"), "text": text(i), "version": "1",
                }))
                .unwrap()
            })
            .collect();
        core.memorize_documents(batch)?;
    }
    let mut search = Vec::new();
    for i in 0..searches {
        let started = Instant::now();
        core.search_documents(
            &format!("{} {}", words[i % words.len()], i % 97),
            &SearchOptions::default(),
        )?;
        search.push(started.elapsed().as_secs_f64() * 1e6);
    }
    let mut delete = Vec::new();
    for i in 0..deletes {
        let started = Instant::now();
        core.delete_document(&format!("doc-{}", i * (documents / deletes.max(1)).max(1)))?;
        delete.push(started.elapsed().as_secs_f64() * 1e6);
    }
    let mut search_after = Vec::new();
    for i in 0..searches {
        let started = Instant::now();
        core.search_documents(
            &format!("{} {}", words[i % words.len()], i % 97),
            &SearchOptions::default(),
        )?;
        search_after.push(started.elapsed().as_secs_f64() * 1e6);
    }
    let started = Instant::now();
    if std::env::var_os("PURGE_SCRUB").is_some() {
        core.compact_scrubbing()?;
    } else {
        core.compact()?;
    }
    let compact_ms = started.elapsed().as_secs_f64() * 1e3;
    println!(
        "{}",
        serde_json::json!({
            "documents": documents,
            "search_before_deletes": summary(search),
            "delete": summary(delete),
            "search_after_deletes": summary(search_after),
            "scrubbing_compaction": std::env::var_os("PURGE_SCRUB").is_some(),
            "compact_ms": compact_ms,
        })
    );
    Ok(())
}
