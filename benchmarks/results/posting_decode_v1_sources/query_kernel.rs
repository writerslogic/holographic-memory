// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use holographic_memory::core::documents::{LexicalExecution, LexicalOptions, LexicalQuery};
use holographic_memory::HmsCore;
use serde::Deserialize;
use std::io::{BufRead, Write};

#[derive(Deserialize)]
struct Request {
    query: String,
    k: usize,
    count: bool,
    repeats: usize,
    verify: bool,
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("index path");
    let core = HmsCore::new(1024, Some(path), None)?;
    println!("ready");
    std::io::stdout().flush()?;
    for line in std::io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        let query = LexicalQuery::parse(&request.query)?;
        let options = LexicalOptions {
            k: request.k,
            count: request.count,
            ..Default::default()
        };
        let expected = core.search_lexical(&query, &options)?;
        if request.verify {
            assert_eq!(
                expected,
                core.search_lexical(
                    &query,
                    &LexicalOptions {
                        execution: LexicalExecution::Exhaustive,
                        ..options.clone()
                    }
                )?,
                "{}",
                request.query
            );
        }
        let start = std::time::Instant::now();
        for _ in 0..request.repeats {
            std::hint::black_box(core.search_lexical(std::hint::black_box(&query), &options)?);
        }
        let ns = start.elapsed().as_nanos();
        let bytes = serde_json::to_vec(&expected)?;
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        println!(
            "{}",
            serde_json::json!({"ns":ns,"hash":hash,"count":expected.count})
        );
        std::io::stdout().flush()?;
    }
    Ok(())
}
