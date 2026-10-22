// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::HmsCore;
use crate::core::documents::{LexicalExecution, LexicalOptions, LexicalQuery};
use anyhow::{Context, Result};
use std::time::Instant;

#[test]
#[ignore = "opens the diagnostic index supplied by HMS_MEMORY_INDEX"]
fn reopened_index_memory_census() -> Result<()> {
    let path = std::env::var("HMS_MEMORY_INDEX").context("set HMS_MEMORY_INDEX")?;
    let output = std::env::var("HMS_MEMORY_CENSUS_OUT").context("set HMS_MEMORY_CENSUS_OUT")?;
    let opened = Instant::now();
    let core = HmsCore::new(1024, Some(path.clone()), None)?;
    let open_seconds = opened.elapsed().as_secs_f64();
    let status = if cfg!(target_os = "linux") {
        Some(std::fs::read_to_string("/proc/self/status")?)
    } else {
        None
    };
    let census = core.documents.read().memory_census().report();
    let mut ranked_checks = Vec::new();
    if let Some(queries) = std::env::var_os("HMS_MEMORY_QUERIES") {
        let queries: Vec<String> = serde_json::from_slice(&std::fs::read(queries)?)?;
        for text in queries {
            let query = LexicalQuery::parse(&text)?;
            for k in [0, 10, 1000] {
                let options = LexicalOptions {
                    k,
                    count: true,
                    execution: LexicalExecution::Exhaustive,
                    ..Default::default()
                };
                let reference = core.search_lexical(&query, &options)?;
                for execution in [
                    LexicalExecution::DocumentAtATime,
                    LexicalExecution::BlockMaxWand,
                ] {
                    let actual = core.search_lexical(
                        &query,
                        &LexicalOptions {
                            execution,
                            ..options.clone()
                        },
                    )?;
                    assert_eq!(actual, reference, "{text:?}, k={k}, {execution:?}");
                }
                ranked_checks.push(serde_json::json!({"query": text, "k": k, "result": reference}));
            }
        }
    }
    let report = serde_json::json!({
        "index": path,
        "open_seconds": open_seconds,
        "linux_status_after_open": status,
        "census": census,
        "ranked_checks": ranked_checks,
    });
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
