// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Memory API: facts that change over time, queried as of a moment, with history kept.
//! Run: `cargo run --release --manifest-path examples/quickstart/rust/Cargo.toml --bin memory`

use std::sync::Arc;

use anyhow::{ensure, Result};
use writerslogic_warm::core::memory_api::{Memory, RecallOptions, RememberInput};
use writerslogic_warm::WarmCore;

fn main() -> Result<()> {
    // --8<-- [start:memory]
    let dir = tempfile::tempdir()?;
    let core = Arc::new(WarmCore::new(
        4096,
        Some(dir.path().display().to_string()),
        None,
    )?);
    let memory = Memory::new(core);

    memory.remember(RememberInput {
        id: Some("home-1".into()),
        text: "Alice lives in Paris".into(),
        subject: Some("alice/city".into()),
        valid_from: Some(1_577_836_800_000), // 2020-01-01
        ..Default::default()
    })?;
    memory.remember(RememberInput {
        id: Some("home-2".into()),
        text: "Alice lives in Berlin".into(),
        subject: Some("alice/city".into()),
        valid_from: Some(1_735_689_600_000), // 2025-01-01
        supersedes: Some(vec!["home-1".into()]),
        ..Default::default()
    })?;

    let question = "where does Alice live";
    let now = memory.recall(question, &RecallOptions::default())?;
    let then = memory.recall(
        question,
        &RecallOptions {
            as_of: Some(1_600_000_000_000), // 2020-09-13
            ..Default::default()
        },
    )?;
    println!("now:  {}", now[0].text.as_deref().unwrap_or(""));
    println!("then: {}", then[0].text.as_deref().unwrap_or(""));
    // --8<-- [end:memory]
    ensure!(
        now.len() == 1 && now[0].id == "home-2",
        "current fact must be Berlin only"
    );
    ensure!(then[0].id == "home-1", "as of 2020 the fact is Paris");
    Ok(())
}
