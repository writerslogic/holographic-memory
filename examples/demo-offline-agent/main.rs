// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! A personal agent's memory, fully offline and embedding-free.
//!
//! Run from a clean clone with one command: `cargo run --example demo-offline-agent`.
//!
//! No model, no network, no API key: retrieval is stemmed BM25 over the stored passages, and
//! relations use the meaning memory. Nothing in the run depends on the clock, the machine or a
//! random seed, so the transcript printed on stdout is identical on every run. CI diffs it
//! against `examples/demo-offline-agent/expected-output.txt`.

use anyhow::{ensure, Result};
use serde_json::json;
use writerslogic_warm::core::WarmConfig;
use writerslogic_warm::{DocumentInput, SearchOptions, WarmCore};

const DIMENSIONS: u32 = 4096;

fn note(id: &str, kind: &str, day: &str, text: &str) -> DocumentInput {
    DocumentInput {
        id: id.to_owned(),
        text: text.to_owned(),
        source_uri: Some(format!("agent://{kind}/{id}")),
        version: Some("1".to_owned()),
        metadata: Some(json!({ "kind": kind, "day": day })),
        chunk_words: None,
        overlap_words: None,
        store_text: None,
        embeddings: None,
        title_weight: None,
    }
}

fn config() -> WarmConfig {
    let mut config = WarmConfig::default();
    config.meaning.enabled = true;
    config
}

/// Prints the ranked document ids for `query` and returns them.
fn recall(memory: &WarmCore, query: &str, options: &SearchOptions) -> Result<Vec<String>> {
    let ids: Vec<String> = memory
        .search_documents(query, options)?
        .into_iter()
        .map(|hit| hit.document_id)
        .collect();
    println!("recall {query:?} -> {ids:?}");
    Ok(ids)
}

fn main() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let top3 = SearchOptions {
        k: Some(3),
        ..Default::default()
    };

    println!("== session 1: the agent learns ==");
    let memory = WarmCore::new(DIMENSIONS, Some(path.clone()), Some(config()))?;
    let notes = vec![
        note(
            "pref-drink",
            "fact",
            "2026-03-01",
            "Mira drinks black coffee every morning before work.",
        ),
        note(
            "pref-music",
            "fact",
            "2026-03-01",
            "Mira listens to jazz while she writes code.",
        ),
        note(
            "allergy",
            "fact",
            "2026-03-01",
            "Mira is allergic to peanuts and carries an epinephrine pen.",
        ),
        note(
            "trip-lisbon",
            "episode",
            "2026-03-04",
            "Mira flew to Lisbon for the quarterly planning offsite with the Acme team.",
        ),
        note(
            "trip-berlin",
            "episode",
            "2026-03-11",
            "Mira visited Berlin to demo the sync engine at a robotics meetup.",
        ),
        note(
            "bug-report",
            "episode",
            "2026-03-12",
            "Mira fixed the flaky sync test by pinning the timezone in the CI runner.",
        ),
        note(
            "api-key",
            "episode",
            "2026-03-13",
            "Temporary staging token for the deploy script was pasted into the chat.",
        ),
        note(
            "book",
            "episode",
            "2026-03-14",
            "Mira finished reading a novel about lighthouse keepers and recommended it to Sam.",
        ),
    ];
    let count = notes.len();
    memory.memorize_documents(notes)?;
    let relations = [
        ("r1", "mira", "works_at", "acme"),
        ("r2", "acme", "based_in", "lisbon"),
        ("r3", "mira", "manager", "sam"),
    ];
    for (id, subject, relation, object) in relations {
        memory.memorize_triplet(id.into(), subject.into(), relation.into(), object.into())?;
    }
    println!(
        "stored {count} documents and {} relations",
        memory.meaning_triple_count()
    );

    println!("\n== recall by content ==");
    ensure!(recall(&memory, "what does Mira drink in the morning", &top3)?[0] == "pref-drink");
    ensure!(recall(&memory, "peanut allergy", &top3)?[0] == "allergy");
    ensure!(recall(&memory, "flaky test in the CI runner", &top3)?[0] == "bug-report");

    println!("\n== recall restricted by metadata ==");
    ensure!(recall(&memory, "jazz music", &top3)?[0] == "pref-music");
    let episodes = SearchOptions {
        filter: Some(json!({ "kind": "episode" })),
        ..top3.clone()
    };
    let hits = recall(&memory, "jazz music", &episodes)?;
    ensure!(
        hits.is_empty(),
        "the only jazz note is a fact, so the episode filter excludes it"
    );
    ensure!(recall(&memory, "Mira trip planning", &episodes)?[0] == "trip-lisbon");

    println!("\n== a fact changes: the same id replaces the old version ==");
    let mut update = note(
        "pref-drink",
        "fact",
        "2026-03-15",
        "Mira switched from coffee to green tea after the offsite.",
    );
    update.version = Some("2".to_owned());
    memory.memorize_document(update)?;
    let hit = memory.search_documents("Mira drink tea", &top3)?;
    println!("pref-drink is now version {}", hit[0].version);
    ensure!(hit[0].document_id == "pref-drink" && hit[0].version == "2");
    ensure!(
        recall(&memory, "black", &top3)?.is_empty(),
        "the old wording must be gone"
    );

    println!("\n== forgetting: the agent deletes a pasted secret ==");
    ensure!(memory.delete_document("api-key")?);
    ensure!(recall(&memory, "staging token deploy script", &top3)?.is_empty());
    memory.compact()?;

    println!("\n== relations: structured multi-hop queries ==");
    let works = memory.multi_hop_with_limits("mira", &["works_at"], Default::default())?;
    println!("mira works_at -> {}", works[0].entity_id);
    ensure!(works[0].entity_id == "acme");
    let city =
        memory.multi_hop_with_limits("mira", &["works_at", "based_in"], Default::default())?;
    println!("mira works_at, based_in -> {}", city[0].entity_id);
    ensure!(city[0].entity_id == "lisbon");

    memory.flush()?;
    drop(memory);

    println!("\n== session 2: the process restarted, the memory is still there ==");
    let memory = WarmCore::new(DIMENSIONS, Some(path), Some(config()))?;
    println!("reopened with {} relations", memory.meaning_triple_count());
    ensure!(recall(&memory, "green tea", &top3)?[0] == "pref-drink");
    ensure!(recall(&memory, "lighthouse novel recommended", &top3)?[0] == "book");
    ensure!(recall(&memory, "staging token deploy script", &top3)?.is_empty());
    ensure!(
        memory.multi_hop_with_limits("mira", &["works_at", "based_in"], Default::default())?[0]
            .entity_id
            == "lisbon"
    );
    println!("\nok");
    Ok(())
}
