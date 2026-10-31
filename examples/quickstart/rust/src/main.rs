// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Five-minute quickstart: ingest a corpus, search it, compare with SQLite FTS5, encrypt it.
//! Run from the repository root: `cargo run --release --manifest-path examples/quickstart/rust/Cargo.toml`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use writerslogic_warm::core::WarmConfig;
use writerslogic_warm::{DocumentInput, WarmCore, SearchOptions};
use rusqlite::Connection;

const DIMENSIONS: u32 = 4096;

struct Doc {
    id: String,
    title: String,
    text: String,
}

fn quickstart_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn load_corpus() -> Result<Vec<Doc>> {
    let raw = std::fs::read_to_string(quickstart_dir().join("corpus.jsonl"))?;
    raw.lines()
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line)?;
            let field = |k: &str| v[k].as_str().map(str::to_owned).context(k.to_owned());
            Ok(Doc {
                id: field("id")?,
                title: field("title")?,
                text: field("text")?,
            })
        })
        .collect()
}

fn to_input(doc: &Doc) -> DocumentInput {
    DocumentInput {
        id: doc.id.clone(),
        text: doc.text.clone(),
        source_uri: Some(format!("udhr#{}", doc.title)),
        version: None,
        metadata: None,
        chunk_words: None,
        overlap_words: None,
        store_text: None,
        embeddings: None,
        title_weight: None,
    }
}

/// FTS5 treats a bare multi-word query as AND; OR of the words is the fair lexical baseline.
fn fts5_query(query: &str) -> String {
    let words: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{w}\""))
        .collect();
    words.join(" OR ")
}

fn main() -> Result<()> {
    let corpus = load_corpus()?;
    let queries: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(
        quickstart_dir().join("queries.json"),
    )?)?;

    // --8<-- [start:ingest]
    // 1. Ingest. A store is a directory (`None` would use the current directory); it persists
    // there and reopens with the same arguments.
    let store_dir = tempfile::tempdir()?;
    let warm = WarmCore::new(
        DIMENSIONS,
        Some(store_dir.path().display().to_string()),
        None,
    )?;
    let start = Instant::now();
    let inputs: Vec<DocumentInput> = corpus.iter().map(to_input).collect();
    warm.memorize_documents(inputs)?;
    println!(
        "ingested {} documents in {:.1?}",
        corpus.len(),
        start.elapsed()
    );
    // --8<-- [end:ingest]

    // --8<-- [start:compare]
    // The same corpus in SQLite FTS5 (bm25 ranking), for comparison.
    let db = Connection::open_in_memory()?;
    db.execute_batch("CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, text);")?;
    for doc in &corpus {
        db.execute(
            "INSERT INTO docs (id, text) VALUES (?1, ?2)",
            (&doc.id, &doc.text),
        )?;
    }

    // 2. Search both engines with the same queries.
    println!("\n{:<58} {:<9} {:<9} expected", "query", "WARM", "FTS5");
    let (mut warm_hits, mut fts_hits) = (0, 0);
    for q in &queries {
        let query = q["query"].as_str().context("query")?;
        let expect = q["expect"].as_str().context("expect")?;
        let options = SearchOptions {
            k: Some(3),
            ..Default::default()
        };
        let results = warm.search_documents(query, &options)?;
        let warm_top = results
            .first()
            .map_or("-", |r| r.document_id.as_str())
            .to_owned();
        let fts_top: String = db
            .query_row(
                "SELECT id FROM docs WHERE docs MATCH ?1 ORDER BY bm25(docs) LIMIT 1",
                [fts5_query(query)],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "-".to_owned());
        warm_hits += usize::from(warm_top == expect);
        fts_hits += usize::from(fts_top == expect);
        println!("{query:<58} {warm_top:<9} {fts_top:<9} {expect}");
    }
    println!(
        "top-1 correct: WARM {warm_hits}/{n}, FTS5 {fts_hits}/{n}",
        n = queries.len()
    );
    ensure!(
        warm_hits >= 5,
        "WARM found the expected article for only {warm_hits} queries"
    );
    // --8<-- [end:compare]

    // --8<-- [start:hit]
    // Hits carry the passage and its byte range in the source text.
    let hit = &warm.search_documents("asylum from persecution", &SearchOptions::default())?[0];
    println!(
        "\n{} bytes {}..{} score {:.3}: {}",
        hit.document_id,
        hit.start_byte,
        hit.end_byte,
        hit.score,
        hit.text.as_deref().unwrap_or("")
    );
    // --8<-- [end:hit]

    // --8<-- [start:encrypt]
    // 3. Encrypt: the same corpus in a persistent store with AES-256-GCM (Argon2id key).
    let secret_dir = tempfile::tempdir()?;
    let secret_path = secret_dir.path().display().to_string();
    let mut config = WarmConfig::default();
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some("correct horse battery staple".into());

    let sealed = WarmCore::new(DIMENSIONS, Some(secret_path.clone()), Some(config.clone()))?;
    sealed.memorize_documents(corpus.iter().map(to_input).collect())?;
    sealed.flush()?;
    println!("\nsecurity status: {}", sealed.security_status());
    drop(sealed);
    let needle = b"periodic holidays with pay";
    let leaks = files_containing(secret_dir.path(), needle)?;
    println!("encrypted store, files holding the passage: {leaks:?}");
    // With encryption on, no store file may hold the passage text, the file-backed text segment
    // (`documents.text`) included.
    ensure!(leaks.is_empty(), "plaintext passage found in {leaks:?}");
    println!("no store file holds the passage text");

    let reopened = WarmCore::new(DIMENSIONS, Some(secret_path.clone()), Some(config.clone()))?;
    let hit = &reopened.search_documents("paid holidays", &SearchOptions::default())?[0];
    println!("reopened with the passphrase, top hit: {}", hit.document_id);
    drop(reopened);

    config.security.encryption_passphrase = Some("wrong passphrase".into());
    let Err(error) = WarmCore::new(DIMENSIONS, Some(secret_path), Some(config)) else {
        anyhow::bail!("opening with the wrong passphrase must fail");
    };
    println!("opening with a wrong passphrase fails: {error}");
    ensure!(
        !error.to_string().contains("already open"),
        "failed for the wrong reason: {error}"
    );
    // --8<-- [end:encrypt]
    Ok(())
}

/// File names under `dir` whose bytes contain `needle`.
fn files_containing(dir: &Path, needle: &[u8]) -> Result<Vec<String>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            found.extend(files_containing(&path, needle)?);
        } else if std::fs::read(&path)?
            .windows(needle.len())
            .any(|w| w == needle)
        {
            found.push(
                path.file_name()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            );
        }
    }
    Ok(found)
}
