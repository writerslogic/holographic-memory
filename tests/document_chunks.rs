// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde_json::json;
use writerslogic_warm::core::documents::{EncodeMode, IngestOptions};
use writerslogic_warm::{core::WarmConfig, DocumentInput, WarmCore};

fn input(id: &str, text: &str, store_text: bool) -> DocumentInput {
    serde_json::from_value(json!({
        "id": id, "text": text, "version": "v1", "sourceUri": "local://original",
        "metadata": {"kind": "note"}, "chunkWords": 2, "overlapWords": 1,
        "storeText": store_text
    }))
    .unwrap()
}

#[test]
fn stored_chunks_preserve_spans_text_and_metadata_across_reopen() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = WarmConfig::default();
    config.document_checkpoint.min_journal_bytes = 0;
    let mut core = WarmCore::new(
        256,
        Some(dir.path().display().to_string()),
        Some(config.clone()),
    )?;
    let options = IngestOptions {
        encode: EncodeMode::Lexical,
        ..Default::default()
    };
    core.memorize_document_with(input("doc", "  alpha  beta\ngamma delta  ", true), &options)?;
    core.memorize_document_with(input("no-text", "alpha beta", false), &options)?;
    core.memorize_document_with(input("unicode", "  café🙂\tβ ", true), &options)?;
    for phase in 0..3 {
        if phase == 1 {
            core.compact()?;
        }
        if phase == 2 {
            drop(core);
            core = WarmCore::new(
                256,
                Some(dir.path().display().to_string()),
                Some(config.clone()),
            )?;
        }
        for (index, (start, end, text)) in [
            (2, 13, "alpha  beta"),
            (9, 19, "beta\ngamma"),
            (14, 25, "gamma delta"),
        ]
        .into_iter()
        .enumerate()
        {
            let chunk = core.read_document_chunk("doc", index)?.unwrap();
            assert_eq!(chunk.revision, core.revision().to_string());
            assert_eq!(
                (chunk.index, chunk.start_byte, chunk.end_byte),
                (index, start, end)
            );
            assert_eq!(chunk.text.as_deref(), Some(text));
            assert_eq!(chunk.document.id, "doc");
            assert_eq!(chunk.document.chunk_count, 3);
            assert_eq!(chunk.document.version, "v1");
            assert_eq!(chunk.document.metadata, json!({"kind": "note"}));
            assert_eq!(
                chunk.document.source_uri.as_deref(),
                Some("local://original")
            );
        }
        let unicode = core.read_document_chunk("unicode", 0)?.unwrap();
        assert_eq!(unicode.text.as_deref(), Some("café🙂\tβ"));
        assert_eq!((unicode.start_byte, unicode.end_byte), (2, 14));
        assert!(core
            .read_document_chunk("no-text", 0)?
            .unwrap()
            .text
            .is_none());
        for index in [3, 4, usize::MAX] {
            assert!(core.read_document_chunk("doc", index)?.is_none());
        }
        assert!(core.read_document_chunk("absent", 0)?.is_none());
    }
    let previous = core.read_document_chunk("doc", 0)?.unwrap();
    core.memorize_document_with(input("doc", "new passage", true), &options)?;
    let updated = core.read_document_chunk("doc", 0)?.unwrap();
    assert_eq!(updated.document.version, previous.document.version);
    assert_ne!(updated.revision, previous.revision);
    assert_eq!(updated.text.as_deref(), Some("new passage"));
    core.delete_document("doc")?;
    assert!(core.read_document_chunk("doc", 0)?.is_none());
    Ok(())
}

#[cfg(feature = "security")]
#[test]
fn encrypted_chunk_reads_work_live_and_from_a_sealed_checkpoint() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = WarmConfig::default();
    config.document_checkpoint.min_journal_bytes = 0;
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some("chunk-fixture".into());
    let core = WarmCore::new(
        256,
        Some(dir.path().display().to_string()),
        Some(config.clone()),
    )?;
    let text = "x".repeat(64 * 1024);
    core.memorize_document_with(
        input("sealed", &text, true),
        &IngestOptions {
            encode: EncodeMode::Lexical,
            ..Default::default()
        },
    )?;
    assert_eq!(
        core.read_document_chunk("sealed", 0)?
            .unwrap()
            .text
            .as_deref(),
        Some(text.as_str())
    );
    drop(core);
    let core = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config))?;
    assert_eq!(
        core.read_document_chunk("sealed", 0)?
            .unwrap()
            .text
            .as_deref(),
        Some(text.as_str())
    );
    Ok(())
}

#[test]
fn closing_after_damaged_text_cache_preserves_journal_recovery() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = WarmConfig::default();
    config.document_checkpoint.min_journal_bytes = 0;
    let core = WarmCore::new(
        256,
        Some(dir.path().display().to_string()),
        Some(config.clone()),
    )?;
    core.memorize_document_with(
        input("doc", "alpha beta", true),
        &IngestOptions {
            encode: EncodeMode::Lexical,
            ..Default::default()
        },
    )?;
    std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("documents.text"))?
        .set_len(8)?;
    let closed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(core)));
    assert!(
        closed.is_ok(),
        "closing must not panic when an optional checkpoint cannot read the text cache"
    );
    let reopened = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config))?;
    assert_eq!(
        reopened.get_document("doc").unwrap().text.as_deref(),
        Some("alpha beta")
    );
    Ok(())
}

#[test]
fn same_length_cache_damage_cannot_enter_a_checkpoint() -> Result<()> {
    reject_damaged_cache(false, true)
}

#[test]
fn same_length_cache_damage_cannot_enter_a_compacted_generation() -> Result<()> {
    reject_damaged_cache(true, true)
}

#[test]
fn same_length_cache_damage_cannot_enter_compacted_journal_frames() -> Result<()> {
    reject_damaged_cache(true, false)
}

fn reject_damaged_cache(compact: bool, checkpoints: bool) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = WarmConfig::default();
    config.document_checkpoint.min_journal_bytes = 0;
    config.document_checkpoint.enabled = checkpoints;
    let core = WarmCore::new(
        256,
        Some(dir.path().display().to_string()),
        Some(config.clone()),
    )?;
    core.memorize_document_with(
        input("doc", "alpha beta", true),
        &IngestOptions {
            encode: EncodeMode::Lexical,
            ..Default::default()
        },
    )?;
    let path = dir.path().join("documents.text");
    let mut cache = std::fs::read(&path)?;
    assert_eq!(&cache[8..], b"alpha beta");
    cache[8] = b'o';
    std::fs::write(path, cache)?;
    let error = core.read_document_chunk("doc", 0).unwrap_err();
    assert!(format!("{error:#}").contains("chunk text checksum mismatch"));
    if compact {
        assert!(core.compact().is_err());
    }
    drop(core);
    let reopened = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config))?;
    assert_eq!(
        reopened.get_document("doc").unwrap().text.as_deref(),
        Some("alpha beta")
    );
    Ok(())
}
