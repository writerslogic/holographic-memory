// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use holographic_memory::core::{admin::reencode_documents, encoding::TextEncoder, HmsConfig};
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde::Deserialize;

fn config(encoder: TextEncoder) -> HmsConfig {
    HmsConfig {
        text_encoder: encoder,
        ..Default::default()
    }
}

fn document(id: &str, text: &str) -> DocumentInput {
    serde_json::from_value(serde_json::json!({
        "id": id, "text": text, "sourceUri": "original.md", "version": "revision-2",
        "metadata": {"project": "migration"}, "chunkWords": 5, "overlapWords": 1
    }))
    .unwrap()
}

#[derive(Deserialize)]
struct Golden {
    text: String,
    dim: u32,
    majority: Vec<u32>,
    capped_union: Vec<u32>,
}

#[test]
fn encoders_match_prechange_vectors_at_density_boundaries() -> Result<()> {
    let goldens: Vec<Golden> = serde_json::from_str(include_str!("fixtures/text-encoder-v1.json"))?;
    for encoder in [
        TextEncoder::MultiscaleWordsV1,
        TextEncoder::MultiscaleCappedUnionV1,
    ] {
        if encoder == TextEncoder::MultiscaleCappedUnionV1 && !cfg!(feature = "experimental") {
            continue;
        }
        for dim in [1, 255, 256, 257, 1024, 4096, 16384] {
            let dir = tempfile::tempdir()?;
            let hms = HmsCore::new(
                dim,
                Some(dir.path().display().to_string()),
                Some(config(encoder)),
            )?;
            for row in goldens.iter().filter(|row| row.dim == dim) {
                let expected = match encoder {
                    TextEncoder::MultiscaleWordsV1 => &row.majority,
                    TextEncoder::MultiscaleCappedUnionV1 => &row.capped_union,
                };
                let actual = hms.encode_text(&row.text);
                assert_eq!(actual.indices(), expected, "{encoder:?}, D{dim}");
                assert!(actual.indices().len() <= (dim as usize / 256).max(1));
                assert!(actual.indices().windows(2).all(|pair| pair[0] < pair[1]));
                assert!(actual.indices().iter().all(|&index| index < dim));
            }
        }
    }
    Ok(())
}

#[test]
fn capped_union_requires_experimental_before_creating_a_store() -> Result<()> {
    if cfg!(feature = "experimental") {
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let target = dir.path().join("unavailable");
    let error = HmsCore::new(
        1024,
        Some(target.display().to_string()),
        Some(config(TextEncoder::MultiscaleCappedUnionV1)),
    )
    .err()
    .expect("experimental encoder must be unavailable");
    assert!(error.to_string().contains("experimental feature"));
    assert!(!target.exists());
    Ok(())
}

#[test]
fn incompatible_encoders_preserve_schema_and_recover_after_compaction() -> Result<()> {
    if !cfg!(feature = "experimental") {
        return Ok(());
    }
    for dim in [1024, 4096, 16384] {
        for encoder in [
            TextEncoder::MultiscaleWordsV1,
            TextEncoder::MultiscaleCappedUnionV1,
        ] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().display().to_string();
            let input = document(
                "manual",
                "Café records explain backups and safe recovery steps.",
            );
            let chunks = holographic_memory::chunk_document(&input)?;
            let hms = HmsCore::new(dim, Some(path.clone()), Some(config(encoder)))?;
            hms.memorize_document(input)?;
            for chunk in &chunks {
                let found = hms.query(&hms.encode_text(&chunk.text), 100);
                assert!(found
                    .iter()
                    .any(|r| r.id == chunk.id && r.similarity == 1.0));
            }
            let before = hms.search_documents("backups", &SearchOptions::default())?;
            hms.compact()?;
            drop(hms);
            let schema = std::fs::read(dir.path().join("store.json"))?;
            let other = match encoder {
                TextEncoder::MultiscaleWordsV1 => TextEncoder::MultiscaleCappedUnionV1,
                TextEncoder::MultiscaleCappedUnionV1 => TextEncoder::MultiscaleWordsV1,
            };
            assert!(HmsCore::new(dim, Some(path.clone()), Some(config(other))).is_err());
            assert_eq!(std::fs::read(dir.path().join("store.json"))?, schema);
            let hms = HmsCore::new(dim, Some(path), Some(config(encoder)))?;
            assert_eq!(
                serde_json::to_value(hms.search_documents("backups", &SearchOptions::default())?)?,
                serde_json::to_value(before)?
            );
            for chunk in &chunks {
                assert!(hms
                    .query(&hms.encode_text(&chunk.text), 100)
                    .iter()
                    .any(|r| r.id == chunk.id && r.similarity == 1.0));
            }
        }
    }
    Ok(())
}

#[test]
fn reencoding_changes_only_the_new_store_and_rejects_incomplete_input() -> Result<()> {
    if !cfg!(feature = "experimental") {
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let original_path = dir.path().join("original-store");
    let target = dir.path().join("union-store");
    let source = dir.path().join("source.jsonl");
    let input = document(
        "manual",
        "Café records explain backups and safe recovery steps.",
    );
    let old = HmsCore::new(4096, Some(original_path.display().to_string()), None)?;
    old.memorize_document(input.clone())?;
    let expected = old.search_documents("backups", &SearchOptions::default())?;
    drop(old);
    let old_schema = std::fs::read(original_path.join("store.json"))?;
    let line = format!("{}\n", serde_json::to_string(&input)?);
    let invalid = format!("{line}{{incomplete\n");
    std::fs::write(&source, &invalid)?;
    let union_config = config(TextEncoder::MultiscaleCappedUnionV1);
    assert!(reencode_documents(&source, &target, 4096, union_config.clone()).is_err());
    assert!(!target.exists());
    assert_eq!(std::fs::read_to_string(&source)?, invalid);
    std::fs::write(&source, &line)?;
    let inspection = reencode_documents(&source, &target, 4096, union_config.clone())?;
    assert!(inspection.checksums_verified);
    assert_eq!(std::fs::read_to_string(&source)?, line);
    assert_eq!(std::fs::read(original_path.join("store.json"))?, old_schema);
    let hms = HmsCore::new(
        4096,
        Some(target.display().to_string()),
        Some(union_config.clone()),
    )?;
    assert_eq!(
        serde_json::to_value(hms.search_documents("backups", &SearchOptions::default())?)?,
        serde_json::to_value(&expected)?
    );
    let target_schema = std::fs::read(target.join("store.json"))?;
    assert!(reencode_documents(&source, &target, 4096, union_config).is_err());
    assert_eq!(std::fs::read(target.join("store.json"))?, target_schema);
    let old = HmsCore::new(4096, Some(original_path.display().to_string()), None)?;
    assert_eq!(
        serde_json::to_value(old.search_documents("backups", &SearchOptions::default())?)?,
        serde_json::to_value(expected)?
    );
    Ok(())
}

#[test]
fn abrupt_exit_child() -> Result<()> {
    let Ok(path) = std::env::var("HMS_TEST_ENCODER_RECOVERY_ROOT") else {
        return Ok(());
    };
    let hms = HmsCore::new(
        4096,
        Some(path),
        Some(config(TextEncoder::MultiscaleCappedUnionV1)),
    )?;
    hms.memorize_document(document(
        "keep",
        "Acknowledged backups survive abrupt process exits.",
    ))?;
    hms.memorize_document(document(
        "remove",
        "Deleted obsolete backups must stay absent.",
    ))?;
    hms.delete_document("remove")?;
    std::process::exit(73);
}

#[test]
fn new_encoder_recovers_acknowledged_writes_after_abrupt_exit() -> Result<()> {
    if !cfg!(feature = "experimental") {
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let child = std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", "abrupt_exit_child"])
        .env("HMS_TEST_ENCODER_RECOVERY_ROOT", dir.path())
        .output()?;
    assert_eq!(child.status.code(), Some(73), "{:?}", child);
    let hms = HmsCore::new(
        4096,
        Some(dir.path().display().to_string()),
        Some(config(TextEncoder::MultiscaleCappedUnionV1)),
    )?;
    let found = hms.search_documents("backups", &SearchOptions::default())?;
    assert!(!found.is_empty());
    assert!(found.iter().all(|r| r.document_id == "keep"));
    assert!(found
        .iter()
        .all(|r| r.source_uri.as_deref() == Some("original.md")));
    hms.memorize_document(document(
        "new",
        "Recovery permits further acknowledged writes.",
    ))?;
    hms.compact()?;
    Ok(())
}
