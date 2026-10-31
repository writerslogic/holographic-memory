// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use tempfile::tempdir;

fn legacy(path: &Path, version: u32) -> Result<()> {
    std::fs::create_dir_all(path)?;
    let manifest = FormatManifest {
        magic: FORMAT_MAGIC.into(),
        version,
        segment_size: if version == 3 {
            SEGMENT_SIZE
        } else {
            LEGACY_SEGMENT_SIZE
        },
    };
    std::fs::write(path.join(FORMAT_MANIFEST), serde_json::to_vec(&manifest)?)?;
    Ok(())
}

#[test]
fn legacy_frames_geometry_and_compaction_survive_reopening() -> Result<()> {
    for version in [1, 2, 3] {
        let size = if version == 3 {
            SEGMENT_SIZE
        } else {
            LEGACY_SEGMENT_SIZE
        };
        let root = tempdir()?;
        let live = root.path().join("live");
        legacy(&live, version)?;
        let arena = PersistentArena::new(&live)?;
        let current = PersistentArena::new(root.path().join("current"))?;
        for payload in [b"first".as_slice(), &[17; 4097], b"last"] {
            assert_eq!(arena.write_slice(payload)?, current.write_slice(payload)?);
        }
        let length = arena.stats().used_bytes;
        assert_eq!(arena.read_slice(0, length)?, current.read_slice(0, length)?);
        assert_eq!(arena.stats().format_version, version.max(2));
        assert_eq!(arena.stats().capacity_bytes, size);
        drop(arena);
        let arena = PersistentArena::new(&live)?;
        assert_eq!(arena.write_slice(b"after restart")?, length);
        let stage = arena.new_generation_dir()?;
        let mut staged = StagedArena::new(&stage, Some(&arena))?;
        for payload in [b"first".as_slice(), &[17; 4097], b"last", b"after restart"] {
            staged.write_slice(payload)?;
        }
        let expected = arena.read_slice(0, arena.stats().used_bytes)?;
        arena.replace_with_compacted(staged.finish()?.prepare_compacted()?, false)?;
        assert_eq!(arena.read_slice(0, expected.len())?, expected);
        drop(arena);
        let arena = PersistentArena::new(&live)?;
        assert_eq!(arena.stats().capacity_bytes, size);
        assert_eq!(arena.read_slice(0, expected.len())?, expected);
        let manifest: FormatManifest =
            serde_json::from_slice(&std::fs::read(live.join(FORMAT_MANIFEST))?)?;
        assert_eq!(manifest.version, version);
        assert_eq!(manifest.segment_size, size);
    }
    Ok(())
}

#[test]
fn premanifest_store_retains_legacy_geometry_and_complete_frames() -> Result<()> {
    let root = tempdir()?;
    legacy(root.path(), 2)?;
    let arena = PersistentArena::new(root.path())?;
    let offset = arena.write_slice(b"pre-manifest record")?;
    let expected = arena.read_slice(0, arena.stats().used_bytes)?;
    drop(arena);
    std::fs::remove_file(root.path().join(FORMAT_MANIFEST))?;
    let arena = PersistentArena::new(root.path())?;
    assert_eq!(arena.read_slice(0, expected.len())?, expected);
    assert_eq!(arena.stats().capacity_bytes, LEGACY_SEGMENT_SIZE);
    assert_eq!(arena.read_frame(offset)?.0, b"pre-manifest record");
    arena.write_slice(b"new record")?;
    Ok(())
}

#[test]
fn mismatched_compaction_geometry_preserves_the_acknowledged_generation() -> Result<()> {
    for legacy_live in [false, true] {
        let root = tempdir()?;
        let live = root.path().join("live");
        let stage = root.path().join("stage");
        legacy(if legacy_live { &live } else { &stage }, 2)?;
        let arena = PersistentArena::new(&live)?;
        arena.write_slice(b"acknowledged")?;
        let checkpoint = arena.checkpoint();
        let staged = PersistentArena::new(&stage)?;
        staged.write_slice(b"incompatible")?;
        assert!(arena
            .replace_with_compacted(staged.prepare_compacted()?, false)
            .is_err());
        assert_eq!(arena.checkpoint(), checkpoint);
        assert!(!live.join("CURRENT").exists());
        drop(arena);
        assert_eq!(
            PersistentArena::new(&live)?.read_frame(0)?.0,
            b"acknowledged"
        );
    }
    Ok(())
}

#[test]
fn manifest_geometry_boundaries_fail_before_segment_mutation() -> Result<()> {
    for version in [1, 2, 3, 4] {
        let expected = if version != 3 {
            LEGACY_SEGMENT_SIZE
        } else {
            SEGMENT_SIZE
        };
        for size in [expected - 1, expected, expected + 1] {
            let root = tempdir()?;
            let arena_path = root.path().join("vectors_data.bin");
            std::fs::create_dir(&arena_path)?;
            let manifest = FormatManifest {
                magic: FORMAT_MAGIC.into(),
                version,
                segment_size: size,
            };
            std::fs::write(
                arena_path.join(FORMAT_MANIFEST),
                serde_json::to_vec(&manifest)?,
            )?;
            if size == expected {
                drop(PersistentArena::new(&arena_path)?);
                let inspection = crate::core::admin::inspect_store(root.path(), true)?;
                assert_eq!(inspection.format_version, version);
                assert_eq!(
                    inspection.allocated_bytes,
                    if version == 4 && INCREMENTAL_ALLOCATION {
                        SEGMENT_SIZE
                    } else {
                        expected
                    } as u64
                );
            } else {
                let segment = arena_path.join("seg_0.bin");
                std::fs::write(&segment, b"must remain untouched")?;
                assert!(PersistentArena::new(&arena_path).is_err());
                assert!(crate::core::admin::inspect_store(root.path(), true).is_err());
                assert_eq!(std::fs::read(segment)?, b"must remain untouched");
            }
        }
    }
    Ok(())
}

#[test]
fn legacy_plain_and_encrypted_engine_recovery_preserves_queries_and_geometry() -> Result<()> {
    use crate::core::{
        documents::{DocumentInput, SearchOptions},
        WarmConfig, WarmCore,
    };
    for version in [1, 2] {
        for encrypted in [false, true]
            .into_iter()
            .filter(|value| !value || cfg!(feature = "security"))
        {
            let root = tempdir()?;
            let mut config = WarmConfig::default();
            config.security.encryption_enabled = encrypted;
            if encrypted {
                config.security.encryption_passphrase = Some("legacy-layout-fixture".into());
            }
            crate::core::schema::validate_store(root.path(), 1024, &config)?;
            legacy(&root.path().join("vectors_data.bin"), version)?;
            let path = root.path().display().to_string();
            let core = WarmCore::new(1024, Some(path.clone()), Some(config.clone()))?;
            let document: DocumentInput = serde_json::from_value(serde_json::json!({
                "id":"legacy", "text":"needle durable legacy document", "metadata":{"unicode":"文書"}
            }))?;
            core.memorize_document(document)?;
            let before =
                serde_json::to_value(core.search_documents("needle", &SearchOptions::default())?)?;
            assert!(!before.as_array().unwrap().is_empty());
            drop(core);
            let core = WarmCore::new(1024, Some(path.clone()), Some(config.clone()))?;
            assert_eq!(
                serde_json::to_value(core.search_documents("needle", &SearchOptions::default())?)?,
                before
            );
            core.compact()?;
            assert_eq!(core.storage_health().format_version, 2);
            assert_eq!(core.arena.stats().capacity_bytes, LEGACY_SEGMENT_SIZE);
            drop(core);
            let core = WarmCore::new(1024, Some(path), Some(config))?;
            assert_eq!(
                serde_json::to_value(core.search_documents("needle", &SearchOptions::default())?)?,
                before
            );
            assert_eq!(core.arena.stats().capacity_bytes, LEGACY_SEGMENT_SIZE);
            let inspection = crate::core::admin::inspect_store(root.path(), true)?;
            assert_eq!(inspection.format_version, version);
            assert_eq!(inspection.allocated_bytes, LEGACY_SEGMENT_SIZE as u64);
        }
    }
    Ok(())
}

#[test]
fn compacted_store_without_root_manifest_keeps_published_generation_geometry() -> Result<()> {
    for version in [1, 2, 3, 4] {
        let root = tempdir()?;
        if version < 4 {
            legacy(root.path(), version)?;
        }
        let arena = PersistentArena::new(root.path())?;
        arena.write_slice(b"durable frame")?;
        let expected_capacity = arena.stats().capacity_bytes;
        let stage = arena.new_generation_dir()?;
        let mut staged = StagedArena::new(&stage, Some(&arena))?;
        staged.write_slice(b"durable frame")?;
        arena.replace_with_compacted(staged.finish()?.prepare_compacted()?, false)?;
        drop(arena);
        std::fs::remove_file(root.path().join(FORMAT_MANIFEST))?;
        let restored = PersistentArena::new(root.path())?;
        assert_eq!(restored.stats().capacity_bytes, expected_capacity);
        assert_eq!(restored.read_frame(0)?.0, b"durable frame");
        let manifest: FormatManifest =
            serde_json::from_slice(&std::fs::read(root.path().join(FORMAT_MANIFEST))?)?;
        assert_eq!(manifest.version, version);
    }
    Ok(())
}

#[test]
fn legacy_compaction_tail_survives_each_plain_and_encrypted_publication_interruption() -> Result<()>
{
    use crate::core::{documents::SearchOptions, WarmCore};
    for version in [1, 2, 3] {
        for encrypted in [false, true]
            .into_iter()
            .filter(|value| !value || cfg!(feature = "security"))
        {
            let mut config = crate::core::engine::fixture_config(2);
            config.security.encryption_enabled = encrypted;
            if encrypted {
                config.security.encryption_passphrase = Some("compaction-replay-fixture".into());
            }
            for phase in ["staged", "published", "mapped"] {
                let root = tempdir()?;
                crate::core::schema::validate_store(root.path(), 1024, &config)?;
                legacy(&root.path().join("vectors_data.bin"), version)?;
                let output = std::process::Command::new(std::env::current_exe()?)
                    .args([
                        "--exact",
                        "core::engine::compaction::tests::compaction_replay_crash_child",
                    ])
                    .env("WARM_TEST_CORE_REPLAY_ROOT", root.path())
                    .env("WARM_TEST_CORE_REPLAY_ENCRYPTED", encrypted.to_string())
                    .env("WARM_TEST_COMPACTION_PHASE", phase)
                    .output()?;
                assert_eq!(
                    output.status.code(),
                    Some(73),
                    "{version}/{encrypted}/{phase}: {output:?}"
                );
                let core = WarmCore::new(
                    1024,
                    Some(root.path().display().to_string()),
                    Some(config.clone()),
                )?;
                let options = SearchOptions {
                    embedding: Some(vec![1.0, 0.0]),
                    ..Default::default()
                };
                let results = core.search_documents("needle", &options)?;
                let mut ids: Vec<_> = results
                    .iter()
                    .map(|result| result.document_id.as_str())
                    .collect();
                ids.sort_unstable();
                assert_eq!(ids, ["insert", "replace"]);
                assert!(results.iter().all(|result| result.version == "new"
                    && result.source_uri.as_deref() == Some("source:new")
                    && result.semantic_score == Some(1.0)));
                assert_eq!(core.relation_count(), 0);
                assert_eq!(core.vector_count(), 3);
                assert_eq!(
                    core.query(&core.encode_text("replacement"), 1)[0].id,
                    "vector"
                );
                assert_eq!(
                    core.arena.stats().capacity_bytes,
                    if version == 3 {
                        SEGMENT_SIZE
                    } else {
                        LEGACY_SEGMENT_SIZE
                    }
                );
                core.compact()?;
                assert_eq!(
                    serde_json::to_value(core.search_documents("needle", &options)?)?,
                    serde_json::to_value(results)?
                );
                assert_eq!(
                    core.arena.stats().capacity_bytes,
                    if version == 3 {
                        SEGMENT_SIZE
                    } else {
                        LEGACY_SEGMENT_SIZE
                    }
                );
            }
        }
    }
    Ok(())
}

#[test]
fn oversized_read_only_mapping_cannot_extend_declared_segment_bounds() -> Result<()> {
    for version in [2, 3] {
        let root = tempdir()?;
        legacy(root.path(), version)?;
        let arena = PersistentArena::new(root.path())?;
        let size = arena.segment_size;
        drop(arena);
        let first = OpenOptions::new()
            .write(true)
            .open(root.path().join("seg_0.bin"))?;
        first.set_len((size + 1) as u64)?;
        let last = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.path().join("seg_1.bin"))?;
        last.set_len(size as u64)?;
        first.sync_all()?;
        last.sync_all()?;
        let restored = PersistentArena::new(root.path())?;
        for length in [0, 1, 2] {
            assert_eq!(restored.read_slice(size - 1, length).is_ok(), length <= 1);
        }
        assert!(restored.read_slice(size - 1, usize::MAX).is_err());
    }
    Ok(())
}
