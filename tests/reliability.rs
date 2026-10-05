// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use holographic_memory::{core::HmsConfig, EntangledHVec, HmsCore};

#[test]
fn opposite_embeddings_remain_distinct() {
    let dense = [0.1, -0.3, 0.7, 0.2];
    let negative: Vec<f32> = dense.iter().map(|x| -x).collect();
    let a = EntangledHVec::from_dense(&dense, 4096);
    let b = EntangledHVec::from_dense(&negative, 4096);
    assert!(a.similarity(&b) < 0.1);
}

#[test]
fn documented_triplets_answer_structural_and_multihop_queries() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = HmsConfig::default();
    config.meaning.enabled = true;
    let hms = HmsCore::new(16384, Some(dir.path().display().to_string()), Some(config))?;
    hms.memorize_triplet(
        "t1".into(),
        "paris".into(),
        "capital_of".into(),
        "france".into(),
    )?;
    hms.memorize_triplet("t2".into(), "john".into(), "father".into(), "mark".into())?;
    hms.memorize_triplet("t3".into(), "mark".into(), "father".into(), "bob".into())?;
    assert_eq!(hms.meaning_triple_count(), 3);
    let s = hms.encode_text("paris");
    let r = hms.encode_text("capital_of");
    let result = hms.structural_query(&[("subject", &s), ("relation", &r)], "object");
    assert_eq!(result.first().map(|r| r.entity_id.as_str()), Some("france"));
    assert_eq!(
        hms.multi_hop("john", &["father", "father"])[0].entity_id,
        "bob"
    );
    Ok(())
}

#[test]
fn relation_deletion_survives_restart() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let hms = HmsCore::new(4096, Some(path.clone()), None)?;
    let relation = serde_json::from_value(serde_json::json!({
        "sourceId": "a", "relationType": "knows", "targetId": "b",
        "properties": null, "validFrom": 0, "validTo": 0
    }))?;
    hms.add_relation(&relation)?;
    hms.remove_relation("a", "knows", "b")?;
    assert_eq!(hms.relation_count(), 0);
    hms.flush()?;
    drop(hms);
    assert_eq!(HmsCore::new(4096, Some(path), None)?.relation_count(), 0);
    Ok(())
}

#[test]
fn incompatible_dimensions_are_rejected() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    drop(HmsCore::new(4096, Some(path.clone()), None)?);
    assert!(HmsCore::new(8192, Some(path), None).is_err());
    Ok(())
}

#[cfg(not(feature = "security"))]
#[test]
fn unavailable_encryption_is_rejected() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = HmsConfig::default();
    config.security.encryption_enabled = true;
    assert!(HmsCore::new(4096, Some(dir.path().display().to_string()), Some(config)).is_err());
    Ok(())
}

#[test]
fn post_training_insert_survives_restart() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let hms = HmsCore::new(4096, Some(path.clone()), None)?;
    for i in 0..1000 {
        hms.memorize(
            format!("base-{i}"),
            EntangledHVec::new_deterministic(4096, i),
        )?;
    }
    hms.train_nsg()?;
    let newest = EntangledHVec::new_deterministic(4096, 999999);
    hms.memorize("newest".into(), newest.clone())?;
    hms.flush()?;
    drop(hms);
    let hms = HmsCore::new(4096, Some(path), None)?;
    assert_eq!(hms.query(&newest, 1)[0].id, "newest");
    Ok(())
}

#[test]
fn concurrent_updates_and_compaction_replay_identically() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let hms = std::sync::Arc::new(HmsCore::new(4096, Some(path.clone()), None)?);
    let writer = std::sync::Arc::clone(&hms);
    let worker = std::thread::spawn(move || -> anyhow::Result<()> {
        for i in 0..90 {
            let id = format!("item-{i}");
            writer.memorize(id.clone(), EntangledHVec::new_deterministic(4096, i))?;
            writer.memorize(id.clone(), EntangledHVec::new_deterministic(4096, i + 1000))?;
            if i % 3 == 0 {
                writer.delete(&id)?;
            }
        }
        Ok(())
    });
    for _ in 0..8 {
        hms.compact()?;
    }
    worker.join().expect("writer must complete")?;
    assert_eq!(hms.vector_count(), 60);
    drop(hms);
    let hms = HmsCore::new(4096, Some(path), None)?;
    assert_eq!(hms.vector_count(), 60);
    for i in (0..90).filter(|i| i % 3 != 0) {
        let vector = EntangledHVec::new_deterministic(4096, i + 1000);
        assert_eq!(hms.query(&vector, 1)[0].id, format!("item-{i}"));
    }
    Ok(())
}

#[test]
fn invalid_batch_cannot_commit_a_prefix() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let hms = HmsCore::new(4096, Some(dir.path().display().to_string()), None)?;
    let items = [
        holographic_memory::MemorizeBatchItem {
            id: "valid".into(),
            text: "first item".into(),
        },
        holographic_memory::MemorizeBatchItem {
            id: "".into(),
            text: "invalid id".into(),
        },
    ];
    assert!(hms.memorize_batch(&items).is_err());
    assert_eq!(hms.vector_count(), 0);
    Ok(())
}

#[test]
fn replacing_and_deleting_meaning_removes_owned_facts_after_compaction() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let mut config = HmsConfig::default();
    config.meaning.enabled = true;
    config.meaning.auto_decompose = true;
    let hms = HmsCore::new(4096, Some(path.clone()), Some(config.clone()))?;
    hms.memorize_meaning("source", "Paris is a city. France is a country.")?;
    let original = hms.meaning_triple_count();
    assert!(original >= 2);
    hms.memorize_meaning("source", "Berlin is a city.")?;
    assert_eq!(hms.meaning_triple_count(), 1);
    let atoms = hms.meaning_atom_count();
    hms.compact()?;
    drop(hms);
    let hms = HmsCore::new(4096, Some(path.clone()), Some(config.clone()))?;
    assert_eq!(hms.meaning_atom_count(), atoms);
    assert_eq!(hms.meaning_triple_count(), 1);
    hms.delete("source")?;
    assert_eq!(hms.meaning_triple_count(), 0);
    drop(hms);
    let hms = HmsCore::new(4096, Some(path), Some(config))?;
    assert_eq!(hms.meaning_triple_count(), 0);
    assert_eq!(hms.vector_count(), 0);
    Ok(())
}

#[test]
fn corrupt_ann_cache_does_not_prevent_recovery_of_durable_data() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let hms = HmsCore::new(4096, Some(path.clone()), None)?;
    for i in 0..1000 {
        hms.memorize(format!("v{i}"), EntangledHVec::new_deterministic(4096, i))?;
    }
    hms.train_nsg()?;
    hms.flush()?;
    drop(hms);
    std::fs::write(dir.path().join("nsg_index.bin"), b"corrupted cache")?;
    let hms = HmsCore::new(4096, Some(path), None)?;
    assert_eq!(hms.vector_count(), 1000);
    assert_eq!(
        hms.query(&EntangledHVec::new_deterministic(4096, 123), 1)[0].id,
        "v123"
    );
    Ok(())
}
