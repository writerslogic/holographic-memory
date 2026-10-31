// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use writerslogic_warm::{core::WarmConfig, EntangledHVec, WarmCore};

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
    let mut config = WarmConfig::default();
    config.meaning.enabled = true;
    let warm = WarmCore::new(16384, Some(dir.path().display().to_string()), Some(config))?;
    warm.memorize_triplet(
        "t1".into(),
        "paris".into(),
        "capital_of".into(),
        "france".into(),
    )?;
    warm.memorize_triplet("t2".into(), "john".into(), "father".into(), "mark".into())?;
    warm.memorize_triplet("t3".into(), "mark".into(), "father".into(), "bob".into())?;
    assert_eq!(warm.meaning_triple_count(), 3);
    let s = warm.encode_text("paris");
    let r = warm.encode_text("capital_of");
    let result = warm.structural_query(&[("subject", &s), ("relation", &r)], "object");
    assert_eq!(result.first().map(|r| r.entity_id.as_str()), Some("france"));
    assert_eq!(
        warm.multi_hop("john", &["father", "father"])[0].entity_id,
        "bob"
    );
    Ok(())
}

#[test]
fn relation_deletion_survives_restart() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let warm = WarmCore::new(4096, Some(path.clone()), None)?;
    let relation = serde_json::from_value(serde_json::json!({
        "sourceId": "a", "relationType": "knows", "targetId": "b",
        "properties": null, "validFrom": 0, "validTo": 0
    }))?;
    warm.add_relation(&relation)?;
    warm.remove_relation("a", "knows", "b")?;
    assert_eq!(warm.relation_count(), 0);
    warm.flush()?;
    drop(warm);
    assert_eq!(WarmCore::new(4096, Some(path), None)?.relation_count(), 0);
    Ok(())
}

#[test]
fn incompatible_dimensions_are_rejected() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    drop(WarmCore::new(4096, Some(path.clone()), None)?);
    assert!(WarmCore::new(8192, Some(path), None).is_err());
    Ok(())
}

#[cfg(not(feature = "security"))]
#[test]
fn unavailable_encryption_is_rejected() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = WarmConfig::default();
    config.security.encryption_enabled = true;
    assert!(WarmCore::new(4096, Some(dir.path().display().to_string()), Some(config)).is_err());
    Ok(())
}

#[test]
fn post_training_insert_survives_restart() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let warm = WarmCore::new(4096, Some(path.clone()), None)?;
    for i in 0..1000 {
        warm.memorize(
            format!("base-{i}"),
            EntangledHVec::new_deterministic(4096, i),
        )?;
    }
    warm.train_nsg()?;
    let newest = EntangledHVec::new_deterministic(4096, 999999);
    warm.memorize("newest".into(), newest.clone())?;
    warm.flush()?;
    drop(warm);
    let warm = WarmCore::new(4096, Some(path), None)?;
    assert_eq!(warm.query(&newest, 1)[0].id, "newest");
    Ok(())
}

#[test]
fn concurrent_updates_and_compaction_replay_identically() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let warm = std::sync::Arc::new(WarmCore::new(4096, Some(path.clone()), None)?);
    let writer = std::sync::Arc::clone(&warm);
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
        warm.compact()?;
    }
    worker.join().expect("writer must complete")?;
    assert_eq!(warm.vector_count(), 60);
    drop(warm);
    let warm = WarmCore::new(4096, Some(path), None)?;
    assert_eq!(warm.vector_count(), 60);
    for i in (0..90).filter(|i| i % 3 != 0) {
        let vector = EntangledHVec::new_deterministic(4096, i + 1000);
        assert_eq!(warm.query(&vector, 1)[0].id, format!("item-{i}"));
    }
    Ok(())
}

#[test]
fn invalid_batch_cannot_commit_a_prefix() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let warm = WarmCore::new(4096, Some(dir.path().display().to_string()), None)?;
    let items = [
        writerslogic_warm::MemorizeBatchItem {
            id: "valid".into(),
            text: "first item".into(),
        },
        writerslogic_warm::MemorizeBatchItem {
            id: "".into(),
            text: "invalid id".into(),
        },
    ];
    assert!(warm.memorize_batch(&items).is_err());
    assert_eq!(warm.vector_count(), 0);
    Ok(())
}

#[test]
fn replacing_and_deleting_meaning_removes_owned_facts_after_compaction() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let mut config = WarmConfig::default();
    config.meaning.enabled = true;
    config.meaning.auto_decompose = true;
    let warm = WarmCore::new(4096, Some(path.clone()), Some(config.clone()))?;
    warm.memorize_meaning("source", "Paris is a city. France is a country.")?;
    let original = warm.meaning_triple_count();
    assert!(original >= 2);
    warm.memorize_meaning("source", "Berlin is a city.")?;
    assert_eq!(warm.meaning_triple_count(), 1);
    let atoms = warm.meaning_atom_count();
    warm.compact()?;
    drop(warm);
    let warm = WarmCore::new(4096, Some(path.clone()), Some(config.clone()))?;
    assert_eq!(warm.meaning_atom_count(), atoms);
    assert_eq!(warm.meaning_triple_count(), 1);
    warm.delete("source")?;
    assert_eq!(warm.meaning_triple_count(), 0);
    drop(warm);
    let warm = WarmCore::new(4096, Some(path), Some(config))?;
    assert_eq!(warm.meaning_triple_count(), 0);
    assert_eq!(warm.vector_count(), 0);
    Ok(())
}

#[test]
fn corrupt_ann_cache_does_not_prevent_recovery_of_durable_data() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let warm = WarmCore::new(4096, Some(path.clone()), None)?;
    for i in 0..1000 {
        warm.memorize(format!("v{i}"), EntangledHVec::new_deterministic(4096, i))?;
    }
    warm.train_nsg()?;
    warm.flush()?;
    drop(warm);
    std::fs::write(dir.path().join("nsg_index.bin"), b"corrupted cache")?;
    let warm = WarmCore::new(4096, Some(path), None)?;
    assert_eq!(warm.vector_count(), 1000);
    assert_eq!(
        warm.query(&EntangledHVec::new_deterministic(4096, 123), 1)[0].id,
        "v123"
    );
    Ok(())
}

fn assert_best_first(results: &[writerslogic_warm::RetrievalResult], expected_top: &str) {
    let listed: Vec<String> = results
        .iter()
        .map(|r| format!("{}={:.3}", r.id, r.similarity))
        .collect();
    assert_eq!(results.len(), 5, "{listed:?}");
    assert_eq!(results[0].id, expected_top, "{listed:?}");
    assert!(
        results
            .windows(2)
            .all(|w| w[0].similarity >= w[1].similarity),
        "results are not in descending similarity: {listed:?}"
    );
    assert!(results[0].similarity > results[4].similarity, "{listed:?}");
}

#[test]
fn every_query_route_returns_best_match_first() -> anyhow::Result<()> {
    const DIM: usize = 16384;
    let scalar = |i: usize| EntangledHVec::from_scalar(i as f64, 0.0, 4000.0, DIM);
    let query = scalar(20);

    // Exact scan (fewer than 1,000 vectors), then the inverted index once past it.
    let dir = tempfile::tempdir()?;
    let single = WarmCore::new(DIM as u32, Some(dir.path().display().to_string()), None)?;
    for i in 0..40 {
        single.memorize(format!("s-{i}"), scalar(i))?;
    }
    assert_best_first(&single.query(&query, 5), "s-20");
    assert_best_first(&single.federated_query(&[], &query, 5)?, "s-20");
    for i in 40..1200 {
        single.memorize(format!("s-{i}"), scalar(i))?;
    }
    assert_best_first(&single.query(&query, 5), "s-20");

    // Multi-shard merge.
    let dir = tempfile::tempdir()?;
    let mut config = WarmConfig::default();
    config.shard.enabled = true;
    config.shard.shard_count = 4;
    let sharded = WarmCore::new(
        DIM as u32,
        Some(dir.path().display().to_string()),
        Some(config),
    )?;
    for i in 0..40 {
        sharded.memorize(format!("s-{i}"), scalar(i))?;
    }
    assert_best_first(&sharded.query(&query, 5), "s-20");
    Ok(())
}
