// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use writerslogic_warm::core::WarmConfig;
use writerslogic_warm::{DocumentInput, EmbeddingSpace, SearchOptions, WarmCore};

fn input(id: &str, text: &str) -> DocumentInput {
    serde_json::from_value(serde_json::json!({
        "id": id, "text": text, "sourceUri": "notes.md", "version": "1",
        "metadata": {"project": "alpha"}, "chunkWords": 5, "overlapWords": 1
    }))
    .unwrap()
}

#[test]
fn document_lifecycle_preserves_sources_filters_and_versions() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let warm = WarmCore::new(4096, Some(path.clone()), None)?;
    let original = input(
        "manual",
        "Café records explain backups. Restore deleted documents from verified snapshots safely.",
    );
    let chunks = writerslogic_warm::chunk_document(&original)?;
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        assert_eq!(
            &original.text[chunk.start_byte as usize..chunk.end_byte as usize],
            chunk.text
        );
    }
    warm.memorize_document(original)?;
    let results = warm.search_documents("restore snapshots", &SearchOptions::default())?;
    assert_eq!(results[0].document_id, "manual");
    assert_eq!(results[0].source_uri.as_deref(), Some("notes.md"));
    assert!(results[0].text.as_deref().unwrap().contains("snapshots"));
    let filtered = SearchOptions {
        filter: Some(serde_json::json!({"project": "other"})),
        ..Default::default()
    };
    assert!(warm.search_documents("restore", &filtered)?.is_empty());
    let mut replacement = input("manual", "Updated indexing instructions.");
    replacement.version = Some("2".into());
    warm.memorize_document(replacement)?;
    assert!(warm
        .search_documents("snapshots", &SearchOptions::default())?
        .is_empty());
    warm.compact()?;
    drop(warm);
    let warm = WarmCore::new(4096, Some(path.clone()), None)?;
    let results = warm.search_documents("indexing", &SearchOptions::default())?;
    assert_eq!(results[0].version, "2");
    assert_eq!(warm.vector_count(), 1);
    assert!(warm.delete_document("manual")?);
    drop(warm);
    let warm = WarmCore::new(4096, Some(path), None)?;
    assert_eq!(warm.vector_count(), 0);
    assert!(warm
        .search_documents("indexing", &SearchOptions::default())?
        .is_empty());
    Ok(())
}

#[test]
fn semantic_candidates_use_cosine_and_filter_before_ranking() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let config = WarmConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "fixture".into(),
            revision: "1".into(),
            dimensions: 3,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    let warm = WarmCore::new(4096, Some(dir.path().display().to_string()), Some(config))?;
    let mut a = input("positive", "An automobile.");
    a.embeddings = Some(vec![vec![1.0, 0.1, 0.0]]);
    warm.memorize_document(a)?;
    let mut b = input("opposite", "Something unrelated.");
    b.embeddings = Some(vec![vec![-1.0, -0.1, 0.0]]);
    warm.memorize_document(b)?;
    let options = SearchOptions {
        embedding: Some(vec![1.0, 0.0, 0.0]),
        k: Some(1),
        ..Default::default()
    };
    let results = warm.search_documents("car", &options)?;
    assert_eq!(results[0].document_id, "positive");
    assert!(results[0].semantic_score.unwrap() > 0.99);
    let options = SearchOptions {
        filter: Some(serde_json::json!({"project": "missing"})),
        ..options
    };
    assert!(warm.search_documents("car", &options)?.is_empty());
    Ok(())
}

#[test]
fn malformed_document_update_does_not_replace_existing_content() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let warm = WarmCore::new(4096, Some(dir.path().display().to_string()), None)?;
    warm.memorize_document(input("d", "Keep this original."))?;
    let mut invalid = input("d", "Invalid replacement.");
    invalid.overlap_words = Some(5);
    assert!(warm.memorize_document(invalid).is_err());
    assert_eq!(
        warm.search_documents("original", &SearchOptions::default())?[0].document_id,
        "d"
    );
    Ok(())
}

#[test]
fn reencoding_is_atomic_and_preserves_original_sources() -> anyhow::Result<()> {
    use writerslogic_warm::core::admin::reencode_documents;
    let dir = tempfile::tempdir()?;
    let source = dir.path().join("source.jsonl");
    let target = dir.path().join("new-store");
    let content = "{\"id\":\"manual\",\"text\":\"Restore a verified backup.\"}\n";
    std::fs::write(&source, format!("{content}{{invalid json}}\n"))?;
    assert!(reencode_documents(&source, &target, 4096, WarmConfig::default()).is_err());
    assert!(!target.exists());
    assert!(std::fs::read_to_string(&source)?.contains("invalid json"));
    std::fs::write(&source, content)?;
    reencode_documents(&source, &target, 4096, WarmConfig::default())?;
    assert_eq!(std::fs::read_to_string(&source)?, content);
    let warm = WarmCore::new(4096, Some(target.display().to_string()), None)?;
    assert_eq!(
        warm.search_documents("backup", &SearchOptions::default())?[0].document_id,
        "manual"
    );
    Ok(())
}

#[test]
fn bounded_document_inputs_fail_without_committing() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let warm = WarmCore::new(4096, Some(dir.path().display().to_string()), None)?;
    assert!(warm
        .memorize_document(input("long-token", &"x".repeat(65537)))
        .is_err());
    let mut document = input("metadata", "Short content.");
    document.metadata = Some(serde_json::json!({"oversized": "x".repeat(65536)}));
    assert!(warm.memorize_document(document).is_err());
    assert_eq!(warm.vector_count(), 0);
    Ok(())
}

#[test]
fn document_readers_observe_atomic_versions_during_compaction() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().display().to_string();
    let warm = WarmCore::new(4096, Some(path.clone()), None)?;
    let mut first = input("changing", "current revision1");
    first.version = Some("1".into());
    warm.memorize_document(first)?;
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| -> anyhow::Result<()> {
        let reader = scope.spawn(|| {
            let mut observations = Vec::new();
            for _ in 0..20 {
                barrier.wait();
                observations.push(
                    warm.search_documents("current revision1 revision2", &SearchOptions::default()),
                );
                barrier.wait();
            }
            observations
        });
        let mut updates = Vec::new();
        for i in 0usize..20 {
            barrier.wait();
            let version = 1 + i % 2;
            let mut replacement = input("changing", &format!("current revision{version}"));
            replacement.version = Some(version.to_string());
            updates.push(warm.memorize_document(replacement));
            let compacted = if i.is_multiple_of(5) {
                warm.compact()
            } else {
                Ok(())
            };
            barrier.wait();
            updates.push(compacted.map(|()| 0));
        }
        for observation in reader.join().unwrap() {
            let found = observation?;
            assert_eq!(found.len(), 1);
            assert!(found[0]
                .text
                .as_ref()
                .unwrap()
                .contains(&format!("revision{}", found[0].version)));
        }
        for update in updates {
            update?;
        }
        Ok(())
    })?;
    assert!(warm.delete_document("changing")?);
    warm.compact()?;
    drop(warm);
    let restored = WarmCore::new(4096, Some(path), None)?;
    assert!(restored
        .search_documents("current revision2", &SearchOptions::default())?
        .is_empty());
    Ok(())
}
