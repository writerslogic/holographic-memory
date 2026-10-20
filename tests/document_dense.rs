// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use holographic_memory::core::HmsConfig;
use holographic_memory::{DocumentInput, EmbeddingSpace, HmsCore, SearchOptions};
use serde_json::{json, Value};

fn config(dimensions: usize) -> HmsConfig {
    HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "fixture".into(),
            revision: "1".into(),
            dimensions,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    }
}

fn document(id: usize, dimensions: usize) -> DocumentInput {
    let vector: Vec<_> = (0..dimensions)
        .map(|coordinate| ((id * 7 + coordinate * 13) as f64 * 0.3).sin())
        .collect();
    serde_json::from_value(json!({"id": format!("d{id:04}"), "version": "old", "text": format!("running common topic{} rare{id}", id % 7), "sourceUri": format!("source{}", id % 3), "metadata": {"group": id % 2, "precision": ((id + 1) as f64 * 0.3).sin()}, "storeText": !id.is_multiple_of(13), "embeddings": [vector]})).unwrap()
}

fn queries(core: &HmsCore) -> Result<Vec<Value>> {
    let mut results = Vec::new();
    for filter in [None, Some(json!({"group": 1})), Some(json!({"group": 99}))] {
        for fusion in ["blend", "rrf"] {
            for k in [1, 9, 10, 11] {
                for (lexical, semantic) in [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0)] {
                    let options = SearchOptions {
                        k: Some(k),
                        candidate_limit: Some(16),
                        filter: filter.clone(),
                        fusion: Some(fusion.into()),
                        embedding: Some(vec![1.0, 0.3, -0.2, 0.1, 0.0, -0.1, 0.0, 0.2]),
                        min_semantic_score: Some(-1.0),
                        lexical_weight: Some(lexical),
                        semantic_weight: Some(semantic),
                        ..Default::default()
                    };
                    results.push(serde_json::to_value(
                        core.search_documents("running rare3", &options)?,
                    )?);
                }
            }
        }
    }
    Ok(results)
}

#[test]
fn trained_document_backend_preserves_hybrid_fields_and_filtered_results() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let core = HmsCore::new(
        1024,
        Some(directory.path().display().to_string()),
        Some(config(8)),
    )?;
    core.memorize_documents((0..256).map(|id| document(id, 8)).collect())?;
    let expected = queries(&core)?;
    assert!(!core.document_index_trained());
    core.train_document_index()?;
    assert!(core.document_index_trained());
    assert_eq!(queries(&core)?, expected);
    core.maintain_indices()?;
    assert_eq!(queries(&core)?, expected);
    Ok(())
}

#[test]
fn mutation_and_restart_rebuild_the_document_cache_without_stale_results() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().display().to_string();
    let core = HmsCore::new(1024, Some(path.clone()), Some(config(8)))?;
    core.memorize_documents((0..256).map(|id| document(id, 8)).collect())?;
    core.train_document_index()?;
    let mut replacement = document(3, 8);
    replacement.version = Some("new".into());
    replacement.text = "replacement".into();
    replacement.embeddings = None;
    core.memorize_document(replacement)?;
    assert!(core.delete_document("d0007")?);
    assert!(!core.document_index_trained());
    assert_eq!(core.document_embedding_count(), 254);
    let expected = queries(&core)?;
    core.maintain_indices()?;
    assert!(core.document_index_trained());
    assert_eq!(queries(&core)?, expected);
    core.compact()?;
    assert_eq!(queries(&core)?, expected);
    drop(core);
    let core = HmsCore::new(1024, Some(path), Some(config(8)))?;
    assert!(!core.document_index_trained());
    assert_eq!(queries(&core)?, expected);
    core.train_document_index()?;
    assert_eq!(queries(&core)?, expected);
    Ok(())
}

#[test]
fn document_index_threshold_and_dimension_limits_have_exact_fallbacks() -> Result<()> {
    for count in [127, 128, 129] {
        let directory = tempfile::tempdir()?;
        let core = HmsCore::new(
            256,
            Some(directory.path().display().to_string()),
            Some(config(8)),
        )?;
        core.memorize_documents((0..count).map(|id| document(id, 8)).collect())?;
        let options = SearchOptions {
            embedding: Some(vec![1.0; 8]),
            k: Some(10),
            ..Default::default()
        };
        let expected = serde_json::to_value(core.search_documents("", &options)?)?;
        core.train_document_index()?;
        assert_eq!(core.document_index_trained(), count >= 128);
        assert_eq!(
            serde_json::to_value(core.search_documents("", &options)?)?,
            expected
        );
    }
    for dimensions in [4095, 4096, 4097] {
        let directory = tempfile::tempdir()?;
        let core = HmsCore::new(
            256,
            Some(directory.path().display().to_string()),
            Some(config(dimensions)),
        )?;
        let mut vector = vec![0.0; dimensions];
        vector[0] = 1.0;
        let inputs = (0..128)
            .map(|id| {
                serde_json::from_value(
                    json!({"id": id.to_string(), "text": "needle", "embeddings": [vector]}),
                )
                .unwrap()
            })
            .collect();
        core.memorize_documents(inputs)?;
        let options = SearchOptions {
            embedding: Some(vector),
            k: Some(10),
            ..Default::default()
        };
        let expected = serde_json::to_value(core.search_documents("", &options)?)?;
        core.train_document_index()?;
        assert_eq!(core.document_index_trained(), dimensions <= 4096);
        assert_eq!(
            serde_json::to_value(core.search_documents("", &options)?)?,
            expected
        );
    }
    Ok(())
}
