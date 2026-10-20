// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use holographic_memory::{core::HmsConfig, DocumentInput, EmbeddingSpace, HmsCore, SearchOptions};
use serde_json::json;

#[test]
fn lexical_top_candidates_preserve_ties_and_scores_through_slot_reuse_and_replay(
) -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().display().to_string();
    let config = HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "fixture".into(),
            revision: "1".into(),
            dimensions: 2,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    let core = HmsCore::new(1024, Some(path.clone()), Some(config.clone()))?;
    for i in (0..12).rev() {
        let document: DocumentInput = serde_json::from_value(json!({
            "id": format!("d{i:02}"), "text": "equal running candidate",
            "sourceUri": "source", "metadata": {"group": i % 2, "nested": {"x": 1}},
            "embeddings": [[1.0, 1.0]],
        }))?;
        core.memorize_document(document)?;
    }
    core.delete_document("d00")?;
    let replacement: DocumentInput = serde_json::from_value(json!({
        "id": "d00", "version": "2", "text": "equal running candidate",
        "sourceUri": "source", "metadata": {"group": 0, "nested": {"x": 1}},
        "embeddings": [[1.0, 1.0]],
    }))?;
    core.memorize_document(replacement)?;
    let mut snapshots = Vec::new();
    for limit in [9, 10, 11] {
        for fusion in ["blend", "rrf"] {
            for filter in [None, Some(json!({})), Some(json!({"nested": {"x": 1}}))] {
                for weight in [1.0, f64::from_bits(1)] {
                    let options = SearchOptions {
                        k: Some(9),
                        candidate_limit: Some(limit),
                        embedding: Some(vec![1.0, 0.0]),
                        lexical_weight: Some(weight),
                        semantic_weight: Some(0.0),
                        fusion: Some(fusion.into()),
                        filter: filter.clone(),
                        ..Default::default()
                    };
                    let results = core.search_documents("running", &options)?;
                    assert_eq!(results.len(), 9);
                    for (i, result) in results.iter().enumerate() {
                        assert_eq!(result.document_id, format!("d{i:02}"));
                        assert_eq!(result.version, if i == 0 { "2" } else { "1" });
                        assert!(result.semantic_score.is_some_and(|value| value > 0.7));
                    }
                    snapshots.push((options, serde_json::to_value(results)?));
                }
            }
        }
    }
    core.compact()?;
    drop(core);
    let core = HmsCore::new(1024, Some(path), Some(config))?;
    for (options, expected) in snapshots {
        assert_eq!(
            serde_json::to_value(core.search_documents("running", &options)?)?,
            expected
        );
    }
    assert!(core
        .search_documents("unknown", &SearchOptions::default())?
        .is_empty());
    Ok(())
}
