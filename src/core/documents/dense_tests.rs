// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::core::{documents, encoding::TextEncoder};
use serde_json::{json, Value};

fn document(id: usize, vectors: &[Vec<f64>]) -> StoredDocument {
    let input = serde_json::from_value(json!({
        "id": format!("d{id:04}"), "version": "old", "sourceUri": format!("source{}", id % 3),
        "text": (0..vectors.len()).map(|chunk| format!("running common rare{id} part{chunk}")).collect::<Vec<_>>().join(" "),
        "chunkWords": 4, "overlapWords": 0, "metadata": {"group": id % 2}, "embeddings": vectors,
    })).unwrap();
    documents::prepare(input, 256, Some(vectors[0].len()), TextEncoder::default()).unwrap()
}

fn store(count: usize, dimensions: usize) -> DocumentStore {
    let mut store = DocumentStore::default();
    for id in (0..count).rev() {
        let vector = (0..dimensions)
            .map(|coordinate| ((id * 17 + coordinate * 31) as f64 * 0.17).sin())
            .collect();
        let document = document(id, &[vector]);
        store.insert(document.id.clone(), document);
    }
    store
}

fn train(store: &DocumentStore, dimensions: usize) -> TrainedDenseIndex {
    build_dense_index(capture_dense_index(store, dimensions).unwrap().unwrap())
        .unwrap()
        .unwrap()
}

fn exhaustive(
    index: &DenseIndex,
    query: &[f64],
    limit: usize,
    minimum: f64,
) -> Vec<(String, usize, u64)> {
    let mut hits: Vec<_> = index
        .rows
        .iter()
        .map(|row| {
            let score = query
                .iter()
                .zip(row.embedding())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            (row, score)
        })
        .filter(|(_, score)| *score >= minimum)
        .collect();
    hits.sort_by(|(a, sa), (b, sb)| sb.total_cmp(sa).then(a.order.cmp(&b.order)));
    hits.truncate(limit);
    hits.into_iter()
        .map(|(row, score)| (row.document.id.clone(), row.chunk, score.to_bits()))
        .collect()
}

fn actual(hits: Vec<DenseHit>) -> Vec<(String, usize, u64)> {
    hits.into_iter()
        .map(|hit| (hit.document.id.clone(), hit.chunk, hit.score.to_bits()))
        .collect()
}

#[test]
fn exact_candidates_match_exhaustive_scores_at_rank_and_threshold_boundaries() {
    let store = store(513, 9);
    let trained = train(&store, 9);
    let index = &trained.index;
    for raw in [
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        vec![
            1.0,
            f64::MIN_POSITIVE,
            -1.0,
            f64::from_bits(1),
            0.0,
            -0.0,
            0.1,
            -0.1,
            0.5,
        ],
    ] {
        let query = documents::normalize_embedding(&raw, 9).unwrap();
        let boundary = index.rows[173]
            .embedding()
            .iter()
            .zip(&query)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .clamp(-1.0, 1.0);
        for minimum in [
            -1.0,
            0.0,
            0.5,
            1.0,
            boundary.next_down().max(-1.0),
            boundary,
            boundary.next_up().min(1.0),
        ] {
            for limit in [1, 9, 10, 11, 63, 64, 65, 100, 512, 513, 514] {
                assert_eq!(
                    actual(index.search(&query, limit, minimum).0),
                    exhaustive(index, &query, limit, minimum),
                    "limit={limit} minimum={minimum}"
                );
            }
        }
        for (node_id, node) in index.nodes.iter().enumerate() {
            let upper = index.upper(node_id, &query);
            for row in &index.rows[node.start..node.end] {
                let score = query
                    .iter()
                    .zip(row.embedding())
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    .clamp(-1.0, 1.0);
                assert!(
                    upper >= score,
                    "node={node_id} upper={upper:?} score={score:?}"
                );
            }
        }
    }
}

#[test]
fn outward_bounds_cover_cancellation_subnormals_signed_zero_and_large_legacy_values() {
    let mut store = store(257, 8);
    let tiny = f64::from_bits(1);
    for id in 0..257 {
        let mut document = store.remove(&format!("d{id:04}")).unwrap().as_ref().clone();
        document.chunks[0].embedding = Some(match id % 7 {
            0 => vec![1.0, -1.0, tiny, -tiny, 0.0, -0.0, 0.0, 0.0],
            1 => vec![tiny, tiny, -tiny, -tiny, 0.0, 0.0, -0.0, 0.0],
            2 => vec![f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, 0.0, 0.0, 0.0, 0.0],
            3 => vec![-f64::MAX, -f64::MAX, f64::MAX, f64::MAX, 0.0, 0.0, 0.0, 0.0],
            4 => vec![
                1.0_f64.next_up(),
                -1.0,
                1.0_f64.next_down(),
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            5 => vec![0.0; 8],
            _ => vec![1.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0],
        });
        store.insert(document.id.clone(), document);
    }
    let trained = train(&store, 8);
    let index = &trained.index;
    for raw in [
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0; 8],
        vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, -1.0, tiny, tiny, 0.0, 0.0, 0.0, 0.0],
        vec![tiny, 1.0, tiny, -1.0, -0.0, 0.0, 0.0, 0.0],
    ] {
        let query = documents::normalize_embedding(&raw, 8).unwrap();
        for minimum in [-1.0, -tiny, -0.0, 0.0, tiny, 1.0_f64.next_down(), 1.0] {
            for limit in [1, 9, 64, 65, 256, 257, 258] {
                assert_eq!(
                    actual(index.search(&query, limit, minimum).0),
                    exhaustive(index, &query, limit, minimum)
                );
            }
        }
        for (node_id, node) in index.nodes.iter().enumerate() {
            let upper = index.upper(node_id, &query);
            assert!(upper.is_finite());
            for row in &index.rows[node.start..node.end] {
                let score = query
                    .iter()
                    .zip(row.embedding())
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    .clamp(-1.0, 1.0);
                assert!(
                    upper >= score,
                    "node={node_id} upper={upper:?} score={score:?}"
                );
            }
        }
    }
}

#[test]
fn separated_fixture_prunes_exact_dots_and_preserves_equal_boundary_ties() {
    let mut store = DocumentStore::default();
    for id in (0..512).rev() {
        let vector = if id < 64 {
            vec![1.0, 0.0, 0.0, 0.0]
        } else if id < 128 {
            vec![0.8, 0.6, 0.0, 0.0]
        } else {
            vec![-1.0, (id % 5) as f64 / 100.0, 0.0, 0.0]
        };
        let document = document(id, &[vector]);
        store.insert(document.id.clone(), document);
    }
    let trained = train(&store, 4);
    for row in &trained.index.rows {
        assert!(Arc::ptr_eq(
            &row.document,
            &store.documents[&row.document.id]
        ));
        assert_eq!(
            row.embedding().as_ptr(),
            store.documents[&row.document.id].chunks[row.chunk]
                .embedding
                .as_ref()
                .unwrap()
                .as_ptr()
        );
    }
    let query = [1.0, 0.0, 0.0, 0.0];
    let (hits, dots) = trained.index.search(&query, 10, 1.0);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.document.id.clone())
            .collect::<Vec<_>>(),
        (0..10).map(|id| format!("d{id:04}")).collect::<Vec<_>>()
    );
    assert_eq!(actual(hits), exhaustive(&trained.index, &query, 10, 1.0));
    assert!(dots < 512, "exact dots={dots}");
    assert_eq!(dots, 64);
    println!("separated_fixture exact_dot_evaluations={dots} exhaustive_dot_evaluations=512");
    let mut equal = DocumentStore::default();
    for id in (0..129).rev() {
        let document = document(id, &[vec![1.0, 0.0, 0.0, 0.0]]);
        equal.insert(document.id.clone(), document);
    }
    let trained = train(&equal, 4);
    let (hits, dots) = trained.index.search(&query, 11, 1.0);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.document.id.clone())
            .collect::<Vec<_>>(),
        (0..11).map(|id| format!("d{id:04}")).collect::<Vec<_>>()
    );
    assert_eq!(dots, 129);
    assert_eq!(actual(hits), exhaustive(&trained.index, &query, 11, 1.0));
}

fn result(
    store: &DocumentStore,
    text: &str,
    options: &documents::SearchOptions,
    dimensions: usize,
) -> Value {
    serde_json::to_value(documents::search(store, text, options, Some(dimensions)).unwrap())
        .unwrap()
}

#[test]
fn hybrid_fields_and_filtered_statistics_match_independent_scan_with_legacy_terms() {
    let mut store = store(256, 4);
    let mut replacement = document(7, &[vec![-1.0, 0.0, 0.0, 0.0]]);
    replacement.chunks[0].terms.insert("ghost".into(), 0);
    store.insert(replacement.id.clone(), replacement);
    let mut absent = document(3, &[vec![1.0, 0.0, 0.0, 0.0]]);
    absent.chunks[0].embedding = None;
    absent.chunks[0].stemmed = false;
    absent.chunks[0].terms = documents::terms("running common rare3 ghost", false).into();
    absent.chunks[0].terms.insert("ghost".into(), 0);
    store.insert(absent.id.clone(), absent);
    let trained = train(&store, 4);
    assert!(publish_dense_index(&mut store, &trained));
    for fusion in ["blend", "rrf"] {
        for (lexical_weight, semantic_weight) in [(1.0, 1.0), (1.0, 0.25), (0.0, 1.0), (1.0, 0.0)] {
            for filter in [
                None,
                Some(json!({})),
                Some(json!({"group": 1})),
                Some(json!({"group": 99})),
            ] {
                for minimum in [-1.0, 0.0, 0.5, 1.0] {
                    let options = documents::SearchOptions {
                        k: Some(11),
                        candidate_limit: Some(11),
                        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
                        lexical_weight: Some(lexical_weight),
                        semantic_weight: Some(semantic_weight),
                        min_semantic_score: Some(minimum),
                        fusion: Some(fusion.into()),
                        filter: filter.clone(),
                        ..Default::default()
                    };
                    let expected = serde_json::to_value(
                        documents::search_impl(
                            &store,
                            "running rare7 rare3 ghost",
                            &options,
                            Some(4),
                            false,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                    assert_eq!(
                        result(&store, "running rare7 rare3 ghost", &options, 4),
                        expected
                    );
                }
            }
        }
    }
    let options = documents::SearchOptions {
        k: Some(1),
        candidate_limit: Some(1),
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        lexical_weight: Some(2.0),
        semantic_weight: Some(0.25),
        min_semantic_score: Some(1.0),
        ..Default::default()
    };
    let hits = documents::search(&store, "rare7", &options, Some(4)).unwrap();
    assert_eq!(hits[0].document_id, "d0007");
    assert_eq!(hits[0].semantic_score, Some(-1.0));
    for options in [
        documents::SearchOptions {
            source_uri: Some("source1".into()),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            ..Default::default()
        },
        documents::SearchOptions {
            document_ids: Some(vec!["d0007".into(), "d0003".into(), "d0007".into()]),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            ..Default::default()
        },
    ] {
        let expected = serde_json::to_value(
            documents::search_impl(&store, "running rare7 rare3", &options, Some(4), false)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result(&store, "running rare7 rare3", &options, 4), expected);
    }
}

#[test]
fn captured_index_survives_replacement_delete_and_slot_reuse_while_stale_build_is_rejected() {
    let mut store = store(256, 4);
    let trained = train(&store, 4);
    assert!(publish_dense_index(&mut store, &trained));
    let options = documents::SearchOptions {
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        min_semantic_score: Some(-1.0),
        ..Default::default()
    };
    let prepared = documents::prepare_search("running rare3", &options, Some(4)).unwrap();
    let captured = documents::capture_search(&store, vec![prepared]);
    let expected_old = result(&store, "running rare3", &options, 4);
    let old = Arc::clone(&store.documents["d0003"]);
    let generation = store.generation;
    let mut replacement = document(3, &[vec![-1.0, 0.0, 0.0, 0.0]]);
    replacement.version = "new".into();
    store.insert(replacement.id.clone(), replacement);
    assert!(store.generation > generation);
    assert!(!store.dense_index_trained());
    assert!(store.remove("d0007").is_some());
    let fresh = document(999, &[vec![0.0, 1.0, 0.0, 0.0]]);
    store.insert(fresh.id.clone(), fresh);
    assert_eq!(
        serde_json::to_value(documents::execute_search(&captured, 0)).unwrap(),
        expected_old
    );
    assert!(!Arc::ptr_eq(&old, &store.documents["d0003"]));
    assert!(!publish_dense_index(&mut store, &trained));
    let stale = train(&store, 4);
    let document = document(1000, &[vec![0.0, -1.0, 0.0, 0.0]]);
    store.insert(document.id.clone(), document);
    assert!(!publish_dense_index(&mut store, &stale));
    let expected = serde_json::to_value(
        documents::search_impl(&store, "running rare3", &options, Some(4), false).unwrap(),
    )
    .unwrap();
    let rebuilt = train(&store, 4);
    assert!(publish_dense_index(&mut store, &rebuilt));
    assert_eq!(result(&store, "running rare3", &options, 4), expected);
    drop(store);
    assert_eq!(
        serde_json::to_value(documents::execute_search(&captured, 0)).unwrap(),
        expected_old
    );
}

#[test]
fn registry_rebuild_preserves_epoch_lineage_and_embedding_owners() {
    for count in [1023, 1024, 1025] {
        let mut store = DocumentStore::default();
        for id in 0..count {
            let document = document(id, &[vec![1.0, id as f64 / 1000.0]]);
            store.insert(document.id.clone(), document);
        }
        let original = train(&store, 2);
        assert!(publish_dense_index(&mut store, &original));
        let survivor = Arc::clone(&store.documents[&format!("d{:04}", count - 1)]);
        let mut epoch = store.generation;
        for id in 0..600 {
            assert!(store.remove(&format!("d{id:04}")).is_some());
            assert!(store.generation > epoch);
            epoch = store.generation;
        }
        assert_eq!(store.embedded_count(), count - 600);
        assert_eq!(store.embedded_documents.len(), count - 600);
        assert!(Arc::ptr_eq(&survivor, &store.documents[&survivor.id]));
        assert_eq!(store.slot_count < count, count == 1025);
        assert!(!publish_dense_index(&mut store, &original));
        let rebuilt = train(&store, 2);
        assert!(publish_dense_index(&mut store, &rebuilt));
        assert!(rebuilt
            .index
            .rows
            .iter()
            .all(|row| Arc::ptr_eq(&row.document, &store.documents[&row.document.id])));
    }
}

#[test]
fn invalid_shapes_and_bounded_auxiliary_allocations_fail_before_building() {
    for count in [127, 128, 129] {
        let store = store(count, 2);
        assert_eq!(
            capture_dense_index(&store, 2).unwrap().is_some(),
            count >= 128
        );
    }
    let store = store(128, 2);
    assert!(capture_dense_index(&store, 0).unwrap().is_none());
    assert!(capture_dense_index(&store, 4097).unwrap().is_none());
    for dimensions in [1, 3] {
        assert!(
            build_dense_index(capture_dense_index(&store, dimensions).unwrap().unwrap()).is_err()
        );
    }
    let mut invalid = store.documents["d0000"].as_ref().clone();
    invalid.chunks[0].embedding.as_mut().unwrap()[0] = f64::NAN;
    let mut malformed = store;
    malformed.insert(invalid.id.clone(), invalid);
    assert!(build_dense_index(capture_dense_index(&malformed, 2).unwrap().unwrap()).is_err());
    for (count, dimensions) in [(MAX_ROWS + 1, 1), (MAX_ROWS, 4096), (100_000, 2048)] {
        assert!(build_dense_index(DenseIndexSnapshot {
            generation: 0,
            dimensions,
            embedded_count: count,
            documents: Vec::new()
        })
        .is_err());
    }
}

#[test]
fn same_generation_publication_is_idempotent_and_mutation_rejects_both_old_builds() {
    let mut store = store(128, 2);
    let first = train(&store, 2);
    let second = train(&store, 2);
    assert!(!Arc::ptr_eq(&first.index, &second.index));
    assert!(publish_dense_index(&mut store, &first));
    assert!(publish_dense_index(&mut store, &second));
    assert!(Arc::ptr_eq(
        store.dense_index.as_ref().unwrap(),
        &first.index
    ));
    let document = document(3, &[vec![-1.0, 0.0]]);
    store.insert(document.id.clone(), document);
    assert!(!publish_dense_index(&mut store, &first));
    assert!(!publish_dense_index(&mut store, &second));
    assert!(!store.dense_index_trained());
}
