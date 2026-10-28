// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::engine::HmsCore;
use crate::core::entangled::EntangledHVec;

#[test]
fn test_determinism() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(1000, Some(dir.path().display().to_string()), None).unwrap();
    let v1 = hms.encode_text("hello world");
    let v2 = hms.encode_text("hello world");
    assert!((v1.similarity(&v2) - 1.0).abs() < 0.0001);
}

#[test]
fn test_store_rejects_concurrent_engines_and_reopens_after_drop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display().to_string();
    let first = HmsCore::new(1000, Some(path.clone()), None).unwrap();
    let error = HmsCore::new(1000, Some(path.clone()), None)
        .err()
        .expect("a second writable engine must be rejected");
    assert!(error.to_string().contains("already open"));
    drop(first);
    HmsCore::new(1000, Some(path), None).expect("store lock must release after teardown");
}

#[test]
fn test_from_dense_produces_sparse() {
    let dense: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
    let e = EntangledHVec::from_dense(&dense, 1000);
    assert_eq!(e.dim, 1000);
    // Should have ~dim/256 active indices
    let expected = 1000 / 256;
    assert_eq!(e.indices.len(), expected);
}

// === Batch Memorization ===

#[test]
fn test_batch_memorize() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    // Simulate what memorize_batch does: parallel encode, sequential insert
    let items = vec![
        ("b1", "first batch item"),
        ("b2", "second batch item"),
        ("b3", "third batch item"),
    ];
    for (id, text) in &items {
        let vec = hms.encode_text(text);
        hms.memorize(id.to_string(), vec).unwrap();
    }
    assert_eq!(hms.vector_count(), 3);

    let q = hms.encode_text("first batch item");
    let results = hms.query(&q, 3);
    assert!(!results.is_empty());
}

// === Custom Config ===

#[test]
fn test_custom_concept_config() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.concepts.similarity_threshold = 0.5;
    config.concepts.min_cluster_size = 5;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    // With high threshold and min size, clusters are harder to form
    for i in 0..10 {
        let vec = hms.encode_text(&format!("config test document {}", i));
        hms.memorize(format!("cfg_{}", i), vec).unwrap();
    }

    let concepts = hms.synthesize_concepts();
    // With strict thresholds, fewer or no concepts should form
    // (this validates the config is actually used)
    for c in &concepts {
        assert!(
            c.member_count >= 5,
            "Min cluster size should be 5, got {}",
            c.member_count
        );
    }
}

// === Diffusion factorizer ===

#[test]
fn test_factorize_returns_factors() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    // Memorize some domain items
    let colors = vec!["red", "blue", "green"];
    let shapes = vec!["circle", "square", "triangle"];
    for c in &colors {
        let v = hms.encode_text(c);
        hms.memorize(c.to_string(), v).unwrap();
    }
    for s in &shapes {
        let v = hms.encode_text(s);
        hms.memorize(s.to_string(), v).unwrap();
    }

    // Create a composite: red * circle
    let red_vec = hms.encode_text("red");
    let circle_vec = hms.encode_text("circle");
    let product = red_vec.bind(&circle_vec);

    let domains = vec![
        colors
            .iter()
            .map(|s| hms.encode_text(s))
            .collect::<Vec<EntangledHVec>>(),
        shapes
            .iter()
            .map(|s| hms.encode_text(s))
            .collect::<Vec<EntangledHVec>>(),
    ];

    let results = hms.factorize_diffusion(&product, &domains, 20);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_concept_synthesis() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(1000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    // Create 20 slightly-varied versions of a "base" concept
    // Use the same seed with small variations to create tight clusters
    let base = hms.encode_text("base concept");
    for i in 0..20 {
        // Create variants by permuting base slightly
        let variant = if i == 0 {
            base.clone()
        } else {
            // Bind with a "small" perturbation (most indices shared)
            let perturb = EntangledHVec::new_deterministic(1000, 10000 + i);
            // Bundle many copies of base with one perturbation to stay close
            EntangledHVec::bundle(&[
                base.clone(),
                base.clone(),
                base.clone(),
                base.clone(),
                perturb,
            ])
        };
        hms.memorize(format!("var_{}", i), variant).unwrap();
    }

    let concepts = hms.synthesize_concepts();
    // We expect at least one synthesized concept representing the cluster
    assert!(
        !concepts.is_empty(),
        "Should synthesize at least one concept"
    );
    assert!(
        concepts[0].coherence > 0.7,
        "Synthesized concept should have high coherence, got {}",
        concepts[0].coherence
    );
}

#[test]
fn test_concept_refinement_stability() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.concepts.similarity_threshold = 0.3;
    config.concepts.min_cluster_size = 3;
    let hms = HmsCore::new(
        1000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    // Create two distinct clusters so refinement has work to do
    let base_a = hms.encode_text("alpha cluster topic");
    let base_b = hms.encode_text("beta different subject");
    for i in 0..10 {
        let perturb = EntangledHVec::new_deterministic(1000, 20000 + i);
        let va = EntangledHVec::bundle(&[
            base_a.clone(),
            base_a.clone(),
            base_a.clone(),
            base_a.clone(),
            perturb.clone(),
        ]);
        hms.memorize(format!("a_{}", i), va).unwrap();

        let perturb_b = EntangledHVec::new_deterministic(1000, 30000 + i);
        let vb = EntangledHVec::bundle(&[
            base_b.clone(),
            base_b.clone(),
            base_b.clone(),
            base_b.clone(),
            perturb_b,
        ]);
        hms.memorize(format!("b_{}", i), vb).unwrap();
    }

    let concepts = hms.synthesize_concepts();
    assert!(
        !concepts.is_empty(),
        "Should synthesize at least one concept"
    );
    // With tight clusters, refinement should converge (stable = true)
    for c in &concepts {
        assert!(
            c.stable,
            "Concept with {} members should be stable after refinement",
            c.member_count
        );
        assert!(
            c.coherence > 0.5,
            "Coherence should be decent after refinement, got {}",
            c.coherence
        );
    }
}

// === NSG Integration Tests ===

#[test]
fn test_nsg_train_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.nsg.max_degree = 8;

    config.nsg.ef_construction = 16;

    let hms = HmsCore::new(
        1000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..50 {
        let vec = hms.encode_text(&format!("nsg document {}", i));
        hms.memorize(format!("nsg_{}", i), vec).unwrap();
    }

    assert!(!hms.nsg_trained());
    hms.train_nsg().unwrap();
    assert!(hms.nsg_trained());

    let q = hms.encode_text("nsg document 0");
    let results = hms.query(&q, 5);
    assert!(!results.is_empty(), "NSG query should return results");
}

#[test]
fn test_nsg_maintenance_at_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.nsg.auto_threshold = 30;
    config.nsg.max_degree = 8;

    config.nsg.ef_construction = 16;

    let hms = HmsCore::new(
        1000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..29 {
        let vec = hms.encode_text(&format!("auto nsg {}", i));
        hms.memorize(format!("ansg_{}", i), vec).unwrap();
    }
    assert!(!hms.nsg_trained());

    let vec = hms.encode_text("auto nsg 29");
    hms.memorize("ansg_29".to_string(), vec).unwrap();
    assert!(!hms.nsg_trained(), "ingestion must not synchronously train");
    assert!(hms.index_status().nsg_training_recommended);
    hms.maintain_indices().unwrap();
    assert!(hms.nsg_trained());
}

// === Phase 4 Integration Tests ===

#[test]
fn test_adaptive_routing_e2e() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.nsg.max_degree = 8;

    config.nsg.ef_construction = 16;

    let hms = HmsCore::new(
        1000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..50 {
        let vec = hms.encode_text(&format!("routing item {}", i));
        hms.memorize(format!("rt_{}", i), vec).unwrap();
    }

    // Train NSG — queries should now route through NSG
    hms.train_nsg().unwrap();
    assert!(hms.nsg_trained());

    let q = hms.encode_text("routing item 0");
    let results = hms.query(&q, 5);
    assert!(
        !results.is_empty(),
        "Adaptive routing should return results via NSG"
    );
}

// === Knowledge Graph Tests ===

#[test]
fn test_triplet_memorize_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    hms.memorize_triplet(
        "paris_capital".to_string(),
        "Paris".to_string(),
        "is_capital_of".to_string(),
        "France".to_string(),
    )
    .unwrap();
    hms.memorize_triplet(
        "berlin_capital".to_string(),
        "Berlin".to_string(),
        "is_capital_of".to_string(),
        "Germany".to_string(),
    )
    .unwrap();
    hms.memorize_triplet(
        "tokyo_capital".to_string(),
        "Tokyo".to_string(),
        "is_capital_of".to_string(),
        "Japan".to_string(),
    )
    .unwrap();

    let results = hms
        .query_triplet("Paris".to_string(), "is_capital_of".to_string(), 3)
        .unwrap();
    assert!(!results.is_empty(), "Triplet query should return results");
    assert!(
        results.iter().any(|r| r.id == "paris_capital"),
        "Paris triplet should appear in top-3 results"
    );
}

// === Sequence Tests ===

#[test]
fn test_sequence_memorize_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    hms.memorize_sequence(
        "recipe_1".to_string(),
        &[
            "preheat oven".to_string(),
            "mix ingredients".to_string(),
            "pour into pan".to_string(),
            "bake for thirty minutes".to_string(),
        ],
    )
    .unwrap();
    hms.memorize_sequence(
        "recipe_2".to_string(),
        &[
            "boil water".to_string(),
            "add pasta".to_string(),
            "drain and serve".to_string(),
        ],
    )
    .unwrap();

    // Query with a partial sequence match
    let q = hms.encode_text("preheat oven").permute(0);
    let results = hms.query(&q, 2);
    assert!(!results.is_empty(), "Sequence query should return results");
}

// === Scalar Query Tests ===

#[test]
fn test_scalar_query_ordering() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    for i in 0..20 {
        let val = i as f64 * 5.0;
        hms.memorize_scalar(format!("temp_{}", i), val, 0.0, 100.0)
            .unwrap();
    }

    // Query near value 50 — should return items closest to 50
    let q = EntangledHVec::from_scalar(50.0, 0.0, 100.0, 10_000);
    let results = hms.query(&q, 5);
    assert!(!results.is_empty(), "Scalar query should return results");

    // Top results should cluster around value 50 (idx 10)
    let top_idx: usize = results[0]
        .id
        .strip_prefix("temp_")
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (5..=15).contains(&top_idx),
        "Top scalar result should be near value 50 (idx 10), got idx {}",
        top_idx
    );
}

// === Component Analysis Tests ===

#[test]
fn test_analyze_components() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    for i in 0..30 {
        let vec = hms.encode_text(&format!("component analysis document {}", i));
        hms.memorize(format!("ca_{}", i), vec).unwrap();
    }

    let vec = hms.encode_text("component analysis document 0");
    let results = hms.analyze_components(&vec);
    assert!(
        !results.is_empty(),
        "analyze_components should return results"
    );
    assert!(
        results.iter().all(|r| r.similarity > 0.05),
        "All results should exceed similarity threshold"
    );
}

// === Delete Tests ===

#[test]
fn test_delete_existing() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    let vec = hms.encode_text("hello");
    hms.memorize("hello".to_string(), vec).unwrap();
    assert_eq!(hms.vector_count(), 1);

    assert!(hms.delete("hello").unwrap());
    assert_eq!(hms.vector_count(), 0);

    let q = hms.encode_text("hello");
    let results = hms.query(&q, 5);
    assert!(
        results.is_empty(),
        "Deleted vector should not appear in results"
    );
}

#[test]
fn test_delete_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().display().to_string()), None).unwrap();
    assert!(!hms.delete("no_such_id").unwrap());
}

#[test]
fn test_basic_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    {
        let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
        let v = hms.encode_text("persist me");
        hms.memorize("p1".to_string(), v).unwrap();
        assert_eq!(hms.vector_count(), 1);
    }

    let hms = HmsCore::new(10_000, Some(path), None).unwrap();
    assert_eq!(hms.vector_count(), 1);
}

#[test]
fn test_delete_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    {
        let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
        let v1 = hms.encode_text("keep me");
        let v2 = hms.encode_text("delete me");
        hms.memorize("keep".to_string(), v1).unwrap();
        hms.memorize("del".to_string(), v2).unwrap();
        assert_eq!(hms.vector_count(), 2);
        hms.delete("del").unwrap();
        assert_eq!(hms.vector_count(), 1);
    }

    let hms = HmsCore::new(10_000, Some(path), None).unwrap();
    assert_eq!(hms.vector_count(), 1);
    let q = hms.encode_text("keep me");
    let results = hms.query(&q, 5);
    assert!(!results.is_empty(), "Kept vector should survive restart");
}

#[test]
fn test_delete_and_rememorize() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    let v1 = hms.encode_text("version 1");
    hms.memorize("doc".to_string(), v1).unwrap();
    hms.delete("doc").unwrap();

    let v2 = hms.encode_text("version 2");
    hms.memorize("doc".to_string(), v2.clone()).unwrap();
    assert_eq!(hms.vector_count(), 1);

    let q = hms.encode_text("version 2");
    let results = hms.query(&q, 1);
    assert_eq!(results[0].id, "doc");
}

// === Compact Tests ===

#[test]
fn test_compact_basic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
    for i in 0..50 {
        let vec = hms.encode_text(&format!("item {}", i));
        hms.memorize(format!("id_{}", i), vec).unwrap();
    }
    // Delete half
    for i in 0..25 {
        hms.delete(&format!("id_{}", i)).unwrap();
    }
    assert_eq!(hms.vector_count(), 25);

    hms.compact().unwrap();

    // Verify all live vectors still queryable
    let q = hms.encode_text("item 30");
    let results = hms.query(&q, 5);
    assert!(!results.is_empty(), "Should find results after compaction");
    assert_eq!(hms.vector_count(), 25);
}

#[test]
fn test_compact_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    {
        let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
        for i in 0..20 {
            let vec = hms.encode_text(&format!("doc {}", i));
            hms.memorize(format!("d_{}", i), vec).unwrap();
        }
        for i in 0..10 {
            hms.delete(&format!("d_{}", i)).unwrap();
        }
        hms.compact().unwrap();
    }

    // Re-open from compacted arena
    let hms = HmsCore::new(10_000, Some(path), None).unwrap();
    assert_eq!(hms.vector_count(), 10);
}

// === Shard Tests ===

#[test]
fn test_multi_shard_insert_query() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.shard.enabled = true;
    config.shard.shard_count = 4;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..100 {
        let vec = hms.encode_text(&format!("shard document {}", i));
        hms.memorize(format!("sd_{}", i), vec).unwrap();
    }

    assert_eq!(hms.vector_count(), 100);

    let q = hms.encode_text("shard document 0");
    let results = hms.query(&q, 5);
    assert!(
        !results.is_empty(),
        "Multi-shard query should return results"
    );
}

#[test]
fn test_multi_shard_delete() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.shard.enabled = true;
    config.shard.shard_count = 4;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..20 {
        let vec = hms.encode_text(&format!("item {}", i));
        hms.memorize(format!("m_{}", i), vec).unwrap();
    }
    assert_eq!(hms.vector_count(), 20);

    for i in 0..10 {
        assert!(hms.delete(&format!("m_{}", i)).unwrap());
    }
    assert_eq!(hms.vector_count(), 10);
}

#[test]
fn test_auto_shard_trigger() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.shard.enabled = true;
    config.shard.shard_count = 0; // auto
    config.shard.auto_threshold = 50;
    config.shard.target_shard_size = 25;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..50 {
        let vec = hms.encode_text(&format!("auto shard {}", i));
        hms.memorize(format!("as_{}", i), vec).unwrap();
    }

    assert_eq!(hms.vector_count(), 50);

    hms.maintain_indices().unwrap();
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("shard_meta.json")).unwrap())
            .unwrap();
    assert_eq!(metadata["shard_count"], 2);

    // Verify queries still work after auto-sharding
    let q = hms.encode_text("auto shard 0");
    let results = hms.query(&q, 5);
    assert!(!results.is_empty(), "Should find results after auto-shard");
}

// === Audit Integration Tests ===

#[test]
fn test_audit_records_operations() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.security.audit_enabled = true;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    let v = hms.encode_text("audit test");
    hms.memorize("aud_1".to_string(), v).unwrap();
    hms.delete("aud_1").unwrap();

    let entries = hms.audit_since(0).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].op, crate::core::audit::AuditOp::Memorize);
    assert_eq!(entries[1].op, crate::core::audit::AuditOp::Delete);
}

#[test]
fn test_audit_disabled_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().display().to_string()), None).unwrap();
    let entries = hms.audit_since(0).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn test_audit_compact_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.security.audit_enabled = true;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    let v = hms.encode_text("compact audit");
    hms.memorize("ca_1".to_string(), v).unwrap();
    hms.compact().unwrap();

    let entries = hms.audit_since(0).unwrap();
    assert!(entries
        .iter()
        .any(|e| e.op == crate::core::audit::AuditOp::Compact));
}

// === Graph Integration Tests ===

#[test]
fn test_graph_add_and_traverse() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    // Memorize some nodes
    for city in &["paris", "france", "europe"] {
        let v = hms.encode_text(city);
        hms.memorize(city.to_string(), v).unwrap();
    }

    hms.add_relation(&crate::core::types::Relation {
        source_id: "paris".into(),
        relation_type: "is_in".into(),
        target_id: "france".into(),
        properties: None,
        valid_from: 0.0,
        valid_to: 0.0,
    })
    .unwrap();
    hms.add_relation(&crate::core::types::Relation {
        source_id: "france".into(),
        relation_type: "is_in".into(),
        target_id: "europe".into(),
        properties: None,
        valid_from: 0.0,
        valid_to: 0.0,
    })
    .unwrap();

    assert_eq!(hms.relation_count(), 2);

    let paths = hms.traverse("paris", Some("is_in"), 3, 0.0);
    assert!(!paths.is_empty());
    let targets: Vec<&str> = paths
        .iter()
        .flat_map(|p| p.hops.iter().map(|h| h.node_id.as_str()))
        .collect();
    assert!(targets.contains(&"france"));
    assert!(targets.contains(&"europe"));
}

#[test]
fn test_graph_transitive_inference() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    for name in &["a", "b", "c"] {
        let v = hms.encode_text(name);
        hms.memorize(name.to_string(), v).unwrap();
    }

    hms.declare_relation_type(crate::core::types::RelationType {
        name: "contains".into(),
        transitive: true,
        symmetric: false,
    })
    .unwrap();

    hms.add_relation(&crate::core::types::Relation {
        source_id: "a".into(),
        relation_type: "contains".into(),
        target_id: "b".into(),
        properties: None,
        valid_from: 0.0,
        valid_to: 0.0,
    })
    .unwrap();
    hms.add_relation(&crate::core::types::Relation {
        source_id: "b".into(),
        relation_type: "contains".into(),
        target_id: "c".into(),
        properties: None,
        valid_from: 0.0,
        valid_to: 0.0,
    })
    .unwrap();

    let paths = hms.traverse("a", Some("contains"), 3, 0.0);
    // Should have inferred single-hop a->c
    let inferred = paths
        .iter()
        .find(|p| p.hops.len() == 1 && p.hops[0].node_id == "c");
    assert!(inferred.is_some(), "Should infer transitive a->c");
}

#[test]
fn test_graph_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    {
        let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
        let v = hms.encode_text("node_a");
        hms.memorize("node_a".to_string(), v).unwrap();
        hms.add_relation(&crate::core::types::Relation {
            source_id: "node_a".into(),
            relation_type: "links_to".into(),
            target_id: "node_b".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        })
        .unwrap();
    }

    let hms = HmsCore::new(10_000, Some(path), None).unwrap();
    assert_eq!(hms.relation_count(), 1);
    let out = hms.outgoing_relations("node_a", None, 0.0);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].target_id, "node_b");
}

#[test]
fn test_graph_temporal_query() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    hms.add_relation(&crate::core::types::Relation {
        source_id: "alice".into(),
        relation_type: "works_at".into(),
        target_id: "acme".into(),
        properties: None,
        valid_from: 1000.0,
        valid_to: 2000.0,
    })
    .unwrap();
    hms.add_relation(&crate::core::types::Relation {
        source_id: "alice".into(),
        relation_type: "works_at".into(),
        target_id: "globex".into(),
        properties: None,
        valid_from: 2001.0,
        valid_to: 0.0,
    })
    .unwrap();

    let at_1500 = hms.outgoing_relations("alice", Some("works_at"), 1500.0);
    assert_eq!(at_1500.len(), 1);
    assert_eq!(at_1500[0].target_id, "acme");

    let at_3000 = hms.outgoing_relations("alice", Some("works_at"), 3000.0);
    assert_eq!(at_3000.len(), 1);
    assert_eq!(at_3000[0].target_id, "globex");
}

#[test]
fn test_federated_query() {
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let path1 = dir1.path().to_string_lossy().to_string();
    let path2 = dir2.path().to_string_lossy().to_string();

    // Populate two separate instances
    {
        let hms1 = HmsCore::new(10_000, Some(path1.clone()), None).unwrap();
        let v = hms1.encode_text("federated doc alpha");
        hms1.memorize("alpha".to_string(), v).unwrap();
    }
    {
        let hms2 = HmsCore::new(10_000, Some(path2.clone()), None).unwrap();
        let v = hms2.encode_text("federated doc beta");
        hms2.memorize("beta".to_string(), v).unwrap();
    }

    // Query from instance 1, federating with instance 2
    let hms1 = HmsCore::new(10_000, Some(path1), None).unwrap();
    let q = hms1.encode_text("federated doc");
    let results = hms1.federated_query(&[path2], &q, 5).unwrap();
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"alpha"), "Should find local result");
    assert!(ids.contains(&"beta"), "Should find federated result");
}

#[test]
fn test_hopfield_query_e2e() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    let topics = [
        "quantum physics experiments in laboratories",
        "medieval history of european kingdoms",
        "tropical marine biology and coral reefs",
        "classical piano compositions by beethoven",
        "volcanic geology and tectonic plate movements",
        "abstract algebra and group theory proofs",
        "impressionist painting techniques with oils",
        "arctic wildlife migration patterns observed",
        "distributed computing consensus algorithms",
        "ancient roman architecture and aqueducts",
    ];
    for (i, topic) in topics.iter().enumerate() {
        let v = hms.encode_text(topic);
        hms.memorize(format!("doc_{}", i), v).unwrap();
    }

    let q = hms.encode_text("volcanic geology and tectonic plate movements");
    let results = hms.query_hopfield(&q, 10);
    assert!(!results.is_empty(), "Hopfield query should return results");
    let top3_ids: Vec<&str> = results.iter().take(3).map(|r| r.id.as_str()).collect();
    assert!(
        top3_ids.contains(&"doc_4"),
        "Exact match should be in top-3 Hopfield results, got {:?}",
        top3_ids
    );
}

#[test]
fn test_graph_compact_preserves_relations() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    let v = hms.encode_text("compactable");
    hms.memorize("node1".to_string(), v).unwrap();
    hms.add_relation(&crate::core::types::Relation {
        source_id: "node1".into(),
        relation_type: "related".into(),
        target_id: "node2".into(),
        properties: None,
        valid_from: 0.0,
        valid_to: 0.0,
    })
    .unwrap();

    hms.compact().unwrap();
    assert_eq!(hms.relation_count(), 1);
    assert_eq!(hms.vector_count(), 1);
}

// === Analogy Tests ===

#[test]
fn test_find_analogy() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    for word in &[
        "walking", "talking", "running", "walked", "talked", "runner",
    ] {
        let v = hms.encode_text(word);
        hms.memorize(word.to_string(), v).unwrap();
    }

    let results = hms.find_analogy("walking", "walked", "talking", 5);
    assert!(!results.is_empty(), "Analogy should return results");
}

// === IVF Persistence ===

#[test]
fn test_ivf_index_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    {
        let mut config = crate::core::config::HmsConfig::default();
        config.ivf.n_clusters = 8;
        config.ivf.n_landmarks = 64;
        config.ivf.d_reduced = 16;
        config.ivf.n_probe = 8;

        let hms = HmsCore::new(10_000, Some(path.clone()), Some(config)).unwrap();
        for i in 0..200 {
            let v = hms.encode_text(&format!("ivf persist {}", i));
            hms.memorize(format!("ip_{}", i), v).unwrap();
        }
        hms.train_ivf().unwrap();
        assert!(hms.ivf_trained());
    }

    let mut config = crate::core::config::HmsConfig::default();
    config.ivf.n_clusters = 8;
    config.ivf.n_landmarks = 64;
    config.ivf.d_reduced = 16;
    config.ivf.n_probe = 8;
    let hms = HmsCore::new(10_000, Some(path), Some(config)).unwrap();
    assert!(
        hms.ivf_trained(),
        "IVF should be loaded from disk on re-open"
    );
    let q = hms.encode_text("ivf persist 0");
    let results = hms.query(&q, 5);
    assert!(!results.is_empty(), "IVF query should work after reload");
}

// === Query Batch Tests ===

#[test]
fn test_query_batch() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    for i in 0..20 {
        let v = hms.encode_text(&format!("batch query doc {}", i));
        hms.memorize(format!("bq_{}", i), v).unwrap();
    }

    let queries: Vec<EntangledHVec> = (0..3)
        .map(|i| hms.encode_text(&format!("batch query doc {}", i)))
        .collect();
    let batch_results = hms.query_batch(&queries, 3);
    assert_eq!(
        batch_results.len(),
        3,
        "Should return one result set per query"
    );
    for results in &batch_results {
        assert!(!results.is_empty(), "Each query should return results");
    }
}

// === Memorize Vector Tests ===

#[test]
fn test_memorize_vector_through_core() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    let dense: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
    hms.memorize_vector("dense_1".to_string(), &dense).unwrap();
    assert_eq!(hms.vector_count(), 1);

    let q = EntangledHVec::from_dense(&dense, 10_000);
    let results = hms.query(&q, 1);
    assert_eq!(results[0].id, "dense_1");
}

// === Readability Integration ===

#[test]
fn test_readability_through_core() {
    let dir = tempfile::tempdir().unwrap();
    let hms = HmsCore::new(10_000, Some(dir.path().display().to_string()), None).unwrap();
    let metrics = hms.analyze_text("The cat sat on the mat.");
    assert!(metrics.word_count > 0);
    let score = hms.calculate_readability(&metrics);
    assert!(
        score > 50.0,
        "Simple sentence should be highly readable (got {:.1})",
        score
    );
}

// === Index Persistence Tests ===

#[test]
fn test_nsg_index_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    {
        let mut config = crate::core::config::HmsConfig::default();
        config.nsg.max_degree = 8;
        config.nsg.ef_construction = 16;

        let hms = HmsCore::new(10_000, Some(path.clone()), Some(config)).unwrap();
        for i in 0..50 {
            let v = hms.encode_text(&format!("persist nsg {}", i));
            hms.memorize(format!("pn_{}", i), v).unwrap();
        }
        hms.train_nsg().unwrap();
        assert!(hms.nsg_trained());
    }

    // Re-open and verify NSG was reloaded
    let mut config = crate::core::config::HmsConfig::default();
    config.nsg.max_degree = 8;
    config.nsg.ef_construction = 16;
    let hms = HmsCore::new(10_000, Some(path), Some(config)).unwrap();
    assert!(
        hms.nsg_trained(),
        "NSG should be loaded from disk on re-open"
    );
    let q = hms.encode_text("persist nsg 0");
    let results = hms.query(&q, 5);
    assert!(!results.is_empty(), "NSG query should work after reload");
}

// === NSG Accuracy ===

#[test]
fn test_nsg_accuracy_matches_brute_force() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.nsg.max_degree = 8;
    config.nsg.ef_construction = 16;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..50 {
        let v = hms.encode_text(&format!("accuracy test {}", i));
        hms.memorize(format!("acc_{}", i), v).unwrap();
    }

    // Query before training (brute force)
    let q = hms.encode_text("accuracy test 0");
    let brute_results = hms.query(&q, 1);

    hms.train_nsg().unwrap();

    // Query after training (NSG)
    let nsg_results = hms.query(&q, 1);

    assert_eq!(
        brute_results[0].id, nsg_results[0].id,
        "NSG top-1 should match brute-force top-1"
    );
}

// === Multi-Shard Merge Verification ===

#[test]
fn test_multi_shard_results_from_multiple_shards() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.shard.enabled = true;
    config.shard.shard_count = 4;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..100 {
        let v = hms.encode_text(&format!("multi verify {}", i));
        hms.memorize(format!("mv_{}", i), v).unwrap();
    }

    let q = hms.encode_text("multi verify");
    let results = hms.query(&q, 20);
    assert!(
        results.len() >= 10,
        "Should return many results from across shards"
    );

    // Verify results come from at least 2 different shards
    let mut shard_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for r in &results {
        shard_ids.insert((crate::core::hash::fx64(r.id.as_bytes()) as usize) % 4);
    }
    assert!(
        shard_ids.len() >= 2,
        "Results should come from multiple shards (got {} distinct shards)",
        shard_ids.len()
    );
}

// === Compact Multi-Shard ===

#[test]
fn test_compact_multi_shard() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.shard.enabled = true;
    config.shard.shard_count = 2;

    let hms = HmsCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..30 {
        let v = hms.encode_text(&format!("compact shard {}", i));
        hms.memorize(format!("cs_{}", i), v).unwrap();
    }
    for i in 0..15 {
        hms.delete(&format!("cs_{}", i)).unwrap();
    }
    assert_eq!(hms.vector_count(), 15);

    hms.compact().unwrap();
    assert_eq!(hms.vector_count(), 15);

    let q = hms.encode_text("compact shard 20");
    let results = hms.query(&q, 5);
    assert!(
        !results.is_empty(),
        "Should find results after multi-shard compact"
    );
}

// === IVF Integration Tests ===

#[test]
fn test_manual_ivf_train_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.ivf.n_clusters = 8;
    config.ivf.n_landmarks = 64;
    config.ivf.d_reduced = 16;
    config.ivf.n_probe = 8;

    let hms = HmsCore::new(
        10000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..200 {
        let vec = hms.encode_text(&format!("document number {}", i));
        hms.memorize(format!("doc_{}", i), vec).unwrap();
    }

    assert!(!hms.ivf_trained());
    hms.train_ivf().unwrap();
    assert!(hms.ivf_trained());

    let q = hms.encode_text("document number 0");
    let results = hms.query(&q, 10);
    assert!(!results.is_empty(), "IVF query should return results");
}

#[test]
fn test_ivf_maintenance_at_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::HmsConfig::default();
    config.ivf.enabled = true;
    config.ivf.auto_threshold = 50;
    config.ivf.n_clusters = 8;
    config.ivf.n_landmarks = 64;
    config.ivf.d_reduced = 16;
    config.ivf.n_probe = 8;

    let hms = HmsCore::new(
        1000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    for i in 0..49 {
        let vec = hms.encode_text(&format!("auto item {}", i));
        hms.memorize(format!("auto_{}", i), vec).unwrap();
    }
    assert!(!hms.ivf_trained(), "Should not be trained before threshold");

    // Reaching the threshold requests maintenance outside ingestion.
    let vec = hms.encode_text("auto item 49");
    hms.memorize("auto_49".to_string(), vec).unwrap();
    assert!(!hms.ivf_trained());
    assert!(hms.index_status().ivf_training_recommended);
    hms.maintain_indices().unwrap();
    assert!(hms.ivf_trained());
}
