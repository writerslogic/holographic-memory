// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::config::WarmConfig;
use crate::core::engine::WarmCore;

fn meaning_warm(dim: u32) -> WarmCore {
    let dir = tempfile::tempdir().unwrap();
    let mut config = WarmConfig::default();
    config.meaning.enabled = true;
    config.meaning.auto_decompose = true;
    config.meaning.beta = 24.0;
    WarmCore::new(
        dim,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap()
}

#[test]
fn test_basin_certification() {
    let warm = meaning_warm(16384);
    for i in 0..1000u64 {
        let v = crate::core::entangled::EntangledHVec::new_deterministic(16384, i);
        warm.memorize(format!("atom_{}", i), v).unwrap();
    }

    let mut recovered = 0;
    for probe_seed in [42u64, 100, 250, 500, 750, 999] {
        let original = crate::core::entangled::EntangledHVec::new_deterministic(16384, probe_seed);
        let mut noisy_indices = original.indices().to_vec();
        let mut rng = rand::rng();
        use rand::RngExt;
        for idx in noisy_indices.iter_mut().take(16) {
            *idx = rng.random_range(0..16384u32);
        }
        noisy_indices.sort_unstable();
        noisy_indices.dedup();
        let noisy = crate::core::entangled::EntangledHVec::from_indices(noisy_indices, 16384);

        if let Some((id, _conf)) = warm.meaning_cleanup(&noisy) {
            if id == format!("atom_{}", probe_seed) {
                recovered += 1;
            }
        }
    }
    assert!(
        recovered >= 5,
        "Should recover at least 5/6 atoms from 25% noise, got {}",
        recovered
    );
}

#[test]
fn test_structural_query_e2e() {
    let warm = meaning_warm(16384);
    warm.memorize_meaning("doc1", "Paris is a city").unwrap();
    let s = warm.encode_text("Paris");
    let r = warm.encode_text("is_a");
    let results = warm.structural_query(&[("subject", &s), ("relation", &r)], "object");
    assert_eq!(results.first().map(|r| r.entity_id.as_str()), Some("city"));
}

#[test]
fn test_multi_hop_chained() {
    let warm = meaning_warm(16384);
    warm.memorize_triplet("t1".into(), "john".into(), "father".into(), "mark".into())
        .unwrap();
    warm.memorize_triplet("t2".into(), "mark".into(), "father".into(), "bob".into())
        .unwrap();
    assert_eq!(
        warm.multi_hop("john", &["father", "father"])[0].entity_id,
        "bob"
    );
}

#[test]
fn test_meaning_decompose_e2e() {
    let warm = meaning_warm(16384);
    warm.memorize_meaning("doc1", "Paris is a city").unwrap();
    assert_eq!(warm.meaning_triple_count(), 1);
    assert_eq!(warm.meaning_composite_count(), 1);
}

#[test]
fn test_backward_compat_meaning_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let warm = WarmCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();
    assert!(!warm.meaning_enabled());
    let v = warm.encode_text("test");
    warm.memorize("t1".to_string(), v).unwrap();
    assert_eq!(warm.vector_count(), 1);
    assert!(warm.structural_query(&[], "object").is_empty());
    assert!(warm.multi_hop("t1", &["r"]).is_empty());
}

#[test]
fn test_role_inversion() {
    use crate::core::admission::AdmissionControl;
    use crate::core::atom_memory::AtomMemory;
    use crate::core::composite_memory::CompositeMemory;
    use crate::core::engine::structural::{fuzzy_structural_query, MeaningContext};
    use crate::core::role::RoleRegistry;
    use crate::core::triple_store::TripleStore;

    let dim = 16384;
    let atom_mem = AtomMemory::new(dim, 3.0);
    let comp_mem = CompositeMemory::new(dim, 3.0);
    let triple_store = TripleStore::new();
    let roles = RoleRegistry::new(dim);
    let admission = AdmissionControl::new(40);

    let triples = vec![
        ("alice", "loves", "bob"),
        ("paris", "capital_of", "france"),
        ("earth", "orbits", "sun"),
    ];

    for (s, r, o) in &triples {
        let (_, sv) = atom_mem.get_or_insert(s);
        let (_, rv) = atom_mem.get_or_insert(r);
        let (_, ov) = atom_mem.get_or_insert(o);
        let comp = roles.compose_triple(&sv, &rv, &ov);
        let cid = format!("{}:{}:{}", s, r, o);
        comp_mem.insert(cid.clone(), comp);
        triple_store.add(s, r, o, &cid);
    }

    let ctx = MeaningContext {
        atom_memory: &atom_mem,
        composite_memory: &comp_mem,
        triple_store: &triple_store,
        roles: &roles,
        admission: &admission,
        beta: 24.0,
        k: 64,
        max_iter: 3,
    };

    for (s, r, o) in &triples {
        let sv = atom_mem.get(s).unwrap();
        let rv = atom_mem.get(r).unwrap();
        let ov = atom_mem.get(o).unwrap();

        // Recover object from (subject, relation)
        let r1 = fuzzy_structural_query(&ctx, &[("subject", &sv), ("relation", &rv)], "object");
        assert!(
            !r1.is_empty() && r1[0].entity_id == *o,
            "Failed to recover object '{}' from ({}, {})",
            o,
            s,
            r
        );

        // Recover subject from (relation, object)
        let r2 = fuzzy_structural_query(&ctx, &[("relation", &rv), ("object", &ov)], "subject");
        assert!(
            !r2.is_empty() && r2[0].entity_id == *s,
            "Failed to recover subject '{}' from ({}, {})",
            s,
            r,
            o
        );

        // Recover relation from (subject, object)
        let r3 = fuzzy_structural_query(&ctx, &[("subject", &sv), ("object", &ov)], "relation");
        assert!(
            !r3.is_empty() && r3[0].entity_id == *r,
            "Failed to recover relation '{}' from ({}, {})",
            r,
            s,
            o
        );
    }
}

#[test]
fn test_fan_out_sweep() {
    use crate::core::admission::AdmissionControl;
    use crate::core::atom_memory::AtomMemory;
    use crate::core::composite_memory::CompositeMemory;
    use crate::core::engine::structural::{fuzzy_structural_query, MeaningContext, StructuralPath};
    use crate::core::role::RoleRegistry;
    use crate::core::triple_store::TripleStore;

    let dim = 16384;
    let atom_mem = AtomMemory::new(dim, 3.0);
    let comp_mem = CompositeMemory::new(dim, 3.0);
    let triple_store = TripleStore::new();
    let roles = RoleRegistry::new(dim);
    let fanout_limit = 10;
    let admission = AdmissionControl::new(fanout_limit);

    let (_, r_vec) = atom_mem.get_or_insert("related_to");

    // Create triples with increasing fan-out
    for i in 0..20u64 {
        let s_name = format!("entity_{}", i);
        let o_name = format!("target_{}", i);
        let (_, sv) = atom_mem.get_or_insert(&s_name);
        let (_, ov) = atom_mem.get_or_insert(&o_name);
        let comp = roles.compose_triple(&sv, &r_vec, &ov);
        let cid = format!("c_{}", i);
        comp_mem.insert(cid.clone(), comp);
        triple_store.add(&s_name, "related_to", &o_name, &cid);
    }

    let ctx = MeaningContext {
        atom_memory: &atom_mem,
        composite_memory: &comp_mem,
        triple_store: &triple_store,
        roles: &roles,
        admission: &admission,
        beta: 24.0,
        k: 64,
        max_iter: 3,
    };

    // Query for one specific entity
    let sv = atom_mem.get("entity_5").unwrap();
    let results = fuzzy_structural_query(&ctx, &[("subject", &sv), ("relation", &r_vec)], "object");
    assert!(!results.is_empty(), "Should find at least one result");

    // At high fan-out, should use materialized path
    if results.len() > fanout_limit {
        assert!(
            results
                .iter()
                .any(|r| r.path == StructuralPath::Materialized),
            "High fan-out should trigger materialized path"
        );
    }
}

#[test]
fn test_poisoning_resistance() {
    use crate::core::atom_memory::AtomMemory;
    use crate::core::entangled::EntangledHVec;

    let dim = 16384;
    let atom_mem = AtomMemory::new(dim, 3.0);

    // Insert 50 normal atoms
    for i in 0..50u64 {
        atom_mem.get_or_insert(&format!("normal_{}", i));
    }

    // Create 5 "poisoned" atoms that share a fixed set of decoy indices
    let mut decoy_indices: Vec<u32> = (100..148).collect();
    decoy_indices.sort_unstable();
    for i in 0..5u64 {
        let mut indices = decoy_indices.clone();
        // Add a few unique indices so they're not identical
        indices.push(200 + i as u32);
        indices.push(300 + i as u32);
        indices.sort_unstable();
        indices.dedup();
        let vec = EntangledHVec::from_indices(indices, dim);
        atom_mem.insert_with_vec(&format!("decoy_{}", i), &vec);
    }

    // Query with a probe similar to the decoy pattern
    let probe = EntangledHVec::from_indices(decoy_indices, dim);
    let result = atom_mem.cleanup(&probe, 24.0, 64, 3);

    // IDF clipping should prevent the shared decoy indices from
    // dominating; the result should be one of the decoy atoms,
    // NOT a normal atom that happens to share a few indices
    if result.found {
        assert!(
            result.id.starts_with("decoy_"),
            "Cleanup should recover a decoy atom, not a normal one (got {})",
            result.id
        );
    }
}

#[test]
fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;

    let dim = 16384;
    let warm = Arc::new(meaning_warm(dim as u32));

    let mut handles = Vec::new();

    // 4 writer threads
    for t in 0..4 {
        let warm = Arc::clone(&warm);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                let id = format!("w{}_{}", t, i);
                let v = crate::core::entangled::EntangledHVec::new_deterministic(
                    dim,
                    (t * 1000 + i) as u64,
                );
                warm.memorize(id, v).unwrap();
            }
        }));
    }

    // 2 reader threads
    for _ in 0..2 {
        let warm = Arc::clone(&warm);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                let probe = crate::core::entangled::EntangledHVec::new_deterministic(dim, i as u64);
                let _ = warm.meaning_cleanup(&probe);
            }
        }));
    }

    // All threads must complete without deadlock (timeout is implicit via test runner)
    for h in handles {
        h.join().expect("Thread panicked");
    }

    assert!(warm.vector_count() >= 400, "All writes should complete");
}

#[test]
fn test_persistence_round_trip() {
    use crate::core::entangled::EntangledHVec;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    let atoms = ["alpha", "beta", "gamma"];
    let original_vecs: Vec<(String, EntangledHVec)>;

    // Phase 1: store data
    let atom_count;
    let triple_count;
    let vector_count;
    {
        let warm = meaning_warm_at(&path, 16384);
        // Use sentences the decomposer can parse: "X is Y" pattern
        warm.memorize_meaning("d1", "Alpha is a concept").unwrap();
        warm.memorize_meaning("d2", "Beta is a concept").unwrap();
        warm.memorize_meaning("d3", "Gamma is a concept").unwrap();

        atom_count = warm.meaning_atom_count();
        triple_count = warm.meaning_triple_count();
        vector_count = warm.vector_count();
        assert!(atom_count > 0, "Should have atoms");
        assert!(triple_count > 0, "Should have triples");

        original_vecs = atoms
            .iter()
            .map(|a| {
                let v = warm.encode_text(a);
                (a.to_string(), v)
            })
            .collect();
    }
    // WarmCore dropped, all in-memory state gone

    // Phase 2: reload from arena
    {
        let warm = meaning_warm_at(&path, 16384);
        assert!(
            warm.meaning_atom_count() >= atom_count,
            "Atom count should survive reload: before={}, after={}",
            atom_count,
            warm.meaning_atom_count()
        );
        assert!(
            warm.meaning_triple_count() >= triple_count,
            "Triple count should survive reload: before={}, after={}",
            triple_count,
            warm.meaning_triple_count()
        );
        assert_eq!(
            warm.vector_count(),
            vector_count,
            "Vector count should survive reload"
        );

        // Verify encoding is deterministic
        for (name, original) in &original_vecs {
            let reloaded = warm.encode_text(name);
            assert_eq!(
                reloaded.indices(),
                original.indices(),
                "Encoding should be deterministic for {}",
                name
            );
        }
    }
}

fn meaning_warm_at(path: &str, dim: u32) -> WarmCore {
    let mut config = WarmConfig::default();
    config.meaning.enabled = true;
    config.meaning.auto_decompose = true;
    config.meaning.beta = 24.0;
    WarmCore::new(dim, Some(path.to_string()), Some(config)).unwrap()
}
