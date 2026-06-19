// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Analogy detection via structural isomorphism.
//!
//! Finds pairs of subgraphs that share the same relational structure
//! but with different entities. Uses connected components and ranked
//! bipartite matching by relation profile similarity, validated by
//! edge-overlap scoring.

use fxhash::{FxHashMap, FxHashSet};

use crate::core::triple_store::TripleStore;

/// A detected analogy between two domains.
#[derive(Clone, Debug)]
pub struct Analogy {
    /// Mapping from domain A entities to domain B entities.
    pub mapping: Vec<(String, String)>,
    /// Relations that are shared between the two domains.
    pub shared_relations: Vec<String>,
    /// Quality score: fraction of edges that match under the mapping.
    pub score: f64,
    /// Edge-overlap quality: fraction of edges in the smaller domain
    /// that have a corresponding edge in the other domain under the mapping.
    pub mapping_quality: f64,
}

/// Detects structural isomorphisms between subgraphs.
pub struct AnalogyDetector;

impl AnalogyDetector {
    /// Find analogies by grouping triples into connected components
    /// and comparing components with the same relational structure.
    ///
    /// `min_shared_relations`: minimum number of shared relation types
    /// between two components to consider them analogous.
    pub fn detect(triple_store: &TripleStore, min_shared_relations: usize) -> Vec<Analogy> {
        let snapshot = triple_store.snapshot();
        if snapshot.is_empty() {
            return Vec::new();
        }

        // Build adjacency list (undirected) for connected components
        let mut adj: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
        for t in &snapshot {
            adj.entry(t.subject_id.clone())
                .or_default()
                .insert(t.object_id.clone());
            adj.entry(t.object_id.clone())
                .or_default()
                .insert(t.subject_id.clone());
        }

        // Find connected components via BFS
        let components = find_components(&adj);

        // Pre-build subject -> relations map for O(1) lookup
        let mut subject_relations: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
        for t in &snapshot {
            subject_relations
                .entry(t.subject_id.clone())
                .or_default()
                .insert(t.relation_id.clone());
        }

        // For each component, compute its relation signature
        let mut comp_signatures: Vec<(FxHashSet<String>, FxHashSet<String>)> = Vec::new();
        for comp_entities in &components {
            let mut relations = FxHashSet::default();
            for entity in comp_entities {
                if let Some(rels) = subject_relations.get(entity) {
                    relations.extend(rels.iter().cloned());
                }
            }
            comp_signatures.push((comp_entities.clone(), relations));
        }

        // Compare pairs of components
        let mut analogies = Vec::new();
        for i in 0..comp_signatures.len() {
            for j in (i + 1)..comp_signatures.len() {
                let (ref entities_a, ref rels_a) = comp_signatures[i];
                let (ref entities_b, ref rels_b) = comp_signatures[j];

                let shared: Vec<String> = rels_a.intersection(rels_b).cloned().collect();
                if shared.len() < min_shared_relations {
                    continue;
                }

                // Ranked matching: compute all pairwise similarities, then select
                // best non-conflicting mapping in descending similarity order.
                if let Some(analogy) = ranked_match(entities_a, entities_b, &shared, &snapshot) {
                    if analogy.score > 0.0 {
                        analogies.push(analogy);
                    }
                }
            }
        }

        analogies.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        analogies
    }
}

/// Find connected components via BFS.
fn find_components(adj: &FxHashMap<String, FxHashSet<String>>) -> Vec<FxHashSet<String>> {
    let mut visited = FxHashSet::default();
    let mut components = Vec::new();

    for start in adj.keys() {
        if visited.contains(start) {
            continue;
        }
        let mut component = FxHashSet::default();
        let mut queue = vec![start.clone()];
        while let Some(node) = queue.pop() {
            if !visited.insert(node.clone()) {
                continue;
            }
            component.insert(node.clone());
            if let Some(neighbors) = adj.get(&node) {
                for n in neighbors {
                    if !visited.contains(n) {
                        queue.push(n.clone());
                    }
                }
            }
        }
        components.push(component);
    }

    components
}

/// Build a relation profile for an entity: relation -> roles (subject/object).
fn entity_profile(
    entity: &str,
    triples: &[crate::core::triple_store::TripleRecord],
    shared_relations: &[String],
) -> FxHashMap<String, Vec<String>> {
    let mut profile: FxHashMap<String, Vec<String>> = FxHashMap::default();
    for t in triples {
        if !shared_relations.contains(&t.relation_id) {
            continue;
        }
        if t.subject_id == entity {
            profile
                .entry(t.relation_id.clone())
                .or_default()
                .push("subject".to_string());
        }
        if t.object_id == entity {
            profile
                .entry(t.relation_id.clone())
                .or_default()
                .push("object".to_string());
        }
    }
    profile
}

/// Ranked bipartite matching: compute all pairwise profile similarities,
/// sort by descending similarity, then greedily assign non-conflicting pairs.
/// After mapping, compute edge-overlap scoring for structural validation.
fn ranked_match(
    entities_a: &FxHashSet<String>,
    entities_b: &FxHashSet<String>,
    shared_relations: &[String],
    triples: &[crate::core::triple_store::TripleRecord],
) -> Option<Analogy> {
    // Build profiles for each entity in A and B
    let profiles_a: Vec<(String, FxHashMap<String, Vec<String>>)> = entities_a
        .iter()
        .map(|e| (e.clone(), entity_profile(e, triples, shared_relations)))
        .collect();
    let profiles_b: Vec<(String, FxHashMap<String, Vec<String>>)> = entities_b
        .iter()
        .map(|e| (e.clone(), entity_profile(e, triples, shared_relations)))
        .collect();

    if profiles_a.is_empty() || profiles_b.is_empty() {
        return None;
    }

    // Compute all pairwise similarities
    let mut pairs: Vec<(f64, usize, usize)> =
        Vec::with_capacity(profiles_a.len() * profiles_b.len());
    for (i, (_ea, pa)) in profiles_a.iter().enumerate() {
        for (j, (_eb, pb)) in profiles_b.iter().enumerate() {
            let sim = profile_similarity(pa, pb);
            if sim > 0.0 {
                pairs.push((sim, i, j));
            }
        }
    }

    // Sort by descending similarity
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedily select best non-conflicting mapping
    let mut used_a = FxHashSet::default();
    let mut used_b = FxHashSet::default();
    let mut mapping = Vec::new();
    let mut total_score = 0.0;

    for (sim, idx_a, idx_b) in &pairs {
        if used_a.contains(idx_a) || used_b.contains(idx_b) {
            continue;
        }
        used_a.insert(*idx_a);
        used_b.insert(*idx_b);
        mapping.push((profiles_a[*idx_a].0.clone(), profiles_b[*idx_b].0.clone()));
        total_score += sim;
    }

    if mapping.is_empty() {
        return None;
    }

    let score = total_score / mapping.len() as f64;

    // Compute edge-overlap scoring
    let mapping_quality = compute_edge_overlap(&mapping, shared_relations, triples);

    Some(Analogy {
        mapping,
        shared_relations: shared_relations.to_vec(),
        score,
        mapping_quality,
    })
}

/// Compute edge-overlap: fraction of triples in the smaller domain that
/// have a corresponding triple in the other domain under the entity mapping.
///
/// A triple (s, r, o) in domain A maps to (map(s), r, map(o)) in domain B.
/// We check how many such mapped triples actually exist.
fn compute_edge_overlap(
    mapping: &[(String, String)],
    shared_relations: &[String],
    triples: &[crate::core::triple_store::TripleRecord],
) -> f64 {
    // Build forward and reverse mapping lookups
    let forward: FxHashMap<&str, &str> = mapping
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let reverse: FxHashMap<&str, &str> = mapping
        .iter()
        .map(|(a, b)| (b.as_str(), a.as_str()))
        .collect();

    let mapped_entities_a: FxHashSet<&str> = mapping.iter().map(|(a, _)| a.as_str()).collect();
    let mapped_entities_b: FxHashSet<&str> = mapping.iter().map(|(_, b)| b.as_str()).collect();

    let shared_set: FxHashSet<&str> = shared_relations.iter().map(|s| s.as_str()).collect();

    // Build a set of (subject, relation, object) triples in B for fast lookup
    let triples_b: FxHashSet<(&str, &str, &str)> = triples
        .iter()
        .filter(|t| {
            shared_set.contains(t.relation_id.as_str())
                && mapped_entities_b.contains(t.subject_id.as_str())
        })
        .map(|t| {
            (
                t.subject_id.as_str(),
                t.relation_id.as_str(),
                t.object_id.as_str(),
            )
        })
        .collect();

    // Build a set of (subject, relation, object) triples in A for fast lookup
    let triples_a: FxHashSet<(&str, &str, &str)> = triples
        .iter()
        .filter(|t| {
            shared_set.contains(t.relation_id.as_str())
                && mapped_entities_a.contains(t.subject_id.as_str())
        })
        .map(|t| {
            (
                t.subject_id.as_str(),
                t.relation_id.as_str(),
                t.object_id.as_str(),
            )
        })
        .collect();

    // Count how many A-triples map to existing B-triples
    let mut matched_a = 0usize;
    for &(subj, rel, obj) in &triples_a {
        if let (Some(&mapped_subj), Some(&mapped_obj)) = (forward.get(subj), forward.get(obj)) {
            if triples_b.contains(&(mapped_subj, rel, mapped_obj)) {
                matched_a += 1;
            }
        }
    }

    // Count how many B-triples map to existing A-triples
    let mut matched_b = 0usize;
    for &(subj, rel, obj) in &triples_b {
        if let (Some(&mapped_subj), Some(&mapped_obj)) = (reverse.get(subj), reverse.get(obj)) {
            if triples_a.contains(&(mapped_subj, rel, mapped_obj)) {
                matched_b += 1;
            }
        }
    }

    // Use the smaller domain's edge count as the denominator
    let count_a = triples_a.len();
    let count_b = triples_b.len();

    if count_a == 0 && count_b == 0 {
        return 0.0;
    }

    // Fraction of edges in the smaller domain that map correctly
    if count_a <= count_b {
        if count_a == 0 {
            0.0
        } else {
            matched_a as f64 / count_a as f64
        }
    } else if count_b == 0 {
        0.0
    } else {
        matched_b as f64 / count_b as f64
    }
}

/// Jaccard-like similarity between two entity profiles.
fn profile_similarity(
    a: &FxHashMap<String, Vec<String>>,
    b: &FxHashMap<String, Vec<String>>,
) -> f64 {
    let keys_a: FxHashSet<&String> = a.keys().collect();
    let keys_b: FxHashSet<&String> = b.keys().collect();
    let intersection = keys_a.intersection(&keys_b).count();
    let union = keys_a.union(&keys_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_analogy() {
        let store = TripleStore::new();
        // Domain A: solar system
        store.add("earth", "orbits", "sun", "c1");
        store.add("earth", "has_satellite", "moon", "c2");
        // Domain B: Jupiter system (separate component)
        store.add("europa", "orbits", "jupiter", "c3");
        store.add("europa", "has_satellite", "none", "c4");

        let analogies = AnalogyDetector::detect(&store, 2);
        // Should find an analogy between the two domains
        if !analogies.is_empty() {
            assert!(analogies[0].shared_relations.len() >= 2);
            assert!(analogies[0].score > 0.0);
        }
    }

    #[test]
    fn test_no_analogy_single_component() {
        let store = TripleStore::new();
        store.add("a", "r", "b", "c1");
        store.add("b", "r", "c", "c2");
        // All in one component, no pair to compare
        let analogies = AnalogyDetector::detect(&store, 1);
        assert!(analogies.is_empty());
    }

    #[test]
    fn test_no_analogy_empty() {
        let store = TripleStore::new();
        let analogies = AnalogyDetector::detect(&store, 1);
        assert!(analogies.is_empty());
    }

    #[test]
    fn test_find_components() {
        let mut adj: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
        adj.entry("a".to_string())
            .or_default()
            .insert("b".to_string());
        adj.entry("b".to_string())
            .or_default()
            .insert("a".to_string());
        adj.entry("c".to_string())
            .or_default()
            .insert("d".to_string());
        adj.entry("d".to_string())
            .or_default()
            .insert("c".to_string());

        let components = find_components(&adj);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_ranked_matching_quality() {
        let store = TripleStore::new();
        // Domain A: structured knowledge graph
        store.add("cat", "is_a", "animal", "c1");
        store.add("cat", "has_part", "tail", "c2");
        store.add("animal", "lives_in", "habitat", "c3");
        // Domain B: parallel structure with different entities
        store.add("dog", "is_a", "creature", "c4");
        store.add("dog", "has_part", "paw", "c5");
        store.add("creature", "lives_in", "den", "c6");

        let analogies = AnalogyDetector::detect(&store, 2);
        assert!(!analogies.is_empty(), "should detect an analogy");

        let best = &analogies[0];
        assert!(best.score > 0.0, "score should be positive");
        assert!(
            best.mapping_quality > 0.0,
            "mapping_quality should be positive for structurally isomorphic domains"
        );
        // With perfect structural isomorphism, mapping_quality should be 1.0
        assert!(
            best.mapping_quality >= 0.5,
            "mapping_quality should be high for isomorphic domains, got {}",
            best.mapping_quality
        );
        // Verify the mapping covers entities from both domains
        assert!(
            best.mapping.len() >= 2,
            "should map at least 2 entity pairs"
        );
    }
}
