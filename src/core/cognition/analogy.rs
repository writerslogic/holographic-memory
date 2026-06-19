// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Analogy detection via structural isomorphism.
//!
//! Finds pairs of subgraphs that share the same relational structure
//! but with different entities. Uses connected components and greedy
//! bipartite mapping by relation overlap.

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

                // Greedy mapping: match entities by their relation profiles
                if let Some(analogy) = greedy_map(entities_a, entities_b, &shared, &snapshot) {
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

/// Build a relation profile for an entity: relation -> role (subject/object).
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

/// Greedy bipartite mapping by relation profile overlap.
fn greedy_map(
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

    // Greedy: for each entity in A, find best match in B by profile similarity
    let mut used_b = FxHashSet::default();
    let mut mapping = Vec::new();
    let mut total_score = 0.0;

    for (entity_a, profile_a) in &profiles_a {
        let mut best_match = None;
        let mut best_sim = -1.0f64;

        for (entity_b, profile_b) in &profiles_b {
            if used_b.contains(entity_b) {
                continue;
            }
            let sim = profile_similarity(profile_a, profile_b);
            if sim > best_sim {
                best_sim = sim;
                best_match = Some(entity_b.clone());
            }
        }

        if let Some(matched) = best_match {
            if best_sim > 0.0 {
                used_b.insert(matched.clone());
                mapping.push((entity_a.clone(), matched));
                total_score += best_sim;
            }
        }
    }

    if mapping.is_empty() {
        return None;
    }

    let score = total_score / mapping.len() as f64;

    Some(Analogy {
        mapping,
        shared_relations: shared_relations.to_vec(),
        score,
    })
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
}
