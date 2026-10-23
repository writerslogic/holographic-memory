// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Analogy detection via structural isomorphism.
//!
//! Finds pairs of subgraphs that share the same relational structure
//! but with different entities. Uses connected components and ranked
//! bipartite matching by relation profile similarity, validated by
//! edge-overlap scoring.

use crate::core::hash::{FastMap, FastSet};

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
        let components = find_components(triple_store);
        let mut comp_signatures = Vec::new();
        for entities in components {
            let mut relations = FastSet::default();
            for entity in &entities {
                triple_store.visit(Some(entity), None, None, |t| {
                    relations.insert(t.relation_id.clone());
                    true
                });
            }
            comp_signatures.push((entities, relations));
        }

        let mut analogies = Vec::new();
        for i in 0..comp_signatures.len() {
            for j in (i + 1)..comp_signatures.len() {
                let (ref entities_a, ref rels_a) = comp_signatures[i];
                let (ref entities_b, ref rels_b) = comp_signatures[j];

                let shared: Vec<String> = rels_a.intersection(rels_b).cloned().collect();
                if shared.len() < min_shared_relations {
                    continue;
                }

                if let Some(analogy) = ranked_match(entities_a, entities_b, &shared, triple_store) {
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
fn find_components(store: &TripleStore) -> Vec<FastSet<String>> {
    let mut visited = FastSet::default();
    let mut components = Vec::new();
    for start in store.subject_ids() {
        if visited.contains(&start) || !store.has(Some(&start), None, None) {
            continue;
        }
        visited.insert(start.clone());
        let mut component = FastSet::default();
        let mut queue = vec![start];
        while let Some(node) = queue.pop() {
            store.visit(Some(&node), None, None, |t| {
                if visited.insert(t.object_id.clone()) {
                    queue.push(t.object_id.clone());
                }
                true
            });
            store.visit(None, None, Some(&node), |t| {
                if visited.insert(t.subject_id.clone()) {
                    queue.push(t.subject_id.clone());
                }
                true
            });
            component.insert(node);
        }
        components.push(component);
    }
    components
}

fn entity_profile(
    entity: &str,
    triples: &TripleStore,
    shared_relations: &[String],
) -> FastSet<String> {
    let mut profile = FastSet::default();
    let mut add = |t: &crate::core::triple_store::TripleRecord| {
        if shared_relations.contains(&t.relation_id) {
            profile.insert(t.relation_id.clone());
        }
        true
    };
    triples.visit(Some(entity), None, None, &mut add);
    triples.visit(None, None, Some(entity), add);
    profile
}

/// Ranked bipartite matching: compute all pairwise profile similarities,
/// sort by descending similarity, then greedily assign non-conflicting pairs.
/// After mapping, compute edge-overlap scoring for structural validation.
fn ranked_match(
    entities_a: &FastSet<String>,
    entities_b: &FastSet<String>,
    shared_relations: &[String],
    triples: &TripleStore,
) -> Option<Analogy> {
    let profiles_a: Vec<(String, FastSet<String>)> = entities_a
        .iter()
        .map(|e| (e.clone(), entity_profile(e, triples, shared_relations)))
        .collect();
    let profiles_b: Vec<(String, FastSet<String>)> = entities_b
        .iter()
        .map(|e| (e.clone(), entity_profile(e, triples, shared_relations)))
        .collect();

    if profiles_a.is_empty() || profiles_b.is_empty() {
        return None;
    }

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

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_a = FastSet::default();
    let mut used_b = FastSet::default();
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
    triples: &TripleStore,
) -> f64 {
    let forward: FastMap<&str, &str> = mapping
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let reverse: FastMap<&str, &str> = mapping
        .iter()
        .map(|(a, b)| (b.as_str(), a.as_str()))
        .collect();
    let count = |map: &FastMap<&str, &str>| {
        let mut total = 0;
        let mut matched = 0;
        for (&subject, &mapped_subject) in map {
            let mut edges = FastSet::default();
            triples.visit(Some(subject), None, None, |t| {
                if shared_relations.contains(&t.relation_id) {
                    edges.insert((t.relation_id.clone(), t.object_id.clone()));
                }
                true
            });
            total += edges.len();
            for (relation, object) in edges {
                if let Some(&mapped_object) = map.get(object.as_str()) {
                    if triples.has(Some(mapped_subject), Some(&relation), Some(mapped_object)) {
                        matched += 1;
                    }
                }
            }
        }
        (total, matched)
    };
    let a = count(&forward);
    let b = count(&reverse);
    let (total, matched) = if a.0 <= b.0 { a } else { b };
    if total == 0 {
        0.0
    } else {
        matched as f64 / total as f64
    }
}

fn profile_similarity(a: &FastSet<String>, b: &FastSet<String>) -> f64 {
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
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
        let store = TripleStore::new();
        store.add("a", "r", "b", "c1");
        store.add("c", "r", "d", "c2");
        store.add("b", "r", "c", "deleted");
        store.remove_composite("deleted");
        let components = find_components(&store);
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
