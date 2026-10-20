// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Epistemic gap detection.
//!
//! Compares an entity's relation profile against its peers to find
//! missing relations. For example, if most cities have a `country` relation
//! but city X does not, that is a gap.

use fxhash::{FxHashMap, FxHashSet};

use crate::core::triple_store::TripleStore;

/// A detected knowledge gap: an entity is missing a relation that its peers have.
#[derive(Clone, Debug)]
pub struct KnowledgeGap {
    /// The entity missing the relation.
    pub entity: String,
    /// The relation that is missing.
    pub missing_relation: String,
    /// Peer entities that do have this relation (evidence).
    pub peers_with_relation: Vec<String>,
    /// Fraction of peers that have this relation (0.0 - 1.0).
    pub peer_coverage: f64,
}

/// Detects knowledge gaps by comparing entity profiles to peer profiles.
pub struct GapDetector;

impl GapDetector {
    /// Find entities that are missing relations their peers have.
    ///
    /// Two entities are peers if they share at least `min_shared_relations` relations.
    /// A gap is reported when `>= min_peer_coverage` fraction of peers have a relation
    /// that the entity lacks.
    pub fn detect(
        triple_store: &TripleStore,
        min_shared_relations: usize,
        min_peer_coverage: f64,
    ) -> Vec<KnowledgeGap> {
        let snapshot = triple_store.snapshot();

        // Build entity -> set of relations (as subject)
        let mut entity_rels: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
        for t in &snapshot {
            entity_rels
                .entry(t.subject_id.clone())
                .or_default()
                .insert(t.relation_id.clone());
        }

        // Build relation -> set of subjects (for peer lookup)
        let mut rel_subjects: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
        for t in &snapshot {
            rel_subjects
                .entry(t.relation_id.clone())
                .or_default()
                .insert(t.subject_id.clone());
        }

        let mut gaps = Vec::new();

        for (entity, rels) in &entity_rels {
            // Find peers: other entities sharing >= min_shared_relations
            let mut peer_candidates: FxHashMap<String, usize> = FxHashMap::default();
            for rel in rels {
                if let Some(subjects) = rel_subjects.get(rel) {
                    for peer in subjects {
                        if peer != entity {
                            *peer_candidates.entry(peer.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }

            let peers: Vec<String> = peer_candidates
                .into_iter()
                .filter(|&(_, count)| count >= min_shared_relations)
                .map(|(id, _)| id)
                .collect();

            if peers.is_empty() {
                continue;
            }

            // Collect all relations that peers have (O(1) lookup via entity_rels)
            let mut peer_rel_counts: FxHashMap<String, usize> = FxHashMap::default();
            for peer in &peers {
                if let Some(peer_rels) = entity_rels.get(peer) {
                    for rel in peer_rels {
                        *peer_rel_counts.entry(rel.clone()).or_insert(0) += 1;
                    }
                }
            }

            // Check for gaps: relations the entity doesn't have but peers do
            let peer_count = peers.len();
            for (rel, count) in &peer_rel_counts {
                if rels.contains(rel) {
                    continue;
                }
                let coverage = *count as f64 / peer_count as f64;
                if coverage >= min_peer_coverage {
                    let peers_with: Vec<String> = peers
                        .iter()
                        .filter(|p| entity_rels.get(*p).is_some_and(|pr| pr.contains(rel)))
                        .cloned()
                        .collect();

                    gaps.push(KnowledgeGap {
                        entity: entity.clone(),
                        missing_relation: rel.clone(),
                        peers_with_relation: peers_with,
                        peer_coverage: coverage,
                    });
                }
            }
        }

        gaps.sort_by(|a, b| {
            b.peer_coverage
                .partial_cmp(&a.peer_coverage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        gaps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gaps() {
        let store = TripleStore::new();
        // Paris and Berlin are in Europe and are capitals
        store.add("paris", "capital_of", "france", "c1");
        store.add("paris", "located_in", "europe", "c2");
        store.add("berlin", "capital_of", "germany", "c3");
        store.add("berlin", "located_in", "europe", "c4");
        // Tokyo is a capital but missing located_in
        store.add("tokyo", "capital_of", "japan", "c5");

        let gaps = GapDetector::detect(&store, 1, 0.5);

        let tokyo_gap = gaps
            .iter()
            .find(|g| g.entity == "tokyo" && g.missing_relation == "located_in");
        assert!(tokyo_gap.is_some(), "Tokyo should be missing located_in");
        let gap = tokyo_gap.unwrap();
        assert!(gap.peer_coverage >= 0.5);
        assert!(!gap.peers_with_relation.is_empty());
    }

    #[test]
    fn test_no_gaps_when_complete() {
        let store = TripleStore::new();
        store.add("a", "r1", "x", "c1");
        store.add("b", "r1", "y", "c2");
        // Both have the same relation, no gaps
        let gaps = GapDetector::detect(&store, 1, 0.5);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_no_gaps_empty_store() {
        let store = TripleStore::new();
        let gaps = GapDetector::detect(&store, 1, 0.5);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_high_coverage_threshold() {
        let store = TripleStore::new();
        store.add("a", "r1", "x", "c1");
        store.add("a", "r2", "x", "c2");
        store.add("b", "r1", "y", "c3");
        // b is missing r2, but only 1 peer has it (100% of 1 peer)
        // With min_shared=1, a is a peer of b
        let gaps = GapDetector::detect(&store, 1, 1.0);
        let b_gap = gaps
            .iter()
            .find(|g| g.entity == "b" && g.missing_relation == "r2");
        assert!(b_gap.is_some());
    }
}
