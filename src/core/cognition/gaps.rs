// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Epistemic gap detection.
//!
//! Compares an entity's relation profile against its peers to find
//! missing relations. For example, if most cities have a `country` relation
//! but city X does not, that is a gap.

use crate::core::hash::{FastMap, FastSet};

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
        let relations = |entity: &str| {
            let mut rels = FastSet::default();
            triple_store.visit(Some(entity), None, None, |t| {
                rels.insert(t.relation_id.clone());
                true
            });
            rels
        };
        let mut gaps = Vec::new();
        for entity in triple_store.subject_ids() {
            let rels = relations(&entity);
            let mut peer_candidates: FastMap<String, usize> = FastMap::default();
            for rel in &rels {
                let mut subjects = FastSet::default();
                triple_store.visit(None, Some(rel), None, |t| {
                    subjects.insert(t.subject_id.clone());
                    true
                });
                for peer in subjects {
                    if peer != entity {
                        *peer_candidates.entry(peer).or_default() += 1;
                    }
                }
            }
            let peers: Vec<_> = peer_candidates
                .into_iter()
                .filter(|(_, count)| *count >= min_shared_relations)
                .map(|(peer, _)| peer)
                .collect();
            let mut evidence: FastMap<String, Vec<String>> = FastMap::default();
            for peer in &peers {
                for rel in relations(peer) {
                    if !rels.contains(&rel) {
                        evidence.entry(rel).or_default().push(peer.clone());
                    }
                }
            }
            for (missing_relation, peers_with_relation) in evidence {
                let peer_coverage = peers_with_relation.len() as f64 / peers.len() as f64;
                if peer_coverage >= min_peer_coverage {
                    gaps.push(KnowledgeGap {
                        entity: entity.clone(),
                        missing_relation,
                        peers_with_relation,
                        peer_coverage,
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
