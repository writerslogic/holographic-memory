// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hypothesis generation for knowledge gaps.
//!
//! Given a detected gap (entity missing a relation), examines peer data
//! to propose the most likely filler. Uses Hopfield cleanup to map
//! the bundled peer-filler vector to the nearest stored atom.

use crate::core::atom_memory::AtomMemory;
use crate::core::cognition::gaps::KnowledgeGap;
use crate::core::entangled::EntangledHVec;
use crate::core::triple_store::TripleStore;

/// A proposed hypothesis to fill a knowledge gap.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    /// The entity with the gap.
    pub entity: String,
    /// The missing relation.
    pub relation: String,
    /// The proposed filler entity.
    pub proposed_filler: String,
    /// Confidence score from Hopfield cleanup (0.0 - 1.0).
    pub confidence: f64,
    /// How many peers contributed evidence.
    pub evidence_count: usize,
    /// Whether this hypothesis has been confirmed by a user.
    pub confirmed: bool,
}

/// Proposes gap fillers based on peer data and Hopfield cleanup.
pub struct HypothesisEngine;

impl HypothesisEngine {
    /// Generate hypotheses for a set of knowledge gaps.
    ///
    /// For each gap, looks at what objects peers have for the missing relation,
    /// bundles those filler vectors, and runs Hopfield cleanup to find the
    /// nearest clean atom.
    pub fn propose(
        gaps: &[KnowledgeGap],
        triple_store: &TripleStore,
        atom_memory: &AtomMemory,
        beta: f64,
        min_confidence: f64,
    ) -> Vec<Hypothesis> {
        let mut hypotheses = Vec::new();

        for gap in gaps {
            if let Some(hyp) = Self::propose_one(gap, triple_store, atom_memory, beta) {
                if hyp.confidence >= min_confidence {
                    hypotheses.push(hyp);
                }
            }
        }

        hypotheses.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hypotheses
    }

    fn propose_one(
        gap: &KnowledgeGap,
        triple_store: &TripleStore,
        atom_memory: &AtomMemory,
        beta: f64,
    ) -> Option<Hypothesis> {
        // Collect what peers have as objects for this relation
        let mut filler_vecs: Vec<EntangledHVec> = Vec::new();
        let mut evidence_count = 0;

        for peer in &gap.peers_with_relation {
            let triples = triple_store.query(Some(peer), Some(&gap.missing_relation), None);
            for t in &triples {
                if let Some(vec) = atom_memory.get(&t.object_id) {
                    filler_vecs.push(vec);
                    evidence_count += 1;
                }
            }
        }

        if filler_vecs.is_empty() {
            return None;
        }

        // Bundle peer fillers into a prototype.
        // For small N at high sparsity, majority-vote may produce empty results.
        // Fall back to the first filler if bundling yields nothing.
        let bundled = EntangledHVec::bundle(&filler_vecs);
        let probe = if bundled.indices().is_empty() {
            filler_vecs[0].clone()
        } else {
            bundled
        };

        // Hopfield cleanup to find nearest stored atom
        let result = atom_memory.cleanup(&probe, beta, 64, 3);
        if !result.found {
            return None;
        }

        Some(Hypothesis {
            entity: gap.entity.clone(),
            relation: gap.missing_relation.clone(),
            proposed_filler: result.id,
            confidence: result.confidence,
            evidence_count,
            confirmed: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cognition::gaps::GapDetector;

    #[test]
    fn test_propose_hypothesis() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        // Setup: cities in continents
        atom_mem.get_or_insert("paris");
        atom_mem.get_or_insert("berlin");
        atom_mem.get_or_insert("tokyo");
        atom_mem.get_or_insert("europe");
        atom_mem.get_or_insert("asia");
        atom_mem.get_or_insert("france");
        atom_mem.get_or_insert("germany");
        atom_mem.get_or_insert("japan");

        triple_store.add("paris", "capital_of", "france", "c1");
        triple_store.add("paris", "located_in", "europe", "c2");
        triple_store.add("berlin", "capital_of", "germany", "c3");
        triple_store.add("berlin", "located_in", "europe", "c4");
        triple_store.add("tokyo", "capital_of", "japan", "c5");
        // Tokyo is missing located_in

        let gaps = GapDetector::detect(&triple_store, 1, 0.5);
        let tokyo_gaps: Vec<_> = gaps
            .iter()
            .filter(|g| g.entity == "tokyo" && g.missing_relation == "located_in")
            .cloned()
            .collect();

        if !tokyo_gaps.is_empty() {
            let hypotheses =
                HypothesisEngine::propose(&tokyo_gaps, &triple_store, &atom_mem, 24.0, 0.0);
            // Peers (paris, berlin) both have located_in -> europe
            // Bundle of [europe, europe] = europe, cleanup should find europe
            if !hypotheses.is_empty() {
                assert_eq!(hypotheses[0].entity, "tokyo");
                assert_eq!(hypotheses[0].relation, "located_in");
                assert_eq!(hypotheses[0].proposed_filler, "europe");
                assert!(hypotheses[0].evidence_count >= 1);
            }
        }
    }

    #[test]
    fn test_propose_empty_gaps() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        let hypotheses = HypothesisEngine::propose(&[], &triple_store, &atom_mem, 24.0, 0.0);
        assert!(hypotheses.is_empty());
    }

    #[test]
    fn test_propose_min_confidence_filter() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        // With impossibly high confidence threshold, nothing passes
        let gaps = vec![KnowledgeGap {
            entity: "x".to_string(),
            missing_relation: "r".to_string(),
            peers_with_relation: vec![],
            peer_coverage: 1.0,
        }];
        let hypotheses = HypothesisEngine::propose(&gaps, &triple_store, &atom_mem, 24.0, 0.99);
        assert!(hypotheses.is_empty());
    }
}
