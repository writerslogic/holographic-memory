// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Distributional refinement of atom vectors.
//!
//! Atoms start as random vectors (deterministic from string hash). Over time,
//! atoms that appear in similar relational contexts (same relations, same
//! co-occurring peers) should become more similar. This module refines atom
//! vectors by blending them toward their abstraction prototypes, producing
//! self-organizing semantic embeddings without any external model.
//!
//! The refinement step runs as part of the cognition loop or is called
//! explicitly via the Governor. It requires write access to AtomMemory.

use fxhash::FxHashMap;

use crate::core::atom_memory::AtomMemory;
use crate::core::entangled::EntangledHVec;
use crate::core::triple_store::TripleStore;

/// Result of a refinement pass.
#[derive(Clone, Debug, Default)]
pub struct RefinementReport {
    /// Number of atoms that were refined.
    pub atoms_refined: usize,
    /// Number of atoms skipped (too few contexts).
    pub atoms_skipped: usize,
    /// Average number of context relations per refined atom.
    pub avg_context_depth: f64,
}

/// Configuration for distributional refinement.
#[derive(Clone, Debug)]
pub struct RefinerConfig {
    /// Blending factor: 0.0 = keep original, 1.0 = fully replace with context.
    /// Default 0.15 — conservative blending to avoid catastrophic drift.
    pub alpha: f64,
    /// Minimum number of distinct relations an atom must participate in
    /// before refinement applies. Atoms with fewer contexts keep their
    /// original random vectors.
    pub min_context_relations: usize,
    /// Minimum number of peer atoms sharing a relation for that relation's
    /// context to contribute. Filters out singleton relations.
    pub min_peers_per_relation: usize,
}

impl Default for RefinerConfig {
    fn default() -> Self {
        Self {
            alpha: 0.15,
            min_context_relations: 2,
            min_peers_per_relation: 2,
        }
    }
}

/// Refines atom vectors by blending them toward distributional context vectors.
///
/// For each atom A that participates in relations {R1, R2, ...}:
/// 1. For each relation Ri, collect all peer atoms that also participate in Ri
/// 2. Build a context vector by frequency-counting indices across peer atoms
/// 3. Blend: refined_A = merge(original_A, context_vector, alpha)
///
/// The merge preserves sparsity by taking the top-k indices from the
/// weighted union of original and context indices.
pub struct DistributionalRefiner;

impl DistributionalRefiner {
    /// Run one refinement pass over all atoms.
    ///
    /// This modifies atom vectors in-place. Each atom's vector is blended
    /// toward the centroid of its relational peers. Requires write access
    /// to AtomMemory (call via Governor, not from the read-only cognition loop).
    pub fn refine(
        atom_memory: &AtomMemory,
        triple_store: &TripleStore,
        config: &RefinerConfig,
    ) -> RefinementReport {
        let snapshot = triple_store.snapshot();
        if snapshot.is_empty() {
            return RefinementReport::default();
        }

        // Build: atom -> set of relations it participates in (as subject)
        let mut atom_relations: FxHashMap<String, Vec<String>> = FxHashMap::default();
        for t in &snapshot {
            atom_relations
                .entry(t.subject_id.clone())
                .or_default()
                .push(t.relation_id.clone());
        }
        // Deduplicate
        for rels in atom_relations.values_mut() {
            rels.sort();
            rels.dedup();
        }

        // Build: relation -> list of subject atoms
        let mut relation_subjects: FxHashMap<String, Vec<String>> = FxHashMap::default();
        for t in &snapshot {
            relation_subjects
                .entry(t.relation_id.clone())
                .or_default()
                .push(t.subject_id.clone());
        }
        for subjects in relation_subjects.values_mut() {
            subjects.sort();
            subjects.dedup();
        }

        let mut report = RefinementReport::default();
        let mut total_context_depth: usize = 0;

        // For each atom, compute its context vector and blend
        for (atom_id, relations) in &atom_relations {
            if relations.len() < config.min_context_relations {
                report.atoms_skipped += 1;
                continue;
            }

            let original = match atom_memory.get(atom_id) {
                Some(v) => v,
                None => continue,
            };

            // Collect peer vectors across all relations this atom participates in
            let mut peer_index_freq: FxHashMap<u32, f64> = FxHashMap::default();
            let mut contributing_relations = 0usize;

            for rel in relations {
                let peers = match relation_subjects.get(rel) {
                    Some(p) => p,
                    None => continue,
                };
                // Exclude self, check minimum peer count
                let other_peers: Vec<&String> =
                    peers.iter().filter(|p| *p != atom_id).collect();
                if other_peers.len() < config.min_peers_per_relation {
                    continue;
                }
                contributing_relations += 1;

                // Weight each peer's indices by 1/num_peers for this relation
                let weight = 1.0 / other_peers.len() as f64;
                for peer_id in &other_peers {
                    if let Some(peer_vec) = atom_memory.get(peer_id) {
                        for &idx in peer_vec.indices() {
                            *peer_index_freq.entry(idx).or_insert(0.0) += weight;
                        }
                    }
                }
            }

            if contributing_relations == 0 {
                report.atoms_skipped += 1;
                continue;
            }

            total_context_depth += contributing_relations;

            // Build blended vector via slot allocation.
            // Reserve (1 - alpha) fraction of slots for original indices,
            // and alpha fraction for the highest-frequency context indices.
            // This guarantees context indices actually appear in the result.
            let dim = original.dim;
            let target_count = (dim / 256).max(1);
            let ctx_slots = ((target_count as f64) * config.alpha).round() as usize;
            let orig_slots = target_count.saturating_sub(ctx_slots);

            // Top context indices by frequency (excluding those already in original)
            let orig_set: fxhash::FxHashSet<u32> =
                original.indices().iter().copied().collect();
            let mut ctx_scored: Vec<(u32, f64)> = peer_index_freq
                .iter()
                .filter(|(idx, _)| !orig_set.contains(idx))
                .map(|(&idx, &freq)| (idx, freq))
                .collect();
            ctx_scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ctx_scored.truncate(ctx_slots);

            // Keep top original indices (by posting frequency if available,
            // otherwise just first orig_slots)
            let mut orig_indices: Vec<u32> = original.indices().to_vec();
            orig_indices.truncate(orig_slots);

            // Merge
            let mut new_indices: Vec<u32> = orig_indices;
            new_indices.extend(ctx_scored.iter().map(|(idx, _)| *idx));
            new_indices.sort_unstable();
            new_indices.dedup();
            new_indices.truncate(target_count);

            let refined = EntangledHVec::from_indices(new_indices, dim);

            // Only update if the refined vector is meaningfully different
            if original.similarity(&refined) < 0.999 {
                atom_memory.delete(atom_id);
                atom_memory.load_atom(atom_id.clone(), refined);
                report.atoms_refined += 1;
            } else {
                report.atoms_skipped += 1;
            }
        }

        if report.atoms_refined > 0 {
            report.avg_context_depth = total_context_depth as f64 / report.atoms_refined as f64;
            atom_memory.rebuild_indices();
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_converges_cooccurring_atoms() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        // Create atoms
        atom_mem.get_or_insert("paris");
        atom_mem.get_or_insert("berlin");
        atom_mem.get_or_insert("tokyo");
        atom_mem.get_or_insert("france");
        atom_mem.get_or_insert("germany");
        atom_mem.get_or_insert("japan");
        atom_mem.get_or_insert("europe");

        // paris and berlin share two relations; tokyo shares one
        triple_store.add("paris", "capital_of", "france", "c1");
        triple_store.add("berlin", "capital_of", "germany", "c2");
        triple_store.add("tokyo", "capital_of", "japan", "c3");
        triple_store.add("paris", "located_in", "europe", "c4");
        triple_store.add("berlin", "located_in", "europe", "c5");

        let paris_before = atom_mem.get("paris").unwrap();
        let berlin_before = atom_mem.get("berlin").unwrap();
        let sim_before = paris_before.similarity(&berlin_before);

        // Run refinement
        let config = RefinerConfig {
            alpha: 0.3,
            min_context_relations: 2,
            min_peers_per_relation: 2,
        };
        let report = DistributionalRefiner::refine(&atom_mem, &triple_store, &config);

        assert!(report.atoms_refined > 0, "Should refine at least some atoms");

        let paris_after = atom_mem.get("paris").unwrap();
        let berlin_after = atom_mem.get("berlin").unwrap();
        let sim_after = paris_after.similarity(&berlin_after);

        assert!(
            sim_after > sim_before,
            "paris and berlin should become more similar after refinement: before={:.4}, after={:.4}",
            sim_before,
            sim_after
        );
    }

    #[test]
    fn test_refine_skips_low_context_atoms() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        atom_mem.get_or_insert("lonely");
        atom_mem.get_or_insert("x");
        triple_store.add("lonely", "r", "x", "c1");

        let config = RefinerConfig {
            min_context_relations: 2,
            ..Default::default()
        };
        let report = DistributionalRefiner::refine(&atom_mem, &triple_store, &config);
        assert_eq!(report.atoms_refined, 0);
        assert!(report.atoms_skipped > 0);
    }

    #[test]
    fn test_refine_empty_store() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        let config = RefinerConfig::default();
        let report = DistributionalRefiner::refine(&atom_mem, &triple_store, &config);
        assert_eq!(report.atoms_refined, 0);
        assert_eq!(report.atoms_skipped, 0);
    }

    #[test]
    fn test_refine_preserves_vector_sparsity() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        for city in &["a", "b", "c", "d", "e"] {
            atom_mem.get_or_insert(city);
        }
        for country in &["x", "y", "z", "w", "v"] {
            atom_mem.get_or_insert(country);
        }
        atom_mem.get_or_insert("continent");

        // All share two relations
        for (city, country) in [("a", "x"), ("b", "y"), ("c", "z"), ("d", "w"), ("e", "v")] {
            triple_store.add(city, "capital_of", country, &format!("c_{}", city));
            triple_store.add(city, "in", "continent", &format!("l_{}", city));
        }

        let config = RefinerConfig {
            alpha: 0.3,
            min_context_relations: 2,
            min_peers_per_relation: 2,
        };
        let report = DistributionalRefiner::refine(&atom_mem, &triple_store, &config);

        // Check sparsity is maintained
        let target = dim / 256;
        for city in &["a", "b", "c", "d", "e"] {
            let vec = atom_mem.get(city).unwrap();
            assert!(
                vec.indices().len() <= target + 1,
                "Refined vector for {} has {} indices, expected <= {}",
                city,
                vec.indices().len(),
                target + 1
            );
        }
        assert!(report.atoms_refined >= 3);
    }

    #[test]
    fn test_multiple_refinement_rounds() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        atom_mem.get_or_insert("a");
        atom_mem.get_or_insert("b");
        atom_mem.get_or_insert("c");
        atom_mem.get_or_insert("x");
        atom_mem.get_or_insert("y");

        triple_store.add("a", "r1", "x", "c1");
        triple_store.add("b", "r1", "y", "c2");
        triple_store.add("c", "r1", "x", "c3");
        triple_store.add("a", "r2", "y", "c4");
        triple_store.add("b", "r2", "x", "c5");
        triple_store.add("c", "r2", "y", "c6");

        let config = RefinerConfig {
            alpha: 0.2,
            min_context_relations: 2,
            min_peers_per_relation: 2,
        };

        let sim_before = atom_mem.get("a").unwrap().similarity(&atom_mem.get("b").unwrap());

        // Run 3 rounds of refinement
        for _ in 0..3 {
            DistributionalRefiner::refine(&atom_mem, &triple_store, &config);
        }

        let sim_after = atom_mem.get("a").unwrap().similarity(&atom_mem.get("b").unwrap());
        assert!(
            sim_after > sim_before,
            "Multiple rounds should increase similarity: before={:.4}, after={:.4}",
            sim_before,
            sim_after
        );
    }
}
