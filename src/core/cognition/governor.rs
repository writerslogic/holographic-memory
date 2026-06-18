// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Memory governor: consolidation, forgetting, and maintenance.
//!
//! Consolidates near-duplicate composites, forgets stale entries,
//! and triggers IDF refresh. All write operations require explicit
//! invocation; nothing runs automatically.

use crate::core::atom_memory::AtomMemory;
use crate::core::cognition::refiner::{DistributionalRefiner, RefinementReport, RefinerConfig};
use crate::core::composite_memory::CompositeMemory;
use crate::core::triple_store::TripleStore;

/// Result of a governance cycle.
#[derive(Clone, Debug, Default)]
pub struct GovernanceReport {
    /// Number of near-duplicate composites merged.
    pub composites_merged: usize,
    /// Number of stale composites forgotten.
    pub composites_forgotten: usize,
    /// Number of stale atoms forgotten.
    pub atoms_forgotten: usize,
    /// Whether IDF weights were refreshed.
    pub idf_refreshed: bool,
    /// Distributional refinement results.
    pub refinement: RefinementReport,
}

/// Configuration for governance operations.
#[derive(Clone, Debug)]
pub struct GovernorConfig {
    /// Similarity threshold above which two composites are considered duplicates.
    pub duplicate_threshold: f64,
    /// Maximum number of composites to scan per consolidation pass.
    pub max_scan_size: usize,
    /// Atoms referenced by zero live triples are candidates for forgetting.
    pub forget_unreferenced_atoms: bool,
    /// Enable distributional refinement of atom vectors.
    pub refine_atoms: bool,
    /// Configuration for distributional refinement.
    pub refiner: RefinerConfig,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            duplicate_threshold: 0.95,
            max_scan_size: 1000,
            forget_unreferenced_atoms: false,
            refine_atoms: false,
            refiner: RefinerConfig::default(),
        }
    }
}

/// Manages memory health: deduplication, forgetting, and index refresh.
/// All operations are explicit (called by user or by promote_insights).
pub struct MemoryGovernor;

impl MemoryGovernor {
    /// Run a full governance cycle: consolidate, forget, refresh.
    pub fn govern(
        atom_memory: &AtomMemory,
        composite_memory: &CompositeMemory,
        triple_store: &TripleStore,
        config: &GovernorConfig,
    ) -> GovernanceReport {
        let composites_merged = Self::consolidate_composites(composite_memory, config);
        let composites_forgotten = Self::forget_stale_composites(composite_memory, triple_store);

        let atoms_forgotten = if config.forget_unreferenced_atoms {
            Self::forget_unreferenced_atoms(atom_memory, triple_store)
        } else {
            0
        };

        let refinement = if config.refine_atoms {
            DistributionalRefiner::refine(atom_memory, triple_store, &config.refiner)
        } else {
            RefinementReport::default()
        };

        Self::refresh_indices(atom_memory, composite_memory);

        GovernanceReport {
            composites_merged,
            composites_forgotten,
            atoms_forgotten,
            idf_refreshed: true,
            refinement,
        }
    }

    /// Find near-duplicate composites and remove the lower-indexed duplicate.
    /// Returns the number of composites merged (deleted).
    pub fn consolidate_composites(
        composite_memory: &CompositeMemory,
        config: &GovernorConfig,
    ) -> usize {
        let all = composite_memory.inner().all_vectors();
        let scan_limit = all.len().min(config.max_scan_size);
        let mut merged = 0;
        let mut deleted: fxhash::FxHashSet<String> = fxhash::FxHashSet::default();

        for i in 0..scan_limit {
            let (_, ref id_i, ref vec_i) = all[i];
            if deleted.contains(id_i) {
                continue;
            }
            for entry in all.iter().take(scan_limit).skip(i + 1) {
                let (_, ref id_j, ref vec_j) = *entry;
                if deleted.contains(id_j) {
                    continue;
                }
                if vec_i.similarity(vec_j) >= config.duplicate_threshold {
                    composite_memory.delete(id_j);
                    deleted.insert(id_j.clone());
                    merged += 1;
                }
            }
        }

        merged
    }

    /// Remove composites that have no corresponding live triple.
    /// Returns the number of composites forgotten.
    pub fn forget_stale_composites(
        composite_memory: &CompositeMemory,
        triple_store: &TripleStore,
    ) -> usize {
        let all = composite_memory.inner().all_vectors();
        let mut forgotten = 0;

        for (_, id, _) in &all {
            let triples = triple_store.by_composite_id(id);
            if triples.is_empty() {
                composite_memory.delete(id);
                forgotten += 1;
            }
        }

        forgotten
    }

    /// Remove atoms not referenced by any live triple.
    /// Returns the number of atoms forgotten.
    pub fn forget_unreferenced_atoms(
        atom_memory: &AtomMemory,
        triple_store: &TripleStore,
    ) -> usize {
        let snapshot = triple_store.snapshot();
        let mut referenced = fxhash::FxHashSet::default();
        for t in &snapshot {
            referenced.insert(t.subject_id.clone());
            referenced.insert(t.relation_id.clone());
            referenced.insert(t.object_id.clone());
        }

        let all_atoms = atom_memory.inner().all_vectors();
        let mut forgotten = 0;

        for (_, id, _) in &all_atoms {
            if !referenced.contains(id) {
                atom_memory.delete(id);
                forgotten += 1;
            }
        }

        forgotten
    }

    /// Rebuild posting lists and IDF weights for both memories.
    pub fn refresh_indices(atom_memory: &AtomMemory, composite_memory: &CompositeMemory) {
        atom_memory.rebuild_indices();
        composite_memory.rebuild_indices();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::entangled::EntangledHVec;

    #[test]
    fn test_consolidate_duplicates() {
        let dim = 16384;
        let comp_mem = CompositeMemory::new(dim, 3.0);

        let v1 = EntangledHVec::new_deterministic(dim, 42);
        comp_mem.insert("c1".to_string(), v1.clone());
        comp_mem.insert("c2".to_string(), v1.clone()); // exact duplicate

        let config = GovernorConfig {
            duplicate_threshold: 0.95,
            max_scan_size: 100,
            ..Default::default()
        };

        let merged = MemoryGovernor::consolidate_composites(&comp_mem, &config);
        assert!(merged >= 1);
    }

    #[test]
    fn test_forget_stale_composites() {
        let dim = 16384;
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        let v1 = EntangledHVec::new_deterministic(dim, 1);
        let v2 = EntangledHVec::new_deterministic(dim, 2);
        comp_mem.insert("c_live".to_string(), v1);
        comp_mem.insert("c_stale".to_string(), v2);

        triple_store.add("a", "r", "b", "c_live");
        // c_stale has no triple

        let forgotten = MemoryGovernor::forget_stale_composites(&comp_mem, &triple_store);
        assert_eq!(forgotten, 1);
    }

    #[test]
    fn test_forget_unreferenced_atoms() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        atom_mem.get_or_insert("used_atom");
        atom_mem.get_or_insert("orphan_atom");
        triple_store.add("used_atom", "r", "x", "c1");

        let forgotten = MemoryGovernor::forget_unreferenced_atoms(&atom_mem, &triple_store);
        // orphan_atom is not referenced, but "r" and "x" also aren't atoms
        // Only orphan_atom should be forgotten
        assert!(forgotten >= 1);
    }

    #[test]
    fn test_full_govern_cycle() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        atom_mem.get_or_insert("a");
        let v = EntangledHVec::new_deterministic(dim, 1);
        comp_mem.insert("c1".to_string(), v);
        triple_store.add("a", "r", "b", "c1");

        let config = GovernorConfig::default();
        let report = MemoryGovernor::govern(&atom_mem, &comp_mem, &triple_store, &config);
        assert!(report.idf_refreshed);
    }

    #[test]
    fn test_govern_empty() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        let config = GovernorConfig::default();
        let report = MemoryGovernor::govern(&atom_mem, &comp_mem, &triple_store, &config);
        assert_eq!(report.composites_merged, 0);
        assert_eq!(report.composites_forgotten, 0);
        assert!(report.idf_refreshed);
    }
}
