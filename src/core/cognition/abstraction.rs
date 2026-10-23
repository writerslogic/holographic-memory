// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Concept invention via abstraction.
//!
//! When N entities share the same relation pattern, bundles their atom
//! vectors (using `EntangledHVec::bundle`) to create a prototype/category atom.

use crate::core::hash::FastMap;

use crate::core::atom_memory::AtomMemory;
use crate::core::entangled::EntangledHVec;
use crate::core::triple_store::TripleStore;

/// A discovered abstraction: a set of entities that share a common pattern.
#[derive(Clone, Debug)]
pub struct Abstraction {
    /// Auto-generated name for the concept (e.g., "concept:capital_of:subject").
    pub name: String,
    /// The relation pattern shared by these entities.
    pub relation: String,
    /// The role (subject or object) that the members fill.
    pub role: AbstractionRole,
    /// The member entities.
    pub members: Vec<String>,
    /// The bundled prototype vector.
    pub prototype: EntangledHVec,
}

/// Which role the abstracted entities fill in the pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbstractionRole {
    Subject,
    Object,
}

/// Creates concept atoms by bundling entities that share structural patterns.
pub struct AbstractionEngine;

impl AbstractionEngine {
    /// Find groups of entities that share a relation pattern and bundle them.
    ///
    /// `min_members`: minimum number of entities sharing a relation to form a concept.
    /// Returns abstractions for both subject-side and object-side groupings.
    pub fn discover(
        triple_store: &TripleStore,
        atom_memory: &AtomMemory,
        min_members: usize,
    ) -> Vec<Abstraction> {
        let mut results = Vec::new();
        for relation in triple_store.relation_ids() {
            let mut subjects = crate::core::hash::FastSet::default();
            let mut objects = crate::core::hash::FastSet::default();
            triple_store.visit(None, Some(&relation), None, |t| {
                subjects.insert(t.subject_id.clone());
                objects.insert(t.object_id.clone());
                true
            });
            for (role, members) in [
                (AbstractionRole::Subject, subjects),
                (AbstractionRole::Object, objects),
            ] {
                if members.len() < min_members {
                    continue;
                }
                let mut members: Vec<_> = members.into_iter().collect();
                members.sort();
                if let Some(prototype) = Self::bundle_members(&members, atom_memory) {
                    let suffix = if role == AbstractionRole::Subject {
                        "subject"
                    } else {
                        "object"
                    };
                    results.push(Abstraction {
                        name: format!("concept:{relation}:{suffix}"),
                        relation: relation.clone(),
                        role,
                        members,
                        prototype,
                    });
                }
            }
        }

        results
    }

    /// Build a prototype vector from member atoms.
    ///
    /// For large groups (>= 10), uses majority-vote bundling.
    /// For small groups, uses weighted index frequency counting with top-k
    /// selection, since majority-vote at high sparsity (64/16384) produces
    /// empty results when members are independent.
    fn bundle_members(members: &[String], atom_memory: &AtomMemory) -> Option<EntangledHVec> {
        let mut freq: FastMap<u32, u32> = FastMap::default();
        let mut dim = None;
        let mut count = 0u32;
        for member in members {
            if let Some(vector) = atom_memory.get(member) {
                dim.get_or_insert(vector.dim);
                count += 1;
                for &index in vector.indices() {
                    *freq.entry(index).or_default() += 1;
                }
            }
        }
        let dim = dim?;
        if count >= 10 {
            let mut selected: Vec<_> = freq
                .into_iter()
                .filter(|(_, n)| *n >= count.div_ceil(2))
                .collect();
            selected.sort_unstable_by_key(|&(index, _)| index);
            return Some(EntangledHVec::from_selected(
                selected,
                dim,
                (dim / 256).max(1),
            ));
        }
        let target_count = (dim / 256).max(1);
        let mut scored: Vec<(u32, u32)> = freq.into_iter().collect();
        if scored.len() > target_count {
            scored
                .select_nth_unstable_by(target_count - 1, |a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            scored.truncate(target_count);
        }
        let mut indices: Vec<u32> = scored.into_iter().map(|(idx, _)| idx).collect();
        indices.sort_unstable();
        Some(EntangledHVec::from_indices(indices, dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_prototypes_preserve_majority_selection() {
        let atoms = AtomMemory::new(256, 3.0);
        let mut members = Vec::new();
        for i in 0..11 {
            let id = i.to_string();
            atoms.insert_with_vec(&id, &EntangledHVec::from_indices(vec![1, 2, 3, 4 + i], 256));
            members.push(id);
            if members.len() >= 10 {
                let vectors: Vec<_> = members.iter().map(|id| atoms.get(id).unwrap()).collect();
                assert_eq!(
                    AbstractionEngine::bundle_members(&members, &atoms)
                        .unwrap()
                        .indices(),
                    EntangledHVec::bundle(&vectors).indices()
                );
            }
        }
    }

    #[test]
    fn test_discover_abstractions() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        // Insert atoms
        atom_mem.get_or_insert("paris");
        atom_mem.get_or_insert("berlin");
        atom_mem.get_or_insert("tokyo");
        atom_mem.get_or_insert("france");
        atom_mem.get_or_insert("germany");
        atom_mem.get_or_insert("japan");

        // All three are capitals
        triple_store.add("paris", "capital_of", "france", "c1");
        triple_store.add("berlin", "capital_of", "germany", "c2");
        triple_store.add("tokyo", "capital_of", "japan", "c3");

        let abstractions = AbstractionEngine::discover(&triple_store, &atom_mem, 3);
        assert!(!abstractions.is_empty());

        // Should have a subject-side concept for capital_of (cities)
        let cities = abstractions
            .iter()
            .find(|a| a.relation == "capital_of" && a.role == AbstractionRole::Subject);
        assert!(cities.is_some());
        let cities = cities.unwrap();
        assert_eq!(cities.members.len(), 3);
        assert!(
            !cities.prototype.indices().is_empty(),
            "Prototype should have indices via frequency counting"
        );
    }

    #[test]
    fn test_discover_below_threshold() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        atom_mem.get_or_insert("a");
        atom_mem.get_or_insert("b");
        triple_store.add("a", "r", "b", "c1");

        let abstractions = AbstractionEngine::discover(&triple_store, &atom_mem, 5);
        assert!(abstractions.is_empty());
    }

    #[test]
    fn test_discover_empty_store() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        let abstractions = AbstractionEngine::discover(&triple_store, &atom_mem, 1);
        assert!(abstractions.is_empty());
    }
}
