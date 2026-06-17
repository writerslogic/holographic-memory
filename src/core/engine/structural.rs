// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::admission::{AdmissionControl, AdmissionDecision};
use crate::core::atom_memory::AtomMemory;
use crate::core::composite_memory::CompositeMemory;
use crate::core::entangled::EntangledHVec;
use crate::core::indexed_memory::hopfield_cleanup;
use crate::core::role::RoleRegistry;
use crate::core::triple_store::TripleStore;

#[derive(Clone, Debug)]
pub struct StructuralResult {
    pub entity_id: String,
    pub confidence: f64,
    pub path: StructuralPath,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StructuralPath {
    Algebraic,
    Materialized,
}

pub fn fuzzy_structural_query(
    atom_memory: &AtomMemory,
    composite_memory: &CompositeMemory,
    triple_store: &TripleStore,
    roles: &RoleRegistry,
    known: &[(&str, &EntangledHVec)],
    target_role: &str,
    admission: &AdmissionControl,
    beta: f64,
    k: usize,
    max_iter: usize,
) -> Vec<StructuralResult> {
    let query = match roles.compose(known) {
        Ok(q) => q,
        Err(_) => return Vec::new(),
    };

    let matches = composite_memory.overlap_scan(&query);
    if matches.is_empty() {
        return Vec::new();
    }

    let max_score = matches.iter().map(|m| m.1).fold(0.0f32, f32::max);
    let threshold = max_score * 0.3;
    let significant: Vec<(u32, f32)> = matches
        .into_iter()
        .filter(|&(_, score)| score > threshold)
        .collect();

    let fan_out = significant.len();
    match admission.check(fan_out) {
        AdmissionDecision::Algebraic => {
            algebraic_path(&significant, &query, composite_memory, atom_memory, roles, target_role, beta, k, max_iter)
        }
        AdmissionDecision::MaterializedLookup => {
            materialized_path(known, target_role, triple_store, composite_memory, &significant)
        }
    }
}

fn algebraic_path(
    matches: &[(u32, f32)],
    query: &EntangledHVec,
    composite_memory: &CompositeMemory,
    atom_memory: &AtomMemory,
    roles: &RoleRegistry,
    target_role: &str,
    beta: f64,
    k: usize,
    max_iter: usize,
) -> Vec<StructuralResult> {
    let mut results: Vec<StructuralResult> = Vec::new();

    for &(comp_idx, _) in matches {
        let (_, composite_vec) = match composite_memory.get_by_idx(comp_idx) {
            Some(v) => v,
            None => continue,
        };

        let residual = composite_vec.bind(query);
        let dim = residual.dim;
        let target_shift = roles.shift_for(target_role).unwrap_or(0);
        let unshifted = if target_shift == 0 {
            residual
        } else {
            residual.permute(dim - target_shift)
        };

        let cleanup = hopfield_cleanup(&unshifted, atom_memory.inner(), beta, k, max_iter);
        if cleanup.found {
            if let Some(existing) = results.iter_mut().find(|r| r.entity_id == cleanup.id) {
                existing.confidence = 1.0 - (1.0 - existing.confidence) * (1.0 - cleanup.confidence);
            } else {
                results.push(StructuralResult {
                    entity_id: cleanup.id,
                    confidence: cleanup.confidence,
                    path: StructuralPath::Algebraic,
                });
            }
        }
    }

    results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn materialized_path(
    known: &[(&str, &EntangledHVec)],
    target_role: &str,
    triple_store: &TripleStore,
    composite_memory: &CompositeMemory,
    significant: &[(u32, f32)],
) -> Vec<StructuralResult> {
    let mut results = Vec::new();

    for &(comp_idx, score) in significant {
        let (comp_id, _) = match composite_memory.get_by_idx(comp_idx) {
            Some(v) => v,
            None => continue,
        };

        let triples = triple_store.query(None, None, None);
        for t in &triples {
            if t.composite_id == comp_id {
                let entity = match target_role {
                    "subject" => &t.subject_id,
                    "relation" => &t.relation_id,
                    "object" => &t.object_id,
                    _ => continue,
                };
                if !results.iter().any(|r: &StructuralResult| r.entity_id == *entity) {
                    results.push(StructuralResult {
                        entity_id: entity.clone(),
                        confidence: score as f64 / 128.0,
                        path: StructuralPath::Materialized,
                    });
                }
            }
        }
    }

    results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structural_query_algebraic() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        let roles = RoleRegistry::new(dim);
        let admission = AdmissionControl::new(40);

        let (_, s_vec) = atom_mem.get_or_insert("paris");
        let (_, r_vec) = atom_mem.get_or_insert("capital_of");
        let (_, o_vec) = atom_mem.get_or_insert("france");

        let composite = roles.compose_triple(&s_vec, &r_vec, &o_vec);
        let comp_id = format!("triple_0");
        comp_mem.insert(comp_id.clone(), composite);
        triple_store.add("paris", "capital_of", "france", &comp_id);

        let results = fuzzy_structural_query(
            &atom_mem, &comp_mem, &triple_store, &roles,
            &[("subject", &s_vec), ("relation", &r_vec)],
            "object",
            &admission, 24.0, 64, 3,
        );

        assert!(!results.is_empty(), "Should find france");
        assert_eq!(results[0].entity_id, "france");
        assert_eq!(results[0].path, StructuralPath::Algebraic);
    }

    #[test]
    fn test_structural_query_role_inversion() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        let roles = RoleRegistry::new(dim);
        let admission = AdmissionControl::new(40);

        let (_, s_vec) = atom_mem.get_or_insert("john");
        let (_, r_vec) = atom_mem.get_or_insert("loves");
        let (_, o_vec) = atom_mem.get_or_insert("mary");

        let composite = roles.compose_triple(&s_vec, &r_vec, &o_vec);
        let comp_id = "triple_jlm".to_string();
        comp_mem.insert(comp_id.clone(), composite);
        triple_store.add("john", "loves", "mary", &comp_id);

        // Query object given subject + relation
        let r1 = fuzzy_structural_query(
            &atom_mem, &comp_mem, &triple_store, &roles,
            &[("subject", &s_vec), ("relation", &r_vec)],
            "object", &admission, 24.0, 64, 3,
        );
        assert!(!r1.is_empty());
        assert_eq!(r1[0].entity_id, "mary");

        // Query subject given relation + object
        let r2 = fuzzy_structural_query(
            &atom_mem, &comp_mem, &triple_store, &roles,
            &[("relation", &r_vec), ("object", &o_vec)],
            "subject", &admission, 24.0, 64, 3,
        );
        assert!(!r2.is_empty());
        assert_eq!(r2[0].entity_id, "john");
    }
}
