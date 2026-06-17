// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::admission::{AdmissionControl, AdmissionDecision};
use crate::core::atom_memory::AtomMemory;
use crate::core::composite_memory::CompositeMemory;
use crate::core::entangled::EntangledHVec;
use crate::core::indexed_memory::hopfield_cleanup;
use crate::core::role::RoleRegistry;
use crate::core::triple_store::TripleStore;

pub struct MeaningContext<'a> {
    pub atom_memory: &'a AtomMemory,
    pub composite_memory: &'a CompositeMemory,
    pub triple_store: &'a TripleStore,
    pub roles: &'a RoleRegistry,
    pub admission: &'a AdmissionControl,
    pub beta: f64,
    pub k: usize,
    pub max_iter: usize,
}

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
    ctx: &MeaningContext<'_>,
    known: &[(&str, &EntangledHVec)],
    target_role: &str,
) -> Vec<StructuralResult> {
    let query = match ctx.roles.compose(known) {
        Ok(q) => q,
        Err(_) => return Vec::new(),
    };

    let matches = ctx.composite_memory.overlap_scan(&query);
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
    match ctx.admission.check(fan_out) {
        AdmissionDecision::Algebraic => algebraic_path(&significant, &query, ctx, target_role),
        AdmissionDecision::MaterializedLookup => materialized_path(
            target_role,
            ctx.triple_store,
            ctx.composite_memory,
            &significant,
        ),
    }
}

fn algebraic_path(
    matches: &[(u32, f32)],
    query: &EntangledHVec,
    ctx: &MeaningContext<'_>,
    target_role: &str,
) -> Vec<StructuralResult> {
    let mut results: Vec<StructuralResult> = Vec::new();

    for &(comp_idx, _) in matches {
        let (_, composite_vec) = match ctx.composite_memory.get_by_idx(comp_idx) {
            Some(v) => v,
            None => continue,
        };

        let residual = composite_vec.bind(query);
        let dim = residual.dim;
        let target_shift = ctx.roles.shift_for(target_role).unwrap_or(0);
        let unshifted = if target_shift == 0 {
            residual
        } else {
            residual.permute(dim - target_shift)
        };

        let cleanup = hopfield_cleanup(
            &unshifted,
            ctx.atom_memory.inner(),
            ctx.beta,
            ctx.k,
            ctx.max_iter,
        );
        if cleanup.found {
            if let Some(existing) = results.iter_mut().find(|r| r.entity_id == cleanup.id) {
                existing.confidence =
                    1.0 - (1.0 - existing.confidence) * (1.0 - cleanup.confidence);
            } else {
                results.push(StructuralResult {
                    entity_id: cleanup.id,
                    confidence: cleanup.confidence,
                    path: StructuralPath::Algebraic,
                });
            }
        }
    }

    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

fn materialized_path(
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

        for t in triple_store.by_composite_id(&comp_id) {
            let entity = match target_role {
                "subject" => &t.subject_id,
                "relation" => &t.relation_id,
                "object" => &t.object_id,
                _ => continue,
            };
            if !results
                .iter()
                .any(|r: &StructuralResult| r.entity_id == *entity)
            {
                results.push(StructuralResult {
                    entity_id: entity.clone(),
                    confidence: score as f64 / 128.0,
                    path: StructuralPath::Materialized,
                });
            }
        }
    }

    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx<'a>(
        atom_mem: &'a AtomMemory,
        comp_mem: &'a CompositeMemory,
        triple_store: &'a TripleStore,
        roles: &'a RoleRegistry,
        admission: &'a AdmissionControl,
    ) -> MeaningContext<'a> {
        MeaningContext {
            atom_memory: atom_mem,
            composite_memory: comp_mem,
            triple_store,
            roles,
            admission,
            beta: 24.0,
            k: 64,
            max_iter: 3,
        }
    }

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
        let comp_id = "triple_0".to_string();
        comp_mem.insert(comp_id.clone(), composite);
        triple_store.add("paris", "capital_of", "france", &comp_id);

        let ctx = make_ctx(&atom_mem, &comp_mem, &triple_store, &roles, &admission);
        let results =
            fuzzy_structural_query(&ctx, &[("subject", &s_vec), ("relation", &r_vec)], "object");

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

        let ctx = make_ctx(&atom_mem, &comp_mem, &triple_store, &roles, &admission);

        let r1 =
            fuzzy_structural_query(&ctx, &[("subject", &s_vec), ("relation", &r_vec)], "object");
        assert!(!r1.is_empty());
        assert_eq!(r1[0].entity_id, "mary");

        let r2 =
            fuzzy_structural_query(&ctx, &[("relation", &r_vec), ("object", &o_vec)], "subject");
        assert!(!r2.is_empty());
        assert_eq!(r2[0].entity_id, "john");
    }
}
