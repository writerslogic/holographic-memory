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
    let Ok(query) = ctx.roles.compose(known) else {
        return Vec::new();
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
        let Some((_, composite_vec)) = ctx.composite_memory.get_by_idx(comp_idx) else {
            continue;
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
        let Some((comp_id, _)) = composite_memory.get_by_idx(comp_idx) else {
            continue;
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
pub(super) mod tests {
    use super::*;

    /// Empty meaning components with the engine's query parameters.
    pub(crate) struct Fixture {
        pub(crate) atoms: AtomMemory,
        pub(crate) triples: TripleStore,
        composites: CompositeMemory,
        roles: RoleRegistry,
        admission: AdmissionControl,
    }

    impl Fixture {
        pub(crate) fn new() -> Self {
            const DIM: usize = 16384;
            Self {
                atoms: AtomMemory::new(DIM, 3.0),
                triples: TripleStore::new(),
                composites: CompositeMemory::new(DIM, 3.0),
                roles: RoleRegistry::new(DIM),
                admission: AdmissionControl::new(40),
            }
        }

        /// Stores `subject relation object` as a role-bound composite named `id` and as a
        /// symbolic triple; returns the three atom vectors.
        pub(crate) fn bind(
            &self,
            subject: &str,
            relation: &str,
            object: &str,
            id: &str,
        ) -> [EntangledHVec; 3] {
            let [(_, s), (_, r), (_, o)] =
                [subject, relation, object].map(|atom| self.atoms.get_or_insert(atom));
            self.composites
                .insert(id.to_string(), self.roles.compose_triple(&s, &r, &o));
            self.triples.add(subject, relation, object, id);
            [s, r, o]
        }

        pub(crate) fn context(&self) -> MeaningContext<'_> {
            MeaningContext {
                atom_memory: &self.atoms,
                composite_memory: &self.composites,
                triple_store: &self.triples,
                roles: &self.roles,
                admission: &self.admission,
                beta: 24.0,
                k: 64,
                max_iter: 3,
            }
        }
    }

    #[test]
    fn algebraic_unbinding_recovers_either_unknown_role() {
        let fixture = Fixture::new();
        let [subject, relation, object] = fixture.bind("john", "loves", "mary", "triple_jlm");
        let context = fixture.context();
        for (target, known, expected) in [
            (
                "object",
                [("subject", &subject), ("relation", &relation)],
                "mary",
            ),
            (
                "subject",
                [("relation", &relation), ("object", &object)],
                "john",
            ),
        ] {
            let results = fuzzy_structural_query(&context, &known, target);
            assert_eq!(
                results
                    .first()
                    .map(|r| (r.entity_id.as_str(), r.path.clone())),
                Some((expected, StructuralPath::Algebraic)),
                "{target}"
            );
        }
    }
}
