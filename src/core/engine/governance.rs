// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};

use super::{HmsCore, Mutation};
use crate::core::atom_memory::AtomMemory;
use crate::core::cognition::governor::{
    GovernanceConflict, GovernanceReport, GovernorConfig, MemoryGovernor,
};
use crate::core::composite_memory::CompositeMemory;
use crate::core::entangled::EntangledHVec;
use crate::core::query_limits::{QueryBudget, QueryLimits};
use crate::core::triple_store::{TripleRecord, TripleStore};

const MAX_OPERATIONS: usize = 4096;
const MAX_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;

struct Snapshot {
    revision: u64,
    atoms: BTreeMap<String, EntangledHVec>,
    composites: BTreeMap<String, EntangledHVec>,
    triples: Vec<TripleRecord>,
    mirrored: BTreeSet<String>,
}

struct Plan {
    revision: u64,
    mutations: Vec<Mutation>,
    payload: Vec<u8>,
    report: GovernanceReport,
}

fn same(left: &EntangledHVec, right: &EntangledHVec) -> bool {
    left.dim == right.dim && left.indices() == right.indices()
}

fn append(
    mutations: &mut Vec<Mutation>,
    budget: &mut QueryBudget,
    mutation: Mutation,
) -> Result<()> {
    ensure!(
        mutations.len() < MAX_OPERATIONS,
        "governance exceeds the 4096-operation atomic transaction limit"
    );
    budget.frontier(mutations.len() + 1)?;
    mutations.push(mutation);
    Ok(())
}

impl HmsCore {
    /// Plans governance on an isolated snapshot, then commits one bounded transaction if
    /// its revision is still current. A conflict changes no governance state and may be retried.
    pub fn govern_memory_with_limits(&self, limits: QueryLimits) -> Result<GovernanceReport> {
        self.govern_with_hook(limits, || {})
    }

    fn govern_with_hook(
        &self,
        limits: QueryLimits,
        after_plan: impl FnOnce(),
    ) -> Result<GovernanceReport> {
        if !self.meaning_enabled() {
            return Ok(GovernanceReport::default());
        }
        ensure!(
            self.schema >= crate::core::format::MEANING_MAINTENANCE_SCHEMA_VERSION,
            "durable governance requires store schema 5; use hms-format upgrade to create an upgraded store"
        );
        let mut budget = QueryBudget::new(limits);
        // Two shadow indexes, rebuild overlap and IDF scratch scale with the ambient dimension.
        budget.reserve(self.dimensions.saturating_mul(96))?;
        budget.work(self.dimensions.saturating_mul(12))?;
        let snapshot = self.governance_snapshot(&mut budget)?;
        let plan = self.governance_plan(snapshot, &mut budget)?;
        after_plan();
        ensure!(
            self.commit_serialized_if_revision(&plan.mutations, &plan.payload, plan.revision)?,
            GovernanceConflict
        );
        Ok(plan.report)
    }

    fn governance_snapshot(&self, budget: &mut QueryBudget) -> Result<Snapshot> {
        let _transaction = self.mutation_gate.read();
        let revision = self.revision();
        let atoms = self
            .atom_memory
            .as_ref()
            .context("meaning atom store is missing")?
            .inner()
            .bounded_snapshot(budget)?
            .into_iter()
            .collect();
        let composites: BTreeMap<_, _> = self
            .composite_memory
            .as_ref()
            .context("meaning composite store is missing")?
            .inner()
            .bounded_snapshot(budget)?
            .into_iter()
            .collect();
        let triples = self
            .triple_store
            .as_ref()
            .context("meaning triple store is missing")?
            .bounded_snapshot(budget)?;
        let mut mirrored = BTreeSet::new();
        self.shards.read().try_for_each_shard(|shard| {
            for (id, vector) in shard.vectors.read().iter() {
                budget.work(1)?;
                if composites
                    .get(id)
                    .is_some_and(|composite| same(composite, vector))
                {
                    budget.reserve(id.len().saturating_mul(2).saturating_add(64))?;
                    mirrored.insert(id.clone());
                }
            }
            Ok(())
        })?;
        Ok(Snapshot {
            revision,
            atoms,
            composites,
            triples,
            mirrored,
        })
    }

    fn governance_plan(&self, snapshot: Snapshot, budget: &mut QueryBudget) -> Result<Plan> {
        let config = GovernorConfig {
            duplicate_threshold: self.config.cognition.governor_duplicate_threshold,
            max_scan_size: self.config.cognition.governor_max_scan_size,
            forget_unreferenced_atoms: self.config.cognition.governor_forget_unreferenced,
            refine_atoms: self.config.cognition.refine_atoms,
            ..Default::default()
        };
        let scan = config.max_scan_size.min(snapshot.composites.len());
        let owned: BTreeSet<_> = snapshot
            .triples
            .iter()
            .map(|triple| triple.composite_id.as_str())
            .collect();
        let eligible = snapshot
            .composites
            .keys()
            .take(scan)
            .filter(|id| !owned.contains(id.as_str()))
            .count();
        let composite_width = snapshot
            .composites
            .values()
            .take(scan)
            .map(|vector| vector.indices().len())
            .max()
            .unwrap_or(0)
            .max(1);
        let activations = snapshot
            .atoms
            .values()
            .chain(snapshot.composites.values())
            .fold(0usize, |total, vector| {
                total.saturating_add(vector.indices().len())
            });
        let refinement = if config.refine_atoms {
            let width = snapshot
                .atoms
                .values()
                .map(|vector| vector.indices().len())
                .max()
                .unwrap_or(0)
                .max((self.dimensions / 256).max(1));
            snapshot
                .atoms
                .len()
                .saturating_mul(snapshot.triples.len())
                .saturating_mul(width)
                .saturating_mul(3)
        } else {
            0
        };
        budget.work(
            eligible
                .saturating_mul(scan)
                .saturating_mul(composite_width)
                .saturating_mul(3)
                .saturating_add(activations.saturating_mul(4))
                .saturating_add(snapshot.atoms.len())
                .saturating_add(snapshot.composites.len())
                .saturating_add(snapshot.triples.len().saturating_mul(4))
                .saturating_add(refinement),
        )?;
        let atoms = AtomMemory::new(self.dimensions, self.config.meaning.idf_clip_factor);
        let composites = CompositeMemory::new(self.dimensions, self.config.meaning.idf_clip_factor);
        let triples = TripleStore::new();
        for (id, vector) in &snapshot.atoms {
            atoms.insert_with_vec(id, vector);
        }
        for (id, vector) in &snapshot.composites {
            composites.insert(id.clone(), vector.clone());
        }
        for triple in &snapshot.triples {
            ensure!(
                snapshot.composites.contains_key(&triple.composite_id),
                "governance found a live triple without its composite; repair the store first"
            );
            triples.load_triple(triple.clone());
        }
        let report = MemoryGovernor::govern(&atoms, &composites, &triples, &config);
        let mut mutations = Vec::new();
        let mut changed = BTreeSet::new();
        for (id, old) in &snapshot.atoms {
            match atoms.get(id) {
                None => append(
                    &mut mutations,
                    budget,
                    Mutation::DeleteAtom { id: id.clone() },
                )?,
                Some(vector) if !same(old, &vector) => {
                    changed.insert(id.as_str());
                    append(
                        &mut mutations,
                        budget,
                        Mutation::Atom {
                            id: id.clone(),
                            vector,
                        },
                    )?;
                }
                Some(_) => {}
            }
        }
        for id in snapshot.composites.keys() {
            if composites.inner().get(id).is_none() {
                ensure!(
                    !triples.has_composite(id),
                    "governance cannot discard an owned composite"
                );
                append(
                    &mut mutations,
                    budget,
                    Mutation::DeleteComposite { id: id.clone() },
                )?;
            }
        }
        let mut owners: BTreeMap<&str, Vec<&TripleRecord>> = BTreeMap::new();
        for triple in &snapshot.triples {
            owners.entry(&triple.composite_id).or_default().push(triple);
        }
        let roles = self
            .role_registry
            .as_ref()
            .context("meaning roles are missing")?;
        for (id, owners) in owners {
            if !owners.iter().any(|owner| {
                [&owner.subject_id, &owner.relation_id, &owner.object_id]
                    .iter()
                    .any(|atom| changed.contains(atom.as_str()))
            }) {
                continue;
            }
            ensure!(
                owners.len() == 1 && snapshot.mirrored.contains(id),
                "governance cannot refine an ambiguous composite/vector owner"
            );
            budget.work(1)?;
            let owner = owners[0];
            let subject = atoms
                .get(&owner.subject_id)
                .context("refinement lost a referenced subject")?;
            let relation = atoms
                .get(&owner.relation_id)
                .context("refinement lost a referenced relation")?;
            let object = atoms
                .get(&owner.object_id)
                .context("refinement lost a referenced object")?;
            let vector = roles.compose_triple(&subject, &relation, &object);
            append(
                &mut mutations,
                budget,
                Mutation::Composite {
                    id: id.to_owned(),
                    vector,
                },
            )?;
        }
        append(&mut mutations, budget, Mutation::RefreshMeaningIndices {})?;
        let json_bytes = crate::core::documents::serialized_size(&mutations, MAX_PAYLOAD_BYTES)
            .context("governance transaction exceeds 32 MiB")?;
        budget.reserve(
            json_bytes
                .saturating_add(mutations.len().saturating_mul(8))
                .saturating_add(64),
        )?;
        let payload = self.transaction_payload(&mutations)?;
        ensure!(
            payload.len() <= MAX_PAYLOAD_BYTES,
            "governance transaction exceeds 32 MiB"
        );
        Ok(Plan {
            revision: snapshot.revision,
            mutations,
            payload,
            report,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{entangled::EntangledHVec, HmsConfig};

    fn config() -> HmsConfig {
        let mut config = HmsConfig::default();
        config.meaning.enabled = true;
        config.cognition.governor_forget_unreferenced = true;
        config
    }

    #[test]
    fn governance_keeps_distinct_live_triple_owners() {
        let directory = tempfile::tempdir().unwrap();
        let core = HmsCore::new(
            16384,
            Some(directory.path().display().to_string()),
            Some(config()),
        )
        .unwrap();
        for index in 0..512 {
            core.memorize_triplet(
                format!("triple-{index:03}"),
                "alice".into(),
                "knows".into(),
                "bob".into(),
            )
            .unwrap();
        }
        core.govern_memory().unwrap();
        assert_eq!(core.meaning_composite_count(), 512);
        assert_eq!(core.meaning_triple_count(), 512);
        for triple in core.triple_store.as_ref().unwrap().snapshot() {
            assert!(core
                .composite_memory
                .as_ref()
                .unwrap()
                .inner()
                .get(&triple.composite_id)
                .is_some());
        }
    }

    #[test]
    fn governance_atom_alias_deletion_is_durable_without_deleting_the_raw_vector() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().display().to_string();
        let core = HmsCore::new(1024, Some(path.clone()), Some(config())).unwrap();
        let vector = EntangledHVec::from_indices(vec![7, 9], 1024);
        core.memorize("alias".into(), vector.clone()).unwrap();
        core.govern_memory().unwrap();
        assert!(core.atom_memory.as_ref().unwrap().get("alias").is_none());
        assert_eq!(
            core.shards.read().get_vector("alias").unwrap().indices(),
            vector.indices()
        );
        drop(core);
        let core = HmsCore::new(1024, Some(path.clone()), Some(config())).unwrap();
        assert!(core.atom_memory.as_ref().unwrap().get("alias").is_none());
        assert_eq!(
            core.shards.read().get_vector("alias").unwrap().indices(),
            vector.indices()
        );
        core.compact().unwrap();
        drop(core);
        let core = HmsCore::new(1024, Some(path), Some(config())).unwrap();
        assert!(core.atom_memory.as_ref().unwrap().get("alias").is_none());
        assert_eq!(
            core.shards.read().get_vector("alias").unwrap().indices(),
            vector.indices()
        );
    }

    fn fingerprint(core: &HmsCore) -> serde_json::Value {
        let entries = |memory: &crate::core::indexed_memory::IndexedMemory| {
            let mut entries: Vec<_> = memory
                .all_vectors()
                .into_iter()
                .map(|(_, id, vector)| (id, vector.indices().to_vec()))
                .collect();
            entries.sort_unstable();
            entries
        };
        let mut vectors: Vec<_> = core
            .shards
            .read()
            .collect_all_patterns()
            .into_iter()
            .map(|(id, vector)| (id, vector.indices().to_vec()))
            .collect();
        vectors.sort_unstable();
        let mut triples: Vec<_> = core
            .triple_store
            .as_ref()
            .unwrap()
            .snapshot()
            .into_iter()
            .map(|t| (t.subject_id, t.relation_id, t.object_id, t.composite_id))
            .collect();
        triples.sort_unstable();
        serde_json::json!({"atoms": entries(core.atom_memory.as_ref().unwrap().inner()), "composites": entries(core.composite_memory.as_ref().unwrap().inner()), "vectors": vectors, "triples": triples})
    }

    #[test]
    fn governance_refinement_rebinds_owned_composites_and_survives_replay_and_compaction() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().display().to_string();
        let mut config = config();
        config.cognition.refine_atoms = true;
        let core = HmsCore::new(16384, Some(path.clone()), Some(config.clone())).unwrap();
        for (city, country) in [
            ("paris", "france"),
            ("berlin", "germany"),
            ("tokyo", "japan"),
        ] {
            core.memorize_triplet(
                format!("{city}:capital"),
                city.into(),
                "capital_of".into(),
                country.into(),
            )
            .unwrap();
            core.memorize_triplet(
                format!("{city}:located"),
                city.into(),
                "located_in".into(),
                "continent".into(),
            )
            .unwrap();
        }
        let before = fingerprint(&core);
        let report = core.govern_memory().unwrap();
        assert!(report.refinement.atoms_refined > 0);
        let after = fingerprint(&core);
        assert_ne!(before["atoms"], after["atoms"]);
        for triple in core.triple_store.as_ref().unwrap().snapshot() {
            let atoms = core.atom_memory.as_ref().unwrap();
            let vector = core.role_registry.as_ref().unwrap().compose_triple(
                &atoms.get(&triple.subject_id).unwrap(),
                &atoms.get(&triple.relation_id).unwrap(),
                &atoms.get(&triple.object_id).unwrap(),
            );
            assert!(same(
                &vector,
                &core
                    .composite_memory
                    .as_ref()
                    .unwrap()
                    .inner()
                    .get(&triple.composite_id)
                    .unwrap()
            ));
            assert!(same(
                &vector,
                &core.shards.read().get_vector(&triple.composite_id).unwrap()
            ));
        }
        drop(core);
        let core = HmsCore::new(16384, Some(path.clone()), Some(config.clone())).unwrap();
        assert_eq!(fingerprint(&core), after);
        core.compact().unwrap();
        drop(core);
        let core = HmsCore::new(16384, Some(path), Some(config)).unwrap();
        assert_eq!(fingerprint(&core), after);
    }

    #[test]
    fn governance_cas_rejects_a_concurrent_writer_without_publishing_the_plan() {
        let directory = tempfile::tempdir().unwrap();
        let core = std::sync::Arc::new(
            HmsCore::new(
                1024,
                Some(directory.path().display().to_string()),
                Some(config()),
            )
            .unwrap(),
        );
        core.memorize("alias".into(), EntangledHVec::from_indices(vec![1], 1024))
            .unwrap();
        let gate = std::sync::Arc::new(std::sync::Barrier::new(2));
        let (writer_core, writer_gate) = (core.clone(), gate.clone());
        let writer = std::thread::spawn(move || {
            writer_gate.wait();
            writer_core
                .memorize("writer".into(), EntangledHVec::from_indices(vec![2], 1024))
                .unwrap();
            writer_gate.wait();
        });
        let error = core
            .govern_with_hook(QueryLimits::default(), || {
                gate.wait();
                gate.wait();
            })
            .unwrap_err();
        writer.join().unwrap();
        assert!(error.downcast_ref::<GovernanceConflict>().is_some());
        assert!(core.atom_memory.as_ref().unwrap().get("alias").is_some());
        assert!(core.shards.read().get_vector("writer").is_some());
        core.govern_memory().unwrap();
        assert!(core.atom_memory.as_ref().unwrap().get("alias").is_none());
    }

    #[test]
    fn governance_transaction_limit_is_atomic_at_4096_operations() {
        for atoms in [4094, 4095, 4096] {
            let directory = tempfile::tempdir().unwrap();
            let core = HmsCore::new(
                1024,
                Some(directory.path().display().to_string()),
                Some(config()),
            )
            .unwrap();
            let mutations: Vec<_> = (0..atoms)
                .map(|index| Mutation::Atom {
                    id: format!("a{index:04}"),
                    vector: EntangledHVec::from_indices(vec![1], 1024),
                })
                .collect();
            core.commit(&mutations).unwrap();
            let revision = core.revision();
            let result = core.govern_memory();
            if atoms == 4096 {
                assert!(result.unwrap_err().to_string().contains("4096-operation"));
                assert_eq!(core.meaning_atom_count(), atoms);
                assert_eq!(core.revision(), revision);
            } else {
                result.unwrap();
                assert_eq!(core.meaning_atom_count(), 0);
            }
        }
    }

    #[test]
    fn governance_budget_errors_leave_authoritative_state_unchanged() {
        let directory = tempfile::tempdir().unwrap();
        let core = HmsCore::new(
            1024,
            Some(directory.path().display().to_string()),
            Some(config()),
        )
        .unwrap();
        core.memorize("alias".into(), EntangledHVec::from_indices(vec![1], 1024))
            .unwrap();
        let before = fingerprint(&core);
        for limits in [
            QueryLimits {
                max_work: 0,
                ..Default::default()
            },
            QueryLimits {
                max_bytes: 0,
                ..Default::default()
            },
            QueryLimits {
                max_bytes: 4096,
                ..Default::default()
            },
        ] {
            assert!(core.govern_memory_with_limits(limits).is_err());
            assert_eq!(fingerprint(&core), before);
        }
    }
}
