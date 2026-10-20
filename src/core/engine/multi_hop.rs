// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::atom_memory::AtomMemory;
use crate::core::rules::RuleStore;
use crate::core::triple_store::TripleStore;

use super::structural::{fuzzy_structural_query, MeaningContext};

#[derive(Clone, Debug)]
pub struct MultiHopResult {
    pub entity_id: String,
    pub confidence: f64,
    pub method: MultiHopMethod,
    pub hops: Vec<HopDetail>,
}

#[derive(Clone, Debug)]
pub enum MultiHopMethod {
    RuleRewrite { rule_name: String },
    ChainedLookup,
    SingleAlgebraic,
}

#[derive(Clone, Debug)]
pub struct HopDetail {
    pub from_entity: String,
    pub relation: String,
    pub to_entity: String,
    pub confidence: f64,
}

pub fn multi_hop_query(
    start_entity: &str,
    relations: &[&str],
    ctx: &MeaningContext<'_>,
    rule_store: &RuleStore,
    max_depth: usize,
) -> Vec<MultiHopResult> {
    if relations.is_empty() || relations.len() > max_depth {
        return Vec::new();
    }

    if relations.len() == 1 {
        return single_hop(start_entity, relations[0], ctx);
    }

    if relations.len() == 2 {
        if let Some(rule) = rule_store.find_rule(relations[0], relations[1]) {
            return rule_rewrite(start_entity, &rule.output_relation, &rule.name, ctx);
        }
    }

    chained_lookup(start_entity, relations, ctx.atom_memory, ctx.triple_store)
}

fn single_hop(start_entity: &str, relation: &str, ctx: &MeaningContext<'_>) -> Vec<MultiHopResult> {
    let s_vec = match ctx.atom_memory.get(start_entity) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let (_, r_vec) = ctx.atom_memory.get_or_insert(relation);

    let results =
        fuzzy_structural_query(ctx, &[("subject", &s_vec), ("relation", &r_vec)], "object");

    results
        .into_iter()
        .map(|r| MultiHopResult {
            entity_id: r.entity_id.clone(),
            confidence: r.confidence,
            method: MultiHopMethod::SingleAlgebraic,
            hops: vec![HopDetail {
                from_entity: start_entity.to_string(),
                relation: relation.to_string(),
                to_entity: r.entity_id,
                confidence: r.confidence,
            }],
        })
        .collect()
}

fn rule_rewrite(
    start_entity: &str,
    output_relation: &str,
    rule_name: &str,
    ctx: &MeaningContext<'_>,
) -> Vec<MultiHopResult> {
    let s_vec = match ctx.atom_memory.get(start_entity) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let (_, r_vec) = ctx.atom_memory.get_or_insert(output_relation);

    let results =
        fuzzy_structural_query(ctx, &[("subject", &s_vec), ("relation", &r_vec)], "object");

    results
        .into_iter()
        .map(|r| MultiHopResult {
            entity_id: r.entity_id.clone(),
            confidence: r.confidence,
            method: MultiHopMethod::RuleRewrite {
                rule_name: rule_name.to_string(),
            },
            hops: vec![HopDetail {
                from_entity: start_entity.to_string(),
                relation: output_relation.to_string(),
                to_entity: r.entity_id,
                confidence: r.confidence,
            }],
        })
        .collect()
}

/// Evidence-weighted confidence for a single hop.
/// When an entity has N matches for a relation, each match gets confidence
/// 1/N — a unique match is high confidence, one of many is low.
/// Base decay of 0.9 per hop is applied on top.
const HOP_DECAY: f64 = 0.9;

fn chained_lookup(
    start_entity: &str,
    relations: &[&str],
    _atom_memory: &AtomMemory,
    triple_store: &TripleStore,
) -> Vec<MultiHopResult> {
    let mut current_entities = vec![start_entity.to_string()];
    let mut all_hops: Vec<Vec<HopDetail>> = vec![Vec::new()];

    for &relation in relations {
        let mut next_entities = Vec::new();
        let mut next_hops = Vec::new();

        for (i, entity) in current_entities.iter().enumerate() {
            let triples = triple_store.query(Some(entity), Some(relation), None);
            let fan_out = triples.len();
            if fan_out == 0 {
                continue;
            }
            // Evidence-weighted: unique match = high confidence, many matches = lower each
            let evidence_weight = 1.0 / (fan_out as f64).sqrt();
            let hop_confidence = HOP_DECAY * evidence_weight.min(1.0);

            for t in &triples {
                let hop = HopDetail {
                    from_entity: entity.clone(),
                    relation: relation.to_string(),
                    to_entity: t.object_id.clone(),
                    confidence: hop_confidence,
                };
                let mut hops = all_hops.get(i).cloned().unwrap_or_default();
                hops.push(hop);
                next_hops.push(hops);
                next_entities.push(t.object_id.clone());
            }
        }

        if next_entities.is_empty() {
            return Vec::new();
        }

        current_entities = next_entities;
        all_hops = next_hops;
    }

    let mut results: Vec<MultiHopResult> = current_entities
        .into_iter()
        .zip(all_hops)
        .map(|(entity, hops)| {
            let chain_confidence = hops.iter().map(|h| h.confidence).product();
            MultiHopResult {
                entity_id: entity,
                confidence: chain_confidence,
                method: MultiHopMethod::ChainedLookup,
                hops,
            }
        })
        .collect();

    // Rank by confidence descending
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
    use crate::core::admission::AdmissionControl;
    use crate::core::composite_memory::CompositeMemory;
    use crate::core::engine::structural::MeaningContext;
    use crate::core::role::RoleRegistry;
    use crate::core::rules::CompositionRule;

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
    fn test_multi_hop_chained() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        let roles = RoleRegistry::new(dim);
        let rule_store = RuleStore::new();
        let admission = AdmissionControl::new(40);

        atom_mem.get_or_insert("john");
        atom_mem.get_or_insert("mark");
        atom_mem.get_or_insert("bob");
        atom_mem.get_or_insert("father");

        triple_store.add("john", "father", "mark", "c1");
        triple_store.add("mark", "father", "bob", "c2");

        let ctx = make_ctx(&atom_mem, &comp_mem, &triple_store, &roles, &admission);
        let results = multi_hop_query("john", &["father", "father"], &ctx, &rule_store, 10);

        assert!(!results.is_empty());
        assert_eq!(results[0].entity_id, "bob");
        assert!(matches!(results[0].method, MultiHopMethod::ChainedLookup));
        assert_eq!(results[0].hops.len(), 2);
    }

    #[test]
    fn test_multi_hop_rule_rewrite() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        let roles = RoleRegistry::new(dim);
        let rule_store = RuleStore::new();
        let admission = AdmissionControl::new(40);

        let (_, s_vec) = atom_mem.get_or_insert("john");
        let (_, r_vec) = atom_mem.get_or_insert("grandfather");
        let (_, o_vec) = atom_mem.get_or_insert("bob");

        rule_store.add_rule(CompositionRule {
            name: "grandfather_rule".to_string(),
            input_relations: vec!["father".to_string(), "father".to_string()],
            output_relation: "grandfather".to_string(),
        });

        let composite = roles.compose_triple(&s_vec, &r_vec, &o_vec);
        comp_mem.insert("grandfather_triple".to_string(), composite);
        triple_store.add("john", "grandfather", "bob", "grandfather_triple");

        let ctx = make_ctx(&atom_mem, &comp_mem, &triple_store, &roles, &admission);
        let results = multi_hop_query("john", &["father", "father"], &ctx, &rule_store, 10);

        assert!(!results.is_empty());
        assert_eq!(results[0].entity_id, "bob");
        assert!(matches!(
            results[0].method,
            MultiHopMethod::RuleRewrite { .. }
        ));
    }

    #[test]
    fn test_evidence_weighted_confidence() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        let roles = RoleRegistry::new(dim);
        let rule_store = RuleStore::new();
        let admission = AdmissionControl::new(40);

        atom_mem.get_or_insert("a");
        atom_mem.get_or_insert("b");
        atom_mem.get_or_insert("c");
        atom_mem.get_or_insert("d");

        // Unique chain: a -> b (only match) -> c (only match)
        triple_store.add("a", "r", "b", "c1");
        triple_store.add("b", "r", "c", "c2");

        let ctx = make_ctx(&atom_mem, &comp_mem, &triple_store, &roles, &admission);
        let unique = multi_hop_query("a", &["r", "r"], &ctx, &rule_store, 10);
        assert!(!unique.is_empty());
        let unique_conf = unique[0].confidence;

        // Ambiguous chain: a -> {b, d} (2 matches)
        triple_store.add("a", "r", "d", "c3");
        let ambig = multi_hop_query("a", &["r", "r"], &ctx, &rule_store, 10);
        // The path through b should now have lower confidence (fan-out=2 at first hop)
        let b_path = ambig.iter().find(|r| r.entity_id == "c");
        assert!(b_path.is_some());
        assert!(
            b_path.unwrap().confidence < unique_conf,
            "Ambiguous path should have lower confidence: ambig={:.4}, unique={:.4}",
            b_path.unwrap().confidence,
            unique_conf
        );
    }

    #[test]
    fn test_multi_hop_ranked_by_confidence() {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let comp_mem = CompositeMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();
        let roles = RoleRegistry::new(dim);
        let rule_store = RuleStore::new();
        let admission = AdmissionControl::new(40);

        atom_mem.get_or_insert("start");
        // Two paths: start -> a -> end, start -> b -> end
        // Path through a: unique at both hops
        // Path through b: b has 3 outgoing "r2" relations (lower confidence)
        triple_store.add("start", "r1", "a", "c1");
        triple_store.add("start", "r1", "b", "c2");
        triple_store.add("a", "r2", "end", "c3");
        triple_store.add("b", "r2", "end", "c4");
        triple_store.add("b", "r2", "noise1", "c5");
        triple_store.add("b", "r2", "noise2", "c6");

        let ctx = make_ctx(&atom_mem, &comp_mem, &triple_store, &roles, &admission);
        let results = multi_hop_query("start", &["r1", "r2"], &ctx, &rule_store, 10);

        // "end" should appear, reached via both paths
        let ends: Vec<_> = results.iter().filter(|r| r.entity_id == "end").collect();
        assert!(!ends.is_empty());

        // Results should be sorted by confidence (highest first)
        for w in results.windows(2) {
            assert!(w[0].confidence >= w[1].confidence);
        }
    }
}
