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
    use crate::core::engine::structural::tests::Fixture;
    use crate::core::rules::CompositionRule;

    #[test]
    fn chained_lookup_follows_each_hop() {
        let fixture = Fixture::new();
        for atom in ["john", "mark", "bob", "father"] {
            fixture.atoms.get_or_insert(atom);
        }
        fixture.triples.add("john", "father", "mark", "c1");
        fixture.triples.add("mark", "father", "bob", "c2");

        let results = multi_hop_query(
            "john",
            &["father", "father"],
            &fixture.context(),
            &RuleStore::new(),
            10,
        );

        assert_eq!(results[0].entity_id, "bob");
        assert!(matches!(results[0].method, MultiHopMethod::ChainedLookup));
        assert_eq!(results[0].hops.len(), 2);
    }

    #[test]
    fn composition_rule_rewrites_two_hops_into_one_relation() {
        let fixture = Fixture::new();
        let rule_store = RuleStore::new();
        rule_store.add_rule(CompositionRule {
            name: "grandfather_rule".to_string(),
            input_relations: vec!["father".to_string(), "father".to_string()],
            output_relation: "grandfather".to_string(),
        });
        fixture.bind("john", "grandfather", "bob", "grandfather_triple");

        let results = multi_hop_query(
            "john",
            &["father", "father"],
            &fixture.context(),
            &rule_store,
            10,
        );

        assert_eq!(results[0].entity_id, "bob");
        assert!(matches!(
            results[0].method,
            MultiHopMethod::RuleRewrite { .. }
        ));
    }

    #[test]
    fn fan_out_at_a_hop_lowers_chain_confidence() {
        let fixture = Fixture::new();
        let rule_store = RuleStore::new();
        for atom in ["a", "b", "c", "d"] {
            fixture.atoms.get_or_insert(atom);
        }
        fixture.triples.add("a", "r", "b", "c1");
        fixture.triples.add("b", "r", "c", "c2");
        let unique = multi_hop_query("a", &["r", "r"], &fixture.context(), &rule_store, 10);
        let unique_confidence = unique[0].confidence;

        fixture.triples.add("a", "r", "d", "c3");
        let ambiguous = multi_hop_query("a", &["r", "r"], &fixture.context(), &rule_store, 10);
        let through_b = ambiguous
            .iter()
            .find(|r| r.entity_id == "c")
            .expect("the path through b still reaches c");
        assert!(
            through_b.confidence < unique_confidence,
            "ambiguous {:.4} >= unique {unique_confidence:.4}",
            through_b.confidence
        );
    }

    #[test]
    fn results_rank_by_confidence() {
        let fixture = Fixture::new();
        fixture.atoms.get_or_insert("start");
        // start -> a -> end is unique at both hops; b also reaches end, among three objects.
        for (subject, relation, object, id) in [
            ("start", "r1", "a", "c1"),
            ("start", "r1", "b", "c2"),
            ("a", "r2", "end", "c3"),
            ("b", "r2", "end", "c4"),
            ("b", "r2", "noise1", "c5"),
            ("b", "r2", "noise2", "c6"),
        ] {
            fixture.triples.add(subject, relation, object, id);
        }

        let results = multi_hop_query(
            "start",
            &["r1", "r2"],
            &fixture.context(),
            &RuleStore::new(),
            10,
        );

        assert!(results.iter().any(|r| r.entity_id == "end"));
        assert!(results
            .windows(2)
            .all(|w| w[0].confidence >= w[1].confidence));
    }
}
