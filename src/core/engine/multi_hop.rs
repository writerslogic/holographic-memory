// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::query_limits::{QueryBudget, QueryLimitExceeded, QueryLimits};
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
    if relations.len() > max_depth {
        return Vec::new();
    }
    multi_hop_query_with_limits(
        start_entity,
        relations,
        ctx,
        rule_store,
        max_depth,
        QueryLimits::default(),
    )
    .expect("multi-hop query budget exhausted; use multi_hop_with_limits")
}

pub fn multi_hop_query_with_limits(
    start_entity: &str,
    relations: &[&str],
    ctx: &MeaningContext<'_>,
    rule_store: &RuleStore,
    max_depth: usize,
    limits: QueryLimits,
) -> Result<Vec<MultiHopResult>, QueryLimitExceeded> {
    if relations.is_empty() || relations.len() > max_depth {
        return if relations.is_empty() {
            Ok(Vec::new())
        } else {
            Err(QueryLimitExceeded {
                resource: "hop depth",
                limit: max_depth,
            })
        };
    }
    let mut budget = QueryBudget::new(limits);
    if relations.len() == 1 {
        preflight_algebraic(ctx, &mut budget)?;
        return bound_algebraic(single_hop(start_entity, relations[0], ctx), &mut budget);
    }

    if relations.len() == 2 {
        if let Some(rule) = rule_store.find_rule(relations[0], relations[1]) {
            preflight_algebraic(ctx, &mut budget)?;
            return bound_algebraic(
                rule_rewrite(start_entity, &rule.output_relation, &rule.name, ctx),
                &mut budget,
            );
        }
    }

    chained_lookup(start_entity, relations, ctx.triple_store, &mut budget)
}

impl super::WarmCore {
    /// Runs the same multi-hop query with explicit work, frontier and byte limits. Budget
    /// exhaustion is an error; distinct paths to an endpoint remain distinct results.
    pub fn multi_hop_with_limits(
        &self,
        start: &str,
        relations: &[&str],
        limits: QueryLimits,
    ) -> anyhow::Result<Vec<MultiHopResult>> {
        let _transaction = self.mutation_gate.read();
        let (Some(ctx), Some(rules)) = (self.meaning_ctx(), &self.rule_store) else {
            return Ok(Vec::new());
        };
        Ok(multi_hop_query_with_limits(
            start,
            relations,
            &ctx,
            rules,
            self.config.meaning.max_hop_depth,
            limits,
        )?)
    }
}

fn preflight_algebraic(
    ctx: &MeaningContext<'_>,
    budget: &mut QueryBudget,
) -> Result<(), QueryLimitExceeded> {
    let atoms = ctx.atom_memory.inner().vectors_read();
    let composites = ctx.composite_memory.inner().vectors_read();
    budget.frontier(composites.len())?;
    budget.work(
        composites
            .len()
            .saturating_mul(atoms.len().saturating_add(1))
            .saturating_mul(ctx.max_iter.saturating_add(1)),
    )?;
    for (id, vector) in atoms.iter().chain(composites.iter()) {
        budget.reserve(
            id.len()
                .saturating_mul(2)
                .saturating_add(vector.indices().len().saturating_mul(64))
                .saturating_add(128),
        )?;
    }
    Ok(())
}

fn bound_algebraic(
    results: Vec<MultiHopResult>,
    budget: &mut QueryBudget,
) -> Result<Vec<MultiHopResult>, QueryLimitExceeded> {
    budget.frontier(results.len())?;
    for result in &results {
        budget.reserve(path_bytes(&result.entity_id, &result.hops))?;
    }
    Ok(results)
}

fn single_hop(start_entity: &str, relation: &str, ctx: &MeaningContext<'_>) -> Vec<MultiHopResult> {
    algebraic_hop(start_entity, relation, ctx, || {
        MultiHopMethod::SingleAlgebraic
    })
}

fn rule_rewrite(
    start_entity: &str,
    output_relation: &str,
    rule_name: &str,
    ctx: &MeaningContext<'_>,
) -> Vec<MultiHopResult> {
    algebraic_hop(start_entity, output_relation, ctx, || {
        MultiHopMethod::RuleRewrite {
            rule_name: rule_name.to_string(),
        }
    })
}

fn algebraic_hop(
    start_entity: &str,
    relation: &str,
    ctx: &MeaningContext<'_>,
    method: impl Fn() -> MultiHopMethod,
) -> Vec<MultiHopResult> {
    let Some(s_vec) = ctx.atom_memory.get(start_entity) else {
        return Vec::new();
    };
    let Some(r_vec) = ctx.atom_memory.get(relation) else {
        return Vec::new();
    };
    fuzzy_structural_query(ctx, &[("subject", &s_vec), ("relation", &r_vec)], "object")
        .into_iter()
        .map(|r| MultiHopResult {
            entity_id: r.entity_id.clone(),
            confidence: r.confidence,
            method: method(),
            hops: vec![HopDetail {
                from_entity: start_entity.to_string(),
                relation: relation.to_string(),
                to_entity: r.entity_id,
                confidence: r.confidence,
            }],
        })
        .collect()
}

/// Evidence-weighted confidence for a single hop.
/// When an entity has N matches for a relation, each match gets confidence
/// 1/sqrt(N) — a unique match is high confidence, one of many is low.
/// Base decay of 0.9 per hop is applied on top.
const HOP_DECAY: f64 = 0.9;

fn chained_lookup(
    start_entity: &str,
    relations: &[&str],
    triple_store: &TripleStore,
    budget: &mut QueryBudget,
) -> Result<Vec<MultiHopResult>, QueryLimitExceeded> {
    let bytes = path_bytes(start_entity, &[]);
    budget.frontier(1)?;
    budget.reserve(bytes)?;
    let mut current = vec![(start_entity.to_owned(), Vec::<HopDetail>::new(), bytes)];

    for &relation in relations {
        let mut next = Vec::new();
        for (entity, hops, _) in &current {
            let mut fan_out = 0usize;
            triple_store.visit_bounded(Some(entity), Some(relation), None, budget, |_, _| {
                fan_out += 1;
                Ok(())
            })?;
            if fan_out == 0 {
                continue;
            }
            let evidence_weight = 1.0 / (fan_out as f64).sqrt();
            let hop_confidence = HOP_DECAY * evidence_weight.min(1.0);
            triple_store.visit_bounded(
                Some(entity),
                Some(relation),
                None,
                budget,
                |t, budget| {
                    budget.work(hops.len().saturating_add(1))?;
                    budget.frontier(next.len().saturating_add(1))?;
                    let bytes = path_bytes(&t.object_id, hops).saturating_add(hop_bytes(
                        entity,
                        relation,
                        &t.object_id,
                    ));
                    budget.reserve(bytes)?;
                    let mut path = hops.clone();
                    path.push(HopDetail {
                        from_entity: entity.clone(),
                        relation: relation.to_owned(),
                        to_entity: t.object_id.clone(),
                        confidence: hop_confidence,
                    });
                    next.push((t.object_id.clone(), path, bytes));
                    Ok(())
                },
            )?;
        }
        if next.is_empty() {
            return Ok(Vec::new());
        }
        for (_, _, bytes) in &current {
            budget.release(*bytes);
        }
        current = next;
    }

    let mut results: Vec<MultiHopResult> = current
        .into_iter()
        .map(|(entity, hops, _)| {
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
    Ok(results)
}

fn hop_bytes(from: &str, relation: &str, to: &str) -> usize {
    2 * std::mem::size_of::<HopDetail>()
        + from
            .len()
            .saturating_add(relation.len())
            .saturating_add(to.len())
            .saturating_mul(2)
}

fn path_bytes(entity: &str, hops: &[HopDetail]) -> usize {
    hops.iter().fold(
        entity
            .len()
            .saturating_mul(2)
            .saturating_add(2 * std::mem::size_of::<MultiHopResult>()),
        |bytes, hop| {
            bytes.saturating_add(hop_bytes(&hop.from_entity, &hop.relation, &hop.to_entity))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::engine::structural::tests::Fixture;
    use crate::core::rules::CompositionRule;

    #[test]
    fn bounded_chains_preserve_distinct_paths_at_work_and_frontier_boundaries() {
        let fixture = Fixture::new();
        for (from, to, id) in [
            ("a", "b", "ab"),
            ("a", "c", "ac"),
            ("b", "d", "bd"),
            ("c", "d", "cd"),
        ] {
            fixture.triples.add(from, "r", to, id);
        }
        let rules = RuleStore::new();
        for work in [13, 14, 15] {
            let result = multi_hop_query_with_limits(
                "a",
                &["r", "r"],
                &fixture.context(),
                &rules,
                10,
                QueryLimits {
                    max_work: work,
                    ..Default::default()
                },
            );
            if work == 13 {
                assert_eq!(result.unwrap_err().resource, "work");
            } else {
                let paths = result.unwrap();
                assert_eq!(paths.len(), 2);
                assert!(paths
                    .iter()
                    .all(|path| path.entity_id == "d" && path.hops.len() == 2));
                assert_ne!(paths[0].hops[0].to_entity, paths[1].hops[0].to_entity);
            }
        }
        for frontier in [1, 2, 3] {
            let result = multi_hop_query_with_limits(
                "a",
                &["r", "r"],
                &fixture.context(),
                &rules,
                10,
                QueryLimits {
                    max_frontier: frontier,
                    ..Default::default()
                },
            );
            if frontier == 1 {
                assert_eq!(result.unwrap_err().resource, "frontier");
            } else {
                assert_eq!(result.unwrap().len(), 2);
            }
        }
        let error = multi_hop_query_with_limits(
            "a",
            &["r", "r"],
            &fixture.context(),
            &rules,
            10,
            QueryLimits {
                max_bytes: 1,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(error.resource, "bytes");
        fixture.triples.add("a", "unrelated", "z", "az");
        let error = multi_hop_query_with_limits(
            "a",
            &["r", "r"],
            &fixture.context(),
            &rules,
            10,
            QueryLimits {
                max_work: 14,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(error.resource, "work");
    }

    #[test]
    fn cyclic_branching_queries_fail_explicitly_before_frontier_growth() {
        let fixture = Fixture::new();
        for from in ["a", "b", "c"] {
            for to in ["a", "b", "c"] {
                fixture.triples.add(from, "r", to, &format!("{from}{to}"));
            }
        }
        let error = multi_hop_query_with_limits(
            "a",
            &["r"; 10],
            &fixture.context(),
            &RuleStore::new(),
            10,
            QueryLimits {
                max_frontier: 8,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(error.resource, "frontier");
    }

    #[test]
    fn querying_an_unknown_relation_does_not_create_an_atom() {
        let fixture = Fixture::new();
        fixture.atoms.get_or_insert("a");
        let count = fixture.atoms.count();
        assert!(
            multi_hop_query("a", &["unknown"], &fixture.context(), &RuleStore::new(), 10)
                .is_empty()
        );
        assert_eq!(fixture.atoms.count(), count);
    }

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
