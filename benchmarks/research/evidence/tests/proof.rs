// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use warm_research_evidence::proof::*;
use std::collections::BTreeSet;

fn node(id: &str, supports: &[&[&str]]) -> ProofNode {
    ProofNode {
        id: id.into(),
        supports: supports
            .iter()
            .map(|support| support.iter().map(|id| (*id).into()).collect())
            .collect(),
    }
}

fn source(id: &str, cost: u64) -> PrimitiveSource {
    PrimitiveSource {
        id: id.into(),
        cost,
    }
}

fn problem(
    nodes: Vec<ProofNode>,
    sources: Vec<PrimitiveSource>,
    targets: &[&str],
    protected: &[&str],
) -> ProofProblem {
    ProofProblem {
        nodes,
        sources,
        targets: targets.iter().map(|id| (*id).into()).collect(),
        protected: protected.iter().map(|id| (*id).into()).collect(),
    }
}

fn complete(problem: &ProofProblem) -> ProofCutResult {
    plan_forgetting(
        problem,
        SearchLimits {
            max_sources: MAX_SOURCES,
            max_steps: MAX_SEARCH_STEPS,
        },
    )
    .unwrap()
}

fn closure(problem: &ProofProblem, removed: &BTreeSet<String>) -> BTreeSet<String> {
    let mut derived = BTreeSet::new();
    loop {
        let before = derived.len();
        for node in &problem.nodes {
            if !removed.contains(&node.id)
                && node
                    .supports
                    .iter()
                    .any(|support| support.iter().all(|premise| derived.contains(premise)))
            {
                derived.insert(node.id.clone());
            }
        }
        if derived.len() == before {
            return derived;
        }
    }
}

fn check_evidence(problem: &ProofProblem, removed: &[String], evidence: &DerivabilityEvidence) {
    let removed: BTreeSet<_> = removed.iter().cloned().collect();
    let expected = closure(problem, &removed);
    let mut witnessed = BTreeSet::new();
    for witness in &evidence.witnesses {
        let node = problem
            .nodes
            .iter()
            .find(|node| node.id == witness.node_id)
            .unwrap();
        assert!(!removed.contains(&node.id));
        assert_eq!(node.supports[witness.support_index], witness.premises);
        assert!(witness
            .premises
            .iter()
            .all(|premise| witnessed.contains(premise)));
        assert!(witnessed.insert(node.id.clone()));
    }
    assert_eq!(witnessed, expected);
    for (labels, statuses) in [
        (&problem.targets, &evidence.targets),
        (&problem.protected, &evidence.protected),
    ] {
        assert_eq!(labels.len(), statuses.len());
        for (id, status) in labels.iter().zip(statuses) {
            assert_eq!(id, &status.node_id);
            assert_eq!(status.derivable, expected.contains(id));
        }
    }
}

#[test]
fn diamond_shared_source_beats_cutting_both_branches() {
    let input = problem(
        vec![
            node("root", &[&[]]),
            node("left", &[&["root"]]),
            node("right", &[&["root"]]),
            node("target", &[&["left"], &["right"]]),
        ],
        vec![source("root", 7)],
        &["target"],
        &[],
    );
    let result = complete(&input);
    assert_eq!(result.status, ProofCutStatus::Solved);
    let plan = result.plan.unwrap();
    assert_eq!(plan.removed_sources, ["root"]);
    assert_eq!(plan.cost, 7);
    check_evidence(&input, &plan.removed_sources, &plan.evidence);
    check_evidence(&input, &[], result.baseline.as_ref().unwrap());
}

#[test]
fn alternate_proofs_require_cutting_every_path() {
    let input = problem(
        vec![
            node("a", &[&[]]),
            node("b", &[&[]]),
            node("target", &[&["a"], &["b"]]),
        ],
        vec![source("b", 5), source("a", 2)],
        &["target"],
        &[],
    );
    let result = complete(&input);
    let plan = result.plan.unwrap();
    assert_eq!(plan.removed_sources, ["a", "b"]);
    assert_eq!(plan.cost, 7);
    check_evidence(&input, &plan.removed_sources, &plan.evidence);
}

#[test]
fn conjunction_chooses_cheapest_operand_and_preserves_independent_proof() {
    let input = problem(
        vec![
            node("a", &[&[]]),
            node("b", &[&[]]),
            node("safe", &[&[]]),
            node("target", &[&["a", "b"]]),
            node("kept", &[&["a"], &["safe"]]),
        ],
        vec![source("a", 2), source("b", 8)],
        &["target"],
        &["kept"],
    );
    let result = complete(&input);
    let plan = result.plan.unwrap();
    assert_eq!(plan.removed_sources, ["a"]);
    assert!(plan.evidence.protected[0].derivable);
    assert!(!plan.evidence.targets[0].derivable);
    check_evidence(&input, &plan.removed_sources, &plan.evidence);
}

#[test]
fn unsupported_cycles_and_self_loops_cannot_bootstrap_truth() {
    let input = problem(
        vec![
            node("a", &[&["b"]]),
            node("b", &[&["a"]]),
            node("self", &[&["self"]]),
        ],
        vec![],
        &["a", "b", "self"],
        &[],
    );
    let result = complete(&input);
    let plan = result.plan.unwrap();
    assert!(plan.removed_sources.is_empty());
    assert!(plan.evidence.witnesses.is_empty());
    check_evidence(&input, &[], &plan.evidence);
}

#[test]
fn seeded_cycle_loses_truth_after_its_only_root_is_cut() {
    let input = problem(
        vec![
            node("root", &[&[]]),
            node("a", &[&["root"], &["b"]]),
            node("b", &[&["a"]]),
        ],
        vec![source("root", 3)],
        &["b"],
        &[],
    );
    let result = complete(&input);
    let plan = result.plan.unwrap();
    assert_eq!(plan.removed_sources, ["root"]);
    assert!(plan.evidence.witnesses.is_empty());
    assert_eq!(result.baseline.as_ref().unwrap().witnesses.len(), 3);
    check_evidence(&input, &[], result.baseline.as_ref().unwrap());
}

#[test]
fn protection_can_force_a_more_expensive_cut_or_make_every_cut_infeasible() {
    let mut input = problem(
        vec![
            node("a", &[&[]]),
            node("b", &[&[]]),
            node("target", &[&["a", "b"]]),
        ],
        vec![source("a", 1), source("b", 9)],
        &["target"],
        &["a"],
    );
    assert_eq!(complete(&input).plan.unwrap().removed_sources, ["b"]);
    input.protected.push("b".into());
    let result = complete(&input);
    assert_eq!(result.status, ProofCutStatus::Infeasible);
    assert_eq!(result.completed_candidates, result.total_candidates);
    assert!(result.plan.is_none());
    input.protected = vec!["target".into()];
    assert_eq!(complete(&input).status, ProofCutStatus::Infeasible);
}

#[test]
fn undeletable_assertion_and_initially_false_protection_are_infeasible() {
    for input in [
        problem(vec![node("target", &[&[]])], vec![], &["target"], &[]),
        problem(vec![node("unknown", &[])], vec![], &[], &["unknown"]),
    ] {
        let result = complete(&input);
        assert_eq!(result.status, ProofCutStatus::Infeasible);
        assert!(result.plan.is_none());
    }
}

#[test]
fn zero_cost_ties_prefer_fewer_sources_then_lexicographic_ids() {
    let input = problem(
        vec![
            node("a", &[&[]]),
            node("b", &[&[]]),
            node("c", &[&[]]),
            node("target", &[&["a", "b"]]),
        ],
        vec![source("c", 0), source("b", 0), source("a", 0)],
        &["target"],
        &[],
    );
    let result = complete(&input);
    let plan = result.plan.unwrap();
    assert_eq!(plan.cost, 0);
    assert_eq!(plan.removed_sources, ["a"]);
    let mut reordered = input.clone();
    reordered.nodes.reverse();
    reordered.sources.reverse();
    assert_eq!(complete(&reordered).plan.unwrap().removed_sources, ["a"]);
}

#[test]
fn search_budget_boundaries_never_mislabel_an_incomplete_search() {
    let input = problem(
        vec![node("a", &[&[]]), node("target", &[&["a"]])],
        vec![source("a", 1)],
        &["target"],
        &[],
    );
    let finished = complete(&input);
    for budget in [
        0,
        finished.steps_used - 1,
        finished.steps_used,
        finished.steps_used + 1,
    ] {
        let result = plan_forgetting(
            &input,
            SearchLimits {
                max_sources: 1,
                max_steps: budget,
            },
        )
        .unwrap();
        assert!(result.steps_used <= budget);
        if budget < finished.steps_used {
            assert_eq!(result.status, ProofCutStatus::BudgetExhausted);
            assert!(result.completed_candidates < result.total_candidates);
        } else {
            assert_eq!(result, finished);
        }
    }
    let bounded = plan_forgetting(
        &input,
        SearchLimits {
            max_sources: 0,
            max_steps: MAX_SEARCH_STEPS,
        },
    )
    .unwrap();
    assert_eq!(bounded.status, ProofCutStatus::BudgetExhausted);
    assert_eq!(bounded.completed_candidates, 0);
    assert_eq!(bounded.steps_used, 0);
}

#[test]
fn exhausted_search_retains_only_a_fully_verified_feasible_incumbent() {
    let input = problem(
        vec![node("root", &[&[]]), node("target", &[])],
        vec![source("root", 0)],
        &["target"],
        &[],
    );
    let result = plan_forgetting(
        &input,
        SearchLimits {
            max_sources: 1,
            max_steps: 4,
        },
    )
    .unwrap();
    assert_eq!(result.status, ProofCutStatus::BudgetExhausted);
    assert_eq!(result.completed_candidates, 1);
    let plan = result.plan.unwrap();
    assert!(plan.removed_sources.is_empty());
    check_evidence(&input, &plan.removed_sources, &plan.evidence);
}

#[test]
fn unknown_references_duplicates_and_nonprimitive_sources_are_rejected() {
    let base = problem(
        vec![node("a", &[&[]]), node("b", &[&["a"]])],
        vec![source("a", 1)],
        &["b"],
        &[],
    );
    let mut invalid = Vec::new();
    let mut input = base.clone();
    input.nodes.push(input.nodes[0].clone());
    invalid.push(input);
    let mut input = base.clone();
    input.nodes[1].supports = vec![vec!["missing".into()]];
    invalid.push(input);
    let mut input = base.clone();
    input.nodes[1].supports = vec![vec!["a".into(), "a".into()]];
    invalid.push(input);
    let mut input = base.clone();
    input.nodes[1].supports.push(vec!["a".into()]);
    invalid.push(input);
    let mut input = base.clone();
    input.sources.push(source("a", 1));
    invalid.push(input);
    let mut input = base.clone();
    input.sources = vec![source("missing", 1)];
    invalid.push(input);
    let mut input = base.clone();
    input.sources = vec![source("b", 1)];
    invalid.push(input);
    let mut input = base.clone();
    input.nodes[0].supports.clear();
    invalid.push(input);
    let mut input = base.clone();
    input.nodes[0].supports.push(vec!["b".into()]);
    invalid.push(input);
    let mut input = base.clone();
    input.targets.push("b".into());
    invalid.push(input);
    let mut input = base.clone();
    input.protected = vec!["a".into(), "a".into()];
    invalid.push(input);
    let mut input = base.clone();
    input.targets = vec!["missing".into()];
    invalid.push(input);
    let mut input = base;
    input.protected = vec!["missing".into()];
    invalid.push(input);
    for input in invalid {
        assert!(plan_forgetting(&input, SearchLimits::default()).is_err());
    }
    let duplicate_permuted_support = problem(
        vec![
            node("a", &[&[]]),
            node("b", &[&[]]),
            node("target", &[&["a", "b"], &["b", "a"]]),
        ],
        vec![],
        &[],
        &[],
    );
    assert!(plan_forgetting(&duplicate_permuted_support, SearchLimits::default()).is_err());
}

#[test]
fn ids_costs_and_serialized_unsigned_ranges_are_validated() {
    for length in [MAX_ID_BYTES - 1, MAX_ID_BYTES, MAX_ID_BYTES + 1] {
        let input = problem(vec![node(&"x".repeat(length), &[&[]])], vec![], &[], &[]);
        assert_eq!(
            plan_forgetting(&input, SearchLimits::default()).is_ok(),
            length <= MAX_ID_BYTES
        );
    }
    let unicode_id = "é".repeat(MAX_ID_BYTES / 2);
    assert!(plan_forgetting(
        &problem(vec![node(&unicode_id, &[])], vec![], &[], &[]),
        SearchLimits::default()
    )
    .is_ok());
    assert!(plan_forgetting(
        &problem(vec![node(&format!("{unicode_id}a"), &[])], vec![], &[], &[]),
        SearchLimits::default()
    )
    .is_err());
    for id in ["", " ", " leading", "trailing ", "line\nbreak", "nul\0byte"] {
        assert!(plan_forgetting(
            &problem(vec![node(id, &[])], vec![], &[], &[]),
            SearchLimits::default()
        )
        .is_err());
    }
    let mut input = problem(
        vec![node("a", &[&[]]), node("b", &[&[]])],
        vec![source("a", u64::MAX), source("b", 0)],
        &["a"],
        &[],
    );
    assert_eq!(complete(&input).plan.unwrap().cost, u64::MAX);
    input.sources[1].cost = 1;
    assert!(plan_forgetting(&input, SearchLimits::default()).is_err());
    assert!(serde_json::from_str::<PrimitiveSource>(r#"{"id":"a","cost":-1}"#).is_err());
    assert!(
        serde_json::from_str::<PrimitiveSource>(r#"{"id":"a","cost":18446744073709551616}"#)
            .is_err()
    );
    assert!(
        serde_json::from_str::<ProofNode>(r#"{"id":"a","supports":[],"unexpected":true}"#).is_err()
    );
}

#[test]
fn node_source_and_search_limit_boundaries_are_explicit() {
    for count in [MAX_NODES - 1, MAX_NODES, MAX_NODES + 1] {
        let input = problem(
            (0..count).map(|i| node(&format!("n{i}"), &[])).collect(),
            vec![],
            &[],
            &[],
        );
        assert_eq!(
            plan_forgetting(&input, SearchLimits::default()).is_ok(),
            count <= MAX_NODES
        );
    }
    for count in [MAX_SOURCES - 1, MAX_SOURCES, MAX_SOURCES + 1] {
        let input = problem(
            (0..count).map(|i| node(&format!("n{i}"), &[&[]])).collect(),
            (0..count).map(|i| source(&format!("n{i}"), 1)).collect(),
            &[],
            &[],
        );
        let result = plan_forgetting(
            &input,
            SearchLimits {
                max_sources: MAX_SOURCES,
                max_steps: 0,
            },
        );
        assert_eq!(result.is_ok(), count <= MAX_SOURCES);
        if let Ok(result) = result {
            assert_eq!(result.status, ProofCutStatus::BudgetExhausted);
            assert_eq!(result.total_candidates, 1u64 << count);
        }
    }
    for limit in [MAX_SEARCH_STEPS - 1, MAX_SEARCH_STEPS, MAX_SEARCH_STEPS + 1] {
        assert_eq!(
            plan_forgetting(
                &ProofProblem::default(),
                SearchLimits {
                    max_sources: 0,
                    max_steps: limit
                }
            )
            .is_ok(),
            limit <= MAX_SEARCH_STEPS
        );
    }
    assert!(plan_forgetting(
        &ProofProblem::default(),
        SearchLimits {
            max_sources: MAX_SOURCES + 1,
            max_steps: 1
        }
    )
    .is_err());
    for field in [true, false] {
        for count in [MAX_NODES - 1, MAX_NODES] {
            let mut input = problem(
                (0..MAX_NODES)
                    .map(|i| node(&format!("n{i}"), &[]))
                    .collect(),
                vec![],
                &[],
                &[],
            );
            let selected = (0..count).map(|i| format!("n{i}")).collect();
            if field {
                input.targets = selected;
            } else {
                input.protected = selected;
            }
            assert!(plan_forgetting(
                &input,
                SearchLimits {
                    max_sources: 0,
                    max_steps: 0
                }
            )
            .is_ok());
        }
        let mut input = problem(vec![node("a", &[])], vec![], &[], &[]);
        if field {
            input.targets = vec!["a".into(); MAX_NODES + 1];
        } else {
            input.protected = vec!["a".into(); MAX_NODES + 1];
        }
        assert!(plan_forgetting(&input, SearchLimits::default()).is_err());
    }
}

#[test]
fn support_and_premise_allocation_boundaries_are_validated_before_search() {
    let no_search = SearchLimits {
        max_sources: MAX_SOURCES,
        max_steps: 0,
    };
    let mut input = ProofProblem::default();
    for i in 0..MAX_NODES {
        input
            .nodes
            .push(node(&format!("n{i}"), &[&[], &["n0"], &["n1"], &["n2"]]));
    }
    input.nodes.last_mut().unwrap().supports.pop();
    assert!(plan_forgetting(&input, no_search).is_ok());
    input
        .nodes
        .last_mut()
        .unwrap()
        .supports
        .push(vec!["n2".into()]);
    assert!(plan_forgetting(&input, no_search).is_ok());
    input.nodes[0].supports.push(vec!["n0".into(), "n1".into()]);
    assert!(plan_forgetting(&input, no_search).is_err());

    for count in [
        MAX_PREMISES_PER_SUPPORT - 1,
        MAX_PREMISES_PER_SUPPORT,
        MAX_PREMISES_PER_SUPPORT + 1,
    ] {
        let mut input = problem(
            (0..count).map(|i| node(&format!("n{i}"), &[])).collect(),
            vec![],
            &[],
            &[],
        );
        input.nodes.push(ProofNode {
            id: "head".into(),
            supports: vec![(0..count).map(|i| format!("n{i}")).collect()],
        });
        assert_eq!(
            plan_forgetting(&input, no_search).is_ok(),
            count <= MAX_PREMISES_PER_SUPPORT
        );
    }

    let premises: Vec<_> = (0..64).map(|i| format!("p{i}")).collect();
    let mut input = problem(
        premises.iter().map(|id| node(id, &[])).collect(),
        vec![],
        &[],
        &[],
    );
    for i in 0..256 {
        input.nodes.push(ProofNode {
            id: format!("h{i}"),
            supports: vec![premises.clone()],
        });
    }
    input.nodes.last_mut().unwrap().supports[0].pop();
    assert!(plan_forgetting(&input, no_search).is_ok());
    input.nodes.last_mut().unwrap().supports[0].push("p63".into());
    assert!(plan_forgetting(&input, no_search).is_ok());
    input
        .nodes
        .last_mut()
        .unwrap()
        .supports
        .push(vec!["p0".into()]);
    assert!(plan_forgetting(&input, no_search).is_err());
}

#[test]
fn aggregate_id_byte_limit_has_exact_boundaries() {
    let premises: Vec<_> = (0..64)
        .map(|i| format!("p{i:03}{}", "x".repeat(252)))
        .collect();
    let mut input = problem(
        premises.iter().map(|id| node(id, &[])).collect(),
        vec![],
        &[],
        &[],
    );
    for i in 0..128 {
        let count = if i < 126 {
            64
        } else if i == 126 {
            60
        } else {
            0
        };
        input.nodes.push(ProofNode {
            id: format!("head{i:04}"),
            supports: vec![premises[..count].to_vec()],
        });
    }
    let bytes: usize = input
        .nodes
        .iter()
        .map(|n| n.id.len() + n.supports.iter().flatten().map(String::len).sum::<usize>())
        .sum();
    assert_eq!(bytes, MAX_TOTAL_ID_BYTES);
    let no_search = SearchLimits {
        max_sources: MAX_SOURCES,
        max_steps: 0,
    };
    input.nodes.last_mut().unwrap().id.pop();
    assert!(plan_forgetting(&input, no_search).is_ok());
    input.nodes.last_mut().unwrap().id.push('7');
    assert!(plan_forgetting(&input, no_search).is_ok());
    input.nodes.last_mut().unwrap().id.push('0');
    assert!(plan_forgetting(&input, no_search).is_err());
}

fn oracle(problem: &ProofProblem) -> Option<(u64, usize, Vec<String>)> {
    let mut best = None;
    for bits in 0..(1usize << problem.sources.len()) {
        let removed: BTreeSet<_> = problem
            .sources
            .iter()
            .enumerate()
            .filter(|(i, _)| bits & (1usize << *i) != 0)
            .map(|(_, s)| s.id.clone())
            .collect();
        let derived = closure(problem, &removed);
        if problem.targets.iter().any(|id| derived.contains(id))
            || problem.protected.iter().any(|id| !derived.contains(id))
        {
            continue;
        }
        let cost: u64 = problem
            .sources
            .iter()
            .filter(|s| removed.contains(&s.id))
            .map(|s| s.cost)
            .sum();
        let key = (cost, removed.len(), removed.into_iter().collect::<Vec<_>>());
        if best.as_ref().is_none_or(|current| key < *current) {
            best = Some(key);
        }
    }
    best
}

#[test]
fn generated_graphs_match_an_independent_bruteforce_fixed_point_oracle() {
    let mut random = 17u64;
    let mut next = || {
        random = random
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        random >> 32
    };
    for case in 0..96 {
        let mut input = ProofProblem::default();
        for i in 0..4 {
            input.nodes.push(node(&format!("n{i}"), &[&[]]));
            input.sources.push(source(&format!("n{i}"), next() % 5));
        }
        for i in 4..10 {
            let mut supports = BTreeSet::new();
            for _ in 0..(next() % 4) {
                let premises: BTreeSet<_> = (0..(next() % 4))
                    .map(|_| format!("n{}", next() % 10))
                    .collect();
                supports.insert(premises.into_iter().collect::<Vec<_>>());
            }
            input.nodes.push(ProofNode {
                id: format!("n{i}"),
                supports: supports.into_iter().collect(),
            });
        }
        input.targets = if case % 3 == 0 {
            vec!["n8".into(), "n9".into()]
        } else {
            vec!["n9".into()]
        };
        if case % 2 == 0 {
            input.protected.push(format!("n{}", next() % 10));
        }
        let expected = oracle(&input);
        let result = complete(&input);
        assert_eq!(result.completed_candidates, result.total_candidates);
        match (expected, result.plan) {
            (Some(key), Some(plan)) => {
                assert_eq!(result.status, ProofCutStatus::Solved);
                assert_eq!(
                    (
                        plan.cost,
                        plan.removed_sources.len(),
                        plan.removed_sources.clone()
                    ),
                    key,
                    "case {case}"
                );
                check_evidence(&input, &plan.removed_sources, &plan.evidence);
            }
            (None, None) => assert_eq!(result.status, ProofCutStatus::Infeasible),
            (expected, actual) => panic!("case {case}: oracle {expected:?}, planner {actual:?}"),
        }
        check_evidence(&input, &[], result.baseline.as_ref().unwrap());
    }
}
