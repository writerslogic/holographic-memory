// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bounded source-cut planning over declared monotone proofs. This module does not delete
//! storage or model unrestricted inference. An empty support asserts a root; no supports
//! means no proof. A source names a primitive root whose assertion a cut can disable.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_NODES: usize = 1024;
pub const MAX_SOURCES: usize = 20;
pub const MAX_SUPPORTS: usize = 4096;
pub const MAX_PREMISES_PER_SUPPORT: usize = 64;
pub const MAX_PREMISE_REFERENCES: usize = 16_384;
pub const MAX_ID_BYTES: usize = 256;
pub const MAX_TOTAL_ID_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_SEARCH_STEPS: u64 = 100_000_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofNode {
    pub id: String,
    /// Alternative proofs, each requiring every named premise. Cycles cannot assert roots.
    pub supports: Vec<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimitiveSource {
    /// Must name a node with exactly one support, which must be empty.
    pub id: String,
    pub cost: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofProblem {
    pub nodes: Vec<ProofNode>,
    pub sources: Vec<PrimitiveSource>,
    pub targets: Vec<String>,
    pub protected: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchLimits {
    /// A smaller configured bound returns BudgetExhausted without starting the search.
    pub max_sources: usize,
    /// Charges candidate entry, node reset, support initialization and premise propagation.
    /// Zero is permitted and performs no search. Validation is separately bounded above.
    pub max_steps: u64,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            max_sources: MAX_SOURCES,
            max_steps: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofCutStatus {
    /// All candidates were evaluated; the plan minimizes cost, then size, then source IDs.
    Solved,
    /// All candidates were evaluated and none satisfied both target and protection checks.
    Infeasible,
    /// Search is incomplete. A returned plan is feasible but is not certified optimal.
    BudgetExhausted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeDerivability {
    pub node_id: String,
    pub derivable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivationWitness {
    pub node_id: String,
    /// Index into this node's input supports.
    pub support_index: usize,
    pub premises: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivabilityEvidence {
    /// Exactly the least fixed point, in firing order: every premise has an earlier witness.
    pub witnesses: Vec<DerivationWitness>,
    pub targets: Vec<NodeDerivability>,
    pub protected: Vec<NodeDerivability>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofCutPlan {
    pub removed_sources: Vec<String>,
    pub cost: u64,
    pub evidence: DerivabilityEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofCutResult {
    pub status: ProofCutStatus,
    pub plan: Option<ProofCutPlan>,
    /// The no-deletion fixed point, absent if even that candidate could not finish.
    pub baseline: Option<DerivabilityEvidence>,
    pub completed_candidates: u64,
    pub total_candidates: u64,
    pub steps_used: u64,
}

struct Rule {
    head: usize,
    support_index: usize,
    premise_count: usize,
}

struct Compiled<'a> {
    problem: &'a ProofProblem,
    rules: Vec<Rule>,
    dependents: Vec<Vec<usize>>,
    source_bits: Vec<Option<u32>>,
    sources: Vec<&'a PrimitiveSource>,
    targets: Vec<usize>,
    protected: Vec<usize>,
}

fn charge_id(id: &str, bytes: &mut usize) -> Result<()> {
    ensure!(
        !id.is_empty()
            && id.len() <= MAX_ID_BYTES
            && id.trim() == id
            && !id.chars().any(char::is_control),
        "proof IDs require 1..={MAX_ID_BYTES} bytes without controls or edge whitespace"
    );
    *bytes = bytes
        .checked_add(id.len())
        .context("proof ID bytes overflow")?;
    ensure!(*bytes <= MAX_TOTAL_ID_BYTES, "proof ID bytes exceed limit");
    Ok(())
}

fn compile(problem: &ProofProblem) -> Result<Compiled<'_>> {
    ensure!(problem.nodes.len() <= MAX_NODES, "too many proof nodes");
    ensure!(
        problem.sources.len() <= MAX_SOURCES,
        "too many primitive sources"
    );
    ensure!(problem.targets.len() <= MAX_NODES, "too many proof targets");
    ensure!(
        problem.protected.len() <= MAX_NODES,
        "too many protected nodes"
    );
    let mut bytes = 0usize;
    let mut support_count = 0usize;
    let mut premise_count = 0usize;
    let mut ids = BTreeMap::new();
    for (index, node) in problem.nodes.iter().enumerate() {
        charge_id(&node.id, &mut bytes)?;
        ensure!(
            ids.insert(node.id.as_str(), index).is_none(),
            "duplicate proof node"
        );
        support_count = support_count
            .checked_add(node.supports.len())
            .context("proof support count overflow")?;
        ensure!(support_count <= MAX_SUPPORTS, "too many proof supports");
        let mut alternatives = BTreeSet::new();
        for support in &node.supports {
            ensure!(
                support.len() <= MAX_PREMISES_PER_SUPPORT,
                "too many premises in support"
            );
            premise_count = premise_count
                .checked_add(support.len())
                .context("proof premise count overflow")?;
            ensure!(
                premise_count <= MAX_PREMISE_REFERENCES,
                "too many proof premise references"
            );
            let mut premises = BTreeSet::new();
            for premise in support {
                charge_id(premise, &mut bytes)?;
                ensure!(
                    premises.insert(premise.as_str()),
                    "duplicate premise in support"
                );
            }
            ensure!(
                alternatives.insert(premises),
                "duplicate alternative support"
            );
        }
    }
    let mut source_ids = BTreeSet::new();
    let mut total_cost = 0u64;
    let mut sources: Vec<_> = problem.sources.iter().collect();
    for source in &sources {
        charge_id(&source.id, &mut bytes)?;
        ensure!(
            source_ids.insert(source.id.as_str()),
            "duplicate primitive source"
        );
        let node = &problem.nodes[*ids
            .get(source.id.as_str())
            .context("unknown primitive source")?];
        ensure!(
            node.supports.len() == 1 && node.supports[0].is_empty(),
            "primitive source must have exactly one empty support"
        );
        total_cost = total_cost
            .checked_add(source.cost)
            .context("primitive source costs overflow")?;
    }
    sources.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    let mut select = |labels: &[String]| -> Result<Vec<usize>> {
        let mut seen = BTreeSet::new();
        labels
            .iter()
            .map(|id| {
                charge_id(id, &mut bytes)?;
                ensure!(seen.insert(id.as_str()), "duplicate selected proof node");
                ids.get(id.as_str())
                    .copied()
                    .context("unknown selected proof node")
            })
            .collect()
    };
    let targets = select(&problem.targets)?;
    let protected = select(&problem.protected)?;
    let mut source_bits = vec![None; problem.nodes.len()];
    for (bit, source) in sources.iter().enumerate() {
        source_bits[ids[source.id.as_str()]] = Some(1u32 << bit);
    }
    let mut rules = Vec::with_capacity(support_count);
    let mut dependents = vec![Vec::new(); problem.nodes.len()];
    for (head, node) in problem.nodes.iter().enumerate() {
        for (support_index, support) in node.supports.iter().enumerate() {
            let rule = rules.len();
            for premise in support {
                let index = *ids.get(premise.as_str()).context("unknown proof premise")?;
                dependents[index].push(rule);
            }
            rules.push(Rule {
                head,
                support_index,
                premise_count: support.len(),
            });
        }
    }
    Ok(Compiled {
        problem,
        rules,
        dependents,
        source_bits,
        sources,
        targets,
        protected,
    })
}

struct Budget {
    used: u64,
    limit: u64,
}

impl Budget {
    fn charge(&mut self) -> bool {
        if self.used == self.limit {
            false
        } else {
            self.used += 1;
            true
        }
    }
}

struct Evaluation {
    remaining: Vec<usize>,
    derived: Vec<bool>,
    witnesses: Vec<Option<usize>>,
    order: Vec<usize>,
}

impl Evaluation {
    fn new(graph: &Compiled<'_>) -> Self {
        Self {
            remaining: vec![0; graph.rules.len()],
            derived: vec![false; graph.problem.nodes.len()],
            witnesses: vec![None; graph.problem.nodes.len()],
            order: Vec::with_capacity(graph.problem.nodes.len()),
        }
    }

    fn derive(&mut self, node: usize, rule: usize) {
        if !self.derived[node] {
            self.derived[node] = true;
            self.witnesses[node] = Some(rule);
            self.order.push(node);
        }
    }

    fn run(&mut self, graph: &Compiled<'_>, removed: u32, budget: &mut Budget) -> bool {
        self.order.clear();
        for node in 0..self.derived.len() {
            if !budget.charge() {
                return false;
            }
            self.derived[node] = false;
            self.witnesses[node] = None;
        }
        for (index, rule) in graph.rules.iter().enumerate() {
            if !budget.charge() {
                return false;
            }
            self.remaining[index] = rule.premise_count;
            if rule.premise_count == 0
                && graph.source_bits[rule.head].is_none_or(|bit| removed & bit == 0)
            {
                self.derive(rule.head, index);
            }
        }
        let mut cursor = 0;
        while cursor < self.order.len() {
            let node = self.order[cursor];
            cursor += 1;
            for &rule_index in &graph.dependents[node] {
                if !budget.charge() {
                    return false;
                }
                let remaining = &mut self.remaining[rule_index];
                assert!(*remaining > 0, "each distinct premise propagates once");
                *remaining -= 1;
                if *remaining == 0 {
                    self.derive(graph.rules[rule_index].head, rule_index);
                }
            }
        }
        true
    }

    fn feasible(&self, graph: &Compiled<'_>) -> bool {
        graph.targets.iter().all(|&node| !self.derived[node])
            && graph.protected.iter().all(|&node| self.derived[node])
    }

    fn evidence(&self, graph: &Compiled<'_>) -> DerivabilityEvidence {
        let selected = |nodes: &[usize]| {
            nodes
                .iter()
                .map(|&node| NodeDerivability {
                    node_id: graph.problem.nodes[node].id.clone(),
                    derivable: self.derived[node],
                })
                .collect()
        };
        DerivabilityEvidence {
            witnesses: self
                .order
                .iter()
                .map(|&node| {
                    let rule =
                        &graph.rules[self.witnesses[node].expect("derived node has witness")];
                    DerivationWitness {
                        node_id: graph.problem.nodes[node].id.clone(),
                        support_index: rule.support_index,
                        premises: graph.problem.nodes[node].supports[rule.support_index].clone(),
                    }
                })
                .collect(),
            targets: selected(&graph.targets),
            protected: selected(&graph.protected),
        }
    }
}

/// Search every source subset under the configured limits. Malformed or oversized input is
/// an error. Budget exhaustion can return the best fully evaluated feasible plan so far;
/// only Solved certifies minimum cost, and only Infeasible certifies that no plan exists.
/// Ties prefer fewer deletions and then lexicographically sorted source IDs. Input bounds
/// cover planner-owned allocations; callers must separately bound any serialized input.
pub fn plan_forgetting(problem: &ProofProblem, limits: SearchLimits) -> Result<ProofCutResult> {
    ensure!(
        limits.max_sources <= MAX_SOURCES,
        "source search limit exceeds hard bound"
    );
    ensure!(
        limits.max_steps <= MAX_SEARCH_STEPS,
        "step search limit exceeds hard bound"
    );
    let graph = compile(problem)?;
    let total_candidates = 1u64 << graph.sources.len();
    let mut result = ProofCutResult {
        status: ProofCutStatus::BudgetExhausted,
        plan: None,
        baseline: None,
        completed_candidates: 0,
        total_candidates,
        steps_used: 0,
    };
    if graph.sources.len() > limits.max_sources {
        return Ok(result);
    }
    let mut budget = Budget {
        used: 0,
        limit: limits.max_steps,
    };
    let mut evaluation = Evaluation::new(&graph);
    for candidate in 0..total_candidates {
        if !budget.charge() || !evaluation.run(&graph, candidate as u32, &mut budget) {
            result.steps_used = budget.used;
            return Ok(result);
        }
        result.completed_candidates += 1;
        if candidate == 0 {
            result.baseline = Some(evaluation.evidence(&graph));
        }
        if !evaluation.feasible(&graph) {
            continue;
        }
        let mut cost = 0u64;
        let mut removed_sources = Vec::new();
        for (bit, source) in graph.sources.iter().enumerate() {
            if candidate & (1u64 << bit) != 0 {
                cost = cost
                    .checked_add(source.cost)
                    .context("candidate deletion cost overflow")?;
                removed_sources.push(source.id.clone());
            }
        }
        let better = result.plan.as_ref().is_none_or(|best| {
            (cost, removed_sources.len(), &removed_sources)
                < (best.cost, best.removed_sources.len(), &best.removed_sources)
        });
        if better {
            result.plan = Some(ProofCutPlan {
                removed_sources,
                cost,
                evidence: evaluation.evidence(&graph),
            });
        }
    }
    result.status = if result.plan.is_some() {
        ProofCutStatus::Solved
    } else {
        ProofCutStatus::Infeasible
    };
    result.steps_used = budget.used;
    Ok(result)
}
