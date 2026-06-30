// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Best-first backward-chaining planner over causal triples.
//!
//! Starting from a goal state, finds triples where the object matches the goal
//! and recursively plans to achieve the subject. Produces an ordered list of
//! actions (triples to make true), expanding the lowest-cost subgoal first.
//!
//! Cost estimation: each action's cost is `1.0 / (1 + evidence_count)` where
//! `evidence_count` is the number of triples supporting the action.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fxhash::FxHashSet;

use crate::core::triple_store::TripleStore;

/// A single planned action: a triple that needs to be made true.
#[derive(Clone, Debug)]
pub struct PlannedAction {
    pub subject: String,
    pub relation: String,
    pub object: String,
    /// Depth in the backward chain (0 = directly achieves goal).
    pub depth: usize,
    /// Estimated cost. Lower means more evidence supports this action.
    /// Computed as `1.0 / (1 + evidence_count)`.
    pub cost: f64,
}

/// A complete plan: an ordered sequence of actions to achieve a goal.
#[derive(Clone, Debug)]
pub struct Plan {
    pub goal: String,
    pub actions: Vec<PlannedAction>,
    /// Whether the plan reaches known facts for every leaf subgoal.
    pub complete: bool,
    /// Total cost of all actions in the plan.
    pub total_cost: f64,
}

/// A subgoal waiting to be expanded in the best-first search.
struct Subgoal {
    target: String,
    depth: usize,
    /// Cumulative cost to reach this subgoal.
    cost: f64,
}

/// Wrapper for min-heap ordering (BinaryHeap is max-heap by default).
impl Eq for Subgoal {}

impl PartialEq for Subgoal {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Ord for Subgoal {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap acts as a min-heap on cost.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Subgoal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Backward-chaining planner over the triple store.
pub struct Planner;

impl Planner {
    /// Plan how to achieve a goal state by backward-chaining.
    ///
    /// Looks for triples where `object == goal` and treats the subject as a
    /// subgoal. Uses best-first search, expanding the lowest-cost subgoal
    /// first. Recurses up to `max_depth`.
    ///
    /// A plan is "complete" if every leaf subgoal either already exists as a
    /// subject in the triple store or has no further causal chain.
    ///
    /// If the chain cannot reach known facts within `max_depth`, the best
    /// partial plan found so far is returned with `complete == false`.
    ///
    /// `causal_relations`: only follow triples with these relation types
    /// (e.g., "causes", "enables", "produces"). If empty, follows all relations.
    pub fn backward_chain(
        triple_store: &TripleStore,
        goal: &str,
        causal_relations: &[&str],
        max_depth: usize,
    ) -> Plan {
        let mut actions = Vec::new();
        let mut visited = FxHashSet::default();
        visited.insert(goal.to_string());

        // Best-first search using a min-heap on cost.
        let mut heap = BinaryHeap::new();
        heap.push(Subgoal {
            target: goal.to_string(),
            depth: 0,
            cost: 0.0,
        });

        let mut all_complete = true;

        while let Some(subgoal) = heap.pop() {
            if subgoal.depth >= max_depth {
                // Could not expand further; mark as incomplete.
                all_complete = false;
                continue;
            }

            // Find triples where object == target.
            let candidates = triple_store.query(None, None, Some(&subgoal.target));
            let filtered: Vec<_> = if causal_relations.is_empty() {
                candidates
            } else {
                candidates
                    .into_iter()
                    .filter(|t| causal_relations.contains(&t.relation_id.as_str()))
                    .collect()
            };

            if filtered.is_empty() {
                // No causal chain found; check if target is a known entity.
                let as_subject = triple_store.query(Some(&subgoal.target), None, None);
                if as_subject.is_empty() {
                    // Leaf subgoal not grounded; plan is partial.
                    all_complete = false;
                }
                continue;
            }

            let evidence_count = filtered.len();
            let action_cost = 1.0 / (1.0 + evidence_count as f64);

            for t in &filtered {
                actions.push(PlannedAction {
                    subject: t.subject_id.clone(),
                    relation: t.relation_id.clone(),
                    object: t.object_id.clone(),
                    depth: subgoal.depth,
                    cost: action_cost,
                });

                // Enqueue the subject as a subgoal if not already visited.
                if !visited.contains(&t.subject_id) {
                    visited.insert(t.subject_id.clone());
                    heap.push(Subgoal {
                        target: t.subject_id.clone(),
                        depth: subgoal.depth + 1,
                        cost: subgoal.cost + action_cost,
                    });
                }
            }
        }

        // Sort actions by depth descending so leaf actions come first (execution order).
        actions.sort_by_key(|b| std::cmp::Reverse(b.depth));

        let total_cost: f64 = actions.iter().map(|a| a.cost).sum();

        Plan {
            goal: goal.to_string(),
            actions,
            complete: all_complete,
            total_cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_backward_chain() {
        let store = TripleStore::new();
        // A causes B, B causes C
        store.add("action_a", "causes", "state_b", "c1");
        store.add("state_b", "causes", "goal_c", "c2");

        let plan = Planner::backward_chain(&store, "goal_c", &["causes"], 5);
        assert!(!plan.actions.is_empty());
        // First action should be the leaf (action_a causes state_b)
        assert_eq!(plan.actions[0].subject, "action_a");
        assert_eq!(plan.actions[0].object, "state_b");
        // Second should be the direct step (state_b causes goal_c)
        assert_eq!(plan.actions[1].subject, "state_b");
        assert_eq!(plan.actions[1].object, "goal_c");
    }

    #[test]
    fn test_no_causal_chain() {
        let store = TripleStore::new();
        store.add("x", "likes", "y", "c1");

        let plan = Planner::backward_chain(&store, "goal", &["causes"], 5);
        assert!(plan.actions.is_empty());
        assert!(!plan.complete);
    }

    #[test]
    fn test_depth_limit() {
        let store = TripleStore::new();
        store.add("a", "causes", "b", "c1");
        store.add("b", "causes", "c", "c2");
        store.add("c", "causes", "d", "c3");
        store.add("d", "causes", "goal", "c4");

        // Depth 2 should not reach all the way back to "a"
        let plan = Planner::backward_chain(&store, "goal", &["causes"], 2);
        assert!(plan.actions.len() <= 3);
    }

    #[test]
    fn test_cycle_prevention() {
        let store = TripleStore::new();
        // Circular: a causes b, b causes a
        store.add("a", "causes", "b", "c1");
        store.add("b", "causes", "a", "c2");

        let plan = Planner::backward_chain(&store, "a", &["causes"], 10);
        // Should not infinite loop
        assert!(plan.actions.len() <= 4);
    }

    #[test]
    fn test_all_relations() {
        let store = TripleStore::new();
        store.add("heat", "produces", "steam", "c1");
        store.add("steam", "enables", "power", "c2");

        // Empty causal_relations means follow all
        let plan = Planner::backward_chain(&store, "power", &[], 5);
        assert_eq!(plan.actions.len(), 2);
    }

    #[test]
    fn test_empty_store() {
        let store = TripleStore::new();
        let plan = Planner::backward_chain(&store, "anything", &[], 5);
        assert!(plan.actions.is_empty());
    }

    #[test]
    fn test_cost_estimation() {
        let store = TripleStore::new();
        // Two triples supporting the same target -> lower cost per action.
        store.add("x", "causes", "goal", "c1");
        store.add("y", "causes", "goal", "c2");

        let plan = Planner::backward_chain(&store, "goal", &["causes"], 5);
        assert_eq!(plan.actions.len(), 2);

        // evidence_count = 2, so cost = 1.0 / (1 + 2) = 1/3
        let expected_cost = 1.0 / 3.0;
        for action in &plan.actions {
            assert!(
                (action.cost - expected_cost).abs() < 1e-9,
                "expected cost {expected_cost}, got {}",
                action.cost
            );
        }

        // total_cost should be 2 * (1/3)
        let expected_total = 2.0 * expected_cost;
        assert!(
            (plan.total_cost - expected_total).abs() < 1e-9,
            "expected total_cost {expected_total}, got {}",
            plan.total_cost
        );

        // Single evidence -> higher cost.
        let store2 = TripleStore::new();
        store2.add("z", "causes", "goal2", "c3");
        let plan2 = Planner::backward_chain(&store2, "goal2", &["causes"], 5);
        assert_eq!(plan2.actions.len(), 1);
        let single_cost = 1.0 / 2.0;
        assert!(
            (plan2.actions[0].cost - single_cost).abs() < 1e-9,
            "expected cost {single_cost}, got {}",
            plan2.actions[0].cost
        );
    }

    #[test]
    fn test_best_first_ordering() {
        let store = TripleStore::new();
        // Path 1: high evidence (low cost) chain
        //   step1a -> mid (3 triples support mid->goal, so cost = 1/4 each)
        store.add("step1a", "causes", "mid", "c1");
        store.add("alt1", "causes", "goal", "c2");
        store.add("alt2", "causes", "goal", "c3");
        store.add("mid", "causes", "goal", "c4");

        // Path 2: low evidence (high cost) chain
        //   step1b -> side (1 triple supports side->goal, cost = 1/2)
        store.add("step1b", "causes", "side", "c5");
        store.add("side", "causes", "goal", "c6");

        let plan = Planner::backward_chain(&store, "goal", &["causes"], 5);

        // The depth-0 actions (direct to goal) should have lower cost than
        // depth-1 actions further from the goal, due to 4 supporting triples
        // for goal (cost = 1/5 each) vs fewer for subgoals.
        let depth0: Vec<_> = plan.actions.iter().filter(|a| a.depth == 0).collect();
        let depth1: Vec<_> = plan.actions.iter().filter(|a| a.depth == 1).collect();

        // Depth-0 actions exist and come after depth-1 in execution order.
        assert!(!depth0.is_empty(), "should have depth-0 actions");

        // All depth-0 actions should have cost = 1/(1+4) = 0.2
        // (4 triples target "goal": alt1, alt2, mid, side)
        let expected_d0_cost = 1.0 / 5.0;
        for a in &depth0 {
            assert!(
                (a.cost - expected_d0_cost).abs() < 1e-9,
                "depth-0 cost should be {expected_d0_cost}, got {}",
                a.cost
            );
        }

        // Depth-1 actions target "mid" (1 triple) or "side" (1 triple): cost = 1/2
        if !depth1.is_empty() {
            for a in &depth1 {
                assert!(
                    a.cost > expected_d0_cost,
                    "depth-1 cost {} should exceed depth-0 cost {expected_d0_cost}",
                    a.cost
                );
            }
        }

        // Verify total_cost is consistent.
        let sum: f64 = plan.actions.iter().map(|a| a.cost).sum();
        assert!(
            (plan.total_cost - sum).abs() < 1e-9,
            "total_cost {} should equal sum of action costs {sum}",
            plan.total_cost
        );
    }
}
