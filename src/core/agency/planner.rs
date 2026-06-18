// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Backward-chaining planner over causal triples.
//!
//! Starting from a goal state, finds triples where the object matches the goal
//! and recursively plans to achieve the subject. Produces an ordered list of
//! actions (triples to make true).

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
}

/// A complete plan: an ordered sequence of actions to achieve a goal.
#[derive(Clone, Debug)]
pub struct Plan {
    pub goal: String,
    pub actions: Vec<PlannedAction>,
    pub complete: bool,
}

/// Backward-chaining planner over the triple store.
pub struct Planner;

impl Planner {
    /// Plan how to achieve a goal state by backward-chaining.
    ///
    /// Looks for triples where `object == goal` and treats the subject as a
    /// subgoal. Recurses up to `max_depth`. A plan is "complete" if every
    /// leaf subgoal either already exists as a subject in the triple store
    /// or has no further causal chain.
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

        let complete = Self::chain_step(
            triple_store,
            goal,
            causal_relations,
            max_depth,
            0,
            &mut actions,
            &mut visited,
        );

        // Reverse so leaf actions come first (execution order)
        actions.reverse();

        Plan {
            goal: goal.to_string(),
            actions,
            complete,
        }
    }

    fn chain_step(
        triple_store: &TripleStore,
        target: &str,
        causal_relations: &[&str],
        max_depth: usize,
        depth: usize,
        actions: &mut Vec<PlannedAction>,
        visited: &mut FxHashSet<String>,
    ) -> bool {
        if depth >= max_depth {
            return false;
        }

        // Find triples where object == target
        let candidates = triple_store.query(None, None, Some(target));
        let filtered: Vec<_> = if causal_relations.is_empty() {
            candidates
        } else {
            candidates
                .into_iter()
                .filter(|t| causal_relations.contains(&t.relation_id.as_str()))
                .collect()
        };

        if filtered.is_empty() {
            // No causal chain found; check if target exists as a known entity
            let as_subject = triple_store.query(Some(target), None, None);
            return !as_subject.is_empty();
        }

        let mut any_complete = false;

        for t in &filtered {
            actions.push(PlannedAction {
                subject: t.subject_id.clone(),
                relation: t.relation_id.clone(),
                object: t.object_id.clone(),
                depth,
            });

            // Recurse on the subject (precondition) if not already visited
            if !visited.contains(&t.subject_id) {
                visited.insert(t.subject_id.clone());
                let sub_complete = Self::chain_step(
                    triple_store,
                    &t.subject_id,
                    causal_relations,
                    max_depth,
                    depth + 1,
                    actions,
                    visited,
                );
                if sub_complete {
                    any_complete = true;
                }
            } else {
                any_complete = true;
            }
        }

        any_complete
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
}
