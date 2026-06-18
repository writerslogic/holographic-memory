// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Goal storage and utility scoring.
//!
//! Goals are hypervectors composed from desired state descriptions.
//! Each goal has a utility score: `u(goal) = relevance * urgency * (1 - cost)`.
//! GoalStore is separate from AtomMemory to keep goal vectors distinct from
//! knowledge atoms.

use fxhash::FxHashMap;
use parking_lot::RwLock;

use crate::core::entangled::EntangledHVec;

/// A goal with its metadata and utility components.
#[derive(Clone, Debug)]
pub struct Goal {
    pub name: String,
    pub description: String,
    pub vector: EntangledHVec,
    pub relevance: f64,
    pub urgency: f64,
    pub cost: f64,
    pub active: bool,
}

impl Goal {
    pub fn utility(&self) -> f64 {
        self.relevance * self.urgency * (1.0 - self.cost.clamp(0.0, 0.99))
    }
}

struct GoalStoreInner {
    goals: Vec<Goal>,
    by_name: FxHashMap<String, usize>,
}

/// Stores goals separately from the knowledge base.
pub struct GoalStore {
    inner: RwLock<GoalStoreInner>,
}

impl Default for GoalStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalStore {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(GoalStoreInner {
                goals: Vec::new(),
                by_name: FxHashMap::default(),
            }),
        }
    }

    pub fn add(&self, goal: Goal) -> usize {
        let mut inner = self.inner.write();
        let idx = inner.goals.len();
        inner.by_name.insert(goal.name.clone(), idx);
        inner.goals.push(goal);
        idx
    }

    pub fn get(&self, name: &str) -> Option<Goal> {
        let inner = self.inner.read();
        let idx = *inner.by_name.get(name)?;
        inner.goals.get(idx).cloned()
    }

    pub fn deactivate(&self, name: &str) -> bool {
        let mut inner = self.inner.write();
        if let Some(&idx) = inner.by_name.get(name) {
            inner.goals[idx].active = false;
            true
        } else {
            false
        }
    }

    pub fn active_goals(&self) -> Vec<Goal> {
        self.inner
            .read()
            .goals
            .iter()
            .filter(|g| g.active)
            .cloned()
            .collect()
    }

    /// Return active goals sorted by utility (highest first).
    pub fn prioritized(&self) -> Vec<Goal> {
        let mut active = self.active_goals();
        active.sort_by(|a, b| {
            b.utility()
                .partial_cmp(&a.utility())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        active
    }

    /// Compute overlap between a vector and the highest-priority active goal.
    /// Returns 0.0 if no active goals.
    pub fn goal_relevance(&self, vec: &EntangledHVec) -> f64 {
        let goals = self.prioritized();
        if goals.is_empty() {
            return 0.0;
        }
        goals[0].vector.similarity(vec)
    }

    pub fn count(&self) -> usize {
        self.inner.read().goals.len()
    }

    pub fn active_count(&self) -> usize {
        self.inner.read().goals.iter().filter(|g| g.active).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_goal(name: &str, seed: u64, relevance: f64, urgency: f64, cost: f64) -> Goal {
        Goal {
            name: name.to_string(),
            description: format!("Goal: {}", name),
            vector: EntangledHVec::new_deterministic(16384, seed),
            relevance,
            urgency,
            cost,
            active: true,
        }
    }

    #[test]
    fn test_utility_scoring() {
        let g = make_goal("test", 1, 0.8, 0.9, 0.2);
        let u = g.utility();
        let expected = 0.8 * 0.9 * (1.0 - 0.2);
        assert!((u - expected).abs() < 0.001);
    }

    #[test]
    fn test_utility_clamps_cost() {
        let g = make_goal("test", 1, 1.0, 1.0, 1.5);
        assert!(g.utility() > 0.0, "cost should be clamped to 0.99");
    }

    #[test]
    fn test_add_get() {
        let store = GoalStore::new();
        store.add(make_goal("learn_rust", 1, 0.9, 0.8, 0.3));
        let g = store.get("learn_rust").unwrap();
        assert_eq!(g.name, "learn_rust");
        assert!(g.active);
    }

    #[test]
    fn test_deactivate() {
        let store = GoalStore::new();
        store.add(make_goal("g1", 1, 1.0, 1.0, 0.0));
        assert_eq!(store.active_count(), 1);
        store.deactivate("g1");
        assert_eq!(store.active_count(), 0);
    }

    #[test]
    fn test_prioritized_ordering() {
        let store = GoalStore::new();
        store.add(make_goal("low", 1, 0.1, 0.1, 0.0));
        store.add(make_goal("high", 2, 1.0, 1.0, 0.0));
        store.add(make_goal("mid", 3, 0.5, 0.5, 0.0));
        let pri = store.prioritized();
        assert_eq!(pri[0].name, "high");
        assert_eq!(pri[1].name, "mid");
        assert_eq!(pri[2].name, "low");
    }

    #[test]
    fn test_goal_relevance() {
        let store = GoalStore::new();
        let g = make_goal("g1", 42, 1.0, 1.0, 0.0);
        let vec = g.vector.clone();
        store.add(g);
        let rel = store.goal_relevance(&vec);
        assert!(
            (rel - 1.0).abs() < 0.001,
            "Same vector should have relevance ~1.0"
        );
    }

    #[test]
    fn test_empty_store() {
        let store = GoalStore::new();
        assert_eq!(store.count(), 0);
        assert_eq!(store.active_count(), 0);
        assert!(store.get("x").is_none());
        let v = EntangledHVec::new_deterministic(16384, 1);
        assert_eq!(store.goal_relevance(&v), 0.0);
    }
}
