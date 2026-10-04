// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use fxhash::FxHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositionRule {
    pub name: String,
    pub input_relations: Vec<String>,
    pub output_relation: String,
}

pub struct RuleStore {
    rules: RwLock<Vec<CompositionRule>>,
    by_chain: RwLock<FxHashMap<(String, String), usize>>,
}

impl RuleStore {
    pub fn new() -> Self {
        Self {
            rules: RwLock::new(Vec::new()),
            by_chain: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn add_rule(&self, rule: CompositionRule) -> usize {
        let mut rules = self.rules.write();
        let idx = rules
            .iter()
            .position(|stored| stored.name == rule.name)
            .unwrap_or(rules.len());
        let mut by_chain = self.by_chain.write();
        by_chain.retain(|_, owner| *owner != idx);
        if rule.input_relations.len() == 2 {
            by_chain.insert(
                (
                    rule.input_relations[0].clone(),
                    rule.input_relations[1].clone(),
                ),
                idx,
            );
        }
        if idx == rules.len() {
            rules.push(rule);
        } else {
            rules[idx] = rule;
        }
        idx
    }

    pub fn find_rule(&self, rel1: &str, rel2: &str) -> Option<CompositionRule> {
        let rules = self.rules.read();
        let by_chain = self.by_chain.read();
        let idx = by_chain.get(&(rel1.to_string(), rel2.to_string()))?;
        rules.get(*idx).cloned()
    }

    pub fn all_rules(&self) -> Vec<CompositionRule> {
        self.rules.read().clone()
    }

    pub fn count(&self) -> usize {
        self.rules.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_add_find() {
        let store = RuleStore::new();
        store.add_rule(CompositionRule {
            name: "grandfather".to_string(),
            input_relations: vec!["father".to_string(), "father".to_string()],
            output_relation: "grandfather".to_string(),
        });

        let found = store.find_rule("father", "father").unwrap();
        assert_eq!(found.output_relation, "grandfather");
        assert!(store.find_rule("mother", "father").is_none());
    }
}
