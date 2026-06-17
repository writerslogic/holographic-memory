// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

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
        let idx = rules.len();
        if rule.input_relations.len() == 2 {
            self.by_chain.write().insert(
                (
                    rule.input_relations[0].clone(),
                    rule.input_relations[1].clone(),
                ),
                idx,
            );
        }
        rules.push(rule);
        idx
    }

    pub fn find_rule(&self, rel1: &str, rel2: &str) -> Option<CompositionRule> {
        let by_chain = self.by_chain.read();
        let idx = by_chain.get(&(rel1.to_string(), rel2.to_string()))?;
        self.rules.read().get(*idx).cloned()
    }

    pub fn all_rules(&self) -> Vec<CompositionRule> {
        self.rules.read().clone()
    }

    pub fn count(&self) -> usize {
        self.rules.read().len()
    }

    pub fn load_rule(&self, rule: CompositionRule) {
        self.add_rule(rule);
    }

    pub fn serialize_rule(rule: &CompositionRule) -> Vec<u8> {
        let json = serde_json::to_vec(rule).unwrap_or_default();
        let mut buf = Vec::with_capacity(1 + 4 + json.len());
        buf.push(0xFE);
        buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);
        buf
    }

    pub fn deserialize_rule(data: &[u8]) -> Option<CompositionRule> {
        if data.len() < 5 || data[0] != 0xFE {
            return None;
        }
        let len = u32::from_le_bytes(data[1..5].try_into().ok()?) as usize;
        let json = data.get(5..5 + len)?;
        serde_json::from_slice(json).ok()
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

    #[test]
    fn test_rule_serialize_roundtrip() {
        let rule = CompositionRule {
            name: "test_rule".to_string(),
            input_relations: vec!["r1".to_string(), "r2".to_string()],
            output_relation: "r3".to_string(),
        };
        let data = RuleStore::serialize_rule(&rule);
        let parsed = RuleStore::deserialize_rule(&data).unwrap();
        assert_eq!(parsed.name, "test_rule");
        assert_eq!(parsed.output_relation, "r3");
    }
}
