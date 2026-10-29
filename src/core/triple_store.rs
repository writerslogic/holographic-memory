// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::hash::FastMap;
use crate::core::query_limits::{QueryBudget, QueryLimitExceeded};
use crate::core::sync::RwLock;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TripleRecord {
    pub subject_id: String,
    pub relation_id: String,
    pub object_id: String,
    pub composite_id: String,
    pub deleted: bool,
}

pub struct TripleStore {
    triples: RwLock<Vec<TripleRecord>>,
    by_subject: RwLock<FastMap<String, Vec<usize>>>,
    by_relation: RwLock<FastMap<String, Vec<usize>>>,
    by_object: RwLock<FastMap<String, Vec<usize>>>,
    by_composite: RwLock<FastMap<String, Vec<usize>>>,
}

impl TripleStore {
    pub fn new() -> Self {
        Self {
            triples: RwLock::new(Vec::new()),
            by_subject: RwLock::new(FastMap::default()),
            by_relation: RwLock::new(FastMap::default()),
            by_object: RwLock::new(FastMap::default()),
            by_composite: RwLock::new(FastMap::default()),
        }
    }

    pub fn add(&self, subject: &str, relation: &str, object: &str, composite_id: &str) -> usize {
        let mut triples = self.triples.write();
        let idx = triples.len();
        triples.push(TripleRecord {
            subject_id: subject.to_string(),
            relation_id: relation.to_string(),
            object_id: object.to_string(),
            composite_id: composite_id.to_string(),
            deleted: false,
        });
        self.by_subject
            .write()
            .entry(subject.to_string())
            .or_default()
            .push(idx);
        self.by_relation
            .write()
            .entry(relation.to_string())
            .or_default()
            .push(idx);
        self.by_object
            .write()
            .entry(object.to_string())
            .or_default()
            .push(idx);
        self.by_composite
            .write()
            .entry(composite_id.to_string())
            .or_default()
            .push(idx);
        idx
    }

    #[allow(dead_code)]
    pub fn remove(&self, subject: &str, relation: &str, object: &str) -> bool {
        let triples = self.triples.read();
        let indices = self.by_subject.read();
        if let Some(idxs) = indices.get(subject) {
            for &idx in idxs {
                let t = &triples[idx];
                if !t.deleted && t.relation_id == relation && t.object_id == object {
                    drop(triples);
                    drop(indices);
                    self.triples.write()[idx].deleted = true;
                    return true;
                }
            }
        }
        false
    }

    pub fn remove_composite(&self, id: &str) {
        let mut triples = self.triples.write();
        if let Some(indices) = self.by_composite.read().get(id) {
            for &i in indices {
                triples[i].deleted = true;
            }
        }
    }

    pub fn query(
        &self,
        subject: Option<&str>,
        relation: Option<&str>,
        object: Option<&str>,
    ) -> Vec<TripleRecord> {
        let mut result = Vec::new();
        self.visit(subject, relation, object, |t| {
            result.push(t.clone());
            true
        });
        result
    }

    // Callbacks run under read locks and must not reenter the store.
    pub(crate) fn visit(
        &self,
        subject: Option<&str>,
        relation: Option<&str>,
        object: Option<&str>,
        mut visitor: impl FnMut(&TripleRecord) -> bool,
    ) {
        let triples = self.triples.read();
        let mut accept = |t: &TripleRecord| {
            t.deleted
                || subject.is_some_and(|s| t.subject_id != s)
                || relation.is_some_and(|r| t.relation_id != r)
                || object.is_some_and(|o| t.object_id != o)
                || visitor(t)
        };
        let indexed = subject
            .map(|s| (&self.by_subject, s))
            .or_else(|| relation.map(|r| (&self.by_relation, r)))
            .or_else(|| object.map(|o| (&self.by_object, o)));
        if let Some((index, key)) = indexed {
            if let Some(indices) = index.read().get(key) {
                for &i in indices {
                    if !accept(&triples[i]) {
                        break;
                    }
                }
            }
        } else {
            for t in triples.iter() {
                if !accept(t) {
                    break;
                }
            }
        }
    }

    pub(crate) fn has(
        &self,
        subject: Option<&str>,
        relation: Option<&str>,
        object: Option<&str>,
    ) -> bool {
        let mut found = false;
        self.visit(subject, relation, object, |_| {
            found = true;
            false
        });
        found
    }

    pub(crate) fn visit_bounded(
        &self,
        subject: Option<&str>,
        relation: Option<&str>,
        object: Option<&str>,
        budget: &mut QueryBudget,
        mut visitor: impl FnMut(&TripleRecord, &mut QueryBudget) -> Result<(), QueryLimitExceeded>,
    ) -> Result<(), QueryLimitExceeded> {
        let triples = self.triples.read();
        let mut accept = |t: &TripleRecord| {
            budget.work(1)?;
            if !t.deleted
                && subject.is_none_or(|s| t.subject_id == s)
                && relation.is_none_or(|r| t.relation_id == r)
                && object.is_none_or(|o| t.object_id == o)
            {
                visitor(t, budget)?;
            }
            Ok(())
        };
        let indexed = subject
            .map(|s| (&self.by_subject, s))
            .or_else(|| relation.map(|r| (&self.by_relation, r)))
            .or_else(|| object.map(|o| (&self.by_object, o)));
        if let Some((index, key)) = indexed {
            if let Some(indices) = index.read().get(key) {
                for &i in indices {
                    accept(&triples[i])?;
                }
            }
        } else {
            for triple in triples.iter() {
                accept(triple)?;
            }
        }
        Ok(())
    }

    pub(crate) fn has_composite(&self, id: &str) -> bool {
        let triples = self.triples.read();
        self.by_composite
            .read()
            .get(id)
            .is_some_and(|indices| indices.iter().any(|&i| !triples[i].deleted))
    }

    pub(crate) fn subject_ids(&self) -> Vec<String> {
        self.by_subject.read().keys().cloned().collect()
    }
    pub(crate) fn relation_ids(&self) -> Vec<String> {
        self.by_relation.read().keys().cloned().collect()
    }

    pub fn by_composite_id(&self, composite_id: &str) -> Vec<TripleRecord> {
        let triples = self.triples.read();
        let by_comp = self.by_composite.read();
        match by_comp.get(composite_id) {
            Some(idxs) => idxs
                .iter()
                .map(|&i| &triples[i])
                .filter(|t| !t.deleted)
                .cloned()
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn count(&self) -> usize {
        self.triples.read().iter().filter(|t| !t.deleted).count()
    }

    pub fn snapshot(&self) -> Vec<TripleRecord> {
        self.triples
            .read()
            .iter()
            .filter(|t| !t.deleted)
            .cloned()
            .collect()
    }

    pub(crate) fn bounded_snapshot(
        &self,
        budget: &mut QueryBudget,
    ) -> Result<Vec<TripleRecord>, QueryLimitExceeded> {
        let mut snapshot = Vec::new();
        self.visit_bounded(None, None, None, budget, |triple, budget| {
            let bytes = [
                &triple.subject_id,
                &triple.relation_id,
                &triple.object_id,
                &triple.composite_id,
            ]
            .iter()
            .fold(512usize, |total, id| {
                total.saturating_add(id.len().saturating_mul(4))
            });
            budget.reserve(bytes)?;
            snapshot.push(triple.clone());
            Ok(())
        })?;
        Ok(snapshot)
    }

    pub fn load_triple(&self, record: TripleRecord) {
        self.add(
            &record.subject_id,
            &record.relation_id,
            &record.object_id,
            &record.composite_id,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_add_query() {
        let store = TripleStore::new();
        store.add("paris", "capital_of", "france", "c1");
        store.add("berlin", "capital_of", "germany", "c2");
        store.add("paris", "located_in", "europe", "c3");

        let results = store.query(Some("paris"), Some("capital_of"), None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object_id, "france");

        let all_paris = store.query(Some("paris"), None, None);
        assert_eq!(all_paris.len(), 2);

        let all_capital = store.query(None, Some("capital_of"), None);
        assert_eq!(all_capital.len(), 2);
        store.remove_composite("c1");
        assert!(!store.has(Some("paris"), Some("capital_of"), None));
        assert!(!store.has(Some("missing"), None, None));
        let mut seen = 0;
        store.visit(None, None, None, |_| {
            seen += 1;
            false
        });
        assert_eq!(seen, 1);
    }
}
