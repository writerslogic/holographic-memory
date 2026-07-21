// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::wire;
use fxhash::FxHashMap;
use parking_lot::RwLock;

const TRIPLE_MAGIC: u8 = wire::magic::TRIPLE;

#[derive(Clone, Debug)]
pub struct TripleRecord {
    pub subject_id: String,
    pub relation_id: String,
    pub object_id: String,
    pub composite_id: String,
    pub deleted: bool,
}

pub struct TripleStore {
    triples: RwLock<Vec<TripleRecord>>,
    by_subject: RwLock<FxHashMap<String, Vec<usize>>>,
    by_relation: RwLock<FxHashMap<String, Vec<usize>>>,
    by_object: RwLock<FxHashMap<String, Vec<usize>>>,
    by_composite: RwLock<FxHashMap<String, Vec<usize>>>,
}

impl TripleStore {
    pub fn new() -> Self {
        Self {
            triples: RwLock::new(Vec::new()),
            by_subject: RwLock::new(FxHashMap::default()),
            by_relation: RwLock::new(FxHashMap::default()),
            by_object: RwLock::new(FxHashMap::default()),
            by_composite: RwLock::new(FxHashMap::default()),
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

    pub fn query(
        &self,
        subject: Option<&str>,
        relation: Option<&str>,
        object: Option<&str>,
    ) -> Vec<TripleRecord> {
        let triples = self.triples.read();

        let candidate_indices: Option<Vec<usize>> = if let Some(s) = subject {
            self.by_subject.read().get(s).cloned()
        } else if let Some(r) = relation {
            self.by_relation.read().get(r).cloned()
        } else if let Some(o) = object {
            self.by_object.read().get(o).cloned()
        } else {
            None
        };

        let candidates: Vec<&TripleRecord> = match candidate_indices {
            Some(idxs) => idxs.iter().map(|&i| &triples[i]).collect(),
            None => triples.iter().collect(),
        };

        candidates
            .into_iter()
            .filter(|t| {
                !t.deleted
                    && subject.is_none_or(|s| t.subject_id == s)
                    && relation.is_none_or(|r| t.relation_id == r)
                    && object.is_none_or(|o| t.object_id == o)
            })
            .cloned()
            .collect()
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

    #[allow(dead_code)]
    pub fn all_for_subject(&self, subject: &str) -> Vec<TripleRecord> {
        self.query(Some(subject), None, None)
    }

    #[allow(dead_code)]
    pub fn all_for_relation(&self, relation: &str) -> Vec<TripleRecord> {
        self.query(None, Some(relation), None)
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

    pub fn load_triple(&self, record: TripleRecord) {
        self.add(
            &record.subject_id,
            &record.relation_id,
            &record.object_id,
            &record.composite_id,
        );
    }

    pub fn serialize_triple(record: &TripleRecord) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(TRIPLE_MAGIC);
        for field in &[
            &record.subject_id,
            &record.relation_id,
            &record.object_id,
            &record.composite_id,
        ] {
            wire::write_lp_str(&mut buf, field);
        }
        buf
    }

    pub fn deserialize_triple(data: &[u8]) -> Option<TripleRecord> {
        if data.is_empty() || data[0] != TRIPLE_MAGIC {
            return None;
        }
        let mut pos = 1;
        let mut fields = Vec::with_capacity(4);
        for _ in 0..4 {
            let (s, next) = wire::read_lp_str(data, pos)?;
            pos = next;
            fields.push(s);
        }
        Some(TripleRecord {
            subject_id: fields.remove(0),
            relation_id: fields.remove(0),
            object_id: fields.remove(0),
            composite_id: fields.remove(0),
            deleted: false,
        })
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

        let all_paris = store.all_for_subject("paris");
        assert_eq!(all_paris.len(), 2);

        let all_capital = store.all_for_relation("capital_of");
        assert_eq!(all_capital.len(), 2);
    }

    #[test]
    fn test_triple_remove() {
        let store = TripleStore::new();
        store.add("a", "r", "b", "c1");
        assert_eq!(store.count(), 1);
        assert!(store.remove("a", "r", "b"));
        assert_eq!(store.count(), 0);
        assert!(!store.remove("a", "r", "b"));
    }

    #[test]
    fn test_triple_serialize_roundtrip() {
        let record = TripleRecord {
            subject_id: "paris".to_string(),
            relation_id: "capital_of".to_string(),
            object_id: "france".to_string(),
            composite_id: "comp_1".to_string(),
            deleted: false,
        };
        let data = TripleStore::serialize_triple(&record);
        let parsed = TripleStore::deserialize_triple(&data).unwrap();
        assert_eq!(parsed.subject_id, "paris");
        assert_eq!(parsed.relation_id, "capital_of");
        assert_eq!(parsed.object_id, "france");
        assert_eq!(parsed.composite_id, "comp_1");
    }
}
