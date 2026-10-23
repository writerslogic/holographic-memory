// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Pattern scanning over the triple store.
//!
//! Groups triples by relation, counts co-occurring subject/object atoms,
//! and surfaces recurring structural regularities.

use crate::core::hash::FastSet;

use crate::core::triple_store::TripleStore;

/// A recurring pattern: entities sharing the same relation with similar peers.
#[derive(Clone, Debug)]
pub struct RelationPattern {
    /// The relation this pattern was found on.
    pub relation: String,
    /// Subject atoms that appear in this relation.
    pub subjects: Vec<String>,
    /// Object atoms that appear in this relation.
    pub objects: Vec<String>,
    /// How many triples use this relation.
    pub frequency: usize,
}

/// Co-occurrence of two entities appearing together across triples.
#[derive(Clone, Debug)]
pub struct CoOccurrence {
    pub entity_a: String,
    pub entity_b: String,
    /// Number of triples where both appear (as subject/object in same relation group).
    pub count: usize,
}

/// Finds recurring structural patterns in a triple store using read-only access.
pub struct PatternScanner;

impl PatternScanner {
    /// Group all triples by relation and return patterns with frequency >= `min_freq`.
    pub fn scan_relation_patterns(
        triple_store: &TripleStore,
        min_freq: usize,
    ) -> Vec<RelationPattern> {
        let mut patterns = Vec::new();
        for relation in triple_store.relation_ids() {
            let mut subjects = FastSet::default();
            let mut objects = FastSet::default();
            let mut frequency = 0;
            triple_store.visit(None, Some(&relation), None, |t| {
                frequency += 1;
                subjects.insert(t.subject_id.clone());
                objects.insert(t.object_id.clone());
                true
            });
            if frequency == 0 || frequency < min_freq {
                continue;
            }
            let mut subjects: Vec<_> = subjects.into_iter().collect();
            let mut objects: Vec<_> = objects.into_iter().collect();
            subjects.sort();
            objects.sort();
            patterns.push(RelationPattern {
                relation,
                subjects,
                objects,
                frequency,
            });
        }

        patterns.sort_by_key(|p| std::cmp::Reverse(p.frequency));
        patterns
    }

    /// Find entities that co-occur as subjects across multiple relations.
    /// Two subjects co-occur if they share at least `min_shared` relations.
    pub fn find_co_occurrences(triple_store: &TripleStore, min_shared: usize) -> Vec<CoOccurrence> {
        let mut entities = Vec::new();
        for entity in triple_store.subject_ids() {
            let mut relations = FastSet::default();
            triple_store.visit(Some(&entity), None, None, |t| {
                relations.insert(t.relation_id.clone());
                true
            });
            if relations.is_empty() {
                continue;
            }
            let mut relations: Vec<_> = relations.into_iter().collect();
            relations.sort();
            entities.push((entity, relations));
        }
        let mut results = Vec::new();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let shared = count_shared_sorted(&entities[i].1, &entities[j].1);
                if shared >= min_shared {
                    results.push(CoOccurrence {
                        entity_a: entities[i].0.clone(),
                        entity_b: entities[j].0.clone(),
                        count: shared,
                    });
                }
            }
        }

        results.sort_by_key(|r| std::cmp::Reverse(r.count));
        results
    }
}

/// Count shared elements between two sorted slices.
fn count_shared_sorted(a: &[String], b: &[String]) -> usize {
    let (mut i, mut j) = (0, 0);
    let mut count = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> TripleStore {
        let store = TripleStore::new();
        store.add("paris", "capital_of", "france", "c1");
        store.add("berlin", "capital_of", "germany", "c2");
        store.add("tokyo", "capital_of", "japan", "c3");
        store.add("paris", "located_in", "europe", "c4");
        store.add("berlin", "located_in", "europe", "c5");
        store
    }

    #[test]
    fn test_scan_relation_patterns() {
        let store = make_store();
        let patterns = PatternScanner::scan_relation_patterns(&store, 2);
        assert!(!patterns.is_empty());
        let capital = patterns
            .iter()
            .find(|p| p.relation == "capital_of")
            .unwrap();
        assert_eq!(capital.frequency, 3);
        assert_eq!(capital.subjects.len(), 3);
        assert_eq!(capital.objects.len(), 3);
    }

    #[test]
    fn test_scan_below_threshold() {
        let store = make_store();
        let patterns = PatternScanner::scan_relation_patterns(&store, 10);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_co_occurrences() {
        let store = make_store();
        // paris and berlin both have capital_of and located_in
        let co = PatternScanner::find_co_occurrences(&store, 2);
        assert!(!co.is_empty());
        let pair = co.iter().find(|c| {
            (c.entity_a == "paris" && c.entity_b == "berlin")
                || (c.entity_a == "berlin" && c.entity_b == "paris")
        });
        assert!(pair.is_some());
        assert_eq!(pair.unwrap().count, 2);
    }

    #[test]
    fn test_co_occurrences_empty() {
        let store = TripleStore::new();
        let co = PatternScanner::find_co_occurrences(&store, 1);
        assert!(co.is_empty());
    }
}
