// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use fxhash::FxHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::types::{GraphPath, PathHop, Relation, RelationType};
use super::wire;

/// Serialized relation for arena persistence.
/// Format: [MAGIC:u8][source_len:u16][source][type_len:u16][type][target_len:u16][target]
///         [valid_from:u64][valid_to:u64][props_len:u32][props_bytes]
///
/// Written under [`wire::magic::RELATION`]. Historically relations shared
/// `0xFE` with composition rules; readers still accept that legacy magic
/// ([`wire::magic::RELATION_LEGACY`]) so pre-migration logs continue to load.
const RELATION_MAGIC: u8 = wire::magic::RELATION;
const RELATION_MAGIC_LEGACY: u8 = wire::magic::RELATION_LEGACY;

/// In-memory graph index over explicit relations.
pub struct RelationStore {
    /// All relations, indexed by position.
    relations: RwLock<Vec<StoredRelation>>,
    /// source_id -> indices into relations vec.
    by_source: RwLock<FxHashMap<String, Vec<usize>>>,
    /// target_id -> indices into relations vec.
    by_target: RwLock<FxHashMap<String, Vec<usize>>>,
    /// relation_type -> indices into relations vec.
    by_type: RwLock<FxHashMap<String, Vec<usize>>>,
    /// Declared relation types with inference properties.
    types: RwLock<FxHashMap<String, RelationType>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredRelation {
    source_id: String,
    relation_type: String,
    target_id: String,
    properties: Option<String>,
    valid_from: f64,
    valid_to: f64,
    deleted: bool,
}

impl StoredRelation {
    fn is_valid_at(&self, timestamp_ms: f64) -> bool {
        if self.deleted {
            return false;
        }
        if timestamp_ms == 0.0 {
            return self.valid_to == 0.0;
        }
        let from_ok = self.valid_from == 0.0 || timestamp_ms >= self.valid_from;
        let to_ok = self.valid_to == 0.0 || timestamp_ms <= self.valid_to;
        from_ok && to_ok
    }

    fn to_relation(&self) -> Relation {
        Relation {
            source_id: self.source_id.clone(),
            relation_type: self.relation_type.clone(),
            target_id: self.target_id.clone(),
            properties: self.properties.clone(),
            valid_from: self.valid_from,
            valid_to: self.valid_to,
        }
    }
}

impl RelationStore {
    pub fn new() -> Self {
        Self {
            relations: RwLock::new(Vec::new()),
            by_source: RwLock::new(FxHashMap::default()),
            by_target: RwLock::new(FxHashMap::default()),
            by_type: RwLock::new(FxHashMap::default()),
            types: RwLock::new(FxHashMap::default()),
        }
    }

    /// Declare a relation type with inference semantics.
    pub fn declare_type(&self, rel_type: RelationType) {
        self.types.write().insert(rel_type.name.clone(), rel_type);
    }

    /// Add a relation to the store. Returns the index.
    pub fn add(&self, rel: &Relation) -> usize {
        let stored = StoredRelation {
            source_id: rel.source_id.clone(),
            relation_type: rel.relation_type.clone(),
            target_id: rel.target_id.clone(),
            properties: rel.properties.clone(),
            valid_from: rel.valid_from,
            valid_to: rel.valid_to,
            deleted: false,
        };

        let mut rels = self.relations.write();
        let idx = rels.len();
        rels.push(stored);

        self.by_source
            .write()
            .entry(rel.source_id.clone())
            .or_default()
            .push(idx);
        self.by_target
            .write()
            .entry(rel.target_id.clone())
            .or_default()
            .push(idx);
        self.by_type
            .write()
            .entry(rel.relation_type.clone())
            .or_default()
            .push(idx);

        idx
    }

    /// Remove a relation by matching source, type, target. Returns true if found.
    pub fn remove(&self, source_id: &str, relation_type: &str, target_id: &str) -> bool {
        let source_indices = self.by_source.read();
        let indices = match source_indices.get(source_id) {
            Some(v) => v.clone(),
            None => return false,
        };
        drop(source_indices);

        let mut rels = self.relations.write();
        for &idx in &indices {
            let r = &mut rels[idx];
            if !r.deleted && r.relation_type == relation_type && r.target_id == target_id {
                r.deleted = true;
                return true;
            }
        }
        false
    }

    /// Get all outgoing relations from a node, optionally filtered by type and time.
    pub fn outgoing(
        &self,
        source_id: &str,
        relation_type: Option<&str>,
        at_time: f64,
    ) -> Vec<Relation> {
        let source_indices = self.by_source.read();
        let indices = match source_indices.get(source_id) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let rels = self.relations.read();
        indices
            .iter()
            .filter_map(|&idx| {
                let r = &rels[idx];
                if !r.is_valid_at(at_time) {
                    return None;
                }
                if let Some(rt) = relation_type {
                    if r.relation_type != rt {
                        return None;
                    }
                }
                Some(r.to_relation())
            })
            .collect()
    }

    /// Get all incoming relations to a node.
    pub fn incoming(
        &self,
        target_id: &str,
        relation_type: Option<&str>,
        at_time: f64,
    ) -> Vec<Relation> {
        let target_indices = self.by_target.read();
        let indices = match target_indices.get(target_id) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let rels = self.relations.read();
        indices
            .iter()
            .filter_map(|&idx| {
                let r = &rels[idx];
                if !r.is_valid_at(at_time) {
                    return None;
                }
                if let Some(rt) = relation_type {
                    if r.relation_type != rt {
                        return None;
                    }
                }
                Some(r.to_relation())
            })
            .collect()
    }

    /// BFS traversal from start_id following relations of given type (or all if None).
    /// Returns paths up to max_depth hops. Includes inferred edges from symmetric/transitive types.
    pub fn traverse(
        &self,
        start_id: &str,
        relation_type: Option<&str>,
        max_depth: u32,
        at_time: f64,
        similarity_fn: &dyn Fn(&str, &str) -> f64,
    ) -> Vec<GraphPath> {
        let mut results = Vec::new();
        let mut visited = FxHashMap::<String, u32>::default();
        visited.insert(start_id.to_string(), 0);

        // BFS queue: (node_id, current_path, depth)
        let mut queue: VecDeque<(String, Vec<PathHop>, u32)> = VecDeque::new();
        queue.push_back((start_id.to_string(), Vec::new(), 0));

        let types = self.types.read();

        while let Some((node_id, path, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Collect neighbors: direct outgoing + symmetric incoming
            let mut neighbors: Vec<(String, String)> = Vec::new();

            for rel in self.outgoing(&node_id, relation_type, at_time) {
                neighbors.push((rel.target_id, rel.relation_type));
            }

            // Symmetric: incoming edges of symmetric types also count as outgoing
            if let Some(rt) = relation_type {
                if let Some(type_def) = types.get(rt) {
                    if type_def.symmetric {
                        for rel in self.incoming(&node_id, Some(rt), at_time) {
                            neighbors.push((rel.source_id, rel.relation_type));
                        }
                    }
                }
            } else {
                // No type filter: check each incoming relation's type for symmetry
                let incoming_indices = self.by_target.read();
                if let Some(indices) = incoming_indices.get(&node_id) {
                    let rels = self.relations.read();
                    for &idx in indices {
                        let r = &rels[idx];
                        if !r.is_valid_at(at_time) || r.deleted {
                            continue;
                        }
                        if let Some(td) = types.get(&r.relation_type) {
                            if td.symmetric {
                                neighbors.push((r.source_id.clone(), r.relation_type.clone()));
                            }
                        }
                    }
                }
            }

            for (next_id, rel_type) in neighbors {
                let next_depth = depth + 1;
                let prev_depth = visited.get(&next_id).copied();
                if prev_depth.is_some_and(|d| d <= next_depth) {
                    continue;
                }
                visited.insert(next_id.clone(), next_depth);

                let sim = similarity_fn(&node_id, &next_id);
                let mut new_path = path.clone();
                new_path.push(PathHop {
                    node_id: next_id.clone(),
                    relation_type: rel_type,
                    similarity: sim,
                });

                let score = new_path.iter().map(|h| h.similarity).product::<f64>();
                results.push(GraphPath {
                    hops: new_path.clone(),
                    score,
                });

                if next_depth < max_depth {
                    queue.push_back((next_id, new_path, next_depth));
                }
            }
        }

        // Transitive inference: if we have A->B->C for a transitive type, add A->C
        if let Some(rt) = relation_type {
            if let Some(type_def) = types.get(rt) {
                if type_def.transitive {
                    let mut inferred = Vec::new();
                    for path in &results {
                        if path.hops.len() >= 2 {
                            let final_node = &path.hops.last().expect("len >= 2").node_id;
                            let sim = similarity_fn(start_id, final_node);
                            inferred.push(GraphPath {
                                hops: vec![PathHop {
                                    node_id: final_node.clone(),
                                    relation_type: rt.to_string(),
                                    similarity: sim,
                                }],
                                score: sim,
                            });
                        }
                    }
                    results.extend(inferred);
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Serialize a relation for arena persistence.
    pub fn serialize_relation(rel: &Relation) -> Vec<u8> {
        let props = rel.properties.as_deref().unwrap_or("").as_bytes();

        let len = 1
            + 2
            + rel.source_id.len()
            + 2
            + rel.relation_type.len()
            + 2
            + rel.target_id.len()
            + 8
            + 8
            + 4
            + props.len();
        let mut buf = Vec::with_capacity(len);
        buf.push(RELATION_MAGIC);
        wire::write_lp_str(&mut buf, &rel.source_id);
        wire::write_lp_str(&mut buf, &rel.relation_type);
        wire::write_lp_str(&mut buf, &rel.target_id);
        buf.extend_from_slice(&(rel.valid_from as u64).to_le_bytes());
        buf.extend_from_slice(&(rel.valid_to as u64).to_le_bytes());
        buf.extend_from_slice(&(props.len() as u32).to_le_bytes());
        buf.extend_from_slice(props);
        buf
    }

    /// Deserialize a relation from arena bytes. Returns None if not a relation frame.
    pub fn deserialize_relation(data: &[u8]) -> Option<Relation> {
        match data.first() {
            Some(&m) if m == RELATION_MAGIC || m == RELATION_MAGIC_LEGACY => {}
            _ => return None,
        }
        let mut pos = 1;

        let (source, next) = wire::read_lp_str(data, pos)?;
        pos = next;

        let (rtype, next) = wire::read_lp_str(data, pos)?;
        pos = next;

        let (target, next) = wire::read_lp_str(data, pos)?;
        pos = next;

        let valid_from = u64::from_le_bytes(data.get(pos..pos + 8)?.try_into().ok()?) as f64;
        pos += 8;
        let valid_to = u64::from_le_bytes(data.get(pos..pos + 8)?.try_into().ok()?) as f64;
        pos += 8;

        let props_len = u32::from_le_bytes(data.get(pos..pos + 4)?.try_into().ok()?) as usize;
        pos += 4;
        let properties = if props_len > 0 {
            Some(
                std::str::from_utf8(data.get(pos..pos + props_len)?)
                    .ok()?
                    .to_string(),
            )
        } else {
            None
        };

        Some(Relation {
            source_id: source,
            relation_type: rtype,
            target_id: target,
            properties,
            valid_from,
            valid_to,
        })
    }

    /// Total number of live (non-deleted) relations.
    pub fn count(&self) -> usize {
        self.relations.read().iter().filter(|r| !r.deleted).count()
    }

    /// Load a relation from deserialized arena data (called during log replay).
    pub fn load_relation(&self, rel: &Relation) {
        self.add(rel);
    }

    /// Snapshot all live relations for compaction.
    pub fn snapshot(&self) -> Vec<Relation> {
        self.relations
            .read()
            .iter()
            .filter(|r| !r.deleted)
            .map(|r| r.to_relation())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_sim(_a: &str, _b: &str) -> f64 {
        0.8
    }

    #[test]
    fn test_add_and_query_outgoing() {
        let store = RelationStore::new();
        store.add(&Relation {
            source_id: "paris".into(),
            relation_type: "capital_of".into(),
            target_id: "france".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });
        store.add(&Relation {
            source_id: "berlin".into(),
            relation_type: "capital_of".into(),
            target_id: "germany".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });

        let out = store.outgoing("paris", Some("capital_of"), 0.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].target_id, "france");

        let out_all = store.outgoing("paris", None, 0.0);
        assert_eq!(out_all.len(), 1);

        let none = store.outgoing("tokyo", None, 0.0);
        assert!(none.is_empty());
    }

    #[test]
    fn test_remove_relation() {
        let store = RelationStore::new();
        store.add(&Relation {
            source_id: "a".into(),
            relation_type: "knows".into(),
            target_id: "b".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });
        assert_eq!(store.count(), 1);
        assert!(store.remove("a", "knows", "b"));
        assert_eq!(store.count(), 0);
        assert!(!store.remove("a", "knows", "b"));
    }

    #[test]
    fn test_symmetric_traversal() {
        let store = RelationStore::new();
        store.declare_type(RelationType {
            name: "friend_of".into(),
            transitive: false,
            symmetric: true,
        });
        store.add(&Relation {
            source_id: "alice".into(),
            relation_type: "friend_of".into(),
            target_id: "bob".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });

        // Bob should see Alice via symmetric inference
        let paths = store.traverse("bob", Some("friend_of"), 1, 0.0, &dummy_sim);
        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.hops[0].node_id == "alice"));
    }

    #[test]
    fn test_transitive_inference() {
        let store = RelationStore::new();
        store.declare_type(RelationType {
            name: "is_in".into(),
            transitive: true,
            symmetric: false,
        });
        store.add(&Relation {
            source_id: "paris".into(),
            relation_type: "is_in".into(),
            target_id: "france".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });
        store.add(&Relation {
            source_id: "france".into(),
            relation_type: "is_in".into(),
            target_id: "europe".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });

        let paths = store.traverse("paris", Some("is_in"), 3, 0.0, &dummy_sim);
        // Should find both direct (paris->france) and transitive (paris->europe)
        let targets: Vec<&str> = paths
            .iter()
            .flat_map(|p| p.hops.iter().map(|h| h.node_id.as_str()))
            .collect();
        assert!(targets.contains(&"france"));
        assert!(targets.contains(&"europe"));

        // Transitive inference should produce a 1-hop inferred path to europe
        let inferred_single_hop = paths
            .iter()
            .find(|p| p.hops.len() == 1 && p.hops[0].node_id == "europe");
        assert!(
            inferred_single_hop.is_some(),
            "Should have inferred single-hop to europe"
        );
    }

    #[test]
    fn test_multi_hop_traversal() {
        let store = RelationStore::new();
        store.add(&Relation {
            source_id: "a".into(),
            relation_type: "links".into(),
            target_id: "b".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });
        store.add(&Relation {
            source_id: "b".into(),
            relation_type: "links".into(),
            target_id: "c".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });
        store.add(&Relation {
            source_id: "c".into(),
            relation_type: "links".into(),
            target_id: "d".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });

        let paths = store.traverse("a", Some("links"), 3, 0.0, &dummy_sim);
        let max_hops = paths.iter().map(|p| p.hops.len()).max().unwrap_or(0);
        assert_eq!(max_hops, 3); // a->b->c->d
    }

    #[test]
    fn test_temporal_filter() {
        let store = RelationStore::new();
        store.add(&Relation {
            source_id: "a".into(),
            relation_type: "employs".into(),
            target_id: "b".into(),
            properties: None,
            valid_from: 1000.0,
            valid_to: 2000.0,
        });
        store.add(&Relation {
            source_id: "a".into(),
            relation_type: "employs".into(),
            target_id: "c".into(),
            properties: None,
            valid_from: 1500.0,
            valid_to: 0.0, // still active
        });

        // At time 1200: only b
        let at_1200 = store.outgoing("a", None, 1200.0);
        assert_eq!(at_1200.len(), 1);
        assert_eq!(at_1200[0].target_id, "b");

        // At time 1700: both
        let at_1700 = store.outgoing("a", None, 1700.0);
        assert_eq!(at_1700.len(), 2);

        // At time 2500: only c (b expired)
        let at_2500 = store.outgoing("a", None, 2500.0);
        assert_eq!(at_2500.len(), 1);
        assert_eq!(at_2500[0].target_id, "c");
    }

    #[test]
    fn test_serialize_deserialize_relation() {
        let rel = Relation {
            source_id: "paris".into(),
            relation_type: "capital_of".into(),
            target_id: "france".into(),
            properties: Some("{\"pop\":2_000_000}".into()),
            valid_from: 1000.0,
            valid_to: 0.0,
        };
        let bytes = RelationStore::serialize_relation(&rel);
        let parsed = RelationStore::deserialize_relation(&bytes).unwrap();
        assert_eq!(parsed.source_id, "paris");
        assert_eq!(parsed.relation_type, "capital_of");
        assert_eq!(parsed.target_id, "france");
        assert_eq!(parsed.properties.as_deref(), Some("{\"pop\":2_000_000}"));
        assert_eq!(parsed.valid_from, 1000.0);
        assert_eq!(parsed.valid_to, 0.0);
    }

    #[test]
    fn test_relation_magic_migration() {
        let rel = Relation {
            source_id: "paris".into(),
            relation_type: "capital_of".into(),
            target_id: "france".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        };

        // New writes use the distinct relation magic (0xFA), not the rule byte.
        let mut bytes = RelationStore::serialize_relation(&rel);
        assert_eq!(bytes[0], wire::magic::RELATION);
        assert_eq!(bytes[0], 0xFA);
        assert_ne!(bytes[0], wire::magic::RULE);
        let parsed = RelationStore::deserialize_relation(&bytes).unwrap();
        assert_eq!(parsed.source_id, "paris");
        assert_eq!(parsed.target_id, "france");

        // A hand-crafted legacy-0xFE relation (as older logs wrote it) still
        // deserializes so existing arenas load.
        bytes[0] = wire::magic::RELATION_LEGACY;
        assert_eq!(bytes[0], 0xFE);
        let legacy = RelationStore::deserialize_relation(&bytes).unwrap();
        assert_eq!(legacy.source_id, "paris");
        assert_eq!(legacy.relation_type, "capital_of");
        assert_eq!(legacy.target_id, "france");
    }

    #[test]
    fn test_snapshot() {
        let store = RelationStore::new();
        store.add(&Relation {
            source_id: "a".into(),
            relation_type: "r".into(),
            target_id: "b".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });
        store.add(&Relation {
            source_id: "c".into(),
            relation_type: "r".into(),
            target_id: "d".into(),
            properties: None,
            valid_from: 0.0,
            valid_to: 0.0,
        });
        store.remove("a", "r", "b");

        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].source_id, "c");
    }
}
