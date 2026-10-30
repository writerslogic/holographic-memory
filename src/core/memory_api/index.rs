// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The in-memory index over fact stamps. It is derived state: rebuilt from the documents when
//! the store's revision moves in a way the memory API did not itself account for, and never
//! persisted. It holds only what recall and policies cannot read from a result's own
//! metadata: the reverse supersession links, the subject groups and the entity postings.
//! A store that uses none of these has an index with no entries.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::stamp::Stamp;
use super::types::Millis;
use crate::core::engine::memory_access::DocumentView;
use crate::core::HmsCore;

/// A fact that supersedes another, with when that takes effect.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Edge {
    pub by: String,
    /// Valid-time effect: the superseding fact's effective-from.
    pub effective: Millis,
    /// Knowledge-time effect: when the superseding fact was ingested.
    pub ingest: Millis,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Member {
    pub id: String,
    pub effective: Millis,
    pub ingest: Millis,
}

#[derive(Default)]
pub(crate) struct Index {
    /// The store revision this index reflects; none when it must be rebuilt.
    pub(crate) revision: Option<u64>,
    windowed: usize,
    superseded_by: HashMap<String, Vec<Edge>>,
    predecessors: HashMap<String, Vec<String>>,
    by_subject: HashMap<String, Vec<Member>>,
    by_entity: HashMap<String, Vec<String>>,
}

impl Index {
    /// Every entry, sorted: equal for two indexes that answer every lookup alike.
    #[cfg(test)]
    pub(crate) fn canonical(&self) -> Vec<String> {
        let mut out = vec![format!("windowed {}", self.windowed)];
        for (id, edges) in &self.superseded_by {
            out.extend(edges.iter().map(|e| format!("by {id} {e:?}")));
        }
        for (id, olds) in &self.predecessors {
            out.extend(olds.iter().map(|old| format!("old {id} {old}")));
        }
        for (subject, members) in &self.by_subject {
            out.extend(members.iter().map(|m| format!("subject {subject} {m:?}")));
        }
        for (entity, ids) in &self.by_entity {
            out.extend(ids.iter().map(|id| format!("entity {entity} {id}")));
        }
        out.sort_unstable();
        out
    }

    /// Whether any fact has a validity window or a supersession link, that is, whether
    /// recall must evaluate more than "everything is current".
    pub(crate) fn constrained(&self) -> bool {
        self.windowed > 0 || !self.superseded_by.is_empty()
    }

    pub(crate) fn edges(&self, id: &str) -> &[Edge] {
        self.superseded_by.get(id).map_or(&[], Vec::as_slice)
    }

    pub(super) fn capture_edges(
        &self,
        budget: &mut crate::core::query_limits::QueryBudget,
    ) -> anyhow::Result<HashMap<String, Vec<Edge>>> {
        let mut captured = HashMap::new();
        for (id, edges) in &self.superseded_by {
            budget.work(1 + edges.len())?;
            budget.reserve(256 + id.len() + edges.len() * std::mem::size_of::<Edge>())?;
            for edge in edges {
                budget.reserve(edge.by.len())?;
            }
            captured.insert(id.clone(), edges.clone());
        }
        Ok(captured)
    }

    /// Candidates of a subject group, ordered by effect, ingest, id.
    pub(crate) fn members(&self, subject: &str) -> Vec<Member> {
        let mut members = self.by_subject.get(subject).cloned().unwrap_or_default();
        members.sort_by(|a, b| (a.effective, a.ingest, &a.id).cmp(&(b.effective, b.ingest, &b.id)));
        members
    }

    /// The effect and ingest time of a member of a subject group.
    pub(crate) fn member(&self, subject: &str, id: &str) -> Option<(Millis, Millis)> {
        self.by_subject
            .get(subject)?
            .iter()
            .find(|member| member.id == id)
            .map(|member| (member.effective, member.ingest))
    }

    pub(crate) fn entity_ids(&self, entity: &str) -> &[String] {
        self.by_entity.get(entity).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn subject_ids(&self, subject: &str) -> Vec<String> {
        self.by_subject
            .get(subject)
            .map(|members| members.iter().map(|m| m.id.clone()).collect())
            .unwrap_or_default()
    }

    fn add_edge(&mut self, old: &str, by: &str, effective: Millis, ingest: Millis) {
        let edges = self.superseded_by.entry(old.to_owned()).or_default();
        if edges.iter().any(|edge| edge.by == by) {
            return;
        }
        edges.push(Edge {
            by: by.to_owned(),
            effective,
            ingest,
        });
        let olds = self.predecessors.entry(by.to_owned()).or_default();
        if !olds.iter().any(|existing| existing == old) {
            olds.push(old.to_owned());
        }
    }

    /// Adds a fact. `lookup` supplies the effect and ingest time of a superseder named in the
    /// stamp's `superseded_by`; a superseder it does not know is ignored.
    pub(crate) fn insert(
        &mut self,
        id: &str,
        stamp: &Stamp,
        lookup: &dyn Fn(&str) -> Option<(Millis, Millis)>,
    ) {
        if stamp.windowed() {
            self.windowed += 1;
        }
        let effective = stamp.effective_from();
        for old in &stamp.supersedes {
            self.add_edge(old, id, effective, stamp.ingest);
        }
        for by in &stamp.superseded_by {
            if let Some((by_effective, by_ingest)) = lookup(by) {
                self.add_edge(id, by, by_effective, by_ingest);
            }
        }
        if let Some(subject) = &stamp.subject {
            let members = self.by_subject.entry(subject.clone()).or_default();
            members.retain(|member| member.id != id);
            members.push(Member {
                id: id.to_owned(),
                effective,
                ingest: stamp.ingest,
            });
        }
        for entity in &stamp.entities {
            let ids = self.by_entity.entry(entity.clone()).or_default();
            if !ids.iter().any(|existing| existing == id) {
                ids.push(id.to_owned());
            }
        }
    }

    /// Removes a fact with every link to and from it.
    pub(crate) fn remove(&mut self, id: &str, stamp: &Stamp) {
        if stamp.windowed() {
            self.windowed = self.windowed.saturating_sub(1);
        }
        if let Some(olds) = self.predecessors.remove(id) {
            for old in olds {
                if let Some(edges) = self.superseded_by.get_mut(&old) {
                    edges.retain(|edge| edge.by != id);
                    if edges.is_empty() {
                        self.superseded_by.remove(&old);
                    }
                }
            }
        }
        if let Some(edges) = self.superseded_by.remove(id) {
            for edge in edges {
                if let Some(olds) = self.predecessors.get_mut(&edge.by) {
                    olds.retain(|old| old != id);
                    if olds.is_empty() {
                        self.predecessors.remove(&edge.by);
                    }
                }
            }
        }
        if let Some(subject) = &stamp.subject {
            if let Some(members) = self.by_subject.get_mut(subject) {
                members.retain(|member| member.id != id);
                if members.is_empty() {
                    self.by_subject.remove(subject);
                }
            }
        }
        for entity in &stamp.entities {
            if let Some(ids) = self.by_entity.get_mut(entity) {
                ids.retain(|existing| existing != id);
                if ids.is_empty() {
                    self.by_entity.remove(entity);
                }
            }
        }
    }

    /// Facts `id` supersedes, directly.
    pub(crate) fn direct_predecessors(&self, id: &str) -> &[String] {
        self.predecessors.get(id).map_or(&[], Vec::as_slice)
    }

    /// Whether `from` supersedes `target`, directly or through a chain.
    pub(crate) fn supersedes_transitively(&self, from: &str, target: &str) -> bool {
        self.walk(from, |index, id| index.direct_predecessors(id).to_vec())
            .contains(target)
    }

    /// The surviving facts `start` supersedes once the facts in `dead` are gone: its direct
    /// predecessors, with each dead one replaced by what it superseded, transitively.
    pub(crate) fn bridge(&self, start: &str, dead: &BTreeSet<String>) -> BTreeSet<String> {
        let mut survivors = BTreeSet::new();
        let mut seen = HashSet::new();
        let mut stack: Vec<String> = self.direct_predecessors(start).to_vec();
        while let Some(next) = stack.pop() {
            if !seen.insert(next.clone()) {
                continue;
            }
            if dead.contains(&next) {
                stack.extend(self.direct_predecessors(&next).iter().cloned());
            } else {
                survivors.insert(next);
            }
        }
        survivors
    }

    /// Every fact `id` supersedes, directly or not.
    pub(crate) fn ancestors(&self, id: &str) -> BTreeSet<String> {
        self.walk(id, |index, id| index.direct_predecessors(id).to_vec())
    }

    /// Every fact that supersedes `id`, directly or not.
    pub(crate) fn descendants(&self, id: &str) -> BTreeSet<String> {
        self.walk(id, |index, id| {
            index.edges(id).iter().map(|edge| edge.by.clone()).collect()
        })
    }

    /// `id` and every fact connected to it by supersession in either direction.
    pub(crate) fn component(&self, id: &str) -> BTreeSet<String> {
        let mut seen = BTreeSet::from([id.to_owned()]);
        let mut frontier = vec![id.to_owned()];
        while let Some(next) = frontier.pop() {
            let neighbours = self
                .direct_predecessors(&next)
                .iter()
                .cloned()
                .chain(self.edges(&next).iter().map(|edge| edge.by.clone()));
            for neighbour in neighbours.collect::<Vec<_>>() {
                if seen.insert(neighbour.clone()) {
                    frontier.push(neighbour);
                }
            }
        }
        seen
    }

    /// The facts reachable from `start` through `step`, excluding `start`.
    fn walk(&self, start: &str, step: impl Fn(&Self, &str) -> Vec<String>) -> BTreeSet<String> {
        let mut seen = HashSet::from([start.to_owned()]);
        let mut found = BTreeSet::new();
        let mut frontier = vec![start.to_owned()];
        while let Some(next) = frontier.pop() {
            for neighbour in step(self, &next) {
                if seen.insert(neighbour.clone()) {
                    found.insert(neighbour.clone());
                    frontier.push(neighbour);
                }
            }
        }
        found
    }

    /// Rebuilds the index from the store's documents.
    pub(crate) fn rebuild(core: &HmsCore) -> Self {
        let mut stamps: Vec<(String, Stamp)> = Vec::new();
        let revision = core.scan_documents(|document: DocumentView<'_>| {
            if let Some(stamp) = Stamp::from_parts(document.version, document.metadata) {
                if stamp.windowed()
                    || stamp.subject.is_some()
                    || !stamp.supersedes.is_empty()
                    || !stamp.superseded_by.is_empty()
                    || !stamp.entities.is_empty()
                {
                    stamps.push((document.id.to_owned(), stamp));
                }
            }
        });
        let known: HashMap<&str, (Millis, Millis)> = stamps
            .iter()
            .map(|(id, stamp)| (id.as_str(), (stamp.effective_from(), stamp.ingest)))
            .collect();
        let lookup = |id: &str| known.get(id).copied();
        let mut index = Self::default();
        // Stamps naming a missing predecessor would create an edge keyed by a fact that does
        // not exist; the edge is harmless (nothing looks it up) and is dropped with the
        // superseder if that is forgotten.
        for (id, stamp) in &stamps {
            index.insert(id, stamp, &lookup);
        }
        index.revision = Some(revision);
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_edge_capture_enforces_work_and_byte_boundaries() {
        use crate::core::query_limits::{QueryBudget, QueryLimitExceeded, QueryLimits};
        let mut index = Index::default();
        index.insert("new", &stamp(2, &["old"]), &|_| None);
        let bytes = 256 + 6 + std::mem::size_of::<Edge>();
        for limit in [bytes - 1, bytes, bytes + 1] {
            let mut budget = QueryBudget::new(QueryLimits {
                max_bytes: limit,
                ..Default::default()
            });
            let result = index.capture_edges(&mut budget);
            assert_eq!(result.is_ok(), limit >= bytes);
            if let Err(error) = result {
                assert_eq!(
                    error.downcast_ref::<QueryLimitExceeded>().unwrap().resource,
                    "bytes"
                );
            }
        }
        for limit in [1, 2, 3] {
            let mut budget = QueryBudget::new(QueryLimits {
                max_work: limit,
                ..Default::default()
            });
            assert_eq!(index.capture_edges(&mut budget).is_ok(), limit >= 2);
        }
    }

    fn stamp(ingest: Millis, supersedes: &[&str]) -> Stamp {
        Stamp {
            ingest,
            supersedes: supersedes.iter().map(|s| s.to_string()).collect(),
            ..Stamp::default()
        }
    }

    #[test]
    fn insert_and_remove_are_inverse_and_chains_are_transitive() {
        let mut index = Index::default();
        let none = |_: &str| None;
        index.insert("b", &stamp(2, &["a"]), &none);
        index.insert("c", &stamp(3, &["b"]), &none);
        assert!(index.supersedes_transitively("c", "a"));
        assert!(!index.supersedes_transitively("a", "c"));
        assert_eq!(
            index.component("a"),
            BTreeSet::from(["a".into(), "b".into(), "c".into()])
        );
        assert_eq!(index.descendants("a").len(), 2);
        assert_eq!(index.ancestors("c").len(), 2);
        index.remove("c", &stamp(3, &["b"]));
        assert!(index.edges("b").is_empty());
        index.remove("b", &stamp(2, &["a"]));
        assert!(!index.constrained());
    }

    #[test]
    fn superseded_by_hint_without_a_known_superseder_is_ignored() {
        let mut index = Index::default();
        let hinted = Stamp {
            ingest: 1,
            superseded_by: vec!["ghost".into()],
            ..Stamp::default()
        };
        index.insert("old", &hinted, &|_| None);
        assert!(index.edges("old").is_empty());
        index.insert("old", &hinted, &|id| (id == "ghost").then_some((9, 9)));
        assert_eq!(index.edges("old").len(), 1);
    }
}
