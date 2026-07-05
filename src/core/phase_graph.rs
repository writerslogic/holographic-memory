// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Phasor relational memory: a holographic connection store on a quantized-phase
//! substrate, validated in the `relation-algebra` and `holographic-retrieval`
//! experiment binaries.
//!
//! Each entity is a `dim`-vector of phases in `Z_N`; a relation is a rotation
//! `theta in Z_N`; binding is phase addition `(phi + theta) mod N`. This gives a
//! genuine **relation algebra** the sparse-binary `ConnectionGraph` cannot
//! express — rotations form a group, so an inverse relation is a negated angle
//! and composition is angle addition. A fact `(subject, relation, object)` is
//! stored as a **trace phase** `t = (subject + relation + object) mod N`,
//! accumulated per dimension into an integer **phase histogram** `hist[dim][t]`.
//!
//! The trace is symmetric across its slots, so a single field answers three query
//! kinds by solving for the missing slot:
//!   - [`score`](PhaseGraph::score): verify a full triple.
//!   - [`retrieve_object`](PhaseGraph::retrieve_object): recover `o` from `(s, r)`.
//!   - [`retrieve_subject`](PhaseGraph::retrieve_subject): recover `s` from
//!     `(r, o)` — i.e. the inverse relation, for free.
//!
//! Everything is integer, so the field is a **deterministic fold** of the event
//! stream: state is bit-exactly replayable and the chain digest is tamper-evident
//! (the same verifiability contract as [`super::connection_graph`]). This is what
//! makes phasor holography compatible with provenance — the float-vs-verifiability
//! tension dissolves under phase quantization.
//!
//! # Honest costs
//! The histogram is dense (`dim * N` integers) — far larger than a sparse bloom.
//! Retrieval ranks over all known entities (`O(entities * dim)`); a production
//! path would prune candidates with the ANN index. Experimental and opt-in.

use fxhash::FxHashMap;

use crate::core::entangled::hash_u64;

/// Store weight per asserted fact.
const W_STORE: u32 = 1;

/// An ordered, replayable state-change event. Facts are referenced by their
/// string ids so a verifier recomputes the affected phases from scratch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Store(String, String, String),
}

/// A phasor relational memory over a `dim`-dimensional, `n_phases`-quantized
/// phase field.
pub struct PhaseGraph {
    dim: usize,
    n_phases: u32,
    hist: Vec<u32>, // dim * n_phases, row-major: hist[d * n_phases + phase]
    entities: Vec<String>,
    entity_set: FxHashMap<String, ()>,
    events: Vec<Event>,
    chain: u64,
}

impl PhaseGraph {
    /// Create an empty phasor memory. Panics if `dim` or `n_phases` is 0.
    pub fn new(dim: usize, n_phases: u32) -> Self {
        assert!(
            dim > 0 && n_phases > 0,
            "phase graph dim/n_phases must be > 0"
        );
        Self {
            dim,
            n_phases,
            hist: vec![0u32; dim * n_phases as usize],
            entities: Vec::new(),
            entity_set: FxHashMap::default(),
            events: Vec::new(),
            chain: 0,
        }
    }

    /// Deterministic phase for entity `id` at dimension `d`.
    fn phase(&self, id: &str, d: usize) -> u32 {
        (hash_u64(str_seed(id), d as u64) % self.n_phases as u64) as u32
    }

    /// Deterministic rotation for relation `rel` at dimension `d`.
    fn rot(&self, rel: &str, d: usize) -> u32 {
        (hash_u64(str_seed(rel) ^ 0x0010_7A70, d as u64) % self.n_phases as u64) as u32
    }

    /// Trace phase of `(subject, relation, object)` at dimension `d`.
    fn trace(&self, s: &str, r: &str, o: &str, d: usize) -> u32 {
        (self.phase(s, d) + self.rot(r, d) + self.phase(o, d)) % self.n_phases
    }

    fn track(&mut self, id: &str) {
        if self.entity_set.insert(id.to_string(), ()).is_none() {
            self.entities.push(id.to_string());
        }
    }

    /// Assert a fact into the memory.
    pub fn relate(&mut self, subject: &str, relation: &str, object: &str) {
        for d in 0..self.dim {
            let t = self.trace(subject, relation, object, d) as usize;
            self.hist[d * self.n_phases as usize + t] += W_STORE;
        }
        self.track(subject);
        self.track(object);
        let ev = Event::Store(
            subject.to_string(),
            relation.to_string(),
            object.to_string(),
        );
        self.chain = chain_step(self.chain, &ev);
        self.events.push(ev);
    }

    /// Raw holographic score of a full triple: how strongly the field agrees with
    /// its trace, summed over dimensions.
    fn raw_score(&self, s: &str, r: &str, o: &str) -> u64 {
        (0..self.dim)
            .map(|d| {
                let t = self.trace(s, r, o, d) as usize;
                self.hist[d * self.n_phases as usize + t] as u64
            })
            .sum()
    }

    /// Presence score of a triple in `[0, 1]`-ish: raw agreement minus the
    /// expected coincidence floor, normalized by dimension. A stored triple sits
    /// well above an absent one.
    pub fn score(&self, subject: &str, relation: &str, object: &str) -> f64 {
        let raw = self.raw_score(subject, relation, object) as f64;
        let total: u64 = self.hist.iter().map(|&c| c as u64).sum();
        let floor = total as f64 / self.n_phases as f64; // expected coincidences
        (raw - floor) / self.dim as f64
    }

    /// Recover the object of `(subject, relation)` by ranking known entities.
    /// Returns `None` if the memory is empty.
    pub fn retrieve_object(&self, subject: &str, relation: &str) -> Option<&str> {
        self.rank(|g, cand| g.raw_score(subject, relation, cand))
    }

    /// Recover the subject of `(relation, object)` — the inverse relation query,
    /// answered from the same field.
    pub fn retrieve_subject(&self, relation: &str, object: &str) -> Option<&str> {
        self.rank(|g, cand| g.raw_score(cand, relation, object))
    }

    fn rank<F: Fn(&Self, &str) -> u64>(&self, score_of: F) -> Option<&str> {
        let mut best: Option<&str> = None;
        let mut best_s = 0u64;
        for cand in &self.entities {
            let s = score_of(self, cand);
            if best.is_none() || s > best_s {
                best_s = s;
                best = Some(cand);
            }
        }
        best
    }

    /// Multi-hop reasoning: follow a relation path from `start`, retrieving the
    /// object at each hop and feeding it to the next. Returns the final entity, or
    /// `None` if a hop finds nothing. This is chained associative retrieval, not a
    /// single composed rotation: for a general (non-learned) store the entities
    /// do not satisfy `object == rotate(subject, r)`, so one-shot composition is
    /// unavailable. Per-hop retrieval error therefore compounds — longer paths
    /// degrade faster under load.
    pub fn retrieve_path(&self, start: &str, relations: &[&str]) -> Option<String> {
        let mut current = start.to_string();
        for r in relations {
            current = self.retrieve_object(&current, r)?.to_string();
        }
        Some(current)
    }

    /// The tamper-evident chain digest over all events so far.
    pub fn chain_digest(&self) -> u64 {
        self.chain
    }

    /// The recorded event stream.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Verify that the current field and chain digest are exactly the
    /// deterministic replay of the event log.
    pub fn verify(&self) -> bool {
        let mut g = Self::new(self.dim, self.n_phases);
        for ev in &self.events {
            match ev {
                Event::Store(s, r, o) => {
                    for d in 0..g.dim {
                        let t = g.trace(s, r, o, d) as usize;
                        g.hist[d * g.n_phases as usize + t] += W_STORE;
                    }
                }
            }
            g.chain = chain_step(g.chain, ev);
        }
        g.hist == self.hist && g.chain == self.chain
    }
}

/// Deterministic 64-bit seed for a string id (FNV-1a through splitmix).
fn str_seed(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash_u64(h, s.len() as u64)
}

fn chain_step(prev: u64, ev: &Event) -> u64 {
    match ev {
        Event::Store(s, r, o) => hash_u64(prev ^ 0x5701, str_seed(&format!("{s}\u{1}{r}\u{1}{o}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_triple_scores_above_absent() {
        let mut g = PhaseGraph::new(1024, 256);
        g.relate("paris", "capital_of", "france");
        assert!(
            g.score("paris", "capital_of", "france") > g.score("berlin", "capital_of", "spain")
        );
    }

    #[test]
    fn retrieves_object_and_subject() {
        let mut g = PhaseGraph::new(1024, 256);
        g.relate("paris", "capital_of", "france");
        g.relate("berlin", "capital_of", "germany");
        g.relate("madrid", "capital_of", "spain");
        // forward: (paris, capital_of, ?) -> france
        assert_eq!(g.retrieve_object("paris", "capital_of"), Some("france"));
        // inverse (subject recovery from the same field): (?, capital_of, germany) -> berlin
        assert_eq!(g.retrieve_subject("capital_of", "germany"), Some("berlin"));
    }

    #[test]
    fn retrieval_holds_under_load() {
        let mut g = PhaseGraph::new(1024, 256);
        for i in 0..400 {
            g.relate(&format!("s{i}"), "rel", &format!("o{i}"));
        }
        // A sampled fact must still round-trip after 400 superposed facts.
        assert_eq!(g.retrieve_object("s137", "rel"), Some("o137"));
        assert_eq!(g.retrieve_subject("rel", "o42"), Some("s42"));
    }

    #[test]
    fn multi_hop_path_reasoning() {
        let mut g = PhaseGraph::new(1024, 256);
        // alice --friend--> bob --employer--> acme --located_in--> paris
        g.relate("alice", "friend", "bob");
        g.relate("bob", "employer", "acme");
        g.relate("acme", "located_in", "paris");
        // add distractors so ranking is non-trivial
        for i in 0..200 {
            g.relate(&format!("x{i}"), "friend", &format!("y{i}"));
        }
        assert_eq!(
            g.retrieve_path("alice", &["friend", "employer", "located_in"])
                .as_deref(),
            Some("paris")
        );
        // single hop still works
        assert_eq!(
            g.retrieve_path("alice", &["friend"]).as_deref(),
            Some("bob")
        );
    }

    #[test]
    fn replay_is_bit_exact_and_tamper_evident() {
        let mut g = PhaseGraph::new(512, 256);
        g.relate("a", "r", "b");
        g.relate("c", "r", "d");
        assert!(g.verify());

        let mut tampered = g.events().to_vec();
        tampered.push(Event::Store("x".into(), "r".into(), "y".into()));
        let mut h = PhaseGraph::new(512, 256);
        for ev in &tampered {
            let Event::Store(s, r, o) = ev;
            h.relate(s, r, o);
        }
        assert_ne!(h.chain_digest(), g.chain_digest());
    }
}
