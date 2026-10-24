// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Phasor relational memory: a holographic connection store on a quantized-phase
//! substrate.
//!
//! Each entity has distinct subject and object phase vectors in `Z_N`; a
//! relation contributes a rotation in `Z_N`. A fact `(subject, relation, object)`
//! is stored as a trace phase `t = (subject_role(s) + relation(r) + object_role(o))
//! mod N`, accumulated per dimension into an integer phase histogram
//! `hist[dim][t]`. Separating the entity roles preserves edge direction even
//! though phase addition itself is commutative.
//!
//! A single field answers three query kinds by scoring candidates in their
//! corresponding roles:
//!   - [`score`](PhaseGraph::score): verify a full triple.
//!   - [`retrieve_object`](PhaseGraph::retrieve_object): recover `o` from `(s, r)`.
//!   - [`retrieve_subject`](PhaseGraph::retrieve_subject): recover `s` from
//!     `(r, o)`.
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

use crate::core::hash::FastMap;

use crate::core::entangled::hash_u64;

/// Store weight per asserted fact.
const W_STORE: u32 = 1;
const SUBJECT_ROLE: u64 = 0x5355_424A_4543_5400;
const OBJECT_ROLE: u64 = 0x4F42_4A45_4354_0000;
const RELATION_ROLE: u64 = 0x0010_7A70;

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
    entity_set: FastMap<String, ()>,
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
        let cells = dim
            .checked_mul(n_phases as usize)
            .expect("phase graph histogram dimensions overflow");
        Self {
            dim,
            n_phases,
            hist: vec![0u32; cells],
            entities: Vec::new(),
            entity_set: FastMap::default(),
            events: Vec::new(),
            chain: 0,
        }
    }

    /// Trace phase of `(subject, relation, object)` at dimension `d`.
    fn trace(&self, seeds: [u64; 3], d: usize) -> u32 {
        let n = self.n_phases as u64;
        ((hash_u64(seeds[0], d as u64) % n
            + hash_u64(seeds[1], d as u64) % n
            + hash_u64(seeds[2], d as u64) % n)
            % n) as u32
    }

    fn track(&mut self, id: &str) {
        if self.entity_set.insert(id.to_string(), ()).is_none() {
            self.entities.push(id.to_string());
        }
    }

    /// Assert a fact into the memory.
    pub fn relate(&mut self, subject: &str, relation: &str, object: &str) {
        let seeds = trace_seeds(subject, relation, object);
        for d in 0..self.dim {
            let t = self.trace(seeds, d) as usize;
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
        let seeds = trace_seeds(s, r, o);
        (0..self.dim)
            .map(|d| {
                let t = self.trace(seeds, d) as usize;
                self.hist[d * self.n_phases as usize + t] as u64
            })
            .sum()
    }

    /// Presence score of a triple in `[0, 1]`-ish: raw agreement minus the
    /// expected coincidence floor, normalized by dimension. A stored triple sits
    /// well above an absent one.
    pub fn score(&self, subject: &str, relation: &str, object: &str) -> f64 {
        let raw = self.raw_score(subject, relation, object) as f64;
        let floor = self.events.len() as f64 * W_STORE as f64 / self.n_phases as f64;
        raw / self.dim as f64 - floor
    }

    /// Recover the object of `(subject, relation)` by ranking known entities.
    /// Returns `None` if the memory is empty.
    pub fn retrieve_object(&self, subject: &str, relation: &str) -> Option<&str> {
        self.rank(
            [
                str_seed(subject) ^ SUBJECT_ROLE,
                str_seed(relation) ^ RELATION_ROLE,
            ],
            OBJECT_ROLE,
        )
    }

    /// Recover the subject of `(relation, object)` — the inverse relation query,
    /// answered from the same field.
    pub fn retrieve_subject(&self, relation: &str, object: &str) -> Option<&str> {
        self.rank(
            [
                str_seed(relation) ^ RELATION_ROLE,
                str_seed(object) ^ OBJECT_ROLE,
            ],
            SUBJECT_ROLE,
        )
    }

    fn rank(&self, known: [u64; 2], candidate_role: u64) -> Option<&str> {
        if self.entities.is_empty() {
            return None;
        }
        let n = self.n_phases as u64;
        let cue: Vec<u32> = (0..self.dim)
            .map(|d| {
                ((hash_u64(known[0], d as u64) % n + hash_u64(known[1], d as u64) % n) % n) as u32
            })
            .collect();
        let mut best: Option<&str> = None;
        let mut best_s = 0u64;
        for cand in &self.entities {
            let seed = str_seed(cand) ^ candidate_role;
            let s = cue
                .iter()
                .enumerate()
                .map(|(d, &phase)| {
                    let t = (phase as u64 + hash_u64(seed, d as u64) % n) % n;
                    self.hist[d * self.n_phases as usize + t as usize] as u64
                })
                .sum();
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

    /// A genuine cryptographic commitment to the event log: SHA-256 over a
    /// canonical, length-prefixed serialization of every event in order. Unlike
    /// [`chain_digest`](Self::chain_digest) (a fast, non-cryptographic 64-bit
    /// integrity check), this is collision-resistant, so it is the digest to sign
    /// (Ed25519) or publish as a checkpoint for verifiable, tamper-evident
    /// mutation. Reordering or altering any event changes it.
    #[cfg(feature = "security")]
    pub fn sha256_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for ev in &self.events {
            let Event::Store(s, r, o) = ev;
            h.update([0x53]); // 'S'
            for field in [s.as_str(), r.as_str(), o.as_str()] {
                h.update((field.len() as u32).to_le_bytes());
                h.update(field.as_bytes());
            }
        }
        h.finalize().into()
    }

    /// Verify that the current field and chain digest are exactly the
    /// deterministic replay of the event log.
    pub fn verify(&self) -> bool {
        let mut g = Self::new(self.dim, self.n_phases);
        for ev in &self.events {
            match ev {
                Event::Store(s, r, o) => {
                    let seeds = trace_seeds(s, r, o);
                    for d in 0..g.dim {
                        let t = g.trace(seeds, d) as usize;
                        g.hist[d * g.n_phases as usize + t] += W_STORE;
                    }
                }
            }
            g.chain = chain_step(g.chain, ev);
        }
        g.hist == self.hist && g.chain == self.chain
    }
}

fn trace_seeds(s: &str, r: &str, o: &str) -> [u64; 3] {
    [
        str_seed(s) ^ SUBJECT_ROLE,
        str_seed(r) ^ RELATION_ROLE,
        str_seed(o) ^ OBJECT_ROLE,
    ]
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
    fn directed_triple_does_not_assert_its_reverse() {
        let mut g = PhaseGraph::new(1024, 256);
        g.relate("alice", "parent_of", "bob");
        assert!(g.score("alice", "parent_of", "bob") > 0.9);
        assert!(g.score("bob", "parent_of", "alice").abs() < 0.1);
        assert!(g.verify());
    }

    #[test]
    fn repeated_relation_path_preserves_direction() {
        let mut g = PhaseGraph::new(1024, 256);
        g.relate("alice", "parent_of", "bob");
        g.relate("bob", "parent_of", "carol");
        g.relate("carol", "parent_of", "dave");

        assert_eq!(g.retrieve_object("bob", "parent_of"), Some("carol"));
        assert_eq!(g.retrieve_subject("parent_of", "bob"), Some("alice"));
        assert_eq!(
            g.retrieve_path("alice", &["parent_of", "parent_of", "parent_of"])
                .as_deref(),
            Some("dave")
        );
        assert!(g.verify());
    }

    #[test]
    fn factored_queries_match_exhaustive_scoring() {
        for dim in [1, 17, 128] {
            for n_phases in [1, 15, 16, 17, 255, 256, 257] {
                let mut g = PhaseGraph::new(dim, n_phases);
                assert_eq!(g.retrieve_object("absent", "rel"), None);
                assert_eq!(g.score("absent", "rel", "absent"), 0.0);
                for i in 0..20 {
                    g.relate(&format!("entity-{i}"), "rel", &format!("entity-{}", i + 1));
                }
                for i in [0, 9, 19, 21] {
                    let id = format!("entity-{i}");
                    let first_max = |scores: Vec<u64>| {
                        let maximum = scores.iter().max().unwrap();
                        g.entities[scores.iter().position(|score| score == maximum).unwrap()]
                            .as_str()
                    };
                    let objects = g
                        .entities
                        .iter()
                        .map(|o| g.raw_score(&id, "rel", o))
                        .collect();
                    let subjects = g
                        .entities
                        .iter()
                        .map(|s| g.raw_score(s, "rel", &id))
                        .collect();
                    assert_eq!(g.retrieve_object(&id, "rel"), Some(first_max(objects)));
                    assert_eq!(g.retrieve_subject("rel", &id), Some(first_max(subjects)));
                    let total: u64 = g.hist.iter().map(|&cell| cell as u64).sum();
                    let expected = (g.raw_score(&id, "rel", "entity-10") as f64
                        - total as f64 / n_phases as f64)
                        / dim as f64;
                    assert!((g.score(&id, "rel", "entity-10") - expected).abs() < 1e-12);
                }
                assert!(g.verify());
            }
        }
    }

    #[test]
    #[should_panic(expected = "phase graph histogram dimensions overflow")]
    fn rejects_overflowing_histogram_dimensions() {
        PhaseGraph::new(usize::MAX / 2 + 1, 2);
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

    #[cfg(feature = "security")]
    #[test]
    fn sha256_digest_is_deterministic_and_tamper_evident() {
        let mut g = PhaseGraph::new(256, 256);
        g.relate("a", "r", "b");
        g.relate("c", "r", "d");
        let digest = g.sha256_digest();

        // Same events reproduce the same commitment.
        let mut same = PhaseGraph::new(256, 256);
        same.relate("a", "r", "b");
        same.relate("c", "r", "d");
        assert_eq!(same.sha256_digest(), digest);

        // Altering an assertion changes the commitment.
        let mut tampered = PhaseGraph::new(256, 256);
        tampered.relate("a", "r", "b");
        tampered.relate("c", "r", "EVIL");
        assert_ne!(tampered.sha256_digest(), digest);
    }
}
