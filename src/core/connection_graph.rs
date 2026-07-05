// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Living connection-graph: a plastic, event-sourced holographic relation store.
//!
//! This is the engine-level form of the mechanisms validated in the
//! `plastic-graph` and `path-plasticity` experiment binaries (see
//! `docs/PREREGISTRATION-binding-readout.md` §10-14 and the
//! `living-connection-graph` design note). It is **experimental** and **opt-in**;
//! it does not touch the sparse-binary `bind`/`similarity` core.
//!
//! The unit of memory is the **relation** `(subject, relation, object)`, not the
//! node. Each relation is a *compositional* edge — the union of a role-permuted
//! subject and a role-permuted object — so relations that share a `(subject,
//! relation)` prefix share substructure, which is what lets the store generalize
//! to relations it never saw (a property a cache lacks; measured in §14 of the
//! preregistration).
//!
//! Reads mutate: querying a relation optionally **strengthens** it and lets the
//! untouched background **decay** (the "observer effect"). Every mutation is an
//! ordered [`Event`]; the interference field is a **deterministic integer fold**
//! of the event stream, so state is bit-exactly replayable and the running chain
//! digest is tamper-evident. This reconciles mutate-on-read with verifiability:
//! the mutation is an auditable event, not a hidden write. (The chain hash here
//! is a deterministic 64-bit digest; a production deployment substitutes the
//! SHA-256 + Ed25519 checkpoint machinery HMS already ships under `provenance`.)
//!
//! # Honest scope
//! Plasticity is a **working-set** mechanism: it keeps frequently-queried
//! relations sharp *by forgetting the rest* (see [`tests::forgetting_is_real`]),
//! and its generalization *is* inductive inference — the same mechanism that can
//! surface relations that were never asserted. It helps under load and is
//! neutral-to-slightly-negative without it. Use accordingly.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::core::entangled::{hash_u64, EntangledHVec};

/// Default multiplicative decay applied to the whole field per decay step: 7/8.
const DEFAULT_DECAY_NUM: i64 = 7;
const DEFAULT_DECAY_DEN: i64 = 8;
/// Field density: active indices per entity vector = dim / this.
const DENSITY_DENOM: usize = 256;
const W_STORE: i64 = 1;
const W_STRENGTHEN: i64 = 2;

/// An ordered, replayable state-change event. Relations are referenced by their
/// string ids so a verifier can recompute the affected indices from scratch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Event {
    /// Assert a relation `(subject, relation, object)`.
    Store(String, String, String),
    /// Reinforce a relation (produced by a mutating query).
    Strengthen(String, String, String),
    /// Multiplicatively decay the whole field one step.
    Decay,
}

/// Configuration for the plastic dynamics.
#[derive(Clone, Copy, Debug)]
pub struct GraphConfig {
    /// Whether a query reinforces the queried relation (the observer effect).
    pub strengthen_on_read: bool,
    /// Decay the field once every `decay_interval` mutating queries. 0 disables.
    pub decay_interval: usize,
    pub decay_num: i64,
    pub decay_den: i64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            strengthen_on_read: true,
            decay_interval: 16,
            decay_num: DEFAULT_DECAY_NUM,
            decay_den: DEFAULT_DECAY_DEN,
        }
    }
}

/// A plastic, event-sourced holographic relation store.
pub struct ConnectionGraph {
    dim: usize,
    field: Vec<i64>,
    events: Vec<Event>,
    chain: u64,
    cfg: GraphConfig,
    queries_since_decay: usize,
    /// Durable append-only event log. `Some` when opened with a path; every
    /// mutation is appended so the state survives restart (the event log IS the
    /// state). Best-effort: an IO failure logs a warning and leaves the in-memory
    /// state intact rather than aborting the operation.
    persist: Option<BufWriter<File>>,
}

impl ConnectionGraph {
    /// Create an empty in-memory graph over a `dim`-dimensional field.
    /// Panics if `dim` is 0.
    pub fn new(dim: usize) -> Self {
        Self::with_config(dim, GraphConfig::default())
    }

    /// Create an empty in-memory graph with explicit plastic dynamics.
    pub fn with_config(dim: usize, cfg: GraphConfig) -> Self {
        assert!(dim > 0, "connection graph dim must be non-zero");
        Self {
            dim,
            field: vec![0i64; dim],
            events: Vec::new(),
            chain: 0,
            cfg,
            queries_since_decay: 0,
            persist: None,
        }
    }

    /// Open a durable graph backed by an append-only event log at `path`. If the
    /// file exists, its events are replayed to reconstruct the exact prior state;
    /// subsequent mutations are appended. This is the persistence path: because
    /// the field is a deterministic fold of the log, restoring == replaying.
    pub fn open<P: AsRef<Path>>(dim: usize, cfg: GraphConfig, path: P) -> std::io::Result<Self> {
        assert!(dim > 0, "connection graph dim must be non-zero");
        let path: PathBuf = path.as_ref().to_path_buf();
        let mut g = Self::with_config(dim, cfg);
        if path.exists() {
            let events = read_events(&path)?;
            for ev in &events {
                g.apply(ev);
                g.chain = chain_step(g.chain, ev);
            }
            g.events = events;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        g.persist = Some(BufWriter::new(file));
        Ok(g)
    }

    /// Deterministic sparse vector for an entity id (no codebook stored).
    fn entity_vec(&self, id: &str) -> EntangledHVec {
        EntangledHVec::new_with_density(self.dim, DENSITY_DENOM, str_seed(id))
    }

    /// Role-specific permutation masks for a relation type.
    fn masks(&self, relation: &str) -> (u32, u32) {
        let s = str_seed(relation);
        let subj = (hash_u64(s, 0x5B_1EC7) % self.dim as u64) as u32 | 1;
        let obj = (hash_u64(s, 0x0B_1EC7) % self.dim as u64) as u32 | 1;
        (subj, obj)
    }

    /// Compositional edge index set: union of the role-permuted subject and
    /// object. Edges sharing `(subject, relation)` share the subject half.
    fn edge_indices(&self, subject: &str, relation: &str, object: &str) -> Vec<u32> {
        let (subj_mask, obj_mask) = self.masks(relation);
        let s = self.entity_vec(subject);
        let o = self.entity_vec(object);
        let mut v: Vec<u32> = s.indices().iter().map(|&i| i ^ subj_mask).collect();
        v.extend(o.indices().iter().map(|&i| i ^ obj_mask));
        v.sort_unstable();
        v.dedup();
        v
    }

    fn fold_edge(&mut self, indices: &[u32], w: i64) {
        for &i in indices {
            self.field[i as usize] += w;
        }
    }

    fn apply_decay(&mut self) {
        for v in self.field.iter_mut() {
            *v = *v * self.cfg.decay_num / self.cfg.decay_den;
        }
    }

    /// Fold a single event into the field (no recording/persistence). Shared by
    /// live mutation, replay, and open.
    fn apply(&mut self, ev: &Event) {
        match ev {
            Event::Store(s, r, o) => {
                let idx = self.edge_indices(s, r, o);
                self.fold_edge(&idx, W_STORE);
            }
            Event::Strengthen(s, r, o) => {
                let idx = self.edge_indices(s, r, o);
                self.fold_edge(&idx, W_STRENGTHEN);
            }
            Event::Decay => self.apply_decay(),
        }
    }

    fn record(&mut self, ev: Event) {
        self.chain = chain_step(self.chain, &ev);
        if let Some(writer) = self.persist.as_mut() {
            if let Err(e) = append_event(writer, &ev) {
                tracing::warn!("connection graph persist append failed: {e}");
            }
        }
        self.events.push(ev);
    }

    /// Assert a relation into the store.
    pub fn store(&mut self, subject: &str, relation: &str, object: &str) {
        let idx = self.edge_indices(subject, relation, object);
        self.fold_edge(&idx, W_STORE);
        self.record(Event::Store(
            subject.to_string(),
            relation.to_string(),
            object.to_string(),
        ));
    }

    /// Pure (non-mutating) score of a relation against the field: mean field
    /// amplitude over its edge indices, background-corrected by the field mean.
    /// Higher means "more present". Absent relations sit near zero.
    pub fn score(&self, subject: &str, relation: &str, object: &str) -> f64 {
        let idx = self.edge_indices(subject, relation, object);
        self.score_indices(&idx)
    }

    fn score_indices(&self, idx: &[u32]) -> f64 {
        if idx.is_empty() {
            return 0.0;
        }
        let sum: i64 = idx.iter().map(|&i| self.field[i as usize]).sum();
        (sum as f64 / idx.len() as f64) - self.global_mean()
    }

    fn global_mean(&self) -> f64 {
        self.field.iter().sum::<i64>() as f64 / self.field.len() as f64
    }

    /// Integer presence gate: does the edge's mean amplitude exceed the field
    /// mean by at least half a store-weight? An absent relation samples the
    /// background, so its edge-mean sits *at* the field mean with only sampling
    /// noise (~0.1); a stored relation carries +`W_STORE` on every one of its
    /// indices, so its edge-mean is ~1 above background. The half-store margin
    /// cleanly separates the two and suppresses coincidental-overlap fabrication.
    /// Cross-multiplied in i128 so the decision is exact and platform-independent
    /// (keeps replay bit-exact).
    ///
    /// Condition: `edge_mean - field_mean >= W_STORE/2`, i.e.
    /// `2*(edge_sum*field_len - field_total*edge_len) >= W_STORE*edge_len*field_len`.
    ///
    /// Why an *absolute* margin and not a scale-invariant `k·σ` (SNR) gate: an SNR
    /// gate is scale-invariant under decay, but it cannot BOOTSTRAP — a
    /// freshly-stored relation adds only +W_STORE per index, which is well under a
    /// few σ against a dense multi-relation background, so it is never recognized
    /// as present, never reinforced, and plasticity-under-load dies. Tried it; it
    /// broke `plasticity_holds_discrimination_under_load`. The absolute margin's
    /// tradeoff is the opposite (its threshold's meaning drifts as the field
    /// decays), but that manifests as a *coherent forgetting policy* — a relation
    /// decayed below half a store-weight has genuinely gone cold — which is the
    /// behavior we want. A gate that bootstraps AND tracks decay is open work.
    fn is_present(&self, idx: &[u32]) -> bool {
        if idx.is_empty() {
            return false;
        }
        let edge_sum = idx.iter().map(|&i| self.field[i as usize]).sum::<i64>() as i128;
        let field_total = self.field.iter().sum::<i64>() as i128;
        let edge_len = idx.len() as i128;
        let field_len = self.field.len() as i128;
        2 * (edge_sum * field_len - field_total * edge_len)
            >= (W_STORE as i128) * edge_len * field_len
    }

    /// Query a relation. Returns its presence score. This is a *mutating* read
    /// (the observer effect), but **presence-gated**: it reinforces the queried
    /// relation only if it is actually present (above the field mean), so
    /// querying an absent relation does not fabricate it. When it does reinforce,
    /// it periodically decays the untouched background. Only mutations are
    /// recorded as events — a miss is free (no write, no read-amplification).
    pub fn query(&mut self, subject: &str, relation: &str, object: &str) -> f64 {
        let idx = self.edge_indices(subject, relation, object);
        let s = self.score_indices(&idx);
        if self.cfg.strengthen_on_read && self.is_present(&idx) {
            self.fold_edge(&idx, W_STRENGTHEN);
            self.record(Event::Strengthen(
                subject.to_string(),
                relation.to_string(),
                object.to_string(),
            ));
            if self.cfg.decay_interval > 0 {
                self.queries_since_decay += 1;
                if self.queries_since_decay >= self.cfg.decay_interval {
                    self.apply_decay();
                    self.record(Event::Decay);
                    self.queries_since_decay = 0;
                }
            }
        }
        s
    }

    /// The running tamper-evident chain digest over all events so far.
    pub fn chain_digest(&self) -> u64 {
        self.chain
    }

    /// The recorded event stream (the audit log of every state change).
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Deterministically rebuild the field and chain digest from an event stream.
    /// Used to verify that stored state is exactly the fold of its signed log.
    pub fn replay(dim: usize, cfg: GraphConfig, events: &[Event]) -> (Vec<i64>, u64) {
        let mut g = Self::with_config(dim, cfg);
        for ev in events {
            g.apply(ev);
            g.chain = chain_step(g.chain, ev);
        }
        (g.field, g.chain)
    }

    /// Verify that this graph's current field and chain digest are exactly the
    /// deterministic replay of its own event log. Tampering with any event
    /// (order or content) changes the digest and fails this check.
    pub fn verify(&self) -> bool {
        let (field, chain) = Self::replay(self.dim, self.cfg, &self.events);
        field == self.field && chain == self.chain
    }
}

/// Deterministic 64-bit seed for a string id (FNV-1a folded through the
/// splitmix finalizer for avalanche).
fn str_seed(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash_u64(h, s.len() as u64)
}

/// One step of the tamper-evident event chain. Deterministic; a production
/// deployment replaces this with SHA-256 over a canonical event encoding plus
/// periodic Ed25519 checkpoint signatures (available under the `provenance`
/// feature).
fn chain_step(prev: u64, ev: &Event) -> u64 {
    let (tag, payload) = match ev {
        Event::Store(s, r, o) => (0x5701u64, str_seed(&format!("{s}\u{1}{r}\u{1}{o}"))),
        Event::Strengthen(s, r, o) => (0x5731u64, str_seed(&format!("{s}\u{1}{r}\u{1}{o}"))),
        Event::Decay => (0xDECAu64, 0),
    };
    hash_u64(prev ^ tag, payload)
}

/// Append one length-framed, bincode-encoded event to the durable log.
fn append_event(writer: &mut BufWriter<File>, ev: &Event) -> std::io::Result<()> {
    let bytes = bincode::serialize(ev)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}

/// Read all length-framed events from a durable log. A torn or corrupt final
/// record (e.g. a crash mid-append) stops replay at the last complete event
/// rather than failing, so a partial write never loses committed history.
fn read_events(path: &Path) -> std::io::Result<Vec<Event>> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    let mut events = Vec::new();
    let mut off = 0;
    while off + 4 <= buf.len() {
        let len = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) as usize;
        off += 4;
        if off + len > buf.len() {
            break; // truncated tail from an interrupted append
        }
        match bincode::deserialize::<Event>(&buf[off..off + len]) {
            Ok(ev) => events.push(ev),
            Err(_) => break, // corrupt record: stop at last good event
        }
        off += len;
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_plasticity() -> GraphConfig {
        GraphConfig {
            strengthen_on_read: false,
            decay_interval: 0,
            ..GraphConfig::default()
        }
    }

    #[test]
    fn stored_scores_above_absent() {
        let mut g = ConnectionGraph::with_config(4096, no_plasticity());
        g.store("paris", "capital_of", "france");
        let present = g.score("paris", "capital_of", "france");
        let absent = g.score("berlin", "capital_of", "spain");
        assert!(present > absent, "present {present} !> absent {absent}");
    }

    #[test]
    fn persists_and_reloads_across_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cg.log");

        {
            let mut g = ConnectionGraph::open(4096, no_plasticity(), &path).unwrap();
            g.store("paris", "capital_of", "france");
            g.store("berlin", "capital_of", "germany");
            // BufWriter flushes on drop.
        }

        // Reopen: the durable event log must reconstruct the exact prior state.
        let g = ConnectionGraph::open(4096, no_plasticity(), &path).unwrap();
        assert!(
            g.verify(),
            "reopened graph must equal the replay of its log"
        );
        let present = g.score("paris", "capital_of", "france");
        let absent = g.score("tokyo", "capital_of", "spain");
        assert!(
            present > absent,
            "reloaded relation ({present}) must outscore an absent one ({absent})"
        );
    }

    #[test]
    fn replay_is_bit_exact_and_tamper_evident() {
        let mut g = ConnectionGraph::new(4096);
        g.store("a", "r", "b");
        g.store("c", "r", "d");
        g.query("a", "r", "b");
        g.query("a", "r", "b");
        assert!(g.verify(), "replay of own log must reproduce state exactly");

        // Tamper: drop the first event -> digest must change.
        let mut tampered = g.events().to_vec();
        tampered.remove(0);
        let (_, chain) = ConnectionGraph::replay(4096, GraphConfig::default(), &tampered);
        assert_ne!(
            chain,
            g.chain_digest(),
            "tamper must change the chain digest"
        );
    }

    #[test]
    fn plasticity_holds_discrimination_under_load() {
        // A saturating static field loses the ability to tell a stored relation
        // from an absent one; a plastic field that keeps querying the hot
        // relation retains it. We compare the hot relation's margin (present vs
        // absent) under static vs plastic at high load.
        let dim = 8192;
        let load = 400; // many distractor relations -> saturation
        let hot = ("A", "knows", "B");
        let absent = ("A", "knows", "Z"); // shares (A, knows) prefix, not stored

        let mut stat = ConnectionGraph::with_config(dim, no_plasticity());
        let mut plas = ConnectionGraph::new(dim);
        stat.store(hot.0, hot.1, hot.2);
        plas.store(hot.0, hot.1, hot.2);
        for i in 0..load {
            let s = format!("s{i}");
            let o = format!("o{i}");
            stat.store(&s, "knows", &o);
            plas.store(&s, "knows", &o);
        }
        // Warm up the plastic field by querying the hot relation.
        for _ in 0..80 {
            plas.query(hot.0, hot.1, hot.2);
        }

        let stat_margin =
            stat.score(hot.0, hot.1, hot.2) - stat.score(absent.0, absent.1, absent.2);
        let plas_margin =
            plas.score(hot.0, hot.1, hot.2) - plas.score(absent.0, absent.1, absent.2);
        assert!(
            plas_margin > stat_margin,
            "plastic margin {plas_margin} must exceed static {stat_margin} under load"
        );
    }

    #[test]
    fn generalizes_through_shared_substructure() {
        // The anti-cache property: after storing+strengthening (A, r, B), an
        // UNSTORED relation sharing the (A, r) prefix must score above a fully
        // unrelated unstored relation. A cache would score both at the floor.
        let dim = 8192;
        let mut g = ConnectionGraph::new(dim);
        g.store("A", "knows", "B");
        for _ in 0..40 {
            g.query("A", "knows", "B");
        }
        let shared = g.score("A", "knows", "NEVER_STORED"); // shares (A, knows)
        let unrelated = g.score("X", "likes", "Y"); // shares nothing
        assert!(
            shared > unrelated,
            "shared-substructure {shared} must exceed unrelated {unrelated} (else it is a cache)"
        );
    }

    #[test]
    fn forgetting_is_real() {
        // Honest cost: a relation that is never queried decays out. We store and
        // repeatedly query a HOT set (present -> reinforced -> decay fires) while
        // "cold" is stored but never touched. Its score must drop below fresh.
        let dim = 8192;
        let mut g = ConnectionGraph::new(dim);
        g.store("cold", "rel", "thing");
        let fresh = g.score("cold", "rel", "thing");
        for i in 0..20 {
            g.store(&format!("hot{i}"), "rel", "x");
        }
        for _ in 0..40 {
            for i in 0..20 {
                g.query(&format!("hot{i}"), "rel", "x");
            }
        }
        let after = g.score("cold", "rel", "thing");
        assert!(
            after < fresh,
            "un-queried relation must decay: after {after} !< fresh {fresh}"
        );
    }

    #[test]
    fn absent_query_does_not_confabulate() {
        // The presence-gating decision: querying an absent, unrelated relation
        // must NOT reinforce it (no fabrication) and must emit no event (no
        // read-amplification). Only genuinely-present relations are reinforced.
        let dim = 8192;
        let mut g = ConnectionGraph::new(dim);
        g.store("paris", "capital_of", "france");
        let events_before = g.events().len();
        for _ in 0..50 {
            g.query("zzz", "unrelated_rel", "qqq");
        }
        assert_eq!(
            g.events().len(),
            events_before,
            "absent queries must not mutate the store"
        );
        assert!(
            g.score("zzz", "unrelated_rel", "qqq") < g.score("paris", "capital_of", "france"),
            "the absent relation must stay absent after being queried"
        );

        // A present relation, by contrast, IS reinforced on query (emits events).
        g.query("paris", "capital_of", "france");
        assert!(
            g.events().len() > events_before,
            "present query must reinforce"
        );
    }
}
