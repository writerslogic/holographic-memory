// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic synthetic agent-memory corpus for the Part B kill test.
//!
//! # Mechanism
//! A fact is a role tuple `(agent, verb, patient, cause, time)` over fixed
//! vocabularies. Documents come in base/twin pairs: the twin renders every fact
//! of the base with agent and patient swapped, in the same voice, with the same
//! filler sentences in the same order, so base and twin have IDENTICAL token
//! multisets and differ only in structure. Facts are rendered in active voice
//! (`agent verb patient cause time`) or passive voice
//! (`patient was verb by agent cause time`); the verbs are regular, so past and
//! participle coincide and the two voices differ only by `was` / `by`.
//! Each document has `k in {4, 10, 20}` facts (cycled by pair index) plus
//! `k / 2` irrelevant filler sentences that reuse person names but a disjoint
//! verb set, so they add lexical noise without creating fact triples.
//!
//! # Uniqueness invariant
//! No ordered triple `(agent, verb, patient)` appears in two base documents and
//! no base triple's swap appears anywhere in the base half; therefore a query
//! built from the swapped triple of a base fact is present ONLY in the twin.
//!
//! # Queries
//! * present: a fact of a random document, rendered as its core triple with
//!   the document's voice with probability `p_voice_match` (else the other
//!   voice), optionally followed by its cause phrase;
//! * absent: the role-swapped triple of a fact of a random document; the
//!   query has the same token multiset as a sentence of that document but the
//!   fact is false for every candidate.
//!
//! # Candidate sets
//! Each query carries `candidates` document ids: the source document, the twin
//! (present queries only; the twin is excluded for absent queries because it
//! holds the swapped fact), and the documents with the highest unique-word
//! overlap with the query text, shuffled. Ties break by document id.
//!
//! All randomness comes from a splitmix64 stream seeded by `SynthConfig::seed`.

use std::collections::HashSet;

use serde::Serialize;

/// Person names (agents and patients share the pool so swaps stay plausible).
pub const NAMES: &[&str] = &[
    "alice", "bob", "carol", "dan", "erin", "frank", "grace", "heidi", "ivan", "judy", "karl",
    "laura", "mallory", "niaj", "oscar", "peggy", "quentin", "rupert", "sybil", "trent", "ursula",
    "victor", "wendy", "xavier", "yves", "zoe", "aaron", "bianca", "caleb", "delia", "ethan",
    "fiona", "gavin", "hana", "isaac", "jade", "kevin", "lena", "marcus", "nora", "owen", "priya",
    "quinn", "rosa", "simon", "tara", "umar", "vera", "wade", "xena", "yuri", "zara", "adrian",
    "bella", "cyrus", "dora", "elias", "farah", "george", "holly", "igor", "jonah", "kira", "liam",
    "maya", "nathan", "olga", "pablo", "rhea", "sofia", "tomas", "uma", "vince", "willa", "yara",
    "zane", "amir", "brooke", "colin", "dina", "emil", "freya", "hugo", "iris", "jorge", "kate",
    "leo", "mira", "noel", "otto", "petra", "ravi", "stella", "theo", "vidya", "walter",
];

/// Fact verbs; regular past forms so the passive participle is the same token.
pub const VERBS: &[&str] = &[
    "warned", "hired", "called", "paid", "visited", "sued", "trained", "followed", "invited",
    "replaced", "thanked", "blamed", "interviewed", "mentored", "briefed", "audited",
];

/// Cause phrases.
pub const CAUSES: &[&str] = &[
    "because of the audit", "after the merger", "over the budget", "during the strike",
    "before the deadline", "despite the warning", "after the election", "because of the leak",
    "over the contract", "during the outage", "after the review", "because of the delay",
    "over the invoice", "during the trial", "after the inspection", "because of the error",
    "over the patent", "during the launch", "after the hearing", "because of the shortage",
];

/// Time phrases.
pub const TIMES: &[&str] = &[
    "on monday", "on tuesday", "on wednesday", "on thursday", "on friday", "in january",
    "in march", "in june", "in august", "in october", "last week", "last month", "last spring",
    "last winter", "at noon", "at midnight", "this morning", "yesterday evening", "two days ago",
    "a year earlier",
];

/// Filler verbs (disjoint from `VERBS`).
pub const FILLER_VERBS: &[&str] = &[
    "attended", "joined", "skipped", "hosted", "chaired", "missed", "recorded", "scheduled",
];
/// Filler events.
pub const FILLER_EVENTS: &[&str] = &[
    "briefing", "workshop", "seminar", "auction", "rehearsal", "ceremony", "tournament",
    "exhibit", "lecture", "banquet", "parade", "summit",
];
/// Filler places.
pub const FILLER_PLACES: &[&str] = &[
    "lobby", "archive", "lab", "atrium", "cellar", "garden", "hangar", "library", "studio",
    "theater", "warehouse", "chapel",
];

/// Facts per document, cycled by base/twin pair index.
pub const FACTS_PER_DOC: [usize; 3] = [4, 10, 20];

/// Splitmix64 generator.
pub struct Rng(u64);

impl Rng {
    /// A generator seeded by `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        crate::hv::splitmix64(&mut self.0)
    }

    /// Uniform integer in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// Bernoulli draw with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        unit < p
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

/// A role tuple and the voice it is rendered in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Fact {
    /// Index into `NAMES`.
    pub agent: u16,
    /// Index into `VERBS`.
    pub verb: u16,
    /// Index into `NAMES`.
    pub patient: u16,
    /// Index into `CAUSES`.
    pub cause: u16,
    /// Index into `TIMES`.
    pub time: u16,
    /// Rendered in passive voice.
    pub passive: bool,
}

/// `(agent, verb, patient)`.
pub type Triple = (u16, u16, u16);

impl Fact {
    /// The ordered role triple.
    pub fn triple(&self) -> Triple {
        (self.agent, self.verb, self.patient)
    }

    /// The same fact with agent and patient exchanged.
    pub fn swapped(&self) -> Fact {
        Fact {
            agent: self.patient,
            patient: self.agent,
            ..*self
        }
    }

    /// Core clause in the requested voice, optionally with the cause phrase.
    pub fn render_core(&self, passive: bool, with_cause: bool) -> String {
        let agent = NAMES[usize::from(self.agent)];
        let verb = VERBS[usize::from(self.verb)];
        let patient = NAMES[usize::from(self.patient)];
        let mut s = if passive {
            format!("{patient} was {verb} by {agent}")
        } else {
            format!("{agent} {verb} {patient}")
        };
        if with_cause {
            s.push(' ');
            s.push_str(CAUSES[usize::from(self.cause)]);
        }
        s
    }

    /// Full document sentence (voice from `self.passive`, cause and time appended).
    pub fn render(&self) -> String {
        let mut s = self.render_core(self.passive, true);
        s.push(' ');
        s.push_str(TIMES[usize::from(self.time)]);
        s
    }
}

/// One synthetic document.
#[derive(Clone, Debug, Serialize)]
pub struct SynthDoc {
    /// Document id (`2p` base, `2p + 1` twin).
    pub id: u32,
    /// The paired document with the identical token multiset.
    pub twin: u32,
    /// Rendered text.
    pub text: String,
    /// Facts in rendering order.
    pub facts: Vec<Fact>,
    /// Ordered triples present in this document.
    #[serde(skip)]
    pub triples: HashSet<Triple>,
}

/// One synthetic query.
#[derive(Clone, Debug, Serialize)]
pub struct SynthQuery {
    /// Query id.
    pub id: u32,
    /// Source document.
    pub doc: u32,
    /// Whether the queried fact is present in `doc`.
    pub present: bool,
    /// Query surface form.
    pub text: String,
    /// The role-swapped surface form (used as the structured foil).
    pub swapped_text: String,
    /// The queried ordered triple.
    pub triple: Triple,
    /// Whether the query voice matches the document's rendering of the fact.
    pub voice_match: bool,
    /// Query rendered in passive voice.
    pub passive: bool,
    /// Candidate document ids (shuffled; contains `doc`).
    pub candidates: Vec<u32>,
}

/// Generator parameters.
#[derive(Clone, Debug, Serialize)]
pub struct SynthConfig {
    /// Seed of the splitmix64 stream.
    pub seed: u64,
    /// Total documents (base + twin); rounded down to an even number.
    pub n_docs: usize,
    /// Present and absent queries each.
    pub n_queries_per_class: usize,
    /// Candidate set size per query.
    pub candidates: usize,
    /// Probability that a document fact is rendered in passive voice.
    pub p_passive: f64,
    /// Probability that a query uses the document's voice for the queried fact.
    pub p_voice_match: f64,
    /// Probability that a query appends the cause phrase.
    pub p_query_cause: f64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            seed: 0x5EED_C0FF_EE00_0002,
            n_docs: 2000,
            n_queries_per_class: 500,
            candidates: 20,
            p_passive: 0.3,
            p_voice_match: 0.7,
            p_query_cause: 0.5,
        }
    }
}

/// Generated corpus and queries.
pub struct SynthSet {
    /// Documents, indexed by id.
    pub docs: Vec<SynthDoc>,
    /// Present queries followed by absent queries.
    pub queries: Vec<SynthQuery>,
}

fn filler(rng: &mut Rng) -> String {
    let name = NAMES[rng.below(NAMES.len())];
    let verb = FILLER_VERBS[rng.below(FILLER_VERBS.len())];
    let event = FILLER_EVENTS[rng.below(FILLER_EVENTS.len())];
    let place = FILLER_PLACES[rng.below(FILLER_PLACES.len())];
    format!("{name} {verb} the {event} in the {place}")
}

fn render_doc(sentences: &[String]) -> String {
    let mut text = sentences.join(". ");
    text.push('.');
    text
}

fn word_set(text: &str) -> HashSet<&str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Generates the corpus and both query classes.
pub fn generate(cfg: &SynthConfig) -> SynthSet {
    let mut rng = Rng::new(cfg.seed);
    let n_pairs = cfg.n_docs / 2;
    let mut used: HashSet<Triple> = HashSet::new();
    let mut docs: Vec<SynthDoc> = Vec::with_capacity(n_pairs * 2);

    for p in 0..n_pairs {
        let k = FACTS_PER_DOC[p % FACTS_PER_DOC.len()];
        let mut facts = Vec::with_capacity(k);
        while facts.len() < k {
            let agent = rng.below(NAMES.len()) as u16;
            let patient = rng.below(NAMES.len()) as u16;
            let verb = rng.below(VERBS.len()) as u16;
            if agent == patient
                || used.contains(&(agent, verb, patient))
                || used.contains(&(patient, verb, agent))
            {
                continue;
            }
            used.insert((agent, verb, patient));
            used.insert((patient, verb, agent));
            facts.push(Fact {
                agent,
                verb,
                patient,
                cause: rng.below(CAUSES.len()) as u16,
                time: rng.below(TIMES.len()) as u16,
                passive: rng.chance(cfg.p_passive),
            });
        }
        let fillers: Vec<String> = (0..k / 2).map(|_| filler(&mut rng)).collect();
        let mut order: Vec<usize> = (0..k + fillers.len()).collect();
        rng.shuffle(&mut order);

        let swapped: Vec<Fact> = facts.iter().map(Fact::swapped).collect();
        let build = |set: &[Fact]| -> (String, Vec<Fact>) {
            let mut sentences = Vec::with_capacity(order.len());
            let mut ordered = Vec::with_capacity(set.len());
            for &slot in &order {
                if slot < set.len() {
                    sentences.push(set[slot].render());
                    ordered.push(set[slot]);
                } else {
                    sentences.push(fillers[slot - set.len()].clone());
                }
            }
            (render_doc(&sentences), ordered)
        };
        let base_id = (2 * p) as u32;
        let twin_id = base_id + 1;
        let (base_text, base_facts) = build(&facts);
        let (twin_text, twin_facts) = build(&swapped);
        docs.push(SynthDoc {
            id: base_id,
            twin: twin_id,
            triples: base_facts.iter().map(Fact::triple).collect(),
            text: base_text,
            facts: base_facts,
        });
        docs.push(SynthDoc {
            id: twin_id,
            twin: base_id,
            triples: twin_facts.iter().map(Fact::triple).collect(),
            text: twin_text,
            facts: twin_facts,
        });
    }

    let word_sets: Vec<HashSet<&str>> = docs.iter().map(|d| word_set(&d.text)).collect();
    let mut queries = Vec::with_capacity(cfg.n_queries_per_class * 2);
    let mut next_id = 0u32;
    for present in [true, false] {
        for _ in 0..cfg.n_queries_per_class {
            let doc = rng.below(docs.len());
            let source = docs[doc].facts[rng.below(docs[doc].facts.len())];
            let passive = if rng.chance(cfg.p_voice_match) {
                source.passive
            } else {
                !source.passive
            };
            let with_cause = rng.chance(cfg.p_query_cause);
            let target = if present { source } else { source.swapped() };
            let text = target.render_core(passive, with_cause);
            let swapped_text = target.swapped().render_core(passive, with_cause);

            let twin = docs[doc].twin as usize;
            let query_words = word_set(&text);
            let mut ranked: Vec<(usize, usize)> = (0..docs.len())
                .filter(|&d| d != doc && d != twin)
                .map(|d| {
                    let overlap = query_words
                        .iter()
                        .filter(|w| word_sets[d].contains(*w))
                        .count();
                    (overlap, d)
                })
                .collect();
            ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut candidates: Vec<u32> = vec![doc as u32];
            if present {
                candidates.push(twin as u32);
            }
            for &(_, d) in ranked.iter().take(cfg.candidates.saturating_sub(candidates.len())) {
                candidates.push(d as u32);
            }
            rng.shuffle(&mut candidates);

            queries.push(SynthQuery {
                id: next_id,
                doc: doc as u32,
                present,
                text,
                swapped_text,
                triple: target.triple(),
                voice_match: passive == source.passive,
                passive,
                candidates,
            });
            next_id += 1;
        }
    }
    drop(word_sets);

    SynthSet { docs, queries }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multiset(text: &str) -> Vec<&str> {
        let mut v: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn twins_share_the_token_multiset_and_queries_respect_presence() {
        let cfg = SynthConfig {
            n_docs: 60,
            n_queries_per_class: 30,
            ..SynthConfig::default()
        };
        let set = generate(&cfg);
        assert_eq!(set.docs.len(), 60);
        for d in set.docs.iter().step_by(2) {
            let twin = &set.docs[d.twin as usize];
            assert_eq!(multiset(&d.text), multiset(&twin.text));
            assert_ne!(d.text, twin.text);
        }
        for q in &set.queries {
            assert_eq!(q.candidates.len(), cfg.candidates);
            assert!(q.candidates.contains(&q.doc));
            let holds = |d: u32| set.docs[d as usize].triples.contains(&q.triple);
            assert_eq!(holds(q.doc), q.present);
            if !q.present {
                assert!(q.candidates.iter().all(|&d| !holds(d)));
            }
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let cfg = SynthConfig {
            n_docs: 20,
            n_queries_per_class: 5,
            ..SynthConfig::default()
        };
        let a = generate(&cfg);
        let b = generate(&cfg);
        assert_eq!(a.docs[3].text, b.docs[3].text);
        assert_eq!(a.queries[7].candidates, b.queries[7].candidates);
    }
}
