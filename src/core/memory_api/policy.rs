// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Pluggable policies. Each is a trait with a documented default that does nothing
//! destructive and nothing surprising, a built-in or two, and a serde-friendly spec so
//! bindings can select a built-in by name. Rust callers may pass any implementation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};

use super::types::{FactStatus, Millis, RememberInput};

/// The source of "now" for ingest times and for recall without `as_of`.
pub trait Clock: Send + Sync {
    fn now_millis(&self) -> Millis;
}

/// The system clock. Default.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_millis(&self) -> Millis {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| {
                i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
            })
    }
}

/// A clock that only moves when told to; for deterministic runs and tests.
#[derive(Debug, Default)]
pub struct ManualClock(AtomicI64);

impl ManualClock {
    pub fn new(now: Millis) -> Self {
        Self(AtomicI64::new(now))
    }

    pub fn set(&self, now: Millis) {
        self.0.store(now, Ordering::SeqCst);
    }

    pub fn advance(&self, by: Millis) {
        self.0.fetch_add(by, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now_millis(&self) -> Millis {
        self.0.load(Ordering::SeqCst)
    }
}

/// A fact as supersession policies see it.
#[derive(Clone, Debug, PartialEq)]
pub struct FactView {
    pub id: String,
    pub subject: Option<String>,
    /// When the fact takes effect on the valid-time axis: valid-from, else event time, else
    /// ingest time.
    pub effective_from: Millis,
    pub ingest_time: Millis,
}

/// The links a policy asks for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resolution {
    /// Existing facts the incoming fact supersedes.
    pub supersedes: Vec<String>,
    /// Existing facts that already supersede the incoming fact: it arrived late and is born
    /// history.
    pub superseded_by: Vec<String>,
}

/// Decides, when a fact with a subject is remembered, which stored facts about the same
/// subject it supersedes or is superseded by. Explicit `supersedes` ids on the input always
/// apply in addition. Ids returned must come from `same_subject`; others are an error.
///
/// Default: [`ExplicitOnly`].
pub trait SupersessionPolicy: Send + Sync {
    /// `same_subject` holds the stored facts of the incoming fact's subject ordered by
    /// effect, ingest time and id, including earlier facts of the same batch.
    fn resolve(&self, incoming: &FactView, same_subject: &[FactView]) -> Resolution;
}

/// Supersession happens only where the caller names it. Default.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExplicitOnly;

impl SupersessionPolicy for ExplicitOnly {
    fn resolve(&self, _: &FactView, _: &[FactView]) -> Resolution {
        Resolution::default()
    }
}

/// Treats a subject as a single-valued slot on the valid-time axis: a fact supersedes the
/// latest earlier-or-equal fact of its subject, and is itself superseded by the earliest
/// later one. Arrival order does not matter, so a late-ingested older fact becomes history
/// instead of overriding newer knowledge. Use subjects that name one slot ("alice/city"),
/// not a topic that holds several facts.
#[derive(Clone, Copy, Debug, Default)]
pub struct LatestBySubject;

impl SupersessionPolicy for LatestBySubject {
    fn resolve(&self, incoming: &FactView, same_subject: &[FactView]) -> Resolution {
        let mut resolution = Resolution::default();
        let before = same_subject
            .iter()
            .rfind(|fact| fact.id != incoming.id && fact.effective_from <= incoming.effective_from);
        if let Some(fact) = before {
            resolution.supersedes.push(fact.id.clone());
        }
        let after = same_subject
            .iter()
            .find(|fact| fact.id != incoming.id && fact.effective_from > incoming.effective_from);
        if let Some(fact) = after {
            resolution.superseded_by.push(fact.id.clone());
        }
        resolution
    }
}

/// What a recency policy sees of a fact.
#[derive(Clone, Copy, Debug)]
pub struct RecencyContext {
    pub now: Millis,
    pub event_time: Option<Millis>,
    pub valid_from: Option<Millis>,
    pub ingest_time: Option<Millis>,
    pub status: FactStatus,
}

/// Multiplies a result's retrieval score. Default: [`NoRecency`], which leaves scores
/// exactly as document search returned them.
pub trait RecencyPolicy: Send + Sync {
    /// True when the policy never changes a score; recall then skips it and keeps the
    /// document search order.
    fn is_identity(&self) -> bool {
        false
    }

    /// A finite, nonnegative multiplier.
    fn factor(&self, fact: &RecencyContext) -> f64;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoRecency;

impl RecencyPolicy for NoRecency {
    fn is_identity(&self) -> bool {
        true
    }

    fn factor(&self, _: &RecencyContext) -> f64 {
        1.0
    }
}

/// Which time a recency decay measures from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecencyBasis {
    /// Event time, else valid-from, else ingest time. Default.
    #[default]
    Event,
    Ingest,
}

/// `0.5^(age / half_life)`, never below `floor`. Facts without a time, and facts newer than
/// the evaluation time, are not decayed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExponentialRecency {
    pub half_life_ms: Millis,
    pub basis: RecencyBasis,
    pub floor: f64,
}

impl RecencyPolicy for ExponentialRecency {
    fn factor(&self, fact: &RecencyContext) -> f64 {
        let at = match self.basis {
            RecencyBasis::Event => fact.event_time.or(fact.valid_from).or(fact.ingest_time),
            RecencyBasis::Ingest => fact.ingest_time,
        };
        let Some(at) = at else { return 1.0 };
        let age = fact.now.saturating_sub(at);
        if age <= 0 {
            return 1.0;
        }
        (0.5f64.powf(age as f64 / self.half_life_ms as f64)).max(self.floor)
    }
}

/// A stored fact as maintenance policies see it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FactSummary {
    pub id: String,
    /// Present only when the policy's `needs_text` is true.
    pub text: Option<String>,
    pub status: FactStatus,
    pub ingest_time: Option<Millis>,
    pub event_time: Option<Millis>,
    pub valid_from: Option<Millis>,
    pub valid_to: Option<Millis>,
    pub effective_from: Millis,
    pub subject: Option<String>,
    /// The earliest time a superseder takes effect, when one exists.
    pub superseded_at: Option<Millis>,
}

/// Chooses facts to remove. Default: [`KeepAll`].
pub trait ForgettingPolicy: Send + Sync {
    fn needs_text(&self) -> bool {
        false
    }

    /// Ids to forget, from the facts stored at `now`.
    fn select(&self, facts: &[FactSummary], now: Millis) -> Vec<String>;
}

/// Never forgets. Default.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeepAll;

impl ForgettingPolicy for KeepAll {
    fn select(&self, _: &[FactSummary], _: Millis) -> Vec<String> {
        Vec::new()
    }
}

/// Forgets facts whose validity ended more than `grace_ms` ago.
#[derive(Clone, Copy, Debug)]
pub struct ExpiredFor {
    pub grace_ms: Millis,
}

impl ForgettingPolicy for ExpiredFor {
    fn select(&self, facts: &[FactSummary], now: Millis) -> Vec<String> {
        facts
            .iter()
            .filter(|fact| {
                fact.valid_to
                    .is_some_and(|to| now.saturating_sub(to) > self.grace_ms)
            })
            .map(|fact| fact.id.clone())
            .collect()
    }
}

/// Forgets superseded facts whose superseder took effect more than `age_ms` ago: prunes old
/// history while current facts are untouched.
#[derive(Clone, Copy, Debug)]
pub struct SupersededFor {
    pub age_ms: Millis,
}

impl ForgettingPolicy for SupersededFor {
    fn select(&self, facts: &[FactSummary], now: Millis) -> Vec<String> {
        facts
            .iter()
            .filter(|fact| {
                fact.status == FactStatus::Superseded
                    && fact
                        .superseded_at
                        .is_some_and(|at| now.saturating_sub(at) > self.age_ms)
            })
            .map(|fact| fact.id.clone())
            .collect()
    }
}

/// Keeps at most `limit` facts: forgets the non-current ones first, oldest ingested first,
/// then the oldest current ones.
#[derive(Clone, Copy, Debug)]
pub struct MaxFacts {
    pub limit: usize,
}

impl ForgettingPolicy for MaxFacts {
    fn select(&self, facts: &[FactSummary], _: Millis) -> Vec<String> {
        let excess = facts.len().saturating_sub(self.limit);
        let mut order: Vec<&FactSummary> = facts.iter().collect();
        order.sort_by(|a, b| {
            let rank = |fact: &FactSummary| fact.status == FactStatus::Current;
            (rank(a), a.ingest_time, &a.id).cmp(&(rank(b), b.ingest_time, &b.id))
        });
        order
            .into_iter()
            .take(excess)
            .map(|fact| fact.id.clone())
            .collect()
    }
}

/// One consolidation step: a new fact that supersedes `sources`.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsolidationAction {
    pub sources: Vec<String>,
    /// The merged fact. Its `supersedes` is filled with `sources`.
    pub summary: RememberInput,
    /// Ask for the sources to be removed. Ignored unless the run allows destructive
    /// consolidation.
    pub forget_sources: bool,
}

/// Plans merges of related facts. Default: [`NoConsolidation`]. A plan only ever adds facts
/// that supersede their sources, so history survives unless the caller opts into removal.
pub trait ConsolidationPolicy: Send + Sync {
    fn needs_text(&self) -> bool {
        false
    }

    fn plan(&self, facts: &[FactSummary], now: Millis) -> Vec<ConsolidationAction>;
}

/// Plans nothing. Default.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoConsolidation;

impl ConsolidationPolicy for NoConsolidation {
    fn plan(&self, _: &[FactSummary], _: Millis) -> Vec<ConsolidationAction> {
        Vec::new()
    }
}

/// Merges current facts whose text is identical after trimming and case folding (and that
/// share a subject) into one fact superseding all of them.
#[derive(Clone, Copy, Debug, Default)]
pub struct CollapseDuplicates;

impl ConsolidationPolicy for CollapseDuplicates {
    fn needs_text(&self) -> bool {
        true
    }

    fn plan(&self, facts: &[FactSummary], _: Millis) -> Vec<ConsolidationAction> {
        let mut groups: BTreeMap<(Option<&str>, String), Vec<&FactSummary>> = BTreeMap::new();
        for fact in facts.iter().filter(|f| f.status == FactStatus::Current) {
            let Some(text) = fact.text.as_deref() else {
                continue;
            };
            let key = text
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            groups
                .entry((fact.subject.as_deref(), key))
                .or_default()
                .push(fact);
        }
        groups
            .into_values()
            .filter(|group| group.len() > 1)
            .map(|mut group| {
                group.sort_by(|a, b| {
                    (a.effective_from, a.ingest_time, &a.id).cmp(&(
                        b.effective_from,
                        b.ingest_time,
                        &b.id,
                    ))
                });
                let newest = group[group.len() - 1];
                let sources: Vec<String> = group.iter().map(|fact| fact.id.clone()).collect();
                ConsolidationAction {
                    summary: RememberInput {
                        text: newest.text.clone().unwrap_or_default(),
                        event_time: newest.event_time,
                        valid_from: newest.valid_from,
                        valid_to: newest.valid_to,
                        subject: newest.subject.clone(),
                        supersedes: Some(sources.clone()),
                        ..RememberInput::default()
                    },
                    sources,
                    forget_sources: false,
                }
            })
            .collect()
    }
}

/// Finds the entities a fact mentions. Optional; default none. The result is stored with
/// the fact and matched exactly by `RecallOptions::entities`. This and [`MemoryObserver`]
/// are the extension points for a graph or entity layer; nothing in the store requires one.
pub trait EntityExtractor: Send + Sync {
    fn entities(&self, text: &str) -> Vec<String>;
}

impl<F: Fn(&str) -> Vec<String> + Send + Sync> EntityExtractor for F {
    fn entities(&self, text: &str) -> Vec<String> {
        self(text)
    }
}

/// Receives memory events after they are durable. Default: none. Callbacks run on the
/// calling thread and must not call back into the same `Memory`'s writers.
pub trait MemoryObserver: Send + Sync {
    fn on_remembered(&self, _fact: &super::types::Remembered, _input: &RememberInput) {}
    fn on_forgotten(&self, _ids: &[String]) {}
}

/// Built-in supersession policies by name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SupersessionSpec {
    /// Only explicit `supersedes`. Default.
    #[default]
    Explicit,
    /// [`LatestBySubject`].
    LatestBySubject,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "kebab-case",
    tag = "kind",
    rename_all_fields = "camelCase"
)]
pub enum RecencySpec {
    /// Scores are untouched. Default.
    #[default]
    None,
    Exponential {
        half_life_ms: Millis,
        #[serde(default)]
        basis: RecencyBasis,
        #[serde(default)]
        floor: f64,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "kebab-case",
    tag = "kind",
    rename_all_fields = "camelCase"
)]
pub enum ForgettingSpec {
    /// Never forgets. Default.
    #[default]
    KeepAll,
    ExpiredFor {
        grace_ms: Millis,
    },
    SupersededFor {
        age_ms: Millis,
    },
    MaxFacts {
        limit: usize,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConsolidationSpec {
    /// Plans nothing. Default.
    #[default]
    None,
    CollapseDuplicates,
}
