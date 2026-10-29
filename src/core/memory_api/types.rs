// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Plain data types of the memory API. Every type is serde-serializable (camelCase) so the
//! HTTP, Node and Python surfaces carry them unchanged. Times are Unix milliseconds.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Unix time in milliseconds.
pub type Millis = i64;

/// One fact to remember.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "server", schemars(extend("required" = ["text"])))]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct RememberInput {
    /// Caller-chosen id. Absent: a deterministic id derived from the content, so retrying the
    /// same call stores the same fact once. Ids may not start with `hms:`.
    pub id: Option<String>,
    #[cfg_attr(feature = "server", schemars(length(min = 1)))]
    pub text: String,
    /// When the thing the fact describes happened (for example a conversation's date). It is
    /// not the ingest time, which the store records itself.
    pub event_time: Option<Millis>,
    /// Start of the interval in which the fact holds. Absent: unbounded.
    pub valid_from: Option<Millis>,
    /// End (exclusive) of the interval in which the fact holds. Absent: unbounded.
    pub valid_to: Option<Millis>,
    /// What the fact is about; the key supersession policies match on.
    pub subject: Option<String>,
    /// Ids of facts this one replaces. They are kept and stay queryable as history.
    pub supersedes: Option<Vec<String>>,
    /// Entities the fact mentions; matched exactly by `RecallOptions::entities`.
    pub entities: Option<Vec<String>>,
    /// Caller metadata (a JSON object); exact top-level matches work as recall filters.
    /// The key `_mem` is reserved.
    pub metadata: Option<Value>,
    /// A dense embedding of the text, when the store has an embedding space.
    pub embedding: Option<Vec<f64>>,
    pub source_uri: Option<String>,
}

/// What `remember` did with one fact.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Remembered {
    pub id: String,
    /// False when the identical fact was already stored under this id (an idempotent retry).
    pub created: bool,
    /// Facts this one supersedes, explicit and policy-resolved.
    pub supersedes: Vec<String>,
    /// Facts that already supersede this one (a late-arriving older fact).
    pub superseded_by: Vec<String>,
    pub ingest_time: Millis,
}

/// A fact's standing at the time recall evaluates it.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FactStatus {
    Current,
    /// A newer fact supersedes it; it stays stored as history.
    Superseded,
    /// Its validity interval ended.
    Expired,
    /// Its validity interval has not started.
    NotYetValid,
}

/// Which facts recall returns.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HistoryMode {
    /// Only facts that are current at the evaluation time. The default.
    #[default]
    Current,
    /// Every fact, with its status; ranking is unchanged.
    All,
    /// Only facts that are not current: superseded, expired or not yet valid.
    Historical,
}

#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct RecallOptions {
    /// Results to return; the configured default (10) when absent. At most 1000.
    #[cfg_attr(feature = "server", schemars(range(min = 1, max = 1000)))]
    pub k: Option<u32>,
    /// Evaluate validity and supersession at this time. Absent: the clock's now.
    pub as_of: Option<Millis>,
    /// Evaluate what the store knew at this ingest time: later facts and later supersessions
    /// are ignored. Absent: everything stored.
    pub known_at: Option<Millis>,
    /// Absent: the configured default.
    pub history: Option<HistoryMode>,
    /// Only facts about this subject.
    pub subject: Option<String>,
    /// Only facts mentioning at least one of these entities.
    pub entities: Option<Vec<String>>,
    /// Exact top-level metadata matches, as in document search.
    pub filter: Option<Value>,
    pub embedding: Option<Vec<f64>>,
    pub lexical_weight: Option<f64>,
    pub semantic_weight: Option<f64>,
    pub min_semantic_score: Option<f64>,
    /// `"blend"` (default) or `"rrf"`, as in document search.
    pub fusion: Option<String>,
    /// Candidates ranked before the validity filter; the default is document search's.
    /// Raise it when most top candidates are history.
    pub candidate_limit: Option<u32>,
    /// Attach an [`Explanation`] to every result. Costs nothing when false.
    pub explain: bool,
}

/// One fact returned by recall.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Recalled {
    pub id: String,
    pub text: Option<String>,
    /// The document search score after the recency policy.
    pub score: f64,
    pub status: FactStatus,
    pub ingest_time: Option<Millis>,
    pub event_time: Option<Millis>,
    pub valid_from: Option<Millis>,
    pub valid_to: Option<Millis>,
    pub subject: Option<String>,
    pub entities: Vec<String>,
    pub supersedes: Vec<String>,
    pub superseded_by: Vec<String>,
    /// The caller's metadata, without the reserved `_mem` key.
    pub metadata: Value,
    pub source_uri: Option<String>,
    pub explanation: Option<Explanation>,
}

#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChainRelation {
    /// This fact (transitively) supersedes the linked one.
    Predecessor,
    /// The linked fact (transitively) supersedes this one.
    Successor,
}

#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChainLink {
    pub id: String,
    pub relation: ChainRelation,
    /// When the link takes effect on the valid-time axis.
    pub effective_from: Millis,
    pub ingest_time: Millis,
}

/// Why a fact matched and how it was adjusted.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Explanation {
    /// False when the fact was not among the ranked candidates for the query (explain
    /// still reports its terms, validity and chain).
    pub retrieved: bool,
    /// 1-based position among the ranked candidates, after the recency policy.
    pub rank: Option<usize>,
    pub lexical_score: Option<f64>,
    pub semantic_score: Option<f64>,
    /// The fused document search score before the recency policy.
    pub retrieval_score: Option<f64>,
    pub lexical_weight: f64,
    pub semantic_weight: f64,
    pub fusion: String,
    /// Query words that match the fact's text (after stemming).
    pub matched_terms: Vec<String>,
    /// The recency policy's multiplier; 1 when none is configured.
    pub recency_factor: f64,
    /// The multiplier for a non-current fact (`historicalWeight`); 1 for a current one.
    pub historical_factor: f64,
    pub final_score: Option<f64>,
    pub status: FactStatus,
    pub as_of: Millis,
    pub known_at: Option<Millis>,
    pub valid_from: Option<Millis>,
    pub valid_to: Option<Millis>,
    pub within_validity: bool,
    pub chain: Vec<ChainLink>,
}

/// A stored fact with its history links, as `get` and `history` return it.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FactRecord {
    pub id: String,
    pub text: Option<String>,
    pub status: FactStatus,
    pub ingest_time: Option<Millis>,
    pub event_time: Option<Millis>,
    pub valid_from: Option<Millis>,
    pub valid_to: Option<Millis>,
    /// When the fact takes effect on the valid-time axis.
    pub effective_from: Millis,
    pub subject: Option<String>,
    pub entities: Vec<String>,
    pub supersedes: Vec<String>,
    pub superseded_by: Vec<String>,
    pub metadata: Value,
}

/// What `forget` removes.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum ForgetScope {
    /// These fact ids.
    Ids { ids: Vec<String> },
    /// Every fact about a subject.
    Subject { subject: String },
    /// Every fact mentioning an entity.
    Entity { entity: String },
    /// Facts ingested before this time.
    IngestedBefore { time: Millis },
    /// Facts whose event time (else valid-from, else ingest time) is before this time.
    EventBefore { time: Millis },
    /// Facts whose caller metadata contains all these top-level values.
    Matching { filter: Value },
}

#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct ForgetOptions {
    /// Also remove every fact linked to a matched one by supersession, in both directions,
    /// so no version of the chain survives. Default: only the matched facts; a successor's
    /// removal makes its predecessor current again.
    pub include_chain: bool,
    /// Report what would be removed without removing it.
    pub dry_run: bool,
    /// Compact the store afterwards. Deleted text stays in the append-only journal until a
    /// compaction rewrites it; this makes the removal physical.
    pub purge: bool,
}

#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForgetReport {
    /// Ids removed (or, for a dry run, that would be removed), sorted.
    pub forgotten: Vec<String>,
    pub dry_run: bool,
    pub purged: bool,
}

/// What a consolidation run did or would do.
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsolidationReport {
    /// Facts written; each supersedes its sources.
    pub created: Vec<Remembered>,
    /// Sources removed. Empty unless destructive consolidation was explicitly allowed.
    pub forgotten: Vec<String>,
    pub dry_run: bool,
}

#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct ConsolidateOptions {
    pub dry_run: bool,
    /// Honor a policy's request to remove consolidated sources. Without it sources remain as
    /// superseded history.
    pub allow_destructive: bool,
}
