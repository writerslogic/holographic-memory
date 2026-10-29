// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! A high-level memory API over the document engine: `remember`, `recall`, `forget` and
//! `explain`, with temporal validity, supersession, and opt-in consolidation and forgetting.
//!
//! # Design
//!
//! A fact is an ordinary document. Its memory fields live in one reserved metadata key
//! (`_mem`, see `stamp`), so the API adds no persisted structure: stores written before it
//! open unchanged, and journaling, replay and encryption apply as for any document. The
//! engine's documents, vectors and raw search stay reachable through [`Memory::core`];
//! nothing here is required to use them.
//!
//! Two time axes are kept apart. Valid time is when a fact holds (`valid_from`, `valid_to`,
//! and the event time it describes); knowledge time is when the store ingested it. Recall
//! evaluates validity "as of" a time and, optionally, what was `known_at` an ingest time.
//!
//! Supersession links are written once, on the superseding fact, and never edit the old one,
//! so history is never rewritten or lost. The reverse links live in a derived in-memory
//! index that is rebuilt from the documents whenever the store changes behind the API's back.
//!
//! Every policy is a trait with a default that changes nothing: supersession is explicit,
//! recency does not rescale scores, nothing is forgotten, nothing is consolidated. Each has
//! a serde spec in [`MemoryConfig`] for bindings and may be replaced with any implementation
//! through the `with_*` builders.

/// Fails with [`MemoryError::Invalid`] unless the condition holds.
macro_rules! check {
    ($cond:expr, $($arg:tt)+) => {
        if !($cond) {
            return Err(anyhow::Error::new($crate::core::memory_api::MemoryError::Invalid(
                format!($($arg)+),
            )));
        }
    };
}

/// Fails with [`MemoryError::Invalid`].
macro_rules! invalid {
    ($($arg:tt)+) => {
        return Err(anyhow::Error::new($crate::core::memory_api::MemoryError::Invalid(
            format!($($arg)+),
        )))
    };
}

#[cfg(feature = "conversation-memory")]
mod conversation;
mod index;
mod json;
mod maintenance;
mod policy;
mod recall;
mod stamp;
mod types;
mod write;

use crate::core::sync::RwLock;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::HmsCore;
use index::Index;

#[cfg(feature = "conversation-memory")]
pub use conversation::*;
pub use json::OPERATIONS;
pub use policy::*;
pub use types::*;

/// An error the caller can act on. Anything that is not one of these (journal I/O, a
/// poisoned invariant) reaches the caller as a plain `anyhow` error; downcast with
/// `error.downcast_ref::<MemoryError>()` to tell the two apart.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// The request is malformed or breaks a rule (bad field, unknown superseded id, cycle).
    #[error("{0}")]
    Invalid(String),
    /// The named fact does not exist.
    #[error("{0}")]
    NotFound(String),
    /// The id exists with different content and `onIdConflict` is `reject`.
    #[error("{0}")]
    Conflict(String),
    /// Concurrent writes prevented a coherent plan or recall; retry the operation.
    #[error("{0}")]
    Concurrent(String),
}

/// What `remember` does when an id exists with different content.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IdConflict {
    /// Fail. Default: nothing is overwritten by accident.
    #[default]
    Reject,
    /// Overwrite the stored fact. This discards its previous text; use supersession to keep
    /// history instead.
    Replace,
}

/// Plain-data configuration; every field has a default that changes no behavior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// Results per recall when `k` is absent.
    pub default_k: u32,
    /// Which facts recall returns when `history` is absent.
    pub default_history: HistoryMode,
    pub supersession: SupersessionSpec,
    pub recency: RecencySpec,
    /// Multiplier on the score of non-current facts when recall includes them.
    pub historical_weight: f64,
    pub forgetting: ForgettingSpec,
    pub consolidation: ConsolidationSpec,
    pub on_id_conflict: IdConflict,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            default_k: 10,
            default_history: HistoryMode::Current,
            supersession: SupersessionSpec::Explicit,
            recency: RecencySpec::None,
            historical_weight: 1.0,
            forgetting: ForgettingSpec::KeepAll,
            consolidation: ConsolidationSpec::None,
            on_id_conflict: IdConflict::Reject,
        }
    }
}

impl MemoryConfig {
    pub fn validate(&self) -> Result<()> {
        check!(
            (1..=1000).contains(&self.default_k),
            "defaultK must be 1..=1000"
        );
        check!(
            self.historical_weight.is_finite() && self.historical_weight >= 0.0,
            "historicalWeight must be finite and nonnegative"
        );
        if let RecencySpec::Exponential {
            half_life_ms,
            floor,
            ..
        } = self.recency
        {
            check!(half_life_ms > 0, "recency halfLifeMs must be positive");
            check!(
                (0.0..=1.0).contains(&floor),
                "recency floor must be in [0, 1]"
            );
        }
        if let ForgettingSpec::MaxFacts { limit } = self.forgetting {
            check!(limit > 0, "forgetting limit must be positive");
        }
        Ok(())
    }
}

/// The memory API over one engine.
///
/// Safe to share across threads. Policy plans use revision-conditional commits; reads retry
/// when their document and policy snapshots differ. Persistent commits are serialized.
/// Several `Memory` values over one core stay correct: each notices the others' writes by
/// the store revision and rebuilds its index, at the cost of one scan.
pub struct Memory {
    core: Arc<HmsCore>,
    config: MemoryConfig,
    supersession: Arc<dyn SupersessionPolicy>,
    /// Whether the supersession policy can return links; the default never does, so writes
    /// skip building its candidates.
    supersession_active: bool,
    recency: Arc<dyn RecencyPolicy>,
    forgetting: Arc<dyn ForgettingPolicy>,
    consolidation: Arc<dyn ConsolidationPolicy>,
    entities: Option<Arc<dyn EntityExtractor>>,
    observer: Option<Arc<dyn MemoryObserver>>,
    clock: Arc<dyn Clock>,
    index: RwLock<Index>,
}

impl Memory {
    /// A memory over `core` with the default, do-nothing policies.
    pub fn new(core: Arc<HmsCore>) -> Self {
        Self::with_config(core, MemoryConfig::default()).expect("the default config is valid")
    }

    pub fn with_config(core: Arc<HmsCore>, config: MemoryConfig) -> Result<Self> {
        config.validate()?;
        let supersession_active = config.supersession != SupersessionSpec::Explicit;
        let supersession: Arc<dyn SupersessionPolicy> = match config.supersession {
            SupersessionSpec::Explicit => Arc::new(ExplicitOnly),
            SupersessionSpec::LatestBySubject => Arc::new(LatestBySubject),
        };
        let recency: Arc<dyn RecencyPolicy> = match config.recency {
            RecencySpec::None => Arc::new(NoRecency),
            RecencySpec::Exponential {
                half_life_ms,
                basis,
                floor,
            } => Arc::new(ExponentialRecency {
                half_life_ms,
                basis,
                floor,
            }),
        };
        let forgetting: Arc<dyn ForgettingPolicy> = match config.forgetting {
            ForgettingSpec::KeepAll => Arc::new(KeepAll),
            ForgettingSpec::ExpiredFor { grace_ms } => Arc::new(ExpiredFor { grace_ms }),
            ForgettingSpec::SupersededFor { age_ms } => Arc::new(SupersededFor { age_ms }),
            ForgettingSpec::MaxFacts { limit } => Arc::new(MaxFacts { limit }),
        };
        let consolidation: Arc<dyn ConsolidationPolicy> = match config.consolidation {
            ConsolidationSpec::None => Arc::new(NoConsolidation),
            ConsolidationSpec::CollapseDuplicates => Arc::new(CollapseDuplicates),
        };
        Ok(Self {
            core,
            config,
            supersession,
            supersession_active,
            recency,
            forgetting,
            consolidation,
            entities: None,
            observer: None,
            clock: Arc::new(SystemClock),
            index: RwLock::new(Index::default()),
        })
    }

    /// Replaces the supersession policy.
    pub fn with_supersession(mut self, policy: Arc<dyn SupersessionPolicy>) -> Self {
        self.supersession = policy;
        self.supersession_active = true;
        self
    }

    pub fn with_recency(mut self, policy: Arc<dyn RecencyPolicy>) -> Self {
        self.recency = policy;
        self
    }

    pub fn with_forgetting(mut self, policy: Arc<dyn ForgettingPolicy>) -> Self {
        self.forgetting = policy;
        self
    }

    pub fn with_consolidation(mut self, policy: Arc<dyn ConsolidationPolicy>) -> Self {
        self.consolidation = policy;
        self
    }

    pub fn with_entity_extractor(mut self, extractor: Arc<dyn EntityExtractor>) -> Self {
        self.entities = Some(extractor);
        self
    }

    pub fn with_observer(mut self, observer: Arc<dyn MemoryObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Replaces the clock, for deterministic ingest times.
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// The engine underneath, for documents, vectors and raw search. Facts are documents
    /// with the fact id as the document id.
    pub fn core(&self) -> &Arc<HmsCore> {
        &self.core
    }

    fn with_snapshot<T>(&self, mut read: impl FnMut(&Index) -> Result<T>) -> Result<T> {
        for _ in 0..16 {
            let Some(revision) = self.planning_revision() else {
                continue;
            };
            let result = {
                let index = self.index.read();
                if index.revision != Some(revision) {
                    continue;
                }
                read(&index)
            };
            if self.core.memory_revision_matches(revision) {
                return result;
            }
        }
        Err(MemoryError::Concurrent(
            "memory changed during the snapshot; retry the operation".into(),
        )
        .into())
    }

    fn planning_revision(&self) -> Option<u64> {
        self.ensure_fresh();
        let revision = self.core.revision();
        (self.index.read().revision == Some(revision)).then_some(revision)
    }

    /// Brings the index up to the store's current revision.
    fn ensure_fresh(&self) {
        let revision = self.core.revision();
        if self.index.read().revision == Some(revision) {
            return;
        }
        let mut index = self.index.write();
        if index.revision == Some(self.core.revision()) {
            return;
        }
        *index = Index::rebuild(&self.core);
    }

    /// Records that this process's own commit moved the store from `before` by exactly
    /// `commits` revisions and the index already reflects it; anything else (another writer
    /// interleaved, or the change was not tracked) leaves the index to rebuild. `apply`
    /// returns false, before changing the index, when it cannot track the change.
    fn settle(&self, before: u64, commits: u64, apply: impl FnOnce(&mut Index) -> bool) {
        let after = self.core.revision();
        let mut index = self.index.write();
        index.revision =
            (index.revision == Some(before) && after == before + commits && apply(&mut index))
                .then_some(after);
    }
}

#[cfg(test)]
mod tests;
