// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Baselines and the challenge method behind one shape, [`Answer`]. Every method reads the
//! same metadata (`store::parse_meta`); none is given anything the others lack.
//!
//! | name | mechanism |
//! |------|-----------|
//! | [`GLOBAL_BM25`] | the engine's document search over the whole store, top 4 |
//! | [`SCOPED_BM25_ENGINE`] | memory API `recall` restricted to the family subject, every BM25 hit that is valid as of the query and scope-compatible, ordered latest-first, top 4 |
//! | [`SCOPED_BM25_LOCAL`] | the same contract over the arena block with the in-crate BM25-lite; the faster of the two scoped variants is the latency reference |
//! | [`SCOPED_LOOKUP`] | the scoped exact lookup without the challenge pass: valid claims whose scope covers the query, most specific then latest, top 4; no retraction, supersession or conflict handling |
//! | [`CHALLENGE`] | `Family::answer` |
//! | SQLite | skipped: `rusqlite` is not a dependency of the root crate |
//!
//! A baseline always reports `Status::Resolved` with the first claim of its list as the
//! answer: that is what a blended ranking is, and on a gold `UnresolvedConflict` it counts
//! as wrong.

use anyhow::{Context, Result};
use holographic_memory::core::memory_api::{HistoryMode, Memory, RecallOptions};
use holographic_memory::{HmsCore, SearchOptions};
use serde::{Deserialize, Serialize};

use crate::contract::{
    tokenize, AssertionKind, Family, Millis, QueryInput, RetrievalUnit, Status, EVIDENCE_BUDGET,
};
use crate::store::parse_meta;
use crate::synth::Query;

pub const GLOBAL_BM25: &str = "global_bm25_top4";
pub const SCOPED_BM25_ENGINE: &str = "scoped_bm25_latest_first_engine";
pub const SCOPED_BM25_LOCAL: &str = "scoped_bm25_latest_first_local";
pub const SCOPED_LOOKUP: &str = "scoped_lookup_no_challenge";
pub const CHALLENGE: &str = "challenge";

/// Every method the experiment runs, in report order.
pub const METHODS: [&str; 5] = [
    GLOBAL_BM25,
    SCOPED_BM25_ENGINE,
    SCOPED_BM25_LOCAL,
    SCOPED_LOOKUP,
    CHALLENGE,
];

/// The non-challenge baselines the accuracy margin is measured against.
pub const NON_CHALLENGE: [&str; 4] = [
    GLOBAL_BM25,
    SCOPED_BM25_ENGINE,
    SCOPED_BM25_LOCAL,
    SCOPED_LOOKUP,
];

/// What any method returns for one query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    pub status: Status,
    /// The claim the method answers with; a blended ranking's is its first claim.
    pub primary: Option<String>,
    /// The competing claims when `status` is `UnresolvedConflict`.
    pub branches: Vec<String>,
    /// Every record returned, in order.
    pub items: Vec<String>,
}

impl Answer {
    pub fn absent() -> Self {
        Self {
            status: Status::Resolved,
            primary: None,
            branches: Vec::new(),
            items: Vec::new(),
        }
    }

    /// A ranking: the first claim is the answer.
    pub fn blended(ranked: Vec<(String, AssertionKind)>) -> Self {
        let primary = ranked
            .iter()
            .find(|(_, kind)| *kind == AssertionKind::Claim)
            .map(|(id, _)| id.clone());
        Self {
            status: Status::Resolved,
            primary,
            branches: Vec::new(),
            items: ranked.into_iter().map(|(id, _)| id).collect(),
        }
    }

    pub fn from_unit(unit: RetrievalUnit) -> Self {
        Self {
            status: unit.status,
            primary: unit.primary_claim,
            branches: match unit.status {
                Status::UnresolvedConflict => unit.branches,
                Status::Resolved => Vec::new(),
            },
            items: unit.items,
        }
    }
}

/// (a) Global BM25 top 4 over every record in the store.
pub fn global_bm25(core: &HmsCore, query: &Query) -> Result<Answer> {
    let options = SearchOptions {
        k: Some(EVIDENCE_BUDGET as u32),
        ..SearchOptions::default()
    };
    let hits = core
        .search_documents(&query.text, &options)
        .with_context(|| format!("document search failed for {}", query.id))?;
    let mut ranked: Vec<(String, AssertionKind)> = Vec::with_capacity(hits.len());
    for hit in hits {
        if ranked.iter().any(|(id, _)| *id == hit.document_id) {
            continue;
        }
        let meta = parse_meta(&hit.metadata)
            .with_context(|| format!("document {} has no challenge metadata", hit.document_id))?;
        ranked.push((hit.document_id, meta.kind));
    }
    ranked.truncate(EVIDENCE_BUDGET);
    Ok(Answer::blended(ranked))
}

fn latest_first(mut rows: Vec<(Millis, u32, String, AssertionKind)>) -> Answer {
    rows.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));
    rows.truncate(EVIDENCE_BUDGET);
    Answer::blended(rows.into_iter().map(|(_, _, id, kind)| (id, kind)).collect())
}

/// (b) Scope-filtered latest-first BM25 top 4 through the memory API's recall: the family's
/// subject restricts the search, every BM25 hit that is valid as of the query and whose
/// scope does not contradict the query's is kept, ordered latest-first. Supersession is
/// deliberately not consulted: recency is this baseline's proxy for currency.
pub fn scoped_bm25_engine(memory: &Memory, query: &Query) -> Result<Answer> {
    let options = RecallOptions {
        k: Some(64),
        as_of: Some(query.as_of),
        history: Some(HistoryMode::All),
        subject: Some(query.key.subject()),
        ..RecallOptions::default()
    };
    let hits = memory
        .recall(&query.text, &options)
        .with_context(|| format!("recall failed for {}", query.id))?;
    let mut rows = Vec::with_capacity(hits.len());
    for hit in hits {
        if hit.score <= 0.0 {
            continue;
        }
        let meta = parse_meta(&hit.metadata)
            .with_context(|| format!("fact {} has no challenge metadata", hit.id))?;
        if !meta.valid_at(query.as_of) || !meta.scope.compatible(&query.scope) {
            continue;
        }
        rows.push((meta.valid_from.unwrap_or(Millis::MIN), meta.seq, hit.id, meta.kind));
    }
    Ok(latest_first(rows))
}

/// (b') The same contract as [`scoped_bm25_engine`] over the arena block with BM25-lite.
pub fn scoped_bm25_local(family: &Family, query: &Query) -> Answer {
    let tokens = tokenize(&query.text);
    let rows = family
        .assertions()
        .iter()
        .enumerate()
        .filter(|(index, a)| {
            a.valid_at(query.as_of)
                && a.scope.compatible(&query.scope)
                && family.lexical_score(*index, &tokens) > 0.0
        })
        .map(|(_, a)| (a.effective(), a.seq, a.id.clone(), a.kind))
        .collect();
    latest_first(rows)
}

/// (c) The scoped exact lookup without the challenge pass: claims valid as of the query
/// whose scope covers the query scope, most specific first, then latest. Retractions,
/// supersession markers and concurrent ties are ignored.
pub fn scoped_lookup_no_challenge(family: &Family, query: &Query) -> Answer {
    let mut candidates: Vec<usize> = (0..family.len())
        .filter(|&index| {
            let a = &family.assertions()[index];
            a.is_claim() && a.valid_at(query.as_of) && a.scope.covers(&query.scope)
        })
        .collect();
    candidates.sort_by(|&a, &b| family.rank_key(b).cmp(&family.rank_key(a)));
    candidates.truncate(EVIDENCE_BUDGET);
    Answer::blended(
        candidates
            .into_iter()
            .map(|index| {
                let a = &family.assertions()[index];
                (a.id.clone(), a.kind)
            })
            .collect(),
    )
}

/// The answer-and-challenge contract.
pub fn challenge(family: &Family, query: &Query) -> Answer {
    Answer::from_unit(family.answer(&QueryInput {
        key: &query.key,
        scope: &query.scope,
        as_of: query.as_of,
        text: &query.text,
    }))
}
