// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_CLAIMS: usize = 4096;
pub const MAX_SOURCES: usize = 1024;
pub const MAX_SOURCE_BYTES: usize = 64 * 1024;
pub const MAX_TOTAL_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_WORLDS: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub version: String,
    pub text: String,
    pub observed_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub source_id: String,
    pub source_version: String,
    pub start: usize,
    pub end: usize,
    pub quote: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Interval {
    pub start: i64,
    pub end: i64,
}

impl Interval {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.start < self.end, "interval requires start < end");
        Ok(())
    }

    pub fn contains(&self, other: Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    pub fn overlaps(&self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Actual,
    Planned,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Amount {
    pub minor: i64,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub event_key: String,
    pub event_candidates: Vec<String>,
    pub value: Option<Amount>,
    pub event_time: Option<Interval>,
    pub valid_from: i64,
    pub valid_to: Option<i64>,
    pub positive: bool,
    pub modality: Modality,
    pub evidence: Vec<Span>,
    pub supersedes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub subject: String,
    pub predicate: String,
    pub modality: Modality,
    pub event_window: Option<Interval>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    CountDistinct,
    Sum {
        unit: String,
    },
    Compare {
        left_event: String,
        right_event: String,
        unit: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub selection: Selection,
    pub operation: Operation,
    pub as_of: i64,
    pub known_at: Option<i64>,
    pub max_worlds: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Gap {
    NoMatchingEvidence,
    MissingValue { claim_id: String },
    UnknownTime { claim_id: String },
    ConflictingValues { event_id: String },
    IdentityBudget,
    MissingEvent { event_id: String },
    CounterEvidence { claim_id: String },
    EvidenceBudget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerStatus {
    Resolved,
    Bounded,
    Unresolved,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Answer {
    pub status: AnswerStatus,
    pub lower: Option<i64>,
    pub upper: Option<i64>,
    pub unit: String,
    pub observed_only: bool,
    pub worlds: usize,
    pub claim_ids: Vec<String>,
    pub evidence: Vec<Span>,
    pub gaps: Vec<Gap>,
}

pub fn validate_id(value: &str) -> Result<()> {
    ensure!(
        !value.trim().is_empty() && value.len() <= 256 && !value.chars().any(char::is_control),
        "identity requires 1..=256 bytes without control characters"
    );
    Ok(())
}

pub fn validate_query(query: &Query) -> Result<()> {
    validate_id(&query.selection.subject)?;
    validate_id(&query.selection.predicate)?;
    ensure!(
        (1..=MAX_WORLDS).contains(&query.max_worlds),
        "max_worlds requires 1..=256"
    );
    if let Some(window) = query.selection.event_window {
        window.validate()?;
    }
    match &query.operation {
        Operation::CountDistinct => {}
        Operation::Sum { unit } => validate_id(unit)?,
        Operation::Compare {
            left_event,
            right_event,
            unit,
        } => {
            validate_id(left_event)?;
            validate_id(right_event)?;
            validate_id(unit)?;
        }
    }
    Ok(())
}

pub fn validate_sources(sources: &[Source]) -> Result<()> {
    ensure!(sources.len() <= MAX_SOURCES, "too many sources");
    let mut ids = BTreeSet::new();
    let mut bytes = 0usize;
    for source in sources {
        validate_id(&source.id)?;
        validate_id(&source.version)?;
        ensure!(ids.insert(&source.id), "duplicate source identity");
        ensure!(
            !source.text.trim().is_empty() && source.text.len() <= MAX_SOURCE_BYTES,
            "source requires 1..=65536 text bytes"
        );
        bytes = bytes
            .checked_add(source.text.len())
            .context("source byte overflow")?;
        ensure!(bytes <= MAX_TOTAL_BYTES, "sources exceed 8 MiB");
    }
    Ok(())
}

pub fn validate_claims(claims: &[Claim], sources: &[Source]) -> Result<()> {
    validate_sources(sources)?;
    ensure!(claims.len() <= MAX_CLAIMS, "too many claims");
    let sources: BTreeMap<_, _> = sources.iter().map(|s| (s.id.as_str(), s)).collect();
    let mut ids = BTreeSet::new();
    let mut alternatives = BTreeMap::new();
    let mut quote_bytes = 0usize;
    for claim in claims {
        for id in [
            &claim.id,
            &claim.subject,
            &claim.predicate,
            &claim.event_key,
        ] {
            validate_id(id)?;
        }
        ensure!(ids.insert(&claim.id), "duplicate claim identity");
        ensure!(
            (1..=8).contains(&claim.event_candidates.len()),
            "event candidates require 1..=8 entries"
        );
        let mut candidates = BTreeSet::new();
        for candidate in &claim.event_candidates {
            validate_id(candidate)?;
            ensure!(candidates.insert(candidate), "duplicate event candidate");
        }
        if let Some(previous) = alternatives.insert(&claim.event_key, candidates.clone()) {
            ensure!(
                previous == candidates,
                "event identity alternatives disagree"
            );
        }
        ensure!(
            claim.valid_to.is_none_or(|end| end > claim.valid_from),
            "invalid claim validity interval"
        );
        if let Some(time) = claim.event_time {
            time.validate()?;
        }
        if let Some(value) = &claim.value {
            validate_id(&value.unit)?;
        }
        ensure!(claim.supersedes.len() <= 64, "too many supersession links");
        let mut links = BTreeSet::new();
        for id in &claim.supersedes {
            validate_id(id)?;
            ensure!(
                id != &claim.id && links.insert(id),
                "invalid supersession link"
            );
        }
        ensure!(
            (1..=8).contains(&claim.evidence.len()),
            "claim evidence requires 1..=8 spans"
        );
        let mut spans = BTreeSet::new();
        for span in &claim.evidence {
            validate_id(&span.source_id)?;
            validate_id(&span.source_version)?;
            let source = sources
                .get(span.source_id.as_str())
                .context("evidence source is absent")?;
            ensure!(
                source.version == span.source_version,
                "evidence source version is stale"
            );
            ensure!(
                span.start < span.end
                    && source.text.get(span.start..span.end) == Some(span.quote.as_str()),
                "evidence quote does not match source byte range"
            );
            ensure!(spans.insert(span), "duplicate evidence span");
            quote_bytes = quote_bytes
                .checked_add(span.quote.len())
                .context("quote byte overflow")?;
            ensure!(
                quote_bytes <= MAX_TOTAL_BYTES,
                "evidence quotes exceed 8 MiB"
            );
        }
    }
    Ok(())
}
