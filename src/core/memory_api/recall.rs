// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `recall`, `explain`, `get` and `history`.
//!
//! The default path is document search plus one metadata parse and one map lookup per
//! result. When nothing in the store has a validity window or a supersession link and no
//! recency policy is set, recall asks document search for exactly `k` results with the
//! options the caller gave, so its ranking and scores are document search's own. Otherwise it
//! ranks the candidate limit's worth of candidates and filters them, which is why a store
//! with much history may need a larger `candidate_limit`.

use anyhow::Result;
use std::collections::{BTreeSet, HashSet};

use super::index::{Edge, Index};
use super::stamp::{user_metadata, Stamp};
use super::types::*;
use super::{Memory, MemoryError, RecencyContext};
use crate::core::documents::{terms, DocumentResult, SearchOptions};

/// Searches tried before a recall settles for an index that may trail the searched revision.
const SNAPSHOT_ATTEMPTS: usize = 4;

/// A fact's status at `now`, or none when `known_at` predates its ingest.
pub(super) fn evaluate(
    stamp: Option<&Stamp>,
    edges: &[Edge],
    now: Millis,
    known_at: Option<Millis>,
) -> Option<FactStatus> {
    if let (Some(known), Some(stamp)) = (known_at, stamp) {
        if stamp.ingest > known {
            return None;
        }
    }
    let superseded = edges
        .iter()
        .any(|edge| edge.effective <= now && known_at.is_none_or(|known| edge.ingest <= known));
    if superseded {
        return Some(FactStatus::Superseded);
    }
    let Some(stamp) = stamp else {
        return Some(FactStatus::Current);
    };
    if stamp.valid_to.is_some_and(|to| to <= now) {
        return Some(FactStatus::Expired);
    }
    if stamp.valid_from.is_some_and(|from| from > now) {
        return Some(FactStatus::NotYetValid);
    }
    Some(FactStatus::Current)
}

struct Ranked {
    hit: DocumentResult,
    stamp: Option<Stamp>,
    status: FactStatus,
    recency: f64,
    historical: f64,
    score: f64,
}

struct Run {
    ranked: Vec<Ranked>,
    now: Millis,
    lexical_weight: f64,
    semantic_weight: f64,
    fusion: String,
}

fn sorted_ids(edges: &[Edge]) -> Vec<String> {
    let mut edges: Vec<&Edge> = edges.iter().collect();
    edges.sort_by(|a, b| (a.effective, a.ingest, &a.by).cmp(&(b.effective, b.ingest, &b.by)));
    edges.into_iter().map(|edge| edge.by.clone()).collect()
}

fn matched_terms(query: &str, text: &str) -> Vec<String> {
    let stems = terms(text, true);
    let raw = terms(text, false);
    let mut matched: Vec<String> = Vec::new();
    for token in query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        let lower = token.to_lowercase();
        let stem = terms(&lower, true).into_keys().next();
        let hit = raw.contains_key(&lower) || stem.is_some_and(|stem| stems.contains_key(&stem));
        if hit && !matched.contains(&lower) {
            matched.push(lower);
        }
    }
    matched
}

impl Memory {
    /// Facts relevant to `query`, best first.
    ///
    /// By default only facts that are current now are returned, ranked as document search
    /// ranks them. `options` selects history, an `as_of` or `known_at` time, a subject or
    /// entities, and per-result explanations.
    pub fn recall(&self, query: &str, options: &RecallOptions) -> Result<Vec<Recalled>> {
        let k = options.k.unwrap_or(self.config.default_k);
        check!((1..=1000).contains(&k), "k must be 1..=1000");
        let history = options.history.unwrap_or(self.config.default_history);
        self.ensure_fresh();
        let Some(mut run) = self.run(query, options, history, k, false)? else {
            return Ok(Vec::new());
        };
        let index = self.index.read();
        let explain = options.explain;
        let ranked = std::mem::take(&mut run.ranked);
        Ok(ranked
            .into_iter()
            .enumerate()
            .map(|(position, ranked)| {
                let explanation = explain.then(|| {
                    self.explanation(&index, &run, query, options, &ranked, Some(position + 1))
                });
                self.recalled(&index, ranked, explanation)
            })
            .collect())
    }

    /// Why `id` does or does not come back for `query`: its lexical and semantic scores,
    /// matched terms, recency and validity adjustments and supersession chain.
    ///
    /// The rank is among the ranked candidates with history included, so it can differ from
    /// the position in a current-only recall. A fact outside the candidates is still
    /// explained, with `retrieved: false`.
    pub fn explain(&self, query: &str, id: &str, options: &RecallOptions) -> Result<Explanation> {
        self.ensure_fresh();
        let (version, metadata, text) = self
            .core
            .document_snapshot(id)
            .ok_or_else(|| MemoryError::NotFound(format!("unknown fact {id:?}")))?;
        let k = options.k.unwrap_or(self.config.default_k).min(1000);
        let limit = options
            .candidate_limit
            .unwrap_or_else(|| 100.max(k))
            .clamp(1, 1000);
        let run = self.run(query, options, HistoryMode::All, limit, true)?;
        let index = self.index.read();
        if let Some(run) = &run {
            if let Some(position) = run.ranked.iter().position(|r| r.hit.document_id == id) {
                let ranked = &run.ranked[position];
                return Ok(self.explanation(
                    &index,
                    run,
                    query,
                    options,
                    ranked,
                    Some(position + 1),
                ));
            }
        }
        let stamp = Stamp::from_parts(&version, &metadata);
        let now = options.as_of.unwrap_or_else(|| self.clock.now_millis());
        let status = evaluate(stamp.as_ref(), index.edges(id), now, options.known_at)
            .unwrap_or(FactStatus::Current);
        Ok(Explanation {
            retrieved: false,
            rank: None,
            lexical_score: None,
            semantic_score: None,
            retrieval_score: None,
            lexical_weight: options.lexical_weight.unwrap_or(1.0),
            semantic_weight: options.semantic_weight.unwrap_or(1.0),
            fusion: options.fusion.clone().unwrap_or_else(|| "blend".into()),
            matched_terms: text
                .as_deref()
                .map(|text| matched_terms(query, text))
                .unwrap_or_default(),
            recency_factor: 1.0,
            historical_factor: 1.0,
            final_score: None,
            status,
            as_of: now,
            known_at: options.known_at,
            valid_from: stamp.as_ref().and_then(|s| s.valid_from),
            valid_to: stamp.as_ref().and_then(|s| s.valid_to),
            within_validity: stamp.as_ref().is_none_or(|s| {
                s.valid_from.is_none_or(|from| from <= now) && s.valid_to.is_none_or(|to| now < to)
            }),
            chain: self.chain(&index, id),
        })
    }

    /// One stored fact, or none.
    pub fn get(&self, id: &str) -> Result<Option<FactRecord>> {
        self.ensure_fresh();
        let index = self.index.read();
        Ok(self.record(&index, id, self.clock.now_millis()))
    }

    /// Every version of the fact `id` belongs to: it, what it superseded and what superseded
    /// it, in effect order, each with its status now.
    pub fn history(&self, id: &str) -> Result<Vec<FactRecord>> {
        self.ensure_fresh();
        let index = self.index.read();
        let now = self.clock.now_millis();
        if !self.core.has_document(id) {
            return Err(MemoryError::NotFound(format!("unknown fact {id:?}")).into());
        }
        let mut records: Vec<FactRecord> = index
            .component(id)
            .iter()
            .filter_map(|member| self.record(&index, member, now))
            .collect();
        records.sort_by(|a, b| {
            (a.effective_from, a.ingest_time, &a.id).cmp(&(b.effective_from, b.ingest_time, &b.id))
        });
        Ok(records)
    }

    fn record(&self, index: &Index, id: &str, now: Millis) -> Option<FactRecord> {
        let (version, metadata, text) = self.core.document_snapshot(id)?;
        let stamp = Stamp::from_parts(&version, &metadata);
        let edges = index.edges(id);
        let status = evaluate(stamp.as_ref(), edges, now, None).unwrap_or(FactStatus::Current);
        let ingest_time = stamp.as_ref().map(|s| s.ingest);
        let known = stamp.unwrap_or_default();
        Some(FactRecord {
            id: id.to_owned(),
            text,
            status,
            ingest_time,
            event_time: known.event,
            valid_from: known.valid_from,
            valid_to: known.valid_to,
            effective_from: known.effective_from(),
            subject: known.subject,
            entities: known.entities,
            supersedes: index_predecessors(index, id, &known.supersedes),
            superseded_by: sorted_ids(edges),
            metadata: user_metadata(metadata),
        })
    }

    /// Searches, evaluates and ranks. None when no fact can qualify.
    ///
    /// A fact's status is judged from the supersession index, so the index must describe the
    /// same store revision the search read. A commit landing between the two (this process's
    /// own `remember` before it settles the index, or a write through the core) would otherwise
    /// show a fact next to an unindexed successor and report both as current. After a miss the
    /// index is brought to the store's revision and the search repeats; it takes no writer lock,
    /// so an observer callback running under one may still recall. After `SNAPSHOT_ATTEMPTS`
    /// the last attempt is used as is.
    fn run(
        &self,
        query: &str,
        options: &RecallOptions,
        history: HistoryMode,
        k: u32,
        force_wide: bool,
    ) -> Result<Option<Run>> {
        let mut attempt = 1;
        loop {
            let (run, consistent) = self.run_once(query, options, history, k, force_wide)?;
            if consistent || attempt == SNAPSHOT_ATTEMPTS {
                return Ok(run);
            }
            attempt += 1;
            self.ensure_fresh();
        }
    }

    /// One search and evaluation, and whether the index matched the revision searched.
    fn run_once(
        &self,
        query: &str,
        options: &RecallOptions,
        history: HistoryMode,
        k: u32,
        force_wide: bool,
    ) -> Result<(Option<Run>, bool)> {
        let revision = self.core.revision();
        let (indexed, constrained, restrict) = {
            let index = self.index.read();
            (
                index.revision,
                index.constrained(),
                self.restriction(&index, options),
            )
        };
        let indexed = indexed == Some(revision);
        let restrict = match restrict {
            Some(ids) if ids.is_empty() => return Ok((None, indexed)),
            other => other,
        };
        let needs_status = constrained || options.known_at.is_some();
        if history == HistoryMode::Historical && !needs_status && !force_wide {
            return Ok((None, indexed));
        }
        let filter = history != HistoryMode::All && needs_status;
        let reorder =
            !self.recency.is_identity() || (needs_status && self.config.historical_weight != 1.0);
        let wide = filter || reorder || force_wide;
        let fetch = if wide {
            k.max(options.candidate_limit.unwrap_or_else(|| 100.max(k)))
                .min(1000)
        } else {
            k
        };
        let search = SearchOptions {
            k: Some(fetch),
            candidate_limit: options.candidate_limit,
            filter: options.filter.clone(),
            document_ids: restrict,
            embedding: options.embedding.clone(),
            lexical_weight: options.lexical_weight,
            semantic_weight: options.semantic_weight,
            min_semantic_score: options.min_semantic_score,
            fusion: options.fusion.clone(),
            ..SearchOptions::default()
        };
        let hits = self
            .core
            .search_documents(query, &search)
            .map_err(|error| MemoryError::Invalid(format!("{error}")))?;
        let now = options.as_of.unwrap_or_else(|| self.clock.now_millis());
        let index = self.index.read();
        // Commits bump the revision inside the gate that excludes the search's capture, so an
        // unchanged revision across the search means the capture saw exactly `revision`.
        let consistent =
            indexed && index.revision == Some(revision) && self.core.revision() == revision;
        let mut seen: HashSet<String> = HashSet::with_capacity(hits.len());
        let mut ranked = Vec::with_capacity(hits.len());
        for hit in hits {
            // Without reordering the result is the first `k` kept hits in search order.
            if !reorder && !force_wide && ranked.len() >= k as usize {
                break;
            }
            if !seen.insert(hit.document_id.clone()) {
                continue;
            }
            let stamp = Stamp::from_parts(&hit.version, &hit.metadata);
            let edges = index.edges(&hit.document_id);
            let Some(status) = evaluate(stamp.as_ref(), edges, now, options.known_at) else {
                continue;
            };
            let keep = match history {
                HistoryMode::All => true,
                HistoryMode::Current => status == FactStatus::Current,
                HistoryMode::Historical => status != FactStatus::Current,
            };
            if !keep {
                continue;
            }
            let recency = if self.recency.is_identity() {
                1.0
            } else {
                self.recency.factor(&RecencyContext {
                    now,
                    event_time: stamp.as_ref().and_then(|s| s.event),
                    valid_from: stamp.as_ref().and_then(|s| s.valid_from),
                    ingest_time: stamp.as_ref().map(|s| s.ingest),
                    status,
                })
            };
            let historical = if status == FactStatus::Current {
                1.0
            } else {
                self.config.historical_weight
            };
            let score = if reorder {
                hit.score * recency * historical
            } else {
                hit.score
            };
            ranked.push(Ranked {
                hit,
                stamp,
                status,
                recency,
                historical,
                score,
            });
        }
        if reorder {
            ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
        if !force_wide {
            ranked.truncate(k as usize);
        }
        let run = Run {
            ranked,
            now,
            lexical_weight: options.lexical_weight.unwrap_or(1.0),
            semantic_weight: options.semantic_weight.unwrap_or(1.0),
            fusion: options.fusion.clone().unwrap_or_else(|| "blend".into()),
        };
        Ok((Some(run), consistent))
    }

    /// Ids a subject or entity filter limits recall to; none when unrestricted.
    fn restriction(&self, index: &Index, options: &RecallOptions) -> Option<Vec<String>> {
        let mut allowed: Option<BTreeSet<String>> = None;
        if let Some(subject) = &options.subject {
            allowed = Some(index.subject_ids(subject.trim()).into_iter().collect());
        }
        if let Some(entities) = &options.entities {
            let mut union = BTreeSet::new();
            for entity in entities {
                union.extend(index.entity_ids(entity.trim()).iter().cloned());
            }
            allowed = Some(match allowed {
                Some(existing) => existing.intersection(&union).cloned().collect(),
                None => union,
            });
        }
        allowed.map(|ids| ids.into_iter().collect())
    }

    fn recalled(
        &self,
        index: &Index,
        ranked: Ranked,
        explanation: Option<Explanation>,
    ) -> Recalled {
        let Ranked {
            hit,
            stamp,
            status,
            score,
            ..
        } = ranked;
        let ingest_time = stamp.as_ref().map(|s| s.ingest);
        let known = stamp.unwrap_or_default();
        Recalled {
            superseded_by: sorted_ids(index.edges(&hit.document_id)),
            id: hit.document_id,
            text: hit.text,
            score,
            status,
            ingest_time,
            event_time: known.event,
            valid_from: known.valid_from,
            valid_to: known.valid_to,
            subject: known.subject,
            entities: known.entities,
            supersedes: known.supersedes,
            metadata: user_metadata(hit.metadata),
            source_uri: hit.source_uri,
            explanation,
        }
    }

    fn explanation(
        &self,
        index: &Index,
        run: &Run,
        query: &str,
        options: &RecallOptions,
        ranked: &Ranked,
        rank: Option<usize>,
    ) -> Explanation {
        let stamp = ranked.stamp.as_ref();
        let id = &ranked.hit.document_id;
        Explanation {
            retrieved: true,
            rank,
            lexical_score: Some(ranked.hit.lexical_score),
            semantic_score: ranked.hit.semantic_score,
            retrieval_score: Some(ranked.hit.score),
            lexical_weight: run.lexical_weight,
            semantic_weight: run.semantic_weight,
            fusion: run.fusion.clone(),
            matched_terms: ranked
                .hit
                .text
                .as_deref()
                .map(|text| matched_terms(query, text))
                .unwrap_or_default(),
            recency_factor: ranked.recency,
            historical_factor: ranked.historical,
            final_score: Some(ranked.score),
            status: ranked.status,
            as_of: run.now,
            known_at: options.known_at,
            valid_from: stamp.and_then(|s| s.valid_from),
            valid_to: stamp.and_then(|s| s.valid_to),
            within_validity: stamp.is_none_or(|s| {
                s.valid_from.is_none_or(|from| from <= run.now)
                    && s.valid_to.is_none_or(|to| run.now < to)
            }),
            chain: self.chain(index, id),
        }
    }

    /// The facts `id` supersedes and the facts that supersede it, transitively, in effect
    /// order.
    fn chain(&self, index: &Index, id: &str) -> Vec<ChainLink> {
        let mut links = Vec::new();
        for (relation, members) in [
            (ChainRelation::Predecessor, index.ancestors(id)),
            (ChainRelation::Successor, index.descendants(id)),
        ] {
            for member in members {
                let Some((version, metadata, _)) = self.core.document_snapshot(&member) else {
                    continue;
                };
                let stamp = Stamp::from_parts(&version, &metadata).unwrap_or_default();
                links.push(ChainLink {
                    id: member,
                    relation,
                    effective_from: stamp.effective_from(),
                    ingest_time: stamp.ingest,
                });
            }
        }
        links.sort_by(|a, b| {
            (a.effective_from, a.ingest_time, &a.id).cmp(&(b.effective_from, b.ingest_time, &b.id))
        });
        links
    }
}

/// The facts `id` supersedes: its own stamp's list, which the index mirrors.
fn index_predecessors(index: &Index, id: &str, stamped: &[String]) -> Vec<String> {
    let mut ids: BTreeSet<String> = stamped.iter().cloned().collect();
    ids.extend(index.direct_predecessors(id).iter().cloned());
    ids.into_iter().collect()
}
