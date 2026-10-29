// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in forgetting and consolidation driven by policies. Nothing here runs unless called;
//! the default policies plan nothing.

use anyhow::Result;

use super::recall::evaluate;
use super::stamp::{stable_hash, Stamp};
use super::types::*;
use super::{ConsolidationPolicy, FactSummary, ForgettingPolicy, Memory, MemoryError};

impl Memory {
    /// Every stored fact as a maintenance policy sees it, ordered by id. Plain documents
    /// (without memory fields) are not facts and are left out.
    /// Panics when concurrent writes prevent a coherent snapshot; use [`Self::try_summaries`]
    /// to handle that retryable failure.
    pub fn summaries(&self, with_text: bool) -> Vec<FactSummary> {
        self.try_summaries(with_text)
            .expect("coherent memory summaries")
    }

    pub fn try_summaries(&self, with_text: bool) -> Result<Vec<FactSummary>> {
        let now = self.clock.now_millis();
        self.summaries_snapshot(with_text, now)
            .map(|(_, facts)| facts)
    }

    fn summaries_snapshot(&self, with_text: bool, now: Millis) -> Result<(u64, Vec<FactSummary>)> {
        self.with_snapshot(|index| {
            let mut facts = Vec::new();
            let revision = self.core.scan_documents(|document| {
                let Some(stamp) = Stamp::from_parts(document.version, document.metadata) else {
                    return;
                };
                let edges = index.edges(document.id);
                let status =
                    evaluate(Some(&stamp), edges, now, None).unwrap_or(FactStatus::Current);
                facts.push(FactSummary {
                    id: document.id.to_owned(),
                    text: if with_text { document.text() } else { None },
                    status,
                    ingest_time: Some(stamp.ingest),
                    event_time: stamp.event,
                    valid_from: stamp.valid_from,
                    valid_to: stamp.valid_to,
                    effective_from: stamp.effective_from(),
                    subject: stamp.subject.clone(),
                    superseded_at: edges.iter().map(|edge| edge.effective).min(),
                });
            });
            facts.sort_by(|a, b| a.id.cmp(&b.id));
            check!(
                index.revision == Some(revision),
                "snapshot revision changed"
            );
            Ok((revision, facts))
        })
    }

    /// Forgets what the configured forgetting policy selects. With the default policy that
    /// is nothing.
    pub fn apply_forgetting(&self, options: &ForgetOptions) -> Result<ForgetReport> {
        self.apply_forgetting_with(self.forgetting.as_ref(), options)
    }

    pub fn apply_forgetting_with(
        &self,
        policy: &dyn ForgettingPolicy,
        options: &ForgetOptions,
    ) -> Result<ForgetReport> {
        let now = self.clock.now_millis();
        let mut completed = std::collections::BTreeSet::new();
        for _ in 0..16 {
            let (revision, facts) = self.summaries_snapshot(policy.needs_text(), now)?;
            let known: std::collections::HashSet<&str> =
                facts.iter().map(|fact| fact.id.as_str()).collect();
            let mut ids: Vec<String> = policy
                .select(&facts, now)
                .into_iter()
                .filter(|id| known.contains(id.as_str()))
                .collect();
            ids.sort_unstable();
            ids.dedup();
            if let Some(report) = self.forget_at_revision(
                &ForgetScope::Ids { ids },
                options,
                revision,
                &mut completed,
            )? {
                return Ok(report);
            }
        }
        Err(MemoryError::Concurrent(
            "memory changed during forgetting policy evaluation; retry the operation".into(),
        )
        .into())
    }

    /// Runs the configured consolidation policy. With the default policy that is nothing.
    pub fn consolidate(&self, options: &ConsolidateOptions) -> Result<ConsolidationReport> {
        self.consolidate_with(self.consolidation.as_ref(), options)
    }

    /// Writes each planned summary as a fact that supersedes its sources, so the sources stay
    /// as history. Sources are removed only when the action asks and `allow_destructive` is
    /// set. Summary ids derive from their sources, so repeating a run adds nothing.
    pub fn consolidate_with(
        &self,
        policy: &dyn ConsolidationPolicy,
        options: &ConsolidateOptions,
    ) -> Result<ConsolidationReport> {
        let now = self.clock.now_millis();
        for _ in 0..16 {
            let (revision, facts) = self.summaries_snapshot(policy.needs_text(), now)?;
            let mut inputs = Vec::new();
            let mut doomed = Vec::new();
            for action in policy.plan(&facts, now) {
                if action.sources.is_empty() {
                    continue;
                }
                let mut sources = action.sources.clone();
                sources.sort_unstable();
                sources.dedup();
                let mut summary = action.summary;
                summary.supersedes = Some(sources.clone());
                if summary.id.is_none() {
                    let parts: Vec<&[u8]> = sources.iter().map(|s| s.as_bytes()).collect();
                    summary.id = Some(format!("mem:merged:{}", stable_hash(&parts)));
                }
                if action.forget_sources && options.allow_destructive {
                    doomed.extend(sources);
                }
                inputs.push(summary);
            }
            if inputs.is_empty() {
                if !self.core.memory_revision_matches(revision) {
                    continue;
                }
                return Ok(ConsolidationReport {
                    created: Vec::new(),
                    forgotten: Vec::new(),
                    dry_run: options.dry_run,
                });
            }
            if options.dry_run {
                if !self.core.memory_revision_matches(revision) {
                    continue;
                }
                let now_for_plan = now;
                let created = inputs
                    .iter()
                    .map(|input| Remembered {
                        id: input.id.clone().unwrap_or_default(),
                        created: true,
                        supersedes: input.supersedes.clone().unwrap_or_default(),
                        superseded_by: Vec::new(),
                        ingest_time: now_for_plan,
                    })
                    .collect();
                doomed.sort_unstable();
                doomed.dedup();
                return Ok(ConsolidationReport {
                    created,
                    forgotten: doomed,
                    dry_run: true,
                });
            }
            let Some(created) = self.remember_at_revision(inputs, now, revision)? else {
                continue;
            };
            let forgotten = if doomed.is_empty() {
                Vec::new()
            } else {
                let expected =
                    revision.wrapping_add(u64::from(created.iter().any(|fact| fact.created)));
                let cleanup = self.forget_at_revision(
                &ForgetScope::Ids { ids: doomed },
                &ForgetOptions::default(),
                expected,
                &mut Default::default(),
            ).and_then(|report| report.ok_or_else(|| MemoryError::Concurrent(
                "consolidation committed; sources changed before cleanup; retained sources require reconciliation".into()
            ).into()));
                crate::core::error::after_commit("consolidation source cleanup", cleanup)?.forgotten
            };
            return Ok(ConsolidationReport {
                created,
                forgotten,
                dry_run: false,
            });
        }
        Err(MemoryError::Concurrent(
            "memory changed during consolidation policy evaluation; retry the operation".into(),
        )
        .into())
    }
}
