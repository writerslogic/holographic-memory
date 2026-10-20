// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in forgetting and consolidation driven by policies. Nothing here runs unless called;
//! the default policies plan nothing.

use anyhow::Result;

use super::recall::evaluate;
use super::stamp::{stable_hash, Stamp};
use super::types::*;
use super::{ConsolidationPolicy, FactSummary, ForgettingPolicy, Memory};

impl Memory {
    /// Every stored fact as a maintenance policy sees it, ordered by id. Plain documents
    /// (without memory fields) are not facts and are left out.
    pub fn summaries(&self, with_text: bool) -> Vec<FactSummary> {
        self.ensure_fresh();
        let now = self.clock.now_millis();
        let index = self.index.read();
        let mut facts = Vec::new();
        self.core.scan_documents(|document| {
            let Some(stamp) = Stamp::from_parts(document.version, document.metadata) else {
                return;
            };
            let edges = index.edges(document.id);
            let status = evaluate(Some(&stamp), edges, now, None).unwrap_or(FactStatus::Current);
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
        facts
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
        let facts = self.summaries(policy.needs_text());
        let known: std::collections::HashSet<&str> =
            facts.iter().map(|fact| fact.id.as_str()).collect();
        let mut ids: Vec<String> = policy
            .select(&facts, now)
            .into_iter()
            .filter(|id| known.contains(id.as_str()))
            .collect();
        ids.sort();
        ids.dedup();
        self.forget(&ForgetScope::Ids { ids }, options)
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
        let facts = self.summaries(policy.needs_text());
        let mut inputs = Vec::new();
        let mut doomed = Vec::new();
        for action in policy.plan(&facts, now) {
            if action.sources.is_empty() {
                continue;
            }
            let mut sources = action.sources.clone();
            sources.sort();
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
            return Ok(ConsolidationReport {
                created: Vec::new(),
                forgotten: Vec::new(),
                dry_run: options.dry_run,
            });
        }
        if options.dry_run {
            let now_for_plan = self.clock.now_millis();
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
            doomed.sort();
            doomed.dedup();
            return Ok(ConsolidationReport {
                created,
                forgotten: doomed,
                dry_run: true,
            });
        }
        let created = self.remember_many(inputs)?;
        let forgotten = if doomed.is_empty() {
            Vec::new()
        } else {
            self.forget(&ForgetScope::Ids { ids: doomed }, &ForgetOptions::default())?
                .forgotten
        };
        Ok(ConsolidationReport {
            created,
            forgotten,
            dry_run: false,
        })
    }
}
