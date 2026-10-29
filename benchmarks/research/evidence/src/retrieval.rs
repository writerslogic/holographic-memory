// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::contract::{
    validate_claims, validate_query, Answer, AnswerStatus, Claim, Gap, Query, Source, MAX_SOURCES,
    MAX_TOTAL_BYTES,
};
use crate::{execute::execute, store::EvidenceStore};

pub const EMPTY_PACKET_BYTES: usize = b"{\"claims\":[],\"sources\":[]}".len();

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalLimits {
    pub max_sources: usize,
    pub max_source_bytes: usize,
    pub max_packet_bytes: usize,
    pub max_rounds: usize,
    pub initial_k: usize,
    pub repair_k: usize,
}

impl Default for RetrievalLimits {
    fn default() -> Self {
        Self {
            max_sources: 32,
            max_source_bytes: 64 * 1024,
            max_packet_bytes: 128 * 1024,
            max_rounds: 3,
            initial_k: 4,
            repair_k: 8,
        }
    }
}

impl RetrievalLimits {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_SOURCES).contains(&self.max_sources),
            "invalid source budget"
        );
        ensure!(
            (1..=MAX_TOTAL_BYTES).contains(&self.max_source_bytes),
            "invalid source byte budget"
        );
        ensure!(
            (EMPTY_PACKET_BYTES..=MAX_TOTAL_BYTES).contains(&self.max_packet_bytes),
            "invalid serialized packet budget"
        );
        ensure!(
            (1..=8).contains(&self.max_rounds),
            "round budget requires 1..=8"
        );
        ensure!(
            (1..=1000).contains(&self.initial_k),
            "initial_k requires 1..=1000"
        );
        ensure!(
            (1..=MAX_SOURCES).contains(&self.repair_k),
            "invalid repair_k"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalStep {
    pub phase: String,
    pub candidates: Vec<String>,
    pub admitted: Vec<String>,
    pub skipped_for_budget: Vec<String>,
    pub gaps_before: Vec<Gap>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalReport {
    pub query: Query,
    pub terms: String,
    pub answer: Answer,
    pub sources: Vec<Source>,
    pub claims: Vec<Claim>,
    pub steps: Vec<RetrievalStep>,
    pub source_bytes: usize,
    pub catalogue_candidates: usize,
    pub catalogue_source_bytes: usize,
    pub packet_bytes: usize,
    pub limits: RetrievalLimits,
}

#[derive(Default)]
struct Packet {
    sources: BTreeMap<String, Source>,
    claims: BTreeMap<String, Claim>,
}

impl Packet {
    fn sources(&self) -> Vec<Source> {
        self.sources.values().cloned().collect()
    }
    fn claims(&self) -> Vec<Claim> {
        self.claims.values().cloned().collect()
    }
    fn bytes(&self) -> usize {
        self.sources.values().map(|s| s.text.len()).sum()
    }

    fn admit(
        &mut self,
        store: &EvidenceStore,
        ids: &[String],
        limits: RetrievalLimits,
        step: &mut RetrievalStep,
    ) -> Result<()> {
        for id in ids {
            if self.sources.contains_key(id) {
                continue;
            }
            let (sources, claims) = store.load(std::slice::from_ref(id))?;
            ensure!(
                sources.iter().any(|s| &s.id == id),
                "retrieval omitted requested source"
            );
            let mut next_sources = self.sources.clone();
            for source in sources {
                if let Some(old) = next_sources.insert(source.id.clone(), source.clone()) {
                    ensure!(old == source, "retrieval changed a source within a query");
                }
            }
            let bytes = next_sources.values().try_fold(0usize, |total, s| {
                total
                    .checked_add(s.text.len())
                    .ok_or_else(|| anyhow::anyhow!("packet byte overflow"))
            })?;
            if next_sources.len() > limits.max_sources || bytes > limits.max_source_bytes {
                step.skipped_for_budget.push(id.clone());
                continue;
            }
            let mut next_claims = self.claims.clone();
            for claim in claims {
                if let Some(old) = next_claims.insert(claim.id.clone(), claim.clone()) {
                    ensure!(old == claim, "retrieval changed a claim within a query");
                }
            }
            validate_claims(
                &next_claims.values().cloned().collect::<Vec<_>>(),
                &next_sources.values().cloned().collect::<Vec<_>>(),
            )?;
            if packet_size(
                &next_sources.values().collect::<Vec<_>>(),
                &next_claims.values().collect::<Vec<_>>(),
            )? > limits.max_packet_bytes
            {
                step.skipped_for_budget.push(id.clone());
                continue;
            }
            step.admitted.extend(
                next_sources
                    .keys()
                    .filter(|id| !self.sources.contains_key(*id))
                    .cloned(),
            );
            self.sources = next_sources;
            self.claims = next_claims;
        }
        Ok(())
    }
}

fn packet_size<S: Serialize, C: Serialize>(sources: &S, claims: &C) -> Result<usize> {
    Ok(serde_json::to_vec(&serde_json::json!({"sources": sources, "claims": claims}))?.len())
}

pub fn answer_with_repair(
    store: &EvidenceStore,
    terms: &str,
    query: &Query,
    limits: RetrievalLimits,
) -> Result<RetrievalReport> {
    validate_query(query)?;
    limits.validate()?;
    ensure!(
        !terms.trim().is_empty() && terms.len() <= 8192,
        "search terms require 1..=8192 bytes"
    );
    let mut packet = Packet::default();
    let candidates = store.lexical(terms, limits.initial_k)?;
    ensure!(
        candidates.len() <= limits.initial_k,
        "lexical provider exceeded its limit"
    );
    let mut first = RetrievalStep {
        phase: "lexical".into(),
        candidates: candidates.clone(),
        admitted: vec![],
        skipped_for_budget: vec![],
        gaps_before: vec![],
    };
    packet.admit(store, &candidates, limits, &mut first)?;
    let mut steps = vec![first];
    let mut answer = execute(query, &packet.claims(), &packet.sources())?;
    let (indexed_sources, indexed_claims) = store.snapshot()?;
    let catalogue_source_bytes = indexed_sources.iter().map(|source| source.text.len()).sum();
    let catalogue = store.matching(&query.selection, MAX_SOURCES)?;
    ensure!(
        catalogue.len() <= MAX_SOURCES,
        "catalogue exceeded the source limit"
    );
    let catalogue_candidates = catalogue.len();
    if limits.max_rounds > 1 {
        // Exact scope enumeration adds witnesses even when an initial observed answer resolves.
        let mut visited: std::collections::BTreeSet<String> =
            packet.sources.keys().cloned().collect();
        for _ in 1..limits.max_rounds {
            let mut candidates: Vec<_> = catalogue
                .iter()
                .filter(|id| !visited.contains(*id))
                .cloned()
                .collect();
            if candidates.is_empty() {
                break;
            }
            // Named missing events are first; other matching sources provide counterevidence.
            let missing: Vec<_> = answer
                .gaps
                .iter()
                .filter_map(|gap| {
                    if let Gap::MissingEvent { event_id } = gap {
                        Some(event_id)
                    } else {
                        None
                    }
                })
                .collect();
            if !missing.is_empty() {
                candidates.sort_by_key(|id| {
                    !indexed_claims.iter().any(|c| {
                        c.evidence[0].source_id == *id
                            && c.event_candidates
                                .iter()
                                .any(|event| missing.contains(&event))
                    })
                });
            }
            candidates.truncate(limits.repair_k);
            visited.extend(candidates.iter().cloned());
            let mut step = RetrievalStep {
                phase: "scope_repair".into(),
                candidates: candidates.clone(),
                admitted: vec![],
                skipped_for_budget: vec![],
                gaps_before: answer.gaps.clone(),
            };
            packet.admit(store, &candidates, limits, &mut step)?;
            steps.push(step);
            answer = execute(query, &packet.claims(), &packet.sources())?;
        }
    }
    let remaining = catalogue.iter().any(|id| !packet.sources.contains_key(id));
    if remaining || steps.iter().any(|s| !s.skipped_for_budget.is_empty()) {
        if !answer.gaps.contains(&Gap::EvidenceBudget) {
            answer.gaps.push(Gap::EvidenceBudget);
        }
        answer.status = AnswerStatus::Unresolved;
        answer.lower = None;
        answer.upper = None;
    }
    let packet_bytes = packet_size(&packet.sources(), &packet.claims())?;
    Ok(RetrievalReport {
        query: query.clone(),
        terms: terms.to_owned(),
        answer,
        sources: packet.sources(),
        claims: packet.claims(),
        steps,
        source_bytes: packet.bytes(),
        catalogue_candidates,
        catalogue_source_bytes,
        packet_bytes,
        limits,
    })
}
