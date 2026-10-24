// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{
    aggregation_intent, invalid, label, query_words, round_index, sha256, source_id,
    ConversationEvidence, ConversationMemory, ConversationRecallOptions, ConversationRole,
    ConversationSource, Millis, MAX_CONVERSATION_BYTES, MAX_CONVERSATION_METADATA_BYTES,
    MAX_CONVERSATION_RECORDS,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct ConversationContextLimits {
    pub max_bytes: usize,
    pub max_sources: usize,
    /// Actual original rounds, including all returned premise sources.
    pub max_rounds: usize,
    /// Restrict expansion to counting/aggregation text. False permits any query.
    pub aggregation_only: bool,
}

impl Default for ConversationContextLimits {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024,
            max_sources: 64,
            max_rounds: 50,
            aggregation_only: true,
        }
    }
}

impl ConversationContextLimits {
    fn validate(&self) -> Result<()> {
        if !(2..=MAX_CONVERSATION_BYTES).contains(&self.max_bytes)
            || !(1..=MAX_CONVERSATION_RECORDS).contains(&self.max_sources)
            || !(1..=MAX_CONVERSATION_RECORDS).contains(&self.max_rounds)
        {
            return Err(invalid("invalid conversation expansion bounds"));
        }
        Ok(())
    }
}

/// Lossless source context with shared session identities and ordinal premise references.
///
/// Each turn is `[session, index, source ID, role, observation time, text, premises]`.
/// A premise is a local ordinal or an unresolved original record ID.
/// This is an optional context encoding, not a retrieval unit or a sufficiency claim.
/// The digest checks the reconstructed sources; it does not authenticate their author.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConversationContext {
    schema_version: u32,
    sufficiency: ContextSufficiency,
    sources_sha256: String,
    sessions: Vec<(String, String)>,
    turns: Vec<ContextTurn>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ContextTurn(
    usize,
    usize,
    String,
    ConversationRole,
    Millis,
    String,
    Vec<ContextPremise>,
);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum ContextPremise {
    Local(usize),
    Unresolved(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ContextSufficiency {
    Unverified,
}

fn premise_id(id: &str) -> bool {
    id.strip_prefix("conversation:source:")
        .is_some_and(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn premise_closure(
    manifest: &super::Manifest,
    positions: &BTreeMap<String, usize>,
    root: usize,
) -> Result<Vec<usize>> {
    let mut pending = vec![root];
    let mut visited = vec![false; manifest.sources.len()];
    let mut indices = Vec::new();
    while let Some(index) = pending.pop() {
        let entry = manifest
            .sources
            .get(index)
            .ok_or_else(|| invalid("conversation expansion source ordinal differs"))?;
        if visited[index] {
            continue;
        }
        visited[index] = true;
        indices.push(index);
        for id in &entry.premise_record_ids {
            let premise = positions
                .get(id.as_str())
                .copied()
                .filter(|premise| *premise < index)
                .ok_or_else(|| invalid("conversation expansion premise is missing or forward"))?;
            if !visited[premise] {
                pending.push(premise);
            }
        }
    }
    indices.sort_unstable();
    Ok(indices)
}

impl ConversationMemory {
    /// Opt-in candidate census; returned evidence keeps its original retrieval budget.
    pub fn recall_with_aggregation_context(
        &self,
        query: &str,
        options: &ConversationRecallOptions,
        limits: &ConversationContextLimits,
    ) -> Result<(ConversationEvidence, ConversationContext)> {
        limits.validate()?;
        let seed = self.recall(query, options)?;
        let candidates = if !limits.aggregation_only || aggregation_intent(&query_words(query)) {
            self.recall(
                query,
                &ConversationRecallOptions {
                    k: 128,
                    candidate_limit: 1000,
                    max_sources: MAX_CONVERSATION_RECORDS,
                    max_bytes: MAX_CONVERSATION_BYTES,
                    max_rounds: None,
                    ..options.clone()
                },
            )?
        } else {
            seed.clone()
        };
        let context = self.expand_aggregation_context(query, &seed, &candidates, limits)?;
        Ok((seed, context))
    }

    /// Experimental enumeration evidence expansion. Retains every seed source, then admits
    /// immediate candidate closures in score order before enriching admitted anchors with
    /// transitive premises within the remaining compact byte and raw-round limits.
    /// Exact stored premises may be recovered even when they were not search hits.
    /// Candidate acquisition is caller-controlled. Neither exhausted candidates nor the
    /// returned mention set establishes an exhaustive event count.
    pub fn expand_aggregation_context(
        &self,
        query: &str,
        seed: &ConversationEvidence,
        candidates: &ConversationEvidence,
        limits: &ConversationContextLimits,
    ) -> Result<ConversationContext> {
        limits.validate()?;
        if query.trim().is_empty()
            || query.len() > 8192
            || candidates.anchors.len() > MAX_CONVERSATION_RECORDS
            || candidates.sources.len() > MAX_CONVERSATION_RECORDS
            || seed.sources.len() > limits.max_sources
        {
            return Err(invalid("invalid conversation expansion bounds"));
        }
        let mut current = seed.pack_context(limits.max_bytes)?;
        let expand = !limits.aggregation_only || aggregation_intent(&query_words(query));
        if expand {
            ConversationContext::from_sources(&candidates.sources, MAX_CONVERSATION_BYTES)?;
        }
        let mut available = BTreeMap::new();
        for source in seed
            .sources
            .iter()
            .chain(candidates.sources.iter().filter(|_| expand))
        {
            if let Some(previous) = available.insert(source.record_id.clone(), source) {
                if previous != source {
                    return Err(invalid("conflicting conversation expansion source"));
                }
            }
        }
        if available.len() > MAX_CONVERSATION_RECORDS {
            return Err(invalid(
                "conversation expansion source allocation exceeds limit",
            ));
        }
        let mut ordered: Vec<_> = available.values().copied().collect();
        ordered.sort_by(|a, b| {
            (&a.session_id, &a.version, a.turn_index).cmp(&(
                &b.session_id,
                &b.version,
                b.turn_index,
            ))
        });
        let mut rounds = BTreeMap::new();
        let mut manifest = None;
        let mut source_bytes = 0usize;
        for source in ordered {
            source_bytes = source_bytes
                .checked_add(source.text.len())
                .filter(|size| *size <= MAX_CONVERSATION_BYTES)
                .ok_or_else(|| invalid("conversation expansion source allocation exceeds limit"))?;
            if manifest.as_ref().is_none_or(|m: &super::Manifest| {
                m.session_id != source.session_id || m.version != source.version
            }) {
                manifest = Some(self.manifest(&source.session_id, &source.version)?);
            }
            let manifest = manifest.as_ref().expect("source manifest loaded");
            if self.hydrate(&source.record_id, manifest)?.as_ref() != Some(source) {
                return Err(invalid("conversation expansion source missing or changed"));
            }
            rounds.insert(
                source.record_id.clone(),
                (
                    source.session_id.clone(),
                    source.version.clone(),
                    round_index(manifest, source.turn_index),
                ),
            );
        }
        let mut selected: BTreeMap<_, _> = seed
            .sources
            .iter()
            .map(|source| (source.record_id.clone(), source.clone()))
            .collect();
        let mut selected_rounds: HashSet<_> =
            selected.keys().map(|id| rounds[id].clone()).collect();
        if selected_rounds.len() > limits.max_rounds {
            return Err(invalid("conversation seed exceeds raw round budget"));
        }
        if !expand {
            return Ok(current);
        }
        let mut anchors: Vec<_> = candidates.anchors.iter().enumerate().collect();
        for (_, anchor) in &anchors {
            let Some(source) = available.get(&anchor.record_id) else {
                return Err(invalid("conversation expansion anchor source is missing"));
            };
            if !anchor.score.is_finite()
                || anchor.score < 0.0
                || anchor.source_id != source.source_id
                || anchor.session_id != source.session_id
                || anchor.version != source.version
                || anchor.turn_index != source.turn_index
            {
                return Err(invalid("invalid conversation expansion anchor"));
            }
        }
        anchors.sort_by(|(a, x), (b, y)| y.score.total_cmp(&x.score).then(a.cmp(b)));
        for (_, anchor) in &anchors {
            if anchor.score == 0.0 || selected.contains_key(&anchor.record_id) {
                continue;
            }
            let source = available[&anchor.record_id];
            let mut trial = selected.clone();
            let mut trial_rounds = selected_rounds.clone();
            for id in std::iter::once(&source.record_id).chain(&source.premise_record_ids) {
                let Some(source) = available.get(id) else {
                    return Err(invalid("conversation expansion anchor premise is missing"));
                };
                trial.insert(id.clone(), (**source).clone());
                trial_rounds.insert(rounds[id].clone());
            }
            if trial.len() > limits.max_sources || trial_rounds.len() > limits.max_rounds {
                continue;
            }
            let context = ConversationContext::from_sources_inner(
                &trial.values().cloned().collect::<Vec<_>>(),
            )?;
            if serde_json::to_vec(&context)?.len() > limits.max_bytes {
                continue;
            }
            current = context;
            selected = trial;
            selected_rounds = trial_rounds;
        }
        let mut positions = BTreeMap::new();
        for (_, anchor) in anchors {
            if anchor.score == 0.0 || !selected.contains_key(&anchor.record_id) {
                continue;
            }
            let source = available[&anchor.record_id];
            if manifest
                .as_ref()
                .is_none_or(|m| m.session_id != source.session_id || m.version != source.version)
            {
                manifest = Some(self.manifest(&source.session_id, &source.version)?);
                positions.clear();
            }
            let manifest = manifest.as_ref().expect("source manifest loaded");
            if positions.is_empty() {
                positions.extend(
                    manifest
                        .sources
                        .iter()
                        .enumerate()
                        .map(|(index, entry)| (entry.record_id.clone(), index)),
                );
            }
            let additions: Vec<_> = premise_closure(manifest, &positions, source.turn_index)?
                .into_iter()
                .filter(|index| !selected.contains_key(&manifest.sources[*index].record_id))
                .collect();
            if additions.is_empty() || selected.len() + additions.len() > limits.max_sources {
                continue;
            }
            let mut trial_rounds = selected_rounds.clone();
            for index in &additions {
                trial_rounds.insert((
                    source.session_id.clone(),
                    source.version.clone(),
                    round_index(manifest, *index),
                ));
            }
            if trial_rounds.len() > limits.max_rounds {
                continue;
            }
            let mut trial = selected.clone();
            let mut text_bytes: usize = trial.values().map(|source| source.text.len()).sum();
            for index in additions {
                let id = &manifest.sources[index].record_id;
                let premise = match available.get(id) {
                    Some(source) => (**source).clone(),
                    None => self.hydrate(id, manifest)?.ok_or_else(|| {
                        invalid("conversation expansion anchor premise is missing")
                    })?,
                };
                text_bytes = text_bytes.checked_add(premise.text.len()).ok_or_else(|| {
                    invalid("conversation expansion source allocation exceeds limit")
                })?;
                if text_bytes > limits.max_bytes {
                    break;
                }
                trial.insert(id.clone(), premise);
            }
            if text_bytes > limits.max_bytes {
                continue;
            }
            let context = ConversationContext::from_sources_inner(
                &trial.values().cloned().collect::<Vec<_>>(),
            )?;
            if serde_json::to_vec(&context)?.len() > limits.max_bytes {
                continue;
            }
            current = context;
            selected = trial;
            selected_rounds = trial_rounds;
        }
        Ok(current)
    }
}

impl ConversationEvidence {
    /// Pack exactly the returned sources, preserving unresolved canonical premise links.
    /// Anchors, scores and drops remain in the evidence receipt, outside reader context.
    pub fn pack_context(&self, max_bytes: usize) -> Result<ConversationContext> {
        ConversationContext::from_sources(&self.sources, max_bytes)
    }
}

impl ConversationContext {
    pub fn from_sources(sources: &[ConversationSource], max_bytes: usize) -> Result<Self> {
        if !(2..=MAX_CONVERSATION_BYTES).contains(&max_bytes) {
            return Err(invalid("invalid conversation context bounds"));
        }
        let context = Self::from_sources_inner(sources)?;
        if serde_json::to_vec(&context)?.len() > max_bytes {
            return Err(invalid("conversation context exceeds byte budget"));
        }
        Ok(context)
    }

    fn from_sources_inner(sources: &[ConversationSource]) -> Result<Self> {
        if sources.len() > MAX_CONVERSATION_RECORDS {
            return Err(invalid("invalid conversation context bounds"));
        }
        let mut bytes = 0usize;
        for source in sources {
            bytes = bytes
                .checked_add(source.text.len())
                .filter(|size| *size <= MAX_CONVERSATION_BYTES)
                .ok_or_else(|| invalid("conversation context source allocation exceeds limit"))?;
            if !label(&source.session_id)
                || !label(&source.version)
                || !label(&source.source_id)
                || source.turn_index >= MAX_CONVERSATION_RECORDS
                || source.text.len() > MAX_CONVERSATION_METADATA_BYTES
                || source.premise_record_ids.len() > 4
                || source.premise_record_ids.iter().any(|id| !premise_id(id))
            {
                return Err(invalid("invalid conversation context source"));
            }
        }
        let mut ordered: Vec<_> = sources.iter().collect();
        ordered.sort_by(|a, b| {
            (&a.session_id, &a.version, a.turn_index).cmp(&(
                &b.session_id,
                &b.version,
                b.turn_index,
            ))
        });
        let sessions: Vec<_> = ordered
            .iter()
            .map(|source| (source.session_id.clone(), source.version.clone()))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let session_index: BTreeMap<_, _> = sessions
            .iter()
            .enumerate()
            .map(|(index, identity)| (identity.clone(), index))
            .collect();
        let source_index: BTreeMap<_, _> = ordered
            .iter()
            .enumerate()
            .map(|(index, source)| (source.record_id.as_str(), index))
            .collect();
        if source_index.len() != ordered.len() {
            return Err(invalid("duplicate conversation context source"));
        }
        let mut turns = Vec::with_capacity(ordered.len());
        for source in &ordered {
            if source.record_id
                != source_id(
                    &source.session_id,
                    &source.version,
                    source.turn_index,
                    &source.source_id,
                )?
            {
                return Err(invalid("conversation context source identity differs"));
            }
            let premises = source
                .premise_record_ids
                .iter()
                .map(|id| match source_index.get(id.as_str()) {
                    Some(index) => ContextPremise::Local(*index),
                    None => ContextPremise::Unresolved(id.clone()),
                })
                .collect();
            turns.push(ContextTurn(
                session_index[&(source.session_id.clone(), source.version.clone())],
                source.turn_index,
                source.source_id.clone(),
                source.role,
                source.observation_time,
                source.text.clone(),
                premises,
            ));
        }
        let context = Self {
            schema_version: 1,
            sufficiency: ContextSufficiency::Unverified,
            sources_sha256: sha256(&serde_json::to_vec(&ordered)?),
            sessions,
            turns,
        };
        context.sources()?;
        Ok(context)
    }

    /// Decode a bounded frame and check identities, closure, order and source digest.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_CONVERSATION_BYTES {
            return Err(invalid("conversation context exceeds frame limit"));
        }
        let context: Self = serde_json::from_slice(bytes)
            .map_err(|_| invalid("invalid conversation context frame"))?;
        context.sources()?;
        Ok(context)
    }

    /// Canonical links whose sources are absent from this context. No sources are invented.
    pub fn unresolved_premise_ids(&self) -> Result<Vec<String>> {
        let sources = self.sources()?;
        let available: HashSet<_> = sources.iter().map(|source| &source.record_id).collect();
        Ok(sources
            .iter()
            .flat_map(|source| &source.premise_record_ids)
            .filter(|id| !available.contains(id))
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect())
    }

    /// Reconstruct immutable source records in canonical session/version/turn order.
    pub fn sources(&self) -> Result<Vec<ConversationSource>> {
        if self.schema_version != 1
            || self.sources_sha256.len() != 64
            || !self
                .sources_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || self.turns.len() > MAX_CONVERSATION_RECORDS
            || self.sessions.len() > self.turns.len()
            || self
                .sessions
                .iter()
                .any(|(session, version)| !label(session) || !label(version))
            || !self.sessions.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(invalid("invalid conversation context metadata"));
        }
        let mut sources: Vec<ConversationSource> = Vec::with_capacity(self.turns.len());
        let mut identities = HashSet::new();
        let mut used_sessions = HashSet::new();
        let mut bytes = 0usize;
        let mut previous = None;
        for turn in &self.turns {
            let ContextTurn(session, index, id, role, time, text, premises) = turn;
            let Some((session_id, version)) = self.sessions.get(*session) else {
                return Err(invalid("conversation context session is missing"));
            };
            bytes = bytes
                .checked_add(text.len())
                .filter(|size| *size <= MAX_CONVERSATION_BYTES)
                .ok_or_else(|| invalid("conversation context source allocation exceeds limit"))?;
            if *index >= MAX_CONVERSATION_RECORDS
                || !label(id)
                || text.len() > MAX_CONVERSATION_METADATA_BYTES
                || premises.len() > 4
                || !identities.insert((*session, id))
                || previous.is_some_and(|position| position >= (*session, *index))
            {
                return Err(invalid("invalid conversation context source"));
            }
            let mut premise_record_ids = Vec::with_capacity(premises.len());
            let mut last_premise = None;
            for premise in premises {
                match premise {
                    ContextPremise::Unresolved(id) => {
                        if !premise_id(id) {
                            return Err(invalid("invalid unresolved conversation premise"));
                        }
                        premise_record_ids.push(id.clone());
                    }
                    ContextPremise::Local(premise) => {
                        let Some(source) = sources.get(*premise) else {
                            return Err(invalid(
                                "conversation context premise is missing or forward",
                            ));
                        };
                        if source.session_id != *session_id
                            || source.version != *version
                            || source.turn_index >= *index
                            || last_premise.is_some_and(|position| position >= *premise)
                        {
                            return Err(invalid(
                                "conversation context premise scope or order differs",
                            ));
                        }
                        last_premise = Some(*premise);
                        premise_record_ids.push(source.record_id.clone());
                    }
                }
            }
            if premise_record_ids.iter().collect::<HashSet<_>>().len() != premise_record_ids.len() {
                return Err(invalid("duplicate conversation context premise"));
            }
            sources.push(ConversationSource {
                record_id: source_id(session_id, version, *index, id)?,
                source_id: id.clone(),
                session_id: session_id.clone(),
                version: version.clone(),
                turn_index: *index,
                role: *role,
                observation_time: *time,
                text: text.clone(),
                premise_record_ids,
            });
            previous = Some((*session, *index));
            used_sessions.insert(*session);
        }
        let available: HashSet<_> = sources.iter().map(|source| &source.record_id).collect();
        if self.turns.iter().any(|turn| {
            turn.6.iter().any(|premise| {
                matches!(premise, ContextPremise::Unresolved(id) if available.contains(id))
            })
        }) {
            return Err(invalid("conversation premise must use its local ordinal"));
        }
        if used_sessions.len() != self.sessions.len()
            || sha256(&serde_json::to_vec(&sources)?) != self.sources_sha256
        {
            return Err(invalid("conversation context source digest differs"));
        }
        Ok(sources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_capacity_accepts_both_sides_of_the_last_valid_ordinal() {
        for count in [
            MAX_CONVERSATION_RECORDS - 1,
            MAX_CONVERSATION_RECORDS,
            MAX_CONVERSATION_RECORDS + 1,
        ] {
            let sources: Vec<_> = (0..count)
                .map(|index| {
                    let session_id = format!("session-{}", index / 2);
                    let source = format!("source-{index}");
                    ConversationSource {
                        record_id: source_id(&session_id, "v1", index % 2, &source).unwrap(),
                        source_id: source,
                        session_id,
                        version: "v1".into(),
                        turn_index: index % 2,
                        role: ConversationRole::User,
                        observation_time: 0,
                        text: "literal observation".into(),
                        premise_record_ids: Vec::new(),
                    }
                })
                .collect();
            let context = ConversationContext::from_sources(&sources, MAX_CONVERSATION_BYTES);
            assert_eq!(context.is_ok(), count <= MAX_CONVERSATION_RECORDS);
            if let Ok(context) = context {
                assert_eq!(context.sources().unwrap().len(), count);
            }
        }
    }
}
