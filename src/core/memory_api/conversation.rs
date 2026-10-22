// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use super::{HistoryMode, IdConflict, Memory, MemoryError, Millis, RecallOptions, RememberInput};
use crate::core::HmsCore;

pub const MAX_CONVERSATION_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_CONVERSATION_METADATA_BYTES: usize = 60 * 1024;
pub const MAX_CONVERSATION_RECORDS: usize = 4096;
const CHUNK_BYTES: usize = 1024;
const CHUNK_WORDS: usize = 256;
const SCHEMA: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConversationRole {
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConversationTurn {
    pub source_id: String,
    pub role: ConversationRole,
    pub text: String,
    pub observation_time: Millis,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConversationInput {
    pub session_id: String,
    pub version: String,
    pub turns: Vec<ConversationTurn>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationRemembered {
    pub session_id: String,
    pub version: String,
    pub source_count: usize,
    pub chunk_count: usize,
    pub created: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct ConversationRecallOptions {
    /// Roles eligible as retrieval anchors; required user premises may also be included.
    pub roles: Vec<ConversationRole>,
    /// Distinct source candidates per role after deduplicating matching chunks.
    pub k: u32,
    /// Native chunks returned per role, from `k` through 1000. Installed model stages may
    /// use a larger internal candidate pool.
    pub candidate_limit: u32,
    pub max_sources: usize,
    /// UTF8 bytes of the JSON sources array, including brackets and separators; not tokens.
    pub max_bytes: usize,
    /// Distinct rounds, including required premises. None preserves the unbounded profile.
    pub max_rounds: Option<usize>,
    /// Assistant RRF weight for mixed-role queries that do not ask about an assistant reply.
    pub assistant_weight: f64,
    /// Session redundancy penalty in [0, 1], applied only to counting/aggregation queries.
    pub session_diversity: f64,
}

impl Default for ConversationRecallOptions {
    fn default() -> Self {
        Self {
            roles: vec![ConversationRole::User, ConversationRole::Assistant],
            k: 32,
            candidate_limit: 128,
            max_sources: 64,
            max_bytes: 64 * 1024,
            max_rounds: None,
            assistant_weight: 1.0,
            session_diversity: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConversationSource {
    pub record_id: String,
    pub source_id: String,
    pub session_id: String,
    pub version: String,
    pub turn_index: usize,
    pub role: ConversationRole,
    pub observation_time: Millis,
    pub text: String,
    pub premise_record_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationAnchor {
    pub record_id: String,
    pub source_id: String,
    pub session_id: String,
    pub version: String,
    pub turn_index: usize,
    /// Weighted reciprocal rank fusion, with constant 60, over the requested roles.
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationDrop {
    pub record_id: String,
    pub source_id: String,
    pub session_id: String,
    pub version: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationEvidence {
    pub sources: Vec<ConversationSource>,
    pub anchors: Vec<ConversationAnchor>,
    pub dropped: Vec<ConversationDrop>,
    pub serialized_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestEntry {
    record_id: String,
    source_id: String,
    turn_index: usize,
    role: ConversationRole,
    observation_time: Millis,
    text_sha256: String,
    premise_record_ids: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    kind: String,
    conversation_schema: u32,
    session_id: String,
    version: String,
    input_sha256: String,
    chunk_bytes: usize,
    chunk_words: usize,
    sources: Vec<ManifestEntry>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredSource {
    kind: String,
    conversation_schema: u32,
    source: ConversationSource,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Chunk {
    kind: String,
    conversation_schema: u32,
    session_id: String,
    version: String,
    source_id: String,
    source_record_id: String,
    turn_index: usize,
    role: ConversationRole,
    observation_time: Millis,
    chunk_index: usize,
    start_byte: usize,
    end_byte: usize,
}

struct Candidate {
    chunk: Chunk,
    text_sha256: String,
    score: f64,
    first_seen: usize,
}

fn query_words(query: &str) -> Vec<String> {
    query
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn aggregation_intent(words: &[String]) -> bool {
    words.iter().any(|word| word == "total")
        || words.windows(2).any(|pair| {
            matches!(
                (pair[0].as_str(), pair[1].as_str()),
                ("how", "many") | ("list", "all" | "every") | ("count" | "number", "of")
            )
        })
}

fn assistant_intent(words: &[String]) -> bool {
    words
        .iter()
        .any(|word| matches!(word.as_str(), "you" | "your" | "assistant"))
        && words.iter().any(|word| {
            matches!(word.as_str(), "say" | "said" | "give" | "gave")
                || ["suggest", "recommend", "mention", "provid"]
                    .iter()
                    .any(|prefix| word.starts_with(prefix))
        })
}

fn round_index(manifest: &Manifest, turn: usize) -> usize {
    manifest.sources[..=turn]
        .iter()
        .rposition(|entry| entry.role == ConversationRole::User)
        .unwrap_or(0)
}

/// Owns the core so one memory writer controls this immutable source namespace. Share the
/// wrapper through `Arc`; configure local stages before moving the core into it.
pub struct ConversationMemory {
    memory: Memory,
}

fn invalid(message: &str) -> anyhow::Error {
    MemoryError::Invalid(message.to_owned()).into()
}

fn label(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn record_id(kind: &str, parts: &impl Serialize) -> Result<String> {
    Ok(format!(
        "conversation:{kind}:{}",
        sha256(&serde_json::to_vec(&(SCHEMA, kind, parts))?)
    ))
}

fn source_id(session: &str, version: &str, index: usize, source: &str) -> Result<String> {
    record_id("source", &(session, version, index, source))
}

fn metadata(value: &impl Serialize) -> Result<Value> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_CONVERSATION_METADATA_BYTES {
        return Err(invalid("conversation metadata exceeds 60 KiB"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn fact(id: String, text: String, metadata: Value) -> RememberInput {
    RememberInput {
        id: Some(id),
        text,
        metadata: Some(metadata),
        ..RememberInput::default()
    }
}

fn premises(entries: &[ManifestEntry], index: usize) -> Vec<String> {
    let mut selected: Vec<usize> = entries[..index]
        .iter()
        .filter(|entry| entry.role == ConversationRole::User)
        .take(2)
        .map(|entry| entry.turn_index)
        .collect();
    let user = if entries[index].role == ConversationRole::User {
        Some(index)
    } else {
        entries[..index]
            .iter()
            .rev()
            .find(|entry| entry.role == ConversationRole::User)
            .map(|entry| entry.turn_index)
    };
    if let Some(user) = user {
        if user != index {
            selected.push(user);
        }
        if user > 0 && entries[user - 1].role == ConversationRole::Assistant {
            selected.push(user - 1);
        }
    }
    selected.sort_unstable();
    selected.dedup();
    selected
        .into_iter()
        .map(|index| entries[index].record_id.clone())
        .collect()
}

fn spans(text: &str) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let (mut start, mut words, mut in_word) = (0, 0, false);
    for (offset, ch) in text.char_indices() {
        let begins_word = !ch.is_whitespace() && !in_word;
        if offset + ch.len_utf8() - start > CHUNK_BYTES || begins_word && words == CHUNK_WORDS {
            result.push((start, offset));
            start = offset;
            words = 0;
            in_word = false;
        }
        if !ch.is_whitespace() && !in_word {
            words += 1;
        }
        in_word = !ch.is_whitespace();
    }
    if start < text.len() {
        result.push((start, text.len()));
    }
    result
}

impl ConversationMemory {
    pub fn new(core: HmsCore) -> Result<Self> {
        #[cfg(feature = "local-models")]
        if let Some(stages) = core.model_stages() {
            if stages.extract_facts {
                return Err(invalid(
                    "disable user-fact extraction before storing conversation sources",
                ));
            }
            if stages.embedder.as_ref().is_some_and(|embedder| {
                embedder.query_task() != crate::core::models::EmbeddingQueryTask::Conversation
            }) {
                return Err(invalid("configure the conversation embedding query task before storing conversation sources"));
            }
        }
        let memory = Memory::new(Arc::new(core));
        if memory.config().on_id_conflict != IdConflict::Reject {
            return Err(invalid(
                "conversation memory requires rejecting ID conflicts",
            ));
        }
        Ok(Self { memory })
    }

    pub fn remember(&self, input: ConversationInput) -> Result<ConversationRemembered> {
        if !label(&input.session_id)
            || !label(&input.version)
            || input.turns.is_empty()
            || input.turns.len() >= MAX_CONVERSATION_RECORDS
        {
            return Err(invalid("invalid conversation identity or turn count"));
        }
        let mut total = 0usize;
        let mut unique = HashSet::new();
        let mut entries = Vec::with_capacity(input.turns.len());
        for (index, turn) in input.turns.iter().enumerate() {
            if !label(&turn.source_id) || !unique.insert(&turn.source_id) {
                return Err(invalid("invalid or duplicate conversation source ID"));
            }
            total = total
                .checked_add(turn.text.len())
                .filter(|size| *size <= MAX_CONVERSATION_BYTES)
                .ok_or_else(|| invalid("conversation source bytes exceed 2 MiB"))?;
            if turn.text.len() > MAX_CONVERSATION_METADATA_BYTES {
                return Err(invalid("conversation source metadata exceeds 60 KiB"));
            }
            entries.push(ManifestEntry {
                record_id: source_id(&input.session_id, &input.version, index, &turn.source_id)?,
                source_id: turn.source_id.clone(),
                turn_index: index,
                role: turn.role,
                observation_time: turn.observation_time,
                text_sha256: sha256(turn.text.as_bytes()),
                premise_record_ids: Vec::new(),
            });
        }
        for index in 0..entries.len() {
            entries[index].premise_record_ids = premises(&entries, index);
        }
        let manifest = Manifest {
            kind: "conversation-manifest".into(),
            conversation_schema: SCHEMA,
            session_id: input.session_id.clone(),
            version: input.version.clone(),
            input_sha256: sha256(&serde_json::to_vec(&input)?),
            chunk_bytes: CHUNK_BYTES,
            chunk_words: CHUNK_WORDS,
            sources: entries.clone(),
        };
        let mut records = vec![fact(
            record_id("manifest", &(&input.session_id, &input.version))?,
            "Conversation manifest".into(),
            metadata(&manifest)?,
        )];
        let mut chunk_count = 0;
        for (turn, entry) in input.turns.iter().zip(&entries) {
            let source = ConversationSource {
                record_id: entry.record_id.clone(),
                source_id: entry.source_id.clone(),
                session_id: input.session_id.clone(),
                version: input.version.clone(),
                turn_index: entry.turn_index,
                role: entry.role,
                observation_time: entry.observation_time,
                text: turn.text.clone(),
                premise_record_ids: entry.premise_record_ids.clone(),
            };
            records.push(fact(
                entry.record_id.clone(),
                "Conversation source".into(),
                metadata(&StoredSource {
                    kind: "conversation-source".into(),
                    conversation_schema: SCHEMA,
                    source,
                })?,
            ));
            for (chunk_index, (start_byte, end_byte)) in spans(&turn.text).into_iter().enumerate() {
                let text = &turn.text[start_byte..end_byte];
                if text.trim().is_empty() {
                    continue;
                }
                let chunk = Chunk {
                    kind: "conversation-chunk".into(),
                    conversation_schema: SCHEMA,
                    session_id: input.session_id.clone(),
                    version: input.version.clone(),
                    source_id: entry.source_id.clone(),
                    source_record_id: entry.record_id.clone(),
                    turn_index: entry.turn_index,
                    role: entry.role,
                    observation_time: entry.observation_time,
                    chunk_index,
                    start_byte,
                    end_byte,
                };
                records.push(fact(
                    record_id("chunk", &(&entry.record_id, chunk_index))?,
                    text.to_owned(),
                    metadata(&chunk)?,
                ));
                chunk_count += 1;
                if records.len() > MAX_CONVERSATION_RECORDS {
                    return Err(invalid("conversation exceeds 4096 stored records"));
                }
            }
        }
        if records.len() > MAX_CONVERSATION_RECORDS {
            return Err(invalid("conversation exceeds 4096 stored records"));
        }
        let remembered = self.memory.remember_many(records)?;
        Ok(ConversationRemembered {
            session_id: input.session_id,
            version: input.version,
            source_count: input.turns.len(),
            chunk_count,
            created: remembered[0].created,
        })
    }

    fn manifest(&self, session: &str, version: &str) -> Result<Manifest> {
        let record = self
            .memory
            .get(&record_id("manifest", &(session, version))?)?
            .ok_or_else(|| invalid("conversation manifest is missing"))?;
        let manifest: Manifest = serde_json::from_value(record.metadata)
            .map_err(|_| invalid("invalid conversation manifest"))?;
        if manifest.kind != "conversation-manifest"
            || manifest.conversation_schema != SCHEMA
            || manifest.session_id != session
            || manifest.version != version
            || manifest.chunk_bytes != CHUNK_BYTES
            || manifest.chunk_words != CHUNK_WORDS
            || manifest.sources.is_empty()
            || manifest.sources.len() >= MAX_CONVERSATION_RECORDS
        {
            return Err(invalid("conversation manifest scope differs"));
        }
        let mut unique = HashSet::new();
        for (index, entry) in manifest.sources.iter().enumerate() {
            if entry.turn_index != index
                || !label(&entry.source_id)
                || !unique.insert(&entry.source_id)
                || entry.record_id != source_id(session, version, index, &entry.source_id)?
                || entry.premise_record_ids != premises(&manifest.sources, index)
            {
                return Err(invalid(
                    "conversation manifest occurrence or premises differ",
                ));
            }
        }
        Ok(manifest)
    }

    fn hydrate(&self, id: &str, manifest: &Manifest) -> Result<Option<ConversationSource>> {
        let Some(record) = self.memory.get(id)? else {
            return Ok(None);
        };
        let stored: StoredSource = serde_json::from_value(record.metadata)
            .map_err(|_| invalid("invalid canonical conversation source"))?;
        let source = stored.source;
        let Some(entry) = manifest.sources.get(source.turn_index) else {
            return Err(invalid("canonical conversation source ordinal differs"));
        };
        if stored.kind != "conversation-source"
            || stored.conversation_schema != SCHEMA
            || source.session_id != manifest.session_id
            || source.version != manifest.version
            || source.record_id != id
            || source.record_id != entry.record_id
            || source.source_id != entry.source_id
            || source.role != entry.role
            || source.observation_time != entry.observation_time
            || source.premise_record_ids != entry.premise_record_ids
            || sha256(source.text.as_bytes()) != entry.text_sha256
        {
            return Err(invalid("canonical conversation source provenance differs"));
        }
        Ok(Some(source))
    }

    pub fn recall(
        &self,
        query: &str,
        options: &ConversationRecallOptions,
    ) -> Result<ConversationEvidence> {
        if query.trim().is_empty()
            || query.len() > 8192
            || !(1..=1000).contains(&options.k)
            || !(options.k..=1000).contains(&options.candidate_limit)
            || options.roles.is_empty()
            || options.roles.len() > 2
            || options.roles.iter().copied().collect::<HashSet<_>>().len() != options.roles.len()
            || !(1..=MAX_CONVERSATION_RECORDS).contains(&options.max_sources)
            || !(2..=MAX_CONVERSATION_BYTES).contains(&options.max_bytes)
            || options
                .max_rounds
                .is_some_and(|limit| !(1..=MAX_CONVERSATION_RECORDS).contains(&limit))
            || !options.assistant_weight.is_finite()
            || !(0.0..=4.0).contains(&options.assistant_weight)
            || !options.session_diversity.is_finite()
            || !(0.0..=1.0).contains(&options.session_diversity)
        {
            return Err(invalid("invalid conversation recall bounds"));
        }
        let mut candidates: BTreeMap<String, Candidate> = BTreeMap::new();
        let mut first_seen = 0;
        let words = query_words(query);
        let diversity = if aggregation_intent(&words) {
            options.session_diversity
        } else {
            0.0
        };
        for role in &options.roles {
            let weight = if *role == ConversationRole::Assistant
                && options.roles.len() > 1
                && !assistant_intent(&words)
            {
                options.assistant_weight
            } else {
                1.0
            };
            if weight == 0.0 {
                continue;
            }
            // Model stages can widen their pool beyond the requested returned chunk count.
            let hits = self.memory.recall(query, &RecallOptions {
                k: Some(options.candidate_limit), candidate_limit: None, history: Some(HistoryMode::All),
                filter: Some(json!({"kind": "conversation-chunk", "conversation_schema": SCHEMA, "role": role})),
                ..RecallOptions::default()
            })?;
            let mut seen = HashSet::new();
            for hit in hits {
                let chunk: Chunk = serde_json::from_value(hit.metadata)
                    .map_err(|_| invalid("invalid conversation retrieval metadata"))?;
                if chunk.kind != "conversation-chunk"
                    || chunk.conversation_schema != SCHEMA
                    || chunk.role != *role
                    || !label(&chunk.session_id)
                    || !label(&chunk.version)
                    || !label(&chunk.source_id)
                    || chunk.source_record_id
                        != source_id(
                            &chunk.session_id,
                            &chunk.version,
                            chunk.turn_index,
                            &chunk.source_id,
                        )?
                    || hit.id != record_id("chunk", &(&chunk.source_record_id, chunk.chunk_index))?
                {
                    return Err(invalid("conversation retrieval provenance differs"));
                }
                if !seen.insert(chunk.source_record_id.clone()) {
                    continue;
                }
                if seen.len() > options.k as usize {
                    break;
                }
                let score = weight / (60.0 + seen.len() as f64);
                if let Some(candidate) = candidates.get_mut(&chunk.source_record_id) {
                    candidate.score += score;
                } else {
                    let text = hit
                        .text
                        .ok_or_else(|| invalid("conversation retrieval text is missing"))?;
                    candidates.insert(
                        chunk.source_record_id.clone(),
                        Candidate {
                            chunk,
                            text_sha256: sha256(text.as_bytes()),
                            score,
                            first_seen,
                        },
                    );
                    first_seen += 1;
                }
            }
        }
        let mut candidates: Vec<_> = candidates.into_values().collect();
        candidates.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.first_seen.cmp(&b.first_seen))
        });
        let maximum_score = candidates.first().map_or(1.0, |candidate| candidate.score);
        let mut evidence = ConversationEvidence {
            sources: Vec::new(),
            anchors: Vec::new(),
            dropped: Vec::new(),
            serialized_bytes: 2,
        };
        let mut selected = HashSet::new();
        let mut selected_rounds = HashSet::new();
        let mut selected_sessions = HashSet::new();
        while !candidates.is_empty() {
            let position = if diversity > 0.0 {
                candidates
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let score = |candidate: &Candidate| {
                            candidate.score / maximum_score
                                - if selected_sessions.contains(&(
                                    candidate.chunk.session_id.clone(),
                                    candidate.chunk.version.clone(),
                                )) {
                                    diversity
                                } else {
                                    0.0
                                }
                        };
                        score(a)
                            .total_cmp(&score(b))
                            .then(b.first_seen.cmp(&a.first_seen))
                    })
                    .map_or(0, |(index, _)| index)
            } else {
                0
            };
            let candidate = candidates.remove(position);
            let chunk = &candidate.chunk;
            let manifest = self.manifest(&chunk.session_id, &chunk.version)?;
            let Some(source) = self.hydrate(&chunk.source_record_id, &manifest)? else {
                evidence.dropped.push(drop_for(chunk, "missing_source"));
                continue;
            };
            if source.source_id != chunk.source_id
                || source.turn_index != chunk.turn_index
                || source.role != chunk.role
                || source.observation_time != chunk.observation_time
                || chunk.start_byte >= chunk.end_byte
                || !source.text.is_char_boundary(chunk.start_byte)
                || !source.text.is_char_boundary(chunk.end_byte)
                || chunk.end_byte > source.text.len()
                || sha256(
                    source.text[chunk.start_byte..chunk.end_byte]
                        .trim()
                        .as_bytes(),
                ) != candidate.text_sha256
            {
                return Err(invalid(
                    "search chunk differs from original conversation source",
                ));
            }
            let mut group = Vec::new();
            let mut missing = false;
            for premise in &source.premise_record_ids {
                let Some(premise) = self.hydrate(premise, &manifest)? else {
                    missing = true;
                    break;
                };
                if premise.turn_index >= source.turn_index {
                    return Err(invalid(
                        "conversation premise is outside its source scope or order",
                    ));
                }
                group.push(premise);
            }
            if missing {
                evidence.dropped.push(drop_for(chunk, "missing_premise"));
                continue;
            }
            group.push(source);
            group.sort_by_key(|source| source.turn_index);
            group.retain(|source| !selected.contains(&source.record_id));
            let new_rounds: HashSet<_> = group
                .iter()
                .map(|source| {
                    (
                        source.session_id.clone(),
                        source.version.clone(),
                        round_index(&manifest, source.turn_index),
                    )
                })
                .filter(|round| !selected_rounds.contains(round))
                .collect();
            if options
                .max_rounds
                .is_some_and(|limit| selected_rounds.len() + new_rounds.len() > limit)
            {
                evidence.dropped.push(drop_for(chunk, "max_rounds"));
                continue;
            }
            let source_count = evidence
                .sources
                .len()
                .checked_add(group.len())
                .ok_or_else(|| invalid("conversation source allocation overflow"))?;
            if source_count > options.max_sources {
                evidence.dropped.push(drop_for(chunk, "max_sources"));
                continue;
            }
            let mut size = evidence.serialized_bytes;
            for (index, source) in group.iter().enumerate() {
                size = size
                    .checked_add(serde_json::to_vec(source)?.len())
                    .and_then(|value| {
                        value.checked_add(usize::from(!evidence.sources.is_empty() || index > 0))
                    })
                    .ok_or_else(|| invalid("conversation evidence size overflow"))?;
            }
            if size > options.max_bytes {
                evidence.dropped.push(drop_for(chunk, "max_bytes"));
                continue;
            }
            evidence.serialized_bytes = size;
            selected_rounds.extend(new_rounds);
            selected_sessions.insert((chunk.session_id.clone(), chunk.version.clone()));
            evidence.anchors.push(ConversationAnchor {
                record_id: chunk.source_record_id.clone(),
                source_id: chunk.source_id.clone(),
                session_id: chunk.session_id.clone(),
                version: chunk.version.clone(),
                turn_index: chunk.turn_index,
                score: candidate.score,
            });
            for source in group {
                selected.insert(source.record_id.clone());
                evidence.sources.push(source);
            }
        }
        evidence.sources.sort_by(|a, b| {
            (&a.session_id, &a.version, a.turn_index).cmp(&(
                &b.session_id,
                &b.version,
                b.turn_index,
            ))
        });
        Ok(evidence)
    }
}

fn drop_for(chunk: &Chunk, reason: &str) -> ConversationDrop {
    ConversationDrop {
        record_id: chunk.source_record_id.clone(),
        source_id: chunk.source_id.clone(),
        session_id: chunk.session_id.clone(),
        version: chunk.version.clone(),
        reason: reason.into(),
    }
}
