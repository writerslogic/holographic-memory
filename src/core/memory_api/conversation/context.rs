// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{
    invalid, label, sha256, source_id, ConversationEvidence, ConversationRole, ConversationSource,
    Millis, MAX_CONVERSATION_BYTES, MAX_CONVERSATION_METADATA_BYTES, MAX_CONVERSATION_RECORDS,
};

/// Lossless source context with shared session identities and ordinal premise references.
///
/// Each turn is `[session, index, source ID, role, observation time, text, premises]`.
/// This is an optional context encoding, not a retrieval unit or a sufficiency claim.
/// The digest checks the reconstructed sources; it does not authenticate their author.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConversationContext {
    schema_version: u32,
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
    Vec<usize>,
);

impl ConversationEvidence {
    /// Pack exactly the returned sources, including their complete premise closures.
    /// Anchors, scores and drops remain in the evidence receipt, outside reader context.
    pub fn pack_context(&self, max_bytes: usize) -> Result<ConversationContext> {
        ConversationContext::from_sources(&self.sources, max_bytes)
    }
}

impl ConversationContext {
    pub fn from_sources(sources: &[ConversationSource], max_bytes: usize) -> Result<Self> {
        if !(2..=MAX_CONVERSATION_BYTES).contains(&max_bytes)
            || sources.len() > MAX_CONVERSATION_RECORDS
        {
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
                || source
                    .premise_record_ids
                    .iter()
                    .any(|id| id.len() != "conversation:source:".len() + 64)
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
                .map(|id| {
                    source_index
                        .get(id.as_str())
                        .copied()
                        .ok_or_else(|| invalid("conversation context premise is missing"))
                })
                .collect::<Result<_>>()?;
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
            sources_sha256: sha256(&serde_json::to_vec(&ordered)?),
            sessions,
            turns,
        };
        context.sources()?;
        if serde_json::to_vec(&context)?.len() > max_bytes {
            return Err(invalid("conversation context exceeds byte budget"));
        }
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
