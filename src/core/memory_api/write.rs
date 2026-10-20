// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `remember` and `forget`.

use anyhow::Result;
use serde_json::{Map, Value};
use std::collections::{BTreeSet, HashMap};

use super::index::Index;
use super::stamp::{stable_hash, Placement, Stamp, KEY};
use super::types::*;
use super::{FactView, IdConflict, Memory, MemoryError};
use crate::core::documents::{chunk_document, DocumentInput};

const MAX_BATCH: usize = 4096;
const MAX_TEXT_BYTES: usize = 64 * 1024;
const CHUNK_WORDS: u32 = 4096;
const MAX_ID_BYTES: usize = 4096;
const MAX_LABEL_BYTES: usize = 1024;
const MAX_LABELS: usize = 256;
const MAX_LINKS: usize = 256;

/// One input after validation and policy resolution.
struct Planned {
    id: String,
    input: RememberInput,
    stamp: Stamp,
    /// The identical fact was already stored.
    idempotent: bool,
    /// The stored fact this one overwrites under `IdConflict::Replace`.
    replaces: bool,
    document: Option<DocumentInput>,
}

/// Stores the stamp where it fits and returns the document version to use: the compact
/// stamp, or none (the engine default) when it went to the `_mem` metadata key.
fn apply_placement(stamp: &Stamp, metadata: &mut Value) -> Option<String> {
    if let Some(map) = metadata.as_object_mut() {
        map.remove(KEY);
    }
    match stamp.placement() {
        Placement::Version(version) => Some(version),
        Placement::Metadata(value) => {
            metadata[KEY] = value;
            None
        }
    }
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_ID_BYTES
        && !id.chars().any(char::is_control)
        && !id.starts_with("hms:")
}

fn label(value: &str, what: &str) -> Result<String> {
    let value = value.trim();
    check!(
        !value.is_empty() && value.len() <= MAX_LABEL_BYTES && !value.contains('\0'),
        "{what} must contain 1..={MAX_LABEL_BYTES} bytes"
    );
    Ok(value.to_owned())
}

fn push_unique(list: &mut Vec<String>, value: String) {
    if !list.contains(&value) {
        list.push(value);
    }
}

fn time_part(value: Option<Millis>) -> Vec<u8> {
    value.map_or_else(|| b"-".to_vec(), |v| v.to_string().into_bytes())
}

impl Memory {
    /// Remembers one fact. See [`Memory::remember_many`].
    pub fn remember(&self, input: RememberInput) -> Result<Remembered> {
        let mut outcomes = self.remember_many(vec![input])?;
        Ok(outcomes.remove(0))
    }

    /// Remembers facts in one atomic transaction: every fact is stored or none is. Later
    /// inputs see earlier ones for supersession. Retrying a call is safe: a fact whose id and
    /// content already match is reported with `created: false` and stored once. Without an
    /// explicit id the id is a hash of the content, so a retried call is still idempotent.
    pub fn remember_many(&self, inputs: Vec<RememberInput>) -> Result<Vec<Remembered>> {
        check!(
            !inputs.is_empty() && inputs.len() <= MAX_BATCH,
            "remember requires 1..={MAX_BATCH} facts"
        );
        let _writer = self.writer.lock();
        self.ensure_fresh();
        let now = self.clock.now_millis();
        let planned = {
            let index = self.index.read();
            let mut planned: Vec<Planned> = Vec::with_capacity(inputs.len());
            let mut seen: HashMap<String, usize> = HashMap::new();
            for input in inputs {
                let fact = self.plan_one(&index, &planned, input, now)?;
                check!(
                    seen.insert(fact.id.clone(), planned.len()).is_none(),
                    "fact id {:?} appears twice in one call",
                    fact.id
                );
                planned.push(fact);
            }
            planned
        };
        let documents: Vec<DocumentInput> = planned
            .iter()
            .filter_map(|fact| fact.document.clone())
            .collect();
        let before = self.core.revision();
        let mut stale = false;
        if !documents.is_empty() {
            self.core.memorize_documents(documents)?;
            stale = planned.iter().any(|fact| fact.replaces);
        }
        let written: Vec<&Planned> = planned
            .iter()
            .filter(|fact| fact.document.is_some())
            .collect();
        if !written.is_empty() {
            if stale {
                self.index.write().revision = None;
            } else {
                self.settle(before, 1, |index| {
                    let batch: HashMap<&str, (Millis, Millis)> = written
                        .iter()
                        .map(|f| (f.id.as_str(), (f.stamp.effective_from(), f.stamp.ingest)))
                        .collect();
                    for fact in &written {
                        let lookup = |id: &str| {
                            batch.get(id).copied().or_else(|| {
                                fact.stamp
                                    .subject
                                    .as_deref()
                                    .and_then(|subject| index.member(subject, id))
                            })
                        };
                        // `lookup` borrows the index immutably, so resolve before inserting.
                        let hints: Vec<(String, Millis, Millis)> = fact
                            .stamp
                            .superseded_by
                            .iter()
                            .filter_map(|id| lookup(id).map(|(e, i)| (id.clone(), e, i)))
                            .collect();
                        let known: HashMap<String, (Millis, Millis)> =
                            hints.into_iter().map(|(id, e, i)| (id, (e, i))).collect();
                        index.insert(&fact.id, &fact.stamp, &|id| known.get(id).copied());
                    }
                });
            }
        }
        let outcomes: Vec<Remembered> = planned
            .iter()
            .map(|fact| Remembered {
                id: fact.id.clone(),
                created: !fact.idempotent,
                supersedes: fact.stamp.supersedes.clone(),
                superseded_by: fact.stamp.superseded_by.clone(),
                ingest_time: fact.stamp.ingest,
            })
            .collect();
        if let Some(observer) = &self.observer {
            for (fact, outcome) in planned.iter().zip(&outcomes) {
                if outcome.created {
                    observer.on_remembered(outcome, &fact.input);
                }
            }
        }
        Ok(outcomes)
    }

    fn plan_one(
        &self,
        index: &Index,
        earlier: &[Planned],
        mut input: RememberInput,
        now: Millis,
    ) -> Result<Planned> {
        check!(
            !input.text.trim().is_empty() && input.text.len() <= MAX_TEXT_BYTES,
            "text must contain 1..={MAX_TEXT_BYTES} bytes"
        );
        if let (Some(from), Some(to)) = (input.valid_from, input.valid_to) {
            check!(from < to, "validFrom must be before validTo");
        }
        let user_metadata = match input.metadata.take() {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(map)) => map,
            Some(_) => invalid!("metadata must be a JSON object"),
        };
        check!(
            !user_metadata.contains_key(KEY),
            "metadata key {KEY:?} is reserved"
        );
        let subject = input
            .subject
            .as_deref()
            .map(|s| label(s, "subject"))
            .transpose()?;
        let mut entities = Vec::new();
        for entity in input.entities.iter().flatten() {
            push_unique(&mut entities, label(entity, "entity")?);
        }
        if let Some(extractor) = &self.entities {
            for entity in extractor.entities(&input.text) {
                if let Ok(entity) = label(&entity, "entity") {
                    push_unique(&mut entities, entity);
                }
            }
        }
        check!(
            entities.len() <= MAX_LABELS,
            "at most {MAX_LABELS} entities"
        );
        let mut explicit = Vec::new();
        for id in input.supersedes.iter().flatten() {
            check!(valid_id(id), "supersedes holds an invalid fact id");
            push_unique(&mut explicit, id.clone());
        }
        explicit.sort();
        check!(
            explicit.len() <= MAX_LINKS,
            "at most {MAX_LINKS} supersedes"
        );
        let metadata_bytes = serde_json::to_vec(&Value::Object(user_metadata.clone()))?;
        let embedding_bytes: Vec<u8> = input
            .embedding
            .iter()
            .flatten()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let hash = stable_hash(&[
            input.text.as_bytes(),
            &metadata_bytes,
            &time_part(input.event_time),
            &time_part(input.valid_from),
            &time_part(input.valid_to),
            subject.as_deref().unwrap_or("").as_bytes(),
            explicit.join("\u{1}").as_bytes(),
            entities.join("\u{1}").as_bytes(),
            input.source_uri.as_deref().unwrap_or("").as_bytes(),
            &embedding_bytes,
        ]);
        let id = match input.id.as_deref() {
            Some(id) => {
                check!(
                    valid_id(id),
                    "id must be 1..={MAX_ID_BYTES} bytes, without control characters or the \"hms:\" prefix"
                );
                id.to_owned()
            }
            None => format!("mem:{hash}"),
        };
        check!(!explicit.contains(&id), "a fact cannot supersede itself");

        let mut replaces = false;
        let mut idempotent_stamp = None;
        if let Some((version, metadata, _)) = self.core.document_snapshot(&id) {
            match Stamp::from_parts(&version, &metadata) {
                Some(stamp) if stamp.hash.as_deref() == Some(hash.as_str()) => {
                    idempotent_stamp = Some(stamp);
                }
                existing => {
                    if self.config.on_id_conflict != IdConflict::Replace {
                        return Err(MemoryError::Conflict(format!(
                            "fact {id:?} already exists with different content; use another id \
                             or set onIdConflict to \"replace\""
                        ))
                        .into());
                    }
                    check!(
                        existing.is_some() || index.edges(&id).is_empty(),
                        "document {id:?} is not a memory fact and is superseded by others"
                    );
                    replaces = true;
                }
            }
        }
        if let Some(stamp) = idempotent_stamp {
            return Ok(Planned {
                id,
                input,
                stamp,
                idempotent: true,
                replaces: false,
                document: None,
            });
        }

        for old in &explicit {
            check!(
                self.core.has_document(old) || earlier.iter().any(|fact| &fact.id == old),
                "cannot supersede unknown fact {old:?}"
            );
        }
        let mut supersedes = explicit;
        let mut superseded_by: Vec<String> = Vec::new();
        if self.supersession_active {
            if let Some(subject) = &subject {
                let mut group: Vec<FactView> = index
                    .members(subject)
                    .into_iter()
                    .filter(|member| member.id != id)
                    .map(|member| FactView {
                        id: member.id,
                        subject: Some(subject.clone()),
                        effective_from: member.effective,
                        ingest_time: member.ingest,
                    })
                    .collect();
                for fact in earlier
                    .iter()
                    .filter(|f| f.stamp.subject.as_ref() == Some(subject))
                {
                    group.retain(|member| member.id != fact.id);
                    group.push(FactView {
                        id: fact.id.clone(),
                        subject: Some(subject.clone()),
                        effective_from: fact.stamp.effective_from(),
                        ingest_time: fact.stamp.ingest,
                    });
                }
                group.sort_by(|a, b| {
                    (a.effective_from, a.ingest_time, &a.id).cmp(&(
                        b.effective_from,
                        b.ingest_time,
                        &b.id,
                    ))
                });
                let incoming = FactView {
                    id: id.clone(),
                    subject: Some(subject.clone()),
                    effective_from: input.valid_from.or(input.event_time).unwrap_or(now),
                    ingest_time: now,
                };
                let resolution = self.supersession.resolve(&incoming, &group);
                for linked in resolution
                    .supersedes
                    .iter()
                    .chain(&resolution.superseded_by)
                {
                    check!(
                        group.iter().any(|member| &member.id == linked),
                        "supersession policy returned {linked:?}, which is not a stored fact of subject {subject:?}"
                    );
                }
                for old in resolution.supersedes {
                    push_unique(&mut supersedes, old);
                }
                for newer in resolution.superseded_by {
                    push_unique(&mut superseded_by, newer);
                }
                supersedes.sort();
                superseded_by.sort();
            }
        }
        for old in &supersedes {
            check!(
                !index.supersedes_transitively(old, &id),
                "superseding {old:?} would make a cycle"
            );
        }
        for newer in &superseded_by {
            check!(
                !index.supersedes_transitively(&id, newer),
                "being superseded by {newer:?} would make a cycle"
            );
        }

        let stamp = Stamp {
            ingest: now,
            event: input.event_time,
            valid_from: input.valid_from,
            valid_to: input.valid_to,
            subject,
            supersedes,
            superseded_by,
            entities,
            hash: Some(hash),
        };
        let mut metadata = Value::Object(user_metadata);
        let version = apply_placement(&stamp, &mut metadata);
        let document = DocumentInput {
            id: id.clone(),
            text: input.text.clone(),
            source_uri: input.source_uri.clone(),
            version,
            metadata: Some(metadata),
            chunk_words: Some(CHUNK_WORDS),
            overlap_words: Some(0),
            store_text: Some(true),
            embeddings: input.embedding.clone().map(|embedding| vec![embedding]),
            title_weight: None,
        };
        let chunks = chunk_document(&document)
            .map_err(|e| MemoryError::Invalid(format!("invalid fact: {e}")))?;
        check!(
            chunks.len() == 1,
            "text must fit one chunk of at most {CHUNK_WORDS} words; remember it as several facts"
        );
        Ok(Planned {
            id,
            input,
            stamp,
            idempotent: false,
            replaces,
            document: Some(document),
        })
    }

    /// Removes facts. Idempotent: repeating a call removes nothing more. Removal is one
    /// journal transaction per 4096 facts, so a crash between transactions leaves a prefix
    /// removed and the same call finishes the rest.
    ///
    /// Without `include_chain` only the matched facts go, and a removed successor makes its
    /// predecessor current again. Deleted text stays in the append-only journal until the
    /// store compacts; `purge` compacts afterwards.
    pub fn forget(&self, scope: &ForgetScope, options: &ForgetOptions) -> Result<ForgetReport> {
        let _writer = self.writer.lock();
        self.ensure_fresh();
        let mut ids = self.resolve_scope(scope)?;
        if options.include_chain {
            let index = self.index.read();
            let mut chained = BTreeSet::new();
            for id in &ids {
                chained.extend(index.component(id));
            }
            ids = chained;
        }
        let mut stamps: Vec<(String, Stamp)> = Vec::new();
        for id in &ids {
            if let Some((version, metadata, _)) = self.core.document_snapshot(id) {
                stamps.push((
                    id.clone(),
                    Stamp::from_parts(&version, &metadata).unwrap_or_default(),
                ));
            }
        }
        let forgotten: Vec<String> = stamps.iter().map(|(id, _)| id.clone()).collect();
        if options.dry_run || forgotten.is_empty() {
            return Ok(ForgetReport {
                forgotten,
                dry_run: options.dry_run,
                purged: false,
            });
        }
        let rewrites = self.bridge_survivors(&forgotten)?;
        let before = self.core.revision();
        let mut commits = 0;
        // Survivors are re-linked before anything is deleted: a crash between the two leaves
        // extra but valid links, and the same call finishes the job.
        for batch in rewrites.chunks(MAX_BATCH) {
            self.core.memorize_documents(batch.to_vec())?;
            commits += 1;
        }
        let (_, delete_commits) = self.core.delete_documents(&forgotten)?;
        commits += delete_commits;
        if rewrites.is_empty() {
            self.settle(before, commits, |index| {
                for (id, stamp) in &stamps {
                    index.remove(id, stamp);
                }
            });
        } else {
            self.index.write().revision = None;
        }
        let mut purged = false;
        if options.purge {
            self.core.compact()?;
            self.index.write().revision = None;
            purged = true;
        }
        if let Some(observer) = &self.observer {
            observer.on_forgotten(&forgotten);
        }
        Ok(ForgetReport {
            forgotten,
            dry_run: false,
            purged,
        })
    }

    /// Rewrites of the surviving facts whose links run through facts about to be removed, so
    /// that removing a fact from the middle of a chain does not make an older version current
    /// again: each survivor that superseded a removed fact now supersedes what that fact did.
    fn bridge_survivors(&self, forgotten: &[String]) -> Result<Vec<DocumentInput>> {
        let dead: BTreeSet<String> = forgotten.iter().cloned().collect();
        let index = self.index.read();
        let mut affected: BTreeSet<String> = BTreeSet::new();
        for id in forgotten {
            affected.extend(
                index
                    .edges(id)
                    .iter()
                    .map(|edge| edge.by.clone())
                    .filter(|by| !dead.contains(by)),
            );
        }
        let mut rewrites = Vec::new();
        for survivor in affected {
            let Some(parts) = self.core.document_parts(&survivor) else {
                continue;
            };
            let Some(mut stamp) = Stamp::from_parts(&parts.version, &parts.metadata) else {
                continue;
            };
            let mut supersedes: BTreeSet<String> = stamp
                .supersedes
                .iter()
                .filter(|id| !dead.contains(*id))
                .cloned()
                .collect();
            supersedes.extend(index.bridge(&survivor, &dead));
            let supersedes: Vec<String> = supersedes.into_iter().collect();
            let hints: Vec<String> = stamp
                .superseded_by
                .iter()
                .filter(|id| !dead.contains(*id))
                .cloned()
                .collect();
            let mut current = stamp.supersedes.clone();
            current.sort();
            if supersedes == current && hints == stamp.superseded_by {
                continue;
            }
            stamp.supersedes = supersedes;
            stamp.superseded_by = hints;
            let mut metadata = parts.metadata;
            let version = apply_placement(&stamp, &mut metadata);
            rewrites.push(DocumentInput {
                id: survivor,
                text: parts.text.unwrap_or_default(),
                source_uri: parts.source_uri,
                version,
                metadata: Some(metadata),
                chunk_words: Some(CHUNK_WORDS),
                overlap_words: Some(0),
                store_text: Some(true),
                embeddings: parts.embedding.map(|embedding| vec![embedding]),
                title_weight: None,
            });
        }
        Ok(rewrites)
    }

    fn resolve_scope(&self, scope: &ForgetScope) -> Result<BTreeSet<String>> {
        match scope {
            ForgetScope::Ids { ids } => {
                check!(
                    ids.iter().all(|id| valid_id(id)),
                    "ids must be valid fact ids"
                );
                Ok(ids.iter().cloned().collect())
            }
            ForgetScope::Subject { subject } => Ok(self
                .index
                .read()
                .subject_ids(subject.trim())
                .into_iter()
                .collect()),
            ForgetScope::Entity { entity } => Ok(self
                .index
                .read()
                .entity_ids(entity.trim())
                .iter()
                .cloned()
                .collect()),
            ForgetScope::IngestedBefore { time } => {
                Ok(self.scan_ids(|_, stamp| stamp.is_some_and(|stamp| stamp.ingest < *time)))
            }
            ForgetScope::EventBefore { time } => Ok(self.scan_ids(|_, stamp| {
                stamp.is_some_and(|stamp| {
                    stamp.event.or(stamp.valid_from).unwrap_or(stamp.ingest) < *time
                })
            })),
            ForgetScope::Matching { filter } => {
                let Some(filter) = filter.as_object().filter(|f| !f.is_empty()) else {
                    invalid!("filter must be a non-empty JSON object");
                };
                check!(
                    !filter.contains_key(KEY),
                    "metadata key {KEY:?} is reserved"
                );
                let mut ids = BTreeSet::new();
                self.core.scan_documents(|document| {
                    if Stamp::from_parts(document.version, document.metadata).is_some()
                        && filter
                            .iter()
                            .all(|(key, value)| document.metadata.get(key) == Some(value))
                    {
                        ids.insert(document.id.to_owned());
                    }
                });
                Ok(ids)
            }
        }
    }

    fn scan_ids(&self, keep: impl Fn(&str, Option<&Stamp>) -> bool) -> BTreeSet<String> {
        let mut ids = BTreeSet::new();
        self.core.scan_documents(|document| {
            let stamp = Stamp::from_parts(document.version, document.metadata);
            if stamp.is_some() && keep(document.id, stamp.as_ref()) {
                ids.insert(document.id.to_owned());
            }
        });
        ids
    }
}
