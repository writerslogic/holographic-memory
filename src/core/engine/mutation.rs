// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{HmsCore, ShardSet};
use crate::core::{
    entangled::EntangledHVec,
    rules::CompositionRule,
    types::{Relation, RelationType},
};

pub(super) const TRANSACTION_MAGIC: &[u8] = crate::core::format::TRANSACTION_MAGIC;
pub(super) const MEANING_PREFIX: &str = "hms:meaning:";

#[derive(Clone, Serialize, Deserialize)]
pub(super) enum Mutation {
    Vector {
        id: String,
        vector: EntangledHVec,
    },
    Atom {
        id: String,
        vector: EntangledHVec,
    },
    Composite {
        id: String,
        vector: EntangledHVec,
    },
    StoredTriple(crate::core::triple_store::TripleRecord),
    Document(Arc<crate::core::documents::StoredDocument>),
    Derived {
        id: String,
        children: Vec<String>,
    },
    DeleteDocument {
        id: String,
    },
    Delete {
        id: String,
    },
    Triplet {
        id: String,
        subject: String,
        relation: String,
        object: String,
    },
    Relation(Relation),
    RemoveRelation {
        source: String,
        relation: String,
        target: String,
    },
    RelationType(RelationType),
    Rule(CompositionRule),
}

/// A batch of mutations with what [`HmsCore::prepare_mutations`] computed for each of them.
/// Installation takes the prepared values out, so the batch is passed by reference and its
/// length stays readable afterwards.
pub(super) struct PreparedBatch<'a> {
    mutations: &'a [Mutation],
    prepared: std::cell::RefCell<Vec<Prepared>>,
}

impl PreparedBatch<'_> {
    pub(super) fn len(&self) -> usize {
        self.mutations.len()
    }
}

enum Prepared {
    None,
    Vector {
        id: String,
        vector: EntangledHVec,
    },
    Document {
        document: Arc<crate::core::documents::StoredDocument>,
        /// The stored document the batch was prepared against, if any.
        previous: Option<Arc<crate::core::documents::StoredDocument>>,
        /// Chunk ids of `previous` that `document` no longer has.
        stale: Vec<String>,
        chunks: Vec<(String, EntangledHVec)>,
    },
}

/// Chunk ids of `previous` absent from `document`.
fn stale_chunks(
    previous: Option<&crate::core::documents::StoredDocument>,
    document: &crate::core::documents::StoredDocument,
) -> Vec<String> {
    let Some(previous) = previous else {
        return Vec::new();
    };
    let ids: std::collections::BTreeSet<_> = document
        .chunks
        .iter()
        .map(|chunk| chunk.id.as_str())
        .collect();
    previous
        .chunks
        .iter()
        .filter(|chunk| !ids.contains(chunk.id.as_str()))
        .map(|chunk| chunk.id.clone())
        .collect()
}

pub(crate) fn validate_id(id: &str) -> Result<()> {
    ensure!(
        !id.is_empty() && id.len() <= u16::MAX as usize && !id.contains('\0'),
        "IDs must contain 1..=65535 UTF-8 bytes and no null characters"
    );
    Ok(())
}

pub(super) fn validate_public_id(id: &str) -> Result<()> {
    validate_id(id)?;
    ensure!(
        !id.starts_with(MEANING_PREFIX) && !id.starts_with(crate::core::documents::CHUNK_PREFIX),
        "hms:chunk: and hms:meaning: IDs are reserved"
    );
    Ok(())
}

impl HmsCore {
    pub(super) fn commit(&self, mutations: &[Mutation]) -> Result<()> {
        self.commit_with_append(mutations, |payload| self.arena_write(payload).map(|_| ()))
    }

    fn commit_with_append(
        &self,
        mutations: &[Mutation],
        append: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !mutations.is_empty() && mutations.len() <= 4096,
            "transaction requires 1..=4096 operations"
        );
        for mutation in mutations {
            self.validate_mutation(mutation)?;
        }
        let mut payload = Vec::with_capacity(TRANSACTION_MAGIC.len() + 128);
        payload.extend_from_slice(TRANSACTION_MAGIC);
        serde_json::to_writer(&mut payload, mutations)?;
        // Lookups, clones and the chunk diff of every mutation are prepared while readers are
        // still served and before this writer queues; the exclusive window only installs them
        // (a document replaced by another writer in between is re-diffed there).
        let mutations = &self.prepare_mutations(mutations);
        let transaction = self.mutation_gate.begin_write();
        append(&payload)?;
        let _transaction = transaction.publish();
        let shards = self.shards.write();
        self.apply_mutations(mutations, &shards)?;
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    pub(super) fn validate_mutation(&self, mutation: &Mutation) -> Result<()> {
        match mutation {
            Mutation::Vector { id, vector }
            | Mutation::Atom { id, vector }
            | Mutation::Composite { id, vector } => {
                validate_id(id)?;
                ensure!(
                    vector.dim == self.dimensions
                        && vector
                            .indices()
                            .last()
                            .is_none_or(|&i| (i as usize) < self.dimensions)
                        && vector.indices().windows(2).all(|p| p[0] < p[1]),
                    "invalid vector dimensions or indices"
                );
            }
            Mutation::Document(document) => {
                validate_id(&document.id)?;
                ensure!(
                    !document.chunks.is_empty() && document.chunks.len() <= 4096,
                    "invalid document chunk count"
                );
                for chunk in &document.chunks {
                    validate_id(&chunk.id)?;
                    ensure!(
                        chunk.vector.dim == self.dimensions
                            && chunk
                                .vector
                                .indices()
                                .last()
                                .is_none_or(|&i| (i as usize) < self.dimensions)
                            && chunk.vector.indices().windows(2).all(|p| p[0] < p[1])
                            && chunk.start_byte < chunk.end_byte
                            && chunk.end_byte as usize
                                <= crate::core::documents::MAX_DOCUMENT_BYTES
                            && (chunk.end_byte - chunk.start_byte) as usize
                                <= crate::core::documents::MAX_CHUNK_BYTES
                            && chunk
                                .text
                                .as_ref()
                                .is_none_or(|text| text.len()
                                    == (chunk.end_byte - chunk.start_byte) as usize),
                        "document vector dimension mismatch"
                    );
                    if let Some(embedding) = &chunk.embedding {
                        let dim = self
                            .config
                            .embedding_space
                            .as_ref()
                            .map_or(0, |s| s.dimensions);
                        crate::core::documents::normalize_embedding(embedding, dim)?;
                    }
                }
            }
            Mutation::Derived { id, children } => {
                validate_public_id(id)?;
                let prefix = format!("{MEANING_PREFIX}{}:{id}:", id.len());
                ensure!(
                    children.len() <= 4094
                        && children.iter().all(|child| child.starts_with(&prefix)),
                    "invalid derived meaning ownership"
                );
                for child in children {
                    validate_id(child)?;
                }
            }
            Mutation::DeleteDocument { id } => validate_id(id)?,
            Mutation::StoredTriple(t) => {
                ensure!(
                    self.meaning_enabled(),
                    "stored triple requires meaning memory"
                );
                for id in [&t.subject_id, &t.relation_id, &t.object_id, &t.composite_id] {
                    validate_id(id)?;
                }
            }
            Mutation::Delete { id } => validate_id(id)?,
            Mutation::Triplet {
                id,
                subject,
                relation,
                object,
            } => {
                for id in [id, subject, relation, object] {
                    validate_id(id)?;
                }
                ensure!(
                    self.meaning_enabled(),
                    "triplet ingestion requires meaningEnabled"
                );
            }
            Mutation::Relation(r) => {
                for id in [&r.source_id, &r.relation_type, &r.target_id] {
                    validate_id(id)?;
                }
                ensure!(
                    r.valid_from.is_finite()
                        && r.valid_to.is_finite()
                        && r.valid_from >= 0.0
                        && r.valid_to >= 0.0
                        && (r.valid_to == 0.0 || r.valid_to >= r.valid_from),
                    "invalid relation validity interval"
                );
            }
            Mutation::RemoveRelation {
                source,
                relation,
                target,
            } => {
                for id in [source, relation, target] {
                    validate_id(id)?;
                }
            }
            Mutation::RelationType(r) => validate_id(&r.name)?,
            Mutation::Rule(r) => {
                ensure!(
                    self.meaning_enabled(),
                    "composition rules require meaningEnabled"
                );
                validate_id(&r.name)?;
                validate_id(&r.output_relation)?;
                ensure!(
                    !r.input_relations.is_empty(),
                    "rule requires input relations"
                );
                for id in &r.input_relations {
                    validate_id(id)?;
                }
            }
        }
        Ok(())
    }

    /// Everything applying a batch needs that can be computed before its exclusive window:
    /// the clones handed to the shards, each replaced document and the chunk ids it loses.
    /// Validation is the caller's ([`Self::commit`] and replay validate first).
    pub(super) fn prepare_mutations<'a>(&self, mutations: &'a [Mutation]) -> PreparedBatch<'a> {
        let prepared = mutations
            .iter()
            .map(|mutation| match mutation {
                Mutation::Vector { id, vector } | Mutation::Composite { id, vector } => {
                    Prepared::Vector {
                        id: id.clone(),
                        vector: vector.clone(),
                    }
                }
                Mutation::Document(document) => {
                    let previous = self.documents.read().get(&document.id).cloned();
                    let stale = stale_chunks(previous.as_deref(), document);
                    let chunks = document
                        .chunks
                        .iter()
                        .map(|chunk| (chunk.id.clone(), chunk.vector.clone()))
                        .collect();
                    Prepared::Document {
                        document: Arc::clone(document),
                        previous,
                        stale,
                        chunks,
                    }
                }
                _ => Prepared::None,
            })
            .collect();
        PreparedBatch {
            mutations,
            prepared: std::cell::RefCell::new(prepared),
        }
    }

    /// Installs a prepared batch into the live state (once: the prepared values are taken).
    /// A document whose stored predecessor is no longer the one the batch was prepared
    /// against (another writer replaced it in between) has its stale chunks re-diffed here.
    pub(super) fn apply_mutations(
        &self,
        batch: &PreparedBatch<'_>,
        shards: &ShardSet,
    ) -> Result<()> {
        let prepared = std::mem::take(&mut *batch.prepared.borrow_mut());
        ensure!(
            prepared.len() == batch.len(),
            "prepared batch already installed"
        );
        for (mutation, prepared) in batch.mutations.iter().zip(prepared) {
            match (mutation, prepared) {
                (
                    Mutation::Vector { id, vector },
                    Prepared::Vector {
                        id: key,
                        vector: value,
                    },
                ) => {
                    self.remove_derived_from_memory(id, shards)?;
                    if let Some(composites) = &self.composite_memory {
                        composites.delete(id);
                    }
                    if let Some(triples) = &self.triple_store {
                        triples.remove_composite(id);
                    }
                    shards.insert(key, value, self.dimensions)?;
                    if let Some(atoms) = &self.atom_memory {
                        atoms.insert_with_vec(id, vector);
                    }
                }
                (Mutation::Atom { id, vector }, _) => {
                    if let Some(atoms) = &self.atom_memory {
                        atoms.insert_with_vec(id, vector);
                    }
                }
                (
                    Mutation::Composite { id, vector },
                    Prepared::Vector {
                        id: key,
                        vector: value,
                    },
                ) => {
                    if let Some(composites) = &self.composite_memory {
                        composites.insert(id.clone(), vector.clone());
                    }
                    shards.insert(key, value, self.dimensions)?;
                }
                (Mutation::StoredTriple(record), _) => {
                    if let Some(triples) = &self.triple_store {
                        triples.load_triple(record.clone());
                    }
                }
                (
                    Mutation::Document(_),
                    Prepared::Document {
                        document,
                        previous,
                        mut stale,
                        chunks,
                    },
                ) => {
                    let current = self.documents.read().get(&document.id).cloned();
                    let unchanged = match (&previous, &current) {
                        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                        (None, None) => true,
                        _ => false,
                    };
                    if !unchanged {
                        stale = stale_chunks(current.as_deref(), &document);
                    }
                    for chunk in &stale {
                        shards.remove(chunk, self.dimensions)?;
                    }
                    for (chunk, vector) in chunks {
                        shards.insert(chunk, vector, self.dimensions)?;
                    }
                    self.documents.write().insert(document.id.clone(), document);
                }
                (
                    Mutation::Vector { .. } | Mutation::Composite { .. } | Mutation::Document(_),
                    _,
                ) => {
                    unreachable!("mutation prepared with the wrong variant")
                }
                (Mutation::Derived { id, children }, _) => {
                    self.derived.write().insert(id.clone(), children.clone());
                }
                (Mutation::DeleteDocument { id }, _) => {
                    self.remove_document_from_memory(id, shards)?;
                }
                (Mutation::Delete { id }, _) => {
                    self.remove_derived_from_memory(id, shards)?;
                    self.remove_memory_id(id, shards)?;
                }
                (
                    Mutation::Triplet {
                        id,
                        subject,
                        relation,
                        object,
                    },
                    _,
                ) => {
                    self.remove_derived_from_memory(id, shards)?;
                    let atoms = self
                        .atom_memory
                        .as_ref()
                        .expect("validated meaning configuration");
                    let composites = self
                        .composite_memory
                        .as_ref()
                        .expect("validated meaning configuration");
                    let triples = self
                        .triple_store
                        .as_ref()
                        .expect("validated meaning configuration");
                    let roles = self
                        .role_registry
                        .as_ref()
                        .expect("validated meaning configuration");
                    let (_, s) = atoms.get_or_insert(subject);
                    let (_, r) = atoms.get_or_insert(relation);
                    let (_, o) = atoms.get_or_insert(object);
                    let vector = roles.compose_triple(&s, &r, &o);
                    composites.insert(id.clone(), vector.clone());
                    triples.remove_composite(id);
                    triples.add(subject, relation, object, id);
                    shards.insert(id.clone(), vector, self.dimensions)?;
                }
                (Mutation::Relation(r), _) => {
                    self.graph
                        .remove(&r.source_id, &r.relation_type, &r.target_id);
                    self.graph.add(r);
                }
                (
                    Mutation::RemoveRelation {
                        source,
                        relation,
                        target,
                    },
                    _,
                ) => {
                    self.graph.remove(source, relation, target);
                }
                (Mutation::RelationType(r), _) => self.graph.declare_type(r.clone()),
                (Mutation::Rule(r), _) => {
                    self.rule_store
                        .as_ref()
                        .expect("validated meaning configuration")
                        .add_rule(r.clone());
                }
            }
        }
        Ok(())
    }

    fn remove_derived_from_memory(&self, id: &str, shards: &ShardSet) -> Result<()> {
        if let Some(children) = self.derived.write().remove(id) {
            for child in children {
                self.remove_memory_id(&child, shards)?;
            }
        }
        Ok(())
    }

    fn remove_memory_id(&self, id: &str, shards: &ShardSet) -> Result<()> {
        shards.remove(id, self.dimensions)?;
        if let Some(atoms) = &self.atom_memory {
            atoms.delete(id);
        }
        if let Some(composites) = &self.composite_memory {
            composites.delete(id);
        }
        if let Some(triples) = &self.triple_store {
            triples.remove_composite(id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{documents, EmbeddingSpace, HmsConfig};
    use crate::{DocumentInput, SearchOptions};
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    fn config(encrypted: bool) -> HmsConfig {
        let mut config = HmsConfig {
            embedding_space: Some(EmbeddingSpace {
                model: "publication-fixture".into(),
                revision: "1".into(),
                dimensions: 2,
                normalization: "l2".into(),
                metric: "cosine".into(),
            }),
            ..Default::default()
        };
        config.security.encryption_enabled = encrypted;
        if encrypted {
            config.security.encryption_passphrase = Some("publication-test-fixture".into());
        }
        config
    }

    fn modes() -> impl Iterator<Item = bool> {
        [false, true]
            .into_iter()
            .filter(|encrypted| !encrypted || cfg!(feature = "security"))
    }

    fn document(id: &str, version: &str) -> DocumentInput {
        serde_json::from_value(json!({
            "id":id,"text":format!("needle source {version}"),"version":version,
            "sourceUri":format!("source:{version}"),"metadata":{"revision":version},
            "embeddings":[if version == "old" { vec![1.0,0.0] } else { vec![0.0,1.0] }],
        }))
        .unwrap()
    }

    /// A batch prepared against one stored version of a document and installed after another
    /// writer replaced that version must diff its stale chunks against the version actually
    /// stored, or the replaced version's extra chunk vectors would survive as orphans.
    #[test]
    fn prepared_batch_rediffs_a_document_replaced_before_installation() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let core = HmsCore::new(
            1024,
            Some(directory.path().display().to_string()),
            Some(config(false)),
        )?;
        core.memorize_documents(vec![document("a", "old")])?;
        let prepare = |input: DocumentInput| -> Result<Vec<Mutation>> {
            let stored = documents::prepare(
                input,
                core.dimensions,
                core.config.embedding_space.as_ref().map(|s| s.dimensions),
                core.config.text_encoder,
            )?;
            Ok(vec![Mutation::Document(Arc::new(stored))])
        };
        let short = prepare(document("a", "new"))?;
        let batch = core.prepare_mutations(&short);
        // Another writer replaces "a" with a two-chunk version (300 words at the default 256
        // words per chunk) in between.
        let long: DocumentInput = serde_json::from_value(json!({
            "id":"a","text":"needle ".repeat(300),"version":"long",
            "sourceUri":"source:long","metadata":{"revision":"long"},
            "embeddings":[vec![0.0,1.0], vec![1.0,0.0]],
        }))?;
        core.memorize_documents(vec![long])?;
        let stored = core.documents.read().get("a").cloned().expect("stored");
        assert_eq!(
            stored.chunks.len(),
            2,
            "the long version splits into two chunks"
        );
        assert_eq!(core.shards.read().count(), 2);
        {
            let _transaction = core.mutation_gate.write();
            let shards = core.shards.write();
            core.apply_mutations(&batch, &shards)?;
        }
        let stored = core.documents.read().get("a").cloned().expect("stored");
        assert_eq!(stored.version, "new");
        assert_eq!(stored.chunks.len(), 1);
        assert_eq!(
            core.shards.read().count(),
            1,
            "the replaced version's second chunk vector was removed"
        );
        assert!(
            core.apply_mutations(&batch, &core.shards.read()).is_err(),
            "a batch installs once"
        );
        Ok(())
    }

    fn replacements(core: &HmsCore) -> Result<Vec<Mutation>> {
        ["a", "b"]
            .into_iter()
            .map(|id| {
                documents::prepare(
                    document(id, "new"),
                    core.dimensions,
                    Some(2),
                    core.config.text_encoder,
                )
                .map(|document| Mutation::Document(Arc::new(document)))
            })
            .collect()
    }

    #[test]
    fn shared_documents_preserve_legacy_wire_bytes_and_owner_lifetime() -> Result<()> {
        #[derive(Serialize)]
        enum Legacy<'a> {
            Document(&'a documents::StoredDocument),
        }
        let directory = tempfile::tempdir()?;
        let core = HmsCore::new(
            1024,
            Some(directory.path().display().to_string()),
            Some(config(false)),
        )?;
        let mutations = replacements(&core)?;
        let legacy = mutations
            .iter()
            .map(|mutation| match mutation {
                Mutation::Document(document) => Legacy::Document(document),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&legacy)?;
        assert_eq!(serde_json::to_vec(&mutations)?, bytes);
        {
            let mut omitted = serde_json::to_value(&legacy)?;
            let document = omitted[0]["Document"].as_object_mut().unwrap();
            document.remove("source_uri");
            let chunk = document["chunks"][0].as_object_mut().unwrap();
            chunk.remove("embedding");
            chunk.remove("text");
            chunk.remove("stemmed");
            let decoded: Vec<Mutation> = serde_json::from_value(omitted)?;
            let Mutation::Document(legacy_document) = &decoded[0] else {
                unreachable!()
            };
            assert!(legacy_document.source_uri.is_none());
            assert!(legacy_document.chunks[0].embedding.is_none());
            assert!(legacy_document.chunks[0].text.is_none());
            assert!(!legacy_document.chunks[0].stemmed);
        }
        let restored: Vec<Mutation> = serde_json::from_slice(&bytes)?;
        assert_eq!(serde_json::to_vec(&restored)?, bytes);
        core.commit(&restored)?;
        for mutation in &restored {
            let Mutation::Document(document) = mutation else {
                unreachable!()
            };
            // The store installs the committed document with its chunk text moved to the
            // text segment; serialized with the text resolved it is byte-identical to the
            // committed one.
            let store = core.documents.read();
            let stored = store.get(&document.id).unwrap();
            assert!(stored
                .chunks
                .iter()
                .all(|c| c.text.is_none() && c.text_at.is_some()));
            let segment = store.text_segment();
            assert_eq!(
                serde_json::to_vec(&crate::core::documents::DocumentWithText {
                    document: stored,
                    segment: segment.as_deref(),
                })?,
                serde_json::to_vec(document.as_ref())?
            );
        }
        drop(restored);
        let before = current(&core, "new")?;
        assert_eq!(before.as_array().unwrap().len(), 2);
        core.memorize_document(document("a", "old"))?;
        let after = core.search_documents(
            "needle",
            &SearchOptions {
                source_uri: Some("source:new".into()),
                filter: Some(json!({"revision":"new"})),
                embedding: Some(vec![0.0, 1.0]),
                ..Default::default()
            },
        )?;
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].document_id, "b");
        assert_eq!(after[0].version, "new");
        assert_eq!(after[0].source_uri.as_deref(), Some("source:new"));
        assert_eq!(after[0].text.as_deref(), Some("needle source new"));
        assert_eq!(after[0].semantic_score, Some(1.0));
        assert_eq!(serde_json::to_vec(&mutations)?, bytes);
        Ok(())
    }

    #[test]
    fn owned_document_buffers_stay_compact_through_publication_and_recovery() -> Result<()> {
        fn reserved(value: &str) -> String {
            let mut result = String::with_capacity(64);
            result.push_str(value);
            result
        }
        fn metadata(value: &serde_json::Value) {
            match value {
                serde_json::Value::Array(values) => {
                    assert_eq!(values.capacity(), values.len());
                    for value in values {
                        metadata(value);
                    }
                }
                serde_json::Value::Object(values) => {
                    for (key, value) in values {
                        assert_eq!(key.capacity(), key.len());
                        metadata(value);
                    }
                }
                serde_json::Value::String(value) => assert_eq!(value.capacity(), value.len()),
                _ => {}
            }
        }
        fn compact(document: &documents::StoredDocument) {
            assert_eq!(document.id.capacity(), document.id.len());
            assert_eq!(document.version.capacity(), document.version.len());
            if let Some(uri) = &document.source_uri {
                assert_eq!(uri.capacity(), uri.len());
            }
            metadata(&document.metadata);
            assert_eq!(document.chunks.capacity(), document.chunks.len());
            for chunk in &document.chunks {
                assert_eq!(chunk.id.capacity(), chunk.id.len());
                if let Some(text) = &chunk.text {
                    assert_eq!(text.capacity(), text.len());
                }
                assert_eq!(chunk.vector.indices.capacity(), chunk.vector.indices.len());
                if let Some(values) = &chunk.embedding {
                    assert_eq!(values.capacity(), values.len());
                }
                chunk.terms.assert_valid_storage();
            }
        }
        for encrypted in modes() {
            for (vsa, dense, count) in [
                (1024, 127, 3),
                (1024, 128, 3),
                (1024, 129, 3),
                (1536, 128, 3),
                (1024, 128, 4),
                (1024, 128, 5),
            ] {
                let directory = tempfile::tempdir()?;
                let path = directory.path().display().to_string();
                let mut config = config(encrypted);
                config.embedding_space.as_mut().unwrap().dimensions = dense;
                let core = HmsCore::new(vsa, Some(path.clone()), Some(config.clone()))?;
                let mut tags = Vec::with_capacity(64);
                tags.extend((0..count).map(|value| json!(value)));
                let mut values = serde_json::Map::new();
                values.insert(reserved("tags"), serde_json::Value::Array(tags));
                values.insert(
                    reserved("empty"),
                    serde_json::Value::Array(Vec::with_capacity(64)),
                );
                values.insert(
                    reserved("label"),
                    serde_json::Value::String(reserved("µ😀")),
                );
                let mut embedding = vec![0.0; dense];
                embedding[0] = 1.0;
                let input = DocumentInput {
                    id: reserved("buffers"),
                    text: "needle source".into(),
                    source_uri: Some(reserved("source:buffers")),
                    version: Some(reserved("1")),
                    metadata: Some(serde_json::Value::Object(values)),
                    embeddings: Some(vec![embedding.clone()]),
                    chunk_words: Some(4096),
                    overlap_words: Some(0),
                    store_text: Some(true),
                    title_weight: None,
                };
                let mut replacement = input.clone();
                replacement.version = Some(reserved("2"));
                core.memorize_document(input)?;
                let held = Arc::clone(core.documents.read().get("buffers").unwrap());
                compact(&held);
                let original_wire = serde_json::to_vec(&held)?;
                core.memorize_document(replacement)?;
                assert_eq!(serde_json::to_vec(&held)?, original_wire);
                let wire = {
                    let documents = core.documents.read();
                    let document = documents.get("buffers").unwrap();
                    compact(document);
                    serde_json::to_vec(document)?
                };
                let options = SearchOptions {
                    k: Some(1),
                    candidate_limit: Some(1),
                    source_uri: Some("source:buffers".into()),
                    embedding: Some(embedding),
                    ..Default::default()
                };
                let before = serde_json::to_value(core.search_documents("needle", &options)?)?;
                assert_eq!(before.as_array().unwrap().len(), 1);
                core.flush()?;
                drop(core);
                let core = HmsCore::new(vsa, Some(path.clone()), Some(config.clone()))?;
                {
                    let documents = core.documents.read();
                    let document = documents.get("buffers").unwrap();
                    compact(document);
                    assert_eq!(serde_json::to_vec(document)?, wire);
                }
                assert_eq!(
                    serde_json::to_value(core.search_documents("needle", &options)?)?,
                    before
                );
                core.compact()?;
                drop(core);
                let core = HmsCore::new(vsa, Some(path), Some(config))?;
                compact(core.documents.read().get("buffers").unwrap());
                assert_eq!(
                    serde_json::to_value(core.search_documents("needle", &options)?)?,
                    before
                );
            }
        }
        Ok(())
    }

    #[test]
    fn journal_replay_keeps_length_sized_chunk_storage() -> Result<()> {
        for encrypted in modes() {
            for count in [3, 4, 5] {
                let directory = tempfile::tempdir()?;
                let path = directory.path().display().to_string();
                let core = HmsCore::new(1024, Some(path.clone()), Some(config(encrypted)))?;
                let input: DocumentInput = serde_json::from_value(json!({
                    "id":"capacity", "version":"1", "sourceUri":"source:capacity",
                    "text": (0..count).map(|i| format!("needle term{i}")).collect::<Vec<_>>().join(" "),
                    "metadata":{"chunks":count}, "chunkWords":2, "overlapWords":0,
                    "embeddings":vec![vec![1.0,0.0]; count],
                }))?;
                core.memorize_document(input)?;
                let options = SearchOptions {
                    k: Some(10),
                    candidate_limit: Some(10),
                    source_uri: Some("source:capacity".into()),
                    filter: Some(json!({"chunks":count})),
                    embedding: Some(vec![1.0, 0.0]),
                    ..Default::default()
                };
                let before = serde_json::to_value(core.search_documents("needle", &options)?)?;
                assert_eq!(before.as_array().unwrap().len(), count);
                let wire = serde_json::to_vec(core.documents.read().get("capacity").unwrap())?;
                core.flush()?;
                drop(core);
                let core = HmsCore::new(1024, Some(path.clone()), Some(config(encrypted)))?;
                {
                    let documents = core.documents.read();
                    let document = documents.get("capacity").unwrap();
                    assert_eq!(document.chunks.len(), count);
                    assert_eq!(document.chunks.capacity(), count);
                    assert_eq!(serde_json::to_vec(document)?, wire);
                }
                assert_eq!(
                    serde_json::to_value(core.search_documents("needle", &options)?)?,
                    before
                );
                core.compact()?;
                drop(core);
                let core = HmsCore::new(1024, Some(path), Some(config(encrypted)))?;
                assert_eq!(
                    core.documents
                        .read()
                        .get("capacity")
                        .unwrap()
                        .chunks
                        .capacity(),
                    count
                );
                assert_eq!(
                    serde_json::to_value(core.search_documents("needle", &options)?)?,
                    before
                );
            }
        }
        Ok(())
    }

    #[test]
    fn replacement_chunk_transitions_remove_obsolete_vectors_and_replay_exactly() -> Result<()> {
        for encrypted in modes() {
            let directory = tempfile::tempdir()?;
            let path = directory.path().display().to_string();
            let core = HmsCore::new(1024, Some(path.clone()), Some(config(encrypted)))?;
            for chunks in [2, 1, 2, 3, 2] {
                let text = ["needle alpha", "needle beta", "needle gamma"][..chunks].join(" ");
                let input: DocumentInput = serde_json::from_value(json!({
                    "id":"changing", "text": text,
                    "version": chunks.to_string(), "sourceUri":format!("source:{chunks}"),
                    "metadata":{"chunks":chunks}, "chunkWords":2,"overlapWords":0,
                    "embeddings":vec![vec![1.0,0.0]; chunks],
                }))?;
                core.memorize_document(input)?;
                let documents = core.documents.read();
                assert_eq!(documents.get("changing").unwrap().chunks.len(), chunks);
                let shards = core.shards.read();
                for chunk in 0..3 {
                    let id = format!("hms:chunk:8:changing:{chunk}");
                    assert_eq!(shards.get_vector(&id).is_some(), chunk < chunks);
                }
            }
            let options = SearchOptions {
                embedding: Some(vec![1.0, 0.0]),
                k: Some(10),
                ..Default::default()
            };
            let before = serde_json::to_value(core.search_documents("needle", &options)?)?;
            core.flush()?;
            drop(core);
            let core = HmsCore::new(1024, Some(path.clone()), Some(config(encrypted)))?;
            assert_eq!(
                serde_json::to_value(core.search_documents("needle", &options)?)?,
                before
            );
            assert!(core
                .shards
                .read()
                .get_vector("hms:chunk:8:changing:2")
                .is_none());
            core.compact()?;
            drop(core);
            let core = HmsCore::new(1024, Some(path), Some(config(encrypted)))?;
            assert_eq!(
                serde_json::to_value(core.search_documents("needle", &options)?)?,
                before
            );
            assert!(core.delete_document("changing")?);
            assert!(core.search_documents("needle", &options)?.is_empty());
        }
        Ok(())
    }

    fn current(core: &HmsCore, version: &str) -> Result<serde_json::Value> {
        let found = core.search_documents(
            "needle",
            &SearchOptions {
                source_uri: Some(format!("source:{version}")),
                filter: Some(json!({"revision":version})),
                embedding: Some(if version == "old" {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }),
                ..Default::default()
            },
        )?;
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|row| row.version == version
            && row.text.as_deref() == Some(&format!("needle source {version}"))
            && row.semantic_score == Some(1.0)));
        Ok(serde_json::to_value(found)?)
    }

    fn reader_progress(after_append: bool) -> Result<()> {
        for encrypted in modes() {
            let directory = tempfile::tempdir()?;
            let path = directory.path().display().to_string();
            let core = Arc::new(HmsCore::new(
                1024,
                Some(path.clone()),
                Some(config(encrypted)),
            )?);
            core.memorize_documents(vec![document("a", "old"), document("b", "old")])?;
            let anchor = core.encode_text("anchor");
            core.memorize("anchor".into(), anchor.clone())?;
            let mutations = replacements(&core)?;
            let checkpoint = core.arena.checkpoint();
            let (paused_tx, paused_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            let writer_core = Arc::clone(&core);
            let writer = std::thread::spawn(move || {
                writer_core.commit_with_append(&mutations, |payload| {
                    if after_append {
                        writer_core.arena_write(payload)?;
                    }
                    paused_tx.send(())?;
                    resume_rx.recv_timeout(Duration::from_secs(15))?;
                    if !after_append {
                        writer_core.arena_write(payload)?;
                    }
                    Ok(())
                })
            });
            paused_rx.recv_timeout(Duration::from_secs(10))?;
            if after_append {
                let (payload, _) = core.arena_read_frame(checkpoint.1)?;
                assert!(payload.starts_with(TRANSACTION_MAGIC));
                let decoded: Vec<Mutation> =
                    serde_json::from_slice(&payload[TRANSACTION_MAGIC.len()..])?;
                assert_eq!(decoded.len(), 2);
                assert!(decoded.iter().all(|mutation| matches!(mutation,
                    Mutation::Document(document) if document.version == "new")));
            } else {
                assert_eq!(core.arena.checkpoint(), checkpoint);
            }
            let (document_tx, document_rx) = mpsc::channel();
            let reader_core = Arc::clone(&core);
            let reader = std::thread::spawn(move || {
                document_tx.send(reader_core.search_documents(
                    "needle",
                    &SearchOptions {
                        embedding: Some(vec![1.0, 0.0]),
                        ..Default::default()
                    },
                ))
            });
            let (vector_tx, vector_rx) = mpsc::channel();
            let reader_core = Arc::clone(&core);
            let query_anchor = anchor.clone();
            let vector_reader =
                std::thread::spawn(move || vector_tx.send(reader_core.query(&query_anchor, 10)));
            let (queued_tx, queued_rx) = mpsc::channel();
            let (committed_tx, committed_rx) = mpsc::channel();
            let second_core = Arc::clone(&core);
            let second = std::thread::spawn(move || -> Result<()> {
                queued_tx.send(())?;
                second_core.memorize("second".into(), anchor)?;
                committed_tx.send(())?;
                Ok(())
            });
            queued_rx.recv_timeout(Duration::from_secs(5))?;
            let (compacted_tx, compacted_rx) = mpsc::channel();
            let compact_core = Arc::clone(&core);
            let compactor = std::thread::spawn(move || -> Result<()> {
                compact_core.compact()?;
                compacted_tx.send(())?;
                Ok(())
            });
            let documents = document_rx.recv_timeout(Duration::from_secs(2));
            let vectors = vector_rx.recv_timeout(Duration::from_secs(2));
            let writer_waited = committed_rx.try_recv().is_err();
            let compactor_waited = compacted_rx.try_recv().is_err();
            resume_tx.send(())?;
            writer.join().unwrap()?;
            second.join().unwrap()?;
            compactor.join().unwrap()?;
            reader.join().unwrap().unwrap();
            vector_reader.join().unwrap().unwrap();
            assert!(
                writer_waited && compactor_waited,
                "publication exclusion was lost"
            );
            let documents = documents.expect("durable append blocked document queries")?;
            assert_eq!(documents.len(), 2);
            assert!(documents.iter().all(|row| row.version == "old"
                && row.text.as_deref() == Some("needle source old")
                && row.source_uri.as_deref() == Some("source:old")
                && row.metadata == json!({"revision":"old"})
                && row.semantic_score == Some(1.0)));
            assert!(vectors
                .expect("durable append blocked vector queries")
                .iter()
                .any(|row| row.id == "anchor"));
            let before = current(&core, "new")?;
            drop(core);
            let restored = HmsCore::new(1024, Some(path), Some(config(encrypted)))?;
            assert_eq!(current(&restored, "new")?, before);
            assert!(restored.shards.read().get_vector("second").is_some());
        }
        Ok(())
    }

    #[test]
    fn pending_append_allows_coherent_queries_and_excludes_other_writers() -> Result<()> {
        reader_progress(false)
    }

    #[test]
    fn durable_unpublished_batch_allows_old_queries_and_blocks_compaction() -> Result<()> {
        reader_progress(true)
    }

    #[test]
    fn append_failure_preserves_live_state_and_allows_retry() -> Result<()> {
        for encrypted in modes() {
            let directory = tempfile::tempdir()?;
            let path = directory.path().display().to_string();
            let core = HmsCore::new(1024, Some(path.clone()), Some(config(encrypted)))?;
            core.memorize_documents(vec![document("a", "old"), document("b", "old")])?;
            let before = current(&core, "old")?;
            let checkpoint = core.arena.checkpoint();
            let mutations = replacements(&core)?;
            let expected = [TRANSACTION_MAGIC, &serde_json::to_vec(&mutations)?].concat();
            let revision = core.revision.load(std::sync::atomic::Ordering::Acquire);
            assert!(core
                .commit_with_append(&mutations, |payload| {
                    assert_eq!(payload, expected);
                    anyhow::bail!("injected append failure")
                })
                .is_err());
            assert_eq!(core.arena.checkpoint(), checkpoint);
            assert_eq!(
                core.revision.load(std::sync::atomic::Ordering::Acquire),
                revision
            );
            assert_eq!(current(&core, "old")?, before);
            core.commit_with_append(&mutations, |payload| {
                assert_eq!(payload, expected);
                core.arena_write(payload).map(|_| ())
            })?;
            assert_eq!(core.arena_read_frame(checkpoint.1)?.0, expected);
            let before = current(&core, "new")?;
            drop(core);
            let restored = HmsCore::new(1024, Some(path), Some(config(encrypted)))?;
            assert_eq!(current(&restored, "new")?, before);
        }
        Ok(())
    }

    #[test]
    fn complete_transaction_bytes_match_legacy_for_all_mutations() -> Result<()> {
        for encrypted in modes() {
            let directory = tempfile::tempdir()?;
            let path = directory.path().display().to_string();
            let mut configuration = config(encrypted);
            configuration.meaning.enabled = true;
            let core = HmsCore::new(1024, Some(path.clone()), Some(configuration.clone()))?;
            let vector = core.encode_text("needle café \"\\\n");
            let mut input = document("文書\"\\\n", "new");
            input.metadata = Some(json!({
                "escaped\"\\\n": [null, true, false, "é\u{0001}", [], {}],
                "floats": [-0.0, f64::MIN_POSITIVE, f64::MAX, 1.2345678901234567],
                "integers": [i64::MIN, u64::MAX],
            }));
            let document = Arc::new(documents::prepare(
                input,
                core.dimensions,
                Some(2),
                core.config.text_encoder,
            )?);
            let mutations = vec![
                Mutation::Vector {
                    id: "vector".into(),
                    vector: vector.clone(),
                },
                Mutation::Atom {
                    id: "atom".into(),
                    vector: vector.clone(),
                },
                Mutation::Composite {
                    id: "composite".into(),
                    vector,
                },
                Mutation::StoredTriple(crate::core::triple_store::TripleRecord {
                    subject_id: "s".into(),
                    relation_id: "r".into(),
                    object_id: "o".into(),
                    composite_id: "composite".into(),
                    deleted: false,
                }),
                Mutation::Document(Arc::clone(&document)),
                Mutation::Derived {
                    id: "derived".into(),
                    children: vec!["hms:meaning:7:derived:child".into()],
                },
                Mutation::DeleteDocument {
                    id: "absent-document".into(),
                },
                Mutation::Delete {
                    id: "absent-vector".into(),
                },
                Mutation::Triplet {
                    id: "triple".into(),
                    subject: "s".into(),
                    relation: "r".into(),
                    object: "o".into(),
                },
                Mutation::Relation(Relation {
                    source_id: "s".into(),
                    relation_type: "r".into(),
                    target_id: "o".into(),
                    properties: Some("\"\\\n文書".into()),
                    valid_from: f64::MIN_POSITIVE,
                    valid_to: f64::MAX,
                }),
                Mutation::RemoveRelation {
                    source: "absent".into(),
                    relation: "r".into(),
                    target: "o".into(),
                },
                Mutation::RelationType(RelationType {
                    name: "r".into(),
                    transitive: true,
                    symmetric: false,
                }),
                Mutation::Rule(CompositionRule {
                    name: "rule".into(),
                    input_relations: vec!["r".into(), "r".into()],
                    output_relation: "rr".into(),
                }),
            ];
            let expected = [TRANSACTION_MAGIC, &serde_json::to_vec(&mutations)?].concat();
            let checkpoint = core.arena.checkpoint();
            core.commit_with_append(&mutations, |payload| {
                assert_eq!(payload, expected);
                core.arena_write(payload).map(|_| ())
            })?;
            assert_eq!(core.arena_read_frame(checkpoint.1)?.0, expected);
            let options = SearchOptions {
                embedding: Some(vec![0.0, 1.0]),
                ..Default::default()
            };
            let before = serde_json::to_value(core.search_documents("needle", &options)?)?;
            assert_eq!(before.as_array().unwrap().len(), 1);
            drop(core);
            let core = HmsCore::new(1024, Some(path.clone()), Some(configuration.clone()))?;
            assert_eq!(core.arena_read_frame(checkpoint.1)?.0, expected);
            assert_eq!(
                serde_json::to_value(core.search_documents("needle", &options)?)?,
                before
            );
            core.compact()?;
            drop(core);
            let core = HmsCore::new(1024, Some(path), Some(configuration))?;
            assert_eq!(
                serde_json::to_value(core.search_documents("needle", &options)?)?,
                before
            );
            assert_eq!(
                serde_json::to_vec(&mutations)?,
                &expected[TRANSACTION_MAGIC.len()..]
            );
        }
        Ok(())
    }

    #[test]
    fn transaction_limits_and_validation_precede_append() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let core = HmsCore::new(1024, Some(directory.path().display().to_string()), None)?;
        let checkpoint = core.arena.checkpoint();
        let revision = core.revision.load(std::sync::atomic::Ordering::Acquire);
        for count in [0, 1, 4095, 4096, 4097] {
            let mutations = vec![
                Mutation::Delete {
                    id: "absent\"\\\n文書".into()
                };
                count
            ];
            let expected = [TRANSACTION_MAGIC, &serde_json::to_vec(&mutations)?].concat();
            let called = std::cell::Cell::new(false);
            let error = core
                .commit_with_append(&mutations, |payload| {
                    called.set(true);
                    assert_eq!(payload, expected);
                    anyhow::bail!("injected append failure")
                })
                .unwrap_err();
            assert_eq!(called.get(), (1..=4096).contains(&count));
            assert!(error.to_string().contains(if called.get() {
                "injected append failure"
            } else {
                "transaction requires"
            }));
            assert_eq!(core.arena.checkpoint(), checkpoint);
            assert_eq!(
                core.revision.load(std::sync::atomic::Ordering::Acquire),
                revision
            );
        }
        let invalid = [
            Mutation::Delete { id: "valid".into() },
            Mutation::Delete { id: String::new() },
        ];
        let error = core
            .commit_with_append(&invalid, |_| panic!("invalid transaction reached append"))
            .unwrap_err();
        assert!(error.to_string().contains("IDs must"));
        assert_eq!(core.arena.checkpoint(), checkpoint);
        Ok(())
    }

    #[test]
    fn publication_exit_child() -> Result<()> {
        let Some(path) = std::env::var_os("HMS_MUTATION_PUBLICATION_TEST_PATH") else {
            return Ok(());
        };
        let phase: usize = std::env::var("HMS_MUTATION_PUBLICATION_TEST_PHASE")?.parse()?;
        ensure!(phase <= 2, "invalid publication interruption phase");
        let encrypted = std::env::var("HMS_MUTATION_PUBLICATION_TEST_ENCRYPTED")? == "true";
        let core = HmsCore::new(
            1024,
            Some(PathBuf::from(path).display().to_string()),
            Some(config(encrypted)),
        )?;
        core.commit_with_append(&replacements(&core)?, |payload| {
            if phase == 0 {
                std::process::exit(77);
            }
            core.arena_write(payload)?;
            if phase == 1 {
                std::process::exit(78);
            }
            Ok(())
        })?;
        std::process::exit(79);
    }

    #[test]
    fn interrupted_append_and_publication_recover_whole_batches() -> Result<()> {
        for encrypted in modes() {
            for phase in 0..=2 {
                let directory = tempfile::tempdir()?;
                let path = directory.path().display().to_string();
                let core = HmsCore::new(1024, Some(path.clone()), Some(config(encrypted)))?;
                core.memorize_documents(vec![document("a", "old"), document("b", "old")])?;
                drop(core);
                let child = std::process::Command::new(std::env::current_exe()?)
                    .args([
                        "--exact",
                        "core::engine::mutation::tests::publication_exit_child",
                        "--nocapture",
                    ])
                    .env("HMS_MUTATION_PUBLICATION_TEST_PATH", &path)
                    .env("HMS_MUTATION_PUBLICATION_TEST_PHASE", phase.to_string())
                    .env(
                        "HMS_MUTATION_PUBLICATION_TEST_ENCRYPTED",
                        encrypted.to_string(),
                    )
                    .output()?;
                assert_eq!(
                    child.status.code(),
                    Some(77 + phase),
                    "{}",
                    String::from_utf8_lossy(&child.stderr)
                );
                let restored = HmsCore::new(1024, Some(path.clone()), Some(config(encrypted)))?;
                let version = if phase == 0 { "old" } else { "new" };
                let before = current(&restored, version)?;
                assert!(restored
                    .search_documents(
                        "needle",
                        &SearchOptions {
                            source_uri: Some(format!(
                                "source:{}",
                                if version == "old" { "new" } else { "old" }
                            )),
                            ..Default::default()
                        }
                    )?
                    .is_empty());
                restored.compact()?;
                drop(restored);
                let restored = HmsCore::new(1024, Some(path), Some(config(encrypted)))?;
                assert_eq!(current(&restored, version)?, before);
            }
        }
        Ok(())
    }
}
