// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use super::{mutation::Mutation, WarmCore};
use crate::core::documents::StoredDocument;
use crate::core::storage::StagedArena;

/// Removes an abandoned staging generation; disarmed once publication begins.
struct StagingDir {
    path: std::path::PathBuf,
    armed: std::cell::Cell<bool>,
}

impl StagingDir {
    fn new(path: std::path::PathBuf) -> Self {
        Self {
            path,
            armed: std::cell::Cell::new(true),
        }
    }
    fn path(&self) -> &std::path::Path {
        &self.path
    }
    fn disarm(&self) {
        self.armed.set(false);
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if self.armed.get() {
            // An abandoned generation is never published, and a leftover is scrubbed or ignored
            // by the next compaction, so a failed removal only costs disk space.
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

struct Snapshot {
    checkpoint: (String, usize),
    mutations: Vec<Mutation>,
    documents: Vec<Arc<StoredDocument>>,
    /// The store's text segment at snapshot time: chunk text moved there is written back
    /// inline into the compaction frames.
    text: Option<Arc<crate::core::documents::TextSegment>>,
    /// The document store as the new generation's checkpoint (sealed for an encrypted store),
    /// in place of `documents` frames, when the store takes one (`image::encode_checkpoint`).
    document_checkpoint: Option<Vec<u8>>,
}

#[derive(Serialize)]
enum DocumentMutation<'a> {
    Document(crate::core::documents::DocumentWithText<'a>),
}

impl WarmCore {
    pub fn compact(&self) -> Result<()> {
        self.compact_impl(|| {}, || {}, false)
    }

    /// Compacts and then physically erases what the compaction replaced: the old generation
    /// and any unreferenced leftovers are overwritten with zeros and synced before they are
    /// unlinked. The text segment is scrubbed per deleted document at delete time.
    ///
    /// Afterwards nothing derived depends on the purged documents or on the order documents
    /// arrived in (see `canonicalize_derived_state`), so every query, status field
    /// other than the opaque counters, and file other than the generation's name match a store
    /// that never held them and was purged at the same point.
    pub fn compact_scrubbing(&self) -> Result<()> {
        self.sync_text_segment()?;
        self.compact_impl(|| {}, || {}, true)
    }

    /// Rebuilds the state a delete leaves shaped by history rather than by the live set: the
    /// text segment (zeroed records of deleted text, offsets that count it) and the slots, in
    /// id order as a reopen of the compacted journal lays them out; a published vector
    /// snapshot, whose graph still holds the deleted rows' vertices and the edges they shaped
    /// (rebuilt with the same graph choice); and the dense document index a delete dropped.
    fn canonicalize_derived_state(&self, documents: bool) -> Result<()> {
        if documents {
            self.canonicalize_documents()?;
        }
        if let Some(snapshot) = self.vector_index.get() {
            let graph_min_rows = if snapshot.has_graph() { 0 } else { usize::MAX };
            self.rebuild_vector_index_waiting(graph_min_rows)?;
        }
        let rebuild_dense = {
            let mut documents = self.documents.write();
            documents.discard_stale_dense_index();
            documents.dense_index_recommended()
        };
        if rebuild_dense {
            self.train_dense_index()?;
        }
        Ok(())
    }

    /// The text segment and slots of [`Self::canonicalize_derived_state`].
    fn canonicalize_documents(&self) -> Result<()> {
        self.hydrate_documents()?;
        let _transaction = self.mutation_gate.maintenance();
        self.documents
            .write()
            .rebuild_canonical(&self.storage_path.join("documents.text"))
    }

    fn compact_impl(
        &self,
        after_snapshot: impl FnOnce(),
        before_publication: impl FnOnce(),
        scrub: bool,
    ) -> Result<()> {
        let _compaction = self.compaction.lock();
        // A checkpoint records the slots as they are, so a scrubbing compaction that writes one
        // lays them out canonically first: its checkpoint is then the one a store that never
        // held the purged documents writes.
        let canonical_first = scrub && self.takes_checkpoint();
        if canonical_first {
            self.canonicalize_documents()?;
        }
        let staging = StagingDir::new(self.arena.new_generation_dir()?);
        let staged_name = staging
            .path()
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .context("staging generation has no name")?;
        let snapshot = self.compaction_snapshot(&staged_name)?;
        let (generation, mut cursor) = snapshot.checkpoint;
        after_snapshot();
        let mut arena = StagedArena::new(staging.path(), Some(&self.arena))?;
        if let Some(bytes) = &snapshot.document_checkpoint {
            // Durable (file and directory synced) before the generation can be published.
            crate::core::durable_file::atomic_write(
                &staging.path().join(super::image::FILE),
                bytes,
            )?;
        }
        for mutation in snapshot.mutations {
            self.write_compaction_frame(&mut arena, &[mutation])?;
        }
        let mut payload = Vec::new();
        for document in &snapshot.documents {
            // A document deleted since the snapshot has had its text zeroed; the delete is in
            // the tail, which the replay below applies, so it is simply not carried over.
            {
                let store = self.documents();
                if store.is_deleted(document) {
                    continue;
                }
                let binary = match self.journal {
                    super::journal::JournalFormat::Binary => {
                        super::journal::binary_document_payload(document, snapshot.text.as_deref())?
                    }
                    super::journal::JournalFormat::Json => None,
                };
                if let Some(binary) = binary {
                    payload = binary;
                } else {
                    // The JSON form carries any document; a binary store reads both.
                    payload.clear();
                    payload.extend_from_slice(super::mutation::TRANSACTION_MAGIC);
                    serde_json::to_writer(
                        &mut payload,
                        &[DocumentMutation::Document(
                            crate::core::documents::DocumentWithText {
                                document,
                                segment: snapshot.text.as_deref(),
                            },
                        )],
                    )?;
                }
            }
            arena.write_slice(&self.maybe_encrypt(&payload)?)?;
        }
        let mut prepared = arena.finish()?.prepare_compacted()?;
        for _ in 0..2 {
            let end = {
                let _transaction = self.mutation_gate.maintenance();
                self.compaction_end(&generation)?
            };
            self.arena
                .replay_compaction_tail(&mut prepared, &mut cursor, end)?;
        }
        before_publication();
        let _transaction = self.mutation_gate.maintenance();
        let end = self.compaction_end(&generation)?;
        self.arena
            .replay_compaction_tail(&mut prepared, &mut cursor, end)?;
        let audit_intent = self.audit_intent(crate::core::audit::AuditOp::Compact, "");
        if let Some(intent) = &audit_intent {
            let mutations = [Mutation::AuditPending(intent.clone())];
            self.validate_audit_transitions(&mutations)?;
            prepared.append_frame(&self.maybe_encrypt(&self.transaction_payload(&mutations)?)?)?;
        }
        staging.disarm();
        self.arena.replace_with_compacted(prepared, scrub)?;
        if let Some(intent) = audit_intent {
            self.audit_pending.write().push_back(intent);
        }
        crate::core::audit_effects::crash_point("compacted");
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        // The arena pointer has moved, so writers may continue: the lease ends here. Index
        // persistence binds the checkpoint read under the lease, holds the shards only while
        // encoding, and writes the files with no lock held; the persist call that follows
        // then finds identical metadata and writes nothing.
        let checkpoint = self.arena.checkpoint();
        _transaction.release();
        crate::core::error::after_commit(
            "compaction index persistence",
            self.persist_indices_detached(checkpoint),
        )?;
        crate::core::error::after_commit(
            "compaction index persistence",
            self.persist_indices(&self.shards.read()),
        )?;
        self.finish_audit(crate::core::audit::AuditOp::Compact, "")?;
        if scrub {
            // Still under the compaction lock: a concurrent compaction must not capture
            // documents whose text offsets the canonical rebuild is replacing.
            crate::core::error::after_commit(
                "compaction derived-state reconciliation",
                self.canonicalize_derived_state(!canonical_first),
            )?;
        }
        Ok(())
    }

    fn compaction_snapshot(&self, staged_name: &str) -> Result<Snapshot> {
        // Built before any lock is held: the snapshot below reads the store under the shards
        // guard, and a store still served from its checkpoint must not be rebuilt there.
        self.hydrate_documents()?;
        let _transaction = self.mutation_gate.maintenance();
        let shards = self.shards.read();
        let shard_vectors = shards.count();
        let mut vectors = Vec::new();
        let composite_ids: HashSet<_> = self
            .composite_memory
            .as_ref()
            .map(|memory| {
                memory
                    .inner()
                    .all_vectors()
                    .into_iter()
                    .map(|(_, id, _)| id)
                    .collect()
            })
            .unwrap_or_default();
        shards.try_for_each_shard(|shard| {
            for (id, vector) in shard.vectors.read().iter() {
                if !id.starts_with(crate::core::documents::CHUNK_PREFIX)
                    && !composite_ids.contains(id)
                {
                    vectors.push((id.clone(), vector.clone()));
                }
            }
            Ok(())
        })?;
        // The shard map's iteration order depends on its insertion and removal history; id
        // order makes the compacted journal a function of the live vectors alone.
        vectors.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let atoms = self
            .atom_memory
            .as_ref()
            .map(|memory| memory.inner().all_vectors());
        let atom_ids: HashSet<_> = atoms
            .iter()
            .flatten()
            .map(|(_, id, _)| id.as_str())
            .collect();
        let mut mutations = Vec::new();
        for (id, vector) in vectors {
            let absent_atom =
                (atoms.is_some() && !atom_ids.contains(id.as_str())).then(|| id.clone());
            mutations.push(Mutation::Vector { id, vector });
            if let Some(id) = absent_atom {
                anyhow::ensure!(
                    self.schema >= crate::core::format::MEANING_MAINTENANCE_SCHEMA_VERSION,
                    "preserving scoped meaning state requires schema 5; use warm-format upgrade"
                );
                mutations.push(Mutation::DeleteAtom { id });
            }
        }
        drop(atom_ids);
        let (documents, text, document_checkpoint) = {
            let store = self.documents.read();
            let checkpoint = if shard_vectors == 0 {
                self.encode_checkpoint(&store, staged_name)?
            } else {
                None
            };
            let documents = if checkpoint.is_some() {
                Vec::new()
            } else {
                store.shared_values().collect()
            };
            (documents, store.text_segment(), checkpoint)
        };
        mutations.extend(self.graph.snapshot().into_iter().map(Mutation::Relation));
        mutations.extend(
            self.graph
                .type_snapshot()
                .into_iter()
                .map(Mutation::RelationType),
        );
        if let Some(atoms) = atoms {
            mutations.extend(
                atoms
                    .into_iter()
                    .map(|(_, id, vector)| Mutation::Atom { id, vector }),
            );
        }
        if let Some(composites) = &self.composite_memory {
            mutations.extend(
                composites
                    .inner()
                    .all_vectors()
                    .into_iter()
                    .map(|(_, id, vector)| Mutation::Composite { id, vector }),
            );
        }
        if let Some(triples) = &self.triple_store {
            mutations.extend(triples.snapshot().into_iter().map(Mutation::StoredTriple));
        }
        if let Some(rules) = &self.rule_store {
            mutations.extend(rules.all_rules().into_iter().map(Mutation::Rule));
        }
        mutations.extend(
            self.derived
                .read()
                .iter()
                .map(|(id, children)| Mutation::Derived {
                    id: id.clone(),
                    children: children.clone(),
                }),
        );
        mutations.extend(
            self.audit_pending
                .read()
                .iter()
                .cloned()
                .map(Mutation::AuditPending),
        );
        Ok(Snapshot {
            checkpoint: self.arena.checkpoint(),
            mutations,
            documents,
            text,
            document_checkpoint,
        })
    }

    fn write_compaction_frame(
        &self,
        arena: &mut StagedArena,
        mutations: &[Mutation],
    ) -> Result<()> {
        let payload = self.transaction_payload(mutations)?;
        arena.write_slice(&self.maybe_encrypt(&payload)?)?;
        Ok(())
    }

    fn compaction_end(&self, generation: &str) -> Result<usize> {
        let checkpoint = self.arena.checkpoint();
        ensure!(
            checkpoint.0 == generation,
            "arena generation changed during compaction"
        );
        Ok(checkpoint.1)
    }

    /// Persists the indices for `checkpoint` without holding the writer lease: the shards are
    /// read only while the indices are encoded, and the files are written afterwards.
    fn persist_indices_detached(&self, checkpoint: (String, usize)) -> Result<()> {
        let encoded = self.encode_indices(&self.shards.read(), checkpoint)?;
        self.write_indices(encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    use crate::core::documents::{DocumentInput, SearchOptions};
    use crate::core::WarmConfig;
    use serde_json::json;

    #[test]
    fn audit_intents_survive_snapshot_and_concurrent_compaction_tail() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut config = WarmConfig::default();
        config.security.audit_enabled = true;
        config.document_checkpoint.enabled = false;
        let mut core = WarmCore::new(
            256,
            Some(dir.path().display().to_string()),
            Some(config.clone()),
        )?;
        core.audit_effects
            .as_mut()
            .unwrap()
            .make_read_only(dir.path())?;
        let write = |id: &str| {
            let error = core
                .memorize(id.into(), crate::EntangledHVec::new_deterministic(256, 7))
                .unwrap_err();
            assert!(error.is::<crate::CommittedMutationError>());
        };
        write("snapshot");
        let error = core
            .compact_impl(|| write("tail"), || {}, false)
            .unwrap_err();
        assert!(error.is::<crate::CommittedMutationError>());
        let intents: Vec<_> = core
            .audit_pending
            .read()
            .iter()
            .map(|intent| intent.0.clone())
            .collect();
        assert_eq!(intents.len(), 3);
        drop(core);
        let core = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config))?;
        assert_eq!(core.audit_since(0)?, intents);
        assert_eq!(core.shards.read().count(), 2);
        assert_eq!(core.pending_audit_effects(), 0);
        Ok(())
    }

    fn config() -> WarmConfig {
        super::super::fixture_config(2)
    }

    fn configurations() -> impl Iterator<Item = WarmConfig> {
        [false, true]
            .into_iter()
            .filter(|encrypted| !encrypted || cfg!(feature = "security"))
            .map(|encrypted| {
                let mut config = config();
                config.security.encryption_enabled = encrypted;
                if encrypted {
                    config.security.encryption_passphrase =
                        Some("compaction-replay-fixture".into());
                }
                config
            })
    }

    fn document(id: &str, version: &str) -> DocumentInput {
        serde_json::from_value(json!({
            "id": id, "version": version, "text": "needle durable source",
            "sourceUri": format!("source:{version}"), "metadata": {"version": version},
            "embeddings": [[1.0, 0.0]],
        }))
        .unwrap()
    }

    #[test]
    fn scrubbing_releases_deleted_dense_owners_even_below_training_threshold() -> Result<()> {
        for count in [128, 129] {
            for config in configurations() {
                let directory = tempfile::tempdir()?;
                let path = directory.path().display().to_string();
                let core = WarmCore::new(1024, Some(path.clone()), Some(config.clone()))?;
                core.memorize_documents(
                    (0..count)
                        .map(|i| document(&format!("row-{i:03}"), "1"))
                        .collect(),
                )?;
                core.train_document_index()?;
                let owner = Arc::clone(core.documents.read().get("row-000").unwrap());
                core.delete_document("row-000")?;
                assert!(Arc::strong_count(&owner) > 1);
                core.compact_scrubbing()?;
                assert_eq!(Arc::strong_count(&owner), 1);
                assert_eq!(core.documents.read().dense_index_trained(), count > 128);
                drop(core);
                let reopened = WarmCore::new(1024, Some(path), Some(config))?;
                assert!(!reopened.has_document("row-000"));
                assert_eq!(reopened.document_count(), count - 1);
            }
        }
        Ok(())
    }

    #[test]
    fn published_compaction_failure_reports_a_committed_outcome_and_reopens() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = Some(dir.path().display().to_string());
        let core = WarmCore::new(256, path.clone(), Some(config()))?;
        core.memorize_document(document("kept", "v1"))?;
        let before = core.revision();
        let error = core
            .compact_impl(
                || {},
                || {
                    std::fs::create_dir(dir.path().join("indices.json")).unwrap();
                },
                false,
            )
            .unwrap_err();
        assert!(error.is::<crate::CommittedMutationError>(), "{error:#}");
        assert!(core.revision() != before);
        assert!(core.has_document("kept"));
        std::fs::remove_dir(dir.path().join("indices.json"))?;
        drop(core);
        let core = WarmCore::new(256, path, Some(config()))?;
        assert_eq!(
            core.get_document("kept").unwrap().text.as_deref(),
            Some("needle durable source")
        );
        core.compact()?;
        Ok(())
    }

    fn seed(core: &WarmCore) -> Result<()> {
        core.memorize_document(document("replace", "old"))?;
        core.memorize_document(document("delete", "old"))?;
        core.memorize("vector".into(), core.encode_text("original"))?;
        core.memorize("remove".into(), core.encode_text("removed"))?;
        core.add_relation(&serde_json::from_value(json!({
            "sourceId": "a", "relationType": "knows", "targetId": "b",
            "properties": null, "validFrom": 0, "validTo": 0,
        }))?)?;
        Ok(())
    }

    fn mutate(core: &WarmCore) -> Result<()> {
        core.memorize_documents(vec![document("replace", "new"), document("insert", "new")])?;
        core.delete_document("delete")?;
        core.memorize("vector".into(), core.encode_text("replacement"))?;
        core.delete("remove")?;
        core.remove_relation("a", "knows", "b")?;
        Ok(())
    }

    fn assert_current(core: &WarmCore) -> Result<serde_json::Value> {
        let options = SearchOptions {
            embedding: Some(vec![1.0, 0.0]),
            source_uri: Some("source:new".into()),
            filter: Some(json!({"version": "new"})),
            ..Default::default()
        };
        let results = core.search_documents("needle", &options)?;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.version == "new"
            && r.source_uri.as_deref() == Some("source:new")
            && r.semantic_score == Some(1.0)));
        assert_eq!(core.relation_count(), 0);
        assert_eq!(core.vector_count(), 3);
        assert_eq!(
            core.query(&core.encode_text("replacement"), 1)[0].id,
            "vector"
        );
        Ok(serde_json::to_value(results)?)
    }

    #[test]
    fn compaction_replays_acknowledged_writes_and_preserves_batch_frames() -> Result<()> {
        for config in configurations() {
            let directory = tempfile::tempdir()?;
            let path = directory.path().display().to_string();
            let core = Arc::new(WarmCore::new(
                1024,
                Some(path.clone()),
                Some(config.clone()),
            )?);
            seed(&core)?;
            let writer_core = Arc::clone(&core);
            let (start_tx, start_rx) = mpsc::channel();
            let (done_tx, done_rx) = mpsc::channel();
            let writer = std::thread::spawn(move || -> Result<()> {
                start_rx.recv()?;
                mutate(&writer_core)?;
                assert_current(&writer_core)?;
                done_tx.send(())?;
                Ok(())
            });
            let compacted = core.compact_impl(
                || {
                    start_tx.send(()).unwrap();
                    done_rx
                        .recv_timeout(Duration::from_secs(5))
                        .expect("compaction snapshot blocked mutations or queries");
                },
                || {},
                false,
            );
            writer.join().unwrap()?;
            compacted?;
            let expected = assert_current(&core)?;
            let mut offset = 0;
            let mut batch_frames = 0;
            while offset < core.arena.stats().used_bytes {
                let (data, _) = core.arena_read_frame(offset)?;
                let operations = super::super::journal::decode(&data)?;
                if operations.len() == 2 {
                    batch_frames += 1;
                    assert!(operations
                        .iter()
                        .all(|op| matches!(op, Mutation::Document(_))));
                }
                offset = core.arena.next_offset(offset)?;
            }
            assert_eq!(batch_frames, 1);
            drop(core);
            let restored = WarmCore::new(1024, Some(path), Some(config))?;
            assert_eq!(assert_current(&restored)?, expected);
        }
        Ok(())
    }

    #[test]
    fn compaction_snapshot_shares_document_payloads_after_replacement() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let core = WarmCore::new(
            1024,
            Some(directory.path().display().to_string()),
            Some(config()),
        )?;
        core.memorize_document(document("replace", "old"))?;
        let snapshot = core.compaction_snapshot("gen-test")?;
        let owner = core.documents.read().shared_values().next().unwrap();
        assert!(Arc::ptr_eq(&owner, &snapshot.documents[0]));
        let dense_pointer = owner.chunks[0].embedding.as_ref().unwrap().as_ptr();
        core.memorize_document(document("replace", "new"))?;
        assert_eq!(snapshot.documents[0].version, "old");
        assert_eq!(
            snapshot.documents[0].chunks[0]
                .embedding
                .as_ref()
                .unwrap()
                .as_ptr(),
            dense_pointer
        );
        Ok(())
    }

    #[test]
    fn compaction_replay_crash_child() -> Result<()> {
        let Ok(root) = crate::compatibility::var("WARM_TEST_CORE_REPLAY_ROOT") else {
            return Ok(());
        };
        let encrypted =
            crate::compatibility::var("WARM_TEST_CORE_REPLAY_ENCRYPTED").as_deref() == Ok("true");
        let config = configurations()
            .find(|config| config.security.encryption_enabled == encrypted)
            .unwrap();
        let core = WarmCore::new(1024, Some(root), Some(config))?;
        seed(&core)?;
        core.compact_impl(|| mutate(&core).unwrap(), || {}, false)
    }

    #[test]
    fn compaction_replayed_writes_survive_each_publication_interruption() -> Result<()> {
        for config in configurations() {
            for phase in ["staged", "published", "mapped"] {
                let directory = tempfile::tempdir()?;
                let output = std::process::Command::new(std::env::current_exe()?)
                    .args([
                        "--exact",
                        "core::engine::compaction::tests::compaction_replay_crash_child",
                    ])
                    .env("WARM_TEST_CORE_REPLAY_ROOT", directory.path())
                    .env(
                        "WARM_TEST_CORE_REPLAY_ENCRYPTED",
                        config.security.encryption_enabled.to_string(),
                    )
                    .env("WARM_TEST_COMPACTION_PHASE", phase)
                    .output()?;
                assert_eq!(output.status.code(), Some(73), "{phase}: {output:?}");
                let core = WarmCore::new(
                    1024,
                    Some(directory.path().display().to_string()),
                    Some(config.clone()),
                )?;
                assert_current(&core)?;
                core.memorize_document(document("after-recovery", "recovered"))?;
                core.compact()?;
            }
        }
        Ok(())
    }

    fn lexical(id: &str, version: &str, text: &str) -> DocumentInput {
        serde_json::from_value(json!({"id": id, "version": version, "text": text})).unwrap()
    }

    fn lexical_options() -> crate::core::documents::IngestOptions {
        crate::core::documents::IngestOptions {
            encode: crate::core::documents::EncodeMode::Lexical,
            ..Default::default()
        }
    }

    /// Hits of a few queries over every lexical fixture document, scores as bits.
    fn lexical_answers(core: &WarmCore) -> Result<Vec<String>> {
        let mut out = Vec::new();
        for query in ["alpha", "beta gamma", "delta", "after"] {
            for hit in core.search_documents(query, &SearchOptions::default())? {
                out.push(format!(
                    "{query}:{}:{}:{:x}",
                    hit.document_id,
                    hit.version,
                    hit.score.to_bits()
                ));
            }
        }
        Ok(out)
    }

    #[test]
    fn checkpoint_compaction_crash_child() -> Result<()> {
        let Ok(root) = crate::compatibility::var("WARM_TEST_CHECKPOINT_ROOT") else {
            return Ok(());
        };
        let encrypted =
            crate::compatibility::var("WARM_TEST_CHECKPOINT_ENCRYPTED").as_deref() == Ok("true");
        let mut config = configurations()
            .find(|config| config.security.encryption_enabled == encrypted)
            .unwrap();
        config.embedding_space = None;
        // The parent wrote the first checkpoint; this compaction replaces it and is cut short.
        let core = WarmCore::new(1024, Some(root), Some(config))?;
        let options = lexical_options();
        core.compact_impl(
            || {
                core.memorize_documents_with(vec![lexical("replace", "2", "beta beta")], &options)
                    .unwrap();
                core.delete_document("delete").unwrap();
            },
            || {},
            false,
        )
    }

    /// A checkpoint compaction interrupted at each publication step reopens to the state every
    /// acknowledged write reached, exactly as a store without checkpoints replays it.
    #[test]
    fn checkpoint_compaction_survives_each_publication_interruption() -> Result<()> {
        for config in configurations() {
            let mut config = config;
            config.embedding_space = None;
            let reference = {
                let directory = tempfile::tempdir()?;
                let mut plain = config.clone();
                plain.document_checkpoint.enabled = false;
                let core = WarmCore::new(
                    1024,
                    Some(directory.path().display().to_string()),
                    Some(plain),
                )?;
                let options = lexical_options();
                core.memorize_documents_with(
                    vec![
                        lexical("keep", "1", "alpha beta gamma"),
                        lexical("replace", "2", "beta beta"),
                    ],
                    &options,
                )?;
                lexical_answers(&core)?
            };
            for phase in ["staged", "published", "mapped"] {
                let directory = tempfile::tempdir()?;
                {
                    let core = WarmCore::new(
                        1024,
                        Some(directory.path().display().to_string()),
                        Some(config.clone()),
                    )?;
                    core.memorize_documents_with(
                        vec![
                            lexical("keep", "1", "alpha beta gamma"),
                            lexical("replace", "1", "alpha delta"),
                            lexical("delete", "1", "gamma delta delta"),
                        ],
                        &lexical_options(),
                    )?;
                    core.compact()?;
                }
                let output = std::process::Command::new(std::env::current_exe()?)
                    .args([
                        "--exact",
                        "core::engine::compaction::tests::checkpoint_compaction_crash_child",
                    ])
                    .env("WARM_TEST_CHECKPOINT_ROOT", directory.path())
                    .env(
                        "WARM_TEST_CHECKPOINT_ENCRYPTED",
                        config.security.encryption_enabled.to_string(),
                    )
                    .env("WARM_TEST_COMPACTION_PHASE", phase)
                    .output()?;
                assert_eq!(output.status.code(), Some(73), "{phase}: {output:?}");
                let generation = crate::core::durable_file::arena_generation(
                    &directory.path().join("vectors_data.bin"),
                )?;
                assert!(
                    generation.join(super::super::image::FILE).exists(),
                    "{phase}: the live generation holds a checkpoint"
                );
                let core = WarmCore::new(
                    1024,
                    Some(directory.path().display().to_string()),
                    Some(config.clone()),
                )?;
                assert_eq!(lexical_answers(&core)?, reference, "{phase}");
                assert!(!core.has_document("delete"), "{phase}");
            }
        }
        Ok(())
    }

    #[test]
    fn concurrent_compactors_serialize_without_losing_intervening_mutations() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().display().to_string();
        let core = Arc::new(WarmCore::new(1024, Some(path.clone()), Some(config()))?);
        seed(&core)?;
        let other_core = Arc::clone(&core);
        let (start_tx, start_rx) = mpsc::channel();
        let other = std::thread::spawn(move || -> Result<()> {
            start_rx.recv()?;
            other_core.compact()
        });
        core.compact_impl(
            || {
                start_tx.send(()).unwrap();
                assert!(core.compaction.try_lock().is_none());
                mutate(&core).unwrap();
            },
            || {},
            false,
        )?;
        other.join().unwrap()?;
        let expected = assert_current(&core)?;
        drop(core);
        let restored = WarmCore::new(1024, Some(path), Some(config()))?;
        assert_eq!(assert_current(&restored)?, expected);
        Ok(())
    }

    #[test]
    fn replay_rejects_partial_frames_and_accepts_exact_capture_boundaries() -> Result<()> {
        for delta in [-1_isize, 0, 1] {
            let directory = tempfile::tempdir()?;
            let core = WarmCore::new(1024, Some(directory.path().display().to_string()), None)?;
            let mut cursor = core.arena.stats().used_bytes;
            core.memorize("after".into(), core.encode_text("acknowledged"))?;
            let stage = tempfile::tempdir_in(directory.path())?;
            let staged = StagedArena::new(stage.path(), None)?.finish()?;
            let mut prepared = staged.prepare_compacted()?;
            let end = core
                .arena
                .stats()
                .used_bytes
                .checked_add_signed(delta)
                .unwrap();
            let result = core
                .arena
                .replay_compaction_tail(&mut prepared, &mut cursor, end);
            assert_eq!(result.is_ok(), delta == 0);
            assert!(!core.storage_path.join("vectors_data.bin/CURRENT").exists());
            assert_eq!(
                core.query(&core.encode_text("acknowledged"), 1)[0].id,
                "after"
            );
        }
        Ok(())
    }

    #[test]
    fn compaction_writes_after_catchup_survive_publication_and_restart() -> Result<()> {
        for config in configurations() {
            let directory = tempfile::tempdir()?;
            let path = directory.path().display().to_string();
            let core = WarmCore::new(1024, Some(path.clone()), Some(config.clone()))?;
            let initial = crate::EntangledHVec::new_deterministic(1024, 1);
            let during = crate::EntangledHVec::new_deterministic(1024, 2);
            let late = crate::EntangledHVec::new_deterministic(1024, 3);
            assert_ne!(initial.indices(), late.indices());
            assert_ne!(during.indices(), late.indices());
            core.memorize("vector".into(), initial)?;
            core.compact_impl(
                || core.memorize("vector".into(), during).unwrap(),
                || {
                    core.memorize("vector".into(), late.clone()).unwrap();
                    core.memorize_document(document("late", "late")).unwrap();
                },
                false,
            )?;
            let expected =
                serde_json::to_value(core.search_documents("needle", &Default::default())?)?;
            assert_eq!(expected[0]["version"], "late");
            drop(core);
            let restored = WarmCore::new(1024, Some(path), Some(config))?;
            assert_eq!(
                restored
                    .shards
                    .read()
                    .get_vector("vector")
                    .unwrap()
                    .indices(),
                late.indices()
            );
            assert_eq!(
                serde_json::to_value(restored.search_documents("needle", &Default::default())?)?,
                expected
            );
        }
        Ok(())
    }
}
