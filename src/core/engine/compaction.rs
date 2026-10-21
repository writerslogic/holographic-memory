// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{ensure, Result};
use serde::Serialize;

use super::{mutation::Mutation, HmsCore};
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
}

#[derive(Serialize)]
enum DocumentMutation<'a> {
    Document(crate::core::documents::DocumentWithText<'a>),
}

impl HmsCore {
    pub fn compact(&self) -> Result<()> {
        self.compact_with_hook(|| {})
    }

    /// Compacts and then physically erases what the compaction replaced: the old generation
    /// and any unreferenced leftovers are overwritten with zeros and synced before they are
    /// unlinked. The text segment is scrubbed per deleted document at delete time.
    pub fn compact_scrubbing(&self) -> Result<()> {
        self.sync_text_segment()?;
        self.compact_impl(|| {}, || {}, true)
    }

    fn compact_with_hook(&self, after_snapshot: impl FnOnce()) -> Result<()> {
        self.compact_with_hooks(after_snapshot, || {})
    }

    fn compact_with_hooks(
        &self,
        after_snapshot: impl FnOnce(),
        before_publication: impl FnOnce(),
    ) -> Result<()> {
        self.compact_impl(after_snapshot, before_publication, false)
    }

    fn compact_impl(
        &self,
        after_snapshot: impl FnOnce(),
        before_publication: impl FnOnce(),
        scrub: bool,
    ) -> Result<()> {
        let _compaction = self.compaction.lock();
        let staging = StagingDir::new(self.arena.new_generation_dir()?);
        let snapshot = self.compaction_snapshot()?;
        let (generation, mut cursor) = snapshot.checkpoint;
        after_snapshot();
        let mut arena = StagedArena::new(staging.path(), Some(&self.arena))?;
        for mutation in snapshot.mutations {
            self.write_compaction_frame(&mut arena, &[mutation])?;
        }
        let mut payload = Vec::new();
        for document in &snapshot.documents {
            // A document deleted since the snapshot has had its text zeroed; the delete is in
            // the tail, which the replay below applies, so it is simply not carried over.
            {
                let store = self.documents.read();
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
        staging.disarm();
        if scrub {
            self.arena.replace_with_compacted_scrubbing(prepared)?;
        } else {
            self.arena.replace_with_compacted(prepared)?;
        }
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        // The arena pointer has moved, so writers may continue: the lease ends here. Index
        // persistence binds the checkpoint read under the lease, holds the shards only while
        // encoding, and writes the files with no lock held; the persist call that follows
        // then finds identical metadata and writes nothing.
        let checkpoint = self.arena.checkpoint();
        _transaction.release();
        self.persist_indices_detached(checkpoint)?;
        self.persist_indices(&self.shards.read())?;
        if let Some(audit) = &self.audit {
            audit.record(
                crate::core::audit::AuditOp::Compact,
                "",
                self.sign_fn().as_deref(),
            )?;
        }
        Ok(())
    }

    fn compaction_snapshot(&self) -> Result<Snapshot> {
        let _transaction = self.mutation_gate.maintenance();
        let shards = self.shards.read();
        let mut mutations = Vec::new();
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
                    mutations.push(Mutation::Vector {
                        id: id.clone(),
                        vector: vector.clone(),
                    });
                }
            }
            Ok(())
        })?;
        let (documents, text) = {
            let store = self.documents.read();
            (store.shared_values().collect(), store.text_segment())
        };
        mutations.extend(self.graph.snapshot().into_iter().map(Mutation::Relation));
        mutations.extend(
            self.graph
                .type_snapshot()
                .into_iter()
                .map(Mutation::RelationType),
        );
        if let Some(atoms) = &self.atom_memory {
            mutations.extend(
                atoms
                    .inner()
                    .all_vectors()
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
        Ok(Snapshot {
            checkpoint: self.arena.checkpoint(),
            mutations,
            documents,
            text,
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
    use crate::core::{EmbeddingSpace, HmsConfig};
    use serde_json::json;

    fn config() -> HmsConfig {
        HmsConfig {
            embedding_space: Some(EmbeddingSpace {
                model: "fixture".into(),
                revision: "1".into(),
                dimensions: 2,
                normalization: "l2".into(),
                metric: "cosine".into(),
            }),
            ..Default::default()
        }
    }

    fn configurations() -> impl Iterator<Item = HmsConfig> {
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

    fn seed(core: &HmsCore) -> Result<()> {
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

    fn mutate(core: &HmsCore) -> Result<()> {
        core.memorize_documents(vec![document("replace", "new"), document("insert", "new")])?;
        core.delete_document("delete")?;
        core.memorize("vector".into(), core.encode_text("replacement"))?;
        core.delete("remove")?;
        core.remove_relation("a", "knows", "b")?;
        Ok(())
    }

    fn assert_current(core: &HmsCore) -> Result<serde_json::Value> {
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
            let core = Arc::new(HmsCore::new(
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
            let compacted = core.compact_with_hook(|| {
                start_tx.send(()).unwrap();
                done_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("compaction snapshot blocked mutations or queries");
            });
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
            let restored = HmsCore::new(1024, Some(path), Some(config))?;
            assert_eq!(assert_current(&restored)?, expected);
        }
        Ok(())
    }

    #[test]
    fn compaction_snapshot_shares_document_payloads_after_replacement() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let core = HmsCore::new(
            1024,
            Some(directory.path().display().to_string()),
            Some(config()),
        )?;
        core.memorize_document(document("replace", "old"))?;
        let snapshot = core.compaction_snapshot()?;
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
        let Ok(root) = std::env::var("HMS_TEST_CORE_REPLAY_ROOT") else {
            return Ok(());
        };
        let encrypted = std::env::var("HMS_TEST_CORE_REPLAY_ENCRYPTED").as_deref() == Ok("true");
        let config = configurations()
            .find(|config| config.security.encryption_enabled == encrypted)
            .unwrap();
        let core = HmsCore::new(1024, Some(root), Some(config))?;
        seed(&core)?;
        core.compact_with_hook(|| mutate(&core).unwrap())
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
                    .env("HMS_TEST_CORE_REPLAY_ROOT", directory.path())
                    .env(
                        "HMS_TEST_CORE_REPLAY_ENCRYPTED",
                        config.security.encryption_enabled.to_string(),
                    )
                    .env("HMS_TEST_COMPACTION_PHASE", phase)
                    .output()?;
                assert_eq!(output.status.code(), Some(73), "{phase}: {output:?}");
                let core = HmsCore::new(
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

    #[test]
    fn concurrent_compactors_serialize_without_losing_intervening_mutations() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().display().to_string();
        let core = Arc::new(HmsCore::new(1024, Some(path.clone()), Some(config()))?);
        seed(&core)?;
        let other_core = Arc::clone(&core);
        let (start_tx, start_rx) = mpsc::channel();
        let other = std::thread::spawn(move || -> Result<()> {
            start_rx.recv()?;
            other_core.compact()
        });
        core.compact_with_hook(|| {
            start_tx.send(()).unwrap();
            assert!(core.compaction.try_lock().is_none());
            mutate(&core).unwrap();
        })?;
        other.join().unwrap()?;
        let expected = assert_current(&core)?;
        drop(core);
        let restored = HmsCore::new(1024, Some(path), Some(config()))?;
        assert_eq!(assert_current(&restored)?, expected);
        Ok(())
    }

    #[test]
    fn replay_rejects_partial_frames_and_accepts_exact_capture_boundaries() -> Result<()> {
        for delta in [-1_isize, 0, 1] {
            let directory = tempfile::tempdir()?;
            let core = HmsCore::new(1024, Some(directory.path().display().to_string()), None)?;
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
            let core = HmsCore::new(1024, Some(path.clone()), Some(config.clone()))?;
            let initial = crate::EntangledHVec::new_deterministic(1024, 1);
            let during = crate::EntangledHVec::new_deterministic(1024, 2);
            let late = crate::EntangledHVec::new_deterministic(1024, 3);
            assert_ne!(initial.indices(), late.indices());
            assert_ne!(during.indices(), late.indices());
            core.memorize("vector".into(), initial)?;
            core.compact_with_hooks(
                || core.memorize("vector".into(), during).unwrap(),
                || {
                    core.memorize("vector".into(), late.clone()).unwrap();
                    core.memorize_document(document("late", "late")).unwrap();
                },
            )?;
            let expected =
                serde_json::to_value(core.search_documents("needle", &Default::default())?)?;
            assert_eq!(expected[0]["version"], "late");
            drop(core);
            let restored = HmsCore::new(1024, Some(path), Some(config))?;
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
