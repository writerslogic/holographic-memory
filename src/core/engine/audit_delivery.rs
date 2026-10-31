// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};

use super::{Mutation, WarmCore};
use crate::core::audit::AuditOp;
use crate::core::audit_effects::{AuditIntent, MAX_PENDING};

impl WarmCore {
    /// Number of committed audit records still awaiting durable acknowledgement.
    /// Legacy schemas have no recoverable audit queue and return zero.
    pub fn pending_audit_effects(&self) -> usize {
        self.audit_pending.read().len()
    }

    /// Whether this store schema supports recoverable audit delivery.
    pub fn durable_audit_delivery(&self) -> bool {
        self.audit_effects.is_some()
    }

    pub(super) fn audit_intent(&self, op: AuditOp, id: &str) -> Option<AuditIntent> {
        (self.config.security.audit_enabled && self.audit_effects.is_some())
            .then(|| AuditIntent::prepare(op, id, self.sign_fn().as_deref()))
    }

    pub(super) fn validate_audit_transitions(&self, mutations: &[Mutation]) -> Result<()> {
        if !mutations.iter().any(|mutation| {
            matches!(
                mutation,
                Mutation::AuditPending(_) | Mutation::AuditAcknowledged { .. }
            )
        }) {
            return Ok(());
        }
        let mut ids = self
            .audit_pending
            .read()
            .iter()
            .map(AuditIntent::id)
            .collect::<Result<std::collections::VecDeque<_>>>()?;
        for mutation in mutations {
            match mutation {
                Mutation::AuditPending(intent) => {
                    let id = intent.id()?;
                    ensure!(
                        ids.len() < MAX_PENDING,
                        "audit delivery backlog is full; flush audit effects before writing"
                    );
                    ensure!(
                        !ids.contains(&id),
                        "duplicate pending audit effect identity"
                    );
                    ids.push_back(id);
                }
                Mutation::AuditAcknowledged { id } => {
                    ensure!(
                        ids.front() == Some(id),
                        "audit acknowledgement is out of order"
                    );
                    ids.pop_front();
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn commit_audited(
        &self,
        mut mutations: Vec<Mutation>,
        op: AuditOp,
        id: &str,
    ) -> Result<()> {
        let intent = self.audit_intent(op, id);
        if let Some(intent) = intent {
            mutations.push(Mutation::AuditPending(intent));
        }
        crate::core::audit_effects::crash_point("before_commit");
        self.commit(&mutations)?;
        crate::core::audit_effects::crash_point("committed");
        self.finish_audit(op, id)
    }

    pub(super) fn finish_audit(&self, op: AuditOp, id: &str) -> Result<()> {
        if self.audit_effects.is_some() {
            crate::core::error::after_commit("audit delivery", self.flush_audit())
        } else if let Some(audit) = &self.audit {
            crate::core::error::after_commit(
                "audit append",
                audit.record(op, id, self.sign_fn().as_deref()),
            )
        } else {
            Ok(())
        }
    }

    /// Deliver committed audit records without repeating their business operations.
    /// Schema 6 persists these records with the mutation and resumes delivery on open.
    /// Earlier schemas have no recoverable delivery queue.
    pub fn flush_audit(&self) -> Result<()> {
        let Some(stream) = &self.audit_effects else {
            return Ok(());
        };
        let _delivery = self.audit_delivery.maintenance();
        let Some(target) = self
            .audit_pending
            .read()
            .back()
            .map(AuditIntent::id)
            .transpose()?
        else {
            return Ok(());
        };
        loop {
            let intent = self
                .audit_pending
                .read()
                .front()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("audit delivery queue lost its target"))?;
            let id = intent.id()?;
            stream.deliver(&intent)?;
            self.commit(&[Mutation::AuditAcknowledged { id }])?;
            crate::core::audit_effects::crash_point("acknowledged");
            if id == target {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::audit_effects::{AuditEffectStream, FILE_NAME};
    use crate::core::config::WarmConfig;
    use crate::EntangledHVec;
    use std::path::Path;

    fn open(path: &Path) -> Result<WarmCore> {
        let mut config = WarmConfig::default();
        config.security.audit_enabled = true;
        config.document_checkpoint.enabled = false;
        WarmCore::new(256, Some(path.display().to_string()), Some(config))
    }

    #[test]
    fn failed_delivery_survives_compaction_and_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut core = open(dir.path())?;
        core.audit_effects
            .as_mut()
            .unwrap()
            .make_read_only(dir.path())?;
        let vector = EntangledHVec::new_deterministic(256, 7);
        let error = core.memorize("durable".into(), vector.clone()).unwrap_err();
        assert!(error.is::<crate::CommittedMutationError>());
        let error = core.compact().unwrap_err();
        assert!(error.is::<crate::CommittedMutationError>());
        let ids = core
            .audit_pending
            .read()
            .iter()
            .map(AuditIntent::id)
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(ids.len(), 2);
        drop(core);
        let core = open(dir.path())?;
        assert_eq!(core.query(&vector, 10).unwrap()[0].id, "durable");
        let entries = core.audit_since(0)?;
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.effect_id.unwrap())
                .collect::<Vec<_>>(),
            ids
        );
        assert!(core.audit_pending.read().is_empty());
        core.flush_audit()?;
        drop(core);
        let core = open(dir.path())?;
        assert_eq!(core.audit_since(0)?, entries);
        Ok(())
    }

    #[test]
    fn delivered_without_acknowledgement_is_recovered_once() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let core = open(dir.path())?;
        let intent = core
            .audit_intent(AuditOp::Memorize, "unacknowledged")
            .unwrap();
        core.commit(&[
            Mutation::Vector {
                id: "unacknowledged".into(),
                vector: EntangledHVec::new_deterministic(256, 9),
            },
            Mutation::AuditPending(intent.clone()),
        ])?;
        core.audit_effects.as_ref().unwrap().deliver(&intent)?;
        drop(core);
        let core = open(dir.path())?;
        assert_eq!(core.audit_since(0)?, vec![intent.0]);
        assert!(core.audit_pending.read().is_empty());
        Ok(())
    }

    #[test]
    fn backlog_bound_rejects_business_write_before_commit() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let core = open(dir.path())?;
        let intents: Vec<_> = (0..MAX_PENDING - 1)
            .map(|n| {
                Mutation::AuditPending(core.audit_intent(AuditOp::Delete, &n.to_string()).unwrap())
            })
            .collect();
        core.commit(&intents)?;
        core.commit(&[Mutation::AuditPending(
            core.audit_intent(AuditOp::Delete, "last").unwrap(),
        )])?;
        let before = core.arena.checkpoint();
        let error = core
            .memorize(
                "must-not-commit".into(),
                EntangledHVec::new_deterministic(256, 1),
            )
            .unwrap_err();
        assert!(!error.is::<crate::CommittedMutationError>());
        assert!(error.to_string().contains("backlog is full"));
        assert_eq!(core.arena.checkpoint(), before);
        assert!(core.shards.read().get_vector("must-not-commit").is_none());
        Ok(())
    }

    #[test]
    fn missing_history_is_not_silently_recreated() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let core = open(dir.path())?;
        core.memorize("written".into(), EntangledHVec::new_deterministic(256, 7))?;
        drop(core);
        std::fs::remove_file(dir.path().join(FILE_NAME))?;
        assert!(open(dir.path())
            .err()
            .unwrap()
            .to_string()
            .contains("audit effect stream is missing"));
        Ok(())
    }

    #[test]
    fn missing_history_is_rejected_after_checkpoint_only_compaction() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut config = WarmConfig::default();
        config.security.audit_enabled = true;
        config.document_checkpoint.min_journal_bytes = usize::MAX;
        let path = dir.path().display().to_string();
        let core = WarmCore::new(256, Some(path.clone()), Some(config.clone()))?;
        core.memorize_document_with(
            serde_json::from_value(serde_json::json!({
                "id": "checkpoint", "text": "preserved document"
            }))?,
            &crate::core::documents::IngestOptions {
                encode: crate::core::documents::EncodeMode::Lexical,
                ..Default::default()
            },
        )?;
        core.compact()?;
        assert_eq!(core.audit_since(0)?.len(), 1);
        drop(core);
        config.security.audit_enabled = false;
        let core = WarmCore::new(256, Some(path.clone()), Some(config.clone()))?;
        core.compact()?;
        assert_eq!(core.arena.stats().used_bytes, 0);
        drop(core);
        std::fs::remove_file(dir.path().join(FILE_NAME))?;
        assert!(WarmCore::new(256, Some(path), Some(config))
            .err()
            .unwrap()
            .to_string()
            .contains("audit effect stream is missing"));
        Ok(())
    }

    #[test]
    fn initialized_empty_store_does_not_recreate_missing_history() -> Result<()> {
        let dir = tempfile::tempdir()?;
        drop(open(dir.path())?);
        std::fs::remove_file(dir.path().join(FILE_NAME))?;
        assert!(open(dir.path())
            .err()
            .unwrap()
            .to_string()
            .contains("audit effect stream is missing"));
        Ok(())
    }

    #[test]
    fn concurrent_delivery_preserves_commit_order() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let core = std::sync::Arc::new(open(dir.path())?);
        let start = std::sync::Arc::new(std::sync::Barrier::new(5));
        let threads: Vec<_> = (0..4)
            .map(|n| {
                let core = core.clone();
                let start = start.clone();
                std::thread::spawn(move || -> Result<()> {
                    start.wait();
                    core.memorize(
                        format!("writer-{n}"),
                        EntangledHVec::new_deterministic(256, n),
                    )?;
                    core.delete(&format!("writer-{n}"))?;
                    Ok(())
                })
            })
            .collect();
        start.wait();
        for thread in threads {
            thread.join().unwrap()?;
        }
        let entries = core.audit_since(0)?;
        assert_eq!(entries.len(), 8);
        for n in 0..4 {
            let hash = crate::core::security::hash_id(&format!("writer-{n}"));
            assert_eq!(
                entries
                    .iter()
                    .filter(|entry| entry.id_hash == hash)
                    .map(|entry| entry.op)
                    .collect::<Vec<_>>(),
                vec![AuditOp::Memorize, AuditOp::Delete]
            );
        }
        assert!(core.audit_pending.read().is_empty());
        drop(core);
        assert_eq!(open(dir.path())?.audit_since(0)?, entries);
        Ok(())
    }

    #[test]
    fn crash_child() -> Result<()> {
        let Ok(path) = std::env::var("WARM_AUDIT_CRASH_STORE") else {
            return Ok(());
        };
        let core = open(Path::new(&path))?;
        core.memorize(
            "crash-write".into(),
            EntangledHVec::new_deterministic(256, 5),
        )?;
        if std::env::var("WARM_AUDIT_CRASH_AT").ok().as_deref() == Some("compacted") {
            core.compact()?;
        }
        panic!("crash point was not reached");
    }

    #[test]
    fn process_crashes_at_commit_write_sync_and_ack_recover_once() -> Result<()> {
        for phase in [
            "initialized",
            "before_commit",
            "committed",
            "written",
            "synced",
            "acknowledged",
            "compacted",
        ] {
            let dir = tempfile::tempdir()?;
            let status = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "core::engine::audit_delivery::tests::crash_child",
                    "--nocapture",
                ])
                .env("WARM_AUDIT_CRASH_STORE", dir.path())
                .env("WARM_AUDIT_CRASH_AT", phase)
                .output()?;
            assert_eq!(
                status.status.code(),
                Some(85),
                "{phase}: {}",
                String::from_utf8_lossy(&status.stderr)
            );
            for _ in 0..2 {
                let core = open(dir.path())?;
                assert_eq!(
                    core.shards.read().get_vector("crash-write").is_some(),
                    !matches!(phase, "before_commit" | "initialized")
                );
                let expected = match phase {
                    "before_commit" | "initialized" => 0,
                    "compacted" => 2,
                    _ => 1,
                };
                assert_eq!(core.audit_since(0)?.len(), expected, "{phase}");
                assert!(core.audit_pending.read().is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn repaired_stream_can_flush_without_repeating_the_mutation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut core = open(dir.path())?;
        core.audit_effects
            .as_mut()
            .unwrap()
            .make_read_only(dir.path())?;
        assert!(core
            .memorize("once".into(), EntangledHVec::new_deterministic(256, 1))
            .is_err());
        let pending = core.audit_pending.read().front().unwrap().clone();
        core.audit_effects = Some(AuditEffectStream::open(dir.path())?);
        core.flush_audit()?;
        assert_eq!(core.audit_since(0)?, vec![pending.0]);
        assert_eq!(core.shards.read().count(), 1);
        Ok(())
    }

    #[test]
    fn pending_effects_survive_document_checkpoint_without_hydration() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut config = WarmConfig::default();
        config.security.audit_enabled = true;
        config.document_checkpoint.min_journal_bytes = usize::MAX;
        let mut core = WarmCore::new(
            256,
            Some(dir.path().display().to_string()),
            Some(config.clone()),
        )?;
        let document = serde_json::from_value(serde_json::json!({
            "id": "lexical", "version": "1", "text": "durable recovery evidence"
        }))?;
        core.memorize_document_with(
            document,
            &crate::core::documents::IngestOptions {
                encode: crate::core::documents::EncodeMode::Lexical,
                ..Default::default()
            },
        )?;
        core.audit_effects
            .as_mut()
            .unwrap()
            .make_read_only(dir.path())?;
        assert!(core
            .compact()
            .unwrap_err()
            .is::<crate::CommittedMutationError>());
        assert_eq!(core.audit_pending.read().len(), 1);
        drop(core);
        let core = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config))?;
        assert!(core.serves_from_checkpoint());
        let hits = core.search_documents("recovery", &Default::default())?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document_id, "lexical");
        assert_eq!(core.audit_since(0)?.len(), 1);
        assert!(core.audit_pending.read().is_empty());
        assert!(core.serves_from_checkpoint());
        Ok(())
    }

    #[test]
    fn verified_snapshot_preserves_pending_delivery_and_source_state() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let source = dir.path().join("source");
        let snapshot = dir.path().join("snapshot");
        let core = open(&source)?;
        let intent = core
            .audit_intent(AuditOp::Delete, "snapshot-effect")
            .unwrap();
        core.commit(&[Mutation::AuditPending(intent.clone())])?;
        drop(core);
        let history = std::fs::read(source.join(FILE_NAME))?;
        let report = crate::core::format::verify_path(&source, &Default::default())?;
        assert_eq!(
            report.verdict,
            crate::core::format::Verdict::Ok,
            "{:?}",
            report.findings
        );
        crate::core::admin::migrate_store(&source, &snapshot)?;
        assert_eq!(std::fs::read(source.join(FILE_NAME))?, history);
        assert_eq!(std::fs::read(snapshot.join(FILE_NAME))?, history);
        let restored = open(&snapshot)?;
        assert_eq!(restored.audit_since(0)?, vec![intent.0.clone()]);
        assert_eq!(restored.pending_audit_effects(), 0);
        assert_eq!(std::fs::read(source.join(FILE_NAME))?, history);
        assert_eq!(open(&source)?.audit_since(0)?, vec![intent.0]);
        Ok(())
    }

    #[cfg(feature = "security")]
    #[test]
    fn recovery_preserves_the_original_signature_after_key_change() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut config = WarmConfig::default();
        config.security.audit_enabled = true;
        config.security.signing_enabled = true;
        config.document_checkpoint.enabled = false;
        let core = WarmCore::new(
            256,
            Some(dir.path().display().to_string()),
            Some(config.clone()),
        )?;
        let old_key =
            crate::core::security::SigningManager::new(&dir.path().join("hms_signing.key"))?;
        let intent = core
            .audit_intent(AuditOp::Delete, "original-signer")
            .unwrap();
        core.commit(&[Mutation::AuditPending(intent.clone())])?;
        drop(core);
        config.security.key_path = Some(dir.path().join("replacement.key").display().to_string());
        let core = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config))?;
        let entries = core.audit_since(0)?;
        assert_eq!(entries, vec![intent.0]);
        old_key.verify(&entries[0].signable_bytes(), &entries[0].signature)?;
        let replacement =
            crate::core::security::SigningManager::new(&dir.path().join("replacement.key"))?;
        assert!(replacement
            .verify(&entries[0].signable_bytes(), &entries[0].signature)
            .is_err());
        Ok(())
    }
}
