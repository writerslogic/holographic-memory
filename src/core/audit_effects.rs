// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use super::audit::{AuditEntry, AuditOp, SignFnRef, ENTRY_SIZE};
use super::durable_file::{
    atomic_create_regular, open_regular, open_regular_read_write, positioned_read,
    positioned_write, sync_mapped_file,
};

pub(crate) const MAX_PENDING: usize = 1024;
pub(crate) const FILE_NAME: &str = "audit.effects.bin";
pub(crate) const MAGIC: &[u8; 8] = b"WARMAU02";
pub(crate) const RECORD_SIZE: usize = 16 + ENTRY_SIZE + 4;

pub(crate) fn crash_point(phase: &str) {
    #[cfg(test)]
    if std::env::var("WARM_AUDIT_CRASH_AT").ok().as_deref() == Some(phase) {
        std::process::exit(85);
    }
    let _ = phase;
}

pub(crate) fn verify(path: &Path) -> Result<u64> {
    inspect(&open_regular(path, false)?)
}

fn inspect(file: &File) -> Result<u64> {
    let mut magic = [0; MAGIC.len()];
    positioned_read(file, &mut magic, 0).context("reading audit effect header")?;
    ensure!(&magic == MAGIC, "unsupported audit effect format");
    let length = file.metadata()?.len();
    ensure!(
        length >= MAGIC.len() as u64,
        "truncated audit effect header"
    );
    let trailing = (length - MAGIC.len() as u64) % RECORD_SIZE as u64;
    let mut offset = MAGIC.len() as u64;
    let mut scratch = vec![0; 512 * RECORD_SIZE];
    while offset < length - trailing {
        let count = (length - trailing - offset).min(scratch.len() as u64) as usize;
        positioned_read(file, &mut scratch[..count], offset)?;
        for (index, bytes) in scratch[..count]
            .as_chunks::<RECORD_SIZE>()
            .0
            .iter()
            .enumerate()
        {
            AuditIntent::decode(bytes).with_context(|| {
                format!(
                    "audit effect at offset {}",
                    offset + (index * RECORD_SIZE) as u64
                )
            })?;
        }
        offset += count as u64;
    }
    Ok(trailing)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct AuditIntent(pub AuditEntry);

impl AuditIntent {
    pub(crate) fn prepare(op: AuditOp, id: &str, sign: SignFnRef<'_>) -> Self {
        Self(AuditEntry::prepare(op, id, Some(rand::random()), sign))
    }

    pub(crate) fn id(&self) -> Result<[u8; 16]> {
        self.0
            .effect_id
            .context("audit intent has no effect identity")
    }

    fn encode(&self) -> Result<[u8; RECORD_SIZE]> {
        let mut bytes = [0; RECORD_SIZE];
        bytes[..16].copy_from_slice(&self.id()?);
        bytes[16..16 + ENTRY_SIZE].copy_from_slice(&self.0.to_bytes());
        let checksum = crc32fast::hash(&bytes[..RECORD_SIZE - 4]);
        bytes[RECORD_SIZE - 4..].copy_from_slice(&checksum.to_le_bytes());
        Ok(bytes)
    }

    pub(crate) fn decode(bytes: &[u8; RECORD_SIZE]) -> Result<Self> {
        let expected = u32::from_le_bytes(bytes[RECORD_SIZE - 4..].try_into()?);
        ensure!(
            crc32fast::hash(&bytes[..RECORD_SIZE - 4]) == expected,
            "audit effect checksum mismatch"
        );
        let mut entry = AuditEntry::from_bytes(bytes[16..16 + ENTRY_SIZE].try_into()?)
            .context("audit effect has an invalid operation")?;
        entry.effect_id = Some(bytes[..16].try_into()?);
        Ok(Self(entry))
    }
}

/// Delivery requires the engine's ordered audit-delivery lease. Readers only observe synced bytes.
pub(crate) struct AuditEffectStream {
    file: File,
    published_len: AtomicU64,
}

impl AuditEffectStream {
    #[cfg(test)]
    pub(crate) fn make_read_only(&mut self, store: &Path) -> Result<()> {
        self.file = open_regular(&store.join(FILE_NAME), false)?;
        Ok(())
    }
    pub(crate) fn open(store: &Path) -> Result<Self> {
        let path = store.join(FILE_NAME);
        let file = match open_regular_read_write(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                atomic_create_regular(&path, MAGIC)?;
                open_regular_read_write(&path)?
            }
            Err(error) => return Err(error.into()),
        };
        inspect(&file)?;
        let stream = Self {
            file,
            published_len: AtomicU64::new(MAGIC.len() as u64),
        };
        let length = stream.repair_tail()?;
        stream.last(length)?;
        stream.published_len.store(length, Ordering::Release);
        Ok(stream)
    }

    fn sync(&self) -> Result<()> {
        self.file.sync_data().context("syncing audit effects")?;
        sync_mapped_file(&self.file).context("syncing audit effects to storage")?;
        Ok(())
    }

    fn repair_tail(&self) -> Result<u64> {
        let length = self.file.metadata()?.len();
        ensure!(
            length >= self.published_len.load(Ordering::Acquire),
            "audit effect history was truncated while open"
        );
        let whole = length - (length - MAGIC.len() as u64) % RECORD_SIZE as u64;
        if whole != length {
            self.file
                .set_len(whole)
                .context("repairing audit effect tail")?;
            self.sync()?;
        }
        Ok(whole)
    }

    fn last(&self, length: u64) -> Result<Option<AuditIntent>> {
        if length == MAGIC.len() as u64 {
            return Ok(None);
        }
        let mut bytes = [0; RECORD_SIZE];
        positioned_read(&self.file, &mut bytes, length - RECORD_SIZE as u64)?;
        AuditIntent::decode(&bytes).map(Some)
    }

    /// The drainer stops before another delivery if acknowledgement fails, so only the last record can be pending.
    pub(crate) fn deliver(&self, intent: &AuditIntent) -> Result<()> {
        let bytes = intent.encode()?;
        let length = self.repair_tail()?;
        if let Some(last) = self.last(length)? {
            if last.id()? == intent.id()? {
                ensure!(
                    last == *intent,
                    "audit effect identity has conflicting payloads"
                );
                self.sync()?;
                self.published_len.store(length, Ordering::Release);
                return Ok(());
            }
        }
        let next = length
            .checked_add(RECORD_SIZE as u64)
            .context("audit effect file length overflow")?;
        positioned_write(&self.file, &bytes, length).context("appending audit effect")?;
        crash_point("written");
        self.sync()?;
        crash_point("synced");
        self.published_len.store(next, Ordering::Release);
        Ok(())
    }

    pub(crate) fn entries_since(&self, timestamp_ms: u64) -> Result<Vec<AuditEntry>> {
        let length = self.published_len.load(Ordering::Acquire);
        let mut at = MAGIC.len() as u64;
        let mut scratch = vec![0; 512 * RECORD_SIZE];
        let mut entries = Vec::new();
        while at < length {
            let count = (length - at).min(scratch.len() as u64) as usize;
            positioned_read(&self.file, &mut scratch[..count], at)?;
            for (index, bytes) in scratch[..count]
                .as_chunks::<RECORD_SIZE>()
                .0
                .iter()
                .enumerate()
            {
                let intent = AuditIntent::decode(bytes).with_context(|| {
                    format!(
                        "audit effect at offset {}",
                        at + (index * RECORD_SIZE) as u64
                    )
                })?;
                if intent.0.timestamp_ms >= timestamp_ms {
                    entries.push(intent.0);
                }
            }
            at += count as u64;
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_delivery_preserves_identity_across_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let intent = AuditIntent::prepare(AuditOp::Memorize, "document", None);
        for _ in 0..3 {
            let stream = AuditEffectStream::open(dir.path())?;
            stream.deliver(&intent)?;
            assert_eq!(stream.entries_since(0)?, vec![intent.0.clone()]);
            assert_eq!(
                stream.file.metadata()?.len(),
                (MAGIC.len() + RECORD_SIZE) as u64
            );
        }
        Ok(())
    }

    #[test]
    fn identical_legacy_payloads_with_distinct_effects_are_not_deduplicated() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let stream = AuditEffectStream::open(dir.path())?;
        let first = AuditIntent::prepare(AuditOp::Delete, "same", None);
        let mut second = first.clone();
        second.0.effect_id = Some([0x55; 16]);
        assert_ne!(first.id()?, second.id()?);
        assert_eq!(first.0.to_bytes(), second.0.to_bytes());
        stream.deliver(&first)?;
        stream.deliver(&second)?;
        assert_eq!(stream.entries_since(0)?.len(), 2);
        Ok(())
    }

    #[test]
    fn conflicting_identity_and_complete_corruption_are_errors() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let stream = AuditEffectStream::open(dir.path())?;
        let original = AuditIntent::prepare(AuditOp::Memorize, "original", None);
        stream.deliver(&original)?;
        let mut conflict = original.clone();
        conflict.0.id_hash[0] ^= 1;
        assert!(stream.deliver(&conflict).is_err());
        let mut bytes = original.encode()?;
        bytes[16] ^= 1;
        positioned_write(&stream.file, &bytes, MAGIC.len() as u64)?;
        assert!(stream.entries_since(u64::MAX).is_err());
        drop(stream);
        assert!(AuditEffectStream::open(dir.path()).is_err());
        Ok(())
    }

    #[test]
    fn partial_delivery_is_repaired_and_complete_delivery_is_retained() -> Result<()> {
        for length in [1, RECORD_SIZE - 1, RECORD_SIZE, RECORD_SIZE + 1] {
            let dir = tempfile::tempdir()?;
            let stream = AuditEffectStream::open(dir.path())?;
            let intent = AuditIntent::prepare(AuditOp::Compact, "", None);
            let bytes = intent.encode()?.repeat(2);
            positioned_write(&stream.file, &bytes[..length], MAGIC.len() as u64)?;
            stream.deliver(&intent)?;
            assert_eq!(stream.entries_since(0)?, vec![intent.0]);
            assert_eq!(
                stream.file.metadata()?.len(),
                (MAGIC.len() + RECORD_SIZE) as u64
            );
        }
        Ok(())
    }

    #[test]
    fn intent_deserialization_bounds_signature_shape() -> Result<()> {
        let intent = AuditIntent::prepare(AuditOp::Memorize, "document", None);
        let value = serde_json::to_value(&intent)?;
        assert_eq!(
            serde_json::from_value::<AuditIntent>(value.clone())?,
            intent
        );
        for length in [31, 33] {
            let mut malformed = value.clone();
            malformed["signature"][0] = serde_json::json!(vec![0u8; length]);
            assert!(serde_json::from_value::<AuditIntent>(malformed).is_err());
        }
        Ok(())
    }

    #[test]
    fn corruption_before_the_last_record_is_not_hidden_by_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let stream = AuditEffectStream::open(dir.path())?;
        let intent = AuditIntent::prepare(AuditOp::Memorize, "first", None);
        stream.deliver(&intent)?;
        stream.deliver(&AuditIntent::prepare(AuditOp::Delete, "second", None))?;
        let mut bytes = intent.encode()?;
        bytes[16] ^= 1;
        positioned_write(&stream.file, &bytes, MAGIC.len() as u64)?;
        drop(stream);
        assert!(AuditEffectStream::open(dir.path()).is_err());
        assert!(verify(&dir.path().join(FILE_NAME)).is_err());
        Ok(())
    }

    #[test]
    fn filtered_queries_preserve_records_across_buffer_boundaries() -> Result<()> {
        for count in [511usize, 512, 513] {
            let dir = tempfile::tempdir()?;
            let mut bytes = MAGIC.to_vec();
            for index in 0..count {
                let mut intent = AuditIntent::prepare(AuditOp::Memorize, "record", None);
                intent.0.timestamp_ms = index as u64;
                bytes.extend_from_slice(&intent.encode()?);
            }
            std::fs::write(dir.path().join(FILE_NAME), bytes)?;
            let stream = AuditEffectStream::open(dir.path())?;
            let entries = stream.entries_since(count as u64 - 2)?;
            assert_eq!(
                entries
                    .iter()
                    .map(|entry| entry.timestamp_ms)
                    .collect::<Vec<_>>(),
                vec![count as u64 - 2, count as u64 - 1]
            );
        }
        Ok(())
    }

    #[cfg(feature = "security")]
    #[test]
    fn signature_binds_delivery_identity_and_original_payload() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let signer = crate::core::security::SigningManager::new(&dir.path().join("signing.key"))?;
        let intent = AuditIntent::prepare(
            AuditOp::Memorize,
            "signed",
            Some(&|message| signer.sign(message)),
        );
        let message = intent.0.signable_bytes();
        assert!(message.starts_with(b"WARM-AUDIT-2\0"));
        assert_eq!(&message[13..29], intent.id()?.as_slice());
        signer.verify(&message, &intent.0.signature)?;
        let mut altered = intent.clone();
        altered.0.effect_id.as_mut().unwrap()[0] ^= 1;
        assert!(signer
            .verify(&altered.0.signable_bytes(), &altered.0.signature)
            .is_err());
        altered = intent.clone();
        altered.0.op = AuditOp::Delete;
        assert!(signer
            .verify(&altered.0.signable_bytes(), &altered.0.signature)
            .is_err());
        Ok(())
    }
}
