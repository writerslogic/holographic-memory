// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::sync::Mutex;
use anyhow::{anyhow, Result};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

use super::security::hash_id;

pub type SignatureBytes = [u8; 64];
pub type SignFnRef<'a> = Option<&'a dyn Fn(&[u8]) -> SignatureBytes>;

/// Operation types for audit entries.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditOp {
    Memorize = 1,
    Delete = 2,
    Compact = 3,
}

impl AuditOp {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Memorize),
            2 => Some(Self::Delete),
            3 => Some(Self::Compact),
            _ => None,
        }
    }
}

/// A single audit log entry.
/// Fixed layout: [timestamp_ms: u64][op: u8][id_hash: 32][sig: 64] = 105 bytes.
/// When signing is disabled, sig is all zeros.
#[derive(Clone, Debug)]
pub struct AuditEntry {
    pub timestamp_ms: u64,
    pub op: AuditOp,
    pub id_hash: [u8; 32],
    pub signature: [u8; 64],
}

const ENTRY_SIZE: usize = 8 + 1 + 32 + 64; // 105 bytes

impl AuditEntry {
    fn to_bytes(&self) -> [u8; ENTRY_SIZE] {
        let mut buf = [0u8; ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        buf[8] = self.op as u8;
        buf[9..41].copy_from_slice(&self.id_hash);
        buf[41..105].copy_from_slice(&self.signature);
        buf
    }

    fn from_bytes(buf: &[u8; ENTRY_SIZE]) -> Option<Self> {
        let timestamp_ms = u64::from_le_bytes(buf[0..8].try_into().expect("8-byte slice"));
        let op = AuditOp::from_u8(buf[8])?;
        let mut id_hash = [0u8; 32];
        id_hash.copy_from_slice(&buf[9..41]);
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&buf[41..105]);
        Some(Self {
            timestamp_ms,
            op,
            id_hash,
            signature,
        })
    }
}

/// Append-only audit log stored at `{storage_path}/audit.bin`.
pub struct AuditLog {
    file: Mutex<File>,
}

impl AuditLog {
    pub fn new(storage_path: &std::path::Path) -> Result<Self> {
        let path = storage_path.join("audit.bin");
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)?;
        Self::repair_torn_tail(&file)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    fn repair_torn_tail(file: &File) -> Result<()> {
        let len = file.metadata()?.len();
        let whole = len - len % ENTRY_SIZE as u64;
        if whole != len {
            file.set_len(whole)
                .map_err(|e| anyhow!("Audit tail repair failed: {e}"))?;
            file.sync_data()
                .map_err(|e| anyhow!("Audit tail repair sync failed: {e}"))?;
        }
        Ok(())
    }

    /// Record an operation. The `sign_fn` is called with the entry's
    /// signable bytes (timestamp + op + id_hash = 41 bytes) and should
    /// return a 64-byte signature, or None if signing is disabled.
    pub fn record(&self, op: AuditOp, id: &str, sign_fn: SignFnRef<'_>) -> Result<()> {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut bytes = AuditEntry {
            timestamp_ms,
            op,
            id_hash: hash_id(id),
            signature: [0u8; 64],
        }
        .to_bytes();

        if let Some(sign) = sign_fn {
            let signature = sign(&bytes[..41]);
            bytes[41..].copy_from_slice(&signature);
        }

        let mut file = self.file.lock();
        // A prior failed write can leave a partial record without closing this handle.
        Self::repair_torn_tail(&file)?;
        file.write_all(&bytes)
            .map_err(|e| anyhow!("Audit write failed: {}", e))?;
        file.sync_data()
            .map_err(|e| anyhow!("Audit sync failed: {}", e))?;
        Ok(())
    }

    /// Read all entries with timestamp >= `since_ms`.
    /// Reject malformed complete records even outside the requested time range.
    pub fn entries_since(&self, since_ms: u64) -> Result<Vec<AuditEntry>> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(0))?;

        let metadata = file.metadata()?;
        let file_len = usize::try_from(metadata.len())
            .map_err(|_| anyhow!("audit log is too large to read into memory"))?;
        if file_len == 0 {
            return Ok(Vec::new());
        }

        let entry_count = file_len / ENTRY_SIZE;
        let mut remaining = entry_count * ENTRY_SIZE;
        let mut buf = vec![0u8; remaining.min((64 * 1024 / ENTRY_SIZE) * ENTRY_SIZE)];
        let mut entries = Vec::new();
        while remaining > 0 {
            let read = remaining.min(buf.len());
            file.read_exact(&mut buf[..read]).map_err(|e| {
                anyhow!(
                    "Audit read failed at offset {}: {e}",
                    entry_count * ENTRY_SIZE - remaining
                )
            })?;
            for (index, chunk) in buf[..read].as_chunks::<ENTRY_SIZE>().0.iter().enumerate() {
                let entry = AuditEntry::from_bytes(chunk).ok_or_else(|| {
                    anyhow!(
                        "Audit record has invalid operation at offset {}",
                        entry_count * ENTRY_SIZE - remaining + index * ENTRY_SIZE
                    )
                })?;
                if entry.timestamp_ms >= since_ms {
                    entries.push(entry);
                }
            }
            remaining -= read;
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_audit_record_and_read() {
        let dir = tempdir().unwrap();
        let log = AuditLog::new(dir.path()).unwrap();

        log.record(AuditOp::Memorize, "doc_1", None).unwrap();
        log.record(AuditOp::Delete, "doc_2", None).unwrap();
        log.record(AuditOp::Compact, "", None).unwrap();

        let entries = log.entries_since(0).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].op, AuditOp::Memorize);
        assert_eq!(entries[1].op, AuditOp::Delete);
        assert_eq!(entries[2].op, AuditOp::Compact);
    }

    #[test]
    fn test_audit_since_filter() {
        let dir = tempdir().unwrap();
        let log = AuditLog::new(dir.path()).unwrap();

        let now_ms = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
        };
        log.record(AuditOp::Memorize, "a", None).unwrap();
        let first = log.entries_since(0).unwrap()[0].timestamp_ms;
        // The second entry must land in a later millisecond than the first.
        while now_ms() <= first {
            std::hint::spin_loop();
        }
        log.record(AuditOp::Delete, "b", None).unwrap();

        assert_eq!(log.entries_since(first).unwrap().len(), 2);
        let entries = log.entries_since(first + 1).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].op, AuditOp::Delete);
    }

    #[test]
    fn test_audit_entry_roundtrip() {
        let entry = AuditEntry {
            timestamp_ms: 1234567890,
            op: AuditOp::Memorize,
            id_hash: [0xAB; 32],
            signature: [0xCD; 64],
        };
        let bytes = entry.to_bytes();
        let parsed = AuditEntry::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.timestamp_ms, entry.timestamp_ms);
        assert_eq!(parsed.op, entry.op);
        assert_eq!(parsed.id_hash, entry.id_hash);
        assert_eq!(parsed.signature, entry.signature);
    }

    #[test]
    fn test_audit_persistence() {
        let dir = tempdir().unwrap();

        {
            let log = AuditLog::new(dir.path()).unwrap();
            log.record(AuditOp::Memorize, "persist_test", None).unwrap();
        }

        let log = AuditLog::new(dir.path()).unwrap();
        let entries = log.entries_since(0).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].op, AuditOp::Memorize);
        assert_eq!(entries[0].id_hash, hash_id("persist_test"));
    }

    #[test]
    fn malformed_operations_are_reported_even_outside_the_timestamp_filter() {
        for index in [623, 624, 625] {
            for op in [0, 4, u8::MAX] {
                let dir = tempdir().unwrap();
                let log = AuditLog::new(dir.path()).unwrap();
                let entry = AuditEntry {
                    timestamp_ms: 1,
                    op: AuditOp::Memorize,
                    id_hash: [0; 32],
                    signature: [0; 64],
                }
                .to_bytes();
                let mut bytes = entry.repeat(index + 2);
                bytes[index * ENTRY_SIZE + 8] = op;
                std::fs::write(dir.path().join("audit.bin"), bytes).unwrap();
                for since in [0, 1, 2] {
                    let error = log.entries_since(since).unwrap_err().to_string();
                    assert!(error.contains(&format!("offset {}", index * ENTRY_SIZE)));
                    assert!(error.contains("invalid operation"));
                }
            }
        }
    }

    #[test]
    fn subsequent_append_repairs_a_torn_tail_without_reopening() {
        for written in [1, ENTRY_SIZE - 1, ENTRY_SIZE, ENTRY_SIZE + 1] {
            let dir = tempdir().unwrap();
            let log = AuditLog::new(dir.path()).unwrap();
            log.record(AuditOp::Memorize, "before", None).unwrap();
            let interrupted = AuditEntry {
                timestamp_ms: 1,
                op: AuditOp::Delete,
                id_hash: hash_id("interrupted"),
                signature: [0; 64],
            }
            .to_bytes();
            log.file
                .lock()
                .write_all(&interrupted.repeat(2)[..written])
                .unwrap();
            log.record(AuditOp::Memorize, "after", None).unwrap();
            let expected_count = 2 + written / ENTRY_SIZE;
            let entries = log.entries_since(0).unwrap();
            assert_eq!(entries.len(), expected_count);
            assert_eq!(entries[0].id_hash, hash_id("before"));
            assert_eq!(entries.last().unwrap().id_hash, hash_id("after"));
            let before_reopen = std::fs::read(dir.path().join("audit.bin")).unwrap();
            assert_eq!(before_reopen.len(), expected_count * ENTRY_SIZE);
            if written >= ENTRY_SIZE {
                assert_eq!(&before_reopen[ENTRY_SIZE..2 * ENTRY_SIZE], &interrupted);
            }
            drop(log);
            let reopened = AuditLog::new(dir.path()).unwrap();
            assert_eq!(reopened.entries_since(0).unwrap().len(), expected_count);
            assert_eq!(
                std::fs::read(dir.path().join("audit.bin")).unwrap(),
                before_reopen
            );
        }
    }

    #[test]
    fn failed_tail_repair_prevents_append() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit.bin");
        let bytes = [7; ENTRY_SIZE - 1];
        std::fs::write(&path, bytes).unwrap();
        let log = AuditLog {
            file: Mutex::new(File::open(&path).unwrap()),
        };
        let error = log.record(AuditOp::Delete, "new", None).unwrap_err();
        assert!(error.to_string().contains("Audit tail repair failed"));
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }

    #[test]
    fn concurrent_appends_preserve_records_after_tail_repair() {
        let dir = tempdir().unwrap();
        let log = AuditLog::new(dir.path()).unwrap();
        log.record(AuditOp::Memorize, "before", None).unwrap();
        log.file.lock().write_all(&[7; ENTRY_SIZE - 1]).unwrap();
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for index in 0..8 {
                let log = &log;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    log.record(AuditOp::Delete, &index.to_string(), None)
                        .unwrap();
                });
            }
        });
        let entries = log.entries_since(0).unwrap();
        assert_eq!(entries.len(), 9);
        assert_eq!(entries[0].id_hash, hash_id("before"));
        for index in 0..8 {
            assert_eq!(
                entries
                    .iter()
                    .filter(|entry| entry.id_hash == hash_id(&index.to_string()))
                    .count(),
                1
            );
        }
        assert_eq!(
            std::fs::metadata(dir.path().join("audit.bin"))
                .unwrap()
                .len(),
            9 * ENTRY_SIZE as u64
        );
    }
}
