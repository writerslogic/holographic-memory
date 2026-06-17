// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
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
        let timestamp_ms = u64::from_le_bytes(buf[0..8].try_into().unwrap());
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
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    /// Record an operation. The `sign_fn` is called with the entry's
    /// signable bytes (timestamp + op + id_hash = 41 bytes) and should
    /// return a 64-byte signature, or None if signing is disabled.
    pub fn record(&self, op: AuditOp, id: &str, sign_fn: SignFnRef<'_>) -> Result<()> {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let id_hash = hash_id(id);

        let mut signable = [0u8; 41];
        signable[0..8].copy_from_slice(&timestamp_ms.to_le_bytes());
        signable[8] = op as u8;
        signable[9..41].copy_from_slice(&id_hash);

        let signature = match sign_fn {
            Some(f) => f(&signable),
            None => [0u8; 64],
        };

        let entry = AuditEntry {
            timestamp_ms,
            op,
            id_hash,
            signature,
        };

        let bytes = entry.to_bytes();
        let mut file = self.file.lock();
        file.write_all(&bytes)
            .map_err(|e| anyhow!("Audit write failed: {}", e))?;
        file.sync_data()
            .map_err(|e| anyhow!("Audit sync failed: {}", e))?;
        Ok(())
    }

    /// Read all entries with timestamp >= `since_ms`.
    pub fn entries_since(&self, since_ms: u64) -> Result<Vec<AuditEntry>> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(0))?;

        let metadata = file.metadata()?;
        let file_len = metadata.len() as usize;
        if file_len == 0 {
            return Ok(Vec::new());
        }

        let entry_count = file_len / ENTRY_SIZE;
        let mut buf = vec![0u8; file_len];
        file.read_exact(&mut buf)?;

        let mut entries = Vec::new();
        for i in 0..entry_count {
            let offset = i * ENTRY_SIZE;
            let chunk: &[u8; ENTRY_SIZE] = buf[offset..offset + ENTRY_SIZE]
                .try_into()
                .map_err(|_| anyhow!("Audit entry read failed at offset {}", offset))?;
            if let Some(entry) = AuditEntry::from_bytes(chunk) {
                if entry.timestamp_ms >= since_ms {
                    entries.push(entry);
                }
            }
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

        log.record(AuditOp::Memorize, "a", None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        std::thread::sleep(std::time::Duration::from_millis(10));
        log.record(AuditOp::Delete, "b", None).unwrap();

        let entries = log.entries_since(cutoff).unwrap();
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
    }
}
