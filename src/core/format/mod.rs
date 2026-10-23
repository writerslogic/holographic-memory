// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! The persisted format of an HMS store: its version constants, an offline verifier and the
//! migration framework behind `hms-format`.
//!
//! The byte-level specification is `docs/file-format/`; this module is its executable form.
//! Everything that has a version number is defined here once ([`FormatVersions`]), and
//! `tests/format_golden.rs` fails when a constant moves without a matching golden store.
//!
//! - [`scan`] holds the pure byte-slice readers (frames, manifests, headers). They never
//!   panic and never allocate from an untrusted length, so the fuzz target can drive them.
//! - [`verify`] validates a store directory without modifying it.
//! - [`upgrade`] is the migration registry and the steps that exist today.

use serde::Serialize;

pub mod scan;
pub mod upgrade;
pub mod verify;

pub use upgrade::{plan_upgrade, upgrade_store, UpgradePlan, UpgradeReport, UpgradeStep};
pub use verify::{verify_path, VerifyOptions};

/// Version of the standalone persisted vector update directory.
pub const VECTOR_UPDATE_FORMAT_VERSION: u32 = 1;
/// Vector checkpoint manifest magic.
pub const VECTOR_MANIFEST_MAGIC: &[u8; 8] = b"HMSVUP01";
/// Full-precision vector row magic.
pub const VECTOR_ROWS_MAGIC: &[u8; 8] = b"HMSVRW01";
/// Vector update log magic.
pub const VECTOR_LOG_MAGIC: &[u8; 8] = b"HMSVLG01";

/// Version of the arena manifest (`vectors_data.bin/format.json`) this build writes.
pub const STORAGE_FORMAT_CURRENT: u32 = super::storage::STORAGE_FORMAT_VERSION;
/// Oldest arena manifest version this build opens. Version 1 (no manifest) is read as 2.
pub const STORAGE_FORMAT_OLDEST: u32 = 2;
/// `version` field of the `store.json` a new store gets. Schema 2 stores write
/// [`TRANSACTION_MAGIC`] frames, schema 3 stores [`BINARY_TRANSACTION_MAGIC`] frames; a store
/// keeps the schema it was created with. Schema 4 stores also write binary frames and may hold
/// a document checkpoint in place of the frames before it (`docs/file-format/CHECKPOINT.md`).
pub const STORE_SCHEMA_VERSION: u32 = 4;
/// First `store.json` schema whose journal generations may hold a document checkpoint.
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 4;
/// Oldest `store.json` schema this build opens.
pub const STORE_SCHEMA_OLDEST: u32 = 2;
/// Prefix of a JSON journal frame payload (schema 2 stores); the digit is the journal format
/// version.
pub const TRANSACTION_MAGIC: &[u8] = b"HMS-TXN-2\0";
/// Prefix of a binary journal frame payload (schema 3 stores).
pub const BINARY_TRANSACTION_MAGIC: &[u8] = b"HMS-TXN-3\0";
/// Bytes shared by every journal format version: `HMS-TXN-`, then ASCII digits, then NUL.
pub const TRANSACTION_MAGIC_PREFIX: &[u8] = b"HMS-TXN-";
/// Header layout version of `documents.text` this build writes.
pub const TEXT_SEGMENT_VERSION: u8 = 2;
/// First four bytes of `documents.text`.
pub const TEXT_SEGMENT_MAGIC: &[u8; 4] = b"HMST";
/// Length of the `documents.text` header.
pub const TEXT_SEGMENT_HEADER_LEN: usize = 8;
/// Bytes of a journal frame header.
pub const FRAME_HEADER_LEN: usize = super::storage::HEADER_SIZE;
/// Version word of the end-of-log marker header.
pub const END_MARK: u32 = super::storage::END_MARK;
/// Largest decompressed frame payload.
pub const MAX_FRAME_PAYLOAD: usize = super::storage::MAX_RAW_FRAME_SIZE;
/// `magic` of `format.json`.
pub const MANIFEST_MAGIC: &str = super::storage::FORMAT_MAGIC;
/// Bytes of one `audit.bin` entry.
pub const AUDIT_ENTRY_LEN: usize = 105;
/// Bytes of `encryption.salt`.
pub const SALT_LEN: usize = 16;
/// Argon2id memory cost in KiB, iteration count and lanes used to derive the AES-256 key.
pub const KDF_MEMORY_KIB: u32 = 19 * 1024;
/// See [`KDF_MEMORY_KIB`].
pub const KDF_ITERATIONS: u32 = 2;
/// See [`KDF_MEMORY_KIB`].
pub const KDF_LANES: u32 = 1;

/// Every version number a store carries, as this build reads and writes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FormatVersions {
    pub storage_format_current: u32,
    pub storage_format_oldest: u32,
    pub store_schema: u32,
    pub transaction_magic: &'static str,
    pub text_segment: u8,
}

/// The versions this build writes.
pub const CURRENT: FormatVersions = FormatVersions {
    storage_format_current: STORAGE_FORMAT_CURRENT,
    storage_format_oldest: STORAGE_FORMAT_OLDEST,
    store_schema: STORE_SCHEMA_VERSION,
    transaction_magic: "HMS-TXN-3",
    text_segment: TEXT_SEGMENT_VERSION,
};

/// How bad a finding is. The order is the order of precedence for the exit code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Severity {
    /// Worth knowing, nothing to do.
    Info,
    /// The store opens: the engine repairs it on open (a torn tail) or ignores it (a cache).
    Warning,
    /// Data or structure is damaged: the engine refuses to open, or would serve less than
    /// was committed.
    Error,
    /// The store is not damaged but this build cannot read it (a newer format, or one that
    /// was never openable).
    Unsupported,
}

/// One observation about a store.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    pub severity: Severity,
    /// Stable identifier, listed in `docs/file-format/VERIFY.md`.
    pub code: &'static str,
    /// Path relative to the store directory.
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u64>,
    pub message: String,
}

/// Overall verdict of a verification, and the exit code of `hms-format verify`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Verdict {
    /// 0: every check passed.
    Ok,
    /// 10: the store opens; the engine repairs a torn tail on open or discards a bad cache.
    Recoverable,
    /// 20: committed data or structure is damaged.
    Corrupt,
    /// 30: written by a newer format, or never openable by this build.
    Unsupported,
}

impl Verdict {
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Ok => 0,
            Self::Recoverable => 10,
            Self::Corrupt => 20,
            Self::Unsupported => 30,
        }
    }
}

/// Versions read from a store's files; `None` where the file is absent or unreadable.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreVersions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_format: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store_schema: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_encoder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_encoder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u64>,
    /// Journal format versions seen in frame payloads.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub transaction_formats: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_segment: Option<u8>,
}

/// Result of verifying one store directory.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub path: String,
    pub versions: StoreVersions,
    pub segments: usize,
    pub frames: usize,
    pub mutations: usize,
    /// Documents live after replaying the journal; absent when payloads were not readable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_documents: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_vectors: Option<usize>,
    /// Documents carrying a memory-API stamp (`mem1:` version or `_mem` metadata).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_stamped_documents: Option<usize>,
    /// True when every frame payload was decrypted and authenticated (or the store is not
    /// encrypted), so the replay figures describe the content.
    pub payloads_authenticated: bool,
    /// Checks that ran and passed, by name.
    pub checks: Vec<&'static str>,
    pub findings: Vec<Finding>,
    pub verdict: Verdict,
    pub exit_code: i32,
    /// For a server data directory: one report per tenant store, keyed by tenant name.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tenants: Vec<TenantReport>,
}

/// A tenant store inside a server data directory.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TenantReport {
    pub name: String,
    pub report: Report,
}

impl Report {
    pub(crate) fn new(path: &std::path::Path) -> Self {
        Self {
            path: path.display().to_string(),
            versions: StoreVersions::default(),
            segments: 0,
            frames: 0,
            mutations: 0,
            live_documents: None,
            live_vectors: None,
            memory_stamped_documents: None,
            payloads_authenticated: false,
            checks: Vec::new(),
            findings: Vec::new(),
            verdict: Verdict::Ok,
            exit_code: 0,
            tenants: Vec::new(),
        }
    }

    pub(crate) fn add(
        &mut self,
        severity: Severity,
        code: &'static str,
        file: impl Into<String>,
        offset: Option<u64>,
        message: impl Into<String>,
    ) {
        self.findings.push(Finding {
            severity,
            code,
            file: file.into(),
            offset,
            message: message.into(),
        });
    }

    /// Computes the verdict from the findings (and those of every tenant).
    pub(crate) fn finish(&mut self) {
        let mut worst = self.findings.iter().map(|f| f.severity).max();
        for tenant in &self.tenants {
            worst = worst.max(Some(match tenant.report.verdict {
                Verdict::Ok => Severity::Info,
                Verdict::Recoverable => Severity::Warning,
                Verdict::Corrupt => Severity::Error,
                Verdict::Unsupported => Severity::Unsupported,
            }));
        }
        self.verdict = match worst {
            None | Some(Severity::Info) => Verdict::Ok,
            Some(Severity::Warning) => Verdict::Recoverable,
            Some(Severity::Error) => Verdict::Corrupt,
            Some(Severity::Unsupported) => Verdict::Unsupported,
        };
        self.exit_code = self.verdict.exit_code();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constants this module republishes are the ones the writers use.
    #[test]
    fn constants_match_the_writers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("documents.text");
        crate::core::documents::TextSegment::create(
            &path,
            #[cfg(feature = "security")]
            None,
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], TEXT_SEGMENT_MAGIC);
        assert_eq!(bytes[4], TEXT_SEGMENT_VERSION);
        assert_eq!(bytes.len(), TEXT_SEGMENT_HEADER_LEN);
        assert_eq!(TRANSACTION_MAGIC, b"HMS-TXN-2\0");
        assert_eq!(BINARY_TRANSACTION_MAGIC, b"HMS-TXN-3\0");
        assert!(TRANSACTION_MAGIC.starts_with(TRANSACTION_MAGIC_PREFIX));
        assert!(BINARY_TRANSACTION_MAGIC.starts_with(TRANSACTION_MAGIC_PREFIX));
        assert_eq!(
            CURRENT.transaction_magic.as_bytes(),
            &BINARY_TRANSACTION_MAGIC[..9]
        );
        assert_eq!(
            crate::core::engine::journal::JournalFormat::for_schema(STORE_SCHEMA_VERSION),
            crate::core::engine::journal::JournalFormat::Binary
        );
    }
}
