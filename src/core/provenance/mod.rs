// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

pub mod access;
pub mod c2pa_manifest;
pub mod cawg;
pub mod cose;
pub mod did;
pub mod jumbf;
pub mod keri;
pub mod scitt;
pub mod sigstore;
pub mod status_list;
pub mod trust;
pub mod types;
pub mod vc;

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use trust::StatusLookup;
use trust::TrustStore;

use types::{
    DeletionRecord, ProvenanceRecord, StoreManifest, VerificationDetail, VerificationResult,
};

pub struct TripleProvenanceParams<'a> {
    pub fact_id: &'a str,
    pub content: &'a [u8],
    pub dimensions: u32,
    pub subject: &'a str,
    pub relation: &'a str,
    pub object: &'a str,
    pub source_uri: Option<&'a str>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct LogEntry {
    prev_hash: String,
    fact_id: String,
    record: ProvenanceRecord,
}

/// [`LogEntry`] borrowing its record, serialized to the same bytes.
#[derive(serde::Serialize)]
struct LogEntryRef<'a> {
    prev_hash: String,
    fact_id: &'a str,
    record: &'a ProvenanceRecord,
}

/// Only `Verified` authenticates the complete available log against its signed head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogIntegrity {
    Verified,
    /// No log or anchor exists. This does not authenticate an empty history.
    Empty,
    Unanchored,
    Corrupt,
    /// An in-memory manager has no persistent history to verify.
    Volatile,
}

struct ChainedAppend<'a> {
    _head: parking_lot::MutexGuard<'a, [u8; 32]>,
    anchor: Result<()>,
}

impl ChainedAppend<'_> {
    fn finish(self) -> Result<()> {
        super::error::after_commit("provenance signed head", self.anchor)
    }
}

pub struct ProvenanceManager {
    signing_key: SigningKey,
    issuer_did: String,
    status_list_url: String,
    records: RwLock<crate::core::hash::FastMap<String, ProvenanceRecord>>,
    chain_head: Mutex<[u8; 32]>,
    logical_clock: std::sync::atomic::AtomicU64,
    log_path: Option<PathBuf>,
    revocation_bitstring: RwLock<Vec<u8>>,
    kel: RwLock<keri::KeyEventLog>,
    #[cfg(feature = "provenance-scitt")]
    scitt_endpoint: Option<String>,
}

impl ProvenanceManager {
    pub fn new(key_path: &Path, storage_path: Option<&Path>) -> Result<Self> {
        let signing_key = super::security::SigningManager::new(key_path)?.into_signing_key();
        let issuer_did = did::did_key_from_ed25519(&signing_key.verifying_key().to_bytes());
        let log_path = storage_path.map(|p| p.join("provenance_log.jsonl"));
        let (records, chain_head, mut clock_init) = Self::load_log(log_path.as_deref())?;
        let revocation_bitstring = Self::load_revocation_bitstring(log_path.as_deref())?;
        if let Some((byte, bits)) = revocation_bitstring
            .iter()
            .enumerate()
            .rev()
            .find(|(_, b)| **b != 0)
        {
            let bit = 7 - bits.leading_zeros() as u64;
            clock_init = clock_init.max(byte as u64 * 8 + bit + 1);
        }
        let mut kel = match storage_path {
            Some(p) => match keri::KeyEventLog::with_path(&p.join("kel.json")) {
                Ok(kel) => kel,
                // `persist` replaces the file atomically, but versions before it did not, so a
                // crash there can have left a torn file; refusing to open would turn that crash
                // into an outage.
                Err(e) => {
                    tracing::warn!("KEL unreadable, starting an in-memory log: {e}");
                    keri::KeyEventLog::default()
                }
            },
            None => keri::KeyEventLog::new(),
        };
        if kel.events().is_empty() {
            if let Err(e) = kel.inception(&signing_key, None) {
                tracing::warn!("KEL inception failed: {e}");
            }
        }

        Ok(Self {
            signing_key,
            status_list_url: vc::status_list_credential_url(&issuer_did),
            issuer_did,
            records: RwLock::new(records),
            chain_head: Mutex::new(chain_head),
            logical_clock: std::sync::atomic::AtomicU64::new(clock_init),
            log_path: log_path.clone(),
            revocation_bitstring: RwLock::new(revocation_bitstring),
            kel: RwLock::new(kel),
            #[cfg(feature = "provenance-scitt")]
            scitt_endpoint: None,
        })
    }

    pub fn from_signing_key(signing_key: SigningKey) -> Self {
        let issuer_did = did::did_key_from_ed25519(&signing_key.verifying_key().to_bytes());
        let mut kel = keri::KeyEventLog::new();
        if let Err(e) = kel.inception(&signing_key, None) {
            tracing::warn!("KEL inception failed: {e}");
        }
        Self {
            signing_key,
            status_list_url: vc::status_list_credential_url(&issuer_did),
            issuer_did,
            records: RwLock::new(crate::core::hash::FastMap::default()),
            chain_head: Mutex::new([0u8; 32]),
            logical_clock: std::sync::atomic::AtomicU64::new(0),
            log_path: None,
            revocation_bitstring: RwLock::new(Vec::new()),
            kel: RwLock::new(kel),
            #[cfg(feature = "provenance-scitt")]
            scitt_endpoint: None,
        }
    }

    pub fn with_status_list_url(mut self, url: String) -> Result<Self> {
        status_list::validate_url(&url)?;
        let legacy = vc::status_list_credential_url(&self.issuer_did);
        for record in self.records.read().values() {
            if let Some(ref json) = record.vc_json {
                let credential: vc::FactCredential = serde_json::from_str(json)?;
                if let Some(status) = credential.credential_status {
                    anyhow::ensure!(
                        status.status_list_credential == url
                            || status.status_list_credential == legacy,
                        "configured status list URL differs from an issued credential"
                    );
                }
            }
        }
        self.status_list_url = url;
        Ok(self)
    }

    fn reserve_sequences(&self, count: usize) -> Result<u64> {
        use std::sync::atomic::Ordering::SeqCst;

        let count = u64::try_from(count)?;
        let mut next = self.logical_clock.load(SeqCst);
        loop {
            let end = next
                .checked_add(count)
                .filter(|end| *end <= status_list::MAX_BYTES as u64 * 8)
                .ok_or_else(|| anyhow::anyhow!("credential status sequence space is exhausted"))?;
            match self
                .logical_clock
                .compare_exchange_weak(next, end, SeqCst, SeqCst)
            {
                Ok(_) => return Ok(next),
                Err(current) => next = current,
            }
        }
    }

    #[cfg(feature = "provenance-scitt")]
    pub fn with_scitt_endpoint(mut self, endpoint: String) -> Self {
        self.scitt_endpoint = Some(endpoint);
        self
    }

    pub fn issuer_did(&self) -> &str {
        &self.issuer_did
    }

    /// Appends `record` to the log, when there is one, then makes it the live record for
    /// `fact_id`. A failed append leaves both unchanged; a signed-head failure returns a
    /// `CommittedMutationError` after publishing the durable record.
    pub fn store_record(&self, fact_id: &str, record: ProvenanceRecord) -> Result<()> {
        let next = record
            .sequence
            .checked_add(1)
            .filter(|end| *end <= status_list::MAX_BYTES as u64 * 8)
            .ok_or_else(|| anyhow::anyhow!("credential status sequence space is exhausted"))?;
        let append = match self.log_path {
            Some(ref path) => {
                Some(
                    self.append_chained(path, 1, &self.signing_key, |_, prev_hash| {
                        serde_json::to_string(&LogEntryRef {
                            prev_hash,
                            fact_id,
                            record: &record,
                        })
                    })?,
                )
            }
            None => None,
        };
        self.logical_clock
            .fetch_max(next, std::sync::atomic::Ordering::SeqCst);
        self.records.write().insert(fact_id.to_string(), record);
        if let Some(append) = append {
            append.finish()?;
        }
        Ok(())
    }

    /// Appends `count` lines to the hash-chained log at `path`, line `i` built by `line_for`
    /// from the hex of its predecessor's hash, then advances the head once and signs it with
    /// `head_key`. The head stays locked throughout so concurrent writers cannot chain two
    /// lines to the same predecessor, and it advances only after the lines are written and
    /// synced: a failed append returns the error with the head unchanged and no torn line left
    /// behind. On success the head lock is returned: callers update the live records before
    /// releasing it, so those updates happen in log order and match a replay of the log.
    fn append_chained(
        &self,
        path: &Path,
        count: usize,
        head_key: &SigningKey,
        mut line_for: impl FnMut(usize, String) -> serde_json::Result<String>,
    ) -> Result<ChainedAppend<'_>> {
        let mut head = self.chain_head.lock();
        let mut new_head = *head;
        let mut buf = String::new();
        for i in 0..count {
            let line = line_for(i, hex_encode(&new_head))?;
            new_head = Sha256::digest(line.as_bytes()).into();
            buf.push_str(&line);
            buf.push('\n');
        }
        append_synced(path, buf.as_bytes())
            .map_err(|e| anyhow::anyhow!("provenance log write failed: {e}"))?;
        // The lines are durable, so the head advances even if re-signing it fails below;
        // `verify_log_integrity` reports a stale signed head until the next append.
        *head = new_head;
        let anchor = write_signed_head(head_key, &new_head, path);
        Ok(ChainedAppend {
            _head: head,
            anchor,
        })
    }

    pub fn get_record(&self, fact_id: &str) -> Option<ProvenanceRecord> {
        self.records.read().get(fact_id).cloned()
    }

    pub fn record_count(&self) -> usize {
        self.records.read().len()
    }

    pub fn record_deletion(&self, fact_id: &str, reason: Option<&str>) -> Result<DeletionRecord> {
        let timestamp_ms = unix_millis();

        let mut claim_payload = Vec::new();
        let cbor_claim = ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("action".to_string()),
                ciborium::Value::Text("delete".to_string()),
            ),
            (
                ciborium::Value::Text("fact_id".to_string()),
                ciborium::Value::Text(fact_id.to_string()),
            ),
            (
                ciborium::Value::Text("timestamp_ms".to_string()),
                ciborium::Value::Integer(timestamp_ms.into()),
            ),
        ]);
        ciborium::into_writer(&cbor_claim, &mut claim_payload)
            .map_err(|e| anyhow::anyhow!("CBOR encoding failed: {e}"))?;

        let cose_envelope = cose::sign_payload(&self.signing_key, &claim_payload)?;

        let record = DeletionRecord {
            fact_id: fact_id.to_string(),
            deleted_by: Some(self.issuer_did.clone()),
            timestamp_ms,
            reason: reason.map(String::from),
            cose_envelope: Some(cose_envelope),
        };

        let append = match self.log_path {
            Some(ref path) => {
                Some(
                    self.append_chained(path, 1, &self.signing_key, |_, prev_hash| {
                        serde_json::to_string(&serde_json::json!({
                            "prev_hash": prev_hash,
                            "action": "delete",
                            "fact_id": fact_id,
                            "deletion": record,
                        }))
                    })?,
                )
            }
            None => None,
        };
        self.records.write().remove(fact_id);
        if let Some(append) = append {
            append.finish()?;
        }

        Ok(record)
    }

    pub fn rotate_key(&mut self, new_key_path: &Path) -> Result<String> {
        let old_did = self.issuer_did.clone();
        let new_signing_key =
            super::security::SigningManager::new(new_key_path)?.into_signing_key();
        let new_did = did::did_key_from_ed25519(&new_signing_key.verifying_key().to_bytes());

        let mut rotation_payload = Vec::new();
        let cbor = ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("action".to_string()),
                ciborium::Value::Text("key_rotation".to_string()),
            ),
            (
                ciborium::Value::Text("old_did".to_string()),
                ciborium::Value::Text(old_did.clone()),
            ),
            (
                ciborium::Value::Text("new_did".to_string()),
                ciborium::Value::Text(new_did.clone()),
            ),
        ]);
        ciborium::into_writer(&cbor, &mut rotation_payload)
            .map_err(|e| anyhow::anyhow!("CBOR encoding failed: {e}"))?;
        let rotation_envelope = cose::sign_payload(&self.signing_key, &rotation_payload)?;

        // The old key signs the rotation line; the new key signs the head after it, since the
        // manager verifies the head with its current key, here and after a reopen with the new
        // key file.
        let anchor = if let Some(ref path) = self.log_path {
            self.append_chained(path, 1, &new_signing_key, |_, prev_hash| {
                serde_json::to_string(&serde_json::json!({
                    "prev_hash": prev_hash,
                    "action": "key_rotation",
                    "old_did": old_did,
                    "new_did": new_did,
                    "cose_envelope": rotation_envelope,
                }))
            })?
            .finish()
        } else {
            Ok(())
        };

        self.signing_key = new_signing_key;
        self.issuer_did = new_did.clone();
        anchor?;
        Ok(new_did)
    }

    pub fn verify_log_integrity(&self) -> Result<bool> {
        Ok(self.log_integrity()? == LogIntegrity::Verified)
    }

    /// Missing evidence never verifies; filesystem failures remain errors. Replay tolerance
    /// is deliberately separate from verification of the complete signed history.
    pub fn log_integrity(&self) -> Result<LogIntegrity> {
        let Some(ref path) = self.log_path else {
            return Ok(LogIntegrity::Volatile);
        };
        let known_head = *self.chain_head.lock();
        let signed_bytes = match std::fs::read(path.with_extension("head")) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).context("failed to read provenance signed head"),
        };
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(if signed_bytes.is_some() || known_head != [0; 32] {
                    LogIntegrity::Corrupt
                } else {
                    LogIntegrity::Empty
                });
            }
            Err(e) => return Err(e).context("failed to open provenance log"),
        };
        let mut expected_prev = [0u8; 32];
        let mut computed_head = [0u8; 32];
        let mut has_entries = false;
        for line in std::io::BufReader::new(file).lines() {
            let line = line.map_err(|e| anyhow::anyhow!("log read error: {e}"))?;
            let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) else {
                return Ok(LogIntegrity::Corrupt);
            };
            let Some(prev) = obj.get("prev_hash").and_then(|v| v.as_str()) else {
                return Ok(LogIntegrity::Corrupt);
            };
            let Ok(declared) = hex_decode(prev) else {
                return Ok(LogIntegrity::Corrupt);
            };
            if declared != expected_prev {
                return Ok(LogIntegrity::Corrupt);
            }
            computed_head = Sha256::digest(line.as_bytes()).into();
            expected_prev = computed_head;
            has_entries = true;
        }
        anyhow::ensure!(
            *self.chain_head.lock() == known_head,
            "provenance log changed during verification; retry"
        );
        if computed_head != known_head {
            return Ok(LogIntegrity::Corrupt);
        }
        let Some(signed_bytes) = signed_bytes else {
            return Ok(if has_entries {
                LogIntegrity::Unanchored
            } else {
                LogIntegrity::Empty
            });
        };
        match cose::verify_and_extract(&self.signing_key.verifying_key(), &signed_bytes) {
            Ok(stored_head) if stored_head.as_slice() == computed_head => {
                Ok(LogIntegrity::Verified)
            }
            _ => Ok(LogIntegrity::Corrupt),
        }
    }

    /// Replays the log into the live records and the chain head. A missing log is an empty
    /// one; an unreadable one is an error, since chaining new entries onto a guessed head
    /// would break the log.
    fn load_log(
        path: Option<&Path>,
    ) -> Result<(
        crate::core::hash::FastMap<String, ProvenanceRecord>,
        [u8; 32],
        u64,
    )> {
        let mut next_sequence = 0;
        let mut map = crate::core::hash::FastMap::default();
        let mut head = [0u8; 32];
        let Some(path) = path else {
            return Ok((map, head, next_sequence));
        };
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((map, head, next_sequence))
            }
            Err(e) => return Err(anyhow::anyhow!("failed to open provenance log: {e}")),
        };
        for line in std::io::BufReader::new(file).lines() {
            let line = match line {
                Ok(line) => line,
                // A crash can tear the last line mid-character; replay what precedes it.
                Err(e) if e.kind() == std::io::ErrorKind::InvalidData => continue,
                Err(e) => return Err(anyhow::anyhow!("log read error: {e}")),
            };
            head = Sha256::digest(line.as_bytes()).into();
            if let Ok(entry) = serde_json::from_str::<LogEntry>(&line) {
                let after = entry
                    .record
                    .sequence
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("provenance sequence is exhausted"))?;
                next_sequence = next_sequence.max(after);
                map.insert(entry.fact_id, entry.record);
            } else if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) {
                if obj.get("action").and_then(|a| a.as_str()) == Some("delete") {
                    if let Some(fid) = obj.get("fact_id").and_then(|f| f.as_str()) {
                        map.remove(fid);
                    }
                }
            }
        }
        Ok((map, head, next_sequence))
    }

    /// A missing status list means nothing is revoked. An unreadable one is an error rather
    /// than an empty list, which would report every revoked credential as valid.
    fn load_revocation_bitstring(log_path: Option<&Path>) -> Result<Vec<u8>> {
        let Some(path) = log_path else {
            return Ok(Vec::new());
        };
        match std::fs::File::open(path.with_extension("statuslist")) {
            Ok(file) => {
                let mut bits = Vec::new();
                file.take((status_list::MAX_BYTES + 1) as u64)
                    .read_to_end(&mut bits)?;
                anyhow::ensure!(
                    bits.len() <= status_list::MAX_BYTES,
                    "revocation status list exceeds size limit"
                );
                Ok(bits)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(anyhow::anyhow!(
                "failed to read revocation status list: {e}"
            )),
        }
    }

    pub fn revoke_credential(&self, status_index: u64) -> Result<()> {
        // The index is caller-supplied: one past `usize` or past what can be allocated is an
        // error, not an overflow or an allocation abort.
        let out_of_range = || anyhow::anyhow!("status index {status_index} is out of range");
        let byte_idx = usize::try_from(status_index / 8).map_err(|_| out_of_range())?;
        let len = byte_idx.checked_add(1).ok_or_else(out_of_range)?;
        anyhow::ensure!(
            len <= status_list::MAX_BYTES,
            "status index exceeds list size limit"
        );
        let bit_idx = (status_index % 8) as u8;
        // The write guard covers the update and its persistence: concurrent revocations
        // reach the file in the order they change the list, so a later write never carries
        // an older list, and a failed write leaves the live list as the file has it.
        let mut bits = self.revocation_bitstring.write();
        let mut next = Vec::new();
        next.try_reserve(len.max(bits.len()))
            .map_err(|_| out_of_range())?;
        next.extend_from_slice(&bits);
        if next.len() < len {
            next.resize(len, 0);
        }
        next[byte_idx] |= 1 << bit_idx;
        if let Some(ref path) = self.log_path {
            // Replaced atomically: a torn list would read back as shorter, un-revoking the
            // credentials past the tear.
            super::durable_file::atomic_write(&path.with_extension("statuslist"), &next)
                .map_err(|e| anyhow::anyhow!("failed to persist revocation bitstring: {e}"))?;
        }
        *bits = next;
        Ok(())
    }

    pub fn is_revoked(&self, status_index: u64) -> bool {
        trust::status_bit(&self.revocation_bitstring.read(), status_index)
    }

    fn published_status_list_len(&self, bits_len: usize) -> Option<usize> {
        let highest_issued = self
            .logical_clock
            .load(std::sync::atomic::Ordering::SeqCst)
            .saturating_sub(1);
        let issued_bytes = usize::try_from(highest_issued / 8).ok()?.checked_add(1)?;
        let length = status_list::MIN_BYTES.max(issued_bytes).max(bits_len);
        (length <= status_list::MAX_BYTES).then_some(length)
    }

    fn local_status(&self, index: u64) -> StatusLookup {
        let bits = self.revocation_bitstring.read();
        let Some(length) = self.published_status_list_len(bits.len()) else {
            return StatusLookup::OutOfRange;
        };
        if index / 8 >= length as u64 {
            StatusLookup::OutOfRange
        } else {
            StatusLookup::Value(trust::status_bit(&bits, index))
        }
    }

    /// Create a full provenance record for a stored fact.
    pub fn create_fact_provenance(
        &self,
        fact_id: &str,
        content: &[u8],
        source_uri: Option<&str>,
    ) -> Result<ProvenanceRecord> {
        let content_hash: [u8; 32] = Sha256::digest(content).into();
        let mut record = ProvenanceRecord::now();
        record.source_uri = Some(source_uri.unwrap_or("").to_string());
        record.source_hash = Some(content_hash);
        record.issuer_did = Some(self.issuer_did.clone());
        record.sequence = self.reserve_sequences(1)?;

        let cose_envelope = cose::sign_fact_claim(
            &self.signing_key,
            fact_id,
            &content_hash,
            source_uri,
            record.timestamp_ms,
        )?;
        record.cose_envelope = Some(cose_envelope);

        let credential = vc::create_fact_credential(
            &self.issuer_did,
            fact_id,
            &content_hash,
            0, // dimensions filled by caller
            source_uri,
            None,
            record.sequence,
        )
        .with_status_list_url(&self.status_list_url);
        let signed_vc = vc::sign_credential(&self.signing_key, credential)?;
        record.vc_json = Some(serde_json::to_string(&signed_vc)?);

        #[cfg(feature = "provenance-scitt")]
        if let Some(ref endpoint) = self.scitt_endpoint {
            let claim = scitt::FactClaim {
                fact_id: fact_id.to_string(),
                content_hash: content_hash.to_vec(),
                timestamp_ms: record.timestamp_ms,
                source_uri: source_uri.map(String::from),
                store_id: None,
            };
            let statement = scitt::create_signed_statement(&self.signing_key, &claim)?;
            match scitt::submit_statement(endpoint, &statement) {
                Ok(receipt) => record.scitt_receipt = Some(receipt.receipt_bytes),
                Err(e) => tracing::warn!("SCITT submission failed: {e}"),
            }
        }

        self.store_record(fact_id, record.clone())?;
        Ok(record)
    }

    /// Create a full provenance record for a stored triple.
    pub fn create_triple_provenance(
        &self,
        params: &TripleProvenanceParams<'_>,
    ) -> Result<ProvenanceRecord> {
        let content_hash: [u8; 32] = Sha256::digest(params.content).into();
        let mut record = ProvenanceRecord::now();
        record.source_uri = params.source_uri.map(String::from);
        record.source_hash = Some(content_hash);
        record.issuer_did = Some(self.issuer_did.clone());
        record.sequence = self.reserve_sequences(1)?;

        let cose_envelope = cose::sign_fact_claim(
            &self.signing_key,
            params.fact_id,
            &content_hash,
            params.source_uri,
            record.timestamp_ms,
        )?;
        record.cose_envelope = Some(cose_envelope);

        let credential = vc::create_fact_credential(
            &self.issuer_did,
            params.fact_id,
            &content_hash,
            params.dimensions,
            params.source_uri,
            Some((params.subject, params.relation, params.object)),
            record.sequence,
        )
        .with_status_list_url(&self.status_list_url);
        let signed_vc = vc::sign_credential(&self.signing_key, credential)?;
        record.vc_json = Some(serde_json::to_string(&signed_vc)?);

        self.store_record(params.fact_id, record.clone())?;
        Ok(record)
    }

    pub fn create_batch_provenance(
        &self,
        items: &[(&str, &[u8], Option<&str>)],
    ) -> Result<Vec<ProvenanceRecord>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let leaf_hashes: Vec<[u8; 32]> = items
            .iter()
            .map(|(_, content, _)| Sha256::digest(content).into())
            .collect();
        let hashing = MerkleHashing::Rfc6962;
        let leaf_nodes: Vec<[u8; 32]> = leaf_hashes.iter().map(|h| hashing.leaf(h)).collect();
        let levels = Self::merkle_levels(&leaf_nodes, hashing);
        let merkle_root = levels[levels.len() - 1][0];
        let root_envelope = cose::sign_payload(
            &self.signing_key,
            &batch_root_claim(&merkle_root, leaf_nodes.len())?,
        )?;

        // One contiguous block of sequence numbers, then the per-record credential signatures
        // (most of a batch's cost) in parallel on the rayon pool, collected in item order.
        let first_sequence = self.reserve_sequences(items.len())?;
        let records = items
            .par_iter()
            .enumerate()
            .map(
                |(i, (fact_id, _, source_uri))| -> Result<ProvenanceRecord> {
                    let content_hash = leaf_hashes[i];
                    let mut record = ProvenanceRecord::now();
                    record.source_uri = Some(source_uri.unwrap_or("").to_string());
                    record.source_hash = Some(content_hash);
                    record.issuer_did = Some(self.issuer_did.clone());
                    record.sequence = first_sequence + i as u64;
                    record.cose_envelope = Some(root_envelope.clone());
                    record.merkle_proof = Some(Self::merkle_proof(&levels, i));

                    let credential = vc::create_fact_credential(
                        &self.issuer_did,
                        fact_id,
                        &content_hash,
                        0,
                        *source_uri,
                        None,
                        record.sequence,
                    )
                    .with_status_list_url(&self.status_list_url);
                    let signed_vc = vc::sign_credential(&self.signing_key, credential)?;
                    record.vc_json = Some(serde_json::to_string(&signed_vc)?);
                    Ok(record)
                },
            )
            .collect::<Result<Vec<_>>>()?;
        // One append, sync and head signature for the whole batch.
        let append = match self.log_path {
            Some(ref path) => Some(self.append_chained(
                path,
                records.len(),
                &self.signing_key,
                |i, prev_hash| {
                    serde_json::to_string(&LogEntryRef {
                        prev_hash,
                        fact_id: items[i].0,
                        record: &records[i],
                    })
                },
            )?),
            None => None,
        };
        let mut live = self.records.write();
        for ((fact_id, _, _), record) in items.iter().zip(&records) {
            live.insert(fact_id.to_string(), record.clone());
        }
        drop(live);
        if let Some(append) = append {
            append.finish()?;
        }
        Ok(records)
    }

    /// The Merkle tree's levels, leaves first and the root alone last.
    fn merkle_levels(leaves: &[[u8; 32]], hashing: MerkleHashing) -> Vec<Vec<[u8; 32]>> {
        let mut levels = vec![if leaves.is_empty() {
            vec![[0u8; 32]]
        } else {
            leaves.to_vec()
        }];
        loop {
            let top = &levels[levels.len() - 1];
            if top.len() <= 1 {
                return levels;
            }
            let parents = Self::merkle_parents(top, hashing);
            levels.push(parents);
        }
    }

    /// One Merkle level up: the node hash of each adjacent pair, an odd last node carried
    /// unhashed.
    fn merkle_parents(level: &[[u8; 32]], hashing: MerkleHashing) -> Vec<[u8; 32]> {
        level
            .chunks(2)
            .map(|pair| match pair {
                [left, right] => hashing.node(left, right),
                _ => pair[0],
            })
            .collect()
    }

    /// Leaf `index`'s inclusion proof, read from the levels of [`Self::merkle_levels`]: each
    /// level's sibling, flagged when it sits to the left; an odd last node has none.
    fn merkle_proof(levels: &[Vec<[u8; 32]>], index: usize) -> Vec<(bool, [u8; 32])> {
        let mut idx = index;
        let mut proof = Vec::new();
        for level in &levels[..levels.len() - 1] {
            if let Some(sibling) = level.get(idx ^ 1) {
                proof.push((idx % 2 == 1, *sibling));
            }
            idx /= 2;
        }
        proof
    }

    /// Create a signed manifest for the entire store.
    pub fn create_store_manifest(
        &self,
        store_id: &str,
        store_data: &[u8],
        fact_count: usize,
        dimensions: u32,
        title: Option<&str>,
    ) -> Result<StoreManifest> {
        let store_hash: [u8; 32] = Sha256::digest(store_data).into();

        let params = c2pa_manifest::ManifestParams {
            store_id,
            fact_count,
            dimensions,
            store_hash: &store_hash,
            title,
            ingredients: Vec::new(),
        };

        let manifest = c2pa_manifest::create_manifest(&self.signing_key, &params)?;
        let cose_envelope = c2pa_manifest::sign_manifest(&self.signing_key, &manifest)?;

        let manifest_label = format!("urn:hms:manifest:{store_id}");
        let claim_json = serde_json::to_value(&manifest)?;
        let assertions: Vec<(&str, serde_json::Value)> = manifest
            .assertions
            .iter()
            .map(|a| (a.label.as_str(), a.data.clone()))
            .collect();
        let jumbf_box = jumbf::build_c2pa_manifest_box(
            &manifest_label,
            claim_json,
            cose_envelope.clone(),
            assertions,
        );
        let jumbf_bytes = jumbf_box.encode()?;

        Ok(StoreManifest {
            manifest_id: manifest_label,
            issuer_did: Some(self.issuer_did.clone()),
            created_ms: unix_millis(),
            fact_count,
            store_hash,
            cose_envelope: Some(cose_envelope),
            jumbf_manifest: Some(jumbf_bytes),
        })
    }

    fn resolve_cose_verifying_key(
        record: &ProvenanceRecord,
        fallback: &ed25519_dalek::VerifyingKey,
    ) -> ed25519_dalek::VerifyingKey {
        if let Some(ref envelope) = record.cose_envelope {
            if let Ok(kid) = cose::extract_key_id(envelope) {
                if kid.len() == 32 {
                    if let Ok(key) = ed25519_dalek::VerifyingKey::from_bytes(
                        kid.as_slice().try_into().unwrap_or(&[0u8; 32]),
                    ) {
                        return key;
                    }
                }
            }
        }
        if let Some(ref issuer_did) = record.issuer_did {
            if let Ok(pk_bytes) = did::ed25519_from_did_key(issuer_did) {
                if let Ok(key) = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes) {
                    return key;
                }
            }
        }
        *fallback
    }

    /// Authenticity check: verifies the record's signatures **and** requires the
    /// key that produced them to be present in `trust`. A record whose embedded
    /// key is not trusted verifies as `valid == false` even if its signature is
    /// internally consistent. Use [`ProvenanceManager::self_trust`] to trust
    /// only this manager's own key.
    pub fn verify_fact_provenance(
        &self,
        record: &ProvenanceRecord,
        trust: &TrustStore,
    ) -> Result<VerificationResult> {
        let mut result = self.verify_signatures(record, Some(trust))?;
        let fallback = self.signing_key.verifying_key();
        let key = Self::resolve_cose_verifying_key(record, &fallback);
        result
            .details
            .push(trust_anchor_detail(trust.is_trusted(&key)));
        result.valid = result.details.iter().all(|d| d.passed) && !result.details.is_empty();
        Ok(result)
    }

    /// Integrity-only check: verifies the record's signatures against the key
    /// **embedded in the record itself**. This proves the record has not been
    /// altered since signing, but says nothing about *who* signed it — a record
    /// signed by any key passes. For authenticity use
    /// [`ProvenanceManager::verify_fact_provenance`] with a trust anchor.
    pub fn verify_fact_provenance_self_consistency(
        &self,
        record: &ProvenanceRecord,
    ) -> Result<VerificationResult> {
        self.verify_signatures(record, None)
    }

    /// The signature, revocation and binding checks of a record. A credential citing this
    /// manager's status list is revoked when the live list or `trust`'s copy says so; one
    /// citing another list is checked against `trust`'s copy of that list. Without one it
    /// falls back to the live list (credentials issued before a key rotation cite the list
    /// under the old DID), unless `trust` requires status lists.
    fn verify_signatures(
        &self,
        record: &ProvenanceRecord,
        trust: Option<&TrustStore>,
    ) -> Result<VerificationResult> {
        let legacy_status_list = vc::status_list_credential_url(&self.issuer_did);
        let mut details = Vec::new();
        let fallback = self.signing_key.verifying_key();
        let verifying_key = Self::resolve_cose_verifying_key(record, &fallback);

        let mut payload = None;
        if let Some(ref envelope) = record.cose_envelope {
            let outcome = cose::verify_and_extract(&verifying_key, envelope);
            details.push(outcome_detail("cose_signature", &outcome));
            payload = outcome.ok();
        }

        let mut credential = None;
        if let Some(ref vc_json) = record.vc_json {
            details.push(outcome_detail(
                "vc_proof",
                &vc::verify_credential_json(vc_json),
            ));
            match serde_json::from_str::<vc::FactCredential>(vc_json) {
                Ok(parsed) => {
                    details.push(outcome_detail(
                        "vc_semantics",
                        &vc::validate_fact_credential(&parsed),
                    ));
                    details.push(revocation_detail(
                        &parsed,
                        trust.is_some_and(TrustStore::requires_status_lists),
                        |url, index| {
                            let trusted = trust
                                .map_or(StatusLookup::Missing, |t| t.status_lookup(url, index));
                            if url == self.status_list_url || url == legacy_status_list {
                                match (self.local_status(index), trusted) {
                                    (StatusLookup::Value(local), StatusLookup::Value(remote)) => {
                                        StatusLookup::Value(local || remote)
                                    }
                                    (StatusLookup::Value(local), _) => StatusLookup::Value(local),
                                    (StatusLookup::OutOfRange, StatusLookup::Value(remote)) => {
                                        StatusLookup::Value(remote)
                                    }
                                    (StatusLookup::OutOfRange, StatusLookup::Stale) => {
                                        StatusLookup::Stale
                                    }
                                    _ => StatusLookup::OutOfRange,
                                }
                            } else {
                                trusted
                            }
                        },
                    ));
                    credential = Some(parsed);
                }
                Err(e) => details.push(VerificationDetail {
                    check: "vc_parse".to_string(),
                    passed: false,
                    message: Some(e.to_string()),
                }),
            }
        }
        if !details.is_empty() {
            details.push(outcome_detail(
                "record_binding",
                &check_binding(
                    record,
                    payload.as_deref().map(|p| (p, &verifying_key)),
                    credential.as_ref(),
                ),
            ));
        }

        let all_passed = details.iter().all(|d| d.passed);

        Ok(VerificationResult {
            valid: all_passed && !details.is_empty(),
            issuer_did: record.issuer_did.clone(),
            timestamp_ms: record.timestamp_ms,
            details,
        })
    }

    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    /// This manager's current verifying (public) key.
    pub fn verifying_key(&self) -> ed25519_dalek::VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// A trust store containing only this manager's own key. Pass it to the
    /// verify methods to authenticate records this manager signed.
    /// A trust store holding this manager's key and a snapshot of its revocation status list.
    pub fn self_trust(&self) -> TrustStore {
        let mut trust = TrustStore::trusting_key(&self.signing_key.verifying_key());
        let mut bits = self.revocation_bitstring.read().clone();
        if let Some(length) = self.published_status_list_len(bits.len()) {
            bits.resize(length, 0);
        }
        trust.trust_status_list(&self.status_list_url, bits.clone());
        let legacy_url = vc::status_list_credential_url(&self.issuer_did);
        if legacy_url != self.status_list_url {
            trust.trust_status_list(&legacy_url, bits);
        }
        trust
    }

    pub fn status_list_credential(&self) -> Result<status_list::StatusListCredential> {
        let highest_issued = self
            .logical_clock
            .load(std::sync::atomic::Ordering::SeqCst)
            .saturating_sub(1);
        let bits = self.revocation_bitstring.read().clone();
        status_list::create_status_list_credential(
            &self.signing_key,
            &self.status_list_url,
            &bits,
            highest_issued,
        )
    }

    pub fn create_sigstore_bundle(
        &self,
        content: &[u8],
        identity: Option<&str>,
    ) -> Result<sigstore::SigstoreBundle> {
        sigstore::create_local_bundle(&self.signing_key, content, identity)
    }

    /// Verify a Sigstore bundle against a caller-supplied trusted key. The key
    /// embedded in the bundle is ignored: authenticity comes from `trusted_key`,
    /// not from the bundle itself.
    pub fn verify_sigstore_bundle(
        &self,
        bundle: &sigstore::SigstoreBundle,
        content: &[u8],
        trusted_key: &ed25519_dalek::VerifyingKey,
    ) -> Result<()> {
        sigstore::verify_bundle_with_key(bundle, content, trusted_key)
    }

    /// Create a CAWG ICA identity assertion. `referenced` is `(url, alg, raw_hash)` and
    /// MUST include the hard binding. Returns the `cawg.identity` assertion JSON.
    pub fn create_cawg_assertion(
        &self,
        referenced: &[(String, String, Vec<u8>)],
        display_name: &str,
    ) -> Result<serde_json::Value> {
        cawg::create_identity_assertion_ica(&self.signing_key, referenced, display_name)
    }

    /// Verify a standalone `cawg.identity` ICA signature and issuer trust, returning
    /// the embedded VC. Complete identity assertion validation also requires the
    /// authenticated containing C2PA claim and its referenced assertions.
    pub fn verify_cawg_assertion(
        &self,
        assertion: &serde_json::Value,
        trust: &TrustStore,
    ) -> Result<serde_json::Value> {
        let embedded = cawg::ica_embedded_bytes(assertion)?;
        let vc = cawg::verify_cawg_ica(&embedded)?;
        let issuer = vc
            .get("issuer")
            .and_then(|v| {
                v.as_str()
                    .or_else(|| v.get("id").and_then(|id| id.as_str()))
            })
            .ok_or_else(|| anyhow::anyhow!("CAWG assertion VC has no issuer"))?;
        let key_bytes = did::ed25519_from_did_jwk(issuer)?;
        let key = ed25519_dalek::VerifyingKey::from_bytes(&key_bytes)
            .map_err(|e| anyhow::anyhow!("invalid CAWG issuer key: {e}"))?;
        if !trust.is_trusted(&key) {
            return Err(anyhow::anyhow!("CAWG issuer is not trusted: {issuer}"));
        }
        Ok(vc)
    }

    pub fn keri_event_log(&self) -> &RwLock<keri::KeyEventLog> {
        &self.kel
    }

    pub fn create_keri_interaction(
        &self,
        anchors: Vec<serde_json::Value>,
    ) -> Result<keri::KeyEvent> {
        self.kel
            .write()
            .interaction(&self.signing_key, anchors)
            .cloned()
    }

    pub fn verify_keri_log(&self) -> Result<()> {
        self.kel.read().verify()
    }

    /// Verify a store manifest's COSE envelope and optionally check the store hash.
    fn resolve_manifest_verifying_key(
        manifest: &StoreManifest,
        fallback: &ed25519_dalek::VerifyingKey,
    ) -> ed25519_dalek::VerifyingKey {
        if let Some(ref envelope) = manifest.cose_envelope {
            if let Ok(kid) = cose::extract_key_id(envelope) {
                if let Ok(key) = ed25519_dalek::VerifyingKey::from_bytes(
                    kid.as_slice().try_into().unwrap_or(&[0u8; 32]),
                ) {
                    return key;
                }
            }
        }
        if let Some(ref issuer_did) = manifest.issuer_did {
            if let Ok(pk_bytes) = did::ed25519_from_did_key(issuer_did) {
                if let Ok(key) = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes) {
                    return key;
                }
            }
        }
        *fallback
    }

    /// Authenticity check for a store manifest: verifies its COSE signature and
    /// optional store hash, and requires the signing key to be present in
    /// `trust`. See [`ProvenanceManager::verify_fact_provenance`] for the
    /// self-consistency vs. authenticity distinction.
    pub fn verify_store_manifest(
        &self,
        manifest: &StoreManifest,
        store_data: Option<&[u8]>,
        trust: &TrustStore,
    ) -> Result<VerificationResult> {
        let mut result = self.verify_store_manifest_self_consistency(manifest, store_data)?;
        let fallback = self.signing_key.verifying_key();
        let key = Self::resolve_manifest_verifying_key(manifest, &fallback);
        result
            .details
            .push(trust_anchor_detail(trust.is_trusted(&key)));
        result.valid = result.details.iter().all(|d| d.passed) && !result.details.is_empty();
        Ok(result)
    }

    /// Integrity-only check for a store manifest: verifies signatures against
    /// the key embedded in the manifest. Does not establish authenticity.
    pub fn verify_store_manifest_self_consistency(
        &self,
        manifest: &StoreManifest,
        store_data: Option<&[u8]>,
    ) -> Result<VerificationResult> {
        let mut details = Vec::new();
        let fallback = self.signing_key.verifying_key();
        let verifying_key = Self::resolve_manifest_verifying_key(manifest, &fallback);

        if let Some(ref envelope) = manifest.cose_envelope {
            match cose::verify_and_extract(&verifying_key, envelope) {
                Ok(payload) => {
                    details.push(VerificationDetail {
                        check: "manifest_cose_signature".to_string(),
                        passed: true,
                        message: None,
                    });

                    // The signature only proves the key signed *some* payload: it must be a
                    // store manifest for this manifest's store hash, or an envelope signed for
                    // a fact, a head or another store verifies when pasted in.
                    let parsed = serde_json::from_slice::<c2pa_manifest::HmsManifest>(&payload)
                        .map_err(|e| {
                            anyhow::anyhow!("signed payload is not a store manifest: {e}")
                        });
                    let binding = parsed
                        .as_ref()
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .and_then(|hms_manifest| {
                            let signed = c2pa_manifest::signed_store_hash(hms_manifest)?;
                            if signed == hex_encode(&manifest.store_hash) {
                                Ok(())
                            } else {
                                Err(anyhow::anyhow!(
                                    "signed store hash differs from the manifest's"
                                ))
                            }
                        });
                    details.push(outcome_detail("manifest_binding", &binding));
                    if let Ok(ref hms_manifest) = parsed {
                        if let Some(data) = store_data {
                            let hash_ok = c2pa_manifest::validate_store_hash(hms_manifest, data)
                                .unwrap_or(false);
                            details.push(VerificationDetail {
                                check: "store_hash".to_string(),
                                passed: hash_ok,
                                message: if hash_ok {
                                    None
                                } else {
                                    Some("store hash mismatch".to_string())
                                },
                            });
                        }
                    }
                }
                Err(e) => details.push(VerificationDetail {
                    check: "manifest_cose_signature".to_string(),
                    passed: false,
                    message: Some(e.to_string()),
                }),
            }
        }

        let all_passed = details.iter().all(|d| d.passed);

        Ok(VerificationResult {
            valid: all_passed && !details.is_empty(),
            issuer_did: manifest.issuer_did.clone(),
            timestamp_ms: manifest.created_ms,
            details,
        })
    }
}

/// Verify a provenance record without a ProvenanceManager instance.
///
/// Verifies the record's signatures and requires its signing identity —
/// resolved from the COSE key_id, or the record's issuer DID for a vc-only
/// record — to be present in `trust`. As with
/// [`ProvenanceManager::verify_fact_provenance`], a record whose signing key is
/// not trusted verifies as `valid == false`.
pub fn verify_record(record: &ProvenanceRecord, trust: &TrustStore) -> Result<VerificationResult> {
    let mut details = Vec::new();

    let mut signed_payload = None;
    if let Some(ref envelope) = record.cose_envelope {
        let kid = cose::extract_key_id(envelope)?;
        let pk: [u8; 32] = kid
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("COSE key_id is not 32 bytes"))?;
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk)
            .map_err(|e| anyhow::anyhow!("invalid public key in COSE: {e}"))?;
        let outcome = cose::verify_and_extract(&verifying_key, envelope);
        details.push(outcome_detail("cose_signature", &outcome));
        signed_payload = outcome.ok().map(|payload| (payload, verifying_key));
    }

    let mut credential = None;
    if let Some(ref vc_json) = record.vc_json {
        details.push(outcome_detail(
            "vc_proof",
            &vc::verify_credential_json(vc_json),
        ));
        match serde_json::from_str::<vc::FactCredential>(vc_json) {
            Ok(parsed) => {
                details.push(outcome_detail(
                    "vc_semantics",
                    &vc::validate_fact_credential(&parsed),
                ));
                details.push(revocation_detail(
                    &parsed,
                    trust.requires_status_lists(),
                    |url, index| trust.status_lookup(url, index),
                ));
                credential = Some(parsed);
            }
            Err(e) => details.push(VerificationDetail {
                check: "vc_parse".to_string(),
                passed: false,
                message: Some(e.to_string()),
            }),
        }
    }
    if !details.is_empty() {
        details.push(outcome_detail(
            "record_binding",
            &check_binding(
                record,
                signed_payload.as_ref().map(|(p, k)| (p.as_slice(), k)),
                credential.as_ref(),
            ),
        ));
    }

    // Trust anchoring applies to the whole record, not just its COSE envelope:
    // resolve the signing identity from the COSE key_id, falling back to the
    // record's issuer DID (so a vc-only record is anchored too), and require it
    // to be trusted. Without the fallback, a record carrying only a self-signed
    // VC would bypass the trust check entirely.
    if !details.is_empty() {
        let signer_key = record
            .cose_envelope
            .as_ref()
            .and_then(|e| cose::extract_key_id(e).ok())
            .and_then(|kid| <[u8; 32]>::try_from(kid.as_slice()).ok())
            .or_else(|| {
                record
                    .issuer_did
                    .as_deref()
                    .and_then(|d| did::ed25519_from_did_key(d).ok())
            })
            .and_then(|pk| ed25519_dalek::VerifyingKey::from_bytes(&pk).ok());
        let trusted = signer_key.as_ref().is_some_and(|k| trust.is_trusted(k));
        details.push(trust_anchor_detail(trusted));
    }

    let all_passed = details.iter().all(|d| d.passed);
    Ok(VerificationResult {
        valid: all_passed && !details.is_empty(),
        issuer_did: record.issuer_did.clone(),
        timestamp_ms: record.timestamp_ms,
        details,
    })
}

/// The `revocation_status` detail of a credential. `lookup(status list URL, index)` answers
/// whether the index is revoked, missing, or outside the known bitstring.
fn revocation_detail(
    credential: &vc::FactCredential,
    require_list: bool,
    lookup: impl Fn(&str, u64) -> StatusLookup,
) -> VerificationDetail {
    let (passed, message) = match credential.credential_status {
        None => (true, None),
        Some(ref status)
            if status.status_type != "BitstringStatusListEntry"
                || status.status_purpose != "revocation" =>
        {
            (
                false,
                Some("unsupported credential status type or purpose".to_string()),
            )
        }
        Some(ref status) => match status.status_list_index.parse::<u64>() {
            Err(_) => (
                false,
                Some(format!(
                    "status list index {:?} is not a number",
                    status.status_list_index
                )),
            ),
            Ok(index) => match lookup(&status.status_list_credential, index) {
                StatusLookup::Value(true) => {
                    (false, Some("credential has been revoked".to_string()))
                }
                StatusLookup::Value(false) => (true, None),
                StatusLookup::OutOfRange => {
                    (false, Some("status list index is out of range".to_string()))
                }
                StatusLookup::Stale => (
                    false,
                    Some("trusted status list is stale or has no signed timestamp".to_string()),
                ),
                StatusLookup::Missing => (
                    !require_list,
                    Some(format!(
                        "no trusted status list for {}; revocation not checked",
                        status.status_list_credential
                    )),
                ),
            },
        },
    };
    VerificationDetail {
        check: "revocation_status".to_string(),
        passed,
        message,
    }
}

/// The fields of a fact claim signed by [`cose::sign_fact_claim`].
struct FactClaim {
    id: String,
    content_hash: [u8; 32],
    timestamp_ms: u64,
    source_uri: Option<String>,
}

/// Decodes a fact claim, rejecting any other signed payload (deletions, key rotations, heads
/// and manifests): unknown or repeated keys, missing fields and wrong types are errors.
fn parse_fact_claim(payload: &[u8]) -> Result<FactClaim> {
    let not_a_claim = || anyhow::anyhow!("signed payload is not a fact claim");
    let value: ciborium::Value = ciborium::from_reader(payload).map_err(|_| not_a_claim())?;
    let ciborium::Value::Map(entries) = value else {
        return Err(not_a_claim());
    };
    let (mut id, mut content_hash, mut timestamp_ms, mut source_uri) = (None, None, None, None);
    for (key, value) in entries {
        let ciborium::Value::Text(key) = key else {
            return Err(not_a_claim());
        };
        let duplicate = match (key.as_str(), value) {
            ("id", ciborium::Value::Text(v)) => id.replace(v).is_some(),
            ("content_hash", ciborium::Value::Bytes(v)) => {
                let hash: [u8; 32] = v.as_slice().try_into().map_err(|_| not_a_claim())?;
                content_hash.replace(hash).is_some()
            }
            ("timestamp_ms", ciborium::Value::Integer(v)) => {
                let ms = u64::try_from(v).map_err(|_| not_a_claim())?;
                timestamp_ms.replace(ms).is_some()
            }
            ("source_uri", ciborium::Value::Text(v)) => source_uri.replace(v).is_some(),
            _ => return Err(not_a_claim()),
        };
        if duplicate {
            return Err(not_a_claim());
        }
    }
    Ok(FactClaim {
        id: id.ok_or_else(not_a_claim)?,
        content_hash: content_hash.ok_or_else(not_a_claim)?,
        timestamp_ms: timestamp_ms.ok_or_else(not_a_claim)?,
        source_uri,
    })
}

/// How a batch's Merkle tree hashes its nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MerkleHashing {
    /// Batch records signed before the tree was versioned: the leaf is the content hash itself
    /// and a parent is SHA-256(left || right). A parent's two children are a 64-byte string
    /// whose SHA-256 is a node of the tree, so those 64 bytes verify as content of the batch.
    /// Their envelopes sign the bare 32-byte root.
    Legacy,
    /// RFC 6962 domain separation: a leaf is SHA-256(0x00 || content hash) and a parent is
    /// SHA-256(0x01 || left || right), so no leaf hash can equal a parent hash. Envelopes sign
    /// a [`batch_root_claim`] naming the version and the number of leaves.
    Rfc6962,
}

impl MerkleHashing {
    fn leaf(self, content_hash: &[u8; 32]) -> [u8; 32] {
        match self {
            Self::Legacy => *content_hash,
            Self::Rfc6962 => Sha256::new()
                .chain_update([0x00])
                .chain_update(content_hash)
                .finalize()
                .into(),
        }
    }

    fn node(self, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let hasher = match self {
            Self::Legacy => Sha256::new(),
            Self::Rfc6962 => Sha256::new().chain_update([0x01]),
        };
        hasher
            .chain_update(left)
            .chain_update(right)
            .finalize()
            .into()
    }
}

/// The `type` of a signed batch root claim.
const BATCH_ROOT_TYPE: &str = "hms-batch-root";
/// The batch root claim version signed with [`MerkleHashing::Rfc6962`] trees.
const BATCH_ROOT_VERSION: u64 = 2;

/// The CBOR payload a batch envelope signs: `{type, v, size, root}`, where `size` is the number
/// of leaves, which fixes the tree's shape.
fn batch_root_claim(root: &[u8; 32], size: usize) -> Result<Vec<u8>> {
    let claim = ciborium::Value::Map(vec![
        (
            ciborium::Value::Text("type".to_string()),
            ciborium::Value::Text(BATCH_ROOT_TYPE.to_string()),
        ),
        (
            ciborium::Value::Text("v".to_string()),
            ciborium::Value::Integer(BATCH_ROOT_VERSION.into()),
        ),
        (
            ciborium::Value::Text("size".to_string()),
            ciborium::Value::Integer((size as u64).into()),
        ),
        (
            ciborium::Value::Text("root".to_string()),
            ciborium::Value::Bytes(root.to_vec()),
        ),
    ]);
    let mut payload = Vec::new();
    ciborium::into_writer(&claim, &mut payload)
        .map_err(|e| anyhow::anyhow!("CBOR encoding failed: {e}"))?;
    Ok(payload)
}

/// Decodes a [`batch_root_claim`] into its root and size, or `None` when the payload is not
/// one (so it can be tried as a fact claim). A payload typed as a batch root with any other
/// version, key or field type is an error.
fn parse_batch_root_claim(payload: &[u8]) -> Result<Option<([u8; 32], u64)>> {
    let Ok(ciborium::Value::Map(entries)) = ciborium::from_reader(payload) else {
        return Ok(None);
    };
    let typed = entries
        .iter()
        .any(|(k, v)| k.as_text() == Some("type") && v.as_text() == Some(BATCH_ROOT_TYPE));
    if !typed {
        return Ok(None);
    }
    let malformed = || anyhow::anyhow!("malformed batch root claim");
    if entries.len() != 4 {
        return Err(malformed());
    }
    let (mut version, mut size, mut root) = (None, None, None);
    for (key, value) in entries {
        match (key.as_text().ok_or_else(malformed)?, value) {
            ("type", _) => {}
            ("v", ciborium::Value::Integer(v)) => version = u64::try_from(v).ok(),
            ("size", ciborium::Value::Integer(v)) => size = u64::try_from(v).ok(),
            ("root", ciborium::Value::Bytes(v)) => root = <[u8; 32]>::try_from(v.as_slice()).ok(),
            _ => return Err(malformed()),
        }
    }
    if version != Some(BATCH_ROOT_VERSION) {
        return Err(anyhow::anyhow!("unsupported batch root claim version"));
    }
    Ok(Some((
        root.ok_or_else(malformed)?,
        size.ok_or_else(malformed)?,
    )))
}

/// The root that a leaf node and its inclusion proof hash up to, as
/// [`ProvenanceManager::merkle_proof`] lays proofs out: each sibling flagged when it sits to
/// the left.
fn merkle_root_from_proof(
    hashing: MerkleHashing,
    leaf_node: [u8; 32],
    proof: &[(bool, [u8; 32])],
) -> [u8; 32] {
    proof
        .iter()
        .fold(leaf_node, |node, (sibling_on_left, sibling)| {
            if *sibling_on_left {
                hashing.node(sibling, &node)
            } else {
                hashing.node(&node, sibling)
            }
        })
}

/// Checks that the content hash of a batch record hashes up to `root` through `proof`. For a
/// versioned root the proof may hold at most one sibling per level of a `size`-leaf tree.
fn check_inclusion(
    hashing: MerkleHashing,
    content_hash: &[u8; 32],
    proof: Option<&[(bool, [u8; 32])]>,
    root: &[u8],
    size: Option<u64>,
) -> Result<()> {
    let proof = proof.ok_or_else(|| anyhow::anyhow!("batch record has no inclusion proof"))?;
    if let Some(size) = size {
        let depth = size
            .checked_next_power_of_two()
            .map_or(64, u64::trailing_zeros);
        if size == 0 || proof.len() > depth as usize {
            return Err(anyhow::anyhow!(
                "inclusion proof does not fit a tree of {size} leaves"
            ));
        }
    }
    if merkle_root_from_proof(hashing, hashing.leaf(content_hash), proof).as_slice() != root {
        return Err(anyhow::anyhow!(
            "content hash does not hash up to the signed batch root"
        ));
    }
    Ok(())
}

/// Checks that a record's signed parts speak about this record and agree with each other.
/// Each signature on its own only proves that the key signed *some* payload, so without this a
/// valid envelope or credential signed for another fact (or a head, deletion or batch root)
/// verifies when pasted into a record.
///
/// - A fact-claim envelope must carry the record's content hash, timestamp and source.
/// - A batch envelope signs a bare 32-byte Merkle root: the record's content hash must hash up
///   to it through the record's inclusion proof, and the record must carry its credential,
///   since a bare hash alone cannot be told apart from a signed log head.
/// - The credential must name the record's content hash, source and issuer, the envelope's
///   signer and the fact id of the envelope's claim.
///
/// `ProvenanceRecord` carries no fact id, so the id is bound across envelope and credential;
/// checking it against the caller's id is up to the caller (`urn:hms:fact:<id>` in the
/// credential subject).
fn check_binding(
    record: &ProvenanceRecord,
    envelope: Option<(&[u8], &ed25519_dalek::VerifyingKey)>,
    credential: Option<&vc::FactCredential>,
) -> Result<()> {
    let source_hash = record
        .source_hash
        .ok_or_else(|| anyhow::anyhow!("record has no content hash"))?;
    let record_uri = record.source_uri.as_deref().unwrap_or("");
    let mut claimed_id = None;
    if let Some((payload, _)) = envelope {
        if payload.len() == 32 {
            check_inclusion(
                MerkleHashing::Legacy,
                &source_hash,
                record.merkle_proof.as_deref(),
                payload,
                None,
            )?;
            if credential.is_none() {
                return Err(anyhow::anyhow!("batch record has no credential"));
            }
        } else if let Some((root, size)) = parse_batch_root_claim(payload)? {
            check_inclusion(
                MerkleHashing::Rfc6962,
                &source_hash,
                record.merkle_proof.as_deref(),
                &root,
                Some(size),
            )?;
        } else {
            let claim = parse_fact_claim(payload)?;
            if record.merkle_proof.is_some() {
                return Err(anyhow::anyhow!(
                    "record carries an inclusion proof but its envelope signs a single fact"
                ));
            }
            if claim.content_hash != source_hash {
                return Err(anyhow::anyhow!(
                    "signed content hash differs from the record's"
                ));
            }
            if claim.timestamp_ms != record.timestamp_ms {
                return Err(anyhow::anyhow!(
                    "signed timestamp differs from the record's"
                ));
            }
            // Records written without a source hold an empty one; their claims omit it.
            if claim.source_uri.as_deref().unwrap_or("") != record_uri {
                return Err(anyhow::anyhow!("signed source differs from the record's"));
            }
            claimed_id = Some(claim.id);
        }
    }
    if let Some(credential) = credential {
        let subject = &credential.credential_subject;
        if subject.content_hash != hex_encode(&source_hash) {
            return Err(anyhow::anyhow!(
                "credential content hash differs from the record's"
            ));
        }
        if subject.source_uri.as_deref().unwrap_or("") != record_uri {
            return Err(anyhow::anyhow!(
                "credential source differs from the record's"
            ));
        }
        if record
            .issuer_did
            .as_deref()
            .is_some_and(|issuer| issuer != credential.issuer)
        {
            return Err(anyhow::anyhow!(
                "credential issuer differs from the record's"
            ));
        }
        if let Some((_, signer)) = envelope {
            if did::did_key_from_ed25519(&signer.to_bytes()) != credential.issuer {
                return Err(anyhow::anyhow!(
                    "envelope and credential are signed by different keys"
                ));
            }
        }
        if let Some(id) = claimed_id {
            if subject.id != format!("urn:hms:fact:{id}") {
                return Err(anyhow::anyhow!(
                    "credential and envelope name different facts"
                ));
            }
        }
    }
    Ok(())
}

fn outcome_detail<T, E: std::fmt::Display>(
    check: &str,
    outcome: &Result<T, E>,
) -> VerificationDetail {
    VerificationDetail {
        check: check.to_string(),
        passed: outcome.is_ok(),
        message: outcome.as_ref().err().map(ToString::to_string),
    }
}

fn trust_anchor_detail(trusted: bool) -> VerificationDetail {
    VerificationDetail {
        check: "trust_anchor".to_string(),
        passed: trusted,
        message: (!trusted).then(|| "signing key is not in the trust store".to_string()),
    }
}

/// Milliseconds since the Unix epoch; 0 if the clock reads before it. `u64` milliseconds last
/// until the year 584 million, so the narrowing cast cannot truncate.
fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Signs `head` with `signing_key` and atomically replaces the log's `.head` file with it.
fn write_signed_head(signing_key: &SigningKey, head: &[u8; 32], log_path: &Path) -> Result<()> {
    let head_path = log_path.with_extension("head");
    let signed = cose::sign_payload(signing_key, head).context("signed head signing failed")?;
    super::durable_file::atomic_write(&head_path, &signed).context("signed head write failed")
}

/// Appends `bytes` to the file at `path` and syncs them to the drive. A failed write or sync
/// truncates the file back to its prior length, so later lines never chain after a torn one.
fn append_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let prior = file.metadata()?.len();
    let written = file
        .write_all(bytes)
        .and_then(|()| file.sync_data())
        .and_then(|()| super::durable_file::sync_mapped_file(&file));
    if let Err(e) = written {
        if let Err(truncate) = file.set_len(prior) {
            tracing::warn!("could not drop a partial provenance log append: {truncate}");
        }
        return Err(e.into());
    }
    if prior == 0 {
        // The file may be new: its directory entry must be durable too.
        super::durable_file::sync_directory(path.parent().unwrap_or_else(|| Path::new(".")))?;
    }
    Ok(())
}

/// Lowercase hex, two digits per byte.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(hex: &str) -> Result<[u8; 32]> {
    if hex.len() != 64 {
        return Err(anyhow::anyhow!("invalid hex hash length"));
    }
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        out[i] = u8::from_str_radix(
            std::str::from_utf8(chunk).map_err(|e| anyhow::anyhow!("hex decode: {e}"))?,
            16,
        )
        .map_err(|e| anyhow::anyhow!("hex decode: {e}"))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{tempdir, TempDir};

    /// A manager with a fresh key, logging under its own temporary directory.
    fn manager() -> (TempDir, ProvenanceManager) {
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        (dir, mgr)
    }

    #[test]
    fn full_fact_provenance_flow() {
        let (_dir, mgr) = manager();

        let record = mgr
            .create_fact_provenance(
                "fact-001",
                b"Paris is the capital of France",
                Some("https://example.com"),
            )
            .unwrap();

        assert!(record.cose_envelope.is_some());
        assert!(record.vc_json.is_some());
        assert!(record.issuer_did.is_some());
        assert!(record.issuer_did.as_ref().unwrap().starts_with("did:key:z"));

        let verification = mgr
            .verify_fact_provenance(&record, &mgr.self_trust())
            .unwrap();
        assert!(verification.valid);
        // cose_signature, vc_proof, vc_semantics, revocation_status, record_binding, trust_anchor
        assert_eq!(verification.details.len(), 6);

        assert!(mgr.get_record("fact-001").is_some());
        assert_eq!(mgr.record_count(), 1);
    }

    #[test]
    fn full_triple_provenance_flow() {
        let (_dir, mgr) = manager();

        let record = mgr
            .create_triple_provenance(&TripleProvenanceParams {
                fact_id: "triple-001",
                content: b"paris capital_of france",
                dimensions: 16384,
                subject: "paris",
                relation: "capital_of",
                object: "france",
                source_uri: None,
            })
            .unwrap();

        let vc: vc::FactCredential =
            serde_json::from_str(record.vc_json.as_ref().unwrap()).unwrap();
        assert_eq!(vc.credential_subject.subject_id.as_deref(), Some("paris"));

        let verification = mgr
            .verify_fact_provenance(&record, &mgr.self_trust())
            .unwrap();
        assert!(verification.valid);
        assert!(mgr.get_record("triple-001").is_some());
    }

    #[test]
    fn store_manifest_flow() {
        let (_dir, mgr) = manager();

        let store_data = b"serialized store contents here";
        let manifest = mgr
            .create_store_manifest("store-001", store_data, 42, 16384, Some("Test Store"))
            .unwrap();

        assert!(manifest.cose_envelope.is_some());

        let verification = mgr
            .verify_store_manifest(&manifest, Some(store_data), &mgr.self_trust())
            .unwrap();
        assert!(verification.valid);
        assert!(verification
            .details
            .iter()
            .any(|d| d.check == "store_hash" && d.passed));

        let bad_verification = mgr
            .verify_store_manifest(&manifest, Some(b"wrong"), &mgr.self_trust())
            .unwrap();
        assert!(!bad_verification.valid);
    }

    #[test]
    fn audit_deleted_issuances_never_reuse_a_status_slot_after_reopen() {
        let dir = tempdir().unwrap();
        let key = dir.path().join("test.key");
        let last = {
            let mgr = ProvenanceManager::new(&key, Some(dir.path())).unwrap();
            let records = mgr
                .create_batch_provenance(&[("a", b"a", None), ("b", b"b", None)])
                .unwrap();
            for (id, record) in ["a", "b"].iter().zip(&records) {
                mgr.revoke_credential(record.sequence).unwrap();
                mgr.record_deletion(id, None).unwrap();
            }
            records[1].sequence
        };
        let mgr = ProvenanceManager::new(&key, Some(dir.path())).unwrap();
        let record = mgr.create_fact_provenance("c", b"c", None).unwrap();
        assert!(record.sequence > last);
        assert!(!mgr.is_revoked(record.sequence));
        assert!(verify_record(&record, &mgr.self_trust()).unwrap().valid);
    }

    #[test]
    fn audit_sequence_reservation_fails_without_wrapping() {
        let (_, mgr) = manager();
        mgr.logical_clock
            .store(u64::MAX, std::sync::atomic::Ordering::SeqCst);
        assert!(mgr.create_fact_provenance("a", b"a", None).is_err());
        assert!(mgr
            .create_batch_provenance(&[("b", b"b", None), ("c", b"c", None)])
            .is_err());
    }

    #[test]
    fn records_persist_to_disk() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("test.key");

        let issuer = {
            let mgr = ProvenanceManager::new(&key_path, Some(dir.path())).unwrap();
            mgr.create_fact_provenance("f1", b"content one", None)
                .unwrap();
            mgr.create_fact_provenance("f2", b"content two", None)
                .unwrap();
            assert_eq!(mgr.record_count(), 2);
            mgr.issuer_did().to_string()
        };

        let mgr2 = ProvenanceManager::new(&key_path, Some(dir.path())).unwrap();
        assert_eq!(mgr2.issuer_did(), issuer);
        assert_eq!(mgr2.record_count(), 2);
        assert!(mgr2.get_record("f1").is_some());
        assert!(mgr2.get_record("f2").is_some());
    }

    #[test]
    fn standalone_verify_without_manager() {
        let (_dir, mgr) = manager();
        let record = mgr
            .create_fact_provenance("standalone-001", b"verify without manager", None)
            .unwrap();
        let issuer = record.issuer_did.clone().unwrap();
        drop(mgr);

        let trust = trust::TrustStore::trusting_did(&issuer).unwrap();
        let result = verify_record(&record, &trust).unwrap();
        assert!(result.valid);
        // cose_signature, vc_proof, vc_semantics, revocation_status (unchecked: no list trusted),
        // record_binding, trust_anchor
        assert_eq!(result.details.len(), 6);
    }

    #[test]
    fn tampered_log_detected() {
        let (dir, mgr) = manager();
        mgr.create_fact_provenance("t1", b"one", None).unwrap();
        mgr.create_fact_provenance("t2", b"two", None).unwrap();
        assert!(mgr.verify_log_integrity().unwrap());
        drop(mgr);

        let log_path = dir.path().join("provenance_log.jsonl");
        let contents = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        let tampered = lines[1].replace("\"t2\"", "\"TAMPERED\"");
        std::fs::write(&log_path, format!("{}\n{tampered}\n", lines[0])).unwrap();

        let mgr2 = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        assert!(!mgr2.verify_log_integrity().unwrap());
    }

    #[test]
    fn absent_history_is_not_authenticated_history() {
        let (dir, mgr) = manager();
        assert_eq!(mgr.log_integrity().unwrap(), LogIntegrity::Empty);
        assert!(!mgr.verify_log_integrity().unwrap());
        std::fs::write(dir.path().join("provenance_log.jsonl"), "").unwrap();
        assert_eq!(mgr.log_integrity().unwrap(), LogIntegrity::Empty);
        let volatile = ProvenanceManager::from_signing_key(mgr.signing_key.clone());
        assert_eq!(volatile.log_integrity().unwrap(), LogIntegrity::Volatile);
        assert!(!volatile.verify_log_integrity().unwrap());
    }

    #[test]
    fn missing_truncated_or_reordered_evidence_never_verifies() {
        for damage in [
            "empty",
            "missing",
            "head",
            "both",
            "truncate",
            "reorder",
            "signature",
        ] {
            let (dir, mgr) = manager();
            mgr.create_fact_provenance("one", b"one", None).unwrap();
            mgr.create_fact_provenance("two", b"two", None).unwrap();
            assert_eq!(mgr.log_integrity().unwrap(), LogIntegrity::Verified);
            let log = dir.path().join("provenance_log.jsonl");
            let head = log.with_extension("head");
            let contents = std::fs::read_to_string(&log).unwrap();
            let lines: Vec<_> = contents.lines().collect();
            match damage {
                "empty" => std::fs::write(&log, "").unwrap(),
                "missing" => std::fs::remove_file(&log).unwrap(),
                "head" => std::fs::remove_file(&head).unwrap(),
                "both" => {
                    std::fs::remove_file(&log).unwrap();
                    std::fs::remove_file(&head).unwrap();
                }
                "truncate" => std::fs::write(&log, format!("{}\n", lines[0])).unwrap(),
                "reorder" => std::fs::write(&log, format!("{}\n{}\n", lines[1], lines[0])).unwrap(),
                "signature" => std::fs::write(&head, b"invalid signature").unwrap(),
                _ => unreachable!(),
            }
            assert!(!mgr.verify_log_integrity().unwrap(), "{damage}");
            drop(mgr);
            let reopened =
                ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
            assert!(
                !reopened.verify_log_integrity().unwrap(),
                "reopened {damage}"
            );
        }
    }

    #[test]
    fn unreadable_log_or_head_returns_an_error() {
        for filename in ["provenance_log.jsonl", "provenance_log.head"] {
            let (dir, mgr) = manager();
            mgr.create_fact_provenance("one", b"one", None).unwrap();
            let path = dir.path().join(filename);
            std::fs::remove_file(&path).unwrap();
            std::fs::create_dir(&path).unwrap();
            assert!(mgr.verify_log_integrity().is_err(), "{filename}");
        }
    }

    #[test]
    fn failed_signed_head_reports_committed_state_and_preserves_replay() {
        for operation in ["create", "delete", "batch", "rotate"] {
            let (dir, mut mgr) = manager();
            mgr.create_fact_provenance("one", b"one", None).unwrap();
            let head = dir.path().join("provenance_log.head");
            std::fs::remove_file(&head).unwrap();
            std::fs::create_dir(&head).unwrap();
            let new_key = dir.path().join("rotated.key");
            let error = match operation {
                "create" => mgr.create_fact_provenance("two", b"two", None).unwrap_err(),
                "delete" => mgr.record_deletion("one", None).unwrap_err(),
                "batch" => mgr
                    .create_batch_provenance(&[("two", b"two", None)])
                    .unwrap_err(),
                "rotate" => mgr.rotate_key(&new_key).unwrap_err(),
                _ => unreachable!(),
            };
            assert!(
                error.is::<crate::CommittedMutationError>(),
                "{operation}: {error:#}"
            );
            assert!(mgr.verify_log_integrity().is_err());
            let before = mgr.record_count();
            let issuer = mgr.issuer_did().to_string();
            assert_eq!(mgr.get_record("one").is_some(), operation != "delete");
            assert_eq!(
                mgr.get_record("two").is_some(),
                matches!(operation, "create" | "batch")
            );
            drop(mgr);
            let key = if operation == "rotate" {
                new_key
            } else {
                dir.path().join("test.key")
            };
            let reopened = ProvenanceManager::new(&key, Some(dir.path())).unwrap();
            assert_eq!(reopened.record_count(), before);
            assert_eq!(reopened.issuer_did(), issuer);
            std::fs::remove_dir(&head).unwrap();
            assert_eq!(reopened.log_integrity().unwrap(), LogIntegrity::Unanchored);
            reopened
                .create_fact_provenance("next", b"next", None)
                .unwrap();
            assert!(reopened.verify_log_integrity().unwrap());
        }
    }

    #[test]
    fn key_rotation_preserves_old_records() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("orig.key");
        let mut mgr = ProvenanceManager::new(&key_path, Some(dir.path())).unwrap();
        let old_did = mgr.issuer_did().to_string();

        let record = mgr
            .create_fact_provenance("before-rotate", b"old key", None)
            .unwrap();

        let new_key_path = dir.path().join("rotated.key");
        let new_did = mgr.rotate_key(&new_key_path).unwrap();
        assert_ne!(old_did, new_did);
        assert_eq!(mgr.issuer_did(), new_did);

        let post_record = mgr
            .create_fact_provenance("after-rotate", b"new key", None)
            .unwrap();
        assert_eq!(post_record.issuer_did.as_deref(), Some(new_did.as_str()));

        let mut trust = trust::TrustStore::new();
        trust.trust_did(&old_did).unwrap();
        trust.trust_did(&new_did).unwrap();

        let old_result = verify_record(&record, &trust).unwrap();
        assert!(old_result.valid);

        let new_result = verify_record(&post_record, &trust).unwrap();
        assert!(new_result.valid);

        assert!(mgr.verify_log_integrity().unwrap());
    }

    #[test]
    fn batch_provenance_merkle() {
        let (_dir, mgr) = manager();

        let items: Vec<(&str, &[u8], Option<&str>)> = vec![
            ("b1", b"alpha", None),
            ("b2", b"beta", None),
            ("b3", b"gamma", Some("https://example.com")),
            ("b4", b"delta", None),
        ];

        let records = mgr.create_batch_provenance(&items).unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!(mgr.record_count(), 4);

        let envelope = records[0].cose_envelope.as_ref().unwrap();
        for r in &records {
            assert_eq!(r.cose_envelope.as_ref().unwrap(), envelope);
            assert!(r.merkle_proof.is_some());
            assert!(r.vc_json.is_some());
        }

        let trust = mgr.self_trust();
        for r in &records {
            let result = verify_record(r, &trust).unwrap();
            assert!(result
                .details
                .iter()
                .any(|d| d.check == "vc_proof" && d.passed));
        }
    }

    #[test]
    fn credential_revocation() {
        let (_dir, mgr) = manager();

        let record = mgr
            .create_fact_provenance("fact-rev", b"revocable content", None)
            .unwrap();

        let verification = mgr
            .verify_fact_provenance(&record, &mgr.self_trust())
            .unwrap();
        assert!(verification.valid);
        assert!(verification
            .details
            .iter()
            .any(|d| d.check == "revocation_status" && d.passed));

        let vc: vc::FactCredential =
            serde_json::from_str(record.vc_json.as_ref().unwrap()).unwrap();
        let status_index: u64 = vc
            .credential_status
            .as_ref()
            .unwrap()
            .status_list_index
            .parse()
            .unwrap();
        mgr.revoke_credential(status_index).unwrap();
        assert!(mgr.is_revoked(status_index));

        let verification = mgr
            .verify_fact_provenance(&record, &mgr.self_trust())
            .unwrap();
        assert!(!verification.valid);
        assert!(verification
            .details
            .iter()
            .any(|d| d.check == "revocation_status" && !d.passed));
    }

    #[test]
    fn unreadable_provenance_state_fails_to_open() {
        for name in ["provenance_log.jsonl", "provenance_log.statuslist"] {
            let dir = tempdir().unwrap();
            std::fs::create_dir(dir.path().join(name)).unwrap();
            assert!(
                ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).is_err(),
                "{name}"
            );
        }
    }

    #[test]
    fn failed_log_append_leaves_head_and_records_unchanged() {
        let (dir, mgr) = manager();
        mgr.create_fact_provenance("a", b"first", None).unwrap();
        let log = dir.path().join("provenance_log.jsonl");
        let head_file = log.with_extension("head");
        let head = *mgr.chain_head.lock();
        let signed = std::fs::read(&head_file).unwrap();

        let aside = dir.path().join("aside.jsonl");
        std::fs::rename(&log, &aside).unwrap();
        std::fs::create_dir(&log).unwrap();
        assert!(mgr.create_fact_provenance("b", b"second", None).is_err());
        assert!(mgr.record_deletion("a", None).is_err());
        assert_eq!(*mgr.chain_head.lock(), head);
        assert_eq!(std::fs::read(&head_file).unwrap(), signed);
        assert!(mgr.get_record("b").is_none());
        assert!(mgr.get_record("a").is_some());

        std::fs::remove_dir(&log).unwrap();
        std::fs::rename(&aside, &log).unwrap();
        mgr.create_fact_provenance("c", b"third", None).unwrap();
        assert!(mgr.verify_log_integrity().unwrap());
        drop(mgr);
        let reopened =
            ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        assert!(reopened.verify_log_integrity().unwrap());
        assert!(reopened.get_record("a").is_some());
        assert!(reopened.get_record("b").is_none());
        assert!(reopened.get_record("c").is_some());
    }

    #[test]
    fn batch_log_chains_and_reloads() {
        let (dir, mgr) = manager();
        mgr.create_fact_provenance("single", b"before", None)
            .unwrap();
        let items: Vec<(&str, &[u8], Option<&str>)> = vec![
            ("x", b"one", None),
            ("y", b"two", Some("u")),
            ("z", b"three", None),
        ];
        mgr.create_batch_provenance(&items).unwrap();
        assert!(mgr.verify_log_integrity().unwrap());
        let log = std::fs::read_to_string(dir.path().join("provenance_log.jsonl")).unwrap();
        assert_eq!(log.lines().count(), 4);
        drop(mgr);
        let reopened =
            ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        assert!(reopened.verify_log_integrity().unwrap());
        assert_eq!(reopened.record_count(), 4);
        let two: [u8; 32] = Sha256::digest(b"two").into();
        assert_eq!(reopened.get_record("y").unwrap().source_hash, Some(two));
    }

    /// The per-leaf tree rebuild the shared-levels proofs replaced.
    fn reference_proof(leaves: &[[u8; 32]], index: usize) -> Vec<(bool, [u8; 32])> {
        let mut proof = Vec::new();
        let mut level: Vec<[u8; 32]> = leaves.to_vec();
        let mut idx = index;
        while level.len() > 1 {
            let sibling_idx = if idx.is_multiple_of(2) {
                idx + 1
            } else {
                idx - 1
            };
            if sibling_idx < level.len() {
                proof.push((idx % 2 == 1, level[sibling_idx]));
            }
            level = ProvenanceManager::merkle_parents(&level, MerkleHashing::Rfc6962);
            idx /= 2;
        }
        proof
    }

    #[test]
    fn merkle_proofs_match_per_leaf_rebuild() {
        for n in 1..=65u32 {
            let leaves: Vec<[u8; 32]> = (0..n)
                .map(|i| Sha256::digest(i.to_le_bytes()).into())
                .collect();
            let levels = ProvenanceManager::merkle_levels(&leaves, MerkleHashing::Rfc6962);
            let mut root = leaves.clone();
            while root.len() > 1 {
                root = ProvenanceManager::merkle_parents(&root, MerkleHashing::Rfc6962);
            }
            assert_eq!(levels[levels.len() - 1], root, "n={n}");
            for index in 0..leaves.len() {
                let proof = ProvenanceManager::merkle_proof(&levels, index);
                assert_eq!(
                    proof,
                    reference_proof(&leaves, index),
                    "n={n} index={index}"
                );
                assert_eq!(
                    merkle_root_from_proof(MerkleHashing::Rfc6962, leaves[index], &proof),
                    root[0],
                    "n={n} index={index}"
                );
            }
        }
    }

    #[test]
    fn revocation_survives_reopen() {
        let dir = tempdir().unwrap();
        let key = dir.path().join("test.key");
        let mgr = ProvenanceManager::new(&key, Some(dir.path())).unwrap();
        mgr.revoke_credential(13).unwrap();
        drop(mgr);
        let reopened = ProvenanceManager::new(&key, Some(dir.path())).unwrap();
        assert!(reopened.is_revoked(13));
        assert!(!reopened.is_revoked(12));
        assert!(!reopened.is_revoked(14));
    }

    /// A revocation that cannot be persisted is not applied, so the live list never claims
    /// a revocation a restart would undo.
    #[test]
    fn failed_revocation_write_leaves_the_credential_live() {
        let dir = tempdir().unwrap();
        let key = dir.path().join("test.key");
        let mgr = ProvenanceManager::new(&key, Some(dir.path())).unwrap();
        mgr.revoke_credential(1).unwrap();
        let list = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.extension().is_some_and(|e| e == "statuslist"))
            .unwrap();
        std::fs::remove_file(&list).unwrap();
        std::fs::create_dir(&list).unwrap();
        std::fs::write(list.join("occupied"), b"x").unwrap();
        assert!(mgr.revoke_credential(13).is_err());
        assert!(!mgr.is_revoked(13));
        assert!(mgr.is_revoked(1));
    }

    /// Concurrent revocations all reach the persisted list: none is overwritten by a write of
    /// an older list.
    #[test]
    fn concurrent_revocations_all_persist() {
        let dir = tempdir().unwrap();
        let key = dir.path().join("test.key");
        let mgr = ProvenanceManager::new(&key, Some(dir.path())).unwrap();
        std::thread::scope(|scope| {
            for thread in 0..8u64 {
                let mgr = &mgr;
                scope.spawn(move || {
                    for i in 0..32 {
                        mgr.revoke_credential(thread * 1000 + i).unwrap();
                    }
                });
            }
        });
        drop(mgr);
        let reopened = ProvenanceManager::new(&key, Some(dir.path())).unwrap();
        for thread in 0..8u64 {
            for i in 0..32 {
                assert!(reopened.is_revoked(thread * 1000 + i));
            }
        }
    }

    #[test]
    fn unallocatable_status_index_is_an_error() {
        let (_dir, mgr) = manager();
        assert!(mgr.revoke_credential(u64::MAX).is_err());
        assert!(!mgr.is_revoked(u64::MAX));
        mgr.revoke_credential(9).unwrap();
        assert!(mgr.is_revoked(9));
        assert!(!mgr.is_revoked(8));
    }

    #[test]
    fn untrusted_key_rejected() {
        let (_dir, mgr) = manager();
        let record = mgr
            .create_fact_provenance("fact-trust", b"authentic content", None)
            .unwrap();

        // Signed by the manager and trusting the manager: authentic.
        let ok = mgr
            .verify_fact_provenance(&record, &mgr.self_trust())
            .unwrap();
        assert!(ok.valid);

        // The record's signature is internally valid, but a trust store that
        // does not contain the signing key must reject it as non-authentic,
        // even though every signature check passes.
        let stranger = SigningKey::from_bytes(&rand::random());
        let foreign_trust = trust::TrustStore::trusting_key(&stranger.verifying_key());
        let rejected = mgr.verify_fact_provenance(&record, &foreign_trust).unwrap();
        assert!(!rejected.valid);
        assert!(rejected
            .details
            .iter()
            .any(|d| d.check == "trust_anchor" && !d.passed));

        // Self-consistency, by contrast, still passes: the record is intact.
        let self_consistent = mgr
            .verify_fact_provenance_self_consistency(&record)
            .unwrap();
        assert!(self_consistent.valid);
    }

    #[test]
    fn vc_only_record_requires_trust() {
        // A record carrying only a self-signed VC (no COSE envelope) must still
        // be trust-anchored via its issuer DID, not treated as authentic just
        // because the VC verifies against its own embedded issuer.
        let (_dir, mgr) = manager();
        let record = mgr
            .create_fact_provenance("vc-only", b"content", None)
            .unwrap();

        let mut vc_only = record.clone();
        vc_only.cose_envelope = None;

        let ok = verify_record(&vc_only, &mgr.self_trust()).unwrap();
        assert!(ok.valid);
        assert!(ok
            .details
            .iter()
            .any(|d| d.check == "trust_anchor" && d.passed));

        let stranger = SigningKey::from_bytes(&rand::random());
        let foreign = trust::TrustStore::trusting_key(&stranger.verifying_key());
        let rejected = verify_record(&vc_only, &foreign).unwrap();
        assert!(!rejected.valid);
        assert!(rejected
            .details
            .iter()
            .any(|d| d.check == "trust_anchor" && !d.passed));
    }

    #[test]
    fn cawg_untrusted_issuer_rejected() {
        let (_dir, mgr) = manager();

        let referenced = vec![(
            "self#jumbf=c2pa.assertions/c2pa.hash.data".to_string(),
            "sha256".to_string(),
            vec![0x11u8; 32],
        )];
        let assertion = mgr.create_cawg_assertion(&referenced, "hms agent").unwrap();

        // The manager's own CAWG issuer (did:jwk of its key) is trusted by self_trust.
        assert!(mgr
            .verify_cawg_assertion(&assertion, &mgr.self_trust())
            .is_ok());

        // A trust store that does not contain the issuer must reject it, even
        // though the assertion is internally self-consistent.
        let stranger = SigningKey::from_bytes(&rand::random());
        let foreign = trust::TrustStore::trusting_key(&stranger.verifying_key());
        assert!(mgr.verify_cawg_assertion(&assertion, &foreign).is_err());
    }
}

#[cfg(test)]
mod binding_tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::{tempdir, TempDir};

    fn manager() -> (TempDir, ProvenanceManager) {
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        (dir, mgr)
    }

    fn assert_rejected(mgr: &ProvenanceManager, record: &ProvenanceRecord, what: &str) {
        let trust = mgr.self_trust();
        let own = mgr.verify_fact_provenance(record, &trust).unwrap();
        assert!(!own.valid, "manager accepted {what}: {:?}", own.details);
        let free = verify_record(record, &trust).unwrap();
        assert!(
            !free.valid,
            "verify_record accepted {what}: {:?}",
            free.details
        );
    }

    fn assert_accepted(mgr: &ProvenanceManager, record: &ProvenanceRecord, what: &str) {
        let trust = mgr.self_trust();
        let own = mgr.verify_fact_provenance(record, &trust).unwrap();
        assert!(own.valid, "manager rejected {what}: {:?}", own.details);
        let free = verify_record(record, &trust).unwrap();
        assert!(
            free.valid,
            "verify_record rejected {what}: {:?}",
            free.details
        );
    }

    #[test]
    fn envelopes_signed_for_something_else_do_not_verify() {
        let (dir, mgr) = manager();
        let a = mgr.create_fact_provenance("a", b"alpha", None).unwrap();
        let b = mgr
            .create_fact_provenance("b", b"beta", Some("https://example.com/b"))
            .unwrap();
        assert_accepted(&mgr, &a, "an untouched fact record");
        assert_accepted(&mgr, &b, "an untouched fact record with a source");

        let deletion = mgr.record_deletion("gone", None).unwrap();
        let head = std::fs::read(dir.path().join("provenance_log.head")).unwrap();
        let items: Vec<(&str, &[u8], Option<&str>)> = vec![("x", b"x", None), ("y", b"y", None)];
        let batch = mgr.create_batch_provenance(&items).unwrap();

        let foreign = [
            ("another fact's envelope", b.cose_envelope.clone().unwrap()),
            ("a deletion envelope", deletion.cose_envelope.unwrap()),
            ("the signed log head", head),
            (
                "a batch root envelope",
                batch[0].cose_envelope.clone().unwrap(),
            ),
        ];
        for (what, envelope) in foreign {
            let mut forged = a.clone();
            forged.cose_envelope = Some(envelope);
            assert_rejected(&mgr, &forged, what);
        }

        let mut forged = a.clone();
        forged.source_hash = Some([0x55; 32]);
        assert_rejected(&mgr, &forged, "a record whose content hash was replaced");

        let mut forged = a.clone();
        forged.vc_json = b.vc_json.clone();
        assert_rejected(&mgr, &forged, "another fact's credential");

        let mut forged = a.clone();
        forged.timestamp_ms += 1;
        assert_rejected(&mgr, &forged, "a record whose timestamp was changed");

        let mut forged = b.clone();
        forged.source_uri = Some("https://example.com/other".to_string());
        assert_rejected(&mgr, &forged, "a record whose source was changed");
    }

    #[test]
    fn manifests_verify_only_with_their_own_signed_manifest() {
        let (dir, mgr) = manager();
        let trust = mgr.self_trust();
        let manifest = mgr
            .create_store_manifest("s1", b"store one", 1, 1024, None)
            .unwrap();
        let other = mgr
            .create_store_manifest("s2", b"store two", 1, 1024, None)
            .unwrap();
        assert!(
            mgr.verify_store_manifest(&manifest, None, &trust)
                .unwrap()
                .valid
        );
        assert!(
            mgr.verify_store_manifest(&manifest, Some(b"store one"), &trust)
                .unwrap()
                .valid
        );

        let fact = mgr.create_fact_provenance("f", b"fact", None).unwrap();
        let head = std::fs::read(dir.path().join("provenance_log.head")).unwrap();
        let foreign = [
            (
                "another store's manifest envelope",
                other.cose_envelope.clone().unwrap(),
            ),
            ("a fact envelope", fact.cose_envelope.unwrap()),
            ("the signed log head", head),
        ];
        for (what, envelope) in foreign {
            let mut forged = manifest.clone();
            forged.cose_envelope = Some(envelope);
            let result = mgr.verify_store_manifest(&forged, None, &trust).unwrap();
            assert!(!result.valid, "accepted {what}: {:?}", result.details);
        }

        let mut forged = manifest.clone();
        forged.store_hash = other.store_hash;
        assert!(
            !mgr.verify_store_manifest(&forged, None, &trust)
                .unwrap()
                .valid
        );
    }

    #[test]
    fn batch_records_verify_only_with_their_own_inclusion_proof() {
        let (_dir, mgr) = manager();
        let items: Vec<(&str, &[u8], Option<&str>)> = vec![
            ("b1", b"alpha", None),
            ("b2", b"beta", None),
            ("b3", b"gamma", Some("https://example.com")),
            ("b4", b"delta", None),
            ("b5", b"epsilon", None),
        ];
        let records = mgr.create_batch_provenance(&items).unwrap();
        for r in &records {
            assert_accepted(&mgr, r, "an untouched batch record");
        }

        let mut forged = records[0].clone();
        forged.merkle_proof = records[1].merkle_proof.clone();
        assert_rejected(&mgr, &forged, "a batch record with a sibling's proof");

        let mut forged = records[0].clone();
        forged.merkle_proof = None;
        assert_rejected(&mgr, &forged, "a batch record without a proof");

        let mut forged = records[0].clone();
        forged.merkle_proof.as_mut().unwrap().pop();
        assert_rejected(&mgr, &forged, "a batch record with a truncated proof");

        let mut forged = records[2].clone();
        forged.source_hash = Some(Sha256::digest(b"not in the batch").into());
        assert_rejected(
            &mgr,
            &forged,
            "a batch record whose content is not in the tree",
        );

        // A versioned root cannot be mistaken for anything else the key signs, so the envelope
        // and proof alone prove the content is in the batch.
        let mut cose_only = records[0].clone();
        cose_only.vc_json = None;
        assert_accepted(
            &mgr,
            &cose_only,
            "a batch record with only its envelope and proof",
        );
    }

    #[test]
    fn verify_record_honors_the_trusted_status_list() {
        let (_dir, mgr) = manager();
        let record = mgr.create_fact_provenance("r", b"revocable", None).unwrap();
        assert!(verify_record(&record, &mgr.self_trust()).unwrap().valid);

        let credential: vc::FactCredential =
            serde_json::from_str(record.vc_json.as_ref().unwrap()).unwrap();
        let index: u64 = credential
            .credential_status
            .unwrap()
            .status_list_index
            .parse()
            .unwrap();
        mgr.revoke_credential(index).unwrap();

        let result = verify_record(&record, &mgr.self_trust()).unwrap();
        assert!(!result.valid);
        assert!(result
            .details
            .iter()
            .any(|d| d.check == "revocation_status" && !d.passed));
    }

    #[test]
    fn published_status_list_covers_issued_credentials_and_requires_trust() {
        let key = SigningKey::from_bytes(&[31u8; 32]);
        let mgr = ProvenanceManager::from_signing_key(key)
            .with_status_list_url("https://example.com/credentials/status/1".into())
            .unwrap();
        let record = mgr
            .create_fact_provenance("fact", b"revocable", None)
            .unwrap();
        let credential: vc::FactCredential =
            serde_json::from_str(record.vc_json.as_ref().unwrap()).unwrap();
        let status = credential.credential_status.unwrap();
        assert_eq!(status.status_list_credential, mgr.status_list_url);
        let index: u64 = status.status_list_index.parse().unwrap();
        mgr.revoke_credential(index).unwrap();

        let published = mgr.status_list_credential().unwrap();
        let mut untrusted = TrustStore::new();
        assert!(untrusted.trust_status_list_credential(&published).is_err());
        let mut trust = TrustStore::trusting_did(mgr.issuer_did()).unwrap();
        assert!(trust.trust_status_list_credential(&published).is_err());
        trust
            .authorize_status_list_publisher(&published.id, &mgr.signing_key.verifying_key())
            .unwrap();
        trust.trust_status_list_credential(&published).unwrap();
        assert_eq!(
            trust.is_revoked(&status.status_list_credential, index),
            Some(true)
        );
        assert!(!verify_record(&record, &trust).unwrap().valid);
    }

    #[test]
    fn reopening_rejects_a_different_published_status_url() {
        let (dir, mgr) = manager();
        let mgr = mgr
            .with_status_list_url("https://example.com/status/1".into())
            .unwrap();
        mgr.create_fact_provenance("fact", b"content", None)
            .unwrap();
        drop(mgr);
        let reopened =
            ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        assert!(reopened
            .with_status_list_url("https://example.com/status/2".into())
            .is_err());
        let reopened =
            ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        assert!(reopened
            .with_status_list_url("https://example.com/status/1".into())
            .is_ok());
    }

    /// Without a trusted status list a credential passes its revocation check unchecked by
    /// default, and fails it when the trust store requires status lists.
    #[test]
    fn strict_trust_store_fails_unknown_revocation_status() {
        let (_dir, mgr) = manager();
        let record = mgr.create_fact_provenance("r", b"strict", None).unwrap();
        let mut trust = TrustStore::trusting_did(mgr.issuer_did()).unwrap();
        assert!(verify_record(&record, &trust).unwrap().valid);
        trust.require_status_lists(true);
        let result = verify_record(&record, &trust).unwrap();
        assert!(!result.valid);
        assert!(result
            .details
            .iter()
            .any(|d| d.check == "revocation_status" && !d.passed));
        assert!(
            verify_record(&record, mgr.self_trust().require_status_lists(true))
                .unwrap()
                .valid
        );
    }

    #[test]
    fn status_lookup_rejects_indices_outside_the_published_list() {
        let key = SigningKey::from_bytes(&[43u8; 32]);
        let did = did::did_key_from_ed25519(&key.verifying_key().to_bytes());
        let url = "https://example.com/status/1";
        let mut credential = vc::create_fact_credential(&did, "fact", &[0u8; 32], 1, None, None, 0)
            .with_status_list_url(url);
        let mut trust = TrustStore::new();
        trust.trust_status_list(url, vec![0; status_list::MIN_BYTES]);
        let last = (status_list::MIN_BYTES * 8 - 1) as u64;
        for (index, expected) in [(last - 1, true), (last, true), (last + 1, false)] {
            credential
                .credential_status
                .as_mut()
                .unwrap()
                .status_list_index = index.to_string();
            let result = revocation_detail(&credential, false, |url, index| {
                trust.status_lookup(url, index)
            });
            assert_eq!(result.passed, expected, "index {index}");
        }
        credential
            .credential_status
            .as_mut()
            .unwrap()
            .status_list_index = last.to_string();
        credential
            .credential_status
            .as_mut()
            .unwrap()
            .status_purpose = "suspension".into();
        assert!(
            !revocation_detail(&credential, false, |url, index| {
                trust.status_lookup(url, index)
            })
            .passed
        );
    }

    #[test]
    fn record_verification_rejects_unsigned_credential_extensions() {
        let (_dir, mgr) = manager();
        let mut record = mgr
            .create_fact_provenance("fact", b"content", None)
            .unwrap();
        let mut document: serde_json::Value =
            serde_json::from_str(record.vc_json.as_ref().unwrap()).unwrap();
        document["credentialSubject"]["admin"] = serde_json::Value::Bool(true);
        record.vc_json = Some(serde_json::to_string(&document).unwrap());
        let result = verify_record(&record, &mgr.self_trust()).unwrap();
        assert!(!result.valid);
        assert!(result
            .details
            .iter()
            .any(|detail| detail.check == "vc_proof" && !detail.passed));
    }

    #[test]
    fn signed_credential_semantics_and_status_snapshot_freshness_are_reported() {
        let (_dir, mgr) = manager();
        let mgr = mgr
            .with_status_list_url("https://example.com/status/1".into())
            .unwrap();
        let mut record = mgr
            .create_fact_provenance("fact", b"content", None)
            .unwrap();
        let mut credential: vc::FactCredential =
            serde_json::from_str(record.vc_json.as_ref().unwrap()).unwrap();
        credential.extra.insert(
            "validUntil".into(),
            serde_json::json!("2020-01-01T00:00:00Z"),
        );
        let credential = vc::sign_credential(&mgr.signing_key, credential).unwrap();
        record.vc_json = Some(serde_json::to_string(&credential).unwrap());
        let checked = verify_record(&record, &mgr.self_trust()).unwrap();
        assert!(checked
            .details
            .iter()
            .any(|d| d.check == "vc_proof" && d.passed));
        assert!(checked
            .details
            .iter()
            .any(|d| d.check == "vc_semantics" && !d.passed));

        let original = mgr
            .create_fact_provenance("freshness", b"content", None)
            .unwrap();
        let mut list = mgr.status_list_credential().unwrap();
        list.valid_from = "2020-01-01T00:00:00Z".into();
        list.proof = None;
        list.proof = Some(vc::sign_document(&mgr.signing_key, &list).unwrap());
        let mut trust = TrustStore::trusting_did(mgr.issuer_did()).unwrap();
        trust
            .authorize_status_list_publisher(&list.id, &mgr.signing_key.verifying_key())
            .unwrap();
        trust.trust_status_list_credential(&list).unwrap();
        trust.require_recent_status_lists(60);
        let checked = verify_record(&original, &trust).unwrap();
        assert!(checked
            .details
            .iter()
            .any(|d| d.check == "revocation_status"
                && !d.passed
                && d.message.as_deref().unwrap_or_default().contains("stale")));
    }

    #[test]
    fn concurrent_writers_keep_one_chain_and_the_last_record() {
        let (dir, mgr) = manager();
        let mgr = Arc::new(mgr);
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let mgr = Arc::clone(&mgr);
                std::thread::spawn(move || {
                    for i in 0..20 {
                        let id = format!("shared-{}", (t + i) % 4);
                        match i % 7 {
                            6 => drop(mgr.record_deletion(&id, None).unwrap()),
                            _ => drop(
                                mgr.create_fact_provenance(
                                    &id,
                                    format!("{t}/{i}").as_bytes(),
                                    None,
                                )
                                .unwrap(),
                            ),
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(mgr.verify_log_integrity().unwrap());
        let live: Vec<Option<u64>> = (0..4)
            .map(|k| mgr.get_record(&format!("shared-{k}")).map(|r| r.sequence))
            .collect();
        drop(Arc::try_unwrap(mgr).ok().unwrap());

        let reopened =
            ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        assert!(reopened.verify_log_integrity().unwrap());
        let replayed: Vec<Option<u64>> = (0..4)
            .map(|k| {
                reopened
                    .get_record(&format!("shared-{k}"))
                    .map(|r| r.sequence)
            })
            .collect();
        assert_eq!(live, replayed);
    }

    #[test]
    fn log_and_key_event_log_verify_right_after_rotation_and_after_reopen() {
        let dir = tempdir().unwrap();
        let mut mgr =
            ProvenanceManager::new(&dir.path().join("old.key"), Some(dir.path())).unwrap();
        mgr.create_fact_provenance("before", b"old key", None)
            .unwrap();
        let new_key = dir.path().join("new.key");
        mgr.rotate_key(&new_key).unwrap();
        assert!(mgr.verify_log_integrity().unwrap());
        mgr.verify_keri_log().unwrap();
        drop(mgr);

        let reopened = ProvenanceManager::new(&new_key, Some(dir.path())).unwrap();
        assert!(reopened.verify_log_integrity().unwrap());
        reopened.verify_keri_log().unwrap();
        reopened
            .create_fact_provenance("after", b"new key", None)
            .unwrap();
        assert!(reopened.verify_log_integrity().unwrap());
        reopened.verify_keri_log().unwrap();
    }

    #[test]
    fn interior_nodes_verify_as_content_only_in_legacy_trees() {
        let content: Vec<[u8; 32]> = (0..4u32)
            .map(|i| Sha256::digest(i.to_le_bytes()).into())
            .collect();
        for hashing in [MerkleHashing::Legacy, MerkleHashing::Rfc6962] {
            let leaves: Vec<[u8; 32]> = content.iter().map(|h| hashing.leaf(h)).collect();
            let levels = ProvenanceManager::merkle_levels(&leaves, hashing);
            let root = levels[levels.len() - 1][0];
            // The 64 bytes of leaves 0 and 1 claimed as content, with leaf 0's proof minus its
            // first sibling.
            let mut forged_content = Vec::with_capacity(64);
            forged_content.extend_from_slice(&leaves[0]);
            forged_content.extend_from_slice(&leaves[1]);
            let forged_hash: [u8; 32] = Sha256::digest(&forged_content).into();
            let proof = ProvenanceManager::merkle_proof(&levels, 0);
            let accepted =
                check_inclusion(hashing, &forged_hash, Some(&proof[1..]), &root, Some(4)).is_ok();
            assert_eq!(accepted, hashing == MerkleHashing::Legacy, "{hashing:?}");
            for (i, h) in content.iter().enumerate() {
                let proof = ProvenanceManager::merkle_proof(&levels, i);
                check_inclusion(hashing, h, Some(&proof), &root, Some(4)).unwrap();
            }
        }
    }

    #[test]
    fn new_batches_sign_a_versioned_root_claim() {
        let (_dir, mgr) = manager();
        let items: Vec<(&str, &[u8], Option<&str>)> = vec![
            ("v1", b"one", None),
            ("v2", b"two", None),
            ("v3", b"three", None),
        ];
        let records = mgr.create_batch_provenance(&items).unwrap();
        let payload = cose::verify_and_extract(
            &mgr.signing_key.verifying_key(),
            records[0].cose_envelope.as_ref().unwrap(),
        )
        .unwrap();
        let (_, size) = parse_batch_root_claim(&payload).unwrap().unwrap();
        assert_eq!(size, 3);

        let mut forged = records[0].clone();
        forged
            .merkle_proof
            .as_mut()
            .unwrap()
            .push((false, [0u8; 32]));
        assert_rejected(&mgr, &forged, "a proof longer than the signed tree allows");

        let mut tampered = ciborium::Value::Map(vec![]);
        if let Ok(ciborium::Value::Map(mut entries)) = ciborium::from_reader(payload.as_slice()) {
            for (k, v) in &mut entries {
                if k.as_text() == Some("v") {
                    *v = ciborium::Value::Integer(3u64.into());
                }
            }
            tampered = ciborium::Value::Map(entries);
        }
        let mut bytes = Vec::new();
        ciborium::into_writer(&tampered, &mut bytes).unwrap();
        assert!(parse_batch_root_claim(&bytes).is_err());
    }

    #[test]
    fn records_written_before_the_binding_checks_still_verify() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("testdata/legacy_v1_records.json")).unwrap();
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let trust = TrustStore::trusting_key(&key.verifying_key());
        let mgr = ProvenanceManager::from_signing_key(key);
        let mut records: Vec<ProvenanceRecord> = ["fact", "fact_with_uri", "triple"]
            .iter()
            .map(|k| serde_json::from_value(fixture[k].clone()).unwrap())
            .collect();
        records.extend(
            serde_json::from_value::<Vec<ProvenanceRecord>>(fixture["batch"].clone()).unwrap(),
        );
        assert_eq!(records.len(), 6);
        for r in &records {
            let free = verify_record(r, &trust).unwrap();
            assert!(free.valid, "{:?}", free.details);
            let own = mgr.verify_fact_provenance(r, &trust).unwrap();
            assert!(own.valid, "{:?}", own.details);
        }

        // A legacy batch envelope signs a bare hash, as the log head does: without the
        // credential the record proves nothing about a batch.
        let mut forged = records[3].clone();
        forged.vc_json = None;
        assert!(!verify_record(&forged, &trust).unwrap().valid);
        assert!(!mgr.verify_fact_provenance(&forged, &trust).unwrap().valid);
    }

    /// A Sigstore bundle over caller-chosen bytes must not be a signature a fact claim can
    /// reuse: the bundle signs under its own domain, not the raw bytes.
    #[test]
    fn sigstore_bundle_signature_does_not_forge_a_fact_claim() {
        use base64::Engine;
        use coset::{CborSerializable, CoseSign1Builder, HeaderBuilder};

        let (_dir, mgr) = manager();
        let genuine = mgr.create_fact_provenance("f", b"genuine", None).unwrap();
        let forged_hash: [u8; 32] = Sha256::digest(b"forged").into();
        let claim = ciborium::Value::Map(vec![
            ("id".into(), "f".into()),
            (
                "content_hash".into(),
                ciborium::Value::Bytes(forged_hash.to_vec()),
            ),
            ("timestamp_ms".into(), genuine.timestamp_ms.into()),
        ]);
        let mut payload = Vec::new();
        ciborium::into_writer(&claim, &mut payload).unwrap();
        let mut sign1 = CoseSign1Builder::new()
            .protected(
                HeaderBuilder::new()
                    .algorithm(coset::iana::Algorithm::EdDSA)
                    .key_id(mgr.verifying_key().to_bytes().to_vec())
                    .build(),
            )
            .payload(payload)
            .build();
        let bundle = mgr
            .create_sigstore_bundle(&sign1.tbs_data(b""), None)
            .unwrap();
        sign1.signature = base64::engine::general_purpose::STANDARD
            .decode(&bundle.message_signature.signature)
            .unwrap();

        let mut forged = genuine;
        forged.source_hash = Some(forged_hash);
        forged.vc_json = None;
        forged.cose_envelope = Some(sign1.to_vec().unwrap());
        assert_rejected(
            &mgr,
            &forged,
            "a fact claim signed through a Sigstore bundle",
        );
    }

    /// A credential from another trusted issuer is checked against that issuer's status list,
    /// not this manager's own list.
    #[test]
    fn foreign_revoked_credential_fails_the_manager_check() {
        let (_dir, issuer) = manager();
        let (_dir2, verifier) = manager();
        let record = issuer
            .create_fact_provenance("x", b"foreign", None)
            .unwrap();
        let credential: vc::FactCredential =
            serde_json::from_str(record.vc_json.as_deref().unwrap()).unwrap();
        let index: u64 = credential
            .credential_status
            .as_ref()
            .unwrap()
            .status_list_index
            .parse()
            .unwrap();
        issuer.revoke_credential(index).unwrap();

        let trust = issuer.self_trust();
        assert!(!verify_record(&record, &trust).unwrap().valid);
        let result = verifier.verify_fact_provenance(&record, &trust).unwrap();
        assert!(
            result
                .details
                .iter()
                .any(|d| d.check == "revocation_status" && !d.passed),
            "{:?}",
            result.details
        );
        assert!(!result.valid);

        // The verifier's own list does not revoke a foreign credential either.
        let other = issuer.create_fact_provenance("y", b"other", None).unwrap();
        let other_index: u64 =
            serde_json::from_str::<vc::FactCredential>(other.vc_json.as_deref().unwrap())
                .unwrap()
                .credential_status
                .unwrap()
                .status_list_index
                .parse()
                .unwrap();
        verifier.revoke_credential(other_index).unwrap();
        assert!(
            verifier
                .verify_fact_provenance(&other, &issuer.self_trust())
                .unwrap()
                .valid
        );
    }
}
