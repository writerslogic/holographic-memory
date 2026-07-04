// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

pub mod c2pa_manifest;
pub mod cawg;
pub mod cose;
pub mod did;
pub mod jumbf;
pub mod keri;
pub mod scitt;
pub mod sigstore;
pub mod trust;
pub mod types;
pub mod vc;

use anyhow::Result;
use ed25519_dalek::SigningKey;
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
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

pub struct ProvenanceManager {
    signing_key: SigningKey,
    issuer_did: String,
    records: RwLock<fxhash::FxHashMap<String, ProvenanceRecord>>,
    chain_head: RwLock<[u8; 32]>,
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
        let (records, chain_head) = Self::load_log(log_path.as_deref());
        let clock_init = records
            .values()
            .map(|r| r.sequence)
            .max()
            .map_or(0, |max| max + 1);
        let mut kel = match storage_path {
            Some(p) => keri::KeyEventLog::with_path(&p.join("kel.json")).unwrap_or_default(),
            None => keri::KeyEventLog::new(),
        };
        if kel.events().is_empty() {
            if let Err(e) = kel.inception(&signing_key, None) {
                tracing::warn!("KEL inception failed: {e}");
            }
        }

        Ok(Self {
            signing_key,
            issuer_did,
            records: RwLock::new(records),
            chain_head: RwLock::new(chain_head),
            logical_clock: std::sync::atomic::AtomicU64::new(clock_init),
            log_path: log_path.clone(),
            revocation_bitstring: RwLock::new(Self::load_revocation_bitstring(log_path.as_deref())),
            kel: RwLock::new(kel),
            #[cfg(feature = "provenance-scitt")]
            scitt_endpoint: None,
        })
    }

    pub fn from_signing_key(signing_key: SigningKey) -> Self {
        let issuer_did = did::did_key_from_ed25519(&signing_key.verifying_key().to_bytes());
        let mut kel = keri::KeyEventLog::new();
        let _ = kel.inception(&signing_key, None);
        Self {
            signing_key,
            issuer_did,
            records: RwLock::new(fxhash::FxHashMap::default()),
            chain_head: RwLock::new([0u8; 32]),
            logical_clock: std::sync::atomic::AtomicU64::new(0),
            log_path: None,
            revocation_bitstring: RwLock::new(Vec::new()),
            kel: RwLock::new(kel),
            #[cfg(feature = "provenance-scitt")]
            scitt_endpoint: None,
        }
    }

    fn next_sequence(&self) -> u64 {
        self.logical_clock
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    #[cfg(feature = "provenance-scitt")]
    pub fn with_scitt_endpoint(mut self, endpoint: String) -> Self {
        self.scitt_endpoint = Some(endpoint);
        self
    }

    pub fn issuer_did(&self) -> &str {
        &self.issuer_did
    }

    pub fn store_record(&self, fact_id: &str, record: ProvenanceRecord) {
        if let Some(ref path) = self.log_path {
            let prev_hash = hex_encode(&*self.chain_head.read());
            let entry = LogEntry {
                prev_hash,
                fact_id: fact_id.to_string(),
                record: record.clone(),
            };
            if let Ok(line) = serde_json::to_string(&entry) {
                let new_hash: [u8; 32] = Sha256::digest(line.as_bytes()).into();
                let mut buf = line;
                buf.push('\n');
                if let Err(e) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .and_then(|mut f| f.write_all(buf.as_bytes()))
                {
                    tracing::warn!("provenance log write failed: {e}");
                }
                *self.chain_head.write() = new_hash;
                self.write_signed_head(&new_hash, path);
            }
        }
        self.records.write().insert(fact_id.to_string(), record);
    }

    fn write_signed_head(&self, head: &[u8; 32], log_path: &Path) {
        let head_path = log_path.with_extension("head");
        match cose::sign_payload(&self.signing_key, head) {
            Ok(signed) => {
                if let Err(e) = std::fs::write(head_path, signed) {
                    tracing::warn!("signed head write failed: {e}");
                }
            }
            Err(e) => tracing::warn!("signed head signing failed: {e}"),
        }
    }

    pub fn get_record(&self, fact_id: &str) -> Option<ProvenanceRecord> {
        self.records.read().get(fact_id).cloned()
    }

    pub fn record_count(&self) -> usize {
        self.records.read().len()
    }

    pub fn record_deletion(&self, fact_id: &str, reason: Option<&str>) -> Result<DeletionRecord> {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

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

        self.records.write().remove(fact_id);

        if let Some(ref path) = self.log_path {
            let prev_hash = hex_encode(&*self.chain_head.read());
            let entry_json = serde_json::json!({
                "prev_hash": prev_hash,
                "action": "delete",
                "fact_id": fact_id,
                "deletion": record,
            });
            if let Ok(line) = serde_json::to_string(&entry_json) {
                let new_hash: [u8; 32] = Sha256::digest(line.as_bytes()).into();
                let mut buf = line;
                buf.push('\n');
                if let Err(e) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .and_then(|mut f| f.write_all(buf.as_bytes()))
                {
                    tracing::warn!("deletion log write failed: {e}");
                }
                *self.chain_head.write() = new_hash;
                self.write_signed_head(&new_hash, path);
            }
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

        if let Some(ref path) = self.log_path {
            let prev_hash = hex_encode(&*self.chain_head.read());
            let entry_json = serde_json::json!({
                "prev_hash": prev_hash,
                "action": "key_rotation",
                "old_did": old_did,
                "new_did": new_did,
                "cose_envelope": rotation_envelope,
            });
            if let Ok(line) = serde_json::to_string(&entry_json) {
                let new_hash: [u8; 32] = Sha256::digest(line.as_bytes()).into();
                let mut buf = line;
                buf.push('\n');
                if let Err(e) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .and_then(|mut f| f.write_all(buf.as_bytes()))
                {
                    tracing::warn!("key rotation log write failed: {e}");
                }
                *self.chain_head.write() = new_hash;
                self.write_signed_head(&new_hash, path);
            }
        }

        self.signing_key = new_signing_key;
        self.issuer_did = new_did.clone();
        Ok(new_did)
    }

    pub fn verify_log_integrity(&self) -> Result<bool> {
        let Some(ref path) = self.log_path else {
            return Ok(true);
        };
        let Ok(file) = std::fs::File::open(path) else {
            return Ok(true);
        };
        let mut expected_prev = [0u8; 32];
        let mut computed_head = [0u8; 32];
        let mut has_entries = false;
        for line in std::io::BufReader::new(file).lines() {
            let line = line.map_err(|e| anyhow::anyhow!("log read error: {e}"))?;
            let obj: serde_json::Value =
                serde_json::from_str(&line).map_err(|e| anyhow::anyhow!("log parse error: {e}"))?;
            let prev = obj
                .get("prev_hash")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing prev_hash"))?;
            let declared = hex_decode(prev)?;
            if declared != expected_prev {
                return Ok(false);
            }
            computed_head = Sha256::digest(line.as_bytes()).into();
            expected_prev = computed_head;
            has_entries = true;
        }
        if has_entries {
            let head_path = path.with_extension("head");
            if let Ok(signed_bytes) = std::fs::read(&head_path) {
                let verifying_key = self.signing_key.verifying_key();
                match cose::verify_and_extract(&verifying_key, &signed_bytes) {
                    Ok(stored_head) => {
                        if stored_head.as_slice() != computed_head {
                            return Ok(false);
                        }
                    }
                    Err(_) => return Ok(false),
                }
            }
        }
        Ok(true)
    }

    fn load_log(path: Option<&Path>) -> (fxhash::FxHashMap<String, ProvenanceRecord>, [u8; 32]) {
        let mut map = fxhash::FxHashMap::default();
        let mut head = [0u8; 32];
        let Some(path) = path else {
            return (map, head);
        };
        let Ok(file) = std::fs::File::open(path) else {
            return (map, head);
        };
        for line in std::io::BufReader::new(file).lines() {
            let Ok(line) = line else { continue };
            head = Sha256::digest(line.as_bytes()).into();
            if let Ok(entry) = serde_json::from_str::<LogEntry>(&line) {
                map.insert(entry.fact_id, entry.record);
            } else if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) {
                if obj.get("action").and_then(|a| a.as_str()) == Some("delete") {
                    if let Some(fid) = obj.get("fact_id").and_then(|f| f.as_str()) {
                        map.remove(fid);
                    }
                }
            }
        }
        (map, head)
    }

    fn load_revocation_bitstring(log_path: Option<&Path>) -> Vec<u8> {
        let Some(path) = log_path else {
            return Vec::new();
        };
        let bitstring_path = path.with_extension("statuslist");
        std::fs::read(bitstring_path).unwrap_or_default()
    }

    pub fn revoke_credential(&self, status_index: u64) -> Result<()> {
        let byte_idx = (status_index / 8) as usize;
        let bit_idx = (status_index % 8) as u8;
        {
            let mut bits = self.revocation_bitstring.write();
            if bits.len() <= byte_idx {
                bits.resize(byte_idx + 1, 0);
            }
            bits[byte_idx] |= 1 << bit_idx;
        }
        if let Some(ref path) = self.log_path {
            let bitstring_path = path.with_extension("statuslist");
            let bits = self.revocation_bitstring.read();
            std::fs::write(bitstring_path, &*bits)
                .map_err(|e| anyhow::anyhow!("failed to persist revocation bitstring: {e}"))?;
        }
        Ok(())
    }

    pub fn is_revoked(&self, status_index: u64) -> bool {
        let byte_idx = (status_index / 8) as usize;
        let bit_idx = (status_index % 8) as u8;
        let bits = self.revocation_bitstring.read();
        byte_idx < bits.len() && (bits[byte_idx] & (1 << bit_idx)) != 0
    }

    /// Create a full provenance record for a stored fact.
    pub fn create_fact_provenance(
        &self,
        fact_id: &str,
        content: &[u8],
        source_uri: Option<&str>,
    ) -> Result<ProvenanceRecord> {
        let content_hash: [u8; 32] = Sha256::digest(content).into();
        let mut record = ProvenanceRecord::now().with_source(source_uri.unwrap_or(""), content);
        record.issuer_did = Some(self.issuer_did.clone());
        record.sequence = self.next_sequence();

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
        );
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

        self.store_record(fact_id, record.clone());
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
        record.sequence = self.next_sequence();

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
        );
        let signed_vc = vc::sign_credential(&self.signing_key, credential)?;
        record.vc_json = Some(serde_json::to_string(&signed_vc)?);

        self.store_record(params.fact_id, record.clone());
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
        let merkle_root = Self::merkle_root(&leaf_hashes);
        let root_envelope = cose::sign_payload(&self.signing_key, &merkle_root)?;

        let mut records = Vec::with_capacity(items.len());
        for (i, (fact_id, content, source_uri)) in items.iter().enumerate() {
            let content_hash = leaf_hashes[i];
            let proof = Self::merkle_proof(&leaf_hashes, i);
            let mut record = ProvenanceRecord::now().with_source(source_uri.unwrap_or(""), content);
            record.issuer_did = Some(self.issuer_did.clone());
            record.sequence = self.next_sequence();
            record.cose_envelope = Some(root_envelope.clone());
            record.merkle_proof = Some(proof);

            let credential = vc::create_fact_credential(
                &self.issuer_did,
                fact_id,
                &content_hash,
                0,
                *source_uri,
                None,
                record.sequence,
            );
            let signed_vc = vc::sign_credential(&self.signing_key, credential)?;
            record.vc_json = Some(serde_json::to_string(&signed_vc)?);

            self.store_record(fact_id, record.clone());
            records.push(record);
        }
        Ok(records)
    }

    fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
        if leaves.is_empty() {
            return [0u8; 32];
        }
        let mut level: Vec<[u8; 32]> = leaves.to_vec();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level.chunks(2) {
                if pair.len() == 2 {
                    let mut hasher = Sha256::new();
                    hasher.update(pair[0]);
                    hasher.update(pair[1]);
                    next.push(hasher.finalize().into());
                } else {
                    next.push(pair[0]);
                }
            }
            level = next;
        }
        level[0]
    }

    fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<(bool, [u8; 32])> {
        let mut proof = Vec::new();
        let mut level: Vec<[u8; 32]> = leaves.to_vec();
        let mut idx = index;
        while level.len() > 1 {
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            if sibling_idx < level.len() {
                proof.push((idx % 2 == 1, level[sibling_idx]));
            }
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level.chunks(2) {
                if pair.len() == 2 {
                    let mut hasher = Sha256::new();
                    hasher.update(pair[0]);
                    hasher.update(pair[1]);
                    next.push(hasher.finalize().into());
                } else {
                    next.push(pair[0]);
                }
            }
            level = next;
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
            created_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
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
        let mut result = self.verify_fact_provenance_self_consistency(record)?;
        let fallback = self.signing_key.verifying_key();
        let key = Self::resolve_cose_verifying_key(record, &fallback);
        let trusted = trust.is_trusted(&key);
        result.details.push(VerificationDetail {
            check: "trust_anchor".to_string(),
            passed: trusted,
            message: if trusted {
                None
            } else {
                Some("signing key is not in the trust store".to_string())
            },
        });
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
        let mut details = Vec::new();
        let fallback = self.signing_key.verifying_key();
        let verifying_key = Self::resolve_cose_verifying_key(record, &fallback);

        if let Some(ref envelope) = record.cose_envelope {
            match cose::verify_and_extract(&verifying_key, envelope) {
                Ok(_) => details.push(VerificationDetail {
                    check: "cose_signature".to_string(),
                    passed: true,
                    message: None,
                }),
                Err(e) => details.push(VerificationDetail {
                    check: "cose_signature".to_string(),
                    passed: false,
                    message: Some(e.to_string()),
                }),
            }
        }

        if let Some(ref vc_json) = record.vc_json {
            match serde_json::from_str::<vc::FactCredential>(vc_json) {
                Ok(ref credential) => {
                    match vc::verify_credential(credential) {
                        Ok(()) => details.push(VerificationDetail {
                            check: "vc_proof".to_string(),
                            passed: true,
                            message: None,
                        }),
                        Err(e) => details.push(VerificationDetail {
                            check: "vc_proof".to_string(),
                            passed: false,
                            message: Some(e.to_string()),
                        }),
                    }
                    let revoked = credential
                        .credential_status
                        .as_ref()
                        .and_then(|s| s.status_list_index.parse::<u64>().ok())
                        .is_some_and(|idx| self.is_revoked(idx));
                    details.push(VerificationDetail {
                        check: "revocation_status".to_string(),
                        passed: !revoked,
                        message: if revoked {
                            Some("credential has been revoked".to_string())
                        } else {
                            None
                        },
                    });
                }
                Err(e) => details.push(VerificationDetail {
                    check: "vc_parse".to_string(),
                    passed: false,
                    message: Some(e.to_string()),
                }),
            }
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
    pub fn self_trust(&self) -> TrustStore {
        TrustStore::trusting_key(&self.signing_key.verifying_key())
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

    /// Verify a `cawg.identity` ICA assertion JSON, returning the embedded VC.
    pub fn verify_cawg_assertion(
        &self,
        assertion: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let embedded = cawg::ica_embedded_bytes(assertion)?;
        cawg::verify_cawg_ica(&embedded)
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
        let trusted = trust.is_trusted(&key);
        result.details.push(VerificationDetail {
            check: "trust_anchor".to_string(),
            passed: trusted,
            message: if trusted {
                None
            } else {
                Some("signing key is not in the trust store".to_string())
            },
        });
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

                    if let Ok(hms_manifest) =
                        serde_json::from_slice::<c2pa_manifest::HmsManifest>(&payload)
                    {
                        if let Some(data) = store_data {
                            let hash_ok = c2pa_manifest::validate_store_hash(&hms_manifest, data)
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
/// Resolves the verifying key from the record's COSE key_id, verifies the
/// signatures against it, and requires that key to be present in `trust`. As
/// with [`ProvenanceManager::verify_fact_provenance`], a record whose embedded
/// key is not trusted verifies as `valid == false`.
pub fn verify_record(record: &ProvenanceRecord, trust: &TrustStore) -> Result<VerificationResult> {
    let mut details = Vec::new();

    if let Some(ref envelope) = record.cose_envelope {
        let kid = cose::extract_key_id(envelope)?;
        let pk: [u8; 32] = kid
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("COSE key_id is not 32 bytes"))?;
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk)
            .map_err(|e| anyhow::anyhow!("invalid public key in COSE: {e}"))?;
        match cose::verify_and_extract(&verifying_key, envelope) {
            Ok(_) => details.push(VerificationDetail {
                check: "cose_signature".to_string(),
                passed: true,
                message: None,
            }),
            Err(e) => details.push(VerificationDetail {
                check: "cose_signature".to_string(),
                passed: false,
                message: Some(e.to_string()),
            }),
        }
        let trusted = trust.is_trusted(&verifying_key);
        details.push(VerificationDetail {
            check: "trust_anchor".to_string(),
            passed: trusted,
            message: if trusted {
                None
            } else {
                Some("signing key is not in the trust store".to_string())
            },
        });
    }

    if let Some(ref vc_json) = record.vc_json {
        match serde_json::from_str::<vc::FactCredential>(vc_json) {
            Ok(ref credential) => match vc::verify_credential(credential) {
                Ok(()) => details.push(VerificationDetail {
                    check: "vc_proof".to_string(),
                    passed: true,
                    message: None,
                }),
                Err(e) => details.push(VerificationDetail {
                    check: "vc_proof".to_string(),
                    passed: false,
                    message: Some(e.to_string()),
                }),
            },
            Err(e) => details.push(VerificationDetail {
                check: "vc_parse".to_string(),
                passed: false,
                message: Some(e.to_string()),
            }),
        }
    }

    let all_passed = details.iter().all(|d| d.passed);
    Ok(VerificationResult {
        valid: all_passed && !details.is_empty(),
        issuer_did: record.issuer_did.clone(),
        timestamp_ms: record.timestamp_ms,
        details,
    })
}

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
    use tempfile::tempdir;

    #[test]
    fn full_fact_provenance_flow() {
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();

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
        assert_eq!(verification.details.len(), 4);

        assert!(mgr.get_record("fact-001").is_some());
        assert_eq!(mgr.record_count(), 1);
    }

    #[test]
    fn full_triple_provenance_flow() {
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();

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
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();

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
    fn issuer_did_consistency() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("test.key");
        let mgr1 = ProvenanceManager::new(&key_path, Some(dir.path())).unwrap();
        let mgr2 = ProvenanceManager::new(&key_path, Some(dir.path())).unwrap();
        assert_eq!(mgr1.issuer_did(), mgr2.issuer_did());
    }

    #[test]
    fn records_persist_to_disk() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("test.key");

        {
            let mgr = ProvenanceManager::new(&key_path, Some(dir.path())).unwrap();
            mgr.create_fact_provenance("f1", b"content one", None)
                .unwrap();
            mgr.create_fact_provenance("f2", b"content two", None)
                .unwrap();
            assert_eq!(mgr.record_count(), 2);
        }

        let mgr2 = ProvenanceManager::new(&key_path, Some(dir.path())).unwrap();
        assert_eq!(mgr2.record_count(), 2);
        assert!(mgr2.get_record("f1").is_some());
        assert!(mgr2.get_record("f2").is_some());
    }

    #[test]
    fn standalone_verify_without_manager() {
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        let record = mgr
            .create_fact_provenance("standalone-001", b"verify without manager", None)
            .unwrap();
        let issuer = record.issuer_did.clone().unwrap();
        drop(mgr);

        let trust = trust::TrustStore::trusting_did(&issuer).unwrap();
        let result = verify_record(&record, &trust).unwrap();
        assert!(result.valid);
        assert_eq!(result.details.len(), 3);
    }

    #[test]
    fn hash_chain_integrity() {
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        mgr.create_fact_provenance("c1", b"first", None).unwrap();
        mgr.create_fact_provenance("c2", b"second", None).unwrap();
        mgr.create_fact_provenance("c3", b"third", None).unwrap();
        assert!(mgr.verify_log_integrity().unwrap());
    }

    #[test]
    fn tampered_log_detected() {
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
        mgr.create_fact_provenance("t1", b"one", None).unwrap();
        mgr.create_fact_provenance("t2", b"two", None).unwrap();
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
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();

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
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();

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
    fn untrusted_key_rejected() {
        let dir = tempdir().unwrap();
        let mgr = ProvenanceManager::new(&dir.path().join("test.key"), Some(dir.path())).unwrap();
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
        let stranger = SigningKey::generate(&mut rand::thread_rng());
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
}
