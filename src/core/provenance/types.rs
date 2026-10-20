// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Provenance metadata attached to a stored fact or vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// URI of the original data source.
    pub source_uri: Option<String>,
    /// SHA-256 hash of the original source content.
    pub source_hash: Option<[u8; 32]>,
    /// Milliseconds since epoch when the fact was ingested.
    pub timestamp_ms: u64,
    /// Monotonic sequence number for causal ordering.
    #[serde(default)]
    pub sequence: u64,
    /// DID of the entity asserting this fact.
    pub issuer_did: Option<String>,
    /// COSE Sign1 envelope over the fact payload.
    pub cose_envelope: Option<Vec<u8>>,
    /// Verifiable Credential JSON wrapping this fact.
    pub vc_json: Option<String>,
    /// SCITT transparency receipt (CBOR-encoded).
    pub scitt_receipt: Option<Vec<u8>>,
    /// Merkle proof path for batch-signed records: (is_left, sibling_hash).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merkle_proof: Option<Vec<(bool, [u8; 32])>>,
}

impl ProvenanceRecord {
    pub fn now() -> Self {
        Self {
            source_uri: None,
            source_hash: None,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            sequence: 0,
            issuer_did: None,
            cose_envelope: None,
            vc_json: None,
            scitt_receipt: None,
            merkle_proof: None,
        }
    }

    pub fn with_source(mut self, uri: &str, content: &[u8]) -> Self {
        use sha2::{Digest, Sha256};
        self.source_uri = Some(uri.to_string());
        self.source_hash = Some(Sha256::digest(content).into());
        self
    }
}

/// Store-level provenance manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreManifest {
    /// Unique identifier for this manifest.
    pub manifest_id: String,
    /// DID of the store owner/creator.
    pub issuer_did: Option<String>,
    /// Timestamp when the manifest was created.
    pub created_ms: u64,
    /// Number of facts/vectors in the store at manifest time.
    pub fact_count: usize,
    /// SHA-256 hash of the serialized store contents.
    pub store_hash: [u8; 32],
    /// COSE Sign1 envelope over the manifest payload.
    pub cose_envelope: Option<Vec<u8>>,
    /// JUMBF-encoded C2PA manifest (ISO 19566-5 binary).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jumbf_manifest: Option<Vec<u8>>,
}

/// Provenance record for a fact deletion event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeletionRecord {
    pub fact_id: String,
    pub deleted_by: Option<String>,
    pub timestamp_ms: u64,
    pub reason: Option<String>,
    pub cose_envelope: Option<Vec<u8>>,
}

/// Verification result for a provenance record or manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationResult {
    pub valid: bool,
    pub issuer_did: Option<String>,
    pub timestamp_ms: u64,
    pub details: Vec<VerificationDetail>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationDetail {
    pub check: String,
    pub passed: bool,
    pub message: Option<String>,
}
