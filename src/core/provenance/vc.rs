// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::did::did_key_from_ed25519;

/// A W3C Verifiable Credential Data Model 2.0 credential wrapping an HMS fact.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FactCredential {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    pub id: String,
    pub issuer: String,
    pub valid_from: String,
    pub credential_subject: FactSubject,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_status: Option<CredentialStatus>,
    pub proof: Option<DataIntegrityProof>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialStatus {
    pub id: String,
    #[serde(rename = "type")]
    pub status_type: String,
    pub status_purpose: String,
    pub status_list_index: String,
    pub status_list_credential: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FactSubject {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_uri: Option<String>,
    pub content_hash: String,
    pub encoding_method: String,
    pub dimensions: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataIntegrityProof {
    #[serde(rename = "type")]
    pub proof_type: String,
    pub cryptosuite: String,
    pub created: String,
    pub verification_method: String,
    pub proof_purpose: String,
    pub proof_value: String,
}

/// Create an unsigned VC for a stored fact.
pub fn create_fact_credential(
    issuer_did: &str,
    fact_id: &str,
    content_hash: &[u8; 32],
    dimensions: u32,
    source_uri: Option<&str>,
    triple: Option<(&str, &str, &str)>,
    status_index: u64,
) -> FactCredential {
    let timestamp = chrono_iso8601_now();
    let hash_hex = hex_encode(content_hash);

    let mut subject = FactSubject {
        id: format!("urn:hms:fact:{fact_id}"),
        source_uri: source_uri.map(String::from),
        content_hash: hash_hex,
        encoding_method: "holographic-reduced-representation".to_string(),
        dimensions,
        subject_id: None,
        relation_id: None,
        object_id: None,
    };

    if let Some((s, r, o)) = triple {
        subject.subject_id = Some(s.to_string());
        subject.relation_id = Some(r.to_string());
        subject.object_id = Some(o.to_string());
    }

    let vc_id = format!("urn:uuid:{}", simple_uuid());
    let status = CredentialStatus {
        id: format!("{vc_id}#status"),
        status_type: "BitstringStatusListEntry".to_string(),
        status_purpose: "revocation".to_string(),
        status_list_index: status_index.to_string(),
        status_list_credential: format!("{issuer_did}/status/1"),
    };

    FactCredential {
        context: vec![
            "https://www.w3.org/ns/credentials/v2".to_string(),
            "https://writerslogic.com/ns/hms/v1".to_string(),
        ],
        credential_type: vec![
            "VerifiableCredential".to_string(),
            "HMSFactCredential".to_string(),
        ],
        id: vc_id,
        issuer: issuer_did.to_string(),
        valid_from: timestamp,
        credential_subject: subject,
        credential_status: Some(status),
        proof: None,
    }
}

/// RFC 8785 JSON Canonicalization Scheme.
/// serde_json::Value uses BTreeMap for objects (sorted keys by default).
/// We serialize to Value first, then to compact JSON bytes.
fn jcs_serialize<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    let json_value =
        serde_json::to_value(value).map_err(|e| anyhow!("JCS serialization failed: {e}"))?;
    serde_json::to_vec(&json_value).map_err(|e| anyhow!("JCS serialization failed: {e}"))
}

/// Sign a VC with Ed25519 DataIntegrity proof using eddsa-jcs-2022.
/// Per W3C Data Integrity EdDSA Cryptosuites v1.0:
///   hashData = SHA-256(JCS(proofOptions)) || SHA-256(JCS(unsignedDocument))
///   signature = Ed25519.sign(hashData)
pub fn sign_credential(
    signing_key: &SigningKey,
    mut credential: FactCredential,
) -> Result<FactCredential> {
    let issuer_did = did_key_from_ed25519(&signing_key.verifying_key().to_bytes());
    let created = chrono_iso8601_now();

    credential.proof = None;
    let document_hash = Sha256::digest(jcs_serialize(&credential)?);

    let proof_options = serde_json::json!({
        "type": "DataIntegrityProof",
        "cryptosuite": "eddsa-jcs-2022",
        "created": &created,
        "verificationMethod": format!("{issuer_did}#key-0"),
        "proofPurpose": "assertionMethod"
    });
    let options_hash = Sha256::digest(jcs_serialize(&proof_options)?);

    let mut hash_data = [0u8; 64];
    hash_data[..32].copy_from_slice(&options_hash);
    hash_data[32..].copy_from_slice(&document_hash);
    let signature = signing_key.sign(&hash_data);

    credential.proof = Some(DataIntegrityProof {
        proof_type: "DataIntegrityProof".to_string(),
        cryptosuite: "eddsa-jcs-2022".to_string(),
        created,
        verification_method: format!("{issuer_did}#key-0"),
        proof_purpose: "assertionMethod".to_string(),
        proof_value: multibase::encode(multibase::Base::Base58Btc, signature.to_bytes()),
    });

    Ok(credential)
}

/// Verify a signed VC against the DID:key in its proof.
/// Reconstructs hashData = SHA-256(JCS(proofOptions)) || SHA-256(JCS(document))
/// and verifies the Ed25519 signature over it.
pub fn verify_credential(credential: &FactCredential) -> Result<()> {
    let proof = credential
        .proof
        .as_ref()
        .ok_or_else(|| anyhow!("credential has no proof"))?;

    let vm = &proof.verification_method;
    let did_part = vm
        .split('#')
        .next()
        .ok_or_else(|| anyhow!("invalid verification method"))?;
    let pk_bytes = super::did::ed25519_from_did_key(did_part)?;
    let verifying_key =
        VerifyingKey::from_bytes(&pk_bytes).map_err(|e| anyhow!("invalid public key: {e}"))?;

    let (_, sig_bytes) = multibase::decode(&proof.proof_value)
        .map_err(|e| anyhow!("multibase decode failed: {e}"))?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow!("invalid signature: {e}"))?;

    let mut unsigned = credential.clone();
    unsigned.proof = None;
    let document_hash = Sha256::digest(jcs_serialize(&unsigned)?);

    let proof_options = serde_json::json!({
        "type": &proof.proof_type,
        "cryptosuite": &proof.cryptosuite,
        "created": &proof.created,
        "verificationMethod": &proof.verification_method,
        "proofPurpose": &proof.proof_purpose
    });
    let options_hash = Sha256::digest(jcs_serialize(&proof_options)?);

    let mut hash_data = [0u8; 64];
    hash_data[..32].copy_from_slice(&options_hash);
    hash_data[32..].copy_from_slice(&document_hash);

    verifying_key
        .verify(&hash_data, &signature)
        .map_err(|e| anyhow!("VC signature verification failed: {e}"))
}

fn chrono_iso8601_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let days = secs / 86400;
    let rem = secs % 86400;
    let hours = rem / 3600;
    let minutes = (rem % 3600) / 60;
    let seconds = rem % 60;

    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn days_to_ymd(days_since_epoch: u64) -> (u64, u64, u64) {
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn simple_uuid() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        u32::from_be_bytes(bytes[0..4].try_into().unwrap()),
        u16::from_be_bytes(bytes[4..6].try_into().unwrap()),
        u16::from_be_bytes(bytes[6..8].try_into().unwrap()),
        u16::from_be_bytes(bytes[8..10].try_into().unwrap()),
        u64::from_be_bytes({
            let mut buf = [0u8; 8];
            buf[2..8].copy_from_slice(&bytes[10..16]);
            buf
        }),
    )
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::generate(&mut rand::thread_rng())
    }

    #[test]
    fn create_and_sign_vc() {
        let key = test_keypair();
        let hash = [0xABu8; 32];
        let did = did_key_from_ed25519(&key.verifying_key().to_bytes());

        let vc = create_fact_credential(&did, "fact-001", &hash, 16384, None, None, 0);
        assert_eq!(vc.credential_type[1], "HMSFactCredential");

        let signed = sign_credential(&key, vc).unwrap();
        assert!(signed.proof.is_some());

        verify_credential(&signed).unwrap();
    }

    #[test]
    fn vc_with_triple() {
        let key = test_keypair();
        let hash = [0xCDu8; 32];
        let did = did_key_from_ed25519(&key.verifying_key().to_bytes());

        let vc = create_fact_credential(
            &did,
            "triple-001",
            &hash,
            16384,
            Some("https://example.com/source"),
            Some(("paris", "capital_of", "france")),
            1,
        );
        assert_eq!(vc.credential_subject.subject_id.as_deref(), Some("paris"));
        assert_eq!(
            vc.credential_subject.relation_id.as_deref(),
            Some("capital_of")
        );

        let signed = sign_credential(&key, vc).unwrap();
        verify_credential(&signed).unwrap();
    }

    #[test]
    fn tampered_vc_rejected() {
        let key = test_keypair();
        let hash = [0xABu8; 32];
        let did = did_key_from_ed25519(&key.verifying_key().to_bytes());

        let vc = create_fact_credential(&did, "fact-001", &hash, 16384, None, None, 0);
        let mut signed = sign_credential(&key, vc).unwrap();
        signed.credential_subject.content_hash = "tampered".to_string();

        assert!(verify_credential(&signed).is_err());
    }

    #[test]
    fn vc_json_structure() {
        let key = test_keypair();
        let hash = [0u8; 32];
        let did = did_key_from_ed25519(&key.verifying_key().to_bytes());

        let vc = create_fact_credential(&did, "f1", &hash, 10000, None, None, 0);
        let json_str = serde_json::to_string_pretty(&vc).unwrap();
        assert!(json_str.contains("@context"));
        assert!(json_str.contains("VerifiableCredential"));
        assert!(json_str.contains("holographic-reduced-representation"));
    }
}
