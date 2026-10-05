// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

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

const PROOF_TYPE: &str = "DataIntegrityProof";
const CRYPTOSUITE: &str = "eddsa-jcs-2022";
const PROOF_PURPOSE: &str = "assertionMethod";

fn proof_hash_data<T: Serialize>(
    unsigned: &T,
    proof_options: &serde_json::Value,
) -> Result<[u8; 64]> {
    let mut hash_data = [0u8; 64];
    hash_data[..32].copy_from_slice(&Sha256::digest(jcs_serialize(proof_options)?));
    hash_data[32..].copy_from_slice(&Sha256::digest(jcs_serialize(unsigned)?));
    Ok(hash_data)
}

/// Create an eddsa-jcs-2022 Data Integrity proof over a proof-less document.
/// Per W3C Data Integrity EdDSA Cryptosuites v1.0:
///   hashData = SHA-256(JCS(proofOptions)) || SHA-256(JCS(unsignedDocument))
///   signature = Ed25519.sign(hashData)
pub(crate) fn sign_document<T: Serialize>(
    signing_key: &SigningKey,
    unsigned: &T,
) -> Result<DataIntegrityProof> {
    let issuer_did = did_key_from_ed25519(&signing_key.verifying_key().to_bytes());
    let created = chrono_iso8601_now();
    let verification_method = format!("{issuer_did}#key-0");
    let proof_options = serde_json::json!({
        "type": PROOF_TYPE,
        "cryptosuite": CRYPTOSUITE,
        "created": &created,
        "verificationMethod": &verification_method,
        "proofPurpose": PROOF_PURPOSE
    });
    let signature = signing_key.sign(&proof_hash_data(unsigned, &proof_options)?);

    Ok(DataIntegrityProof {
        proof_type: PROOF_TYPE.to_string(),
        cryptosuite: CRYPTOSUITE.to_string(),
        created,
        verification_method,
        proof_purpose: PROOF_PURPOSE.to_string(),
        proof_value: multibase::encode(multibase::Base::Base58Btc, signature.to_bytes()),
    })
}

/// Verify an eddsa-jcs-2022 proof over a proof-less document and return the
/// issuer's key. The proof's verification method DID must equal `issuer`:
/// without that binding an attacker can tamper a document, re-sign it with
/// their own key, and have it verify under the original issuer's name.
pub(crate) fn verify_document<T: Serialize>(
    unsigned: &T,
    issuer: &str,
    proof: &DataIntegrityProof,
) -> Result<VerifyingKey> {
    if proof.proof_type != PROOF_TYPE
        || proof.cryptosuite != CRYPTOSUITE
        || proof.proof_purpose != PROOF_PURPOSE
    {
        return Err(anyhow!("unsupported proof type, cryptosuite, or purpose"));
    }

    let did_part = proof
        .verification_method
        .split('#')
        .next()
        .unwrap_or(&proof.verification_method);
    let issuer_did = issuer.split('#').next().unwrap_or(issuer);
    if did_part != issuer_did {
        return Err(anyhow!(
            "verification method DID ({did_part}) does not match credential issuer ({issuer_did})"
        ));
    }

    let pk_bytes = super::did::ed25519_from_did_key(did_part)?;
    let verifying_key =
        VerifyingKey::from_bytes(&pk_bytes).map_err(|e| anyhow!("invalid public key: {e}"))?;

    let (_, sig_bytes) = multibase::decode(&proof.proof_value)
        .map_err(|e| anyhow!("multibase decode failed: {e}"))?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow!("invalid signature: {e}"))?;

    let proof_options = serde_json::json!({
        "type": &proof.proof_type,
        "cryptosuite": &proof.cryptosuite,
        "created": &proof.created,
        "verificationMethod": &proof.verification_method,
        "proofPurpose": &proof.proof_purpose
    });
    verifying_key
        .verify(&proof_hash_data(unsigned, &proof_options)?, &signature)
        .map_err(|e| anyhow!("VC signature verification failed: {e}"))?;
    Ok(verifying_key)
}

/// Sign a VC with Ed25519 DataIntegrity proof using eddsa-jcs-2022.
pub fn sign_credential(
    signing_key: &SigningKey,
    mut credential: FactCredential,
) -> Result<FactCredential> {
    credential.proof = None;
    credential.proof = Some(sign_document(signing_key, &credential)?);
    Ok(credential)
}

/// Verify a signed VC against the DID:key in its proof.
pub fn verify_credential(credential: &FactCredential) -> Result<()> {
    let proof = credential
        .proof
        .as_ref()
        .ok_or_else(|| anyhow!("credential has no proof"))?;
    let mut unsigned = credential.clone();
    unsigned.proof = None;
    verify_document(&unsigned, &credential.issuer, proof).map(|_| ())
}

/// Parse a UTC `YYYY-MM-DDTHH:MM:SSZ` timestamp into unix seconds.
pub(crate) fn parse_iso8601_utc(value: &str) -> Result<u64> {
    let b = value.as_bytes();
    let well_formed = b.len() == 20
        && b[4] == b'-'
        && b[7] == b'-'
        && b[10] == b'T'
        && b[13] == b':'
        && b[16] == b':'
        && b[19] == b'Z';
    let field = |range: std::ops::Range<usize>| -> Option<u64> {
        let part = value.get(range)?;
        part.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| part.parse().ok())
            .flatten()
    };
    let parsed = well_formed
        .then(|| {
            Some((
                field(0..4)?,
                field(5..7)?,
                field(8..10)?,
                field(11..13)?,
                field(14..16)?,
                field(17..19)?,
            ))
        })
        .flatten();
    let Some((year, month, day, hour, minute, second)) = parsed else {
        return Err(anyhow!("timestamp must be YYYY-MM-DDTHH:MM:SSZ"));
    };
    if year < 1970 || !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 59 {
        return Err(anyhow!("timestamp field out of range"));
    }
    // Days from civil (Howard Hinnant), the inverse of `days_to_ymd`.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    if day == 0 || days_to_ymd(days) != (year, month, day) {
        return Err(anyhow!("timestamp is not a valid calendar date"));
    }
    Ok(days * 86400 + hour * 3600 + minute * 60 + second)
}

pub(crate) fn chrono_iso8601_now() -> String {
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
    use rand::Rng;
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
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
        SigningKey::from_bytes(&rand::random())
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
    fn issuer_mismatch_rejected() {
        // A victim's credential re-signed by an attacker: the signature is valid
        // for the attacker's key, but the claimed issuer is the victim. Binding
        // the verification method to the issuer must reject this forgery.
        let victim = test_keypair();
        let attacker = test_keypair();
        let victim_did = did_key_from_ed25519(&victim.verifying_key().to_bytes());
        let hash = [0x11u8; 32];

        let vc = create_fact_credential(&victim_did, "forge-001", &hash, 16384, None, None, 0);
        let signed = sign_credential(&attacker, vc).unwrap();

        assert!(verify_credential(&signed).is_err());
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
