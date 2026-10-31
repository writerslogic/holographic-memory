// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::did::did_key_from_ed25519;
use super::hex_encode;

/// A W3C Verifiable Credential Data Model 2.0 credential wrapping an WARM fact.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<DataIntegrityProof>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataIntegrityProof {
    #[serde(rename = "@context", default, skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,
    #[serde(rename = "type")]
    pub proof_type: String,
    pub cryptosuite: String,
    pub created: String,
    pub verification_method: String,
    pub proof_purpose: String,
    pub proof_value: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
    let timestamp = iso8601_now();
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
        extra: serde_json::Map::new(),
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
        status_list_credential: status_list_credential_url(issuer_did),
        extra: serde_json::Map::new(),
    };

    FactCredential {
        context: vec![
            "https://www.w3.org/ns/credentials/v2".to_string(),
            "https://docs.holographic-memory.com/ns/hms/v1.jsonld".to_string(),
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
        extra: serde_json::Map::new(),
    }
}

impl FactCredential {
    pub fn with_status_list_url(mut self, url: &str) -> Self {
        if let Some(ref mut status) = self.credential_status {
            status.status_list_credential = url.to_string();
        }
        self
    }
}

/// The status list credential that `issuer_did`'s fact credentials cite for revocation.
pub(crate) fn status_list_credential_url(issuer_did: &str) -> String {
    format!("{issuer_did}/status/1")
}

fn jcs_serialize(value: &serde_json::Value) -> Result<Vec<u8>> {
    serde_json_canonicalizer::to_vec(value).map_err(|e| anyhow!("JCS serialization failed: {e}"))
}

const PROOF_TYPE: &str = "DataIntegrityProof";
const CRYPTOSUITE: &str = "eddsa-jcs-2022";
const PROOF_PURPOSE: &str = "assertionMethod";

fn proof_hash_data(
    unsigned: &serde_json::Value,
    proof_options: &serde_json::Value,
) -> Result<[u8; 64]> {
    let mut hash_data = [0u8; 64];
    hash_data[..32].copy_from_slice(&Sha256::digest(jcs_serialize(proof_options)?));
    hash_data[32..].copy_from_slice(&Sha256::digest(jcs_serialize(unsigned)?));
    Ok(hash_data)
}

fn legacy_proof_hash_data(
    unsigned: &serde_json::Value,
    proof_options: &serde_json::Value,
) -> Result<[u8; 64]> {
    let mut old_unsigned = unsigned.clone();
    old_unsigned
        .as_object_mut()
        .ok_or_else(|| anyhow!("credential must be a JSON object"))?
        .insert("proof".to_string(), serde_json::Value::Null);
    let mut hash_data = [0u8; 64];
    hash_data[..32].copy_from_slice(&Sha256::digest(serde_json::to_vec(proof_options)?));
    hash_data[32..].copy_from_slice(&Sha256::digest(serde_json::to_vec(&old_unsigned)?));
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
    let mut document = serde_json::to_value(unsigned)?;
    let object = document
        .as_object_mut()
        .ok_or_else(|| anyhow!("credential must be a JSON object"))?;
    object.remove("proof");
    let context = object.get("@context").cloned();
    let issuer_did = did_key_from_ed25519(&signing_key.verifying_key().to_bytes());
    let created = iso8601_now();
    let verification_method = format!(
        "{issuer_did}#{}",
        issuer_did
            .strip_prefix("did:key:")
            .expect("generated did:key")
    );
    let mut proof_options = serde_json::json!({
        "type": PROOF_TYPE,
        "cryptosuite": CRYPTOSUITE,
        "created": &created,
        "verificationMethod": &verification_method,
        "proofPurpose": PROOF_PURPOSE
    });
    if let Some(ref context) = context {
        proof_options["@context"] = context.clone();
    }
    let signature = signing_key.sign(&proof_hash_data(&document, &proof_options)?);

    Ok(DataIntegrityProof {
        context,
        proof_type: PROOF_TYPE.to_string(),
        cryptosuite: CRYPTOSUITE.to_string(),
        created,
        verification_method,
        proof_purpose: PROOF_PURPOSE.to_string(),
        proof_value: multibase::encode(multibase::Base::Base58Btc, signature.to_bytes()),
        extra: serde_json::Map::new(),
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
    parse_datetime_stamp(&proof.created)?;

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
    let canonical_method = format!(
        "{did_part}#{}",
        did_part
            .strip_prefix("did:key:")
            .ok_or_else(|| anyhow!("verification method is not a did:key"))?
    );
    if proof.verification_method != canonical_method
        && !(proof.context.is_none() && proof.verification_method == format!("{did_part}#key-0"))
    {
        return Err(anyhow!(
            "verification method does not identify the issuer's key"
        ));
    }

    let pk_bytes = super::did::ed25519_from_did_key(did_part)?;
    let verifying_key =
        VerifyingKey::from_bytes(&pk_bytes).map_err(|e| anyhow!("invalid public key: {e}"))?;

    if !proof.proof_value.starts_with('z') {
        return Err(anyhow!("VC proofValue must use base58btc multibase"));
    }
    let (_, sig_bytes) = multibase::decode(&proof.proof_value)
        .map_err(|e| anyhow!("multibase decode failed: {e}"))?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow!("invalid signature: {e}"))?;

    let mut document = serde_json::to_value(unsigned)?;
    let object = document
        .as_object_mut()
        .ok_or_else(|| anyhow!("credential must be a JSON object"))?;
    object.remove("proof");
    let mut proof_options = serde_json::to_value(proof)?;
    proof_options
        .as_object_mut()
        .expect("serialized proof is an object")
        .remove("proofValue");
    if let Some(ref context) = proof.context {
        let signed_context = context
            .as_array()
            .ok_or_else(|| anyhow!("proof @context must be an array"))?;
        let document_context = object
            .get("@context")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow!("credential @context must be an array"))?;
        if !document_context.starts_with(signed_context) {
            return Err(anyhow!(
                "proof @context does not prefix credential @context"
            ));
        }
        object.insert("@context".to_string(), context.clone());
        proof_options["@context"] = context.clone();
    }
    let hash_data = proof_hash_data(&document, &proof_options)?;
    if verifying_key.verify(&hash_data, &signature).is_err()
        && (proof.context.is_some()
            || verifying_key
                .verify(
                    &legacy_proof_hash_data(&document, &proof_options)?,
                    &signature,
                )
                .is_err())
    {
        return Err(anyhow!("VC signature verification failed"));
    }
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
    verify_document(&unsigned, &credential.issuer, proof)?;
    validate_fact_credential(credential)
}

pub(crate) fn validate_fact_credential(credential: &FactCredential) -> Result<()> {
    use anyhow::ensure;

    ensure!(
        credential.context.first().map(String::as_str)
            == Some("https://www.w3.org/ns/credentials/v2"),
        "credential has no VC v2 context"
    );
    ensure!(
        credential
            .credential_type
            .iter()
            .any(|t| t == "VerifiableCredential")
            && credential
                .credential_type
                .iter()
                .any(|t| t == "HMSFactCredential"),
        "credential has no VC or WARM fact type"
    );
    ensure!(
        !credential.id.is_empty() && credential.issuer.starts_with("did:key:"),
        "credential id or issuer is invalid"
    );
    let from = parse_datetime_stamp(&credential.valid_from)?;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
    let now = (
        now.as_secs() as i64,
        format!("{:09}", now.subsec_nanos())
            .trim_end_matches('0')
            .to_string(),
    );
    ensure!(from <= now, "credential is not yet valid");
    if let Some(until) = credential.extra.get("validUntil") {
        let until = until
            .as_str()
            .ok_or_else(|| anyhow!("validUntil must be a dateTimeStamp string"))?;
        let until = parse_datetime_stamp(until)?;
        ensure!(from <= until, "credential validity interval is inverted");
        ensure!(now <= until, "credential has expired");
    }
    let subject = &credential.credential_subject;
    ensure!(
        subject.id.starts_with("urn:hms:fact:")
            && subject.id.len() > "urn:hms:fact:".len()
            && subject.content_hash.len() == 64
            && subject.content_hash.bytes().all(|b| b.is_ascii_hexdigit())
            && subject.encoding_method == "holographic-reduced-representation",
        "invalid WARM fact subject"
    );
    let triple_count = [
        subject.subject_id.as_ref(),
        subject.relation_id.as_ref(),
        subject.object_id.as_ref(),
    ]
    .into_iter()
    .filter(Option::is_some)
    .count();
    ensure!(
        triple_count == 0 || triple_count == 3,
        "incomplete fact triple"
    );
    if let Some(status) = &credential.credential_status {
        ensure!(
            status.status_type == "BitstringStatusListEntry"
                && status.status_purpose == "revocation"
                && status.id != status.status_list_credential
                && status.status_list_index.bytes().all(|b| b.is_ascii_digit())
                && !status.status_list_index.is_empty(),
            "invalid revocation status entry"
        );
        status.status_list_index.parse::<u64>()?;
        if !status.status_list_credential.starts_with("did:key:") {
            super::status_list::validate_url(&status.status_list_credential)?;
        }
    }
    Ok(())
}

pub(crate) fn parse_datetime_stamp(value: &str) -> Result<(i64, String)> {
    let invalid = || anyhow!("timestamp must be an RFC 3339 dateTimeStamp");
    let b = value.as_bytes();
    if b.len() < 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
        return Err(invalid());
    }
    let field = |start: usize, end: usize| -> Result<i64> {
        let part = b.get(start..end).ok_or_else(invalid)?;
        if !part.iter().all(u8::is_ascii_digit) {
            return Err(invalid());
        }
        Ok(std::str::from_utf8(part)?.parse()?)
    };
    let (year, month, day) = (field(0, 4)?, field(5, 7)?, field(8, 10)?);
    let (hour, minute, second) = (field(11, 13)?, field(14, 16)?, field(17, 19)?);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=days_in_month[(month - 1) as usize]).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(invalid());
    }
    let mut pos = 19;
    let mut fraction = String::new();
    if b.get(pos) == Some(&b'.') {
        pos += 1;
        let start = pos;
        while b.get(pos).is_some_and(u8::is_ascii_digit) {
            fraction.push(b[pos] as char);
            pos += 1;
        }
        if pos == start {
            return Err(invalid());
        }
    }
    let offset = match b.get(pos..) {
        Some(b"Z") => 0,
        Some(tz) if tz.len() == 6 && (tz[0] == b'+' || tz[0] == b'-') && tz[3] == b':' => {
            let hours = field(pos + 1, pos + 3)?;
            let minutes = field(pos + 4, pos + 6)?;
            if hours > 14 || minutes > 59 || (hours == 14 && minutes != 0) {
                return Err(invalid());
            }
            let sign = if tz[0] == b'+' { 1 } else { -1 };
            sign * (hours * 3600 + minutes * 60)
        }
        _ => return Err(invalid()),
    };
    let adjusted_year = year - i64::from(month <= 2);
    let era = adjusted_year.div_euclid(400);
    let yoe = adjusted_year - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    let seconds = days * 86400 + hour * 3600 + minute * 60 + second - offset;
    Ok((seconds, fraction.trim_end_matches('0').to_string()))
}

pub(crate) fn verify_credential_json(json: &str) -> Result<()> {
    let mut document: serde_json::Value = serde_json::from_str(json)?;
    let object = document
        .as_object_mut()
        .ok_or_else(|| anyhow!("credential must be a JSON object"))?;
    let issuer = object
        .get("issuer")
        .and_then(|value| {
            value
                .as_str()
                .or_else(|| value.get("id").and_then(serde_json::Value::as_str))
        })
        .ok_or_else(|| anyhow!("credential issuer must be a URL or an object with an id URL"))?
        .to_string();
    let proof_value = object
        .remove("proof")
        .ok_or_else(|| anyhow!("credential has no proof"))?;
    let proof: DataIntegrityProof = serde_json::from_value(proof_value.clone())?;
    if serde_json::to_value(&proof)? != proof_value {
        return Err(anyhow!("credential proof contains unsupported fields"));
    }
    verify_document(&document, &issuer, &proof).map(|_| ())
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

/// Current UTC time as an RFC 3339 / ISO 8601 timestamp with second precision.
pub(crate) fn iso8601_now() -> String {
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
    let hex = hex_encode(&bytes);
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
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
        assert_eq!(
            signed.proof.as_ref().unwrap().context,
            Some(serde_json::json!(signed.context))
        );
        assert_eq!(
            signed.proof.as_ref().unwrap().verification_method,
            format!("{did}#{}", did.strip_prefix("did:key:").unwrap())
        );

        verify_credential(&signed).unwrap();
    }

    #[test]
    fn jcs_uses_utf16_property_order_and_ecmascript_numbers() {
        let value = serde_json::json!({"\u{fb33}": 1, "😀": 2, "numbers": [1e30, 4.5]});
        assert_eq!(
            String::from_utf8(jcs_serialize(&value).unwrap()).unwrap(),
            "{\"numbers\":[1e+30,4.5],\"😀\":2,\"דּ\":1}"
        );
    }

    #[test]
    fn proof_context_is_bound_and_can_prefix_later_contexts() {
        let key = test_keypair();
        let did = did_key_from_ed25519(&key.verifying_key().to_bytes());
        let signed = sign_credential(
            &key,
            create_fact_credential(&did, "f1", &[0u8; 32], 1, None, None, 0),
        )
        .unwrap();

        let mut extended = signed.clone();
        extended
            .context
            .push("https://example.com/extension".into());
        verify_credential(&extended).unwrap();

        let mut altered = signed;
        altered.context[0] = "https://example.com/replacement".into();
        assert!(verify_credential(&altered).is_err());
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
