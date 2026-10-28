// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::io::{Read, Write};

use anyhow::{anyhow, ensure, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use flate2::{read::GzDecoder, Compression, GzBuilder};
use serde::{Deserialize, Serialize};

use super::did::did_key_from_ed25519;
use super::vc::{self, DataIntegrityProof};

pub(crate) const MIN_BYTES: usize = 16 * 1024;
pub(crate) const MAX_BYTES: usize = 1024 * 1024;
pub(crate) type StatusTime = (i64, String);

pub(crate) fn current_time() -> Result<StatusTime> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
    Ok((
        i64::try_from(now.as_secs())?,
        format!("{:09}", now.subsec_nanos())
            .trim_end_matches('0')
            .to_string(),
    ))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusListCredential {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    pub id: String,
    pub issuer: String,
    pub valid_from: String,
    pub credential_subject: StatusListSubject,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<DataIntegrityProof>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusListSubject {
    pub id: String,
    #[serde(rename = "type")]
    pub subject_type: String,
    pub status_purpose: String,
    pub encoded_list: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

pub(crate) fn valid_until(credential: &StatusListCredential) -> Result<Option<StatusTime>> {
    credential
        .extra
        .get("validUntil")
        .map(|until| {
            let until = until
                .as_str()
                .ok_or_else(|| anyhow!("status list validUntil must be a dateTimeStamp string"))?;
            vc::parse_datetime_stamp(until)
        })
        .transpose()
}

pub(crate) fn validate_url(url: &str) -> Result<()> {
    ensure!(url.len() <= 2048, "status list URL exceeds 2048 bytes");
    let uri: http::Uri = url
        .parse()
        .map_err(|e| anyhow!("invalid status list URL: {e}"))?;
    let authority = uri
        .authority()
        .ok_or_else(|| anyhow!("status list URL has no authority"))?;
    let authority_text = authority.as_str();
    let port_valid = if authority_text.starts_with('[') {
        authority_text.split_once(']').is_some_and(|(_, tail)| {
            tail.is_empty() || (tail.starts_with(':') && authority.port_u16().is_some())
        })
    } else {
        !authority_text.contains(':') || authority.port_u16().is_some()
    };
    ensure!(
        uri.scheme_str() == Some("https")
            && !authority.host().is_empty()
            && !authority_text.contains('@')
            && !url.contains('#')
            && !url.bytes().any(|b| b <= 0x20 || b == 0x7f)
            && port_valid,
        "status list URL must be an HTTPS URL with a valid authority and no fragment"
    );
    Ok(())
}

pub fn create_status_list_credential(
    signing_key: &SigningKey,
    url: &str,
    revocation_bits: &[u8],
    highest_issued_index: u64,
) -> Result<StatusListCredential> {
    validate_url(url)?;
    let needed = highest_issued_index
        .checked_div(8)
        .and_then(|byte| byte.checked_add(1))
        .and_then(|byte| usize::try_from(byte).ok())
        .ok_or_else(|| anyhow!("status index is out of range"))?;
    let len = MIN_BYTES.max(needed).max(revocation_bits.len());
    ensure!(len <= MAX_BYTES, "status list exceeds {MAX_BYTES} bytes");

    let mut wire = Vec::new();
    wire.try_reserve_exact(len)
        .map_err(|_| anyhow!("cannot allocate status list"))?;
    wire.resize(len, 0);
    for (target, source) in wire.iter_mut().zip(revocation_bits) {
        *target = source.reverse_bits();
    }

    let mut encoder = GzBuilder::new()
        .mtime(0)
        .write(Vec::new(), Compression::default());
    encoder.write_all(&wire)?;
    let compressed = encoder.finish()?;

    let mut credential = StatusListCredential {
        context: vec!["https://www.w3.org/ns/credentials/v2".into()],
        credential_type: vec![
            "VerifiableCredential".into(),
            "BitstringStatusListCredential".into(),
        ],
        id: url.into(),
        issuer: did_key_from_ed25519(&signing_key.verifying_key().to_bytes()),
        valid_from: vc::iso8601_now(),
        credential_subject: StatusListSubject {
            id: format!("{url}#list"),
            subject_type: "BitstringStatusList".into(),
            status_purpose: "revocation".into(),
            encoded_list: multibase::encode(multibase::Base::Base64Url, compressed),
            extra: serde_json::Map::new(),
        },
        proof: None,
        extra: serde_json::Map::new(),
    };
    credential.proof = Some(vc::sign_document(signing_key, &credential)?);
    Ok(credential)
}

pub fn verify_status_list_credential(
    credential: &StatusListCredential,
) -> Result<(VerifyingKey, Vec<u8>)> {
    verify_status_list_credential_at(credential, &current_time()?)
}

pub(crate) fn verify_status_list_credential_at(
    credential: &StatusListCredential,
    now: &StatusTime,
) -> Result<(VerifyingKey, Vec<u8>)> {
    validate_url(&credential.id)?;
    let valid_from = vc::parse_datetime_stamp(&credential.valid_from)?;
    ensure!(&valid_from <= now, "status list is not yet valid");
    if let Some(until) = valid_until(credential)? {
        ensure!(
            valid_from <= until,
            "status list validity interval is inverted"
        );
        ensure!(now <= &until, "status list has expired");
    }
    ensure!(
        credential.context.first().map(String::as_str)
            == Some("https://www.w3.org/ns/credentials/v2")
            && credential
                .credential_type
                .iter()
                .any(|name| name == "VerifiableCredential")
            && credential
                .credential_type
                .iter()
                .any(|name| name == "BitstringStatusListCredential")
            && credential.credential_subject.id == format!("{}#list", credential.id)
            && credential.credential_subject.subject_type == "BitstringStatusList"
            && credential.credential_subject.status_purpose == "revocation",
        "invalid status list credential fields"
    );
    let proof = credential
        .proof
        .as_ref()
        .ok_or_else(|| anyhow!("status list credential has no proof"))?;
    let mut unsigned = credential.clone();
    unsigned.proof = None;
    let key = vc::verify_document(&unsigned, &credential.issuer, proof)?;

    let encoded = &credential.credential_subject.encoded_list;
    ensure!(
        encoded.len() <= MAX_BYTES * 2 && !encoded.contains('='),
        "encoded status list exceeds size limit"
    );
    let (base, compressed) = multibase::decode(encoded)
        .map_err(|e| anyhow!("status list multibase decode failed: {e}"))?;
    ensure!(
        base == multibase::Base::Base64Url,
        "status list must use unpadded base64url multibase"
    );
    let mut wire = Vec::new();
    GzDecoder::new(compressed.as_slice())
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut wire)
        .map_err(|e| anyhow!("status list GZIP decode failed: {e}"))?;
    ensure!(
        (MIN_BYTES..=MAX_BYTES).contains(&wire.len()),
        "status list uncompressed size is out of range"
    );
    for byte in &mut wire {
        *byte = byte.reverse_bits();
    }
    Ok((key, wire))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[17u8; 32])
    }

    #[test]
    fn status_list_round_trip_and_w3c_bit_order() {
        let mut internal = vec![0u8; MIN_BYTES + 1];
        for index in [0, 7, 8, MIN_BYTES * 8 - 1, MIN_BYTES * 8] {
            internal[index / 8] |= 1 << (index % 8);
        }
        let vc = create_status_list_credential(
            &key(),
            "https://example.com/status/1",
            &internal,
            (MIN_BYTES * 8) as u64,
        )
        .unwrap();
        let (base, compressed) = multibase::decode(&vc.credential_subject.encoded_list).unwrap();
        assert_eq!(base, multibase::Base::Base64Url);
        let mut wire = Vec::new();
        GzDecoder::new(compressed.as_slice())
            .read_to_end(&mut wire)
            .unwrap();
        assert_eq!(wire.len(), MIN_BYTES + 1);
        assert_eq!(&wire[..2], &[0x81, 0x80]);
        assert_eq!(wire[MIN_BYTES - 1], 0x01);
        assert_eq!(wire[MIN_BYTES], 0x80);
        let (verified_key, decoded) = verify_status_list_credential(&vc).unwrap();
        assert_eq!(verified_key, key().verifying_key());
        assert_eq!(decoded, internal);
    }

    #[test]
    fn status_list_enforces_allocation_boundary() {
        let max_index = (MAX_BYTES * 8 - 1) as u64;
        for index in [max_index - 1, max_index] {
            let vc =
                create_status_list_credential(&key(), "https://example.com/status/1", &[], index)
                    .unwrap();
            let (_, bits) = verify_status_list_credential(&vc).unwrap();
            assert_eq!(bits.len(), MAX_BYTES);
        }
        assert!(create_status_list_credential(
            &key(),
            "https://example.com/status/1",
            &[],
            max_index + 1,
        )
        .is_err());
    }

    #[test]
    fn status_list_rejects_tampering_and_bad_urls() {
        let mut vc =
            create_status_list_credential(&key(), "https://example.com/status/1", &[], 0).unwrap();
        vc.credential_subject.encoded_list.push('x');
        assert!(verify_status_list_credential(&vc).is_err());
        assert!(
            create_status_list_credential(&key(), "http://example.com/status", &[], 0).is_err()
        );
        assert!(create_status_list_credential(&key(), "https://host/#frag", &[], 0).is_err());
        assert!(create_status_list_credential(&key(), "https://:123/status", &[], 0).is_err());
        assert!(create_status_list_credential(&key(), "https://host:bad/status", &[], 0).is_err());
        assert!(create_status_list_credential(&key(), "https://[::1]:bad/status", &[], 0).is_err());
        assert!(create_status_list_credential(&key(), "https://user@host/status", &[], 0).is_err());
    }

    #[test]
    fn status_list_enforces_signed_validity_window() {
        let mut credential =
            create_status_list_credential(&key(), "https://example.com/status/1", &[], 0).unwrap();
        credential.valid_from = "2020-01-01T00:00:00Z".into();
        let resign = |credential: &mut StatusListCredential| {
            credential.proof = None;
            credential.proof = Some(vc::sign_document(&key(), credential).unwrap());
        };
        credential.extra.insert(
            "validUntil".into(),
            serde_json::json!("2100-01-01T00:00:00Z"),
        );
        resign(&mut credential);
        verify_status_list_credential(&credential).unwrap();

        for (until, message) in [
            (
                serde_json::json!("2019-12-31T23:59:59Z"),
                "validity interval is inverted",
            ),
            (
                serde_json::json!("2020-01-01T00:00:00Z"),
                "status list has expired",
            ),
            (
                serde_json::json!("2020-01-01T00:00:01Z"),
                "status list has expired",
            ),
            (serde_json::json!(42), "dateTimeStamp string"),
            (serde_json::json!("2100-13-01T00:00:00Z"), "RFC 3339"),
        ] {
            credential.extra.insert("validUntil".into(), until);
            resign(&mut credential);
            assert!(verify_status_list_credential(&credential)
                .unwrap_err()
                .to_string()
                .contains(message));
        }
        credential.extra.remove("validUntil");
        credential.valid_from = "2100-01-01T00:00:00.000000001Z".into();
        resign(&mut credential);
        assert!(verify_status_list_credential(&credential)
            .unwrap_err()
            .to_string()
            .contains("not yet valid"));
    }

    #[test]
    fn signed_expiry_boundary_preserves_fractional_precision() {
        let mut credential =
            create_status_list_credential(&key(), "https://example.com/status/1", &[], 0).unwrap();
        credential.valid_from = "2020-01-01T00:00:00Z".into();
        credential.extra.insert(
            "validUntil".into(),
            serde_json::json!("2020-01-01T01:00:01.000000001+01:00"),
        );
        credential.proof = None;
        credential.proof = Some(vc::sign_document(&key(), &credential).unwrap());

        for (instant, accepted) in [
            ("2020-01-01T00:00:01Z", true),
            ("2020-01-01T00:00:01.000000001Z", true),
            ("2020-01-01T00:00:01.000000002Z", false),
        ] {
            let now = vc::parse_datetime_stamp(instant).unwrap();
            let result = verify_status_list_credential_at(&credential, &now);
            if accepted {
                result.unwrap();
            } else {
                assert!(result.unwrap_err().to_string().contains("has expired"));
            }
        }
    }
}
