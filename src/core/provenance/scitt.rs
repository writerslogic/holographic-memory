// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

pub mod receipt;

use anyhow::{anyhow, Result};
use coset::{CoseSign1Builder, HeaderBuilder, TaggedCborSerializable};
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};

const CLAIM_CONTENT_TYPE: &str = "application/cbor";
#[cfg(feature = "provenance-scitt")]
const STATEMENT_CONTENT_TYPE: &str = "application/scitt-statement+cose";
const MAX_STATEMENT_BYTES: usize = 1 << 20;
#[cfg(feature = "provenance-scitt")]
const MAX_RESPONSE_BYTES: u64 = 1 << 20;

/// An issuer-signed WARM fact statement with RFC 9943 CWT issuer/subject headers.
/// This type does not assert registration, receipt validity, or transparency-service trust.
#[derive(Clone, Debug)]
pub struct SignedStatement {
    pub cose_bytes: Vec<u8>,
}

/// Unverified bytes returned by the experimental transparency-service transport. The service
/// response is not evidence of inclusion or trust; use [`receipt::verify_receipt`] with
/// independently provisioned issuer, service and checkpoint policy for that assertion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnverifiedReceipt {
    pub entry_id: String,
    pub receipt_bytes: Vec<u8>,
    pub log_id: Option<String>,
}

/// Compatibility name for the transport artifact; this is never a verified receipt.
pub type TransparencyReceipt = UnverifiedReceipt;

/// SCITT claim payload for a knowledge insertion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactClaim {
    pub fact_id: String,
    pub content_hash: Vec<u8>,
    pub timestamp_ms: u64,
    pub source_uri: Option<String>,
    pub store_id: Option<String>,
}

/// Build a SCITT signed statement for a fact insertion.
pub fn create_signed_statement(
    signing_key: &SigningKey,
    claim: &FactClaim,
) -> Result<SignedStatement> {
    validate_claim(claim)?;
    let mut payload = Vec::new();
    let cbor_claim = ciborium::Value::Map(vec![
        (
            ciborium::Value::Text("fact_id".to_string()),
            ciborium::Value::Text(claim.fact_id.clone()),
        ),
        (
            ciborium::Value::Text("content_hash".to_string()),
            ciborium::Value::Bytes(claim.content_hash.clone()),
        ),
        (
            ciborium::Value::Text("timestamp_ms".to_string()),
            ciborium::Value::Integer(claim.timestamp_ms.into()),
        ),
        (
            ciborium::Value::Text("source_uri".to_string()),
            match &claim.source_uri {
                Some(uri) => ciborium::Value::Text(uri.clone()),
                None => ciborium::Value::Null,
            },
        ),
        (
            ciborium::Value::Text("store_id".to_string()),
            match &claim.store_id {
                Some(id) => ciborium::Value::Text(id.clone()),
                None => ciborium::Value::Null,
            },
        ),
    ]);
    ciborium::into_writer(&cbor_claim, &mut payload)
        .map_err(|e| anyhow!("CBOR encoding failed: {e}"))?;

    let protected = HeaderBuilder::new()
        .algorithm(coset::iana::Algorithm::EdDSA)
        .key_id(signing_key.verifying_key().to_bytes().to_vec())
        .content_type(CLAIM_CONTENT_TYPE.to_string())
        .value(
            15,
            ciborium::Value::Map(vec![
                (
                    ciborium::Value::Integer(1.into()),
                    ciborium::Value::Text(super::did::did_key_from_ed25519(
                        &signing_key.verifying_key().to_bytes(),
                    )),
                ),
                (
                    ciborium::Value::Integer(2.into()),
                    ciborium::Value::Text(claim.fact_id.clone()),
                ),
            ]),
        )
        .build();

    let sign1 = CoseSign1Builder::new()
        .protected(protected)
        .payload(payload)
        .create_signature(b"", |data| signing_key.sign(data).to_bytes().to_vec())
        .build();

    let cose_bytes = sign1
        .to_tagged_vec()
        .map_err(|e| anyhow!("COSE serialization failed: {e}"))?;

    Ok(SignedStatement { cose_bytes })
}

/// Bound on each request to the Transparency Service, so an unresponsive service fails the
/// submission instead of blocking the caller (fact provenance creation) indefinitely.
#[cfg(feature = "provenance-scitt")]
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Submit a statement and obtain bounded, unverified response bytes. This experimental
/// `/entries` adapter is not a general RFC 9943 registration or receipt-verification client.
#[cfg(feature = "provenance-scitt")]
pub fn submit_statement(endpoint: &str, statement: &SignedStatement) -> Result<UnverifiedReceipt> {
    submit_statement_within(endpoint, statement, REQUEST_TIMEOUT)
}

#[cfg(feature = "provenance-scitt")]
fn agent(timeout: std::time::Duration) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .max_redirects(0)
            .build(),
    )
}

#[cfg(feature = "provenance-scitt")]
fn response_bytes(mut response: http::Response<ureq::Body>) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .with_config()
        .limit(MAX_RESPONSE_BYTES + 1)
        .reader()
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_RESPONSE_BYTES,
        "transparency-service response exceeds size limit"
    );
    Ok(bytes)
}

#[cfg(feature = "provenance-scitt")]
fn same_origin_url(endpoint: &str, location: &str) -> Result<String> {
    anyhow::ensure!(
        endpoint.len() <= 8192 && location.len() <= 8192 && !location.is_empty(),
        "invalid transparency-service URL length"
    );
    let base: http::Uri = endpoint.parse()?;
    anyhow::ensure!(
        matches!(base.scheme_str(), Some("http" | "https"))
            && base.authority().is_some_and(|a| !a.as_str().contains('@'))
            && base.query().is_none(),
        "invalid transparency-service endpoint"
    );
    anyhow::ensure!(
        !location.starts_with("//") && !location.contains('#') && !location.contains('\\'),
        "invalid operation URL"
    );
    let url = if location.starts_with("http://") || location.starts_with("https://") {
        location.to_string()
    } else if location.starts_with('/') {
        format!(
            "{}://{}{}",
            base.scheme_str().unwrap(),
            base.authority().unwrap(),
            location
        )
    } else {
        format!("{}/{}", endpoint.trim_end_matches('/'), location)
    };
    let parsed: http::Uri = url.parse()?;
    anyhow::ensure!(
        parsed.scheme() == base.scheme()
            && parsed.authority().is_some_and(|a| a
                .as_str()
                .eq_ignore_ascii_case(base.authority().unwrap().as_str())),
        "cross-origin transparency-service operation URL is forbidden"
    );
    Ok(url)
}

#[cfg(feature = "provenance-scitt")]
fn remaining(deadline: std::time::Instant) -> Result<std::time::Duration> {
    deadline
        .checked_duration_since(std::time::Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| anyhow!("transparency-service operation deadline exceeded"))
}

#[cfg(feature = "provenance-scitt")]
fn submit_statement_within(
    endpoint: &str,
    statement: &SignedStatement,
    timeout: std::time::Duration,
) -> Result<UnverifiedReceipt> {
    anyhow::ensure!(
        statement.cose_bytes.len() <= MAX_STATEMENT_BYTES,
        "statement exceeds size limit"
    );
    let deadline = std::time::Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| anyhow!("invalid operation timeout"))?;
    let url = same_origin_url(
        endpoint,
        &format!("{}/entries", endpoint.trim_end_matches('/')),
    )?;
    let response = agent(remaining(deadline)?)
        .post(&url)
        .header("Content-Type", STATEMENT_CONTENT_TYPE)
        .send(&statement.cose_bytes)
        .map_err(|e| anyhow!("transparency-service submission failed: {e}"))?;
    match response.status().as_u16() {
        200 => Ok(UnverifiedReceipt {
            entry_id: String::new(),
            receipt_bytes: response_bytes(response)?,
            log_id: None,
        }),
        201 => {
            let location = response
                .headers()
                .get("Location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| anyhow!("missing operation Location"))?;
            let operation = same_origin_url(endpoint, location)?;
            poll_for_receipt(endpoint, &operation, deadline)
        }
        status => Err(anyhow!("transparency service returned HTTP {status}")),
    }
}

#[cfg(feature = "provenance-scitt")]
fn poll_for_receipt(
    endpoint: &str,
    operation_url: &str,
    deadline: std::time::Instant,
) -> Result<UnverifiedReceipt> {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1).min(remaining(deadline)?));
        let response = agent(remaining(deadline)?).get(operation_url).call()?;
        anyhow::ensure!(
            response.status().as_u16() == 200,
            "transparency-service operation returned HTTP {}",
            response.status()
        );
        let body: serde_json::Value = serde_json::from_slice(&response_bytes(response)?)?;
        match body.get("status").and_then(|s| s.as_str()) {
            Some("succeeded") => {
                let entry_id = body
                    .get("entryId")
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| anyhow!("missing entryId"))?
                    .to_string();
                let url = same_origin_url(endpoint, &receipt_url(endpoint, &entry_id)?)?;
                let response = agent(remaining(deadline)?).get(&url).call()?;
                anyhow::ensure!(
                    response.status().as_u16() == 200,
                    "receipt request returned HTTP {}",
                    response.status()
                );
                return Ok(UnverifiedReceipt {
                    entry_id,
                    receipt_bytes: response_bytes(response)?,
                    log_id: body.get("logId").and_then(|v| v.as_str()).map(String::from),
                });
            }
            Some("failed") => return Err(anyhow!("transparency-service registration failed")),
            Some("running" | "pending") => {}
            _ => return Err(anyhow!("unknown transparency-service operation status")),
        }
    }
}

/// The receipt URL of `entry_id` under `endpoint`. The id comes from the service's response
/// and becomes one path segment, so it must be a non-empty run of URL-safe characters: a `/`,
/// `..`, `?` or `#` in it would point the receipt request elsewhere.
#[cfg(feature = "provenance-scitt")]
fn receipt_url(endpoint: &str, entry_id: &str) -> Result<String> {
    let segment_ok = !entry_id.is_empty()
        && entry_id.len() <= 1024
        && entry_id != "."
        && entry_id != ".."
        && entry_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~:".contains(&b));
    if !segment_ok {
        return Err(anyhow!(
            "SCITT entry id {entry_id:?} is not a URL path segment"
        ));
    }
    Ok(format!(
        "{}/entries/{entry_id}/receipt",
        endpoint.trim_end_matches('/')
    ))
}

/// Verify a SCITT signed statement's COSE signature (local verification, no TS needed).
pub fn verify_statement(
    verifying_key: &ed25519_dalek::VerifyingKey,
    statement: &SignedStatement,
) -> Result<FactClaim> {
    anyhow::ensure!(
        statement.cose_bytes.len() <= MAX_STATEMENT_BYTES,
        "statement exceeds size limit"
    );
    let sign1 = super::cose::decode_sign1(&statement.cose_bytes)?;

    super::cose::verify_ed25519(&sign1, verifying_key)
        .map_err(|e| anyhow!("SCITT statement verification failed: {e}"))?;

    let payload = sign1
        .payload
        .as_ref()
        .ok_or_else(|| anyhow!("no payload in statement"))?;

    let mut cursor = std::io::Cursor::new(payload);
    let decoded: ciborium::Value =
        ciborium::from_reader(&mut cursor).map_err(|e| anyhow!("CBOR decode failed: {e}"))?;
    anyhow::ensure!(
        cursor.position() == payload.len() as u64,
        "trailing fact claim bytes"
    );

    let claim = parse_fact_claim(&decoded)?;
    validate_claim(&claim)?;
    if let Some((_, value)) = sign1
        .protected
        .header
        .rest
        .iter()
        .find(|(label, _)| label == &coset::Label::Int(15))
    {
        let expected = ciborium::Value::Map(vec![
            (
                ciborium::Value::Integer(1.into()),
                ciborium::Value::Text(super::did::did_key_from_ed25519(&verifying_key.to_bytes())),
            ),
            (
                ciborium::Value::Integer(2.into()),
                ciborium::Value::Text(claim.fact_id.clone()),
            ),
        ]);
        anyhow::ensure!(
            value == &expected,
            "SCITT issuer/subject claims do not bind the signing key and fact"
        );
    }
    Ok(claim)
}

fn validate_claim(claim: &FactClaim) -> Result<()> {
    anyhow::ensure!(
        !claim.fact_id.is_empty()
            && claim.fact_id.len() <= 4096
            && claim.content_hash.len() == 32
            && claim.source_uri.as_ref().is_none_or(|s| s.len() <= 8192)
            && claim.store_id.as_ref().is_none_or(|s| s.len() <= 4096),
        "invalid WARM fact claim"
    );
    Ok(())
}

fn parse_fact_claim(val: &ciborium::Value) -> Result<FactClaim> {
    let map = match val {
        ciborium::Value::Map(m) => m,
        _ => return Err(anyhow!("expected CBOR map")),
    };

    let mut seen = std::collections::HashSet::new();
    for (key, value) in map {
        let ciborium::Value::Text(key) = key else {
            return Err(anyhow!("fact claim key is not text"));
        };
        anyhow::ensure!(seen.insert(key), "duplicate fact claim key");
        if matches!(key.as_str(), "source_uri" | "store_id") {
            anyhow::ensure!(
                matches!(value, ciborium::Value::Null | ciborium::Value::Text(_)),
                "optional fact claim field is not text or null"
            );
        }
    }

    let get_text = |key: &str| -> Option<String> {
        map.iter()
            .find(|(k, _)| k == &ciborium::Value::Text(key.to_string()))
            .and_then(|(_, v)| match v {
                ciborium::Value::Text(s) => Some(s.clone()),
                _ => None,
            })
    };

    let get_bytes = |key: &str| -> Option<Vec<u8>> {
        map.iter()
            .find(|(k, _)| k == &ciborium::Value::Text(key.to_string()))
            .and_then(|(_, v)| match v {
                ciborium::Value::Bytes(b) => Some(b.clone()),
                _ => None,
            })
    };

    let get_int = |key: &str| -> Option<u64> {
        map.iter()
            .find(|(k, _)| k == &ciborium::Value::Text(key.to_string()))
            .and_then(|(_, v)| match v {
                ciborium::Value::Integer(i) => u64::try_from(*i).ok(),
                _ => None,
            })
    };

    Ok(FactClaim {
        fact_id: get_text("fact_id").ok_or_else(|| anyhow!("missing fact_id"))?,
        content_hash: get_bytes("content_hash").ok_or_else(|| anyhow!("missing content_hash"))?,
        timestamp_ms: get_int("timestamp_ms").ok_or_else(|| anyhow!("missing timestamp_ms"))?,
        source_uri: get_text("source_uri"),
        store_id: get_text("store_id"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::from_bytes(&rand::random())
    }

    #[cfg(feature = "provenance-scitt")]
    #[test]
    fn operation_locations_cannot_escape_the_service_origin() {
        assert_eq!(
            same_origin_url("https://ts.example/api", "/operations/a").unwrap(),
            "https://ts.example/operations/a"
        );
        for location in [
            "https://evil.example/a",
            "http://ts.example/a",
            "//evil.example/a",
            "https://ts.example@evil.example/a",
            "a#fragment",
            "",
        ] {
            assert!(
                same_origin_url("https://ts.example/api", location).is_err(),
                "{location}"
            );
        }
    }

    #[test]
    fn claim_hash_and_id_boundaries_are_validated_before_signing() {
        let key = test_keypair();
        for hash_len in [31, 32, 33] {
            let claim = FactClaim {
                fact_id: "f".into(),
                content_hash: vec![0; hash_len],
                timestamp_ms: 0,
                source_uri: None,
                store_id: None,
            };
            assert_eq!(
                create_signed_statement(&key, &claim).is_ok(),
                hash_len == 32
            );
        }
        for length in [4095, 4096, 4097] {
            let claim = FactClaim {
                fact_id: "f".repeat(length),
                content_hash: vec![0; 32],
                timestamp_ms: 0,
                source_uri: None,
                store_id: None,
            };
            assert_eq!(
                create_signed_statement(&key, &claim).is_ok(),
                length <= 4096
            );
        }
    }

    #[test]
    fn signed_statement_roundtrip() {
        let key = test_keypair();
        let claim = FactClaim {
            fact_id: "fact-001".to_string(),
            content_hash: vec![0xAB; 32],
            timestamp_ms: 1719878400000,
            source_uri: Some("https://example.com".to_string()),
            store_id: Some("store-1".to_string()),
        };

        let statement = create_signed_statement(&key, &claim).unwrap();
        let recovered = verify_statement(&key.verifying_key(), &statement).unwrap();

        assert_eq!(recovered.fact_id, "fact-001");
        assert_eq!(recovered.content_hash, vec![0xAB; 32]);
        assert_eq!(recovered.timestamp_ms, 1719878400000);
        assert_eq!(recovered.source_uri.as_deref(), Some("https://example.com"));
    }

    #[cfg(feature = "provenance-scitt")]
    #[test]
    fn receipt_url_takes_the_entry_id_as_one_segment() {
        assert_eq!(
            receipt_url("https://ts.example/", "2.34").unwrap(),
            "https://ts.example/entries/2.34/receipt"
        );
        for bad in ["", "..", "../admin", "a/b", "a?b", "a#b", "a b"] {
            assert!(receipt_url("https://ts.example", bad).is_err(), "{bad:?}");
        }
    }

    /// A service that accepts the connection and never answers fails the submission once
    /// the request timeout passes.
    #[cfg(feature = "provenance-scitt")]
    #[test]
    fn unresponsive_service_times_out() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let statement = create_signed_statement(
            &test_keypair(),
            &FactClaim {
                fact_id: "f".to_string(),
                content_hash: vec![0; 32],
                timestamp_ms: 0,
                source_uri: None,
                store_id: None,
            },
        )
        .unwrap();
        let (done, outcome) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = submit_statement_within(
                &endpoint,
                &statement,
                std::time::Duration::from_millis(300),
            );
            let _ = done.send(result.is_err());
        });
        let _held = listener.accept().unwrap();
        assert!(outcome
            .recv_timeout(std::time::Duration::from_secs(20))
            .expect("submission still blocked"));
    }

    #[test]
    fn wrong_key_rejected() {
        let key = test_keypair();
        let claim = FactClaim {
            fact_id: "f1".to_string(),
            content_hash: vec![0; 32],
            timestamp_ms: 0,
            source_uri: None,
            store_id: None,
        };

        let statement = create_signed_statement(&key, &claim).unwrap();
        let wrong_key = test_keypair();
        assert!(verify_statement(&wrong_key.verifying_key(), &statement).is_err());
    }
}
