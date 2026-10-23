// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use coset::{CborSerializable, CoseSign1, CoseSign1Builder, HeaderBuilder};
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};

const SCITT_CONTENT_TYPE: &str = "application/cbor";

/// A SCITT signed statement: a COSE Sign1 envelope with SCITT-specific headers.
/// Per draft-ietf-scitt-architecture, the payload is the claim (fact hash + metadata).
#[derive(Clone, Debug)]
pub struct SignedStatement {
    pub cose_bytes: Vec<u8>,
}

/// A SCITT transparency receipt returned by the Transparency Service.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransparencyReceipt {
    pub entry_id: String,
    pub receipt_bytes: Vec<u8>,
    pub log_id: Option<String>,
}

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
        .content_type(SCITT_CONTENT_TYPE.to_string())
        .build();

    let sign1 = CoseSign1Builder::new()
        .protected(protected)
        .payload(payload)
        .create_signature(b"", |data| signing_key.sign(data).to_bytes().to_vec())
        .build();

    let cose_bytes = sign1
        .to_vec()
        .map_err(|e| anyhow!("COSE serialization failed: {e}"))?;

    Ok(SignedStatement { cose_bytes })
}

/// Bound on each request to the Transparency Service, so an unresponsive service fails the
/// submission instead of blocking the caller (fact provenance creation) indefinitely.
#[cfg(feature = "provenance-scitt")]
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Submit a signed statement to a SCITT Transparency Service and get a receipt.
#[cfg(feature = "provenance-scitt")]
pub fn submit_statement(
    endpoint: &str,
    statement: &SignedStatement,
) -> Result<TransparencyReceipt> {
    submit_statement_within(endpoint, statement, REQUEST_TIMEOUT)
}

#[cfg(feature = "provenance-scitt")]
fn submit_statement_within(
    endpoint: &str,
    statement: &SignedStatement,
    timeout: std::time::Duration,
) -> Result<TransparencyReceipt> {
    let agent = ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build(),
    );
    let url = format!("{}/entries", endpoint.trim_end_matches('/'));

    let response = agent
        .post(&url)
        .header("Content-Type", SCITT_CONTENT_TYPE)
        .send(&statement.cose_bytes)
        .map_err(|e| anyhow!("SCITT submission failed: {e}"))?;

    let status = response.status().as_u16();
    if status != 201 && status != 200 {
        let body = response.into_body().read_to_string().unwrap_or_default();
        return Err(anyhow!("SCITT service returned {status}: {body}"));
    }

    if status == 201 {
        let location = response
            .headers()
            .get("Location")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let operation_url = if location.starts_with("http") {
            location
        } else {
            format!("{}{}", endpoint.trim_end_matches('/'), location)
        };
        return poll_for_receipt(&agent, endpoint, &operation_url);
    }

    let receipt_bytes = response
        .into_body()
        .read_to_vec()
        .map_err(|e| anyhow!("failed to read receipt: {e}"))?;

    Ok(TransparencyReceipt {
        entry_id: String::new(),
        receipt_bytes,
        log_id: None,
    })
}

#[cfg(feature = "provenance-scitt")]
fn poll_for_receipt(
    agent: &ureq::Agent,
    endpoint: &str,
    operation_url: &str,
) -> Result<TransparencyReceipt> {
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_secs(1));

        let resp = agent
            .get(operation_url)
            .call()
            .map_err(|e| anyhow!("polling failed: {e}"))?;

        if resp.status().as_u16() == 200 {
            let body_str = resp
                .into_body()
                .read_to_string()
                .map_err(|e| anyhow!("response read failed: {e}"))?;
            let body: serde_json::Value =
                serde_json::from_str(&body_str).map_err(|e| anyhow!("JSON parse failed: {e}"))?;

            if body.get("status").and_then(|s| s.as_str()) == Some("succeeded") {
                let entry_id = body
                    .get("entryId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let receipt_url = receipt_url(endpoint, &entry_id)?;
                let receipt_resp = agent
                    .get(&receipt_url)
                    .call()
                    .map_err(|e| anyhow!("receipt fetch failed: {e}"))?;
                let receipt_bytes = receipt_resp
                    .into_body()
                    .read_to_vec()
                    .map_err(|e| anyhow!("receipt read failed: {e}"))?;

                return Ok(TransparencyReceipt {
                    entry_id,
                    receipt_bytes,
                    log_id: body.get("logId").and_then(|v| v.as_str()).map(String::from),
                });
            }
        }
    }
    Err(anyhow!("SCITT receipt polling timed out after 30 seconds"))
}

/// The receipt URL of `entry_id` under `endpoint`. The id comes from the service's response
/// and becomes one path segment, so it must be a non-empty run of URL-safe characters: a `/`,
/// `..`, `?` or `#` in it would point the receipt request elsewhere.
#[cfg(feature = "provenance-scitt")]
fn receipt_url(endpoint: &str, entry_id: &str) -> Result<String> {
    let segment_ok = !entry_id.is_empty()
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
    let sign1 = CoseSign1::from_slice(&statement.cose_bytes)
        .map_err(|e| anyhow!("COSE deserialization failed: {e}"))?;

    super::cose::verify_ed25519(&sign1, verifying_key)
        .map_err(|e| anyhow!("SCITT statement verification failed: {e}"))?;

    let payload = sign1
        .payload
        .as_ref()
        .ok_or_else(|| anyhow!("no payload in statement"))?;

    let decoded: ciborium::Value = ciborium::from_reader(payload.as_slice())
        .map_err(|e| anyhow!("CBOR decode failed: {e}"))?;

    parse_fact_claim(&decoded)
}

fn parse_fact_claim(val: &ciborium::Value) -> Result<FactClaim> {
    let map = match val {
        ciborium::Value::Map(m) => m,
        _ => return Err(anyhow!("expected CBOR map")),
    };

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
