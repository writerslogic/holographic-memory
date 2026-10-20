// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use coset::{CborSerializable, CoseSign1, CoseSign1Builder, HeaderBuilder};
use ed25519_dalek::{Signer, SigningKey, Verifier};
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

/// Submit a signed statement to a SCITT Transparency Service and get a receipt.
#[cfg(feature = "provenance-scitt")]
pub fn submit_statement(
    endpoint: &str,
    statement: &SignedStatement,
) -> Result<TransparencyReceipt> {
    let url = format!("{}/entries", endpoint.trim_end_matches('/'));

    let response = ureq::post(&url)
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
        return poll_for_receipt(endpoint, &operation_url);
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
fn poll_for_receipt(endpoint: &str, operation_url: &str) -> Result<TransparencyReceipt> {
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_secs(1));

        let resp = ureq::get(operation_url)
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

                let receipt_url = format!(
                    "{}/entries/{}/receipt",
                    endpoint.trim_end_matches('/'),
                    entry_id
                );
                let receipt_resp = ureq::get(&receipt_url)
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

/// Verify a SCITT signed statement's COSE signature (local verification, no TS needed).
pub fn verify_statement(
    verifying_key: &ed25519_dalek::VerifyingKey,
    statement: &SignedStatement,
) -> Result<FactClaim> {
    let sign1 = CoseSign1::from_slice(&statement.cose_bytes)
        .map_err(|e| anyhow!("COSE deserialization failed: {e}"))?;

    sign1
        .verify_signature(b"", |sig, data| {
            let signature = ed25519_dalek::Signature::from_slice(sig)
                .map_err(|e| anyhow!("invalid signature: {e}"))?;
            verifying_key
                .verify(data, &signature)
                .map_err(|e| anyhow!("verification failed: {e}"))
        })
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
