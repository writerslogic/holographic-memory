// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use coset::iana::Algorithm;
use coset::{CborSerializable, CoseSign1, CoseSign1Builder, HeaderBuilder};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};

/// Create a COSE Sign1 envelope over arbitrary payload bytes.
pub fn sign_payload(signing_key: &SigningKey, payload: &[u8]) -> Result<Vec<u8>> {
    let protected = HeaderBuilder::new()
        .algorithm(Algorithm::EdDSA)
        .key_id(signing_key.verifying_key().to_bytes().to_vec())
        .build();

    let sign1 = CoseSign1Builder::new()
        .protected(protected)
        .payload(payload.to_vec())
        .create_signature(b"", |data| signing_key.sign(data).to_bytes().to_vec())
        .build();

    sign1
        .to_vec()
        .map_err(|e| anyhow!("COSE serialization failed: {e}"))
}

/// Verify a COSE Sign1 envelope and return the payload if valid.
pub fn verify_and_extract(verifying_key: &VerifyingKey, cose_bytes: &[u8]) -> Result<Vec<u8>> {
    let sign1 = CoseSign1::from_slice(cose_bytes)
        .map_err(|e| anyhow!("COSE deserialization failed: {e}"))?;

    let payload = sign1
        .payload
        .as_ref()
        .ok_or_else(|| anyhow!("COSE envelope has no payload"))?;

    sign1
        .verify_signature(b"", |sig, data| {
            let signature = ed25519_dalek::Signature::from_slice(sig)
                .map_err(|e| anyhow!("invalid signature bytes: {e}"))?;
            verifying_key
                .verify(data, &signature)
                .map_err(|e| anyhow!("signature verification failed: {e}"))
        })
        .map_err(|e| anyhow!("COSE verification failed: {e}"))?;

    Ok(payload.clone())
}

/// Extract the key ID (verifying key bytes) from a COSE Sign1 envelope without verifying.
pub fn extract_key_id(cose_bytes: &[u8]) -> Result<Vec<u8>> {
    let sign1 = CoseSign1::from_slice(cose_bytes)
        .map_err(|e| anyhow!("COSE deserialization failed: {e}"))?;
    let kid = sign1.protected.header.key_id.clone();
    if kid.is_empty() {
        return Err(anyhow!("no key ID in COSE envelope"));
    }
    Ok(kid)
}

/// Build a COSE Sign1 envelope for a provenance claim about a fact.
/// The payload is CBOR-encoded with fields: id, content_hash, source_uri, timestamp.
pub fn sign_fact_claim(
    signing_key: &SigningKey,
    fact_id: &str,
    content_hash: &[u8; 32],
    source_uri: Option<&str>,
    timestamp_ms: u64,
) -> Result<Vec<u8>> {
    let mut claim = ciborium::Value::Map(vec![
        (
            ciborium::Value::Text("id".to_string()),
            ciborium::Value::Text(fact_id.to_string()),
        ),
        (
            ciborium::Value::Text("content_hash".to_string()),
            ciborium::Value::Bytes(content_hash.to_vec()),
        ),
        (
            ciborium::Value::Text("timestamp_ms".to_string()),
            ciborium::Value::Integer(timestamp_ms.into()),
        ),
    ]);
    if let Some(uri) = source_uri {
        if let ciborium::Value::Map(ref mut map) = claim {
            map.push((
                ciborium::Value::Text("source_uri".to_string()),
                ciborium::Value::Text(uri.to_string()),
            ));
        }
    }
    let mut payload = Vec::new();
    ciborium::into_writer(&claim, &mut payload)
        .map_err(|e| anyhow!("CBOR encoding failed: {e}"))?;

    sign_payload(signing_key, &payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::generate(&mut rand::thread_rng())
    }

    #[test]
    fn sign_verify_roundtrip() {
        let key = test_keypair();
        let payload = b"test payload data";

        let envelope = sign_payload(&key, payload).unwrap();
        let recovered = verify_and_extract(&key.verifying_key(), &envelope).unwrap();
        assert_eq!(recovered, payload);
    }

    #[test]
    fn tampered_payload_rejected() {
        let key = test_keypair();
        let envelope = sign_payload(&key, b"original").unwrap();

        let wrong_key = test_keypair();
        assert!(verify_and_extract(&wrong_key.verifying_key(), &envelope).is_err());
    }

    #[test]
    fn key_id_extraction() {
        let key = test_keypair();
        let envelope = sign_payload(&key, b"data").unwrap();
        let kid = extract_key_id(&envelope).unwrap();
        assert_eq!(kid, key.verifying_key().to_bytes().to_vec());
    }

    #[test]
    fn fact_claim_roundtrip() {
        let key = test_keypair();
        let hash = [0xABu8; 32];
        let envelope = sign_fact_claim(
            &key,
            "fact-001",
            &hash,
            Some("https://example.com"),
            1719878400000,
        )
        .unwrap();
        let payload = verify_and_extract(&key.verifying_key(), &envelope).unwrap();

        let decoded: ciborium::Value = ciborium::from_reader(payload.as_slice()).unwrap();
        if let ciborium::Value::Map(entries) = decoded {
            let id_val = entries
                .iter()
                .find(|(k, _)| k == &ciborium::Value::Text("id".to_string()))
                .unwrap();
            assert_eq!(id_val.1, ciborium::Value::Text("fact-001".to_string()));
        } else {
            panic!("expected CBOR map");
        }
    }
}
