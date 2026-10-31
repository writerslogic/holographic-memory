// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use coset::iana::Algorithm;
use coset::{
    AsCborValue, CborSerializable, CoseSign1, CoseSign1Builder, HeaderBuilder,
    TaggedCborSerializable,
};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};

/// Create a COSE Sign1 envelope over a payload of at most 16 MiB minus 1024 header bytes.
pub fn sign_payload(signing_key: &SigningKey, payload: &[u8]) -> Result<Vec<u8>> {
    anyhow::ensure!(
        payload.len() <= (16 << 20) - 1024,
        "COSE payload exceeds size limit"
    );
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

pub(crate) fn decode_sign1(bytes: &[u8]) -> Result<CoseSign1> {
    anyhow::ensure!(bytes.len() <= 16 << 20, "COSE envelope exceeds size limit");
    let mut cursor = std::io::Cursor::new(bytes);
    let value: ciborium::Value = ciborium::de::from_reader_with_recursion_limit(&mut cursor, 32)?;
    anyhow::ensure!(
        cursor.position() == bytes.len() as u64,
        "trailing bytes after COSE envelope"
    );
    let value = match value {
        ciborium::Value::Tag(18, value) => *value,
        value => value,
    };
    let ciborium::Value::Array(fields) = value else {
        return Err(anyhow!("COSE envelope is not an array"));
    };
    anyhow::ensure!(fields.len() == 4, "invalid COSE_Sign1 shape");
    let ciborium::Value::Bytes(raw) = &fields[0] else {
        return Err(anyhow!("protected headers are not bytes"));
    };
    let protected = if raw.is_empty() {
        ciborium::Value::Map(Vec::new())
    } else {
        let mut cursor = std::io::Cursor::new(raw);
        let headers = ciborium::de::from_reader_with_recursion_limit(&mut cursor, 32)?;
        anyhow::ensure!(
            cursor.position() == raw.len() as u64,
            "trailing protected header bytes"
        );
        headers
    };
    let ciborium::Value::Map(protected) = protected else {
        return Err(anyhow!("protected headers are not a map"));
    };
    let ciborium::Value::Map(unprotected) = &fields[1] else {
        return Err(anyhow!("unprotected headers are not a map"));
    };
    let critical = ciborium::Value::Integer(2.into());
    anyhow::ensure!(
        !unprotected.iter().any(|(k, _)| k == &critical)
            && !protected
                .iter()
                .any(|(k, _)| unprotected.iter().any(|(u, _)| k == u)),
        "COSE header maps overlap or crit is unprotected"
    );
    if let Some((_, value)) = protected.iter().find(|(key, _)| key == &critical) {
        anyhow::ensure!(
            matches!(value, ciborium::Value::Array(labels) if !labels.is_empty()),
            "COSE critical list must be nonempty"
        );
    }
    CoseSign1::from_tagged_slice(bytes)
        .or_else(|_| CoseSign1::from_slice(bytes))
        .map_err(|e| anyhow!("COSE deserialization failed: {e}"))
}

/// Verify a COSE Sign1 envelope and return the payload if valid.
pub fn verify_and_extract(verifying_key: &VerifyingKey, cose_bytes: &[u8]) -> Result<Vec<u8>> {
    let sign1 = decode_sign1(cose_bytes)?;

    let payload = sign1
        .payload
        .as_ref()
        .ok_or_else(|| anyhow!("COSE envelope has no payload"))?;

    verify_ed25519(&sign1, verifying_key).map_err(|e| anyhow!("COSE verification failed: {e}"))?;

    Ok(payload.clone())
}

/// Verify the Ed25519 WARM profile: a protected EdDSA algorithm, disjoint header maps,
/// understood critical headers and strict point/signature validation (RFC 9052 sections 3–4).
pub(crate) fn verify_ed25519(sign1: &CoseSign1, verifying_key: &VerifyingKey) -> Result<()> {
    use coset::{iana::HeaderParameter, Label, RegisteredLabelWithPrivate};
    if sign1.protected.header.alg != Some(RegisteredLabelWithPrivate::Assigned(Algorithm::EdDSA))
        || sign1.unprotected.alg.is_some()
    {
        return Err(anyhow!("COSE requires a protected EdDSA algorithm"));
    }
    let protected = sign1
        .protected
        .header
        .clone()
        .to_cbor_value()
        .map_err(|e| anyhow!("invalid protected headers: {e}"))?;
    let unprotected = sign1
        .unprotected
        .clone()
        .to_cbor_value()
        .map_err(|e| anyhow!("invalid unprotected headers: {e}"))?;
    let ciborium::Value::Map(protected) = protected else {
        unreachable!()
    };
    let ciborium::Value::Map(unprotected) = unprotected else {
        unreachable!()
    };
    if !sign1.unprotected.crit.is_empty()
        || protected
            .iter()
            .any(|(k, _)| unprotected.iter().any(|(u, _)| u == k))
    {
        return Err(anyhow!(
            "COSE header maps overlap or contain unprotected critical headers"
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    for label in &sign1.protected.header.crit {
        let key = match label {
            RegisteredLabelWithPrivate::Assigned(
                p @ (HeaderParameter::Alg | HeaderParameter::ContentType | HeaderParameter::Kid),
            ) => Label::Int(*p as i64),
            _ => return Err(anyhow!("COSE has an unsupported critical header")),
        };
        let cbor = key
            .clone()
            .to_cbor_value()
            .map_err(|e| anyhow!("invalid critical header: {e}"))?;
        if !seen.insert(key) || !protected.iter().any(|(k, _)| k == &cbor) {
            return Err(anyhow!("COSE critical header is duplicated or absent"));
        }
    }
    sign1.verify_signature(b"", |sig, data| {
        let signature = ed25519_dalek::Signature::from_slice(sig)
            .map_err(|e| anyhow!("invalid signature bytes: {e}"))?;
        verifying_key
            .verify_strict(data, &signature)
            .map_err(|e| anyhow!("signature verification failed: {e}"))
    })
}

/// Extract the key ID (verifying key bytes) from a COSE Sign1 envelope without verifying.
pub fn extract_key_id(cose_bytes: &[u8]) -> Result<Vec<u8>> {
    let sign1 = decode_sign1(cose_bytes)?;
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
        SigningKey::from_bytes(&rand::random())
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
    fn audit_critical_and_algorithm_headers_fail_closed() {
        let key = test_keypair();
        let signed = |protected: coset::Header, unprotected: coset::Header| {
            CoseSign1Builder::new()
                .protected(protected)
                .unprotected(unprotected)
                .payload(b"data".to_vec())
                .create_signature(b"", |data| key.sign(data).to_bytes().to_vec())
                .build()
                .to_vec()
                .unwrap()
        };
        let alg = || HeaderBuilder::new().algorithm(Algorithm::EdDSA).build();
        assert!(verify_and_extract(
            &key.verifying_key(),
            &signed(coset::Header::default(), alg())
        )
        .is_err());
        assert!(verify_and_extract(&key.verifying_key(), &signed(alg(), alg())).is_err());
        let mut critical = alg();
        critical
            .crit
            .push(coset::RegisteredLabelWithPrivate::Text("unknown".into()));
        critical.rest.push((
            coset::Label::Text("unknown".into()),
            ciborium::Value::Bool(true),
        ));
        assert!(verify_and_extract(
            &key.verifying_key(),
            &signed(critical, coset::Header::default())
        )
        .is_err());
        let mut absent = alg();
        absent
            .crit
            .push(coset::RegisteredLabelWithPrivate::Assigned(
                coset::iana::HeaderParameter::ContentType,
            ));
        assert!(verify_and_extract(
            &key.verifying_key(),
            &signed(absent, coset::Header::default())
        )
        .is_err());
    }

    #[test]
    fn empty_critical_lists_and_overlapping_empty_headers_are_rejected() {
        let key = test_keypair();
        let envelope = sign_payload(&key, b"data").unwrap();
        for unprotected in [
            ciborium::Value::Map(vec![(
                ciborium::Value::Integer(2.into()),
                ciborium::Value::Array(vec![]),
            )]),
            ciborium::Value::Map(vec![(
                ciborium::Value::Integer(4.into()),
                ciborium::Value::Bytes(vec![]),
            )]),
        ] {
            let mut raw: ciborium::Value = ciborium::from_reader(envelope.as_slice()).unwrap();
            let ciborium::Value::Array(ref mut fields) = raw else {
                unreachable!()
            };
            fields[1] = unprotected;
            let mut bytes = Vec::new();
            ciborium::into_writer(&raw, &mut bytes).unwrap();
            assert!(verify_and_extract(&key.verifying_key(), &bytes).is_err());
        }
    }

    #[test]
    fn wrong_key_or_tampered_payload_is_rejected() {
        let key = test_keypair();
        let envelope = sign_payload(&key, b"original").unwrap();
        let wrong_key = test_keypair();
        assert!(verify_and_extract(&wrong_key.verifying_key(), &envelope).is_err());

        let mut sign1 = CoseSign1::from_slice(&envelope).unwrap();
        sign1.payload = Some(b"tampered".to_vec());
        let tampered = sign1.to_vec().unwrap();
        assert!(verify_and_extract(&key.verifying_key(), &tampered).is_err());
    }

    #[test]
    fn envelope_naming_another_algorithm_is_rejected() {
        let key = test_keypair();
        let mut sign1 = CoseSign1::from_slice(&sign_payload(&key, b"data").unwrap()).unwrap();
        let protected = HeaderBuilder::new()
            .algorithm(Algorithm::ES256)
            .key_id(key.verifying_key().to_bytes().to_vec())
            .build();
        let resigned = CoseSign1Builder::new()
            .protected(protected)
            .payload(sign1.payload.take().unwrap())
            .create_signature(b"", |data| key.sign(data).to_bytes().to_vec())
            .build()
            .to_vec()
            .unwrap();
        assert!(verify_and_extract(&key.verifying_key(), &resigned).is_err());
    }

    /// Under the identity point as public key, the signature (R = identity, s = 0) satisfies
    /// the plain verification equation for every message.
    #[test]
    fn small_order_key_signature_is_rejected() {
        let mut identity = [0u8; 32];
        identity[0] = 1;
        let weak = VerifyingKey::from_bytes(&identity).unwrap();
        let mut sign1 = CoseSign1Builder::new()
            .protected(HeaderBuilder::new().algorithm(Algorithm::EdDSA).build())
            .payload(b"any statement".to_vec())
            .build();
        let mut signature = vec![0u8; 64];
        signature[0] = 1;
        sign1.signature = signature;
        let envelope = sign1.to_vec().unwrap();
        assert!(verify_and_extract(&weak, &envelope).is_err());
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

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn verifies_cogmem_cross_conformance_vector() {
        // Cross-implementation conformance: a COSE_Sign1 EdDSA signed statement produced
        // by cogmem (Python / cbor2). WARM must verify it and recover the payload, proving
        // the two projects emit byte-compatible signed statements. See tests/vectors/.
        let pk = unhex("4b5f52db17ebdeb555101922e89beac9b43e864086b02e4529951d7f491f0cfa");
        let cose = unhex("845838a3012703706170706c69636174696f6e2f63626f720458204b5f52db17ebdeb555101922e89beac9b43e864086b02e4529951d7f491f0cfaa05887a6636973736c6469643a6b65793a74657374686d656d6f7279496468766563746f722d316a6d656d6f7279547970656472756c65656576656e7467637265617465646d73746174656d656e74486173685820000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f6b74696d657374616d704d731b0000019070bdb80058402fc837df236fa9d4d3a8bad98148ea020b4ea7bbb717d57819675eaee65264b280f5ec8d22f21dc9531f91fe4571305f84f05396cfde33ac897eac52a4937e09");
        let expected = unhex("a6636973736c6469643a6b65793a74657374686d656d6f7279496468766563746f722d316a6d656d6f7279547970656472756c65656576656e7467637265617465646d73746174656d656e74486173685820000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f6b74696d657374616d704d731b0000019070bdb800");
        let vk = VerifyingKey::from_bytes(&pk.try_into().unwrap()).unwrap();
        let payload = verify_and_extract(&vk, &cose).unwrap();
        assert_eq!(payload, expected);
    }
}
