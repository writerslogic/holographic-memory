// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, ensure, Result};
use ciborium::Value;
use ed25519_dalek::VerifyingKey;
use sha2::{Digest, Sha256};

use super::{SignedStatement, UnverifiedReceipt};
use crate::core::provenance::{cose, did, trust::TrustStore};

/// A caller-authenticated current checkpoint. Its acquisition, freshness and consistency
/// with earlier checkpoints are outside this verifier.
pub struct ReceiptPolicy<'a> {
    pub issuer_key: &'a VerifyingKey,
    pub issuer_trust: &'a TrustStore,
    pub service_key: &'a VerifyingKey,
    pub service_trust: &'a TrustStore,
    pub service_issuer: &'a str,
    pub subject: &'a str,
    pub current_root: [u8; 32],
    pub checkpoint_not_before_ms: u64,
    pub checkpoint_valid_until_ms: u64,
    pub now_ms: u64,
}

/// A statement included under the caller's pinned checkpoint and trust policy.
/// This does not establish append-only consistency, non-equivocation or revocation status.
#[derive(Debug)]
pub struct VerifiedReceipt {
    root: [u8; 32],
    claim: super::FactClaim,
}
impl VerifiedReceipt {
    pub fn root(&self) -> &[u8; 32] {
        &self.root
    }
    pub fn claim(&self) -> &super::FactClaim {
        &self.claim
    }
}

fn value_at(headers: &[(coset::Label, Value)], label: i64) -> Result<&Value> {
    let mut values = headers
        .iter()
        .filter(|(k, _)| *k == coset::Label::Int(label));
    let (_, value) = values
        .next()
        .ok_or_else(|| anyhow!("missing header {label}"))?;
    ensure!(values.next().is_none(), "duplicate header {label}");
    Ok(value)
}

fn claims(value: &Value, issuer: &str, subject: &str) -> Result<()> {
    let Value::Map(map) = value else {
        return Err(anyhow!("CWT claims must be a map"));
    };
    ensure!(map.len() == 2, "unsupported CWT claims");
    for (label, expected) in [(1, issuer), (2, subject)] {
        let mut matches = map
            .iter()
            .filter(|(k, _)| *k == Value::Integer(label.into()));
        let (_, val) = matches.next().ok_or_else(|| anyhow!("missing CWT claim"))?;
        ensure!(
            matches.next().is_none() && val == &Value::Text(expected.to_owned()),
            "CWT identity mismatch"
        );
    }
    Ok(())
}

/// Verify a detached RFC 9942 RFC9162_SHA256 receipt for an WARM Ed25519 signed
/// statement. This profile requires a single inclusion proof and an externally
/// authenticated current root. Unsupported proof types and algorithms fail closed.
pub fn verify_receipt(
    statement: &SignedStatement,
    receipt: &UnverifiedReceipt,
    policy: &ReceiptPolicy<'_>,
) -> Result<VerifiedReceipt> {
    ensure!(
        policy.checkpoint_not_before_ms <= policy.now_ms
            && policy.now_ms < policy.checkpoint_valid_until_ms,
        "checkpoint is not current"
    );
    ensure!(
        receipt.receipt_bytes.len() <= 8192,
        "receipt exceeds size limit"
    );
    ensure!(
        policy.issuer_trust.is_trusted(policy.issuer_key)
            && policy.service_trust.is_trusted(policy.service_key),
        "untrusted issuer or service"
    );
    ensure!(
        !policy.subject.is_empty()
            && policy.subject.len() <= 4096
            && !policy.service_issuer.is_empty()
            && policy.service_issuer.len() <= 1024,
        "invalid identity"
    );
    let claim = super::verify_statement(policy.issuer_key, statement)?;
    ensure!(
        claim.fact_id == policy.subject,
        "statement subject mismatch"
    );
    let s = cose::decode_sign1(&statement.cose_bytes)?;
    ensure!(
        s.protected.header.content_type
            == Some(coset::RegisteredLabel::Text(
                super::CLAIM_CONTENT_TYPE.into()
            )),
        "unsupported statement media type"
    );
    ensure!(
        s.unprotected == coset::Header::default(),
        "statement must have empty unprotected headers"
    );
    ensure!(
        s.protected.header.key_id == policy.issuer_key.to_bytes(),
        "statement key identifier mismatch"
    );
    claims(
        value_at(&s.protected.header.rest, 15)?,
        &did::did_key_from_ed25519(&policy.issuer_key.to_bytes()),
        policy.subject,
    )?;
    let mut r = cose::decode_sign1(&receipt.receipt_bytes)?;
    ensure!(r.payload.is_none(), "receipt payload must be detached");
    ensure!(
        r.protected.header.key_id == policy.service_key.to_bytes(),
        "service key identifier mismatch"
    );
    ensure!(
        value_at(&r.protected.header.rest, 395)? == &Value::Integer(1.into()),
        "unsupported VDS"
    );
    claims(
        value_at(&r.protected.header.rest, 15)?,
        policy.service_issuer,
        policy.subject,
    )?;
    let vdp = value_at(&r.unprotected.rest, 396)?;
    let Value::Map(map) = vdp else {
        return Err(anyhow!("invalid proof map"));
    };
    ensure!(
        map.len() == 1 && map[0].0 == Value::Integer((-1).into()),
        "unsupported proof type"
    );
    let Value::Array(proofs) = &map[0].1 else {
        return Err(anyhow!("invalid proofs"));
    };
    ensure!(proofs.len() == 1, "exactly one proof required");
    let Value::Bytes(encoded) = &proofs[0] else {
        return Err(anyhow!("invalid proof encoding"));
    };
    ensure!(encoded.len() <= 2304, "proof exceeds size limit");
    let mut cursor = std::io::Cursor::new(encoded);
    let proof: Value = ciborium::de::from_reader_with_recursion_limit(&mut cursor, 8)?;
    ensure!(
        cursor.position() == encoded.len() as u64,
        "trailing proof bytes"
    );
    let Value::Array(parts) = proof else {
        return Err(anyhow!("invalid proof"));
    };
    ensure!(parts.len() == 3, "invalid proof shape");
    let unsigned = |v: &Value| -> Result<u64> {
        match v {
            Value::Integer(i) => u64::try_from(*i).map_err(|_| anyhow!("invalid proof integer")),
            _ => Err(anyhow!("invalid proof integer")),
        }
    };
    let size = unsigned(&parts[0])?;
    let index = unsigned(&parts[1])?;
    ensure!(size > 1 && index < size, "invalid tree position");
    let Value::Array(path) = &parts[2] else {
        return Err(anyhow!("invalid proof path"));
    };
    ensure!(
        !path.is_empty() && path.len() <= 64,
        "invalid proof path length"
    );
    let hash = inclusion_root(&statement.cose_bytes, size, index, path)?;
    ensure!(
        hash == policy.current_root,
        "proof does not match current checkpoint"
    );
    r.payload = Some(hash.to_vec());
    cose::verify_ed25519(&r, policy.service_key)?;
    Ok(VerifiedReceipt { root: hash, claim })
}

fn inclusion_root(entry: &[u8], size: u64, index: u64, path: &[Value]) -> Result<[u8; 32]> {
    ensure!(
        size > 1 && index < size && !path.is_empty() && path.len() <= 64,
        "invalid proof bounds"
    );
    let mut leaf = Sha256::new();
    leaf.update([0]);
    leaf.update(entry);
    let mut hash: [u8; 32] = leaf.finalize().into();
    let mut fn_index = index;
    let mut sn = size - 1;
    for sibling in path {
        let Value::Bytes(sibling) = sibling else {
            return Err(anyhow!("invalid path hash"));
        };
        ensure!(
            sibling.len() == 32 && sn != 0,
            "invalid path hash or length"
        );
        let mut h = Sha256::new();
        h.update([1]);
        if fn_index & 1 == 1 || fn_index == sn {
            h.update(sibling);
            h.update(hash);
            while fn_index & 1 == 0 && fn_index != 0 {
                fn_index >>= 1;
                sn >>= 1;
            }
        } else {
            h.update(hash);
            h.update(sibling);
        }
        hash = h.finalize().into();
        fn_index >>= 1;
        sn >>= 1;
    }
    ensure!(sn == 0, "incomplete proof");
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coset::{CoseSign1Builder, HeaderBuilder, TaggedCborSerializable};
    use ed25519_dalek::{Signer, SigningKey};

    fn merkle(leaves: &[Vec<u8>]) -> [u8; 32] {
        if leaves.len() == 1 {
            return Sha256::digest([&[0][..], &leaves[0]].concat()).into();
        }
        let split = 1usize << (usize::BITS - (leaves.len() - 1).leading_zeros() - 1);
        Sha256::digest(
            [
                &[1][..],
                &merkle(&leaves[..split]),
                &merkle(&leaves[split..]),
            ]
            .concat(),
        )
        .into()
    }

    fn proof(leaves: &[Vec<u8>], index: usize) -> Vec<Value> {
        if leaves.len() == 1 {
            return Vec::new();
        }
        let split = 1usize << (usize::BITS - (leaves.len() - 1).leading_zeros() - 1);
        let (mut path, sibling) = if index < split {
            (proof(&leaves[..split], index), merkle(&leaves[split..]))
        } else {
            (
                proof(&leaves[split..], index - split),
                merkle(&leaves[..split]),
            )
        };
        path.push(Value::Bytes(sibling.to_vec()));
        path
    }

    #[test]
    fn iterative_verifier_matches_recursive_rfc9162_tree_oracle() {
        for size in 2..=17 {
            let leaves: Vec<_> = (0..size).map(|i| vec![i as u8; i + 1]).collect();
            for index in 0..size {
                let path = proof(&leaves, index);
                assert_eq!(
                    inclusion_root(&leaves[index], size as u64, index as u64, &path).unwrap(),
                    merkle(&leaves)
                );
                assert!(inclusion_root(&leaves[index], size as u64, size as u64, &path).is_err());
                let mut extra = path.clone();
                extra.push(Value::Bytes(vec![0; 32]));
                assert!(inclusion_root(&leaves[index], size as u64, index as u64, &extra).is_err());
                let mut short = path;
                short.pop();
                assert!(inclusion_root(&leaves[index], size as u64, index as u64, &short).is_err());
            }
        }
        for length in [63, 64, 65] {
            let path = vec![Value::Bytes(vec![0; 32]); length];
            assert_eq!(
                inclusion_root(b"leaf", u64::MAX, 0, &path).is_ok(),
                length == 64
            );
        }
        for length in [31, 32, 33] {
            assert_eq!(
                inclusion_root(b"leaf", 2, 0, &[Value::Bytes(vec![0; length])]).is_ok(),
                length == 32
            );
        }
    }

    fn fixture(
        subject: &str,
    ) -> (
        SignedStatement,
        UnverifiedReceipt,
        SigningKey,
        SigningKey,
        [u8; 32],
    ) {
        let issuer = SigningKey::from_bytes(&[7; 32]);
        let service = SigningKey::from_bytes(&[8; 32]);
        let statement = super::super::create_signed_statement(
            &issuer,
            &super::super::FactClaim {
                fact_id: subject.into(),
                content_hash: vec![1; 32],
                timestamp_ms: 1,
                source_uri: None,
                store_id: None,
            },
        )
        .unwrap();
        let leaf = Sha256::digest([&[0u8][..], &statement.cose_bytes].concat());
        let sibling = [4u8; 32];
        let root: [u8; 32] =
            Sha256::digest([&[1u8][..], leaf.as_slice(), &sibling].concat()).into();
        let proof = Value::Array(vec![
            Value::Integer(2.into()),
            Value::Integer(0.into()),
            Value::Array(vec![Value::Bytes(sibling.to_vec())]),
        ]);
        let mut proof_bytes = Vec::new();
        ciborium::into_writer(&proof, &mut proof_bytes).unwrap();
        let protected = HeaderBuilder::new()
            .algorithm(coset::iana::Algorithm::EdDSA)
            .key_id(service.verifying_key().to_bytes().to_vec())
            .value(395, Value::Integer(1.into()))
            .value(
                15,
                Value::Map(vec![
                    (Value::Integer(1.into()), Value::Text("service".into())),
                    (Value::Integer(2.into()), Value::Text(subject.into())),
                ]),
            )
            .build();
        let unprotected = HeaderBuilder::new()
            .value(
                396,
                Value::Map(vec![(
                    Value::Integer((-1).into()),
                    Value::Array(vec![Value::Bytes(proof_bytes)]),
                )]),
            )
            .build();
        let sign1 = CoseSign1Builder::new()
            .protected(protected)
            .unprotected(unprotected)
            .create_detached_signature(&root, b"", |data| service.sign(data).to_bytes().to_vec())
            .build();
        let receipt = UnverifiedReceipt {
            entry_id: "1".into(),
            log_id: None,
            receipt_bytes: sign1.to_tagged_vec().unwrap(),
        };
        (statement, receipt, issuer, service, root)
    }

    #[test]
    fn generated_subject_boundaries_are_supported_by_receipt_verification() {
        for length in [1023, 1024, 1025, 4095, 4096] {
            let subject = "s".repeat(length);
            let (statement, receipt, issuer, service, root) = fixture(&subject);
            let ik = issuer.verifying_key();
            let sk = service.verifying_key();
            let it = TrustStore::trusting_key(&ik);
            let st = TrustStore::trusting_key(&sk);
            let policy = ReceiptPolicy {
                issuer_key: &ik,
                issuer_trust: &it,
                service_key: &sk,
                service_trust: &st,
                service_issuer: "service",
                subject: &subject,
                current_root: root,
                checkpoint_not_before_ms: 1,
                checkpoint_valid_until_ms: 3,
                now_ms: 2,
            };
            assert!(
                verify_receipt(&statement, &receipt, &policy).is_ok(),
                "length {length}"
            );
        }
    }

    #[test]
    fn pinned_receipt_requires_both_trust_and_current_checkpoint() {
        let (statement, mut receipt, issuer, service, root) = fixture("fact");
        let ik = issuer.verifying_key();
        let sk = service.verifying_key();
        let it = TrustStore::trusting_key(&ik);
        let st = TrustStore::trusting_key(&sk);
        let mut policy = ReceiptPolicy {
            issuer_key: &ik,
            issuer_trust: &it,
            service_key: &sk,
            service_trust: &st,
            service_issuer: "service",
            subject: "fact",
            current_root: root,
            checkpoint_not_before_ms: 1,
            checkpoint_valid_until_ms: 3,
            now_ms: 2,
        };
        assert!(verify_receipt(&statement, &receipt, &policy).is_ok());
        policy.now_ms = 0;
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
        policy.now_ms = 3;
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
        policy.now_ms = 2;
        policy.current_root = [0; 32];
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
        policy.current_root = root;
        let empty = TrustStore::new();
        policy.service_trust = &empty;
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
        policy.service_trust = &st;
        policy.subject = "other";
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
        policy.subject = "fact";
        let other_statement = super::super::create_signed_statement(
            &issuer,
            &super::super::FactClaim {
                fact_id: "fact".into(),
                content_hash: vec![2; 32],
                timestamp_ms: 1,
                source_uri: None,
                store_id: None,
            },
        )
        .unwrap();
        assert!(verify_receipt(&other_statement, &receipt, &policy).is_err());
        let empty_issuer = TrustStore::new();
        policy.issuer_trust = &empty_issuer;
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
        policy.issuer_trust = &it;
        let original = receipt.receipt_bytes.clone();
        let mut unknown_critical = cose::decode_sign1(&original).unwrap();
        unknown_critical.protected.original_data = None;
        unknown_critical
            .protected
            .header
            .crit
            .push(coset::RegisteredLabelWithPrivate::Text("unknown".into()));
        unknown_critical
            .protected
            .header
            .rest
            .push((coset::Label::Text("unknown".into()), Value::Bool(true)));
        receipt.receipt_bytes = unknown_critical.to_tagged_vec().unwrap();
        assert!(verify_receipt(&statement, &receipt, &policy)
            .unwrap_err()
            .to_string()
            .contains("critical"));
        receipt.receipt_bytes = original.clone();
        let last = receipt.receipt_bytes.len() - 1;
        receipt.receipt_bytes[last] ^= 1;
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
        receipt.receipt_bytes = original;
        receipt.receipt_bytes.resize(8193, 0);
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
        receipt.receipt_bytes.truncate(12);
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
    }
}
