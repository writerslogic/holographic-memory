// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};

use super::did;

/// CAWG Identity Assertion per Creator Assertions Working Group spec 1.1.
/// Binds a verified creator identity to a C2PA manifest via a signed claim.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IdentityAssertion {
    pub signer_payload: SignerPayload,
    pub signature: String,
    pub pad: Option<Vec<u8>>,
}

/// The payload that the identity signer signs over.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignerPayload {
    pub referenced_assertions: Vec<HashedUri>,
    pub sig_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<IdentityClaim>,
}

/// A hashed URI reference to a C2PA assertion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashedUri {
    pub url: String,
    pub alg: String,
    pub hash: String,
}

/// The identity claim within a CAWG assertion.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IdentityClaim {
    pub did: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

/// Create a CAWG identity assertion binding a DID to referenced manifest assertions.
pub fn create_identity_assertion(
    signing_key: &SigningKey,
    referenced_assertions: Vec<HashedUri>,
    display_name: Option<&str>,
    provider: Option<&str>,
) -> Result<IdentityAssertion> {
    let issuer_did = did::did_key_from_ed25519(&signing_key.verifying_key().to_bytes());

    let payload = SignerPayload {
        referenced_assertions,
        sig_type: "cawg.ed25519".to_string(),
        identity: Some(IdentityClaim {
            did: issuer_did,
            display_name: display_name.map(String::from),
            provider: provider.map(String::from),
        }),
    };

    let payload_bytes = serde_json::to_vec(&payload)
        .map_err(|e| anyhow::anyhow!("CAWG payload serialization failed: {e}"))?;
    let signature = signing_key.sign(&payload_bytes);
    let sig_multibase = multibase::encode(multibase::Base::Base58Btc, signature.to_bytes());

    Ok(IdentityAssertion {
        signer_payload: payload,
        signature: sig_multibase,
        pad: None,
    })
}

/// Verify a CAWG identity assertion signature.
pub fn verify_identity_assertion(assertion: &IdentityAssertion) -> Result<()> {
    let identity = assertion
        .signer_payload
        .identity
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no identity in assertion"))?;

    let pk_bytes = did::ed25519_from_did_key(&identity.did)
        .or_else(|_| {
            // Try resolving as did:web via a provided DID document
            Err(anyhow::anyhow!(
                "did:web resolution requires DID document; use verify_identity_assertion_with_key"
            ))
        })
        .or_else(|_| did::ed25519_from_did_key(&identity.did))?;

    let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes)
        .map_err(|e| anyhow::anyhow!("invalid public key: {e}"))?;

    let (_, sig_bytes) = multibase::decode(&assertion.signature)
        .map_err(|e| anyhow::anyhow!("multibase decode failed: {e}"))?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow::anyhow!("invalid signature: {e}"))?;

    let payload_bytes = serde_json::to_vec(&assertion.signer_payload)
        .map_err(|e| anyhow::anyhow!("payload serialization failed: {e}"))?;

    ed25519_dalek::Verifier::verify(&verifying_key, &payload_bytes, &signature)
        .map_err(|e| anyhow::anyhow!("CAWG signature verification failed: {e}"))
}

/// Verify using an explicitly provided verifying key (for did:web or other non-did:key methods).
pub fn verify_identity_assertion_with_key(
    assertion: &IdentityAssertion,
    verifying_key: &ed25519_dalek::VerifyingKey,
) -> Result<()> {
    let (_, sig_bytes) = multibase::decode(&assertion.signature)
        .map_err(|e| anyhow::anyhow!("multibase decode failed: {e}"))?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow::anyhow!("invalid signature: {e}"))?;

    let payload_bytes = serde_json::to_vec(&assertion.signer_payload)
        .map_err(|e| anyhow::anyhow!("payload serialization failed: {e}"))?;

    ed25519_dalek::Verifier::verify(verifying_key, &payload_bytes, &signature)
        .map_err(|e| anyhow::anyhow!("CAWG signature verification failed: {e}"))
}

/// Build a HashedUri from a label and content bytes.
pub fn hash_assertion(label: &str, content: &[u8]) -> HashedUri {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(content);
    HashedUri {
        url: format!("self#jumbf=c2pa.assertions/{label}"),
        alg: "sha256".to_string(),
        hash: hash.iter().map(|b| format!("{b:02x}")).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::generate(&mut rand::thread_rng())
    }

    #[test]
    fn create_and_verify_identity_assertion() {
        let key = test_keypair();
        let refs = vec![hash_assertion("c2pa.hash.data", b"test content")];

        let assertion =
            create_identity_assertion(&key, refs, Some("Alice"), Some("writerslogic.com")).unwrap();

        assert!(assertion.signer_payload.identity.is_some());
        assert_eq!(
            assertion
                .signer_payload
                .identity
                .as_ref()
                .unwrap()
                .display_name
                .as_deref(),
            Some("Alice")
        );

        verify_identity_assertion(&assertion).unwrap();
    }

    #[test]
    fn tampered_assertion_rejected() {
        let key = test_keypair();
        let refs = vec![hash_assertion("c2pa.hash.data", b"content")];
        let mut assertion = create_identity_assertion(&key, refs, None, None).unwrap();
        assertion.signer_payload.sig_type = "tampered".to_string();
        assert!(verify_identity_assertion(&assertion).is_err());
    }

    #[test]
    fn verify_with_explicit_key() {
        let key = test_keypair();
        let refs = vec![hash_assertion("c2pa.actions", b"action data")];
        let assertion = create_identity_assertion(&key, refs, None, None).unwrap();
        verify_identity_assertion_with_key(&assertion, &key.verifying_key()).unwrap();

        let wrong_key = test_keypair();
        assert!(
            verify_identity_assertion_with_key(&assertion, &wrong_key.verifying_key()).is_err()
        );
    }
}
