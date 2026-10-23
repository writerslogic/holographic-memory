// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use base64::engine::{general_purpose::GeneralPurpose, DecodePaddingMode, GeneralPurposeConfig};
use base64::{alphabet, Engine};
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A Sigstore bundle containing a signature, verification material, and optional
/// transparency log entry. Follows the Sigstore bundle format v0.3.
/// For local-only operation, the Rekor log entry is omitted; for online mode
/// (behind provenance-sigstore feature), it can be populated via submission.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SigstoreBundle {
    pub media_type: String,
    pub verification_material: VerificationMaterial,
    pub message_signature: MessageSignature,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tlog_entries: Option<Vec<TlogEntry>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationMaterial {
    pub public_key: PublicKeyInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub certificate_chain: Option<CertificateChain>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicKeyInfo {
    pub algorithm: String,
    pub raw_bytes: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CertificateChain {
    pub certificates: Vec<CertificateEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CertificateEntry {
    pub raw_bytes: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageSignature {
    pub message_digest: MessageDigest,
    pub signature: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageDigest {
    pub algorithm: String,
    pub digest: String,
}

/// A Rekor transparency log entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TlogEntry {
    pub log_index: u64,
    pub log_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integrated_time: Option<u64>,
    pub inclusion_proof: Option<serde_json::Value>,
    pub canonicalized_body: String,
}

/// Create a Sigstore bundle for a content payload using local Ed25519 signing.
/// This produces a bundle without Rekor/Fulcio (keyful signing mode).
pub fn create_local_bundle(
    signing_key: &SigningKey,
    content: &[u8],
    identity: Option<&str>,
) -> Result<SigstoreBundle> {
    let digest = Sha256::digest(content);
    let digest_hex = super::hex_encode(&digest);

    let signature = signing_key.sign(content);
    let sig_base64 = base64_encode(&signature.to_bytes());

    let pk_bytes = signing_key.verifying_key().to_bytes();
    let pk_base64 = base64_encode(&pk_bytes);

    Ok(SigstoreBundle {
        media_type: "application/vnd.dev.sigstore.bundle.v0.3+json".to_string(),
        verification_material: VerificationMaterial {
            public_key: PublicKeyInfo {
                algorithm: "ed25519".to_string(),
                raw_bytes: pk_base64,
                identity: identity.map(String::from),
            },
            certificate_chain: None,
        },
        message_signature: MessageSignature {
            message_digest: MessageDigest {
                algorithm: "sha256".to_string(),
                digest: digest_hex,
            },
            signature: sig_base64,
        },
        tlog_entries: None,
    })
}

/// Verify a Sigstore bundle's signature against the embedded public key.
/// Does NOT verify Rekor inclusion or certificate chains (those require online access).
pub fn verify_bundle(bundle: &SigstoreBundle, content: &[u8]) -> Result<()> {
    let digest = Sha256::digest(content);
    let digest_hex = super::hex_encode(&digest);
    if digest_hex != bundle.message_signature.message_digest.digest {
        return Err(anyhow!("content digest mismatch"));
    }

    let pk_bytes = base64_decode(&bundle.verification_material.public_key.raw_bytes)?;
    if pk_bytes.len() != 32 {
        return Err(anyhow!("invalid public key length: {}", pk_bytes.len()));
    }
    let mut pk_arr = [0u8; 32];
    pk_arr.copy_from_slice(&pk_bytes);
    let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk_arr)
        .map_err(|e| anyhow!("invalid public key: {e}"))?;

    let sig_bytes = base64_decode(&bundle.message_signature.signature)?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow!("invalid signature: {e}"))?;

    ed25519_dalek::Verifier::verify(&verifying_key, content, &signature)
        .map_err(|e| anyhow!("signature verification failed: {e}"))
}

/// Verify a bundle against a specific verifying key (ignores embedded key).
pub fn verify_bundle_with_key(
    bundle: &SigstoreBundle,
    content: &[u8],
    verifying_key: &ed25519_dalek::VerifyingKey,
) -> Result<()> {
    let digest = Sha256::digest(content);
    let digest_hex = super::hex_encode(&digest);
    if digest_hex != bundle.message_signature.message_digest.digest {
        return Err(anyhow!("content digest mismatch"));
    }

    let sig_bytes = base64_decode(&bundle.message_signature.signature)?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow!("invalid signature: {e}"))?;

    ed25519_dalek::Verifier::verify(verifying_key, content, &signature)
        .map_err(|e| anyhow!("signature verification failed: {e}"))
}

/// Standard-alphabet base64, written padded. Protobuf JSON (the bundle encoding) accepts
/// padded and unpadded input, so both decode; non-canonical trailing bits and impossible
/// lengths are errors.
const B64: GeneralPurpose = GeneralPurpose::new(
    &alphabet::STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

fn base64_encode(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

fn base64_decode(input: &str) -> Result<Vec<u8>> {
    B64.decode(input)
        .map_err(|e| anyhow!("invalid base64: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decoding_is_strict_about_shape() {
        let bytes: Vec<u8> = (0u8..=255).collect();
        for len in 0..8 {
            let encoded = base64_encode(&bytes[..len]);
            assert_eq!(base64_decode(&encoded).unwrap(), &bytes[..len]);
            assert_eq!(
                base64_decode(encoded.trim_end_matches('=')).unwrap(),
                &bytes[..len]
            );
        }
        assert_eq!(base64_encode(b"hi"), "aGk=");
        // One dangling character, non-zero trailing bits and padding mid-string were accepted
        // by the hand-written decoder.
        for bad in ["aGk=a", "aGl=", "aGj", "a", "aG=k", "aGk*"] {
            assert!(base64_decode(bad).is_err(), "{bad:?} decoded");
        }
    }

    fn test_keypair() -> SigningKey {
        SigningKey::from_bytes(&rand::random())
    }

    #[test]
    fn local_bundle_roundtrip() {
        let key = test_keypair();
        let content = b"knowledge store data to sign";
        let bundle = create_local_bundle(&key, content, Some("user@example.com")).unwrap();

        assert_eq!(
            bundle.media_type,
            "application/vnd.dev.sigstore.bundle.v0.3+json"
        );
        assert!(bundle.tlog_entries.is_none());
        assert_eq!(
            bundle.verification_material.public_key.identity.as_deref(),
            Some("user@example.com")
        );

        verify_bundle(&bundle, content).unwrap();
    }

    #[test]
    fn wrong_content_rejected() {
        let key = test_keypair();
        let bundle = create_local_bundle(&key, b"original", None).unwrap();
        assert!(verify_bundle(&bundle, b"tampered").is_err());
    }

    #[test]
    fn verify_with_explicit_key() {
        let key = test_keypair();
        let content = b"test data";
        let bundle = create_local_bundle(&key, content, None).unwrap();
        verify_bundle_with_key(&bundle, content, &key.verifying_key()).unwrap();

        let wrong_key = test_keypair();
        assert!(verify_bundle_with_key(&bundle, content, &wrong_key.verifying_key()).is_err());
    }
}
