// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
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
    let digest_hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();

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
    let digest_hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
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
    let digest_hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    if digest_hex != bundle.message_signature.message_digest.digest {
        return Err(anyhow!("content digest mismatch"));
    }

    let sig_bytes = base64_decode(&bundle.message_signature.signature)?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow!("invalid signature: {e}"))?;

    ed25519_dalek::Verifier::verify(verifying_key, content, &signature)
        .map_err(|e| anyhow!("signature verification failed: {e}"))
}

fn base64_encode(bytes: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode(input: &str) -> Result<Vec<u8>> {
    let input = input.trim_end_matches('=');
    let mut result = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for c in input.chars() {
        let val = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => return Err(anyhow!("invalid base64 character: {c}")),
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::generate(&mut rand::thread_rng())
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

    #[test]
    fn base64_roundtrip() {
        let original = b"hello world 12345";
        let encoded = base64_encode(original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn bundle_serialization() {
        let key = test_keypair();
        let bundle = create_local_bundle(&key, b"data", None).unwrap();
        let json = serde_json::to_string(&bundle).unwrap();
        let recovered: SigstoreBundle = serde_json::from_str(&json).unwrap();
        verify_bundle(&recovered, b"data").unwrap();
    }
}
