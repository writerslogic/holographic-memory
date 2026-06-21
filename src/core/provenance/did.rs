// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};

const ED25519_MULTICODEC: [u8; 2] = [0xed, 0x01];

/// Generate a did:key URI from an Ed25519 public key (32 bytes).
pub fn did_key_from_ed25519(public_key: &[u8; 32]) -> String {
    let mut prefixed = Vec::with_capacity(2 + 32);
    prefixed.extend_from_slice(&ED25519_MULTICODEC);
    prefixed.extend_from_slice(public_key);
    let encoded = multibase::encode(multibase::Base::Base58Btc, &prefixed);
    format!("did:key:{encoded}")
}

/// Extract the Ed25519 public key bytes from a did:key URI.
pub fn ed25519_from_did_key(did: &str) -> Result<[u8; 32]> {
    let multibase_part = did
        .strip_prefix("did:key:")
        .ok_or_else(|| anyhow!("not a did:key URI"))?;
    let (_, decoded) =
        multibase::decode(multibase_part).map_err(|e| anyhow!("multibase decode failed: {e}"))?;
    if decoded.len() != 34
        || decoded[0] != ED25519_MULTICODEC[0]
        || decoded[1] != ED25519_MULTICODEC[1]
    {
        return Err(anyhow!("not an Ed25519 did:key (wrong prefix or length)"));
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&decoded[2..34]);
    Ok(key)
}

/// Create a did:web URI from a domain and optional path segments.
/// did:web encodes `:` as domain separators and `path` segments after the domain.
/// Example: did:web:example.com or did:web:example.com:users:alice
pub fn did_web_from_domain(domain: &str, path: Option<&[&str]>) -> String {
    let encoded_domain = domain.replace(':', "%3A");
    match path {
        Some(segments) if !segments.is_empty() => {
            let path_str = segments.join(":");
            format!("did:web:{encoded_domain}:{path_str}")
        }
        _ => format!("did:web:{encoded_domain}"),
    }
}

/// Convert a did:web URI to the HTTPS URL where its DID document should be hosted.
/// Per the did:web spec:
///   did:web:example.com → https://example.com/.well-known/did.json
///   did:web:example.com:path:to → https://example.com/path/to/did.json
pub fn did_web_to_url(did: &str) -> Result<String> {
    let domain_path = did
        .strip_prefix("did:web:")
        .ok_or_else(|| anyhow!("not a did:web URI"))?;
    let parts: Vec<&str> = domain_path.split(':').collect();
    let domain = parts[0].replace("%3A", ":");
    if parts.len() == 1 {
        Ok(format!("https://{domain}/.well-known/did.json"))
    } else {
        let path = parts[1..].join("/");
        Ok(format!("https://{domain}/{path}/did.json"))
    }
}

/// Create a DID document for a did:web or did:key identity with an Ed25519 verification key.
pub fn create_did_document(did: &str, public_key: &[u8; 32]) -> serde_json::Value {
    let key_multibase = {
        let mut prefixed = Vec::with_capacity(34);
        prefixed.extend_from_slice(&ED25519_MULTICODEC);
        prefixed.extend_from_slice(public_key);
        multibase::encode(multibase::Base::Base58Btc, &prefixed)
    };
    let key_id = format!("{did}#key-0");
    serde_json::json!({
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/multikey/v1"
        ],
        "id": did,
        "verificationMethod": [{
            "id": &key_id,
            "type": "Multikey",
            "controller": did,
            "publicKeyMultibase": key_multibase
        }],
        "authentication": [&key_id],
        "assertionMethod": [&key_id]
    })
}

/// Extract the Ed25519 public key from a DID (supports both did:key and did:web documents).
/// For did:key, extracts directly from the URI.
/// For did:web, requires the DID document JSON to be passed in.
pub fn ed25519_from_did_document(doc: &serde_json::Value) -> Result<[u8; 32]> {
    let methods = doc
        .get("verificationMethod")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("no verificationMethod in DID document"))?;
    let method = methods
        .first()
        .ok_or_else(|| anyhow!("empty verificationMethod array"))?;
    let multibase_key = method
        .get("publicKeyMultibase")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("no publicKeyMultibase in verification method"))?;
    let (_, decoded) =
        multibase::decode(multibase_key).map_err(|e| anyhow!("multibase decode failed: {e}"))?;
    if decoded.len() != 34
        || decoded[0] != ED25519_MULTICODEC[0]
        || decoded[1] != ED25519_MULTICODEC[1]
    {
        return Err(anyhow!("not an Ed25519 key (wrong multicodec prefix)"));
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&decoded[2..34]);
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn did_key_roundtrip() {
        let pk = [0xABu8; 32];
        let did = did_key_from_ed25519(&pk);
        assert!(did.starts_with("did:key:z"));
        let recovered = ed25519_from_did_key(&did).unwrap();
        assert_eq!(recovered, pk);
    }

    #[test]
    fn reject_bad_prefix() {
        assert!(ed25519_from_did_key("did:web:example.com").is_err());
    }

    #[test]
    fn did_web_simple() {
        let did = did_web_from_domain("example.com", None);
        assert_eq!(did, "did:web:example.com");
        let url = did_web_to_url(&did).unwrap();
        assert_eq!(url, "https://example.com/.well-known/did.json");
    }

    #[test]
    fn did_web_with_path() {
        let did = did_web_from_domain("example.com", Some(&["users", "alice"]));
        assert_eq!(did, "did:web:example.com:users:alice");
        let url = did_web_to_url(&did).unwrap();
        assert_eq!(url, "https://example.com/users/alice/did.json");
    }

    #[test]
    fn did_web_with_port() {
        let did = did_web_from_domain("localhost%3A8080", None);
        assert_eq!(did, "did:web:localhost%3A8080");
        let url = did_web_to_url(&did).unwrap();
        assert_eq!(url, "https://localhost:8080/.well-known/did.json");
    }

    #[test]
    fn did_document_roundtrip() {
        let pk = [0xCDu8; 32];
        let did = did_web_from_domain("example.com", None);
        let doc = create_did_document(&did, &pk);
        let recovered = ed25519_from_did_document(&doc).unwrap();
        assert_eq!(recovered, pk);
        assert_eq!(doc["id"], "did:web:example.com");
    }
}
