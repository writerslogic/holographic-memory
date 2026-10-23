// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use base64::engine::{general_purpose::GeneralPurpose, DecodePaddingMode, GeneralPurposeConfig};
use base64::{alphabet, Engine};

const ED25519_MULTICODEC: [u8; 2] = [0xed, 0x01];

/// base64url engine matching c2pa-rs: URL-safe alphabet, padding ignored on decode.
const B64URL: GeneralPurpose = GeneralPurpose::new(
    &alphabet::URL_SAFE,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

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
    ed25519_multicodec_key(&decoded)
        .ok_or_else(|| anyhow!("not an Ed25519 did:key (wrong prefix or length)"))
}

/// The key inside an Ed25519 multicodec value: the 2-byte prefix followed by exactly 32 bytes.
fn ed25519_multicodec_key(decoded: &[u8]) -> Option<[u8; 32]> {
    decoded
        .strip_prefix(ED25519_MULTICODEC.as_slice())?
        .try_into()
        .ok()
}

/// Build a did:jwk URI from an Ed25519 public key (32 bytes), the inverse of
/// [`ed25519_from_did_jwk`]. The JWK JSON is compact with SORTED keys
/// (`{"crv":"Ed25519","kty":"OKP","x":"<base64url-nopad pubkey>"}`); the method id is
/// that JSON base64url-encoded WITH canonical '=' padding, matching cogmem's
/// `agent_did_jwk` so the two emit byte-identical issuer DIDs.
pub fn did_jwk_from_ed25519(public_key: &[u8; 32]) -> String {
    // base64url, no padding, for the JWK `x` parameter.
    let b64url_nopad = GeneralPurpose::new(
        &alphabet::URL_SAFE,
        GeneralPurposeConfig::new().with_encode_padding(false),
    );
    let x = b64url_nopad.encode(public_key);
    // Compact JWK with sorted keys: crv, kty, x.
    let jwk_json = format!("{{\"crv\":\"Ed25519\",\"kty\":\"OKP\",\"x\":\"{x}\"}}");
    // Canonical '=' padding on the method-specific id (c2pa-rs / cogmem requirement).
    let b64url_pad = GeneralPurpose::new(&alphabet::URL_SAFE, GeneralPurposeConfig::new());
    let enc = b64url_pad.encode(jwk_json.as_bytes());
    format!("did:jwk:{enc}")
}

/// Extract the Ed25519 public key bytes from a did:jwk URI. The method id is a
/// base64url-encoded JWK; decoding tolerates the canonical padding c2pa-rs emits.
pub fn ed25519_from_did_jwk(did: &str) -> Result<[u8; 32]> {
    let encoded = did
        .strip_prefix("did:jwk:")
        .ok_or_else(|| anyhow!("not a did:jwk URI"))?;
    let json = B64URL
        .decode(encoded)
        .map_err(|e| anyhow!("did:jwk base64url decode failed: {e}"))?;
    let jwk: serde_json::Value =
        serde_json::from_slice(&json).map_err(|e| anyhow!("did:jwk JWK parse failed: {e}"))?;
    if jwk.get("kty").and_then(|v| v.as_str()) != Some("OKP") {
        return Err(anyhow!("did:jwk is not an OKP key"));
    }
    if jwk.get("crv").and_then(|v| v.as_str()) != Some("Ed25519") {
        return Err(anyhow!("did:jwk is not an Ed25519 key"));
    }
    let x = jwk
        .get("x")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("did:jwk JWK has no x parameter"))?;
    let key_bytes = B64URL
        .decode(x)
        .map_err(|e| anyhow!("did:jwk x base64url decode failed: {e}"))?;
    let key: [u8; 32] = key_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("did:jwk x is not a 32-byte Ed25519 key"))?;
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
///   did:web:example.com → <https://example.com/.well-known/did.json>
///   did:web:example.com:path:to → <https://example.com/path/to/did.json>
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
    ed25519_multicodec_key(&decoded)
        .ok_or_else(|| anyhow!("not an Ed25519 key (wrong multicodec prefix)"))
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
    fn did_key_accepts_only_a_32_byte_ed25519_multicodec_key() {
        for (prefix, len, valid) in [
            (ED25519_MULTICODEC, 31, false),
            (ED25519_MULTICODEC, 32, true),
            (ED25519_MULTICODEC, 33, false),
            ([0xec, 0x01], 32, false),
        ] {
            let mut bytes = prefix.to_vec();
            bytes.extend(std::iter::repeat_n(0xAB, len));
            let did = format!(
                "did:key:{}",
                multibase::encode(multibase::Base::Base58Btc, &bytes)
            );
            assert_eq!(
                ed25519_from_did_key(&did).is_ok(),
                valid,
                "prefix {prefix:?}, key length {len}"
            );
        }
    }

    #[test]
    fn did_jwk_from_ed25519_roundtrips() {
        let pk = [0x42u8; 32];
        let did = did_jwk_from_ed25519(&pk);
        assert!(did.starts_with("did:jwk:"));
        let recovered = ed25519_from_did_jwk(&did).unwrap();
        assert_eq!(recovered, pk);
    }

    #[test]
    fn did_jwk_resolves_vector_issuer() {
        let issuer = "did:jwk:eyJjcnYiOiJFZDI1NTE5Iiwia3R5IjoiT0tQIiwieCI6IkE2RUh2X1BPRUw0ZGNOMFk1MHZBbVdmazFqQ2JwUTFmSGR5R1pCSlZNYmcifQ==";
        let expected = "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8";
        let key = ed25519_from_did_jwk(issuer).unwrap();
        let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, expected);
    }

    #[test]
    fn did_web_maps_domain_path_and_port_to_its_document_url() {
        let users_alice: &[&str] = &["users", "alice"];
        for (domain, path, did, url) in [
            (
                "example.com",
                None,
                "did:web:example.com",
                "https://example.com/.well-known/did.json",
            ),
            (
                "example.com",
                Some(users_alice),
                "did:web:example.com:users:alice",
                "https://example.com/users/alice/did.json",
            ),
            (
                "localhost%3A8080",
                None,
                "did:web:localhost%3A8080",
                "https://localhost:8080/.well-known/did.json",
            ),
        ] {
            assert_eq!(did_web_from_domain(domain, path), did);
            assert_eq!(did_web_to_url(did).unwrap(), url, "{did}");
        }
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
