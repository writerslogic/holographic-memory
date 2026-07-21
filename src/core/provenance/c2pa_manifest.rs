// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::cawg;
use super::did::did_key_from_ed25519;

/// HMS knowledge store manifest using C2PA assertion labels for semantic interop.
/// Signed via COSE Sign1 and also encoded as JUMBF (ISO 19566-5) in StoreManifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HmsManifest {
    pub claim_generator: String,
    pub claim_generator_info: Vec<ClaimGeneratorInfo>,
    pub title: String,
    pub format: String,
    pub instance_id: String,
    pub assertions: Vec<Assertion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ingredients: Vec<Ingredient>,
    pub signature_info: Option<SignatureInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaimGeneratorInfo {
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Assertion {
    pub label: String,
    pub data: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignatureInfo {
    pub alg: String,
    pub issuer: String,
    pub time: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ingredient {
    pub title: String,
    pub format: String,
    pub instance_id: String,
    pub relationship: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

/// Parameters for creating an HMS store manifest.
pub struct ManifestParams<'a> {
    pub store_id: &'a str,
    pub fact_count: usize,
    pub dimensions: u32,
    pub store_hash: &'a [u8; 32],
    pub title: Option<&'a str>,
    pub ingredients: Vec<Ingredient>,
}

/// Create an HMS manifest with C2PA-style assertions.
pub fn create_manifest(
    signing_key: &SigningKey,
    params: &ManifestParams<'_>,
) -> Result<HmsManifest> {
    let did = did_key_from_ed25519(&signing_key.verifying_key().to_bytes());
    let timestamp = iso8601_now();
    let hash_hex: String = params
        .store_hash
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    let assertions = vec![
        Assertion {
            label: "stds.schema-org.CreativeWork".to_string(),
            data: serde_json::json!({
                "@type": "CreativeWork",
                "author": [{
                    "@type": "Person",
                    "identifier": &did,
                }],
            }),
        },
        Assertion {
            label: "c2pa.actions".to_string(),
            data: serde_json::json!({
                "actions": [{
                    "action": "c2pa.created",
                    "softwareAgent": {
                        "name": "holographic-memory",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia",
                    "description": "Knowledge store created via holographic reduced representations",
                }],
            }),
        },
        Assertion {
            label: "c2pa.hash.data".to_string(),
            data: serde_json::json!({
                "name": "store_contents",
                "alg": "sha256",
                "hash": hash_hex,
            }),
        },
        Assertion {
            label: "hms.store.metadata".to_string(),
            data: serde_json::json!({
                "storeId": params.store_id,
                "factCount": params.fact_count,
                "dimensions": params.dimensions,
                "encodingMethod": "holographic-reduced-representation",
                "vectorType": "entangled-sparse-binary",
            }),
        },
        Assertion {
            label: "c2pa.ai".to_string(),
            data: serde_json::json!({
                "model": {
                    "name": "holographic-memory",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "description": "Content encoded into hyperdimensional vectors via algebraic operations (FFT-based circular convolution). Original content cannot be reconstructed from stored vectors.",
            }),
        },
    ];

    Ok(HmsManifest {
        claim_generator: format!("holographic-memory/{}", env!("CARGO_PKG_VERSION")),
        claim_generator_info: vec![ClaimGeneratorInfo {
            name: "holographic-memory".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            icon: None,
        }],
        title: params.title.unwrap_or("HMS Knowledge Store").to_string(),
        format: "application/x-hms-store".to_string(),
        instance_id: format!("urn:hms:store:{}", params.store_id),
        assertions,
        ingredients: params.ingredients.clone(),
        signature_info: Some(SignatureInfo {
            alg: "EdDSA".to_string(),
            issuer: did,
            time: timestamp,
        }),
    })
}

/// Serialize manifest to crJSON sidecar format and sign it.
pub fn sign_manifest(signing_key: &SigningKey, manifest: &HmsManifest) -> Result<Vec<u8>> {
    let json_bytes =
        serde_json::to_vec(manifest).map_err(|e| anyhow!("manifest serialization failed: {e}"))?;

    super::cose::sign_payload(signing_key, &json_bytes)
}

/// Verify a signed manifest and return it.
pub fn verify_manifest(
    verifying_key: &ed25519_dalek::VerifyingKey,
    signed_bytes: &[u8],
) -> Result<HmsManifest> {
    let payload = super::cose::verify_and_extract(verifying_key, signed_bytes)?;
    serde_json::from_slice(&payload).map_err(|e| anyhow!("manifest deserialization failed: {e}"))
}

/// Validate that a manifest's store hash matches the actual store contents.
pub fn validate_store_hash(manifest: &HmsManifest, store_data: &[u8]) -> Result<bool> {
    let hash_assertion = manifest
        .assertions
        .iter()
        .find(|a| a.label == "c2pa.hash.data")
        .ok_or_else(|| anyhow!("manifest has no hash assertion"))?;

    let expected_hex = hash_assertion
        .data
        .get("hash")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("hash assertion has no hash field"))?;

    let actual_hash = Sha256::digest(store_data);
    let actual_hex: String = actual_hash.iter().map(|b| format!("{b:02x}")).collect();

    Ok(actual_hex == expected_hex)
}

/// Every provenance layer for an AI-agent-produced asset.
pub struct AgentProvenanceParams<'a> {
    pub asset: &'a [u8],
    pub agent_credential: &'a serde_json::Value,
    pub memory_provenance: &'a [u8],
    /// `c2pa.process-proof` — CPoE proof-of-effort (SWF/VDF + Merkle + jitter).
    pub process_proof: Option<&'a serde_json::Value>,
    /// `c2pa.attestation` evidence for the signing environment (e.g. a RATS EAT).
    pub attestation: Option<&'a serde_json::Value>,
    /// `cawg.training-and-data-mining` usage policy.
    pub training_policy: Option<&'a serde_json::Value>,
}

/// Create a complete C2PA manifest for AI-agent-produced content, embedding every
/// provenance layer: the default HMS assertions, a CAWG identity assertion naming the
/// AI agent (did:key / COSE), cogmem memory provenance, an optional `c2pa.process-proof`
/// (CPoE), a `c2pa.soft_binding` fingerprint for durability, an optional
/// `cawg.training-and-data-mining` policy, and an optional `c2pa.attestation` bound to
/// the partial-claim hash. The claim is signed by `claim_key`; the agent identity
/// assertion by `agent_key`.
pub fn create_agent_provenance_manifest(
    claim_key: &SigningKey,
    agent_key: &SigningKey,
    params: &AgentProvenanceParams<'_>,
) -> Result<(HmsManifest, Vec<u8>)> {
    let store_hash: [u8; 32] = Sha256::digest(params.asset).into();
    let mp = ManifestParams {
        store_id: "agent-content",
        fact_count: 0,
        dimensions: 0,
        store_hash: &store_hash,
        title: Some("AI-agent-produced content"),
        ingredients: Vec::new(),
    };
    let mut manifest = create_manifest(claim_key, &mp)?;

    let hard = manifest
        .assertions
        .iter()
        .find(|a| a.label == "c2pa.hash.data")
        .ok_or_else(|| anyhow!("manifest missing hard binding"))?;
    let hard_hash: [u8; 32] = Sha256::digest(serde_json::to_vec(&hard.data)?).into();
    let referenced = vec![(
        "self#jumbf=c2pa.assertions/c2pa.hash.data".to_string(),
        "sha256".to_string(),
        hard_hash.to_vec(),
    )];

    let agent_assertion =
        cawg::create_identity_assertion_ica(agent_key, &referenced, "cogmem agent")?;
    manifest.assertions.push(Assertion {
        label: "cawg.identity".to_string(),
        data: agent_assertion,
    });

    let mem_hex: String = params
        .memory_provenance
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    manifest.assertions.push(Assertion {
        label: "cogmem.memory.provenance".to_string(),
        data: serde_json::json!({ "alg": "EdDSA", "coseHex": mem_hex }),
    });

    if let Some(pp) = params.process_proof {
        manifest.assertions.push(Assertion {
            label: "c2pa.process-proof".to_string(),
            data: pp.clone(),
        });
    }

    // PLACEHOLDER soft binding. This is a SHA-256 digest, NOT a perceptual fingerprint:
    // it will not survive any edit, so it provides no durability yet. A real durable
    // soft binding requires an approved perceptual/watermark algorithm (unimplemented).
    let digest: String = Sha256::digest(params.asset)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    manifest.assertions.push(Assertion {
        label: "c2pa.soft_binding".to_string(),
        data: serde_json::json!({
            "alg": "sha256-placeholder",
            "durable": false,
            "blocks": [{ "value": digest }]
        }),
    });

    if let Some(tp) = params.training_policy {
        manifest.assertions.push(Assertion {
            label: "cawg.training-and-data-mining".to_string(),
            data: tp.clone(),
        });
    }

    // Attestation binds to the hash of the partial claim (every assertion signed so far).
    if let Some(att) = params.attestation {
        let partial_hash: String = Sha256::digest(serde_json::to_vec(&manifest.assertions)?)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        manifest.assertions.push(Assertion {
            label: "c2pa.attestation".to_string(),
            data: serde_json::json!({
                "attType": "c2pa.RATS",
                "alg": "sha256",
                "partialClaimHash": partial_hash,
                "evidence": att.clone(),
            }),
        });
    }

    let signed = sign_manifest(claim_key, &manifest)?;
    Ok((manifest, signed))
}

/// Verify the claim signature AND the embedded CAWG agent identity assertion.
pub fn verify_agent_provenance_manifest(
    claim_vk: &ed25519_dalek::VerifyingKey,
    signed_bytes: &[u8],
) -> Result<HmsManifest> {
    let manifest = verify_manifest(claim_vk, signed_bytes)?;
    let agent = manifest
        .assertions
        .iter()
        .find(|a| a.label == "cawg.identity")
        .ok_or_else(|| anyhow!("manifest has no cawg.identity assertion"))?;
    let embedded = cawg::ica_embedded_bytes(&agent.data)?;
    cawg::verify_cawg_ica(&embedded)?;

    // If an attestation is present, confirm it binds to the partial claim.
    if let Some(att) = manifest
        .assertions
        .iter()
        .find(|a| a.label == "c2pa.attestation")
    {
        let claimed = att
            .data
            .get("partialClaimHash")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("attestation missing partialClaimHash"))?;
        let partial: Vec<&Assertion> = manifest
            .assertions
            .iter()
            .filter(|a| a.label != "c2pa.attestation")
            .collect();
        let recomputed: String = Sha256::digest(serde_json::to_vec(&partial)?)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if recomputed != claimed {
            return Err(anyhow!(
                "attestation does not bind to the manifest's partial claim"
            ));
        }
    }
    Ok(manifest)
}

fn iso8601_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let days = secs / 86400;
    let rem = secs % 86400;
    let (year, month, day) = days_to_ymd(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn days_to_ymd(days_since_epoch: u64) -> (u64, u64, u64) {
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::generate(&mut rand::thread_rng())
    }

    #[test]
    fn create_and_sign_manifest() {
        let key = test_keypair();
        let store_hash = Sha256::digest(b"test store data");
        let store_hash_arr: [u8; 32] = store_hash.into();

        let params = ManifestParams {
            store_id: "store-001",
            fact_count: 42,
            dimensions: 16384,
            store_hash: &store_hash_arr,
            title: Some("Test Knowledge Store"),
            ingredients: Vec::new(),
        };

        let manifest = create_manifest(&key, &params).unwrap();
        assert_eq!(manifest.assertions.len(), 5);
        assert!(manifest.signature_info.is_some());

        let signed = sign_manifest(&key, &manifest).unwrap();
        let recovered = verify_manifest(&key.verifying_key(), &signed).unwrap();
        assert_eq!(recovered.title, "Test Knowledge Store");
    }

    #[test]
    fn agent_provenance_manifest_roundtrip() {
        let claim_key = test_keypair();
        let agent_key = test_keypair();
        let vc = serde_json::json!({
            "type": ["VerifiableCredential", "AgentIdentityCredential"],
            "credentialSubject": { "actorType": "ai-agent" }
        });
        let process_proof = serde_json::json!({
            "merkleRoot": "abcd", "input": "seed", "durationMs": 1500, "iterations": 100000
        });
        let attestation = serde_json::json!({ "eat": "<EAT-token>", "platform": "cogmem-runtime" });
        let training =
            serde_json::json!({ "entries": [{ "use": "aiTraining", "constraint": "notAllowed" }] });
        let params = AgentProvenanceParams {
            asset: b"agent-made content",
            agent_credential: &vc,
            memory_provenance: b"fake-cose-memory-statement",
            process_proof: Some(&process_proof),
            attestation: Some(&attestation),
            training_policy: Some(&training),
        };
        let (manifest, signed) =
            create_agent_provenance_manifest(&claim_key, &agent_key, &params).unwrap();
        for label in [
            "cawg.identity",
            "cogmem.memory.provenance",
            "c2pa.process-proof",
            "c2pa.soft_binding",
            "cawg.training-and-data-mining",
            "c2pa.attestation",
        ] {
            assert!(
                manifest.assertions.iter().any(|a| a.label == label),
                "missing assertion: {label}"
            );
        }
        let m = verify_agent_provenance_manifest(&claim_key.verifying_key(), &signed).unwrap();
        assert_eq!(m.title, "AI-agent-produced content");

        // Tampering with the attestation's partial-claim binding must be rejected.
        let mut tampered = manifest.clone();
        for a in tampered.assertions.iter_mut() {
            if a.label == "c2pa.attestation" {
                a.data["partialClaimHash"] = serde_json::json!("deadbeef");
            }
        }
        let bad = sign_manifest(&claim_key, &tampered).unwrap();
        assert!(verify_agent_provenance_manifest(&claim_key.verifying_key(), &bad).is_err());
    }

    #[test]
    fn validate_hash_match() {
        let key = test_keypair();
        let store_data = b"knowledge store contents";
        let store_hash: [u8; 32] = Sha256::digest(store_data).into();

        let params = ManifestParams {
            store_id: "s1",
            fact_count: 1,
            dimensions: 10000,
            store_hash: &store_hash,
            title: None,
            ingredients: Vec::new(),
        };

        let manifest = create_manifest(&key, &params).unwrap();
        assert!(validate_store_hash(&manifest, store_data).unwrap());
        assert!(!validate_store_hash(&manifest, b"wrong data").unwrap());
    }

    #[test]
    fn manifest_has_required_assertions() {
        let key = test_keypair();
        let hash = [0u8; 32];
        let params = ManifestParams {
            store_id: "s1",
            fact_count: 0,
            dimensions: 16384,
            store_hash: &hash,
            title: None,
            ingredients: Vec::new(),
        };

        let manifest = create_manifest(&key, &params).unwrap();
        let labels: Vec<&str> = manifest
            .assertions
            .iter()
            .map(|a| a.label.as_str())
            .collect();
        assert!(labels.contains(&"stds.schema-org.CreativeWork"));
        assert!(labels.contains(&"c2pa.actions"));
        assert!(labels.contains(&"c2pa.hash.data"));
        assert!(labels.contains(&"c2pa.ai"));
        assert!(labels.contains(&"hms.store.metadata"));
    }

    #[test]
    fn wrong_key_rejects_manifest() {
        let key = test_keypair();
        let hash = [0u8; 32];
        let params = ManifestParams {
            store_id: "s1",
            fact_count: 0,
            dimensions: 10000,
            store_hash: &hash,
            title: None,
            ingredients: Vec::new(),
        };

        let manifest = create_manifest(&key, &params).unwrap();
        let signed = sign_manifest(&key, &manifest).unwrap();

        let wrong_key = test_keypair();
        assert!(verify_manifest(&wrong_key.verifying_key(), &signed).is_err());
    }

    #[test]
    fn manifest_with_ingredients() {
        let key = test_keypair();
        let hash = [0u8; 32];
        let params = ManifestParams {
            store_id: "s2",
            fact_count: 10,
            dimensions: 16384,
            store_hash: &hash,
            title: None,
            ingredients: vec![Ingredient {
                title: "parent-store".to_string(),
                format: "application/x-hms-store".to_string(),
                instance_id: "urn:hms:store:s1".to_string(),
                relationship: "parentOf".to_string(),
                hash: Some("abc123".to_string()),
            }],
        };

        let manifest = create_manifest(&key, &params).unwrap();
        assert_eq!(manifest.ingredients.len(), 1);
        assert_eq!(manifest.ingredients[0].relationship, "parentOf");

        let signed = sign_manifest(&key, &manifest).unwrap();
        let recovered = verify_manifest(&key.verifying_key(), &signed).unwrap();
        assert_eq!(recovered.ingredients.len(), 1);
    }
}
