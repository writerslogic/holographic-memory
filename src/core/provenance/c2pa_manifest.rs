// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

use super::cawg;
use super::did::did_key_from_ed25519;
use super::hex_encode;
use super::vc::iso8601_now;

/// WARM knowledge store manifest using C2PA assertion labels for semantic interop.
/// Signed via COSE Sign1 and also encoded as JUMBF (ISO 19566-5) in StoreManifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WarmManifest {
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

/// Parameters for creating an WARM store manifest.
pub struct ManifestParams<'a> {
    pub store_id: &'a str,
    pub fact_count: usize,
    pub dimensions: u32,
    pub store_hash: &'a [u8; 32],
    pub title: Option<&'a str>,
    pub ingredients: Vec<Ingredient>,
}

/// Create an WARM manifest with C2PA-style assertions.
pub fn create_manifest(
    signing_key: &SigningKey,
    params: &ManifestParams<'_>,
) -> Result<WarmManifest> {
    let did = did_key_from_ed25519(&signing_key.verifying_key().to_bytes());
    let timestamp = iso8601_now();
    let hash_hex = hex_encode(params.store_hash);

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
                        "name": "writerslogic-warm",
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
                    "name": "writerslogic-warm",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "description": "Content encoded into hyperdimensional vectors via algebraic operations (FFT-based circular convolution). Original content cannot be reconstructed from stored vectors.",
            }),
        },
    ];

    Ok(WarmManifest {
        claim_generator: format!("writerslogic-warm/{}", env!("CARGO_PKG_VERSION")),
        claim_generator_info: vec![ClaimGeneratorInfo {
            name: "writerslogic-warm".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            icon: None,
        }],
        title: params.title.unwrap_or("WARM Knowledge Store").to_string(),
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
pub fn sign_manifest(signing_key: &SigningKey, manifest: &WarmManifest) -> Result<Vec<u8>> {
    let json_bytes =
        serde_json::to_vec(manifest).map_err(|e| anyhow!("manifest serialization failed: {e}"))?;

    super::cose::sign_payload(signing_key, &json_bytes)
}

/// Verify a signed manifest and return it.
pub fn verify_manifest(
    verifying_key: &ed25519_dalek::VerifyingKey,
    signed_bytes: &[u8],
) -> Result<WarmManifest> {
    let payload = super::cose::verify_and_extract(verifying_key, signed_bytes)?;
    serde_json::from_slice(&payload).map_err(|e| anyhow!("manifest deserialization failed: {e}"))
}

/// Validate that a manifest's store hash matches the actual store contents.
pub fn validate_store_hash(manifest: &WarmManifest, store_data: &[u8]) -> Result<bool> {
    let expected_hex = signed_store_hash(manifest)?;
    let actual_hash = Sha256::digest(store_data);
    let actual_hex = hex_encode(&actual_hash);

    Ok(actual_hex == expected_hex)
}

/// The store hash, in hex, that the manifest's `c2pa.hash.data` assertion carries.
pub(crate) fn signed_store_hash(manifest: &WarmManifest) -> Result<&str> {
    let hash_assertion = manifest
        .assertions
        .iter()
        .find(|a| a.label == "c2pa.hash.data")
        .ok_or_else(|| anyhow!("manifest has no hash assertion"))?;
    hash_assertion
        .data
        .get("hash")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("hash assertion has no hash field"))
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
    /// `cawg.training-mining` usage policy.
    pub training_policy: Option<&'a serde_json::Value>,
}

fn validate_agent_credential(
    claim_key: &SigningKey,
    agent_key: &SigningKey,
    credential: &serde_json::Value,
) -> Result<()> {
    let mut pending = vec![(credential, 0_u8)];
    let mut visited = 0_usize;
    while let Some((value, depth)) = pending.pop() {
        visited += 1;
        anyhow::ensure!(
            depth <= 32 && visited <= 4096,
            "agent credential nesting or node limit exceeded"
        );
        match value {
            serde_json::Value::Array(values) => {
                anyhow::ensure!(
                    values.len() <= 128,
                    "agent credential collection limit exceeded"
                );
                pending.extend(values.iter().map(|value| (value, depth + 1)));
            }
            serde_json::Value::Object(values) => {
                anyhow::ensure!(
                    values.len() <= 128,
                    "agent credential collection limit exceeded"
                );
                pending.extend(values.values().map(|value| (value, depth + 1)));
            }
            _ => {}
        }
    }
    struct Bounded(Vec<u8>);
    impl std::io::Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > 65_536 - self.0.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "agent credential exceeds 65536 bytes",
                ));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut serialized = Bounded(Vec::new());
    serde_json::to_writer(&mut serialized, credential)?;
    let issuer = credential["issuer"]
        .as_str()
        .or_else(|| credential["issuer"]["id"].as_str());
    let expected_issuer = did_key_from_ed25519(&claim_key.verifying_key().to_bytes());
    anyhow::ensure!(
        issuer == Some(expected_issuer.as_str()),
        "agent credential issuer is not the authenticated manifest issuer"
    );
    let expected_agent = did_key_from_ed25519(&agent_key.verifying_key().to_bytes());
    let expected_jwk = super::did::did_jwk_from_ed25519(&agent_key.verifying_key().to_bytes());
    let subject = credential["credentialSubject"]["id"].as_str();
    anyhow::ensure!(
        subject == Some(expected_agent.as_str()) || subject == Some(expected_jwk.as_str()),
        "agent credential subject does not control the identity signing key"
    );
    anyhow::ensure!(
        credential["@context"]
            .as_array()
            .is_some_and(|values| values.first().and_then(serde_json::Value::as_str)
                == Some("https://www.w3.org/ns/credentials/v2")),
        "agent credential must use the VC 2.0 context"
    );
    anyhow::ensure!(
        credential["type"]
            .as_array()
            .is_some_and(|values| values.len() <= 8
                && values
                    .iter()
                    .any(|v| v.as_str() == Some("VerifiableCredential"))
                && values
                    .iter()
                    .any(|v| v.as_str() == Some("AgentIdentityCredential"))),
        "agent credential type is invalid"
    );
    let valid_from = credential["validFrom"]
        .as_str()
        .ok_or_else(|| anyhow!("agent credential is missing validFrom"))?;
    let valid_from = super::vc::parse_iso8601_utc(valid_from)?;
    let now = super::vc::parse_iso8601_utc(&iso8601_now())?;
    anyhow::ensure!(valid_from <= now, "agent credential is not yet valid");
    if let Some(until) = credential.get("validUntil") {
        let until = until
            .as_str()
            .ok_or_else(|| anyhow!("agent credential validUntil must be text"))?;
        let until = super::vc::parse_iso8601_utc(until)?;
        anyhow::ensure!(
            until > valid_from && until > now,
            "agent credential validity interval is invalid or expired"
        );
    }
    anyhow::ensure!(
        credential.get("credentialStatus").is_none(),
        "agent credential status requires a trusted caller verifier"
    );
    super::vc::verify_credential_json(std::str::from_utf8(&serialized.0)?)
}

/// Create an WARM agent-provenance sidecar after authenticating a current agent
/// credential issued by the claim key. This admission check does not establish
/// external issuer authority or full CAWG conformance.
pub fn create_agent_provenance_manifest(
    claim_key: &SigningKey,
    agent_key: &SigningKey,
    params: &AgentProvenanceParams<'_>,
) -> Result<(WarmManifest, Vec<u8>)> {
    validate_agent_credential(claim_key, agent_key, params.agent_credential)?;
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

    let mem_hex = hex_encode(params.memory_provenance);
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
    let digest = hex_encode(&Sha256::digest(params.asset));
    manifest.assertions.push(Assertion {
        label: "c2pa.soft_binding".to_string(),
        data: serde_json::json!({
            "alg": "sha256-placeholder",
            "durable": false,
            "blocks": [{ "value": digest }]
        }),
    });

    if let Some(tp) = params.training_policy {
        validate_training_policy(tp)?;
        manifest.assertions.push(Assertion {
            label: "cawg.training-mining".to_string(),
            data: tp.clone(),
        });
    }

    // Attestation binds to the hash of the partial claim (every assertion signed so far).
    if let Some(att) = params.attestation {
        let partial_hash = hex_encode(&Sha256::digest(serde_json::to_vec(&manifest.assertions)?));
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

fn validate_training_policy(policy: &serde_json::Value) -> Result<()> {
    let root = policy
        .as_object()
        .ok_or_else(|| anyhow!("CAWG training policy must be an object"))?;
    anyhow::ensure!(
        root.len() == 1 && root.contains_key("entries"),
        "CAWG training policy must contain only entries"
    );
    let entries = root["entries"]
        .as_object()
        .ok_or_else(|| anyhow!("CAWG training entries must be an object"))?;
    anyhow::ensure!(
        (1..=4).contains(&entries.len()),
        "CAWG training entries count is out of range"
    );
    for (name, entry) in entries {
        anyhow::ensure!(
            matches!(
                name.as_str(),
                "cawg.data_mining"
                    | "cawg.ai_inference"
                    | "cawg.ai_training"
                    | "cawg.ai_generative_training"
            ),
            "unsupported CAWG training entry"
        );
        let entry = entry
            .as_object()
            .ok_or_else(|| anyhow!("CAWG training entry must be an object"))?;
        anyhow::ensure!(
            entry.len() <= 2
                && entry.contains_key("use")
                && entry.keys().all(|k| k == "use" || k == "constraint_info"),
            "invalid CAWG training entry fields"
        );
        anyhow::ensure!(
            matches!(
                entry["use"].as_str(),
                Some("allowed" | "notAllowed" | "constrained")
            ),
            "invalid CAWG training use"
        );
        if let Some(info) = entry.get("constraint_info") {
            let info = info
                .as_str()
                .ok_or_else(|| anyhow!("CAWG constraint_info must be text"))?;
            anyhow::ensure!(
                (1..=2048).contains(&info.len()),
                "CAWG constraint_info length is out of range"
            );
        }
    }
    Ok(())
}

/// Verify the signed WARM sidecar and bind every ICA reference to an assertion in it.
/// Asset content and issuer trust require separate checks by the caller.
pub fn verify_agent_provenance_manifest(
    claim_vk: &ed25519_dalek::VerifyingKey,
    signed_bytes: &[u8],
) -> Result<WarmManifest> {
    let manifest = verify_manifest(claim_vk, signed_bytes)?;
    let mut assertions = HashMap::new();
    for assertion in &manifest.assertions {
        anyhow::ensure!(
            assertions
                .insert(assertion.label.as_str(), assertion)
                .is_none(),
            "manifest repeats assertion label {}",
            assertion.label
        );
    }
    let hard = assertions
        .get("c2pa.hash.data")
        .ok_or_else(|| anyhow!("manifest missing hard binding"))?;
    anyhow::ensure!(
        hard.data.get("alg").and_then(serde_json::Value::as_str) == Some("sha256"),
        "manifest hard binding uses an unsupported hash algorithm"
    );
    let asset_hash = signed_store_hash(&manifest)?;
    anyhow::ensure!(
        asset_hash.len() == 64 && asset_hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "manifest hard binding has an invalid SHA-256 digest"
    );
    let identities: Vec<&str> = assertions
        .keys()
        .copied()
        .filter(|label| *label == "cawg.identity" || label.starts_with("cawg.identity__"))
        .collect();
    anyhow::ensure!(
        !identities.is_empty(),
        "manifest has no cawg.identity assertion"
    );
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::new();
    for label in &identities {
        let embedded = cawg::ica_embedded_bytes(&assertions[*label].data)?;
        let (vc, referenced) = cawg::verify_cawg_ica_references(&embedded)?;
        let asset = &vc["credentialSubject"]["c2paAsset"];
        for field in [
            "expected_partial_claim",
            "expected_claim_generator",
            "expected_countersigners",
        ] {
            anyhow::ensure!(
                asset.get(field).is_none(),
                "CAWG {field} needs C2PA claim context unavailable in the WARM sidecar"
            );
        }
        anyhow::ensure!(
            vc.get("credentialStatus").is_none(),
            "CAWG credential status needs a trusted resolver"
        );
        let mut has_hard_binding = false;
        for reference in referenced {
            anyhow::ensure!(
                reference.alg.as_deref() == Some("sha256") && reference.hash.len() == 32,
                "unsupported CAWG assertion hash algorithm or length"
            );
            let assertion = assertions
                .get(reference.label())
                .ok_or_else(|| anyhow!("CAWG reference is absent from the signed manifest"))?;
            let expected_url = format!("self#jumbf=c2pa.assertions/{}", assertion.label);
            anyhow::ensure!(
                reference.url == expected_url,
                "CAWG reference URL does not match this manifest"
            );
            let expected_hash: [u8; 32] =
                Sha256::digest(serde_json::to_vec(&assertion.data)?).into();
            anyhow::ensure!(
                reference.hash == expected_hash,
                "CAWG reference hash differs from the signed assertion"
            );
            if assertion.label == "c2pa.hash.data" {
                has_hard_binding = true;
            }
            if assertion.label == "cawg.identity" || assertion.label.starts_with("cawg.identity__")
            {
                edges.entry(*label).or_default().push(&assertion.label);
            }
        }
        anyhow::ensure!(
            has_hard_binding,
            "CAWG identity assertion does not reference this manifest's hard binding"
        );
    }
    for start in &identities {
        let mut pending = vec![*start];
        let mut visited = HashSet::new();
        while let Some(label) = pending.pop() {
            if !visited.insert(label) {
                continue;
            }
            if let Some(neighbors) = edges.get(label) {
                for neighbor in neighbors {
                    anyhow::ensure!(neighbor != start, "CAWG identity references form a cycle");
                    pending.push(neighbor);
                }
            }
        }
    }

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
        let recomputed = hex_encode(&Sha256::digest(serde_json::to_vec(&partial)?));
        if recomputed != claimed {
            return Err(anyhow!(
                "attestation does not bind to the manifest's partial claim"
            ));
        }
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::from_bytes(&rand::random())
    }

    fn agent_credential(issuer: &SigningKey, agent: &SigningKey) -> serde_json::Value {
        let mut credential = serde_json::json!({
            "@context": ["https://www.w3.org/ns/credentials/v2"],
            "type": ["VerifiableCredential", "AgentIdentityCredential"],
            "issuer": did_key_from_ed25519(&issuer.verifying_key().to_bytes()),
            "validFrom": iso8601_now(),
            "credentialSubject": {"id": did_key_from_ed25519(&agent.verifying_key().to_bytes()), "actorType": "ai-agent"}
        });
        credential["proof"] =
            serde_json::to_value(super::super::vc::sign_document(issuer, &credential).unwrap())
                .unwrap();
        credential
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
        let vc = agent_credential(&claim_key, &agent_key);
        let process_proof = serde_json::json!({
            "merkleRoot": "abcd", "input": "seed", "durationMs": 1500, "iterations": 100000
        });
        let attestation = serde_json::json!({ "eat": "<EAT-token>", "platform": "cogmem-runtime" });
        let training = serde_json::json!({
            "entries": {"cawg.ai_training": {"use": "notAllowed"}}
        });
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
            "cawg.training-mining",
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
    fn agent_credential_is_authenticated_before_manifest_generation() {
        let (issuer, agent) = (test_keypair(), test_keypair());
        let valid = agent_credential(&issuer, &agent);
        validate_agent_credential(&issuer, &agent, &valid).unwrap();
        let mut unsigned = valid.clone();
        unsigned.as_object_mut().unwrap().remove("proof");
        let mut tampered = valid.clone();
        tampered["credentialSubject"]["actorType"] = serde_json::json!("tampered");
        let mut future = valid.clone();
        future["validFrom"] = serde_json::json!("2999-01-01T00:00:00Z");
        let mut expired = valid.clone();
        expired["validUntil"] = serde_json::json!("2000-01-01T00:00:00Z");
        let mut status = valid.clone();
        status["credentialStatus"] = serde_json::json!({"id": "https://example.com/status"});
        let wrong_agent = agent_credential(&issuer, &test_keypair());
        let wrong_issuer = agent_credential(&test_keypair(), &agent);
        for rejected in [
            &unsigned,
            &tampered,
            &future,
            &expired,
            &status,
            &wrong_agent,
            &wrong_issuer,
        ] {
            let params = AgentProvenanceParams {
                asset: b"asset",
                agent_credential: rejected,
                memory_provenance: b"memory",
                process_proof: None,
                attestation: None,
                training_policy: None,
            };
            assert!(create_agent_provenance_manifest(&issuer, &agent, &params).is_err());
        }
    }

    #[test]
    fn agent_credential_structure_and_byte_limits_are_enforced() {
        let (issuer, agent) = (test_keypair(), test_keypair());
        let valid = agent_credential(&issuer, &agent);
        for depth in [31, 32, 33] {
            let mut credential = valid.clone();
            let mut nested = serde_json::json!(null);
            for _ in 1..depth {
                nested = serde_json::json!({"nested": nested});
            }
            credential["extra"] = nested;
            credential["proof"] = serde_json::to_value(
                super::super::vc::sign_document(&issuer, &credential).unwrap(),
            )
            .unwrap();
            assert_eq!(
                validate_agent_credential(&issuer, &agent, &credential).is_ok(),
                depth <= 32
            );
        }
        for limit in [65_535, 65_536, 65_537] {
            let mut credential = valid.clone();
            credential["extra"] = serde_json::json!("");
            for attempt in 0..128 {
                let size = serde_json::to_vec(&credential).unwrap().len();
                if size == limit {
                    break;
                }
                credential["nonce"] = serde_json::json!(format!("{attempt:03}"));
                let size = serde_json::to_vec(&credential).unwrap().len();
                let current = credential["extra"].as_str().unwrap().len();
                let length = if size < limit {
                    current + limit - size
                } else {
                    current - (size - limit)
                };
                credential["extra"] = serde_json::json!("x".repeat(length));
                credential["proof"] = serde_json::to_value(
                    super::super::vc::sign_document(&issuer, &credential).unwrap(),
                )
                .unwrap();
            }
            assert_eq!(serde_json::to_vec(&credential).unwrap().len(), limit);
            assert_eq!(
                validate_agent_credential(&issuer, &agent, &credential).is_ok(),
                limit <= 65_536
            );
        }
    }

    #[test]
    fn training_policy_uses_ratified_cawg_shape_and_bounds() {
        for len in [2047, 2048] {
            let policy = serde_json::json!({
                "entries": {"cawg.data_mining": {
                    "use": "constrained",
                    "constraint_info": "x".repeat(len)
                }}
            });
            validate_training_policy(&policy).unwrap();
        }
        let too_long = serde_json::json!({
            "entries": {"cawg.data_mining": {
                "use": "constrained", "constraint_info": "x".repeat(2049)
            }}
        });
        assert!(validate_training_policy(&too_long).is_err());
        for malformed in [
            serde_json::json!({"entries": []}),
            serde_json::json!({"entries": {}}),
            serde_json::json!({"entries": {"cawg.ai_training": {"use": "maybe"}}}),
            serde_json::json!({"entries": {"cawg.unknown": {"use": "allowed"}}}),
        ] {
            assert!(validate_training_policy(&malformed).is_err());
        }
    }

    /// An agent's identity assertion signs one asset's hard binding; carried into a manifest
    /// for another asset it no longer verifies.
    #[test]
    fn agent_identity_assertion_does_not_transplant_to_another_asset() {
        let (agent_key, claim_key) = (test_keypair(), test_keypair());
        let vc = agent_credential(&claim_key, &agent_key);
        let params = |asset| AgentProvenanceParams {
            asset,
            agent_credential: &vc,
            memory_provenance: b"memory",
            process_proof: None,
            attestation: None,
            training_policy: None,
        };
        let (genuine, _) =
            create_agent_provenance_manifest(&claim_key, &agent_key, &params(b"genuine")).unwrap();
        let other_key = test_keypair();
        let other_credential = agent_credential(&claim_key, &other_key);
        let other_params = AgentProvenanceParams {
            agent_credential: &other_credential,
            ..params(b"other")
        };
        let (mut other, _) =
            create_agent_provenance_manifest(&claim_key, &other_key, &other_params).unwrap();
        let identity = |m: &WarmManifest| {
            m.assertions
                .iter()
                .position(|a| a.label == "cawg.identity")
                .unwrap()
        };
        let lifted = genuine.assertions[identity(&genuine)].clone();
        let slot = identity(&other);
        other.assertions[slot] = lifted;
        let signed = sign_manifest(&claim_key, &other).unwrap();
        assert!(verify_agent_provenance_manifest(&claim_key.verifying_key(), &signed).is_err());
    }

    #[test]
    fn agent_identity_references_match_the_authenticated_manifest() {
        let (agent_key, claim_key) = (test_keypair(), test_keypair());
        let credential = agent_credential(&claim_key, &agent_key);
        let params = AgentProvenanceParams {
            asset: b"agent content",
            agent_credential: &credential,
            memory_provenance: b"memory",
            process_proof: None,
            attestation: None,
            training_policy: None,
        };
        let (manifest, _) =
            create_agent_provenance_manifest(&claim_key, &agent_key, &params).unwrap();
        let hard = manifest
            .assertions
            .iter()
            .find(|assertion| assertion.label == "c2pa.hash.data")
            .unwrap();
        let hard_hash = Sha256::digest(serde_json::to_vec(&hard.data).unwrap()).to_vec();
        let valid_hard = (
            "self#jumbf=c2pa.assertions/c2pa.hash.data".to_string(),
            "sha256".to_string(),
            hard_hash.clone(),
        );
        for refs in [
            vec![valid_hard.clone(), valid_hard.clone()],
            vec![
                valid_hard.clone(),
                (
                    "self#jumbf=c2pa.assertions/c2pa.actions".to_string(),
                    "sha256".to_string(),
                    vec![0x55; 32],
                ),
            ],
            vec![(
                "self#jumbf=another-manifest/c2pa.assertions/c2pa.hash.data".to_string(),
                "sha256".to_string(),
                hard_hash.clone(),
            )],
        ] {
            let mut forged = manifest.clone();
            forged
                .assertions
                .iter_mut()
                .find(|assertion| assertion.label == "cawg.identity")
                .unwrap()
                .data = cawg::create_identity_assertion_ica(&agent_key, &refs, "agent").unwrap();
            let signed = sign_manifest(&claim_key, &forged).unwrap();
            assert!(verify_agent_provenance_manifest(&claim_key.verifying_key(), &signed).is_err());
        }
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
