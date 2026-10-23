// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use base64::Engine;
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};

use super::did;
use super::vc::iso8601_now;

/// A hashed URI reference to a C2PA assertion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashedUri {
    pub url: String,
    pub alg: String,
    pub hash: String,
}

/// Build a HashedUri from a label and content bytes (hex-encoded SHA-256 digest).
pub fn hash_assertion(label: &str, content: &[u8]) -> HashedUri {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(content);
    HashedUri {
        url: format!("self#jumbf=c2pa.assertions/{label}"),
        alg: "sha256".to_string(),
        hash: super::hex_encode(&hash),
    }
}

/// The CAWG ICA `sig_type` for the identity claims aggregation credential.
const CAWG_ICA_SIG_TYPE: &str = "cawg.identity_claims_aggregation";
const VC_CONTEXT: &str = "https://www.w3.org/ns/credentials/v2";
const ICA_CONTEXT: &str = "https://cawg.io/identity/1.1/ica/context/";
const ICA_CREDENTIAL_TYPE: &str = "IdentityClaimsAggregationCredential";
const COSE_CONTENT_TYPE_VC: &str = "application/vc";

/// Create a CAWG Identity Claims Aggregation (ICA) identity assertion — the production
/// path c2pa-rs ships. Builds the cogmem ICA shape: a `signer_payload`
/// `{referenced_assertions, sig_type}`, a W3C VC v2 of type
/// `IdentityClaimsAggregationCredential` issued by the agent's did:jwk, secured by a
/// tag-18 COSE_Sign1 (EdDSA, `application/vc`) over the VC JSON, embedded as
/// `{signer_payload, signature, pad1}`. `referenced` is `(url, alg, raw_hash)` and MUST
/// include the hard binding (a `c2pa.hash.*` assertion). Returns the JSON wrapper stored
/// under the `cawg.identity` assertion label: `{"alg":"EdDSA","ica":"<base64 CBOR>"}`,
/// whose `ica` field is the embedded IdentityAssertion CBOR that [`verify_cawg_ica`]
/// reads back.
pub fn create_identity_assertion_ica(
    agent_key: &SigningKey,
    referenced: &[(String, String, Vec<u8>)],
    display_name: &str,
) -> Result<serde_json::Value> {
    use ciborium::value::Value;

    let issuer_did = did::did_jwk_from_ed25519(&agent_key.verifying_key().to_bytes());
    let now = iso8601_now();

    // signer_payload: CBOR map with raw bytestring hashes (carried in the assertion).
    let sp_cbor_refs: Vec<Value> = referenced
        .iter()
        .map(|(url, alg, hash)| {
            Value::Map(vec![
                (Value::Text("url".into()), Value::Text(url.clone())),
                (Value::Text("alg".into()), Value::Text(alg.clone())),
                (Value::Text("hash".into()), Value::Bytes(hash.clone())),
            ])
        })
        .collect();
    let signer_payload = Value::Map(vec![
        (
            Value::Text("referenced_assertions".into()),
            Value::Array(sp_cbor_refs),
        ),
        (
            Value::Text("sig_type".into()),
            Value::Text(CAWG_ICA_SIG_TYPE.into()),
        ),
    ]);

    // c2paAsset: the SignerPayload in JSON form with STANDARD base64 hashes.
    let json_refs: Vec<serde_json::Value> = referenced
        .iter()
        .map(|(url, alg, hash)| {
            serde_json::json!({
                "url": url,
                "alg": alg,
                "hash": base64::engine::general_purpose::STANDARD.encode(hash),
            })
        })
        .collect();
    let c2pa_asset = serde_json::json!({
        "referenced_assertions": json_refs,
        "sig_type": CAWG_ICA_SIG_TYPE,
    });

    // W3C VC v2 IdentityClaimsAggregationCredential.
    let vc = serde_json::json!({
        "@context": [VC_CONTEXT, ICA_CONTEXT],
        "type": ["VerifiableCredential", ICA_CREDENTIAL_TYPE],
        "issuer": issuer_did,
        "validFrom": now,
        "credentialSubject": {
            "verifiedIdentities": [{
                "type": "writersproof.ai_agent",
                "name": display_name,
                "verifiedAt": now,
                "provider": { "id": "https://writersproof.com", "name": "WritersProof" },
            }],
            "c2paAsset": c2pa_asset,
        },
    });
    let vc_bytes =
        serde_json::to_vec(&vc).map_err(|e| anyhow::anyhow!("VC serialization failed: {e}"))?;

    // Tag-18 COSE_Sign1 over the VC JSON: protected {alg EdDSA, content_type
    // application/vc}, empty external_aad.
    let cose = sign_cose_vc(agent_key, &vc_bytes)?;

    // Embedded IdentityAssertion: {signer_payload, signature, pad1}.
    let assertion = Value::Map(vec![
        (Value::Text("signer_payload".into()), signer_payload),
        (Value::Text("signature".into()), Value::Bytes(cose)),
        (Value::Text("pad1".into()), Value::Bytes(Vec::new())),
    ]);
    let mut assertion_cbor = Vec::new();
    ciborium::into_writer(&assertion, &mut assertion_cbor)
        .map_err(|e| anyhow::anyhow!("identity assertion CBOR encoding failed: {e}"))?;

    Ok(serde_json::json!({
        "alg": "EdDSA",
        "ica": base64::engine::general_purpose::STANDARD.encode(&assertion_cbor),
    }))
}

/// Read the embedded IdentityAssertion CBOR bytes from a `cawg.identity` assertion JSON
/// produced by [`create_identity_assertion_ica`] (the base64-encoded `ica` field).
pub fn ica_embedded_bytes(assertion: &serde_json::Value) -> Result<Vec<u8>> {
    let ica = assertion
        .get("ica")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("cawg.identity assertion has no ica field"))?;
    base64::engine::general_purpose::STANDARD
        .decode(ica)
        .map_err(|e| anyhow::anyhow!("ica base64 decode failed: {e}"))
}

/// Tag-18 COSE_Sign1 over the VC JSON: protected {alg EdDSA, content_type
/// application/vc}, empty external_aad, non-detached payload.
fn sign_cose_vc(signing_key: &SigningKey, payload: &[u8]) -> Result<Vec<u8>> {
    use coset::iana;
    use coset::{CoseSign1Builder, HeaderBuilder, TaggedCborSerializable};

    let protected = HeaderBuilder::new()
        .algorithm(iana::Algorithm::EdDSA)
        .content_type(COSE_CONTENT_TYPE_VC.to_string())
        .build();

    let sign1 = CoseSign1Builder::new()
        .protected(protected)
        .payload(payload.to_vec())
        .create_signature(b"", |data| signing_key.sign(data).to_bytes().to_vec())
        .build();

    sign1
        .to_tagged_vec()
        .map_err(|e| anyhow::anyhow!("COSE_Sign1 tagged serialization failed: {e}"))
}

/// Verify a CAWG Identity Claims Aggregation (ICA) identity assertion from its raw CBOR
/// bytes — the `cawg.identity` assertion box content `{signer_payload, signature, pad1}`.
/// The signature is a tag-18 COSE_Sign1 (`application/vc`) over the W3C VC JSON, signed by
/// the did:jwk issuer. Checks: the COSE signature is valid under the issuer DID, the
/// assertion's `signer_payload` equals the `c2paAsset` the signed VC carries, sig_type is
/// ICA, and a hard binding is referenced. Returns the verified, parsed VC JSON.
pub fn verify_cawg_ica(assertion_cbor: &[u8]) -> Result<serde_json::Value> {
    verify_cawg_ica_references(assertion_cbor).map(|(vc, _)| vc)
}

/// An assertion an identity assertion signs over: its URL, hash algorithm and raw hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReferencedAssertion {
    pub url: String,
    pub alg: Option<String>,
    pub hash: Vec<u8>,
}

impl ReferencedAssertion {
    /// The assertion label the URL ends in (`c2pa.hash.data` for
    /// `self#jumbf=c2pa.assertions/c2pa.hash.data`).
    pub fn label(&self) -> &str {
        self.url.rsplit_once('/').map_or("", |(_, label)| label)
    }
}

/// [`verify_cawg_ica`], also returning the signed referenced assertions.
pub(crate) fn verify_cawg_ica_references(
    assertion_cbor: &[u8],
) -> Result<(serde_json::Value, Vec<ReferencedAssertion>)> {
    use ciborium::value::Value;
    use coset::{CoseSign1, TaggedCborSerializable};

    let assertion: Value = ciborium::from_reader(assertion_cbor)
        .map_err(|e| anyhow::anyhow!("identity assertion CBOR parse failed: {e}"))?;
    let map = match &assertion {
        Value::Map(entries) => entries,
        _ => return Err(anyhow::anyhow!("identity assertion is not a CBOR map")),
    };
    let get = |key: &str| -> Option<&Value> {
        map.iter()
            .find(|(k, _)| matches!(k, Value::Text(t) if t == key))
            .map(|(_, v)| v)
    };

    let signer_payload = match get("signer_payload") {
        Some(payload @ Value::Map(_)) => cbor_to_json(payload)?,
        _ => return Err(anyhow::anyhow!("missing signer_payload map")),
    };

    let sig_bytes = match get("signature") {
        Some(Value::Bytes(b)) => b,
        _ => return Err(anyhow::anyhow!("missing signature byte string")),
    };

    let sign1 = CoseSign1::from_tagged_slice(sig_bytes)
        .map_err(|e| anyhow::anyhow!("COSE_Sign1 deserialization failed: {e}"))?;
    let payload = sign1
        .payload
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("COSE envelope has no payload"))?;
    let vc: serde_json::Value = serde_json::from_slice(payload)
        .map_err(|e| anyhow::anyhow!("VC JSON parse failed: {e}"))?;

    let issuer = vc
        .get("issuer")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("VC has no issuer"))?;
    let key = ed25519_dalek::VerifyingKey::from_bytes(&super::did::ed25519_from_did_jwk(issuer)?)
        .map_err(|e| anyhow::anyhow!("invalid issuer key: {e}"))?;
    super::cose::verify_ed25519(&sign1, &key)
        .map_err(|e| anyhow::anyhow!("CAWG ICA verification failed: {e}"))?;

    // The outer signer_payload is not signed; the copy inside the VC is. Everything below
    // reads the signed copy, and the two must agree so a reader of either sees the same refs.
    let c2pa_asset = &normalize_hashes(
        vc.get("credentialSubject")
            .and_then(|s| s.get("c2paAsset"))
            .ok_or_else(|| anyhow::anyhow!("VC has no credentialSubject.c2paAsset"))?,
    )?;
    if *c2pa_asset != signer_payload {
        return Err(anyhow::anyhow!(
            "signer_payload differs from the signed c2paAsset"
        ));
    }

    if c2pa_asset.get("sig_type").and_then(|v| v.as_str()) != Some(CAWG_ICA_SIG_TYPE) {
        return Err(anyhow::anyhow!(
            "sig_type is not cawg.identity_claims_aggregation"
        ));
    }
    let referenced = c2pa_asset
        .get("referenced_assertions")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing referenced_assertions array"))?
        .iter()
        .map(|item| {
            let field = |name: &str| item.get(name).and_then(|v| v.as_str());
            let hash =
                field("hash").ok_or_else(|| anyhow::anyhow!("referenced assertion has no hash"))?;
            Ok(ReferencedAssertion {
                url: field("url")
                    .ok_or_else(|| anyhow::anyhow!("referenced assertion has no url"))?
                    .to_string(),
                alg: field("alg").map(str::to_string),
                hash: base64::engine::general_purpose::STANDARD
                    .decode(hash)
                    .map_err(|e| anyhow::anyhow!("referenced assertion hash: {e}"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if !referenced
        .iter()
        .any(|r| r.label().starts_with("c2pa.hash."))
    {
        return Err(anyhow::anyhow!("missing required hard binding"));
    }

    Ok((vc, referenced))
}

/// `asset` with each referenced assertion's hash as padded standard base64, the form
/// [`cbor_to_json`] gives byte strings. Producers write the hash as base64 (cogmem, this
/// crate) or as an array of byte values (c2pa-rs); both name the same bytes.
fn normalize_hashes(asset: &serde_json::Value) -> Result<serde_json::Value> {
    use base64::engine::{
        general_purpose::GeneralPurpose, DecodePaddingMode, GeneralPurposeConfig,
    };
    const LENIENT: GeneralPurpose = GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
    );
    let mut asset = asset.clone();
    let refs = asset
        .get_mut("referenced_assertions")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| anyhow::anyhow!("missing referenced_assertions array"))?;
    for item in refs {
        let hash = item
            .get_mut("hash")
            .ok_or_else(|| anyhow::anyhow!("referenced assertion has no hash"))?;
        let bytes = match &*hash {
            serde_json::Value::String(b64) => LENIENT
                .decode(b64)
                .map_err(|e| anyhow::anyhow!("referenced assertion hash: {e}"))?,
            serde_json::Value::Array(values) => values
                .iter()
                .map(|v| v.as_u64().and_then(|b| u8::try_from(b).ok()))
                .collect::<Option<Vec<u8>>>()
                .ok_or_else(|| anyhow::anyhow!("referenced assertion hash is not bytes"))?,
            _ => return Err(anyhow::anyhow!("referenced assertion hash is not bytes")),
        };
        *hash = base64::engine::general_purpose::STANDARD
            .encode(bytes)
            .into();
    }
    Ok(asset)
}

/// A CBOR value as the JSON a VC carries it in: byte strings become standard base64.
fn cbor_to_json(value: &ciborium::value::Value) -> Result<serde_json::Value> {
    use ciborium::value::Value;
    Ok(match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Text(t) => serde_json::Value::String(t.clone()),
        Value::Bytes(b) => {
            serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(b))
        }
        Value::Integer(i) => {
            let i = i128::from(*i);
            if let Ok(u) = u64::try_from(i) {
                u.into()
            } else {
                i64::try_from(i)
                    .map_err(|_| anyhow::anyhow!("signer_payload integer out of range"))?
                    .into()
            }
        }
        Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(cbor_to_json).collect::<Result<_>>()?)
        }
        Value::Map(entries) => {
            let mut object = serde_json::Map::new();
            for (k, v) in entries {
                let Value::Text(k) = k else {
                    return Err(anyhow::anyhow!("signer_payload has a non-text key"));
                };
                if object.insert(k.clone(), cbor_to_json(v)?).is_some() {
                    return Err(anyhow::anyhow!("signer_payload repeats key {k:?}"));
                }
            }
            serde_json::Value::Object(object)
        }
        _ => {
            return Err(anyhow::anyhow!(
                "signer_payload holds an unsupported CBOR value"
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::from_bytes(&rand::random())
    }

    fn ica_refs() -> Vec<(String, String, Vec<u8>)> {
        vec![
            (
                "self#jumbf=c2pa.assertions/c2pa.hash.data".to_string(),
                "sha256".to_string(),
                vec![0x11u8; 32],
            ),
            (
                "self#jumbf=c2pa.assertions/cogmem.memory.provenance".to_string(),
                "sha256".to_string(),
                vec![0x22u8; 32],
            ),
        ]
    }

    #[test]
    fn create_and_verify_ica_assertion() {
        let key = test_keypair();
        let assertion = create_identity_assertion_ica(&key, &ica_refs(), "cogmem agent").unwrap();
        let embedded = ica_embedded_bytes(&assertion).unwrap();
        let vc = verify_cawg_ica(&embedded).unwrap();

        let expected_issuer = did::did_jwk_from_ed25519(&key.verifying_key().to_bytes());
        assert_eq!(vc["issuer"].as_str().unwrap(), expected_issuer);
        assert!(vc["type"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "IdentityClaimsAggregationCredential"));
        assert_eq!(
            vc["credentialSubject"]["verifiedIdentities"][0]["name"]
                .as_str()
                .unwrap(),
            "cogmem agent"
        );
    }

    #[test]
    fn ica_assertion_tamper_rejected() {
        let key = test_keypair();
        let assertion = create_identity_assertion_ica(&key, &ica_refs(), "cogmem agent").unwrap();
        let mut embedded = ica_embedded_bytes(&assertion).unwrap();
        // Corrupt the COSE signature bytes near the tail — must be rejected.
        let last = embedded.len() - 1;
        embedded[last] ^= 0xff;
        assert!(verify_cawg_ica(&embedded).is_err());
    }

    #[test]
    fn ica_assertion_requires_hard_binding() {
        let key = test_keypair();
        let refs = vec![(
            "self#jumbf=c2pa.assertions/cogmem.memory.provenance".to_string(),
            "sha256".to_string(),
            vec![0x33u8; 32],
        )];
        let assertion = create_identity_assertion_ica(&key, &refs, "cogmem agent").unwrap();
        let embedded = ica_embedded_bytes(&assertion).unwrap();
        assert!(verify_cawg_ica(&embedded).is_err());
    }

    /// Re-encodes an embedded identity assertion with `edit` applied to its outer
    /// signer_payload, leaving the signature untouched.
    fn with_signer_payload(
        embedded: &[u8],
        edit: impl FnOnce(&mut Vec<(ciborium::Value, ciborium::Value)>),
    ) -> Vec<u8> {
        use ciborium::Value;
        let mut assertion: Value = ciborium::from_reader(embedded).unwrap();
        let Value::Map(entries) = &mut assertion else {
            panic!("not a map")
        };
        let (_, Value::Map(payload)) = entries
            .iter_mut()
            .find(|(k, _)| k == &Value::Text("signer_payload".into()))
            .unwrap()
        else {
            panic!("no signer_payload")
        };
        edit(payload);
        let mut out = Vec::new();
        ciborium::into_writer(&assertion, &mut out).unwrap();
        out
    }

    #[test]
    fn unsigned_signer_payload_must_match_the_signed_one() {
        use ciborium::Value;
        let key = test_keypair();
        let assertion = create_identity_assertion_ica(&key, &ica_refs(), "cogmem agent").unwrap();
        let embedded = ica_embedded_bytes(&assertion).unwrap();

        // Retargeted at another asset's hard binding: the signature still verifies, but the
        // payload no longer says what was signed.
        let retargeted = with_signer_payload(&embedded, |payload| {
            let (_, Value::Array(refs)) = &mut payload[0] else {
                panic!("no referenced_assertions")
            };
            let Value::Map(first) = &mut refs[0] else {
                panic!("not a map")
            };
            first[2].1 = Value::Bytes(vec![0x99; 32]);
        });
        assert!(verify_cawg_ica(&retargeted).is_err());

        let extended = with_signer_payload(&embedded, |payload| {
            payload.push((Value::Text("roles".into()), Value::Array(Vec::new())));
        });
        assert!(verify_cawg_ica(&extended).is_err());
    }

    #[test]
    fn c2pa_asset_hashes_compare_as_bytes() {
        let asset = serde_json::json!({
            "referenced_assertions": [
                {"url": "a", "hash": "AAEC"},
                {"url": "b", "hash": "AAE"},
                {"url": "c", "hash": [0, 1]},
            ],
            "sig_type": CAWG_ICA_SIG_TYPE,
        });
        let normalized = normalize_hashes(&asset).unwrap();
        let hashes: Vec<_> = normalized["referenced_assertions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["hash"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(hashes, ["AAEC", "AAE=", "AAE="]);
        let mut bad = asset;
        bad["referenced_assertions"][2]["hash"] = serde_json::json!([256]);
        assert!(normalize_hashes(&bad).is_err());
    }

    #[test]
    fn hard_binding_is_matched_by_label_not_substring() {
        let key = test_keypair();
        let refs = vec![(
            "self#jumbf=c2pa.assertions/x?c2pa.hash.data".to_string(),
            "sha256".to_string(),
            vec![0x44u8; 32],
        )];
        let assertion = create_identity_assertion_ica(&key, &refs, "cogmem agent").unwrap();
        let embedded = ica_embedded_bytes(&assertion).unwrap();
        assert!(verify_cawg_ica(&embedded).is_err());
    }

    #[test]
    fn cross_verifies_cogmem_ica_vector() {
        fn unhex(s: &str) -> Vec<u8> {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect()
        }
        // A cawg.identity (ICA) assertion produced by cogmem (Python / cbor2). HMS verifies
        // the tag-18 COSE over the VC from the raw bytes — cross-implementation conformance.
        let assertion = unhex("a3647061643140697369676e61747572655903f7d28453a20127036e6170706c69636174696f6e2f7663a059039b7b2240636f6e74657874223a5b2268747470733a2f2f7777772e77332e6f72672f6e732f63726564656e7469616c732f7632222c2268747470733a2f2f636177672e696f2f6964656e746974792f312e312f6963612f636f6e746578742f225d2c2263726564656e7469616c5375626a656374223a7b22633270614173736574223a7b227265666572656e6365645f617373657274696f6e73223a5b7b22616c67223a22736861323536222c2268617368223a2241414543417751464267634943516f4c4441304f4478415245684d554652595847426b61477877644868383d222c2275726c223a2273656c66236a756d62663d633270612e617373657274696f6e732f633270612e686173682e64617461227d2c7b22616c67223a22736861323536222c2268617368223a22494345694979516c4a69636f4b536f724c4330754c7a41784d6a4d304e5459334f446b364f7a7739506a383d222c2275726c223a2273656c66236a756d62663d633270612e617373657274696f6e732f636f676d656d2e6d656d6f72792e70726f76656e616e6365227d5d2c227369675f74797065223a22636177672e6964656e746974795f636c61696d735f6167677265676174696f6e227d2c2276657269666965644964656e746974696573223a5b7b226e616d65223a22636f676d656d206167656e74222c2270726f7669646572223a7b226964223a2268747470733a2f2f7772697465727370726f6f662e636f6d222c226e616d65223a225772697465727350726f6f66227d2c2274797065223a22636177672e616666696c696174696f6e222c22757269223a2268747470733a2f2f7772697465727370726f6f662e636f6d2f6167656e74732f636f676d656d222c2276657269666965644174223a22323032362d30312d30315430303a30303a30302b30303a3030227d5d7d2c22697373756572223a226469643a6a776b3a65794a6a636e59694f694a465a4449314e5445354969776961335235496a6f69543074514969776965434936496b45325255683258314250525577305a474e4f4d466b314d485a426256646d617a467151324a775554466d5347523552317043536c5a4e596d636966513d3d222c2274797065223a5b2256657269666961626c6543726564656e7469616c222c224964656e74697479436c61696d734167677265676174696f6e43726564656e7469616c225d2c2276616c696446726f6d223a22323032362d30312d30315430303a30303a30302b30303a3030227d5840957d00066cd6d5f6918f18d9f6330decbc0b1df3664d0be7316865581fd77c169a229196116e1a957b521f9eb11900ab60a0adfadc780b5d586de34f6395380d6e7369676e65725f7061796c6f6164a2687369675f747970657820636177672e6964656e746974795f636c61696d735f6167677265676174696f6e757265666572656e6365645f617373657274696f6e7382a363616c67667368613235366375726c782973656c66236a756d62663d633270612e617373657274696f6e732f633270612e686173682e6461746164686173685820000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1fa363616c67667368613235366375726c783373656c66236a756d62663d633270612e617373657274696f6e732f636f676d656d2e6d656d6f72792e70726f76656e616e636564686173685820202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f");
        let vc = verify_cawg_ica(&assertion).unwrap();
        assert!(vc["issuer"].as_str().unwrap().starts_with("did:jwk:"));
        assert!(vc["type"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "IdentityClaimsAggregationCredential"));
    }
}
