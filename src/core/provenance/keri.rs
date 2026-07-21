// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::did;

/// KERI event types per Key Event Receipt Infrastructure spec (IETF draft-ssmith-keri).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventType {
    Inception,
    Rotation,
    Interaction,
}

/// A KERI key event in the Key Event Log (KEL).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyEvent {
    #[serde(rename = "v")]
    pub version: String,
    #[serde(rename = "t")]
    pub event_type: EventType,
    #[serde(rename = "d")]
    pub digest: String,
    #[serde(rename = "i")]
    pub identifier: String,
    #[serde(rename = "s")]
    pub sequence: u64,
    #[serde(rename = "kt")]
    pub key_threshold: u64,
    #[serde(rename = "k")]
    pub keys: Vec<String>,
    #[serde(rename = "nt", skip_serializing_if = "Option::is_none")]
    pub next_threshold: Option<u64>,
    #[serde(rename = "n", skip_serializing_if = "Option::is_none")]
    pub next_keys_digest: Option<String>,
    #[serde(rename = "p", skip_serializing_if = "Option::is_none")]
    pub prior_digest: Option<String>,
    #[serde(rename = "a", default, skip_serializing_if = "Vec::is_empty")]
    pub anchors: Vec<serde_json::Value>,
    pub signature: String,
}

/// A local Key Event Log for tracking key lifecycle.
#[derive(Default)]
pub struct KeyEventLog {
    events: Vec<KeyEvent>,
    path: Option<std::path::PathBuf>,
}

impl KeyEventLog {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            path: None,
        }
    }

    pub fn with_path(path: &std::path::Path) -> Result<Self> {
        let events = if path.exists() {
            let data =
                std::fs::read_to_string(path).map_err(|e| anyhow!("failed to read KEL: {e}"))?;
            serde_json::from_str(&data).map_err(|e| anyhow!("failed to parse KEL: {e}"))?
        } else {
            Vec::new()
        };
        Ok(Self {
            events,
            path: Some(path.to_path_buf()),
        })
    }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.path {
            let data = serde_json::to_vec_pretty(&self.events)
                .map_err(|e| anyhow!("KEL serialization failed: {e}"))?;
            std::fs::write(path, data).map_err(|e| anyhow!("KEL write failed: {e}"))?;
        }
        Ok(())
    }

    pub fn events(&self) -> &[KeyEvent] {
        &self.events
    }

    pub fn current_keys(&self) -> Option<&[String]> {
        self.events.last().map(|e| e.keys.as_slice())
    }

    pub fn latest_event(&self) -> Option<&KeyEvent> {
        self.events.last()
    }

    /// Create an inception event establishing the identifier with an initial key.
    pub fn inception(
        &mut self,
        signing_key: &SigningKey,
        next_key_digest: Option<&str>,
    ) -> Result<&KeyEvent> {
        if !self.events.is_empty() {
            return Err(anyhow!("KEL already has an inception event"));
        }

        let pk_bytes = signing_key.verifying_key().to_bytes();
        let identifier = did::did_key_from_ed25519(&pk_bytes);
        let current_key = did::did_key_from_ed25519(&pk_bytes);

        let mut event = KeyEvent {
            version: "KERI10JSON".to_string(),
            event_type: EventType::Inception,
            digest: String::new(),
            identifier: identifier.clone(),
            sequence: 0,
            key_threshold: 1,
            keys: vec![current_key],
            next_threshold: next_key_digest.map(|_| 1),
            next_keys_digest: next_key_digest.map(String::from),
            prior_digest: None,
            anchors: Vec::new(),
            signature: String::new(),
        };

        event.digest = self_addressing_digest(&event)?;
        event.signature = sign_event(signing_key, &event)?;

        self.events.push(event);
        self.persist()?;
        Ok(self.events.last().unwrap())
    }

    /// Create a rotation event replacing the current key with a new key.
    pub fn rotation(
        &mut self,
        current_key: &SigningKey,
        new_key: &SigningKey,
        next_key_digest: Option<&str>,
    ) -> Result<&KeyEvent> {
        let prior = self
            .events
            .last()
            .ok_or_else(|| anyhow!("no prior event for rotation"))?;

        let new_pk = new_key.verifying_key().to_bytes();
        let new_did = did::did_key_from_ed25519(&new_pk);

        let mut event = KeyEvent {
            version: "KERI10JSON".to_string(),
            event_type: EventType::Rotation,
            digest: String::new(),
            identifier: prior.identifier.clone(),
            sequence: prior.sequence + 1,
            key_threshold: 1,
            keys: vec![new_did],
            next_threshold: next_key_digest.map(|_| 1),
            next_keys_digest: next_key_digest.map(String::from),
            prior_digest: Some(prior.digest.clone()),
            anchors: Vec::new(),
            signature: String::new(),
        };

        event.digest = self_addressing_digest(&event)?;
        event.signature = sign_event(current_key, &event)?;

        self.events.push(event);
        self.persist()?;
        Ok(self.events.last().unwrap())
    }

    /// Create an interaction event anchoring data (seals) without rotating keys.
    pub fn interaction(
        &mut self,
        signing_key: &SigningKey,
        anchors: Vec<serde_json::Value>,
    ) -> Result<&KeyEvent> {
        let prior = self
            .events
            .last()
            .ok_or_else(|| anyhow!("no prior event for interaction"))?;

        let mut event = KeyEvent {
            version: "KERI10JSON".to_string(),
            event_type: EventType::Interaction,
            digest: String::new(),
            identifier: prior.identifier.clone(),
            sequence: prior.sequence + 1,
            key_threshold: prior.key_threshold,
            keys: prior.keys.clone(),
            next_threshold: prior.next_threshold,
            next_keys_digest: prior.next_keys_digest.clone(),
            prior_digest: Some(prior.digest.clone()),
            anchors,
            signature: String::new(),
        };

        event.digest = self_addressing_digest(&event)?;
        event.signature = sign_event(signing_key, &event)?;

        self.events.push(event);
        self.persist()?;
        Ok(self.events.last().unwrap())
    }

    /// Verify the entire KEL chain: each event's digest and signature, plus chain linkage.
    pub fn verify(&self) -> Result<()> {
        if self.events.is_empty() {
            return Err(anyhow!("empty KEL"));
        }

        if self.events[0].event_type != EventType::Inception {
            return Err(anyhow!("first event must be inception"));
        }

        let mut current_keys = Vec::new();

        for (i, event) in self.events.iter().enumerate() {
            if event.sequence != i as u64 {
                return Err(anyhow!(
                    "sequence gap: expected {i}, got {}",
                    event.sequence
                ));
            }

            // A KeyEvent carries a single signature, so it cannot represent an
            // m-of-n multisig. Reject kt>1 rather than silently accepting one
            // signature as satisfying a multisig threshold.
            if event.key_threshold > 1 {
                return Err(anyhow!(
                    "multisig key threshold ({}) not supported at event {i}",
                    event.key_threshold
                ));
            }

            let computed = self_addressing_digest(event)?;
            if computed != event.digest {
                return Err(anyhow!("digest mismatch at event {i}"));
            }

            if i > 0 {
                let prior = &self.events[i - 1];
                if event
                    .prior_digest
                    .as_ref()
                    .is_none_or(|d| d != &prior.digest)
                {
                    return Err(anyhow!("chain break at event {i}"));
                }

                // Pre-rotation: a rotation may only install keys the prior event
                // committed to via its next-key digest. This is KERI's core
                // security property — a compromised current key cannot rotate to
                // an attacker-chosen key.
                if event.event_type == EventType::Rotation {
                    let committed = prior.next_keys_digest.as_ref().ok_or_else(|| {
                        anyhow!("rotation at event {i} but prior event pre-committed no next key")
                    })?;
                    if &next_keys_digest(&event.keys)? != committed {
                        return Err(anyhow!(
                            "rotation at event {i} installs keys not matching the prior commitment"
                        ));
                    }
                }
            }

            if i == 0 {
                verify_event_signature(event, &event.keys)?;
            } else {
                verify_event_signature(event, &current_keys)?;
            }

            current_keys.clone_from(&event.keys);
        }

        Ok(())
    }
}

/// Canonical digest committing to a next authorized key set: SHA-256 over the
/// concatenated raw Ed25519 public keys decoded from each `did:key`, hex-encoded.
/// For a single key this equals SHA-256(pk_bytes), matching how callers derive
/// the pre-rotation commitment passed to `inception`/`rotation`.
fn next_keys_digest(keys: &[String]) -> Result<String> {
    let mut hasher = Sha256::new();
    for key_did in keys {
        let pk = did::ed25519_from_did_key(key_did)?;
        hasher.update(pk);
    }
    let hash = hasher.finalize();
    Ok(hash.iter().map(|b| format!("{b:02x}")).collect())
}

fn self_addressing_digest(event: &KeyEvent) -> Result<String> {
    let mut e = event.clone();
    e.digest = String::new();
    e.signature = String::new();
    let serialized =
        serde_json::to_vec(&e).map_err(|e| anyhow!("event serialization failed: {e}"))?;
    let hash = Sha256::digest(&serialized);
    Ok(hash.iter().map(|b| format!("{b:02x}")).collect())
}

fn sign_event(signing_key: &SigningKey, event: &KeyEvent) -> Result<String> {
    let mut e = event.clone();
    e.signature = String::new();
    let serialized =
        serde_json::to_vec(&e).map_err(|e| anyhow!("event serialization failed: {e}"))?;
    let signature = signing_key.sign(&serialized);
    Ok(multibase::encode(
        multibase::Base::Base58Btc,
        signature.to_bytes(),
    ))
}

fn verify_event_signature(event: &KeyEvent, signer_keys: &[String]) -> Result<()> {
    let mut e = event.clone();
    e.signature = String::new();
    let serialized =
        serde_json::to_vec(&e).map_err(|e| anyhow!("event serialization failed: {e}"))?;

    let (_, sig_bytes) =
        multibase::decode(&event.signature).map_err(|e| anyhow!("multibase decode failed: {e}"))?;
    let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow!("invalid signature: {e}"))?;

    for key_did in signer_keys {
        if let Ok(pk_bytes) = did::ed25519_from_did_key(key_did) {
            if let Ok(vk) = VerifyingKey::from_bytes(&pk_bytes) {
                if vk.verify(&serialized, &signature).is_ok() {
                    return Ok(());
                }
            }
        }
    }

    Err(anyhow!("no authorized key verified the signature"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> SigningKey {
        SigningKey::generate(&mut rand::thread_rng())
    }

    fn next_key_digest(key: &SigningKey) -> String {
        let pk = key.verifying_key().to_bytes();
        let hash = Sha256::digest(pk);
        hash.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn inception_and_verify() {
        let key = test_keypair();
        let mut kel = KeyEventLog::new();
        let event = kel.inception(&key, None).unwrap();

        assert_eq!(event.event_type, EventType::Inception);
        assert_eq!(event.sequence, 0);
        assert!(!event.digest.is_empty());
        kel.verify().unwrap();
    }

    #[test]
    fn rotation_and_verify() {
        let key1 = test_keypair();
        let key2 = test_keypair();
        let mut kel = KeyEventLog::new();

        let nkd = next_key_digest(&key2);
        kel.inception(&key1, Some(&nkd)).unwrap();
        let rot = kel.rotation(&key1, &key2, None).unwrap();

        assert_eq!(rot.event_type, EventType::Rotation);
        assert_eq!(rot.sequence, 1);
        assert!(rot.prior_digest.is_some());
        kel.verify().unwrap();
    }

    #[test]
    fn interaction_anchors_data() {
        let key = test_keypair();
        let mut kel = KeyEventLog::new();
        kel.inception(&key, None).unwrap();

        let seal = serde_json::json!({"i": "fact-123", "d": "abc123"});
        let ixn = kel.interaction(&key, vec![seal.clone()]).unwrap();

        assert_eq!(ixn.event_type, EventType::Interaction);
        assert_eq!(ixn.anchors.len(), 1);
        kel.verify().unwrap();
    }

    #[test]
    fn tampered_event_rejected() {
        let key = test_keypair();
        let mut kel = KeyEventLog::new();
        kel.inception(&key, None).unwrap();
        kel.interaction(&key, Vec::new()).unwrap();

        kel.events[1].anchors = vec![serde_json::json!({"tampered": true})];
        assert!(kel.verify().is_err());
    }

    #[test]
    fn rotation_to_uncommitted_key_rejected() {
        // Inception pre-commits to key2, but the holder rotates to key3 instead.
        // Pre-rotation enforcement must reject the chain even though the
        // rotation event is validly signed by the current key.
        let key1 = test_keypair();
        let key2 = test_keypair();
        let key3 = test_keypair();
        let mut kel = KeyEventLog::new();

        kel.inception(&key1, Some(&next_key_digest(&key2))).unwrap();
        kel.rotation(&key1, &key3, None).unwrap();

        assert!(kel.verify().is_err());
    }

    #[test]
    fn rotation_without_precommitment_rejected() {
        // Inception commits no next key, so any rotation is unauthorized.
        let key1 = test_keypair();
        let key2 = test_keypair();
        let mut kel = KeyEventLog::new();

        kel.inception(&key1, None).unwrap();
        kel.rotation(&key1, &key2, None).unwrap();

        assert!(kel.verify().is_err());
    }

    #[test]
    fn multisig_threshold_rejected() {
        let key = test_keypair();
        let mut kel = KeyEventLog::new();
        kel.inception(&key, None).unwrap();
        kel.events[0].key_threshold = 2;

        assert!(kel.verify().is_err());
    }

    #[test]
    fn double_inception_rejected() {
        let key = test_keypair();
        let mut kel = KeyEventLog::new();
        kel.inception(&key, None).unwrap();
        assert!(kel.inception(&key, None).is_err());
    }

    #[test]
    fn full_lifecycle() {
        let key1 = test_keypair();
        let key2 = test_keypair();
        let key3 = test_keypair();
        let mut kel = KeyEventLog::new();

        kel.inception(&key1, Some(&next_key_digest(&key2))).unwrap();
        kel.interaction(&key1, vec![serde_json::json!({"seal": "a"})])
            .unwrap();
        kel.rotation(&key1, &key2, Some(&next_key_digest(&key3)))
            .unwrap();
        kel.interaction(&key2, vec![serde_json::json!({"seal": "b"})])
            .unwrap();
        kel.rotation(&key2, &key3, None).unwrap();

        assert_eq!(kel.events().len(), 5);
        kel.verify().unwrap();

        let current = kel.current_keys().unwrap();
        let key3_did = did::did_key_from_ed25519(&key3.verifying_key().to_bytes());
        assert_eq!(current, &[key3_did]);
    }

    #[test]
    fn persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let kel_path = dir.path().join("kel.json");

        let key1 = test_keypair();
        let key2 = test_keypair();

        {
            let mut kel = KeyEventLog::with_path(&kel_path).unwrap();
            kel.inception(&key1, Some(&next_key_digest(&key2))).unwrap();
            kel.interaction(&key1, vec![serde_json::json!({"fact": "f1"})])
                .unwrap();
            kel.rotation(&key1, &key2, None).unwrap();
            assert_eq!(kel.events().len(), 3);
        }

        let loaded = KeyEventLog::with_path(&kel_path).unwrap();
        assert_eq!(loaded.events().len(), 3);
        loaded.verify().unwrap();
        let key2_did = did::did_key_from_ed25519(&key2.verifying_key().to_bytes());
        assert_eq!(loaded.current_keys().unwrap(), &[key2_did]);
    }
}
