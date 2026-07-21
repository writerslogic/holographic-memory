// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Caller-supplied trust anchors for provenance verification.
//!
//! Signature checks alone only prove a record is *internally self-consistent*:
//! the key embedded in the record verifies the signature the record carries.
//! Authenticity requires an independent trust decision — "do I trust the party
//! that holds this key?". A [`TrustStore`] is the set of keys/issuers the caller
//! has decided to trust; verification is authentic only when the key that
//! produced a signature is present in the store.

use std::collections::HashSet;

use anyhow::Result;
use ed25519_dalek::VerifyingKey;

use super::did;

/// A set of trusted Ed25519 verification keys.
///
/// A record is treated as authentic only if the key that produced its
/// signature is in this set. Keys can be added directly or via their
/// `did:key` representation.
#[derive(Default, Clone)]
pub struct TrustStore {
    keys: HashSet<[u8; 32]>,
}

impl TrustStore {
    /// An empty trust store. It trusts nothing, so every record verifies as
    /// non-authentic until keys are added.
    pub fn new() -> Self {
        Self::default()
    }

    /// Trust a raw verifying key.
    pub fn trust_key(&mut self, key: &VerifyingKey) -> &mut Self {
        self.keys.insert(key.to_bytes());
        self
    }

    /// Trust the holder of a `did:key` identifier. Errors if the DID is not a
    /// valid Ed25519 `did:key`.
    pub fn trust_did(&mut self, did_key: &str) -> Result<&mut Self> {
        let pk = did::ed25519_from_did_key(did_key)?;
        self.keys.insert(pk);
        Ok(self)
    }

    /// Trust the holder of a `did:jwk` identifier (used by CAWG ICA issuers).
    /// Errors if the DID is not a valid Ed25519 `did:jwk`.
    pub fn trust_did_jwk(&mut self, did_jwk: &str) -> Result<&mut Self> {
        let pk = did::ed25519_from_did_jwk(did_jwk)?;
        self.keys.insert(pk);
        Ok(self)
    }

    /// Convenience constructor: a store trusting a single verifying key.
    pub fn trusting_key(key: &VerifyingKey) -> Self {
        let mut s = Self::new();
        s.trust_key(key);
        s
    }

    /// Convenience constructor: a store trusting a single `did:key` issuer.
    pub fn trusting_did(did_key: &str) -> Result<Self> {
        let mut s = Self::new();
        s.trust_did(did_key)?;
        Ok(s)
    }

    /// Whether the given key is trusted.
    pub fn is_trusted(&self, key: &VerifyingKey) -> bool {
        self.keys.contains(&key.to_bytes())
    }

    /// Whether the store trusts nothing.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of distinct trusted keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn key() -> SigningKey {
        SigningKey::generate(&mut rand::thread_rng())
    }

    #[test]
    fn empty_trusts_nothing() {
        let store = TrustStore::new();
        assert!(store.is_empty());
        assert!(!store.is_trusted(&key().verifying_key()));
    }

    #[test]
    fn trusts_added_key_only() {
        let trusted = key();
        let other = key();
        let store = TrustStore::trusting_key(&trusted.verifying_key());
        assert!(store.is_trusted(&trusted.verifying_key()));
        assert!(!store.is_trusted(&other.verifying_key()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn trust_via_did_roundtrips() {
        let k = key();
        let did = did::did_key_from_ed25519(&k.verifying_key().to_bytes());
        let store = TrustStore::trusting_did(&did).unwrap();
        assert!(store.is_trusted(&k.verifying_key()));
    }
}
