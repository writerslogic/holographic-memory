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

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use ed25519_dalek::VerifyingKey;

use super::{did, status_list};

/// A set of trusted Ed25519 verification keys.
///
/// A record is treated as authentic only if the key that produced its
/// signature is in this set. Keys can be added directly or via their
/// `did:key` representation.
#[derive(Default, Clone)]
pub struct TrustStore {
    keys: HashSet<[u8; 32]>,
    /// Revocation bitstrings, keyed by the status list credential URL that credentials cite.
    status_lists: HashMap<String, StatusListSnapshot>,
    status_list_publishers: HashMap<String, [u8; 32]>,
    /// Whether a credential citing a status list this store does not hold fails its
    /// revocation check instead of passing unchecked.
    require_status_lists: bool,
    max_status_list_age_seconds: Option<u64>,
}

#[derive(Clone)]
struct StatusListSnapshot {
    bits: Vec<u8>,
    signed_at: Option<i64>,
    valid_until: Option<status_list::StatusTime>,
}

/// Whether bit `index` of a revocation bitstring is set: byte `index / 8`, bit `index % 8`
/// counted from the least significant bit. Indices past the end are not revoked.
pub(crate) fn status_bit(bits: &[u8], index: u64) -> bool {
    usize::try_from(index / 8)
        .ok()
        .and_then(|byte_idx| bits.get(byte_idx))
        .is_some_and(|byte| byte & (1 << (index % 8)) != 0)
}

pub(crate) enum StatusLookup {
    Missing,
    Stale,
    OutOfRange,
    Value(bool),
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

    /// Trust `bits` as the current revocation bitstring of the status list published at
    /// `credential_url`, replacing any list trusted for that URL. Credentials citing that URL
    /// then verify as revoked when their bit is set.
    pub fn trust_status_list(&mut self, credential_url: &str, bits: Vec<u8>) -> &mut Self {
        self.status_lists.insert(
            credential_url.to_string(),
            StatusListSnapshot {
                bits,
                signed_at: None,
                valid_until: None,
            },
        );
        self
    }

    pub fn authorize_status_list_publisher(
        &mut self,
        credential_url: &str,
        key: &VerifyingKey,
    ) -> Result<&mut Self> {
        super::status_list::validate_url(credential_url)?;
        anyhow::ensure!(
            self.is_trusted(key),
            "status list publisher key is not trusted"
        );
        self.status_list_publishers
            .insert(credential_url.to_string(), key.to_bytes());
        Ok(self)
    }

    pub fn trust_status_list_credential(
        &mut self,
        credential: &super::status_list::StatusListCredential,
    ) -> Result<&mut Self> {
        self.trust_status_list_credential_at(credential, &status_list::current_time()?)
    }

    fn trust_status_list_credential_at(
        &mut self,
        credential: &status_list::StatusListCredential,
        now: &status_list::StatusTime,
    ) -> Result<&mut Self> {
        let (key, bits) = status_list::verify_status_list_credential_at(credential, now)?;
        anyhow::ensure!(
            self.is_trusted(&key),
            "status list credential issuer is not trusted"
        );
        anyhow::ensure!(
            self.status_list_publishers.get(&credential.id) == Some(&key.to_bytes()),
            "status list credential issuer is not authorized for this URL"
        );
        let signed_at = super::vc::parse_datetime_stamp(&credential.valid_from)?.0;
        let valid_until = status_list::valid_until(credential)?;
        if let Some(old) = self.status_lists.get(&credential.id) {
            if let Some(old_at) = old.signed_at {
                anyhow::ensure!(
                    signed_at >= old_at,
                    "status list snapshot is older than the trusted one"
                );
                anyhow::ensure!(
                    old.bits.iter().enumerate().all(|(i, old_byte)| bits
                        .get(i)
                        .is_some_and(|new_byte| new_byte & old_byte == *old_byte)),
                    "status list cannot clear a revocation bit"
                );
            }
        }
        self.status_lists.insert(
            credential.id.clone(),
            StatusListSnapshot {
                bits,
                signed_at: Some(signed_at),
                valid_until,
            },
        );
        Ok(self)
    }

    /// Whether the status list trusted for `credential_url` revokes `index`, or `None` when no
    /// list is trusted for that URL.
    pub fn is_revoked(&self, credential_url: &str, index: u64) -> Option<bool> {
        match self.status_lookup(credential_url, index) {
            StatusLookup::Value(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn status_lookup(&self, credential_url: &str, index: u64) -> StatusLookup {
        let now = status_list::current_time().ok();
        self.status_lookup_at(credential_url, index, now.as_ref())
    }

    fn status_lookup_at(
        &self,
        credential_url: &str,
        index: u64,
        now: Option<&status_list::StatusTime>,
    ) -> StatusLookup {
        let Some(snapshot) = self.status_lists.get(credential_url) else {
            return StatusLookup::Missing;
        };
        if let Some(until) = snapshot.valid_until.as_ref() {
            if now.is_none_or(|now| now > until) {
                return StatusLookup::Stale;
            }
        }
        if let Some(max_age) = self.max_status_list_age_seconds {
            let now = now.and_then(|(seconds, _)| u64::try_from(*seconds).ok());
            let fresh = snapshot
                .signed_at
                .and_then(|signed| u64::try_from(signed).ok())
                .zip(now)
                .is_some_and(|(signed, now)| signed <= now && now - signed <= max_age);
            if !fresh {
                return StatusLookup::Stale;
            }
        }
        let Some(byte) = usize::try_from(index / 8)
            .ok()
            .and_then(|i| snapshot.bits.get(i))
        else {
            return StatusLookup::OutOfRange;
        };
        StatusLookup::Value(byte & (1 << (index % 8)) != 0)
    }

    /// Require an imported, signed status list whose `validFrom` is no older than
    /// this many seconds. The caller supplies authenticated snapshots; HMS makes no fetch.
    pub fn require_recent_status_lists(&mut self, max_age_seconds: u64) -> &mut Self {
        self.max_status_list_age_seconds = Some(max_age_seconds);
        self.require_status_lists = true;
        self
    }

    /// Makes a credential whose status list this store does not hold fail verification
    /// (`revocation_status` not passed) instead of passing with "revocation not checked".
    /// Off by default.
    pub fn require_status_lists(&mut self, required: bool) -> &mut Self {
        self.require_status_lists = required;
        self
    }

    /// Whether [`Self::require_status_lists`] is on.
    pub fn requires_status_lists(&self) -> bool {
        self.require_status_lists
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
        SigningKey::from_bytes(&rand::random())
    }

    #[test]
    fn status_list_import_requires_authorized_url_publisher() {
        let publisher = key();
        let other = key();
        let url = "https://example.com/status/1";
        let mut trust = TrustStore::new();
        assert!(trust
            .authorize_status_list_publisher(url, &publisher.verifying_key())
            .is_err());
        trust.trust_key(&publisher.verifying_key());
        trust.trust_key(&other.verifying_key());
        trust
            .authorize_status_list_publisher(url, &publisher.verifying_key())
            .unwrap();
        let wrong =
            super::super::status_list::create_status_list_credential(&other, url, &[], 0).unwrap();
        assert!(trust.trust_status_list_credential(&wrong).is_err());
        assert!(trust.is_revoked(url, 0).is_none());
        let right =
            super::super::status_list::create_status_list_credential(&publisher, url, &[1], 0)
                .unwrap();
        trust.trust_status_list_credential(&right).unwrap();
        assert_eq!(trust.is_revoked(url, 0), Some(true));
    }

    #[test]
    fn cached_signed_status_list_expires_at_each_lookup() {
        let publisher = key();
        let url = "https://example.com/status/1";
        let mut credential =
            status_list::create_status_list_credential(&publisher, url, &[1], 1).unwrap();
        credential.valid_from = "2020-01-01T00:00:00Z".into();
        credential.extra.insert(
            "validUntil".into(),
            serde_json::json!("2020-01-01T01:00:01.000000001+01:00"),
        );
        credential.proof = None;
        credential.proof = Some(super::super::vc::sign_document(&publisher, &credential).unwrap());
        let imported_at = super::super::vc::parse_datetime_stamp("2020-01-01T00:00:00Z").unwrap();

        for strict in [false, true] {
            let mut trust = TrustStore::trusting_key(&publisher.verifying_key());
            trust.require_status_lists(strict);
            trust
                .authorize_status_list_publisher(url, &publisher.verifying_key())
                .unwrap();
            trust
                .trust_status_list_credential_at(&credential, &imported_at)
                .unwrap();
            for max_age in [None, Some(60)] {
                trust.max_status_list_age_seconds = max_age;
                for (instant, accepted) in [
                    ("2020-01-01T00:00:01Z", true),
                    ("2020-01-01T00:00:01.000000001Z", true),
                    ("2020-01-01T00:00:01.000000002Z", false),
                ] {
                    let now = super::super::vc::parse_datetime_stamp(instant).unwrap();
                    for index in [0, 1] {
                        let result = trust.status_lookup_at(url, index, Some(&now));
                        if accepted {
                            assert!(
                                matches!(result, StatusLookup::Value(value) if value == (index == 0))
                            );
                        } else {
                            assert!(matches!(result, StatusLookup::Stale));
                        }
                    }
                }
            }
            assert!(matches!(
                trust.status_lookup_at(url, 0, None),
                StatusLookup::Stale
            ));
        }
    }

    #[test]
    fn missing_signed_expiry_keeps_raw_and_signed_status_semantics() {
        let publisher = key();
        let url = "https://example.com/status/1";
        let mut credential =
            status_list::create_status_list_credential(&publisher, url, &[], 0).unwrap();
        credential.valid_from = "2020-01-01T00:00:00Z".into();
        credential.proof = None;
        credential.proof = Some(super::super::vc::sign_document(&publisher, &credential).unwrap());
        let imported_at = super::super::vc::parse_datetime_stamp("2020-01-01T00:00:00Z").unwrap();
        let later = super::super::vc::parse_datetime_stamp("2020-01-01T00:00:02Z").unwrap();
        let mut trust = TrustStore::trusting_key(&publisher.verifying_key());
        trust
            .authorize_status_list_publisher(url, &publisher.verifying_key())
            .unwrap();
        trust
            .trust_status_list_credential_at(&credential, &imported_at)
            .unwrap();
        assert!(matches!(
            trust.status_lookup_at(url, 0, Some(&later)),
            StatusLookup::Value(false)
        ));
        trust.require_recent_status_lists(1);
        assert!(matches!(
            trust.status_lookup_at(url, 0, Some(&later)),
            StatusLookup::Stale
        ));
        trust.trust_status_list(url, vec![1]);
        trust.max_status_list_age_seconds = None;
        assert!(matches!(
            trust.status_lookup_at(url, 0, None),
            StatusLookup::Value(true)
        ));
        trust.require_recent_status_lists(1);
        assert!(matches!(
            trust.status_lookup_at(url, 0, Some(&later)),
            StatusLookup::Stale
        ));
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
