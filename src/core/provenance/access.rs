// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Credential-gated agent admission.
//!
//! An [`AccessCredential`] is a W3C Verifiable Credential in which a trusted
//! issuer grants an agent `did:key` a set of [`Permission`]s. An
//! [`AgentRegistry`] admits an agent only after verifying the credential's
//! eddsa-jcs-2022 proof, its validity window, and that the signing key belongs
//! to an issuer the host has explicitly trusted.
//!
//! The registry decides; it does not authenticate the caller. A host exposing
//! HMS to remote agents must separately prove that the party presenting a DID
//! controls its key (for example a signed challenge) before calling
//! [`AgentRegistry::authorize`].

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, ensure, Result};
use ed25519_dalek::SigningKey;
use fxhash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::did::{did_key_from_ed25519, ed25519_from_did_key};
use super::trust::TrustStore;
use super::vc::{self, DataIntegrityProof};

pub const ACCESS_CREDENTIAL_TYPE: &str = "HmsAccessCredential";
const VC_TYPE: &str = "VerifiableCredential";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Permission {
    Read,
    Write,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccessCredential {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    pub id: String,
    pub issuer: String,
    pub valid_from: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    pub credential_subject: AccessSubject,
    pub proof: Option<DataIntegrityProof>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccessSubject {
    /// The agent's Ed25519 `did:key`.
    pub id: String,
    pub permissions: Vec<Permission>,
}

/// Issue a signed access credential for `subject_did`. `valid_until` is a UTC
/// `YYYY-MM-DDTHH:MM:SSZ` timestamp; `None` means the grant does not expire.
pub fn issue_access_credential(
    signing_key: &SigningKey,
    subject_did: &str,
    permissions: &[Permission],
    valid_until: Option<&str>,
) -> Result<AccessCredential> {
    ed25519_from_did_key(subject_did)?;
    ensure!(
        !permissions.is_empty(),
        "credential must grant a permission"
    );
    if let Some(until) = valid_until {
        vc::parse_iso8601_utc(until)?;
    }
    let nonce: [u8; 16] = rand::random();
    let id: String = nonce.iter().map(|b| format!("{b:02x}")).collect();

    let mut credential = AccessCredential {
        context: vec![
            "https://www.w3.org/ns/credentials/v2".to_string(),
            "https://writerslogic.com/ns/hms/v1".to_string(),
        ],
        credential_type: vec![VC_TYPE.to_string(), ACCESS_CREDENTIAL_TYPE.to_string()],
        id: format!("urn:hms:access:{id}"),
        issuer: did_key_from_ed25519(&signing_key.verifying_key().to_bytes()),
        valid_from: vc::chrono_iso8601_now(),
        valid_until: valid_until.map(str::to_string),
        credential_subject: AccessSubject {
            id: subject_did.to_string(),
            permissions: permissions.to_vec(),
        },
        proof: None,
    };
    credential.proof = Some(vc::sign_document(signing_key, &credential)?);
    Ok(credential)
}

struct Grant {
    permissions: Vec<Permission>,
    expires_at: Option<u64>,
}

/// Agents admitted by credentials from trusted issuers. Trusts no issuer, and
/// therefore admits no agent, until [`AgentRegistry::trust_issuer`] is called.
#[derive(Default)]
pub struct AgentRegistry {
    issuers: TrustStore,
    grants: FxHashMap<String, Grant>,
}

impl AgentRegistry {
    pub fn new(issuers: TrustStore) -> Self {
        Self {
            issuers,
            grants: FxHashMap::default(),
        }
    }

    /// Trust credentials signed by the holder of this Ed25519 `did:key`.
    pub fn trust_issuer(&mut self, issuer_did: &str) -> Result<()> {
        self.issuers.trust_did(issuer_did).map(|_| ())
    }

    /// Verify a credential and record its grant, replacing any earlier grant
    /// for the same agent.
    pub fn admit(&mut self, credential: &AccessCredential) -> Result<()> {
        self.admit_at(credential, unix_now())
    }

    pub fn admit_at(&mut self, credential: &AccessCredential, now: u64) -> Result<()> {
        let proof = credential
            .proof
            .as_ref()
            .ok_or_else(|| anyhow!("credential has no proof"))?;
        ensure!(
            [VC_TYPE, ACCESS_CREDENTIAL_TYPE]
                .iter()
                .all(|t| credential.credential_type.iter().any(|c| c == t)),
            "credential is not an {ACCESS_CREDENTIAL_TYPE}"
        );
        ensure!(
            !credential.credential_subject.permissions.is_empty(),
            "credential grants no permission"
        );
        ed25519_from_did_key(&credential.credential_subject.id)?;

        let mut unsigned = credential.clone();
        unsigned.proof = None;
        let issuer_key = vc::verify_document(&unsigned, &credential.issuer, proof)?;
        ensure!(
            self.issuers.is_trusted(&issuer_key),
            "credential issuer is not trusted"
        );

        ensure!(
            vc::parse_iso8601_utc(&credential.valid_from)? <= now,
            "credential is not yet valid"
        );
        let expires_at = credential
            .valid_until
            .as_deref()
            .map(vc::parse_iso8601_utc)
            .transpose()?;
        ensure!(expires_at.is_none_or(|t| now < t), "credential has expired");

        self.grants.insert(
            credential.credential_subject.id.clone(),
            Grant {
                permissions: credential.credential_subject.permissions.clone(),
                expires_at,
            },
        );
        Ok(())
    }

    /// Remove an agent's grant. Returns whether one existed.
    pub fn revoke(&mut self, agent_did: &str) -> bool {
        self.grants.remove(agent_did).is_some()
    }

    /// Check that `agent_did` currently holds `permission`.
    pub fn authorize(&self, agent_did: &str, permission: Permission) -> Result<()> {
        self.authorize_at(agent_did, permission, unix_now())
    }

    pub fn authorize_at(&self, agent_did: &str, permission: Permission, now: u64) -> Result<()> {
        let grant = self
            .grants
            .get(agent_did)
            .ok_or_else(|| anyhow!("agent has no admitted credential"))?;
        ensure!(
            grant.expires_at.is_none_or(|t| now < t),
            "agent credential has expired"
        );
        ensure!(
            grant.permissions.contains(&permission),
            "agent credential does not grant {permission:?}"
        );
        Ok(())
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> SigningKey {
        SigningKey::from_bytes(&rand::random())
    }

    fn did(key: &SigningKey) -> String {
        did_key_from_ed25519(&key.verifying_key().to_bytes())
    }

    fn registry_trusting(issuer: &SigningKey) -> AgentRegistry {
        let mut registry = AgentRegistry::default();
        registry.trust_issuer(&did(issuer)).unwrap();
        registry
    }

    #[test]
    fn trusted_credential_grants_only_listed_permissions() {
        let (issuer, agent) = (key(), did(&key()));
        let mut registry = registry_trusting(&issuer);
        let credential =
            issue_access_credential(&issuer, &agent, &[Permission::Read], None).unwrap();

        assert!(registry.authorize(&agent, Permission::Read).is_err());
        registry.admit(&credential).unwrap();
        registry.authorize(&agent, Permission::Read).unwrap();
        assert!(registry.authorize(&agent, Permission::Write).is_err());

        assert!(registry.revoke(&agent));
        assert!(registry.authorize(&agent, Permission::Read).is_err());
    }

    #[test]
    fn untrusted_or_self_issued_credential_is_rejected() {
        let (issuer, attacker) = (key(), key());
        let mut registry = registry_trusting(&issuer);
        let self_issued =
            issue_access_credential(&attacker, &did(&attacker), &[Permission::Read], None).unwrap();
        assert!(registry.admit(&self_issued).is_err());
        assert!(AgentRegistry::default().admit(&self_issued).is_err());
    }

    #[test]
    fn tampered_or_unsigned_credential_is_rejected() {
        let (issuer, agent) = (key(), did(&key()));
        let mut registry = registry_trusting(&issuer);
        let credential =
            issue_access_credential(&issuer, &agent, &[Permission::Read], None).unwrap();

        let mut escalated = credential.clone();
        escalated
            .credential_subject
            .permissions
            .push(Permission::Write);
        assert!(registry.admit(&escalated).is_err());

        let mut retargeted = credential.clone();
        retargeted.credential_subject.id = did(&key());
        assert!(registry.admit(&retargeted).is_err());

        // Re-signed by an attacker while still naming the trusted issuer.
        let mut forged = escalated.clone();
        forged.proof = None;
        forged.proof = Some(vc::sign_document(&key(), &forged).unwrap());
        assert!(registry.admit(&forged).is_err());

        let mut unsigned = credential.clone();
        unsigned.proof = None;
        assert!(registry.admit(&unsigned).is_err());

        let mut garbage = credential;
        garbage.proof.as_mut().unwrap().proof_value = "z1111".to_string();
        assert!(registry.admit(&garbage).is_err());
    }

    #[test]
    fn validity_window_is_enforced_at_its_boundaries() {
        let (issuer, agent) = (key(), did(&key()));
        let mut registry = registry_trusting(&issuer);
        let until = "2100-01-01T00:00:00Z";
        let expiry = vc::parse_iso8601_utc(until).unwrap();
        let credential =
            issue_access_credential(&issuer, &agent, &[Permission::Read], Some(until)).unwrap();
        let issued = vc::parse_iso8601_utc(&credential.valid_from).unwrap();

        assert!(registry.admit_at(&credential, issued - 1).is_err());
        assert!(registry.admit_at(&credential, expiry).is_err());
        registry.admit_at(&credential, issued).unwrap();

        registry
            .authorize_at(&agent, Permission::Read, expiry - 1)
            .unwrap();
        assert!(registry
            .authorize_at(&agent, Permission::Read, expiry)
            .is_err());
        assert!(registry
            .authorize_at(&agent, Permission::Read, expiry + 1)
            .is_err());
    }

    #[test]
    fn timestamps_round_trip_and_reject_malformed_input() {
        assert_eq!(vc::parse_iso8601_utc("1970-01-01T00:00:00Z").unwrap(), 0);
        assert_eq!(
            vc::parse_iso8601_utc("2024-02-29T12:30:15Z").unwrap(),
            1_709_209_815
        );
        for bad in [
            "2023-02-29T00:00:00Z",
            "2024-13-01T00:00:00Z",
            "2024-01-01T24:00:00Z",
            "2024-01-01 00:00:00Z",
            "2024-01-01T00:00:00+00:00",
            "+024-01-01T00:00:00Z",
            "",
        ] {
            assert!(vc::parse_iso8601_utc(bad).is_err(), "{bad}");
        }
    }
}
