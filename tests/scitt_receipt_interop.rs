// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "provenance")]

use base64::Engine;
use ed25519_dalek::VerifyingKey;
use holographic_memory::core::provenance::{
    scitt::{
        receipt::{verify_receipt, ReceiptPolicy},
        SignedStatement, UnverifiedReceipt,
    },
    trust::TrustStore,
};
use serde_json::Value;

#[test]
fn independently_signed_python_receipts_require_inclusion_identity_trust_and_freshness() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/scitt_receipt_python_v1.json")).unwrap();
    let decode = |value: &Value| {
        base64::engine::general_purpose::STANDARD
            .decode(value.as_str().unwrap())
            .unwrap()
    };
    let issuer =
        VerifyingKey::from_bytes(&decode(&fixture["issuer_public"]).try_into().unwrap()).unwrap();
    let service =
        VerifyingKey::from_bytes(&decode(&fixture["service_public"]).try_into().unwrap()).unwrap();
    let issuer_trust = TrustStore::trusting_key(&issuer);
    let service_trust = TrustStore::trusting_key(&service);
    let mut policy = ReceiptPolicy {
        issuer_key: &issuer,
        issuer_trust: &issuer_trust,
        service_key: &service,
        service_trust: &service_trust,
        service_issuer: fixture["service_issuer"].as_str().unwrap(),
        subject: fixture["subject"].as_str().unwrap(),
        current_root: decode(&fixture["current_root"]).try_into().unwrap(),
        checkpoint_not_before_ms: fixture["checkpoint_not_before_ms"].as_u64().unwrap(),
        checkpoint_valid_until_ms: fixture["checkpoint_valid_until_ms"].as_u64().unwrap(),
        now_ms: fixture["now_ms"].as_u64().unwrap(),
    };
    let statement = SignedStatement {
        cose_bytes: decode(&fixture["statement"]),
    };
    for vector in fixture["vectors"].as_array().unwrap() {
        let receipt = UnverifiedReceipt {
            entry_id: "independent".into(),
            log_id: None,
            receipt_bytes: decode(&vector["receipt"]),
        };
        let result = verify_receipt(&statement, &receipt, &policy);
        assert_eq!(
            result.is_ok(),
            vector["valid"].as_bool().unwrap(),
            "{}: {result:?}",
            vector["name"]
        );
        if let Ok(verified) = result {
            assert_eq!(verified.claim().fact_id, policy.subject);
            assert_eq!(
                verified.claim().source_uri.as_deref(),
                Some("fixture:transcript/independent")
            );
            assert_eq!(verified.root(), &policy.current_root);
        }
    }
    let receipt = UnverifiedReceipt {
        entry_id: "independent".into(),
        log_id: None,
        receipt_bytes: decode(&fixture["vectors"][0]["receipt"]),
    };
    let empty = TrustStore::new();
    policy.issuer_trust = &empty;
    assert!(verify_receipt(&statement, &receipt, &policy).is_err());
    policy.issuer_trust = &issuer_trust;
    policy.service_trust = &empty;
    assert!(verify_receipt(&statement, &receipt, &policy).is_err());
    policy.service_trust = &service_trust;
    policy.current_root[0] ^= 1;
    assert!(verify_receipt(&statement, &receipt, &policy).is_err());
    policy.current_root[0] ^= 1;
    for now in [
        policy.checkpoint_not_before_ms - 1,
        policy.checkpoint_valid_until_ms,
        policy.checkpoint_valid_until_ms + 1,
    ] {
        policy.now_ms = now;
        assert!(verify_receipt(&statement, &receipt, &policy).is_err());
    }
}
