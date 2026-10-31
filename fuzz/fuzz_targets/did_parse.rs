// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! DID strings and DID documents arrive from signed statements and remote documents. Every
//! parser must reject malformed input; whatever key a DID yields must survive a round trip.

use writerslogic_warm::core::provenance::did;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    if let Ok(key) = did::ed25519_from_did_key(&text) {
        assert_eq!(
            did::ed25519_from_did_key(&did::did_key_from_ed25519(&key)).ok(),
            Some(key)
        );
    }
    if let Ok(key) = did::ed25519_from_did_jwk(&text) {
        assert_eq!(
            did::ed25519_from_did_jwk(&did::did_jwk_from_ed25519(&key)).ok(),
            Some(key)
        );
    }
    let _ = did::did_web_to_url(&text);
    if let Ok(document) = serde_json::from_slice::<serde_json::Value>(data) {
        let _ = did::ed25519_from_did_document(&document);
    }
});
