// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Reopening an encrypted store with the wrong passphrase names the passphrase.
#![cfg(feature = "security")]

use serde_json::json;
use writerslogic_warm::{core::WarmConfig, DocumentInput, WarmCore};

fn config(passphrase: &str) -> WarmConfig {
    let mut config = WarmConfig::default();
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some(passphrase.to_owned());
    config
}

#[test]
fn wrong_passphrase_error_mentions_the_passphrase() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    {
        let core = WarmCore::new(1024, Some(path.clone()), Some(config("right one"))).unwrap();
        let doc: DocumentInput =
            serde_json::from_value(json!({ "id": "d1", "text": "alpha beta" })).unwrap();
        core.memorize_document(doc).unwrap();
    }

    let error = match WarmCore::new(1024, Some(path.clone()), Some(config("wrong passphrase"))) {
        Ok(_) => panic!("opening with the wrong passphrase must fail"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("passphrase"), "{error}");
    assert!(!error.contains("right one"), "{error}");

    // the right passphrase still opens it
    WarmCore::new(1024, Some(path), Some(config("right one"))).unwrap();
}
