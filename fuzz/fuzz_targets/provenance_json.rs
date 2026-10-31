// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! JSON provenance documents from outside: W3C verifiable credentials, Sigstore bundles and
//! KERI key event logs (a file read by `KeyEventLog::with_path`). All must be rejected or
//! verified without panicking.

use std::sync::OnceLock;

use writerslogic_warm::core::provenance::keri::KeyEventLog;
use writerslogic_warm::core::provenance::sigstore::{verify_bundle, SigstoreBundle};
use writerslogic_warm::core::provenance::vc::{verify_credential, FactCredential};
use libfuzzer_sys::fuzz_target;

fn log_path() -> &'static std::path::Path {
    // One file, overwritten by every iteration.
    static DIR: OnceLock<(tempfile::TempDir, std::path::PathBuf)> = OnceLock::new();
    let (_, path) = DIR.get_or_init(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kel.json");
        (dir, path)
    });
    path
}

fuzz_target!(|data: &[u8]| {
    if let Ok(credential) = serde_json::from_slice::<FactCredential>(data) {
        let _ = verify_credential(&credential);
    }
    if let Ok(bundle) = serde_json::from_slice::<SigstoreBundle>(data) {
        let _ = verify_bundle(&bundle, b"content");
    }
    let path = log_path();
    if std::fs::write(path, data).is_ok() {
        if let Ok(log) = KeyEventLog::with_path(path) {
            let _ = log.verify();
        }
    }
});
