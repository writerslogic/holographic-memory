// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! With encryption on, no file under the store directory may hold document text, and the
//! text segment must authenticate what it returns.
#![cfg(feature = "security")]

use holographic_memory::{core::HmsConfig, DocumentInput, HmsCore, SearchOptions};
use serde_json::json;
use std::path::Path;

const MARKER: &str = "zqxjmarkerplaintextcanary";

fn config(passphrase: Option<&str>) -> HmsConfig {
    let mut config = HmsConfig::default();
    config.security.encryption_enabled = passphrase.is_some();
    config.security.encryption_passphrase = passphrase.map(str::to_owned);
    config
}

fn document(id: &str, text: &str) -> DocumentInput {
    serde_json::from_value(json!({ "id": id, "text": text })).unwrap()
}

fn files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn leaks(dir: &Path) -> Vec<String> {
    let mut all = Vec::new();
    files(dir, &mut all);
    all.into_iter()
        .filter(|p| {
            std::fs::read(p)
                .unwrap()
                .windows(MARKER.len())
                .any(|w| w == MARKER.as_bytes())
        })
        .map(|p| p.display().to_string())
        .collect()
}

fn hit(core: &HmsCore) -> Option<String> {
    core.search_documents(MARKER, &SearchOptions::default())
        .unwrap()
        .into_iter()
        .find_map(|r| r.text)
}

#[test]
fn encrypted_store_never_writes_document_text_in_the_clear() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    let text = format!("alpha beta {MARKER} gamma delta");
    {
        let core = HmsCore::new(1024, Some(path.clone()), Some(config(Some("pw-one")))).unwrap();
        core.memorize_document(document("d1", &text)).unwrap();
        assert_eq!(hit(&core).as_deref(), Some(text.as_str()));
        assert!(leaks(dir.path()).is_empty(), "before compact");
        core.compact().unwrap();
        assert!(leaks(dir.path()).is_empty(), "after compact");
        assert_eq!(hit(&core).as_deref(), Some(text.as_str()));
    }
    let core = HmsCore::new(1024, Some(path.clone()), Some(config(Some("pw-one")))).unwrap();
    assert_eq!(hit(&core).as_deref(), Some(text.as_str()));
    assert!(leaks(dir.path()).is_empty(), "after reopen");
    drop(core);

    // Wrong key and no key both fail closed.
    assert!(HmsCore::new(1024, Some(path.clone()), Some(config(Some("pw-two")))).is_err());
    assert!(HmsCore::new(1024, Some(path), Some(config(None))).is_err());
}

#[test]
fn unencrypted_store_still_opens_and_searches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    let text = format!("one {MARKER} two");
    {
        let core = HmsCore::new(1024, Some(path.clone()), Some(config(None))).unwrap();
        core.memorize_document(document("d1", &text)).unwrap();
    }
    let core = HmsCore::new(1024, Some(path), Some(config(None))).unwrap();
    assert_eq!(hit(&core).as_deref(), Some(text.as_str()));
    // The detector sees plaintext where encryption is off.
    assert!(leaks(dir.path())
        .iter()
        .any(|p| p.ends_with("documents.text")));
}

#[test]
fn tampered_text_record_is_not_returned() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();
    let text = format!("one {MARKER} two");
    let core = HmsCore::new(1024, Some(path), Some(config(Some("pw-one")))).unwrap();
    core.memorize_document(document("d1", &text)).unwrap();
    let segment = dir.path().join("documents.text");
    let mut bytes = std::fs::read(&segment).unwrap();
    assert!(bytes.len() > 32);
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&segment, &bytes).unwrap();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| hit(&core)));
    assert!(
        !matches!(outcome, Ok(Some(_))),
        "tampered record must not be returned as text"
    );
}
