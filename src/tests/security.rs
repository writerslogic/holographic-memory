// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::audit::AuditOp;
use crate::core::engine::WarmCore;
use crate::core::security::SigningManager;

#[test]
fn test_signed_audit_entries() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::WarmConfig::default();
    config.security.signing_enabled = true;
    config.security.audit_enabled = true;

    let warm = WarmCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    let v = warm.encode_text("signed audit test");
    warm.memorize("sig_1".to_string(), v).unwrap();
    warm.delete("sig_1").unwrap();

    let entries = warm.audit_since(0).unwrap();
    assert_eq!(entries.len(), 2);

    // All entries should be signed (non-zero signature)
    for entry in &entries {
        assert_ne!(entry.signature, [0u8; 64], "Entry should be signed");
    }

    // Verify signatures using the same key
    let key_path = dir.path().join("hms_signing.key");
    let mgr = SigningManager::new(&key_path).unwrap();
    for entry in &entries {
        let mut signable = [0u8; 41];
        signable[0..8].copy_from_slice(&entry.timestamp_ms.to_le_bytes());
        signable[8] = entry.op as u8;
        signable[9..41].copy_from_slice(&entry.id_hash);
        mgr.verify(&signable, &entry.signature)
            .expect("Signature verification failed");
    }
}

#[test]
fn test_signed_compact_audit() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::core::config::WarmConfig::default();
    config.security.signing_enabled = true;
    config.security.audit_enabled = true;

    let warm = WarmCore::new(
        10_000,
        Some(dir.path().to_string_lossy().to_string()),
        Some(config),
    )
    .unwrap();

    let v = warm.encode_text("compact signed");
    warm.memorize("cs_1".to_string(), v).unwrap();
    warm.compact().unwrap();

    let entries = warm.audit_since(0).unwrap();
    let compact_entry = entries.iter().find(|e| e.op == AuditOp::Compact).unwrap();
    assert_ne!(compact_entry.signature, [0u8; 64]);

    let key_path = dir.path().join("hms_signing.key");
    let mgr = SigningManager::new(&key_path).unwrap();
    let mut signable = [0u8; 41];
    signable[0..8].copy_from_slice(&compact_entry.timestamp_ms.to_le_bytes());
    signable[8] = compact_entry.op as u8;
    signable[9..41].copy_from_slice(&compact_entry.id_hash);
    mgr.verify(&signable, &compact_entry.signature)
        .expect("Compact signature verification failed");
}

#[test]
fn test_encrypted_arena_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    {
        let mut config = crate::core::config::WarmConfig::default();
        config.security.encryption_enabled = true;
        config.security.encryption_passphrase = Some("test-passphrase-123".to_string());

        let warm = WarmCore::new(10_000, Some(path.clone()), Some(config)).unwrap();
        let v = warm.encode_text("encrypted data");
        warm.memorize("enc_1".to_string(), v).unwrap();
        let v2 = warm.encode_text("more encrypted data");
        warm.memorize("enc_2".to_string(), v2).unwrap();
        assert_eq!(warm.vector_count(), 2);
    }

    // Reopen with same passphrase — should decrypt and recover
    let mut config = crate::core::config::WarmConfig::default();
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some("test-passphrase-123".to_string());

    let warm = WarmCore::new(10_000, Some(path), Some(config)).unwrap();
    assert_eq!(warm.vector_count(), 2);

    let q = warm.encode_text("encrypted data");
    let results = warm.query(&q, 1);
    assert_eq!(results[0].id, "enc_1");
}

#[test]
fn test_encrypted_compact_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    let mut config = crate::core::config::WarmConfig::default();
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some("compact-test".to_string());

    let warm = WarmCore::new(10_000, Some(path.clone()), Some(config.clone())).unwrap();
    for i in 0..10 {
        let v = warm.encode_text(&format!("encrypted compact {}", i));
        warm.memorize(format!("ec_{}", i), v).unwrap();
    }
    for i in 0..5 {
        warm.delete(&format!("ec_{}", i)).unwrap();
    }
    warm.compact().unwrap();
    assert_eq!(warm.vector_count(), 5);

    // Reopen after compaction
    drop(warm);
    let warm = WarmCore::new(10_000, Some(path), Some(config)).unwrap();
    assert_eq!(warm.vector_count(), 5);
}
