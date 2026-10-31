// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "security", unix))]

use anyhow::Result;
use writerslogic_warm::core::WarmConfig;
use writerslogic_warm::{CommittedMutationError, EntangledHVec, WarmCore};

#[test]
fn legacy_failed_audit_sync_does_not_hide_durable_create_or_delete() -> Result<()> {
    let dir = tempfile::tempdir()?;
    std::os::unix::fs::symlink("/dev/null", dir.path().join("audit.bin"))?;
    let mut config = WarmConfig::default();
    config.security.audit_enabled = true;
    let path = dir.path().display().to_string();
    drop(WarmCore::new(
        1024,
        Some(path.clone()),
        Some(config.clone()),
    )?);
    let schema_path = dir.path().join("store.json");
    let mut schema: serde_json::Value = serde_json::from_slice(&std::fs::read(&schema_path)?)?;
    schema["version"] = 5.into();
    let bytes = serde_json::to_vec_pretty(&schema)?;
    std::fs::write(schema_path, &bytes)?;
    std::fs::write(
        dir.path().join("store.json.crc32"),
        format!("{:08x}\n", crc32fast::hash(&bytes)),
    )?;
    let core = WarmCore::new(1024, Some(path.clone()), Some(config.clone()))?;
    let vector = EntangledHVec::new_deterministic(1024, 7);
    for id in ["deleted", "survivor"] {
        let error = core.memorize(id.into(), vector.clone()).unwrap_err();
        assert_eq!(
            error
                .downcast_ref::<CommittedMutationError>()
                .unwrap()
                .effect,
            "audit append"
        );
        assert!(core.query(&vector, 10).iter().any(|hit| hit.id == id));
    }
    let error = core.delete("deleted").unwrap_err();
    assert_eq!(
        error
            .downcast_ref::<CommittedMutationError>()
            .unwrap()
            .effect,
        "audit append"
    );
    assert!(!core
        .query(&vector, 10)
        .iter()
        .any(|hit| hit.id == "deleted"));
    drop(core);
    std::fs::remove_file(dir.path().join("audit.bin"))?;
    let core = WarmCore::new(1024, Some(path), Some(config))?;
    let hits = core.query(&vector, 10);
    assert!(hits.iter().any(|hit| hit.id == "survivor"));
    assert!(!hits.iter().any(|hit| hit.id == "deleted"));
    assert!(core.audit_since(0)?.is_empty());
    core.memorize("next-operation".into(), vector)?;
    assert_eq!(core.audit_since(0)?.len(), 1);
    Ok(())
}
