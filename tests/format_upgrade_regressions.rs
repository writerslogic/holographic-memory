// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

mod format_common;
use format_common::goldens;
use holographic_memory::core::format::{upgrade_store, verify_path, Verdict, VerifyOptions};

#[test]
fn mixed_version_upgrade_keeps_current_tenant_arena() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let source = root.path().join("source");
    let output = root.path().join("output");
    let fixtures = goldens()?;
    fixtures
        .iter()
        .find(|g| g.name == "v3-fixed64")
        .unwrap()
        .unpack(&source.join("tenants/6f6c64"))?;
    fixtures
        .iter()
        .find(|g| g.name == "v4-plain")
        .unwrap()
        .unpack(&source.join("tenants/6e6577"))?;
    assert_eq!(
        verify_path(&source, &VerifyOptions::default())?.verdict,
        Verdict::Ok
    );
    let report = upgrade_store(&source, &output, false)?;
    assert!(
        output.join("tenants/6e6577/vectors_data.bin").is_dir(),
        "current tenant's journal was omitted; output verdict {:?}",
        report.verification.map(|r| r.verdict)
    );
    Ok(())
}

#[test]
fn scanner_rejects_incomplete_stored_triple() {
    let input = br#"[{"StoredTriple":{}}]"#;
    let mut replay = holographic_memory::core::format::scan::Replay::default();
    assert!(holographic_memory::core::format::scan::check_transaction(
        input,
        Some(256),
        &mut replay
    )
    .is_err());
}

#[test]
fn nul_prefixed_document_survives_compaction() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().display().to_string();
    let core = holographic_memory::HmsCore::new(256, Some(path.clone()), None)?;
    let input = serde_json::from_value(serde_json::json!({
        "id": "live", "text": format!("{}live searchable text", "\0".repeat(16))
    }))?;
    core.memorize_documents(vec![input])?;
    assert!(core.has_document("live"));
    core.compact()?;
    drop(core);
    let core = holographic_memory::HmsCore::new(256, Some(path), None)?;
    assert!(
        core.has_document("live"),
        "unscrubbed live document was discarded during compaction"
    );
    Ok(())
}

#[test]
fn verify_agrees_with_replay_on_missing_fields() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let fixtures = goldens()?;
    let fixture = fixtures.iter().find(|g| g.name == "v4-plain").unwrap();
    let store = root.path().join("store");
    fixture.unpack(&store)?;
    let segment = store.join("vectors_data.bin/seg_0.bin");
    let len = std::fs::metadata(&segment)?.len();
    let payload = b"HMS-TXN-2\0[{\"StoredTriple\":{}}]";
    let mut bytes = crc32fast::hash(payload).to_le_bytes().to_vec();
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&[0; 12]);
    bytes.extend_from_slice(&0x454e444cu32.to_le_bytes());
    std::fs::write(&segment, bytes)?;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&segment)?
        .set_len(len)?;
    let report = verify_path(&store, &VerifyOptions::default())?;
    let opened = holographic_memory::HmsCore::new(
        fixture.dimensions(),
        Some(store.display().to_string()),
        Some(fixture.config()?),
    );
    assert!(opened.is_err());
    assert_ne!(
        report.verdict,
        Verdict::Ok,
        "verifier accepts a transaction the engine cannot deserialize"
    );
    Ok(())
}

#[test]
fn recoverable_current_tenant_does_not_block_data_dir_upgrade() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let source = root.path().join("source");
    let output = root.path().join("output");
    let fixtures = goldens()?;
    fixtures
        .iter()
        .find(|g| g.name == "v3-fixed64")
        .unwrap()
        .unpack(&source.join("tenants/6f6c64"))?;
    let current = fixtures.iter().find(|g| g.name == "v4-plain").unwrap();
    current.unpack(&source.join("tenants/6e6577"))?;
    let segment = source.join("tenants/6e6577/vectors_data.bin/seg_0.bin");
    let mut bytes = std::fs::read(&segment)?;
    let mark = 0x454e444cu32.to_le_bytes();
    let at = bytes
        .windows(16)
        .rposition(|header| header[..12].iter().all(|&b| b == 0) && header[12..] == mark)
        .expect("end marker");
    bytes[at..at + 16].fill(0xAB);
    std::fs::write(&segment, &bytes)?;
    let before = verify_path(&source, &VerifyOptions::default())?;
    assert_eq!(before.verdict, Verdict::Recoverable);
    let report = upgrade_store(&source, &output, false)?;
    let verification = report.verification.unwrap();
    assert_eq!(verification.verdict, Verdict::Recoverable);
    assert_eq!(verification.frames, before.frames);
    let tenant = output.join("tenants/6e6577");
    assert!(tenant.join("vectors_data.bin/seg_0.bin").is_file());
    let core = holographic_memory::HmsCore::new(
        current.dimensions(),
        Some(tenant.display().to_string()),
        Some(current.config()?),
    )?;
    drop(core);
    assert_eq!(
        verify_path(&output, &VerifyOptions::default())?.verdict,
        Verdict::Ok
    );
    assert_eq!(std::fs::read(segment)?, bytes);
    Ok(())
}
