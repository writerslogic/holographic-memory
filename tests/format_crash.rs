// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Crash-consistency claims of `docs/file-format/CRASH-CONSISTENCY.md`, tested on committed
//! golden stores.

mod format_common;

use std::path::Path;

use anyhow::{ensure, Result};
use format_common::*;
use writerslogic_warm::core::format::{scan, verify_path, Verdict, VerifyOptions};
use writerslogic_warm::WarmCore;

fn unpacked(name: &str) -> Result<(tempfile::TempDir, std::path::PathBuf, Golden)> {
    let golden = goldens()?
        .into_iter()
        .find(|g| g.name == name)
        .expect("golden");
    let scratch = tempfile::tempdir()?;
    let store = scratch.path().join("store");
    golden.unpack(&store)?;
    Ok((scratch, store, golden))
}

fn open(store: &Path, golden: &Golden) -> Result<WarmCore> {
    WarmCore::new(
        golden.dimensions(),
        Some(store.display().to_string()),
        Some(golden.config()?),
    )
}

/// A crash can stop an append at any byte. For every byte of the last frame and the marker
/// after it, with the rest of the file zero or garbage: the store verifies as ok or
/// recoverable, opens, holds exactly the transactions before the torn one, and accepts a new
/// append that survives a reopen.
#[test]
fn torn_append_at_every_byte() -> Result<()> {
    let (_keep, pristine, golden) = unpacked("v4-plain")?;
    let segment = pristine.join("vectors_data.bin/seg_0.bin");
    let bytes = std::fs::read(&segment)?;
    let mut frames = Vec::new();
    let end = scan::scan_segment(&bytes, true, |f, _| frames.push(*f)).end_offset;
    let last = *frames.last().unwrap();
    assert_eq!(last.end(), end);
    // The last frame deletes "doc-obsolete".
    let mut cuts = 0;
    for garbage in [false, true] {
        for cut in (last.offset..=end + 16).filter(|c| !garbage || c % 3 == 0) {
            let scratch = tempfile::tempdir()?;
            let store = scratch.path().join("store");
            golden.unpack(&store)?;
            let path = store.join("vectors_data.bin/seg_0.bin");
            let mut torn = bytes.clone();
            torn[cut..end + 16].fill(0);
            if garbage && cut < end {
                let stop = (cut + 24).min(end + 16);
                torn[cut..stop].fill(0xA5);
            }
            std::fs::write(&path, &torn)?;
            let report = verify_path(&store, &VerifyOptions::default())?;
            let complete = cut >= end;
            ensure!(
                matches!(report.verdict, Verdict::Ok | Verdict::Recoverable),
                "cut {cut}: {:?}",
                report.findings
            );
            if !complete && (cut != last.offset || garbage) {
                ensure!(
                    report.verdict == Verdict::Recoverable,
                    "cut {cut}: {:?}",
                    report.verdict
                );
            }
            {
                let core = open(&store, &golden)?;
                // The delete is present only if its frame was complete.
                ensure!(core.has_document("doc-obsolete") != complete, "cut {cut}");
                core.memorize_document(serde_json::from_value(serde_json::json!({
                    "id": "after-crash", "text": "appended after recovery", "version": "1",
                    "metadata": {}
                }))?)?;
            }
            let core = open(&store, &golden)?;
            ensure!(
                core.has_document("after-crash") && core.has_document("doc-harbor"),
                "cut {cut}"
            );
            drop(core);
            let after = verify_path(&store, &VerifyOptions::default())?;
            ensure!(
                after.verdict == Verdict::Ok,
                "cut {cut}: {:?}",
                after.findings
            );
            cuts += 1;
        }
    }
    assert!(cuts > 60);
    Ok(())
}

/// A leftover `indices.json.crc32` (written by earlier releases, and by a crash between the
/// old sidecar and file writes) does not stop the store opening.
#[test]
fn stale_indices_sidecar_is_ignored() -> Result<()> {
    let (_keep, store, golden) = unpacked("v4-plain")?;
    std::fs::write(store.join("indices.json.crc32"), "deadbeef\n")?;
    let report = verify_path(&store, &VerifyOptions::default())?;
    assert_eq!(report.verdict, Verdict::Ok, "{:?}", report.findings);
    let core = open(&store, &golden)?;
    assert!(core.has_document("doc-harbor"));
    Ok(())
}

/// A damaged cache is discarded, not fatal: flipped bytes in the IVF file or an unparsable
/// `indices.json` leave an openable store with the same documents.
#[test]
fn damaged_caches_are_discarded() -> Result<()> {
    let (_keep, store, golden) = unpacked("v4-indices")?;
    let ivf = store.join("ivf_index.bin");
    let mut bytes = std::fs::read(&ivf)?;
    bytes[10] ^= 0xff;
    std::fs::write(&ivf, bytes)?;
    let report = verify_path(&store, &VerifyOptions::default())?;
    assert_eq!(
        report.verdict,
        Verdict::Recoverable,
        "{:?}",
        report.findings
    );
    assert!(open(&store, &golden)?.has_document("doc-harbor"));
    std::fs::write(store.join("indices.json"), b"{not json")?;
    let report = verify_path(&store, &VerifyOptions::default())?;
    assert_eq!(report.verdict, Verdict::Recoverable);
    assert!(open(&store, &golden)?.has_document("doc-harbor"));
    Ok(())
}

/// A partial audit entry is reported, cut on the next open, and later entries stay aligned.
#[test]
fn audit_torn_entry_is_repaired() -> Result<()> {
    let (_keep, store, golden) = unpacked("v4-audit")?;
    let audit = store.join("audit.bin");
    let before = std::fs::read(&audit)?;
    assert_eq!(before.len() % 105, 0);
    let mut torn = before.clone();
    torn.extend_from_slice(&[7u8; 50]);
    std::fs::write(&audit, &torn)?;
    let report = verify_path(&store, &VerifyOptions::default())?;
    assert_eq!(report.verdict, Verdict::Recoverable);
    assert!(report.findings.iter().any(|f| f.code == "audit-torn-tail"));
    let core = open(&store, &golden)?;
    let count = core.audit_since(0)?.len();
    core.memorize(
        "audit-new".into(),
        writerslogic_warm::EntangledHVec::new_deterministic(1024, 5),
    )?;
    assert_eq!(core.audit_since(0)?.len(), count + 1);
    drop(core);
    assert_eq!(std::fs::read(&audit)?.len() % 105, 0);
    Ok(())
}

/// The salt exists, has the specified length and leaves no temporary file behind.
#[cfg(feature = "security")]
#[test]
fn salt_is_created_whole_before_the_first_frame() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = writerslogic_warm::core::WarmConfig::default();
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some("p".into());
    let core = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config))?;
    assert_eq!(
        std::fs::metadata(dir.path().join("encryption.salt"))?.len(),
        16
    );
    assert_eq!(core.vector_count(), 0);
    let strays: Vec<_> = std::fs::read_dir(dir.path())?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp"))
        .collect();
    assert!(strays.is_empty());
    Ok(())
}
