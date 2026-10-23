// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! A purge is physically complete when it returns: no deleted text remains in any file under
//! the store directory, including after an abrupt process death and a reopen.

use std::path::Path;
use std::sync::Arc;

use holographic_memory::core::memory_api::*;
use holographic_memory::HmsCore;

const DIM: u32 = 4096;
const MARKERS: [&str; 3] = ["zyxquartzmarker", "wvutfeldsparmarker", "obsidianmarkerqq"];

fn memory_at(path: &Path) -> Memory {
    let core = Arc::new(HmsCore::new(DIM, Some(path.display().to_string()), None).unwrap());
    Memory::new(core).with_clock(Arc::new(ManualClock::new(1)))
}

fn files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Every file under `dir` containing `needle` or its first ten bytes (a posting dictionary
/// or delta list that stored terms or prefixes in the clear would match the fragment).
fn leaks(dir: &Path, needle: &str) -> Vec<String> {
    let mut all = Vec::new();
    files(dir, &mut all);
    let fragment = &needle.as_bytes()[..needle.len().min(10)];
    all.into_iter()
        .filter(|path| {
            std::fs::read(path)
                .map(|bytes| bytes.windows(fragment.len()).any(|w| w == fragment))
                .unwrap_or(false)
        })
        .map(|path| path.display().to_string())
        .collect()
}

fn remember(memory: &Memory, id: &str, text: &str) {
    memory
        .remember(RememberInput {
            id: Some(id.into()),
            text: text.into(),
            ..RememberInput::default()
        })
        .unwrap();
}

fn assert_clean(dir: &Path, when: &str) {
    for marker in MARKERS {
        let found = leaks(dir, marker);
        assert!(found.is_empty(), "{marker} present {when}: {found:?}");
    }
}

fn purge(memory: &Memory, ids: &[&str]) {
    let report = memory
        .forget(
            &ForgetScope::Ids {
                ids: ids.iter().map(|id| id.to_string()).collect(),
            },
            &ForgetOptions {
                purge: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(report.purged);
}

fn seed(memory: &Memory) {
    for (i, marker) in MARKERS.iter().enumerate() {
        remember(memory, &format!("d{i}"), &format!("secret note {marker}"));
    }
    remember(memory, "keep", "an unrelated survivor basalt");
}

#[test]
fn purge_leaves_no_marker_in_any_file_before_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let memory = memory_at(dir.path());
    seed(&memory);
    assert!(
        !leaks(dir.path(), MARKERS[0]).is_empty(),
        "the fixture must write the marker"
    );
    purge(&memory, &["d0", "d1", "d2"]);
    assert_clean(dir.path(), "after purge, before reopen");
    assert!(!leaks(dir.path(), "survivor basalt").is_empty());
    drop(memory);
    assert_clean(dir.path(), "after reopen");
}

#[test]
fn forget_without_purge_scrubs_the_text_segment_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let memory = memory_at(dir.path());
    seed(&memory);
    memory
        .forget(
            &ForgetScope::Ids {
                ids: vec!["d0".into()],
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    // The append-only journal keeps the frame until a purge; the text segment does not.
    let bytes = std::fs::read(dir.path().join("documents.text")).unwrap();
    let has = |needle: &str| bytes.windows(needle.len()).any(|w| w == needle.as_bytes());
    assert!(!has(MARKERS[0]));
    assert!(has(MARKERS[1]), "other documents stay readable");
    assert!(memory.core().has_document("d1"));
}

/// Child half of the crash test: ingests, purges, dies without unwinding.
#[test]
fn purge_child() {
    let Ok(path) = std::env::var("HMS_PURGE_CRASH_DIR") else {
        return;
    };
    let memory = memory_at(Path::new(&path));
    seed(&memory);
    purge(&memory, &["d0", "d1", "d2"]);
    println!("acked purge");
    std::process::abort();
}

#[test]
fn purged_text_stays_gone_after_an_abrupt_death_and_restart() {
    let dir = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "purge_child", "--nocapture", "--test-threads=1"])
        .env("HMS_PURGE_CRASH_DIR", dir.path())
        .output()
        .unwrap();
    assert!(!output.status.success(), "the child must die by abort");
    assert!(String::from_utf8_lossy(&output.stdout).contains("acked purge"));
    assert_clean(dir.path(), "after the crash");
    let memory = memory_at(dir.path());
    assert!(!memory.core().has_document("d0"));
    assert!(memory.core().has_document("keep"));
    drop(memory);
    assert_clean(dir.path(), "after restart");
}

#[test]
fn purge_erases_a_generation_left_by_an_interrupted_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let memory = memory_at(dir.path());
    seed(&memory);
    // A compaction that died after staging leaves a generation nobody references.
    let orphan = dir.path().join("vectors_data.bin").join("gen-1-1");
    std::fs::create_dir(&orphan).unwrap();
    std::fs::write(orphan.join("seg_0.bin"), format!("frame {}", MARKERS[2])).unwrap();
    assert!(!leaks(dir.path(), MARKERS[2]).is_empty());
    purge(&memory, &["d0", "d1", "d2"]);
    assert!(!orphan.exists());
    assert_clean(dir.path(), "after purge with an orphaned generation");
}

#[test]
fn replaced_text_is_zeroed_in_the_segment_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let core = HmsCore::new(DIM, Some(dir.path().display().to_string()), None).unwrap();
    let put = |id: &str, version: &str, text: &str| {
        let input = serde_json::from_value(serde_json::json!({
            "id": id, "version": version, "text": text,
        }))
        .unwrap();
        core.memorize_document(input).unwrap();
    };
    let in_segment = |needle: &str| {
        let bytes = std::fs::read(dir.path().join("documents.text")).unwrap();
        bytes.windows(needle.len()).any(|w| w == needle.as_bytes())
    };
    put("a", "1", &format!("first draft {}", MARKERS[0]));
    // Same chunk count, then a different one.
    put("a", "2", "second draft with other words");
    assert!(!in_segment(MARKERS[0]), "text survived a same-shape update");
    assert!(in_segment("second draft"));
    put("b", "1", &format!("draft {}", MARKERS[1]));
    let long = vec!["filler"; 400].join(" ");
    put("b", "2", &long);
    assert!(!in_segment(MARKERS[1]), "text survived a reshaping update");
    assert!(in_segment("filler filler"));
    core.compact_scrubbing().unwrap();
    assert_clean(dir.path(), "after the updates are purged");
}

#[cfg(feature = "security")]
#[test]
fn encrypted_store_purges_and_keeps_survivors_readable() {
    use holographic_memory::core::HmsConfig;
    let dir = tempfile::tempdir().unwrap();
    let mut config = HmsConfig::default();
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some("purge-fixture".into());
    let open = |config: &HmsConfig| {
        let core = HmsCore::new(
            DIM,
            Some(dir.path().display().to_string()),
            Some(config.clone()),
        )
        .unwrap();
        Memory::new(Arc::new(core)).with_clock(Arc::new(ManualClock::new(1)))
    };
    let memory = open(&config);
    seed(&memory);
    purge(&memory, &["d0", "d1", "d2"]);
    assert_clean(dir.path(), "after an encrypted purge");
    assert!(!memory.core().has_document("d0"));
    drop(memory);
    let memory = open(&config);
    assert!(memory.core().has_document("keep"));
    assert_clean(dir.path(), "after an encrypted restart");
}

/// A store served from its journal checkpoint (`documents.ckpt`) forgets through the memory API
/// like one replaying its journal: a plain forget hides the document at once and a purge leaves
/// no marker in any file, the old checkpoint included, while the survivor stays searchable.
/// Checkpoints hold lexical-only documents, so the fixture stores those through the engine.
#[test]
fn forget_and_purge_work_on_a_store_served_from_its_checkpoint() -> anyhow::Result<()> {
    use holographic_memory::core::HmsConfig;
    let dir = tempfile::tempdir().unwrap();
    let mut config = HmsConfig::default();
    config.document_checkpoint.min_journal_bytes = 0;
    let open = || {
        let core = HmsCore::new(
            DIM,
            Some(dir.path().display().to_string()),
            Some(config.clone()),
        )
        .unwrap();
        Memory::new(Arc::new(core)).with_clock(Arc::new(ManualClock::new(1)))
    };
    {
        let memory = open();
        let lexical = holographic_memory::core::documents::IngestOptions {
            encode: holographic_memory::core::documents::EncodeMode::Lexical,
            ..Default::default()
        };
        let mut texts: Vec<(String, String)> = MARKERS
            .iter()
            .enumerate()
            .map(|(i, marker)| (format!("d{i}"), format!("secret note {marker}")))
            .collect();
        texts.push(("keep".into(), "an unrelated survivor basalt".into()));
        for (id, text) in texts {
            memory.core().memorize_document_with(
                serde_json::from_value(serde_json::json!({"id": id, "text": text}))?,
                &lexical,
            )?;
        }
    }
    let mut all = Vec::new();
    files(dir.path(), &mut all);
    assert!(
        all.iter().any(|path| path.ends_with("documents.ckpt")),
        "closing writes a checkpoint: {all:?}"
    );

    let memory = open();
    assert!(memory.core().serves_from_checkpoint());
    memory
        .forget(
            &ForgetScope::Ids {
                ids: vec!["d0".into()],
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    assert!(!memory.core().has_document("d0"));
    assert!(memory.core().has_document("d1"));
    // The purge's scrubbing compaction also drops the forgotten d0 from the checkpoint.
    purge(&memory, &["d1", "d2"]);
    assert_clean(dir.path(), "after purging a checkpoint-served store");
    assert!(memory.core().has_document("keep"));
    drop(memory);
    let memory = open();
    assert_clean(dir.path(), "after reopening");
    assert!(!memory.core().has_document("d0"));
    let hits = memory
        .core()
        .search_documents("survivor basalt", &Default::default())
        .unwrap();
    assert_eq!(
        hits.first().map(|hit| hit.document_id.as_str()),
        Some("keep")
    );
    Ok(())
}
