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
