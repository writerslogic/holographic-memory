// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Persisted wire formats: manifests with their CRC sidecars, the append-only arena frames,
//! the document log and the indices. A valid store is built once; every input corrupts one
//! of its files (overwrite, truncate, append or replace) and the store must then verify and
//! open to a clean error or a working store, never panic, hang or allocate from a length
//! read off disk.
//!
//! Input layout: file selector (1 byte), mode (1 byte), offset (4 bytes LE), payload.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use holographic_memory::core::admin::verify_store;
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use libfuzzer_sys::fuzz_target;

const DIMENSIONS: u32 = 256;

struct Template {
    _dir: tempfile::TempDir,
    root: PathBuf,
    files: Vec<PathBuf>,
}

fn files_under(root: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(root)
        .expect("read template")
        .map(|e| e.expect("entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            files_under(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create");
    for entry in std::fs::read_dir(from).expect("read") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).expect("copy");
        }
    }
}

fn template() -> &'static Template {
    static TEMPLATE: OnceLock<Template> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("store");
        {
            let core =
                HmsCore::new(DIMENSIONS, Some(root.display().to_string()), None).expect("create");
            for (id, text) in [
                ("backups", "Restore backups from verified snapshots."),
                ("rivers", "Rivers carry water to the sea across the plain."),
                (
                    "gone",
                    "A document that is deleted before the store closes.",
                ),
            ] {
                let input: DocumentInput = serde_json::from_value(serde_json::json!({
                    "id": id, "text": text, "sourceUri": format!("{id}.md"),
                    "metadata": {"topic": id}
                }))
                .expect("input");
                core.memorize_document(input).expect("memorize");
            }
            core.delete_document("gone").expect("delete");
            core.compact().expect("compact");
        }
        let mut files = Vec::new();
        files_under(&root, &mut files);
        Template {
            _dir: dir,
            root,
            files,
        }
    })
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let template = template();
    let scratch = tempfile::tempdir().expect("scratch");
    let store = scratch.path().join("store");
    copy_tree(&template.root, &store);

    let file = &template.files[data[0] as usize % template.files.len()];
    let target = store.join(file.strip_prefix(&template.root).expect("relative"));
    let offset = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as usize;
    let payload = &data[6..];
    let mut bytes = std::fs::read(&target).expect("read target");
    match data[1] % 4 {
        0 => {
            let at = offset % bytes.len().max(1);
            let end = (at + payload.len()).min(bytes.len());
            bytes[at..end].copy_from_slice(&payload[..end - at]);
        }
        1 => bytes.truncate(offset % (bytes.len() + 1)),
        2 => bytes.extend_from_slice(payload),
        _ => bytes = payload.to_vec(),
    }
    std::fs::write(&target, bytes).expect("write target");

    let _ = verify_store(&store, None);
    if let Ok(core) = HmsCore::new(DIMENSIONS, Some(store.display().to_string()), None) {
        let _ = core.search_documents("restore backups snapshots", &SearchOptions::default());
        let _ = core.has_document("backups");
    }
});
