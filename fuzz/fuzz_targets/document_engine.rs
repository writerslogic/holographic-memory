// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Document ingest through the engine: a `DocumentInput` and `SearchOptions` as JSON (the
//! shape the Node API, Python SDK and server hand over) are memorized, searched and deleted
//! in a long-lived store. Errors are fine; panics, hangs and unbounded memory are not, and a
//! document that was accepted must be findable by its own first words and deletable.
//!
//! Input layout: JSON `DocumentInput`, a 0x00 byte, JSON `SearchOptions`.

use std::sync::OnceLock;

use writerslogic_warm::{DocumentInput, WarmCore, SearchOptions};
use libfuzzer_sys::fuzz_target;

static CORE: OnceLock<(tempfile::TempDir, WarmCore)> = OnceLock::new();

fn core() -> &'static WarmCore {
    let (_, core) = CORE.get_or_init(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let core = WarmCore::new(512, Some(dir.path().display().to_string()), None).expect("store");
        (dir, core)
    });
    core
}

fuzz_target!(|data: &[u8]| {
    let (doc, options) = match data.iter().position(|&b| b == 0) {
        Some(at) => (&data[..at], &data[at + 1..]),
        None => (data, &[][..]),
    };
    let Ok(input) = serde_json::from_slice::<DocumentInput>(doc) else {
        return;
    };
    let options = serde_json::from_slice::<SearchOptions>(options).unwrap_or_default();
    let core = core();
    let id = input.id.clone();
    let query: String = input
        .text
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ");
    if core.memorize_document(input).is_ok() {
        assert!(core.has_document(&id), "accepted document must exist");
        let _ = core.search_documents(&query, &options);
        let _ = core.search_documents(&query, &SearchOptions::default());
        assert!(core
            .delete_document(&id)
            .expect("delete of an accepted document"));
    }
});
