// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! The connection graph is rebuilt by replaying an append-only, length-prefixed bincode event
//! log. A truncated or corrupt log must stop at the last good event, never panic or allocate
//! from a length read off the file, and the rebuilt graph must verify against its own log.

use std::sync::OnceLock;

use holographic_memory::core::connection_graph::{ConnectionGraph, GraphConfig};
use libfuzzer_sys::fuzz_target;

fn log_path() -> &'static std::path::Path {
    // One file, overwritten by every iteration.
    static DIR: OnceLock<(tempfile::TempDir, std::path::PathBuf)> = OnceLock::new();
    let (_, path) = DIR.get_or_init(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("graph.log");
        (dir, path)
    });
    path
}

fuzz_target!(|data: &[u8]| {
    let path = log_path();
    if std::fs::write(path, data).is_err() {
        return;
    }
    if let Ok(graph) = ConnectionGraph::open(64, GraphConfig::default(), path) {
        assert!(graph.verify());
    }
});
