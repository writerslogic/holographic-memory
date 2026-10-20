// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Stderr logging. stdout carries the stdio transport's protocol and must stay clean. Callers
//! pass operation names, ids' lengths and codes, never memory text, queries or secrets.

use std::sync::atomic::{AtomicU8, Ordering};

static LEVEL: AtomicU8 = AtomicU8::new(1);

pub fn set_level(name: &str) -> Result<(), String> {
    let level = match name {
        "error" => 0,
        "warn" => 1,
        "info" => 2,
        "debug" => 3,
        other => {
            return Err(format!(
                "unknown log level {other:?}; use error, warn, info or debug"
            ))
        }
    };
    LEVEL.store(level, Ordering::Relaxed);
    Ok(())
}

fn emit(level: u8, tag: &str, message: &str) {
    if level <= LEVEL.load(Ordering::Relaxed) {
        eprintln!("hms-mcp {tag}: {message}");
    }
}

pub fn error(message: &str) {
    emit(0, "error", message);
}
pub fn warn(message: &str) {
    emit(1, "warn", message);
}
pub fn info(message: &str) {
    emit(2, "info", message);
}
pub fn debug(message: &str) {
    emit(3, "debug", message);
}
