// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! hms-server request surface: the single and batch document bodies, the query body and the
//! tenant header, through the server's own parsing and validation (the first input byte picks
//! the surface). The server source is compiled in as a module; its `fuzz_request` entry point
//! exists only under `--cfg fuzzing`, which cargo-fuzz sets.

#[path = "../../src/bin/hms-server.rs"]
#[allow(dead_code, unused_imports, unexpected_cfgs)]
mod server;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    server::fuzz_request(data);
});
