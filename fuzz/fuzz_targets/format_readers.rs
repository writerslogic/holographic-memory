// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;

// Every persisted-format reader (frames, payloads, manifests, headers, audit, caches) on
// arbitrary bytes: must return, never panic, never allocate from an untrusted length.
fuzz_target!(|data: &[u8]| {
    writerslogic_warm::core::format::scan::probe(data);
});
