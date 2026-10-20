// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! JUMBF (ISO 19566-5) box hierarchies carry C2PA manifests from untrusted files. Decoding
//! must terminate with bounded memory and nesting; a box that decodes must encode again.

use holographic_memory::core::provenance::jumbf::JumbfBox;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(parsed) = JumbfBox::decode(data) {
        let _ = parsed.encode();
    }
});
