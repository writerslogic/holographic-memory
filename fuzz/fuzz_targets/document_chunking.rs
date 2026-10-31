// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Document ingest, chunking stage. `DocumentInput` arrives as JSON from the Node API, the
//! Python SDK and the server. Whatever it contains, chunking must not panic, and every chunk
//! must be a valid slice of the source text on char boundaries.

use writerslogic_warm::{chunk_document, DocumentInput};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(input) = serde_json::from_slice::<DocumentInput>(data) else {
        return;
    };
    let Ok(chunks) = chunk_document(&input) else {
        return;
    };
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        let (start, end) = (chunk.start_byte as usize, chunk.end_byte as usize);
        assert!(start <= end && end <= input.text.len());
        assert_eq!(input.text.get(start..end), Some(chunk.text.as_str()));
    }
});
