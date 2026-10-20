// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Text produced by an on-device language model is untrusted input to the fact-extraction
//! and query-rewriting stages. The parsers must tolerate any string.

use holographic_memory::core::models::prompts::{
    parse_date, parse_facts, parse_json, parse_query_plan,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let parsed = parse_json(&text);
    let _ = parse_facts(parsed.as_ref(), 8);
    let _ = parse_query_plan(parsed.as_ref());
    let _ = parse_date(&text);
});
