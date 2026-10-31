// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Question parsing for the operand retriever takes free text typed by a user.

use writerslogic_warm::core::operand_retriever::parse_question;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = parse_question(&String::from_utf8_lossy(data));
});
