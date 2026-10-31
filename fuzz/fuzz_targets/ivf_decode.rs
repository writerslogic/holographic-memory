// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! The quantized IVF index is persisted and reloaded from bytes. Arbitrary bytes must be
//! rejected or decode to an index that re-encodes to exactly the same bytes.

use writerslogic_warm::core::qgraph::ivf::QuantizedIvf;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(index) = QuantizedIvf::from_bytes(data) {
        assert_eq!(
            index.to_bytes(),
            data,
            "decoded index must re-encode identically"
        );
    }
});
