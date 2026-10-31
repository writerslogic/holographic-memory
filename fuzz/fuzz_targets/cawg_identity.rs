// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! CAWG identity assertions are CBOR maps holding a tag-18 COSE_Sign1 over a W3C VC whose
//! issuer key is a did:jwk named inside the untrusted document itself.

use writerslogic_warm::core::provenance::cawg::verify_cawg_ica;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = verify_cawg_ica(data);
});
