// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Signed envelopes from outside: bare COSE_Sign1, the signed HMS manifest and the SCITT
//! signed statement. A fixed key verifies them, so only a correctly signed seed gets past the
//! signature; every other input must be rejected without panicking.

use ed25519_dalek::SigningKey;
use holographic_memory::core::provenance::{c2pa_manifest, cose, scitt};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let key = SigningKey::from_bytes(&[7u8; 32]).verifying_key();
    let _ = cose::verify_and_extract(&key, data);
    let _ = c2pa_manifest::verify_manifest(&key, data);
    let statement = scitt::SignedStatement {
        cose_bytes: data.to_vec(),
    };
    let _ = scitt::verify_statement(&key, &statement);
});
