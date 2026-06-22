// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-implementation verification of the cogmem C2PA agent-credential sample.
//!
//! The bytes in `examples/fixtures/*.cose` are the exact COSE/SCITT cognition
//! statements embedded in cogmem's public C2PA sample
//! (`cogmem/examples/c2pa-agent-credential/agent-content.c2pa`). This binary
//! re-verifies them using holographic-memory's own, independent provenance
//! implementation — proving the shared substrate is byte-compatible across
//! implementations, not just within one.
//!
//! Run: `cargo run --features provenance --example verify_cogmem_sample`

use ciborium::value::Value;
use ed25519_dalek::VerifyingKey;
use holographic_memory::core::provenance::{cose, did};

const MEMORY: &[u8] = include_bytes!("fixtures/cogmem.memory.provenance.cose");
const REASONING: &[u8] = include_bytes!("fixtures/crosstalk.orchestration.audit.cose");

fn field<'a>(claim: &'a Value, key: &str) -> Option<&'a Value> {
    match claim {
        Value::Map(entries) => entries
            .iter()
            .find(|(k, _)| matches!(k, Value::Text(t) if t == key))
            .map(|(_, v)| v),
        _ => None,
    }
}

fn text(claim: &Value, key: &str) -> String {
    match field(claim, key) {
        Some(Value::Text(t)) => t.clone(),
        _ => "?".to_string(),
    }
}

/// Verify a COSE_Sign1 statement and confirm its issuer DID is bound to the
/// signing key (the `iss` in the payload must derive from the key that signed it).
fn verify(label: &str, cose_bytes: &[u8]) -> anyhow::Result<Value> {
    let kid = cose::extract_key_id(cose_bytes)?;
    let key_bytes: [u8; 32] = kid
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("kid must be a 32-byte ed25519 key"))?;
    let verifying_key = VerifyingKey::from_bytes(&key_bytes)?;
    let payload = cose::verify_and_extract(&verifying_key, cose_bytes)?;
    let claim: Value = ciborium::from_reader(payload.as_slice())?;

    let iss = text(&claim, "iss");
    let derived = did::did_key_from_ed25519(&key_bytes);
    if iss != derived {
        anyhow::bail!("issuer binding mismatch: claim iss {iss} != signing key {derived}");
    }
    println!("  VERIFIED {label}");
    println!("           issuer {iss} (matches the signing key)");
    Ok(claim)
}

fn main() -> anyhow::Result<()> {
    println!(
        "holographic-memory independently verifying the cogmem C2PA sample's \
         cognition statements:\n"
    );

    let mem = verify("memory    cogmem.memory.provenance", MEMORY)?;
    println!(
        "           attests: memory '{}' ({}, {})",
        text(&mem, "memoryId"),
        text(&mem, "memoryType"),
        text(&mem, "event")
    );

    let rsn = verify("reasoning crosstalk.orchestration.audit", REASONING)?;
    let turns = match field(&rsn, "turn_count") {
        Some(Value::Integer(i)) => {
            let n: i128 = (*i).into();
            n.to_string()
        }
        _ => "?".to_string(),
    };
    println!(
        "           attests: session '{}', {} turns",
        text(&rsn, "session_id"),
        turns
    );

    println!(
        "\nPASS: both cognition statements verify under holographic-memory — \
         identical bytes,\n      independent implementation. \
         Cross-implementation conformance confirmed."
    );
    Ok(())
}
