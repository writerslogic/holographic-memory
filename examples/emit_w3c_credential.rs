// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use ed25519_dalek::SigningKey;
use holographic_memory::core::provenance::{did, status_list, vc};

fn main() -> anyhow::Result<()> {
    let key = SigningKey::from_bytes(&rand::random());
    let did = did::did_key_from_ed25519(&key.verifying_key().to_bytes());
    let url = "https://example.com/credentials/status/1";
    let fact = vc::create_fact_credential(&did, "interop-example", &[0x42; 32], 64, None, None, 7)
        .with_status_list_url(url);
    let credential = vc::sign_credential(&key, fact)?;
    let revoked_fact =
        vc::create_fact_credential(&did, "interop-revoked", &[0x24; 32], 64, None, None, 8)
            .with_status_list_url(url);
    let revoked_credential = vc::sign_credential(&key, revoked_fact)?;
    let status_list = status_list::create_status_list_credential(&key, url, &[0, 1], 8)?;
    println!(
        "{}",
        serde_json::json!({
            "credential": credential,
            "revokedCredential": revoked_credential,
            "statusList": status_list
        })
    );
    Ok(())
}
