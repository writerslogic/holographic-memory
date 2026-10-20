// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Key derivation (and, behind `inwasm-aes`, the AEAD) for encrypted snapshots. WebCrypto
//! has no Argon2, so the memory-hard KDF lives here; AES-GCM itself is done by WebCrypto
//! unless measurement says otherwise (docs/runtimes/wasm/ENCRYPTION.md).

use wasm_bindgen::prelude::*;

const MAX_MEMORY_KIB: u32 = 1 << 20;
const MAX_PASSES: u32 = 64;
const MAX_LANES: u32 = 16;

/// Argon2id: a 32-byte key from a passphrase and a salt of at least 16 bytes.
#[wasm_bindgen(js_name = argon2idKey)]
pub fn argon2id_key(
    passphrase: &[u8],
    salt: &[u8],
    memory_kib: u32,
    passes: u32,
    lanes: u32,
) -> Result<Vec<u8>, JsError> {
    if passphrase.is_empty()
        || salt.len() < 16
        || !(8..=MAX_MEMORY_KIB).contains(&memory_kib)
        || !(1..=MAX_PASSES).contains(&passes)
        || !(1..=MAX_LANES).contains(&lanes)
    {
        return Err(JsError::new(
            "argon2id needs a passphrase, a 16-byte salt, memory 8..=2^20 KiB, passes 1..=64, lanes 1..=16",
        ));
    }
    let params = argon2::Params::new(memory_kib, passes, lanes, Some(32))
        .map_err(|e| JsError::new(&format!("argon2id parameters: {e}")))?;
    let mut key = vec![0u8; 32];
    argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params)
        .hash_password_into(passphrase, salt, &mut key)
        .map_err(|e| JsError::new(&format!("argon2id failed: {e}")))?;
    Ok(key)
}

#[cfg(feature = "inwasm-aes")]
mod aes {
    use aes_gcm::{
        aead::{Aead, KeyInit, Payload},
        Aes256Gcm, Nonce,
    };
    use wasm_bindgen::prelude::*;

    fn cipher(key: &[u8], nonce: &[u8]) -> Result<Aes256Gcm, JsError> {
        if nonce.len() != 12 {
            return Err(JsError::new("AES-GCM nonce must be 12 bytes"));
        }
        Aes256Gcm::new_from_slice(key).map_err(|_| JsError::new("AES-256 key must be 32 bytes"))
    }

    /// AES-256-GCM seal: returns ciphertext followed by the 16-byte tag.
    #[wasm_bindgen(js_name = aesGcmSeal)]
    pub fn seal(key: &[u8], nonce: &[u8], aad: &[u8], data: &[u8]) -> Result<Vec<u8>, JsError> {
        cipher(key, nonce)?
            .encrypt(Nonce::from_slice(nonce), Payload { msg: data, aad })
            .map_err(|_| JsError::new("encryption failed"))
    }

    #[wasm_bindgen(js_name = aesGcmOpen)]
    pub fn open(key: &[u8], nonce: &[u8], aad: &[u8], data: &[u8]) -> Result<Vec<u8>, JsError> {
        cipher(key, nonce)?
            .decrypt(Nonce::from_slice(nonce), Payload { msg: data, aad })
            .map_err(|_| JsError::new("decryption failed: wrong key or corrupted data"))
    }
}
