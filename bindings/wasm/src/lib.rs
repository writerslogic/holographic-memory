// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Browser demo binding: a standalone reimplementation of the sparse
//! hypervector set algebra and the keyed coordinate mask. It is not the HMS
//! engine and its vectors are not interchangeable with the engine's.

use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha8Rng;

const DEFAULT_RHO_DENOM: usize = 256;

#[wasm_bindgen]
pub struct WasmEntangledHVec {
    dim: usize,
    indices: Vec<u32>,
}

#[wasm_bindgen]
impl WasmEntangledHVec {
    #[wasm_bindgen(constructor)]
    pub fn new_deterministic(dim: usize, seed: u64) -> Self {
        let active_count = (dim / DEFAULT_RHO_DENOM).max(1);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut indices = Vec::with_capacity(active_count);
        for _ in 0..active_count {
            indices.push(rng.gen_range(0..dim as u32));
        }
        indices.sort_unstable();
        indices.dedup();
        // Simplified fill for demo purposes
        while indices.len() < active_count {
             indices.push(rng.gen_range(0..dim as u32));
             indices.sort_unstable();
             indices.dedup();
        }
        Self { dim, indices }
    }

    pub fn bind(&self, other: &WasmEntangledHVec) -> Self {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        
        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] < other.indices[j] {
                result.push(self.indices[i]);
                i += 1;
            } else if self.indices[i] > other.indices[j] {
                result.push(other.indices[j]);
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
        while i < self.indices.len() {
            result.push(self.indices[i]);
            i += 1;
        }
        while j < other.indices.len() {
            result.push(other.indices[j]);
            j += 1;
        }

        Self {
            dim: self.dim,
            indices: result,
        }
    }

    pub fn similarity(&self, other: &WasmEntangledHVec) -> f64 {
        let mut intersection = 0;
        let mut i = 0;
        let mut j = 0;
        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] < other.indices[j] {
                i += 1;
            } else if self.indices[i] > other.indices[j] {
                j += 1;
            } else {
                intersection += 1;
                i += 1;
                j += 1;
            }
        }
        let union = self.indices.len() + other.indices.len() - intersection;
        if union == 0 { return 1.0; }
        intersection as f64 / union as f64
    }
}

/// Toy lexical encoder: the union of one deterministic vector per lowercase
/// word, so texts sharing words share active coordinates.
#[wasm_bindgen]
pub fn encode_text(dim: usize, text: &str) -> WasmEntangledHVec {
    let mut indices: Vec<u32> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .flat_map(|w| {
            WasmEntangledHVec::new_deterministic(dim, fxhash::hash64(&w.to_lowercase())).indices
        })
        .collect();
    indices.sort_unstable();
    indices.dedup();
    WasmEntangledHVec { dim, indices }
}

const MAX_DIM: usize = 1 << 20;

/// A secret permutation of the coordinate space derived from a passphrase with
/// Argon2id. Obfuscation, not encryption: every pairwise similarity is
/// preserved and therefore visible to whoever holds the masked vectors.
#[wasm_bindgen]
pub struct WasmVectorMask {
    forward: Vec<u32>,
}

#[wasm_bindgen]
impl WasmVectorMask {
    #[wasm_bindgen(constructor)]
    pub fn derive(passphrase: &str, salt: &str, dim: usize) -> Result<WasmVectorMask, JsError> {
        if passphrase.is_empty() || salt.len() < 16 || !(1..=MAX_DIM).contains(&dim) {
            return Err(JsError::new(
                "passphrase must be non-empty, salt at least 16 bytes, dim in 1..=2^20",
            ));
        }
        let mut key = [0u8; 32];
        argon2::Argon2::default()
            .hash_password_into(passphrase.as_bytes(), salt.as_bytes(), &mut key)
            .map_err(|e| JsError::new(&format!("key derivation failed: {e}")))?;

        let mut prefix = Sha256::new();
        prefix.update(b"hms-vector-mask-v1");
        prefix.update(&key);
        prefix.update((dim as u64).to_le_bytes());

        let mut ranked: Vec<([u8; 32], u32)> = (0..dim as u32)
            .map(|idx| {
                let mut h = prefix.clone();
                h.update(idx.to_le_bytes());
                (h.finalize().into(), idx)
            })
            .collect();
        ranked.sort_unstable();

        let mut forward = vec![0u32; dim];
        for (rank, (_, idx)) in ranked.iter().enumerate() {
            forward[*idx as usize] = rank as u32;
        }
        Ok(WasmVectorMask { forward })
    }

    pub fn apply(&self, vector: &WasmEntangledHVec) -> Result<WasmEntangledHVec, JsError> {
        if vector.dim != self.forward.len() {
            return Err(JsError::new("vector dimension does not match mask"));
        }
        let mut indices: Vec<u32> = vector
            .indices
            .iter()
            .map(|&idx| self.forward[idx as usize])
            .collect();
        indices.sort_unstable();
        Ok(WasmEntangledHVec {
            dim: vector.dim,
            indices,
        })
    }
}

#[wasm_bindgen]
impl WasmEntangledHVec {
    /// Active coordinates, for display.
    pub fn indices(&self) -> Vec<u32> {
        self.indices.clone()
    }
}
