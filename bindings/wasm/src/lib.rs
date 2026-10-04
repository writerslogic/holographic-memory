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
