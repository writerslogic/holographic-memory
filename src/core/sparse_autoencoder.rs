// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Sparse autoencoder for projecting dense embeddings into HMS sparse space.
//!
//! Inference-only: takes a pre-trained weight matrix and bias, applies
//! linear projection + ReLU + top-k sparsification to produce an
//! EntangledHVec. Training is done externally (Python/PyTorch).
//!
//! Architecture: dense_input (d_in) → W (d_in × d_hms) → ReLU → top-k → binary

use crate::core::entangled::{EntangledHVec, DEFAULT_RHO_DENOM};

pub struct SparseAutoencoder {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    output_dim: usize,
}

impl SparseAutoencoder {
    pub fn new(weights: Vec<Vec<f32>>, bias: Vec<f32>, output_dim: usize) -> Self {
        assert!(!weights.is_empty(), "weights must be non-empty");
        assert_eq!(
            weights[0].len(),
            output_dim,
            "weight columns must match output_dim"
        );
        assert_eq!(bias.len(), output_dim, "bias length must match output_dim");
        Self {
            weights,
            bias,
            output_dim,
        }
    }

    /// Random projection encoder (no training needed).
    /// Uses Achlioptas sparse ternary projection: P(+1)=P(-1)=1/6, P(0)=2/3.
    /// Deterministic given seed.
    pub fn random_projection(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        use crate::core::entangled::hash_u64;
        let mut weights = Vec::with_capacity(input_dim);
        for i in 0..input_dim {
            let mut row = vec![0.0f32; output_dim];
            for (j, cell) in row.iter_mut().enumerate() {
                let r = hash_u64(seed.wrapping_add(i as u64), j as u64) % 6;
                *cell = match r {
                    0 => 1.0,
                    5 => -1.0,
                    _ => 0.0,
                };
            }
            weights.push(row);
        }
        let bias = vec![0.0f32; output_dim];
        Self {
            weights,
            bias,
            output_dim,
        }
    }

    /// Encode a dense vector into a sparse HMS vector.
    pub fn encode(&self, input: &[f32]) -> EntangledHVec {
        let active_count = (self.output_dim / DEFAULT_RHO_DENOM).max(1);

        let mut activations: Vec<(u32, f32)> = Vec::with_capacity(self.output_dim);
        for j in 0..self.output_dim {
            let mut sum = self.bias[j];
            for (i, &val) in input.iter().enumerate() {
                if i < self.weights.len() {
                    sum += val * self.weights[i][j];
                }
            }
            let relu = sum.max(0.0);
            if relu > 0.0 {
                activations.push((j as u32, relu));
            }
        }

        if activations.len() > active_count {
            activations.select_nth_unstable_by(active_count - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            activations.truncate(active_count);
        }

        let mut indices: Vec<u32> = activations.into_iter().map(|(idx, _)| idx).collect();
        indices.sort_unstable();
        EntangledHVec::from_indices(indices, self.output_dim)
    }

    /// Batch encode multiple vectors.
    pub fn encode_batch(&self, inputs: &[Vec<f32>]) -> Vec<EntangledHVec> {
        inputs.iter().map(|inp| self.encode(inp)).collect()
    }

    pub fn input_dim(&self) -> usize {
        self.weights.len()
    }
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_projection_produces_correct_density() {
        let sae = SparseAutoencoder::random_projection(128, 16384, 42);
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let encoded = sae.encode(&input);
        let expected = 16384 / DEFAULT_RHO_DENOM;
        assert_eq!(
            encoded.indices().len(),
            expected,
            "Should have {} active indices, got {}",
            expected,
            encoded.indices().len()
        );
    }

    #[test]
    fn test_similar_inputs_produce_similar_outputs() {
        let sae = SparseAutoencoder::random_projection(64, 16384, 42);
        let a: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let mut b = a.clone();
        b[0] += 0.01;
        b[1] -= 0.01;

        let ea = sae.encode(&a);
        let eb = sae.encode(&b);
        let sim = ea.similarity(&eb);
        assert!(
            sim > 0.5,
            "Similar inputs should produce similar outputs, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_different_inputs_produce_different_outputs() {
        let sae = SparseAutoencoder::random_projection(64, 16384, 42);
        let a: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let b: Vec<f32> = (0..64).map(|i| -(i as f32) / 64.0).collect();

        let ea = sae.encode(&a);
        let eb = sae.encode(&b);
        let sim = ea.similarity(&eb);
        assert!(
            sim < 0.3,
            "Different inputs should produce different outputs, got {:.4}",
            sim
        );
    }
}
