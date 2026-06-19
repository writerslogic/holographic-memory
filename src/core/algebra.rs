// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Trait boundary for holographic vector algebra.
//!
//! Defines the algebraic operations that any vector representation must support.
//! Currently implemented by `EntangledHVec` (sparse binary). Future implementations
//! include a geometric algebra type using the Clifford geometric product.
//!
//! Call sites are NOT yet generic over this trait — they use `EntangledHVec`
//! directly. The trait exists as a contract: any future representation must
//! implement these operations to be a drop-in replacement. Migration to
//! trait-generic code happens incrementally when the second implementation lands.

use std::fmt::Debug;

/// Core algebraic operations for holographic memory vectors.
///
/// The three fundamental operations (bind, bundle, similarity) correspond to:
/// - **Bind**: Compositional association (role-filler binding, sequence encoding).
///   Must be approximately invertible: `a.bind(b).bind(b) ≈ a`.
/// - **Bundle**: Superposition of multiple items into a single representation.
///   Must preserve similarity: `bundle([a, a, a, b]).similarity(a) > bundle([a, a, a, b]).similarity(b)` is not guaranteed, but `bundle` of many copies of `a` should be close to `a`.
/// - **Similarity**: Retrieval metric. Higher = more similar.
///
/// These map to different algebra depending on representation:
/// - Sparse binary (EntangledHVec): XOR, majority vote, Jaccard
/// - HRR (future): circular convolution, addition, cosine
/// - Clifford (future): geometric product, addition, grade-0 projection
pub trait HolographicAlgebra: Clone + Debug + Send + Sync {
    /// Dimensionality of the vector space.
    fn dim(&self) -> usize;

    /// Bind two vectors (compositional association).
    /// For sparse binary: symmetric difference (XOR).
    /// For HRR: circular convolution.
    /// For Clifford: geometric product.
    fn bind(&self, other: &Self) -> Self;

    /// Bundle multiple vectors into a superposition.
    /// For sparse binary: majority vote with threshold.
    /// For HRR/Clifford: element-wise addition + normalization.
    fn bundle(vectors: &[Self]) -> Self
    where
        Self: Sized;

    /// Similarity between two vectors. Range: [0, 1] where 1 = identical.
    /// For sparse binary: Jaccard index.
    /// For HRR: cosine similarity.
    /// For Clifford: normalized grade-0 inner product.
    fn similarity(&self, other: &Self) -> f64;

    /// Positional permutation (shift indices by `shifts` modulo dim).
    /// Used for position-sensitive encoding (n-grams, sequences).
    fn permute(&self, shifts: usize) -> Self;

    /// Create a deterministic vector from a seed.
    /// Same seed + same dim must always produce the same vector.
    fn from_seed(dim: usize, seed: u64) -> Self
    where
        Self: Sized;

    /// Create from sorted active indices (sparse representation).
    /// For dense representations, this sets the given positions to 1.0.
    fn from_active_indices(indices: Vec<u32>, dim: usize) -> Self
    where
        Self: Sized;

    /// Return active indices (for sparse representations).
    /// Dense representations should return indices of non-zero elements.
    fn active_indices(&self) -> &[u32];
}

/// Implement the trait for the existing sparse binary representation.
impl HolographicAlgebra for super::entangled::EntangledHVec {
    fn dim(&self) -> usize {
        self.dim
    }

    fn bind(&self, other: &Self) -> Self {
        self.bind(other)
    }

    fn bundle(vectors: &[Self]) -> Self {
        Self::bundle(vectors)
    }

    fn similarity(&self, other: &Self) -> f64 {
        self.similarity(other)
    }

    fn permute(&self, shifts: usize) -> Self {
        self.permute(shifts)
    }

    fn from_seed(dim: usize, seed: u64) -> Self {
        Self::new_deterministic(dim, seed)
    }

    fn from_active_indices(indices: Vec<u32>, dim: usize) -> Self {
        Self::from_indices(indices, dim)
    }

    fn active_indices(&self) -> &[u32] {
        self.indices()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_algebra_contract<T: HolographicAlgebra>(dim: usize) {
        // Determinism
        let a = T::from_seed(dim, 42);
        let b = T::from_seed(dim, 42);
        assert!(
            (a.similarity(&b) - 1.0).abs() < 1e-10,
            "Same seed must produce identical vectors"
        );

        // Self-similarity
        assert!(
            (a.similarity(&a) - 1.0).abs() < 1e-10,
            "Self-similarity must be 1.0"
        );

        // Bind approximate involution: (a ⊕ b) ⊕ b ≈ a
        let c = T::from_seed(dim, 99);
        let ac = a.bind(&c);
        let recovered = ac.bind(&c);
        assert!(
            recovered.similarity(&a) > 0.9,
            "Bind should be approximately invertible"
        );

        // Bundle majority: bundle of 5×a + 2×noise should be close to a
        let noise1 = T::from_seed(dim, 200);
        let noise2 = T::from_seed(dim, 300);
        let vecs = vec![
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            noise1,
            noise2,
        ];
        let bundled = T::bundle(&vecs);
        let sim_a = bundled.similarity(&a);
        let sim_random = bundled.similarity(&T::from_seed(dim, 400));
        assert!(
            sim_a > sim_random,
            "Bundle majority should be closer to majority element"
        );

        // Permute produces different vector
        let permuted = a.permute(1);
        assert!(
            a.similarity(&permuted) < 0.9,
            "Permuted vector should differ from original"
        );
    }

    #[test]
    fn test_entangled_hvec_satisfies_contract() {
        test_algebra_contract::<super::super::entangled::EntangledHVec>(16384);
    }
}
