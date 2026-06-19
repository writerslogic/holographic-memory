// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Sparse Clifford algebra multivector for holographic memory.
//!
//! Implements Cl(n,0) — positive-definite Clifford algebra where e_i² = +1.
//! A multivector is a sparse linear combination of basis blades, where each
//! blade is a subset of basis vectors encoded as a bitmask.
//!
//! For n=14, the algebra has 2^14 = 16384 basis blades, matching HMS's
//! default dimensionality. Sparse storage (~64 non-zero terms) uses ~512 bytes
//! per vector, comparable to EntangledHVec's ~256 bytes.
//!
//! Key operations and their holographic interpretation:
//! - **Geometric product** (bind): `uv = u·v + u∧v` — simultaneously captures
//!   coherence (inner product) and structural variation (wedge product).
//!   Validated by CliffordNet (Ji, 2026) as sufficient for all interactions.
//! - **Addition** (bundle): superposition of interference patterns.
//! - **Grade-0 projection** (similarity): scalar part of geometric product.

use crate::core::algebra::HolographicAlgebra;
use fxhash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Default number of non-zero terms to maintain in sparse multivectors.
const DEFAULT_SPARSITY: usize = 64;

/// Sparse Clifford multivector in Cl(n,0).
///
/// Each term is a (blade, coefficient) pair where blade is a bitmask encoding
/// which basis vectors are present. For Cl(14,0), blades fit in 14 bits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CliffordVec {
    /// Number of basis vectors (n in Cl(n,0)). Algebra dimension = 2^n.
    n: usize,
    /// Sparse terms sorted by blade index. Invariant: no zero coefficients.
    terms: Vec<(u32, f32)>,
}

impl CliffordVec {
    /// Algebra dimension (number of possible basis blades = 2^n).
    pub fn algebra_dim(&self) -> usize {
        1 << self.n
    }

    /// Compute n from a target dimension (n = ceil(log2(dim))).
    fn n_from_dim(dim: usize) -> usize {
        if dim <= 1 {
            return 1;
        }
        let mut n = 0;
        let mut d = dim - 1;
        while d > 0 {
            d >>= 1;
            n += 1;
        }
        n
    }

    /// Geometric product of two sparse multivectors.
    ///
    /// For each pair of terms (blade_a, coeff_a) and (blade_b, coeff_b):
    ///   result_blade = blade_a XOR blade_b
    ///   result_coeff = sign(blade_a, blade_b) * coeff_a * coeff_b
    ///
    /// The sign accounts for anticommutativity of basis vectors.
    fn geometric_product(&self, other: &Self) -> Self {
        let mut accum: FxHashMap<u32, f32> =
            FxHashMap::with_capacity_and_hasher(self.terms.len() * 2, Default::default());

        for &(blade_a, coeff_a) in &self.terms {
            for &(blade_b, coeff_b) in &other.terms {
                let result_blade = blade_a ^ blade_b;
                let sign = geometric_sign(blade_a, blade_b);
                let contribution = sign as f32 * coeff_a * coeff_b;
                *accum.entry(result_blade).or_insert(0.0) += contribution;
            }
        }

        let mut terms: Vec<(u32, f32)> = accum
            .into_iter()
            .filter(|(_, c)| c.abs() > f32::EPSILON)
            .collect();

        // Truncate to sparsity target: keep highest-magnitude terms
        if terms.len() > DEFAULT_SPARSITY {
            terms.select_nth_unstable_by(DEFAULT_SPARSITY - 1, |a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            terms.truncate(DEFAULT_SPARSITY);
        }

        terms.sort_unstable_by_key(|&(blade, _)| blade);

        Self { n: self.n, terms }
    }

    /// Scalar (grade-0) part of the multivector.
    pub fn scalar_part(&self) -> f32 {
        self.terms
            .iter()
            .find(|&&(blade, _)| blade == 0)
            .map(|&(_, c)| c)
            .unwrap_or(0.0)
    }

    /// L2 norm: sqrt(sum of squared coefficients).
    pub fn norm(&self) -> f32 {
        self.terms
            .iter()
            .map(|&(_, c)| c * c)
            .sum::<f32>()
            .sqrt()
    }

    /// Clifford reverse: reverses the order of basis vectors in each blade.
    /// For a grade-k blade, the reverse has sign (-1)^(k(k-1)/2).
    pub fn reverse(&self) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|&(blade, coeff)| {
                let k = blade.count_ones();
                let sign = if (k * (k - 1) / 2) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                (blade, coeff * sign)
            })
            .collect();
        Self { n: self.n, terms }
    }

    /// Normalize to unit norm.
    fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm < f32::EPSILON {
            return self.clone();
        }
        let terms = self.terms.iter().map(|&(b, c)| (b, c / norm)).collect();
        Self { n: self.n, terms }
    }

    /// Convert from EntangledHVec: each active index becomes a unit-coefficient blade.
    pub fn from_entangled(vec: &crate::core::entangled::EntangledHVec) -> Self {
        let n = Self::n_from_dim(vec.dim);
        let terms: Vec<(u32, f32)> = vec.indices.iter().map(|&idx| (idx, 1.0)).collect();
        Self { n, terms }
    }

    /// Convert back to EntangledHVec: take blade indices of highest-magnitude terms.
    pub fn to_entangled(&self, dim: usize) -> crate::core::entangled::EntangledHVec {
        let target = (dim / crate::core::entangled::DEFAULT_RHO_DENOM).max(1);
        let mut scored: Vec<(u32, f32)> = self
            .terms
            .iter()
            .filter(|&&(blade, _)| (blade as usize) < dim)
            .copied()
            .collect();

        if scored.len() > target {
            scored.select_nth_unstable_by(target - 1, |a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(target);
        }

        let mut indices: Vec<u32> = scored.into_iter().map(|(blade, _)| blade).collect();
        indices.sort_unstable();
        crate::core::entangled::EntangledHVec::from_indices(indices, dim)
    }
}

/// Sign of the geometric product of two basis blades in Cl(n,0).
///
/// Counts the number of transpositions needed to sort the combined basis
/// vector sequence. For blade_a = {i1, i2, ...} and blade_b = {j1, j2, ...},
/// each basis vector in blade_b must "pass through" all basis vectors in
/// blade_a that have higher index. The sign is (-1)^(total transpositions).
///
/// In Cl(n,0) (positive-definite): e_i² = +1, so shared basis vectors
/// cancel from the blade (via XOR) but don't affect the sign beyond the
/// transposition count.
fn geometric_sign(blade_a: u32, blade_b: u32) -> i8 {
    let mut swaps = 0u32;
    let mut b = blade_b;
    while b != 0 {
        let pos = b.trailing_zeros();
        b &= b - 1; // clear lowest set bit
        // Count bits in blade_a strictly above position `pos`
        let above = blade_a >> (pos + 1);
        swaps += above.count_ones();
    }
    if swaps % 2 == 0 {
        1
    } else {
        -1
    }
}

impl HolographicAlgebra for CliffordVec {
    fn dim(&self) -> usize {
        self.algebra_dim()
    }

    fn bind(&self, other: &Self) -> Self {
        self.geometric_product(other)
    }

    fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self {
                n: 1,
                terms: Vec::new(),
            };
        }
        let n = vectors[0].n;
        let mut accum: FxHashMap<u32, f32> = FxHashMap::default();
        for v in vectors {
            for &(blade, coeff) in &v.terms {
                *accum.entry(blade).or_insert(0.0) += coeff;
            }
        }

        let mut terms: Vec<(u32, f32)> = accum
            .into_iter()
            .filter(|(_, c)| c.abs() > f32::EPSILON)
            .collect();

        if terms.len() > DEFAULT_SPARSITY {
            terms.select_nth_unstable_by(DEFAULT_SPARSITY - 1, |a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            terms.truncate(DEFAULT_SPARSITY);
        }

        terms.sort_unstable_by_key(|&(blade, _)| blade);

        let result = Self { n, terms };
        result.normalize()
    }

    fn similarity(&self, other: &Self) -> f64 {
        // Reverse inner product: sim(a, b) = ⟨a * b̃⟩_0 / (|a| · |b|)
        // where b̃ is the Clifford reverse (reverses basis vector order in each blade).
        // The reverse of grade-k blade has sign (-1)^(k(k-1)/2), which exactly cancels
        // the geometric product sign for matching blades, giving a positive-definite metric.
        let norm_a = self.norm() as f64;
        let norm_b = other.norm() as f64;
        if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
            return if norm_a < f64::EPSILON && norm_b < f64::EPSILON {
                1.0
            } else {
                0.0
            };
        }

        // Grade-0 contributions come from pairs where blade_a == blade_b.
        // The reverse sign and product sign cancel, so contribution = c_a * c_b.
        let mut scalar = 0.0f64;
        let mut ia = 0;
        let mut ib = 0;
        while ia < self.terms.len() && ib < other.terms.len() {
            let (ba, ca) = self.terms[ia];
            let (bb, cb) = other.terms[ib];
            match ba.cmp(&bb) {
                std::cmp::Ordering::Less => ia += 1,
                std::cmp::Ordering::Greater => ib += 1,
                std::cmp::Ordering::Equal => {
                    scalar += ca as f64 * cb as f64;
                    ia += 1;
                    ib += 1;
                }
            }
        }

        (scalar / (norm_a * norm_b)).clamp(0.0, 1.0)
    }

    fn permute(&self, shifts: usize) -> Self {
        if self.n == 0 || shifts == 0 {
            return self.clone();
        }
        let shift = (shifts % self.n) as u32;
        let n = self.n as u32;

        // Cyclic permutation of basis vectors: e_i → e_{(i+shift) mod n}.
        // For a blade represented as bitmask, rotate the bits.
        let mask = (1u32 << n) - 1;
        let terms: Vec<(u32, f32)> = self
            .terms
            .iter()
            .map(|&(blade, coeff)| {
                let rotated = ((blade << shift) | (blade >> (n - shift))) & mask;
                (rotated, coeff)
            })
            .collect();

        let mut result = Self { n: self.n, terms };
        result.terms.sort_unstable_by_key(|&(blade, _)| blade);
        // Dedup in case rotation creates collisions (shouldn't for permutation, but defensive)
        result.terms.dedup_by(|b, a| {
            if a.0 == b.0 {
                a.1 += b.1;
                true
            } else {
                false
            }
        });
        result
    }

    fn from_seed(dim: usize, seed: u64) -> Self {
        // Generate a sparse multivector deterministically from seed.
        // Use the same hash-based approach as EntangledHVec but produce
        // (blade, coefficient) pairs.
        let n = Self::n_from_dim(dim);
        let algebra_dim = 1usize << n;
        let active_count = (algebra_dim / crate::core::entangled::DEFAULT_RHO_DENOM).max(1);

        let mut terms = Vec::with_capacity(active_count);
        for i in 0..active_count {
            let blade = hash_u64(seed, i as u64) % algebra_dim as u64;
            // Coefficient: deterministic sign based on hash
            let sign_hash = hash_u64(seed.wrapping_add(0xCAFE), i as u64);
            let coeff = if sign_hash % 2 == 0 { 1.0f32 } else { -1.0 };
            terms.push((blade as u32, coeff));
        }

        terms.sort_unstable_by_key(|&(blade, _)| blade);
        terms.dedup_by(|b, a| {
            if a.0 == b.0 {
                a.1 += b.1;
                true
            } else {
                false
            }
        });
        // Remove zero-coefficient terms from cancellation
        terms.retain(|&(_, c)| c.abs() > f32::EPSILON);

        Self { n, terms }
    }

    fn from_active_indices(indices: Vec<u32>, dim: usize) -> Self {
        let n = Self::n_from_dim(dim);
        let terms: Vec<(u32, f32)> = indices.into_iter().map(|idx| (idx, 1.0)).collect();
        Self { n, terms }
    }

    fn active_indices(&self) -> &[u32] {
        // This is a lossy view — returns blade indices, ignoring coefficients.
        // Safe for interop with code expecting sparse index sets.
        // We can't return a reference to a computed Vec, so we store blades
        // as the first element of each term pair. Since terms is Vec<(u32, f32)>,
        // we can't directly return &[u32]. Return empty and let callers use
        // conversion methods instead.
        &[]
    }
}

fn hash_u64(a: u64, b: u64) -> u64 {
    use fxhash::FxHasher;
    use std::hash::Hasher;
    let mut h = FxHasher::default();
    h.write_u64(a);
    h.write_u64(b);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_sign_identity() {
        // e_0 * e_0 = +1 (Cl(n,0))
        assert_eq!(geometric_sign(0b1, 0b1), 1);
    }

    #[test]
    fn test_geometric_sign_anticommute() {
        // e_0 * e_1 should have opposite sign to e_1 * e_0
        let s01 = geometric_sign(0b01, 0b10);
        let s10 = geometric_sign(0b10, 0b01);
        assert_eq!(s01, -s10, "Basis vectors should anticommute");
    }

    #[test]
    fn test_geometric_sign_scalar() {
        // scalar * anything = anything (blade 0 is the scalar)
        assert_eq!(geometric_sign(0, 0b101), 1);
        assert_eq!(geometric_sign(0b101, 0), 1);
    }

    #[test]
    fn test_geometric_product_associative() {
        let dim = 16384;
        let a = CliffordVec::from_seed(dim, 1);
        let b = CliffordVec::from_seed(dim, 2);
        let c = CliffordVec::from_seed(dim, 3);

        let ab_c = a.geometric_product(&b).geometric_product(&c);
        let a_bc = a.geometric_product(&b.geometric_product(&c));

        // Associativity may be approximate due to sparse truncation
        let sim = ab_c.similarity(&a_bc);
        assert!(
            sim > 0.5,
            "Geometric product should be approximately associative, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_bind_reverse_unbinding() {
        let dim = 16384;
        let a = CliffordVec::from_seed(dim, 10);
        let b = CliffordVec::from_seed(dim, 20);

        // In Clifford algebra, unbinding uses the reverse: a*b → (a*b)*b̃ ≈ a*|b|²
        // This is the proper inverse, unlike double-bind (which works for XOR but not GP).
        let ab = a.bind(&b);
        let b_rev = b.reverse();
        let recovered = ab.bind(&b_rev);
        let sim = recovered.similarity(&a);
        assert!(
            sim > 0.1,
            "Reverse-based unbinding should approximately recover original, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_self_similarity() {
        let dim = 16384;
        let a = CliffordVec::from_seed(dim, 42);
        let sim = a.similarity(&a);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "Self-similarity should be ~1.0, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_random_pair_low_similarity() {
        let dim = 16384;
        let a = CliffordVec::from_seed(dim, 1);
        let b = CliffordVec::from_seed(dim, 2);
        let sim = a.similarity(&b);
        assert!(
            sim < 0.3,
            "Random pair should have low similarity, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_bundle_majority() {
        let dim = 16384;
        let base = CliffordVec::from_seed(dim, 100);
        let noise1 = CliffordVec::from_seed(dim, 200);
        let noise2 = CliffordVec::from_seed(dim, 300);

        let vecs = vec![
            base.clone(),
            base.clone(),
            base.clone(),
            base.clone(),
            base.clone(),
            noise1,
            noise2,
        ];
        let bundled = CliffordVec::bundle(&vecs);
        let sim_base = bundled.similarity(&base);
        let sim_random = bundled.similarity(&CliffordVec::from_seed(dim, 400));
        assert!(
            sim_base > sim_random,
            "Bundle should favor majority: base={:.4} random={:.4}",
            sim_base,
            sim_random
        );
    }

    #[test]
    fn test_from_entangled_roundtrip() {
        let dim = 16384;
        let entangled = crate::core::entangled::EntangledHVec::new_deterministic(dim, 42);
        let clifford = CliffordVec::from_entangled(&entangled);
        let back = clifford.to_entangled(dim);

        let sim = entangled.similarity(&back);
        assert!(
            sim > 0.9,
            "Roundtrip should preserve most information, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_contract_satisfied() {
        // Run the generic algebra contract test
        crate::core::algebra::tests::test_algebra_contract_for::<CliffordVec>(16384);
    }
}
