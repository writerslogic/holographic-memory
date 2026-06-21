// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Ternary sparse hypervector: {-1, 0, +1} with density rho.
//!
//! Same sparsity as EntangledHVec (1/256 non-zero) but each active index
//! carries a sign bit, roughly doubling information capacity per active
//! position. Similarity uses signed overlap instead of unsigned Jaccard.

use crate::core::algebra::HolographicAlgebra;
use crate::core::entangled::{hash_u64, DEFAULT_RHO_DENOM};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TernaryHVec {
    pub(crate) dim: usize,
    pub(crate) indices: Vec<u32>,
    pub(crate) signs: Vec<i8>,
}

impl TernaryHVec {
    pub fn new_deterministic(dim: usize, seed: u64) -> Self {
        let active_count = (dim / DEFAULT_RHO_DENOM).max(1);
        let mut entries: Vec<(u32, i8)> = Vec::with_capacity(active_count);

        for i in 0..active_count {
            let idx = (hash_u64(seed, i as u64) % dim as u64) as u32;
            let sign = if hash_u64(seed.wrapping_add(0xCAFE), i as u64) % 2 == 0 {
                1i8
            } else {
                -1
            };
            entries.push((idx, sign));
        }

        entries.sort_unstable_by_key(|&(idx, _)| idx);
        entries.dedup_by(|b, a| {
            if a.0 == b.0 {
                a.1 = a.1.saturating_add(b.1).signum();
                true
            } else {
                false
            }
        });
        entries.retain(|&(_, s)| s != 0);

        let max_backfill = active_count * 10;
        let mut counter = 0u64;
        while entries.len() < active_count {
            if counter as usize >= max_backfill {
                break;
            }
            let idx = (hash_u64(seed.wrapping_add(0xDEAD), counter) % dim as u64) as u32;
            let sign = if hash_u64(seed.wrapping_add(0xBEEF), counter) % 2 == 0 {
                1i8
            } else {
                -1
            };
            counter += 1;
            if entries.binary_search_by_key(&idx, |&(i, _)| i).is_err() {
                let pos = entries.partition_point(|&(i, _)| i < idx);
                entries.insert(pos, (idx, sign));
            }
        }

        let (indices, signs): (Vec<u32>, Vec<i8>) = entries.into_iter().unzip();
        Self {
            dim,
            indices,
            signs,
        }
    }

    pub fn from_entries(mut entries: Vec<(u32, i8)>, dim: usize) -> Self {
        entries.sort_unstable_by_key(|&(idx, _)| idx);
        entries.dedup_by(|b, a| {
            if a.0 == b.0 {
                a.1 = a.1.saturating_add(b.1).signum();
                true
            } else {
                false
            }
        });
        entries.retain(|&(_, s)| s != 0);
        let (indices, signs) = entries.into_iter().unzip();
        Self {
            dim,
            indices,
            signs,
        }
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    pub fn signs(&self) -> &[i8] {
        &self.signs
    }

    /// Signed cosine-like similarity: sum(s_a * s_b) for shared indices,
    /// normalized by geometric mean of active counts.
    pub fn similarity(&self, other: &Self) -> f64 {
        let mut dot = 0i64;
        let mut ia = 0;
        let mut ib = 0;
        while ia < self.indices.len() && ib < other.indices.len() {
            match self.indices[ia].cmp(&other.indices[ib]) {
                std::cmp::Ordering::Less => ia += 1,
                std::cmp::Ordering::Greater => ib += 1,
                std::cmp::Ordering::Equal => {
                    dot += self.signs[ia] as i64 * other.signs[ib] as i64;
                    ia += 1;
                    ib += 1;
                }
            }
        }
        let norm = ((self.indices.len() * other.indices.len()) as f64).sqrt();
        if norm < 1e-15 {
            return 0.0;
        }
        (dot as f64 / norm).clamp(-1.0, 1.0)
    }

    /// Bind via signed symmetric difference.
    /// Indices in A only: keep with A's sign.
    /// Indices in B only: keep with B's sign.
    /// Indices in both: cancel (XOR-like).
    pub fn bind(&self, other: &Self) -> Self {
        let mut entries = Vec::with_capacity(self.indices.len() + other.indices.len());
        let mut ia = 0;
        let mut ib = 0;
        while ia < self.indices.len() && ib < other.indices.len() {
            match self.indices[ia].cmp(&other.indices[ib]) {
                std::cmp::Ordering::Less => {
                    entries.push((self.indices[ia], self.signs[ia]));
                    ia += 1;
                }
                std::cmp::Ordering::Greater => {
                    entries.push((other.indices[ib], other.signs[ib]));
                    ib += 1;
                }
                std::cmp::Ordering::Equal => {
                    ia += 1;
                    ib += 1;
                }
            }
        }
        while ia < self.indices.len() {
            entries.push((self.indices[ia], self.signs[ia]));
            ia += 1;
        }
        while ib < other.indices.len() {
            entries.push((other.indices[ib], other.signs[ib]));
            ib += 1;
        }
        let (indices, signs) = entries.into_iter().unzip();
        Self {
            dim: self.dim,
            indices,
            signs,
        }
    }

    /// Bundle via signed majority vote.
    pub fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self {
                dim: 0,
                indices: Vec::new(),
                signs: Vec::new(),
            };
        }
        let dim = vectors[0].dim;
        let n = vectors.len();

        let mut all: Vec<(u32, i8)> = vectors
            .iter()
            .flat_map(|v| v.indices.iter().zip(&v.signs).map(|(&i, &s)| (i, s)))
            .collect();
        all.sort_unstable_by_key(|&(idx, _)| idx);

        if all.is_empty() {
            return Self {
                dim,
                indices: Vec::new(),
                signs: Vec::new(),
            };
        }

        let threshold = (n as i64 + 1) / 2;
        let mut selected: Vec<(u32, i64, u32)> = Vec::new(); // (idx, signed_sum, count)

        let mut current_idx = all[0].0;
        let mut signed_sum = all[0].1 as i64;
        let mut count = 1u32;

        for &(idx, sign) in &all[1..] {
            if idx == current_idx {
                signed_sum += sign as i64;
                count += 1;
            } else {
                if count as i64 >= threshold && signed_sum != 0 {
                    selected.push((current_idx, signed_sum, count));
                }
                current_idx = idx;
                signed_sum = sign as i64;
                count = 1;
            }
        }
        if count as i64 >= threshold && signed_sum != 0 {
            selected.push((current_idx, signed_sum, count));
        }

        let target = (dim / DEFAULT_RHO_DENOM).max(1);
        if selected.len() > target {
            selected.select_nth_unstable_by(target - 1, |a, b| b.2.cmp(&a.2));
            selected.truncate(target);
            selected.sort_unstable_by_key(|&(idx, _, _)| idx);
        }

        let indices: Vec<u32> = selected.iter().map(|&(idx, _, _)| idx).collect();
        let signs: Vec<i8> = selected
            .iter()
            .map(|&(_, s, _)| if s > 0 { 1 } else { -1 })
            .collect();
        Self {
            dim,
            indices,
            signs,
        }
    }

    pub fn permute(&self, shifts: usize) -> Self {
        if self.dim == 0 {
            return self.clone();
        }
        let d = self.dim as u32;
        let shift = (shifts % self.dim) as u32;
        let mut entries: Vec<(u32, i8)> = self
            .indices
            .iter()
            .zip(&self.signs)
            .map(|(&idx, &sign)| ((idx + shift) % d, sign))
            .collect();
        entries.sort_unstable_by_key(|&(idx, _)| idx);
        let (indices, signs) = entries.into_iter().unzip();
        Self {
            dim: self.dim,
            indices,
            signs,
        }
    }

    pub fn to_entangled(&self) -> crate::core::entangled::EntangledHVec {
        crate::core::entangled::EntangledHVec::from_indices(self.indices.clone(), self.dim)
    }

    pub fn from_entangled(e: &crate::core::entangled::EntangledHVec, seed: u64) -> Self {
        let signs: Vec<i8> = e
            .indices()
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if hash_u64(seed, i as u64) % 2 == 0 {
                    1i8
                } else {
                    -1
                }
            })
            .collect();
        Self {
            dim: e.dim,
            indices: e.indices().to_vec(),
            signs,
        }
    }
}

impl HolographicAlgebra for TernaryHVec {
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
        let signs = vec![1i8; indices.len()];
        Self {
            dim,
            indices,
            signs,
        }
    }

    fn active_indices(&self) -> &[u32] {
        &self.indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_similarity() {
        let v = TernaryHVec::new_deterministic(16384, 42);
        let sim = v.similarity(&v);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "Self-similarity should be ~1.0, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_random_pair_low_similarity() {
        let a = TernaryHVec::new_deterministic(16384, 1);
        let b = TernaryHVec::new_deterministic(16384, 2);
        let sim = a.similarity(&b).abs();
        assert!(
            sim < 0.1,
            "Random pair should have near-zero similarity, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_bind_involution() {
        let a = TernaryHVec::new_deterministic(16384, 1);
        let b = TernaryHVec::new_deterministic(16384, 2);
        let recovered = a.bind(&b).bind(&b);
        assert_eq!(recovered.indices, a.indices);
        assert_eq!(recovered.signs, a.signs);
    }

    #[test]
    fn test_bundle_majority() {
        let dim = 16384;
        let base = TernaryHVec::new_deterministic(dim, 100);
        let vecs = vec![
            base.clone(),
            base.clone(),
            base.clone(),
            base.clone(),
            base.clone(),
            TernaryHVec::new_deterministic(dim, 200),
            TernaryHVec::new_deterministic(dim, 300),
        ];
        let bundled = TernaryHVec::bundle(&vecs);
        let sim_base = bundled.similarity(&base);
        let sim_random = bundled.similarity(&TernaryHVec::new_deterministic(dim, 400));
        assert!(sim_base > sim_random, "Bundle should favor majority");
    }

    #[test]
    fn test_signed_discrimination() {
        let dim = 16384;
        let a = TernaryHVec::new_deterministic(dim, 1);
        let mut flipped = a.clone();
        for s in &mut flipped.signs {
            *s *= -1;
        }
        let sim = a.similarity(&flipped);
        assert!(
            sim < -0.5,
            "Sign-flipped should have negative similarity, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_contract() {
        crate::core::algebra::tests::test_algebra_contract_for::<TernaryHVec>(16384);
    }
}
