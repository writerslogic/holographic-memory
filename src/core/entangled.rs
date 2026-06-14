use fxhash::FxHasher;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::hash::Hasher;

/// Default sparsity denominator: rho = 1/256 of dimensions are active.
///
/// At D=16384 this yields ~64 active indices. This is an engineering choice
/// balancing memory (256x smaller than dense) against discrimination; the
/// cited dense ±1 papers (Laiho 2015, Frady 2021) use different representations.
pub(crate) const DEFAULT_RHO_DENOM: usize = 256;

/// Number of entanglement groups (hash-seeded, not learned).
///
/// 64 groups at D=16384 gives ~1 active index per group on average.
/// Groups hash into the full `[0, dim)` range (they do NOT partition the
/// dimension space — collisions across groups are possible).
const NUM_ENTANGLEMENTS: usize = 64;

/// Entangled Sparse Hypervector.
///
/// Instead of a dense bitvec (~50% active), stores only sorted active indices.
/// Sparsity rho = 1/256 means ~D/256 active bits out of D total.
/// Indices are grouped into `NUM_ENTANGLEMENTS` entanglement groups,
/// each generated from a seeded hash.
///
/// Benefits:
/// - 256x less memory per vector at D=16384
/// - Hamming via sorted merge (O(k) where k = active count)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntangledHVec {
    pub(crate) dim: usize,
    /// Sorted active bit indices (rho * dim total).
    pub(crate) indices: Vec<u32>,
}

impl EntangledHVec {
    /// Create a random entangled sparse hypervector.
    /// Uses entanglement groups when dim is large enough, otherwise direct hash generation.
    /// After dedup, if birthday collisions reduced the count below `active_count`,
    /// deterministic backfill slots are generated until the exact count is reached.
    pub fn new_deterministic(dim: usize, seed: u64) -> Self {
        let active_count = (dim / DEFAULT_RHO_DENOM).max(1);
        let mut indices = Vec::with_capacity(active_count);

        let per_group = active_count / NUM_ENTANGLEMENTS;
        if per_group > 0 {
            // Standard entangled generation with groups
            for e in 0..NUM_ENTANGLEMENTS {
                let group_seed = hash_u64(seed, e as u64);
                for p in 0..per_group {
                    let idx = hash_u64(group_seed, p as u64) % dim as u64;
                    indices.push(idx as u32);
                }
            }
        } else {
            // Fallback for small dims: generate active_count indices directly
            for p in 0..active_count {
                let idx = hash_u64(seed, p as u64) % dim as u64;
                indices.push(idx as u32);
            }
        }

        indices.sort_unstable();
        indices.dedup();

        // Birthday collision backfill: if collisions reduced count, fill remaining
        // slots deterministically until we reach exactly active_count.
        let max_backfill_attempts = active_count * 10;
        let mut backfill_counter = 0u64;
        while indices.len() < active_count {
            if backfill_counter as usize >= max_backfill_attempts {
                break;
            }
            let idx = (hash_u64(seed.wrapping_add(0xDEAD), backfill_counter) % dim as u64) as u32;
            backfill_counter += 1;
            if indices.binary_search(&idx).is_err() {
                let pos = indices.partition_point(|&x| x < idx);
                indices.insert(pos, idx);
            }
        }

        Self { dim, indices }
    }

    /// Create directly from a set of active indices.
    /// Indices must be sorted, deduplicated, and within `[0, dim)`.
    pub fn from_indices(indices: Vec<u32>, dim: usize) -> Self {
        debug_assert!(
            indices.windows(2).all(|w| w[0] < w[1]),
            "from_indices: indices must be sorted and deduplicated"
        );
        debug_assert!(
            indices.last().is_none_or(|&last| (last as usize) < dim),
            "from_indices: indices must be within [0, dim)"
        );
        Self { dim, indices }
    }

    /// Create from a dense float vector via sparse ternary sketching.
    /// Selects top rho*dim indices by Achlioptas ternary projection magnitude.
    pub fn from_dense(dense: &[f32], target_dim: usize) -> Self {
        let active_count = (target_dim / DEFAULT_RHO_DENOM).max(1);
        if dense.is_empty() {
            return Self {
                dim: target_dim,
                indices: Vec::new(),
            };
        }

        // Compute random projection dot products.
        // Achlioptas (2003) sparse ternary projection:
        // P(+1) = P(-1) = 1/6, P(0) = 2/3.
        // Preserves JL distances with 3x fewer multiplications.
        // Uses FxHash per (dim, input_pos) pair for deterministic ternary values
        // instead of StdRng per dimension (avoids target_dim RNG inits).
        let mut projections: Vec<(u32, f64)> = (0..target_dim)
            .map(|i| {
                let mut dot_product = 0.0f64;
                for (j, &val) in dense.iter().enumerate() {
                    let r = hash_u64(i as u64, j as u64) % 6;
                    if r == 0 {
                        dot_product += val as f64;
                    } else if r == 5 {
                        dot_product -= val as f64;
                    }
                }
                (i as u32, dot_product.abs())
            })
            .collect();

        // Select top active_count by magnitude
        if projections.len() > active_count {
            projections.select_nth_unstable_by(active_count - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            projections.truncate(active_count);
        }

        let mut indices: Vec<u32> = projections.into_iter().map(|(idx, _)| idx).collect();
        indices.sort_unstable();

        Self {
            dim: target_dim,
            indices,
        }
    }

    /// Encode a scalar into a sparse hypervector via permuted sliding window.
    ///
    /// The normalized value selects a contiguous window of `active_count`
    /// logical slots in `[0, target_dim - active_count]`. Each logical slot
    /// is then scattered across the index space by multiplying with a prime
    /// constant modulo `target_dim` (golden-ratio permutation). This gives:
    /// - Nearby values share most window slots → high Jaccard
    /// - Distant values share few slots → low Jaccard
    /// - Monotonic: similarity decreases with scalar distance
    pub fn from_scalar(value: f64, min_val: f64, max_val: f64, target_dim: usize) -> Self {
        let range = max_val - min_val;
        let normalized = if range.abs() < f64::EPSILON {
            0.5
        } else {
            ((value - min_val) / range).clamp(0.0, 1.0)
        };
        let active_count = (target_dim / DEFAULT_RHO_DENOM).max(1);

        // Map normalized value to a starting offset in the index space.
        // We use (target_dim - active_count) possible start positions.
        let max_offset = (target_dim - active_count) as f64;
        let start_offset = (normalized * max_offset).round() as usize;

        // Deterministic permutation for global spread with local overlap monotonicity.
        // Must be prime so gcd(PERM_PRIME, dim) = 1 for all dim, ensuring a
        // full-period bijection. (The old golden-ratio constant was divisible by 5,
        // restricting output to 20% of dimensions for dim % 5 == 0.)
        const PERM_PRIME: u64 = 0x9E3779B97F4A7C55;

        let mut indices = Vec::with_capacity(active_count);
        for i in 0..active_count {
            let logical_idx = (start_offset + i) as u64;
            // Deterministic shuffle: (logical_idx * prime) % target_dim
            let idx = (logical_idx.wrapping_mul(PERM_PRIME) % target_dim as u64) as u32;
            indices.push(idx);
        }

        indices.sort_unstable();
        // PERM_PRIME is prime ⇒ gcd(PERM_PRIME, target_dim)=1 ⇒ mapping is injective.
        // Dedup is a defensive no-op; no backfill needed (unlike random() where
        // hash collisions across entanglement groups are possible).
        indices.dedup();

        Self {
            dim: target_dim,
            indices,
        }
    }

    /// Active bit indices (sorted).
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Convert sorted indices to deltas (gaps between consecutive indices).
    /// First element is the absolute value.
    pub fn to_deltas(&self) -> Vec<u32> {
        if self.indices.is_empty() {
            return Vec::new();
        }
        let mut deltas = Vec::with_capacity(self.indices.len());
        deltas.push(self.indices[0]);
        for i in 1..self.indices.len() {
            deltas.push(self.indices[i] - self.indices[i - 1]);
        }
        deltas
    }

    /// Restore absolute indices from deltas.
    pub fn from_deltas(deltas: &[u32], dim: usize) -> Self {
        if deltas.is_empty() {
            return Self {
                dim,
                indices: Vec::new(),
            };
        }
        let mut indices = Vec::with_capacity(deltas.len());
        let mut current: u32 = 0;
        for &d in deltas {
            current = match current.checked_add(d) {
                Some(v) if (v as usize) < dim => v,
                _ => break,
            };
            indices.push(current);
        }
        Self { dim, indices }
    }

    /// Hamming distance via sorted-merge intersection count on sparse index sets.
    pub fn hamming(&self, other: &Self) -> u32 {
        let intersection =
            super::intersection::sparse_intersection_count(&self.indices, &other.indices);
        (self.indices.len() + other.indices.len() - 2 * intersection) as u32
    }

    /// Jaccard similarity: |A∩B| / |A∪B|.
    /// Proper metric for sparse index sets (ρ=1/256).
    /// The old 1−hamming/dim collapsed to ~0.992 for all sparse pairs.
    pub fn similarity(&self, other: &Self) -> f64 {
        let intersection =
            super::intersection::sparse_intersection_count(&self.indices, &other.indices);
        let union = self.indices.len() + other.indices.len() - intersection;
        if union == 0 {
            return 1.0;
        }
        intersection as f64 / union as f64
    }

    /// Entangled bind: symmetric XOR of index sets.
    /// For sparse vectors, XOR of active sets = symmetric difference.
    pub fn bind(&self, other: &Self) -> Self {
        let indices = sorted_symmetric_difference(&self.indices, &other.indices);
        Self {
            dim: self.dim,
            indices,
        }
    }

    /// Permute by shifting all indices by `shifts` modulo dim.
    pub fn permute(&self, shifts: usize) -> Self {
        if self.dim == 0 {
            return self.clone();
        }
        let dim = self.dim as u32;
        let shift = (shifts % self.dim) as u32;
        let mut indices: Vec<u32> = self
            .indices
            .iter()
            .map(|&idx| (idx + shift) % dim)
            .collect();
        indices.sort_unstable();
        Self {
            dim: self.dim,
            indices,
        }
    }

    /// Bundle multiple sparse vectors with threshold at rho.
    /// Counts per-index frequency, keeps only those above threshold.
    pub fn bundle<V: Borrow<Self>>(vectors: &[V]) -> Self {
        if vectors.is_empty() {
            return Self {
                dim: 0,
                indices: Vec::new(),
            };
        }
        let dim = vectors[0].borrow().dim;
        let n = vectors.len();

        let mut all_indices: Vec<u32> = vectors
            .iter()
            .flat_map(|v| v.borrow().indices.iter().copied())
            .collect();
        all_indices.sort_unstable();

        if all_indices.is_empty() {
            return Self {
                dim,
                indices: Vec::new(),
            };
        }

        // Run-length count on sorted indices, filtering by threshold while counting.
        let threshold = (n as u32).div_ceil(2);
        let mut selected: Vec<(u32, u32)> = Vec::new();

        let mut current = all_indices[0];
        let mut count: u32 = 1;
        for &idx in &all_indices[1..] {
            if idx == current {
                count += 1;
            } else {
                if count >= threshold {
                    selected.push((current, count));
                }
                current = idx;
                count = 1;
            }
        }
        if count >= threshold {
            selected.push((current, count));
        }

        let target_count = (dim / DEFAULT_RHO_DENOM).max(1);

        // If too many pass threshold, keep top target_count by count
        if selected.len() > target_count {
            selected.select_nth_unstable_by(target_count - 1, |a, b| b.1.cmp(&a.1));
            selected.truncate(target_count);
        }

        let mut indices: Vec<u32> = selected.into_iter().map(|(idx, _)| idx).collect();
        indices.sort_unstable();

        Self { dim, indices }
    }

}

/// Fast non-crypto hash for seed mixing.
fn hash_u64(a: u64, b: u64) -> u64 {
    let mut h = FxHasher::default();
    h.write_u64(a);
    h.write_u64(b);
    h.finish()
}

/// Symmetric difference of two sorted slices (XOR for index sets).
pub(crate) fn sorted_symmetric_difference(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                // In both → cancel (XOR)
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Count of symmetric difference (without collecting).
pub(crate) fn sorted_symmetric_difference_count(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                count += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                count += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    count += a.len() - i;
    count += b.len() - j;
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entangled_self_similarity() {
        let v = EntangledHVec::new_deterministic(16384, 42);
        assert!(
            (v.similarity(&v) - 1.0).abs() < 0.0001,
            "Self-similarity should be 1.0"
        );
    }

    #[test]
    fn test_entangled_random_pair_distance() {
        // Two random sparse vectors should have near-zero Jaccard overlap
        let a = EntangledHVec::new_deterministic(16384, 1);
        let b = EntangledHVec::new_deterministic(16384, 2);
        let sim = a.similarity(&b);
        // Jaccard for random sparse pairs: ~64*64/16384 intersect / ~128 union ≈ 0.002
        assert!(
            sim < 0.05,
            "Random sparse pair Jaccard {:.4} should be near 0",
            sim
        );
    }

    #[test]
    fn test_entangled_bind_involution() {
        let a = EntangledHVec::new_deterministic(16384, 1);
        let b = EntangledHVec::new_deterministic(16384, 2);
        let ab = a.bind(&b);
        let recovered = ab.bind(&b);
        // XOR is involutory: (A⊕B)⊕B = A
        assert_eq!(recovered.indices, a.indices);
    }

    #[test]
    fn test_entangled_bundle_majority() {
        let dim = 16384;
        let base = EntangledHVec::new_deterministic(dim, 100);
        // Create 5 copies of base + 2 random noise
        let mut vecs = vec![base.clone(); 5];
        vecs.push(EntangledHVec::new_deterministic(dim, 200));
        vecs.push(EntangledHVec::new_deterministic(dim, 300));
        let bundled = EntangledHVec::bundle(&vecs);
        // Bundled should be most similar to base (majority)
        let sim_base = bundled.similarity(&base);
        let sim_random = bundled.similarity(&EntangledHVec::new_deterministic(dim, 400));
        assert!(
            sim_base > sim_random,
            "Bundle should be closer to majority element"
        );
    }

    #[test]
    fn test_from_dense_produces_sparse() {
        let dense: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let e = EntangledHVec::from_dense(&dense, 10000);
        assert_eq!(e.dim, 10000);
        // Should have ~dim/256 active indices
        let expected = 10000 / 256;
        assert_eq!(e.indices.len(), expected);
    }

    #[test]
    fn test_from_scalar() {
        let e = EntangledHVec::from_scalar(0.5, 0.0, 1.0, 10000);
        assert_eq!(e.dim, 10000);
        assert!(!e.indices.is_empty());
    }

    #[test]
    fn test_from_scalar_locality() {
        let dim = 16384;
        let base_val = 0.5;
        let v_base = EntangledHVec::from_scalar(base_val, 0.0, 1.0, dim);

        let mut prev_sim = 1.0;
        // Check points moving away from 0.5
        for i in 1..20 {
            let offset = i as f64 * 0.02;
            let v_curr = EntangledHVec::from_scalar(base_val + offset, 0.0, 1.0, dim);
            let sim = v_base.similarity(&v_curr);

            assert!(
                sim <= prev_sim,
                "Monotonicity violation at offset {}: sim={:.4} was prev={:.4}",
                offset,
                sim,
                prev_sim
            );
            prev_sim = sim;
        }
    }

}
