// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Counting holographic membership store.
//!
//! A variant of [`super::bloom_memory::HolographicBloomMemory`] that keeps a
//! per-index HIT COUNT instead of a one-bit OR union. The binary union saturates
//! to all-ones as items accumulate, at which point non-members become
//! indistinguishable from members and discrimination collapses to chance. The
//! count preserves the signal the union throws away -- each member adds +1 to each
//! of its k active indices -- and a Poisson z-score readout recovers that shift far
//! past the binary wall.
//!
//! Measured (D=16384, k=64, 20 seeds, member-vs-non-member AUC): the binary
//! readout walls at AUC<0.95 by ~1000 items and is at chance by ~2000; the counting
//! readout holds AUC>0.95 to ~4000 and degrades gracefully (0.84 at 8000). A ~4x
//! usable-capacity gain at matched dimension. See `docs/PREREGISTRATION-binding-
//! readout.md` §19 and `src/bin/bloom-wall.rs`.
//!
//! Cost: one `u32` per index rather than one bit -- this trades a Bloom filter's
//! bit-compactness for capacity. It remains a deterministic integer fold of its
//! inserts, so the store is an exact, replay-verifiable function of what was added.

use crate::core::entangled::EntangledHVec;

/// A z-score threshold of 3.0 bounds the per-query non-member false-positive rate
/// to roughly 0.1% under the Poisson null (the summed statistic is ~N(0,1) for a
/// non-member), independent of load.
pub const DEFAULT_Z_THRESHOLD: f64 = 3.0;

pub struct CountingBloomMemory {
    dim: usize,
    counts: Vec<u32>,
    /// Σ over inserted items of their active-index count -- the Poisson null rate
    /// is λ = active_insertions / dim.
    active_insertions: u64,
    item_count: usize,
}

impl CountingBloomMemory {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            counts: vec![0; dim],
            active_insertions: 0,
            item_count: 0,
        }
    }

    /// Add one item, incrementing the count at each of its active indices.
    pub fn insert(&mut self, item: &EntangledHVec) {
        for &i in item.indices() {
            // Indices are < dim by construction; the bounds check guards against a
            // mismatched-dimension item rather than trusting the caller.
            if let Some(c) = self.counts.get_mut(i as usize) {
                *c = c.saturating_add(1);
                self.active_insertions += 1;
            }
        }
        self.item_count += 1;
    }

    pub fn insert_batch(&mut self, items: &[EntangledHVec]) {
        for item in items {
            self.insert(item);
        }
    }

    /// Poisson z-score membership statistic.
    ///
    /// Under the non-member null each queried index has count ~ Poisson(λ), so the
    /// summed excess `Σ(count_i − λ)` over the query's k indices is ~N(0, kλ). A
    /// member shifted every one of its indices up by +1, moving the sum by +k. The
    /// returned standardized value `Σ(count_i − λ) / √(kλ)` is therefore ~0 for a
    /// non-member and ~√(k/λ) for a member. (Ranking is invariant to the √(kλ)
    /// scale at fixed k; the normalization is what makes the value an interpretable
    /// z-score for thresholding.)
    pub fn membership_score(&self, item: &EntangledHVec) -> f64 {
        let ix = item.indices();
        if ix.is_empty() {
            return 0.0;
        }
        let lambda = self.active_insertions as f64 / self.dim.max(1) as f64;
        let excess: f64 = ix
            .iter()
            .map(|&i| self.counts.get(i as usize).copied().unwrap_or(0) as f64 - lambda)
            .sum();
        let std = (ix.len() as f64 * lambda).sqrt();
        if std <= f64::EPSILON {
            // Empty or near-empty store: any present index is unambiguous signal.
            return if excess > 0.0 { f64::INFINITY } else { 0.0 };
        }
        excess / std
    }

    /// Whether `item` is a member at the given z-score threshold.
    pub fn contains(&self, item: &EntangledHVec, z_threshold: f64) -> bool {
        self.membership_score(item) >= z_threshold
    }

    /// Candidate indices whose membership z-score meets `z_threshold`, with scores.
    pub fn query_candidates(
        &self,
        candidates: &[EntangledHVec],
        z_threshold: f64,
    ) -> Vec<(usize, f64)> {
        candidates
            .iter()
            .enumerate()
            .map(|(i, item)| (i, self.membership_score(item)))
            .filter(|&(_, z)| z >= z_threshold)
            .collect()
    }

    pub fn item_count(&self) -> usize {
        self.item_count
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Poisson null rate λ = Σk / dim (mean count per index).
    pub fn mean_count(&self) -> f64 {
        self.active_insertions as f64 / self.dim.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(dim: usize, n: u64) -> CountingBloomMemory {
        let mut mem = CountingBloomMemory::new(dim);
        let items: Vec<EntangledHVec> = (0..n)
            .map(|i| EntangledHVec::new_deterministic(dim, i * 100))
            .collect();
        mem.insert_batch(&items);
        mem
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    #[test]
    fn single_member_scores_high_nonmember_near_zero() {
        let dim = 16384;
        let mut mem = CountingBloomMemory::new(dim);
        let item = EntangledHVec::new_deterministic(dim, 42);
        mem.insert(&item);
        let member = mem.membership_score(&item);
        assert!(member > 20.0, "lone member must stand out, got {member:.2}");
        // At near-empty load the null variance √(kλ) is tiny, so an individual
        // non-member sharing an index or two is noisy; the null MEAN over a sample
        // still sits near zero, and the member dwarfs any individual non-member.
        let nms: Vec<f64> = (0..50)
            .map(|j| mem.membership_score(&EntangledHVec::new_deterministic(dim, 900_000 + j)))
            .collect();
        assert!(
            mean(&nms).abs() < 1.0,
            "non-member mean should sit near the null"
        );
        let nm_max = nms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            member > 5.0 * nm_max.max(1.0),
            "member must dwarf non-members"
        );
    }

    #[test]
    fn empty_query_is_zero() {
        let mem = store(16384, 100);
        let empty = EntangledHVec::from_indices(Vec::new(), 16384);
        assert_eq!(mem.membership_score(&empty), 0.0);
    }

    #[test]
    fn discriminates_at_1000_items() {
        // Where the binary readout is already collapsing (AUC~0.87), counting is
        // near-perfect. Assert a comfortable, theory-predicted margin.
        let dim = 16384;
        let mem = store(dim, 1000);
        let members: Vec<f64> = (0..1000)
            .map(|i| mem.membership_score(&EntangledHVec::new_deterministic(dim, i * 100)))
            .collect();
        let nonmembers: Vec<f64> = (0..200)
            .map(|j| mem.membership_score(&EntangledHVec::new_deterministic(dim, 5_000_000 + j)))
            .collect();
        let m = mean(&members);
        let nm = mean(&nonmembers);
        assert!(m > 3.0, "member mean z should clear 3, got {m:.2}");
        assert!(
            nm.abs() < 1.0,
            "non-member mean z should sit near 0, got {nm:.2}"
        );
        let fpr = nonmembers.iter().filter(|&&z| z >= 3.0).count() as f64 / nonmembers.len() as f64;
        assert!(
            fpr < 0.05,
            "non-member FPR@z>3 should be small, got {fpr:.3}"
        );
    }

    #[test]
    fn still_discriminates_where_binary_is_dead() {
        // At 2000 items the binary OR-union readout is at chance (AUC~0.51);
        // counting still separates the means by a clear margin (~4x capacity).
        let dim = 16384;
        let mem = store(dim, 2000);
        let members: Vec<f64> = (0..400)
            .map(|i| mem.membership_score(&EntangledHVec::new_deterministic(dim, i * 100)))
            .collect();
        let nonmembers: Vec<f64> = (0..400)
            .map(|j| mem.membership_score(&EntangledHVec::new_deterministic(dim, 6_000_000 + j)))
            .collect();
        assert!(
            mean(&members) - mean(&nonmembers) > 2.0,
            "counting must still separate at 2000 items: members {:.2} vs non {:.2}",
            mean(&members),
            mean(&nonmembers)
        );
    }
}
