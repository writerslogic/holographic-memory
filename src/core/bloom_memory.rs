// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Holographic Bloom Memory.
//!
//! Stores items via Bloom-filter bundling (set union) and retrieves them
//! with density-corrected containment similarity. Raw containment has a
//! floor at the bundle density (non-members score ~density), so we subtract
//! that floor and normalize: corrected = (raw - density) / (1 - density).
//!
//! Members score 1.0 (exact containment). Non-members score ~0.0.
//! Capacity at D=16384, density 1/256: hundreds of items with d' > 5.

use crate::core::entangled::EntangledHVec;

pub struct HolographicBloomMemory {
    dim: usize,
    bundle: EntangledHVec,
    item_count: usize,
}

impl HolographicBloomMemory {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            bundle: EntangledHVec::from_indices(Vec::new(), dim),
            item_count: 0,
        }
    }

    pub fn insert(&mut self, item: &EntangledHVec) {
        self.bundle = EntangledHVec::bundle_bloom(&[self.bundle.clone(), item.clone()]);
        self.item_count += 1;
    }

    pub fn insert_batch(&mut self, items: &[EntangledHVec]) {
        let mut all = vec![self.bundle.clone()];
        all.extend(items.iter().cloned());
        self.bundle = EntangledHVec::bundle_bloom(&all);
        self.item_count += items.len();
    }

    /// Density-corrected containment score.
    pub fn contains(&self, item: &EntangledHVec) -> f64 {
        item.corrected_containment(&self.bundle)
    }

    pub fn query_candidates(
        &self,
        candidates: &[EntangledHVec],
        threshold: f64,
    ) -> Vec<(usize, f64)> {
        candidates
            .iter()
            .enumerate()
            .map(|(i, item)| (i, self.contains(item)))
            .filter(|&(_, score)| score >= threshold)
            .collect()
    }

    pub fn item_count(&self) -> usize {
        self.item_count
    }
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn density(&self) -> f64 {
        self.bundle.indices().len() as f64 / self.dim as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_member_retrieval() {
        let dim = 16384;
        let mut mem = HolographicBloomMemory::new(dim);
        let items: Vec<EntangledHVec> = (0..50)
            .map(|i| EntangledHVec::new_deterministic(dim, i * 100))
            .collect();
        mem.insert_batch(&items);

        for (i, item) in items.iter().enumerate() {
            let score = mem.contains(item);
            assert!(
                (score - 1.0).abs() < 1e-10,
                "Member {} should have score 1.0, got {:.6}",
                i,
                score
            );
        }
    }

    #[test]
    fn test_non_member_discrimination() {
        let dim = 16384;
        let mut mem = HolographicBloomMemory::new(dim);
        let items: Vec<EntangledHVec> = (0..100)
            .map(|i| EntangledHVec::new_deterministic(dim, i * 100))
            .collect();
        mem.insert_batch(&items);

        let mut non_member_scores = Vec::new();
        for i in 0..100 {
            let non_member = EntangledHVec::new_deterministic(dim, 999000 + i);
            non_member_scores.push(mem.contains(&non_member));
        }

        let max_score = non_member_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_score < 0.5,
            "Non-member max score should be <0.5, got {:.4}",
            max_score
        );
    }

    #[test]
    fn test_capacity_200_items() {
        let dim = 16384;
        let mut mem = HolographicBloomMemory::new(dim);
        let items: Vec<EntangledHVec> = (0..200)
            .map(|i| EntangledHVec::new_deterministic(dim, i * 100))
            .collect();
        mem.insert_batch(&items);

        let mut members_found = 0;
        for item in &items {
            if mem.contains(item) > 0.5 {
                members_found += 1;
            }
        }
        assert_eq!(members_found, 200, "All 200 members should be found");

        let mut false_positives = 0;
        for i in 0..200 {
            let nm = EntangledHVec::new_deterministic(dim, 777000 + i as u64);
            if mem.contains(&nm) > 0.5 {
                false_positives += 1;
            }
        }
        assert!(
            false_positives < 5,
            "False positives at 200 items should be <5, got {}",
            false_positives
        );
    }

    #[test]
    fn test_capacity_500_items() {
        let dim = 16384;
        let mut mem = HolographicBloomMemory::new(dim);
        let items: Vec<EntangledHVec> = (0..500)
            .map(|i| EntangledHVec::new_deterministic(dim, i * 100))
            .collect();
        mem.insert_batch(&items);

        let mut members_found = 0;
        for item in &items {
            if mem.contains(item) > 0.3 {
                members_found += 1;
            }
        }
        let recall = members_found as f64 / 500.0;
        assert!(
            recall > 0.95,
            "Recall at 500 items should be >0.95, got {:.4}",
            recall
        );
    }
}
