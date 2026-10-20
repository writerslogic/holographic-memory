// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::posting::PostingShard;

fn compute_idf(total_docs: u32, df: u32) -> f32 {
    (1.0 + total_docs as f64 / (1.0 + df as f64)).ln() as f32
}

/// Inverse document frequency weights per dimension.
/// Provides proportional IDF clipping as a poisoning defense.
pub struct IdfWeights {
    weights: Vec<f32>,
    total_docs: u32,
    clip_factor: f64,
}

impl IdfWeights {
    pub fn new(dim: usize, clip_factor: f64) -> Self {
        Self {
            weights: vec![1.0; dim],
            total_docs: 0,
            clip_factor,
        }
    }

    /// Increment total_docs and update weights for the inserted dimensions.
    pub fn update_insert(&mut self, indices: &[u32], postings: &PostingShard) {
        self.total_docs += 1;
        for &idx in indices {
            let df = postings.doc_freq(idx);
            self.weights[idx as usize] = self.raw_idf(df);
        }
    }

    /// Decrement total_docs and update weights for the removed dimensions.
    #[allow(dead_code)]
    pub fn update_remove(&mut self, indices: &[u32], postings: &PostingShard) {
        self.total_docs = self.total_docs.saturating_sub(1);
        for &idx in indices {
            let df = postings.doc_freq(idx);
            self.weights[idx as usize] = self.raw_idf(df);
        }
    }

    /// Recompute all weights from scratch.
    pub fn recompute(&mut self, postings: &PostingShard, total_docs: u32) {
        self.total_docs = total_docs;
        let td = self.total_docs;
        for (i, w) in self.weights.iter_mut().enumerate() {
            let df = postings.doc_freq(i as u32);
            *w = compute_idf(td, df);
        }
        self.clip();
    }

    fn raw_idf(&self, df: u32) -> f32 {
        compute_idf(self.total_docs, df)
    }

    /// Clip weights to clip_factor * median to defeat poisoning attacks.
    pub fn clip(&mut self) {
        let mut sorted: Vec<f32> = self.weights.iter().copied().filter(|w| *w > 0.0).collect();
        if sorted.is_empty() {
            return;
        }
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let cap = (self.clip_factor * median as f64) as f32;
        for w in &mut self.weights {
            if *w > cap {
                *w = cap;
            }
        }
    }

    pub fn weight(&self, dim_index: u32) -> f32 {
        self.weights.get(dim_index as usize).copied().unwrap_or(1.0)
    }

    #[allow(dead_code)]
    pub fn total_docs(&self) -> u32 {
        self.total_docs
    }

    pub fn weights_for(&self, indices: &[u32]) -> Vec<f32> {
        indices.iter().map(|&d| self.weight(d)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tombstone::TombstoneMap;

    #[test]
    fn test_idf_basic() {
        let mut postings = PostingShard::new(100);
        let mut idf = IdfWeights::new(100, 3.0);

        postings.insert(0, &[5, 10]);
        postings.insert(1, &[5]);
        idf.recompute(&postings, 2);

        // dim 5: df=2, dim 10: df=1. Higher df = lower IDF.
        assert!(idf.weight(5) < idf.weight(10));
    }

    #[test]
    fn test_idf_clip() {
        let mut idf = IdfWeights::new(10, 2.0);
        idf.weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0];
        idf.clip();
        // median is 1.0, cap = 2.0 * 1.0 = 2.0
        assert!(idf.weight(8) <= 2.0);
        assert!((idf.weight(0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_idf_recompute() {
        let mut postings = PostingShard::new(20);
        let tombstones = TombstoneMap::new();
        postings.insert(0, &[0, 1, 2]);
        postings.insert(1, &[0, 1]);
        postings.insert(2, &[0]);

        let mut idf = IdfWeights::new(20, 3.0);
        idf.recompute(&postings, 3);

        // dim 0: df=3, dim 1: df=2, dim 2: df=1
        assert!(idf.weight(0) < idf.weight(1));
        assert!(idf.weight(1) < idf.weight(2));
        let _ = tombstones; // used by overlap_counts, not directly here
    }
}
