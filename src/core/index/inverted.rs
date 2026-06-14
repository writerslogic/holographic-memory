use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

use crate::core::types::RetrievalResult;

/// Sparse Inverted Index optimized for fixed-sparsity Jaccard (m = D/256).
#[derive(Default, Serialize, Deserialize)]
pub struct SparseInvertedIndex {
    /// For each dimension [0, D), the sorted list of doc_ids containing it.
    pub postings: Vec<Vec<u32>>,
    /// Fixed sparsity parameter (m).
    pub m: usize,
    /// Total number of dimensions (D).
    pub dimensions: usize,
    /// Number of documents in index.
    pub doc_count: usize,
    /// Document frequencies (count of docs per dimension).
    pub df: Vec<u32>,
}

impl SparseInvertedIndex {
    pub fn new(dimensions: usize, m: usize) -> Self {
        Self {
            postings: vec![Vec::new(); dimensions],
            m,
            dimensions,
            doc_count: 0,
            df: vec![0; dimensions],
        }
    }

    /// Add a document to the index. Assumes doc_ids are added in increasing order.
    pub fn add_doc(&mut self, doc_id: u32, dims: &[u32]) {
        for &dim in dims {
            if (dim as usize) < self.dimensions {
                self.postings[dim as usize].push(doc_id);
                self.df[dim as usize] += 1;
            }
        }
        self.doc_count += 1;
    }

    /// Ensure all posting lists are sorted (if not added monotonically).
    pub fn finalize(&mut self) {
        for list in &mut self.postings {
            list.sort_unstable();
        }
    }

    /// Perform k-NN query using the epoch-stamped accumulator strategy.
    pub fn query(
        &self,
        query_dims: &[u32],
        k: usize,
        accumulator: &mut Accumulator,
    ) -> Vec<RetrievalResult> {
        debug_assert!(
            self.postings
                .iter()
                .all(|list| list.windows(2).all(|w| w[0] <= w[1])),
            "posting lists must be sorted before query"
        );

        if query_dims.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }

        // 1. Order query dimensions by increasing document frequency (heuristic pruning)
        let mut sorted_qdims: Vec<u32> = query_dims.to_vec();
        sorted_qdims
            .sort_unstable_by_key(|&d| self.df.get(d as usize).cloned().unwrap_or(u32::MAX));

        // 2. Clear and increment epoch
        accumulator.next_epoch();

        // 3. Accumulate intersections
        for &dim in &sorted_qdims {
            if let Some(list) = self.postings.get(dim as usize) {
                for &doc_id in list {
                    accumulator.increment(doc_id);
                }
            }
        }

        // 4. Compute Jaccard and extract Top-K
        // Jaccard = inter / (m_q + m_d - inter).
        // With fixed sparsity m_q = m_d = self.m.
        let m_f = self.m as f64;
        let mut heap = BinaryHeap::with_capacity(k + 1);

        // We only need to check doc_ids that were touched
        for &doc_id in &accumulator.touched {
            let inter = accumulator.get_count(doc_id) as f64;
            let denom = 2.0 * m_f - inter;
            let similarity = if denom > f64::EPSILON { inter / denom } else { 1.0 };

            heap.push(RetrievalResult {
                id: doc_id.to_string(), // In practice, we'll map doc_id back to String ID
                similarity,
            });

            if heap.len() > k {
                heap.pop();
            }
        }

        // With inverted Ord (smaller similarity = Greater),
        // into_sorted_vec (ascending Ord) returns DESCENDING similarity.
        heap.into_sorted_vec()
    }
}

/// Thread-local or reusable accumulator to avoid allocations.
pub struct Accumulator {
    counts: Vec<u16>,
    seen: Vec<u32>,
    epoch: u32,
    touched: Vec<u32>,
}

impl Accumulator {
    pub fn new(max_docs: usize) -> Self {
        Self {
            counts: vec![0; max_docs],
            seen: vec![0; max_docs],
            epoch: 0,
            touched: Vec::with_capacity(1024),
        }
    }

    /// Prepare for a new query.
    pub fn next_epoch(&mut self) {
        self.epoch += 1;
        self.touched.clear();
        // If epoch wraps, we must zero the 'seen' array.
        if self.epoch == 0 {
            self.seen.fill(0);
            self.epoch = 1;
        }
    }

    #[inline(always)]
    pub fn increment(&mut self, doc_id: u32) {
        let idx = doc_id as usize;
        if idx >= self.seen.len() {
            // Grow arrays if needed (though max_docs should be pre-sized)
            let new_size = (idx + 1).max(self.seen.len() * 2);
            self.seen.resize(new_size, 0);
            self.counts.resize(new_size, 0);
        }

        if self.seen[idx] != self.epoch {
            self.seen[idx] = self.epoch;
            self.counts[idx] = 1;
            self.touched.push(doc_id);
        } else {
            self.counts[idx] += 1;
        }
    }

    #[inline(always)]
    pub fn get_count(&self, doc_id: u32) -> u16 {
        let idx = doc_id as usize;
        if idx < self.seen.len() && self.seen[idx] == self.epoch {
            self.counts[idx]
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverted_index_basic() {
        let mut index = SparseInvertedIndex::new(1000, 4);
        index.add_doc(0, &[10, 20, 30, 40]);
        index.add_doc(1, &[10, 25, 35, 45]);
        index.add_doc(2, &[10, 20, 50, 60]);
        index.finalize();

        let mut acc = Accumulator::new(10);
        // Query overlaps with doc 0 (3 hits), doc 1 (1 hit), doc 2 (2 hits)
        let results = index.query(&[10, 20, 30], 5, &mut acc);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "0");
        // inter=3, m=4. J = 3 / (4 + 4 - 3) = 3/5 = 0.6
        assert!((results[0].similarity - 0.6).abs() < 1e-6);

        assert_eq!(results[1].id, "2");
        // inter=2, m=4. J = 2 / (4 + 4 - 2) = 2/6 = 0.333
        assert!((results[1].similarity - 0.333333).abs() < 1e-5);
    }
}
