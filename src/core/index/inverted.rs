// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

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
    lengths: Vec<usize>,
}

impl SparseInvertedIndex {
    pub fn new(dimensions: usize, m: usize) -> Self {
        Self {
            postings: vec![Vec::new(); dimensions],
            m,
            dimensions,
            doc_count: 0,
            df: vec![0; dimensions],
            lengths: Vec::new(),
        }
    }

    /// Add a document to the index. Assumes doc_ids are added in increasing order.
    pub fn add_doc(&mut self, doc_id: u32, dims: &[u32]) {
        for &dim in dims {
            if (dim as usize) < self.dimensions {
                let list = &mut self.postings[dim as usize];
                let at = list.partition_point(|&id| id < doc_id);
                list.insert(at, doc_id);
                self.df[dim as usize] += 1;
            }
        }
        self.lengths
            .resize(self.lengths.len().max(doc_id as usize + 1), 0);
        self.lengths[doc_id as usize] = dims.len();
        self.doc_count += 1;
    }

    pub fn remove_doc(&mut self, doc_id: u32, dims: &[u32]) {
        for &dim in dims {
            if let Some(list) = self.postings.get_mut(dim as usize) {
                if let Ok(i) = list.binary_search(&doc_id) {
                    list.remove(i);
                    self.df[dim as usize] -= 1;
                }
            }
        }
        self.doc_count = self.doc_count.saturating_sub(1);
        if let Some(len) = self.lengths.get_mut(doc_id as usize) {
            *len = 0;
        }
    }

    /// Ensure all posting lists are sorted (if not added monotonically).
    pub fn finalize(&mut self) {
        for list in &mut self.postings {
            list.sort_unstable();
        }
    }

    /// Perform k-NN query using the epoch-stamped accumulator strategy: the `k` most similar
    /// document slots, most similar first.
    pub fn query(
        &self,
        query_dims: &[u32],
        k: usize,
        accumulator: &mut Accumulator,
    ) -> Vec<SlotHit> {
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
        let mut sorted_qdims = std::mem::take(&mut accumulator.dims);
        sorted_qdims.clear();
        sorted_qdims.extend_from_slice(query_dims);
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
        accumulator.dims = sorted_qdims;

        // 4. Compute Jaccard and extract Top-K
        // Jaccard = inter / (m_q + m_d - inter).
        // With fixed sparsity m_q = m_d = self.m.
        let mut heap = BinaryHeap::with_capacity(k + 1);

        // We only need to check doc_ids that were touched
        for &doc_id in &accumulator.touched {
            let inter = accumulator.get_count(doc_id) as f64;
            let denom = query_dims.len() as f64 + self.lengths[doc_id as usize] as f64 - inter;
            let similarity = if denom > f64::EPSILON {
                inter / denom
            } else {
                1.0
            };

            let hit = SlotHit {
                slot: doc_id,
                similarity,
            };
            // `Ord` is total over distinct slots, so the k smallest hits (the k best) do not
            // depend on the order the heap saw them: a hit no better than the worst kept one
            // is skipped instead of pushed and popped straight back out.
            if heap.len() < k {
                heap.push(hit);
            } else if let Some(mut worst) = heap.peek_mut() {
                if hit < *worst {
                    *worst = hit;
                }
            }
        }

        // With the reversed `Ord` (smaller similarity = Greater), into_sorted_vec (ascending
        // Ord) returns DESCENDING similarity.
        heap.into_sorted_vec()
    }
}

/// A scored document slot. Ordered as a `RetrievalResult` whose id is the slot's decimal
/// string: similarity reversed, so a `BinaryHeap` keeps the k largest, then ties by the
/// decimal strings' byte order ("10" < "9"), the order the index produced when it
/// returned slot numbers as strings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotHit {
    pub slot: u32,
    pub similarity: f64,
}

impl Eq for SlotHit {}

impl PartialOrd for SlotHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SlotHit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| decimal_cmp(self.slot, other.slot))
    }
}

/// `a.to_string().cmp(&b.to_string())` without allocating.
fn decimal_cmp(a: u32, b: u32) -> std::cmp::Ordering {
    fn digits(mut n: u32, buffer: &mut [u8; 10]) -> &[u8] {
        let mut at = buffer.len();
        loop {
            at -= 1;
            buffer[at] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                return &buffer[at..];
            }
        }
    }
    let (mut x, mut y) = ([0u8; 10], [0u8; 10]);
    digits(a, &mut x).cmp(digits(b, &mut y))
}

/// Thread-local or reusable accumulator to avoid allocations.
pub struct Accumulator {
    counts: Vec<u32>,
    seen: Vec<u32>,
    epoch: u32,
    touched: Vec<u32>,
    /// The query's dimensions in scan order, kept between queries.
    dims: Vec<u32>,
}

impl Accumulator {
    pub fn new(max_docs: usize) -> Self {
        Self {
            counts: vec![0; max_docs],
            seen: vec![0; max_docs],
            epoch: 0,
            touched: Vec::with_capacity(1024),
            dims: Vec::new(),
        }
    }

    /// Prepare for a new query.
    pub fn next_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
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
    pub fn get_count(&self, doc_id: u32) -> u32 {
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
        assert_eq!(results[0].slot, 0);
        // inter=3, m=4. J = 3 / (4 + 4 - 3) = 3/5 = 0.6
        assert!((results[0].similarity - 0.75).abs() < 1e-6);

        assert_eq!(results[1].slot, 2);
        // inter=2, m=4. J = 2 / (4 + 4 - 2) = 2/6 = 0.333
        assert!((results[1].similarity - 0.4).abs() < 1e-5);
    }

    #[test]
    fn decimal_cmp_orders_as_decimal_strings() {
        let mut values = vec![0, 1, 9, 10, 11, 19, 99, 100, 101, 999, 1000, 4096, 65_535];
        values.extend([1_000_000_000, 4_294_967_294, u32::MAX]);
        values.extend((0..200u32).map(|i| i.wrapping_mul(2_654_435_761) >> (i % 29)));
        for &a in &values {
            for &b in &values {
                assert_eq!(
                    decimal_cmp(a, b),
                    a.to_string().cmp(&b.to_string()),
                    "{a} vs {b}"
                );
            }
        }
    }

    /// The scan as it was when hits carried their slot as a decimal `String`.
    fn string_id_query(
        index: &SparseInvertedIndex,
        query_dims: &[u32],
        k: usize,
        accumulator: &mut Accumulator,
    ) -> Vec<crate::core::types::RetrievalResult> {
        let mut sorted_qdims: Vec<u32> = query_dims.to_vec();
        sorted_qdims
            .sort_unstable_by_key(|&d| index.df.get(d as usize).cloned().unwrap_or(u32::MAX));
        accumulator.next_epoch();
        for &dim in &sorted_qdims {
            if let Some(list) = index.postings.get(dim as usize) {
                for &doc_id in list {
                    accumulator.increment(doc_id);
                }
            }
        }
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for &doc_id in &accumulator.touched {
            let inter = accumulator.get_count(doc_id) as f64;
            let denom = query_dims.len() as f64 + index.lengths[doc_id as usize] as f64 - inter;
            let similarity = if denom > f64::EPSILON {
                inter / denom
            } else {
                1.0
            };
            heap.push(crate::core::types::RetrievalResult {
                id: doc_id.to_string(),
                similarity,
            });
            if heap.len() > k {
                heap.pop();
            }
        }
        heap.into_sorted_vec()
    }

    /// Slot hits rank exactly as the string-id hits did, including the decimal-string order
    /// of tied slots across 1-, 2- and 3-digit slot numbers at the k boundary.
    #[test]
    fn slot_hits_match_string_id_ranking() {
        let mut index = SparseInvertedIndex::new(512, 8);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for doc in 0..240u32 {
            // A third of the slots share one of three code sets, so similarities tie.
            let dims: Vec<u32> = if doc % 3 == 0 {
                let base = (doc / 3 % 3) * 7;
                (0..8).map(|i| base + i * 5).collect()
            } else {
                let mut dims: Vec<u32> = (0..8).map(|_| (next() % 512) as u32).collect();
                dims.sort_unstable();
                dims.dedup();
                dims
            };
            index.add_doc(doc, &dims);
        }
        index.finalize();
        let mut acc = Accumulator::new(240);
        let queries: Vec<Vec<u32>> = vec![
            (0..8).map(|i| i * 5).collect(),
            (0..8).map(|i| 7 + i * 5).collect(),
            vec![0, 5, 14, 21, 40, 300, 301, 302],
            (0..16).map(|i| i * 31 % 512).collect(),
        ];
        for query in &queries {
            for k in [1, 5, 9, 10, 11, 12, 40, 81, 100, 239, 240, 241] {
                let expected = string_id_query(&index, query, k, &mut acc);
                let hits = index.query(query, k, &mut acc);
                assert_eq!(hits.len(), expected.len());
                for (hit, want) in hits.iter().zip(&expected) {
                    assert_eq!(hit.slot.to_string(), want.id, "k = {k}");
                    assert_eq!(hit.similarity.to_bits(), want.similarity.to_bits());
                }
            }
        }
    }
}
