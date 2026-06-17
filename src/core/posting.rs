// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use fxhash::FxHashMap;

use super::tombstone::TombstoneMap;

/// Inverted index mapping dimension indices to sorted lists of vector IDs.
/// A single PostingShard covers all D=16384 dimensions.
pub struct PostingShard {
    /// lists[dim_index] = sorted Vec of vector IDs that have this dimension active.
    lists: Vec<Vec<u32>>,
    dim: usize,
}

impl PostingShard {
    pub fn new(dim: usize) -> Self {
        Self {
            lists: (0..dim).map(|_| Vec::new()).collect(),
            dim,
        }
    }

    /// Insert a vector ID into the posting lists for each of its active indices.
    /// Maintains sorted order. Deduplicates (no-op if already present).
    pub fn insert(&mut self, vec_id: u32, indices: &[u32]) {
        for &idx in indices {
            if (idx as usize) < self.dim {
                let list = &mut self.lists[idx as usize];
                match list.binary_search(&vec_id) {
                    Ok(_) => {} // already present
                    Err(pos) => list.insert(pos, vec_id),
                }
            }
        }
    }

    /// Remove a vector ID from posting lists for each of its active indices.
    pub fn remove(&mut self, vec_id: u32, indices: &[u32]) {
        for &idx in indices {
            if (idx as usize) < self.dim {
                let list = &mut self.lists[idx as usize];
                if let Ok(pos) = list.binary_search(&vec_id) {
                    list.remove(pos);
                }
            }
        }
    }

    /// Get the posting list for a dimension index.
    pub fn get_list(&self, dim_index: u32) -> &[u32] {
        if (dim_index as usize) < self.dim {
            &self.lists[dim_index as usize]
        } else {
            &[]
        }
    }

    /// Compute overlap counts: for each vector ID found across the query's
    /// active dimensions, count how many dimensions it shares with the query.
    /// Filters out tombstoned vectors.
    /// Returns (vec_id, overlap_count) sorted by overlap descending.
    pub fn overlap_counts(
        &self,
        query_indices: &[u32],
        tombstones: &TombstoneMap,
    ) -> Vec<(u32, u32)> {
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        for &dim in query_indices {
            for &vec_id in self.get_list(dim) {
                if !tombstones.is_deleted(vec_id) {
                    *counts.entry(vec_id).or_insert(0) += 1;
                }
            }
        }
        let mut result: Vec<(u32, u32)> = counts.into_iter().collect();
        result.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Document frequency: number of vectors containing this dimension.
    pub fn doc_freq(&self, dim_index: u32) -> u32 {
        self.get_list(dim_index).len() as u32
    }

    /// Rebuild all posting lists from a set of (vec_id, indices) pairs.
    pub fn rebuild(&mut self, vectors: &[(u32, &[u32])]) {
        for list in &mut self.lists {
            list.clear();
        }
        for &(vec_id, indices) in vectors {
            self.insert(vec_id, indices);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posting_insert_and_overlap() {
        let mut shard = PostingShard::new(100);
        let tombstones = TombstoneMap::new();

        shard.insert(1, &[0, 5, 10]);
        shard.insert(2, &[5, 10, 20]);
        shard.insert(3, &[0, 20]);

        // Query dims [0, 5, 10]: vec 1 has overlap 3, vec 2 has overlap 2, vec 3 has overlap 1
        let result = shard.overlap_counts(&[0, 5, 10], &tombstones);
        assert_eq!(result[0], (1, 3));
        assert_eq!(result[1], (2, 2));
        assert_eq!(result[2], (3, 1));
    }

    #[test]
    fn test_posting_dedup() {
        let mut shard = PostingShard::new(100);
        shard.insert(1, &[5]);
        shard.insert(1, &[5]); // duplicate
        assert_eq!(shard.get_list(5).len(), 1);
    }

    #[test]
    fn test_posting_remove() {
        let mut shard = PostingShard::new(100);
        shard.insert(1, &[5, 10]);
        shard.remove(1, &[5, 10]);
        assert!(shard.get_list(5).is_empty());
        assert!(shard.get_list(10).is_empty());
    }

    #[test]
    fn test_posting_tombstone_filter() {
        let mut shard = PostingShard::new(100);
        let mut tombstones = TombstoneMap::new();

        shard.insert(1, &[5]);
        shard.insert(2, &[5]);
        tombstones.mark_deleted(1);

        let result = shard.overlap_counts(&[5], &tombstones);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 2);
    }
}
