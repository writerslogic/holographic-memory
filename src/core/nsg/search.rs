use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;

use super::NSGIndex;

pub(crate) fn greedy_search_internal(
    index: &NSGIndex,
    query: &EntangledHVec,
    ef_search: usize,
) -> Vec<(u32, u32)> {
    if index.vectors.is_empty() {
        return vec![];
    }

    let n = index.vectors.len();
    let mut visited = vec![false; n];

    // Min-heap frontier: expand closest-first (Reverse makes BinaryHeap act as min-heap).
    let mut frontier: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(ef_search * 2);
    // Max-heap result set: keeps the ef_search best (closest) candidates.
    // The furthest candidate sits at the top; we pop it when the set is full
    // and a closer candidate arrives. This replaces the O(n log n) sort.
    let mut results: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(ef_search + 1);

    let start = index.navigating_node as usize;
    if start >= n {
        return vec![];
    }

    let start_dist = query.hamming(&index.vectors[start]);
    frontier.push(Reverse((start_dist, start as u32)));
    results.push((start_dist, start as u32));
    visited[start] = true;

    while let Some(Reverse((current_dist, current))) = frontier.pop() {
        // Pruning: if this candidate is already worse than the worst in our
        // result set (which is full), skip expansion entirely.
        if results.len() >= ef_search {
            if let Some(&(worst_dist, _)) = results.peek() {
                if current_dist > worst_dist {
                    continue;
                }
            }
        }

        let current_idx = current as usize;
        if current_idx >= index.neighbors.len() {
            continue;
        }

        for &neighbor in &index.neighbors[current_idx] {
            let ni = neighbor as usize;
            if ni >= n || visited[ni] {
                continue;
            }
            visited[ni] = true;

            let dist = query.hamming(&index.vectors[ni]);

            // Only add to frontier/results if it could improve the result set.
            let dominated =
                results.len() >= ef_search && results.peek().is_some_and(|&(w, _)| dist >= w);
            if dominated {
                continue;
            }

            frontier.push(Reverse((dist, neighbor)));
            results.push((dist, neighbor));

            // Evict the worst if over capacity.
            if results.len() > ef_search {
                results.pop();
            }
        }
    }

    // Drain into a sorted vec (ascending by distance).
    let mut candidates: Vec<(u32, u32)> = results.into_vec();
    candidates.sort_unstable();
    candidates
}

pub(super) fn greedy_search(
    index: &NSGIndex,
    query: &EntangledHVec,
    k: usize,
    ef_search: usize,
) -> Vec<RetrievalResult> {
    let candidates = greedy_search_internal(index, query, ef_search.max(k));

    candidates
        .iter()
        .take(k)
        .map(|&(_dist, idx)| {
            let id = &index.id_map[idx as usize];
            let sim = query.similarity(&index.vectors[idx as usize]);
            RetrievalResult {
                id: id.clone(),
                similarity: sim,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::NSGConfig;
    use crate::core::nsg::training;

    fn build_test_index(n: usize, dim: usize) -> NSGIndex {
        let vectors: Vec<EntangledHVec> = (0..n)
            .map(|i| EntangledHVec::new_deterministic(dim, i as u64))
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("vec_{}", i)).collect();
        let config = NSGConfig {
            max_degree: 8,
            ef_construction: 32,
            auto_threshold: 0,
            seed: 42,
        };
        training::train(&vectors, &ids, &config).unwrap()
    }

    #[test]
    fn greedy_search_finds_self() {
        let index = build_test_index(50, 1000);
        let results = greedy_search(&index, &index.vectors[0], 1, 32);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "vec_0");
    }

    #[test]
    fn greedy_search_ordering() {
        let index = build_test_index(50, 1000);
        let query = EntangledHVec::new_deterministic(1000, 999);
        let results = greedy_search(&index, &query, 10, 32);
        for w in results.windows(2) {
            assert!(
                w[0].similarity >= w[1].similarity,
                "Results should be sorted desc"
            );
        }
    }

    #[test]
    fn greedy_search_returns_k_results() {
        let index = build_test_index(50, 1000);
        let query = EntangledHVec::new_deterministic(1000, 999);
        let results = greedy_search(&index, &query, 5, 32);
        assert_eq!(results.len(), 5, "Should return exactly k results");
    }
}
