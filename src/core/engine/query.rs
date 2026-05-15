use std::collections::BinaryHeap;
use fxhash::FxHashSet;

use rayon::prelude::*;

use super::router::IndexRoute;
use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;

/// Wrapper for BinaryHeap min-heap ordering by similarity.
#[derive(PartialEq)]
struct ScoredEntry {
    similarity: f64,
    id: String,
}

impl Eq for ScoredEntry {}

impl PartialOrd for ScoredEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering: smallest similarity at the top (min-heap)
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl HmsCore {
    /// Query the memory system for the k most similar vectors.
    /// Routes to NSG or IVF if trained, otherwise falls back to brute-force scan.
    pub fn query(&self, query_vec: &EntangledHVec, k: u32) -> Vec<RetrievalResult> {
        let n = self.vector_count.load(std::sync::atomic::Ordering::SeqCst) as usize;

        // 1. Plan retrieval strategy based on collection size and query sparsity.
        let planner = super::router::QueryPlanner::new(
            self.nsg_trained(),
            n > 0, // Inverted index always available if items exist
            self.ivf_trained(),
            n,
            self.dimensions,
        );
        let plan = planner.plan(query_vec, k);

        // NSG/IVF indexes may contain stale entries for deleted IDs.
        // Filter results against the cached live ID set which is authoritative.
        let live_ids = self.live_ids.read();

        match plan.route {
            IndexRoute::NSG => {
                if let Some(ref nsg) = *self.nsg.read() {
                    let results: Vec<RetrievalResult> = nsg
                        .query(query_vec, k as usize, plan.ef_search)
                        .into_iter()
                        .filter(|r| live_ids.contains(&r.id))
                        .collect();
                    if !results.is_empty() {
                        return results;
                    }
                }
            }
            IndexRoute::Inverted => {
                let mut acc = self.accumulator.lock();
                let results = self
                    .inverted
                    .read()
                    .query(&query_vec.indices, k as usize, &mut acc);

                // Map u32 doc_id back to String ID using the registry
                let reg = self.registry.read();
                let mapped: Vec<RetrievalResult> = results
                    .into_iter()
                    .filter_map(|r| {
                        let doc_id: u32 = r.id.parse().ok()?;
                        let (id_str, _) = reg.get(doc_id as usize)?;
                        Some(RetrievalResult {
                            id: id_str.clone(),
                            similarity: r.similarity,
                        })
                    })
                    .collect();

                if !mapped.is_empty() {
                    return mapped;
                }
            }
            IndexRoute::IVF => {
                if let Some(ref ivf) = *self.ivf.read() {
                    if let Ok(candidates) = ivf.query(query_vec, k as usize, plan.n_probe) {
                        // IVF returns PQ-approximate distances on an incompatible scale.
                        // Re-rank candidates by exact Jaccard so all query paths return
                        // the same similarity metric.
                        let reranked =
                            self.rerank_by_exact_similarity(query_vec, &candidates, &live_ids);
                        if !reranked.is_empty() {
                            return reranked;
                        }
                    }
                }
            }
            IndexRoute::BruteForce => {
                return self.brute_force_scan(query_vec, k as usize);
            }
        }

        // Final fallback (should only happen if an index failed or was empty)
        self.brute_force_scan(query_vec, k as usize)
    }

    /// Re-rank IVF candidates by exact Jaccard similarity.
    /// Uses fast id_to_offset lookup for each candidate.
    fn rerank_by_exact_similarity(
        &self,
        query_vec: &EntangledHVec,
        candidates: &[RetrievalResult],
        live_ids: &FxHashSet<String>,
    ) -> Vec<RetrievalResult> {
        let ito = self.id_to_offset.read();

        let mut results: Vec<RetrievalResult> = candidates
            .iter()
            .filter(|c| live_ids.contains(&c.id))
            .filter_map(|c| {
                let offset = *ito.get(&c.id)?;
                let (_, vec) = self.read_entry(offset);
                Some(RetrievalResult {
                    id: c.id.clone(),
                    similarity: query_vec.similarity(&vec),
                })
            })
            .collect();

        results.sort_unstable_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Linear scan over all stored vectors. Uses a min-heap of size k
    /// to avoid sorting all results when k is small relative to n.
    fn brute_force_scan(&self, query_vec: &EntangledHVec, k: usize) -> Vec<RetrievalResult> {
        let registry = self.registry.read();
        let mut heap: BinaryHeap<ScoredEntry> = BinaryHeap::with_capacity(k + 1);

        for (id, offset) in registry.iter() {
            let (_, vec) = self.read_entry(*offset);
            let sim = query_vec.similarity(&vec);

            if heap.len() < k {
                heap.push(ScoredEntry {
                    similarity: sim,
                    id: id.clone(),
                });
            } else if let Some(top) = heap.peek() {
                if sim > top.similarity {
                    heap.pop();
                    heap.push(ScoredEntry {
                        similarity: sim,
                        id: id.clone(),
                    });
                }
            }
        }

        let mut results: Vec<RetrievalResult> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| RetrievalResult {
                id: e.id,
                similarity: e.similarity,
            })
            .collect();
        // into_sorted_vec gives ascending order (min-heap); we want descending
        results.reverse();
        results
    }

    /// Process multiple queries in parallel using rayon.
    /// Returns one result vector per query, in the same order as the input.
    pub fn query_batch(&self, queries: &[EntangledHVec], k: u32) -> Vec<Vec<RetrievalResult>> {
        queries.par_iter().map(|q| self.query(q, k)).collect()
    }

    /// Analyze components of a vector by finding its nearest neighbors.
    /// Returns neighbors with raw Jaccard similarity above a minimum threshold.
    ///
    /// Threshold 0.05 is ~25x the null Jaccard expectation for independent sparse
    /// vectors at rho=1/256: E[J] = rho/(2-rho) ~ 0.002. This filters noise
    /// while retaining weak but genuine associations.
    pub fn analyze_components(&self, vector: &EntangledHVec) -> Vec<RetrievalResult> {
        let neighbors = self.query(vector, 20);
        neighbors
            .into_iter()
            .filter(|r| r.similarity > 0.05)
            .collect()
    }
}
