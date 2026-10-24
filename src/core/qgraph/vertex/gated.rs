// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Bound-gated re-rank (see [`super::SearchKnobs::rerank_z`]).

use std::collections::BinaryHeap;

use super::{key, key_dist, key_id, SearchScratch, VGraph};

/// Rows prefetched ahead of the re-rank cursor.
const AHEAD: usize = 8;
/// Re-scored rows before the gate may stop: the error's mean and spread need a sample.
const SAMPLE: usize = 16;

impl SearchScratch {
    /// Re-scores `self.keys` (sorted by estimate first) in estimate order into `self.results`,
    /// sorted by exact distance, and returns the number re-scored. The first `max(k, 16)` are
    /// always re-scored; each later one only while `est + mean_err - z * sd_err` (errors of the
    /// re-scored rows, in exact-distance units: `est * sq`) is below the `k`-th exact distance.
    /// At most `take` rows, so the gate never re-ranks more than the fixed rule would.
    pub(super) fn rerank_gated(
        &mut self,
        g: &VGraph,
        k: usize,
        take: usize,
        sq: f32,
        z: f32,
    ) -> usize {
        debug_assert!(k > 0 && take > k && take <= self.keys.len() && sq > 0.0);
        if !self.keys.is_sorted() {
            self.keys.sort_unstable();
        }
        let keys = &self.keys[..take];
        for &c in keys.iter().take(AHEAD) {
            g.prefetch_rerank_rows(key_id(c));
        }
        self.results.clear();
        // The `k` best exact keys so far, worst on top.
        let mut best = BinaryHeap::with_capacity(k + 1);
        let (mut n, mut sum, mut sum2) = (0f64, 0f64, 0f64);
        for (i, &c) in keys.iter().enumerate() {
            let id = key_id(c);
            let est = key_dist(c) * sq;
            if i >= k.max(SAMPLE) {
                let mean = sum / n;
                let sd = (sum2 / n - mean * mean).max(0.0).sqrt();
                let bound = f64::from(est) + mean - f64::from(z) * sd;
                let kth = best.peek().map_or(f32::INFINITY, |&w| key_dist(w));
                if bound > f64::from(kth) {
                    break;
                }
            }
            if let Some(&a) = keys.get(i + AHEAD) {
                g.prefetch_rerank_rows(key_id(a));
            }
            let exact = g.fine(&self.qf, id);
            let err = f64::from(exact - est);
            n += 1.0;
            sum += err;
            sum2 += err * err;
            self.results.push((exact, g.original(id)));
            best.push(key(exact, id));
            if best.len() > k {
                best.pop();
            }
        }
        // Ties (duplicate vectors) break by the original id, independent of the layout.
        self.results
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        self.results.len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{SearchKnobs, VGraph, VSearchParams};
    use crate::core::qgraph::{BuildParams, Graph, Metric};

    /// An unreachable gate re-ranks exactly what the fixed rule re-ranks, so it returns the
    /// same ids; a finite gate re-ranks a prefix of those rows (never more), and its first
    /// `k` rows are always re-scored.
    #[test]
    fn gate_is_a_prefix_of_the_fixed_rerank() {
        let rows = crate::core::qgraph::tests::clustered(512, 32, 91);
        for metric in [Metric::Angular, Metric::Euclidean] {
            let graph = Graph::build(
                &rows,
                32,
                &BuildParams {
                    metric,
                    degree: 8,
                    build_ef: 32,
                    ..Default::default()
                },
            )
            .unwrap();
            for bits in [4, 8] {
                let index = VGraph::from_graph_with(&graph, bits, 8, true, true, 4, false).unwrap();
                let gate = |z| SearchKnobs {
                    rerank_z: z,
                    ..SearchKnobs::OFF
                };
                let mut fixed = index.searcher_with(SearchKnobs::OFF);
                let mut open = index.searcher_with(gate(1e30));
                let mut tight = index.searcher_with(gate(0.5));
                let p = VSearchParams {
                    ef: 40,
                    rerank: 40,
                    ..Default::default()
                };
                for q in rows.as_chunks::<32>().0.iter().take(16) {
                    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
                    for k in [1, 5, 10] {
                        fixed.search(q, k, p, &mut a).unwrap();
                        open.search(q, k, p, &mut b).unwrap();
                        tight.search(q, k, p, &mut c).unwrap();
                        assert_eq!(a, b, "{metric:?} {bits} k {k}");
                        assert_eq!(c.len(), a.len());
                        let (all, some) = (&fixed.scratch.results, &tight.scratch.results);
                        assert_eq!(open.scratch.results.len(), all.len());
                        assert!(some.len() >= k && some.len() <= all.len());
                        assert!(some.iter().all(|r| all.contains(r)));
                    }
                }
            }
        }
    }
}
