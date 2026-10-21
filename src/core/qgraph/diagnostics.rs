// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::{invalid, Graph};
use crate::core::error::HmsError;

#[derive(Clone, Copy, Debug)]
struct Candidate(f64, u32);

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// Diagnostic control using uncompressed graph vectors and float64 cosine scoring.
/// This keeps the production graph topology but does not model compact-index memory.
pub struct ExactSearcher<'a> {
    graph: &'a Graph,
    norms: Vec<f64>,
    seen: Vec<u32>,
    epoch: u32,
    candidates: BinaryHeap<Reverse<Candidate>>,
    best: BinaryHeap<Candidate>,
    ranked: Vec<Candidate>,
}

impl<'a> ExactSearcher<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            norms: graph
                .vectors
                .chunks_exact(graph.dim)
                .map(|row| {
                    row.iter()
                        .map(|&v| f64::from(v).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect(),
            seen: vec![0; graph.n],
            epoch: 0,
            candidates: BinaryHeap::new(),
            best: BinaryHeap::new(),
            ranked: Vec::new(),
        }
    }

    fn validate(&self, query: &[f32], k: usize) -> Result<(), HmsError> {
        if query.len() != self.graph.dim
            || query.iter().any(|v| !v.is_finite())
            || query.iter().all(|&v| v == 0.0)
            || k == 0
            || k > self.graph.n
        {
            return Err(invalid("diagnostic search requires finite nonzero query of index dimension and 1 <= k <= n".into()));
        }
        Ok(())
    }

    fn score(&self, query: &[f32], id: u32) -> Candidate {
        let row =
            &self.graph.vectors[id as usize * self.graph.dim..(id as usize + 1) * self.graph.dim];
        let norm = self.norms[id as usize];
        let similarity = if norm == 0.0 {
            0.0
        } else {
            query
                .iter()
                .zip(row)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
                / norm
        };
        Candidate(-similarity, id)
    }

    pub fn rerank(
        &mut self,
        query: &[f32],
        candidates: &[u32],
        k: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), HmsError> {
        self.validate(query, k)?;
        if candidates.iter().any(|&id| id as usize >= self.graph.n) {
            return Err(invalid("diagnostic candidate ID outside index".into()));
        }
        self.ranked.clear();
        for &id in candidates {
            let score = self.score(query, id);
            self.ranked.push(score);
        }
        self.ranked.sort_unstable();
        self.ranked.dedup_by_key(|c| c.1);
        out.clear();
        out.extend(self.ranked.iter().take(k).map(|c| c.1));
        Ok(())
    }

    pub fn flat(&mut self, query: &[f32], k: usize, out: &mut Vec<u32>) -> Result<usize, HmsError> {
        self.validate(query, k)?;
        self.best.clear();
        for id in 0..self.graph.n as u32 {
            let score = self.score(query, id);
            if self.best.len() < k {
                self.best.push(score);
            } else if score < *self.best.peek().expect("nonempty top k") {
                *self.best.peek_mut().expect("nonempty top k") = score;
            }
        }
        self.finish(k, out);
        Ok(self.graph.n)
    }

    fn finish(&mut self, k: usize, out: &mut Vec<u32>) {
        self.ranked.clear();
        self.ranked.extend(self.best.drain());
        self.ranked.sort_unstable();
        out.clear();
        out.extend(self.ranked.iter().take(k).map(|c| c.1));
    }

    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        ef: usize,
        out: &mut Vec<u32>,
    ) -> Result<usize, HmsError> {
        self.validate(query, k)?;
        let g = self.graph;
        let ef = ef.max(k).min(g.n);
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.seen.fill(0);
            self.epoch = 1;
        }
        let mut entry = g.entry;
        let mut evals = 0;
        if !g.layers.is_empty() {
            let mut p = 0;
            let mut best = self.score(query, g.upper_ids[0]);
            evals += 1;
            for layer in g.layers.iter().rev() {
                loop {
                    let mut moved = false;
                    for &u in &layer[p] {
                        let candidate = self.score(query, g.upper_ids[u as usize]);
                        evals += 1;
                        if candidate.0 < best.0 {
                            best = candidate;
                            p = u as usize;
                            moved = true;
                        }
                    }
                    if !moved {
                        break;
                    }
                }
            }
            entry = g.upper_ids[p];
        }
        self.candidates.clear();
        self.best.clear();
        let first = self.score(query, entry);
        evals += 1;
        self.candidates.push(Reverse(first));
        self.best.push(first);
        self.seen[entry as usize] = self.epoch;
        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if self.best.len() >= ef && self.best.peek().is_some_and(|&worst| candidate > worst) {
                break;
            }
            for &id in &g.adj[candidate.1 as usize] {
                if self.seen[id as usize] == self.epoch {
                    continue;
                }
                self.seen[id as usize] = self.epoch;
                let score = self.score(query, id);
                evals += 1;
                if self.best.len() < ef {
                    self.best.push(score);
                } else {
                    let mut worst = self.best.peek_mut().expect("positive beam width");
                    if score >= *worst {
                        continue;
                    }
                    *worst = score;
                }
                self.candidates.push(Reverse(score));
            }
        }
        self.finish(k, out);
        Ok(evals)
    }
}

/// A second probe opposing symmetric int8 query-rounding error. The union still
/// requires ranking with the original query; no unbiased estimator is implied.
pub fn residual_feedback(query: &[f32]) -> Result<Vec<f32>, HmsError> {
    if query.is_empty() || query.iter().any(|v| !v.is_finite()) || query.iter().all(|&v| v == 0.0) {
        return Err(invalid("feedback requires finite nonzero query".into()));
    }
    let maximum = query
        .iter()
        .map(|v| f64::from(*v).abs())
        .fold(0.0, f64::max);
    let scale = 1.0 / 127.0;
    Ok(query
        .iter()
        .map(|&v| {
            let normalized = f64::from(v) / maximum;
            (2.0 * normalized - (normalized / scale).round() * scale) as f32
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::qgraph::{BuildParams, VGraph, VSearchParams};

    #[test]
    fn full_beam_and_rerank_match_flat_with_duplicates_and_zero_rows() {
        let data = [1., 0., 1., 0., 0., 1., -1., 0., 0., 0.];
        let graph = Graph::build(
            &data,
            2,
            &BuildParams {
                degree: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scorer = ExactSearcher::new(&graph);
        let mut flat = Vec::new();
        scorer.flat(&[1., 0.], 5, &mut flat).unwrap();
        assert_eq!(flat, [0, 1, 2, 4, 3]);
        let mut found = Vec::new();
        for ef in [4, 5, 6] {
            scorer.search(&[1., 0.], 5, ef, &mut found).unwrap();
            assert_eq!(found, flat);
        }
        scorer
            .rerank(&[1., 0.], &[4, 3, 2, 1, 0, 1], 5, &mut found)
            .unwrap();
        assert_eq!(found, flat);
        scorer.epoch = u32::MAX;
        scorer.search(&[1., 0.], 5, 5, &mut found).unwrap();
        assert_eq!(found, flat);
        for ids in [&[4][..], &[5][..], &[6][..]] {
            assert_eq!(
                scorer.rerank(&[1., 0.], ids, 1, &mut found).is_ok(),
                ids[0] < 5
            );
        }
        for k in [0, 1, 5, 6] {
            assert_eq!(
                scorer.flat(&[1., 0.], k, &mut found).is_ok(),
                (1..=5).contains(&k)
            );
        }
        for query in [&[0., 0.][..], &[f32::NAN, 1.][..], &[1.][..]] {
            assert!(scorer.search(query, 1, 5, &mut found).is_err());
        }
    }

    #[test]
    fn exporting_full_pool_preserves_production_prefix_without_patience() {
        let data: Vec<f32> = (0..1024).map(|i| ((i * 73 + 11) as f32).sin()).collect();
        let graph = Graph::build(
            &data,
            8,
            &BuildParams {
                degree: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let index = VGraph::from_graph(&graph, true).unwrap();
        let mut searcher = index.searcher();
        let mut short = Vec::new();
        let mut pool = Vec::new();
        let query = &data[16..24];
        for ef in [15, 16, 17, 32] {
            let params = VSearchParams {
                ef,
                patience: 0,
                rerank: 0,
                ..Default::default()
            };
            let n = searcher.search(query, 10, params, &mut short).unwrap();
            let m = searcher.search(query, ef, params, &mut pool).unwrap();
            assert_eq!(short, pool[..10]);
            assert_eq!(n, m);
        }
    }

    #[test]
    fn feedback_handles_extreme_finite_queries_and_rejects_invalid_inputs() {
        for query in [
            vec![f32::MAX, f32::MAX / 3.],
            vec![f32::MIN_POSITIVE, 0.],
            vec![f32::from_bits(1), 0.],
        ] {
            let probe = residual_feedback(&query).unwrap();
            assert!(probe.iter().all(|v| v.is_finite()));
            assert!(probe.iter().any(|&v| v != 0.));
        }
        for query in [vec![], vec![0.], vec![f32::INFINITY], vec![f32::NAN]] {
            assert!(residual_feedback(&query).is_err());
        }
    }
}
