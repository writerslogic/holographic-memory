// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Graph index with one compact code per vertex instead of one code per edge.
//!
//! Every vertex stores its mean-centred vector as 8-bit integers with one scale (locally
//! adaptive scalar quantization, LVQ; Aguerrebere et al., VLDB 2023), next to that scale in one
//! row. Traversal estimates `<q, x> - <q, mean> ~= s_x * s_q * <c_q, c_x>` with an int8 query
//! code and one integer dot product per neighbour, so a neighbour costs one short row fetch
//! (`dim + 4` bytes) rather than a full-precision vector. An optional second 8-bit code of the
//! quantization residual (LVQ-8x8) re-ranks the best `rerank` candidates of the final pool with
//! a float query.
//!
//! Memory per vertex is `4 * degree + dim + 4` bytes, plus `dim + 4` with the residual, against
//! `4 * dim` for the vectors alone in an HNSW index.

use rayon::prelude::*;

use super::kernels::{dot_f32_i8, dot_i8_kernel, prefetch_bytes, DotI8};
use super::{upper_bytes, Candidate, Graph, Visited, NONE};

/// Search parameters of a [`VGraph`].
#[derive(Clone, Copy, Debug)]
pub struct VSearchParams {
    /// Candidate pool size (beam width).
    pub ef: usize,
    /// Number of best pool candidates re-scored with the float query (and the residual code,
    /// when built); 0 returns the pool order.
    pub rerank: usize,
}

pub struct VGraph {
    n: usize,
    dim: usize,
    /// `dim` rounded up to a multiple of 16 (the integer kernel's width).
    padded: usize,
    /// Bytes per code row: `padded` codes then the f32 scale.
    row: usize,
    degree: usize,
    mean: Vec<f32>,
    adj: Vec<u32>,
    codes: Vec<i8>,
    residual: Vec<i8>,
    entry: u32,
    upper_ids: Vec<u32>,
    layers: Vec<Vec<Vec<u32>>>,
}

/// Writes the symmetric 8-bit code of `x` into `code` (zero padded) and returns its scale, so
/// that `x_i ~= scale * code_i`.
fn quantize(x: &[f32], code: &mut [i8]) -> f32 {
    let m = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    code.fill(0);
    if m == 0.0 {
        return 0.0;
    }
    let scale = m / 127.0;
    for (c, &v) in code.iter_mut().zip(x) {
        *c = (v / scale).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

fn scale_of(row: &[i8], padded: usize) -> f32 {
    f32::from_ne_bytes(std::array::from_fn(|i| row[padded + i] as u8))
}

fn set_scale(row: &mut [i8], padded: usize, s: f32) {
    for (c, b) in row[padded..].iter_mut().zip(s.to_ne_bytes()) {
        *c = b as i8;
    }
}

impl VGraph {
    /// Encodes a built graph. `residual` adds the second code used by `rerank`.
    pub fn from_graph(g: &Graph, residual: bool) -> Self {
        let (n, dim) = (g.n, g.dim);
        let padded = dim.next_multiple_of(16);
        let row = padded + 4;
        let mut sum = vec![0f64; dim];
        for x in g.unit.chunks_exact(dim) {
            sum.iter_mut().zip(x).for_each(|(s, &v)| *s += f64::from(v));
        }
        let mut mean: Vec<f32> = sum.iter().map(|s| (s / n as f64) as f32).collect();
        mean.resize(padded, 0.0);

        let mut codes = vec![0i8; n * row];
        let mut res = vec![0i8; if residual { n * row } else { 0 }];
        let encode = |(v, (c, r)): (usize, (&mut [i8], Option<&mut [i8]>))| {
            let mut x: Vec<f32> = g.unit[v * dim..(v + 1) * dim]
                .iter()
                .zip(&mean)
                .map(|(a, m)| a - m)
                .collect();
            let s = quantize(&x, &mut c[..padded]);
            set_scale(c, padded, s);
            if let Some(r) = r {
                x.iter_mut()
                    .zip(&c[..dim])
                    .for_each(|(a, &q)| *a -= s * f32::from(q));
                let sr = quantize(&x, &mut r[..padded]);
                set_scale(r, padded, sr);
            }
        };
        if residual {
            codes
                .par_chunks_mut(row)
                .zip(res.par_chunks_mut(row).map(Some))
                .enumerate()
                .for_each(encode);
        } else {
            codes
                .par_chunks_mut(row)
                .map(|c| (c, None))
                .enumerate()
                .for_each(encode);
        }

        let r = g.degree;
        let mut adj = vec![NONE; n * r];
        for (dst, src) in adj.chunks_exact_mut(r).zip(&g.adj) {
            dst[..src.len()].copy_from_slice(src);
        }
        Self {
            n,
            dim,
            padded,
            row,
            degree: r,
            mean,
            adj,
            codes,
            residual: res,
            entry: g.entry,
            upper_ids: g.upper_ids.clone(),
            layers: g.layers.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn has_residual(&self) -> bool {
        !self.residual.is_empty()
    }

    /// Bytes held by the index, residual included.
    pub fn index_bytes(&self) -> usize {
        self.adj.len() * 4
            + self.codes.len()
            + self.residual.len()
            + self.mean.len() * 4
            + upper_bytes(&self.upper_ids, &self.layers)
    }

    /// Bytes of the residual codes, which only `rerank` reads.
    pub fn residual_bytes(&self) -> usize {
        self.residual.len()
    }

    fn code(&self, v: u32) -> &[i8] {
        let v = v as usize;
        &self.codes[v * self.row..(v + 1) * self.row]
    }

    /// Estimated negative inner product with the query code (up to the per-query terms
    /// `<q, mean>` and `s_q`, which do not change the order).
    #[inline]
    fn est(&self, dot: DotI8, qc: &[i8], v: u32) -> f32 {
        let c = self.code(v);
        -(scale_of(c, self.padded) * dot(qc, &c[..self.padded]) as f32)
    }

    /// Re-rank score with the float (centred) query: negative estimated inner product.
    fn fine(&self, qf: &[f32], v: u32) -> f32 {
        let c = self.code(v);
        let mut ip = scale_of(c, self.padded) * dot_f32_i8(qf, &c[..self.padded]);
        if self.has_residual() {
            let v = v as usize;
            let r = &self.residual[v * self.row..(v + 1) * self.row];
            ip += scale_of(r, self.padded) * dot_f32_i8(qf, &r[..self.padded]);
        }
        -ip
    }

    /// Greedy descent through the upper layers on estimated distances; returns the bottom-layer
    /// entry vertex and the number of estimates.
    fn descend(&self, dot: DotI8, qc: &[i8]) -> (u32, usize) {
        if self.layers.is_empty() {
            return (self.entry, 0);
        }
        let mut p = 0u32;
        let mut best = self.est(dot, qc, self.upper_ids[0]);
        let mut evals = 1;
        for layer in self.layers.iter().rev() {
            loop {
                let mut moved = false;
                for &u in &layer[p as usize] {
                    let d = self.est(dot, qc, self.upper_ids[u as usize]);
                    evals += 1;
                    if d < best {
                        best = d;
                        p = u;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        (self.upper_ids[p as usize], evals)
    }

    pub fn searcher(&self) -> VSearcher<'_> {
        VSearcher {
            index: self,
            dot: dot_i8_kernel(),
            visited: Visited::new(self.n),
            qf: vec![0.0; self.padded],
            qc: vec![0; self.padded],
            fresh: Vec::with_capacity(self.degree),
            pool: Vec::new(),
            results: Vec::new(),
        }
    }
}

/// Per-thread search state for one [`VGraph`].
pub struct VSearcher<'a> {
    index: &'a VGraph,
    dot: DotI8,
    visited: Visited,
    qf: Vec<f32>,
    qc: Vec<i8>,
    fresh: Vec<u32>,
    pool: Vec<Candidate>,
    results: Vec<(f32, u32)>,
}

impl VSearcher<'_> {
    /// Writes the ids of the (approximate) `k` nearest vectors by cosine into `out`, nearest
    /// first, and returns the number of distance estimates (descent, traversal and re-rank).
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        params: VSearchParams,
        out: &mut Vec<u32>,
    ) -> usize {
        let g = self.index;
        assert_eq!(query.len(), g.dim);
        let ef = params.ef.max(k).max(1);
        let dot = self.dot;
        // Only the order matters, so the query is used as given (no centring needed: the
        // `<q, mean>` term is common to all candidates) and its scale is dropped.
        self.qf[..g.dim].copy_from_slice(query);
        quantize(&self.qf, &mut self.qc);

        self.visited.next();
        let (entry, mut evals) = g.descend(dot, &self.qc);
        self.visited.insert(entry);
        self.pool.clear();
        self.pool.push(Candidate {
            dist: g.est(dot, &self.qc, entry),
            id: entry,
            done: false,
        });
        evals += 1;
        let r = g.degree;
        let mut cur = 0;
        while cur < self.pool.len() {
            self.pool[cur].done = true;
            let v = self.pool[cur].id as usize;
            self.fresh.clear();
            for &u in &g.adj[v * r..(v + 1) * r] {
                if u == NONE {
                    break;
                }
                if self.visited.insert(u) {
                    prefetch_bytes(g.code(u));
                    self.fresh.push(u);
                }
            }
            let mut best = usize::MAX;
            for &u in &self.fresh {
                let d = g.est(dot, &self.qc, u);
                evals += 1;
                if self.pool.len() >= ef && d >= self.pool[ef - 1].dist {
                    continue;
                }
                let pos = self.pool.partition_point(|c| c.dist <= d);
                self.pool.insert(
                    pos,
                    Candidate {
                        dist: d,
                        id: u,
                        done: false,
                    },
                );
                self.pool.truncate(ef);
                best = best.min(pos);
            }
            cur = best.min(cur + 1);
            while cur < self.pool.len() && self.pool[cur].done {
                cur += 1;
            }
        }

        self.results.clear();
        let take = params.rerank.max(k).min(self.pool.len());
        if params.rerank > 0 {
            self.results.extend(
                self.pool[..take]
                    .iter()
                    .map(|c| (g.fine(&self.qf, c.id), c.id)),
            );
            evals += take;
            self.results
                .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        } else {
            self.results
                .extend(self.pool[..take].iter().map(|c| (c.dist, c.id)));
        }
        out.clear();
        out.extend(self.results.iter().take(k).map(|r| r.1));
        evals
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{brute_force, clustered, gaussian};
    use super::super::{kernels::SplitMix, BuildParams};
    use super::*;

    /// Per coordinate the 8-bit code is off by at most half a step, and the residual code
    /// shrinks that error by about another factor of 127.
    #[test]
    fn quantization_error_is_bounded() {
        let mut rng = SplitMix(12);
        let x: Vec<f32> = (0..100).map(|_| gaussian(&mut rng)).collect();
        let mut c = vec![0i8; 112];
        let s = quantize(&x, &mut c);
        let mut r: Vec<f32> = x
            .iter()
            .zip(&c)
            .map(|(a, &q)| a - s * f32::from(q))
            .collect();
        assert!(r.iter().all(|e| e.abs() <= s / 2.0 + 1e-6));
        assert!(c[100..].iter().all(|&q| q == 0));
        let mut rc = vec![0i8; 112];
        let sr = quantize(&r, &mut rc);
        r.iter_mut()
            .zip(&rc)
            .for_each(|(a, &q)| *a -= sr * f32::from(q));
        assert!(r.iter().all(|e| e.abs() <= s / 254.0 + 1e-6));
        assert_eq!(quantize(&[0.0; 4], &mut c[..16]), 0.0);
    }

    fn recall(index: &VGraph, data: &[f32], queries: &[f32], d: usize, p: VSearchParams) -> f64 {
        let mut s = index.searcher();
        let mut out = Vec::new();
        let mut hits = 0;
        for q in queries.chunks_exact(d) {
            let truth = brute_force(data, d, q, 10);
            s.search(q, 10, p, &mut out);
            assert_eq!(out.len(), 10);
            hits += out.iter().filter(|x| truth.contains(x)).count();
        }
        hits as f64 / (queries.len() / d * 10) as f64
    }

    #[test]
    fn recall_on_clustered_data() {
        let d = 24;
        let data = clustered(4000, d, 1);
        let queries = clustered(100, d, 2);
        let graph = Graph::build(&data, d, &BuildParams::default());
        let plain = VGraph::from_graph(&graph, false);
        let fine = VGraph::from_graph(&graph, true);
        assert!(plain.index_bytes() < fine.index_bytes());
        let pool = VSearchParams { ef: 64, rerank: 0 };
        let rr = VSearchParams { ef: 64, rerank: 32 };
        let r0 = recall(&plain, &data, &queries, d, pool);
        let r1 = recall(&fine, &data, &queries, d, rr);
        // Same regression guard as the edge-code index on this deliberately hard set.
        assert!(r0 >= 0.9, "recall {r0}");
        assert!(r1 >= r0, "rerank {r1} below pool order {r0}");
    }

    #[test]
    fn build_and_search_are_deterministic() {
        let d = 20;
        let data = clustered(3000, d, 3);
        let params = BuildParams {
            seed: 9,
            ..BuildParams::default()
        };
        let a = VGraph::from_graph(&Graph::build(&data, d, &params), true);
        let b = VGraph::from_graph(&Graph::build(&data, d, &params), true);
        assert!(a.codes == b.codes && a.residual == b.residual && a.adj == b.adj);
        assert_eq!((a.entry, &a.mean), (b.entry, &b.mean));
        let (mut sa, mut sb) = (a.searcher(), b.searcher());
        let (mut oa, mut ob) = (Vec::new(), Vec::new());
        let p = VSearchParams { ef: 32, rerank: 16 };
        for q in data.chunks_exact(d).take(50) {
            sa.search(q, 10, p, &mut oa);
            sb.search(q, 10, p, &mut ob);
            assert_eq!(oa, ob);
        }
    }
}
