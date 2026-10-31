// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `warm-research-witness`: occupancy-balanced rare-witness decoding, a
//! graph-free large-N ANN index whose candidate discovery streams sorted
//! posting lists and never follows a data-dependent pointer.
//!
//! # Mechanism
//!
//! A fixed bank of `M` signed projection channels is derived from `M/2`
//! seeded SRHT directions (`bank.rs`). Per-channel empirical tail quantiles
//! are fitted on a seeded sample of `S` corpus rows so that every channel has
//! the same expected occupancy. A corpus vector is a member of channel `c`
//! when its signed projection value exceeds the channel's threshold `tau_c`,
//! the sample quantile at tail probability `B / M`; the vector therefore has
//! `B` memberships on average (its own count varies). Memberships are stored
//! as one sorted `u32` posting list per channel (`postings.rs`).
//!
//! A query computes its `M/2` projections, picks its `Q` most extreme
//! channels by *quantile rank* (a per-direction z-score pre-filter followed
//! by exact per-channel tail-grid interpolation), streams the `Q` posting
//! lists through a counting threshold-join that emits every id with at least
//! `t` witnesses (`join.rs`), then reranks the emitted candidates with exact
//! cosine in one bulk pass over the f32 rows and returns the top 10.
//!
//! Address discovery is `Q` sequential list streams plus one u8 counter
//! array; the only random accesses are the (mutually independent, therefore
//! overlappable) counter increments and the final candidate row fetches.
//!
//! # Data layout
//!
//! * corpus rows: row-major `n x dim` f32, l2-normalized (rerank storage);
//! * bank: `blocks x dim_pad` sign floats, per-direction `mean` / `inv_std`,
//!   per-channel `GRID_LEVELS` tail quantiles, per-budget `tau[M]`;
//! * postings: CSR `offsets[M + 1]` (u32) over `ids[]` (u32), each channel's
//!   slice strictly increasing;
//! * joiner: `counts[n]` u8, all zero between queries.
//!
//! # Invariants
//!
//! * Every posting list is strictly increasing and each id occurs at most
//!   once per list, so a counter never exceeds `Q <= 255`.
//! * `join_emit` pushes an id exactly when its counter reaches `t`, so the
//!   candidate list has no duplicates and `candidates.len() == |{id : count
//!   >= t}|`.
//! * `reset` walks the same lists as the join, restoring all-zero counters.
//! * All randomness derives from one seed; the timed query path is
//!   single-threaded. Rayon is used only in the build-time projection and
//!   quantile-fit passes, and the merge order is fixed, so builds are
//!   bit-for-bit reproducible across thread counts.

#![allow(clippy::needless_range_loop)]

pub mod bank;
pub mod data;
pub mod join;
pub mod postings;

use serde::Serialize;

use bank::{Bank, SelectScratch};
use join::Joiner;
use postings::Postings;

/// Dot product with eight independent lane accumulators (auto-vectorizes).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail += x * y;
    }
    tail + ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// l2 norm.
pub fn norm_f32(a: &[f32]) -> f32 {
    dot_f32(a, a).sqrt()
}

/// Scale to unit norm; an all-zero vector is left unchanged.
pub fn normalize_in_place(v: &mut [f32]) {
    let n = norm_f32(v);
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Median of a sample (average of the two middle values for even counts).
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Distribution summary of a per-query measurement.
#[derive(Serialize, Clone, Copy, Debug, Default)]
pub struct Summary {
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

/// Sort `values` and summarize them (nearest-rank percentiles).
pub fn summarize(values: &mut [f64]) -> Summary {
    if values.is_empty() {
        return Summary::default();
    }
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = values.len();
    let at = |p: f64| values[(((n - 1) as f64) * p).round() as usize];
    Summary {
        mean: values.iter().sum::<f64>() / n as f64,
        median: at(0.5),
        p95: at(0.95),
        max: values[n - 1],
    }
}

/// Fraction of `truth` ids present in `found`.
pub fn overlap(found: &[(f32, u32)], truth: &[(f32, u32)]) -> f64 {
    if truth.is_empty() {
        return 0.0;
    }
    let hit = truth
        .iter()
        .filter(|(_, t)| found.iter().any(|(_, f)| f == t))
        .count();
    hit as f64 / truth.len() as f64
}

fn insert_sorted(top: &mut Vec<(f32, u32)>, item: (f32, u32), k: usize) {
    let pos = top.partition_point(|e| e.0 > item.0 || (e.0 == item.0 && e.1 < item.1));
    top.insert(pos, item);
    if top.len() > k {
        top.pop();
    }
}

/// Exact top-`k` by cosine (rows are unit norm) for every query row, scored
/// in batches of eight queries per corpus pass. Each result is sorted by
/// score descending, id ascending.
pub fn exact_topk(docs: &[f32], dim: usize, n: usize, queries: &[f32], k: usize) -> Vec<Vec<(f32, u32)>> {
    const BATCH: usize = 8;
    let nq = queries.len() / dim;
    let mut result: Vec<Vec<(f32, u32)>> = Vec::with_capacity(nq);
    for qb in queries.chunks(BATCH * dim) {
        let qs: Vec<&[f32]> = qb.chunks_exact(dim).collect();
        let mut tops: Vec<Vec<(f32, u32)>> = vec![Vec::with_capacity(k + 1); qs.len()];
        for (i, row) in docs[..n * dim].chunks_exact(dim).enumerate() {
            for (j, q) in qs.iter().enumerate() {
                let s = dot_f32(row, q);
                let top = &mut tops[j];
                if top.len() < k || s > top[top.len() - 1].0 {
                    insert_sorted(top, (s, i as u32), k);
                }
            }
        }
        result.extend(tops);
    }
    result
}

/// Exact-cosine rerank of `cands` against `q`; writes the top `top` (score
/// descending, id ascending) into `out`.
pub fn rerank(
    docs: &[f32],
    dim: usize,
    cands: &[u32],
    q: &[f32],
    top: usize,
    scores: &mut Vec<(f32, u32)>,
    out: &mut Vec<(f32, u32)>,
) {
    scores.clear();
    for &id in cands {
        let i = id as usize;
        scores.push((dot_f32(&docs[i * dim..(i + 1) * dim], q), id));
    }
    let cmp = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    if scores.len() > top {
        scores.select_nth_unstable_by(top - 1, cmp);
        scores.truncate(top);
    }
    scores.sort_unstable_by(cmp);
    out.clear();
    out.extend_from_slice(scores);
}

/// Query-time knobs.
#[derive(Clone, Copy, Debug)]
pub struct QueryParams {
    /// Channels selected per query.
    pub q: usize,
    /// Witness threshold.
    pub t: u8,
    /// Results returned.
    pub top: usize,
}

/// Per-query reusable buffers.
pub struct QueryScratch {
    /// `dirs` projections.
    pub proj: Vec<f32>,
    /// Selector buffers.
    pub sel: SelectScratch,
    /// Selected channels.
    pub chans: Vec<u32>,
    /// Emitted candidates.
    pub cands: Vec<u32>,
    /// Rerank scores.
    pub scores: Vec<(f32, u32)>,
}

impl QueryScratch {
    /// Buffers sized for `bank`.
    pub fn new(bank: &Bank) -> Self {
        Self {
            proj: vec![0.0f32; bank.dirs],
            sel: SelectScratch::default(),
            chans: Vec::with_capacity(256),
            cands: Vec::with_capacity(8192),
            scores: Vec::with_capacity(8192),
        }
    }
}

/// What one query touched.
#[derive(Clone, Copy, Debug, Default)]
pub struct Trace {
    /// Bytes of posting ids streamed.
    pub posting_bytes: usize,
    /// Candidates emitted by the join (= rows reranked).
    pub candidates: usize,
}

/// The complete index for one `(M, B)`: bank, thresholds, postings, joiner.
pub struct Index {
    /// Channel bank with fitted tables.
    pub bank: Bank,
    /// Membership budget the postings were built with.
    pub b: usize,
    /// Per-channel membership thresholds (`M` entries).
    pub tau: Vec<f32>,
    /// Sorted posting lists.
    pub postings: Postings,
    /// Counter array (`n` entries).
    pub joiner: Joiner,
}

impl Index {
    /// Bytes of the index proper (postings, bank tables, thresholds); rerank
    /// rows are accounted separately by the caller.
    pub fn index_bytes(&self) -> usize {
        self.postings.bytes() + self.bank.table_bytes() + self.tau.len() * 4
    }

    /// The full query: projection, selection, threshold-join, bulk rerank.
    pub fn query(
        &mut self,
        docs: &[f32],
        qv: &[f32],
        p: &QueryParams,
        s: &mut QueryScratch,
        out: &mut Vec<(f32, u32)>,
    ) -> Trace {
        self.bank.project(qv, &mut s.proj);
        self.bank.select(&s.proj, p.q, &mut s.sel, &mut s.chans);
        s.cands.clear();
        let posting_bytes = self.joiner.join_emit(&self.postings, &s.chans, p.t, &mut s.cands);
        self.joiner.reset(&self.postings, &s.chans);
        rerank(docs, self.bank.dim, &s.cands, qv, p.top, &mut s.scores, out);
        Trace {
            posting_bytes,
            candidates: s.cands.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar_with_remainder() {
        let a: Vec<f32> = (0..21).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..21).map(|i| 1.0 - i as f32 * 0.1).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot_f32(&a, &b) - want).abs() < 1e-3);
    }

    #[test]
    fn exact_topk_and_rerank_agree() {
        let dim = 8;
        let n = 200;
        let mut rng = data::Rng::new(3);
        let mut docs = vec![0.0f32; n * dim];
        for row in docs.chunks_exact_mut(dim) {
            for x in row.iter_mut() {
                *x = rng.uniform() as f32 - 0.5;
            }
            normalize_in_place(row);
        }
        let q: Vec<f32> = docs[..dim].to_vec();
        let gt = exact_topk(&docs, dim, n, &q, 5);
        assert_eq!(gt.len(), 1);
        assert_eq!(gt[0][0].1, 0);
        let all: Vec<u32> = (0..n as u32).collect();
        let mut scores = Vec::new();
        let mut out = Vec::new();
        rerank(&docs, dim, &all, &q, 5, &mut scores, &mut out);
        assert_eq!(out, gt[0]);
    }

    #[test]
    fn summary_percentiles_are_nearest_rank() {
        let mut v: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let s = summarize(&mut v);
        assert_eq!(s.median, 50.0);
        assert_eq!(s.p95, 95.0);
        assert_eq!(s.max, 100.0);
    }
}
