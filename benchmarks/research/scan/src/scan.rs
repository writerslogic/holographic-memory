// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Linear scan kernels and the search pipeline.
//!
//! Each kernel is a single pass over one contiguous array with a cached
//! admission threshold from a bounded [`TopK`] heap; the heap is touched only
//! when a candidate beats the current K-th score. The split kernels are
//! monomorphized over the head word count `HW` (2/4/8 for head 16/32/64) and
//! the residual word count `W` so the head dot and the XOR/AND + popcount loops
//! fully unroll; the int8 and exact kernels use the lane-accumulator dots from
//! the crate root.
//!
//! Pipeline for the approximate methods: transform the query once, scan to a
//! K-candidate heap, exact f32 rerank of the live candidates from
//! [`Primary`], return the top `top` ids in descending score order.
//!
//! Scans never read the liveness flags; the exact scan and the rerank do.

use crate::codes::{
    Basis, Int8Index, Int8Query, Primary, ResidualMode, Scratch, SplitIndex, SplitQuery,
    MAX_HEAD_DIM,
};
use crate::{dot_f32, dot_i8, f16_fast, TopK};

#[inline(always)]
fn head_dot<const HW: usize>(words: &[u64; HW], hq: &[i32; MAX_HEAD_DIM]) -> i32 {
    let mut dot = 0i32;
    for w in 0..HW {
        let b = words[w].to_le_bytes();
        for j in 0..8 {
            dot += (b[j] as i8 as i32) * hq[w * 8 + j];
        }
    }
    dot
}

/// `sym1` kernel: head dot + Hamming distance on `W` words.
#[inline(never)]
fn scan_sym<const HW: usize, const W: usize>(codes: &[u64], q: &SplitQuery, heap: &mut TopK) {
    let stride = HW + 1 + W;
    let qb: &[u64; W] = q.bits[..W]
        .try_into()
        .expect("query bits are sized to the basis word count");
    let hq = &q.head_q;
    let a_q = q.a_q;
    let b_q = q.b_q;
    let c_q = q.c_q;
    let mut thr = heap.threshold();
    for (i, rec) in codes.chunks_exact(stride).enumerate() {
        let head: &[u64; HW] = rec[..HW].try_into().expect("record stride invariant");
        let dot = head_dot::<HW>(head, hq);
        let sc = rec[HW];
        let hs = f16_fast((sc & 0xffff) as u16);
        let rho = f16_fast(((sc >> 16) & 0xffff) as u16);
        let bits: &[u64; W] = rec[HW + 1..HW + 1 + W]
            .try_into()
            .expect("record stride invariant");
        let mut ham = 0u32;
        for w in 0..W {
            ham += (bits[w] ^ qb[w]).count_ones();
        }
        let score = hs * a_q * (dot as f32) + rho * (b_q - c_q * (ham as f32));
        if score > thr {
            heap.push(score, i as u32);
            thr = heap.threshold();
        }
    }
}

/// `asym4` kernel: head dot + four AND/popcount planes on `W` words.
#[inline(never)]
fn scan_asym<const HW: usize, const W: usize>(codes: &[u64], q: &SplitQuery, heap: &mut TopK) {
    let stride = HW + 1 + W;
    let p0: &[u64; W] = q.planes[0][..W].try_into().expect("plane 0 sized to W");
    let p1: &[u64; W] = q.planes[1][..W].try_into().expect("plane 1 sized to W");
    let p2: &[u64; W] = q.planes[2][..W].try_into().expect("plane 2 sized to W");
    let p3: &[u64; W] = q.planes[3][..W].try_into().expect("plane 3 sized to W");
    let hq = &q.head_q;
    let a_q = q.a_q;
    let delta2 = q.delta2;
    let lo2 = q.lo2;
    let zsum = q.zsum;
    let mut thr = heap.threshold();
    for (i, rec) in codes.chunks_exact(stride).enumerate() {
        let head: &[u64; HW] = rec[..HW].try_into().expect("record stride invariant");
        let dot = head_dot::<HW>(head, hq);
        let sc = rec[HW];
        let hs = f16_fast((sc & 0xffff) as u16);
        let rho = f16_fast(((sc >> 16) & 0xffff) as u16);
        let ones = ((sc >> 32) & 0xffff) as f32;
        let bits: &[u64; W] = rec[HW + 1..HW + 1 + W]
            .try_into()
            .expect("record stride invariant");
        let mut s1 = 0u32;
        for w in 0..W {
            let b = bits[w];
            s1 += (b & p0[w]).count_ones()
                + ((b & p1[w]).count_ones() << 1)
                + ((b & p2[w]).count_ones() << 2)
                + ((b & p3[w]).count_ones() << 3);
        }
        let score = hs * a_q * (dot as f32) + rho * (delta2 * (s1 as f32) + lo2 * ones - zsum);
        if score > thr {
            heap.push(score, i as u32);
            thr = heap.threshold();
        }
    }
}

macro_rules! dispatch_words {
    ($hw:literal, $w:expr, $kernel:ident, $codes:expr, $q:expr, $heap:expr) => {
        match $w {
            1 => $kernel::<$hw, 1>($codes, $q, $heap),
            2 => $kernel::<$hw, 2>($codes, $q, $heap),
            3 => $kernel::<$hw, 3>($codes, $q, $heap),
            4 => $kernel::<$hw, 4>($codes, $q, $heap),
            6 => $kernel::<$hw, 6>($codes, $q, $heap),
            8 => $kernel::<$hw, 8>($codes, $q, $heap),
            12 => $kernel::<$hw, 12>($codes, $q, $heap),
            16 => $kernel::<$hw, 16>($codes, $q, $heap),
            24 => $kernel::<$hw, 24>($codes, $q, $heap),
            32 => $kernel::<$hw, 32>($codes, $q, $heap),
            w => unreachable!("residual word count {w} is not in SUPPORTED_WORDS (fit_basis validates it)"),
        }
    };
}

macro_rules! dispatch_kernel {
    ($hw:expr, $w:expr, $kernel:ident, $codes:expr, $q:expr, $heap:expr) => {
        match $hw {
            2 => dispatch_words!(2, $w, $kernel, $codes, $q, $heap),
            4 => dispatch_words!(4, $w, $kernel, $codes, $q, $heap),
            8 => dispatch_words!(8, $w, $kernel, $codes, $q, $heap),
            hw => unreachable!("head word count {hw} is not in {{2, 4, 8}} (fit_basis validates it)"),
        }
    };
}

/// Scan the split index into `heap` (capacity = K). The heap is not cleared.
pub fn scan_split(ix: &SplitIndex, q: &SplitQuery, heap: &mut TopK) {
    assert_eq!(q.mode, ix.mode, "query prepared for a different residual mode");
    let codes = &ix.codes[..ix.n * ix.stride];
    let hw = ix.basis.head_words;
    let w = ix.basis.words;
    match ix.mode {
        ResidualMode::Sym1 => dispatch_kernel!(hw, w, scan_sym, codes, q, heap),
        ResidualMode::Asym4 => dispatch_kernel!(hw, w, scan_asym, codes, q, heap),
    }
}

/// Scan the int8 index into `heap`.
pub fn scan_int8(ix: &Int8Index, q: &Int8Query, heap: &mut TopK) {
    let d = ix.dim;
    let qv = &q.q[..d];
    let qs = q.scale;
    let mut thr = heap.threshold();
    for (i, row) in ix.codes[..ix.n * d].chunks_exact(d).enumerate() {
        let acc = dot_i8(row, qv);
        let score = ix.scales[i] * qs * (acc as f32);
        if score > thr {
            heap.push(score, i as u32);
            thr = heap.threshold();
        }
    }
}

/// Exact f32 scan of the live rows into `heap`.
pub fn scan_exact(p: &Primary, q: &[f32], heap: &mut TopK) {
    let d = p.dim;
    let mut thr = heap.threshold();
    for (i, row) in p.rows[..p.n * d].chunks_exact(d).enumerate() {
        let s = dot_f32(row, q);
        if s > thr && p.live[i] != 0 {
            heap.push(s, i as u32);
            thr = heap.threshold();
        }
    }
}

/// Exact rerank of the live candidates; writes the top ids (descending) to `out`.
pub fn rerank(p: &Primary, q: &[f32], cands: &TopK, top: &mut TopK, out: &mut Vec<u32>) {
    top.clear();
    for &(_, id) in cands.entries() {
        let idx = id as usize;
        if p.live[idx] == 0 {
            continue;
        }
        let s = dot_f32(p.row(idx), q);
        if s > top.threshold() {
            top.push(s, id);
        }
    }
    top.drain_sorted_desc(out);
}

/// Search state shared by every method: heaps, int8 query, result buffer.
pub struct Baseline {
    /// K-candidate heap.
    pub cands: TopK,
    /// Final top heap.
    pub top: TopK,
    /// Quantized int8 query.
    pub iq: Int8Query,
    /// Result ids of the last search.
    pub out: Vec<u32>,
}

impl Baseline {
    /// State for dimension `dim` with candidate capacity `k` and `top` results.
    pub fn new(dim: usize, k: usize, top: usize) -> Self {
        Self {
            cands: TopK::new(k),
            top: TopK::new(top),
            iq: Int8Query::new(dim),
            out: Vec::with_capacity(top),
        }
    }

    /// Change the candidate capacity.
    pub fn set_k(&mut self, k: usize) {
        if self.cands.k() != k {
            self.cands = TopK::new(k);
        }
    }

    /// Int8 search: quantize, scan, rerank.
    pub fn int8(&mut self, ix: &Int8Index, p: &Primary, q: &[f32]) -> &[u32] {
        self.iq.prepare(q);
        self.cands.clear();
        scan_int8(ix, &self.iq, &mut self.cands);
        rerank(p, q, &self.cands, &mut self.top, &mut self.out);
        &self.out
    }

    /// Exact search: f32 scan straight into the top heap.
    pub fn exact(&mut self, p: &Primary, q: &[f32]) -> &[u32] {
        self.top.clear();
        scan_exact(p, q, &mut self.top);
        self.top.drain_sorted_desc(&mut self.out);
        &self.out
    }
}

/// [`Baseline`] plus the split-manifold transform buffers for one basis.
pub struct Searcher {
    /// Shared heaps and buffers.
    pub base: Baseline,
    /// Transform scratch.
    pub scratch: Scratch,
    /// Transformed split query.
    pub sq: SplitQuery,
}

impl Searcher {
    /// State for `basis`/`mode` with candidate capacity `k` and `top` results.
    pub fn new(basis: &Basis, mode: ResidualMode, k: usize, top: usize) -> Self {
        Self {
            base: Baseline::new(basis.dim, k, top),
            scratch: Scratch::new(basis),
            sq: SplitQuery::new(basis, mode),
        }
    }

    /// Change the candidate capacity.
    pub fn set_k(&mut self, k: usize) {
        self.base.set_k(k);
    }

    /// Split-manifold search: transform, scan, rerank.
    pub fn split(&mut self, ix: &SplitIndex, p: &Primary, q: &[f32]) -> &[u32] {
        ix.basis.prepare_query(q, &mut self.scratch, &mut self.sq);
        self.base.cands.clear();
        scan_split(ix, &self.sq, &mut self.base.cands);
        rerank(p, q, &self.base.cands, &mut self.base.top, &mut self.base.out);
        &self.base.out
    }

    /// Int8 search through the shared state.
    pub fn int8(&mut self, ix: &Int8Index, p: &Primary, q: &[f32]) -> &[u32] {
        self.base.int8(ix, p, q)
    }

    /// Exact search through the shared state.
    pub fn exact(&mut self, p: &Primary, q: &[f32]) -> &[u32] {
        self.base.exact(p, q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::{fit_basis, BasisKind, BasisParams};
    use crate::data::{synthetic, SyntheticParams};

    fn fixture(
        mode: ResidualMode,
        head_dim: usize,
        mult: usize,
        bits: usize,
    ) -> (SplitIndex, Int8Index, Primary, Vec<f32>, usize) {
        let d = 96;
        let ds = synthetic(&SyntheticParams {
            n: 2000,
            dim: d,
            clusters: 10,
            n_queries: 20,
            seed: 11,
            spread: 0.75,
            spectrum_alpha: 0.5,
        });
        let basis = fit_basis(
            &BasisParams {
                kind: BasisKind::Pca,
                dim: d,
                head_dim,
                residual_mult: mult,
                residual_bits: bits,
                seed: 11,
                pca_iters: 16,
            },
            &ds.docs[..1000 * d],
        )
        .expect("basis");
        let mut split = SplitIndex::with_capacity(basis, ds.n_docs, mode);
        let mut s = Scratch::new(&split.basis);
        let mut int8 = Int8Index::with_capacity(ds.n_docs, d);
        for i in 0..ds.n_docs {
            split.encode(i, ds.doc(i), &mut s);
            int8.encode(i, ds.doc(i));
        }
        let primary = Primary::new(ds.docs, d);
        (split, int8, primary, ds.queries, ds.n_queries)
    }

    fn brute_top(p: &Primary, q: &[f32], top: usize) -> Vec<u32> {
        let mut all: Vec<(f32, u32)> = (0..p.n)
            .filter(|&i| p.live[i] != 0)
            .map(|i| (dot_f32(p.row(i), q), i as u32))
            .collect();
        all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        all.iter().take(top).map(|e| e.1).collect()
    }

    #[test]
    fn full_candidate_set_reproduces_exact_top10_across_modes_heads_and_mults() {
        for mode in [ResidualMode::Sym1, ResidualMode::Asym4] {
            for (head_dim, mult, bits) in [(16usize, 1usize, 128usize), (32, 2, 256), (64, 1, 64)] {
                let (split, int8, primary, queries, nq) = fixture(mode, head_dim, mult, bits);
                let d = primary.dim;
                let mut srch = Searcher::new(&split.basis, mode, primary.n, 10);
                for qi in 0..nq {
                    let q = &queries[qi * d..(qi + 1) * d];
                    let want = brute_top(&primary, q, 10);
                    assert_eq!(srch.exact(&primary, q), &want[..], "exact mode={mode:?}");
                    assert_eq!(
                        srch.split(&split, &primary, q),
                        &want[..],
                        "split mode={mode:?} head={head_dim} mult={mult}"
                    );
                    assert_eq!(srch.int8(&int8, &primary, q), &want[..], "int8 mode={mode:?}");
                }
            }
        }
    }

    #[test]
    fn small_candidate_set_has_useful_recall_and_deletes_never_leak() {
        let (mut split, mut int8, mut primary, queries, nq) =
            fixture(ResidualMode::Sym1, 16, 1, 128);
        let d = primary.dim;
        let mut srch = Searcher::new(&split.basis, ResidualMode::Sym1, 64, 10);
        let mut hits = 0usize;
        for qi in 0..nq {
            let q = &queries[qi * d..(qi + 1) * d];
            let want = brute_top(&primary, q, 10);
            let got = srch.split(&split, &primary, q).to_vec();
            hits += got.iter().filter(|g| want.contains(g)).count();
        }
        let recall = hits as f64 / (10 * nq) as f64;
        assert!(recall > 0.5, "recall@10 with K=64 on 2k docs = {recall}");

        let dead: Vec<usize> = (0..primary.n).step_by(3).collect();
        for &id in &dead {
            split.delete(id);
            int8.delete(id);
            primary.delete(id);
        }
        for qi in 0..nq {
            let q = &queries[qi * d..(qi + 1) * d];
            for id in srch.split(&split, &primary, q).to_vec() {
                assert!(primary.live[id as usize] != 0, "split leaked deleted {id}");
            }
            for id in srch.int8(&int8, &primary, q).to_vec() {
                assert!(primary.live[id as usize] != 0, "int8 leaked deleted {id}");
            }
            for id in srch.exact(&primary, q).to_vec() {
                assert!(primary.live[id as usize] != 0, "exact leaked deleted {id}");
            }
        }
    }
}
