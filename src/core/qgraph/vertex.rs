// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Graph index with one compact code per vertex instead of one code per edge.
//!
//! Every vertex stores its mean-centred vector as 8-bit integers with one scale (locally
//! adaptive scalar quantization, LVQ; Aguerrebere et al., VLDB 2023). Traversal estimates
//! `<q, x> - <q, mean> ~= s_x * s_q * <c_q, c_x>` with an int8 query code and one integer dot
//! product per neighbour, so a neighbour costs one short code fetch (`dim` bytes, the scales
//! live in their own small array) rather than a full-precision vector. An optional second code
//! of the quantization residual (8-bit, LVQ-8x8, or 4-bit) re-ranks the best `rerank`
//! candidates of the final pool with a float query.
//!
//! Memory per vertex is `4 * degree + dim + 4` bytes, plus `dim + 4` with the 8-bit residual
//! (`dim / 2 + 4` with the 4-bit one), against `4 * dim` for the vectors alone in an HNSW
//! index.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use super::kernels::{
    dot_f32_i4, dot_f32_i8, dot_i4_inline, dot_i8_inline, interleave_for_i4, prefetch,
    prefetch_bytes, raw_one, QueryCode, Rotation, SplitMix,
};
use super::{invalid, upper_bytes, BuildParams, Graph, NONE};
use crate::core::error::HmsError;

/// Search parameters of a [`VGraph`].
#[derive(Clone, Copy, Debug)]
pub struct VSearchParams {
    /// Candidate pool size (beam width).
    pub ef: usize,
    /// Number of best pool candidates re-scored with the float query (and the residual code,
    /// when built); 0 returns the pool order.
    pub rerank: usize,
    /// Early stop: end the traversal once this many consecutive expansions have left the
    /// `k` best estimates unchanged (0 = the pool rule alone). The pool rule still applies.
    pub patience: usize,
    /// With a screen (see [`VGraph::from_graph_with`]): a fresh neighbour gets the 8-bit
    /// estimate only if its 1-bit estimate minus this many standard deviations of the
    /// screen's error could still enter the full pool. Larger values skip less.
    pub screen_sigmas: f32,
    /// Consult the screen when the index has one (false runs the plain search on it).
    pub screen: bool,
}

impl Default for VSearchParams {
    fn default() -> Self {
        Self {
            ef: 64,
            rerank: 0,
            patience: 0,
            screen_sigmas: 1.0,
            screen: true,
        }
    }
}

/// One 128-byte cache line (the line size of Apple cores).
#[repr(C, align(128))]
#[derive(Clone, Copy)]
struct Line([i8; 128]);

/// Zeroed bytes starting on a cache-line boundary.
struct Lines {
    lines: Vec<Line>,
    len: usize,
}

impl Lines {
    fn zeroed(len: usize) -> Self {
        Self {
            lines: vec![Line([0; 128]); len.div_ceil(128)],
            len,
        }
    }

    fn bytes(&self) -> &[i8] {
        // SAFETY: `Line` is `repr(C)` around `[i8; 128]` (size 128, no padding), so `lines` is
        // `128 * lines.len() >= len` initialized, contiguous bytes.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr().cast::<i8>(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [i8] {
        // SAFETY: as in `bytes`, with the exclusive borrow of `lines`.
        unsafe { std::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast::<i8>(), self.len) }
    }
}

pub struct VGraph {
    n: usize,
    dim: usize,
    /// `dim` rounded up to a multiple of 16 (32 with 4-bit codes), the integer kernels' width.
    padded: usize,
    /// Bits per traversal code coordinate: 8, or 4 (two per byte).
    bits: u8,
    /// Bytes of one traversal code: `padded` or `padded / 2`.
    cbytes: usize,
    /// Bytes from one code row to the next: `cbytes`, or with aligned rows the smallest
    /// power of two (up to 64) or multiple of 128 that holds it, so no row straddles a line.
    stride: usize,
    degree: usize,
    mean: Vec<f32>,
    /// Neighbour ids, `degree` per vertex padded with `NONE`; empty with 3-byte ids.
    adj: Vec<u32>,
    /// The same as 3-byte little-endian ids (padded with `NONE3`); empty with `u32` ids.
    adj3: Vec<u8>,
    codes: Lines,
    scales: Vec<f32>,
    /// 0 (none), 4 or 8.
    res_bits: u8,
    /// Bytes per residual row: `padded` (8-bit) or `padded / 2` (4-bit, two per byte).
    res_stride: usize,
    residual: Vec<i8>,
    res_scales: Vec<f32>,
    entry: u32,
    upper_ids: Vec<u32>,
    layers: Vec<Vec<Vec<u32>>>,
    /// Original id of each stored vertex when the vertices were renumbered; empty otherwise.
    ids: Vec<u32>,
    /// 1-bit screen rows: `screen_words` u64 of sign bits of the rotated centred vector, then
    /// one u64 holding the f32 factor (low half) and the popcount (high half); empty without
    /// a screen.
    screen: Vec<u64>,
    screen_words: usize,
    screen_rot: Option<Rotation>,
    /// Standard deviation of (screen estimate - 8-bit estimate), in estimate units, measured
    /// at build on index vectors used as queries.
    screen_sigma: f32,
}

/// Seed of the screen's rotation (fixed: the screen is a property of the encoding).
const SCREEN_SEED: u64 = 0x5C12_EE00;

/// One vertex as encoded in parallel: traversal code and scale, residual code and scale,
/// screen row (empty without a screen).
type EncodedRow = (Vec<i8>, f32, Vec<i8>, f32, Vec<u64>);

/// Inverse of [`key`] for the distance part.
#[inline]
fn key_dist(k: u64) -> f32 {
    let o = (k >> 32) as u32;
    f32::from_bits(if o >> 31 == 1 { o & 0x7FFF_FFFF } else { !o })
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

/// [`quantize`] with 4-bit codes in `-7..=7`, packed two per byte as [`dot_f32_i4`] reads them
/// (`code` holds `code.len() * 2 >= x.len()` values).
fn quantize4(x: &[f32], code: &mut [i8]) -> f32 {
    let m = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    code.fill(0);
    if m == 0.0 {
        return 0.0;
    }
    let scale = m / 7.0;
    let q = |i: usize| {
        x.get(i)
            .map_or(0, |&v| (v / scale).round().clamp(-7.0, 7.0) as i8)
    };
    for (j, c) in code.iter_mut().enumerate() {
        *c = ((q(2 * j) as u8 & 0x0F) | ((q(2 * j + 1) as u8) << 4)) as i8;
    }
    scale
}

/// Breadth-first order of the bottom layer from the entry (unreached vertices follow in id
/// order): neighbours get nearby positions, so a search touches fewer distinct pages and lines.
fn bfs_order(g: &Graph) -> Vec<u32> {
    let mut seen = vec![false; g.n];
    let mut order = Vec::with_capacity(g.n);
    for s in std::iter::once(g.entry as usize).chain(0..g.n) {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut head = order.len();
        order.push(s as u32);
        while head < order.len() {
            let v = order[head] as usize;
            head += 1;
            for &u in &g.adj[v] {
                if !std::mem::replace(&mut seen[u as usize], true) {
                    order.push(u);
                }
            }
        }
    }
    order
}

/// Coordinate `i` of a code row of `bits` bits (0 past its end).
fn code_at(c: &[i8], bits: u8, i: usize) -> i8 {
    if bits == 8 {
        c.get(i).copied().unwrap_or(0)
    } else {
        c.get(i / 2).map_or(0, |&b| {
            if i.is_multiple_of(2) {
                (((b as u8) << 4) as i8) >> 4
            } else {
                b >> 4
            }
        })
    }
}

/// Padding of a 3-byte neighbour list.
const NONE3: u32 = 0x00FF_FFFF;

#[inline]
fn prefetch_u8(bytes: &[u8]) {
    // SAFETY: any initialized `u8` slice is also initialized `i8`s.
    prefetch_bytes(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) });
}

#[inline]
fn prefetch_ids(ids: &[u32]) {
    // SAFETY: any initialized `u32` slice is also `4 * len` initialized bytes.
    prefetch_bytes(unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<i8>(), ids.len() * 4) });
}

/// Orders by distance, then id; `f32` order is kept for every non-NaN distance.
#[inline]
fn key(dist: f32, id: u32) -> u64 {
    let b = dist.to_bits();
    let o = if b >> 31 == 1 { !b } else { b | 0x8000_0000 };
    (u64::from(o) << 32) | u64::from(id)
}

#[inline]
fn key_id(k: u64) -> u32 {
    k as u32
}

impl VGraph {
    /// Encodes a built graph with the 8-bit residual code used by `rerank` (or none), unaligned
    /// rows and the build's vertex order.
    pub fn from_graph(g: &Graph, residual: bool) -> Result<Self, HmsError> {
        Self::from_graph_with(g, 8, if residual { 8 } else { 0 }, false, false, 4, false)
    }

    /// Encodes a built graph. `bits` (8 or 4) sizes the traversal code and `residual_bits`
    /// (0, 4 or 8) the re-rank code of its residual; `align` keeps every code row inside as
    /// few 128-byte lines as its size allows; `reorder` renumbers the vertices in breadth-first
    /// order (search still returns the original ids); `id_bytes` 3 stores neighbour ids in
    /// three bytes (graphs below 2^24 - 1 vertices) instead of 4; `screen` adds a 1-bit
    /// sign code of the rotated centred vector per vertex (RaBitQ-style, with a per-vertex
    /// factor) that the search consults before paying for the 8-bit estimate.
    pub fn from_graph_with(
        g: &Graph,
        bits: u8,
        residual_bits: u8,
        align: bool,
        reorder: bool,
        id_bytes: u8,
        screen: bool,
    ) -> Result<Self, HmsError> {
        if !matches!(bits, 4 | 8) {
            return Err(invalid(format!(
                "vertex code bits must be 4 or 8, not {bits}"
            )));
        }
        if !(id_bytes == 4 || (id_bytes == 3 && (g.n as u64) < u64::from(NONE3))) {
            return Err(invalid(format!(
                "id_bytes must be 4, or 3 below 2^24 - 1 vertices (got {id_bytes} for {} vertices)",
                g.n
            )));
        }
        if !matches!(residual_bits, 0 | 4 | 8) {
            return Err(invalid(format!(
                "residual code bits must be 0, 4 or 8, not {residual_bits}"
            )));
        }
        let (n, dim) = (g.n, g.dim);
        let padded = dim.next_multiple_of(if bits == 4 { 32 } else { 16 });
        let cbytes = if bits == 4 { padded / 2 } else { padded };
        let stride = match (align, cbytes) {
            (false, _) => cbytes,
            (true, c) if c <= 64 => c.next_power_of_two(),
            (true, c) => c.next_multiple_of(128),
        };
        let res_stride = match residual_bits {
            8 => padded,
            4 => padded / 2,
            _ => 0,
        };
        let mut sum = vec![0f64; dim];
        for x in g.unit.chunks_exact(dim) {
            sum.iter_mut().zip(x).for_each(|(s, &v)| *s += f64::from(v));
        }
        let mut mean: Vec<f32> = sum.iter().map(|s| (s / n as f64) as f32).collect();
        mean.resize(padded, 0.0);

        let ids = if reorder { bfs_order(g) } else { Vec::new() };
        let old = |v: usize| if reorder { ids[v] as usize } else { v };
        let mut new_id: Vec<u32> = Vec::new();
        if reorder {
            new_id = vec![0; n];
            for (v, &o) in ids.iter().enumerate() {
                new_id[o as usize] = v as u32;
            }
        }
        let new = |o: u32| if reorder { new_id[o as usize] } else { o };

        let screen_rot = screen.then(|| Rotation::new(dim, 0, SCREEN_SEED));
        let screen_words = screen_rot.as_ref().map_or(0, |r| r.padded() / 64);
        let rows: Vec<EncodedRow> = (0..n)
            .into_par_iter()
            .map(|v| {
                let o = old(v);
                let mut x: Vec<f32> = g.unit[o * dim..(o + 1) * dim]
                    .iter()
                    .zip(&mean)
                    .map(|(a, m)| a - m)
                    .collect();
                let mut srow = Vec::new();
                if let Some(rot) = &screen_rot {
                    let mut xr = vec![0f32; rot.padded()];
                    rot.apply(&x, &mut xr);
                    let (mut norm2, mut abs) = (0f32, 0f32);
                    let mut pop = 0u32;
                    srow = vec![0u64; screen_words + 1];
                    for (i, &a) in xr.iter().enumerate() {
                        norm2 += a * a;
                        abs += a.abs();
                        if a >= 0.0 {
                            srow[i / 64] |= 1u64 << (i % 64);
                            pop += 1;
                        }
                    }
                    // x_r ~= f * sign(x_r) with f = |x_r|^2 / sum |x_r,i| (the RaBitQ factor).
                    let f = if abs > 0.0 { norm2 / abs } else { 0.0 };
                    srow[screen_words] = u64::from(f.to_bits()) | (u64::from(pop) << 32);
                }
                let mut c = vec![0i8; cbytes];
                let s = if bits == 8 {
                    quantize(&x, &mut c)
                } else {
                    quantize4(&x, &mut c)
                };
                let mut r = vec![0i8; res_stride];
                let mut sr = 0.0;
                if residual_bits > 0 {
                    x.iter_mut()
                        .enumerate()
                        .for_each(|(i, a)| *a -= s * f32::from(code_at(&c, bits, i)));
                    sr = if residual_bits == 8 {
                        quantize(&x, &mut r)
                    } else {
                        quantize4(&x, &mut r)
                    };
                }
                (c, s, r, sr, srow)
            })
            .collect();
        let mut codes = Lines::zeroed(n * stride);
        let mut scales = Vec::with_capacity(n);
        let mut residual = Vec::with_capacity(n * res_stride);
        let mut res_scales = Vec::with_capacity(if residual_bits > 0 { n } else { 0 });
        let mut screen_rows = Vec::with_capacity(if screen { n * (screen_words + 1) } else { 0 });
        for ((c, s, r, sr, srow), dst) in rows
            .into_iter()
            .zip(codes.bytes_mut().chunks_exact_mut(stride))
        {
            dst[..cbytes].copy_from_slice(&c);
            scales.push(s);
            if residual_bits > 0 {
                residual.extend_from_slice(&r);
                res_scales.push(sr);
            }
            screen_rows.extend_from_slice(&srow);
        }

        let r = g.degree;
        let mut adj = vec![NONE; n * r];
        for (v, dst) in adj.chunks_exact_mut(r).enumerate() {
            for (d, &u) in dst.iter_mut().zip(&g.adj[old(v)]) {
                *d = new(u);
            }
        }
        let mut adj3 = Vec::new();
        if id_bytes == 3 {
            adj3 = adj
                .iter()
                .flat_map(|&u| (u.min(NONE3)).to_le_bytes()[..3].to_vec())
                .collect();
            adj = Vec::new();
        }
        let mut out = Self {
            n,
            dim,
            padded,
            bits,
            cbytes,
            stride,
            degree: r,
            mean,
            adj,
            adj3,
            codes,
            scales,
            res_bits: residual_bits,
            res_stride,
            residual,
            res_scales,
            entry: new(g.entry),
            upper_ids: g.upper_ids.iter().map(|&u| new(u)).collect(),
            layers: g.layers.clone(),
            ids,
            screen: screen_rows,
            screen_words,
            screen_rot,
            screen_sigma: 0.0,
        };
        if screen {
            out.screen_sigma = out.measure_screen_sigma(g);
        }
        Ok(out)
    }

    /// Standard deviation of the screen estimate's error against the 8-bit estimate over
    /// index vectors used as queries (deterministic sample), in estimate units.
    fn measure_screen_sigma(&self, g: &Graph) -> f32 {
        let (nq, nv) = (128.min(self.n), 1024.min(self.n));
        let mut rng = SplitMix(SCREEN_SEED ^ 1);
        let mut qc = vec![0i8; self.padded];
        let mut qtmp = vec![0i8; self.padded];
        let mut qf = vec![0f32; self.padded];
        let mut qrot = vec![0f32; self.screen_words * 64];
        let mut qs = QueryCode::new(self.screen_words);
        let (mut sum, mut sum2, mut count) = (0f64, 0f64, 0usize);
        for _ in 0..nq {
            let q = (rng.next_u64() % self.n as u64) as usize;
            qf[..self.dim].copy_from_slice(&g.unit[q * self.dim..(q + 1) * self.dim]);
            let sq = self.encode_query(&qf, &mut qc, &mut qtmp, &mut qrot, &mut qs);
            if sq == 0.0 {
                continue;
            }
            for _ in 0..nv {
                let v = (rng.next_u64() % self.n as u64) as u32;
                let d = f64::from(self.screen_est(&qs, v) - self.est(&qc, v));
                sum += d;
                sum2 += d * d;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        let mean = sum / count as f64;
        ((sum2 / count as f64) - mean * mean).max(0.0).sqrt() as f32
    }

    /// Writes the traversal code of `qf` (padded) into `qc` and, with a screen, the rotated
    /// query scaled like the code into `qs`; returns the query's quantization scale.
    fn encode_query(
        &self,
        qf: &[f32],
        qc: &mut [i8],
        qtmp: &mut [i8],
        qrot: &mut [f32],
        qs: &mut QueryCode,
    ) -> f32 {
        let sq = if self.bits == 8 {
            quantize(qf, qc)
        } else {
            let s = quantize(qf, qtmp);
            interleave_for_i4(qtmp, qc);
            s
        };
        if let Some(rot) = &self.screen_rot {
            rot.apply(&qf[..self.dim], qrot);
            if sq > 0.0 {
                // The 8-bit estimate drops the query scale, so the screen works in the same
                // units: rotated query divided by s_q.
                qrot.iter_mut().for_each(|v| *v /= sq);
            }
            qs.encode(qrot);
        }
        sq
    }

    /// True if the index carries the 1-bit screen.
    pub fn has_screen(&self) -> bool {
        !self.screen.is_empty()
    }

    /// Bytes of the screen rows.
    pub fn screen_bytes(&self) -> usize {
        self.screen.len() * 8
    }

    /// 1-bit estimate in the units of [`VGraph::est`]: `-f * sum_i sign_i * q_i`, with the
    /// signed sum from the query's bit planes as `2 * (lo * pop + delta * raw) - sum(q)`.
    #[inline]
    fn screen_est(&self, qs: &QueryCode, v: u32) -> f32 {
        let w = self.screen_words;
        let row = &self.screen[v as usize * (w + 1)..(v as usize + 1) * (w + 1)];
        let raw = raw_one(&row[..w], &qs.planes, w) as f32;
        let meta = row[w];
        let f = f32::from_bits(meta as u32);
        let pop = (meta >> 32) as f32;
        -(f * (2.0 * (qs.lo * pop + qs.delta * raw) - qs.sum))
    }

    #[inline]
    fn prefetch_screen(&self, v: u32) {
        let w = self.screen_words;
        prefetch(&self.screen[v as usize * (w + 1)..(v as usize + 1) * (w + 1)]);
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn has_residual(&self) -> bool {
        self.res_bits > 0
    }

    /// Bytes held by the index, residual and id map included.
    pub fn index_bytes(&self) -> usize {
        self.adj.len() * 4
            + self.adj3.len()
            + self.n * self.stride
            + self.scales.len() * 4
            + self.residual_bytes()
            + self.mean.len() * 4
            + self.ids.len() * 4
            + self.screen_bytes()
            + upper_bytes(&self.upper_ids, &self.layers)
    }

    /// Bytes of the residual codes and their scales, which only `rerank` reads.
    pub fn residual_bytes(&self) -> usize {
        self.residual.len() + self.res_scales.len() * 4
    }

    #[inline]
    fn code(&self, v: u32) -> &[i8] {
        let at = v as usize * self.stride;
        &self.codes.bytes()[at..at + self.cbytes]
    }

    /// Calls `f` on every neighbour of `v`, in list order.
    #[inline]
    fn for_each_neighbour(&self, v: usize, mut f: impl FnMut(u32)) {
        let r = self.degree;
        if self.adj3.is_empty() {
            for &u in &self.adj[v * r..(v + 1) * r] {
                if u == NONE {
                    break;
                }
                f(u);
            }
        } else {
            for b in self.adj3[3 * v * r..3 * (v + 1) * r].as_chunks::<3>().0 {
                let u = u32::from_le_bytes([b[0], b[1], b[2], 0]);
                if u == NONE3 {
                    break;
                }
                f(u);
            }
        }
    }

    #[inline]
    fn prefetch_neighbours(&self, v: usize) {
        let r = self.degree;
        if self.adj3.is_empty() {
            prefetch_ids(&self.adj[v * r..(v + 1) * r]);
        } else {
            prefetch_u8(&self.adj3[3 * v * r..3 * (v + 1) * r]);
        }
    }

    #[inline]
    fn original(&self, v: u32) -> u32 {
        if self.ids.is_empty() {
            v
        } else {
            self.ids[v as usize]
        }
    }

    /// Estimated negative inner product with the query code (up to the per-query terms
    /// `<q, mean>` and `s_q`, which do not change the order).
    #[inline]
    fn est(&self, qc: &[i8], v: u32) -> f32 {
        let dot = if self.bits == 8 {
            dot_i8_inline(qc, self.code(v))
        } else {
            dot_i4_inline(qc, self.code(v))
        };
        -(self.scales[v as usize] * dot as f32)
    }

    /// Re-rank score with the float (centred) query: negative estimated inner product.
    fn fine(&self, qf: &[f32], v: u32) -> f32 {
        let primary = if self.bits == 8 {
            dot_f32_i8(qf, self.code(v))
        } else {
            dot_f32_i4(qf, self.code(v))
        };
        let mut ip = self.scales[v as usize] * primary;
        if self.has_residual() {
            let v = v as usize;
            let r = &self.residual[v * self.res_stride..(v + 1) * self.res_stride];
            let d = if self.res_bits == 8 {
                dot_f32_i8(qf, r)
            } else {
                dot_f32_i4(qf, r)
            };
            ip += self.res_scales[v] * d;
        }
        -ip
    }

    /// Greedy descent through the upper layers on estimated distances; returns the bottom-layer
    /// entry vertex and the number of estimates.
    fn descend(&self, qc: &[i8]) -> (u32, usize) {
        if self.layers.is_empty() {
            return (self.entry, 0);
        }
        let mut p = 0u32;
        let mut best = self.est(qc, self.upper_ids[0]);
        let mut evals = 1;
        for layer in self.layers.iter().rev() {
            loop {
                let mut moved = false;
                for &u in &layer[p as usize] {
                    let d = self.est(qc, self.upper_ids[u as usize]);
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
            visited: Marks {
                marks: vec![0; self.n],
                epoch: 0,
            },
            qf: vec![0.0; self.padded],
            qc: vec![0; self.padded],
            qtmp: vec![0; self.padded],
            qrot: vec![0.0; self.screen_words * 64],
            qs: QueryCode::new(self.screen_words.max(1)),
            fresh: Vec::with_capacity(self.degree),
            survivors: Vec::with_capacity(self.degree),
            screened: 0,
            cand: BinaryHeap::new(),
            best: BinaryHeap::new(),
            keys: Vec::new(),
            topk: Vec::new(),
            results: Vec::new(),
        }
    }
}

/// Visited marks with an 8-bit epoch: a quarter of the cache footprint of `u32` epochs, at the
/// cost of clearing every 255 searches.
struct Marks {
    marks: Vec<u8>,
    epoch: u8,
}

impl Marks {
    fn next(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// True if `id` was not yet visited in this epoch.
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        let m = &mut self.marks[id as usize];
        let fresh = *m != self.epoch;
        *m = self.epoch;
        fresh
    }
}

/// Per-thread search state for one [`VGraph`].
pub struct VSearcher<'a> {
    index: &'a VGraph,
    visited: Marks,
    qf: Vec<f32>,
    /// Query code in the traversal kernel's layout.
    qc: Vec<i8>,
    qtmp: Vec<i8>,
    /// Rotated, scaled query and its bit planes for the screen.
    qrot: Vec<f32>,
    qs: QueryCode,
    fresh: Vec<u32>,
    /// Fresh neighbours that passed the screen.
    survivors: Vec<u32>,
    /// Neighbours the screen skipped in the last search.
    screened: usize,
    /// Unexpanded candidates, nearest on top.
    cand: BinaryHeap<Reverse<u64>>,
    /// The `ef` nearest seen so far, farthest on top.
    best: BinaryHeap<u64>,
    keys: Vec<u64>,
    /// The `k` best keys seen, sorted, when `patience` is set.
    topk: Vec<u64>,
    results: Vec<(f32, u32)>,
}

impl VSearcher<'_> {
    /// Neighbours the screen skipped (no 8-bit estimate) in the last search.
    pub fn last_screened(&self) -> usize {
        self.screened
    }

    /// Writes the ids of the (approximate) `k` nearest vectors by cosine into `out`, nearest
    /// first, and returns the number of distance estimates (descent, traversal and re-rank).
    /// A query of the wrong dimension, with a non-finite value, or all zero (cosine is
    /// undefined) is an error; the pool and heaps are bounded by the vertex count whatever
    /// `ef` is.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        params: VSearchParams,
        out: &mut Vec<u32>,
    ) -> Result<usize, HmsError> {
        let g = self.index;
        if query.len() != g.dim {
            return Err(invalid(format!(
                "query has {} dimensions, the index has {}",
                query.len(),
                g.dim
            )));
        }
        if let Some(at) = query.iter().position(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "non-finite query value at coordinate {at}"
            )));
        }
        if query.iter().all(|&v| v == 0.0) {
            return Err(invalid("zero query: cosine similarity is undefined".into()));
        }
        let ef = params.ef.max(k).max(1);
        // Only the order matters, so the query is used as given (no centring needed: the
        // `<q, mean>` term is common to all candidates) and its scale is dropped.
        self.qf[..g.dim].copy_from_slice(query);
        g.encode_query(
            &self.qf,
            &mut self.qc,
            &mut self.qtmp,
            &mut self.qrot,
            &mut self.qs,
        );
        let qc = &self.qc[..];
        let screen = g.has_screen() && params.screen;
        let margin = params.screen_sigmas * g.screen_sigma;
        self.screened = 0;

        self.visited.next();
        let (entry, mut evals) = g.descend(qc);
        self.visited.insert(entry);
        self.cand.clear();
        self.best.clear();
        let k0 = key(g.est(qc, entry), entry);
        self.cand.push(Reverse(k0));
        self.best.push(k0);
        evals += 1;
        // Best-first expansion of the nearest unexpanded candidate among the `ef` best seen;
        // a candidate that fell out of those is farther than all of them, so it ends the search.
        // With `patience`, the `k` best keys are tracked and the search also ends after that
        // many expansions without a change to them.
        let patience = params.patience;
        self.topk.clear();
        let mut stale = 0usize;
        while let Some(Reverse(c)) = self.cand.pop() {
            if self.best.len() >= ef && self.best.peek().is_some_and(|&w| c > w) {
                break;
            }
            if patience > 0 && stale >= patience {
                break;
            }
            stale += 1;
            if let Some(&Reverse(next)) = self.cand.peek() {
                g.prefetch_neighbours(key_id(next) as usize);
            }
            self.fresh.clear();
            let (visited, fresh) = (&mut self.visited, &mut self.fresh);
            g.for_each_neighbour(key_id(c) as usize, |u| {
                if visited.insert(u) {
                    if screen {
                        g.prefetch_screen(u);
                    } else {
                        prefetch_bytes(g.code(u));
                    }
                    fresh.push(u);
                }
            });
            if screen {
                // A neighbour whose optimistic 1-bit estimate cannot enter the full pool is
                // skipped (it stays visited); the rest get the 8-bit estimate.
                let worst = if self.best.len() >= ef {
                    self.best.peek().map(|&w| key_dist(w))
                } else {
                    None
                };
                self.survivors.clear();
                for &u in &self.fresh {
                    if let Some(w) = worst {
                        if g.screen_est(&self.qs, u) - margin >= w {
                            self.screened += 1;
                            continue;
                        }
                    }
                    prefetch_bytes(g.code(u));
                    self.survivors.push(u);
                }
                std::mem::swap(&mut self.fresh, &mut self.survivors);
            }
            for &u in &self.fresh {
                let kk = key(g.est(qc, u), u);
                evals += 1;
                if self.best.len() < ef {
                    self.best.push(kk);
                } else {
                    let mut worst = self.best.peek_mut().expect("ef >= 1");
                    if kk >= *worst {
                        continue;
                    }
                    *worst = kk;
                }
                self.cand.push(Reverse(kk));
                if patience > 0 && (self.topk.len() < k || kk < self.topk[self.topk.len() - 1]) {
                    let at = self.topk.partition_point(|&t| t < kk);
                    self.topk.insert(at, kk);
                    self.topk.truncate(k);
                    stale = 0;
                }
            }
        }

        self.keys.clear();
        self.keys.extend(self.best.drain());
        self.keys.sort_unstable();
        self.results.clear();
        let take = params.rerank.max(k).min(self.keys.len());
        if params.rerank > 0 {
            // Ties (duplicate vectors) break by the original id, independent of the layout.
            self.results.extend(self.keys[..take].iter().map(|&c| {
                let id = key_id(c);
                (g.fine(&self.qf, id), g.original(id))
            }));
            evals += take;
            self.results
                .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        } else {
            self.results.extend(
                self.keys[..take]
                    .iter()
                    .map(|&c| (0.0, g.original(key_id(c)))),
            );
        }
        out.clear();
        out.extend(self.results.iter().take(k).map(|r| r.1));
        Ok(evals)
    }
}

/// Format 02 adds the build parameters and a fingerprint of the vectors to the header, so a
/// cache written for other data or parameters is rejected instead of silently reused.
const CACHE_MAGIC: &[u8; 8] = b"HMSGRF02";

/// Fingerprint of the normalized vectors a graph was built over.
fn data_fingerprint(unit: &[f32]) -> u64 {
    let mut h = super::Fnv::new();
    unit.iter().for_each(|x| h.word(x.to_bits()));
    h.finish()
}

fn put_u32s(w: &mut impl std::io::Write, xs: &[u32]) -> std::io::Result<()> {
    w.write_all(&(xs.len() as u64).to_le_bytes())?;
    for x in xs {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn get_u64(r: &mut impl std::io::Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn get_u32s(r: &mut impl std::io::Read, max: u64) -> std::io::Result<Vec<u32>> {
    let len = get_u64(r)?;
    if len > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "graph cache: list longer than the graph",
        ));
    }
    let mut bytes = vec![0u8; len as usize * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

fn put_lists(w: &mut impl std::io::Write, lists: &[Vec<u32>]) -> std::io::Result<()> {
    let lens: Vec<u32> = lists.iter().map(|l| l.len() as u32).collect();
    put_u32s(w, &lens)?;
    put_u32s(w, &lists.concat())
}

fn get_lists(r: &mut impl std::io::Read, n: u64) -> std::io::Result<Vec<Vec<u32>>> {
    let lens = get_u32s(r, n)?;
    let flat = get_u32s(r, lens.iter().map(|&l| u64::from(l)).sum())?;
    let mut out = Vec::with_capacity(lens.len());
    let mut at = 0;
    for l in lens {
        out.push(flat[at..at + l as usize].to_vec());
        at += l as usize;
    }
    Ok(out)
}

/// Benchmark support: the graph structure (not the vectors) on disk, so that search
/// experiments on one graph need not rebuild it.
impl Graph {
    /// Writes the header (n, dim, degree, build_ef, alpha, seed, entry, data fingerprint),
    /// the adjacency and the upper layers; the vectors are not stored.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        use std::io::Write;
        w.write_all(CACHE_MAGIC)?;
        let header = [
            self.n as u64,
            self.dim as u64,
            self.degree as u64,
            self.build.build_ef as u64,
            u64::from(self.build.alpha.to_bits()),
            self.build.seed,
            u64::from(self.entry),
            data_fingerprint(&self.unit),
        ];
        for x in header {
            w.write_all(&x.to_le_bytes())?;
        }
        put_lists(&mut w, &self.adj)?;
        put_u32s(&mut w, &self.upper_ids)?;
        w.write_all(&(self.layers.len() as u64).to_le_bytes())?;
        for l in &self.layers {
            put_lists(&mut w, l)?;
        }
        w.flush()
    }

    /// Reads a graph written by [`Graph::save`] for the same `data` (normalized here exactly
    /// as [`Graph::build`] does) and the same `params`. Any mismatch in the vector count,
    /// dimension, degree, build_ef, alpha, seed or the data fingerprint is an error, so a
    /// stale cache never stands in for a build; delete the file to rebuild.
    pub fn load(
        data: &[f32],
        dim: usize,
        path: &std::path::Path,
        params: &BuildParams,
    ) -> std::io::Result<Self> {
        let bad = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        std::io::Read::read_exact(&mut r, &mut magic)?;
        if &magic != CACHE_MAGIC {
            return Err(bad(
                "graph cache: bad magic (an older format or not a cache file); delete it to rebuild"
                    .into(),
            ));
        }
        let mut header = [0u64; 8];
        for x in &mut header {
            *x = get_u64(&mut r)?;
        }
        let [n, d, degree, build_ef, alpha_bits, seed, entry, fingerprint] = header;
        if dim == 0 || !data.len().is_multiple_of(dim) {
            return Err(bad("graph cache: ragged data".into()));
        }
        let mut unit = data.to_vec();
        unit.par_chunks_mut(dim).for_each(|x| {
            let norm = super::dot(x, x).sqrt();
            if norm > 0.0 {
                x.iter_mut().for_each(|o| *o /= norm);
            }
        });
        let checks: [(&str, u64, u64); 7] = [
            ("n", n, (data.len() / dim) as u64),
            ("dim", d, dim as u64),
            ("degree", degree, params.degree as u64),
            ("build_ef", build_ef, params.build_ef as u64),
            ("alpha", alpha_bits, u64::from(params.alpha.to_bits())),
            ("seed", seed, params.seed),
            ("data fingerprint", fingerprint, data_fingerprint(&unit)),
        ];
        if let Some((what, file, want)) = checks.iter().find(|(_, a, b)| a != b) {
            let show = |x: u64| {
                if *what == "alpha" {
                    f32::from_bits(x as u32).to_string()
                } else {
                    format!("{x:#x}")
                }
            };
            return Err(bad(format!(
                "graph cache: {what} {} in the file, {} requested; delete the file to rebuild",
                show(*file),
                show(*want)
            )));
        }
        let n = n as usize;
        if entry >= n as u64 {
            return Err(bad("graph cache: entry out of range".into()));
        }
        let adj = get_lists(&mut r, n as u64)?;
        let upper_ids = get_u32s(&mut r, n as u64)?;
        let nl = get_u64(&mut r)?;
        if nl > 64 {
            return Err(bad("graph cache: too many layers".into()));
        }
        let layers = (0..nl)
            .map(|_| get_lists(&mut r, n as u64))
            .collect::<std::io::Result<Vec<_>>>()?;
        let ok = adj.len() == n
            && adj.iter().flatten().all(|&u| (u as usize) < n)
            && upper_ids.iter().all(|&u| (u as usize) < n)
            && layers.iter().all(|l| {
                l.len() <= upper_ids.len() && l.iter().flatten().all(|&u| (u as usize) < l.len())
            });
        if !ok {
            return Err(bad("graph cache: id out of range".into()));
        }
        Ok(Self {
            n,
            dim,
            degree: degree as usize,
            unit,
            adj,
            entry: entry as u32,
            upper_ids,
            layers,
            build: params.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::kernels::SplitMix;
    use super::super::tests::{brute_force, clustered, gaussian};
    use super::*;

    /// The cache round-trips the structure for the same data and parameters, and rejects a
    /// file written for other data or for any other build parameter.
    #[test]
    fn graph_cache_round_trips_and_rejects_mismatches() {
        let d = 16;
        let data = clustered(600, d, 4);
        let params = BuildParams {
            build_ef: 32,
            degree: 8,
            alpha: 1.1,
            seed: 9,
            ..BuildParams::default()
        };
        let graph = Graph::build(&data, d, &params).expect("build");
        let path = std::env::temp_dir().join(format!("hms-qgraph-cache-{}", std::process::id()));
        graph.save(&path).unwrap();
        let back = Graph::load(&data, d, &path, &params).unwrap();
        assert_eq!(back.structure_hash(), graph.structure_hash());
        assert!(back.adj == graph.adj && back.layers == graph.layers);
        assert_eq!(
            (back.entry, &back.upper_ids),
            (graph.entry, &graph.upper_ids)
        );
        assert_eq!(back.unit, graph.unit);
        let rejects = |data: &[f32], p: &BuildParams, what: &str| {
            let err = Graph::load(data, d, &path, p)
                .err()
                .expect("a mismatched cache was accepted")
                .to_string();
            assert!(err.contains(what), "{what}: {err}");
        };
        for (what, p) in [
            (
                "degree",
                BuildParams {
                    degree: 10,
                    ..params.clone()
                },
            ),
            (
                "build_ef",
                BuildParams {
                    build_ef: 33,
                    ..params.clone()
                },
            ),
            (
                "alpha",
                BuildParams {
                    alpha: 1.0,
                    ..params.clone()
                },
            ),
            (
                "seed",
                BuildParams {
                    seed: 10,
                    ..params.clone()
                },
            ),
        ] {
            rejects(&data, &p, what);
        }
        let mut other = data.clone();
        other[d * 7 + 3] += 0.5;
        rejects(&other, &params, "data fingerprint");
        rejects(&data[..d * 599], &params, "n");
        std::fs::write(&path, b"HMSGRF01").unwrap();
        rejects(&data, &params, "bad magic");
        std::fs::remove_file(&path).unwrap();
    }

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
            s.search(q, 10, p, &mut out).expect("search");
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
        let graph = Graph::build(&data, d, &BuildParams::default()).expect("build");
        let plain = VGraph::from_graph(&graph, false).expect("encode");
        let fine = VGraph::from_graph(&graph, true).expect("encode");
        assert!(plain.index_bytes() < fine.index_bytes());
        let pool = VSearchParams {
            ef: 64,
            ..VSearchParams::default()
        };
        let rr = VSearchParams {
            ef: 64,
            rerank: 32,
            ..VSearchParams::default()
        };
        let r0 = recall(&plain, &data, &queries, d, pool);
        let r1 = recall(&fine, &data, &queries, d, rr);
        // Same regression guard as the edge-code index on this deliberately hard set.
        assert!(r0 >= 0.9, "recall {r0}");
        assert!(r1 >= r0, "rerank {r1} below pool order {r0}");
        let fine4 = VGraph::from_graph_with(&graph, 8, 4, false, false, 4, false).expect("encode");
        assert!(plain.index_bytes() < fine4.index_bytes());
        assert!(fine4.index_bytes() < fine.index_bytes());
        let r2 = recall(&fine4, &data, &queries, d, rr);
        assert!(r2 >= r0 - 0.02, "4-bit rerank {r2} below pool order {r0}");
        let lvq48 = VGraph::from_graph_with(&graph, 4, 8, true, false, 4, false).expect("encode");
        assert!(lvq48.index_bytes() < fine.index_bytes());
        // A sanity bound: 4-bit codes of 24 coordinates are coarse, so a wider pool and re-rank.
        let wide = VSearchParams {
            ef: 96,
            rerank: 64,
            ..VSearchParams::default()
        };
        let r3 = recall(&lvq48, &data, &queries, d, wide);
        assert!(
            r3 >= r0 - 0.05,
            "4-bit traversal with 8-bit residual {r3} vs {r0}"
        );
    }

    /// Renumbering and aligning the rows change the layout, not the search: the same ids come
    /// back (up to ties between equal estimates, which break by the stored id).
    #[test]
    fn reordered_aligned_index_returns_the_same_ids() {
        let d = 24;
        let data = clustered(3000, d, 5);
        let graph = Graph::build(&data, d, &BuildParams::default()).expect("build");
        let a = VGraph::from_graph(&graph, true).expect("encode");
        let b = VGraph::from_graph_with(&graph, 8, 8, true, true, 4, false).expect("encode");
        // 32-byte rows already sit inside one line; the id map is the only extra.
        assert_eq!(b.index_bytes(), a.index_bytes() + b.n * 4);
        let (mut sa, mut sb) = (a.searcher(), b.searcher());
        let (mut oa, mut ob) = (Vec::new(), Vec::new());
        let c = VGraph::from_graph_with(&graph, 8, 8, true, true, 3, false).expect("encode");
        assert_eq!(c.index_bytes(), b.index_bytes() - c.n * c.degree);
        let mut sc = c.searcher();
        let mut oc = Vec::new();
        let mut same = 0;
        for q in data.chunks_exact(d).take(200) {
            for p in [
                VSearchParams {
                    ef: 32,
                    ..VSearchParams::default()
                },
                VSearchParams {
                    ef: 48,
                    rerank: 16,
                    ..VSearchParams::default()
                },
            ] {
                sa.search(q, 10, p, &mut oa).expect("search");
                sb.search(q, 10, p, &mut ob).expect("search");
                sc.search(q, 10, p, &mut oc).expect("search");
                assert_eq!(ob, oc, "3-byte ids changed the search");
                same += usize::from(oa == ob);
            }
        }
        assert!(same >= 396, "{same} of 400 searches agree");
    }

    #[test]
    fn build_and_search_are_deterministic() {
        let d = 20;
        let data = clustered(3000, d, 3);
        let params = BuildParams {
            seed: 9,
            ..BuildParams::default()
        };
        let a = VGraph::from_graph(&Graph::build(&data, d, &params).expect("build"), true)
            .expect("encode");
        let b = VGraph::from_graph(&Graph::build(&data, d, &params).expect("build"), true)
            .expect("encode");
        assert!(a.codes.bytes() == b.codes.bytes() && a.residual == b.residual && a.adj == b.adj);
        assert!(a.scales == b.scales && a.res_scales == b.res_scales);
        assert_eq!((a.entry, &a.mean), (b.entry, &b.mean));
        let (mut sa, mut sb) = (a.searcher(), b.searcher());
        let (mut oa, mut ob) = (Vec::new(), Vec::new());
        let p = VSearchParams {
            ef: 32,
            rerank: 16,
            ..VSearchParams::default()
        };
        for q in data.chunks_exact(d).take(50) {
            sa.search(q, 10, p, &mut oa).expect("search");
            sb.search(q, 10, p, &mut ob).expect("search");
            assert_eq!(oa, ob);
        }
    }

    /// Every failure mode is an error that names its cause; no input panics.
    #[test]
    fn invalid_inputs_are_errors_that_name_the_cause() {
        let d = 16;
        let data = clustered(500, d, 4);
        let err = |r: Result<Graph, HmsError>| match r {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an invalid input was accepted"),
        };
        assert!(err(Graph::build(&[], d, &BuildParams::default())).contains("empty"));
        assert!(err(Graph::build(&data[..d + 1], d, &BuildParams::default())).contains("ragged"));
        assert!(err(Graph::build(&data, 0, &BuildParams::default())).contains("ragged"));
        let mut bad = data.clone();
        bad[3 * d + 2] = f32::NAN;
        let e = err(Graph::build(&bad, d, &BuildParams::default()));
        assert!(e.contains("non-finite") && e.contains("row 3"), "{e}");
        let odd = BuildParams {
            degree: 7,
            ..BuildParams::default()
        };
        assert!(err(Graph::build(&data, d, &odd)).contains("even"));
        let graph = Graph::build(&data, d, &BuildParams::default()).expect("build");
        let e = match VGraph::from_graph_with(&graph, 5, 8, false, false, 4, false) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("bits 5 was accepted"),
        };
        assert!(e.contains("bits"), "{e}");
        let index = VGraph::from_graph(&graph, true).expect("encode");
        let mut s = index.searcher();
        let mut out = Vec::new();
        let p = VSearchParams::default();
        let e = s
            .search(&data[..d - 1], 10, p, &mut out)
            .expect_err("short query")
            .to_string();
        assert!(e.contains("dimension"), "{e}");
        let mut q = data[..d].to_vec();
        q[5] = f32::INFINITY;
        let e = s.search(&q, 10, p, &mut out).expect_err("inf").to_string();
        assert!(e.contains("non-finite") && e.contains('5'), "{e}");
        let e = s
            .search(&vec![0.0; d], 10, p, &mut out)
            .expect_err("zero")
            .to_string();
        assert!(e.contains("zero"), "{e}");
        // Adversarial sizes stay bounded by the vertex count: a huge ef and k finish and
        // return at most n ids.
        let huge = VSearchParams {
            ef: usize::MAX,
            rerank: usize::MAX,
            ..VSearchParams::default()
        };
        let evals = s.search(&data[..d], 10, huge, &mut out).expect("search");
        assert_eq!(out.len(), 10);
        assert!(evals <= 2 * index.len() + 10);
        s.search(&data[..d], 5000, huge, &mut out).expect("search");
        assert!(out.len() <= index.len());
        // A zero data row is allowed and never returned ahead of real neighbours by a NaN.
        let mut with_zero = data.clone();
        with_zero[..d].fill(0.0);
        let g0 = Graph::build(&with_zero, d, &BuildParams::default()).expect("build");
        let i0 = VGraph::from_graph(&g0, true).expect("encode");
        let mut s0 = i0.searcher();
        s0.search(&data[d..2 * d], 10, p, &mut out).expect("search");
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|&v| (v as usize) < i0.len()));
    }

    /// The screen only decides which neighbours get the 8-bit estimate: with a wide margin
    /// it skips nothing and the ids are identical; at one sigma recall stays close.
    #[test]
    fn screen_is_transparent_at_a_wide_margin() {
        let d = 24;
        let data = clustered(3000, d, 5);
        let queries = clustered(100, d, 2);
        let graph = Graph::build(&data, d, &BuildParams::default()).expect("build");
        let plain = VGraph::from_graph(&graph, true).expect("encode");
        let screened =
            VGraph::from_graph_with(&graph, 8, 8, false, false, 4, true).expect("encode");
        assert!(screened.has_screen() && screened.index_bytes() > plain.index_bytes());
        let (mut sa, mut sb) = (plain.searcher(), screened.searcher());
        let (mut oa, mut ob) = (Vec::new(), Vec::new());
        let wide = VSearchParams {
            ef: 64,
            rerank: 16,
            screen_sigmas: 1e6,
            ..VSearchParams::default()
        };
        for q in queries.chunks_exact(d) {
            let ea = sa.search(q, 10, wide, &mut oa).expect("search");
            let eb = sb.search(q, 10, wide, &mut ob).expect("search");
            assert_eq!((ea, &oa), (eb, &ob));
            assert_eq!(sb.last_screened(), 0);
        }
        let one = VSearchParams {
            ef: 64,
            rerank: 16,
            ..VSearchParams::default()
        };
        let r0 = recall(&plain, &data, &queries, d, one);
        let r1 = recall(&screened, &data, &queries, d, one);
        assert!(r1 >= r0 - 0.02, "screened {r1} vs {r0}");
        let mut skipped = 0;
        for q in queries.chunks_exact(d) {
            sb.search(q, 10, one, &mut ob).expect("search");
            skipped += sb.last_screened();
        }
        assert!(skipped > 0, "the screen never skipped a neighbour");
    }
}
