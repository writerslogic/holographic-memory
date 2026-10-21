// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
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

use std::collections::BinaryHeap;
use std::sync::OnceLock;

use rayon::prelude::*;

use super::fastscan::{pack32, scan32, BLOCK};
use super::kernels::{
    dot_f32_i4, dot_f32_i8, dot_i4_inline, dot_i4_inline_x4, dot_i8_inline, interleave_for_i4,
    prefetch, prefetch_bytes, raw_one, QueryCode, Rotation, SplitMix,
};
use super::{invalid, prepare_rows, upper_bytes, BuildParams, Graph, Metric, NONE};
use crate::core::error::HmsError;

mod file;
pub use file::Residency;
use file::Slab;

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
    /// With neighbour blocks: fuse the block estimates a vertex receives from every expanded
    /// vertex that lists it (precision-weighted by the edge residual's scale) and re-admit
    /// it to the pool when the fused estimate qualifies, instead of keeping its first one.
    pub fuse: bool,
    /// Per-row search: estimate only the first this many neighbours of each expanded vertex
    /// (lists keep the build's order, nearest first); 0 = all.
    pub neighbour_cap: usize,
}

impl Default for VSearchParams {
    fn default() -> Self {
        Self {
            ef: 64,
            rerank: 0,
            patience: 0,
            screen_sigmas: 1.0,
            screen: true,
            fuse: false,
            neighbour_cap: 0,
        }
    }
}

/// Runtime knobs of the search loop, each off by default so that the results are those of the
/// plain search. [`VGraph::searcher`] reads them once per process from the environment
/// ([`SearchKnobs::from_env`]); [`VGraph::searcher_with`] takes them explicitly.
///
/// Mechanisms, and what they may change:
/// * `prefetch` (`HMS_QGRAPH_PREFETCH`, 0..=3, results never change): 0 prefetches the next
///   candidate's neighbour list and the code row of every admitted neighbour; 1 adds the
///   admitted neighbours' scales and the rows the re-rank reads; 2 adds the neighbour list of
///   the second-nearest unexpanded candidate; 3 adds a lookahead that, after an expansion,
///   prefetches the code rows (or screen rows) of the next candidate's unvisited neighbours.
/// * `batch_est` (`HMS_QGRAPH_BATCH_EST`, results never change): the admitted neighbours'
///   4-bit rows are estimated four at a time with shared query loads (`dot_i4_inline_x4` in
///   the kernels module); the integer sums are the same.
/// * `frontier` (`HMS_QGRAPH_FRONTIER`, `0.0..=1.0`, 1.0 = off): with a full pool, stop when
///   the nearest unexpanded estimate is past the point `frontier` of the way from the `k`-th
///   best to the `ef`-th best. Fewer expansions, so lower recall at a given `ef`; the
///   recall/QPS curve over `ef` is what to compare.
/// * `block_screen` (`HMS_QGRAPH_BLOCK_SCREEN`, needs [`VGraph::with_blocks`]): the neighbour
///   block screens instead of ordering; survivors fetch their code row and the pool holds row
///   estimates, as in the plain search. A fresh neighbour is skipped when its block estimate,
///   improved by `screen_sigmas` deviations of the calibrated edge error, could not enter the
///   full pool. Off, the block search is the existing one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchKnobs {
    /// Prefetch depth, 0 to 3 (values above 3 act as 3).
    pub prefetch: u8,
    /// Estimate admitted neighbours four rows at a time.
    pub batch_est: bool,
    /// Frontier stop fraction in `0.0..=1.0`; 1.0 (or a non-finite value) is off.
    pub frontier: f32,
    /// Screen with the neighbour blocks and estimate survivors from their rows.
    pub block_screen: bool,
}

impl SearchKnobs {
    /// Every knob off: the plain search.
    pub const OFF: Self = Self {
        prefetch: 0,
        batch_est: false,
        frontier: 1.0,
        block_screen: false,
    };

    /// The knobs set in the environment, read once per process; unset or invalid values
    /// (reported once) take the [`SearchKnobs::OFF`] value.
    pub fn from_env() -> Self {
        static KNOBS: OnceLock<SearchKnobs> = OnceLock::new();
        *KNOBS.get_or_init(|| Self {
            prefetch: env_knob::<u8>("HMS_QGRAPH_PREFETCH").unwrap_or(0).min(3),
            batch_est: env_knob::<u8>("HMS_QGRAPH_BATCH_EST").unwrap_or(0) != 0,
            frontier: env_knob::<f32>("HMS_QGRAPH_FRONTIER")
                .filter(|f| f.is_finite())
                .map_or(1.0, |f| f.clamp(0.0, 1.0)),
            block_screen: env_knob::<u8>("HMS_QGRAPH_BLOCK_SCREEN").unwrap_or(0) != 0,
        })
    }
}

impl Default for SearchKnobs {
    fn default() -> Self {
        Self::OFF
    }
}

/// `name` parsed as `T`: `None` when unset, and when the value does not parse, which is
/// reported.
fn env_knob<T: std::str::FromStr>(name: &str) -> Option<T> {
    let value = std::env::var(name).ok()?;
    match value.trim().parse::<T>() {
        Ok(v) => Some(v),
        Err(_) => {
            eprintln!("{name}={value:?} is not a valid value; using the default");
            None
        }
    }
}

/// `HMS_QGRAPH_HUGEPAGES` set to a non-zero value: the large arrays of an index built in this
/// process (code rows, neighbour lists, scales, residual rows, neighbour blocks) are advised
/// into transparent huge pages before they are first touched, so each random row fetch costs
/// one TLB entry per 2 MiB instead of per 4 KiB. Linux only; a no-op elsewhere. Read once.
fn huge_pages() -> bool {
    static HUGE: OnceLock<bool> = OnceLock::new();
    *HUGE.get_or_init(|| env_knob::<u8>("HMS_QGRAPH_HUGEPAGES").unwrap_or(0) != 0)
}

/// Asks the kernel to back the 2 MiB-aligned interior of `[ptr, ptr + len)` with transparent
/// huge pages (Linux `MADV_HUGEPAGE`). Must run before the pages are first touched, so the
/// page-fault path allocates them huge where THP is `always` or `madvise`; afterwards only
/// `khugepaged` could collapse them, slowly. A no-op outside Linux. Errors are ignored: the
/// advice changes paging policy only, never the mapping or its contents.
fn advise_huge_pages(ptr: *const u8, len: usize) {
    #[cfg(target_os = "linux")]
    {
        const HUGE: usize = 2 << 20;
        const MADV_HUGEPAGE: i32 = 14;
        extern "C" {
            fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
        }
        let start = (ptr as usize).next_multiple_of(HUGE);
        let end = (ptr as usize + len) & !(HUGE - 1);
        if start < end {
            // SAFETY: `[start, end)` lies inside the live allocation `[ptr, ptr + len)` and
            // `madvise` with `MADV_HUGEPAGE` reads or writes nothing in it.
            unsafe {
                madvise(start as *mut std::ffi::c_void, end - start, MADV_HUGEPAGE);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (ptr, len);
}

/// [`advise_huge_pages`] over the capacity of a freshly allocated, still untouched vector when
/// [`huge_pages`] is on.
fn huge_capacity<T>(ptr: *const T, capacity: usize) {
    if huge_pages() {
        advise_huge_pages(ptr.cast::<u8>(), capacity * std::mem::size_of::<T>());
    }
}

/// One 128-byte cache line (the line size of Apple cores).
#[repr(C, align(128))]
#[derive(Clone, Copy)]
struct Line([i8; 128]);

/// Bytes starting on a cache-line boundary (zeroed when built here).
struct Lines {
    /// `len.div_ceil(128)` lines.
    lines: Slab<Line>,
    len: usize,
}

impl Lines {
    fn zeroed(len: usize) -> Self {
        let count = len.div_ceil(128);
        let mut lines = Vec::with_capacity(count);
        huge_capacity(lines.as_ptr(), count);
        lines.resize(count, Line([0; 128]));
        Self {
            lines: lines.into(),
            len,
        }
    }

    #[inline(always)]
    fn bytes(&self) -> &[i8] {
        // SAFETY: `Line` is `repr(C)` around `[i8; 128]` (size 128, no padding), so `lines` is
        // `128 * lines.len() >= len` initialized, contiguous bytes.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr().cast::<i8>(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [i8] {
        let len = self.len;
        let lines = self.lines.as_mut_slice();
        // SAFETY: as in `bytes`, with the exclusive borrow of `lines`.
        unsafe { std::slice::from_raw_parts_mut(lines.as_mut_ptr().cast::<i8>(), len) }
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
    /// Euclidean: `|x|^2 / 2` per stored vertex, the norm term of the distance order; empty
    /// for angular.
    half_norms: Slab<f32>,
    /// Neighbour ids, `degree` per vertex padded with `NONE`; empty with 3-byte ids.
    adj: Slab<u32>,
    /// The same as 3-byte little-endian ids (padded with `NONE3`); empty with `u32` ids.
    adj3: Slab<u8>,
    /// Delta-coded lists (`id_bytes` 0): per vertex, the sorted neighbour ids as LEB128
    /// varints of the zigzag first difference to the vertex and the gaps less one after,
    /// with `adjv_off[v]..adjv_off[v + 1]` the byte span of vertex `v`; empty otherwise.
    adjv: Slab<u8>,
    adjv_off: Slab<u32>,
    codes: Lines,
    scales: Slab<f32>,
    /// 0 (none), 4 or 8.
    res_bits: u8,
    /// Bytes per residual row: `padded` (8-bit) or `padded / 2` (4-bit, two per byte).
    res_stride: usize,
    residual: Slab<i8>,
    res_scales: Slab<f32>,
    entry: u32,
    upper_ids: Vec<u32>,
    layers: Vec<Vec<Vec<u32>>>,
    /// Original id of each stored vertex when the vertices were renumbered; empty otherwise.
    ids: Slab<u32>,
    /// 1-bit screen rows: `screen_words` u64 of sign bits of the rotated centred vector, then
    /// one u64 holding the f32 factor (low half) and the popcount (high half); empty without
    /// a screen.
    screen: Slab<u64>,
    screen_words: usize,
    screen_rot: Option<Rotation>,
    /// Standard deviation of (screen estimate - 8-bit estimate), in estimate units, measured
    /// at build on index vectors used as queries.
    screen_sigma: f32,
    /// Bits per coordinate of the neighbour-block codes (0 = no blocks; 1, 2 or 4). See
    /// [`VGraph::with_blocks`].
    block_bits: u8,
    /// Coordinates of a block code: the block rotation's width.
    block_dim: usize,
    /// Nibble groups per block code: `block_dim * block_bits / 4`.
    block_m: usize,
    /// Blocks per vertex: `degree.div_ceil(32)`.
    blocks_per_vertex: usize,
    /// Per vertex, `blocks_per_vertex` records of `16 * block_m` code bytes in the fast-scan
    /// layout followed by the 32 neighbours' scales as little-endian f32.
    blocks: Lines,
    block_rot: Option<Rotation>,
    /// Error model of the block estimates: `var = block_kappa * s^2 * |q - x_v|^2` for an
    /// edge of scale `s` seen from the expanded vertex `v`, calibrated at build.
    block_kappa: f32,
}

/// Seed of the screen's rotation (fixed: the screen is a property of the encoding).
const SCREEN_SEED: u64 = 0x5C12_EE00;
/// Seed of the neighbour-block rotation.
const BLOCK_SEED: u64 = 0xB10C_5CA0;

/// Bytes of one block record: codes, then 32 scales, then 32 offsets (little-endian f32).
fn block_record_bytes(m: usize) -> usize {
    16 * m + 8 * BLOCK
}

/// Level of coordinate `i` of a block code (see [`encode_block_code`]).
#[inline]
fn block_level(nibbles: &[u8], bits: u8, i: usize) -> f32 {
    match bits {
        1 => {
            if nibbles[i / 4] >> (i % 4) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        }
        2 => 2.0 * f32::from(nibbles[i / 2] >> (2 * (i % 2)) & 3) - 3.0,
        _ => f32::from(nibbles[i]) - 7.0,
    }
}

/// Little-endian f32 number `i` of `bytes`.
#[inline]
fn le_f32(bytes: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([
        bytes[4 * i],
        bytes[4 * i + 1],
        bytes[4 * i + 2],
        bytes[4 * i + 3],
    ])
}

/// Encodes a rotated, centred vector as `m` nibbles for a `bits`-per-coordinate block code
/// and returns its scale, so that coordinate `i ~= scale * level(i)` with the levels
/// `{-1, 1}` (1 bit), `{-3, -1, 1, 3}` (2 bits) or `-7..=7` (4 bits; nibble `level + 7`).
/// The levels of the 2- and 4-bit codes are chosen with the scale among eight candidates
/// that minimizes the squared reconstruction error. The returned scale, for every width, is
/// `|x|^2 / <levels, x>`: it makes the code error orthogonal to `x` itself (for one bit this
/// is the RaBitQ factor `|x|^2 / sum |x_i|`), so inner products with queries that are
/// correlated with `x` are unbiased, where the least-squares scale underestimates them by
/// the relative error energy.
fn encode_block_code(x: &[f32], bits: u8, nibbles: &mut [u8]) -> f32 {
    nibbles.fill(0);
    let m = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    if m == 0.0 {
        return 0.0;
    }
    match bits {
        1 => {
            let (mut norm2, mut abs) = (0f32, 0f32);
            for (i, &v) in x.iter().enumerate() {
                norm2 += v * v;
                abs += v.abs();
                if v >= 0.0 {
                    nibbles[i / 4] |= 1 << (i % 4);
                }
            }
            norm2 / abs
        }
        2 => {
            // Levels {-3, -1, 1, 3} * s; index t = (level + 3) / 2.
            let err = |s: f32| -> f32 {
                x.iter()
                    .map(|&v| {
                        let t = ((v / s + 3.0) / 2.0).round().clamp(0.0, 3.0);
                        let d = v - s * (2.0 * t - 3.0);
                        d * d
                    })
                    .sum()
            };
            let (mut best, mut best_err) = (m / 3.0, f32::INFINITY);
            for k in 0..8 {
                let s = m / 3.0 * (1.0 - 0.1 * k as f32);
                let e = err(s);
                if e < best_err {
                    (best, best_err) = (s, e);
                }
            }
            let mut dot = 0f32;
            for (i, &v) in x.iter().enumerate() {
                let t = ((v / best + 3.0) / 2.0).round().clamp(0.0, 3.0) as u8;
                nibbles[i / 2] |= t << (2 * (i % 2));
                dot += v * (2.0 * f32::from(t) - 3.0);
            }
            unbiased_scale(x, dot, best)
        }
        _ => {
            let err = |s: f32| -> f32 {
                x.iter()
                    .map(|&v| {
                        let d = v - s * (v / s).round().clamp(-7.0, 7.0);
                        d * d
                    })
                    .sum()
            };
            let (mut best, mut best_err) = (m / 7.0, f32::INFINITY);
            for k in 0..8 {
                let s = m / 7.0 * (1.0 - 0.06 * k as f32);
                let e = err(s);
                if e < best_err {
                    (best, best_err) = (s, e);
                }
            }
            let mut dot = 0f32;
            for (i, &v) in x.iter().enumerate() {
                let level = (v / best).round().clamp(-7.0, 7.0);
                nibbles[i] = (level as i8 + 7) as u8;
                dot += v * level;
            }
            unbiased_scale(x, dot, best)
        }
    }
}

/// `|x|^2 / <levels, x>` (see [`encode_block_code`]), or the least-squares scale when the
/// levels do not correlate with `x`.
fn unbiased_scale(x: &[f32], dot: f32, fallback: f32) -> f32 {
    let norm2: f32 = x.iter().map(|v| v * v).sum();
    if dot > 0.0 {
        norm2 / dot
    } else {
        fallback
    }
}

/// Fills the float lookup table `t` (`16 * m` entries) of a rotated query for `bits`-per-
/// coordinate block codes: `t[16 * j + v] = sum over the coordinates of nibble j of
/// q_i * level_i(v)`.
fn block_table(q: &[f32], bits: u8, m: usize, t: &mut [f32]) {
    match bits {
        1 => {
            for j in 0..m {
                let qs = &q[4 * j..4 * j + 4];
                let total: f32 = qs.iter().sum();
                for v in 0..16usize {
                    let on: f32 = (0..4).filter(|b| v >> b & 1 == 1).map(|b| qs[b]).sum();
                    t[16 * j + v] = 2.0 * on - total;
                }
            }
        }
        2 => {
            for j in 0..m {
                let (qa, qb) = (q[2 * j], q[2 * j + 1]);
                for v in 0..16usize {
                    let (ta, tb) = ((v & 3) as f32, (v >> 2) as f32);
                    t[16 * j + v] = qa * (2.0 * ta - 3.0) + qb * (2.0 * tb - 3.0);
                }
            }
        }
        _ => {
            for j in 0..m {
                for v in 0..16usize {
                    t[16 * j + v] = q[j] * (v as f32 - 7.0);
                }
            }
        }
    }
}

/// Per-query block lookup table: `lut` as the scan reads it, with the affine map
/// `entry ~= delta * lut + min_j` so that a code's inner product is
/// `scale * (delta * sum + min_sum)`.
pub struct BlockQuery {
    table: Vec<f32>,
    lut: Vec<u8>,
    delta: f32,
    min_sum: f32,
    /// `1 / s_q`, which puts the block estimate in [`VGraph::est`] units.
    inv_sq: f32,
}

impl BlockQuery {
    fn new(m: usize) -> Self {
        Self {
            table: vec![0.0; 16 * m],
            lut: vec![0; 16 * m],
            delta: 1.0,
            min_sum: 0.0,
            inv_sq: 1.0,
        }
    }

    /// Quantizes the float table to bytes: per group, entries are offset by the group's
    /// minimum; one step `delta` serves every group.
    fn quantize(&mut self, m: usize) {
        let mut range = 0f32;
        let mut min_sum = 0f32;
        for g in self.table.as_chunks::<16>().0 {
            let lo = g.iter().fold(f32::INFINITY, |a, &v| a.min(v));
            let hi = g.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            min_sum += lo;
            range = range.max(hi - lo);
        }
        let delta = if range > 0.0 { range / 255.0 } else { 1.0 };
        for (j, g) in self.table.as_chunks::<16>().0.iter().enumerate() {
            let lo = g.iter().fold(f32::INFINITY, |a, &v| a.min(v));
            for (v, &e) in g.iter().enumerate() {
                self.lut[16 * j + v] = ((e - lo) / delta).round().clamp(0.0, 255.0) as u8;
            }
        }
        debug_assert_eq!(self.lut.len(), 16 * m);
        self.delta = delta;
        self.min_sum = min_sum;
    }
}

/// Inverse of [`key`] for the distance part.
#[inline]
fn key_dist(k: u64) -> f32 {
    let o = (k >> 32) as u32;
    f32::from_bits(if o >> 31 == 1 { o & 0x7FFF_FFFF } else { !o })
}

/// Writes the symmetric 8-bit code of `x` into `code` (zero padded) and returns its scale, so
/// that `x_i ~= scale * code_i`.
pub(super) fn quantize(x: &[f32], code: &mut [i8]) -> f32 {
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
pub(super) fn quantize4(x: &[f32], code: &mut [i8]) -> f32 {
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
pub(super) fn code_at(c: &[i8], bits: u8, i: usize) -> i8 {
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
/// LEB128 varint (7 bits per byte, low bits first).
fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Decodes one LEB128 varint; returns the value and the bytes used.
#[inline]
fn get_varint(bytes: &[u8]) -> (u64, usize) {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= u64::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return (v, i + 1);
        }
    }
    (v, bytes.len())
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

fn prefetch_u8(bytes: &[u8]) {
    // SAFETY: any initialized `u8` slice is also initialized `i8`s.
    prefetch_bytes(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) });
}

#[inline]
fn prefetch_f32(x: &f32) {
    // SAFETY: an initialized `f32` is four initialized bytes.
    prefetch_bytes(unsafe { std::slice::from_raw_parts((x as *const f32).cast::<i8>(), 4) });
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
    /// three bytes (graphs below 2^24 - 1 vertices) instead of 4, and 0 stores each list
    /// sorted and delta-coded as varints (sizes vary; `reorder` makes the deltas small); `screen` adds a 1-bit
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
        if !(id_bytes == 4 || id_bytes == 0 || (id_bytes == 3 && (g.n as u64) < u64::from(NONE3))) {
            return Err(invalid(format!(
                "id_bytes must be 4, 0 (delta varints), or 3 below 2^24 - 1 vertices (got {id_bytes} for {} vertices)",
                g.n
            )));
        }
        if !matches!(residual_bits, 0 | 4 | 8) {
            return Err(invalid(format!(
                "residual code bits must be 0, 4 or 8, not {residual_bits}"
            )));
        }
        let euclidean = g.metric == Metric::Euclidean;
        if euclidean && screen {
            return Err(invalid("the 1-bit screen is angular only".into()));
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
        for x in g.vectors.chunks_exact(dim) {
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
        let half_norms: Vec<f32> = if euclidean {
            (0..n).map(|v| 0.5 * g.norms[old(v)]).collect()
        } else {
            Vec::new()
        };

        let screen_rot = screen.then(|| Rotation::new(dim, 0, SCREEN_SEED));
        let screen_words = screen_rot.as_ref().map_or(0, |r| r.padded() / 64);
        // The centred row of stored vertex `v`, which every code below encodes.
        let centred = |v: usize| -> Vec<f32> {
            let o = old(v);
            g.vectors[o * dim..(o + 1) * dim]
                .iter()
                .zip(&mean)
                .map(|(a, m)| a - m)
                .collect()
        };
        // Codes are written in place, so no per-vertex buffers sit next to the final arrays.
        let mut codes = Lines::zeroed(n * stride);
        let mut scales = Vec::with_capacity(n);
        huge_capacity(scales.as_ptr(), n);
        scales.resize(n, 0f32);
        codes
            .bytes_mut()
            .par_chunks_exact_mut(stride)
            .zip(scales.par_iter_mut())
            .enumerate()
            .for_each(|(v, (dst, s))| {
                let x = centred(v);
                *s = if bits == 8 {
                    quantize(&x, &mut dst[..cbytes])
                } else {
                    quantize4(&x, &mut dst[..cbytes])
                };
            });
        let mut residual = Vec::with_capacity(n * res_stride);
        huge_capacity(residual.as_ptr(), n * res_stride);
        residual.resize(n * res_stride, 0i8);
        let mut res_scales = vec![0f32; if residual_bits > 0 { n } else { 0 }];
        if residual_bits > 0 {
            let code_bytes = codes.bytes();
            residual
                .par_chunks_exact_mut(res_stride)
                .zip(res_scales.par_iter_mut())
                .enumerate()
                .for_each(|(v, (r, sr))| {
                    let mut x = centred(v);
                    let c = &code_bytes[v * stride..v * stride + cbytes];
                    let s = scales[v];
                    x.iter_mut()
                        .enumerate()
                        .for_each(|(i, a)| *a -= s * f32::from(code_at(c, bits, i)));
                    *sr = if residual_bits == 8 {
                        quantize(&x, r)
                    } else {
                        quantize4(&x, r)
                    };
                });
        }
        let mut screen_rows = vec![0u64; if screen { n * (screen_words + 1) } else { 0 }];
        if let Some(rot) = &screen_rot {
            screen_rows
                .par_chunks_exact_mut(screen_words + 1)
                .enumerate()
                .for_each(|(v, srow)| {
                    let x = centred(v);
                    let mut xr = vec![0f32; rot.padded()];
                    rot.apply(&x, &mut xr);
                    let (mut norm2, mut abs) = (0f32, 0f32);
                    let mut pop = 0u32;
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
                });
        }

        let r = g.degree;
        // Stored vertex `v`'s list: the build's list of the vertex, renumbered, at most `r`.
        let list = |v: usize| g.adj[old(v)].iter().take(r).map(|&u| new(u));
        let (mut adj, mut adj3) = (Vec::new(), Vec::new());
        let (mut adjv, mut adjv_off) = (Vec::new(), Vec::new());
        if id_bytes == 3 {
            adj3 = Vec::<u8>::with_capacity(3 * n * r);
            huge_capacity(adj3.as_ptr(), adj3.capacity());
            for v in 0..n {
                let mut count = 0;
                for u in list(v) {
                    adj3.extend_from_slice(&u.min(NONE3).to_le_bytes()[..3]);
                    count += 1;
                }
                for _ in count..r {
                    adj3.extend_from_slice(&NONE3.to_le_bytes()[..3]);
                }
            }
        } else if id_bytes == 4 {
            adj = vec![NONE; n * r];
            for (v, dst) in adj.chunks_exact_mut(r).enumerate() {
                for (d, u) in dst.iter_mut().zip(list(v)) {
                    *d = u;
                }
            }
        } else {
            adjv_off.reserve(n + 1);
            let mut sorted: Vec<u32> = Vec::with_capacity(r);
            for v in 0..n {
                adjv_off.push(adjv.len() as u32);
                sorted.clear();
                sorted.extend(list(v).take_while(|&u| u != NONE));
                sorted.sort_unstable();
                let list = &sorted;
                let mut prev = v as u32;
                for (i, &u) in list.iter().enumerate() {
                    let delta = if i == 0 {
                        zigzag(i64::from(u) - i64::from(prev))
                    } else {
                        u64::from(u - prev - 1)
                    };
                    put_varint(&mut adjv, delta);
                    prev = u;
                }
            }
            adjv_off.push(adjv.len() as u32);
        }
        let entry = new(g.entry);
        let upper_ids = g.upper_ids.iter().map(|&u| new(u)).collect();
        let mut out = Self {
            n,
            dim,
            padded,
            bits,
            cbytes,
            stride,
            degree: r,
            mean,
            half_norms: half_norms.into(),
            adj: adj.into(),
            adj3: adj3.into(),
            adjv: adjv.into(),
            adjv_off: adjv_off.into(),
            codes,
            scales: scales.into(),
            res_bits: residual_bits,
            res_stride,
            residual: residual.into(),
            res_scales: res_scales.into(),
            entry,
            upper_ids,
            layers: g.layers.clone(),
            ids: ids.into(),
            screen: screen_rows.into(),
            screen_words,
            screen_rot,
            screen_sigma: 0.0,
            block_bits: 0,
            block_dim: 0,
            block_m: 0,
            blocks_per_vertex: 0,
            blocks: Lines::zeroed(0),
            block_rot: None,
            block_kappa: 0.0,
        };
        if screen {
            out.screen_sigma = out.measure_screen_sigma(g);
        }
        Ok(out)
    }

    /// Adds neighbour-block codes: every vertex stores, next to its neighbour list, a
    /// `bits`-per-coordinate code of each neighbour's rotated centred vector in the fast-scan
    /// layout (32 codes per block), so one streamed read and one table scan estimate all
    /// neighbours of an expanded vertex instead of one random row fetch per neighbour
    /// (SymphonyQG's layout; Gou et al., SIGMOD 2025). `bits` is 1, 2 or 4.
    pub fn with_blocks(mut self, g: &Graph, bits: u8) -> Result<Self, HmsError> {
        if !matches!(bits, 1 | 2 | 4) {
            return Err(invalid(format!("block bits must be 1, 2 or 4, not {bits}")));
        }
        if !self.half_norms.is_empty() {
            return Err(invalid("neighbour blocks are angular only".into()));
        }
        let (n, dim, r) = (self.n, self.dim, self.degree);
        let rot = Rotation::new(dim, 0, BLOCK_SEED);
        let bd = rot.padded();
        let m = bd * usize::from(bits) / 4;
        let per_vertex = r.div_ceil(BLOCK);
        let record = block_record_bytes(m);
        let ids = &self.ids;
        let old = |v: usize| if ids.is_empty() { v } else { ids[v] as usize };
        let row = |v: usize| &g.vectors[old(v) * dim..(old(v) + 1) * dim];
        let mut blocks = Lines::zeroed(n * per_vertex * record);
        blocks
            .bytes_mut()
            .par_chunks_exact_mut(per_vertex * record)
            .enumerate()
            .for_each(|(v, dst)| {
                // Codes are of the residual `r = x_u - x_v` of each neighbour `u` to the
                // vertex `v` (as in SymphonyQG), so the code error scales with `|r|`; `e`
                // folds in the query-independent terms `<x_v, r> - s * <levels, rot(x_v)>`,
                // which makes the estimate's error `<rot(q - x_v), code error>`, scaled by
                // the query's distance to `v` as well:
                // `<q, x_u - mean> = <q, x_v - mean> + s * <levels, rot(q)> + e`.
                let xv = row(v);
                let mut xr = vec![0f32; bd];
                rot.apply(xv, &mut xr);
                let mut diff = vec![0f32; dim];
                let mut rr = vec![0f32; bd];
                let mut nibbles = vec![0u8; BLOCK * m];
                let mut scales = [0f32; BLOCK];
                let mut offsets = [0f32; BLOCK];
                let mut list: Vec<u32> = Vec::with_capacity(r);
                self.for_each_neighbour(v, |u| list.push(u));
                for (b, rec) in dst.chunks_exact_mut(record).enumerate() {
                    nibbles.fill(0);
                    scales.fill(0.0);
                    offsets.fill(0.0);
                    for i in 0..BLOCK {
                        let Some(&u) = list.get(b * BLOCK + i) else {
                            break;
                        };
                        let xu = row(u as usize);
                        for (d, (a, b)) in diff.iter_mut().zip(xu.iter().zip(xv)) {
                            *d = a - b;
                        }
                        rot.apply(&diff, &mut rr);
                        let nib = &mut nibbles[i * m..(i + 1) * m];
                        let s = encode_block_code(&rr, bits, nib);
                        let dot_lv: f32 = (0..bd).map(|j| block_level(nib, bits, j) * xr[j]).sum();
                        let dot_vr: f32 = xv.iter().zip(&diff).map(|(a, b)| a * b).sum();
                        scales[i] = s;
                        offsets[i] = dot_vr - s * dot_lv;
                    }
                    // SAFETY-free reinterpretation: the record is plain bytes.
                    let rec_u8 = unsafe {
                        std::slice::from_raw_parts_mut(rec.as_mut_ptr().cast::<u8>(), rec.len())
                    };
                    pack32(&nibbles, m, &mut rec_u8[..16 * m]);
                    for (i, (s, e)) in scales.iter().zip(&offsets).enumerate() {
                        let at = 16 * m + 4 * i;
                        rec_u8[at..at + 4].copy_from_slice(&s.to_le_bytes());
                        let at = 16 * m + 4 * BLOCK + 4 * i;
                        rec_u8[at..at + 4].copy_from_slice(&e.to_le_bytes());
                    }
                }
            });
        self.block_bits = bits;
        self.block_dim = bd;
        self.block_m = m;
        self.blocks_per_vertex = per_vertex;
        self.blocks = blocks;
        self.block_rot = Some(rot);
        self.block_kappa = self.measure_block_kappa(g);
        Ok(self)
    }

    /// Unit row `v` (stored numbering) of the build data.
    fn unit_row<'g>(&self, g: &'g Graph, v: usize) -> &'g [f32] {
        let o = if self.ids.is_empty() {
            v
        } else {
            self.ids[v] as usize
        };
        &g.vectors[o * self.dim..(o + 1) * self.dim]
    }

    /// Fits `block_kappa` by least squares on a deterministic sample: the squared error of
    /// the block estimate against the exact centred inner product, regressed on
    /// `s^2 * |q - x_v|^2` (index vectors as queries, random expanded vertices).
    fn measure_block_kappa(&self, g: &Graph) -> f32 {
        let (nq, nv) = (32.min(self.n), 64.min(self.n));
        let mut rng = SplitMix(BLOCK_SEED ^ 1);
        let mut s = self.searcher();
        let (mut num, mut den) = (0f64, 0f64);
        let (mut diag_sum2, mut diag_n) = (0f64, 0f64);
        let mut scan = [0u32; BLOCK];
        for _ in 0..nq {
            let qi = (rng.next_u64() % self.n as u64) as usize;
            let q = self.unit_row(g, qi).to_vec();
            s.qf[..self.dim].copy_from_slice(&q);
            let sq = self.encode_query(&s.qf, &mut s.qc, &mut s.qtmp, &mut s.qrot, &mut s.qs);
            if sq == 0.0 {
                continue;
            }
            self.encode_block_query(&s.qf, sq, &mut s.bqrot, &mut s.bq);
            let centred = |x: &[f32]| -> f32 {
                q.iter()
                    .zip(x)
                    .zip(&self.mean)
                    .map(|((a, b), m)| a * (b - m))
                    .sum()
            };
            let mut list = Vec::with_capacity(self.degree);
            for _ in 0..nv {
                let v = (rng.next_u64() % self.n as u64) as usize;
                let xv = self.unit_row(g, v);
                let ip_v = centred(xv);
                let d2: f32 = q.iter().zip(xv).map(|(a, b)| (a - b) * (a - b)).sum();
                list.clear();
                self.for_each_neighbour(v, |u| list.push(u));
                for b in 0..self.blocks_per_vertex {
                    let (codes, scales, offsets) = self.block(v, b);
                    scan32(codes, &s.bq.lut, self.block_m, &mut scan);
                    for (i, &sum) in scan.iter().enumerate() {
                        let Some(&u) = list.get(b * BLOCK + i) else {
                            break;
                        };
                        let sc = le_f32(scales, i);
                        let est = ip_v
                            + sc * (s.bq.delta * sum as f32 + s.bq.min_sum)
                            + le_f32(offsets, i);
                        let exact = centred(self.unit_row(g, u as usize));
                        let err = f64::from(est - exact);
                        num += err * err;
                        den += f64::from(sc * sc * d2);
                        diag_sum2 += f64::from(exact - ip_v).powi(2);
                        diag_n += 1.0;
                    }
                }
            }
        }
        if std::env::var_os("HMS_BLOCK_DIAG").is_some() && diag_n > 0.0 {
            eprintln!(
                "block diag: bits {} kappa {:.4} rms_err {:.5} rms_residual_ip {:.5} pairs {}",
                self.block_bits,
                num / den.max(1e-30),
                (num / diag_n).sqrt(),
                (diag_sum2 / diag_n).sqrt(),
                diag_n
            );
        }
        if den > 0.0 {
            (num / den) as f32
        } else {
            0.0
        }
    }

    /// True if the index carries neighbour-block codes.
    pub fn has_blocks(&self) -> bool {
        self.block_bits > 0
    }

    /// Bytes of the neighbour-block codes and scales.
    pub fn block_bytes(&self) -> usize {
        self.blocks.len
    }

    /// Block `b` of vertex `v`: its packed codes, its 32 scales and its 32 offsets (both as
    /// little-endian f32 bytes).
    #[inline]
    fn block(&self, v: usize, b: usize) -> (&[u8], &[u8], &[u8]) {
        let record = block_record_bytes(self.block_m);
        let at = (v * self.blocks_per_vertex + b) * record;
        let bytes = self.blocks.bytes();
        // SAFETY: `i8` and `u8` have the same size and alignment; the slice stays in bounds.
        let rec =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr().add(at).cast::<u8>(), record) };
        let (codes, rest) = rec.split_at(16 * self.block_m);
        let (scales, offsets) = rest.split_at(4 * BLOCK);
        (codes, scales, offsets)
    }

    #[inline]
    fn prefetch_blocks(&self, v: usize) {
        let record = block_record_bytes(self.block_m) * self.blocks_per_vertex;
        let at = v * record;
        prefetch_bytes(&self.blocks.bytes()[at..at + record]);
    }

    /// Builds the block lookup table of a query (`qf` padded, `sq` its 8-bit code scale).
    fn encode_block_query(&self, qf: &[f32], sq: f32, qrot: &mut [f32], bq: &mut BlockQuery) {
        let rot = self.block_rot.as_ref().expect("blocks present");
        rot.apply(&qf[..self.dim], qrot);
        block_table(qrot, self.block_bits, self.block_m, &mut bq.table);
        bq.quantize(self.block_m);
        bq.inv_sq = if sq > 0.0 { 1.0 / sq } else { 1.0 };
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
            qf[..self.dim].copy_from_slice(&g.vectors[q * self.dim..(q + 1) * self.dim]);
            let sq = self.encode_query(&qf, &mut qc, &mut qtmp, &mut qrot, &mut qs);
            if sq == 0.0 {
                continue;
            }
            for _ in 0..nv {
                let v = (rng.next_u64() % self.n as u64) as u32;
                let d = f64::from(self.screen_est(&qs, v) - self.est(&qc, v, 1.0));
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
            + self.adjv.len()
            + self.adjv_off.len() * 4
            + self.n * self.stride
            + self.scales.len() * 4
            + self.residual_bytes()
            + self.mean.len() * 4
            + self.half_norms.len() * 4
            + self.ids.len() * 4
            + self.screen_bytes()
            + self.block_bytes()
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
        if !self.adjv_off.is_empty() {
            return self.for_each_delta_neighbour(v, f);
        }
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

    /// [`VGraph::for_each_neighbour`] over a delta-coded list.
    #[cold]
    #[inline(never)]
    fn for_each_delta_neighbour(&self, v: usize, mut f: impl FnMut(u32)) {
        let span = &self.adjv[self.adjv_off[v] as usize..self.adjv_off[v + 1] as usize];
        let mut at = 0;
        let mut prev = v as u32;
        let mut first = true;
        while at < span.len() {
            let (delta, used) = get_varint(&span[at..]);
            at += used;
            let u = if first {
                (i64::from(prev) + unzigzag(delta)) as u32
            } else {
                prev + delta as u32 + 1
            };
            first = false;
            prev = u;
            f(u);
        }
    }

    #[inline]
    fn prefetch_neighbours(&self, v: usize) {
        let r = self.degree;
        if !self.adjv_off.is_empty() {
            prefetch_u8(&self.adjv[self.adjv_off[v] as usize..self.adjv_off[v + 1] as usize]);
        } else if self.adj3.is_empty() {
            prefetch_ids(&self.adj[v * r..(v + 1) * r]);
        } else {
            prefetch_u8(&self.adj3[3 * v * r..3 * (v + 1) * r]);
        }
    }

    /// Prefetches the scale of `v` (its own array, one line per random vertex).
    #[inline]
    fn prefetch_scale(&self, v: u32) {
        prefetch_f32(&self.scales[v as usize]);
    }

    /// Prefetches what [`VGraph::fine`] reads for `v`.
    #[inline]
    fn prefetch_rerank_rows(&self, v: u32) {
        prefetch_bytes(self.code(v));
        self.prefetch_scale(v);
        if self.has_residual() {
            let v = v as usize;
            prefetch_bytes(&self.residual[v * self.res_stride..(v + 1) * self.res_stride]);
            prefetch_f32(&self.res_scales[v]);
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
    /// `<q, mean>` and `s_q`, which do not change the order). Euclidean adds `w |x|^2 / 2`
    /// with `w = 1 / s_q`, the order of `|q - x|^2` in the same units.
    #[inline]
    fn est(&self, qc: &[i8], v: u32, w: f32) -> f32 {
        let dot = if self.bits == 8 {
            dot_i8_inline(qc, self.code(v))
        } else {
            dot_i4_inline(qc, self.code(v))
        };
        self.order(v, self.scales[v as usize] * dot as f32, w)
    }

    /// The distance order of `v` from its estimated (or re-ranked) inner product `ip` in
    /// units of `1 / w`: `-ip` for angular, `w |x|^2 / 2 - ip` for Euclidean.
    #[inline]
    fn order(&self, v: u32, ip: f32, w: f32) -> f32 {
        if self.half_norms.is_empty() {
            -ip
        } else {
            self.half_norms[v as usize] * w - ip
        }
    }

    /// [`VGraph::est`] of four vertices with shared query loads; the same integer sums, so
    /// the same values.
    #[inline]
    fn est4(&self, qc: &[i8], v: [u32; 4], w: f32) -> [f32; 4] {
        let dots = if self.bits == 8 {
            v.map(|u| dot_i8_inline(qc, self.code(u)))
        } else {
            dot_i4_inline_x4(qc, v.map(|u| self.code(u)))
        };
        std::array::from_fn(|i| self.order(v[i], self.scales[v[i] as usize] * dots[i] as f32, w))
    }

    /// Re-rank score with the float query: negative estimated inner product with the
    /// reconstructed centred vector (Euclidean: plus `|x|^2 / 2`).
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
        self.order(v, ip, 1.0)
    }

    /// Greedy descent through the upper layers on estimated distances; returns the bottom-layer
    /// entry vertex and the number of estimates.
    fn descend(&self, qc: &[i8], w: f32) -> (u32, usize) {
        if self.layers.is_empty() {
            return (self.entry, 0);
        }
        let mut p = 0u32;
        let mut best = self.est(qc, self.upper_ids[0], w);
        let mut evals = 1;
        for layer in self.layers.iter().rev() {
            loop {
                let mut moved = false;
                for &u in &layer[p as usize] {
                    let d = self.est(qc, self.upper_ids[u as usize], w);
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

    /// A searcher with the knobs of the environment ([`SearchKnobs::from_env`]).
    pub fn searcher(&self) -> VSearcher<'_> {
        self.searcher_with(SearchKnobs::from_env())
    }

    /// A searcher with explicit [`SearchKnobs`].
    pub fn searcher_with(&self, knobs: SearchKnobs) -> VSearcher<'_> {
        VSearcher {
            knobs,
            index: self,
            visited: Marks {
                marks: vec![0; self.n],
                epoch: 0,
                traces: vec![Trace::default(); if self.has_blocks() { self.n } else { 0 }],
            },
            qf: vec![0.0; self.padded],
            qc: vec![0; self.padded],
            qtmp: vec![0; self.padded],
            qrot: vec![0.0; self.screen_words * 64],
            qs: QueryCode::new(self.screen_words.max(1)),
            bqrot: vec![0.0; self.block_dim],
            bq: BlockQuery::new(self.block_m),
            scan: [0; BLOCK],
            fresh: Vec::with_capacity(self.degree),
            survivors: Vec::with_capacity(self.degree),
            screened: 0,
            pool: Pool {
                keys: Vec::new(),
                expanded: Vec::new(),
                cap: 1,
                cursor: 0,
            },
            expanded: BinaryHeap::new(),
            keys: Vec::new(),
            results: Vec::new(),
        }
    }
}

/// Visited marks with an 8-bit epoch: a quarter of the cache footprint of `u32` epochs, at the
/// cost of clearing every 255 searches.
struct Marks {
    marks: Vec<u8>,
    epoch: u8,
    /// With neighbour blocks, one record per vertex instead of the byte mark.
    traces: Vec<Trace>,
}

/// What one search knows about a vertex met through neighbour blocks: the precision-weighted
/// sum of the block estimates it received (one per expanded vertex listing it), their
/// weight, and whether it was expanded itself.
#[derive(Clone, Copy, Default)]
struct Trace {
    epoch: u8,
    expanded: bool,
    w: f32,
    wsum: f32,
    /// The vertex's key in the pool, or `u64::MAX` when it holds none.
    key: u64,
}

impl Marks {
    fn next(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.traces.fill(Trace::default());
            self.epoch = 1;
        }
    }

    /// The trace of `id` in this epoch, and whether it was just created (first encounter).
    #[inline]
    fn trace(&mut self, id: u32) -> (bool, &mut Trace) {
        let t = &mut self.traces[id as usize];
        let fresh = t.epoch != self.epoch;
        if fresh {
            *t = Trace {
                epoch: self.epoch,
                key: u64::MAX,
                ..Trace::default()
            };
        }
        (fresh, t)
    }

    /// True if `id` was visited in this epoch (no mark is set).
    #[inline]
    fn seen(&self, id: u32) -> bool {
        self.marks[id as usize] == self.epoch
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

/// Best-first candidate pool as one sorted array: the `ef` best keys seen, nearest first,
/// each with an expanded flag, and a cursor at the nearest unexpanded entry. One insert is a
/// binary search and a short move; one expansion step is a scan from the cursor; the pool
/// rule ("the nearest unexpanded candidate is among the `ef` best seen") holds by
/// construction, so the search ends when no unexpanded entry is left. Two binary heaps
/// (queue and bounded pool) cost two sifts per admitted candidate and a pop per expansion,
/// and their results came back unsorted; the array is cache-resident at any practical `ef`.
struct Pool {
    keys: Vec<u64>,
    expanded: Vec<bool>,
    cap: usize,
    cursor: usize,
}

impl Pool {
    fn reset(&mut self, cap: usize) {
        self.keys.clear();
        self.expanded.clear();
        self.cap = cap.max(1);
        self.cursor = 0;
    }

    fn is_full(&self) -> bool {
        self.keys.len() >= self.cap
    }

    /// The worst key held, once the pool is full.
    fn worst(&self) -> Option<u64> {
        self.is_full().then(|| self.keys[self.keys.len() - 1])
    }

    /// Admits `key` unless the pool is full of better ones; returns its position and the
    /// key evicted to make room.
    #[inline]
    fn insert(&mut self, key: u64) -> Option<(usize, Option<u64>)> {
        let n = self.keys.len();
        if n >= self.cap && key >= self.keys[n - 1] {
            return None;
        }
        let pos = self.keys.partition_point(|&t| t < key);
        let mut evicted = None;
        if n >= self.cap {
            evicted = self.keys.pop();
            self.expanded.pop();
        }
        self.keys.insert(pos, key);
        self.expanded.insert(pos, false);
        if pos < self.cursor {
            self.cursor = pos;
        }
        Some((pos, evicted))
    }

    /// Drops `key` if present (a re-admitted vertex's older entry).
    fn remove(&mut self, key: u64) {
        let pos = self.keys.partition_point(|&t| t < key);
        if pos < self.keys.len() && self.keys[pos] == key {
            self.keys.remove(pos);
            self.expanded.remove(pos);
            if pos < self.cursor {
                self.cursor -= 1;
            }
        }
    }

    /// The nearest unexpanded key, now marked expanded.
    #[inline]
    fn next(&mut self) -> Option<u64> {
        while self.cursor < self.keys.len() {
            if !self.expanded[self.cursor] {
                self.expanded[self.cursor] = true;
                return Some(self.keys[self.cursor]);
            }
            self.cursor += 1;
        }
        None
    }

    /// The key [`Pool::next`] would return.
    #[inline]
    fn peek_next(&self) -> Option<u64> {
        self.peek_unexpanded(0)
    }

    /// The `n`-th (from 0) unexpanded key in pool order.
    #[inline]
    fn peek_unexpanded(&self, n: usize) -> Option<u64> {
        (self.cursor..self.keys.len())
            .filter(|&i| !self.expanded[i])
            .nth(n)
            .map(|i| self.keys[i])
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
    /// Query rotated for the neighbour blocks, its lookup table, and one block's sums.
    bqrot: Vec<f32>,
    bq: BlockQuery,
    scan: [u32; BLOCK],
    fresh: Vec<u32>,
    /// Fresh neighbours that passed the screen.
    survivors: Vec<u32>,
    /// Neighbours the screen skipped in the last search.
    screened: usize,
    /// The `ef` best candidates seen, with their expansion state.
    pool: Pool,
    /// With blocks: the best `max(k, rerank)` expanded vertices by their own-code estimate,
    /// farthest on top. Block estimates only order the exploration; results come from here.
    expanded: BinaryHeap<u64>,
    keys: Vec<u64>,
    results: Vec<(f32, u32)>,
    knobs: SearchKnobs,
}

impl VSearcher<'_> {
    /// Neighbours the screen skipped (no 8-bit estimate) in the last search.
    pub fn last_screened(&self) -> usize {
        self.screened
    }

    /// The knobs this searcher runs with.
    pub fn knobs(&self) -> SearchKnobs {
        self.knobs
    }

    /// Row estimates of the admitted neighbours in `fresh`, inserted into the pool in list
    /// order (four rows at a time with `batch_est`; the same values either way); returns the
    /// number of estimates. `stale` resets on an insert among the `k` best when `patience`
    /// is on.
    #[inline]
    fn score_fresh(&mut self, k: usize, patience: usize, stale: &mut usize, w: f32) -> usize {
        let g = self.index;
        let qc = &self.qc[..];
        let n = self.fresh.len();
        let mut i = 0;
        if self.knobs.batch_est {
            while i + 4 <= n {
                let ids = [
                    self.fresh[i],
                    self.fresh[i + 1],
                    self.fresh[i + 2],
                    self.fresh[i + 3],
                ];
                for (e, u) in g.est4(qc, ids, w).into_iter().zip(ids) {
                    if let Some((pos, _)) = self.pool.insert(key(e, u)) {
                        if patience > 0 && pos < k {
                            *stale = 0;
                        }
                    }
                }
                i += 4;
            }
        }
        for &u in &self.fresh[i..] {
            let kk = key(g.est(qc, u, w), u);
            if let Some((pos, _)) = self.pool.insert(kk) {
                if patience > 0 && pos < k {
                    *stale = 0;
                }
            }
        }
        n
    }

    /// Writes the ids of the (approximate) `k` nearest vectors by the index's metric (cosine
    /// or Euclidean) into `out`, nearest first, and returns the number of distance estimates
    /// (descent, traversal and re-rank). A query of the wrong dimension, with a non-finite
    /// value, or all zero on an angular index (cosine is undefined) is an error; the pool and
    /// heaps are bounded by the vertex count whatever `ef` is.
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
        if g.half_norms.is_empty() && query.iter().all(|&v| v == 0.0) {
            return Err(invalid("zero query: cosine similarity is undefined".into()));
        }
        let ef = params.ef.max(k).max(1);
        // Only the order matters, so the query is used as given (no centring needed: the
        // `<q, mean>` term is common to all candidates) and its scale is dropped.
        self.qf[..g.dim].copy_from_slice(query);
        let sq = g.encode_query(
            &self.qf,
            &mut self.qc,
            &mut self.qtmp,
            &mut self.qrot,
            &mut self.qs,
        );
        // Weight of the Euclidean norm term in the estimate's units (unused for angular); a
        // zero query has every inner product zero, so the norms alone order it.
        let w = if sq > 0.0 { 1.0 / sq } else { 1.0 };
        let qc = &self.qc[..];
        let blocks = g.has_blocks();
        if blocks {
            g.encode_block_query(&self.qf, sq, &mut self.bqrot, &mut self.bq);
        }
        let knobs = self.knobs;
        let block_screen = blocks && knobs.block_screen;
        let screen = g.has_screen() && params.screen && !blocks;
        let margin = params.screen_sigmas * g.screen_sigma;
        self.screened = 0;

        self.visited.next();
        let (entry, mut evals) = g.descend(qc, w);
        self.visited.insert(entry);
        let fuse = blocks && params.fuse && !block_screen;
        if fuse {
            self.visited.trace(entry);
        }
        // Query terms of the error model: `|q - x_v|^2 = |q|^2 + 1 - 2 (<q, x_v - mean> + <q, mean>)`
        // for unit index rows.
        let (qn2, qm) = if blocks {
            let q = &self.qf[..g.dim];
            (
                q.iter().map(|a| a * a).sum::<f32>(),
                q.iter().zip(&g.mean).map(|(a, m)| a * m).sum::<f32>(),
            )
        } else {
            (0.0, 0.0)
        };
        self.pool.reset(ef);
        self.expanded.clear();
        let take = params.rerank.max(k).min(g.n);
        let k0 = key(g.est(qc, entry, w), entry);
        self.pool.insert(k0);
        if fuse {
            self.visited.trace(entry).1.key = k0;
        }
        evals += 1;
        // Best-first expansion of the nearest unexpanded candidate among the `ef` best seen;
        // the pool holds exactly those, so the search ends when none is left. With
        // `patience`, it also ends after that many expansions without an insert among the
        // `k` best.
        let patience = params.patience;
        let frontier = if knobs.frontier.is_finite() {
            knobs.frontier.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let mut stale = 0usize;
        while let Some(c) = self.pool.next() {
            if patience > 0 && stale >= patience {
                break;
            }
            if frontier < 1.0 && k > 0 && self.pool.is_full() {
                // Frontier rule: the nearest unexpanded candidate already lies past the point
                // `frontier` of the way from the `k`-th best estimate to the `ef`-th. At 1.0
                // it never fires (every pool key is at most the `ef`-th).
                let len = self.pool.keys.len();
                let kth = key_dist(self.pool.keys[k.min(len) - 1]);
                let worst = key_dist(self.pool.keys[len - 1]);
                if key_dist(c) > kth + frontier * (worst - kth) {
                    break;
                }
            }
            if fuse {
                // A re-admitted vertex may sit in the pool twice; expand it once.
                let (_, t) = self.visited.trace(key_id(c));
                if t.expanded {
                    continue;
                }
                t.expanded = true;
            }
            stale += 1;
            if let Some(next) = self.pool.peek_next() {
                g.prefetch_neighbours(key_id(next) as usize);
                if blocks {
                    g.prefetch_blocks(key_id(next) as usize);
                    if !block_screen {
                        prefetch_bytes(g.code(key_id(next)));
                    }
                }
            }
            if knobs.prefetch >= 2 {
                if let Some(second) = self.pool.peek_unexpanded(1) {
                    g.prefetch_neighbours(key_id(second) as usize);
                    if blocks {
                        g.prefetch_blocks(key_id(second) as usize);
                    }
                }
            }
            if block_screen {
                // The neighbour block screens and the pool holds row estimates, so the popped
                // key carries the row estimate of `v` (`key` round-trips the f32 bits). A fresh
                // neighbour whose block estimate, improved by `screen_sigmas` deviations of
                // the calibrated edge error, could not enter the full pool is skipped (it
                // stays visited); the survivors' rows are prefetched together, then estimated.
                let v = key_id(c) as usize;
                let ip_v = -key_dist(c) * sq;
                let d2_v = (qn2 + 1.0 - 2.0 * (ip_v + qm)).max(1e-6);
                let (delta, min_sum, inv_sq) = (self.bq.delta, self.bq.min_sum, self.bq.inv_sq);
                let worst = self.pool.worst().map(key_dist);
                let sigmas = params.screen_sigmas;
                self.fresh.clear();
                let list = &mut self.fresh;
                g.for_each_neighbour(v, |u| list.push(u));
                self.survivors.clear();
                'screen: for b in 0..g.blocks_per_vertex {
                    let (codes, scales, offsets) = g.block(v, b);
                    scan32(codes, &self.bq.lut, g.block_m, &mut self.scan);
                    for i in 0..BLOCK {
                        let Some(&u) = self.fresh.get(b * BLOCK + i) else {
                            break 'screen;
                        };
                        if !self.visited.insert(u) {
                            continue;
                        }
                        let s = le_f32(scales, i);
                        let ip =
                            ip_v + s * (delta * self.scan[i] as f32 + min_sum) + le_f32(offsets, i);
                        if let Some(w) = worst {
                            let sigma = (g.block_kappa * s * s * d2_v).sqrt();
                            if -(ip + sigmas * sigma) * inv_sq >= w {
                                self.screened += 1;
                                continue;
                            }
                        }
                        prefetch_bytes(g.code(u));
                        if knobs.prefetch >= 1 {
                            g.prefetch_scale(u);
                        }
                        self.survivors.push(u);
                    }
                }
                std::mem::swap(&mut self.fresh, &mut self.survivors);
                evals += self.score_fresh(k, patience, &mut stale, w);
                continue;
            }
            if blocks {
                // The expanded vertex is scored with its own code (the per-row estimate the
                // plain search uses everywhere) into the result pool. One table scan per
                // block then estimates all its neighbours, ordering the exploration; list
                // padding is skipped, and a vertex met again is re-admitted only when its
                // fused estimate improves on the entry it holds.
                let v = key_id(c) as usize;
                let est_v = g.est(&self.qc, v as u32, w);
                let ek = key(est_v, v as u32);
                evals += 1;
                // `<q, x_v - mean>` in float units: the estimate dropped the query scale.
                let ip_v = -est_v * sq;
                let d2_v = (qn2 + 1.0 - 2.0 * (ip_v + qm)).max(1e-6);
                if self.expanded.len() < take {
                    self.expanded.push(ek);
                } else if let Some(mut worst) = self.expanded.peek_mut() {
                    if ek < *worst {
                        *worst = ek;
                    }
                }
                let (delta, min_sum, inv_sq) = (self.bq.delta, self.bq.min_sum, self.bq.inv_sq);
                self.fresh.clear();
                let list = &mut self.fresh;
                g.for_each_neighbour(v, |u| list.push(u));
                'blocks: for b in 0..g.blocks_per_vertex {
                    let (codes, scales, offsets) = g.block(v, b);
                    scan32(codes, &self.bq.lut, g.block_m, &mut self.scan);
                    for i in 0..BLOCK {
                        let Some(&u) = self.fresh.get(b * BLOCK + i) else {
                            break 'blocks;
                        };
                        let s = le_f32(scales, i);
                        let ip =
                            ip_v + s * (delta * self.scan[i] as f32 + min_sum) + le_f32(offsets, i);
                        if !fuse {
                            // Without fusion the block path keeps the plain path's byte marks
                            // (the per-vertex trace records would not stay in cache).
                            if !self.visited.insert(u) {
                                continue;
                            }
                            evals += 1;
                            if let Some((pos, _)) = self.pool.insert(key(-ip * inv_sq, u)) {
                                if patience > 0 && pos < k {
                                    stale = 0;
                                }
                            }
                            continue;
                        }
                        // Precision of this estimate under the calibrated error model.
                        let w = 1.0 / (g.block_kappa * s * s * d2_v + 1e-12);
                        let (fresh, t) = self.visited.trace(u);
                        let ip = if fresh {
                            t.w = w;
                            t.wsum = w * ip;
                            ip
                        } else {
                            if t.expanded {
                                continue;
                            }
                            t.w += w;
                            t.wsum += w * ip;
                            t.wsum / t.w
                        };
                        let kk = key(-ip * inv_sq, u);
                        let old = t.key;
                        evals += 1;
                        if !fresh && kk >= old {
                            continue;
                        }
                        if old != u64::MAX {
                            self.pool.remove(old);
                        }
                        let Some((pos, evicted)) = self.pool.insert(kk) else {
                            if old != u64::MAX {
                                self.visited.trace(u).1.key = u64::MAX;
                            }
                            continue;
                        };
                        self.visited.trace(u).1.key = kk;
                        if let Some(e) = evicted {
                            self.visited.trace(key_id(e)).1.key = u64::MAX;
                        }
                        if patience > 0 && pos < k {
                            stale = 0;
                        }
                    }
                }
                continue;
            }
            self.fresh.clear();
            let (visited, fresh) = (&mut self.visited, &mut self.fresh);
            let scale_too = knobs.prefetch >= 1;
            let admit = |u: u32, visited: &mut Marks, fresh: &mut Vec<u32>| {
                if visited.insert(u) {
                    if screen {
                        g.prefetch_screen(u);
                    } else {
                        prefetch_bytes(g.code(u));
                        if scale_too {
                            g.prefetch_scale(u);
                        }
                    }
                    fresh.push(u);
                }
            };
            if params.neighbour_cap == 0 {
                g.for_each_neighbour(key_id(c) as usize, |u| admit(u, visited, fresh));
            } else {
                let mut seen = 0usize;
                g.for_each_neighbour(key_id(c) as usize, |u| {
                    seen += 1;
                    if seen <= params.neighbour_cap {
                        admit(u, visited, fresh);
                    }
                });
            }
            if screen {
                // A neighbour whose optimistic 1-bit estimate cannot enter the full pool is
                // skipped (it stays visited); the rest get the 8-bit estimate.
                let worst = self.pool.worst().map(key_dist);
                self.survivors.clear();
                for &u in &self.fresh {
                    if let Some(w) = worst {
                        if g.screen_est(&self.qs, u) - margin >= w {
                            self.screened += 1;
                            continue;
                        }
                    }
                    prefetch_bytes(g.code(u));
                    if scale_too {
                        g.prefetch_scale(u);
                    }
                    self.survivors.push(u);
                }
                std::mem::swap(&mut self.fresh, &mut self.survivors);
            }
            evals += self.score_fresh(k, patience, &mut stale, w);
            if knobs.prefetch >= 3 {
                // Lookahead: what the next expansion's admission touches for its unvisited
                // neighbours, one expansion early (its list was prefetched above).
                if let Some(next) = self.pool.peek_next() {
                    let visited = &self.visited;
                    g.for_each_neighbour(key_id(next) as usize, |u| {
                        if !visited.seen(u) {
                            if screen {
                                g.prefetch_screen(u);
                            } else {
                                prefetch_bytes(g.code(u));
                                g.prefetch_scale(u);
                            }
                        }
                    });
                }
            }
        }

        self.keys.clear();
        if blocks && !block_screen {
            self.keys.extend(self.expanded.drain());
            self.keys.sort_unstable();
        } else {
            self.keys.extend_from_slice(&self.pool.keys);
        }
        self.results.clear();
        let take = take.min(self.keys.len());
        if params.rerank > 0 {
            if knobs.prefetch >= 1 {
                for &c in &self.keys[..take] {
                    g.prefetch_rerank_rows(key_id(c));
                }
            }
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
/// cache written for other data or parameters is rejected instead of silently reused; 03 the
/// quantized-search flag; 04 the metric.
const CACHE_MAGIC: &[u8; 8] = b"HMSGRF04";

/// Fingerprint of the prepared vectors (normalized for angular) a graph was built over.
fn data_fingerprint(vectors: &[f32]) -> u64 {
    let mut h = super::Fnv::new();
    vectors.iter().for_each(|x| h.word(x.to_bits()));
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
    /// Writes the header (n, dim, degree, build_ef, alpha, seed, entry, data fingerprint,
    /// quantized_search, metric), the adjacency and the upper layers; the vectors are not
    /// stored.
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
            data_fingerprint(&self.vectors),
            u64::from(self.build.quantized_search),
            self.metric.code(),
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

    /// Reads a graph written by [`Graph::save`] for the same `data` (prepared here exactly
    /// as [`Graph::build`] does) and the same `params`. Any mismatch in the vector count,
    /// dimension, degree, build_ef, alpha, seed, quantized search, metric or the data
    /// fingerprint is an error, so a stale cache never stands in for a build; delete the file
    /// to rebuild.
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
        let mut header = [0u64; 10];
        for x in &mut header {
            *x = get_u64(&mut r)?;
        }
        let [n, d, degree, build_ef, alpha_bits, seed, entry, fingerprint, quantized, metric] =
            header;
        if dim == 0 || !data.len().is_multiple_of(dim) {
            return Err(bad("graph cache: ragged data".into()));
        }
        let mut vectors = data.to_vec();
        let norms = prepare_rows(&mut vectors, dim, params.metric);
        let checks: [(&str, u64, u64); 9] = [
            ("metric", metric, params.metric.code()),
            ("n", n, (data.len() / dim) as u64),
            ("dim", d, dim as u64),
            ("degree", degree, params.degree as u64),
            ("build_ef", build_ef, params.build_ef as u64),
            ("alpha", alpha_bits, u64::from(params.alpha.to_bits())),
            ("seed", seed, params.seed),
            ("data fingerprint", fingerprint, data_fingerprint(&vectors)),
            (
                "quantized_search",
                quantized,
                u64::from(params.quantized_search),
            ),
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
            metric: params.metric,
            vectors,
            norms,
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
        assert_eq!(back.vectors, graph.vectors);
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
            (
                "metric",
                BuildParams {
                    metric: Metric::Euclidean,
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

    /// On clustered rows whose norms spread over [0.1, 10], a Euclidean index (4-bit codes,
    /// 8-bit residual re-rank, reordered 3-byte ids: the benchmark layout) finds the exact
    /// Euclidean neighbours, while an angular index over the same rows, which ranks by
    /// direction alone, clearly does not.
    #[test]
    fn euclidean_index_finds_euclidean_neighbours() {
        let (d, n, k) = (32, 4000, 10);
        let spread = |rows: &mut [f32], salt: usize| {
            let count = rows.len() / d;
            for (i, x) in rows.chunks_exact_mut(d).enumerate() {
                let s = 0.1 * 100f32.powf(((i * 7919 + salt) % count) as f32 / count as f32);
                x.iter_mut().for_each(|v| *v *= s);
            }
        };
        // Queries are held-out rows of the same clusters.
        let mut data = clustered(n + 100, d, 7);
        spread(&mut data, 0);
        let queries = data.split_off(n * d);
        let l2 = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                .sum()
        };
        let recall = |metric: Metric| -> f64 {
            let params = BuildParams {
                metric,
                degree: 32,
                build_ef: 100,
                ..BuildParams::default()
            };
            let g = Graph::build(&data, d, &params).expect("build");
            let index = VGraph::from_graph_with(&g, 4, 8, true, true, 3, false).expect("encode");
            let mut s = index.searcher_with(SearchKnobs::OFF);
            let mut out = Vec::new();
            let mut hits = 0;
            for q in queries.chunks_exact(d) {
                let mut all: Vec<(f64, u32)> = data
                    .chunks_exact(d)
                    .enumerate()
                    .map(|(i, x)| (l2(q, x), i as u32))
                    .collect();
                all.sort_by(|a, b| a.0.total_cmp(&b.0));
                let truth: Vec<u32> = all[..k].iter().map(|t| t.1).collect();
                let params = VSearchParams {
                    ef: 64,
                    rerank: 32,
                    ..VSearchParams::default()
                };
                s.search(q, k, params, &mut out).expect("search");
                hits += out.iter().filter(|x| truth.contains(x)).count();
            }
            hits as f64 / (k * 100) as f64
        };
        let (euclidean, angular) = (recall(Metric::Euclidean), recall(Metric::Angular));
        assert!(euclidean >= 0.95, "Euclidean recall {euclidean}");
        assert!(
            angular < euclidean - 0.3,
            "angular recall {angular} against Euclidean {euclidean}"
        );
    }

    /// A Euclidean index refuses the angular-only encodings and accepts a zero query (its
    /// nearest rows are the ones of smallest norm).
    #[test]
    fn euclidean_index_limits() {
        let d = 16;
        let mut data = clustered(500, d, 2);
        for (i, x) in data.chunks_exact_mut(d).enumerate() {
            x.iter_mut().for_each(|v| *v *= 1.0 + i as f32 / 50.0);
        }
        let params = BuildParams {
            metric: Metric::Euclidean,
            degree: 16,
            build_ef: 64,
            ..BuildParams::default()
        };
        let g = Graph::build(&data, d, &params).expect("build");
        assert!(VGraph::from_graph_with(&g, 8, 0, false, false, 4, true).is_err());
        assert!(super::super::QGraph::from_graph(&g, &params).is_err());
        let blocks = VGraph::from_graph(&g, false)
            .expect("encode")
            .with_blocks(&g, 4);
        assert!(blocks.is_err());
        let index = VGraph::from_graph(&g, true).expect("encode");
        let mut out = Vec::new();
        let params = VSearchParams {
            ef: 500,
            rerank: 500,
            ..VSearchParams::default()
        };
        index
            .searcher_with(SearchKnobs::OFF)
            .search(&[0.0; 16], 1, params, &mut out)
            .expect("zero query");
        let smallest = data
            .chunks_exact(d)
            .enumerate()
            .min_by(|a, b| {
                let n2 = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>();
                n2(a.1).total_cmp(&n2(b.1))
            })
            .map(|m| m.0 as u32);
        assert_eq!(out.first().copied(), smallest);
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

    /// Neighbour-block codes estimate every neighbour from the expanded vertex's record; the
    /// traversal must reach the same recall band as the per-row 8-bit estimates after the
    /// residual re-rank, at every block width, with and without renumbering or 3-byte ids.
    #[test]
    fn block_codes_keep_recall() {
        let d = 24;
        let data = clustered(4000, d, 1);
        let queries = clustered(100, d, 2);
        let graph = Graph::build(&data, d, &BuildParams::default()).expect("build");
        let plain = VGraph::from_graph(&graph, true).expect("encode");
        let rr = VSearchParams {
            ef: 96,
            rerank: 64,
            ..VSearchParams::default()
        };
        let r0 = recall(&plain, &data, &queries, d, rr);
        for bits in [4u8, 2, 1] {
            for (reorder, id_bytes) in [(false, 4u8), (true, 3)] {
                let b = VGraph::from_graph_with(&graph, 8, 8, true, reorder, id_bytes, false)
                    .expect("encode")
                    .with_blocks(&graph, bits)
                    .expect("blocks");
                assert!(b.has_blocks());
                assert!(b.index_bytes() > plain.index_bytes());
                let r = recall(&b, &data, &queries, d, rr);
                // 24 coordinates are few for 1-bit codes (a 1/sqrt(dim) relative error on
                // this deliberately hard set), so the band widens with coarseness; the
                // estimator itself is checked against exact inner products below.
                let slack = match bits {
                    4 => 0.03,
                    2 => 0.06,
                    _ => 0.30,
                };
                assert!(
                    r >= r0 - slack,
                    "{bits}-bit blocks (reorder {reorder}, ids {id_bytes}) recall {r} vs {r0}"
                );
            }
        }
        let e = match VGraph::from_graph(&graph, true)
            .expect("encode")
            .with_blocks(&graph, 3)
        {
            Ok(_) => panic!("3 bits accepted"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("block bits"), "{e}");
    }

    /// The block estimate of a neighbour, in `est` units, must track the exact centred inner
    /// product: Pearson correlation over the neighbours of 64 vertices above a per-width bar.
    #[test]
    fn block_estimates_track_exact_inner_products() {
        let d = 64;
        let data = clustered(3000, d, 5);
        let graph = Graph::build(&data, d, &BuildParams::default()).expect("build");
        // The index works on unit rows; so must the exact products.
        let unit = &graph.vectors;
        for (bits, bar) in [(4u8, 0.998), (2, 0.98), (1, 0.90)] {
            let g = VGraph::from_graph(&graph, false)
                .expect("encode")
                .with_blocks(&graph, bits)
                .expect("blocks");
            let mut s = g.searcher();
            let q: Vec<f32> = unit[7 * d..8 * d].to_vec();
            s.qf[..d].copy_from_slice(&q);
            let sq = g.encode_query(&s.qf, &mut s.qc, &mut s.qtmp, &mut s.qrot, &mut s.qs);
            g.encode_block_query(&s.qf, sq, &mut s.bqrot, &mut s.bq);
            let centred =
                |u: usize| -> f32 { (0..d).map(|j| q[j] * (unit[u * d + j] - g.mean[j])).sum() };
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for v in 0..64usize {
                // The exact `<q, x_v - mean>` stands in for the expanded vertex's estimate.
                let ip_v = centred(v);
                let mut list = Vec::new();
                g.for_each_neighbour(v, |u| list.push(u));
                for b in 0..g.blocks_per_vertex {
                    let (codes, scales, offsets) = g.block(v, b);
                    scan32(codes, &s.bq.lut, g.block_m, &mut s.scan);
                    for i in 0..BLOCK {
                        let Some(&u) = list.get(b * BLOCK + i) else {
                            break;
                        };
                        let ip = ip_v
                            + le_f32(scales, i) * (s.bq.delta * s.scan[i] as f32 + s.bq.min_sum)
                            + le_f32(offsets, i);
                        xs.push(f64::from(ip));
                        ys.push(f64::from(centred(u as usize)));
                    }
                }
            }
            let n = xs.len() as f64;
            let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for (x, y) in xs.iter().zip(&ys) {
                sxy += (x - mx) * (y - my);
                sxx += (x - mx) * (x - mx);
                syy += (y - my) * (y - my);
            }
            let r = sxy / (sxx * syy).sqrt();
            eprintln!(
                "{bits}-bit block estimates: correlation {r:.4} over {} pairs",
                xs.len()
            );
            assert!(
                r > bar,
                "{bits}-bit block estimates: correlation {r} below {bar}"
            );
        }
    }

    /// Delta-coded lists hold the same neighbour sets as the 4-byte ones, in fewer bytes,
    /// and the search returns the same ids; the block index decodes them in the same order
    /// it was built in.
    #[test]
    fn delta_coded_adjacency_round_trips() {
        let d = 24;
        let data = clustered(4000, d, 1);
        let queries = clustered(50, d, 2);
        let graph = Graph::build(&data, d, &BuildParams::default()).expect("build");
        let wide = VGraph::from_graph_with(&graph, 4, 8, true, true, 4, false).expect("encode");
        let delta = VGraph::from_graph_with(&graph, 4, 8, true, true, 0, false).expect("encode");
        assert!(
            delta.index_bytes() < wide.index_bytes(),
            "{} vs {}",
            delta.index_bytes(),
            wide.index_bytes()
        );
        for v in 0..wide.n {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            wide.for_each_neighbour(v, |u| a.push(u));
            delta.for_each_neighbour(v, |u| b.push(u));
            a.sort_unstable();
            assert_eq!(a, b, "vertex {v}");
        }
        let p = VSearchParams {
            ef: 48,
            rerank: 16,
            ..VSearchParams::default()
        };
        let (mut s1, mut s2) = (wide.searcher(), delta.searcher());
        let (mut o1, mut o2) = (Vec::new(), Vec::new());
        for q in queries.chunks_exact(d) {
            s1.search(q, 10, p, &mut o1).expect("search");
            s2.search(q, 10, p, &mut o2).expect("search");
            assert_eq!(o1, o2);
        }
        let blocks = VGraph::from_graph_with(&graph, 8, 8, true, true, 0, false)
            .expect("encode")
            .with_blocks(&graph, 2)
            .expect("blocks");
        assert!(recall(&blocks, &data, &queries, d, p) > 0.8);
        assert_eq!(zigzag(-1), 1);
        assert_eq!(unzigzag(zigzag(-123_456)), -123_456);
        let mut buf = Vec::new();
        put_varint(&mut buf, 300);
        assert_eq!(get_varint(&buf), (300, 2));
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
        assert!(
            a.codes.bytes() == b.codes.bytes()
                && a.residual[..] == b.residual[..]
                && a.adj[..] == b.adj[..]
        );
        assert!(a.scales[..] == b.scales[..] && a.res_scales[..] == b.res_scales[..]);
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
