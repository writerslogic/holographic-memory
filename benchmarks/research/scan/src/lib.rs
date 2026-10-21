// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Graph-free "split-manifold" scan: kill test for the panel's #1 direction.
//!
//! # Mechanism
//!
//! Every document `x` (unit-norm, dimension `D`) is split against a frozen,
//! public, secret-free basis `(mu, U, R)`:
//!
//! * `mu`: centre (mean of a 1,000-vector public calibration sample).
//! * `U`: `H` orthonormal rows (`H x D`, `H` in 16/32/64, `--head-dim`), either
//!   the top-`H` PCA subspace of the calibration sample (subspace iteration) or
//!   a fixed random orthonormal basis (flag-selectable, see [`codes::BasisKind`]).
//! * `R`: `M` independent subsampled randomized Walsh-Hadamard rotations
//!   (`--residual-mult`) on the zero-padded power-of-two dimension `P >= D`:
//!   `R_m v = H diag(s_m) pad(v) / sqrt(P)`, concatenated to `M * P`
//!   coordinates, of which a fixed random subset `S` of size `B` (the residual
//!   bit count, a multiple of 64, `B <= M * P`) is kept.
//!
//! Decomposition: `x - mu = U^T h + r` with head `h = U (x - mu)` (`H` floats) and
//! residual `r` orthogonal to the head subspace. For a query `q`:
//!
//! ```text
//! <q, x> = <q, mu> + <U q, h> + <q_perp, r>,     q_perp = q - U^T (U q)
//! ```
//!
//! `<q, mu>` is constant per query and dropped. The head term is an int8 x int16
//! dot (doc head quantized per document, query head per query). The residual
//! term is estimated from a 1-bit sign code `b = sign(R r)` with a per-document
//! scalar `rho` that makes the RaBitQ-style asymmetric estimator unbiased under
//! the random rotation:
//!
//! ```text
//! y = R r (B kept coords),  rho = (P / B) * |y|^2 / sum_i |y_i|
//! <q_perp, r> ~= rho * sum_i sign(y_i) * z_i,   z = R q_perp
//! ```
//!
//! (`P / B` is right for any `M`: each rotation preserves `<q_perp, r>`, so the
//! sum over all `M * P` coordinates is `M <q_perp, r>` and keeping `B` of them
//! scales it by `B / (M * P)`.)
//!
//! Two query-side evaluations of `sum_i sign(y_i) z_i` are implemented:
//!
//! * `sym1`: query also 1-bit, `sum ~= mean|z| * (B - 2 * hamming(b, sign(z)))`,
//!   one XOR + popcount per 64-bit word.
//! * `asym4`: query quantized to 4 bits per coordinate as four bit planes,
//!   `sum_{b_i = 1} z_i` via four AND + popcount per word plus the stored
//!   popcount of `b` (the standard 4-bit-query / 1-bit-code dot).
//!
//! # Record layout (one contiguous `[u64; HW + 1 + W]` per document)
//!
//! `HW = H / 8` head words, `W = B / 64` residual words:
//!
//! ```text
//! word 0..HW        : H x int8 head (little-endian bytes)
//! word HW           : bits  0..16 head scale  (IEEE half)
//!                     bits 16..32 rho         (IEEE half)
//!                     bits 32..48 popcount(b) (u16, used by asym4 only)
//!                     bits 48..64 reserved (zero)
//! word HW+1..HW+1+W : residual sign bits, coordinate i at word i/64, bit i%64
//! ```
//!
//! `D = 768, H = 16, B = 768` gives `2 + 1 + 12 = 15` words = 120 bytes per
//! document (the primary f32 row for exact rerank lives in a separate array).
//!
//! # Invariants
//!
//! * All stored vectors are l2-normalized; scores are cosines.
//! * `H` is one of [`codes::SUPPORTED_HEAD_DIMS`]; `B % 64 == 0`, `B <= M * P`,
//!   `W` is one of the monomorphized word counts in [`codes::SUPPORTED_WORDS`].
//! * Delete never touches the scan loop: a deleted slot's scalar word is zeroed
//!   (score 0) and the liveness flag in [`codes::Primary`] is cleared; the exact
//!   rerank / final filter is the correctness guarantee.
//! * Deterministic: every random draw comes from [`data::Rng`] seeded from the
//!   CLI seed.
//!
//! No root-crate API is used; the crate is plain Rust with auto-vectorizable
//! loops (lane-accumulator dots, `u64::count_ones` popcounts).

#![allow(clippy::needless_range_loop)]

use std::cmp::Ordering;

pub mod codes;
pub mod data;
pub mod scan;

/// Lane count for the vectorizable dot products.
const LANES: usize = 16;

/// Dot product with 16 independent accumulators so the compiler can
/// vectorize the f32 reduction without reassociating a single scalar chain.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let ra = ca.remainder();
    let rb = cb.remainder();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ca.zip(cb) {
        for j in 0..LANES {
            acc[j] += x[j] * y[j];
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    acc.iter().sum::<f32>() + tail
}

/// Integer dot of two int8 rows with lane accumulators (`i32` cannot overflow:
/// `127 * 127 * len` stays below `2^31` for `len < 133,000`).
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let ra = ca.remainder();
    let rb = cb.remainder();
    let mut acc = [0i32; LANES];
    for (x, y) in ca.zip(cb) {
        for j in 0..LANES {
            acc[j] += (x[j] as i32) * (y[j] as i32);
        }
    }
    let mut tail = 0i32;
    for (x, y) in ra.iter().zip(rb) {
        tail += (*x as i32) * (*y as i32);
    }
    acc.iter().sum::<i32>() + tail
}

/// Euclidean norm.
pub fn norm_f32(a: &[f32]) -> f32 {
    dot_f32(a, a).sqrt()
}

/// Scale `v` to unit norm; a zero vector is left unchanged.
pub fn normalize_in_place(v: &mut [f32]) {
    let n = norm_f32(v);
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// IEEE 754 binary16 encode, round-to-nearest-even, overflow to infinity.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        let nan_payload = if mant != 0 { 0x200 } else { 0 };
        return sign | 0x7c00 | nan_payload;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | (h as u16)
}

/// IEEE 754 binary16 decode (exact, handles subnormals, infinities, NaN).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let v = (mant as f32) * (1.0 / 16_777_216.0);
        return f32::from_bits(sign | v.to_bits());
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// Branch-free binary16 decode for the scan loop. Exact for finite inputs
/// (normals and subnormals); infinities and NaNs are never stored.
#[inline(always)]
pub fn f16_fast(h: u16) -> f32 {
    const TWO_POW_112: f32 = f32::from_bits(0x7780_0000);
    let mag = f32::from_bits(((h as u32) & 0x7fff) << 13) * TWO_POW_112;
    f32::from_bits(mag.to_bits() | (((h as u32) & 0x8000) << 16))
}

/// Bounded top-K keeper: a binary min-heap on the score, so the root is the
/// current admission threshold. `push` is O(log K) only when the candidate
/// beats the root; the scan loop compares against [`TopK::threshold`] first.
pub struct TopK {
    k: usize,
    heap: Vec<(f32, u32)>,
}

impl TopK {
    /// Create a keeper for `k >= 1` entries.
    pub fn new(k: usize) -> Self {
        assert!(k >= 1, "TopK needs k >= 1");
        Self {
            k,
            heap: Vec::with_capacity(k),
        }
    }

    /// Capacity.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Drop all entries.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Current admission threshold (`-inf` until the heap is full).
    #[inline(always)]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::NEG_INFINITY
        } else {
            self.heap[0].0
        }
    }

    /// Insert if the heap is not full or `score` beats the threshold.
    #[inline]
    pub fn push(&mut self, score: f32, id: u32) {
        if self.heap.len() < self.k {
            self.heap.push((score, id));
            self.sift_up(self.heap.len() - 1);
        } else if score > self.heap[0].0 {
            self.heap[0] = (score, id);
            self.sift_down(0);
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.heap[i].0 < self.heap[p].0 {
                self.heap.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            if l >= n {
                break;
            }
            let r = l + 1;
            let mut m = l;
            if r < n && self.heap[r].0 < self.heap[l].0 {
                m = r;
            }
            if self.heap[m].0 < self.heap[i].0 {
                self.heap.swap(i, m);
                i = m;
            } else {
                break;
            }
        }
    }

    /// Entries in heap order (unsorted).
    pub fn entries(&self) -> &[(f32, u32)] {
        &self.heap
    }

    /// Sort the entries by descending score (ties by ascending id), append the
    /// ids to `out`, and empty the heap. Allocation-free after warmup.
    pub fn drain_sorted_desc(&mut self, out: &mut Vec<u32>) {
        self.heap.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        out.clear();
        out.extend(self.heap.iter().map(|e| e.1));
        self.heap.clear();
    }
}

/// Median of a sample (returns 0 for an empty slice).
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_is_exact_on_representable_values() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 65504.0, 6.103_515_6e-5, 5.960_464_5e-8, 0.0024] {
            let h = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(h);
            let fast = f16_fast(h);
            assert_eq!(back.to_bits(), fast.to_bits(), "decoders disagree on {v}");
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-9, "{v} -> {back}");
        }
        assert_eq!(f32_to_f16_bits(1e9), 0x7c00);
        assert!(f16_bits_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn topk_keeps_the_largest_scores() {
        let mut t = TopK::new(3);
        for (i, s) in [0.1f32, 0.9, 0.5, 0.7, 0.2, 0.8].iter().enumerate() {
            if *s > t.threshold() {
                t.push(*s, i as u32);
            }
        }
        let mut out = Vec::new();
        t.drain_sorted_desc(&mut out);
        assert_eq!(out, vec![1, 5, 3]);
    }

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.1).collect();
        let expect: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot_f32(&a, &b) - expect).abs() < 1e-3);
        let ai: Vec<i8> = (0..37).map(|i| (i as i8) - 18).collect();
        let bi: Vec<i8> = (0..37).map(|i| 3 - (i as i8)).collect();
        let expect_i: i32 = ai.iter().zip(&bi).map(|(x, y)| (*x as i32) * (*y as i32)).sum();
        assert_eq!(dot_i8(&ai, &bi), expect_i);
    }
}
