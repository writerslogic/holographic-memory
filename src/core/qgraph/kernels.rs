// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inner loops of the quantized graph: rotation, float dot product, and the bit-plane
//! popcount that turns 1-bit edge codes into inner-product estimates.

/// Bits per query coordinate in the bit-plane estimator.
pub(crate) const QUERY_BITS: usize = 4;
const ROUNDS: usize = 3;

/// SplitMix64: small, seedable and stable across platforms and crate versions.
pub(crate) struct SplitMix(pub(crate) u64);

impl SplitMix {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    #[cfg(test)]
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Random orthogonal transform: zero-pad to a power of two (at least 64), then three rounds
/// of random sign flips followed by a fast Walsh-Hadamard transform. Not Haar-distributed,
/// but close enough for the RaBitQ estimator in practice (checked by the bias test).
pub(crate) struct Rotation {
    dim: usize,
    padded: usize,
    signs: Vec<f32>,
    scale: f32,
}

impl Rotation {
    /// `bits` is the requested output length (0 = automatic); see `BuildParams::code_bits`.
    pub(crate) fn new(dim: usize, bits: usize, seed: u64) -> Self {
        let padded = dim.max(bits).next_power_of_two().max(64);
        let mut rng = SplitMix(seed);
        let signs = (0..ROUNDS * padded)
            .map(|_| if rng.next_u64() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        Self {
            dim,
            padded,
            signs,
            scale: (padded as f32).powf(-0.5 * ROUNDS as f32),
        }
    }

    pub(crate) fn padded(&self) -> usize {
        self.padded
    }

    pub(crate) fn apply(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(out.len(), self.padded);
        out[..self.dim].copy_from_slice(x);
        out[self.dim..].fill(0.0);
        for signs in self.signs.chunks_exact(self.padded) {
            for (o, s) in out.iter_mut().zip(signs) {
                *o *= s;
            }
            fwht(out);
        }
        for o in out.iter_mut() {
            *o *= self.scale;
        }
    }
}

/// Unnormalized in-place Walsh-Hadamard transform; `x.len()` must be a power of two.
fn fwht(x: &mut [f32]) {
    let mut h = 1;
    while h < x.len() {
        for chunk in x.chunks_exact_mut(2 * h) {
            let (a, b) = chunk.split_at_mut(h);
            for (u, v) in a.iter_mut().zip(b.iter_mut()) {
                let t = *u;
                *u = t + *v;
                *v = t - *v;
            }
        }
        h *= 2;
    }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 16];
    let ((ca, ta), (cb, tb)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    let tail: f32 = ta.iter().zip(tb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for ((s, p), q) in acc.iter_mut().zip(x).zip(y) {
            *s += p * q;
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// A query coordinate vector quantized to [`QUERY_BITS`] unsigned bits per coordinate and
/// stored as bit planes: `planes[p * words + w]` holds bit `p` of coordinates `64w..64w+63`.
/// `sum_i b_i x_i ~= lo * popcount(b) + delta * raw(b)` for any bit vector `b`.
pub(crate) struct QueryCode {
    pub(crate) planes: Vec<u64>,
    pub(crate) lo: f32,
    pub(crate) delta: f32,
    /// Sum of the dequantized coordinates.
    pub(crate) sum: f32,
}

impl QueryCode {
    pub(crate) fn new(words: usize) -> Self {
        Self {
            planes: vec![0; QUERY_BITS * words],
            lo: 0.0,
            delta: 1.0,
            sum: 0.0,
        }
    }

    pub(crate) fn encode(&mut self, x: &[f32]) {
        let words = self.planes.len() / QUERY_BITS;
        debug_assert_eq!(x.len(), 64 * words);
        let (lo, hi) = x
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        let levels = ((1 << QUERY_BITS) - 1) as f32;
        let delta = if hi > lo { (hi - lo) / levels } else { 1.0 };
        self.planes.fill(0);
        let mut total = 0u32;
        for (i, &v) in x.iter().enumerate() {
            let q = ((v - lo) / delta).round().clamp(0.0, levels) as u32;
            total += q;
            for p in 0..QUERY_BITS {
                self.planes[p * words + i / 64] |= u64::from((q >> p) & 1) << (i % 64);
            }
        }
        self.lo = lo;
        self.delta = delta;
        self.sum = lo * x.len() as f32 + delta * total as f32;
    }
}

/// For each code row (`words` u64 each) computes `sum_p 2^p * popcount(code & plane_p)`.
pub(crate) fn raw_estimates(codes: &[u64], planes: &[u64], words: usize, out: &mut [u32]) {
    #[cfg(target_arch = "aarch64")]
    match words {
        1 => return neon::raw::<1>(codes, planes, out),
        2 => return neon::raw::<2>(codes, planes, out),
        4 => return neon::raw::<4>(codes, planes, out),
        8 => return neon::raw::<8>(codes, planes, out),
        16 => return neon::raw::<16>(codes, planes, out),
        _ => {}
    }
    raw_scalar(codes, planes, words, out)
}

pub(crate) fn raw_scalar(codes: &[u64], planes: &[u64], words: usize, out: &mut [u32]) {
    assert_eq!(planes.len(), QUERY_BITS * words);
    assert_eq!(codes.len(), out.len() * words);
    for (code, o) in codes.chunks_exact(words).zip(out.iter_mut()) {
        *o = code
            .iter()
            .enumerate()
            .map(|(w, &c)| scalar_word(c, planes, words, w))
            .sum();
    }
}

fn scalar_word(c: u64, planes: &[u64], words: usize, w: usize) -> u32 {
    (0..QUERY_BITS)
        .map(|p| (c & planes[p * words + w]).count_ones() << p)
        .sum()
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::{scalar_word, QUERY_BITS};

    /// [`super::DotI8`] with the SDOT instruction. Handed out only by `dot_i8_kernel` after
    /// detecting `dotprod`.
    #[cfg_attr(target_feature = "dotprod", allow(dead_code))]
    pub(super) fn dot_i8_sdot(a: &[i8], b: &[i8]) -> i32 {
        assert!(a.len() == b.len() && a.len().is_multiple_of(16));
        // SAFETY: the only caller path checked `dotprod` at runtime.
        unsafe { sdot(a, b) }
    }

    #[target_feature(enable = "dotprod")]
    #[inline]
    pub(super) unsafe fn sdot(a: &[i8], b: &[i8]) -> i32 {
        let (mut s0, mut s1) = (vdupq_n_s32(0), vdupq_n_s32(0));
        let mut i = 0;
        // SAFETY (all loads): `i + 16 <= len` (or `i + 32 <= len`) for both slices, whose
        // lengths are equal and a multiple of 16 (asserted by the caller).
        while i + 32 <= a.len() {
            let (x0, y0) = (vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i)));
            let (x1, y1) = (
                vld1q_s8(a.as_ptr().add(i + 16)),
                vld1q_s8(b.as_ptr().add(i + 16)),
            );
            std::arch::asm!(
                "sdot {s0:v}.4s, {x0:v}.16b, {y0:v}.16b",
                "sdot {s1:v}.4s, {x1:v}.16b, {y1:v}.16b",
                s0 = inout(vreg) s0, s1 = inout(vreg) s1,
                x0 = in(vreg) x0, y0 = in(vreg) y0, x1 = in(vreg) x1, y1 = in(vreg) y1,
                options(pure, nomem, nostack, preserves_flags)
            );
            i += 32;
        }
        if i < a.len() {
            let (x0, y0) = (vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i)));
            std::arch::asm!(
                "sdot {s0:v}.4s, {x0:v}.16b, {y0:v}.16b",
                s0 = inout(vreg) s0, x0 = in(vreg) x0, y0 = in(vreg) y0,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        vaddvq_s32(vaddq_s32(s0, s1))
    }

    /// [`super::dot_i4_inline`] with SDOT: each 16 packed bytes become the 16 even and the 16
    /// odd coordinates of a 32-wide block, matched with the query laid out by
    /// [`super::interleave_for_i4`].
    #[target_feature(enable = "dotprod")]
    #[inline]
    #[cfg_attr(not(target_feature = "dotprod"), allow(dead_code))]
    pub(super) unsafe fn sdot4(q: &[i8], p: &[i8]) -> i32 {
        let (mut s0, mut s1) = (vdupq_n_s32(0), vdupq_n_s32(0));
        let mut i = 0;
        // SAFETY (all loads): the caller checked `q.len() == 2 * p.len()` and `p.len() % 16 == 0`,
        // so `i + 16 <= p.len()` and `2 * i + 32 <= q.len()`.
        while i + 16 <= p.len() {
            let x = vld1q_s8(p.as_ptr().add(i));
            let lo = vshrq_n_s8::<4>(vshlq_n_s8::<4>(x));
            let hi = vshrq_n_s8::<4>(x);
            let (qe, qo) = (
                vld1q_s8(q.as_ptr().add(2 * i)),
                vld1q_s8(q.as_ptr().add(2 * i + 16)),
            );
            std::arch::asm!(
                "sdot {s0:v}.4s, {lo:v}.16b, {qe:v}.16b",
                "sdot {s1:v}.4s, {hi:v}.16b, {qo:v}.16b",
                s0 = inout(vreg) s0, s1 = inout(vreg) s1,
                lo = in(vreg) lo, qe = in(vreg) qe, hi = in(vreg) hi, qo = in(vreg) qo,
                options(pure, nomem, nostack, preserves_flags)
            );
            i += 16;
        }
        vaddvq_s32(vaddq_s32(s0, s1))
    }

    /// [`super::DotI8`] with widening multiplies; for cores without `dotprod`.
    #[cfg_attr(target_feature = "dotprod", allow(dead_code))]
    pub(super) fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        assert!(a.len() == b.len() && a.len().is_multiple_of(16));
        let mut i = 0;
        // SAFETY: NEON is part of the aarch64 baseline; `i + 16 <= len` for both slices. Each
        // i16 lane holds one product (|x| <= 128 * 128), and the pairwise adds go to i32.
        unsafe {
            let mut s = vdupq_n_s32(0);
            while i < a.len() {
                let (x, y) = (vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i)));
                s = vpadalq_s16(s, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
                s = vpadalq_s16(s, vmull_high_s8(x, y));
                i += 16;
            }
            vaddvq_s32(s)
        }
    }

    pub(super) fn raw<const W: usize>(codes: &[u64], planes: &[u64], out: &mut [u32]) {
        assert_eq!(QUERY_BITS, 4);
        assert_eq!(planes.len(), 4 * W);
        assert_eq!(codes.len(), out.len() * W);
        for (code, o) in codes.as_chunks::<W>().0.iter().zip(out.iter_mut()) {
            let mut total = 0u32;
            let mut k = 0;
            while k + 2 <= W {
                // SAFETY: NEON is part of the aarch64 baseline. `k + 2 <= W` keeps each 16-byte
                // load inside `code` (length W) and inside plane `p` (planes[p*W..p*W+W], length
                // checked by the assert above). Per byte the weighted sum is at most
                // 8 * (1 + 2 + 4 + 8) = 120, so the u8 additions cannot overflow.
                total += unsafe {
                    let x = vreinterpretq_u8_u64(vld1q_u64(code.as_ptr().add(k)));
                    let p0 = vreinterpretq_u8_u64(vld1q_u64(planes.as_ptr().add(k)));
                    let p1 = vreinterpretq_u8_u64(vld1q_u64(planes.as_ptr().add(W + k)));
                    let p2 = vreinterpretq_u8_u64(vld1q_u64(planes.as_ptr().add(2 * W + k)));
                    let p3 = vreinterpretq_u8_u64(vld1q_u64(planes.as_ptr().add(3 * W + k)));
                    let c0 = vcntq_u8(vandq_u8(x, p0));
                    let c1 = vcntq_u8(vandq_u8(x, p1));
                    let c2 = vcntq_u8(vandq_u8(x, p2));
                    let c3 = vcntq_u8(vandq_u8(x, p3));
                    let s = vaddq_u8(
                        vaddq_u8(c0, vshlq_n_u8::<1>(c1)),
                        vaddq_u8(vshlq_n_u8::<2>(c2), vshlq_n_u8::<3>(c3)),
                    );
                    u32::from(vaddlvq_u8(s))
                };
                k += 2;
            }
            if W % 2 == 1 {
                total += scalar_word(code[W - 1], planes, W, W - 1);
            }
            *o = total;
        }
    }
}

/// Integer dot product of two i8 rows whose length is a multiple of 16.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "dotprod"),
    allow(dead_code)
)]
pub(crate) type DotI8 = fn(&[i8], &[i8]) -> i32;

/// The fastest [`DotI8`] this CPU supports.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "dotprod"),
    allow(dead_code)
)]
pub(crate) fn dot_i8_kernel() -> DotI8 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return neon::dot_i8_sdot;
        }
        neon::dot_i8
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_i8_scalar
}

/// [`DotI8`] resolved at compile time, so the search loop can inline it: SDOT when the build
/// target enables `dotprod` (every Apple core), else the runtime-detected kernel.
#[inline(always)]
pub(crate) fn dot_i8_inline(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    {
        assert!(a.len() == b.len() && a.len().is_multiple_of(16));
        // SAFETY: `dotprod` is enabled for the whole build; the lengths are checked above.
        unsafe { neon::sdot(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
    dot_i8_kernel()(a, b)
}

/// Lays out an i8 query code for [`dot_i4_inline`]: per 32-wide block, its 16 even then its
/// 16 odd coordinates. `q.len()` is a multiple of 32.
pub(crate) fn interleave_for_i4(q: &[i8], out: &mut [i8]) {
    assert!(q.len() == out.len() && q.len().is_multiple_of(32));
    for (src, dst) in q.chunks_exact(32).zip(out.chunks_exact_mut(32)) {
        for j in 0..16 {
            dst[j] = src[2 * j];
            dst[16 + j] = src[2 * j + 1];
        }
    }
}

/// `sum_i q_i * c_i` for a query code laid out by [`interleave_for_i4`] and a row packed as
/// [`dot_f32_i4`] reads it; `q.len() == 2 * packed.len()`, a multiple of 32.
#[inline(always)]
pub(crate) fn dot_i4_inline(q: &[i8], packed: &[i8]) -> i32 {
    assert!(q.len() == 2 * packed.len() && packed.len().is_multiple_of(16));
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    {
        // SAFETY: `dotprod` is enabled for the whole build; the lengths are checked above.
        unsafe { neon::sdot4(q, packed) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
    dot_i4_scalar(q, packed)
}

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "dotprod"))))]
pub(crate) fn dot_i4_scalar(q: &[i8], packed: &[i8]) -> i32 {
    let mut s = 0i32;
    for (qb, pb) in q.chunks_exact(32).zip(packed.chunks_exact(16)) {
        for (j, &x) in pb.iter().enumerate() {
            let lo = (((x as u8) << 4) as i8) >> 4;
            s += i32::from(qb[j]) * i32::from(lo) + i32::from(qb[16 + j]) * i32::from(x >> 4);
        }
    }
    s
}

/// `sum_i a_i * c_i` where `c` is packed two signed 4-bit values per byte, element `2j` in the
/// low nibble of byte `j` and `2j + 1` in the high nibble; `a.len() == 2 * packed.len()`.
pub(crate) fn dot_f32_i4(a: &[f32], packed: &[i8]) -> f32 {
    debug_assert_eq!(a.len(), 2 * packed.len());
    let mut acc = [0f32; 16];
    let ((ca, ta), (cb, tb)) = (a.as_chunks::<16>(), packed.as_chunks::<8>());
    debug_assert!(
        ta.is_empty() && tb.is_empty(),
        "length not a multiple of 16"
    );
    for (x, y) in ca.iter().zip(cb) {
        for (j, &b) in y.iter().enumerate() {
            let lo = f32::from((((b as u8) << 4) as i8) >> 4);
            let hi = f32::from(b >> 4);
            acc[2 * j] += x[2 * j] * lo;
            acc[2 * j + 1] += x[2 * j + 1] * hi;
        }
    }
    acc.iter().sum::<f32>()
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(crate) fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// `sum_i a_i * b_i` for a float row and an i8 row of the same length.
pub(crate) fn dot_f32_i8(a: &[f32], b: &[i8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 16];
    let ((ca, ta), (cb, tb)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    let tail: f32 = ta.iter().zip(tb).map(|(x, &y)| x * f32::from(y)).sum();
    for (x, y) in ca.iter().zip(cb) {
        for ((s, p), &q) in acc.iter_mut().zip(x).zip(y) {
            *s += p * f32::from(q);
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Hint the cache to load every line of `bytes` (128-byte lines on Apple cores).
#[inline]
pub(crate) fn prefetch_bytes(bytes: &[i8]) {
    #[cfg(target_arch = "aarch64")]
    if let Some(last) = bytes.len().checked_sub(1) {
        let base = bytes.as_ptr();
        let mut off = 0;
        loop {
            let at = off.min(last);
            // SAFETY: PRFM is a hint that never faults or writes; `at < bytes.len()`, so the
            // address is inside the live slice.
            unsafe {
                std::arch::asm!(
                    "prfm pldl1keep, [{0}]",
                    in(reg) base.add(at),
                    options(nostack, preserves_flags, readonly)
                );
            }
            if at == last {
                break;
            }
            off += 128;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = bytes;
}

/// Hint the cache to load `words`. A no-op where no prefetch instruction is wired up.
#[inline]
pub(crate) fn prefetch(words: &[u64]) {
    #[cfg(target_arch = "aarch64")]
    for line in words.chunks(16) {
        // SAFETY: PRFM is a hint: it never faults, writes no register or memory, and the
        // address points into a live slice.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) line.as_ptr(),
                options(nostack, preserves_flags, readonly)
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = words;
}

pub(crate) fn as_f32(w: &[u64]) -> &[f32] {
    // SAFETY: u64 alignment satisfies f32 alignment, the f32 view covers exactly the same
    // bytes, every bit pattern is a valid f32, and the lifetime is tied to `w`.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast::<f32>(), w.len() * 2) }
}

pub(crate) fn as_f32_mut(w: &mut [u64]) -> &mut [f32] {
    // SAFETY: as in `as_f32`; the exclusive borrow of `w` is carried over to the view.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<f32>(), w.len() * 2) }
}

pub(crate) fn as_u32(w: &[u64]) -> &[u32] {
    // SAFETY: as in `as_f32`; every bit pattern is a valid u32.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast::<u32>(), w.len() * 2) }
}

pub(crate) fn as_u32_mut(w: &mut [u64]) -> &mut [u32] {
    // SAFETY: as in `as_f32_mut`; every bit pattern is a valid u32.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<u32>(), w.len() * 2) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_popcount_matches_scalar() {
        let mut rng = SplitMix(7);
        for words in 1..=17 {
            let codes: Vec<u64> = (0..32 * words).map(|_| rng.next_u64()).collect();
            let planes: Vec<u64> = (0..QUERY_BITS * words).map(|_| rng.next_u64()).collect();
            let (mut fast, mut slow) = ([0u32; 32], [0u32; 32]);
            raw_estimates(&codes, &planes, words, &mut fast);
            raw_scalar(&codes, &planes, words, &mut slow);
            assert_eq!(fast, slow, "words = {words}");
        }
    }

    #[test]
    fn integer_dot_kernels_match_scalar() {
        let mut rng = SplitMix(13);
        let kernels: Vec<DotI8> = {
            #[cfg(target_arch = "aarch64")]
            {
                let mut k: Vec<DotI8> = vec![neon::dot_i8];
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    k.push(neon::dot_i8_sdot);
                }
                k
            }
            #[cfg(not(target_arch = "aarch64"))]
            vec![dot_i8_kernel()]
        };
        for len in (0..=96).step_by(16) {
            let a: Vec<i8> = (0..len).map(|_| rng.next_u64() as i8).collect();
            let mut b: Vec<i8> = (0..len).map(|_| rng.next_u64() as i8).collect();
            if len > 0 {
                b[0] = -128;
            }
            let want = dot_i8_scalar(&a, &b);
            for k in &kernels {
                assert_eq!(k(&a, &b), want, "len = {len}");
            }
            assert_eq!(dot_i8_inline(&a, &b), want, "len = {len}");
            let af: Vec<f32> = a.iter().map(|&x| f32::from(x)).collect();
            assert_eq!(dot_f32_i8(&af, &b), want as f32);
            let nib: Vec<i8> = b.iter().map(|&x| x >> 4).collect();
            let packed: Vec<i8> = nib
                .chunks_exact(2)
                .map(|p| ((p[0] as u8 & 0x0F) | ((p[1] as u8) << 4)) as i8)
                .collect();
            assert_eq!(dot_f32_i4(&af, &packed), dot_i8_scalar(&a, &nib) as f32);
            if len % 32 == 0 {
                let mut qi = vec![0i8; len];
                interleave_for_i4(&a, &mut qi);
                let want4 = dot_i8_scalar(&a, &nib);
                assert_eq!(dot_i4_scalar(&qi, &packed), want4, "len = {len}");
                assert_eq!(dot_i4_inline(&qi, &packed), want4, "len = {len}");
            }
        }
    }

    #[test]
    fn dot_matches_naive() {
        let mut rng = SplitMix(3);
        for len in 0..70 {
            let a: Vec<f32> = (0..len).map(|_| rng.next_f32() - 0.5).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.next_f32() - 0.5).collect();
            let naive: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y)).sum();
            assert!((f64::from(dot(&a, &b)) - naive).abs() < 1e-5);
        }
    }

    #[test]
    fn rotation_preserves_inner_products() {
        let rot = Rotation::new(100, 0, 11);
        let mut rng = SplitMix(5);
        let a: Vec<f32> = (0..100).map(|_| rng.next_f32() - 0.5).collect();
        let b: Vec<f32> = (0..100).map(|_| rng.next_f32() - 0.5).collect();
        let (mut ra, mut rb) = (vec![0.0; 128], vec![0.0; 128]);
        rot.apply(&a, &mut ra);
        rot.apply(&b, &mut rb);
        assert!((dot(&ra, &rb) - dot(&a, &b)).abs() < 1e-4);
        assert!((dot(&ra, &ra) - dot(&a, &a)).abs() < 1e-4);
    }

    #[test]
    fn query_code_reconstructs_bit_sums() {
        let mut rng = SplitMix(9);
        let x: Vec<f32> = (0..128).map(|_| rng.next_f32() - 0.5).collect();
        let mut qc = QueryCode::new(2);
        qc.encode(&x);
        let code = [rng.next_u64(), rng.next_u64()];
        let mut raw = [0u32; 1];
        raw_estimates(&code, &qc.planes, 2, &mut raw);
        let pop = code.iter().map(|c| c.count_ones()).sum::<u32>() as f32;
        let approx = qc.lo * pop + qc.delta * raw[0] as f32;
        let exact: f32 = (0..128)
            .filter(|i| code[i / 64] >> (i % 64) & 1 == 1)
            .map(|i| x[i])
            .sum();
        // Rounding error is at most delta / 2 per selected coordinate.
        assert!((approx - exact).abs() <= qc.delta / 2.0 * pop + 1e-4);
    }
}
