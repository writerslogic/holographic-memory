// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Inner loops of the quantized graph: rotation, float dot product, and the bit-plane
//! popcount that turns 1-bit edge codes into inner-product estimates.

/// Bits per query coordinate in the bit-plane estimator.
const QUERY_BITS: usize = 4;
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

    #[cfg(feature = "experimental")]
    pub(crate) fn index_bytes(&self) -> usize {
        self.signs.len() * std::mem::size_of::<f32>()
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
pub(crate) fn fwht<T>(x: &mut [T])
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>,
{
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

#[must_use]
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

fn raw_scalar(codes: &[u64], planes: &[u64], words: usize, out: &mut [u32]) {
    assert_eq!(planes.len(), QUERY_BITS * words);
    assert_eq!(codes.len(), out.len() * words);
    for (code, o) in codes.chunks_exact(words).zip(out.iter_mut()) {
        *o = raw_one_scalar(code, planes, words);
    }
}

/// [`raw_estimates`] for one code row: NEON AND + CNT over the bit planes for the row widths
/// the index produces (16 planes x 64 words at most), scalar otherwise.
#[must_use]
#[inline]
pub(crate) fn raw_one(code: &[u64], planes: &[u64], words: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        let mut out = [0u32; 1];
        match words {
            2 => {
                neon::raw::<2>(code, planes, &mut out);
                return out[0];
            }
            4 => {
                neon::raw::<4>(code, planes, &mut out);
                return out[0];
            }
            8 => {
                neon::raw::<8>(code, planes, &mut out);
                return out[0];
            }
            16 => {
                neon::raw::<16>(code, planes, &mut out);
                return out[0];
            }
            _ => {}
        }
    }
    raw_one_scalar(code, planes, words)
}

/// Scalar form of [`raw_one`] (the reference, and the microbenchmark's baseline).
#[inline]
fn raw_one_scalar(code: &[u64], planes: &[u64], words: usize) -> u32 {
    code.iter()
        .enumerate()
        .map(|(w, &c)| scalar_word(c, planes, words, w))
        .sum()
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

    /// [`sdot`] behind a length check, for the kernel tests (after detecting `dotprod`).
    #[cfg(test)]
    pub(super) fn dot_i8_sdot(a: &[i8], b: &[i8]) -> i32 {
        assert!(a.len() == b.len() && a.len().is_multiple_of(16));
        // SAFETY: the only caller detected `dotprod` at runtime; lengths checked above.
        unsafe { sdot(a, b) }
    }

    /// # Safety
    /// The CPU supports `dotprod`; `a` and `b` have the same length, a multiple of 16.
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

    /// [`super::dot_i4_inline`] with SDOT: each 16 packed bytes become the 16 even and the 16 odd
    /// coordinates of a 32-wide block, matched with the query laid out by
    /// [`super::interleave_for_i4`].
    ///
    /// # Safety
    /// The CPU supports `dotprod`; `q.len() == 2 * p.len()` and `p.len()` is a multiple of 16.
    #[target_feature(enable = "dotprod")]
    #[inline]
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

    /// [`sdot4`] with widening multiplies; for cores without `dotprod`. Each i16 lane holds one
    /// product (|c * q| <= 8 * 128) and the pairwise adds go to i32.
    #[cfg_attr(target_feature = "dotprod", allow(dead_code))]
    pub(super) fn dot_i4(q: &[i8], p: &[i8]) -> i32 {
        assert!(q.len() == 2 * p.len() && p.len().is_multiple_of(16));
        let mut i = 0;
        // SAFETY: NEON is part of the aarch64 baseline; `i + 16 <= p.len()` and
        // `2 * i + 32 <= q.len()` by the assert above.
        unsafe {
            let mut s = vdupq_n_s32(0);
            while i < p.len() {
                let x = vld1q_s8(p.as_ptr().add(i));
                let lo = vshrq_n_s8::<4>(vshlq_n_s8::<4>(x));
                let hi = vshrq_n_s8::<4>(x);
                let (qe, qo) = (
                    vld1q_s8(q.as_ptr().add(2 * i)),
                    vld1q_s8(q.as_ptr().add(2 * i + 16)),
                );
                s = vpadalq_s16(s, vmull_s8(vget_low_s8(lo), vget_low_s8(qe)));
                s = vpadalq_s16(s, vmull_high_s8(lo, qe));
                s = vpadalq_s16(s, vmull_s8(vget_low_s8(hi), vget_low_s8(qo)));
                s = vpadalq_s16(s, vmull_high_s8(hi, qo));
                i += 16;
            }
            vaddvq_s32(s)
        }
    }

    /// [`super::dot_i8_inline`] with widening multiplies; for cores without `dotprod`.
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

/// The x86 kernel tier, detected once. `HMS_QGRAPH_KERNEL=scalar|avx2|vnni` (read once, at the
/// first distance call) pins a lower tier than the CPU supports, so every tier can be run and
/// compared on one machine; `auto` or unset takes the best detected tier, and a tier above
/// what the CPU supports falls back to the best detected one.
#[cfg(target_arch = "x86_64")]
mod tier {
    use std::sync::atomic::{AtomicU8, Ordering};

    pub(super) const SCALAR: u8 = 1;
    pub(super) const AVX2: u8 = 2;
    pub(super) const VNNI: u8 = 3;
    static TIER: AtomicU8 = AtomicU8::new(0);

    #[inline(always)]
    pub(super) fn get() -> u8 {
        match TIER.load(Ordering::Relaxed) {
            0 => detect(),
            t => t,
        }
    }

    #[cold]
    fn detect() -> u8 {
        let best = if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
        {
            VNNI
        } else if is_x86_feature_detected!("avx2") {
            AVX2
        } else {
            SCALAR
        };
        let want = match std::env::var("HMS_QGRAPH_KERNEL").as_deref() {
            Ok("scalar") => SCALAR,
            Ok("avx2") => AVX2,
            Ok("vnni") => VNNI,
            Ok("auto") | Err(_) => best,
            Ok(other) => {
                eprintln!("HMS_QGRAPH_KERNEL={other:?} is not scalar|avx2|vnni|auto; using auto");
                best
            }
        };
        let tier = want.min(best);
        TIER.store(tier, Ordering::Relaxed);
        tier
    }
}

/// Integer dot product of two i8 rows whose length is a multiple of 16, dispatched inline so
/// the search loop makes no indirect call: SDOT when the build target enables `dotprod`
/// (every Apple core), else the kernel of the detected tier (one relaxed load, a plain move
/// on x86).
#[must_use]
#[inline(always)]
pub(crate) fn dot_i8_inline(a: &[i8], b: &[i8]) -> i32 {
    assert!(a.len() == b.len() && a.len().is_multiple_of(16));
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    {
        // SAFETY: `dotprod` is enabled for the whole build; the lengths are checked above.
        unsafe { neon::sdot(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
    {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                // SAFETY: `dotprod` was detected at runtime; the lengths are checked above.
                return unsafe { neon::sdot(a, b) };
            }
            neon::dot_i8(a, b)
        }
        #[cfg(target_arch = "x86_64")]
        match tier::get() {
            // SAFETY: the tier is VNNI only after detecting `avx512vnni`, `avx512vl` and `avx2`;
            // the lengths are checked above.
            tier::VNNI => unsafe { x86::dot_i8_vnni(a, b) },
            // SAFETY: the tier is AVX2 only after detecting `avx2`; the lengths are checked above.
            tier::AVX2 => unsafe { x86::dot_i8_avx2(a, b) },
            _ => dot_i8_scalar(a, b),
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        dot_i8_scalar(a, b)
    }
}

/// Lays out an i8 query code for [`dot_i4_inline`]: per 32-wide block, its 16 even then its
/// 16 odd coordinates. `q.len()` is a multiple of 32.
pub(crate) fn interleave_for_i4(q: &[i8], out: &mut [i8]) {
    assert!(q.len() == out.len() && q.len().is_multiple_of(32));
    for (src, dst) in q
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
    {
        for j in 0..16 {
            dst[j] = src[2 * j];
            dst[16 + j] = src[2 * j + 1];
        }
    }
}

/// `sum_i q_i * c_i` for a query code laid out by [`interleave_for_i4`] and a row packed as
/// [`dot_f32_i4`] reads it; `q.len() == 2 * packed.len()`, a multiple of 32.
#[must_use]
#[inline(always)]
pub(crate) fn dot_i4_inline(q: &[i8], packed: &[i8]) -> i32 {
    assert!(q.len() == 2 * packed.len() && packed.len().is_multiple_of(16));
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    {
        // SAFETY: `dotprod` is enabled for the whole build; the lengths are checked above.
        unsafe { neon::sdot4(q, packed) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
    {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: `dotprod` was detected at runtime; the slice lengths were checked above.
            return unsafe { neon::sdot4(q, packed) };
        }
        #[cfg(target_arch = "aarch64")]
        {
            neon::dot_i4(q, packed)
        }
        #[cfg(target_arch = "x86_64")]
        match tier::get() {
            // SAFETY: the tier is VNNI only after detecting `avx512vnni`, `avx512vl` and `avx2`;
            // the slice lengths were checked above.
            tier::VNNI => return unsafe { x86::dot_i4_vnni(q, packed) },
            // SAFETY: the tier is AVX2 or above only after detecting `avx2`.
            tier::AVX2 => return unsafe { x86::dot_i4_avx2(q, packed) },
            _ => {}
        }
        #[cfg(not(target_arch = "aarch64"))]
        dot_i4_scalar(q, packed)
    }
}

/// [`dot_i4_inline`] of one query against four rows of equal length. On AVX2 the query's
/// widened halves are loaded once per 16 packed bytes and the four row loads are in flight
/// together; the sums are the same integers, so every result equals the single call's.
#[must_use]
#[inline(always)]
pub(crate) fn dot_i4_inline_x4(q: &[i8], rows: [&[i8]; 4]) -> [i32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        let len = rows[0].len();
        if rows.iter().all(|r| r.len() == len)
            && q.len() == 2 * len
            && len.is_multiple_of(16)
            && tier::get() >= tier::AVX2
        {
            // SAFETY: the tier is AVX2 or VNNI only after detecting `avx2`; the lengths were
            // checked above.
            return unsafe { x86::dot_i4_avx2_x4(q, rows) };
        }
    }
    [
        dot_i4_inline(q, rows[0]),
        dot_i4_inline(q, rows[1]),
        dot_i4_inline(q, rows[2]),
        dot_i4_inline(q, rows[3]),
    ]
}

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "dotprod"))))]
fn dot_i4_scalar(q: &[i8], packed: &[i8]) -> i32 {
    let mut s = 0i32;
    for (qb, pb) in q
        .as_chunks::<32>()
        .0
        .iter()
        .zip(packed.as_chunks::<16>().0.iter())
    {
        for (j, &x) in pb.iter().enumerate() {
            let lo = (((x as u8) << 4) as i8) >> 4;
            s += i32::from(qb[j]) * i32::from(lo) + i32::from(qb[16 + j]) * i32::from(x >> 4);
        }
    }
    s
}

/// `sum_i a_i * c_i` where `c` is packed two signed 4-bit values per byte, element `2j` in the
/// low nibble of byte `j` and `2j + 1` in the high nibble; `a.len() == 2 * packed.len()`.
#[must_use]
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
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// `sum_i a_i * b_i` for a float row and an i8 row of the same length.
#[must_use]
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
    #[cfg(target_arch = "x86_64")]
    for line in bytes.chunks(64) {
        // SAFETY: PREFETCHT0 is a hint that never faults or writes; the address is inside
        // the live slice.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                line.as_ptr().cast::<i8>(),
            );
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
    #[cfg(target_arch = "x86_64")]
    for line in words.chunks(8) {
        // SAFETY: as in `prefetch_bytes`.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                line.as_ptr().cast::<i8>(),
            );
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = words;
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Signed i8 dot product of two rows of equal length, a multiple of 16, exact over the
    /// whole i8 range: the bytes widen to i16 lanes (`vpmovsxbw`) and `vpmaddwd` sums pairs
    /// of products into i32 (the `maddubs` form would misread -128 after the sign trick).
    ///
    /// # Safety
    /// The CPU supports AVX2; `a` and `b` have the same length.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        // SAFETY (all loads): `i + 16 <= len` for both slices, whose lengths are equal.
        while i + 16 <= a.len() {
            let x = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i).cast()));
            let y = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i).cast()));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
            i += 16;
        }
        horizontal_sum(acc)
    }

    /// [`super::dot_i4_inline`]: each 16 packed bytes become the 16 even and the 16 odd
    /// coordinates of a 32-wide block, matched with the query laid out by
    /// [`super::interleave_for_i4`].
    ///
    /// # Safety
    /// The CPU supports AVX2; `q.len() == 2 * p.len()` and `p.len()` is a multiple of 16.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_i4_avx2(q: &[i8], p: &[i8]) -> i32 {
        let mask = _mm_set1_epi8(0x0f);
        let eight = _mm_set1_epi8(8);
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        // SAFETY (all loads): the caller checked `q.len() == 2 * p.len()` and `p.len() % 16 == 0`,
        // so `i + 16 <= p.len()` and `2 * i + 32 <= q.len()`.
        while i + 16 <= p.len() {
            let x = _mm_loadu_si128(p.as_ptr().add(i).cast());
            // Sign-extend each nibble: ((n & 15) ^ 8) - 8.
            let lo = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(x, mask), eight), eight);
            let hi = _mm_sub_epi8(
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi16::<4>(x), mask), eight),
                eight,
            );
            let qe = _mm256_cvtepi8_epi16(_mm_loadu_si128(q.as_ptr().add(2 * i).cast()));
            let qo = _mm256_cvtepi8_epi16(_mm_loadu_si128(q.as_ptr().add(2 * i + 16).cast()));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(lo), qe));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(hi), qo));
            i += 16;
        }
        horizontal_sum(acc)
    }

    /// [`dot_i4_avx2`] of four rows against one query: the query's widened halves are loaded
    /// once per 16 packed bytes and each row keeps its own accumulator, so the four row loads
    /// are in flight together. Lane by lane the same integer sums as four single calls.
    ///
    /// # Safety
    /// The CPU supports AVX2; the rows have one length, a multiple of 16, and `q.len()` is
    /// twice it.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_i4_avx2_x4(q: &[i8], rows: [&[i8]; 4]) -> [i32; 4] {
        let mask = _mm_set1_epi8(0x0f);
        let eight = _mm_set1_epi8(8);
        let mut acc = [_mm256_setzero_si256(); 4];
        let mut i = 0;
        // SAFETY (all loads): the caller checked that every row has the same length, a
        // multiple of 16, and that `q.len()` is twice it, so `i + 16 <= row.len()` and
        // `2 * i + 32 <= q.len()`.
        while i + 16 <= rows[0].len() {
            let qe = _mm256_cvtepi8_epi16(_mm_loadu_si128(q.as_ptr().add(2 * i).cast()));
            let qo = _mm256_cvtepi8_epi16(_mm_loadu_si128(q.as_ptr().add(2 * i + 16).cast()));
            for (a, row) in acc.iter_mut().zip(rows) {
                let x = _mm_loadu_si128(row.as_ptr().add(i).cast());
                let lo = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(x, mask), eight), eight);
                let hi = _mm_sub_epi8(
                    _mm_xor_si128(_mm_and_si128(_mm_srli_epi16::<4>(x), mask), eight),
                    eight,
                );
                *a = _mm256_add_epi32(*a, _mm256_madd_epi16(_mm256_cvtepi8_epi16(lo), qe));
                *a = _mm256_add_epi32(*a, _mm256_madd_epi16(_mm256_cvtepi8_epi16(hi), qo));
            }
            i += 16;
        }
        [
            horizontal_sum(acc[0]),
            horizontal_sum(acc[1]),
            horizontal_sum(acc[2]),
            horizontal_sum(acc[3]),
        ]
    }

    /// [`dot_i8_avx2`] with `vpdpbusd` on 256-bit registers (AVX512-VNNI with VL): `a` is biased
    /// by 128 into the unsigned operand, so `sum (a + 128) b = a.b + 128 sum b`, and a second
    /// `vpdpbusd` against ones yields `sum b` to subtract. Exact over the whole i8 range. The
    /// 512-bit forms measured no faster on 128 and 256-wide rows (kill test).
    ///
    /// # Safety
    /// The CPU supports AVX512F, AVX512VL, AVX512-VNNI and AVX2; `a` and `b` have the same
    /// length, a multiple of 16.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2")]
    pub(super) unsafe fn dot_i8_vnni(a: &[i8], b: &[i8]) -> i32 {
        let (flip, ones) = (_mm256_set1_epi8(-128), _mm256_set1_epi8(1));
        let (mut acc, mut sum_b) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let mut i = 0;
        // SAFETY (all loads): `i + 32 <= len` in the wide loop, `i + 16 <= len` after it.
        while i + 32 <= a.len() {
            let x = _mm256_xor_si256(_mm256_loadu_si256(a.as_ptr().add(i).cast()), flip);
            let y = _mm256_loadu_si256(b.as_ptr().add(i).cast());
            acc = _mm256_dpbusd_epi32(acc, x, y);
            sum_b = _mm256_dpbusd_epi32(sum_b, ones, y);
            i += 32;
        }
        let mut total = horizontal_sum(acc) - 128 * horizontal_sum(sum_b);
        if i < a.len() {
            total += super::x86::dot_i8_avx2(&a[i..], &b[i..]);
        }
        total
    }

    /// [`dot_i4_avx2`] with `vpdpbusd`: the nibbles are the unsigned operand, offset so that
    /// `n ^ 8` is the signed nibble plus 8, in 0..=15; the query is the signed one. Then
    /// `sum c q = sum (c + 8) q - 8 sum q`, with `sum q` from `vpdpbusd` against ones.
    ///
    /// # Safety
    /// The CPU supports AVX512F, AVX512VL, AVX512-VNNI and AVX2; `q.len() == 2 * p.len()` and
    /// `p.len()` is a multiple of 16.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2")]
    pub(super) unsafe fn dot_i4_vnni(q: &[i8], p: &[i8]) -> i32 {
        let mask = _mm_set1_epi8(0x0f);
        let eight = _mm_set1_epi8(8);
        let ones = _mm_set1_epi8(1);
        let (mut acc, mut sum_q) = (_mm_setzero_si128(), _mm_setzero_si128());
        let mut i = 0;
        // SAFETY (all loads): as in `dot_i4_avx2`.
        while i + 16 <= p.len() {
            let x = _mm_loadu_si128(p.as_ptr().add(i).cast());
            let lo = _mm_xor_si128(_mm_and_si128(x, mask), eight);
            let hi = _mm_xor_si128(_mm_and_si128(_mm_srli_epi16::<4>(x), mask), eight);
            let qe = _mm_loadu_si128(q.as_ptr().add(2 * i).cast());
            let qo = _mm_loadu_si128(q.as_ptr().add(2 * i + 16).cast());
            acc = _mm_dpbusd_epi32(acc, lo, qe);
            acc = _mm_dpbusd_epi32(acc, hi, qo);
            sum_q = _mm_dpbusd_epi32(sum_q, ones, qe);
            sum_q = _mm_dpbusd_epi32(sum_q, ones, qo);
            i += 16;
        }
        let h = |v: __m128i| {
            let s = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
            _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s)))
        };
        h(acc) - 8 * h(sum_q)
    }

    /// # Safety
    /// The CPU supports AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        _mm_cvtsi128_si32(s)
    }
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
    fn popcount_kernels_match_scalar_at_every_width() {
        let seed = 7;
        let mut rng = SplitMix(seed);
        for words in 1..=17 {
            let codes: Vec<u64> = (0..32 * words).map(|_| rng.next_u64()).collect();
            let planes: Vec<u64> = (0..QUERY_BITS * words).map(|_| rng.next_u64()).collect();
            let (mut fast, mut slow) = ([0u32; 32], [0u32; 32]);
            raw_estimates(&codes, &planes, words, &mut fast);
            raw_scalar(&codes, &planes, words, &mut slow);
            assert_eq!(fast, slow, "raw_estimates, words {words}, seed {seed}");
            for (row, (code, want)) in codes.chunks_exact(words).zip(slow).enumerate() {
                assert_eq!(
                    raw_one(code, &planes, words),
                    want,
                    "raw_one, row {row}, words {words}, seed {seed}"
                );
            }
        }
    }

    /// The dispatched entry points and the float-by-integer dots, including the empty row;
    /// the raw kernels are covered by `every_integer_kernel_matches_scalar`.
    #[test]
    fn dispatched_and_float_dots_match_scalar() {
        let seed = 13;
        let mut rng = SplitMix(seed);
        for len in (0..=96).step_by(16) {
            let a = random_i8(&mut rng, len);
            let mut b = random_i8(&mut rng, len);
            if let Some(first) = b.first_mut() {
                *first = -128;
            }
            let want = dot_i8_scalar(&a, &b);
            assert_eq!(dot_i8_inline(&a, &b), want, "len {len}, seed {seed}");
            let af: Vec<f32> = a.iter().map(|&x| f32::from(x)).collect();
            assert_eq!(dot_f32_i8(&af, &b), want as f32, "len {len}, seed {seed}");
            let nib: Vec<i8> = b.iter().map(|&x| x >> 4).collect();
            let packed = pack_nibbles(&b);
            let want4 = dot_i8_scalar(&a, &nib);
            assert_eq!(
                dot_f32_i4(&af, &packed),
                want4 as f32,
                "len {len}, seed {seed}"
            );
            if len % 32 == 0 {
                let mut qi = vec![0i8; len];
                interleave_for_i4(&a, &mut qi);
                assert_eq!(dot_i4_scalar(&qi, &packed), want4, "len {len}, seed {seed}");
                assert_eq!(dot_i4_inline(&qi, &packed), want4, "len {len}, seed {seed}");
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

    /// Kill test for the 1-bit screen (experiment E2): per neighbour,
    /// the screen must cost < 25% of the 8-bit SDOT estimate on a nytimes-sized working set
    /// (300k rows, 256 dims: 256-byte code rows, 40-byte screen rows) with the search loop's
    /// access pattern (64 random fresh neighbours per expansion, prefetched, then estimated).
    /// `cargo test --release raw_screen_microbenchmark -- --ignored --nocapture`; writes JSON
    /// to `$HMS_SCREEN_BENCH_OUT` when set.
    #[test]
    #[ignore = "microbenchmark: run in release with --ignored --nocapture"]
    fn raw_screen_microbenchmark() {
        use std::time::Instant;
        const N: usize = 300_000;
        const DIM: usize = 256;
        const WORDS: usize = DIM / 64;
        const DEG: usize = 64;
        const EXPANSIONS: usize = 20_000;
        let mut rng = SplitMix(3);
        let codes: Vec<i8> = (0..N * DIM)
            .map(|_| (rng.next_u64() & 0xff) as i8)
            .collect();
        let screen: Vec<u64> = (0..N * (WORDS + 1)).map(|_| rng.next_u64()).collect();
        let q: Vec<i8> = (0..DIM).map(|_| (rng.next_u64() & 0xff) as i8).collect();
        let x: Vec<f32> = (0..DIM).map(|_| rng.next_f32() - 0.5).collect();
        let mut qc = QueryCode::new(WORDS);
        qc.encode(&x);
        let ids: Vec<u32> = (0..EXPANSIONS * DEG)
            .map(|_| (rng.next_u64() % N as u64) as u32)
            .collect();
        let code = |v: u32| &codes[v as usize * DIM..(v as usize + 1) * DIM];
        let srow = |v: u32| &screen[v as usize * (WORDS + 1)..(v as usize + 1) * (WORDS + 1)];
        let time = |f: &dyn Fn(&[u32]) -> i64| -> (f64, i64) {
            let mut best = f64::INFINITY;
            let mut acc = 0i64;
            for _ in 0..5 {
                let t = Instant::now();
                for batch in ids.as_chunks::<DEG>().0 {
                    acc = acc.wrapping_add(f(batch));
                }
                best = best.min(t.elapsed().as_secs_f64() * 1e9 / ids.len() as f64);
            }
            (best, acc)
        };
        let est = |batch: &[u32]| -> i64 {
            batch.iter().for_each(|&v| prefetch_bytes(code(v)));
            batch
                .iter()
                .map(|&v| i64::from(dot_i8_inline(&q, code(v))))
                .sum()
        };
        let scalar = |batch: &[u32]| -> i64 {
            batch.iter().for_each(|&v| prefetch(srow(v)));
            batch
                .iter()
                .map(|&v| i64::from(raw_one_scalar(&srow(v)[..WORDS], &qc.planes, WORDS)))
                .sum()
        };
        let neon_row = |batch: &[u32]| -> i64 {
            batch.iter().for_each(|&v| prefetch(srow(v)));
            batch
                .iter()
                .map(|&v| i64::from(raw_one(&srow(v)[..WORDS], &qc.planes, WORDS)))
                .sum()
        };
        let mut gathered = vec![0u64; DEG * WORDS];
        let mut out = vec![0u32; DEG];
        let neon_batch = |batch: &[u32]| -> i64 {
            // SAFETY-free gather: copy the 64 rows into one contiguous block, then one
            // `raw_estimates` pass (the FastScan-style form).
            let mut g = gathered.clone();
            let mut o = out.clone();
            batch.iter().for_each(|&v| prefetch(srow(v)));
            for (i, &v) in batch.iter().enumerate() {
                g[i * WORDS..(i + 1) * WORDS].copy_from_slice(&srow(v)[..WORDS]);
            }
            raw_estimates(&g, &qc.planes, WORDS, &mut o);
            o.iter().map(|&r| i64::from(r)).sum()
        };
        // Loop models: screen every fresh neighbour, then estimate the survivors (about 24%
        // on nytimes, held-out: 76% skipped). `late` prefetches a survivor's code when it
        // survives (the committed loop); `early` prefetches every code with the screen rows.
        let mut sample: Vec<u32> = ids[..4096]
            .iter()
            .map(|&v| raw_one(&srow(v)[..WORDS], &qc.planes, WORDS))
            .collect();
        sample.sort_unstable();
        let cut = sample[sample.len() * 24 / 100];
        let survive = |v: u32| raw_one(&srow(v)[..WORDS], &qc.planes, WORDS) < cut;
        let late = |batch: &[u32]| -> i64 {
            batch.iter().for_each(|&v| prefetch(srow(v)));
            let mut acc = 0i64;
            for &v in batch {
                if survive(v) {
                    prefetch_bytes(code(v));
                    acc += i64::from(dot_i8_inline(&q, code(v)));
                }
            }
            acc
        };
        let late_two_pass = |batch: &[u32]| -> i64 {
            batch.iter().for_each(|&v| prefetch(srow(v)));
            let survivors: Vec<u32> = batch.iter().copied().filter(|&v| survive(v)).collect();
            survivors.iter().for_each(|&v| prefetch_bytes(code(v)));
            survivors
                .iter()
                .map(|&v| i64::from(dot_i8_inline(&q, code(v))))
                .sum()
        };
        let early = |batch: &[u32]| -> i64 {
            batch.iter().for_each(|&v| {
                prefetch(srow(v));
                prefetch_bytes(code(v));
            });
            batch
                .iter()
                .filter(|&&v| survive(v))
                .map(|&v| i64::from(dot_i8_inline(&q, code(v))))
                .sum()
        };
        let (t_est, a) = time(&est);
        let (t_scalar, b) = time(&scalar);
        let (t_neon, c) = time(&neon_row);
        let (t_batch, d) = time(&neon_batch);
        let (t_late, e) = time(&late);
        let (t_late2, e2) = time(&late_two_pass);
        let (t_early, f) = time(&early);
        gathered.clear();
        out.clear();
        assert_eq!(c, d);
        assert_eq!(e, f);
        assert_eq!(e, e2);
        let survivors = ids.iter().filter(|&&v| survive(v)).count() as f64 / ids.len() as f64;
        let report = format!(
            "{{\"n_rows\": {N}, \"dim\": {DIM}, \"degree\": {DEG}, \"expansions\": {EXPANSIONS}, \
             \"ns_per_neighbour\": {{\"est_8bit_sdot\": {t_est:.2}, \"screen_scalar\": {t_scalar:.2}, \
             \"screen_neon_row\": {t_neon:.2}, \"screen_neon_gathered_batch\": {t_batch:.2}}}, \
             \"ratio_to_est\": {{\"screen_scalar\": {:.3}, \"screen_neon_row\": {:.3}, \
             \"screen_neon_gathered_batch\": {:.3}}}, \"kill_rule\": \"screen < 0.25 of est\", \
             \"loop_models_ns_per_fresh_neighbour\": {{\"estimate_all\": {t_est:.2}, \
             \"screen_then_late_prefetch\": {t_late:.2}, \"screen_then_two_pass_prefetch\": {t_late2:.2}, \
             \"screen_with_early_prefetch\": {t_early:.2}}}, \"survivor_share\": {survivors:.3}, \
             \"checksums\": [{a}, {b}, {c}, {e}]}}",
            t_scalar / t_est,
            t_neon / t_est,
            t_batch / t_est
        );
        println!("{report}");
        if let Ok(path) = std::env::var("HMS_SCREEN_BENCH_OUT") {
            std::fs::write(path, format!("{report}\n")).unwrap();
        }
    }

    type I8Kernel = fn(&[i8], &[i8]) -> i32;

    /// Every integer kernel this CPU can run, by name: `(name, i8 dot, i4 dot)`.
    fn int_variants() -> Vec<(&'static str, I8Kernel, I8Kernel)> {
        // SAFETY (every unsafe kernel below): its target features are detected before it is
        // listed, and every caller passes rows of the lengths the kernel requires.
        let mut v: Vec<(&'static str, I8Kernel, I8Kernel)> =
            vec![("scalar", dot_i8_scalar, dot_i4_scalar)];
        #[cfg(target_arch = "aarch64")]
        {
            v.push(("neon_widening", neon::dot_i8, neon::dot_i4));
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                v.push(("sdot", neon::dot_i8_sdot, |q, p| unsafe {
                    neon::sdot4(q, p)
                }));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                v.push((
                    "avx2",
                    |a, b| unsafe { x86::dot_i8_avx2(a, b) },
                    |q, p| unsafe { x86::dot_i4_avx2(q, p) },
                ));
            }
            if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") {
                v.push((
                    "vnni",
                    |a, b| unsafe { x86::dot_i8_vnni(a, b) },
                    |q, p| unsafe { x86::dot_i4_vnni(q, p) },
                ));
            }
        }
        v
    }

    fn random_i8(rng: &mut SplitMix, n: usize) -> Vec<i8> {
        (0..n).map(|_| rng.next_u64() as i8).collect()
    }

    /// Packs signed nibbles (`x >> 4` of each byte) two per byte as [`dot_f32_i4`] reads them.
    fn pack_nibbles(bytes: &[i8]) -> Vec<i8> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| (((p[0] >> 4) as u8 & 0x0F) | (((p[1] >> 4) as u8) << 4)) as i8)
            .collect()
    }

    /// The tail, odd-dimension and extreme-value cases every integer kernel must match the
    /// scalar reference on, bit for bit (lengths are the multiples of 16 the index produces).
    #[test]
    fn every_integer_kernel_matches_scalar() {
        let mut rng = SplitMix(99);
        for (name, i8k, i4k) in int_variants() {
            for len in (16..=512).step_by(16) {
                for round in 0..24 {
                    let mut a = random_i8(&mut rng, len);
                    let mut b = random_i8(&mut rng, len);
                    match round {
                        0 => {
                            a.fill(-128);
                            b.fill(-128);
                        }
                        1 => {
                            a.fill(127);
                            b.fill(-128);
                        }
                        2 => {
                            a.fill(-128);
                            b.fill(127);
                        }
                        3 => {
                            a.fill(0);
                            b.fill(-128);
                        }
                        _ => {}
                    }
                    assert_eq!(i8k(&a, &b), dot_i8_scalar(&a, &b), "{name} i8 len {len}");
                    if len % 32 == 0 {
                        let mut packed = pack_nibbles(&b);
                        if round == 1 {
                            packed.fill(-128); // both nibbles -8
                        } else if round == 2 {
                            packed.fill(0x77); // both nibbles 7
                        }
                        let mut q = vec![0i8; len];
                        interleave_for_i4(&a, &mut q);
                        assert_eq!(
                            i4k(&q, &packed),
                            dot_i4_scalar(&q, &packed),
                            "{name} i4 len {len}"
                        );
                    }
                }
            }
        }
    }

    /// Kill test for every distance kernel: ns per call with the search loop's access
    /// pattern (random fresh rows, prefetched in batches of 64) on a working set far larger
    /// than cache (`cold`), and over rows that stay in L1 (`hot`, the compute bound), for
    /// the row widths the indexes use. A kernel that does not beat the incumbent is dropped.
    /// `cargo test --release dot_kernel_microbench -- --ignored --nocapture`; writes JSON to
    /// `$HMS_PLATFORM_BENCH_OUT` when set.
    #[test]
    #[ignore = "microbenchmark: run in release with --ignored --nocapture"]
    fn dot_kernel_microbench() {
        use std::time::Instant;
        const COLD_ROWS: usize = 400_000;
        const CALLS: usize = 1 << 20;
        let mut rng = SplitMix(3);
        let mut report = Vec::new();
        let mut push = |dim: usize, kind: &str, name: &str, cold: f64, hot: f64| {
            report.push(format!(
                "{{\"dim\": {dim}, \"kernel\": \"{kind}\", \"variant\": \"{name}\", \"ns_cold\": {cold:.3}, \"ns_hot\": {hot:.3}}}"
            ));
        };
        for dim in [128usize, 256] {
            let cold = random_i8(&mut rng, COLD_ROWS * dim);
            let ids: Vec<u32> = (0..CALLS)
                .map(|_| (rng.next_u64() % COLD_ROWS as u64) as u32)
                .collect();
            let hot = random_i8(&mut rng, 64 * dim);
            let q = random_i8(&mut rng, dim);
            let mut qi = vec![0i8; dim];
            interleave_for_i4(&q, &mut qi);
            let cold4 = pack_nibbles(&cold);
            let hot4 = pack_nibbles(&hot);
            for (kind, width) in [("i8", dim), ("i4", dim / 2)] {
                let (rows_cold, rows_hot): (&[i8], &[i8]) = if kind == "i8" {
                    (&cold, &hot)
                } else {
                    (&cold4, &hot4)
                };
                let query: &[i8] = if kind == "i8" { &q } else { &qi };
                let mut ref_sum = None;
                for (name, i8k, i4k) in int_variants() {
                    let k = if kind == "i8" { i8k } else { i4k };
                    let mut best = (f64::INFINITY, f64::INFINITY);
                    let mut sum = 0i64;
                    for _ in 0..7 {
                        let t = Instant::now();
                        let mut s = 0i64;
                        for batch in ids.chunks(64) {
                            for &v in batch {
                                prefetch_bytes(&rows_cold[v as usize * width..][..width]);
                            }
                            for &v in batch {
                                s += i64::from(k(query, &rows_cold[v as usize * width..][..width]));
                            }
                        }
                        best.0 = best.0.min(t.elapsed().as_secs_f64() * 1e9 / CALLS as f64);
                        sum = s;
                        let t = Instant::now();
                        let mut s = 0i64;
                        for _ in 0..CALLS / 64 {
                            for r in rows_hot.chunks_exact(width) {
                                s += i64::from(k(query, r));
                            }
                        }
                        best.1 = best.1.min(t.elapsed().as_secs_f64() * 1e9 / CALLS as f64);
                        std::hint::black_box(s);
                    }
                    assert_eq!(*ref_sum.get_or_insert(sum), sum, "{name} {kind} dim {dim}");
                    push(dim, kind, name, best.0, best.1);
                }
            }
        }
        let json = format!(
            "{{\"arch\": \"{}\", \"results\": [\n  {}\n]}}\n",
            std::env::consts::ARCH,
            report.join(",\n  ").replace("NaN", "null")
        );
        println!("{json}");
        if let Ok(path) = std::env::var("HMS_PLATFORM_BENCH_OUT") {
            std::fs::write(path, json).unwrap();
        }
    }
}
