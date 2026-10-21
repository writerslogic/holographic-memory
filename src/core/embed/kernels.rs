// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Numeric kernels. Every output element is produced by one thread with a fixed reduction
//! order that does not depend on the batch, the thread count or the SIMD width the compiler
//! picks (no reassociation), so results are bit-identical across runs, batch compositions and
//! thread counts on one build. Transcendentals come from `libm`, not the platform.

use rayon::prelude::*;

/// Storage of a weight matrix of shape `[rows, cols]` (row-major, `rows` output features).
#[derive(Clone, Debug)]
pub(crate) enum Weights {
    F32(Vec<f32>),
    /// IEEE half precision bit patterns.
    F16(Vec<u16>),
    /// Symmetric int8 with one f32 scale per `group` consecutive values of a row (`group`
    /// divides the row length). Held as raw bytes and read as `i8`.
    I8 {
        q: Vec<u8>,
        scale: Vec<f32>,
        group: usize,
    },
}

impl Weights {
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Self::F32(w) => w.len() * 4,
            Self::F16(w) => w.len() * 2,
            Self::I8 { q, scale, .. } => q.len() + scale.len() * 4,
        }
    }

    /// Writes row `r` (`cols` values) into `out` as f32.
    pub(crate) fn row_into(&self, r: usize, cols: usize, out: &mut [f32]) {
        let at = r * cols;
        match self {
            Self::F32(w) => out.copy_from_slice(&w[at..at + cols]),
            Self::F16(w) => {
                for (o, &h) in out.iter_mut().zip(&w[at..at + cols]) {
                    *o = f16_to_f32(h);
                }
            }
            Self::I8 { q, scale, group } => {
                let groups = cols / group;
                for (g, (o, qs)) in out
                    .chunks_exact_mut(*group)
                    .zip(q[at..at + cols].chunks_exact(*group))
                    .enumerate()
                {
                    let s = scale[r * groups + g];
                    for (o, &b) in o.iter_mut().zip(qs) {
                        *o = f32::from(b as i8) * s;
                    }
                }
            }
        }
    }
}

/// Exact conversion of an IEEE 754 binary16 bit pattern to f32.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal half: normalize into an f32.
            let shift = frac.leading_zeros() - 21;
            let frac = (frac << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (frac << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// Round-to-nearest-even conversion of f32 to a binary16 bit pattern (overflow saturates to
/// infinity). Used by the pack writer.
pub(crate) fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = frac | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        if rem > mid || (rem == mid && half & 1 == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = sign | ((e as u16) << 10) | (frac >> 13) as u16;
    let rem = frac & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    half
}

const LANES: usize = 16;

/// Dot product with 16 independent accumulators and a fixed pairwise reduction.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; LANES];
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..LANES {
            acc[i] = fma(x[i], y[i], acc[i]);
        }
    }
    let mut tail = 0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    let mut w = LANES / 2;
    while w > 0 {
        for i in 0..w {
            acc[i] += acc[i + w];
        }
        w /= 2;
    }
    acc[0] + tail
}

/// Dot product of short vectors (attention heads): four lane partials, `(l0 + l2) + (l1 + l3)`,
/// then the scalar tail.
#[inline(always)]
pub(crate) fn dot_head(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 4];
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..4 {
            acc[l] = fma(x[l], y[l], acc[l]);
        }
    }
    let mut tail = 0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    ((acc[0] + acc[2]) + (acc[1] + acc[3])) + tail
}

/// `y = x * W^T + bias` for `x: [m, k]`, `W: [n, k]`, `y: [m, n]`.
#[derive(Clone, Debug)]
pub(crate) struct Linear {
    pub(crate) w: Weights,
    pub(crate) bias: Vec<f32>,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

const FEATURES_PER_TASK: usize = 16;
const LANE: usize = 4;

/// `a * b + c`, fused where the target has a hardware FMA (every aarch64 CPU, x86-64 built with
/// the `fma` feature) and as separate multiply and add elsewhere, where a software fused op would
/// be far slower. Results are bit-identical across thread counts and batches on any one build;
/// across architectures they can differ in the last bits.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}
const TOKEN_TILE: usize = 4;

/// `MR` tokens x 4 features. Each output keeps four lane partial sums over `k` in index order and
/// reduces them as `(l0 + l2) + (l1 + l3)`, then adds the scalar tail: the same arithmetic for
/// every `MR`, so an output never depends on which tile it was computed in.
#[inline(always)]
fn tile<const MR: usize>(x: &[f32], k: usize, w: &[f32], out: &mut [[f32; 4]; MR]) {
    let body = k - k % LANE;
    let mut acc = [[[0f32; LANE]; 4]; MR];
    let mut kk = 0;
    while kk < body {
        let mut wv = [[0f32; LANE]; 4];
        for (j, v) in wv.iter_mut().enumerate() {
            v.copy_from_slice(&w[j * k + kk..j * k + kk + LANE]);
        }
        for (i, row) in acc.iter_mut().enumerate() {
            let mut xv = [0f32; LANE];
            xv.copy_from_slice(&x[i * k + kk..i * k + kk + LANE]);
            for (a, w) in row.iter_mut().zip(&wv) {
                for l in 0..LANE {
                    a[l] = fma(xv[l], w[l], a[l]);
                }
            }
        }
        kk += LANE;
    }
    for i in 0..MR {
        for j in 0..4 {
            let a = acc[i][j];
            let mut tail = 0f32;
            for t in body..k {
                tail += x[i * k + t] * w[j * k + t];
            }
            out[i][j] = ((a[0] + a[2]) + (a[1] + a[3])) + tail;
        }
    }
}

impl Linear {
    pub(crate) fn forward(&self, x: &[f32], m: usize, out: &mut [f32], scratch: &mut Vec<f32>) {
        let (n, k) = (self.n, self.k);
        debug_assert_eq!(x.len(), m * k);
        debug_assert_eq!(out.len(), m * n);
        // Feature-major scratch so each task owns a contiguous block; one transpose pass after.
        scratch.clear();
        scratch.resize(n * m, 0.0);
        let task = |(chunk, block): (usize, &mut [f32])| {
            let f0 = chunk * FEATURES_PER_TASK;
            let nf = block.len() / m;
            let padded = nf.div_ceil(4) * 4;
            // Dequantized weight rows of this task, zero rows pad the last feature block.
            let mut wbuf = vec![0f32; padded * k];
            for j in 0..nf {
                self.w.row_into(f0 + j, k, &mut wbuf[j * k..(j + 1) * k]);
            }
            for jb in (0..padded).step_by(4) {
                let w = &wbuf[jb * k..(jb + 4) * k];
                let mut mi = 0;
                while mi + TOKEN_TILE <= m {
                    let mut o = [[0f32; 4]; TOKEN_TILE];
                    tile::<TOKEN_TILE>(&x[mi * k..], k, w, &mut o);
                    for (i, r) in o.iter().enumerate() {
                        for j in 0..4.min(nf.saturating_sub(jb)) {
                            block[(jb + j) * m + mi + i] = r[j] + self.bias[f0 + jb + j];
                        }
                    }
                    mi += TOKEN_TILE;
                }
                while mi < m {
                    let mut o = [[0f32; 4]; 1];
                    tile::<1>(&x[mi * k..], k, w, &mut o);
                    for j in 0..4.min(nf.saturating_sub(jb)) {
                        block[(jb + j) * m + mi] = o[0][j] + self.bias[f0 + jb + j];
                    }
                    mi += 1;
                }
            }
        };
        if m * k * n < 1 << 14 {
            scratch
                .chunks_mut(FEATURES_PER_TASK * m)
                .enumerate()
                .for_each(task);
        } else {
            scratch
                .par_chunks_mut(FEATURES_PER_TASK * m)
                .enumerate()
                .for_each(task);
        }
        for mi in 0..m {
            for f in 0..n {
                out[mi * n + f] = scratch[f * m + mi];
            }
        }
    }
}

/// In-place layer normalization of each `width`-wide row of `x`, fixed accumulation order in f64.
pub(crate) fn layer_norm(x: &mut [f32], width: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(width) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
        let var = row
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / width as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for ((v, &g), &b) in row.iter_mut().zip(gamma).zip(beta) {
            *v = ((f64::from(*v) - mean) * inv) as f32 * g + b;
        }
    }
}

/// `e^x` for `x` in `[-87, 88]` (inputs outside are clamped): range reduction by `ln 2` and a
/// degree-6 polynomial, relative error below 2e-7. Plain f32 arithmetic, so it vectorizes and is
/// identical on every thread; `libm::expf` is several times slower per element.
#[inline(always)]
pub(crate) fn fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 88.0);
    let n = (x * LOG2E).round();
    let r = fma(-n, LN2_LO, fma(-n, LN2_HI, x));
    let mut p = fma(r, 1.0 / 720.0, 1.0 / 120.0);
    p = fma(r, p, 1.0 / 24.0);
    p = fma(r, p, 1.0 / 6.0);
    p = fma(r, p, 0.5);
    p = fma(r, p, 1.0);
    p = fma(r, p, 1.0);
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// GELU with the exact erf form `x * Phi(x)`; erf by Abramowitz and Stegun 7.1.26 (absolute error
/// below 1.5e-7, far under the 1e-3 the int8 weights already cost).
#[inline(always)]
pub(crate) fn gelu(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / fma(0.327_591_1, z.abs(), 1.0);
    let mut poly = fma(1.061_405_4, t, -1.453_152);
    poly = fma(poly, t, 1.421_413_7);
    poly = fma(poly, t, -0.284_496_74);
    poly = fma(poly, t, 0.254_829_6);
    let erf_abs = 1.0 - poly * t * fast_exp(-z * z);
    let erf = if z < 0.0 { -erf_abs } else { erf_abs };
    0.5 * x * (1.0 + erf)
}

/// In-place softmax of one row.
pub(crate) fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in row.iter_mut() {
        *v = fast_exp(*v - max);
    }
    let sum = row.iter().sum::<f32>();
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_every_pattern() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f32_to_f16(f) & 0x7c00 == 0x7c00 && f32_to_f16(f) & 0x3ff != 0);
            } else {
                assert_eq!(f32_to_f16(f), h, "pattern {h:#06x} -> {f}");
            }
        }
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        // 1 + 2^-11 is exactly halfway between 1.0 and 1 + 2^-10: ties to even (1.0).
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        // Just above halfway rounds up.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3c01);
        assert_eq!(f32_to_f16(1e9), 0x7c00);
    }

    #[test]
    fn linear_result_is_independent_of_batching() {
        let (n, k) = (19, 37);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7919 % 251) as f32 - 125.0) / 100.0)
            .collect();
        let lin = Linear {
            w: Weights::F32(w),
            bias: (0..n).map(|i| i as f32 * 0.01).collect(),
            n,
            k,
        };
        let x: Vec<f32> = (0..5 * k)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) / 50.0)
            .collect();
        let mut all = vec![0f32; 5 * n];
        lin.forward(&x, 5, &mut all, &mut Vec::new());
        for r in 0..5 {
            let mut one = vec![0f32; n];
            lin.forward(&x[r * k..(r + 1) * k], 1, &mut one, &mut Vec::new());
            assert_eq!(one, all[r * n..(r + 1) * n]);
        }
    }

    #[test]
    fn gelu_matches_reference_values() {
        assert!((gelu(1.0) - 0.841_344_8).abs() < 1e-6);
        for i in -40..=40 {
            let x = i as f32 * 0.25;
            let want = (x as f64) * 0.5 * (1.0 + libm::erf(x as f64 / std::f64::consts::SQRT_2));
            assert!((f64::from(gelu(x)) - want).abs() < 2e-6, "gelu({x})");
            let e = f64::from(fast_exp(-x.abs()));
            assert!(
                ((e - (-(x.abs() as f64)).exp()) / e).abs() < 5e-7,
                "exp({x})"
            );
        }
        assert!((gelu(-1.0) + 0.158_655_25).abs() < 1e-6);
        assert_eq!(gelu(0.0), 0.0);
    }
}
