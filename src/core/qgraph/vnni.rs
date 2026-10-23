// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Traversal dot products with AVX-VNNI (`vpdpbusd`, 256-bit) in its two encodings: VEX
//! (`avx`, AVX-VNNI: Alder Lake and later client cores, Zen 5) and EVEX (`evex`, AVX512-VNNI
//! with VL: Ice Lake and later servers, Zen 4).
//!
//! `vpdpbusd` multiplies unsigned by signed bytes, so one operand is biased into the unsigned
//! range and the bias is subtracted after the sum. The bias always goes on the row, so the
//! correction is a sum over the query alone: `sum (r + 128) q = r.q + 128 sum q` for 8-bit rows
//! and `sum (c + 8) q = c.q + 8 sum q` for 4-bit rows. [`query_sum`] gives that sum once per
//! query; the kernels take it, so each row costs one `vpdpbusd` per 32 coordinates. Every
//! result is the exact integer sum (no saturation: four products of at most `255 * 128` per
//! lane and step), equal to the scalar kernels'.

/// `sum q`, the correction term of the kernels below.
#[must_use]
pub(crate) fn query_sum(q: &[i8]) -> i32 {
    q.iter().map(|&v| i32::from(v)).sum()
}

/// The kernels for one encoding of `vpdpbusd`: `$features` are the target features that
/// enable it, `$dp256` and `$dp128` its 256- and 128-bit intrinsics.
#[cfg(target_arch = "x86_64")]
macro_rules! kernels {
    ($name:ident, $features:literal, $dp256:ident, $dp128:ident) => {
        pub(crate) mod $name {
            use std::arch::x86_64::*;

            /// `a.b` for a query `a` with `qsum == query_sum(a)` and an 8-bit row `b`.
            ///
            /// # Safety
            /// The CPU supports the module's features; `a` and `b` have the same length, a multiple of 16.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn dot_i8(a: &[i8], b: &[i8], qsum: i32) -> i32 {
                let flip = _mm256_set1_epi8(-128);
                let mut acc = _mm256_setzero_si256();
                let mut i = 0;
                // SAFETY (all loads): `i + 32 <= len` in the loop, `i + 16 <= len` for the tail.
                while i + 32 <= a.len() {
                    let r = _mm256_xor_si256(_mm256_loadu_si256(b.as_ptr().add(i).cast()), flip);
                    let q = _mm256_loadu_si256(a.as_ptr().add(i).cast());
                    acc = $dp256(acc, r, q);
                    i += 32;
                }
                let mut total = hsum256(acc);
                if i < a.len() {
                    let r = _mm_xor_si128(
                        _mm_loadu_si128(b.as_ptr().add(i).cast()),
                        _mm_set1_epi8(-128),
                    );
                    let q = _mm_loadu_si128(a.as_ptr().add(i).cast());
                    total = total.wrapping_add(hsum128($dp128(_mm_setzero_si128(), r, q)));
                }
                total.wrapping_sub(128i32.wrapping_mul(qsum))
            }

            /// The two nibble planes of 16 packed bytes at `p`, each biased to `signed nibble + 8`.
            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn nibbles128(x: __m128i) -> (__m128i, __m128i) {
                let (mask, eight) = (_mm_set1_epi8(0x0f), _mm_set1_epi8(8));
                (
                    _mm_xor_si128(_mm_and_si128(x, mask), eight),
                    _mm_xor_si128(_mm_and_si128(_mm_srli_epi16::<4>(x), mask), eight),
                )
            }

            /// The same for 32 packed bytes (two 16-byte groups, one per 128-bit lane).
            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn nibbles256(x: __m256i) -> (__m256i, __m256i) {
                let (mask, eight) = (_mm256_set1_epi8(0x0f), _mm256_set1_epi8(8));
                (
                    _mm256_xor_si256(_mm256_and_si256(x, mask), eight),
                    _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(x), mask), eight),
                )
            }

            /// The even and odd query halves of two consecutive 16-byte packed groups starting at packed
            /// offset `i`: group `g` reads its 16 even coordinates at `q[2 i + 32 g..]` and its 16 odd
            /// ones 16 bytes later (the [`super::super::kernels::interleave_for_i4`] layout).
            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn query_pair(q: &[i8], i: usize) -> (__m256i, __m256i) {
                let at = |o: usize| _mm_loadu_si128(q.as_ptr().add(2 * i + o).cast());
                (
                    _mm256_set_m128i(at(32), at(0)),
                    _mm256_set_m128i(at(48), at(16)),
                )
            }

            /// `q.c` for a query laid out by [`super::super::kernels::interleave_for_i4`] with
            /// `qsum == query_sum(q)` and a packed 4-bit row `p`.
            ///
            /// # Safety
            /// The CPU supports the module's features; `q.len() == 2 * p.len()` and `p.len()` is a multiple of
            /// 16.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn dot_i4(q: &[i8], p: &[i8], qsum: i32) -> i32 {
                let mut acc = _mm256_setzero_si256();
                let mut i = 0;
                // SAFETY (all loads): `i + 32 <= p.len()` in the loop and `2 * i + 64 <= q.len()`;
                // `i + 16 <= p.len()` and `2 * i + 32 <= q.len()` for the tail.
                while i + 32 <= p.len() {
                    let (lo, hi) = nibbles256(_mm256_loadu_si256(p.as_ptr().add(i).cast()));
                    let (qe, qo) = query_pair(q, i);
                    acc = $dp256(acc, lo, qe);
                    acc = $dp256(acc, hi, qo);
                    i += 32;
                }
                let mut total = hsum256(acc);
                if i < p.len() {
                    let (lo, hi) = nibbles128(_mm_loadu_si128(p.as_ptr().add(i).cast()));
                    let qe = _mm_loadu_si128(q.as_ptr().add(2 * i).cast());
                    let qo = _mm_loadu_si128(q.as_ptr().add(2 * i + 16).cast());
                    let t = $dp128(_mm_setzero_si128(), lo, qe);
                    total = total.wrapping_add(hsum128($dp128(t, hi, qo)));
                }
                total.wrapping_sub(8i32.wrapping_mul(qsum))
            }

            /// [`dot_i4`] of one query against four rows of one length, sharing the query loads.
            ///
            /// # Safety
            /// As [`dot_i4`], for every row.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn dot_i4_x4(q: &[i8], rows: [&[i8]; 4], qsum: i32) -> [i32; 4] {
                let len = rows[0].len();
                let mut acc = [_mm256_setzero_si256(); 4];
                let mut i = 0;
                // SAFETY (all loads): as in `dot_i4`, every row having length `len`.
                while i + 32 <= len {
                    let (qe, qo) = query_pair(q, i);
                    for (a, row) in acc.iter_mut().zip(rows) {
                        let (lo, hi) = nibbles256(_mm256_loadu_si256(row.as_ptr().add(i).cast()));
                        *a = $dp256(*a, lo, qe);
                        *a = $dp256(*a, hi, qo);
                    }
                    i += 32;
                }
                let mut out = acc.map(|a| hsum256(a));
                if i < len {
                    let qe = _mm_loadu_si128(q.as_ptr().add(2 * i).cast());
                    let qo = _mm_loadu_si128(q.as_ptr().add(2 * i + 16).cast());
                    for (o, row) in out.iter_mut().zip(rows) {
                        let (lo, hi) = nibbles128(_mm_loadu_si128(row.as_ptr().add(i).cast()));
                        let t = $dp128(_mm_setzero_si128(), lo, qe);
                        *o = o.wrapping_add(hsum128($dp128(t, hi, qo)));
                    }
                }
                out.map(|o| o.wrapping_sub(8i32.wrapping_mul(qsum)))
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn hsum128(s: __m128i) -> i32 {
                let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
                _mm_cvtsi128_si32(s)
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn hsum256(v: __m256i) -> i32 {
                hsum128(_mm_add_epi32(
                    _mm256_castsi256_si128(v),
                    _mm256_extracti128_si256::<1>(v),
                ))
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
kernels!(
    avx,
    "avx2,avxvnni",
    _mm256_dpbusd_avx_epi32,
    _mm_dpbusd_avx_epi32
);
#[cfg(target_arch = "x86_64")]
kernels!(
    evex,
    "avx2,avx512f,avx512vl,avx512vnni",
    _mm256_dpbusd_epi32,
    _mm_dpbusd_epi32
);

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::core::qgraph::kernels::{interleave_for_i4, SplitMix};

    type Kernels = (
        unsafe fn(&[i8], &[i8], i32) -> i32,
        unsafe fn(&[i8], &[i8], i32) -> i32,
        unsafe fn(&[i8], [&[i8]; 4], i32) -> [i32; 4],
    );

    /// The encodings this CPU runs.
    fn available() -> Vec<(&'static str, Kernels)> {
        let mut out: Vec<(&'static str, Kernels)> = Vec::new();
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avxvnni") {
            out.push(("avx", (avx::dot_i8, avx::dot_i4, avx::dot_i4_x4)));
        }
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni")
        {
            out.push(("evex", (evex::dot_i8, evex::dot_i4, evex::dot_i4_x4)));
        }
        out
    }

    fn scalar_i8(a: &[i8], b: &[i8]) -> i32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| i32::from(x) * i32::from(y))
            .sum()
    }

    /// Signed nibble `j` of a row packed two per byte (even coordinate low).
    fn nibble(p: &[i8], j: usize) -> i32 {
        let byte = p[j / 2] as u8;
        let n = if j.is_multiple_of(2) {
            byte & 15
        } else {
            byte >> 4
        };
        i32::from(((n ^ 8) as i8) - 8)
    }

    fn rows(rng: &mut SplitMix, len: usize, extreme: bool) -> Vec<i8> {
        (0..len)
            .map(|i| match (extreme, i % 2) {
                (true, 0) => -128,
                (true, _) => 127,
                _ => rng.next_u64() as i8,
            })
            .collect()
    }

    #[test]
    fn avxvnni_kernels_equal_the_scalar_sums_at_every_width() {
        let kernels = available();
        if kernels.is_empty() {
            eprintln!("neither VNNI encoding available; skipped");
            return;
        }
        for (name, (dot_i8, dot_i4, dot_i4_x4)) in kernels {
            let mut rng = SplitMix(41);
            for len in (0..=512).step_by(16) {
                for extreme in [false, true] {
                    // 8-bit rows: every length that is a multiple of 16, including 0.
                    let a = rows(&mut rng, len, extreme);
                    let b = rows(&mut rng, len, !extreme);
                    // SAFETY: the encoding's features were detected; equal lengths, multiples of 16.
                    let got = unsafe { dot_i8(&a, &b, query_sum(&a)) };
                    assert_eq!(got, scalar_i8(&a, &b), "{name} i8 len {len}");
                    // 4-bit rows of `len` packed bytes against a `2 len` query.
                    let q = rows(&mut rng, 2 * len, extreme);
                    let mut ql = vec![0i8; 2 * len];
                    interleave_for_i4(&q, &mut ql);
                    let ps: Vec<Vec<i8>> = (0..4).map(|_| rows(&mut rng, len, !extreme)).collect();
                    let want: Vec<i32> = ps
                        .iter()
                        .map(|p| (0..2 * len).map(|j| i32::from(q[j]) * nibble(p, j)).sum())
                        .collect();
                    let qs = query_sum(&ql);
                    for (p, &w) in ps.iter().zip(&want) {
                        // SAFETY: as above; `ql.len() == 2 * p.len()`.
                        assert_eq!(unsafe { dot_i4(&ql, p, qs) }, w, "{name} i4 len {len}");
                    }
                    let four = [&ps[0][..], &ps[1][..], &ps[2][..], &ps[3][..]];
                    // SAFETY: as above, every row of length `len`.
                    assert_eq!(
                        unsafe { dot_i4_x4(&ql, four, qs) }.to_vec(),
                        want,
                        "{name} x4 len {len}"
                    );
                }
            }
        }
    }

    /// Development microbenchmark (`--ignored`): ns per row of the AVX2 and VNNI kernels over
    /// 4,096 rows (L2-resident) at the traversal widths of the ann-benchmarks sets.
    #[test]
    #[ignore]
    fn microbench_dot_kernels() {
        use crate::core::qgraph::kernels::{
            dot_i4_inline, dot_i4_query, dot_i8_inline, dot_i8_query,
        };
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = SplitMix(7);
        for dim in [128usize, 112, 784, 800] {
            let n = 4096;
            let q = rows(&mut rng, dim, false);
            let data = rows(&mut rng, n * dim, false);
            let qs = query_sum(&q);
            let time = |f: &dyn Fn(&[i8]) -> i32, bytes: usize| {
                let mut best = f64::INFINITY;
                for _ in 0..7 {
                    let t = Instant::now();
                    let mut acc = 0i32;
                    for r in data.chunks_exact(dim).take(n) {
                        acc = acc.wrapping_add(f(black_box(&r[..bytes])));
                    }
                    black_box(acc);
                    best = best.min(t.elapsed().as_nanos() as f64 / n as f64);
                }
                best
            };
            let i8_plain = time(&|r| dot_i8_inline(&q, r), dim);
            let i8_query = time(&|r| dot_i8_query(&q, r, qs), dim);
            let (i4_plain, i4_query) = if dim.is_multiple_of(32) {
                (
                    time(&|r| dot_i4_inline(&q, r), dim / 2),
                    time(&|r| dot_i4_query(&q, r, qs), dim / 2),
                )
            } else {
                (f64::NAN, f64::NAN)
            };
            println!(
                "{{\"dim\": {dim}, \"i8_inline_ns\": {i8_plain:.2}, \"i8_query_ns\": {i8_query:.2}, \
                 \"i4_inline_ns\": {i4_plain:.2}, \"i4_query_ns\": {i4_query:.2}}}"
            );
        }
    }
}
