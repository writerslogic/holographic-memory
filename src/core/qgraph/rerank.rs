// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Float-query kernels of the re-rank: a float row against an 8-bit row or a packed 4-bit
//! row. Each SIMD path keeps the arithmetic of the scalar kernels in `kernels`: sixteen lane
//! accumulators, each coordinate's product rounded before it is added to its lane (no fused
//! multiply-add), and the lanes summed in order at the end, so every path returns the scalar
//! kernel's bits and the re-rank order is independent of the machine.

use super::kernels;

/// `sum_i a_i * b_i` for a float row and an i8 row of the same length.
#[inline]
pub(super) fn dot_f32_i8(a: &[f32], b: &[i8]) -> f32 {
    // The SIMD paths load `a` at every offset they load `b` at.
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: `avx2` was detected at runtime.
        return unsafe { x86::dot_f32_i8(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return neon::dot_f32_i8(a, b);
    }
    #[allow(unreachable_code)]
    kernels::dot_f32_i8(a, b)
}

/// `sum_i a_i * c_i` for a float row and a row packed two signed 4-bit values per byte, the
/// even coordinate in the low nibble; `a.len() == 2 * packed.len()`.
#[inline]
pub(super) fn dot_f32_i4(a: &[f32], packed: &[i8]) -> f32 {
    // The SIMD paths load `a` at twice every offset they load `packed` at, eight bytes at a
    // time with no tail.
    assert!(a.len() == 2 * packed.len() && packed.len().is_multiple_of(8));
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: `avx2` was detected at runtime.
        return unsafe { x86::dot_f32_i4(a, packed) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return neon::dot_f32_i4(a, packed);
    }
    #[allow(unreachable_code)]
    kernels::dot_f32_i4(a, packed)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn avx2() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static AVX2: AtomicU8 = AtomicU8::new(0);
    match AVX2.load(Ordering::Relaxed) {
        0 => {
            let yes = is_x86_feature_detected!("avx2");
            AVX2.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
            yes
        }
        t => t == 2,
    }
}

/// The scalar kernels' final step: the sixteen lane sums added in lane order, then the tail.
#[inline(always)]
fn finish(lanes: [f32; 16], a: &[f32], b: &[i8]) -> f32 {
    let tail: f32 = a.iter().zip(b).map(|(x, &y)| x * f32::from(y)).sum();
    lanes.iter().sum::<f32>() + tail
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// # Safety
    /// The CPU supports AVX2; `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_f32_i8(a: &[f32], b: &[i8]) -> f32 {
        let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        // SAFETY (all loads): `i + 16 <= b.len() == a.len()`.
        while i + 16 <= b.len() {
            let y = _mm_loadu_si128(b.as_ptr().add(i).cast());
            let y0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(y));
            let y1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(y)));
            let x0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let x1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x0, y0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(x1, y1));
            i += 16;
        }
        let mut lanes = [0f32; 16];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
        _mm256_storeu_ps(lanes.as_mut_ptr().add(8), acc1);
        super::finish(lanes, &a[i..], &b[i..])
    }

    /// # Safety
    /// The CPU supports AVX2; `a.len() == 2 * p.len()`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_f32_i4(a: &[f32], p: &[i8]) -> f32 {
        let mask = _mm_set1_epi16(0x0f);
        let eight = _mm_set1_epi8(8);
        let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        // SAFETY (all loads): `i + 8 <= p.len()` and `2 * i + 16 <= a.len()`.
        while i + 8 <= p.len() {
            // Eight bytes widened to 16-bit lanes; each lane becomes [low nibble, high
            // nibble] as bytes, the coordinate order of the packing.
            let w = _mm_cvtepu8_epi16(_mm_loadl_epi64(p.as_ptr().add(i).cast()));
            let lo = _mm_and_si128(w, mask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(w), mask);
            let n = _mm_or_si128(lo, _mm_slli_epi16::<8>(hi));
            let c = _mm_sub_epi8(_mm_xor_si128(n, eight), eight);
            let y0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c));
            let y1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(c)));
            let x0 = _mm256_loadu_ps(a.as_ptr().add(2 * i));
            let x1 = _mm256_loadu_ps(a.as_ptr().add(2 * i + 8));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x0, y0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(x1, y1));
            i += 8;
        }
        let mut lanes = [0f32; 16];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
        _mm256_storeu_ps(lanes.as_mut_ptr().add(8), acc1);
        debug_assert_eq!(i, p.len(), "packed length is a multiple of 8");
        super::finish(lanes, &[], &[])
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn widen(c: int8x16_t) -> [float32x4_t; 4] {
        let lo = vmovl_s8(vget_low_s8(c));
        let hi = vmovl_high_s8(c);
        [
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
            vcvtq_f32_s32(vmovl_high_s16(lo)),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
            vcvtq_f32_s32(vmovl_high_s16(hi)),
        ]
    }

    #[inline(always)]
    unsafe fn accumulate(acc: &mut [float32x4_t; 4], a: *const f32, y: [float32x4_t; 4]) {
        for (k, (s, y)) in acc.iter_mut().zip(y).enumerate() {
            // SAFETY: the caller has 16 floats at `a`.
            let x = vld1q_f32(a.add(4 * k));
            *s = vaddq_f32(*s, vmulq_f32(x, y));
        }
    }

    pub(super) fn dot_f32_i8(a: &[f32], b: &[i8]) -> f32 {
        let mut acc = [unsafe { vdupq_n_f32(0.0) }; 4];
        let mut i = 0;
        while i + 16 <= b.len() {
            // SAFETY: `i + 16 <= b.len() == a.len()`.
            unsafe {
                let c = vld1q_s8(b.as_ptr().add(i));
                accumulate(&mut acc, a.as_ptr().add(i), widen(c));
            }
            i += 16;
        }
        let mut lanes = [0f32; 16];
        for (k, s) in acc.iter().enumerate() {
            // SAFETY: `lanes` holds 16 floats.
            unsafe { vst1q_f32(lanes.as_mut_ptr().add(4 * k), *s) };
        }
        super::finish(lanes, &a[i..], &b[i..])
    }

    pub(super) fn dot_f32_i4(a: &[f32], p: &[i8]) -> f32 {
        let mut acc = [unsafe { vdupq_n_f32(0.0) }; 4];
        let mut i = 0;
        while i + 8 <= p.len() {
            // SAFETY: `i + 8 <= p.len()` and `2 * i + 16 <= a.len()`.
            unsafe {
                let w = vld1_u8(p.as_ptr().add(i).cast());
                let lo = vand_u8(w, vdup_n_u8(0x0f));
                let hi = vshr_n_u8::<4>(w);
                // Nibbles in coordinate order, then sign-extended: (n ^ 8) - 8.
                let n = vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
                let c = vsubq_s8(
                    vreinterpretq_s8_u8(veorq_u8(n, vdupq_n_u8(8))),
                    vdupq_n_s8(8),
                );
                accumulate(&mut acc, a.as_ptr().add(2 * i), widen(c));
            }
            i += 8;
        }
        let mut lanes = [0f32; 16];
        for (k, s) in acc.iter().enumerate() {
            // SAFETY: `lanes` holds 16 floats.
            unsafe { vst1q_f32(lanes.as_mut_ptr().add(4 * k), *s) };
        }
        debug_assert_eq!(i, p.len(), "packed length is a multiple of 8");
        super::finish(lanes, &[], &[])
    }
}

#[cfg(test)]
mod tests {
    use super::super::kernels::{self, SplitMix};

    #[test]
    fn rerank_kernels_return_the_scalar_bits() {
        let mut rng = SplitMix(0x5EED_F1E1);
        for &dim in &[16, 32, 128, 160, 800, 1024] {
            for _ in 0..50 {
                let a: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
                let b: Vec<i8> = (0..dim).map(|_| (rng.next_u64() % 255) as i8).collect();
                let p: Vec<i8> = (0..dim / 2).map(|_| rng.next_u64() as i8).collect();
                assert_eq!(
                    super::dot_f32_i8(&a, &b).to_bits(),
                    kernels::dot_f32_i8(&a, &b).to_bits(),
                    "i8 dim {dim}"
                );
                assert_eq!(
                    super::dot_f32_i4(&a, &p).to_bits(),
                    kernels::dot_f32_i4(&a, &p).to_bits(),
                    "i4 dim {dim}"
                );
            }
        }
    }

    #[test]
    #[should_panic]
    fn i8_row_shorter_than_the_query_panics() {
        let _ = super::dot_f32_i8(&[1.0; 16], &[1; 32]);
    }

    #[test]
    #[should_panic]
    fn i4_row_not_half_the_query_panics() {
        let _ = super::dot_f32_i4(&[1.0; 16], &[1; 16]);
    }

    #[test]
    #[should_panic]
    fn i4_row_with_a_partial_block_panics() {
        let _ = super::dot_f32_i4(&[1.0; 20], &[1; 10]);
    }
}
