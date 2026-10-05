// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! AVX2 block intersection of sorted, deduplicated `u32` index sets.

use std::arch::x86_64::*;

/// Intersection count, using AVX2 when the CPU supports it.
#[inline]
pub fn simd_intersection_count(a: &[u32], b: &[u32]) -> Option<usize> {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 && b.len() >= 8 {
        // SAFETY: AVX2 support was just verified at runtime.
        return Some(unsafe { avx2_intersection_count(a, b) });
    }
    None
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_intersection_count(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;

    // Process blocks of 8 u32s
    while i + 8 <= a.len() && j + 8 <= b.len() {
        let max_a = a[i + 7];
        let max_b = b[j + 7];

        if max_a < b[j] {
            i += 8;
        } else if max_b < a[i] {
            j += 8;
        } else {
            let va = _mm256_loadu_si256(a[i..].as_ptr() as *const __m256i);
            let mut v_match = _mm256_setzero_si256();
            for k in 0..8 {
                let vb_elem = _mm256_set1_epi32(b[j + k] as i32);
                let cmp = _mm256_cmpeq_epi32(va, vb_elem);
                v_match = _mm256_or_si256(v_match, cmp);
            }
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(v_match));
            count += mask.count_ones() as usize;

            if max_a <= max_b {
                i += 8;
            }
            if max_b <= max_a {
                j += 8;
            }
        }
    }

    // Tail scalar loop
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            count += 1;
            i += 1;
            j += 1;
        }
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::entangled::EntangledHVec;

    fn scalar(a: &[u32], b: &[u32]) -> usize {
        a.iter().filter(|x| b.binary_search(x).is_ok()).count()
    }

    #[test]
    fn matches_scalar_count_across_sizes_and_overlaps() {
        for (dim, seed) in [(2048usize, 1u64), (16384, 2), (65536, 3)] {
            let a = EntangledHVec::new_deterministic(dim, seed);
            let b = EntangledHVec::new_deterministic(dim, seed + 100);
            let union = EntangledHVec::bundle(&[a.clone(), b.clone()]);
            let dense = EntangledHVec::new_with_density(dim, 8, seed);
            for (x, y) in [
                (&a, &b),
                (&a, &union),
                (&union, &b),
                (&a, &a),
                (&dense, &a),
                (&dense, &union),
            ] {
                if let Some(count) = simd_intersection_count(x.indices(), y.indices()) {
                    assert_eq!(count, scalar(x.indices(), y.indices()));
                }
            }
        }
    }
}
