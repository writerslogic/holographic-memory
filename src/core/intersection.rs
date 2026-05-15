//! Sparse intersection primitives.
//!
//! Galloping intersection for skewed sizes, SIMD-assisted sorted merge for balanced.

/// Hamming distance between two sorted sparse bit-index slices.
///
/// Uses the adaptive intersection kernel underneath, so callers get the best
/// available path without duplicating the symmetric-difference arithmetic.
#[inline]
pub fn sparse_hamming_distance(a: &[u32], b: &[u32]) -> u32 {
    let intersection = sparse_intersection_count(a, b);
    (a.len() + b.len() - 2 * intersection) as u32
}

/// Count common elements in two sorted u32 slices.
///
/// Adaptively picks the best algorithm:
/// - When one slice is much smaller (|a| * 8 < |b|), uses galloping search
///   which is O(|small| * log(|large|)) and beats O(|a| + |b|) merge.
/// - Otherwise uses a sorted merge at O(|a| + |b|), with runtime SIMD
///   dispatch on x86/x86_64 when AVX2 is available.
#[inline]
pub fn sparse_intersection_count(a: &[u32], b: &[u32]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    // Ensure a is the smaller slice for the skew check.
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if small.len() * 8 < large.len() {
        galloping_intersection_count(small, large)
    } else {
        merge_intersection_count(small, large)
    }
}

/// Sorted merge intersection count: O(|a| + |b|).
#[inline]
fn merge_intersection_count(a: &[u32], b: &[u32]) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection above.
            return unsafe { merge_intersection_count_avx2(a, b) };
        }
    }

    merge_intersection_count_scalar(a, b)
}

#[inline]
fn merge_intersection_count_scalar(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn merge_intersection_count_avx2(a: &[u32], b: &[u32]) -> usize {
    use std::arch::x86_64::{
        __m256i, _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_loadu_si256,
        _mm256_movemask_ps, _mm256_set1_epi32,
    };

    let mut count = 0;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        let av = a[i];
        let bv = b[j];

        if av == bv {
            count += 1;
            i += 1;
            j += 1;
        } else if av < bv {
            if i + 8 <= a.len() && a[i + 7] < bv {
                i += 8;
            } else {
                i += 1;
            }
        } else if j + 8 <= b.len() {
            let needle = _mm256_set1_epi32(av as i32);
            let chunk = _mm256_loadu_si256(b[j..].as_ptr() as *const __m256i);
            let eq = _mm256_cmpeq_epi32(needle, chunk);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(eq));
            if mask != 0 {
                count += 1;
                let offset = mask.trailing_zeros() as usize;
                i += 1;
                j += offset + 1;
            } else if b[j + 7] < av {
                j += 8;
            } else {
                j += 1;
            }
        } else {
            j += 1;
        }
    }

    count
}

/// Galloping (exponential search) intersection: O(|small| * log(|large|)).
///
/// For each element in `small`, does an exponential search in `large` to find
/// the matching position. Maintains a cursor into `large` so total work on
/// the large array is bounded by O(|small| * log(|large|/|small|)).
#[inline]
fn galloping_intersection_count(small: &[u32], large: &[u32]) -> usize {
    let mut count = 0;
    let mut lo = 0; // cursor into large; advances monotonically
    for &val in small {
        // Skip elements in large that are less than val using exponential search.
        // First, find a bound by doubling the step.
        let mut step = 1;
        while lo + step < large.len() && large[lo + step] < val {
            step *= 2;
        }
        // Binary search within [lo, min(lo+step, len))
        let hi = (lo + step).min(large.len());
        // Find first position >= val
        lo = binary_search_left(&large[lo..hi], val) + lo;
        if lo < large.len() && large[lo] == val {
            count += 1;
            lo += 1; // move past this match
        }
    }
    count
}

/// Binary search for the leftmost position where `slice[pos] >= target`.
#[inline]
fn binary_search_left(slice: &[u32], target: u32) -> usize {
    let mut lo = 0;
    let mut hi = slice.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if slice[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_intersection() {
        let a = vec![1, 3, 5, 7, 9];
        let b = vec![2, 3, 6, 7, 10];
        assert_eq!(sparse_intersection_count(&a, &b), 2);
    }

    #[test]
    fn test_sparse_intersection_empty() {
        let a: Vec<u32> = vec![];
        let b = vec![1, 2, 3];
        assert_eq!(sparse_intersection_count(&a, &b), 0);
    }

    #[test]
    fn test_galloping_skewed() {
        // Small vs large: should trigger galloping path
        let small = vec![5, 50, 500];
        let large: Vec<u32> = (0..1000).collect();
        assert_eq!(sparse_intersection_count(&small, &large), 3);
    }

    #[test]
    fn test_galloping_no_overlap() {
        let small = vec![1001, 1002, 1003];
        let large: Vec<u32> = (0..1000).collect();
        assert_eq!(sparse_intersection_count(&small, &large), 0);
    }

    #[test]
    fn test_galloping_all_overlap() {
        let small = vec![0, 1, 2, 3, 4];
        let large: Vec<u32> = (0..1000).collect();
        assert_eq!(sparse_intersection_count(&small, &large), 5);
    }

    #[test]
    fn test_merge_balanced() {
        let a: Vec<u32> = (0..100).step_by(2).collect(); // evens
        let b: Vec<u32> = (0..100).step_by(3).collect(); // multiples of 3
        let expected = (0..100u32).filter(|x| x % 2 == 0 && x % 3 == 0).count();
        assert_eq!(sparse_intersection_count(&a, &b), expected);
        assert_eq!(merge_intersection_count_scalar(&a, &b), expected);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection above.
            assert_eq!(unsafe { merge_intersection_count_avx2(&a, &b) }, expected);
        }
    }

    #[test]
    fn test_identical() {
        let a: Vec<u32> = (0..50).collect();
        assert_eq!(sparse_intersection_count(&a, &a), 50);
    }

    #[test]
    fn test_reversed_args() {
        // Ensure order does not matter
        let a = vec![1, 5, 10];
        let b: Vec<u32> = (0..1000).collect();
        assert_eq!(
            sparse_intersection_count(&a, &b),
            sparse_intersection_count(&b, &a)
        );
    }
}
