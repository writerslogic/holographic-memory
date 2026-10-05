// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sparse intersection primitives.
//!
//! Galloping intersection for skewed sizes, sorted merge for balanced.

/// Count common elements in two sorted u32 slices.
///
/// Adaptively picks the best algorithm:
/// - When one slice is much smaller (|a| * 8 < |b|), uses galloping search
///   which is O(|small| * log(|large|)) and beats O(|a| + |b|) merge.
/// - Otherwise uses the standard sorted merge at O(|a| + |b|).
#[inline]
pub fn sparse_intersection_count(a: &[u32], b: &[u32]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    // Ensure a is the smaller slice for the skew check.
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    // Skewed sizes gallop before any block kernel: a block kernel walks the
    // large side linearly.
    if small.len() * 8 < large.len() {
        return galloping_intersection_count(small, large);
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(count) = crate::core::simd_math::simd_intersection_count(small, large) {
        return count;
    }

    merge_intersection_count(small, large)
}

/// Sorted merge intersection count: O(|a| + |b|).
#[inline]
fn merge_intersection_count(a: &[u32], b: &[u32]) -> usize {
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
    }

    #[test]
    fn test_identical() {
        let a: Vec<u32> = (0..50).collect();
        assert_eq!(sparse_intersection_count(&a, &a), 50);
    }

    fn naive(a: &[u32], b: &[u32]) -> usize {
        let set: std::collections::HashSet<u32> = b.iter().copied().collect();
        a.iter().filter(|x| set.contains(x)).count()
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// Sorted, deduplicated sample of `n` values below `universe` (`universe >= n`).
    fn sample(rng: &mut Lcg, n: usize, universe: u64) -> Vec<u32> {
        let mut set = std::collections::BTreeSet::new();
        while set.len() < n {
            set.insert((rng.next() % universe) as u32);
        }
        set.into_iter().collect()
    }

    fn check_all(a: &[u32], b: &[u32]) {
        let want = naive(a, b);
        assert_eq!(sparse_intersection_count(a, b), want, "dispatch a,b");
        assert_eq!(sparse_intersection_count(b, a), want, "dispatch b,a");
        assert_eq!(merge_intersection_count(a, b), want, "merge");
        let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        assert_eq!(galloping_intersection_count(small, large), want, "gallop");
        #[cfg(target_arch = "x86_64")]
        if let Some(c) = crate::core::simd_math::simd_intersection_count(a, b) {
            assert_eq!(c, want, "avx2");
        }
    }

    #[test]
    fn all_kernels_match_naive_count() {
        for seed in 0..300u64 {
            let mut rng = Lcg(seed);
            let na = (rng.next() % 2001) as usize;
            let nb = (rng.next() % 2001) as usize;
            // A universe near max(na, nb) forces heavy overlap; a wide one forces little.
            let universe = (na.max(nb) as u64 + 1) * (1 + rng.next() % 8);
            check_all(
                &sample(&mut rng, na, universe),
                &sample(&mut rng, nb, universe),
            );
        }
    }

    #[test]
    fn kernels_match_on_edge_shapes_and_tails() {
        let mut rng = Lcg(7);
        for na in 0..=17usize {
            for nb in (0..=17usize).chain([64, 65, 600]) {
                for universe in [40u64, 2000, 1 << 20] {
                    let universe = universe.max(na.max(nb) as u64);
                    check_all(
                        &sample(&mut rng, na, universe),
                        &sample(&mut rng, nb, universe),
                    );
                }
            }
        }
        let ident = sample(&mut rng, 777, 5000);
        check_all(&ident, &ident);
        let evens: Vec<u32> = (0..1000).map(|x| x * 2).collect();
        let odds: Vec<u32> = (0..1000).map(|x| x * 2 + 1).collect();
        check_all(&evens, &odds);
        check_all(&[], &odds);
        check_all(&[], &[]);
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
