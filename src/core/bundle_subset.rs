// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Subset-union ("Bloom") bundling: each item contributes only a
//! deterministic `k'`-subset of its active indices to the union, and
//! membership requires all `k'` subset indices to be present. Lowering `k'`
//! slows bundle saturation at the cost of a higher per-query false-positive
//! exponent; see `src/bin/bundle-subset-sweep.rs` for measurements.

use super::entangled::{hash_u64, EntangledHVec};

/// Content hash of an item's own indices, used as the selection key so no
/// external id is needed.
fn item_key(item: &EntangledHVec) -> u64 {
    item.indices()
        .iter()
        .fold(0x5B17_5E7Cu64, |h, &i| hash_u64(h, i as u64))
}

/// Deterministic `k'`-subset of `item`'s active indices, sorted ascending.
/// Picks the `k'` indices with the smallest `hash(key, index)`. If
/// `k' >= item size` the whole index set is returned.
pub fn subset_indices(item: &EntangledHVec, k: usize) -> Vec<u32> {
    let idx = item.indices();
    if k >= idx.len() {
        return idx.to_vec();
    }
    let key = item_key(item);
    let mut ranked: Vec<(u64, u32)> = idx.iter().map(|&i| (hash_u64(key, i as u64), i)).collect();
    ranked.sort_unstable();
    ranked.truncate(k);
    let mut out: Vec<u32> = ranked.into_iter().map(|(_, i)| i).collect();
    out.sort_unstable();
    out
}

/// Union of every item's `k'`-subset. Returns an empty zero-dim vector for
/// empty input, matching `EntangledHVec::bundle_bloom`.
pub fn bundle_subset<V: std::borrow::Borrow<EntangledHVec>>(
    items: &[V],
    k: usize,
) -> EntangledHVec {
    let Some(first) = items.first() else {
        return EntangledHVec::from_indices(Vec::new(), 0);
    };
    let dim = first.borrow().dim;
    let mut all: Vec<u32> = items
        .iter()
        .flat_map(|v| subset_indices(v.borrow(), k))
        .collect();
    all.sort_unstable();
    all.dedup();
    EntangledHVec::from_indices(all, dim)
}

/// True when every index of `item`'s `k'`-subset is present in `bundle`.
pub fn contains_subset(bundle: &EntangledHVec, item: &EntangledHVec, k: usize) -> bool {
    let b = bundle.indices();
    subset_indices(item, k)
        .iter()
        .all(|i| b.binary_search(i).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 16384;

    fn item(seed: u64) -> EntangledHVec {
        EntangledHVec::new_with_density(D, 256, seed)
    }

    #[test]
    fn selection_is_deterministic_and_a_subset() {
        let a = item(7);
        let s1 = subset_indices(&a, 11);
        let s2 = subset_indices(&a.clone(), 11);
        assert_eq!(s1, s2);
        assert_eq!(s1.len(), 11);
        assert!(s1.windows(2).all(|w| w[0] < w[1]));
        assert!(s1.iter().all(|i| a.indices().binary_search(i).is_ok()));
    }

    #[test]
    fn every_inserted_member_tests_positive() {
        let items: Vec<_> = (0..200).map(item).collect();
        for k in [4, 11, 64] {
            let b = bundle_subset(&items, k);
            assert!(items.iter().all(|it| contains_subset(&b, it, k)));
        }
    }

    #[test]
    fn subset_size_clamps_at_item_size() {
        let a = item(3);
        let n = a.indices().len();
        assert_eq!(subset_indices(&a, n), a.indices());
        assert_eq!(subset_indices(&a, n + 100), a.indices());
        assert_eq!(subset_indices(&a, n - 1).len(), n - 1);
        assert!(subset_indices(&a, 0).is_empty());
    }
}
