// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x517c_c1b7_2722_0a95;

/// The multiply-rotate Fx hash with 64-bit words and no seed.
///
/// foldhash replaces it where it measured faster (the bulk-index run map keyed by `&str`).
/// Alternating runs on the 22-core benchmark host, median of 21, foldhash over this hash:
/// random `usize` slot table 1.11x, `u32` score table 1.21x, sequential `usize` set 1.27x,
/// `String` ids 1.08x, random `usize` set 0.96x, `&str` terms 0.79x. Maps keyed by integers or
/// owned strings therefore keep this hash. It is not DoS-resistant and its iteration order is
/// reproducible, so keys must not be attacker-chosen without a bound.
#[derive(Clone, Copy, Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_ne_bytes(*word));
        }
        let mut rest = rest;
        if let Some((word, tail)) = rest.split_first_chunk::<4>() {
            self.add(u64::from(u32::from_ne_bytes(*word)));
            rest = tail;
        }
        for &byte in rest {
            self.add(u64::from(byte));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub(crate) type FastSet<T> = HashSet<T, BuildHasherDefault<FxHasher>>;

/// Fx hash of `bytes`.
///
/// Shard placement and the audit `id_hash` of non-`security` builds depend on these exact
/// values, so the algorithm must not change.
pub(crate) fn fx64(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(bytes);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use std::hash::{BuildHasher, BuildHasherDefault};

    use super::{fx64, FxHasher};

    #[test]
    fn fx64_matches_recorded_values() {
        for (input, expected) in [
            (&b""[..], 0),
            (b"a", 0xe045_6665_d3e6_0275),
            (b"test-id", 0xdd05_573d_9bc9_2c95),
            (b"persist_test", 0xad79_bfe2_e288_eb06),
            (
                b"holographic-memory shard key 1234567",
                0xea19_f7d1_a034_dc79,
            ),
        ] {
            assert_eq!(fx64(input), expected, "input {input:?}");
        }
    }

    #[test]
    fn hash_trait_paths_match_recorded_values() {
        let build = BuildHasherDefault::<FxHasher>::default();
        assert_eq!(build.hash_one(7u32), 0x3a69_4c02_11ee_4a13);
        assert_eq!(build.hash_one("ab"), 0x8e41_c8ab_dad1_f8ab);
    }
}
