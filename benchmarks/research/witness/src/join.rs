// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Counting threshold-join over sorted posting lists.
//!
//! One `u8` counter per corpus id. A join streams the selected channels'
//! lists in order, increments the counter of every id it sees, and emits an
//! id the moment its counter reaches `t` (so the candidate list is
//! duplicate-free and ordered by the position of each id's `t`-th witness).
//! The counter array is 1 byte x n (1.18 MB at glove scale), which stays in
//! L2; the increments are independent random accesses the core can overlap,
//! unlike a graph walk's dependent fetches.
//!
//! # Invariants
//!
//! * Each id occurs at most once per list (lists are strictly increasing), so
//!   a counter is at most the number of selected channels, which the caller
//!   keeps `<= 255`.
//! * The join leaves counters dirty; `reset` with the same channels restores
//!   all zeros. Every public entry point pairs `join_*` with `reset`.

use crate::postings::Postings;

/// Counter array shared by consecutive queries.
pub struct Joiner {
    counts: Vec<u8>,
}

/// Witness statistics of one join: `reached[k - 1]` is the number of ids
/// whose counter reached `k` (i.e. with at least `k` witnesses), for
/// `k = 1..=4`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    /// Ids with `>= k` witnesses, `k = 1..=4`.
    pub reached: [usize; 4],
    /// Bytes of posting ids streamed.
    pub posting_bytes: usize,
}

impl Joiner {
    /// Zeroed counters for `n` ids.
    pub fn new(n: usize) -> Self {
        Self { counts: vec![0u8; n] }
    }

    /// Number of ids covered.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// True when no ids are covered.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Stream `chans` and push every id whose counter reaches exactly `t`
    /// into `out`. Returns the posting bytes streamed. Counters stay dirty.
    pub fn join_emit(&mut self, p: &Postings, chans: &[u32], t: u8, out: &mut Vec<u32>) -> usize {
        let mut bytes = 0usize;
        for &c in chans {
            let list = p.list(c as usize);
            bytes += 4 * list.len();
            for &id in list {
                let e = &mut self.counts[id as usize];
                *e += 1;
                if *e == t {
                    out.push(id);
                }
            }
        }
        bytes
    }

    /// Stream `chans` and tally how many ids reach 1, 2, 3 and 4 witnesses.
    /// Counters stay dirty so the caller can probe them with [`count`].
    ///
    /// [`count`]: Joiner::count
    pub fn join_tally(&mut self, p: &Postings, chans: &[u32]) -> Tally {
        let mut tally = Tally::default();
        for &c in chans {
            let list = p.list(c as usize);
            tally.posting_bytes += 4 * list.len();
            for &id in list {
                let e = &mut self.counts[id as usize];
                *e += 1;
                let k = *e as usize;
                if k <= 4 {
                    tally.reached[k - 1] += 1;
                }
            }
        }
        tally
    }

    /// Current witness count of `id`.
    #[inline]
    pub fn count(&self, id: u32) -> u8 {
        self.counts[id as usize]
    }

    /// Zero the counters touched by `chans`.
    pub fn reset(&mut self, p: &Postings, chans: &[u32]) {
        for &c in chans {
            for &id in p.list(c as usize) {
                self.counts[id as usize] = 0;
            }
        }
    }

    /// True when every counter is zero (test / debug check).
    pub fn is_clean(&self) -> bool {
        self.counts.iter().all(|&c| c == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> Postings {
        // channel 0: {1, 4, 7}, channel 1: {4, 7, 9}, channel 2: {7, 9}, channel 3: {}
        Postings {
            m: 4,
            offsets: vec![0, 3, 6, 8, 8],
            ids: vec![1, 4, 7, 4, 7, 9, 7, 9],
        }
    }

    #[test]
    fn emit_and_tally_agree_and_reset_cleans() {
        let p = toy();
        let mut j = Joiner::new(10);
        let chans = [0u32, 1, 2, 3];
        let mut out = Vec::new();
        let bytes = j.join_emit(&p, &chans, 2, &mut out);
        assert_eq!(bytes, 32);
        assert_eq!(out, vec![4, 7, 9]);
        j.reset(&p, &chans);
        assert!(j.is_clean());
        let t = j.join_tally(&p, &chans);
        assert_eq!(t.reached, [4, 3, 1, 0]);
        assert_eq!(j.count(7), 3);
        assert_eq!(j.count(2), 0);
        j.reset(&p, &chans);
        assert!(j.is_clean());
        out.clear();
        j.join_emit(&p, &chans, 3, &mut out);
        assert_eq!(out, vec![7]);
        j.reset(&p, &chans);
    }
}
