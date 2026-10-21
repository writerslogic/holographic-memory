// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Sorted posting lists in CSR form and their build pass.
//!
//! `offsets[c]..offsets[c + 1]` indexes channel `c`'s ids. Lists are strictly
//! increasing because the build visits corpus rows in id order (windows are
//! sequential, chunks within a window are merged in chunk order, rows within
//! a chunk in row order) and a row joins a channel at most once.
//!
//! The build projects each row once and tests it against every budget's
//! thresholds, so one pass yields one `Postings` per `B`. The projection pass
//! runs on the given rayon pool; the merge is sequential and order-fixed, so
//! the result does not depend on the thread count.

use anyhow::{ensure, Result};
use rayon::prelude::*;
use serde::Serialize;

use crate::bank::{Bank, Thresholds};

/// CSR posting lists over `m` channels.
pub struct Postings {
    /// Channel count.
    pub m: usize,
    /// `m + 1` offsets into `ids`.
    pub offsets: Vec<u32>,
    /// Concatenated per-channel sorted ids.
    pub ids: Vec<u32>,
}

/// Per-channel occupancy distribution.
#[derive(Serialize, Clone, Copy, Debug, Default)]
pub struct Occupancy {
    /// Channels.
    pub channels: usize,
    /// Total memberships (= `ids.len()`).
    pub total: usize,
    /// Mean list length.
    pub mean: f64,
    /// Shortest list.
    pub min: usize,
    /// 5th-percentile list length.
    pub p05: usize,
    /// Median list length.
    pub p50: usize,
    /// 95th-percentile list length.
    pub p95: usize,
    /// Longest list.
    pub max: usize,
    /// Channels with no members.
    pub empty: usize,
}

impl Postings {
    /// Channel `c`'s sorted ids.
    #[inline]
    pub fn list(&self, c: usize) -> &[u32] {
        &self.ids[self.offsets[c] as usize..self.offsets[c + 1] as usize]
    }

    /// Total memberships.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True when no channel has a member.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Bytes of `ids` plus `offsets`.
    pub fn bytes(&self) -> usize {
        4 * (self.ids.len() + self.offsets.len())
    }

    /// True when every list is strictly increasing.
    pub fn is_sorted(&self) -> bool {
        (0..self.m).all(|c| self.list(c).windows(2).all(|w| w[0] < w[1]))
    }

    /// Occupancy distribution over channels.
    pub fn occupancy(&self) -> Occupancy {
        let mut lens: Vec<usize> = (0..self.m).map(|c| self.list(c).len()).collect();
        if lens.is_empty() {
            return Occupancy::default();
        }
        lens.sort_unstable();
        let n = lens.len();
        let at = |p: f64| lens[(((n - 1) as f64) * p).round() as usize];
        Occupancy {
            channels: n,
            total: self.ids.len(),
            mean: self.ids.len() as f64 / n as f64,
            min: lens[0],
            p05: at(0.05),
            p50: at(0.5),
            p95: at(0.95),
            max: lens[n - 1],
            empty: lens.iter().filter(|&&l| l == 0).count(),
        }
    }
}

/// Build one `Postings` per entry of `taus` over the first `n` rows of
/// `docs` (row-major, `bank.dim` wide). `window_rows` bounds the rows whose
/// membership pairs are held in memory at once.
pub fn build(
    bank: &Bank,
    taus: &[Thresholds],
    docs: &[f32],
    n: usize,
    pool: &rayon::ThreadPool,
    window_rows: usize,
) -> Result<Vec<Postings>> {
    let dim = bank.dim;
    let m = bank.m;
    ensure!(docs.len() >= n * dim, "corpus has fewer than {} rows", n);
    ensure!(n <= u32::MAX as usize, "ids must fit in u32");
    ensure!(!taus.is_empty(), "no budgets to build");
    for t in taus {
        ensure!(t.tau.len() == m, "threshold table has {} entries, bank has {} channels", t.tau.len(), m);
    }
    let nb = taus.len();
    let mut lists: Vec<Vec<Vec<u32>>> = taus
        .iter()
        .map(|t| {
            let cap = n * t.b / m + 16;
            (0..m).map(|_| Vec::with_capacity(cap)).collect()
        })
        .collect();

    const CHUNK_ROWS: usize = 1024;
    let window = window_rows.max(CHUNK_ROWS);
    let mut start = 0usize;
    while start < n {
        let end = (start + window).min(n);
        let slab = &docs[start * dim..end * dim];
        let parts: Vec<Vec<Vec<(u32, u32)>>> = pool.install(|| {
            slab.par_chunks(CHUNK_ROWS * dim)
                .enumerate()
                .map_init(
                    || (vec![0.0f32; bank.dirs], Vec::<u32>::with_capacity(256)),
                    |state: &mut (Vec<f32>, Vec<u32>), (ci, rows): (usize, &[f32])| {
                        let (y, mem) = state;
                        let base = start + ci * CHUNK_ROWS;
                        let mut out: Vec<Vec<(u32, u32)>> = (0..nb).map(|_| Vec::new()).collect();
                        for (r, x) in rows.chunks_exact(dim).enumerate() {
                            let id = (base + r) as u32;
                            bank.project(x, &mut y[..]);
                            for (bi, t) in taus.iter().enumerate() {
                                mem.clear();
                                bank.memberships(&y[..], &t.tau[..], mem);
                                out[bi].extend(mem.iter().map(|&c| (c, id)));
                            }
                        }
                        out
                    },
                )
                .collect()
        });
        for part in parts {
            for (bi, pairs) in part.into_iter().enumerate() {
                for (c, id) in pairs {
                    lists[bi][c as usize].push(id);
                }
            }
        }
        start = end;
    }

    let mut result = Vec::with_capacity(nb);
    for per_chan in lists {
        let total: usize = per_chan.iter().map(Vec::len).sum();
        ensure!(total <= u32::MAX as usize, "posting offsets must fit in u32");
        let mut offsets = Vec::with_capacity(m + 1);
        let mut ids = Vec::with_capacity(total);
        offsets.push(0u32);
        for l in &per_chan {
            ids.extend_from_slice(l);
            offsets.push(ids.len() as u32);
        }
        let p = Postings { m, offsets, ids };
        debug_assert!(p.is_sorted());
        result.push(p);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Rng;
    use crate::normalize_in_place;

    #[test]
    fn build_is_sorted_and_thread_count_independent() {
        let dim = 12;
        let n = 3000;
        let mut rng = Rng::new(11);
        let mut docs = vec![0.0f32; n * dim];
        for row in docs.chunks_exact_mut(dim) {
            for x in row.iter_mut() {
                *x = rng.uniform() as f32 - 0.5;
            }
            normalize_in_place(row);
        }
        let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
        let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build().expect("pool");
        let mut bank = Bank::new(64, dim, 5).expect("bank");
        let taus = bank.fit(&docs[..1024 * dim], &[4, 8], &pool1).expect("fit");
        let a = build(&bank, &taus, &docs, n, &pool1, 1024).expect("build");
        let b = build(&bank, &taus, &docs, n, &pool4, 1500).expect("build");
        assert_eq!(a.len(), 2);
        for (x, y) in a.iter().zip(&b) {
            assert!(x.is_sorted());
            assert_eq!(x.offsets, y.offsets);
            assert_eq!(x.ids, y.ids);
        }
        let mean = a[0].len() as f64 / n as f64;
        assert!(mean > 2.0 && mean < 6.0, "mean memberships {mean}");
    }
}
