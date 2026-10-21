// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The channel bank: `M` signed channels from `M/2` SRHT projection
//! directions, per-channel empirical tail tables, membership thresholds and
//! the query-side channel selector.
//!
//! # Projections
//!
//! The input dimension `D` is zero-padded to `Dp = next_power_of_two(D)`.
//! Direction block `b` (`0 <= b < M / (2 Dp)`) owns a seeded sign vector
//! `d_b in {-1, +1}^Dp`; its `Dp` projections are the unnormalized Walsh-
//! Hadamard transform `y_b = H_Dp (d_b * x)`. Direction `m = b * Dp + k` has
//! two signed channels: `2m` fires on the upper tail of `y[m]`, `2m + 1` on
//! the lower tail (its signed value is `-y[m]`). Cost per vector:
//! `(M/2) * log2(Dp)` add/subs plus `M/2` sign multiplies.
//!
//! # Tables (fitted on a seeded sample of `S` corpus rows)
//!
//! Per direction: sample mean and `1/std` (query-side pre-filter only).
//! Per signed channel: `GRID_LEVELS` upper-tail quantiles of the signed
//! value at tail probabilities `2^-(j+1)`, `j = 0..GRID_LEVELS`
//! (nondecreasing in `j`), and one membership threshold `tau` per budget
//! `B`: the empirical quantile at tail probability `p_B = B / M`, so every
//! channel has the same expected occupancy `N * B / M` and a vector has `B`
//! memberships on average (its own count varies).
//!
//! # Selection
//!
//! `select` ranks channels by quantile rank, not raw value: a z-score pass
//! over the `M/2` directions keeps the directions whose `|z|` clears a
//! Gaussian cut sized to pass about `4Q` of them (the cut halves until at
//! least `2Q` pass), then each survivor's channel on the side of the query's
//! sign gets its exact tail-grid score, and the top `Q` by score are kept.
//!
//! # Invariants
//!
//! * A vector (or query) uses at most one of a direction's two channels: the
//!   membership test is `y >= tau[2m]` else `-y >= tau[2m + 1]`, and the
//!   selector scores only the channel on the side of the query's sign.
//! * `grid[c]` is nondecreasing; `grid_score` is monotone in the signed
//!   value and equals `j + frac` where the tail probability is about
//!   `2^-(j + 1 + frac)`.
//! * `select` is deterministic: ties on score break on the channel index.

use anyhow::{ensure, Result};
use rayon::prelude::*;

use crate::data::Rng;

/// Tail-quantile grid levels per signed channel (`2^-1 .. 2^-GRID_LEVELS`).
pub const GRID_LEVELS: usize = 12;

/// Membership thresholds for one budget `B`.
#[derive(Clone, Debug)]
pub struct Thresholds {
    /// Budget (expected memberships per vector).
    pub b: usize,
    /// Tail probability `B / M`.
    pub tail_p: f64,
    /// Sample rank used for the quantile (`round(tail_p * S)`, at least 1).
    pub sample_rank: usize,
    /// Per signed channel: member when the signed value is `>= tau[c]`.
    pub tau: Vec<f32>,
}

/// SRHT channel bank with fitted tables.
pub struct Bank {
    /// Signed channels.
    pub m: usize,
    /// Projection directions (`m / 2`).
    pub dirs: usize,
    /// Input dimension.
    pub dim: usize,
    /// Padded (power of two) dimension.
    pub dim_pad: usize,
    /// SRHT blocks (`dirs / dim_pad`).
    pub blocks: usize,
    signs: Vec<f32>,
    /// Per-direction sample mean.
    pub mean: Vec<f32>,
    /// Per-direction `1 / std` (0 for a degenerate direction).
    pub inv_std: Vec<f32>,
    /// Per signed channel, `GRID_LEVELS` nondecreasing tail quantiles.
    pub grid: Vec<f32>,
}

/// In-place unnormalized Walsh-Hadamard transform; `v.len()` is a power of two.
pub fn fwht(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            let (lo, hi) = v[i..i + 2 * h].split_at_mut(h);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let x = *a;
                let y = *b;
                *a = x + y;
                *b = x - y;
            }
            i += 2 * h;
        }
        h <<= 1;
    }
}

/// Upper-tail standard-normal quantile: `x` with `P(Z >= x) = p`
/// (Acklam's rational approximation, relative error below 1.2e-9).
pub fn normal_upper_tail(p: f64) -> f64 {
    let p = p.clamp(1e-12, 0.5);
    if p >= 0.5 {
        return 0.0;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;
    // Quantile of the lower tail at probability p is -x; evaluate at p directly.
    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    -x
}

/// Selector buffers reused across queries.
#[derive(Default)]
pub struct SelectScratch {
    cand: Vec<(f32, u32)>,
}

struct DirFit {
    mean: f32,
    inv_std: f32,
    grid_plus: [f32; GRID_LEVELS],
    grid_minus: [f32; GRID_LEVELS],
    tau_plus: Vec<f32>,
    tau_minus: Vec<f32>,
}

/// Order statistics of one direction's sample column (sorted in place).
fn fit_direction(col: &mut [f32], grid_ranks: &[usize; GRID_LEVELS], ranks: &[usize]) -> DirFit {
    let n = col.len();
    let mean = col.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = col
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std = var.sqrt();
    let inv_std = if std > 0.0 { (1.0 / std) as f32 } else { 0.0 };
    col.sort_unstable_by(|a, b| a.total_cmp(b));
    // k-th largest is col[n - k]; k-th smallest is col[k - 1].
    let mut grid_plus = [0.0f32; GRID_LEVELS];
    let mut grid_minus = [0.0f32; GRID_LEVELS];
    for j in 0..GRID_LEVELS {
        let k = grid_ranks[j].clamp(1, n);
        grid_plus[j] = col[n - k];
        grid_minus[j] = -col[k - 1];
    }
    let tau_plus = ranks.iter().map(|&k| col[n - k.clamp(1, n)]).collect();
    let tau_minus = ranks.iter().map(|&k| -col[k.clamp(1, n) - 1]).collect();
    DirFit {
        mean: mean as f32,
        inv_std,
        grid_plus,
        grid_minus,
        tau_plus,
        tau_minus,
    }
}

struct Tables {
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    grid: Vec<f32>,
    taus: Vec<Thresholds>,
}

fn fit_tables(bank: &Bank, sample: &[f32], bs: &[usize], pool: &rayon::ThreadPool) -> Result<Tables> {
    let dim = bank.dim;
    let dp = bank.dim_pad;
    ensure!(sample.len() % dim == 0, "sample length is not a multiple of dim");
    let s_n = sample.len() / dim;
    ensure!(s_n >= 64, "need at least 64 sample rows, got {}", s_n);
    ensure!(!bs.is_empty(), "no budgets");
    let mut taus = Vec::with_capacity(bs.len());
    let mut ranks = Vec::with_capacity(bs.len());
    for &b in bs {
        ensure!(b >= 1 && b < bank.m, "budget B={} must be in [1, M={})", b, bank.m);
        let tail_p = b as f64 / bank.m as f64;
        let sample_rank = ((tail_p * s_n as f64).round() as usize).clamp(1, s_n);
        ranks.push(sample_rank);
        taus.push(Thresholds {
            b,
            tail_p,
            sample_rank,
            tau: vec![0.0f32; bank.m],
        });
    }
    let mut grid_ranks = [0usize; GRID_LEVELS];
    for (j, g) in grid_ranks.iter_mut().enumerate() {
        *g = ((s_n as f64 * 2f64.powi(-(j as i32 + 1))).round() as usize).clamp(1, s_n);
    }
    let mut mean = vec![0.0f32; bank.dirs];
    let mut inv_std = vec![0.0f32; bank.dirs];
    let mut grid = vec![0.0f32; bank.m * GRID_LEVELS];
    let mut rows = vec![0.0f32; s_n * dp];
    let mut cols = vec![0.0f32; dp * s_n];
    for blk in 0..bank.blocks {
        pool.install(|| {
            rows.par_chunks_mut(dp)
                .zip(sample.par_chunks(dim))
                .for_each(|(o, x)| bank.project_block(blk, x, o));
        });
        for (i, r) in rows.chunks_exact(dp).enumerate() {
            for k in 0..dp {
                cols[k * s_n + i] = r[k];
            }
        }
        let fits: Vec<DirFit> = pool.install(|| {
            cols.par_chunks_mut(s_n)
                .map(|col| fit_direction(col, &grid_ranks, &ranks))
                .collect()
        });
        for (k, f) in fits.iter().enumerate() {
            let d = blk * dp + k;
            mean[d] = f.mean;
            inv_std[d] = f.inv_std;
            let cp = 2 * d;
            let cm = 2 * d + 1;
            grid[cp * GRID_LEVELS..(cp + 1) * GRID_LEVELS].copy_from_slice(&f.grid_plus);
            grid[cm * GRID_LEVELS..(cm + 1) * GRID_LEVELS].copy_from_slice(&f.grid_minus);
            for (bi, t) in taus.iter_mut().enumerate() {
                t.tau[cp] = f.tau_plus[bi];
                t.tau[cm] = f.tau_minus[bi];
            }
        }
    }
    Ok(Tables {
        mean,
        inv_std,
        grid,
        taus,
    })
}

impl Bank {
    /// A bank of `m` signed channels for `dim`-dimensional input; `m` must be
    /// a positive multiple of `2 * next_power_of_two(dim)`. Tables are zero
    /// until [`fit`](Bank::fit).
    pub fn new(m: usize, dim: usize, seed: u64) -> Result<Self> {
        ensure!(dim > 0, "dim must be positive");
        let dim_pad = dim.next_power_of_two().max(2);
        ensure!(
            m >= 2 * dim_pad && m % (2 * dim_pad) == 0,
            "M={} must be a positive multiple of 2 * dim_pad = {}",
            m,
            2 * dim_pad
        );
        let dirs = m / 2;
        let blocks = dirs / dim_pad;
        let mut rng = Rng::new(seed ^ 0x5352_4854_0000_0000 ^ (m as u64));
        let signs = (0..blocks * dim_pad)
            .map(|_| if rng.next_u64() >> 63 == 0 { 1.0f32 } else { -1.0f32 })
            .collect();
        Ok(Self {
            m,
            dirs,
            dim,
            dim_pad,
            blocks,
            signs,
            mean: vec![0.0f32; dirs],
            inv_std: vec![0.0f32; dirs],
            grid: vec![0.0f32; m * GRID_LEVELS],
        })
    }

    /// Bytes of the signs, direction stats and grid tables.
    pub fn table_bytes(&self) -> usize {
        4 * (self.signs.len() + self.mean.len() + self.inv_std.len() + self.grid.len())
    }

    /// Block `b`'s `dim_pad` projections of `x` (`dim` long) into `out`.
    #[inline]
    pub fn project_block(&self, b: usize, x: &[f32], out: &mut [f32]) {
        let dp = self.dim_pad;
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), dp);
        let sg = &self.signs[b * dp..(b + 1) * dp];
        for i in 0..self.dim {
            out[i] = x[i] * sg[i];
        }
        for v in &mut out[self.dim..] {
            *v = 0.0;
        }
        fwht(out);
    }

    /// All `dirs` projections of `x` into `out`.
    pub fn project(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.dirs);
        for (b, o) in out.chunks_exact_mut(self.dim_pad).enumerate() {
            self.project_block(b, x, o);
        }
    }

    /// Fit the tables on `sample` (row-major, `dim` wide) and return the
    /// membership thresholds for each budget in `bs`.
    pub fn fit(&mut self, sample: &[f32], bs: &[usize], pool: &rayon::ThreadPool) -> Result<Vec<Thresholds>> {
        let t = fit_tables(self, sample, bs, pool)?;
        self.mean = t.mean;
        self.inv_std = t.inv_std;
        self.grid = t.grid;
        Ok(t.taus)
    }

    /// Channels the projection vector `y` belongs to under `tau`, appended
    /// to `out` in increasing channel order (at most one per direction).
    pub fn memberships(&self, y: &[f32], tau: &[f32], out: &mut Vec<u32>) {
        debug_assert_eq!(y.len(), self.dirs);
        debug_assert_eq!(tau.len(), self.m);
        for m in 0..self.dirs {
            let v = y[m];
            if v >= tau[2 * m] {
                out.push((2 * m) as u32);
            } else if -v >= tau[2 * m + 1] {
                out.push((2 * m + 1) as u32);
            }
        }
    }

    /// Tail-grid score of signed value `s` on channel `chan`: `j + frac`
    /// for `grid[j] <= s < grid[j + 1]`, extrapolated linearly past the last
    /// level, negative below the first. Monotone in `s`.
    #[inline]
    pub fn grid_score(&self, chan: usize, s: f32) -> f32 {
        const G: usize = GRID_LEVELS;
        let g = &self.grid[chan * G..(chan + 1) * G];
        if s < g[0] {
            let w = (g[1] - g[0]).max(f32::EPSILON);
            return (s - g[0]) / w;
        }
        let mut j = 0;
        while j + 1 < G && s >= g[j + 1] {
            j += 1;
        }
        if j + 1 < G {
            let w = (g[j + 1] - g[j]).max(f32::EPSILON);
            j as f32 + (s - g[j]) / w
        } else {
            let w = (g[G - 1] - g[G - 2]).max(f32::EPSILON);
            (G - 1) as f32 + (s - g[G - 1]) / w
        }
    }

    /// The query's `q` most extreme channels by quantile rank, written to
    /// `out` in decreasing score order. Requires `1 <= q <= dirs`.
    pub fn select(&self, y: &[f32], q: usize, sc: &mut SelectScratch, out: &mut Vec<u32>) {
        let dirs = self.dirs;
        debug_assert!(q >= 1 && q <= dirs);
        debug_assert_eq!(y.len(), dirs);
        let want = (4 * q).min(dirs);
        let mut zcut = normal_upper_tail(want as f64 / (2.0 * dirs as f64)) as f32;
        loop {
            sc.cand.clear();
            for m in 0..dirs {
                let z = (y[m] - self.mean[m]) * self.inv_std[m];
                let az = z.abs();
                if az >= zcut {
                    let side = usize::from(z < 0.0);
                    sc.cand.push((az, (2 * m + side) as u32));
                }
            }
            if (sc.cand.len() >= want / 2 && sc.cand.len() >= q) || zcut <= 0.0 {
                break;
            }
            zcut = if zcut < 0.25 { 0.0 } else { zcut * 0.5 };
        }
        for c in sc.cand.iter_mut() {
            let chan = c.1 as usize;
            let m = chan >> 1;
            let s = if chan & 1 == 0 { y[m] } else { -y[m] };
            c.0 = self.grid_score(chan, s);
        }
        let cmp = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        if sc.cand.len() > q {
            sc.cand.select_nth_unstable_by(q - 1, cmp);
            sc.cand.truncate(q);
        }
        sc.cand.sort_unstable_by(cmp);
        out.clear();
        out.extend(sc.cand.iter().map(|c| c.1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize_in_place;

    #[test]
    fn fwht_is_orthogonal_up_to_scale() {
        let mut v: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let orig = v.clone();
        let e0: f32 = orig.iter().map(|x| x * x).sum();
        fwht(&mut v);
        let e1: f32 = v.iter().map(|x| x * x).sum();
        assert!((e1 - 16.0 * e0).abs() < 1e-3 * e1.max(1.0));
        fwht(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a / 16.0 - b).abs() < 1e-4);
        }
    }

    #[test]
    fn normal_tail_quantiles_are_close() {
        assert!((normal_upper_tail(0.5)).abs() < 1e-9);
        assert!((normal_upper_tail(0.158_655_25) - 1.0).abs() < 1e-6);
        assert!((normal_upper_tail(0.022_750_13) - 2.0).abs() < 1e-6);
        assert!((normal_upper_tail(0.001_349_898) - 3.0).abs() < 1e-6);
    }

    fn toy_bank() -> (Bank, Vec<Thresholds>, Vec<f32>, usize) {
        let dim = 10;
        let n = 4096;
        let mut rng = Rng::new(9);
        let mut docs = vec![0.0f32; n * dim];
        for row in docs.chunks_exact_mut(dim) {
            for x in row.iter_mut() {
                *x = rng.uniform() as f32 - 0.5;
            }
            normalize_in_place(row);
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
        let mut bank = Bank::new(128, dim, 1).expect("bank");
        let taus = bank.fit(&docs, &[8], &pool).expect("fit");
        (bank, taus, docs, dim)
    }

    #[test]
    fn fitted_tables_balance_occupancy_and_grid_is_monotone() {
        let (bank, taus, docs, dim) = toy_bank();
        assert_eq!(bank.dirs, 64);
        assert_eq!(bank.blocks, 4);
        let s_n = docs.len() / dim;
        let mut y = vec![0.0f32; bank.dirs];
        let mut occ = vec![0usize; bank.m];
        let mut mem = Vec::new();
        for row in docs.chunks_exact(dim) {
            bank.project(row, &mut y);
            mem.clear();
            bank.memberships(&y, &taus[0].tau, &mut mem);
            for &c in &mem {
                occ[c as usize] += 1;
            }
        }
        let expect = taus[0].sample_rank;
        for (c, &o) in occ.iter().enumerate() {
            assert!(o >= expect && o <= expect + 2, "channel {c}: occupancy {o}, expected ~{expect}");
        }
        let total: usize = occ.iter().sum();
        assert!((total as f64 / s_n as f64 - 8.0).abs() < 0.5);
        for c in 0..bank.m {
            let g = &bank.grid[c * GRID_LEVELS..(c + 1) * GRID_LEVELS];
            assert!(g.windows(2).all(|w| w[0] <= w[1]));
            let lo = bank.grid_score(c, g[0]);
            let hi = bank.grid_score(c, g[GRID_LEVELS - 1]);
            assert!(lo < hi);
            assert!(bank.grid_score(c, g[0] - 1.0) < lo);
            assert!(bank.grid_score(c, g[GRID_LEVELS - 1] + 1.0) > hi);
        }
    }

    #[test]
    fn selection_returns_q_distinct_channels_with_matching_sign() {
        let (bank, _taus, docs, dim) = toy_bank();
        let mut y = vec![0.0f32; bank.dirs];
        let mut sc = SelectScratch::default();
        let mut out = Vec::new();
        for q in [1usize, 8, 32, 64] {
            bank.project(&docs[..dim], &mut y);
            bank.select(&y, q, &mut sc, &mut out);
            assert_eq!(out.len(), q);
            let mut dirs: Vec<u32> = out.iter().map(|c| c >> 1).collect();
            dirs.sort_unstable();
            dirs.dedup();
            assert_eq!(dirs.len(), q);
            for &c in &out {
                let m = (c >> 1) as usize;
                if c & 1 == 0 {
                    assert!(y[m] >= bank.mean[m]);
                } else {
                    assert!(y[m] <= bank.mean[m]);
                }
            }
        }
    }
}
