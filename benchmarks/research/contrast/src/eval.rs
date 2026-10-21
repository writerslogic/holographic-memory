// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Metrics and timing helpers.
//!
//! * `ndcg_at_k`: graded nDCG with `gain = qrel score`, `discount = log2(rank + 1)`,
//!   ideal DCG from the sorted qrel gains of the query (pytrec_eval / BEIR form);
//!   documents absent from the qrels gain 0.
//! * `roc_auc`: Mann-Whitney AUC of positives over negatives with half credit for ties.
//! * `threshold_at_tar`: the largest threshold accepting at least `tar` of the
//!   positive scores; `rate_at_or_above` is the acceptance rate of a score set.
//! * `time_median_secs`: single-thread wall-clock median over `repeats` runs of a
//!   closure after `warmup` unmeasured runs.
//! * `minmax`: per-list min-max normalisation into `[0, 1]` (constant lists map to 0).
//!
//! All functions are pure; timing uses `std::time::Instant`.

use std::collections::HashMap;
use std::time::Instant;

/// nDCG@k of `ranked` (document ids, best first) against graded `rels`.
pub fn ndcg_at_k(ranked: &[u32], rels: &HashMap<u32, f64>, k: usize) -> f64 {
    let mut gains: Vec<f64> = rels.values().copied().filter(|&g| g > 0.0).collect();
    if gains.is_empty() {
        return 0.0;
    }
    gains.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = gains
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, g)| g / ((i + 2) as f64).log2())
        .sum();
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, d)| rels.get(d).copied().unwrap_or(0.0) / ((i + 2) as f64).log2())
        .sum();
    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

/// ROC AUC of `pos` scores over `neg` scores (ties count one half).
pub fn roc_auc(pos: &[f64], neg: &[f64]) -> f64 {
    if pos.is_empty() || neg.is_empty() {
        return 0.5;
    }
    let mut sorted_neg: Vec<f64> = neg.to_vec();
    sorted_neg.sort_by(|a, b| a.total_cmp(b));
    let mut wins = 0.0f64;
    for &p in pos {
        let below = sorted_neg.partition_point(|&n| n < p);
        let at_or_below = sorted_neg.partition_point(|&n| n <= p);
        wins += below as f64 + 0.5 * (at_or_below - below) as f64;
    }
    wins / (pos.len() as f64 * neg.len() as f64)
}

/// The largest threshold `t` such that at least `tar` of `pos` satisfy `score >= t`.
pub fn threshold_at_tar(pos: &[f64], tar: f64) -> f64 {
    if pos.is_empty() {
        return f64::INFINITY;
    }
    let mut s: Vec<f64> = pos.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    let need = ((tar * s.len() as f64).ceil() as usize).clamp(1, s.len());
    s[need - 1]
}

/// Fraction of `scores` at or above `threshold`.
pub fn rate_at_or_above(scores: &[f64], threshold: f64) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().filter(|&&s| s >= threshold).count() as f64 / scores.len() as f64
}

/// Median of `values` (empty input yields NaN).
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Arithmetic mean (empty input yields 0).
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Median wall-clock seconds of `f` over `repeats` measured runs after `warmup` runs.
pub fn time_median_secs<F: FnMut()>(warmup: usize, repeats: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(repeats.max(1));
    for _ in 0..repeats.max(1) {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64());
    }
    median(&samples)
}

/// Min-max normalisation into `[0, 1]`; a constant list becomes all zeros.
pub fn minmax(values: &mut [f64]) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in values.iter() {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = hi - lo;
    for v in values.iter_mut() {
        *v = if span > 0.0 && span.is_finite() {
            (*v - lo) / span
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_is_one_for_ideal_ranking_and_zero_for_miss() {
        let rels: HashMap<u32, f64> = [(1u32, 2.0), (2u32, 1.0)].into_iter().collect();
        assert!((ndcg_at_k(&[1, 2, 3], &rels, 10) - 1.0).abs() < 1e-12);
        assert_eq!(ndcg_at_k(&[3, 4], &rels, 10), 0.0);
        let swapped = ndcg_at_k(&[2, 1], &rels, 10);
        assert!(swapped > 0.0 && swapped < 1.0);
    }

    #[test]
    fn auc_and_thresholds() {
        assert!((roc_auc(&[2.0, 3.0], &[0.0, 1.0]) - 1.0).abs() < 1e-12);
        assert!((roc_auc(&[1.0], &[1.0]) - 0.5).abs() < 1e-12);
        let pos = [0.1, 0.2, 0.3, 0.4, 0.5];
        let t = threshold_at_tar(&pos, 0.8);
        assert!((t - 0.2).abs() < 1e-12);
        assert!((rate_at_or_above(&pos, t) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn minmax_maps_constant_to_zero() {
        let mut v = [3.0, 3.0];
        minmax(&mut v);
        assert_eq!(v, [0.0, 0.0]);
        let mut w = [1.0, 3.0, 2.0];
        minmax(&mut w);
        assert_eq!(w, [0.0, 1.0, 0.5]);
    }
}
