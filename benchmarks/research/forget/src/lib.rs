// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `warm-research-forget`: exact deletion from a superposition memory with a
//! verifiable certificate.
//!
//! # Mechanism
//!
//! The engine's sparse hypervectors (`EntangledHVec`, `k ~ dim/256` sorted
//! active indices) are bundled by thresholding a per-index sum, after which no
//! single contribution can be removed. [`bundle::CounterBundle`] stores the
//! per-index integer sum unthresholded (`i16` or `i32`, overflow-checked), so
//! insert is `+1` and delete is `-1` at the item's indices: an exact group
//! inverse. The recall view `{ i : count[i] >= tau }` is derived on demand and
//! equals the engine's `bundle_bloom` of the surviving items at `tau = 1`.
//! [`certificate::DeletionCertificate`] binds SHA-256 digests of the state
//! before, the state after and the deleted item, plus a residual bound that
//! places the item's post-delete similarity inside the null distribution of a
//! random item. A verifier holding the post-delete counts and the item
//! reproduces `state_before` by re-adding the item in a streaming digest, which
//! proves the transition was exactly that subtraction.
//!
//! # Data layout
//!
//! See the module docs of [`bundle`] (counts, ledger, view bookkeeping) and
//! [`certificate`] (digest chain, residual null). The binary (`src/main.rs`)
//! runs the kill test and writes one JSON report whose schema is in `RUN.md`.
//!
//! # Invariants
//!
//! * Deterministic: every item, deleted subset and probe derives from the
//!   `--seed` through the engine's `hash_u64`.
//! * Single-thread timing, warmup then medians over `--runs` repeats.
//! * No operation on a bundle can leave it in a state that is not the fold of
//!   a multiset of inserted items.

pub mod bundle;
pub mod certificate;

use serde::Serialize;

/// Summary moments of a score sample; `sd` is the unbiased sample deviation
/// (0 when `n < 2`).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Sample {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub max: f64,
}

/// Moments of `xs`; an empty slice yields zeros.
pub fn sample(xs: &[f64]) -> Sample {
    let n = xs.len();
    if n == 0 {
        return Sample {
            n: 0,
            mean: 0.0,
            sd: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Sample {
        n,
        mean,
        sd: var.max(0.0).sqrt(),
        min,
        max,
    }
}

/// Two-sample z-score of `a` against the `null` sample under the hypothesis
/// that `a` is drawn from the null: `(mean_a - mean_null) / (sd_null *
/// sqrt(1/n_a + 1/n_null))`. Zero when the null has no spread (a saturated
/// view) or either sample is empty.
pub fn z_group(a: &Sample, null: &Sample) -> f64 {
    if a.n == 0 || null.n == 0 || null.sd <= 0.0 {
        return 0.0;
    }
    let se = null.sd * (1.0 / a.n as f64 + 1.0 / null.n as f64).sqrt();
    (a.mean - null.mean) / se
}

/// Median of `xs` (sorts in place); 0 for an empty slice.
pub fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = xs.len() / 2;
    if xs.len() % 2 == 0 {
        (xs[m - 1] + xs[m]) / 2.0
    } else {
        xs[m]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_moments() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut []), 0.0);
        let s = sample(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.n, 4);
        assert!((s.mean - 2.5).abs() < 1e-12);
        assert!((s.sd - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!((s.min, s.max), (1.0, 4.0));
    }

    #[test]
    fn z_group_is_zero_without_spread_and_scales_with_n() {
        let null = sample(&[1.0; 10]);
        let a = sample(&[2.0; 10]);
        assert_eq!(z_group(&a, &null), 0.0);
        let null = sample(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let a = sample(&[1.0, 1.0, 1.0, 1.0]);
        let z4 = z_group(&a, &null);
        let a16 = sample(&[1.0; 16]);
        let z16 = z_group(&a16, &null);
        assert!(z4 > 0.0 && z16 > z4);
    }
}
