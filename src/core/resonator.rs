// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Resonator network for factorizing bundled compositions.
//!
//! Given a composite vector C = bind(f1, f2, ..., fn) bundled with noise,
//! and codebooks for each factor, iteratively recovers the individual
//! factors. Capacity scales exponentially with the number of factors
//! (Frady et al., 2020), far exceeding majority-vote unbundling.

use crate::core::entangled::EntangledHVec;

pub struct ResonatorConfig {
    pub max_iter: usize,
    pub convergence_threshold: f64,
}

impl Default for ResonatorConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            convergence_threshold: 0.999,
        }
    }
}

pub struct FactorResult {
    pub factor_idx: usize,
    pub codebook_entry: usize,
    pub similarity: f64,
    pub converged: bool,
    pub iterations: usize,
}

/// Factorize a composite vector into its constituent factors.
///
/// `composite`: the bundled/bound vector to factorize
/// `codebooks`: one codebook per factor position, each containing candidate vectors
/// `config`: iteration parameters
///
/// Returns the best-matching codebook entry for each factor.
pub fn resonator_factorize(
    composite: &EntangledHVec,
    codebooks: &[Vec<EntangledHVec>],
    config: &ResonatorConfig,
) -> Vec<FactorResult> {
    let n_factors = codebooks.len();
    if n_factors == 0 {
        return Vec::new();
    }

    // Initialize estimates: first entry of each codebook
    let mut estimates: Vec<usize> = vec![0; n_factors];
    let mut prev_estimates: Vec<usize> = vec![usize::MAX; n_factors];
    let mut converged = false;
    let mut iter = 0;

    while iter < config.max_iter && !converged {
        iter += 1;
        prev_estimates.copy_from_slice(&estimates);

        for f in 0..n_factors {
            // Compute the product of all OTHER factors' current estimates
            let mut other_product = codebooks[0][estimates[0]].clone();
            let mut started = false;
            for (g, cb) in codebooks.iter().enumerate() {
                if g == f { continue; }
                if !started {
                    other_product = cb[estimates[g]].clone();
                    started = true;
                } else {
                    other_product = other_product.bind(&cb[estimates[g]]);
                }
            }

            // Unbind other factors from composite to isolate factor f's contribution
            let isolated = if n_factors == 1 {
                composite.clone()
            } else {
                composite.bind(&other_product)
            };

            // Find best match in factor f's codebook
            let mut best_idx = 0;
            let mut best_sim = f64::NEG_INFINITY;
            for (i, entry) in codebooks[f].iter().enumerate() {
                let sim = isolated.similarity(entry);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
            estimates[f] = best_idx;
        }

        converged = estimates == prev_estimates;
    }

    estimates.iter().enumerate().map(|(f, &entry_idx)| {
        let other_product = compute_other_product(codebooks, &estimates, f);
        let isolated = if n_factors == 1 {
            composite.clone()
        } else {
            composite.bind(&other_product)
        };
        let sim = isolated.similarity(&codebooks[f][entry_idx]);

        FactorResult {
            factor_idx: f,
            codebook_entry: entry_idx,
            similarity: sim,
            converged,
            iterations: iter,
        }
    }).collect()
}

/// Factorize a bundle of multiple bound composites.
///
/// `bundle`: sum of bind(f1_i, f2_i, ...) for multiple items
/// `codebooks`: candidate vectors per factor
/// `n_items`: how many bound composites are in the bundle
///
/// Returns up to n_items sets of factor assignments, found by iteratively
/// subtracting recovered composites from the residual.
pub fn resonator_unbundle(
    bundle: &EntangledHVec,
    codebooks: &[Vec<EntangledHVec>],
    n_items: usize,
    config: &ResonatorConfig,
) -> Vec<Vec<FactorResult>> {
    let mut results = Vec::new();
    let mut residual = bundle.clone();

    for _ in 0..n_items {
        let factors = resonator_factorize(&residual, codebooks, config);
        if factors.is_empty() { break; }

        let best_sim: f64 = factors.iter().map(|f| f.similarity).sum::<f64>() / factors.len() as f64;
        if best_sim < 0.01 { break; }

        // Reconstruct the recovered composite and subtract from residual
        let mut recovered = codebooks[0][factors[0].codebook_entry].clone();
        for f in &factors[1..] {
            recovered = recovered.bind(&codebooks[f.factor_idx][f.codebook_entry]);
        }

        // "Subtract" by binding out the recovered pattern
        // For XOR-based binding, this is the same as binding again
        residual = residual.bind(&recovered);

        results.push(factors);
    }

    results
}

fn compute_other_product(
    codebooks: &[Vec<EntangledHVec>],
    estimates: &[usize],
    exclude: usize,
) -> EntangledHVec {
    let mut product: Option<EntangledHVec> = None;
    for (g, cb) in codebooks.iter().enumerate() {
        if g == exclude { continue; }
        match product {
            None => product = Some(cb[estimates[g]].clone()),
            Some(ref p) => product = Some(p.bind(&cb[estimates[g]])),
        }
    }
    product.unwrap_or_else(|| EntangledHVec::from_indices(Vec::new(), 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_factor_recovery() {
        let dim = 16384;
        let codebook: Vec<EntangledHVec> = (0..10)
            .map(|i| EntangledHVec::new_deterministic(dim, i * 100))
            .collect();

        let target_idx = 3;
        let composite = codebook[target_idx].clone();
        let config = ResonatorConfig::default();

        let results = resonator_factorize(&composite, &[codebook], &config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].codebook_entry, target_idx);
    }

    #[test]
    fn test_two_factor_recovery() {
        let dim = 16384;
        let cb_a: Vec<EntangledHVec> = (0..10)
            .map(|i| EntangledHVec::new_deterministic(dim, i * 100 + 1))
            .collect();
        let cb_b: Vec<EntangledHVec> = (0..10)
            .map(|i| EntangledHVec::new_deterministic(dim, i * 100 + 2))
            .collect();

        let a_idx = 2;
        let b_idx = 7;
        let composite = cb_a[a_idx].bind(&cb_b[b_idx]);
        let config = ResonatorConfig::default();

        let results = resonator_factorize(&composite, &[cb_a, cb_b], &config);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].codebook_entry, a_idx, "Factor 0 should recover index {}", a_idx);
        assert_eq!(results[1].codebook_entry, b_idx, "Factor 1 should recover index {}", b_idx);
    }

    #[test]
    fn test_three_factor_recovery() {
        let dim = 16384;
        let codebooks: Vec<Vec<EntangledHVec>> = (0..3)
            .map(|f| (0..8).map(|i| EntangledHVec::new_deterministic(dim, f * 1000 + i * 100 + 3)).collect())
            .collect();

        let targets = [1, 4, 6];
        let composite = codebooks[0][targets[0]]
            .bind(&codebooks[1][targets[1]])
            .bind(&codebooks[2][targets[2]]);
        let config = ResonatorConfig::default();

        let results = resonator_factorize(&composite, &codebooks, &config);
        assert_eq!(results.len(), 3);
        for (f, &target) in targets.iter().enumerate() {
            assert_eq!(results[f].codebook_entry, target,
                "Factor {} should recover index {}, got {}", f, target, results[f].codebook_entry);
        }
    }
}
