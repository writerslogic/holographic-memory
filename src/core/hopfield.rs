// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hopfield-Fenchel-Young (HFYN) energy-based associative memory retrieval.
//!
//! Implements sparse Hopfield retrieval using Tsallis α-entropy transformations
//! (Santos et al., JMLR 2025). For sparse binary vectors, the update rule is:
//!
//!   scores_i = β · |query ∩ pattern_i|       (intersection count)
//!   weights  = entmax_α(scores)               (sparse attention)
//!   result   = patterns with non-zero weights
//!
//! Key properties:
//! - **Exact retrieval**: for well-separated patterns, converges to a single
//!   pattern in one step (margin m = 1/(α-1), requires Δ ≥ m/β).
//! - **Sparse output**: α > 1 produces exact zeros, naturally determining
//!   the number of relevant results (no fixed k needed).
//! - **Exponential capacity**: O(ζ^(D/2)) patterns recoverable for dimension D.

use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;

/// Hopfield retrieval configuration.
#[derive(Clone, Debug)]
pub struct HopfieldConfig {
    /// Inverse temperature. Higher β = sharper retrieval (fewer results).
    /// Must be > 0. Typical range: 10.0–500.0 (scores are Jaccard in [0,1]).
    pub beta: f64,
    /// Tsallis entropy parameter controlling sparsity of attention weights.
    /// α=1: softmax (dense, no exact zeros — equivalent to standard attention).
    /// α=1.5: 1.5-entmax (moderately sparse).
    /// α=2: sparsemax (maximally sparse, margin m=1).
    pub alpha: f64,
    /// Maximum Hopfield iterations for query refinement.
    /// 1 = single-step retrieval (sufficient for well-separated patterns).
    pub max_iter: usize,
}

impl Default for HopfieldConfig {
    fn default() -> Self {
        Self {
            beta: 100.0,
            alpha: 2.0,
            max_iter: 1,
        }
    }
}

/// Perform Hopfield-HFYN retrieval against a set of stored patterns.
///
/// Returns patterns with non-zero entmax weights, sorted by weight descending.
/// The number of results is determined by the sparsity of the entmax output,
/// not by a fixed k — but at most `max_results` are returned.
pub fn hopfield_query(
    query: &EntangledHVec,
    patterns: &[(String, EntangledHVec)],
    config: &HopfieldConfig,
    max_results: usize,
) -> Vec<RetrievalResult> {
    if patterns.is_empty() || config.beta <= 0.0 {
        return Vec::new();
    }

    // Step 1: Compute similarity scores (Jaccard similarity scaled by β)
    // Using Jaccard instead of raw intersection count because multi-scale
    // encoding produces variable-density vectors; raw counts bias toward denser vectors.
    let scores: Vec<f64> = patterns
        .iter()
        .map(|(_, pat)| config.beta * query.similarity(pat))
        .collect();

    // Step 2: Apply entmax transformation
    let weights = if config.alpha <= 1.0 + f64::EPSILON {
        softmax(&scores)
    } else if (config.alpha - 1.5).abs() < f64::EPSILON {
        entmax_15(&scores)
    } else {
        sparsemax(&scores)
    };

    // Step 3: Collect non-zero weights as retrieval results
    let mut results: Vec<RetrievalResult> = weights
        .iter()
        .enumerate()
        .filter(|(_, &w)| w > 1e-12)
        .map(|(i, &w)| RetrievalResult {
            id: patterns[i].0.clone(),
            similarity: w,
        })
        .collect();

    results.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(max_results);
    results
}

/// Iterative Hopfield retrieval with query refinement.
///
/// Each iteration reconstructs the query as a weighted combination of patterns
/// (via weighted bundle), then re-queries. Converges to an attractor basin.
pub fn hopfield_query_iterative(
    query: &EntangledHVec,
    patterns: &[(String, EntangledHVec)],
    config: &HopfieldConfig,
    max_results: usize,
) -> Vec<RetrievalResult> {
    if config.max_iter <= 1 {
        return hopfield_query(query, patterns, config, max_results);
    }

    let mut current_query = query.clone();

    for _ in 0..config.max_iter - 1 {
        let scores: Vec<f64> = patterns
            .iter()
            .map(|(_, pat)| config.beta * current_query.similarity(pat))
            .collect();

        let weights = if config.alpha <= 1.0 + f64::EPSILON {
            softmax(&scores)
        } else if (config.alpha - 1.5).abs() < f64::EPSILON {
            entmax_15(&scores)
        } else {
            sparsemax(&scores)
        };

        // Reconstruct query as weighted bundle of attended patterns
        current_query = weighted_reconstruct(patterns, &weights, query.dim);
        if current_query.indices.is_empty() {
            break;
        }
    }

    hopfield_query(&current_query, patterns, config, max_results)
}

/// Sparsemax transformation (α=2 entmax).
///
/// Projects scores onto the probability simplex with maximum sparsity.
/// Margin m = 1/(α-1) = 1 for α=2.
///
/// Algorithm (Martins & Astudillo, 2016):
/// 1. Sort scores descending
/// 2. Find support size k = max{j : 1 + j·z_(j) > Σ_{i≤j} z_(i)}
/// 3. Threshold τ = (Σ_{i≤k} z_(i) - 1) / k
/// 4. p_i = max(z_i - τ, 0)
fn sparsemax(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    // Sort indices by score descending
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumsum = 0.0;
    let mut support_size = 0;

    for (j, &idx) in sorted_indices.iter().enumerate() {
        cumsum += scores[idx];
        // 1-indexed: j+1
        if 1.0 + (j + 1) as f64 * scores[idx] > cumsum {
            support_size = j + 1;
        } else {
            break;
        }
    }

    if support_size == 0 {
        // All scores identical or degenerate — uniform
        let uniform = 1.0 / n as f64;
        return vec![uniform; n];
    }

    // Recompute cumsum for the support
    let tau_sum: f64 = sorted_indices[..support_size]
        .iter()
        .map(|&i| scores[i])
        .sum();
    let tau = (tau_sum - 1.0) / support_size as f64;

    scores.iter().map(|&s| (s - tau).max(0.0)).collect()
}

/// 1.5-entmax transformation (α=1.5).
///
/// Intermediate sparsity between softmax and sparsemax.
/// Margin m = 1/(α-1) = 2 for α=1.5.
///
/// Uses bisection to find threshold τ such that Σ max(z_i - τ, 0)^(1/(α-1)) = 1.
/// For α=1.5: Σ max(z_i - τ, 0)^2 = 1.
fn entmax_15(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Bisection for τ: find τ such that Σ max(s_i - τ, 0)^2 = 1
    let mut lo = max_score - 1.0;
    let mut hi = max_score;

    // Expand bounds if needed
    while sum_clipped_sq(scores, lo) < 1.0 {
        lo -= 1.0;
    }

    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        let s = sum_clipped_sq(scores, mid);
        if s > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let tau = (lo + hi) / 2.0;
    scores.iter().map(|&s| (s - tau).max(0.0).powi(2)).collect()
}

fn sum_clipped_sq(scores: &[f64], tau: f64) -> f64 {
    scores.iter().map(|&s| (s - tau).max(0.0).powi(2)).sum()
}

/// Standard softmax (α=1 limit). Dense, no exact zeros.
fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        let uniform = 1.0 / scores.len() as f64;
        return vec![uniform; scores.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Reconstruct a query vector from weighted patterns.
///
/// For sparse binary vectors: count weighted frequency of each index across
/// patterns with non-zero weight, keep indices with highest weighted frequency
/// up to the target active count.
fn weighted_reconstruct(
    patterns: &[(String, EntangledHVec)],
    weights: &[f64],
    dim: usize,
) -> EntangledHVec {
    use crate::core::entangled::DEFAULT_RHO_DENOM;

    let target = (dim / DEFAULT_RHO_DENOM).max(1);
    let mut freq: fxhash::FxHashMap<u32, f64> =
        fxhash::FxHashMap::with_capacity_and_hasher(target * 2, Default::default());

    for (i, (_, pat)) in patterns.iter().enumerate() {
        let w = weights[i];
        if w < 1e-12 {
            continue;
        }
        for &idx in &pat.indices {
            *freq.entry(idx).or_insert(0.0) += w;
        }
    }

    let mut scored: Vec<(u32, f64)> = freq.into_iter().collect();
    if scored.len() > target {
        scored.select_nth_unstable_by(target - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(target);
    }

    let mut indices: Vec<u32> = scored.into_iter().map(|(idx, _)| idx).collect();
    indices.sort_unstable();
    EntangledHVec::from_indices(indices, dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pattern(dim: usize, seed: u64) -> EntangledHVec {
        EntangledHVec::new_deterministic(dim, seed)
    }

    #[test]
    fn test_sparsemax_single() {
        let result = sparsemax(&[5.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparsemax_sparsity() {
        // One dominant score should suppress others
        let result = sparsemax(&[10.0, 1.0, 1.0, 1.0]);
        assert!(result[0] > 0.9, "Dominant score should get most weight");
        let nonzero = result.iter().filter(|&&w| w > 1e-10).count();
        assert!(nonzero < 4, "Sparsemax should produce zeros");
    }

    #[test]
    fn test_sparsemax_uniform() {
        // Equal scores → equal weights
        let result = sparsemax(&[3.0, 3.0, 3.0]);
        for &w in &result {
            assert!(
                (w - 1.0 / 3.0).abs() < 1e-10,
                "Equal scores should give uniform weights"
            );
        }
    }

    #[test]
    fn test_sparsemax_sums_to_one() {
        let result = sparsemax(&[5.0, 3.0, 1.0, 0.5, 0.1]);
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-8,
            "Sparsemax output should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_entmax_15_sparsity() {
        let result = entmax_15(&[10.0, 1.0, 1.0, 1.0]);
        let nonzero = result.iter().filter(|&&w| w > 1e-10).count();
        assert!(nonzero <= 3, "1.5-entmax should produce some zeros");
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "1.5-entmax should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_softmax_no_zeros() {
        let result = softmax(&[10.0, 1.0, 1.0, 1.0]);
        for &w in &result {
            assert!(w > 0.0, "Softmax should never produce exact zeros");
        }
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hopfield_exact_retrieval() {
        let dim = 16384;
        let p1 = make_pattern(dim, 100);
        let p2 = make_pattern(dim, 200);
        let p3 = make_pattern(dim, 300);

        let patterns = vec![
            ("p1".to_string(), p1.clone()),
            ("p2".to_string(), p2.clone()),
            ("p3".to_string(), p3),
        ];

        let config = HopfieldConfig {
            beta: 100.0,
            alpha: 2.0,
            max_iter: 1,
        };

        // Query with p1 itself should retrieve p1 with highest weight
        let results = hopfield_query(&p1, &patterns, &config, 10);
        assert!(!results.is_empty(), "Should retrieve at least one pattern");
        assert_eq!(results[0].id, "p1", "Should retrieve the exact pattern");

        // Query with p2 should retrieve p2
        let results = hopfield_query(&p2, &patterns, &config, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "p2");
    }

    #[test]
    fn test_hopfield_sparse_output() {
        let dim = 16384;
        let patterns: Vec<(String, EntangledHVec)> = (0..20)
            .map(|i| (format!("p{}", i), make_pattern(dim, i * 100 + 1)))
            .collect();

        let config = HopfieldConfig {
            beta: 100.0,
            alpha: 2.0,
            max_iter: 1,
        };

        let query = patterns[5].1.clone();
        let results = hopfield_query(&query, &patterns, &config, 20);

        // Sparsemax should produce far fewer than 20 results
        assert!(
            results.len() < 10,
            "Sparsemax should produce sparse output, got {} results",
            results.len()
        );
    }

    #[test]
    fn test_hopfield_iterative_convergence() {
        let dim = 16384;
        let p1 = make_pattern(dim, 100);
        let p2 = make_pattern(dim, 200);

        // Create a noisy query (bind p1 with random noise)
        let noise = make_pattern(dim, 999);
        let noisy_query = p1.bind(&noise);

        let patterns = vec![
            ("p1".to_string(), p1.clone()),
            ("p2".to_string(), p2),
        ];

        let config = HopfieldConfig {
            beta: 100.0,
            alpha: 2.0,
            max_iter: 3,
        };

        let results = hopfield_query_iterative(&noisy_query, &patterns, &config, 10);
        // Should still be able to retrieve something
        assert!(!results.is_empty(), "Iterative query should return results");
    }
}
