// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use fxhash::FxHashMap;
use rand::{Rng, SeedableRng};

use crate::core::config::DiffusionConfig;
use crate::core::entangled::{EntangledHVec, DEFAULT_RHO_DENOM};

const DIFFUSION_SEED: u64 = 42;

pub(crate) struct DiffusionFactorizer<'a> {
    codebook: &'a [EntangledHVec],
    config: DiffusionConfig,
}

impl<'a> DiffusionFactorizer<'a> {
    fn new(codebook: &'a [EntangledHVec], config: DiffusionConfig) -> Self {
        Self { codebook, config }
    }

    fn noise_schedule(&self) -> Vec<f64> {
        let n = self.config.steps;
        if n <= 1 {
            return vec![self.config.sigma_max];
        }
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                self.config.sigma_max * (self.config.sigma_min / self.config.sigma_max).powf(t)
            })
            .collect()
    }

    /// Jaccard-weighted KDE score function: computes energy gradient on active index set.
    fn score_function_sparse(&self, x: &EntangledHVec, sigma: f64) -> Vec<(u32, f64)> {
        if self.codebook.is_empty() || sigma < 1e-12 {
            return Vec::new();
        }

        let mut total_weight = 0.0f64;
        let mut weights = Vec::with_capacity(self.codebook.len());

        for entry in self.codebook {
            // Jaccard similarity: |A∩B|/|A∪B| — much better dynamic range for sparse vectors
            // than Hamming-based similarity (which is ~0.99 for all sparse pairs).
            let intersection =
                crate::core::intersection::sparse_intersection_count(&x.indices, &entry.indices);
            let union_size = x.indices.len() + entry.indices.len() - intersection;
            let jaccard = if union_size > 0 {
                intersection as f64 / union_size as f64
            } else {
                1.0
            };
            let weight = (-(1.0 - jaccard) / sigma).exp();
            total_weight += weight;
            weights.push(weight);
        }

        if total_weight < 1e-12 {
            return Vec::new();
        }

        // Sparse gradient: codebook indices attract, unsupported x indices repel
        let mut score_map: FxHashMap<u32, f64> = FxHashMap::default();
        score_map.reserve(x.indices.len() * 2);

        for (i, entry) in self.codebook.iter().enumerate() {
            let w = weights[i] / total_weight;
            for &idx in &entry.indices {
                let current_score = score_map.entry(idx).or_insert(0.0f64);
                *current_score += w;
            }
        }

        // Repulsive term: indices in x with no codebook support get pushed out
        for &idx in &x.indices {
            score_map.entry(idx).or_insert(-1.0);
        }

        let mut scores: Vec<(u32, f64)> = score_map.into_iter().collect();
        scores.sort_unstable_by_key(|s| s.0);
        scores
    }

    /// Sparse Langevin step: avoids dense binarization.
    ///
    /// Inclusion/retention thresholds are expressed as fractions of sigma so
    /// they scale with the noise schedule rather than being magic constants.
    fn langevin_step_sparse(
        &self,
        continuous: &mut Vec<(u32, f64)>,
        sigma: f64,
        x: &EntangledHVec,
        step_counter: u64,
    ) {
        /// New indices must exceed this fraction of sigma to be included.
        /// Set to 0.2*sigma so that at sigma_max=0.5 the threshold is 0.1
        /// (matching the previous hardcoded value) and shrinks at lower noise.
        const INCLUSION_THRESHOLD_FRAC: f64 = 0.2;
        /// Existing indices below this fraction of sigma are pruned.
        /// Set to 0.1*sigma so at sigma_max=0.5 the threshold is 0.05.
        const RETENTION_THRESHOLD_FRAC: f64 = 0.1;

        let step_size = self.config.step_size;
        let scores = self.score_function_sparse(x, sigma);
        let inclusion_threshold = sigma * INCLUSION_THRESHOLD_FRAC;
        let retention_threshold = sigma * RETENTION_THRESHOLD_FRAC;

        let mut score_map: FxHashMap<u32, f64> = scores.into_iter().collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(step_counter);

        for (idx, val) in continuous.iter_mut() {
            if let Some(&s) = score_map.get(idx) {
                *val += step_size * s;
                score_map.remove(idx);
            }
            let noise = gaussian_sample(&mut rng);
            // Noise coefficient includes sigma to approximate annealed step-size
            // scaling (Song & Ermon 2019). With fixed eta, sqrt(sigma) provides
            // natural exploration at high noise and refinement at low noise.
            *val += (2.0 * step_size * sigma).sqrt() * noise;
            *val = val.clamp(0.0, 1.0);
        }

        for (idx, s) in score_map {
            let noise = gaussian_sample(&mut rng);
            let val = (step_size * s + (2.0 * step_size * sigma).sqrt() * noise).clamp(0.0, 1.0);
            if val > inclusion_threshold {
                continuous.push((idx, val));
            }
        }

        continuous.retain(|&(_, v)| v > retention_threshold);
    }

    fn binarize_sparse(continuous: &[(u32, f64)], dim: usize) -> EntangledHVec {
        let target_count = (dim / DEFAULT_RHO_DENOM).max(1);
        let mut sorted = continuous.to_vec();
        if sorted.len() > target_count {
            sorted.select_nth_unstable_by(target_count - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted.truncate(target_count);
        }

        let mut indices: Vec<u32> = sorted.into_iter().map(|(idx, _)| idx).collect();
        indices.sort_unstable();

        EntangledHVec::from_indices(indices, dim)
    }

    fn denoise(&self, noisy: &EntangledHVec, seed_offset: u64) -> EntangledHVec {
        let schedule = self.noise_schedule();
        let dim = noisy.dim;
        let mut continuous: Vec<(u32, f64)> = noisy.indices.iter().map(|&idx| (idx, 1.0)).collect();

        let mut step_counter = DIFFUSION_SEED.wrapping_add(seed_offset);
        for &sigma in &schedule {
            for _ in 0..self.config.n_langevin {
                let current = Self::binarize_sparse(&continuous, dim);
                self.langevin_step_sparse(&mut continuous, sigma, &current, step_counter);
                step_counter = step_counter.wrapping_add(1);
            }
        }
        Self::binarize_sparse(&continuous, dim)
    }

    pub fn factorize(
        config: &DiffusionConfig,
        product: &EntangledHVec,
        domain_codebooks: &[Vec<EntangledHVec>],
        max_iter: usize,
    ) -> Vec<Option<EntangledHVec>> {
        if domain_codebooks.is_empty() {
            return vec![];
        }
        let num_factors = domain_codebooks.len();
        let mut estimates: Vec<EntangledHVec> = (0..num_factors)
            .map(|i| {
                EntangledHVec::new_deterministic(product.dim, DIFFUSION_SEED.wrapping_add(i as u64))
            })
            .collect();

        for iter in 0..max_iter {
            for i in 0..num_factors {
                let mut residual = product.clone();
                for (j, est) in estimates.iter().enumerate() {
                    if j != i {
                        residual = residual.bind(est);
                    }
                }

                let domain_factorizer =
                    DiffusionFactorizer::new(&domain_codebooks[i], config.clone());
                estimates[i] = domain_factorizer.denoise(&residual, iter as u64);
            }
        }

        estimates
            .iter()
            .enumerate()
            .map(|(i, est)| {
                if domain_codebooks[i].is_empty() {
                    return None;
                }
                domain_codebooks[i]
                    .iter()
                    .min_by_key(|entry| est.hamming(entry))
                    .cloned()
            })
            .collect()
    }
}

/// Gaussian sample via Box-Muller transform using a seeded PRNG.
/// Langevin dynamics requires Gaussian noise for the correct stationary distribution.
fn gaussian_sample(rng: &mut rand::rngs::StdRng) -> f64 {
    let u1: f64 = rng.gen_range(1e-10..1.0);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_schedule_monotonic() {
        let config = DiffusionConfig::default();
        let factorizer = DiffusionFactorizer::new(&[], config);
        let schedule = factorizer.noise_schedule();
        assert!(!schedule.is_empty());
        for w in schedule.windows(2) {
            assert!(
                w[0] >= w[1],
                "Schedule should be monotonically decreasing: {} < {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn score_points_toward_codebook() {
        let dim = 10_000;
        let entry = EntangledHVec::new_deterministic(dim, 0);
        let query = EntangledHVec::new_deterministic(dim, 1);

        let codebook = [entry.clone()];
        let factorizer = DiffusionFactorizer::new(&codebook, DiffusionConfig::default());
        let scores = factorizer.score_function_sparse(&query, 0.3);

        // Scores for indices in entry should be positive
        let mut entry_hits = 0;
        let mut entry_total = 0;
        for &(idx, score) in &scores {
            if entry.indices.binary_search(&idx).is_ok() {
                entry_total += 1;
                if score > 0.0 {
                    entry_hits += 1;
                }
            }
        }
        if entry_total > 0 {
            assert!(
                entry_hits as f64 / entry_total as f64 > 0.5,
                "Score should mostly point toward codebook entry"
            );
        }
    }

    #[test]
    fn denoise_recovers_entry() {
        let dim = 10_000;
        let entry = EntangledHVec::new_deterministic(dim, 0);

        let noise = EntangledHVec::new_deterministic(dim, 999);
        let noisy = EntangledHVec::bundle(&[entry.clone(), noise]);

        let config = DiffusionConfig {
            steps: 5,
            n_langevin: 3,
            ..Default::default()
        };
        let codebook = [entry.clone()];
        let factorizer = DiffusionFactorizer::new(&codebook, config);
        let denoised = factorizer.denoise(&noisy, 0);

        let sim_before = entry.similarity(&noisy);
        let sim_after = entry.similarity(&denoised);
        assert!(
            sim_after >= sim_before,
            "Denoised should be closer: before={}, after={}",
            sim_before,
            sim_after
        );
    }

    #[test]
    fn factorize_basic() {
        let dim = 10_000;
        let red = EntangledHVec::new_deterministic(dim, 100);
        let blue = EntangledHVec::new_deterministic(dim, 101);
        let circle = EntangledHVec::new_deterministic(dim, 200);
        let square = EntangledHVec::new_deterministic(dim, 201);

        let product = red.bind(&circle);

        let domain_colors = vec![red.clone(), blue.clone()];
        let domain_shapes = vec![circle.clone(), square.clone()];

        let config = DiffusionConfig {
            steps: 8,
            n_langevin: 5,
            step_size: 0.15,
            ..Default::default()
        };
        let factors =
            DiffusionFactorizer::factorize(&config, &product, &[domain_colors, domain_shapes], 5);

        assert_eq!(factors.len(), 2);
        assert!(factors[0].is_some(), "Color factor should be found");
        assert!(factors[1].is_some(), "Shape factor should be found");

        if let Some(ref color) = factors[0] {
            let sim_red = color.similarity(&red);
            let sim_blue = color.similarity(&blue);
            assert!(
                sim_red > sim_blue,
                "Should recover red: sim_red={}, sim_blue={}",
                sim_red,
                sim_blue
            );
        }
        if let Some(ref shape) = factors[1] {
            let sim_circle = shape.similarity(&circle);
            let sim_square = shape.similarity(&square);
            assert!(
                sim_circle > sim_square,
                "Should recover circle: sim_circle={}, sim_square={}",
                sim_circle,
                sim_square
            );
        }
    }

    #[test]
    fn langevin_step_reduces_distance() {
        let dim = 10_000;
        let target = EntangledHVec::new_deterministic(dim, 0);
        let start = EntangledHVec::new_deterministic(dim, 1);

        let codebook = [target.clone()];
        let factorizer = DiffusionFactorizer::new(&codebook, DiffusionConfig::default());

        let mut continuous: Vec<(u32, f64)> = start.indices.iter().map(|&idx| (idx, 1.0)).collect();
        factorizer.langevin_step_sparse(&mut continuous, 0.01, &start, 42);
        let stepped = DiffusionFactorizer::binarize_sparse(&continuous, dim);

        let dist_before = start.hamming(&target);
        let dist_after = stepped.hamming(&target);
        assert!(
            dist_after <= dist_before,
            "Step should reduce distance: before={}, after={}",
            dist_before,
            dist_after
        );
    }
}
