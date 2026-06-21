// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::DVector;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const MAX_ITERS: usize = 50;
const CONVERGENCE_THRESHOLD: f64 = 1e-4;

#[derive(Serialize, Deserialize)]
pub(crate) struct KMeansNystrom {
    pub(crate) centroids: Vec<DVector<f32>>,
    pub(crate) k: usize,
}

impl KMeansNystrom {
    /// Train k-means with k-means++ initialization on projected (float) vectors.
    pub fn train(data: &[DVector<f32>], k: usize, seed: u64) -> Self {
        let centroids = kmeans_plus_plus_init(data, k, seed);
        let mut km = Self { centroids, k };
        km.lloyds(data);
        km
    }

    /// Assign a single point to its nearest centroid. Returns cluster index.
    pub fn assign(&self, point: &DVector<f32>) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, (point - c).norm_squared()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Return the top `n_probe` nearest clusters sorted by distance (ascending).
    pub fn top_clusters(&self, point: &DVector<f32>, n_probe: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, (point - c).norm_squared()))
            .collect();
        let n = n_probe.min(dists.len());
        if n == 0 {
            return vec![];
        }
        dists.select_nth_unstable_by(n.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        dists.truncate(n);
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.into_iter().map(|(i, _)| i).collect()
    }

    fn lloyds(&mut self, data: &[DVector<f32>]) {
        for _ in 0..MAX_ITERS {
            // Assignment
            let assignments: Vec<usize> = data.iter().map(|p| self.assign(p)).collect();

            // Update centroids
            let dim = self.centroids[0].len();
            let mut sums: Vec<DVector<f32>> =
                (0..self.k).map(|_| DVector::<f32>::zeros(dim)).collect();
            let mut counts = vec![0usize; self.k];

            for (i, point) in data.iter().enumerate() {
                let c = assignments[i];
                sums[c] += point;
                counts[c] += 1;
            }

            let mut max_shift: f64 = 0.0;
            for c in 0..self.k {
                if counts[c] > 0 {
                    let new_centroid = &sums[c] / counts[c] as f32;
                    let shift = (&new_centroid - &self.centroids[c]).norm() as f64;
                    if shift > max_shift {
                        max_shift = shift;
                    }
                    self.centroids[c] = new_centroid;
                } else {
                    // Re-seed empty cluster from the point farthest from its centroid
                    let farthest = data
                        .iter()
                        .map(|p| {
                            let d = self
                                .centroids
                                .iter()
                                .map(|c| (p - c).norm_squared())
                                .fold(f32::MAX, f32::min);
                            d
                        })
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some((idx, _)) = farthest {
                        self.centroids[c] = data[idx].clone();
                        max_shift = f64::MAX; // force another iteration
                    }
                }
            }

            if max_shift < CONVERGENCE_THRESHOLD {
                break;
            }
        }
    }
}

/// K-means++ initialization
fn kmeans_plus_plus_init(data: &[DVector<f32>], k: usize, seed: u64) -> Vec<DVector<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = data.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let mut centroids = Vec::with_capacity(k);
    centroids.push(data[rng.gen_range(0..n)].clone());

    let mut dists = vec![f32::MAX; n];

    for _ in 1..k {
        let last = centroids.last().expect("non-empty after initial push");
        for (i, point) in data.iter().enumerate() {
            let d = (point - last).norm_squared();
            if d < dists[i] {
                dists[i] = d;
            }
        }

        let total: f64 = dists.iter().map(|&d| d as f64).sum();
        if total < 1e-12 {
            centroids.push(data[rng.gen_range(0..n)].clone());
            continue;
        }

        let threshold = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            cumulative += d as f64;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_convergence() {
        let mut data = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);
        for center in &[
            DVector::from_vec(vec![0.0f32, 0.0, 0.0, 0.0]),
            DVector::from_vec(vec![10.0, 10.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 10.0, 10.0]),
        ] {
            for _ in 0..50 {
                let noise = DVector::from_fn(4, |_, _| rng.gen::<f32>() * 0.5);
                data.push(center + noise);
            }
        }

        let km = KMeansNystrom::train(&data, 3, 42);
        assert_eq!(km.centroids.len(), 3);

        for c in &km.centroids {
            let near_a = (c - DVector::from_vec(vec![0.0f32, 0.0, 0.0, 0.0])).norm() < 2.0;
            let near_b = (c - DVector::from_vec(vec![10.0f32, 10.0, 0.0, 0.0])).norm() < 2.0;
            let near_c = (c - DVector::from_vec(vec![0.0f32, 0.0, 10.0, 10.0])).norm() < 2.0;
            assert!(
                near_a || near_b || near_c,
                "Centroid {:?} not near any true center",
                c
            );
        }
    }

    #[test]
    fn test_cluster_assignment() {
        let centroids = vec![
            DVector::from_vec(vec![0.0f32, 0.0]),
            DVector::from_vec(vec![10.0, 10.0]),
        ];
        let km = KMeansNystrom { centroids, k: 2 };

        let p1 = DVector::from_vec(vec![0.1f32, 0.1]);
        let p2 = DVector::from_vec(vec![9.9, 9.9]);
        assert_eq!(km.assign(&p1), 0);
        assert_eq!(km.assign(&p2), 1);
    }

    #[test]
    fn test_top_clusters() {
        let centroids = vec![
            DVector::from_vec(vec![0.0f32, 0.0]),
            DVector::from_vec(vec![5.0, 5.0]),
            DVector::from_vec(vec![10.0, 10.0]),
        ];
        let km = KMeansNystrom { centroids, k: 3 };

        let p = DVector::from_vec(vec![4.0f32, 4.0]);
        let top = km.top_clusters(&p, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], 1);
    }
}
