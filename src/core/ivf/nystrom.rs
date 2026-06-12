use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::entangled::EntangledHVec;

#[derive(Serialize, Deserialize)]
pub(crate) struct NystromProjector {
    /// Transform matrix: (n_landmarks × d_reduced) after SVD
    pub(crate) transform: DMatrix<f32>,
    /// Landmark vectors for kernel computation
    pub(crate) landmarks: Vec<EntangledHVec>,
}

impl NystromProjector {
    pub fn train(landmarks: Vec<EntangledHVec>, d_reduced: usize) -> Result<Self> {
        let m = landmarks.len();
        if m < d_reduced {
            return Err(anyhow!(
                "Need at least d_reduced={} landmarks, got {}",
                d_reduced,
                m
            ));
        }

        // Build m×m kernel matrix using exponential Jaccard kernel:
        //   K(i,j) = exp(-(1 - Jaccard(i,j)))
        // Raw Jaccard is NOT positive semi-definite, which breaks Nyström's
        // eigendecomposition. The exponential transform (RBF on Jaccard distance)
        // guarantees PSD by Schoenberg's theorem.
        let mut kernel = DMatrix::<f32>::zeros(m, m);
        for i in 0..m {
            kernel[(i, i)] = 1.0; // exp(0) = 1
            for j in (i + 1)..m {
                let jaccard = landmarks[i].similarity(&landmarks[j]) as f32;
                let k_val = (-(1.0 - jaccard)).exp();
                kernel[(i, j)] = k_val;
                kernel[(j, i)] = k_val;
            }
        }

        // Symmetric eigendecomposition
        let eigen = kernel.symmetric_eigen();

        // Sort eigenvalues descending, filter non-positive, take top d_reduced
        let mut indexed: Vec<(usize, f32)> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Filter: keep only eigenvalues above lambda_max * 1e-6 (relative threshold)
        let lambda_max = indexed.first().map(|&(_, v)| v).unwrap_or(1.0);
        let eig_floor = lambda_max * 1e-6;
        indexed.retain(|&(_, v)| v > eig_floor);
        let d_actual = d_reduced.min(indexed.len());

        let top = &indexed[..d_actual];

        // Build transform: U_d * Λ_d^{-1/2}
        let mut transform = DMatrix::<f32>::zeros(m, d_actual);
        for (col, &(eig_idx, eig_val)) in top.iter().enumerate() {
            let scale = 1.0 / eig_val.sqrt();
            for row in 0..m {
                transform[(row, col)] = eigen.eigenvectors[(row, eig_idx)] * scale;
            }
        }

        Ok(Self {
            transform,
            landmarks,
        })
    }

    /// Project a single vector into the reduced space using exponential Jaccard kernel.
    pub fn project(&self, vec: &EntangledHVec) -> DVector<f32> {
        let m = self.landmarks.len();
        let mut k_vec = DVector::<f32>::zeros(m);
        for (i, lm) in self.landmarks.iter().enumerate() {
            let jaccard = vec.similarity(lm) as f32;
            k_vec[i] = (-(1.0 - jaccard)).exp();
        }
        self.transform.tr_mul(&k_vec)
    }

    /// Project a batch of vectors in parallel.
    pub fn project_batch(&self, vecs: &[EntangledHVec]) -> Vec<DVector<f32>> {
        vecs.par_iter().map(|v| self.project(v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_dimensions() {
        let dim = 1000;
        let d_reduced = 16;
        let landmarks: Vec<EntangledHVec> = (0..32)
            .map(|s| EntangledHVec::new_deterministic(dim, s))
            .collect();
        let proj = NystromProjector::train(landmarks, d_reduced).unwrap();

        let test_vec = EntangledHVec::new_deterministic(dim, 999);
        let projected = proj.project(&test_vec);
        assert_eq!(projected.len(), d_reduced);
    }

    #[test]
    fn test_self_kernel_is_one() {
        let dim = 1000;
        let v = EntangledHVec::new_deterministic(dim, 42);
        let sim = v.similarity(&v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "K(x,x) should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_projection_batch() {
        let dim = 1000;
        let d_reduced = 8;
        let landmarks: Vec<EntangledHVec> = (0..16)
            .map(|s| EntangledHVec::new_deterministic(dim, s))
            .collect();
        let proj = NystromProjector::train(landmarks, d_reduced).unwrap();

        let vecs: Vec<EntangledHVec> = (100..110)
            .map(|s| EntangledHVec::new_deterministic(dim, s))
            .collect();
        let batch = proj.project_batch(&vecs);
        assert_eq!(batch.len(), 10);
        for p in &batch {
            assert_eq!(p.len(), d_reduced);
        }
    }
}
