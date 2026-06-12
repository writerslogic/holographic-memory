use anyhow::Result;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use super::inverted_list::InvertedLists;
use super::kmeans::KMeansNystrom;
use super::nystrom::NystromProjector;
use super::pq::PQEncoder;
use super::IVFIndex;
use crate::core::config::IVFConfig;
use crate::core::entangled::EntangledHVec;

impl IVFIndex {
    /// Train the full IVF pipeline from a set of EntangledHVec vectors.
    pub fn train(
        vectors: &[EntangledHVec],
        ids: &[String],
        dim: usize,
        config: &IVFConfig,
    ) -> Result<Self> {
        let n = vectors.len();
        let n_landmarks = config.n_landmarks.min(n);
        let n_clusters = config.n_clusters.min(n);

        // 1. Sample landmarks (seed from n to vary with data size)
        let landmark_seed = n as u64;
        let mut rng = StdRng::seed_from_u64(landmark_seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let landmark_indices = &indices[..n_landmarks];
        let landmarks: Vec<EntangledHVec> = landmark_indices
            .iter()
            .map(|&i| vectors[i].clone())
            .collect();

        // 2. Nyström projection
        let projector = NystromProjector::train(landmarks, config.d_reduced)?;
        let projected = projector.project_batch(vectors);

        // 3. K-means in projected space (different seed to decorrelate from landmarks)
        let kmeans = KMeansNystrom::train(&projected, n_clusters, landmark_seed.wrapping_add(1));

        // 4. Compute assignments
        let assignments: Vec<usize> = projected.iter().map(|p| kmeans.assign(p)).collect();

        // 5. Train PQ codebooks
        let pq = PQEncoder::train(vectors, dim);

        // 6. Build inverted lists
        let lists = InvertedLists::new();
        for (i, vec) in vectors.iter().enumerate() {
            let cluster = assignments[i];
            let codes = pq.encode(vec);
            lists.append(cluster, &ids[i], &codes)?;
        }

        Ok(IVFIndex {
            projector,
            kmeans,
            pq,
            lists: Some(lists),
            dim,
            trained: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::IVFConfig;

    #[test]
    fn test_end_to_end_train() {
        let dim = 1000;
        let n = 500usize;
        let vectors: Vec<EntangledHVec> = (0..n)
            .map(|s| EntangledHVec::new_deterministic(dim, s as u64))
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("vec_{}", i)).collect();

        let config = IVFConfig {
            enabled: true,
            n_clusters: 16,
            n_landmarks: 64,
            d_reduced: 8,
            n_probe: 4,
            auto_threshold: 0,
        };

        let index = IVFIndex::train(&vectors, &ids, dim, &config).unwrap();
        assert!(index.is_trained());
    }
}
