use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::ConceptCandidate;

impl HmsCore {
    /// Cluster similar vectors and return concept candidates with centroids and coherence scores.
    pub fn synthesize_concepts(&self) -> Vec<ConceptCandidate> {
        let cfg = &self.config.concepts;
        let mut all_ids = Vec::new();
        let mut all_vectors = Vec::new();

        let shards = self.shards.read();
        shards.for_each_shard(|shard| {
            let (ids, vectors) = shard.load_all_vectors();
            all_ids.extend(ids);
            all_vectors.extend(vectors);
        });

        if all_ids.len() < cfg.min_cluster_size {
            return vec![];
        }

        let n = all_vectors.len();
        let mut used = vec![false; n];
        let mut concepts = Vec::new();

        for i in 0..n {
            if used[i] {
                continue;
            }
            let mut cluster = vec![i];
            for j in (i + 1)..n {
                if used[j] {
                    continue;
                }
                if all_vectors[i].similarity(&all_vectors[j]) > cfg.similarity_threshold {
                    cluster.push(j);
                }
            }
            if cluster.len() >= cfg.min_cluster_size {
                for &idx in &cluster {
                    used[idx] = true;
                }
                let cluster_vecs: Vec<&EntangledHVec> =
                    cluster.iter().map(|&idx| &all_vectors[idx]).collect();
                let centroid = EntangledHVec::bundle(&cluster_vecs);
                let coherence: f64 = cluster_vecs
                    .iter()
                    .map(|v| v.similarity(&centroid))
                    .sum::<f64>()
                    / cluster_vecs.len() as f64;
                let member_ids: Vec<String> = cluster.iter().map(|&idx| all_ids[idx].clone()).collect();

                concepts.push(ConceptCandidate {
                    centroid_id: member_ids.first().cloned().unwrap_or_default(),
                    member_count: cluster.len() as u32,
                    coherence,
                    member_ids,
                });
            }
            used[i] = true;
        }
        concepts
    }
}
