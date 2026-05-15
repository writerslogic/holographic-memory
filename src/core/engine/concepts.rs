use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::ConceptCandidate;

/// Minimum Jaccard similarity to merge two vectors into the same concept cluster.
pub const CONCEPT_SIMILARITY_THRESHOLD: f64 = 0.3;

/// Minimum cluster size to emit a concept candidate.
pub const MIN_CONCEPT_CLUSTER_SIZE: usize = 3;

impl HmsCore {
    pub fn synthesize_concepts(&self) -> Vec<ConceptCandidate> {
        let (ids, vectors, _) = self.load_all_vectors();
        if ids.len() < 3 {
            return vec![];
        }

        let n = vectors.len();
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
                if vectors[i].similarity(&vectors[j]) > CONCEPT_SIMILARITY_THRESHOLD {
                    cluster.push(j);
                }
            }
            if cluster.len() >= MIN_CONCEPT_CLUSTER_SIZE {
                for &idx in &cluster {
                    used[idx] = true;
                }
                let cluster_vecs: Vec<&EntangledHVec> =
                    cluster.iter().map(|&idx| &vectors[idx]).collect();
                let centroid = EntangledHVec::bundle(&cluster_vecs);
                let coherence: f64 = cluster_vecs
                    .iter()
                    .map(|v| v.similarity(&centroid))
                    .sum::<f64>()
                    / cluster_vecs.len() as f64;
                let member_ids: Vec<String> = cluster.iter().map(|&idx| ids[idx].clone()).collect();

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
