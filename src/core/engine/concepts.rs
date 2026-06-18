// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use fxhash::{FxHashMap, FxHashSet};

use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::ConceptCandidate;

impl HmsCore {
    /// Cluster similar vectors and return concept candidates with centroids and coherence scores.
    ///
    /// Uses posting-list candidate generation: only vector pairs sharing at least
    /// one active index are compared, avoiding O(n²) all-pairs at high sparsity.
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

        // For large collections, use posting-list candidate generation
        // to avoid O(n²). For small collections, brute-force is fine.
        let use_postings = n > 500;
        let neighbors: Vec<FxHashSet<usize>> = if use_postings {
            let mut postings: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
            for (i, vec) in all_vectors.iter().enumerate() {
                for &idx in vec.indices() {
                    postings.entry(idx).or_default().push(i);
                }
            }
            let mut nbrs = vec![FxHashSet::default(); n];
            for list in postings.values() {
                for (pos, &a) in list.iter().enumerate() {
                    for &b in &list[pos + 1..] {
                        nbrs[a].insert(b);
                        nbrs[b].insert(a);
                    }
                }
            }
            nbrs
        } else {
            Vec::new()
        };

        for i in 0..n {
            if used[i] {
                continue;
            }
            let mut cluster = vec![i];
            if use_postings {
                for &j in &neighbors[i] {
                    if !used[j]
                        && all_vectors[i].similarity(&all_vectors[j]) > cfg.similarity_threshold
                    {
                        cluster.push(j);
                    }
                }
            } else {
                for j in (i + 1)..n {
                    if !used[j]
                        && all_vectors[i].similarity(&all_vectors[j]) > cfg.similarity_threshold
                    {
                        cluster.push(j);
                    }
                }
            }
            if cluster.len() >= cfg.min_cluster_size {
                for &idx in &cluster {
                    used[idx] = true;
                }
                let cluster_vecs: Vec<&EntangledHVec> =
                    cluster.iter().map(|&idx| &all_vectors[idx]).collect();
                let centroid = self.bundle(&cluster_vecs);
                let coherence: f64 = cluster_vecs
                    .iter()
                    .map(|v| v.similarity(&centroid))
                    .sum::<f64>()
                    / cluster_vecs.len() as f64;
                let member_ids: Vec<String> =
                    cluster.iter().map(|&idx| all_ids[idx].clone()).collect();

                concepts.push(ConceptCandidate {
                    centroid_id: member_ids.first().cloned().unwrap_or_default(),
                    member_count: cluster.len() as u32,
                    coherence,
                    member_ids,
                });
            }
        }
        concepts
    }
}
