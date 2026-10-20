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
                    stable: false,
                });
            }
        }

        // Centroid refinement: k-means-style reassignment (up to 3 iterations).
        // Recompute centroids and reassign vectors to closest centroid.
        if concepts.len() >= 2 {
            let cluster_indices: Vec<usize> = (0..n).filter(|&i| used[i]).collect();

            // Build initial assignment: vector index -> concept index
            let mut assignment: FxHashMap<usize, usize> = FxHashMap::default();
            for (ci, concept) in concepts.iter().enumerate() {
                for mid in &concept.member_ids {
                    if let Some(pos) = all_ids.iter().position(|id| id == mid) {
                        assignment.insert(pos, ci);
                    }
                }
            }

            let mut centroids: Vec<EntangledHVec> = concepts
                .iter()
                .enumerate()
                .map(|(ci, _)| {
                    let members: Vec<&EntangledHVec> = cluster_indices
                        .iter()
                        .filter(|&&vi| assignment.get(&vi) == Some(&ci))
                        .map(|&vi| &all_vectors[vi])
                        .collect();
                    self.bundle(&members)
                })
                .collect();

            let mut stable = true;
            for _iter in 0..3 {
                let mut changed = false;
                for &vi in &cluster_indices {
                    let mut best_ci = assignment[&vi];
                    let mut best_sim = all_vectors[vi].similarity(&centroids[best_ci]);
                    for (ci, centroid) in centroids.iter().enumerate() {
                        let sim = all_vectors[vi].similarity(centroid);
                        if sim > best_sim {
                            best_sim = sim;
                            best_ci = ci;
                        }
                    }
                    if best_ci != assignment[&vi] {
                        assignment.insert(vi, best_ci);
                        changed = true;
                    }
                }
                if !changed {
                    stable = true;
                    break;
                }
                stable = false;
                // Recompute centroids after reassignment
                for (ci, centroid) in centroids.iter_mut().enumerate() {
                    let members: Vec<&EntangledHVec> = cluster_indices
                        .iter()
                        .filter(|&&vi| assignment.get(&vi) == Some(&ci))
                        .map(|&vi| &all_vectors[vi])
                        .collect();
                    if !members.is_empty() {
                        *centroid = self.bundle(&members);
                    }
                }
            }

            // Rebuild concepts from refined assignments
            for (ci, concept) in concepts.iter_mut().enumerate() {
                let members: Vec<usize> = cluster_indices
                    .iter()
                    .filter(|&&vi| assignment.get(&vi) == Some(&ci))
                    .copied()
                    .collect();
                if members.is_empty() {
                    concept.member_count = 0;
                    concept.member_ids.clear();
                    concept.coherence = 0.0;
                    concept.stable = stable;
                    continue;
                }
                let member_vecs: Vec<&EntangledHVec> =
                    members.iter().map(|&vi| &all_vectors[vi]).collect();
                let coherence: f64 = member_vecs
                    .iter()
                    .map(|v| v.similarity(&centroids[ci]))
                    .sum::<f64>()
                    / member_vecs.len() as f64;
                concept.member_ids = members.iter().map(|&vi| all_ids[vi].clone()).collect();
                concept.centroid_id = concept.member_ids.first().cloned().unwrap_or_default();
                concept.member_count = members.len() as u32;
                concept.coherence = coherence;
                concept.stable = stable;
            }

            // Remove empty concepts
            concepts.retain(|c| c.member_count > 0);
        } else {
            // Single or zero concepts: trivially stable
            for concept in &mut concepts {
                concept.stable = true;
            }
        }

        concepts
    }
}
