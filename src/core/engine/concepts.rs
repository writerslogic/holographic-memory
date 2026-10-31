// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::hash::{FastMap, FastSet};

use super::WarmCore;
use crate::core::entangled::EntangledHVec;
use crate::core::query_limits::{QueryBudget, QueryLimits};
use crate::core::types::ConceptCandidate;
use anyhow::Result;

impl WarmCore {
    /// Cluster similar vectors and return concept candidates with centroids and coherence scores.
    ///
    /// Large sparse collections generate candidates from postings, one bounded seed
    /// frontier at a time. Common dimensions can still require quadratic comparison work.
    /// Panics on budget exhaustion; use `synthesize_concepts_with_limits` to handle it.
    pub fn synthesize_concepts(&self) -> Vec<ConceptCandidate> {
        self.synthesize_concepts_with_limits(QueryLimits::default())
            .expect("concept query budget exhausted; use synthesize_concepts_with_limits")
    }

    /// Synthesizes complete concepts or reports a work, frontier or byte limit error.
    pub fn synthesize_concepts_with_limits(
        &self,
        limits: QueryLimits,
    ) -> Result<Vec<ConceptCandidate>> {
        let mut budget = QueryBudget::new(limits);
        let cfg = &self.config.concepts;
        let mut all_ids = Vec::new();
        let mut all_vectors = Vec::new();

        {
            let _transaction = self.mutation_gate.read();
            let shards = self.shards.read();
            shards.try_for_each_shard_indexed(|_, shard| {
                let vectors = shard.vectors.read();
                let registry = shard.registry.read();
                for id in registry.iter() {
                    if let Some(vector) = vectors.get(id) {
                        budget.work(1)?;
                        budget.reserve(
                            id.len()
                                .saturating_mul(2)
                                .saturating_add(vector.indices().len().saturating_mul(8))
                                .saturating_add(128),
                        )?;
                        all_ids.push(id.clone());
                        all_vectors.push(vector.clone());
                    }
                }
                Ok(())
            })?;
        }

        if all_ids.len() < cfg.min_cluster_size {
            return Ok(Vec::new());
        }

        let n = all_vectors.len();
        let mut used = vec![false; n];
        let mut concepts = Vec::new();

        let use_postings = n > 500;
        let postings = if use_postings {
            let mut postings: FastMap<u32, Vec<usize>> = FastMap::default();
            for (i, vec) in all_vectors.iter().enumerate() {
                for &idx in vec.indices() {
                    budget.work(1)?;
                    budget.reserve(16 + if postings.contains_key(&idx) { 0 } else { 64 })?;
                    postings.entry(idx).or_default().push(i);
                }
            }
            Some(postings)
        } else {
            None
        };

        for i in 0..n {
            if used[i] {
                continue;
            }
            budget.reserve(16)?;
            let (candidates, candidate_bytes) =
                concept_candidates(i, &all_vectors, postings.as_ref(), &mut budget)?;
            let mut cluster = vec![i];
            for &j in &candidates {
                budget.work(1)?;
                if !used[j] && all_vectors[i].similarity(&all_vectors[j]) > cfg.similarity_threshold
                {
                    budget.reserve(16)?;
                    cluster.push(j);
                }
            }
            drop(candidates);
            budget.release(candidate_bytes);
            if cluster.len() >= cfg.min_cluster_size {
                budget.frontier(concepts.len().saturating_add(1))?;
                for &idx in &cluster {
                    used[idx] = true;
                }
                let cluster_vecs: Vec<&EntangledHVec> =
                    cluster.iter().map(|&idx| &all_vectors[idx]).collect();
                let centroid = bounded_bundle(self, &cluster_vecs, &mut budget)?;
                budget.work(cluster_vecs.len())?;
                let coherence: f64 = cluster_vecs
                    .iter()
                    .map(|v| v.similarity(&centroid))
                    .sum::<f64>()
                    / cluster_vecs.len() as f64;
                let member_ids: Vec<String> = cluster
                    .iter()
                    .map(|&idx| {
                        budget.reserve(all_ids[idx].len().saturating_mul(2).saturating_add(64))?;
                        Ok(all_ids[idx].clone())
                    })
                    .collect::<Result<_>>()?;

                concepts.push(ConceptCandidate {
                    centroid_id: member_ids.first().cloned().unwrap_or_default(),
                    member_count: cluster.len() as u32,
                    coherence,
                    member_ids,
                    stable: false,
                });
            }
            budget.release(cluster.len().saturating_mul(16));
        }

        // k-means-style centroid refinement, up to 3 iterations.
        if concepts.len() >= 2 {
            budget.reserve(n.saturating_mul(160))?;
            let cluster_indices: Vec<usize> = (0..n).filter(|&i| used[i]).collect();
            let by_id: FastMap<_, _> = all_ids
                .iter()
                .enumerate()
                .map(|(index, id)| (id.as_str(), index))
                .collect();

            // Vector index -> concept index.
            let mut assignment: FastMap<usize, usize> = FastMap::default();
            for (ci, concept) in concepts.iter().enumerate() {
                for mid in &concept.member_ids {
                    budget.work(1)?;
                    if let Some(&pos) = by_id.get(mid.as_str()) {
                        assignment.insert(pos, ci);
                    }
                }
            }

            let mut centroids: Vec<EntangledHVec> = concepts
                .iter()
                .enumerate()
                .map(|(ci, _)| {
                    budget.work(cluster_indices.len())?;
                    let members: Vec<&EntangledHVec> = cluster_indices
                        .iter()
                        .filter(|&&vi| assignment.get(&vi) == Some(&ci))
                        .map(|&vi| &all_vectors[vi])
                        .collect();
                    bounded_bundle(self, &members, &mut budget)
                })
                .collect::<Result<_>>()?;

            let mut stable = true;
            for _iter in 0..3 {
                let mut changed = false;
                for &vi in &cluster_indices {
                    let mut best_ci = assignment[&vi];
                    let mut best_sim = all_vectors[vi].similarity(&centroids[best_ci]);
                    for (ci, centroid) in centroids.iter().enumerate() {
                        budget.work(1)?;
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
                for (ci, centroid) in centroids.iter_mut().enumerate() {
                    budget.work(cluster_indices.len())?;
                    let members: Vec<&EntangledHVec> = cluster_indices
                        .iter()
                        .filter(|&&vi| assignment.get(&vi) == Some(&ci))
                        .map(|&vi| &all_vectors[vi])
                        .collect();
                    if !members.is_empty() {
                        *centroid = bounded_bundle(self, &members, &mut budget)?;
                    }
                }
            }

            for (ci, concept) in concepts.iter_mut().enumerate() {
                budget.work(cluster_indices.len())?;
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
                budget.work(member_vecs.len())?;
                let coherence: f64 = member_vecs
                    .iter()
                    .map(|v| v.similarity(&centroids[ci]))
                    .sum::<f64>()
                    / member_vecs.len() as f64;
                concept.member_ids = members
                    .iter()
                    .map(|&vi| {
                        budget.reserve(all_ids[vi].len().saturating_mul(2).saturating_add(64))?;
                        Ok(all_ids[vi].clone())
                    })
                    .collect::<Result<_>>()?;
                concept.centroid_id = concept.member_ids.first().cloned().unwrap_or_default();
                concept.member_count = members.len() as u32;
                concept.coherence = coherence;
                concept.stable = stable;
            }

            concepts.retain(|c| c.member_count > 0);
        } else {
            // Single or zero concepts: trivially stable
            for concept in &mut concepts {
                concept.stable = true;
            }
        }

        Ok(concepts)
    }
}

fn concept_candidates(
    index: usize,
    vectors: &[EntangledHVec],
    postings: Option<&FastMap<u32, Vec<usize>>>,
    budget: &mut QueryBudget,
) -> Result<(Vec<usize>, usize)> {
    let mut bytes = 0usize;
    let mut candidates = if let Some(postings) = postings {
        let mut selected = FastSet::default();
        for active in vectors[index].indices() {
            for &other in &postings[active] {
                budget.work(1)?;
                if other != index && !selected.contains(&other) {
                    budget.frontier(selected.len().saturating_add(1))?;
                    budget.reserve(48)?;
                    bytes += 48;
                    selected.insert(other);
                }
            }
        }
        selected.into_iter().collect::<Vec<_>>()
    } else {
        budget.frontier(vectors.len() - index - 1)?;
        bytes = (vectors.len() - index - 1).saturating_mul(16);
        budget.reserve(bytes)?;
        (index + 1..vectors.len()).collect()
    };
    budget.work(
        candidates
            .len()
            .saturating_mul(candidates.len().max(1).ilog2() as usize + 1),
    )?;
    candidates.sort_unstable();
    Ok((candidates, bytes))
}

fn bounded_bundle(
    core: &WarmCore,
    vectors: &[&EntangledHVec],
    budget: &mut QueryBudget,
) -> Result<EntangledHVec> {
    let indices = vectors.iter().fold(0usize, |total, vector| {
        total.saturating_add(vector.indices().len())
    });
    let scratch = core
        .dimensions
        .saturating_mul(if core.config.privacy.dp_enabled {
            32
        } else {
            16
        })
        .saturating_add(indices.saturating_mul(16));
    budget.work(indices.saturating_add(if core.config.privacy.dp_enabled {
        core.dimensions
    } else {
        0
    }))?;
    budget.reserve(scratch)?;
    let centroid = core.bundle(vectors);
    budget.release(scratch);
    budget.reserve(
        centroid
            .indices()
            .len()
            .saturating_mul(8)
            .saturating_add(64),
    )?;
    Ok(centroid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::query_limits::QueryLimitExceeded;

    fn fixture(distinct: bool) -> (tempfile::TempDir, WarmCore) {
        let directory = tempfile::tempdir().unwrap();
        let mut config = crate::core::WarmConfig::default();
        config.concepts.similarity_threshold = 0.9;
        let core = WarmCore::new(
            1024,
            Some(directory.path().display().to_string()),
            Some(config),
        )
        .unwrap();
        for index in 0..501 {
            let vector = EntangledHVec::from_indices(
                if distinct {
                    vec![0, index + 1]
                } else {
                    vec![0, 1]
                },
                1024,
            );
            core.shards
                .read()
                .insert(format!("v{index:03}"), vector, 1024)
                .unwrap();
        }
        (directory, core)
    }

    #[test]
    fn hot_dimension_concepts_use_one_candidate_frontier_and_preserve_all_members() {
        let (_directory, core) = fixture(false);
        let concepts = core
            .synthesize_concepts_with_limits(QueryLimits {
                max_bytes: 180_000,
                max_work: 100_000,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(concepts.len(), 1);
        assert_eq!(concepts[0].member_count, 501);
        assert_eq!(concepts[0].coherence, 1.0);
        for frontier in [499, 500, 501] {
            let result = core.synthesize_concepts_with_limits(QueryLimits {
                max_frontier: frontier,
                ..Default::default()
            });
            if frontier == 499 {
                assert_eq!(
                    result
                        .err()
                        .unwrap()
                        .downcast_ref::<QueryLimitExceeded>()
                        .unwrap()
                        .resource,
                    "frontier"
                );
            } else {
                assert_eq!(result.unwrap()[0].member_count, 501);
            }
        }
    }

    #[test]
    fn skewed_concept_work_and_input_bytes_fail_without_partial_answers() {
        let (_directory, core) = fixture(true);
        for (limits, resource) in [
            (
                QueryLimits {
                    max_work: 5_000,
                    ..Default::default()
                },
                "work",
            ),
            (
                QueryLimits {
                    max_bytes: 1_000,
                    ..Default::default()
                },
                "bytes",
            ),
        ] {
            let error = core
                .synthesize_concepts_with_limits(limits)
                .err()
                .expect("budget should reject the complete query");
            assert_eq!(
                error.downcast_ref::<QueryLimitExceeded>().unwrap().resource,
                resource
            );
            assert_eq!(core.vector_count(), 501);
        }
    }
}
