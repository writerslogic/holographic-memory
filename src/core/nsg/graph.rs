use anyhow::Result;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::NSGIndex;
use crate::core::entangled::EntangledHVec;

const NN_DESCENT_ITERATIONS: usize = 10;
const CONVERGENCE_EPSILON: f64 = 0.001;

/// NN-Descent for K-NN graph construction.
/// Random init + parallel iterative refinement with early termination.
///
/// The inner loop is parallelized with rayon: each node's candidate scoring
/// is independent, so we compute all updates in parallel and then apply them.
pub(super) fn build_knn_graph(vectors: &[EntangledHVec], k_build: usize, seed: u64) -> Vec<Vec<u32>> {
    let n = vectors.len();
    let k = k_build.min(n.saturating_sub(1));
    if n == 0 {
        return Vec::new();
    }

    // 1. Initialize with K random neighbors per node (sample without replacement)
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut graph: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            let mut neighbors = HashSet::with_capacity(k);
            while neighbors.len() < k {
                let j = rng.gen_range(0..n);
                if j != i {
                    neighbors.insert(j as u32);
                }
            }
            neighbors.into_iter().collect()
        })
        .collect();

    // 2. Parallel iterative refinement (NN-Descent)
    for _ in 0..NN_DESCENT_ITERATIONS {
        let changes = AtomicUsize::new(0);

        // Compute updated neighbor lists in parallel. Each node reads from the
        // current (immutable) snapshot of `graph` and produces its own update.
        let updates: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut candidates = HashSet::new();
                for &neigh in &graph[i] {
                    candidates.insert(neigh);
                    for &neigh_neigh in &graph[neigh as usize] {
                        if neigh_neigh as usize != i {
                            candidates.insert(neigh_neigh);
                        }
                    }
                }

                let mut scored: Vec<(u32, u32)> = candidates
                    .into_iter()
                    .map(|c| (vectors[i].hamming(&vectors[c as usize]), c))
                    .collect();

                scored.sort_unstable();
                let updated: Vec<u32> = scored.iter().take(k).map(|&(_, c)| c).collect();

                if updated != graph[i] {
                    changes.fetch_add(1, Ordering::Relaxed);
                }
                updated
            })
            .collect();

        graph = updates;

        let total_changes = changes.load(Ordering::Relaxed);
        if total_changes == 0 {
            break;
        }
        if n > 0 && (total_changes as f64 / n as f64) < CONVERGENCE_EPSILON {
            break;
        }
    }

    graph
}

pub(super) fn prune_edges(
    vectors: &[EntangledHVec],
    knn_graph: &[Vec<u32>],
    max_degree: usize,
) -> Vec<Vec<u32>> {
    let n = vectors.len();
    let mut pruned = Vec::with_capacity(n);

    for i in 0..n {
        let candidates = &knn_graph[i];

        let mut sorted_candidates: Vec<(u32, u32)> = candidates
            .iter()
            .map(|&j| {
                let d = vectors[i].hamming(&vectors[j as usize]);
                (d, j)
            })
            .collect();
        sorted_candidates.sort_unstable();

        let mut selected: Vec<u32> = Vec::new();
        for &(dist_ic, c) in &sorted_candidates {
            if selected.len() >= max_degree {
                break;
            }
            let pruned_by_existing = selected.iter().any(|&s| {
                let dist_sc = vectors[s as usize].hamming(&vectors[c as usize]);
                dist_sc < dist_ic
            });
            if !pruned_by_existing {
                selected.push(c);
            }
        }
        pruned.push(selected);
    }
    pruned
}

pub(super) fn select_navigating_node(vectors: &[EntangledHVec]) -> u32 {
    if vectors.is_empty() {
        return 0;
    }

    // Sample for approximate centroid when dataset is large (avoids O(n*k) bundle)
    const SAMPLE_CAP: usize = 1024;
    let centroid = if vectors.len() <= SAMPLE_CAP {
        EntangledHVec::bundle(vectors)
    } else {
        let step = vectors.len() / SAMPLE_CAP;
        let sample: Vec<EntangledHVec> = vectors
            .iter()
            .step_by(step)
            .take(SAMPLE_CAP)
            .cloned()
            .collect();
        EntangledHVec::bundle(&sample)
    };

    let mut best_idx = 0u32;
    let mut best_dist = u32::MAX;
    for (i, v) in vectors.iter().enumerate() {
        let d = centroid.hamming(v);
        if d < best_dist {
            best_dist = d;
            best_idx = i as u32;
        }
    }
    best_idx
}

pub(super) fn insert_online(
    index: &mut NSGIndex,
    id: &str,
    vector: &EntangledHVec,
) -> Result<()> {
    if !index.trained {
        return Ok(());
    }

    let new_idx = index.vectors.len() as u32;
    index.vectors.push(vector.clone());
    index.id_map.push(id.to_string());

    let ef = index.config.ef_construction;
    let candidates = super::search::greedy_search_internal(index, vector, ef);

    let max_degree = index.config.max_degree;
    let mut sorted: Vec<(u32, u32)> = candidates;
    sorted.sort_unstable();

    let mut selected: Vec<u32> = Vec::new();
    for &(dist_ic, c) in &sorted {
        if selected.len() >= max_degree {
            break;
        }
        let pruned = selected.iter().any(|&s| {
            let dist_sc = index.vectors[s as usize].hamming(&index.vectors[c as usize]);
            dist_sc < dist_ic
        });
        if !pruned {
            selected.push(c);
        }
    }

    index.neighbors.push(selected.clone());
    for &neighbor in &selected {
        let n = neighbor as usize;
        if n < index.neighbors.len() {
            index.neighbors[n].push(new_idx);
            if index.neighbors[n].len() > max_degree {
                // RNG diversity pruning (same rule as forward edges)
                let v_n = &index.vectors[n];
                let mut sorted_candidates: Vec<(u32, u32)> = index.neighbors[n]
                    .iter()
                    .map(|&j| (v_n.hamming(&index.vectors[j as usize]), j))
                    .collect();
                sorted_candidates.sort_unstable();
                let mut pruned: Vec<u32> = Vec::new();
                for &(dist_nc, c) in &sorted_candidates {
                    if pruned.len() >= max_degree {
                        break;
                    }
                    let pruned_by_existing = pruned.iter().any(|&s| {
                        let dist_sc = index.vectors[s as usize].hamming(&index.vectors[c as usize]);
                        dist_sc < dist_nc
                    });
                    if !pruned_by_existing {
                        pruned.push(c);
                    }
                }
                index.neighbors[n] = pruned;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_graph_structure() {
        let vectors: Vec<EntangledHVec> = (0..20)
            .map(|i| EntangledHVec::new_deterministic(1000, i))
            .collect();
        let graph = build_knn_graph(&vectors, 5, 42);
        assert_eq!(graph.len(), 20);
        for (i, neighbors) in graph.iter().enumerate() {
            assert!(neighbors.len() <= 5);
            assert!(!neighbors.contains(&(i as u32)), "No self-links");
        }
    }

    #[test]
    fn pruning_respects_max_degree() {
        let vectors: Vec<EntangledHVec> = (0..30)
            .map(|i| EntangledHVec::new_deterministic(1000, i))
            .collect();
        let knn = build_knn_graph(&vectors, 10, 42);
        let pruned = prune_edges(&vectors, &knn, 6);
        for neighbors in &pruned {
            assert!(neighbors.len() <= 6, "Degree {} > max 6", neighbors.len());
        }
    }

    #[test]
    fn navigating_node_near_centroid() {
        let vectors: Vec<EntangledHVec> = (0..50)
            .map(|i| EntangledHVec::new_deterministic(1000, i))
            .collect();
        let nav = select_navigating_node(&vectors);
        let centroid = EntangledHVec::bundle(&vectors);
        let nav_sim = vectors[nav as usize].similarity(&centroid);
        let mean_sim: f64 =
            vectors.iter().map(|v| v.similarity(&centroid)).sum::<f64>() / vectors.len() as f64;
        assert!(
            nav_sim >= mean_sim,
            "Nav node should be above-average similarity to centroid"
        );
    }
}
