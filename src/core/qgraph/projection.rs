// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Query-neighborhood projection, following the graph concept in RoarGraph
//! (Chen et al., 2024, <https://arxiv.org/abs/2408.08933>).

use super::{invalid, Adjacency, BuildParams, Graph, Metric, Rows};
use crate::core::error::HmsError;
use nalgebra::DMatrix;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const TRAINING_LIMIT: usize = 32768;
const QUERY_BLOCK: usize = 32;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct ProjectionStats {
    pub training_queries: usize,
    pub neighbors_per_query: usize,
    pub projected_degree: usize,
    pub original_edges: usize,
    pub forward_edges: usize,
    pub reverse_edges: usize,
    pub added_edges: usize,
    pub total_edges: usize,
    pub mean_degree: f64,
    pub max_degree: usize,
    pub degree_capacity: usize,
}

fn allocation<T>(count: usize) -> Result<(), HmsError> {
    if count
        .checked_mul(std::mem::size_of::<T>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(invalid("projection allocation size overflows".into()));
    }
    Ok(())
}

fn reserved<T>(count: usize) -> Result<Vec<T>, HmsError> {
    allocation::<T>(count)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("cannot allocate projection scratch".into()))?;
    Ok(values)
}

fn lists(count: usize) -> Result<Vec<Vec<u32>>, HmsError> {
    let mut values = reserved(count)?;
    values.resize_with(count, Vec::new);
    Ok(values)
}

fn add_edges(total: usize, edges: usize) -> Result<usize, HmsError> {
    total
        .checked_add(edges)
        .ok_or_else(|| invalid("projection edge count overflows".into()))
}

fn select(
    rows: &Rows<'_>,
    vertex: u32,
    candidates: &mut Vec<u32>,
    existing: &[u32],
    additional: &[u32],
    degree: usize,
    alpha: f32,
) -> Result<Vec<u32>, HmsError> {
    candidates.sort_unstable();
    candidates.dedup();
    candidates.retain(|&id| id != vertex && !existing.contains(&id) && !additional.contains(&id));
    let mut scored = reserved(candidates.len())?;
    for &id in candidates.iter() {
        let distance = rows.d2(vertex, id);
        if !distance.is_finite() {
            return Err(invalid("projection base distance is nonfinite".into()));
        }
        scored.push((distance, id));
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let limit = degree.min(scored.len());
    let mut prunable = reserved(scored.len())?;
    prunable.extend_from_slice(&scored);
    let mut selected = rows.prune(vertex, prunable, alpha, limit);
    for &(_, id) in &scored {
        if selected.len() == limit {
            break;
        }
        if !selected.contains(&id) {
            selected.push(id);
        }
    }
    Ok(selected)
}

pub(super) fn validate_queries(
    queries: &[f32],
    dim: usize,
    metric: Metric,
) -> Result<(), HmsError> {
    if dim == 0 || queries.is_empty() || !queries.len().is_multiple_of(dim) {
        return Err(invalid(
            "training queries must be nonempty rows of graph dimension".into(),
        ));
    }
    if queries.iter().any(|value| !value.is_finite()) {
        return Err(invalid("training queries must be finite".into()));
    }
    if metric == Metric::Angular
        && queries
            .chunks_exact(dim)
            .any(|row| row.iter().all(|&value| value == 0.0))
    {
        return Err(invalid("angular training queries must be nonzero".into()));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Nearest(f64, u32);

impl PartialEq for Nearest {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Nearest {}

impl PartialOrd for Nearest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Nearest {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

impl Graph {
    /// Builds a graph with optional construction queries in the graph's input coordinates.
    /// Queries must be independent of evaluation data. The first 32768 rows guide additional
    /// projected edges; all supplied rows are validated. `None` preserves the legacy build.
    /// MIPS callers must augment base rows and append zero to queries before this call.
    pub fn build_with_training_queries(
        data: &[f32],
        dim: usize,
        params: &BuildParams,
        training_queries: Option<&[f32]>,
    ) -> Result<Self, HmsError> {
        let Some(queries) = training_queries else {
            return Self::build(data, dim, params);
        };
        validate_queries(queries, dim, params.metric)?;
        let mut graph = Self::build(data, dim, params)?;
        graph.apply_training_queries(queries)?;
        Ok(graph)
    }

    /// Owned-data equivalent of [`Graph::build_with_training_queries`].
    pub fn build_owned_with_training_queries(
        data: Vec<f32>,
        dim: usize,
        params: &BuildParams,
        training_queries: Option<&[f32]>,
    ) -> Result<Self, HmsError> {
        let Some(queries) = training_queries else {
            return Self::build_owned(data, dim, params);
        };
        validate_queries(queries, dim, params.metric)?;
        let mut graph = Self::build_owned(data, dim, params)?;
        graph.apply_training_queries(queries)?;
        Ok(graph)
    }

    pub(super) fn apply_training_queries(&mut self, queries: &[f32]) -> Result<(), HmsError> {
        if self.n == 1 {
            return Ok(());
        }
        let count = (queries.len() / self.dim).min(TRAINING_LIMIT);
        let queries = &queries[..count * self.dim];
        let neighbors = self.n.min(100);
        let ids = self.training_neighbors(queries, neighbors)?;
        self.project_neighborhoods(Some(queries), &ids, neighbors, (self.n - 1).min(32))?;
        Ok(())
    }

    fn training_neighbors(&self, queries: &[f32], neighbors: usize) -> Result<Vec<u32>, HmsError> {
        let count = queries.len() / self.dim;
        let base_len = self
            .n
            .checked_mul(self.dim)
            .ok_or_else(|| invalid("training base matrix size overflows".into()))?;
        let mut norms: Vec<f64> = reserved(self.n)?;
        for row in self.vectors.chunks_exact(self.dim) {
            norms.push(row.iter().map(|&value| f64::from(value).powi(2)).sum());
        }
        // DMatrix is column-major: columns hold one coordinate across all base rows, so
        // gemm uses matrixmultiply rather than the scalar transpose-multiply path.
        let mut base = reserved(base_len)?;
        for coordinate in 0..self.dim {
            for (vertex, &norm) in norms.iter().enumerate() {
                let value = f64::from(self.vectors[vertex * self.dim + coordinate]);
                base.push(if self.metric == Metric::Angular && norm > 0.0 {
                    value / norm.sqrt()
                } else {
                    value
                });
            }
        }
        let base = DMatrix::from_vec(self.n, self.dim, base);
        let width = count.min(QUERY_BLOCK);
        let query_len = self
            .dim
            .checked_mul(width)
            .ok_or_else(|| invalid("training query block size overflows".into()))?;
        let product_len = self
            .n
            .checked_mul(width)
            .ok_or_else(|| invalid("training product block size overflows".into()))?;
        let result_len = count
            .checked_mul(neighbors)
            .ok_or_else(|| invalid("training neighborhood size overflows".into()))?;
        let mut query_values = reserved(query_len)?;
        query_values.resize(query_len, 0.0f64);
        let mut query_block = DMatrix::from_vec(self.dim, width, query_values);
        let mut product_values = reserved(product_len)?;
        product_values.resize(product_len, 0.0f64);
        let mut products = DMatrix::from_vec(self.n, width, product_values);
        let mut result = reserved(result_len)?;
        let mut best = BinaryHeap::new();
        best.try_reserve(neighbors)
            .map_err(|_| invalid("cannot allocate training neighbor heap".into()))?;
        let mut ordered: Vec<Nearest> = reserved(neighbors)?;
        for block in queries.chunks(query_len) {
            query_block.as_mut_slice().fill(0.0);
            for (value, &input) in query_block.as_mut_slice().iter_mut().zip(block) {
                *value = f64::from(input);
            }
            products.gemm(1.0, &base, &query_block, 0.0);
            for column in 0..block.len() / self.dim {
                best.clear();
                for vertex in 0..self.n {
                    let product = products[(vertex, column)];
                    let score = match self.metric {
                        Metric::Angular => -product,
                        Metric::Euclidean => norms[vertex] - 2.0 * product,
                    };
                    if !score.is_finite() {
                        return Err(invalid("training neighbor score is nonfinite".into()));
                    }
                    let candidate = Nearest(if score == 0.0 { 0.0 } else { score }, vertex as u32);
                    if best.len() < neighbors {
                        best.push(candidate);
                    } else if candidate < *best.peek().expect("nonempty training neighbor heap") {
                        *best.peek_mut().expect("nonempty training neighbor heap") = candidate;
                    }
                }
                ordered.clear();
                ordered.extend(best.drain());
                ordered.sort_unstable();
                result.extend(ordered.iter().map(|candidate| candidate.1));
            }
        }
        Ok(result)
    }

    /// Projects supplied construction-query neighborhoods into additional base-layer edges.
    /// Queries use graph coordinates; callers must transform original MIPS queries first.
    /// Neighbor lists must come exclusively from construction queries, never evaluation data.
    /// Each neighbor row starts with its nearest base ID, which becomes the projection pivot.
    /// Every old edge is retained, with separate forward and reverse degree quotas.
    /// Increased capacity invalidates the original build's cache key; persist a VGraph.
    pub(super) fn project_neighborhoods(
        &mut self,
        training_queries: Option<&[f32]>,
        query_neighbor_ids: &[u32],
        neighbors_per_query: usize,
        projected_degree: usize,
    ) -> Result<ProjectionStats, HmsError> {
        let original_edges = self
            .adj
            .iter()
            .try_fold(0usize, |sum, ids| add_edges(sum, ids.len()))?;
        let mut stats = ProjectionStats {
            original_edges,
            total_edges: original_edges,
            mean_degree: original_edges as f64 / self.n as f64,
            max_degree: self.adj.iter().map(<[u32]>::len).max().unwrap_or(0),
            degree_capacity: self.degree,
            ..ProjectionStats::default()
        };
        let Some(queries) = training_queries else {
            return Ok(stats);
        };
        if self.dim == 0 || queries.is_empty() || !queries.len().is_multiple_of(self.dim) {
            return Err(invalid(
                "projection queries must be nonempty rows of graph dimension".into(),
            ));
        }
        if queries.iter().any(|value| !value.is_finite()) {
            return Err(invalid("projection queries must be finite".into()));
        }
        if !(2..=self.n).contains(&neighbors_per_query)
            || projected_degree == 0
            || projected_degree >= self.n
        {
            return Err(invalid(
                "projection requires 2 <= neighbors_per_query <= n and 1 <= projected_degree < n"
                    .into(),
            ));
        }
        let query_count = queries.len() / self.dim;
        let neighbor_count = query_count
            .checked_mul(neighbors_per_query)
            .ok_or_else(|| invalid("projection neighbor count overflows".into()))?;
        if query_neighbor_ids.len() != neighbor_count {
            return Err(invalid(
                "projection neighbor matrix has the wrong length".into(),
            ));
        }
        allocation::<u32>(neighbor_count)?;
        let degree_capacity = projected_degree
            .checked_mul(2)
            .and_then(|extra| self.degree.checked_add(extra))
            .filter(|&degree| degree < u32::MAX as usize)
            .ok_or_else(|| invalid("projection degree capacity overflows".into()))?;
        let slots = self
            .n
            .checked_mul(degree_capacity)
            .ok_or_else(|| invalid("projection adjacency size overflows".into()))?;
        allocation::<u32>(slots)?;
        allocation::<u32>(self.n)?;

        let mut ordered = reserved(neighbors_per_query)?;
        for ids in query_neighbor_ids.chunks_exact(neighbors_per_query) {
            if ids.iter().any(|&id| id as usize >= self.n) {
                return Err(invalid(
                    "projection neighbor ID is outside the graph".into(),
                ));
            }
            ordered.clear();
            ordered.extend_from_slice(ids);
            ordered.sort_unstable();
            if ordered.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(invalid(
                    "projection neighbor IDs must be unique in every row".into(),
                ));
            }
        }

        let mut candidates = lists(self.n)?;
        for ids in query_neighbor_ids.chunks_exact(neighbors_per_query) {
            let pooled = &mut candidates[ids[0] as usize];
            pooled
                .try_reserve(ids.len() - 1)
                .map_err(|_| invalid("cannot allocate projection candidates".into()))?;
            pooled.extend_from_slice(&ids[1..]);
        }
        let rows = self.rows();
        let mut forward = lists(self.n)?;
        let mut reverse_candidates = lists(self.n)?;
        for vertex in 0..self.n {
            forward[vertex] = select(
                &rows,
                vertex as u32,
                &mut candidates[vertex],
                self.adj.get(vertex),
                &[],
                projected_degree,
                self.build.alpha,
            )?;
            stats.forward_edges = add_edges(stats.forward_edges, forward[vertex].len())?;
            for &neighbor in &forward[vertex] {
                let incoming = &mut reverse_candidates[neighbor as usize];
                incoming
                    .try_reserve(1)
                    .map_err(|_| invalid("cannot allocate projection reverse candidates".into()))?;
                incoming.push(vertex as u32);
            }
        }
        drop(candidates);
        let mut replacement = Adjacency::new(self.n, degree_capacity);
        let mut combined = reserved(degree_capacity)?;
        stats.max_degree = 0;
        for vertex in 0..self.n {
            let reverse = select(
                &rows,
                vertex as u32,
                &mut reverse_candidates[vertex],
                self.adj.get(vertex),
                &forward[vertex],
                projected_degree,
                self.build.alpha,
            )?;
            stats.reverse_edges = add_edges(stats.reverse_edges, reverse.len())?;
            combined.clear();
            combined.extend_from_slice(self.adj.get(vertex));
            combined.extend_from_slice(&forward[vertex]);
            combined.extend_from_slice(&reverse);
            stats.max_degree = stats.max_degree.max(combined.len());
            replacement.set(vertex, &combined);
        }
        stats.added_edges = add_edges(stats.forward_edges, stats.reverse_edges)?;
        stats.total_edges = add_edges(original_edges, stats.added_edges)?;
        stats.training_queries = query_count;
        stats.neighbors_per_query = neighbors_per_query;
        stats.projected_degree = projected_degree;
        stats.mean_degree = stats.total_edges as f64 / self.n as f64;
        stats.degree_capacity = degree_capacity;
        self.adj = replacement;
        self.degree = degree_capacity;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::qgraph::{Residency, SearchKnobs, VGraph, VSearchParams};

    fn scalar_neighbors(graph: &Graph, queries: &[f32], k: usize) -> Vec<u32> {
        let mut result = Vec::new();
        for query in queries.chunks_exact(graph.dim) {
            let mut scored = Vec::new();
            for (id, row) in graph.vectors.chunks_exact(graph.dim).enumerate() {
                let score = if graph.metric == Metric::Euclidean {
                    row.iter()
                        .zip(query)
                        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                        .sum::<f64>()
                } else {
                    let norm = row
                        .iter()
                        .map(|&value| f64::from(value).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if norm == 0.0 {
                        0.0
                    } else {
                        -row.iter()
                            .zip(query)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>()
                            / norm
                    }
                };
                scored.push(Nearest(if score == 0.0 { 0.0 } else { score }, id as u32));
            }
            scored.sort_unstable();
            result.extend(scored.iter().take(k).map(|candidate| candidate.1));
        }
        result
    }

    fn graph() -> Graph {
        let vectors = vec![
            -10.0, 0.0, -11.0, 0.0, -12.0, 0.0, 10.0, 0.0, 11.0, 0.0, 12.0, 0.0,
        ];
        let mut adj = Adjacency::new(6, 2);
        for (vertex, next) in [1, 2, 0, 4, 5, 3].into_iter().enumerate() {
            adj.set(vertex, &[next]);
        }
        Graph {
            n: 6,
            dim: 2,
            degree: 2,
            metric: Metric::Euclidean,
            norms: vectors
                .as_chunks::<2>()
                .0
                .iter()
                .map(|row| row[0] * row[0])
                .collect(),
            vectors,
            adj,
            entry: 0,
            upper_ids: vec![0, 3],
            layers: vec![vec![vec![1], vec![0]]],
            build: BuildParams {
                metric: Metric::Euclidean,
                degree: 2,
                ..BuildParams::default()
            },
        }
    }

    #[test]
    fn absent_queries_preserve_serialized_bytes_even_with_unused_arguments() {
        let directory = tempfile::tempdir().unwrap();
        let before = directory.path().join("before");
        let after = directory.path().join("after");
        let mut graph = graph();
        graph.save(&before).unwrap();
        let stats = graph
            .project_neighborhoods(None, &[u32::MAX], usize::MAX, usize::MAX)
            .unwrap();
        graph.save(&after).unwrap();
        assert_eq!(
            std::fs::read(before).unwrap(),
            std::fs::read(after).unwrap()
        );
        assert_eq!(stats.added_edges, 0);
        assert_eq!(stats.degree_capacity, 2);
    }

    #[test]
    fn query_neighbor_bridge_preserves_old_edges_and_upper_layers() {
        let mut graph = graph();
        let old = graph.adj.to_lists();
        let upper_ids = graph.upper_ids.clone();
        let layers = graph.layers.clone();
        let stats = graph
            .project_neighborhoods(Some(&[0.0, 0.0]), &[0, 3, 4], 3, 1)
            .unwrap();
        assert!(graph.adj.get(0).contains(&3));
        assert!(graph.adj.get(3).contains(&0));
        for (vertex, previous) in old.iter().enumerate() {
            assert!(previous.iter().all(|id| graph.adj.get(vertex).contains(id)));
            let mut unique = graph.adj.get(vertex).to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), graph.adj.get(vertex).len());
            assert!(!unique.contains(&(vertex as u32)));
        }
        assert_eq!(graph.upper_ids, upper_ids);
        assert_eq!(graph.layers, layers);
        assert_eq!(stats.forward_edges, 1);
        assert_eq!(stats.reverse_edges, 1);
        assert_eq!(stats.original_edges, 6);
        assert_eq!(stats.total_edges, 8);
        assert_eq!(stats.max_degree, 2);
        assert_eq!(stats.degree_capacity, 4);
    }

    #[test]
    fn pooling_is_deterministic_and_independent_of_query_order() {
        let mut first = graph();
        let mut second = graph();
        let queries = [0.0, 0.0, 1.0, 0.0];
        let first_stats = first
            .project_neighborhoods(Some(&queries), &[0, 3, 4, 0, 4, 5], 3, 2)
            .unwrap();
        let second_stats = second
            .project_neighborhoods(Some(&queries), &[0, 4, 5, 0, 3, 4], 3, 2)
            .unwrap();
        assert_eq!(first.adj, second.adj);
        assert_eq!(first_stats, second_stats);
        assert!(first.adj.iter().all(|ids| ids.len() <= 6));
    }

    #[test]
    fn invalid_inputs_leave_the_graph_unchanged() {
        let cases: &[(&[f32], &[u32], usize, usize)] = &[
            (&[], &[], 3, 1),
            (&[0.0], &[0, 3, 4], 3, 1),
            (&[f32::NAN, 0.0], &[0, 3, 4], 3, 1),
            (&[f32::INFINITY, 0.0], &[0, 3, 4], 3, 1),
            (&[0.0, 0.0], &[0, 3], 3, 1),
            (&[0.0, 0.0], &[0, 3, 6], 3, 1),
            (&[0.0, 0.0], &[0, 3, 3], 3, 1),
            (&[0.0, 0.0], &[0, 3, 4], 0, 1),
            (&[0.0, 0.0], &[0], 1, 1),
            (&[0.0, 0.0], &[0, 1, 2, 3, 4, 5, 6], 7, 1),
            (&[0.0, 0.0], &[0, 3, 4], 3, 0),
            (&[0.0, 0.0], &[0, 3, 4], 3, 6),
            (&[0.0, 0.0], &[0, 3, 4], 3, usize::MAX),
        ];
        for &(queries, ids, neighbors, degree) in cases {
            let mut graph = graph();
            let before = graph.structure_hash();
            assert!(graph
                .project_neighborhoods(Some(queries), ids, neighbors, degree)
                .is_err());
            assert_eq!(graph.structure_hash(), before);
            assert_eq!(graph.degree, 2);
        }
    }

    #[test]
    fn neighbor_and_projected_degree_upper_boundaries_are_checked() {
        for neighbors in [2, 5, 6] {
            for degree in [4, 5] {
                let mut graph = graph();
                let ids: Vec<u32> = (0..neighbors as u32).collect();
                graph
                    .project_neighborhoods(Some(&[0.0, 0.0]), &ids, neighbors, degree)
                    .unwrap();
                assert_eq!(graph.degree, 2 + 2 * degree);
            }
        }
        let mut graph = graph();
        graph.degree = usize::MAX - 1;
        assert!(graph
            .project_neighborhoods(Some(&[0.0, 0.0]), &[0, 3], 2, 1)
            .is_err());
    }

    #[test]
    fn block_matrix_neighbors_match_scalar_angular_and_euclidean_metrics() {
        let data = [2.0, 0.0, 0.0, 3.0, -1.0, 1.0, 4.0, 4.0, 0.0, 0.0];
        let mut queries = Vec::new();
        for i in 0..33 {
            queries.extend_from_slice(&[i as f32 + 1.0, (i % 5) as f32 - 2.0]);
        }
        for metric in [Metric::Angular, Metric::Euclidean] {
            let params = BuildParams {
                metric,
                degree: 2,
                build_ef: 8,
                ..Default::default()
            };
            let graph = Graph::build(&data, 2, &params).unwrap();
            for neighbors in [1, 4, 5] {
                assert_eq!(
                    graph.training_neighbors(&queries, neighbors).unwrap(),
                    scalar_neighbors(&graph, &queries, neighbors)
                );
            }
        }
    }

    #[test]
    fn asymmetric_mips_neighbors_match_original_variable_norm_inner_products() {
        let original: [[f32; 2]; 5] =
            [[3.0, 0.0], [0.0, 4.0], [-3.0, 0.0], [0.0, -4.0], [0.0, 0.0]];
        let augmented = [
            3.0, 0.0, 4.0, 0.0, 4.0, 3.0, -3.0, 0.0, 4.0, 0.0, -4.0, 3.0, 0.0, 0.0, 5.0,
        ];
        let queries = [2.0, 1.0, 0.0, -1.0, 3.0, 0.0];
        let params = BuildParams {
            metric: Metric::Euclidean,
            degree: 2,
            build_ef: 8,
            ..Default::default()
        };
        let graph = Graph::build(&augmented, 3, &params).unwrap();
        let mut expected = Vec::new();
        for query in queries.as_chunks::<3>().0 {
            let mut ids: Vec<u32> = (0..5).collect();
            let score = |id: u32| {
                original[id as usize]
                    .iter()
                    .zip(query)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
            };
            ids.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
            expected.extend(ids);
        }
        assert_eq!(graph.training_neighbors(&queries, 5).unwrap(), expected);
    }

    #[test]
    fn untrained_optional_builds_preserve_legacy_cache_bytes() {
        let data = graph().vectors;
        let params = BuildParams {
            metric: Metric::Euclidean,
            degree: 2,
            build_ef: 8,
            ..Default::default()
        };
        let legacy = Graph::build(&data, 2, &params).unwrap();
        let borrowed = Graph::build_with_training_queries(&data, 2, &params, None).unwrap();
        let owned =
            Graph::build_owned_with_training_queries(data.clone(), 2, &params, None).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let mut bytes = None;
        for (id, graph) in [legacy, borrowed, owned].iter().enumerate() {
            let path = directory.path().join(id.to_string());
            graph.save(&path).unwrap();
            let current = std::fs::read(&path).unwrap();
            if let Some(previous) = &bytes {
                assert_eq!(previous, &current);
            }
            bytes = Some(current);
            assert_eq!(
                Graph::load(&data, 2, &path, &params)
                    .unwrap()
                    .structure_hash(),
                graph.structure_hash()
            );
        }
    }

    #[test]
    fn optional_training_matches_projection_of_independent_neighbors() {
        let data = graph().vectors;
        let params = BuildParams {
            metric: Metric::Euclidean,
            degree: 2,
            build_ef: 8,
            ..Default::default()
        };
        let queries = [0.0, 0.0, 8.0, 1.0];
        let mut expected = Graph::build(&data, 2, &params).unwrap();
        let ids = scalar_neighbors(&expected, &queries, 6);
        expected
            .project_neighborhoods(Some(&queries), &ids, 6, 5)
            .unwrap();
        let borrowed =
            Graph::build_with_training_queries(&data, 2, &params, Some(&queries)).unwrap();
        let owned =
            Graph::build_owned_with_training_queries(data, 2, &params, Some(&queries)).unwrap();
        assert_eq!(borrowed.structure_hash(), expected.structure_hash());
        assert_eq!(owned.structure_hash(), expected.structure_hash());
        assert_eq!(borrowed.degree, 12);
        let reversed = [8.0, 1.0, 0.0, 0.0];
        let reordered =
            Graph::build_with_training_queries(&borrowed.vectors, 2, &params, Some(&reversed))
                .unwrap();
        assert_eq!(borrowed.structure_hash(), reordered.structure_hash());
    }

    #[test]
    fn invalid_queries_are_rejected_before_build_including_unused_rows() {
        let data = graph().vectors;
        for metric in [Metric::Angular, Metric::Euclidean] {
            let params = BuildParams {
                metric,
                degree: 2,
                ..Default::default()
            };
            for queries in [
                &[][..],
                &[1.0][..],
                &[f32::NAN, 1.0][..],
                &[f32::INFINITY, 1.0][..],
            ] {
                let error = Graph::build_with_training_queries(&[], 2, &params, Some(queries))
                    .err()
                    .expect("invalid training query");
                assert!(error.to_string().contains("training"));
                assert!(Graph::build_owned_with_training_queries(
                    data.clone(),
                    2,
                    &params,
                    Some(queries)
                )
                .is_err());
            }
            let mut queries = vec![1.0; (TRAINING_LIMIT + 1) * 2];
            *queries.last_mut().unwrap() = f32::NAN;
            assert!(Graph::build_with_training_queries(&data, 2, &params, Some(&queries)).is_err());
            if metric == Metric::Angular {
                assert!(
                    Graph::build_with_training_queries(&data, 2, &params, Some(&[0.0, 0.0]))
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn singleton_training_is_valid_and_adds_no_edges() {
        for metric in [Metric::Angular, Metric::Euclidean] {
            let params = BuildParams {
                metric,
                degree: 2,
                ..Default::default()
            };
            let graph =
                Graph::build_with_training_queries(&[3.0, 4.0], 2, &params, Some(&[1.0, 0.0]))
                    .unwrap();
            assert_eq!(graph.degree, 2);
            assert!(graph.adj.get(0).is_empty());
        }
    }

    #[test]
    fn trained_graph_cache_is_rejected_and_encoded_index_round_trips() {
        let data = graph().vectors;
        let params = BuildParams {
            metric: Metric::Euclidean,
            degree: 2,
            build_ef: 8,
            ..Default::default()
        };
        let queries = [0.0, 0.0, 8.0, 1.0];
        let graph = Graph::build_with_training_queries(&data, 2, &params, Some(&queries)).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("cache");
        std::fs::write(&cache, b"preserve existing cache").unwrap();
        let error = graph.save(&cache).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("VGraph"));
        assert_eq!(std::fs::read(cache).unwrap(), b"preserve existing cache");
        let index = VGraph::from_graph(&graph, true).unwrap();
        let path = directory.path().join("trained-index");
        index.save(&path).unwrap();
        let opened = VGraph::open(&path, Residency::Heap).unwrap();
        assert_eq!(index.index_bytes(), opened.index_bytes());
        let (mut first, mut second) = (
            index.searcher_with(SearchKnobs::OFF),
            opened.searcher_with(SearchKnobs::OFF),
        );
        for query in queries.as_chunks::<2>().0 {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let search = VSearchParams {
                ef: 6,
                rerank: 6,
                ..Default::default()
            };
            let first_count = first.search(query, 3, search, &mut a).unwrap();
            let second_count = second.search(query, 3, search, &mut b).unwrap();
            assert_eq!((first_count, a), (second_count, b));
        }
    }
}
