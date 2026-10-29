// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bounding-box tree over the stored `f32` embeddings for filtered and hybrid semantic
//! queries. Rows reference the store's chunks through their documents (no copy of the
//! embeddings); node bounds are the exact coordinate-wise `f32` minimum and maximum of the
//! rows below, and a query prunes a node when the outward-rounded `f64` upper bound of its
//! dot product is below the current k-th score. Leaf rows are scored with
//! [`Embedding::cosine`], the one semantic score of a chunk.

use std::{cmp::Ordering, collections::BinaryHeap, mem::size_of, sync::Arc};

use anyhow::{ensure, Context, Result};

use super::{DocumentStore, Embedding, StoredDocument};

pub(super) const MIN_ROWS: usize = 128;
pub(super) const MAX_ROWS: usize = 250_000;
const MAX_DIMENSIONS: usize = 4096;
const MAX_INPUT_VALUES: usize = 64 * 1024 * 1024;
const MAX_BOUND_BYTES: usize = 64 * 1024 * 1024;
const MAX_AUXILIARY_BYTES: usize = 80 * 1024 * 1024;
const LEAF_ROWS: usize = 64;

pub(crate) struct DenseIndexSnapshot {
    generation: u64,
    dimensions: usize,
    embedded_count: usize,
    documents: Vec<Arc<StoredDocument>>,
}

pub(crate) struct TrainedDenseIndex {
    generation: u64,
    index: Arc<DenseIndex>,
}

pub(super) struct DenseIndex {
    pub(super) generation: u64,
    pub(super) dimensions: usize,
    rows: Vec<DenseRow>,
    nodes: Vec<Node>,
}

struct DenseRow {
    document: Arc<StoredDocument>,
    chunk: usize,
    order: usize,
}

impl DenseRow {
    fn stored(&self) -> &Embedding {
        self.document.chunks[self.chunk]
            .embedding
            .as_ref()
            .expect("indexed chunk lost its embedding")
    }

    fn embedding(&self) -> &[f32] {
        self.stored()
    }
}

/// Exact `f32` bounds of the rows in `start..end`: a minimum or maximum of `f32` values is one
/// of them, so no rounding is involved in the bounds themselves.
struct Node {
    low: Vec<f32>,
    high: Vec<f32>,
    start: usize,
    end: usize,
    children: Option<(usize, usize)>,
}

pub(super) struct DenseHit {
    pub(super) document: Arc<StoredDocument>,
    pub(super) chunk: usize,
    pub(super) score: f64,
}

#[derive(Clone, Copy)]
struct RankedHit {
    row: usize,
    order: usize,
    score: f64,
}

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.order == other.order
    }
}
impl Eq for RankedHit {}
impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.order.cmp(&other.order))
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    node: usize,
    upper: f64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.upper.total_cmp(&other.upper) == Ordering::Equal && self.node == other.node
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper
            .total_cmp(&other.upper)
            .then(other.node.cmp(&self.node))
    }
}

pub(crate) fn capture_dense_index(
    store: &DocumentStore,
    dimensions: usize,
) -> Result<Option<DenseIndexSnapshot>> {
    if !(1..=MAX_DIMENSIONS).contains(&dimensions)
        || store.embedded_count() < MIN_ROWS
        || store.dense_index_trained()
    {
        return Ok(None);
    }
    ensure!(
        store.embedded_count() <= MAX_ROWS,
        "dense index exceeds 250000 embedded chunks"
    );
    let mut documents = Vec::new();
    documents
        .try_reserve_exact(store.embedded_documents.len())
        .context("cannot allocate dense snapshot document owners")?;
    for id in &store.embedded_documents {
        documents.push(Arc::clone(
            store
                .documents
                .get(id)
                .expect("embedded document missing from store"),
        ));
    }
    Ok(Some(DenseIndexSnapshot {
        generation: store.generation,
        dimensions,
        embedded_count: store.embedded_count(),
        documents,
    }))
}

pub(crate) fn build_dense_index(snapshot: DenseIndexSnapshot) -> Result<Option<TrainedDenseIndex>> {
    let dimensions = snapshot.dimensions;
    let count = snapshot.embedded_count;
    if !(1..=MAX_DIMENSIONS).contains(&dimensions) || count < MIN_ROWS {
        return Ok(None);
    }
    ensure!(
        count <= MAX_ROWS,
        "dense index exceeds 250000 embedded chunks"
    );
    ensure!(
        snapshot.documents.len() <= count,
        "dense snapshot has more documents than embedded chunks"
    );
    ensure!(
        count
            .checked_mul(dimensions)
            .is_some_and(|values| values <= MAX_INPUT_VALUES),
        "dense index exceeds 67108864 embedding values"
    );
    let leaves = count.div_ceil(LEAF_ROWS).next_power_of_two();
    let max_nodes = leaves * 2 - 1;
    let bounds = max_nodes
        .checked_mul(dimensions)
        .and_then(|n| n.checked_mul(2 * size_of::<f32>()));
    ensure!(
        bounds.is_some_and(|bytes| bytes <= MAX_BOUND_BYTES),
        "dense index bounds exceed 64 MiB"
    );
    let auxiliary = bounds
        .and_then(|bytes| bytes.checked_add(count * size_of::<DenseRow>()))
        .and_then(|bytes| {
            bytes.checked_add(snapshot.documents.len() * size_of::<Arc<StoredDocument>>())
        })
        .and_then(|bytes| bytes.checked_add(max_nodes * size_of::<Node>()));
    ensure!(
        auxiliary.is_some_and(|bytes| bytes <= MAX_AUXILIARY_BYTES),
        "dense index auxiliary memory exceeds 80 MiB"
    );
    let mut rows = Vec::new();
    rows.try_reserve_exact(count)
        .context("cannot allocate dense index rows")?;
    for document in &snapshot.documents {
        for (chunk, stored) in document.chunks.iter().enumerate() {
            if let Some(embedding) = &stored.embedding {
                ensure!(
                    rows.len() < count,
                    "dense snapshot embedded count is too small"
                );
                ensure!(
                    embedding.len() == dimensions
                        && embedding.iter().all(|value| value.is_finite()),
                    "dense index embedding shape or values are invalid"
                );
                rows.push(DenseRow {
                    document: Arc::clone(document),
                    chunk,
                    order: rows.len(),
                });
            }
        }
    }
    ensure!(
        rows.len() == count,
        "dense index embedded count differs from snapshot"
    );
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(max_nodes)
        .context("cannot allocate dense index nodes")?;
    let mut index = DenseIndex {
        generation: snapshot.generation,
        dimensions,
        rows,
        nodes,
    };
    index.build_node(0, count)?;
    Ok(Some(TrainedDenseIndex {
        generation: snapshot.generation,
        index: Arc::new(index),
    }))
}

pub(crate) fn publish_dense_index(store: &mut DocumentStore, trained: &TrainedDenseIndex) -> bool {
    if store.generation != trained.generation {
        return false;
    }
    if !store.dense_index_trained() {
        store.dense_index = Some(Arc::clone(&trained.index));
    }
    true
}

impl DenseIndex {
    fn build_node(&mut self, start: usize, end: usize) -> Result<usize> {
        let mut low = Vec::new();
        low.try_reserve_exact(self.dimensions)
            .context("cannot allocate dense lower bounds")?;
        low.resize(self.dimensions, f32::INFINITY);
        let mut high = Vec::new();
        high.try_reserve_exact(self.dimensions)
            .context("cannot allocate dense upper bounds")?;
        high.resize(self.dimensions, f32::NEG_INFINITY);
        for row in &self.rows[start..end] {
            for (dimension, &value) in row.embedding().iter().enumerate() {
                low[dimension] = low[dimension].min(value);
                high[dimension] = high[dimension].max(value);
            }
        }
        let split = (0..self.dimensions)
            .max_by(|&a, &b| {
                (high[a] - low[a])
                    .total_cmp(&(high[b] - low[b]))
                    .then(b.cmp(&a))
            })
            .expect("dense index has no dimensions");
        let width = high[split] - low[split];
        let node = self.nodes.len();
        self.nodes.push(Node {
            low,
            high,
            start,
            end,
            children: None,
        });
        if end - start > LEAF_ROWS && width > 0.0 {
            let middle = start + (end - start) / 2;
            self.rows[start..end].select_nth_unstable_by(middle - start, |a, b| {
                a.embedding()[split]
                    .total_cmp(&b.embedding()[split])
                    .then(a.order.cmp(&b.order))
            });
            let left = self.build_node(start, middle)?;
            let right = self.build_node(middle, end)?;
            self.nodes[node].children = Some((left, right));
        }
        Ok(node)
    }

    fn upper(&self, node: usize, query: &[f64]) -> f64 {
        let node = &self.nodes[node];
        let mut upper = 0.0_f64;
        // The `f32` endpoints widen exactly; coordinate containment plus outward rounding then
        // bounds the computed sequential `f64` dot of `Embedding::cosine`, including
        // cancellation and underflow. Equality with the kth score must not be pruned.
        for (dimension, &value) in query.iter().enumerate() {
            let endpoint = f64::from(if value >= 0.0 {
                node.high[dimension]
            } else {
                node.low[dimension]
            });
            upper = (upper + (value * endpoint).next_up()).next_up();
        }
        if upper.is_nan() {
            1.0
        } else {
            upper.clamp(-1.0, 1.0)
        }
    }

    pub(super) fn search(
        &self,
        query: &[f64],
        limit: usize,
        minimum: f64,
    ) -> (Vec<DenseHit>, usize) {
        self.search_where(query, limit, minimum, |_| true)
    }

    pub(super) fn search_where(
        &self,
        query: &[f64],
        limit: usize,
        minimum: f64,
        matches: impl Fn(&StoredDocument) -> bool,
    ) -> (Vec<DenseHit>, usize) {
        assert_eq!(
            query.len(),
            self.dimensions,
            "dense query shape differs from trained index"
        );
        assert!(
            (1..=4096).contains(&limit) && minimum.is_finite() && (-1.0..=1.0).contains(&minimum)
        );
        assert!(query
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 1.0));
        let mut pending = BinaryHeap::new();
        pending.push(Candidate {
            node: 0,
            upper: self.upper(0, query),
        });
        let mut nearest = BinaryHeap::<RankedHit>::with_capacity(limit);
        let mut dots = 0;
        while let Some(candidate) = pending.pop() {
            let threshold = if nearest.len() == limit {
                minimum.max(nearest.peek().expect("full dense heap is empty").score)
            } else {
                minimum
            };
            if candidate.upper < threshold {
                continue;
            }
            let node = &self.nodes[candidate.node];
            if let Some((left, right)) = node.children {
                for child in [left, right] {
                    let upper = self.upper(child, query);
                    if upper >= threshold {
                        pending.push(Candidate { node: child, upper });
                    }
                }
            } else {
                for (row, point) in self.rows[node.start..node.end].iter().enumerate() {
                    if !matches(&point.document) {
                        continue;
                    }
                    let score = point.stored().cosine(query);
                    dots += 1;
                    if !score.is_finite() || score < minimum {
                        continue;
                    }
                    let hit = RankedHit {
                        row: node.start + row,
                        order: point.order,
                        score,
                    };
                    if nearest.len() < limit {
                        nearest.push(hit);
                    } else if hit < *nearest.peek().expect("full dense heap is empty") {
                        *nearest.peek_mut().expect("full dense heap is empty") = hit;
                    }
                }
            }
        }
        let hits = nearest
            .into_sorted_vec()
            .into_iter()
            .map(|hit| {
                let row = &self.rows[hit.row];
                DenseHit {
                    document: Arc::clone(&row.document),
                    chunk: row.chunk,
                    score: hit.score,
                }
            })
            .collect();
        (hits, dots)
    }
}

#[cfg(test)]
#[path = "dense_tests.rs"]
mod tests;
