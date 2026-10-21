// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Vector query dispatch for the document store.
//!
//! Mechanism: a vector-only, unfiltered query does not need the lexical capture in
//! `search.rs`, which clones one `Arc<StoredDocument>` per stored chunk and scores `f64` rows
//! through the document pointers on every query. It is answered from a [`VectorSnapshot`]:
//! an immutable, contiguous `f32` matrix of the embedded chunks captured at one store
//! generation, with an optional vertex graph (`qgraph::VGraph`) over the same rows. Every
//! [`VectorIndex`] maps a query to row ids of that matrix and the snapshot maps rows back to
//! documents. Readers share the snapshot through one `Arc`, so a query holds the store's read
//! lock only to read the generation and the text segment (and, when stale, to validate hits).
//!
//! Data layout: `rows[i]` owns the document of matrix row `i` and the chunk index inside it;
//! `matrix[i * dim .. (i + 1) * dim]` is that chunk's embedding, l2-normalised in `f32` (a
//! zero embedding stays zero and scores 0). `members` holds the address of every document
//! allocation the snapshot covers; the addresses stay unique while the snapshot holds the
//! `Arc`s, so a replaced document (a new allocation under the same id) is never mistaken for
//! the captured one.
//!
//! Consistency: the store bumps `generation` on every insert, replace and delete
//! (`index.rs`), and a snapshot records the generation it was captured at.
//! - current (`store.generation == snapshot.generation`): the rows are the live store, hits
//!   are returned as found.
//! - stale: every hit is validated by identity against the live store under its read lock.
//!   A deleted or replaced document's `Arc` is no longer the one in `store.documents`, so
//!   its rows drop out (the tombstone), and the search over-fetches up to `4k` rows to cover
//!   them. Documents the snapshot does not cover (inserted or replaced since the capture)
//!   are scored exactly in a delta pass over the live documents: one walk over
//!   `store.documents` with a hash probe per document plus a dot product per uncovered
//!   chunk. That pass is bounded by the store size and never costs more than the generic
//!   exact path; when fewer than `k` hits survive, the caller falls back to that path.
//!   `rebuild_recommended` asks maintenance for a fresh capture after
//!   [`REBUILD_AFTER_MUTATIONS`] mutations, so the stale window is bounded by the
//!   maintenance interval.
//! - third implementation slot: [`VectorIndexKind::Scan`] is reserved for the graph-free
//!   scan under research in `benchmarks/research/scan`; nothing implements it here.
//!
//! Scores are `f32` cosines widened to `f64`; ties break on `(document id, chunk)` ascending,
//! like the generic path.

use std::{cmp::Ordering, collections::BinaryHeap, sync::Arc};

use anyhow::{bail, ensure, Context, Result};
use fxhash::FxHashSet;
use parking_lot::{Mutex, MutexGuard, RwLock};
use serde::{Deserialize, Serialize};

use super::index::text_of;
use super::{DocumentResult, DocumentStore, StoredDocument, TextSegment};
use crate::core::qgraph::{BuildParams, Graph, VGraph, VSearchParams};

/// Fewest embedded chunks worth a snapshot; smaller stores take the generic exact path.
pub const SNAPSHOT_MIN_ROWS: usize = 128;
/// Default `graph_min_rows`: a snapshot with fewer rows carries only the exact scan, which
/// beats a graph traversal at that size.
pub const DEFAULT_GRAPH_MIN_ROWS: usize = 4096;
/// Default beam width of a graph query. The accepted qgraph rows reached recall 0.95 at
/// `ef` 736 on glove-100-angular (1.18M vectors, `benchmarks/results/ann_modal_confirm_v1.json`);
/// stores of up to a few hundred thousand embedding rows reach it with a far smaller beam, and
/// `tests/server_ann.rs` holds the default to recall >= 0.95 on its corpus.
pub const DEFAULT_EF: usize = 128;
pub const MAX_EF: usize = 4096;
pub const MAX_K: usize = 1000;
/// Mutations since the capture after which maintenance builds a fresh snapshot.
pub const REBUILD_AFTER_MUTATIONS: u64 = 64;
const MAX_ROWS: usize = 1 << 22;
/// Upper bound on the matrix: 512 Mi `f32` values (2 GiB).
const MAX_MATRIX_VALUES: usize = 512 * 1024 * 1024;

/// Which implementation answered, or may answer, a vector query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorIndexKind {
    /// Exact cosine over the contiguous `f32` matrix.
    Exact,
    /// The vertex graph (`qgraph::VGraph`) with exact re-scoring of the fetched rows.
    Graph,
    /// Reserved for the graph-free scan under research in `benchmarks/research/scan`.
    Scan,
}

/// Route of one vector query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorIndexMode {
    /// The graph when the snapshot has one, the exact scan otherwise.
    #[default]
    Auto,
    /// Always the exact scan (the generic path when no snapshot exists).
    Exact,
    /// The graph, or [`GraphUnavailable`] when none is built.
    Graph,
}

impl VectorIndexMode {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::Auto),
            "exact" => Some(Self::Exact),
            "graph" => Some(Self::Graph),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Exact => "exact",
            Self::Graph => "graph",
        }
    }
}

/// Per-query options of [`crate::HmsCore::search_vectors`].
#[derive(Clone, Debug)]
pub struct VectorQueryOptions {
    pub mode: VectorIndexMode,
    /// Beam width of a graph query (`1..=MAX_EF`); ignored by the exact scan.
    pub ef: usize,
    /// Lowest cosine returned, in `[-1, 1]`.
    pub min_score: f64,
}

impl Default for VectorQueryOptions {
    fn default() -> Self {
        Self {
            mode: VectorIndexMode::Auto,
            ef: DEFAULT_EF,
            min_score: -1.0,
        }
    }
}

/// State of the snapshot relative to the live store.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorIndexStatus {
    /// Embedded chunks the snapshot covers (0 without a snapshot).
    pub rows: usize,
    /// Store generation the snapshot was captured at.
    pub generation: u64,
    /// Whether the snapshot is the live store.
    pub current: bool,
    /// Inserts, replaces and deletes since the capture.
    pub stale_mutations: u64,
    /// Whether the snapshot carries a graph index.
    pub graph: bool,
    /// Whether maintenance should capture a fresh snapshot.
    pub rebuild_recommended: bool,
}

/// A `graph` query reached a store without a graph index.
#[derive(Debug, Clone, Copy)]
pub struct GraphUnavailable;

impl std::fmt::Display for GraphUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no graph index is built; run index maintenance")
    }
}

impl std::error::Error for GraphUnavailable {}

/// A nearest-neighbour index over the rows of one [`VectorSnapshot`].
pub trait VectorIndex: Send + Sync {
    fn kind(&self) -> VectorIndexKind;
    fn rows(&self) -> usize;
    /// The `k` best rows by cosine, best first, as `(row, score)`; `ef` is the beam of an
    /// approximate index and ignored by an exact one. `query` is l2-normalised and has the
    /// snapshot's dimensions.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(u32, f32)>>;
}

#[derive(Clone, Copy)]
struct Ranked {
    score: f32,
    row: u32,
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    /// Greater is worse: a lower score, or the same score at a higher row.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.row.cmp(&other.row))
    }
}

/// Bounded best-`k` set; the heap top is the worst kept candidate.
struct TopK {
    heap: BinaryHeap<Ranked>,
    k: usize,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    fn offer(&mut self, row: u32, score: f32) {
        let candidate = Ranked { score, row };
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(candidate);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if candidate < *worst {
                *worst = candidate;
            }
        }
    }

    fn into_sorted(self) -> Vec<(u32, f32)> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|ranked| (ranked.row, ranked.score))
            .collect()
    }
}

/// Dot product with eight independent accumulators so the loop vectorises.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut lanes = [0.0f32; 8];
    let mut wide_a = a.chunks_exact(8);
    let mut wide_b = b.chunks_exact(8);
    for (x, y) in (&mut wide_a).zip(&mut wide_b) {
        for ((lane, x), y) in lanes.iter_mut().zip(x).zip(y) {
            *lane += x * y;
        }
    }
    let mut sum = lanes.iter().sum::<f32>();
    for (x, y) in wide_a.remainder().iter().zip(wide_b.remainder()) {
        sum += x * y;
    }
    sum
}

/// Writes `values` l2-normalised into `out`; a zero or non-finite vector becomes zeros.
fn write_normalized(values: &[f64], out: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = (value / norm) as f32;
        }
    } else {
        out.fill(0.0);
    }
}

/// Exact cosine over the snapshot matrix, one query per thread.
pub(crate) struct ExactScan {
    dim: usize,
    matrix: Arc<[f32]>,
}

impl ExactScan {
    fn row(&self, row: u32) -> &[f32] {
        let start = row as usize * self.dim;
        &self.matrix[start..start + self.dim]
    }
}

impl VectorIndex for ExactScan {
    fn kind(&self) -> VectorIndexKind {
        VectorIndexKind::Exact
    }

    fn rows(&self) -> usize {
        self.matrix.len() / self.dim
    }

    fn search(&self, query: &[f32], k: usize, _ef: usize) -> Result<Vec<(u32, f32)>> {
        ensure!(
            query.len() == self.dim,
            "query has {} dimensions, the index {}",
            query.len(),
            self.dim
        );
        let mut best = TopK::new(k.min(self.rows()));
        for (row, values) in self.matrix.chunks_exact(self.dim).enumerate() {
            best.offer(row as u32, dot(query, values));
        }
        Ok(best.into_sorted())
    }
}

/// The vertex graph over the snapshot rows; fetched ids are re-scored exactly.
pub(crate) struct GraphIndex {
    graph: VGraph,
    exact: ExactScan,
}

impl VectorIndex for GraphIndex {
    fn kind(&self) -> VectorIndexKind {
        VectorIndexKind::Graph
    }

    fn rows(&self) -> usize {
        self.exact.rows()
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(u32, f32)>> {
        ensure!(
            query.len() == self.exact.dim,
            "query has {} dimensions, the index {}",
            query.len(),
            self.exact.dim
        );
        let k = k.min(self.rows());
        if k == 0 {
            return Ok(Vec::new());
        }
        let ef = ef.clamp(k, MAX_EF.max(k));
        let params = VSearchParams {
            ef,
            rerank: ef,
            ..VSearchParams::default()
        };
        let mut ids = Vec::with_capacity(k);
        self.graph
            .searcher()
            .search(query, k, params, &mut ids)
            .map_err(|e| anyhow::anyhow!("graph search: {e}"))?;
        let mut hits: Vec<(u32, f32)> = ids
            .into_iter()
            .filter(|&id| (id as usize) < self.rows())
            .map(|id| (id, dot(query, self.exact.row(id))))
            .collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        hits.dedup_by(|a, b| a.0 == b.0);
        hits.truncate(k);
        Ok(hits)
    }
}

struct RowRef {
    document: Arc<StoredDocument>,
    chunk: u32,
}

/// The embedded chunks of one store generation with the indexes over them.
pub(crate) struct VectorSnapshot {
    generation: u64,
    dim: usize,
    rows: Vec<RowRef>,
    members: FxHashSet<usize>,
    exact: ExactScan,
    graph: Option<GraphIndex>,
}

impl VectorSnapshot {
    pub(crate) fn rows(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    fn index(&self, mode: VectorIndexMode) -> Result<&dyn VectorIndex> {
        match mode {
            VectorIndexMode::Exact => Ok(&self.exact),
            VectorIndexMode::Graph => self
                .graph
                .as_ref()
                .map(|graph| graph as &dyn VectorIndex)
                .ok_or_else(|| anyhow::Error::new(GraphUnavailable)),
            VectorIndexMode::Auto => Ok(self
                .graph
                .as_ref()
                .map_or(&self.exact as &dyn VectorIndex, |graph| {
                    graph as &dyn VectorIndex
                })),
        }
    }
}

/// The rows copied out of the store under its read lock; [`build_vector_snapshot`] turns it
/// into a snapshot without any lock.
pub(crate) struct VectorCapture {
    generation: u64,
    dim: usize,
    rows: Vec<RowRef>,
    matrix: Vec<f32>,
}

/// Copies every embedded chunk into a contiguous normalised matrix. `None` when the store has
/// fewer than [`SNAPSHOT_MIN_ROWS`] embedded chunks.
pub(crate) fn capture_vector_rows(
    store: &DocumentStore,
    dim: usize,
) -> Result<Option<VectorCapture>> {
    let count = store.embedded_count();
    if dim == 0 || count < SNAPSHOT_MIN_ROWS {
        return Ok(None);
    }
    ensure!(
        count <= MAX_ROWS,
        "vector snapshot exceeds {MAX_ROWS} embedded chunks"
    );
    ensure!(
        count
            .checked_mul(dim)
            .is_some_and(|values| values <= MAX_MATRIX_VALUES),
        "vector snapshot exceeds {MAX_MATRIX_VALUES} embedding values"
    );
    let mut rows = Vec::new();
    rows.try_reserve_exact(count)
        .context("cannot allocate vector snapshot rows")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(count * dim)
        .context("cannot allocate vector snapshot matrix")?;
    for id in &store.embedded_documents {
        let Some(document) = store.documents.get(id) else {
            bail!("embedded document {id:?} is missing from the store");
        };
        for (chunk, stored) in document.chunks.iter().enumerate() {
            let Some(embedding) = &stored.embedding else {
                continue;
            };
            ensure!(
                embedding.len() == dim,
                "stored embedding has {} dimensions, the embedding space {dim}",
                embedding.len()
            );
            ensure!(
                rows.len() < count,
                "store counts {count} embedded chunks but holds more"
            );
            let start = matrix.len();
            matrix.resize(start + dim, 0.0);
            write_normalized(embedding, &mut matrix[start..]);
            rows.push(RowRef {
                document: Arc::clone(document),
                chunk: u32::try_from(chunk).context("chunk index over u32")?,
            });
        }
    }
    ensure!(
        rows.len() == count,
        "store counts {count} embedded chunks but holds {}",
        rows.len()
    );
    Ok(Some(VectorCapture {
        generation: store.generation,
        dim,
        rows,
        matrix,
    }))
}

/// Builds the exact scan and, with at least `graph_min_rows` rows, the vertex graph. The
/// graph build is deterministic (`BuildParams::default()`, fixed seed) and uses the rayon pool.
pub(crate) fn build_vector_snapshot(
    capture: VectorCapture,
    graph_min_rows: usize,
) -> Result<VectorSnapshot> {
    let VectorCapture {
        generation,
        dim,
        rows,
        matrix,
    } = capture;
    let members = rows
        .iter()
        .map(|row| Arc::as_ptr(&row.document) as usize)
        .collect();
    let matrix: Arc<[f32]> = matrix.into();
    let exact = ExactScan {
        dim,
        matrix: Arc::clone(&matrix),
    };
    let graph = if rows.len() >= graph_min_rows.max(SNAPSHOT_MIN_ROWS) {
        let built = Graph::build_owned(matrix.to_vec(), dim, &BuildParams::default())
            .map_err(|e| anyhow::anyhow!("graph build: {e}"))?;
        let encoded =
            VGraph::from_graph(&built, true).map_err(|e| anyhow::anyhow!("graph encode: {e}"))?;
        drop(built);
        ensure!(
            encoded.len() == rows.len(),
            "graph index has {} vertices for {} rows",
            encoded.len(),
            rows.len()
        );
        Some(GraphIndex {
            graph: encoded,
            exact: ExactScan {
                dim,
                matrix: Arc::clone(&matrix),
            },
        })
    } else {
        None
    };
    Ok(VectorSnapshot {
        generation,
        dim,
        rows,
        members,
        exact,
        graph,
    })
}

fn staleness(store_generation: u64, snapshot_generation: u64) -> u64 {
    if store_generation == snapshot_generation {
        0
    } else {
        store_generation.abs_diff(snapshot_generation).max(1)
    }
}

pub(crate) fn vector_index_status(
    snapshot: Option<&VectorSnapshot>,
    store: &DocumentStore,
) -> VectorIndexStatus {
    match snapshot {
        None => VectorIndexStatus {
            rows: 0,
            generation: 0,
            current: false,
            stale_mutations: 0,
            graph: false,
            rebuild_recommended: store.embedded_count() >= SNAPSHOT_MIN_ROWS,
        },
        Some(snapshot) => {
            let stale = staleness(store.generation, snapshot.generation);
            VectorIndexStatus {
                rows: snapshot.rows(),
                generation: snapshot.generation,
                current: stale == 0,
                stale_mutations: stale,
                graph: snapshot.has_graph(),
                rebuild_recommended: stale >= REBUILD_AFTER_MUTATIONS,
            }
        }
    }
}

struct Hit {
    document: Arc<StoredDocument>,
    chunk: u32,
    score: f32,
}

fn to_result(hit: &Hit, segment: Option<&TextSegment>) -> DocumentResult {
    let document = &hit.document;
    let chunk = &document.chunks[hit.chunk as usize];
    let score = f64::from(hit.score).clamp(-1.0, 1.0);
    DocumentResult {
        id: chunk.id.clone(),
        document_id: document.id.clone(),
        source_uri: document.source_uri.clone(),
        version: document.version.clone(),
        metadata: document.metadata.clone(),
        text: text_of(chunk, segment),
        start_byte: chunk.start_byte,
        end_byte: chunk.end_byte,
        score,
        lexical_score: 0.0,
        semantic_score: Some(score),
    }
}

/// Answers an unfiltered vector query from the snapshot. `query` is l2-normalised. `None`
/// means the snapshot is too stale to answer this query (fewer than `k` hits survived
/// validation); the caller takes the generic exact path.
pub(crate) fn search_vector_snapshot(
    snapshot: &VectorSnapshot,
    store: &RwLock<DocumentStore>,
    query: &[f32],
    k: usize,
    options: &VectorQueryOptions,
) -> Result<Option<Vec<DocumentResult>>> {
    ensure!(
        query.len() == snapshot.dim,
        "query has {} dimensions, the index {}",
        query.len(),
        snapshot.dim
    );
    ensure!((1..=MAX_K).contains(&k), "require 1 <= k <= {MAX_K}");
    ensure!(
        (1..=MAX_EF).contains(&options.ef),
        "require 1 <= ef <= {MAX_EF}"
    );
    ensure!(
        options.min_score.is_finite() && (-1.0..=1.0).contains(&options.min_score),
        "minScore must be in [-1,1]"
    );
    let index = snapshot.index(options.mode)?;
    let (generation, segment) = {
        let store = store.read();
        (store.generation, store.text_segment())
    };
    let stale = staleness(generation, snapshot.generation);
    let extra = usize::try_from(stale).unwrap_or(usize::MAX).min(4 * k);
    let fetch = (k + extra).min(snapshot.rows());
    let hits = index.search(query, fetch, options.ef)?;
    let mut found: Vec<Hit> = Vec::with_capacity(fetch);
    if stale == 0 {
        for (row, score) in hits {
            let row = &snapshot.rows[row as usize];
            found.push(Hit {
                document: Arc::clone(&row.document),
                chunk: row.chunk,
                score,
            });
        }
    } else {
        let store = store.read();
        for (row, score) in hits {
            let row = &snapshot.rows[row as usize];
            let live = store
                .get(&row.document.id)
                .is_some_and(|live| Arc::ptr_eq(live, &row.document));
            if live {
                found.push(Hit {
                    document: Arc::clone(&row.document),
                    chunk: row.chunk,
                    score,
                });
            }
        }
        let mut scratch = vec![0.0f32; snapshot.dim];
        for document in store.documents.values() {
            if snapshot
                .members
                .contains(&(Arc::as_ptr(document) as usize))
            {
                continue;
            }
            for (chunk, stored) in document.chunks.iter().enumerate() {
                let Some(embedding) = &stored.embedding else {
                    continue;
                };
                if embedding.len() != snapshot.dim {
                    continue;
                }
                write_normalized(embedding, &mut scratch);
                found.push(Hit {
                    document: Arc::clone(document),
                    chunk: u32::try_from(chunk).context("chunk index over u32")?,
                    score: dot(query, &scratch),
                });
            }
        }
        drop(store);
        if found.len() < k.min(snapshot.rows()) {
            return Ok(None);
        }
    }
    found.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.document.id.cmp(&b.document.id))
            .then(a.chunk.cmp(&b.chunk))
    });
    let results = found
        .iter()
        .filter(|hit| f64::from(hit.score) >= options.min_score)
        .take(k)
        .map(|hit| to_result(hit, segment.as_deref()))
        .collect();
    Ok(Some(results))
}

/// The engine's published snapshot and the build lock.
#[derive(Default)]
pub(crate) struct VectorIndexSlot {
    current: RwLock<Option<Arc<VectorSnapshot>>>,
    building: Mutex<()>,
}

impl VectorIndexSlot {
    pub(crate) fn get(&self) -> Option<Arc<VectorSnapshot>> {
        self.current.read().clone()
    }

    pub(crate) fn publish(&self, snapshot: Arc<VectorSnapshot>) {
        *self.current.write() = Some(snapshot);
    }

    pub(crate) fn clear(&self) {
        *self.current.write() = None;
    }

    /// `None` while another build is running.
    pub(crate) fn try_build(&self) -> Option<MutexGuard<'_, ()>> {
        self.building.try_lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[[f32; 4]]) -> ExactScan {
        let mut values = Vec::new();
        for row in rows {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            values.extend(row.iter().map(|v| v / norm));
        }
        ExactScan {
            dim: 4,
            matrix: values.into(),
        }
    }

    #[test]
    fn exact_scan_orders_by_score_then_row_and_bounds_k() {
        let scan = matrix(&[
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ]);
        let hits = scan.search(&[1.0, 0.0, 0.0, 0.0], 3, 0).unwrap();
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [1, 2, 3]);
        assert!((hits[0].1 - 1.0).abs() < 1e-6 && (hits[2].1 - 0.7071).abs() < 1e-3);
        assert_eq!(scan.search(&[1.0, 0.0, 0.0, 0.0], 10, 0).unwrap().len(), 4);
        assert!(scan.search(&[1.0, 0.0, 0.0], 1, 0).is_err());
    }

    #[test]
    fn dot_matches_the_scalar_sum_with_a_remainder() {
        let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..19).map(|i| 1.0 - i as f32 * 0.1).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - expected).abs() < 1e-4);
    }
}
