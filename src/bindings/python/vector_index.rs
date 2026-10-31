// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Python surface of the quantized graph index (`core::qgraph`): `VectorIndex`.
//!
//! The default encoding is the one `public-bench ann-qgraph` measures: 4-bit traversal codes, an 8-bit
//! residual code for the re-rank, cache-line aligned rows, breadth-first vertex order and
//! 3-byte neighbour ids, over a graph built with quantized construction search. Inputs are
//! numpy arrays; build and search run without the GIL.

use std::sync::Arc;

use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::core::error::WarmError;
use crate::core::qgraph::{
    BuildParams, Consolidation, ConsolidationMode, ConsolidationPolicy, Graph, Metric, Residency,
    UpdatableVGraph, UpdateError, VGraph, VSearchParams, VSearcher,
};

/// Graphs below this many vertices store neighbour ids in three bytes.
const THREE_BYTE_IDS_BELOW: usize = (1 << 24) - 1;

fn err(e: WarmError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A built index with the searcher `query` reuses (its visited marks are one word per vector,
/// so a searcher per query would allocate and clear that much every time).
struct Built {
    // Declared before `graph`, so it is dropped first: it borrows `*graph`.
    searcher: VSearcher<'static>,
    graph: Arc<VGraph>,
}

impl Built {
    fn new(graph: VGraph) -> Self {
        let graph = Arc::new(graph);
        // SAFETY: the searcher borrows the `VGraph` inside the `Arc` allocation, whose address
        // does not change when `Built` moves. `Built` holds a strong reference for as long as
        // the searcher exists (field order drops the searcher first), nothing mutates the graph
        // after this point, and the `'static` borrow never leaves this module: callers only get
        // `&mut VSearcher` through `&mut Built`.
        let shared: &'static VGraph = unsafe { &*Arc::as_ptr(&graph) };
        Self {
            searcher: shared.searcher(),
            graph,
        }
    }
}

/// The distance a `VectorIndex` ranks by.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Distance {
    Angular,
    Euclidean,
    /// Largest inner product first. Each row `x` gains the coordinate `sqrt(M^2 - |x|^2)`, with
    /// `M` the largest row norm, and each query a 0, so that the squared Euclidean distance
    /// `|q|^2 + M^2 - 2 <q, x>` orders rows exactly by descending inner product.
    InnerProduct,
}

impl Distance {
    fn name(self) -> &'static str {
        match self {
            Self::Angular => "angular",
            Self::Euclidean => "euclidean",
            Self::InnerProduct => "ip",
        }
    }

    fn graph_metric(self) -> Metric {
        match self {
            Self::Angular => Metric::Angular,
            Self::Euclidean | Self::InnerProduct => Metric::Euclidean,
        }
    }

    /// Coordinates the graph stores per row of `dim` input coordinates.
    fn graph_dim(self, dim: usize) -> usize {
        match self {
            Self::InnerProduct => dim + 1,
            Self::Angular | Self::Euclidean => dim,
        }
    }
}

/// `rows` (row-major, `dim` wide) with the inner-product coordinate appended to each row.
fn augment_rows(rows: &[f32], dim: usize) -> Vec<f32> {
    let norms2: Vec<f64> = rows
        .chunks_exact(dim)
        .map(|r| r.iter().map(|&v| f64::from(v) * f64::from(v)).sum())
        .collect();
    let max2 = norms2.iter().copied().fold(0.0, f64::max);
    let mut out = Vec::with_capacity(norms2.len() * (dim + 1));
    for (r, n2) in rows.chunks_exact(dim).zip(norms2) {
        out.extend_from_slice(r);
        out.push((max2 - n2).max(0.0).sqrt() as f32);
    }
    out
}

fn augment_queries(rows: &[f32], dim: usize) -> Vec<f32> {
    rows.chunks_exact(dim)
        .flat_map(|q| q.iter().copied().chain([0.0]))
        .collect()
}

/// Approximate nearest-neighbour index over float32 vectors: cosine similarity, Euclidean
/// distance or (maximum) inner product.
///
/// `fit` builds it, `set_query_arguments` sets the search effort, `query` and `batch_query`
/// return the ids (row numbers of the fitted matrix) of the `k` nearest rows, nearest first.
#[pyclass(name = "VectorIndex")]
struct PyVectorIndex {
    metric: Distance,
    degree: usize,
    build_ef: usize,
    code_bits: u8,
    residual_bits: u8,
    compact_adjacency: bool,
    direct_ip: bool,
    pool: Option<rayon::ThreadPool>,
    ef: usize,
    rerank: usize,
    reservoir: usize,
    patience: usize,
    dim: usize,
    built: Option<Built>,
    query_buf: Vec<f32>,
}

impl PyVectorIndex {
    fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    fn params(&self) -> VSearchParams {
        VSearchParams {
            ef: self.ef,
            rerank: self.rerank,
            reservoir: self.reservoir,
            patience: self.patience,
            ..VSearchParams::default()
        }
    }

    fn check_k(k: usize) -> PyResult<()> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be positive"));
        }
        Ok(())
    }
}

/// The rows of a 2-D array, row-major. `as_slice` also accepts Fortran-ordered arrays (in column
/// order), so only a C-contiguous array is copied as one slice.
fn row_major(x: &PyReadonlyArray2<'_, f32>) -> Vec<f32> {
    match x.as_slice() {
        Ok(s) if x.is_c_contiguous() => s.to_vec(),
        _ => x.as_array().iter().copied().collect(),
    }
}

/// The `k` nearest rows of every query by a search of `index` at effort `ef` (re-ranking the
/// whole pool), `k` ids per query in query order. A query with fewer than `k` distinct results
/// is an error, since the projection needs complete neighbour lists.
fn approximate_neighbors(
    index: &VGraph,
    queries: &[f32],
    dim: usize,
    k: usize,
    ef: usize,
) -> Result<Vec<u32>, WarmError> {
    let params = VSearchParams {
        ef: ef.max(k),
        rerank: ef.max(k),
        ..VSearchParams::default()
    };
    let nq = queries.len() / dim;
    let chunk = nq.div_ceil(rayon::current_num_threads() * 16).max(1);
    let mut ids = vec![0u32; nq * k];
    ids.par_chunks_mut(chunk * k)
        .zip(queries.par_chunks(chunk * dim))
        .try_for_each(|(out, rows)| {
            let mut searcher = index.searcher();
            let mut found = Vec::with_capacity(k);
            for (q, slot) in rows.chunks_exact(dim).zip(out.chunks_exact_mut(k)) {
                searcher.search(q, k, params, &mut found)?;
                if found.len() != k {
                    return Err(WarmError::InvalidParam {
                        details: format!(
                            "a training query found {} of {k} neighbours",
                            found.len()
                        ),
                    });
                }
                slot.copy_from_slice(&found);
            }
            Ok(())
        })?;
    Ok(ids)
}

/// The ids as a numpy int64 array.
fn ids_array<'py>(py: Python<'py>, ids: &[u32]) -> Bound<'py, PyArray1<i64>> {
    ids.iter()
        .map(|&i| i64::from(i))
        .collect::<Vec<_>>()
        .into_pyarray(py)
}

#[pymethods]
impl PyVectorIndex {
    /// `metric` is "angular" (cosine), "euclidean" or "ip" (largest inner product first).
    /// `degree` is the out-degree of the graph (even),
    /// `build_ef` the construction search width, `threads` the build and batch-query threads
    /// (0 = one per core). `code_bits` selects 4-bit traversal codes (default) or 8-bit
    /// traversal codes for higher routing precision. `residual_bits` selects 8-bit (default),
    /// 4-bit or no residual storage (0); smaller codes may reduce recall. Reranking still
    /// scores the primary codes with a float query when no residual is stored.
    /// `direct_ip` serves original coordinates with negative-IP scores on the lifted graph.
    #[new]
    #[pyo3(signature = (metric="angular", degree=64, build_ef=500, threads=0, code_bits=4, compact_adjacency=false, direct_ip=false, residual_bits=8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        metric: &str,
        degree: usize,
        build_ef: usize,
        threads: usize,
        code_bits: u8,
        compact_adjacency: bool,
        direct_ip: bool,
        residual_bits: u8,
    ) -> PyResult<Self> {
        let metric = match metric {
            "angular" => Distance::Angular,
            "euclidean" => Distance::Euclidean,
            "ip" => Distance::InnerProduct,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown metric {other:?}; VectorIndex supports 'angular', 'euclidean' and 'ip'"
                )))
            }
        };
        if direct_ip && metric != Distance::InnerProduct {
            return Err(PyValueError::new_err("direct_ip requires metric 'ip'"));
        }
        if degree == 0 || !degree.is_multiple_of(2) {
            return Err(PyValueError::new_err(format!(
                "degree must be positive and even, not {degree}"
            )));
        }
        if build_ef == 0 {
            return Err(PyValueError::new_err("build_ef must be positive"));
        }
        if !matches!(code_bits, 4 | 8) {
            return Err(PyValueError::new_err(format!(
                "code_bits must be 4 or 8, not {code_bits}"
            )));
        }
        if !matches!(residual_bits, 0 | 4 | 8) {
            return Err(PyValueError::new_err(format!(
                "residual_bits must be 0, 4 or 8, not {residual_bits}"
            )));
        }
        let pool = match threads {
            0 => None,
            n => Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| PyRuntimeError::new_err(format!("thread pool: {e}")))?,
            ),
        };
        Ok(Self {
            metric,
            degree,
            build_ef,
            code_bits,
            residual_bits,
            compact_adjacency,
            direct_ip,
            pool,
            ef: 64,
            rerank: 64,
            reservoir: 0,
            patience: 0,
            dim: 0,
            built: None,
            query_buf: Vec::new(),
        })
    }

    /// Builds the index over a 2-D array, optionally adding projected edges from separate
    /// training queries in the original coordinates. Replaces any previous index.
    ///
    /// The first `training_limit` training queries are used. With `training_ef` 0 their
    /// `training_neighbors` nearest rows are found by exact scoring of every row (cost
    /// `training_limit * n`); a positive `training_ef` instead searches an index encoded from
    /// the untrained graph at that effort, which scales to millions of queries. Each row gains
    /// up to `projected_degree` forward and as many reverse edges.
    #[pyo3(signature = (x, training_queries=None, training_limit=32768, training_neighbors=100, projected_degree=32, training_ef=0))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyArrayLike2<'_, f32, AllowTypeChange>,
        training_queries: Option<PyArrayLike2<'_, f32, AllowTypeChange>>,
        training_limit: usize,
        training_neighbors: usize,
        projected_degree: usize,
        training_ef: usize,
    ) -> PyResult<()> {
        let (n, dim) = (x.shape()[0], x.shape()[1]);
        if n == 0 || dim == 0 {
            return Err(PyValueError::new_err(format!(
                "fit expects a non-empty matrix, got shape ({n}, {dim})"
            )));
        }
        if training_limit == 0 || training_neighbors < 2 || projected_degree == 0 {
            return Err(PyValueError::new_err(
                "training_limit and projected_degree must be positive and training_neighbors at least 2",
            ));
        }
        // One copy, made while the array is borrowed under the GIL; the build normalizes it in
        // place and keeps it until the encoding is done. Inner-product rows are augmented
        // straight from a C-contiguous array, so the plain copy never coexists with the
        // augmented one.
        let data = match (self.metric, x.as_slice()) {
            (Distance::InnerProduct, Ok(s)) if x.is_c_contiguous() => augment_rows(s, dim),
            (Distance::InnerProduct, _) => augment_rows(&row_major(&x), dim),
            _ => row_major(&x),
        };
        if data.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "fit base vectors contain non-finite values",
            ));
        }
        let training_data = training_queries
            .map(|queries| {
                let (count, query_dim) = (queries.shape()[0], queries.shape()[1]);
                if count == 0 || query_dim != dim {
                    return Err(PyValueError::new_err(format!(
                        "training queries must be non-empty with {dim} dimensions, got shape ({count}, {query_dim})"
                    )));
                }
                let used = count.min(training_limit) * dim;
                let rows: Vec<f32> = match queries.as_slice() {
                    Ok(s) if queries.is_c_contiguous() => s[..used].to_vec(),
                    _ => queries.as_array().iter().take(used).copied().collect(),
                };
                if rows.iter().any(|v| !v.is_finite()) {
                    return Err(PyValueError::new_err("training queries contain non-finite values"));
                }
                if self.metric == Distance::Angular
                    && rows.chunks_exact(dim).any(|row| row.iter().all(|&v| v == 0.0))
                {
                    return Err(PyValueError::new_err("training queries contain a zero-norm row"));
                }
                Ok(rows)
            })
            .transpose()?;
        let params = BuildParams {
            metric: self.metric.graph_metric(),
            degree: self.degree,
            build_ef: self.build_ef,
            quantized_search: true,
            ..BuildParams::default()
        };
        self.built = None;
        let id_bytes = if n < THREE_BYTE_IDS_BELOW { 3 } else { 4 };
        let metric = self.metric;
        let code_bits = self.code_bits;
        let residual_bits = self.residual_bits;
        let compact_adjacency = self.compact_adjacency;
        let direct_ip = self.direct_ip;
        let graph = py
            .detach(|| {
                self.run(|| {
                    let encode = |g: &Graph| {
                        if direct_ip {
                            VGraph::from_lifted_ip_graph(
                                g,
                                code_bits,
                                residual_bits,
                                true,
                                true,
                                id_bytes,
                            )
                        } else {
                            VGraph::from_graph_with(
                                g,
                                code_bits,
                                residual_bits,
                                true,
                                true,
                                id_bytes,
                                false,
                            )
                        }
                    };
                    let graph_dim = metric.graph_dim(dim);
                    let mut g = Graph::build_owned(data, graph_dim, &params)?;
                    if let Some(queries) = training_data {
                        let lifted = match metric {
                            Distance::InnerProduct => augment_queries(&queries, dim),
                            Distance::Angular | Distance::Euclidean => queries.clone(),
                        };
                        if training_ef == 0 {
                            g.apply_training_queries_with(
                                &lifted,
                                training_limit,
                                training_neighbors,
                                projected_degree,
                            )?;
                        } else {
                            let k = training_neighbors.min(n);
                            let pd = projected_degree.min(n.saturating_sub(1)).max(1);
                            let ids = {
                                let untrained = encode(&g)?;
                                // The encoded index takes queries in its serving coordinates.
                                let (served, served_dim) = if direct_ip {
                                    (&queries, dim)
                                } else {
                                    (&lifted, graph_dim)
                                };
                                approximate_neighbors(
                                    &untrained,
                                    served,
                                    served_dim,
                                    k,
                                    training_ef,
                                )?
                            };
                            if k >= 2 && pd < n {
                                g.project_training_neighbors(&lifted, &ids, k, pd)?;
                            }
                        }
                    }
                    // The graph's float rows are dropped with `g` once the codes are encoded.
                    let mut index = encode(&g)?;
                    if compact_adjacency {
                        index.compact_adjacency()?;
                    }
                    Ok(index)
                })
            })
            .map_err(err)?;
        self.dim = dim;
        self.built = Some(Built::new(graph));
        Ok(())
    }

    /// Search effort: `ef` is the candidate pool size, `rerank` how many of the best pool
    /// candidates are re-scored with the residual code (0 returns the pool order).
    /// `reservoir` retains candidates independently of traversal (0 uses the final beam).
    /// `patience` stops after that many expansions without improving the estimated top k
    /// (0 disables this early stop).
    #[pyo3(signature = (ef, rerank=64, patience=0, reservoir=0))]
    fn set_query_arguments(
        &mut self,
        ef: usize,
        rerank: usize,
        patience: usize,
        reservoir: usize,
    ) -> PyResult<()> {
        if ef == 0 {
            return Err(PyValueError::new_err("ef must be positive"));
        }
        self.ef = ef;
        self.rerank = rerank;
        self.patience = patience;
        self.reservoir = reservoir;
        Ok(())
    }

    /// Ids of the (approximate) `k` nearest rows to `v`, nearest first, as int64. Fewer than
    /// `k` when the index holds fewer rows.
    fn query<'py>(
        &mut self,
        py: Python<'py>,
        v: PyArrayLike1<'py, f32, AllowTypeChange>,
        k: usize,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Self::check_k(k)?;
        let params = self.params();
        let Self {
            built,
            query_buf,
            metric,
            dim,
            direct_ip,
            ..
        } = self;
        let built = built
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("the index is not fitted; call fit first"))?;
        // Copied under the GIL so no other thread can change it during the search.
        query_buf.clear();
        match v.as_slice() {
            Ok(s) => query_buf.extend_from_slice(s),
            Err(_) => query_buf.extend(v.as_array().iter().copied()),
        }
        if *metric == Distance::InnerProduct && !*direct_ip {
            if query_buf.len() != *dim {
                return Err(PyValueError::new_err(format!(
                    "the query has {} dimensions, the index has {dim}",
                    query_buf.len()
                )));
            }
            query_buf.push(0.0);
        }
        let mut out = Vec::with_capacity(k);
        let q = &query_buf[..];
        let searcher = &mut built.searcher;
        py.detach(|| searcher.search(q, k, params, &mut out))
            .map_err(err)?;
        Ok(ids_array(py, &out))
    }

    /// `query` for every row of a 2-D array, in parallel over the index's threads; a list of
    /// int64 arrays in row order.
    fn batch_query<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayLike2<'py, f32, AllowTypeChange>,
        k: usize,
    ) -> PyResult<Vec<Bound<'py, PyArray1<i64>>>> {
        Self::check_k(k)?;
        let built = self
            .built
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("the index is not fitted; call fit first"))?;
        let (nq, dim) = (x.shape()[0], x.shape()[1]);
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "queries have {dim} dimensions, the index has {}",
                self.dim
            )));
        }
        let mut queries = row_major(&x);
        let params = self.params();
        let graph = &*built.graph;
        let metric = self.metric;
        if metric == Distance::InnerProduct && !self.direct_ip {
            queries = augment_queries(&queries, dim);
        }
        let dim = graph.dim();
        let results: Vec<Vec<u32>> = py
            .detach(|| {
                self.run(|| {
                    // A few chunks per thread, one searcher per chunk.
                    let chunk = nq.div_ceil(rayon::current_num_threads() * 4).max(1);
                    queries
                        .par_chunks(chunk * dim)
                        .map(|rows| {
                            let mut s = graph.searcher();
                            rows.chunks_exact(dim)
                                .map(|q| {
                                    let mut out = Vec::with_capacity(k);
                                    s.search(q, k, params, &mut out).map(|_| out)
                                })
                                .collect::<Result<Vec<_>, WarmError>>()
                        })
                        .collect::<Result<Vec<_>, WarmError>>()
                        .map(|c| c.into_iter().flatten().collect())
                })
            })
            .map_err(err)?;
        Ok(results.iter().map(|ids| ids_array(py, ids)).collect())
    }

    /// `batch_query` as one (nq, k) int64 array, row i holding the ids for query i, padded with
    /// -1 when the index returns fewer than `k`.
    fn batch_query_array<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayLike2<'py, f32, AllowTypeChange>,
        k: usize,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        Self::check_k(k)?;
        let built = self
            .built
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("the index is not fitted; call fit first"))?;
        let (nq, dim) = (x.shape()[0], x.shape()[1]);
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "queries have {dim} dimensions, the index has {}",
                self.dim
            )));
        }
        let mut queries = row_major(&x);
        if self.metric == Distance::InnerProduct && !self.direct_ip {
            queries = augment_queries(&queries, dim);
        }
        let params = self.params();
        let graph = &*built.graph;
        let dim = graph.dim();
        let mut ids = vec![-1i64; nq * k];
        py.detach(|| {
            self.run(|| {
                let chunk = nq.div_ceil(rayon::current_num_threads() * 4).max(1);
                ids.par_chunks_mut(chunk * k)
                    .zip(queries.par_chunks(chunk * dim))
                    .try_for_each(|(out, rows)| {
                        let mut s = graph.searcher();
                        let mut found = Vec::with_capacity(k);
                        for (q, slot) in rows.chunks_exact(dim).zip(out.chunks_exact_mut(k)) {
                            s.search(q, k, params, &mut found)?;
                            for (o, &id) in slot.iter_mut().zip(&found) {
                                *o = i64::from(id);
                            }
                        }
                        Ok::<(), WarmError>(())
                    })
            })
        })
        .map_err(err)?;
        let array = numpy::ndarray::Array2::from_shape_vec((nq, k), ids)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    /// Writes the fitted index to `path` (replacing it atomically); `load` reads it back.
    fn save(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        let built = self
            .built
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("the index is not fitted; call fit first"))?;
        let graph = &*built.graph;
        py.detach(|| graph.save(&path))
            .map_err(|e| PyRuntimeError::new_err(format!("save {}: {e}", path.display())))
    }

    /// Replaces the index with one written by `save` from a `VectorIndex` of the same metric
    /// and `direct_ip` setting, read into memory. Restores the serialized code widths;
    /// subsequent `fit` calls reuse those widths.
    fn load(&mut self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        let graph = py
            .detach(|| VGraph::open(&path, Residency::Heap))
            .map_err(|e| PyRuntimeError::new_err(format!("load {}: {e}", path.display())))?;
        let lifted = self.metric == Distance::InnerProduct && !self.direct_ip;
        let dim = graph.dim() - usize::from(lifted);
        if dim == 0 {
            return Err(PyValueError::new_err(
                "the saved index does not match this metric",
            ));
        }
        self.code_bits = graph.code_bits();
        self.residual_bits = graph.residual_bits();
        self.dim = dim;
        self.built = Some(Built::new(graph));
        Ok(())
    }

    /// Logical payload bytes of the encoded index, including row padding. Excludes allocator
    /// overhead, capacities, slab rounding, rotations and query scratch; not RSS/PSS/USS.
    #[getter]
    fn index_bytes(&self) -> usize {
        self.built.as_ref().map_or(0, |b| b.graph.index_bytes())
    }

    /// Component logical payload bytes whose sum is `index_bytes`, or all zeros before fit.
    #[getter]
    fn memory_breakdown(&self) -> std::collections::BTreeMap<String, usize> {
        let bytes = self
            .built
            .as_ref()
            .map_or_else(Default::default, |b| b.graph.memory_breakdown());
        [
            ("adjacency_ids_bytes", bytes.adjacency_ids_bytes),
            ("adjacency_offsets_bytes", bytes.adjacency_offsets_bytes),
            ("traversal_codes_bytes", bytes.traversal_codes_bytes),
            ("traversal_scales_bytes", bytes.traversal_scales_bytes),
            ("residual_codes_bytes", bytes.residual_codes_bytes),
            ("residual_scales_bytes", bytes.residual_scales_bytes),
            ("mean_bytes", bytes.mean_bytes),
            ("norms_bytes", bytes.norms_bytes),
            ("original_ids_bytes", bytes.original_ids_bytes),
            ("screen_bytes", bytes.screen_bytes),
            ("neighbor_blocks_bytes", bytes.neighbor_blocks_bytes),
            ("upper_layers_bytes", bytes.upper_layers_bytes),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value))
        .collect()
    }

    #[getter]
    fn code_bits(&self) -> u8 {
        self.code_bits
    }

    #[getter]
    fn residual_bits(&self) -> u8 {
        self.residual_bits
    }

    /// Dimension of the fitted vectors (0 before `fit`).
    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    fn __len__(&self) -> usize {
        self.built.as_ref().map_or(0, |b| b.graph.len())
    }

    fn __repr__(&self) -> String {
        format!(
            "VectorIndex(metric='{}', degree={}, build_ef={}, ef={}, rerank={}, direct_ip={}, reservoir={}, code_bits={}, residual_bits={}, n={})",
            self.metric.name(),
            self.degree,
            self.build_ef,
            self.ef,
            self.rerank,
            self.direct_ip,
            self.reservoir,
            self.code_bits,
            self.residual_bits,
            self.__len__()
        )
    }
}

fn update_err(e: UpdateError) -> PyErr {
    match e {
        UpdateError::Io(_) | UpdateError::Persistence(_) | UpdateError::RecoveryRequired => {
            PyRuntimeError::new_err(e.to_string())
        }
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// The ids of a 1-D array as u32, rejecting values outside 0..u32::MAX.
fn u32_ids(ids: &PyArrayLike1<'_, i64, AllowTypeChange>) -> PyResult<Vec<u32>> {
    ids.as_array()
        .iter()
        .map(|&i| {
            u32::try_from(i)
                .ok()
                .filter(|&i| i != u32::MAX)
                .ok_or_else(|| PyValueError::new_err(format!("id {i} is outside 0..2^32-1")))
        })
        .collect()
}

/// A durable index that takes inserts, deletes and replacements while it serves queries
/// (`core::qgraph::UpdatableVGraph`): a memory-mapped base graph, an in-memory graph of the
/// rows added since, a log that makes each `commit` durable, and `consolidate`, which rebuilds
/// the base over the live rows or merges the changes into it. Ids are caller-chosen integers
/// in 0..2^32-1.
#[pyclass(name = "UpdatableIndex")]
struct PyUpdatableIndex {
    index: UpdatableVGraph,
    pool: Option<rayon::ThreadPool>,
    ef: usize,
    rerank: usize,
}

impl PyUpdatableIndex {
    fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// The caller's rows, borrowed when they are already a C-contiguous float32 matrix (the
    /// streaming harness passes up to 757,735 x 100 rows per insert: a 303 MB copy otherwise).
    fn rows<'a>(
        &self,
        x: &'a PyArrayLike2<'_, f32, AllowTypeChange>,
        n: usize,
    ) -> PyResult<std::borrow::Cow<'a, [f32]>> {
        let (rows, dim) = (x.shape()[0], x.shape()[1]);
        if rows != n || dim != self.index.dim() {
            return Err(PyValueError::new_err(format!(
                "expected a ({n}, {}) matrix, got ({rows}, {dim})",
                self.index.dim()
            )));
        }
        Ok(match x.as_slice() {
            Ok(rows) if x.is_c_contiguous() => std::borrow::Cow::Borrowed(rows),
            _ => std::borrow::Cow::Owned(row_major(x)),
        })
    }
}

fn thread_pool(threads: usize) -> PyResult<Option<rayon::ThreadPool>> {
    match threads {
        0 => Ok(None),
        n => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map(Some)
            .map_err(|e| PyRuntimeError::new_err(format!("thread pool: {e}"))),
    }
}

#[pymethods]
impl PyUpdatableIndex {
    /// Creates an empty index in the new directory `path`. `metric` is "angular" or
    /// "euclidean"; `degree` (even) and `build_ef` shape the graph; `threads` bounds insert,
    /// consolidation and batch-query parallelism (0 = one per core). `alpha` is the pruning
    /// parameter of the second construction pass, as for `VectorIndex` (1.0: keep only edges
    /// not occluded by a nearer kept neighbour). At degree 32 on 100-d MS Turing, 1.2 keeps
    /// nearer, shorter edges and costs 7-9 points of recall@10 at ef 64-128 on both the
    /// consolidated base and the delta; raise it only with a measured gain at the degree used.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, dim, metric="euclidean", degree=32, build_ef=100, alpha=1.0, threads=0))]
    fn create(
        py: Python<'_>,
        path: std::path::PathBuf,
        dim: usize,
        metric: &str,
        degree: usize,
        build_ef: usize,
        alpha: f32,
        threads: usize,
    ) -> PyResult<Self> {
        let metric = match metric {
            "angular" => Metric::Angular,
            "euclidean" => Metric::Euclidean,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown metric {other:?}; UpdatableIndex supports 'angular' and 'euclidean'"
                )))
            }
        };
        if degree == 0 || !degree.is_multiple_of(2) || build_ef == 0 || dim == 0 {
            return Err(PyValueError::new_err(
                "dim and build_ef must be positive and degree positive and even",
            ));
        }
        if !(alpha.is_finite() && alpha >= 1.0) {
            return Err(PyValueError::new_err("alpha must be at least 1"));
        }
        let params = BuildParams {
            metric,
            degree,
            build_ef,
            alpha,
            ..BuildParams::default()
        };
        let pool = thread_pool(threads)?;
        let index = py
            .detach(|| UpdatableVGraph::create(&path, &[], dim, &params))
            .map_err(update_err)?;
        Ok(Self {
            index,
            pool,
            ef: 64,
            rerank: 64,
        })
    }

    /// Opens an existing index directory and replays its committed batches.
    #[staticmethod]
    #[pyo3(signature = (path, threads=0))]
    fn open(py: Python<'_>, path: std::path::PathBuf, threads: usize) -> PyResult<Self> {
        let pool = thread_pool(threads)?;
        let index = py
            .detach(|| UpdatableVGraph::open(&path))
            .map_err(update_err)?;
        Ok(Self {
            index,
            pool,
            ef: 64,
            rerank: 64,
        })
    }

    /// Search effort, as for `VectorIndex`.
    #[pyo3(signature = (ef, rerank=64))]
    fn set_query_arguments(&mut self, ef: usize, rerank: usize) -> PyResult<()> {
        if ef == 0 {
            return Err(PyValueError::new_err("ef must be positive"));
        }
        self.ef = ef;
        self.rerank = rerank;
        Ok(())
    }

    /// Adds rows under new `ids` and commits. Ids must not be live.
    fn insert(
        &self,
        py: Python<'_>,
        x: PyArrayLike2<'_, f32, AllowTypeChange>,
        ids: PyArrayLike1<'_, i64, AllowTypeChange>,
    ) -> PyResult<()> {
        let ids = u32_ids(&ids)?;
        let rows = self.rows(&x, ids.len())?;
        let dim = self.index.dim();
        let step = self.index.batch_capacity();
        py.detach(|| {
            self.run(|| {
                for (ids, rows) in ids.chunks(step).zip(rows.chunks(step * dim)) {
                    self.index.insert_batch(ids, rows)?;
                    self.index.commit()?;
                }
                Ok(())
            })
        })
        .map_err(update_err)
    }

    /// Removes live `ids` and commits.
    fn delete(&self, py: Python<'_>, ids: PyArrayLike1<'_, i64, AllowTypeChange>) -> PyResult<()> {
        let ids = u32_ids(&ids)?;
        let step = self.index.batch_capacity();
        py.detach(|| {
            for chunk in ids.chunks(step) {
                self.index.delete_batch(chunk)?;
                self.index.commit()?;
            }
            Ok(())
        })
        .map_err(update_err)
    }

    /// Gives each live id in `ids` the matching row of `x` and commits.
    fn replace(
        &self,
        py: Python<'_>,
        x: PyArrayLike2<'_, f32, AllowTypeChange>,
        ids: PyArrayLike1<'_, i64, AllowTypeChange>,
    ) -> PyResult<()> {
        let ids = u32_ids(&ids)?;
        let rows = self.rows(&x, ids.len())?;
        let dim = self.index.dim();
        let step = self.index.batch_capacity();
        py.detach(|| {
            for (ids, rows) in ids.chunks(step).zip(rows.chunks(step * dim)) {
                self.index.replace_batch(ids, rows)?;
                self.index.commit()?;
            }
            Ok(())
        })
        .map_err(update_err)
    }

    /// The `k` nearest live ids of every row of `x`, nearest first, as an (n, k) int64 array;
    /// -1 pads rows when fewer than `k` ids are live.
    fn search<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayLike2<'py, f32, AllowTypeChange>,
        k: usize,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        PyVectorIndex::check_k(k)?;
        let nq = x.shape()[0];
        let queries = self.rows(&x, nq)?;
        let dim = self.index.dim();
        let params = VSearchParams {
            ef: self.ef,
            rerank: self.rerank,
            ..VSearchParams::default()
        };
        let mut out = vec![-1i64; nq * k];
        py.detach(|| {
            self.run(|| {
                out.par_chunks_mut(k)
                    .zip(queries.par_chunks(dim))
                    .try_for_each_init(Vec::new, |found, (row, q)| {
                        self.index.search_into(q, k, params, found)?;
                        for (slot, &id) in row.iter_mut().zip(found.iter()) {
                            *slot = i64::from(id);
                        }
                        Ok::<_, UpdateError>(())
                    })
            })
        })
        .map_err(update_err)?;
        numpy::ndarray::Array2::from_shape_vec((nq, k), out)
            .map(|a| a.into_pyarray(py))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Whether the delta or the deleted share has outgrown the given fractions of the base.
    #[pyo3(signature = (delta_ratio=0.1, tombstone_ratio=0.2, min_delta=1024))]
    fn needs_consolidation(
        &self,
        delta_ratio: f64,
        tombstone_ratio: f64,
        min_delta: usize,
    ) -> PyResult<bool> {
        self.index
            .needs_consolidation(ConsolidationPolicy {
                delta_ratio,
                tombstone_ratio,
                min_delta,
            })
            .map_err(update_err)
    }

    /// Publishes the live rows as a new base; searches keep running meanwhile. By default the
    /// base graph is rebuilt over every live row. With `merge=True` the previous base graph is
    /// kept: deleted vertices are spliced out of the lists that pointed at them and the delta
    /// rows are linked in, unless the rows inserted and deleted by merges since the last
    /// rebuild (counted from open) would reach `rebuild_after` times the live rows, or another
    /// condition of `ConsolidationMode::Merge` asks for a rebuild. Returns "merged",
    /// "rebuilt" or "unchanged".
    #[pyo3(signature = (merge=false, rebuild_after=1.0))]
    fn consolidate(
        &self,
        py: Python<'_>,
        merge: bool,
        rebuild_after: f64,
    ) -> PyResult<&'static str> {
        if rebuild_after.is_nan() || rebuild_after < 0.0 {
            return Err(PyValueError::new_err("rebuild_after must be non-negative"));
        }
        let mode = if merge {
            ConsolidationMode::Merge { rebuild_after }
        } else {
            ConsolidationMode::Rebuild
        };
        let done = py
            .detach(|| self.run(|| self.index.consolidate_with(mode)))
            .map_err(update_err)?;
        Ok(match done {
            Consolidation::Unchanged => "unchanged",
            Consolidation::Merged => "merged",
            Consolidation::Rebuilt => "rebuilt",
        })
    }

    /// Streaming diagnostics, enabled at open/create by WARM_STREAM_DIAGNOSTICS=1.
    fn streaming_diagnostics(
        &self,
        py: Python<'_>,
    ) -> Option<std::collections::BTreeMap<String, f64>> {
        py.detach(|| self.index.streaming_diagnostics())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.index.dim()
    }

    fn __len__(&self) -> usize {
        self.index.len()
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorIndex>()?;
    m.add_class::<PyUpdatableIndex>()
}
