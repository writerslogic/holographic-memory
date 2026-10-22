// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Python surface of the quantized graph index (`core::qgraph`): `VectorIndex`.
//!
//! The encoding is the one `public-bench ann-qgraph` measures: 4-bit traversal codes, an 8-bit
//! residual code for the re-rank, cache-line aligned rows, breadth-first vertex order and
//! 3-byte neighbour ids, over a graph built with quantized construction search. Inputs are
//! numpy arrays; build and search run without the GIL.

use std::sync::Arc;

use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArrayLike1, PyArrayLike2};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::core::error::HmsError;
use crate::core::qgraph::{BuildParams, Graph, Metric, VGraph, VSearchParams, VSearcher};

/// Traversal code bits per coordinate.
const VERTEX_BITS: u8 = 4;
/// Residual code bits per coordinate, read by the re-rank.
const RESIDUAL_BITS: u8 = 8;
/// Graphs below this many vertices store neighbour ids in three bytes.
const THREE_BYTE_IDS_BELOW: usize = (1 << 24) - 1;

fn err(e: HmsError) -> PyErr {
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
    pool: Option<rayon::ThreadPool>,
    ef: usize,
    rerank: usize,
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
    /// (0 = one per core).
    #[new]
    #[pyo3(signature = (metric="angular", degree=64, build_ef=500, threads=0))]
    fn new(metric: &str, degree: usize, build_ef: usize, threads: usize) -> PyResult<Self> {
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
        if degree == 0 || !degree.is_multiple_of(2) {
            return Err(PyValueError::new_err(format!(
                "degree must be positive and even, not {degree}"
            )));
        }
        if build_ef == 0 {
            return Err(PyValueError::new_err("build_ef must be positive"));
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
            pool,
            ef: 64,
            rerank: 64,
            dim: 0,
            built: None,
            query_buf: Vec::new(),
        })
    }

    /// Builds the index over the rows of a 2-D array (converted to float32 if needed).
    /// Replaces any previous index.
    fn fit(&mut self, py: Python<'_>, x: PyArrayLike2<'_, f32, AllowTypeChange>) -> PyResult<()> {
        let (n, dim) = (x.shape()[0], x.shape()[1]);
        if n == 0 || dim == 0 {
            return Err(PyValueError::new_err(format!(
                "fit expects a non-empty matrix, got shape ({n}, {dim})"
            )));
        }
        // One copy, made while the array is borrowed under the GIL; the build normalizes it in
        // place and keeps it until the encoding is done.
        let data = row_major(&x);
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
        let graph = py
            .detach(|| {
                self.run(|| {
                    let data = match metric {
                        Distance::InnerProduct => augment_rows(&data, dim),
                        Distance::Angular | Distance::Euclidean => data,
                    };
                    let g = Graph::build_owned(data, metric.graph_dim(dim), &params)?;
                    // The graph's float rows are dropped here; the index keeps only its codes.
                    VGraph::from_graph_with(
                        &g,
                        VERTEX_BITS,
                        RESIDUAL_BITS,
                        true,
                        true,
                        id_bytes,
                        false,
                    )
                })
            })
            .map_err(err)?;
        self.dim = dim;
        self.built = Some(Built::new(graph));
        Ok(())
    }

    /// Search effort: `ef` is the candidate pool size, `rerank` how many of the best pool
    /// candidates are re-scored with the residual code (0 returns the pool order).
    #[pyo3(signature = (ef, rerank=64))]
    fn set_query_arguments(&mut self, ef: usize, rerank: usize) -> PyResult<()> {
        if ef == 0 {
            return Err(PyValueError::new_err("ef must be positive"));
        }
        self.ef = ef;
        self.rerank = rerank;
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
        if *metric == Distance::InnerProduct {
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
        if metric == Distance::InnerProduct {
            queries = queries
                .chunks_exact(dim)
                .flat_map(|q| q.iter().copied().chain([0.0]))
                .collect();
        }
        let dim = metric.graph_dim(dim);
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
                                .collect::<Result<Vec<_>, HmsError>>()
                        })
                        .collect::<Result<Vec<_>, HmsError>>()
                        .map(|c| c.into_iter().flatten().collect())
                })
            })
            .map_err(err)?;
        Ok(results.iter().map(|ids| ids_array(py, ids)).collect())
    }

    /// Bytes of the encoded index (codes, residuals, adjacency, upper layers).
    #[getter]
    fn index_bytes(&self) -> usize {
        self.built.as_ref().map_or(0, |b| b.graph.index_bytes())
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
            "VectorIndex(metric='{}', degree={}, build_ef={}, ef={}, rerank={}, n={})",
            self.metric.name(),
            self.degree,
            self.build_ef,
            self.ef,
            self.rerank,
            self.__len__()
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorIndex>()
}
