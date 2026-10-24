// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Python surface of the learned-sparse inner-product index (`core::index::sparse_impact`):
//! `SparseIndex`. Documents come from a CSR file in the big-ann-benchmarks `spmat` layout
//! (int64 rows, cols, nnz; int64 indptr; int32 indices; float32 data), memory-mapped so the
//! source matrix is never copied into Python. Build and search run without the GIL.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::core::index::sparse_impact::{
    CsrView, SparseBuildParams, SparseImpactIndex, SparseSearchParams,
};

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
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

/// Top-k inner-product search over non-negative learned sparse vectors.
///
/// `SparseIndex(postings_per_term, block_fraction, summary_energy, threads)` then
/// `build_from_spmat(path)`; `search(indptr, indices, data, k)` takes query rows in CSR form
/// and returns an (n, k) int64 array of document rows, best first, -1 padded.
/// `postings_per_term=0` keeps every posting; with `summary_energy=1`, `query_cut=0` and
/// `heap_factor=1` the search is exact.
#[pyclass(name = "SparseIndex")]
struct PySparseIndex {
    params: SparseBuildParams,
    index: Option<SparseImpactIndex>,
    search: SparseSearchParams,
    pool: Option<rayon::ThreadPool>,
}

impl PySparseIndex {
    fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    fn built(&self) -> PyResult<&SparseImpactIndex> {
        self.index
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("index is not built"))
    }
}

#[pymethods]
impl PySparseIndex {
    #[new]
    #[pyo3(signature = (postings_per_term=4000, block_fraction=0.1, summary_energy=0.4, threads=0, seed=0x5eed))]
    fn new(
        postings_per_term: usize,
        block_fraction: f32,
        summary_energy: f32,
        threads: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if !(block_fraction > 0.0 && block_fraction <= 1.0) {
            return Err(value_err("block_fraction must be in (0, 1]"));
        }
        if !(summary_energy > 0.0 && summary_energy <= 1.0) {
            return Err(value_err("summary_energy must be in (0, 1]"));
        }
        Ok(Self {
            params: SparseBuildParams {
                postings_per_term: if postings_per_term == 0 {
                    usize::MAX
                } else {
                    postings_per_term
                },
                block_fraction,
                summary_energy,
                seed,
                ..SparseBuildParams::default()
            },
            index: None,
            search: SparseSearchParams {
                query_cut: 4,
                heap_factor: 1.1,
            },
            pool: thread_pool(threads)?,
        })
    }

    /// Builds from an `spmat` CSR file (memory-mapped while building).
    fn build_from_spmat(&mut self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        let file = std::fs::File::open(&path)
            .map_err(|e| value_err(format!("{}: {e}", path.display())))?;
        // SAFETY: the map is read-only and lives until the build returns; the index copies
        // every value it keeps. The file must not be truncated meanwhile, as with any reader.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| value_err(format!("{}: {e}", path.display())))?;
        let view = CsrView::from_spmat(&map).map_err(value_err)?;
        let params = self.params;
        let built = py.detach(|| self.run(|| SparseImpactIndex::build(view, params)));
        self.index = Some(built.map_err(value_err)?);
        Ok(())
    }

    /// Query coordinates walked per query (`0` walks all) and the block threshold factor.
    #[pyo3(signature = (query_cut=4, heap_factor=1.1))]
    fn set_search(&mut self, query_cut: usize, heap_factor: f32) -> PyResult<()> {
        if !(heap_factor > 0.0 && heap_factor.is_finite()) {
            return Err(value_err("heap_factor must be positive"));
        }
        self.search = SparseSearchParams {
            query_cut: if query_cut == 0 {
                usize::MAX
            } else {
                query_cut
            },
            heap_factor,
        };
        Ok(())
    }

    fn search<'py>(
        &self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i32>,
        data: PyReadonlyArray1<'py, f32>,
        k: usize,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let index = self.built()?;
        if k == 0 || k > 4096 {
            return Err(value_err("k must be in 1..=4096"));
        }
        let view = CsrView {
            indptr: indptr.as_slice()?,
            indices: indices.as_slice()?,
            data: data.as_slice()?,
            n_dims: index.n_dims(),
        };
        let nq = indptr.len().saturating_sub(1);
        let params = self.search;
        let ids = py
            .detach(|| self.run(|| index.search_batch(view, k, params)))
            .map_err(value_err)?;
        let out: Vec<i64> = ids
            .into_iter()
            .map(|d| if d == u32::MAX { -1 } else { i64::from(d) })
            .collect();
        numpy::ndarray::Array2::from_shape_vec((nq, k), out)
            .map(|a| a.into_pyarray(py))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Bytes held by the index, and its block and kept-posting counts.
    fn stats(&self) -> PyResult<(usize, usize, usize)> {
        let index = self.built()?;
        let (blocks, postings) = index.block_stats();
        Ok((index.memory_bytes(), blocks, postings))
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparseIndex>()
}
