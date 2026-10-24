// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Python surface of the label-filtered index (`core::qgraph::filter`): `FilteredIndex`.
//!
//! Rows are uint8; labels come as the CSR arrays of a scipy sparse matrix (`indptr`,
//! `indices`). Build and search run without the GIL.

use numpy::{AllowTypeChange, IntoPyArray, PyArray2, PyArrayLike1, PyArrayLike2};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::core::qgraph::filter::{FilterBuildParams, FilterSearchParams, FilteredIndex};

#[pyclass(name = "FilteredIndex")]
struct PyFilteredIndex {
    index: FilteredIndex,
    pool: Option<rayon::ThreadPool>,
    params: FilterSearchParams,
}

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn u8_rows<'py>(x: &PyArrayLike2<'py, u8, AllowTypeChange>) -> (Vec<u8>, usize, usize) {
    let shape = x.shape();
    let (n, dim) = (shape[0], shape[1]);
    let ro = x.readonly();
    let rows = match ro.as_slice() {
        Ok(s) if ro.is_c_contiguous() => s.to_vec(),
        _ => ro.as_array().iter().copied().collect(),
    };
    (rows, n, dim)
}

fn csr(
    indptr: &PyArrayLike1<'_, i64, AllowTypeChange>,
    indices: &PyArrayLike1<'_, i64, AllowTypeChange>,
    rows: usize,
) -> PyResult<(Vec<u64>, Vec<u32>)> {
    let p = indptr.readonly();
    let p = p.as_array();
    if p.len() != rows + 1 {
        return Err(PyValueError::new_err(format!(
            "indptr has {} entries, expected {}",
            p.len(),
            rows + 1
        )));
    }
    let ptr: Vec<u64> = p
        .iter()
        .map(|&v| u64::try_from(v).map_err(|_| PyValueError::new_err("negative indptr entry")))
        .collect::<PyResult<_>>()?;
    let i = indices.readonly();
    let ids: Vec<u32> = i
        .as_array()
        .iter()
        .map(|&v| u32::try_from(v).map_err(|_| PyValueError::new_err("label id out of u32 range")))
        .collect::<PyResult<_>>()?;
    Ok((ptr, ids))
}

#[pymethods]
impl PyFilteredIndex {
    /// Builds over `data` (n, dim) uint8 rows whose labels are the CSR rows `indptr`/`indices`
    /// (a scipy `csr_matrix`'s arrays). Labels with more than `ivf_min` members get an inverted
    /// file of about `cluster_size` members per list (at most `max_lists`); labels with at least
    /// `bitset_min` members a membership bitset; labels with an inverted file and at least
    /// `copy_min` members keep their rows copied in list order. `threads` bounds build and search (0 = all).
    #[new]
    #[pyo3(signature = (data, indptr, indices, labels=0, ivf_min=20_000, cluster_size=1_000, max_lists=1_024, bitset_min=20_000, kmeans_iters=6, train_per_list=48, copy_min=usize::MAX, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        data: PyArrayLike2<'_, u8, AllowTypeChange>,
        indptr: PyArrayLike1<'_, i64, AllowTypeChange>,
        indices: PyArrayLike1<'_, i64, AllowTypeChange>,
        labels: usize,
        ivf_min: usize,
        cluster_size: usize,
        max_lists: usize,
        bitset_min: usize,
        kmeans_iters: usize,
        train_per_list: usize,
        copy_min: usize,
        threads: usize,
    ) -> PyResult<Self> {
        let (rows, n, dim) = u8_rows(&data);
        let (ptr, ids) = csr(&indptr, &indices, n)?;
        let pool = match threads {
            0 => None,
            t => Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .map_err(|e| PyRuntimeError::new_err(format!("thread pool: {e}")))?,
            ),
        };
        let params = FilterBuildParams {
            ivf_min,
            cluster_size,
            max_lists,
            bitset_min,
            kmeans_iters,
            train_per_list,
            copy_min,
            ..FilterBuildParams::default()
        };
        let build = || FilteredIndex::build(&rows, dim, &ptr, &ids, labels, &params);
        let index = py
            .detach(|| match &pool {
                Some(p) => p.install(build),
                None => build(),
            })
            .map_err(value_err)?;
        Ok(Self {
            index,
            pool,
            params: FilterSearchParams::default(),
        })
    }

    /// Sets the search parameters: queries whose rarest label has at most `brute_max` members
    /// score every match; others probe that label's lists until `target` matches are scored
    /// and at least `min_lists` lists are probed. Multi-label queries whose rarest label has at
    /// most `intersect_max` members are intersected first and scored exactly when the matches
    /// number at most `brute_max`.
    #[pyo3(signature = (brute_max=20_000, target=5_000, min_lists=1, intersect_max=200_000))]
    fn set_search(
        &mut self,
        brute_max: usize,
        target: usize,
        min_lists: usize,
        intersect_max: usize,
    ) {
        self.params = FilterSearchParams {
            brute_max,
            target,
            min_lists,
            intersect_max,
        };
    }

    /// The `k` nearest live rows of each query row that hold all of its labels (CSR
    /// `indptr`/`indices`), nearest first, as an (nq, k) int64 array padded with -1.
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyArrayLike2<'py, u8, AllowTypeChange>,
        indptr: PyArrayLike1<'py, i64, AllowTypeChange>,
        indices: PyArrayLike1<'py, i64, AllowTypeChange>,
        k: usize,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be positive"));
        }
        let (rows, nq, dim) = u8_rows(&queries);
        if dim != self.index.dim() {
            return Err(PyValueError::new_err(format!(
                "queries have {dim} columns, index {}",
                self.index.dim()
            )));
        }
        let (ptr, ids) = csr(&indptr, &indices, nq)?;
        let mut out = vec![-1i64; nq * k];
        let mut search = || {
            self.index
                .search_batch(&rows, &ptr, &ids, k, &self.params, &mut out)
        };
        py.detach(|| match &self.pool {
            Some(p) => p.install(search),
            None => search(),
        })
        .map_err(value_err)?;
        numpy::ndarray::Array2::from_shape_vec((nq, k), out)
            .map(|a| a.into_pyarray(py))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Marks row `id` deleted; returns whether it was live. Inserting needs a rebuild.
    fn delete(&mut self, id: u32) -> bool {
        self.index.delete(id)
    }

    /// Path counters since the last reset: (brute-force queries, inverted-file queries,
    /// rows scored).
    fn stats(&self) -> (u64, u64, u64) {
        let s = self.index.stats();
        (s.brute, s.ivf, s.scored)
    }

    fn reset_stats(&self) {
        self.index.reset_stats();
    }

    /// Bytes held by the index.
    fn memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }

    /// Labels that have an inverted file.
    fn ivf_labels(&self) -> usize {
        self.index.ivf_labels()
    }

    fn __len__(&self) -> usize {
        self.index.len()
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFilteredIndex>()
}
