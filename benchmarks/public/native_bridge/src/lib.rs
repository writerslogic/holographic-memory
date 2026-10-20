use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use serde::Deserialize;

include!(concat!(env!("OUT_DIR"), "/core.rs"));

use crate::core::qgraph::{BuildParams, Graph, VGraph, VSearchParams, VSearcher};

struct SearchState<'a> {
    searcher: VSearcher<'a>,
    out: Vec<u32>,
}

self_cell::self_cell! {
    struct IndexCell {
        owner: VGraph,
        #[covariant]
        dependent: SearchState,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    degree: usize,
    build_ef: usize,
    alpha: f32,
    seed: u64,
    bits: u8,
    residual_bits: u8,
    align: bool,
    reorder: bool,
    id_bytes: u8,
}

#[pyclass(unsendable)]
struct NativeIndex {
    cell: IndexCell,
    n: usize,
    dim: usize,
    graph_hash: String,
}

impl NativeIndex {
    fn own(index: VGraph, n: usize, dim: usize, graph_hash: String) -> Self {
        Self {
            cell: IndexCell::new(index, |index| SearchState {
                searcher: index.searcher(),
                out: Vec::new(),
            }),
            n,
            dim,
            graph_hash,
        }
    }
}

#[pymethods]
impl NativeIndex {
    #[staticmethod]
    fn build(train: PyReadonlyArray2<'_, f32>, config: &str) -> PyResult<Self> {
        if !train.is_c_contiguous() {
            return Err(PyValueError::new_err("train must be C-contiguous float32"));
        }
        let shape = train.shape();
        let (n, dim) = (shape[0], shape[1]);
        let rows = train
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let config: Config =
            serde_json::from_str(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (index, graph_hash) = pool.install(|| -> PyResult<_> {
            let params = BuildParams {
                degree: config.degree,
                build_ef: config.build_ef,
                alpha: config.alpha,
                seed: config.seed,
                code_bits: 0,
            };
            let graph = Graph::build(rows, dim, &params)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let index = VGraph::from_graph_with(
                &graph,
                config.bits,
                config.residual_bits,
                config.align,
                config.reorder,
                config.id_bytes,
                false,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok((index, format!("{:016x}", graph.structure_hash())))
        })?;
        Ok(Self::own(index, n, dim, graph_hash))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let (index, dim, graph_hash) = VGraph::bridge_load(std::path::Path::new(path))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let n = index.len();
        Ok(Self::own(index, n, dim, graph_hash))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.cell
            .borrow_owner()
            .bridge_save(std::path::Path::new(path), &self.graph_hash)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    fn query<'py>(
        &mut self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef: usize,
        rerank: usize,
        patience: usize,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        if !queries.is_c_contiguous()
            || queries.shape()[1] != self.dim
            || !(1..=self.n).contains(&k)
        {
            return Err(PyValueError::new_err(
                "query requires C-contiguous float32 [n,dim], 1 <= k <= index rows",
            ));
        }
        let nqueries = queries.shape()[0];
        let size = nqueries
            .checked_mul(k)
            .ok_or_else(|| PyValueError::new_err("output size overflow"))?;
        let rows = queries
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut labels = Vec::new();
        labels
            .try_reserve_exact(size)
            .map_err(|_| PyValueError::new_err("output allocation failed"))?;
        self.cell.with_dependent_mut(|_, state| -> PyResult<()> {
            for query in rows.chunks_exact(self.dim) {
                let start = labels.len();
                state
                    .searcher
                    .search(
                        query,
                        k,
                        VSearchParams {
                            ef,
                            rerank,
                            patience,
                            screen: false,
                            ..VSearchParams::default()
                        },
                        &mut state.out,
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                labels.extend(state.out.iter().map(|&id| i64::from(id)));
                labels.resize(start + k, -1);
            }
            Ok(())
        })?;
        let matrix = Array2::from_shape_vec((nqueries, k), labels)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(matrix.into_pyarray(py))
    }

    fn memory(&self) -> (usize, usize, usize) {
        let index = self.cell.borrow_owner();
        let workspace = self.cell.with_dependent(|_, state| {
            state.searcher.bridge_workspace_bytes() + state.out.capacity() * 4
        });
        (
            index.index_bytes(),
            index.bridge_allocated_bytes(),
            workspace,
        )
    }

    #[getter]
    fn graph_hash(&self) -> &str {
        &self.graph_hash
    }
}

#[pymodule]
fn hms_native_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeIndex>()?;
    Ok(())
}
