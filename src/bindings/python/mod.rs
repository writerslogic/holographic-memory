// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings (pyo3) for the deterministic quantized-phase VSA substrate.
//!
//! Exposes the differentiated phase-vector surface -- non-self-inverse binding,
//! graceful similarity, deterministic bundling, and resonator factorization -- as a
//! pip-installable module built with `maturin` under the `python` feature. This is
//! the initial, curated surface (the phase substrate, qFHRR arXiv 2604.25939, and the
//! `VectorIndex` nearest-neighbour index); it will grow as the public API stabilises.
//! See `pyproject.toml` and `.github/workflows/python-native-release.yml`. The module is named
//! `writerslogic_warm`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::phase_hvec::PhaseHVec;
use crate::core::phase_resonator::{phase_resonator_factorize, ResonatorConfig};

const MAX_PHASE_DIM: usize = 65_536;
const MAX_PHASE_ITEMS: usize = 65_536;
const MAX_PHASE_ELEMENTS: usize = 4_194_304;
const MAX_PHASE_FACTORS: usize = 64;
const MAX_PHASE_ITER: usize = 1_000;
const MAX_PHASE_WORK: usize = 100_000_000;

fn phase_limit(value: usize, limit: usize, name: &str) -> PyResult<()> {
    if value > limit {
        return Err(PyValueError::new_err(format!(
            "{name} exceeds limit {limit}"
        )));
    }
    Ok(())
}

fn phase_shape(dim: usize, items: usize) -> PyResult<()> {
    phase_limit(dim, MAX_PHASE_DIM, "phase dimensions")?;
    phase_limit(items, MAX_PHASE_ITEMS, "phase items")?;
    let elements = dim
        .checked_mul(items)
        .ok_or_else(|| PyValueError::new_err("phase element count overflow"))?;
    phase_limit(elements, MAX_PHASE_ELEMENTS, "phase elements")
}

fn resonator_work(dim: usize, items: usize, factors: usize, iterations: usize) -> PyResult<()> {
    phase_shape(dim, items)?;
    phase_limit(factors, MAX_PHASE_FACTORS, "resonator factors")?;
    phase_limit(iterations, MAX_PHASE_ITER, "resonator iterations")?;
    let work = factors
        .checked_mul(factors)
        .and_then(|pairs| items.checked_mul(3)?.checked_add(pairs))
        .and_then(|per_dim| per_dim.checked_mul(dim.max(1)))
        .and_then(|per_iteration| per_iteration.checked_mul(iterations + 1))
        .ok_or_else(|| PyValueError::new_err("resonator work count overflow"))?;
    phase_limit(work, MAX_PHASE_WORK, "resonator work")
}

#[cfg(feature = "local-embed")]
mod embed;
mod filter_index;
mod sparse_index;
mod vector_index;

/// A quantized-phase hypervector: `dim` phases in `Z_N`. Binding is phase-add mod N
/// (not self-inverse), bundling is a deterministic integer fold, similarity is the
/// mean cosine of phase differences.
#[pyclass(name = "PhaseHVec", from_py_object)]
#[derive(Clone)]
struct PyPhaseHVec {
    inner: PhaseHVec,
}

impl PyPhaseHVec {
    fn compat(&self, other: &Self) -> PyResult<()> {
        if self.inner.dim() != other.inner.dim() || self.inner.n() != other.inner.n() {
            return Err(PyValueError::new_err(
                "phase vectors differ in dimension or resolution",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PyPhaseHVec {
    /// A random phase vector of `dim` dimensions at resolution `n` (phase levels,
    /// 2..=65536), deterministic in `seed`.
    #[staticmethod]
    fn random(dim: usize, n: u32, seed: u64) -> PyResult<Self> {
        phase_shape(dim, 1)?;
        if !(2..=65536).contains(&n) {
            return Err(PyValueError::new_err(
                "n (phase resolution) must be in 2..=65536",
            ));
        }
        Ok(Self {
            inner: PhaseHVec::new_random(dim, n, seed),
        })
    }

    /// Bind two vectors (phase-add mod N). Non-self-inverse; inverse is `unbind`.
    fn bind(&self, other: &Self) -> PyResult<Self> {
        self.compat(other)?;
        Ok(Self {
            inner: self.inner.bind(&other.inner),
        })
    }

    /// Unbind (phase-sub mod N). `a.bind(b).unbind(b) == a`.
    fn unbind(&self, other: &Self) -> PyResult<Self> {
        self.compat(other)?;
        Ok(Self {
            inner: self.inner.unbind(&other.inner),
        })
    }

    /// Similarity: mean cosine of phase differences, in [-1, 1].
    fn similarity(&self, other: &Self) -> PyResult<f64> {
        self.compat(other)?;
        Ok(self.inner.similarity(&other.inner))
    }

    /// Deterministic superposition (bundle) of a list of phase vectors.
    #[staticmethod]
    fn bundle(items: Vec<PyRef<'_, PyPhaseHVec>>) -> PyResult<Self> {
        if items.is_empty() {
            return Err(PyValueError::new_err("cannot bundle an empty list"));
        }
        phase_shape(items[0].inner.dim(), items.len())?;
        for item in &items {
            items[0].compat(item)?;
        }
        let inners: Vec<&PhaseHVec> = items.iter().map(|p| &p.inner).collect();
        Ok(Self {
            inner: PhaseHVec::bundle(&inners),
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn n(&self) -> u32 {
        self.inner.n()
    }

    /// The raw phase indices as a list of ints in [0, n).
    fn phases(&self) -> Vec<u16> {
        self.inner.phases().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("PhaseHVec(dim={}, n={})", self.inner.dim(), self.inner.n())
    }
}

/// Factor a composite `bind(f0, f1, ...)` into one codebook index per factor via the
/// deterministic resonator. Returns `(index, similarity)` for each factor.
#[pyfunction]
#[pyo3(signature = (composite, codebooks, max_iter=50))]
fn resonator_factorize(
    composite: &PyPhaseHVec,
    codebooks: Vec<Vec<PyRef<'_, PyPhaseHVec>>>,
    max_iter: usize,
) -> PyResult<Vec<(usize, f64)>> {
    phase_limit(codebooks.len(), MAX_PHASE_FACTORS, "resonator factors")?;
    phase_limit(max_iter, MAX_PHASE_ITER, "resonator iterations")?;
    let mut items = 0usize;
    for book in &codebooks {
        if book.is_empty() {
            return Err(PyValueError::new_err("factor codebooks must not be empty"));
        }
        items = items
            .checked_add(book.len())
            .ok_or_else(|| PyValueError::new_err("codebook size overflow"))?;
        phase_shape(composite.inner.dim(), items)?;
        for entry in book {
            composite.compat(entry)?;
        }
    }
    resonator_work(composite.inner.dim(), items, codebooks.len(), max_iter)?;
    let cbs: Vec<Vec<PhaseHVec>> = codebooks
        .into_iter()
        .map(|cb| cb.into_iter().map(|p| p.inner.clone()).collect())
        .collect();
    let cfg = ResonatorConfig {
        max_iter,
        convergence_threshold: 0.999,
    };
    Ok(phase_resonator_factorize(&composite.inner, &cbs, &cfg)
        .into_iter()
        .map(|r| (r.codebook_entry, r.similarity))
        .collect())
}

#[pymodule]
fn writerslogic_warm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Deterministic, verifiable vector-symbolic (VSA) memory on a quantized-phase substrate.",
    )?;
    m.add_class::<PyPhaseHVec>()?;
    m.add_function(wrap_pyfunction!(resonator_factorize, m)?)?;
    vector_index::register(m)?;
    filter_index::register(m)?;
    sparse_index::register(m)?;
    #[cfg(feature = "local-embed")]
    embed::register(m)?;
    Ok(())
}
