// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings (pyo3) for the deterministic quantized-phase VSA substrate.
//!
//! Exposes the differentiated phase-vector surface -- non-self-inverse binding,
//! graceful similarity, deterministic bundling, and resonator factorization -- as a
//! pip-installable module built with `maturin` under the `python` feature. This is
//! the initial, curated surface (the phase substrate, qFHRR arXiv 2604.25939); it
//! will grow as the public API stabilises. See `pyproject.toml` and
//! `.github/workflows/publish-pypi.yml`. The module is named `holographic_memory`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::phase_hvec::PhaseHVec;
use crate::core::phase_resonator::phase_resonator_factorize;
use crate::core::resonator::ResonatorConfig;

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
    fn bundle(items: Vec<PyPhaseHVec>) -> PyResult<Self> {
        if items.is_empty() {
            return Err(PyValueError::new_err("cannot bundle an empty list"));
        }
        let inners: Vec<PhaseHVec> = items.into_iter().map(|p| p.inner).collect();
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
    codebooks: Vec<Vec<PyPhaseHVec>>,
    max_iter: usize,
) -> Vec<(usize, f64)> {
    let cbs: Vec<Vec<PhaseHVec>> = codebooks
        .into_iter()
        .map(|cb| cb.into_iter().map(|p| p.inner).collect())
        .collect();
    let cfg = ResonatorConfig {
        max_iter,
        convergence_threshold: 0.999,
    };
    phase_resonator_factorize(&composite.inner, &cbs, &cfg)
        .into_iter()
        .map(|r| (r.codebook_entry, r.similarity))
        .collect()
}

#[pymodule]
fn holographic_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Deterministic, verifiable vector-symbolic (VSA) memory on a quantized-phase substrate.",
    )?;
    m.add_class::<PyPhaseHVec>()?;
    m.add_function(wrap_pyfunction!(resonator_factorize, m)?)?;
    Ok(())
}
