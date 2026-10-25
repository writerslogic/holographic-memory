// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Python surface of the local embedder (feature `local-embed`): `Embedder` and `SemanticIndex`.
//! Inference runs without the GIL. The companion package `holographic_memory_models`, when
//! installed, supplies its pinned model pack without a first-use download.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::embed::{EmbedKind, EmbedOptions, Embedder, Pooling, SemanticIndex};

fn err(e: anyhow::Error) -> PyErr {
    PyValueError::new_err(format!("{e:#}"))
}

#[derive(FromPyObject)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

/// A directory of model packs installed by the companion package, if any.
fn companion_dir(py: Python<'_>) -> Option<PathBuf> {
    let module = py.import("holographic_memory_models").ok()?;
    module
        .call_method0("pack_dir")
        .ok()?
        .extract::<PathBuf>()
        .ok()
}

/// A local sentence embedder: pure-Rust CPU inference, no network at inference.
#[pyclass(name = "Embedder")]
struct PyEmbedder {
    inner: Arc<Embedder>,
}

#[pymethods]
impl PyEmbedder {
    /// `model` is a registry id, "default", or the path of a `.hmsemb` pack.
    #[new]
    #[pyo3(signature = (
        model="default", *, sha256=None, max_length=None, pooling=None, normalize=None,
        query_prefix=None, passage_prefix=None, batch_size=32, max_batch_tokens=2048,
        threads=0, offline=None, model_dir=None, cache_dir=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: &str,
        sha256: Option<String>,
        max_length: Option<usize>,
        pooling: Option<&str>,
        normalize: Option<bool>,
        query_prefix: Option<String>,
        passage_prefix: Option<String>,
        batch_size: usize,
        max_batch_tokens: usize,
        threads: usize,
        offline: Option<bool>,
        model_dir: Option<PathBuf>,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let options = EmbedOptions {
            model: model.into(),
            sha256,
            max_length,
            pooling: pooling
                .map(str::parse::<Pooling>)
                .transpose()
                .map_err(err)?,
            normalize,
            query_prefix,
            passage_prefix,
            batch_size,
            max_batch_tokens,
            threads,
            offline,
            model_dir: model_dir.or_else(|| companion_dir(py)),
            cache_dir,
        };
        let inner = py.detach(|| Embedder::new(&options)).map_err(err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Embeds search queries: a list of floats for one string, a list of lists for a list.
    fn embed_query<'py>(&self, py: Python<'py>, texts: OneOrMany) -> PyResult<Bound<'py, PyAny>> {
        self.run(py, texts, EmbedKind::Query)
    }

    /// Embeds passages (the texts being searched).
    fn embed_passages<'py>(
        &self,
        py: Python<'py>,
        texts: OneOrMany,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.run(py, texts, EmbedKind::Passage)
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    #[getter]
    fn model(&self) -> String {
        self.inner.model().into()
    }

    #[getter]
    fn revision(&self) -> String {
        self.inner.revision().into()
    }

    #[getter]
    fn quantization(&self) -> &'static str {
        self.inner.quantization().name()
    }

    #[getter]
    fn max_length(&self) -> usize {
        self.inner.max_length()
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }

    #[getter]
    fn pack_sha256(&self) -> String {
        self.inner.pack_sha256().into()
    }

    #[getter]
    fn pack_path(&self) -> String {
        self.inner.pack_path().display().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Embedder(model={:?}, revision={:?}, quantization={:?}, dimensions={})",
            self.inner.model(),
            self.inner.revision(),
            self.inner.quantization().name(),
            self.inner.dimensions()
        )
    }
}

impl PyEmbedder {
    fn run<'py>(
        &self,
        py: Python<'py>,
        texts: OneOrMany,
        kind: EmbedKind,
    ) -> PyResult<Bound<'py, PyAny>> {
        let embedder = self.inner.clone();
        match texts {
            OneOrMany::One(t) => {
                let v = py
                    .detach(|| embedder.embed(&[t], kind))
                    .map_err(err)?
                    .remove(0);
                Ok(v.into_pyobject(py)?.into_any())
            }
            OneOrMany::Many(ts) => {
                let v = py.detach(|| embedder.embed(&ts, kind)).map_err(err)?;
                Ok(v.into_pyobject(py)?.into_any())
            }
        }
    }
}

/// The given embedder, or a default one loading from the companion package's model packs.
fn embedder_or_default(
    py: Python<'_>,
    embedder: Option<PyRef<'_, PyEmbedder>>,
) -> PyResult<Arc<Embedder>> {
    match embedder {
        Some(e) => Ok(e.inner.clone()),
        None => {
            let options = EmbedOptions {
                model_dir: companion_dir(py),
                ..EmbedOptions::default()
            };
            Ok(Arc::new(
                py.detach(|| Embedder::new(&options)).map_err(err)?,
            ))
        }
    }
}

/// An exact in-memory semantic index: `add`, `search`, `remove`, `save`, `load`.
#[pyclass(name = "SemanticIndex")]
struct PySemanticIndex {
    inner: SemanticIndex,
}

#[pymethods]
impl PySemanticIndex {
    /// Uses `embedder`, or a default `Embedder()` when none is given.
    #[new]
    #[pyo3(signature = (embedder=None))]
    fn new(py: Python<'_>, embedder: Option<PyRef<'_, PyEmbedder>>) -> PyResult<Self> {
        let embedder = embedder_or_default(py, embedder)?;
        Ok(Self {
            inner: SemanticIndex::new(embedder),
        })
    }

    /// Adds or replaces items; `ids` and `texts` are equal-length lists.
    fn add(&mut self, py: Python<'_>, ids: Vec<String>, texts: Vec<String>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.add(&ids, &texts)).map_err(err)
    }

    /// The `k` best `(id, score, text)` triples, best first.
    #[pyo3(signature = (query, k=10))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        k: usize,
    ) -> PyResult<Vec<(String, f32, String)>> {
        let inner = &self.inner;
        let hits = py.detach(|| inner.search(query, k)).map_err(err)?;
        Ok(hits.into_iter().map(|h| (h.id, h.score, h.text)).collect())
    }

    fn remove(&mut self, id: &str) -> bool {
        self.inner.remove(id)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(&path).map_err(err)
    }

    /// Loads a saved index; it must have been built with the same model pack.
    #[staticmethod]
    #[pyo3(signature = (path, embedder=None))]
    fn load(
        py: Python<'_>,
        path: PathBuf,
        embedder: Option<PyRef<'_, PyEmbedder>>,
    ) -> PyResult<Self> {
        let embedder = embedder_or_default(py, embedder)?;
        Ok(Self {
            inner: SemanticIndex::load(&path, embedder).map_err(err)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmbedder>()?;
    m.add_class::<PySemanticIndex>()?;
    Ok(())
}
