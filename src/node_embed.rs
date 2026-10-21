// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Node surface of the local embedder (features `node-api` + `local-embed`): `LocalEmbedder` and
//! `SemanticIndex`. Inference runs on the blocking pool, never on the JS thread. The optional
//! package `holographic-memory-models` supplies model packs offline (`index.js` points
//! `HMS_EMBED_MODEL_DIR` at it).

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::RwLock;

use crate::core::embed::{EmbedKind, EmbedOptions, Embedder, Pooling, SemanticIndex};
use crate::{napi_err, run_async};

// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

#[napi(object)]
#[derive(Default)]
pub struct EmbedOptionsJs {
    /// A registry id, "default", or the path of a `.hmsemb` pack.
    pub model: Option<String>,
    pub sha256: Option<String>,
    pub max_length: Option<u32>,
    /// "mean" or "cls".
    pub pooling: Option<String>,
    pub normalize: Option<bool>,
    pub query_prefix: Option<String>,
    pub passage_prefix: Option<String>,
    pub batch_size: Option<u32>,
    pub max_batch_tokens: Option<u32>,
    pub threads: Option<u32>,
    pub offline: Option<bool>,
    pub model_dir: Option<String>,
    pub cache_dir: Option<String>,
}

impl EmbedOptionsJs {
    fn into_options(self) -> anyhow::Result<EmbedOptions> {
        let d = EmbedOptions::default();
        Ok(EmbedOptions {
            model: self.model.unwrap_or(d.model),
            sha256: self.sha256,
            max_length: self.max_length.map(|v| v as usize),
            pooling: self
                .pooling
                .as_deref()
                .map(str::parse::<Pooling>)
                .transpose()?,
            normalize: self.normalize,
            query_prefix: self.query_prefix,
            passage_prefix: self.passage_prefix,
            batch_size: self.batch_size.map_or(d.batch_size, |v| v as usize),
            max_batch_tokens: self
                .max_batch_tokens
                .map_or(d.max_batch_tokens, |v| v as usize),
            threads: self.threads.map_or(d.threads, |v| v as usize),
            offline: self.offline,
            model_dir: self.model_dir.map(Into::into),
            cache_dir: self.cache_dir.map(Into::into),
        })
    }
}

async fn load(options: Option<EmbedOptionsJs>) -> Result<Arc<Embedder>> {
    let options = options
        .unwrap_or_default()
        .into_options()
        .map_err(napi_err)?;
    run_async(move || Embedder::new(&options).map(Arc::new)).await
}

// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

#[napi]
pub struct LocalEmbedder {
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    inner: Arc<Embedder>,
}

// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

#[napi]
impl LocalEmbedder {
    /// Loads (and on first use downloads) the model: `await LocalEmbedder.load()`.
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    #[napi(factory)]
    pub async fn load(options: Option<EmbedOptionsJs>) -> Result<Self> {
        Ok(Self {
            inner: load(options).await?,
        })
    }

    /// One L2-normalized vector per query text.
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    #[napi]
    pub async fn embed_query(&self, texts: Vec<String>) -> Result<Vec<Float32Array>> {
        self.run(texts, EmbedKind::Query).await
    }

    /// One L2-normalized vector per passage text.
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    #[napi]
    pub async fn embed_passages(&self, texts: Vec<String>) -> Result<Vec<Float32Array>> {
        self.run(texts, EmbedKind::Passage).await
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi(getter)]
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions() as u32
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi(getter)]
    pub fn model(&self) -> String {
        self.inner.model().into()
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi(getter)]
    pub fn revision(&self) -> String {
        self.inner.revision().into()
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi(getter)]
    pub fn quantization(&self) -> String {
        self.inner.quantization().name().into()
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi(getter)]
    pub fn pack_sha256(&self) -> String {
        self.inner.pack_sha256().into()
    }
}

impl LocalEmbedder {
    async fn run(&self, texts: Vec<String>, kind: EmbedKind) -> Result<Vec<Float32Array>> {
        let embedder = self.inner.clone();
        let rows = run_async(move || embedder.embed(&texts, kind)).await?;
        Ok(rows.into_iter().map(Float32Array::new).collect())
    }
}

// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

#[napi(object)]
pub struct SemanticHit {
    pub id: String,
    pub score: f64,
    pub text: String,
}

/// An exact in-memory semantic index.
// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
#[napi(js_name = "SemanticIndex")]
pub struct JsSemanticIndex {
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    inner: Arc<RwLock<SemanticIndex>>,
}

// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

#[napi]
impl JsSemanticIndex {
    /// An empty index over a loaded embedder: `SemanticIndex.create(await LocalEmbedder.load())`.
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    #[napi(factory)]
    pub fn create(embedder: &LocalEmbedder) -> Self {
        Self {
            inner: Arc::new(RwLock::new(SemanticIndex::new(embedder.inner.clone()))),
        }
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi]
    pub async fn add(&self, ids: Vec<String>, texts: Vec<String>) -> Result<()> {
        let index = self.inner.clone();
        run_async(move || index.write().add(&ids, &texts)).await
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi]
    pub async fn search(&self, query: String, k: Option<u32>) -> Result<Vec<SemanticHit>> {
        let index = self.inner.clone();
        let hits = run_async(move || index.read().search(&query, k.unwrap_or(10) as usize)).await?;
        Ok(hits
            .into_iter()
            .map(|h| SemanticHit {
                id: h.id,
                score: f64::from(h.score),
                text: h.text,
            })
            .collect())
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi]
    pub fn remove(&self, id: String) -> bool {
        self.inner.write().remove(&id)
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi]
    pub async fn save(&self, path: String) -> Result<()> {
        let index = self.inner.clone();
        run_async(move || index.read().save(std::path::Path::new(&path))).await
    }

    /// Loads a saved index; it must have been built with the same model pack.
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    #[napi(factory)]
    pub fn load(path: String, embedder: &LocalEmbedder) -> Result<Self> {
        let index = SemanticIndex::load(std::path::Path::new(&path), embedder.inner.clone())
            .map_err(napi_err)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(index)),
        })
    }

    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.read().len() as u32
    }
}
