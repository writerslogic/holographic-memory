// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Node surface of the local embedder (features `node-api` + `local-embed`): `LocalEmbedder` and
//! `SemanticIndex`. Inference runs on the blocking pool, never on the JS thread. The optional
//! package `holographic-memory-models` supplies model packs offline (`index.js` points
//! `HMS_EMBED_MODEL_DIR` at it).

use std::sync::Arc;

use crate::core::sync::RwLock;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::{napi_err, run_async};
use crate::core::embed::{EmbedKind, EmbedLimits, EmbedOptions, Embedder, Pooling, SemanticIndex};

fn checked_count(input: f64, minimum: u32, name: &str) -> anyhow::Result<usize> {
    anyhow::ensure!(
        input.is_finite()
            && input.fract() == 0.0
            && (f64::from(minimum)..=f64::from(u32::MAX)).contains(&input),
        "{name} must be a finite integer in {minimum}..=4294967295"
    );
    Ok(input as usize)
}

#[napi(object)]
#[derive(Default)]
pub struct EmbedLimitsJs {
    pub max_texts: Option<f64>,
    pub max_input_bytes: Option<f64>,
    pub max_tokens: Option<f64>,
    pub max_output_bytes: Option<f64>,
    pub max_concurrent_calls: Option<f64>,
}

impl TryFrom<EmbedLimitsJs> for EmbedLimits {
    type Error = anyhow::Error;

    fn try_from(value: EmbedLimitsJs) -> anyhow::Result<Self> {
        let defaults = Self::default();
        let limit = |input: Option<f64>, default, name| -> anyhow::Result<usize> {
            let Some(input) = input else {
                return Ok(default);
            };
            checked_count(input, 1, name)
        };
        Ok(Self {
            max_texts: limit(value.max_texts, defaults.max_texts, "max_texts")?,
            max_input_bytes: limit(
                value.max_input_bytes,
                defaults.max_input_bytes,
                "max_input_bytes",
            )?,
            max_tokens: limit(value.max_tokens, defaults.max_tokens, "max_tokens")?,
            max_output_bytes: limit(
                value.max_output_bytes,
                defaults.max_output_bytes,
                "max_output_bytes",
            )?,
            max_concurrent_calls: limit(
                value.max_concurrent_calls,
                defaults.max_concurrent_calls,
                "max_concurrent_calls",
            )?,
        })
    }
}

#[napi(object)]
pub struct EmbeddingInputJs {
    pub input_bytes: u32,
    pub retained_bytes: u32,
    pub tokens: u32,
    pub byte_truncated: bool,
    pub token_truncated: bool,
}

#[napi(object)]
pub struct EmbeddingBatchJs {
    pub embeddings: Vec<Float32Array>,
    pub inputs: Vec<EmbeddingInputJs>,
}

// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings

#[napi(object)]
#[derive(Default)]
pub struct EmbedOptionsJs {
    /// A registry id, "default", or the path of a `.hmsemb` pack.
    pub model: Option<String>,
    pub sha256: Option<String>,
    pub max_length: Option<f64>,
    /// "mean" or "cls".
    pub pooling: Option<String>,
    pub normalize: Option<bool>,
    pub query_prefix: Option<String>,
    pub passage_prefix: Option<String>,
    pub batch_size: Option<f64>,
    pub max_batch_tokens: Option<f64>,
    pub limits: Option<EmbedLimitsJs>,
    pub threads: Option<f64>,
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
            max_length: self
                .max_length
                .map(|v| checked_count(v, 2, "max_length"))
                .transpose()?,
            pooling: self
                .pooling
                .as_deref()
                .map(str::parse::<Pooling>)
                .transpose()?,
            normalize: self.normalize,
            query_prefix: self.query_prefix,
            passage_prefix: self.passage_prefix,
            batch_size: self
                .batch_size
                .map(|v| checked_count(v, 1, "batch_size"))
                .transpose()?
                .unwrap_or(d.batch_size),
            max_batch_tokens: self
                .max_batch_tokens
                .map(|v| checked_count(v, 1, "max_batch_tokens"))
                .transpose()?
                .unwrap_or(d.max_batch_tokens),
            limits: self.limits.unwrap_or_default().try_into()?,
            threads: self
                .threads
                .map(|v| checked_count(v, 0, "threads"))
                .transpose()?
                .unwrap_or(d.threads),
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

    #[napi]
    pub async fn embed_query_with_report(&self, texts: Vec<String>) -> Result<EmbeddingBatchJs> {
        self.run_with_report(texts, EmbedKind::Query).await
    }

    #[napi]
    pub async fn embed_passages_with_report(&self, texts: Vec<String>) -> Result<EmbeddingBatchJs> {
        self.run_with_report(texts, EmbedKind::Passage).await
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
    async fn run_with_report(
        &self,
        texts: Vec<String>,
        kind: EmbedKind,
    ) -> Result<EmbeddingBatchJs> {
        let embedder = self.inner.clone();
        let batch = run_async(move || embedder.embed_with_report(&texts, kind)).await?;
        let inputs = batch
            .inputs
            .into_iter()
            .map(|input| {
                Ok(EmbeddingInputJs {
                    input_bytes: u32::try_from(input.input_bytes)
                        .map_err(|e| napi_err(e.into()))?,
                    retained_bytes: u32::try_from(input.retained_bytes)
                        .map_err(|e| napi_err(e.into()))?,
                    tokens: u32::try_from(input.tokens).map_err(|e| napi_err(e.into()))?,
                    byte_truncated: input.byte_truncated,
                    token_truncated: input.token_truncated,
                })
            })
            .collect::<Result<_>>()?;
        Ok(EmbeddingBatchJs {
            embeddings: batch
                .embeddings
                .into_iter()
                .map(Float32Array::new)
                .collect(),
            inputs,
        })
    }

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
