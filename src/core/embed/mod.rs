// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Local text embeddings (feature `local-embed`): a pure-Rust, CPU, deterministic inference
//! engine for BERT-family sentence encoders with int8 or f16 weights.
//!
//! * No network at inference. Weights come from a sha256-verified *pack* file that is found by
//!   path, by `HMS_EMBED_PACK` / `HMS_EMBED_MODEL_DIR`, in the cache, or (feature
//!   `local-embed-download`, never when offline) fetched once from the pinned upstream files.
//! * Outputs are bit-identical for any thread count and any batch composition.
//! * Forward passes use a soft token budget, with one full sequence always admitted. Hard
//!   per-call input/token/output and concurrent-call limits apply independently.
//!
//! ```no_run
//! use holographic_memory::core::embed::{EmbedOptions, Embedder};
//! let embedder = Embedder::new(&EmbedOptions::default())?;
//! let q = embedder.embed_query(&["what is the capital of France?"])?;
//! let d = embedder.embed_passages(&["Paris is the capital of France."])?;
//! let cosine: f32 = q[0].iter().zip(&d[0]).map(|(a, b)| a * b).sum();
//! # Ok::<(), anyhow::Error>(())
//! ```

mod engine;
#[cfg(feature = "local-embed-download")]
mod fetch;
mod index;
mod kernels;
mod limits;
mod locate;
mod pack;
mod registry;
#[doc(hidden)]
pub mod testing;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(feature = "local-embed-download")]
pub use fetch::{download_pack as fetch_pack, ENV_MIRROR};
pub use index::{Hit, SemanticIndex};
pub use limits::EmbedLimits;
pub use locate::{ENV_CACHE, ENV_MODEL_DIR, ENV_OFFLINE, ENV_PACK};
pub use pack::{
    convert_hf_dir, sha256_file, sha256_hex, Arch, Meta, Pooling, Quantization, Sha256Mismatch,
    DEFAULT_INT8_GROUP,
};
pub use registry::{ModelSpec, PinnedFile, DEFAULT_MODEL, MODELS};

/// Whether texts are search queries or the passages being searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedKind {
    Query,
    Passage,
}

/// Preprocessing of one input, in the same order as the returned embeddings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingInput {
    pub input_bytes: usize,
    /// Source bytes retained before tokenization, excluding the model prefix.
    pub retained_bytes: usize,
    pub tokens: usize,
    pub byte_truncated: bool,
    pub token_truncated: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingBatch {
    pub embeddings: Vec<Vec<f32>>,
    pub inputs: Vec<EmbeddingInput>,
}

/// Tunable inference options. Every field has a default; nothing is forced.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct EmbedOptions {
    /// A registry id (see [`MODELS`]), `"default"`, or the path of a `.hmsemb` pack file.
    pub model: String,
    /// Expected sha256 of a pack given by path. Registry models are always checked against
    /// their pinned hash.
    pub sha256: Option<String>,
    /// Longest sequence in tokens, specials included (default: the model's, capped by its
    /// position table). Longer texts are truncated.
    pub max_length: Option<usize>,
    /// Overrides the model's pooling.
    pub pooling: Option<Pooling>,
    /// L2-normalize outputs (default: the model's convention, true for every registry model).
    pub normalize: Option<bool>,
    /// Overrides the model's query prefix (for example `"query: "`).
    pub query_prefix: Option<String>,
    /// Overrides the model's passage prefix.
    pub passage_prefix: Option<String>,
    /// Most sequences stacked into one forward pass.
    pub batch_size: usize,
    /// Soft token budget per forward pass. One sequence may exceed it, up to max_length.
    /// Tokenized inputs and output buffers for the whole call are additional allocations.
    pub max_batch_tokens: usize,
    pub limits: EmbedLimits,
    /// Worker threads of the embedder's private pool; 0 uses the available parallelism.
    pub threads: usize,
    /// Never touch the network. `None` follows `HMS_EMBED_OFFLINE`.
    pub offline: Option<bool>,
    /// Directory of ready packs searched before the cache. `None` follows `HMS_EMBED_MODEL_DIR`.
    pub model_dir: Option<PathBuf>,
    /// Cache directory for downloaded packs. `None` follows `HMS_EMBED_CACHE`, then the
    /// platform cache directory.
    pub cache_dir: Option<PathBuf>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            model: "default".into(),
            sha256: None,
            max_length: None,
            pooling: None,
            normalize: None,
            query_prefix: None,
            passage_prefix: None,
            batch_size: 32,
            max_batch_tokens: 2048,
            limits: EmbedLimits::default(),
            threads: 0,
            offline: None,
            model_dir: None,
            cache_dir: None,
        }
    }
}

impl EmbedOptions {
    fn validate(&self) -> Result<()> {
        ensure!(self.batch_size >= 1, "batch_size must be at least 1");
        ensure!(
            self.max_batch_tokens >= 1,
            "max_batch_tokens must be at least 1"
        );
        self.limits.validate()
    }

    /// Options for a registry model or pack path with everything else defaulted.
    pub fn model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Self::default()
        }
    }
}

/// A loaded sentence encoder. Cheap to share (`Arc<Embedder>`); concurrent calls are safe.
pub struct Embedder {
    encoder: engine::Encoder,
    tokenizer: tokenizers::Tokenizer,
    meta: Meta,
    arch: Arch,
    quantization: Quantization,
    sha256: String,
    pack_bytes: u64,
    pooling: Pooling,
    normalize: bool,
    max_length: usize,
    query_prefix: String,
    passage_prefix: String,
    batch_size: usize,
    max_batch_tokens: usize,
    limits: EmbedLimits,
    active_calls: AtomicUsize,
    pool: rayon::ThreadPool,
    source: PathBuf,
    space: crate::core::schema::EmbeddingSpace,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder")
            .field("model", &self.meta.model)
            .field("revision", &self.meta.revision)
            .field("quantization", &self.quantization)
            .field("dimensions", &self.arch.hidden)
            .finish()
    }
}

impl Embedder {
    /// Resolves the pack for `options.model` (path, env, cache, then download unless offline),
    /// verifies its sha256 and loads it.
    pub fn new(options: &EmbedOptions) -> Result<Self> {
        options.validate()?;
        let located = locate::resolve(options)?;
        let loaded = Self::from_pack(&located.path, located.sha256.as_deref(), options);
        // A damaged cache entry is replaced by a fresh verified download when that is allowed;
        // a file the caller named (path, env, model directory) is never touched.
        let damaged = located.from_cache
            && !locate::offline(options)
            && cfg!(feature = "local-embed-download")
            && loaded
                .as_ref()
                .err()
                .is_some_and(|e| e.downcast_ref::<Sha256Mismatch>().is_some());
        if !damaged {
            return loaded;
        }
        #[cfg(feature = "local-embed-download")]
        {
            let spec = registry::find(&options.model)?;
            let path = fetch::download_pack(spec, &locate::cache_dir(options)?)?;
            Self::from_pack(&path, Some(spec.pack_sha256), options)
        }
        #[cfg(not(feature = "local-embed-download"))]
        loaded
    }

    /// Loads a pack file. `expected_sha256` is checked against the bytes actually parsed.
    pub fn from_pack(
        path: &Path,
        expected_sha256: Option<&str>,
        options: &EmbedOptions,
    ) -> Result<Self> {
        options.validate()?;
        let mut loaded = pack::read(path, expected_sha256)?;
        let encoder = engine::Encoder::from_pack(&mut loaded)?;
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(&loaded.tokenizer)
            .map_err(|e| anyhow::anyhow!("pack tokenizer: {e}"))?;
        let arch = loaded.arch.clone();
        let usable = arch.max_positions - arch.position_offset;
        let max_length = options
            .max_length
            .unwrap_or(loaded.meta.max_length)
            .min(usable);
        ensure!(max_length >= 2, "max_length must be at least 2");
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None);
        let threads = if options.threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            options.threads
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("hms-embed-{i}"))
            .build()
            .context("build the embedder thread pool")?;
        let pooling = options.pooling.unwrap_or(loaded.meta.pooling);
        let normalize = options.normalize.unwrap_or(loaded.meta.normalize);
        let query_prefix = options
            .query_prefix
            .clone()
            .unwrap_or_else(|| loaded.meta.query_prefix.clone());
        let passage_prefix = options
            .passage_prefix
            .clone()
            .unwrap_or_else(|| loaded.meta.passage_prefix.clone());
        let space = crate::core::schema::EmbeddingSpace::effective(
            loaded.meta.model.clone(),
            arch.hidden,
            if normalize { "l2" } else { "none" },
            &(
                "bert-unpadded-exact-gelu-v1",
                &loaded.sha256,
                pooling,
                normalize,
                max_length,
                &query_prefix,
                &passage_prefix,
                "utf8-prefix-32-bytes-per-token-v1",
            ),
        )?;
        Ok(Self {
            space,
            pooling,
            normalize,
            query_prefix,
            passage_prefix,
            max_length,
            batch_size: options.batch_size,
            max_batch_tokens: options.max_batch_tokens,
            limits: options.limits.clone(),
            active_calls: AtomicUsize::new(0),
            quantization: loaded.quantization,
            sha256: loaded.sha256,
            pack_bytes: loaded.file_bytes,
            meta: loaded.meta,
            arch,
            encoder,
            tokenizer,
            pool,
            source: path.to_path_buf(),
        })
    }

    pub fn dimensions(&self) -> usize {
        self.arch.hidden
    }

    pub fn model(&self) -> &str {
        &self.meta.model
    }

    pub fn revision(&self) -> &str {
        &self.meta.revision
    }

    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// sha256 of the pack that was loaded.
    pub fn pack_sha256(&self) -> &str {
        &self.sha256
    }

    pub fn pack_path(&self) -> &Path {
        &self.source
    }

    pub fn pack_bytes(&self) -> u64 {
        self.pack_bytes
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Resident bytes of the model weights.
    pub fn weight_bytes(&self) -> usize {
        self.encoder.weight_bytes()
    }

    /// The store embedding space for documents embedded by this model: the schema check
    /// rejects a store opened with a different model, revision, quantization or dimension.
    pub fn space(&self) -> crate::core::schema::EmbeddingSpace {
        self.space.clone()
    }

    pub fn embed_query<S: AsRef<str> + Sync>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts, EmbedKind::Query)
    }

    pub fn embed_passages<S: AsRef<str> + Sync>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts, EmbedKind::Passage)
    }

    /// One vector per text, in input order (L2-normalized unless disabled).
    /// Text is truncated to max_length tokens after a UTF-8 byte prefix cap of
    /// max(256, 32 * max_length). [`Self::embed_with_report`] exposes both truncation stages.
    pub fn embed<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        kind: EmbedKind,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(self.embed_with_report(texts, kind)?.embeddings)
    }

    pub fn embed_with_report<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        kind: EmbedKind,
    ) -> Result<EmbeddingBatch> {
        let _lease =
            limits::CallLease::acquire(&self.active_calls, self.limits.max_concurrent_calls)?;
        let values = self.validate_inputs(texts, kind)?;
        let mut flat = Vec::new();
        flat.try_reserve_exact(values)
            .context("allocate embedding output")?;
        flat.resize(values, 0f32);
        let inputs = self.run_into(texts, kind, &mut flat)?;
        let mut embeddings = Vec::new();
        embeddings
            .try_reserve_exact(texts.len())
            .context("allocate embedding rows")?;
        for row in flat.chunks_exact(self.dimensions()) {
            let mut copy = Vec::new();
            copy.try_reserve_exact(row.len())
                .context("allocate embedding row")?;
            copy.extend_from_slice(row);
            embeddings.push(copy);
        }
        Ok(EmbeddingBatch { embeddings, inputs })
    }

    pub(crate) fn validate_inputs<S: AsRef<str>>(
        &self,
        texts: &[S],
        kind: EmbedKind,
    ) -> Result<usize> {
        let values = self.limits.output_values(texts.len(), self.dimensions())?;
        let prefix = match kind {
            EmbedKind::Query => &self.query_prefix,
            EmbedKind::Passage => &self.passage_prefix,
        };
        let mut bytes = 0usize;
        for text in texts {
            bytes = bytes
                .checked_add(text.as_ref().len())
                .and_then(|n| n.checked_add(prefix.len()))
                .context("embedding input size overflow")?;
            ensure!(
                bytes <= self.limits.max_input_bytes,
                "embedding input exceeds max_input_bytes"
            );
        }
        Ok(values)
    }

    /// Writes `texts.len() * dimensions()` values into a caller-owned buffer.
    pub fn embed_into<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        kind: EmbedKind,
        out: &mut [f32],
    ) -> Result<()> {
        let _lease =
            limits::CallLease::acquire(&self.active_calls, self.limits.max_concurrent_calls)?;
        let values = self.validate_inputs(texts, kind)?;
        ensure!(
            out.len() == values,
            "output buffer must hold {} values",
            values
        );
        self.run_into(texts, kind, out).map(|_| ())
    }

    fn run_into<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        kind: EmbedKind,
        out: &mut [f32],
    ) -> Result<Vec<EmbeddingInput>> {
        let dim = self.dimensions();
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefix = match kind {
            EmbedKind::Query => &self.query_prefix,
            EmbedKind::Passage => &self.passage_prefix,
        };
        let total_tokens = AtomicUsize::new(0);
        let tokenized: Vec<(Vec<u32>, EmbeddingInput)> = self.pool.install(|| {
            use rayon::prelude::*;
            texts
                .par_iter()
                .map(|t| {
                    let (ids, info) = self.tokenize(prefix, t.as_ref())?;
                    limits::reserve(&total_tokens, ids.len(), self.limits.max_tokens, "tokens")?;
                    Ok((ids, info))
                })
                .collect::<Result<_>>()
        })?;
        // Length-sorted batches keep stacked work uniform; results return to input order.
        let mut order: Vec<usize> = (0..tokenized.len()).collect();
        order.sort_by_key(|&i| (tokenized[i].0.len(), i));
        let mut scratch = engine::Scratch::default();
        let mut at = 0;
        while at < order.len() {
            let mut end = at;
            let mut tokens = 0;
            while end < order.len()
                && end - at < self.batch_size
                && (end == at
                    || tokenized[order[end]].0.len()
                        <= self.max_batch_tokens.saturating_sub(tokens))
            {
                tokens += tokenized[order[end]].0.len();
                end += 1;
            }
            let seqs: Vec<&[u32]> = order[at..end]
                .iter()
                .map(|&i| tokenized[i].0.as_slice())
                .collect();
            let pooled = self
                .pool
                .install(|| self.encoder.encode(&seqs, self.pooling, &mut scratch));
            for (&i, mut v) in order[at..end].iter().zip(pooled) {
                if self.normalize {
                    let norm = v
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                        .sqrt()
                        .max(1e-12);
                    v.iter_mut()
                        .for_each(|x| *x = (f64::from(*x) / norm) as f32);
                }
                out[i * dim..(i + 1) * dim].copy_from_slice(&v);
            }
            at = end;
        }
        Ok(tokenized.into_iter().map(|(_, info)| info).collect())
    }

    fn tokenize(&self, prefix: &str, text: &str) -> Result<(Vec<u32>, EmbeddingInput)> {
        // Truncation happens on tokens, but the tokenizer still scans the whole string, so cap the
        // input first: no model reads more than a few bytes per token it keeps.
        let cap = self.max_length.saturating_mul(32).max(256);
        let mut end = text.len().min(cap);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let joined;
        let input = if prefix.is_empty() {
            &text[..end]
        } else {
            joined = format!("{prefix}{}", &text[..end]);
            joined.as_str()
        };
        let enc = self
            .tokenizer
            .encode(input, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        if ids.is_empty() {
            bail!("tokenizer produced no tokens");
        }
        ids.truncate(self.max_length);
        if ids.iter().any(|&id| id as usize >= self.arch.vocab) {
            bail!(
                "the pack's tokenizer emits ids outside the model vocabulary ({})",
                self.arch.vocab
            );
        }
        let info = EmbeddingInput {
            input_bytes: text.len(),
            retained_bytes: end,
            tokens: ids.len(),
            byte_truncated: end < text.len(),
            token_truncated: !enc.get_overflowing().is_empty(),
        };
        Ok((ids, info))
    }
}
