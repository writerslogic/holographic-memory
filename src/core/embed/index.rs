// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! A small exact semantic index over an [`Embedder`]: add texts, search by meaning, save and
//! load. Brute-force cosine over a flat f32 matrix, so results are exact and deterministic
//! (ties break toward the earlier insertion); sized for up to a few hundred thousand passages.
//! Larger or filtered corpora belong in the document store (`TextEncoder::LocalEmbedV1`).

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

use super::{EmbedKind, Embedder};

const MAGIC: &[u8; 8] = b"HMSSIDX1";
const MAX_HEADER: usize = 1 << 30;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Hit {
    pub id: String,
    pub score: f32,
    pub text: String,
}

#[derive(Serialize, Deserialize)]
struct Header {
    format: u32,
    model: String,
    revision: String,
    pack_sha256: String,
    #[serde(default)]
    embedding_space: Option<crate::core::schema::EmbeddingSpace>,
    dimensions: usize,
    ids: Vec<String>,
    texts: Vec<String>,
}

pub struct SemanticIndex {
    embedder: Arc<Embedder>,
    ids: Vec<String>,
    texts: Vec<String>,
    vectors: Vec<f32>,
    position: HashMap<String, usize>,
}

impl SemanticIndex {
    pub fn new(embedder: Arc<Embedder>) -> Self {
        Self {
            embedder,
            ids: Vec::new(),
            texts: Vec::new(),
            vectors: Vec::new(),
            position: HashMap::new(),
        }
    }

    pub fn embedder(&self) -> &Arc<Embedder> {
        &self.embedder
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Adds or replaces `(id, text)` items, embedding all of them in one batched pass.
    pub fn add<I: AsRef<str>, T: AsRef<str> + Sync>(
        &mut self,
        ids: &[I],
        texts: &[T],
    ) -> Result<()> {
        ensure!(
            ids.len() == texts.len(),
            "ids and texts must have the same length"
        );
        let mut seen = std::collections::HashSet::new();
        for id in ids {
            ensure!(!id.as_ref().is_empty(), "ids must not be empty");
            ensure!(
                seen.insert(id.as_ref()),
                "duplicate id {:?} in one call",
                id.as_ref()
            );
        }
        let vectors = self.embedder.embed(texts, EmbedKind::Passage)?;
        let dim = self.embedder.dimensions();
        for ((id, text), v) in ids.iter().zip(texts).zip(vectors) {
            match self.position.get(id.as_ref()) {
                Some(&at) => {
                    self.texts[at] = text.as_ref().to_string();
                    self.vectors[at * dim..(at + 1) * dim].copy_from_slice(&v);
                }
                None => {
                    self.position
                        .insert(id.as_ref().to_string(), self.ids.len());
                    self.ids.push(id.as_ref().to_string());
                    self.texts.push(text.as_ref().to_string());
                    self.vectors.extend_from_slice(&v);
                }
            }
        }
        Ok(())
    }

    /// Removes an item; returns whether it existed. Later items keep their relative order.
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(at) = self.position.remove(id) else {
            return false;
        };
        let dim = self.embedder.dimensions();
        self.ids.remove(at);
        self.texts.remove(at);
        self.vectors.drain(at * dim..(at + 1) * dim);
        for p in self.position.values_mut().filter(|p| **p > at) {
            *p -= 1;
        }
        true
    }

    /// The `k` most similar items to `query`, best first.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<Hit>> {
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let q = self.embedder.embed_query(&[query])?.remove(0);
        Ok(self.top_k(&q, k))
    }

    /// Like [`search`](Self::search) for a ready query vector (must be `dimensions()` long).
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let dim = self.embedder.dimensions();
        assert_eq!(query.len(), dim, "query vector has the wrong dimension");
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f32, usize)> = self
            .vectors
            .chunks_exact(dim)
            .enumerate()
            .map(|(i, v)| (super::kernels::dot(query, v), i))
            .collect();
        let k = k.min(scored.len());
        let order = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        if k < scored.len() {
            scored.select_nth_unstable_by(k - 1, order);
            scored.truncate(k);
        }
        scored.sort_unstable_by(order);
        scored
            .into_iter()
            .map(|(score, i)| Hit {
                id: self.ids[i].clone(),
                score,
                text: self.texts[i].clone(),
            })
            .collect()
    }

    /// Writes the index atomically (temporary file, then rename).
    pub fn save(&self, path: &Path) -> Result<()> {
        let header = serde_json::to_vec(&Header {
            format: 2,
            model: self.embedder.model().into(),
            revision: self.embedder.revision().into(),
            pack_sha256: self.embedder.pack_sha256().into(),
            embedding_space: Some(self.embedder.space()),
            dimensions: self.embedder.dimensions(),
            ids: self.ids.clone(),
            texts: self.texts.clone(),
        })?;
        let mut bytes = Vec::with_capacity(16 + header.len() + 4 * self.vectors.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        for x in &self.vectors {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        // A temporary file of its own, so no file of any name is truncated or consumed.
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(parent)
            .with_context(|| format!("create a temporary file in {}", parent.display()))?;
        tmp.write_all(&bytes)?;
        tmp.as_file().sync_all()?;
        tmp.persist(path)
            .with_context(|| format!("install {}", path.display()))?;
        crate::core::durable_file::sync_directory(parent)
    }

    /// Loads an index written by [`save`](Self::save). It must have been built with the same
    /// pack (same weights and quantization), because vectors of different models do not compare.
    pub fn load(path: &Path, embedder: Arc<Embedder>) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            bytes.len() >= 16 && &bytes[..8] == MAGIC,
            "{} is not a semantic index",
            path.display()
        );
        let hl = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")) as usize;
        ensure!(
            hl <= MAX_HEADER && 16 + hl <= bytes.len(),
            "index header is invalid"
        );
        let header: Header = serde_json::from_slice(&bytes[16..16 + hl]).context("index header")?;
        ensure!(
            header.format == 2,
            "unsupported or legacy index format {}; rebuild from source texts to bind the effective embedding function",
            header.format
        );
        ensure!(header.embedding_space.as_ref() == Some(&embedder.space()),
            "semantic index embedding function changed; rebuild from original source texts with the intended model and options");
        if header.pack_sha256 != embedder.pack_sha256() {
            bail!(
                "the index was built with {} {} (pack {}) but the embedder is {} {} (pack {}); rebuild it with the same model",
                header.model,
                header.revision,
                header.pack_sha256,
                embedder.model(),
                embedder.revision(),
                embedder.pack_sha256()
            );
        }
        ensure!(
            header.dimensions == embedder.dimensions(),
            "index dimensions do not match the embedder"
        );
        let n = header.ids.len();
        ensure!(
            header.texts.len() == n,
            "index ids and texts differ in length"
        );
        let data = &bytes[16 + hl..];
        let expected = n
            .checked_mul(header.dimensions)
            .and_then(|n| n.checked_mul(4))
            .context("index shape overflows address space")?;
        ensure!(
            data.len() == expected,
            "index vector data has the wrong size"
        );
        let vectors = super::pack::f32s_le(data);
        ensure!(
            vectors.iter().all(|v| v.is_finite()),
            "index contains non-finite values"
        );
        let mut position = HashMap::with_capacity(n);
        for (i, id) in header.ids.iter().enumerate() {
            ensure!(
                position.insert(id.clone(), i).is_none(),
                "index has duplicate id {id:?}"
            );
        }
        Ok(Self {
            embedder,
            ids: header.ids,
            texts: header.texts,
            vectors,
            position,
        })
    }
}
