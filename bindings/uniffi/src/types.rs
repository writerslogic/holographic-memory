// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use holographic_memory as hms;
use serde_json::Value;

use crate::error::HmsError;

/// How a store is opened. Every field but `path` has a default.
#[derive(Clone, Debug, uniffi::Record)]
pub struct StoreOptions {
    /// Hypervector dimensions, at least 256. Fixed when the store is created; reopening with
    /// another value is refused. Larger is more accurate and larger on disk and in memory.
    #[uniffi(default = 4096)]
    pub dimensions: u32,
    /// Passphrase that encrypts the store (journal and chunk-text cache; Argon2id key derivation,
    /// AES-256-GCM). `None` stores plaintext. Fixed when the store is created. Use a random
    /// secret from the platform keystore, not a short user-typed password.
    #[uniffi(default = None)]
    pub encryption_key: Option<String>,
    /// Identifies the embedding model whose vectors callers pass in `embeddings`/`embedding`.
    /// Set together with `embedding_dimensions` to enable semantic and hybrid search.
    #[uniffi(default = None)]
    pub embedding_model: Option<String>,
    #[uniffi(default = None)]
    pub embedding_revision: Option<String>,
    #[uniffi(default = None)]
    pub embedding_dimensions: Option<u32>,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            dimensions: 4096,
            encryption_key: None,
            embedding_model: None,
            embedding_revision: None,
            embedding_dimensions: None,
        }
    }
}

impl StoreOptions {
    pub(crate) fn to_config(&self) -> Result<hms::core::HmsConfig, HmsError> {
        let mut config = hms::core::HmsConfig::default();
        if let Some(key) = &self.encryption_key {
            if key.is_empty() {
                return Err(HmsError::invalid("encryption key must not be empty"));
            }
            config.security.encryption_enabled = true;
            config.security.encryption_passphrase = Some(key.clone());
        }
        if self.embedding_model.is_some()
            || self.embedding_revision.is_some()
            || self.embedding_dimensions.is_some()
        {
            let dimensions = self.embedding_dimensions.ok_or_else(|| {
                HmsError::invalid("embeddingDimensions is required with an embedding model")
            })?;
            config.embedding_space = Some(hms::EmbeddingSpace {
                model: self.embedding_model.clone().unwrap_or_default(),
                revision: self.embedding_revision.clone().unwrap_or_default(),
                dimensions: dimensions as usize,
                normalization: "l2".into(),
                metric: "cosine".into(),
            });
        }
        Ok(config)
    }
}

/// A document to ingest. Re-ingesting an id replaces the document.
#[derive(Clone, Debug, uniffi::Record)]
pub struct DocumentInput {
    pub id: String,
    pub text: String,
    #[uniffi(default = None)]
    pub source_uri: Option<String>,
    #[uniffi(default = None)]
    pub version: Option<String>,
    /// A JSON object. Scalar members can be matched by `SearchOptions.filterJson`.
    #[uniffi(default = None)]
    pub metadata_json: Option<String>,
    /// Words per chunk; the engine default applies when absent.
    #[uniffi(default = None)]
    pub chunk_words: Option<u32>,
    #[uniffi(default = None)]
    pub overlap_words: Option<u32>,
    /// Keep the chunk text so results can return it; the engine default applies when absent.
    #[uniffi(default = None)]
    pub store_text: Option<bool>,
    /// One caller-computed embedding per chunk, in the store's embedding space.
    #[uniffi(default = None)]
    pub embeddings: Option<Vec<Vec<f64>>>,
    /// Counts the terms of the first line this many times in the first chunk, as if the title
    /// were repeated. 1 or absent leaves the document unchanged.
    #[uniffi(default = None)]
    pub title_weight: Option<u32>,
}

impl DocumentInput {
    pub(crate) fn into_core(self) -> Result<hms::DocumentInput, HmsError> {
        Ok(hms::DocumentInput {
            id: self.id,
            text: self.text,
            source_uri: self.source_uri,
            version: self.version,
            metadata: parse_json("metadataJson", self.metadata_json)?,
            chunk_words: self.chunk_words,
            overlap_words: self.overlap_words,
            store_text: self.store_text,
            embeddings: self.embeddings,
            title_weight: self.title_weight,
        })
    }
}

/// A chunk of a document, with its byte offsets into the source text.
#[derive(Clone, Debug, uniffi::Record)]
pub struct DocumentChunk {
    pub id: String,
    pub text: String,
    pub start_byte: u32,
    pub end_byte: u32,
}

impl From<hms::DocumentChunk> for DocumentChunk {
    fn from(chunk: hms::DocumentChunk) -> Self {
        Self {
            id: chunk.id,
            text: chunk.text,
            start_byte: chunk.start_byte,
            end_byte: chunk.end_byte,
        }
    }
}

/// How the lexical and semantic result lists combine.
#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum Fusion {
    /// Adds each list's min-max normalized scores, weighted by the lexical and semantic weights.
    Blend,
    /// Reciprocal rank fusion with k = 60.
    Rrf,
}

/// Search options. Every field has a default.
#[derive(Clone, Debug, Default, uniffi::Record)]
pub struct SearchOptions {
    /// Number of chunks returned.
    #[uniffi(default = None)]
    pub k: Option<u32>,
    /// How many candidates each list contributes before fusion.
    #[uniffi(default = None)]
    pub candidate_limit: Option<u32>,
    /// A JSON object; documents whose metadata members equal its scalar members match.
    #[uniffi(default = None)]
    pub filter_json: Option<String>,
    #[uniffi(default = None)]
    pub source_uri: Option<String>,
    #[uniffi(default = None)]
    pub document_ids: Option<Vec<String>>,
    /// Query embedding in the store's embedding space.
    #[uniffi(default = None)]
    pub embedding: Option<Vec<f64>>,
    #[uniffi(default = None)]
    pub lexical_weight: Option<f64>,
    #[uniffi(default = None)]
    pub semantic_weight: Option<f64>,
    #[uniffi(default = None)]
    pub min_semantic_score: Option<f64>,
    #[uniffi(default = None)]
    pub fusion: Option<Fusion>,
    /// BM25 term-frequency saturation `k1` (default 1.2).
    #[uniffi(default = None)]
    pub bm25_k1: Option<f64>,
    /// BM25 length normalization `b` in [0, 1] (default 0.75).
    #[uniffi(default = None)]
    pub bm25_b: Option<f64>,
}

impl SearchOptions {
    pub(crate) fn into_core(self) -> Result<hms::SearchOptions, HmsError> {
        Ok(hms::SearchOptions {
            k: self.k,
            candidate_limit: self.candidate_limit,
            filter: parse_json("filterJson", self.filter_json)?,
            source_uri: self.source_uri,
            document_ids: self.document_ids,
            embedding: self.embedding,
            lexical_weight: self.lexical_weight,
            semantic_weight: self.semantic_weight,
            min_semantic_score: self.min_semantic_score,
            fusion: self.fusion.map(|f| match f {
                Fusion::Blend => "blend".to_owned(),
                Fusion::Rrf => "rrf".to_owned(),
            }),
            bm25_k1: self.bm25_k1,
            bm25_b: self.bm25_b,
        })
    }
}

/// One matching chunk.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SearchHit {
    pub id: String,
    pub document_id: String,
    pub source_uri: Option<String>,
    pub version: String,
    /// The document's metadata as a JSON object.
    pub metadata_json: String,
    /// The chunk text, when the document was stored with text.
    pub text: Option<String>,
    pub start_byte: u32,
    pub end_byte: u32,
    pub score: f64,
    pub lexical_score: f64,
    pub semantic_score: Option<f64>,
}

impl From<hms::DocumentResult> for SearchHit {
    fn from(result: hms::DocumentResult) -> Self {
        Self {
            id: result.id,
            document_id: result.document_id,
            source_uri: result.source_uri,
            version: result.version,
            metadata_json: result.metadata.to_string(),
            text: result.text,
            start_byte: result.start_byte,
            end_byte: result.end_byte,
            score: result.score,
            lexical_score: result.lexical_score,
            semantic_score: result.semantic_score,
        }
    }
}

/// A point-in-time description of an open store.
#[derive(Clone, Debug, uniffi::Record)]
pub struct StoreStats {
    pub document_count: u64,
    /// Chunk vectors held by the store.
    pub chunk_count: u64,
    pub dimensions: u32,
    /// Chunks that carry a caller-supplied embedding.
    pub embedded_chunk_count: u64,
    pub dense_index_trained: bool,
    pub encrypted: bool,
    /// Bytes of vector log in use and the bytes reserved for it.
    pub log_used_bytes: u64,
    pub log_capacity_bytes: u64,
    /// Sum of the lengths of every file in the store directory.
    pub disk_bytes: u64,
    /// Bytes the file system has allocated for them; smaller than `disk_bytes` where the
    /// preallocated log segment is sparse.
    pub disk_allocated_bytes: u64,
}

/// What a snapshot wrote.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SnapshotInfo {
    pub path: String,
    pub file_count: u32,
    pub total_bytes: u64,
    pub document_count: u64,
}

fn parse_json(name: &str, text: Option<String>) -> Result<Option<Value>, HmsError> {
    let Some(text) = text else {
        return Ok(None);
    };
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| HmsError::invalid(format!("{name} is not valid JSON: {e}")))?;
    if !value.is_object() {
        return Err(HmsError::invalid(format!("{name} must be a JSON object")));
    }
    Ok(Some(value))
}
