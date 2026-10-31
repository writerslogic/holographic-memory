// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use super::{encoding::TextEncoder, entangled::EntangledHVec};

pub(crate) mod binary;
#[cfg(test)]
mod compact_term_tests;
mod packed_terms;
use packed_terms::StoredTerms;

#[cfg(test)]
mod census;
mod dense;
mod image;
mod index;
mod lexical;
mod listing;
mod memory;
mod overlay;
mod positions;
mod postings;
mod search;
mod text;
#[cfg(not(target_arch = "wasm32"))]
mod vector_index;
pub(crate) use dense::{build_dense_index, capture_dense_index, publish_dense_index};
pub(crate) use image::{
    binding as checkpoint_binding, recorded_crc as checkpoint_crc,
    representable as checkpoint_representable, Backing, ServingImage,
    CRC_START as CHECKPOINT_CRC_START, HEADER_LEN as CHECKPOINT_HEADER_LEN,
    MAGIC as CHECKPOINT_MAGIC,
};
pub(crate) use index::DocumentWithText;
pub(crate) use index::{canonical_staging_path, opaque_counter_start, DocumentStore};
pub(crate) use lexical::execute as execute_lexical;
pub use lexical::{LexicalExecution, LexicalHit, LexicalOptions, LexicalQuery, LexicalResults};
pub use listing::DocumentRecord;
pub(crate) use listing::{count, get_record, list_records};
pub use memory::{MemoryDocumentStore, MemoryStoreStats};
pub(crate) use overlay::Base as CheckpointBase;
pub(crate) use search::{
    capture_search, count_matches, execute_search, prepare_search, rank_search, read_ranked,
    validate_search, CapturedSearch, PreparedSearch,
};
pub(crate) use text::TextSegment;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use vector_index::{
    build_vector_snapshot, capture_vector_rows, search_vector_snapshot, vector_index_status,
    VectorIndexSlot,
};
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use vector_index::{rank_vector_snapshot, read_vector_hits};
#[cfg(not(target_arch = "wasm32"))]
pub use vector_index::{
    GraphUnavailable, VectorIndex, VectorIndexKind, VectorIndexMode, VectorIndexStatus,
    VectorQueryOptions, DEFAULT_EF, DEFAULT_GRAPH_MIN_ROWS, MAX_EF, MAX_K, REBUILD_AFTER_MUTATIONS,
    SNAPSHOT_MIN_ROWS,
};

/// Documents, and chunks, one `memorize_documents_with` call accepts.
pub const MAX_BATCH_ITEMS: usize = 4096;

/// A document batch over one of the engine's batch limits (items, chunks, source or metadata
/// bytes, embedding values, transaction bytes). Nothing was written; a smaller batch can succeed.
#[derive(Debug, Clone, Copy)]
pub struct BatchTooLarge(pub &'static str);

impl std::fmt::Display for BatchTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for BatchTooLarge {}
pub(crate) const MAX_DOCUMENT_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_CHUNK_BYTES: usize = 64 * 1024;
pub(crate) const CHUNK_PREFIX: &str = "hms:chunk:";

#[cfg_attr(feature = "node-api", napi_derive::napi(object))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentInput {
    pub id: String,
    pub text: String,
    pub source_uri: Option<String>,
    pub version: Option<String>,
    pub metadata: Option<Value>,
    pub chunk_words: Option<u32>,
    pub overlap_words: Option<u32>,
    pub store_text: Option<bool>,
    pub embeddings: Option<Vec<Vec<f64>>>,
    /// Counts the terms of the text's first line this many times in the first chunk's term
    /// frequencies and word count, as if the title were repeated; 1 or absent leaves the
    /// document exactly as before this option existed.
    pub title_weight: Option<u32>,
}

/// When a document's chunk hypervectors are computed. Terms, postings and text are stored at
/// ingest in every mode, so lexical search sees the document at once; only the vector side
/// (`WarmCore::query` over chunk vectors) differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodeMode {
    /// Encode at ingest: the behavior before this option existed.
    #[default]
    Eager,
    /// Commit the document lexically searchable now; its vectors are encoded from the stored
    /// text on the first vector query, by `maintain_indices`, `train_document_index` or
    /// `encode_pending_documents`, and persisted then. Needs a text encoder and the chunk text
    /// (`storeText` not false).
    Deferred,
    /// Never encode: the document stays outside vector search until memorized again.
    Lexical,
}

impl EncodeMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Deferred => "deferred",
            Self::Lexical => "lexical",
        }
    }
}

impl std::str::FromStr for EncodeMode {
    type Err = anyhow::Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "eager" => Ok(Self::Eager),
            "deferred" => Ok(Self::Deferred),
            "lexical" => Ok(Self::Lexical),
            _ => bail!("encode mode must be \"eager\", \"deferred\" or \"lexical\", not {name:?}"),
        }
    }
}

/// Options of one `memorize_documents_with` call; the default is exactly `memorize_documents`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct IngestOptions {
    pub encode: EncodeMode,
    /// Index each word lowercased but not stemmed, so a document matches only the query words
    /// it contains as written. The default stems every word with the English Snowball stemmer.
    /// Query terms are matched in both forms, so one store may hold documents of either kind.
    pub unstemmed: bool,
    /// Retain token positions for exact, ordered phrase queries. Off by default.
    pub positions: bool,
    /// Index one whole article as one lexical chunk (up to 8 MiB). Requires lexical
    /// encoding and positions; chunkWords/overlapWords do not split the article.
    pub whole_document: bool,
}

/// Persisted state of a chunk's hypervector. Written only when not `Encoded`, so a store
/// without deferred or lexical documents serializes byte for byte as before the flag existed,
/// and a chunk without the field reads as `Encoded`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum VectorState {
    #[default]
    Encoded,
    /// Memorized with [`EncodeMode::Deferred`]; `vector` is empty and the chunk is not in the
    /// vector index until `WarmCore::encode_pending_documents` installs its encoding.
    Pending,
    /// Memorized with [`EncodeMode::Lexical`]; `vector` is empty for good.
    Lexical,
}

impl VectorState {
    pub(crate) fn is_encoded(&self) -> bool {
        *self == Self::Encoded
    }
}

#[cfg_attr(feature = "node-api", napi_derive::napi(object))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentChunk {
    pub id: String,
    pub text: String,
    pub start_byte: u32,
    pub end_byte: u32,
}

#[cfg_attr(feature = "node-api", napi_derive::napi(object))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SearchOptions {
    pub k: Option<u32>,
    pub candidate_limit: Option<u32>,
    pub filter: Option<Value>,
    pub source_uri: Option<String>,
    pub document_ids: Option<Vec<String>>,
    pub embedding: Option<Vec<f64>>,
    pub lexical_weight: Option<f64>,
    pub semantic_weight: Option<f64>,
    pub min_semantic_score: Option<f64>,
    /// How the lexical and semantic lists combine: `"blend"` (default) adds each list's
    /// min-max normalized scores over its top `candidateLimit`, weighted by `lexicalWeight`
    /// and `semanticWeight`; `"rrf"` is reciprocal rank fusion with k = 60.
    pub fusion: Option<String>,
    /// BM25 term-frequency saturation `k1`; the default scores exactly as before this option
    /// existed (1.2).
    pub bm25_k1: Option<f64>,
    /// BM25 length normalization `b` in `[0, 1]`; the default scores exactly as before (0.75).
    pub bm25_b: Option<f64>,
}

/// BM25 constants as the scorer multiplies them: `k1 + 1`, `k1`, `1 - b` and `b`. The default
/// keeps the literal constants the scorer always used, so default scores stay bit-identical.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bm25 {
    pub(crate) k1_plus_one: f64,
    pub(crate) k1: f64,
    pub(crate) one_minus_b: f64,
    pub(crate) b: f64,
}

impl Bm25 {
    pub(crate) const DEFAULT: Self = Self {
        k1_plus_one: 2.2,
        k1: 1.2,
        one_minus_b: 0.25,
        b: 0.75,
    };

    pub(crate) fn from_options(options: &SearchOptions) -> Result<Self> {
        let (Some(k1), Some(b)) = (
            options.bm25_k1.or(Some(Self::DEFAULT.k1)),
            options.bm25_b.or(Some(Self::DEFAULT.b)),
        ) else {
            unreachable!("defaults fill both parameters");
        };
        if options.bm25_k1.is_none() && options.bm25_b.is_none() {
            return Ok(Self::DEFAULT);
        }
        ensure!(
            k1.is_finite() && k1 >= 0.0 && (0.0..=1.0).contains(&b),
            "bm25K1 must be finite and nonnegative and bm25B in [0, 1]"
        );
        Ok(Self {
            k1_plus_one: k1 + 1.0,
            k1,
            one_minus_b: 1.0 - b,
            b,
        })
    }

    /// The length normalization of a chunk of `word_count` words against `average`.
    #[inline]
    pub(crate) fn norm(self, word_count: f64, average: f64) -> f64 {
        self.k1 * (self.one_minus_b + self.b * word_count / average)
    }

    /// The inverse document frequency of a term in `df` of `n` chunks.
    #[inline]
    pub(crate) fn idf(n: f64, df: f64) -> f64 {
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// The weight of a term with inverse document frequency `idf` seen `tf` times in a chunk
    /// whose length normalization is `norm`.
    #[inline]
    pub(crate) fn weight(self, idf: f64, tf: f64, norm: f64) -> f64 {
        if tf == 0.0 {
            0.0
        } else {
            idf * tf * self.k1_plus_one / (tf + norm)
        }
    }
}

#[cfg_attr(feature = "node-api", napi_derive::napi(object))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentResult {
    pub id: String,
    pub document_id: String,
    pub source_uri: Option<String>,
    pub version: String,
    pub metadata: Value,
    pub text: Option<String>,
    pub start_byte: u32,
    pub end_byte: u32,
    pub score: f64,
    pub lexical_score: f64,
    pub semantic_score: Option<f64>,
}

/// A chunk's dense embedding as stored: the l2-normalised values rounded to `f32`, in one
/// shared allocation that the store and every vector snapshot covering the chunk hold
/// together, so a snapshot adds no copy of the matrix (4 bytes per value in memory, against
/// the 8 of the `f64` form plus a 4-byte snapshot copy before this type).
///
/// Scoring: every path (the generic search, the dense index, the vector snapshot's exact
/// re-score) calls [`Embedding::cosine`], which widens each stored value to `f64` and
/// accumulates in `f64`, so one chunk has one semantic score however it was reached. The
/// score is the exact cosine of the stored (`f32`) vector; it differs from the cosine of the
/// client's `f64` vector by at most the rounding of the stored values, `2^-24` relative per
/// coordinate (about `6e-8` worst case over an l2-normalised pair, `2e-9` typical at 384
/// dimensions).
///
/// Serialized form: a JSON array of the `f32` values (shortest round-trip decimals, about
/// half the bytes of the `f64` form). A record written before this type held an array of
/// `f64` values under the same key; both read as an array of numbers rounded to the nearest
/// `f32`, so a store from either version opens, and a store written now still opens with
/// the previous release (which reads the array as `f64`). Chunks without an embedding
/// serialize byte for byte as before.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Embedding(Arc<[f32]>);

impl Embedding {
    /// Rounds l2-normalised `f64` values to the stored `f32` form.
    pub(crate) fn from_f64(values: &[f64]) -> Self {
        Self(values.iter().map(|&value| value as f32).collect())
    }

    pub(crate) fn from_f32(values: Vec<f32>) -> Self {
        Self(values.into())
    }

    /// The stored values widened to `f64` (exact).
    pub(crate) fn to_f64(&self) -> Vec<f64> {
        self.0.iter().map(|&value| f64::from(value)).collect()
    }

    /// The semantic score of this chunk against an l2-normalised `f64` query: the sequential
    /// `f64` dot product over the widened stored values, clamped to the cosine range.
    #[inline]
    pub(crate) fn cosine(&self, query: &[f64]) -> f64 {
        query
            .iter()
            .zip(self.0.iter())
            .map(|(q, &v)| q * f64::from(v))
            .sum::<f64>()
            .clamp(-1.0, 1.0)
    }

    /// Whether both handles share one allocation (a snapshot row and the store's chunk).
    #[cfg(test)]
    pub(crate) fn shares_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Deref for Embedding {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.0
    }
}

impl Serialize for Embedding {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter())
    }
}

impl<'de> Deserialize<'de> for Embedding {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<f32>::deserialize(deserializer).map(Self::from_f32)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct StoredChunk {
    pub id: String,
    /// Inline text, or none when the store moved it to its text segment (`text_at`) or the
    /// document was stored without text.
    pub text: Option<String>,
    /// Location of the text in the store's text segment, for the process that installed it.
    #[serde(skip)]
    pub text_at: Option<(u64, u32)>,
    pub start_byte: u32,
    pub end_byte: u32,
    pub vector: EntangledHVec,
    pub embedding: Option<Embedding>,
    pub terms: StoredTerms,
    pub word_count: usize,
    /// Whether `terms` were stemmed at ingest; chunks stored before stemming existed match
    /// the unstemmed query terms.
    #[serde(default)]
    pub stemmed: bool,
    /// Whether `vector` is the chunk's encoding, awaits deferred encoding, or is deliberately
    /// absent; `Pending` and `Lexical` chunks carry an empty vector and are not in the vector
    /// index. Absent from the serialized form when `Encoded`.
    #[serde(default, skip_serializing_if = "VectorState::is_encoded")]
    pub vector_state: VectorState,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct StoredDocument {
    #[serde(deserialize_with = "deserialize_compact_string")]
    pub id: String,
    #[serde(default, deserialize_with = "deserialize_compact_uri")]
    pub source_uri: Option<String>,
    #[serde(deserialize_with = "deserialize_compact_string")]
    pub version: String,
    #[serde(deserialize_with = "deserialize_compact_metadata")]
    pub metadata: Value,
    #[serde(deserialize_with = "deserialize_compact_chunks")]
    pub chunks: Vec<StoredChunk>,
}

fn deserialize_compact_chunks<'de, D>(deserializer: D) -> Result<Vec<StoredChunk>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut chunks = Vec::<StoredChunk>::deserialize(deserializer)?;
    compact_chunks(&mut chunks);
    Ok(chunks)
}

fn deserialize_compact_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = String::deserialize(deserializer)?;
    compact_string(&mut value);
    Ok(value)
}

fn deserialize_compact_uri<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = Option::<String>::deserialize(deserializer)?;
    if let Some(value) = &mut value {
        compact_string(value);
    }
    Ok(value)
}

fn deserialize_compact_metadata<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = Value::deserialize(deserializer)?;
    compact_metadata(&mut value);
    Ok(value)
}

fn compact_vec<T>(values: &mut Vec<T>) {
    if values.capacity() != values.len() {
        *values = std::mem::take(values).into_boxed_slice().into_vec();
    }
}

fn compact_string(value: &mut String) {
    if value.capacity() != value.len() {
        *value = std::mem::take(value).into_boxed_str().into_string();
    }
}

fn compact_metadata(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values.iter_mut() {
                compact_metadata(value);
            }
            compact_vec(values);
        }
        Value::Object(values) => {
            if values.keys().any(|key| key.capacity() != key.len()) {
                *values = std::mem::take(values)
                    .into_iter()
                    .map(|(mut key, value)| {
                        compact_string(&mut key);
                        (key, value)
                    })
                    .collect();
            }
            for value in values.values_mut() {
                compact_metadata(value);
            }
        }
        Value::String(value) => compact_string(value),
        _ => {}
    }
}

fn compact_chunks(chunks: &mut Vec<StoredChunk>) {
    for chunk in chunks.iter_mut() {
        compact_string(&mut chunk.id);
        if let Some(text) = &mut chunk.text {
            compact_string(text);
        }
        compact_vec(&mut chunk.vector.indices);
    }
    compact_vec(chunks);
}

pub fn chunk_document(input: &DocumentInput) -> Result<Vec<DocumentChunk>> {
    Ok(chunk_spans(input)?
        .into_iter()
        .enumerate()
        .map(|(index, (start_byte, end_byte))| DocumentChunk {
            id: chunk_id(&input.id, index),
            text: input.text[start_byte as usize..end_byte as usize].into(),
            start_byte,
            end_byte,
        })
        .collect())
}

fn chunk_id(document: &str, index: usize) -> String {
    let mut length = [0u8; 20];
    let mut ordinal = [0u8; 20];
    let (length, ordinal) = (
        decimal(document.len(), &mut length),
        decimal(index, &mut ordinal),
    );
    // Sized exactly: the id is stored as is, and a grown buffer would be reallocated to fit.
    let mut id = String::with_capacity(
        CHUNK_PREFIX.len() + length.len() + document.len() + ordinal.len() + 2,
    );
    for part in [CHUNK_PREFIX, length, ":", document, ":", ordinal] {
        id.push_str(part);
    }
    id
}

/// Whether `id` is [`chunk_id`]`(document, index)`, without building it.
fn is_chunk_id(id: &str, document: &str, index: usize) -> bool {
    let mut length = [0u8; 20];
    let mut ordinal = [0u8; 20];
    let (length, ordinal) = (
        decimal(document.len(), &mut length),
        decimal(index, &mut ordinal),
    );
    let mut rest = id;
    for part in [CHUNK_PREFIX, length, ":", document, ":", ordinal] {
        match rest.strip_prefix(part) {
            Some(after) => rest = after,
            None => return false,
        }
    }
    rest.is_empty()
}

/// `value` in decimal, written to the end of `buffer`.
fn decimal(mut value: usize, buffer: &mut [u8; 20]) -> &str {
    let mut at = buffer.len();
    loop {
        at -= 1;
        buffer[at] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    std::str::from_utf8(&buffer[at..]).expect("ASCII digits")
}

/// A JSON sink that counts bytes and fails once they pass `limit`, so a size check allocates
/// nothing.
struct SizeLimit {
    bytes: usize,
    limit: usize,
}

impl std::io::Write for SizeLimit {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .bytes
            .checked_add(bytes.len())
            .filter(|&n| n <= self.limit);
        match next {
            Some(next) => {
                self.bytes = next;
                Ok(bytes.len())
            }
            None => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "document batch size limit exceeded",
            )),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The length of `value` as JSON, or an error once it passes `limit` bytes.
pub(crate) fn serialized_size(value: &impl serde::Serialize, limit: usize) -> Result<usize> {
    let mut sink = SizeLimit { bytes: 0, limit };
    serde_json::to_writer(&mut sink, value)?;
    Ok(sink.bytes)
}

/// The byte span of every chunk [`chunk_document`] cuts `input` into, with the same checks and
/// errors, without copying any text.
pub(crate) fn chunk_spans_with(
    input: &DocumentInput,
    options: &IngestOptions,
) -> Result<Vec<(u32, u32)>> {
    if !options.whole_document {
        return chunk_spans(input);
    }
    ensure!(
        options.encode == EncodeMode::Lexical && options.positions,
        "whole-document ingest requires lexical encoding and positions"
    );
    validate_document_source(input)?;
    ensure!(
        input.text.chars().any(|c| !c.is_whitespace()),
        "document must contain text"
    );
    Ok(vec![(0, input.text.len() as u32)])
}

fn validate_document_source(input: &DocumentInput) -> Result<()> {
    ensure!(
        !input.id.is_empty() && input.id.len() <= 4096 && !input.id.contains('\0'),
        "document id must contain 1..=4096 bytes"
    );
    ensure!(
        input.text.len() <= MAX_DOCUMENT_BYTES,
        "document exceeds 8 MiB"
    );
    Ok(())
}

pub(crate) fn chunk_spans(input: &DocumentInput) -> Result<Vec<(u32, u32)>> {
    validate_document_source(input)?;
    let size = input.chunk_words.unwrap_or(256) as usize;
    let overlap = input
        .overlap_words
        .unwrap_or(32.min(size.saturating_sub(1)) as u32) as usize;
    ensure!(
        (1..=4096).contains(&size) && overlap < size,
        "chunkWords must be 1..=4096 and overlapWords smaller than chunkWords"
    );
    let mut words = Vec::with_capacity(input.text.len() / 6);
    let mut start = None;
    let mut boundary = |i: usize, whitespace: bool| {
        if whitespace {
            if let Some(s) = start.take() {
                words.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    };
    if input.text.is_ascii() {
        // `char::is_whitespace` on ASCII is exactly 0x09..=0x0D and the space.
        for (i, b) in input.text.bytes().enumerate() {
            boundary(i, matches!(b, b' ' | b'\t'..=b'\r'));
        }
    } else {
        for (i, c) in input.text.char_indices() {
            boundary(i, c.is_whitespace());
        }
    }
    if let Some(s) = start {
        words.push((s, input.text.len()));
    }
    ensure!(!words.is_empty(), "document must contain text");
    let mut chunks = Vec::new();
    let mut word = 0;
    while word < words.len() {
        ensure!(
            chunks.len() < 4096,
            "document exceeds 4096 chunks; increase chunkWords"
        );
        let end_word = (word + size).min(words.len());
        let (start_byte, end_byte) = (words[word].0, words[end_word - 1].1);
        ensure!(
            end_byte - start_byte <= MAX_CHUNK_BYTES,
            "chunk exceeds 64 KiB; reduce chunkWords or split long tokens"
        );
        chunks.push((start_byte as u32, end_byte as u32));
        if end_word == words.len() {
            break;
        }
        word += size - overlap;
    }
    Ok(chunks)
}

/// Validates `values` (the store's dimensions, finite, not all zero) and returns them
/// l2-normalised in `f64`. Accepts the client's `f64` values and a stored [`Embedding`]'s
/// `f32` values, which widen exactly.
pub(crate) fn normalize_embedding<T: Copy + Into<f64>>(
    values: &[T],
    dimensions: usize,
) -> Result<Vec<f64>> {
    let values: Vec<f64> = values.iter().map(|&x| x.into()).collect();
    ensure!(
        values.len() == dimensions && values.iter().all(|x| x.is_finite()),
        "embedding dimensions must match the store and values must be finite"
    );
    let scale = values.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    ensure!(scale > 0.0, "zero embeddings are not supported");
    let norm = values
        .iter()
        .map(|x| (x / scale).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(values.iter().map(|x| (x / scale) / norm).collect())
}

fn stemmer() -> &'static rust_stemmers::Stemmer {
    static STEMMER: OnceLock<rust_stemmers::Stemmer> = OnceLock::new();
    STEMMER.get_or_init(|| rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English))
}

/// Lowercased alphanumeric tokens and their counts; with `stem`, each token is reduced by
/// the English Snowball (Porter 2) stemmer, which is what the published BM25 baselines do.
/// The distinct words of a query in one form.
pub(crate) fn query_words(text: &str, stem: bool) -> std::collections::BTreeSet<String> {
    terms(text, stem).into_keys().collect()
}

pub(crate) fn terms(text: &str, stem: bool) -> BTreeMap<String, u32> {
    // Counted in a hash map keyed by borrowed text (one allocation per distinct term, not per
    // occurrence), then sorted once into the ordered map the postings expect.
    // Seeded per process: the keys are caller text.
    let mut counts: std::collections::HashMap<String, u32, foldhash::fast::RandomState> =
        std::collections::HashMap::with_capacity_and_hasher(text.len() / 16, Default::default());
    let mut lower = String::new();
    let mut add = |key: &str| match counts.get_mut(key) {
        Some(count) => *count += 1,
        None => {
            counts.insert(key.to_owned(), 1);
        }
    };
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
    {
        lower.clear();
        if word.is_ascii() {
            lower.push_str(word);
            lower.make_ascii_lowercase();
        } else {
            // `str::to_lowercase`, not per-char mapping: it applies the final-sigma rule.
            lower.push_str(&word.to_lowercase());
        }
        if stem {
            add(&stemmer().stem(&lower));
        } else {
            add(&lower);
        }
    }
    // Keys are unique, so an unstable sort is exact; the map's own stable sort then sees one
    // ascending run and finishes in a single pass.
    let mut sorted: Vec<(String, u32)> = counts.into_iter().collect();
    sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    sorted.into_iter().collect()
}

/// Words a batch interner maps before it starts over; bounds memory on corpora with
/// unbounded vocabularies while keeping the common-word hit rate.
const STEM_MEMO_CAPACITY: usize = 1 << 18;

/// Hash map keyed by caller text, so seeded per process.
type SeededMap<K, V> = std::collections::HashMap<K, V, foldhash::fast::RandomState>;

/// The lexical state of ingest on one thread, carried across a store's batches
/// ([`StemMemoPool`]): every distinct word seen maps to the id of its term (lowercased,
/// Snowball-stemmed), so a repeated word costs one hash lookup and no copy, stemming runs
/// once per distinct word, and a chunk's counts accumulate in an array by term id.
#[derive(Default)]
struct Interner {
    /// Word exactly as it appears in the text -> term id.
    words: WordMap,
    /// Term -> id, so the words of one stem count as one term.
    ids: SeededMap<Box<str>, u32>,
    /// Term by id, and its first eight bytes big-endian (zero-padded): a term never contains
    /// a NUL byte, so comparing keys orders terms as their bytes do except on equal keys.
    terms: Vec<Box<str>>,
    keys: Vec<u64>,
    /// Count per term id for the chunk being counted, and the ids counted so far. Zero and
    /// empty between chunks.
    counts: Vec<u32>,
    counted: Vec<u32>,
    /// The term id of each word of the chunk, in text order, while `record` is set.
    sequence: Vec<u32>,
    record: bool,
    lower: String,
    /// Terms are the lowercased words themselves rather than their stems.
    unstemmed: bool,
}

/// Word -> term id. A word of at most 16 bytes (nearly every word) is keyed by its bytes
/// zero-padded into a `u128`, unambiguous because a word holds no NUL byte, so a lookup
/// compares integers rather than following a pointer to compare strings.
#[derive(Default)]
struct WordMap {
    short: SeededMap<u128, u32>,
    long: SeededMap<Box<str>, u32>,
}

impl WordMap {
    #[inline]
    fn short_key(word: &str) -> Option<u128> {
        debug_assert!(!word.contains('\0'));
        let bytes = word.as_bytes();
        (bytes.len() <= 16).then(|| {
            let mut key = [0u8; 16];
            key[..bytes.len()].copy_from_slice(bytes);
            u128::from_le_bytes(key)
        })
    }

    #[inline]
    fn get(&self, word: &str) -> Option<u32> {
        match Self::short_key(word) {
            Some(key) => self.short.get(&key).copied(),
            None => self.long.get(word).copied(),
        }
    }

    fn insert(&mut self, word: &str, id: u32) {
        match Self::short_key(word) {
            Some(key) => self.short.insert(key, id),
            None => self.long.insert(word.into(), id),
        };
    }

    fn len(&self) -> usize {
        self.short.len() + self.long.len()
    }
}

impl Interner {
    fn term_id(&mut self, word: &str, ascii: bool) -> u32 {
        if let Some(id) = self.words.get(word) {
            return id;
        }
        self.lower.clear();
        if ascii {
            self.lower.push_str(word);
            self.lower.make_ascii_lowercase();
        } else {
            // `str::to_lowercase`, not per-char mapping: it applies the final-sigma rule.
            self.lower.push_str(&word.to_lowercase());
        }
        let stem = if self.unstemmed {
            std::borrow::Cow::Borrowed(self.lower.as_str())
        } else {
            stemmer().stem(&self.lower)
        };
        let id = match self.ids.get(stem.as_ref()) {
            Some(&id) => id,
            None => {
                let id = u32::try_from(self.terms.len()).expect("batch vocabulary over u32");
                let term: Box<str> = stem.into();
                let mut key = [0u8; 8];
                let prefix = term.len().min(8);
                key[..prefix].copy_from_slice(&term.as_bytes()[..prefix]);
                self.keys.push(u64::from_be_bytes(key));
                self.ids.insert(term.clone(), id);
                self.terms.push(term);
                self.counts.push(0);
                id
            }
        };
        self.words.insert(word, id);
        id
    }

    /// Counts the words of `text` `weight` times each, as `terms(text, true)` splits them
    /// (maximal runs of alphanumeric characters); returns how many were counted, weighted.
    fn count(&mut self, text: &str, weight: u32) -> usize {
        let bytes = text.as_bytes();
        let mut counted = 0;
        let mut at = 0;
        let char_at = |at: usize| text[at..].chars().next().expect("char boundary");
        while at < bytes.len() {
            if bytes[at] < 0x80 {
                if !bytes[at].is_ascii_alphanumeric() {
                    at += 1;
                    continue;
                }
            } else {
                let c = char_at(at);
                if !c.is_alphanumeric() {
                    at += c.len_utf8();
                    continue;
                }
            }
            let start = at;
            let mut ascii = true;
            while at < bytes.len() {
                if bytes[at] < 0x80 {
                    if !bytes[at].is_ascii_alphanumeric() {
                        break;
                    }
                    at += 1;
                } else {
                    let c = char_at(at);
                    if !c.is_alphanumeric() {
                        break;
                    }
                    ascii = false;
                    at += c.len_utf8();
                }
            }
            let id = self.term_id(&text[start..at], ascii);
            if self.record {
                self.sequence.push(id);
            }
            let count = &mut self.counts[id as usize];
            if *count == 0 {
                self.counted.push(id);
            }
            *count += weight;
            counted += weight as usize;
        }
        counted
    }

    /// The term table of `text` with its first line's terms counted `title_weight` times in
    /// total, and the weighted word count: what `terms(text, true)` plus the title weighting
    /// produce.
    /// With `positions` (`Some(whole_document)`), the table also carries each term's word
    /// positions, the bytes the reference analyzer (`Positions::from_text`) writes.
    fn chunk_terms(
        &mut self,
        text: &str,
        title_weight: u32,
        stem: bool,
        positions: Option<bool>,
    ) -> (StoredTerms, usize) {
        if self.words.len() >= STEM_MEMO_CAPACITY || self.unstemmed == stem {
            *self = Self {
                unstemmed: !stem,
                ..Self::default()
            };
        }
        self.sequence.clear();
        self.record = positions.is_some();
        let mut words = self.count(text, 1);
        self.record = false;
        if title_weight > 1 {
            words += self.count(text.lines().next().unwrap_or_default(), title_weight - 1);
        }
        let (terms, keys, counts) = (&self.terms, &self.keys, &mut self.counts);
        self.counted.sort_unstable_by(|&a, &b| {
            let (a, b) = (a as usize, b as usize);
            keys[a]
                .cmp(&keys[b])
                .then_with(|| terms[a].as_bytes().cmp(terms[b].as_bytes()))
        });
        let bytes = self
            .counted
            .iter()
            .map(|&id| terms[id as usize].len())
            .sum();
        let mut table = StoredTerms::from_sorted_refs(
            self.counted.len(),
            bytes,
            self.counted
                .iter()
                .map(|&id| (&*terms[id as usize], counts[id as usize])),
        );
        if let Some(whole_document) = positions {
            // Each word's term as its rank among the chunk's sorted terms, through `counts`
            // (cleared below).
            for (rank, &id) in self.counted.iter().enumerate() {
                counts[id as usize] = rank as u32;
            }
            for id in &mut self.sequence {
                *id = counts[*id as usize];
            }
            table.positions = Some(Arc::new(positions::Positions::from_ranks(
                &self.sequence,
                self.counted.len(),
                whole_document,
            )));
            self.sequence.clear();
        }
        for &id in &self.counted {
            counts[id as usize] = 0;
        }
        self.counted.clear();
        (table, words)
    }
}

thread_local! {
    static ACTIVE_STEM_MEMO: std::cell::RefCell<Option<Interner>> =
        const { std::cell::RefCell::new(None) };
}

/// The word interner of an ingest batch. It is installed on a thread only while that thread
/// prepares documents for the batch and is kept by the store between batches
/// ([`StemMemoPool`]), so no thread keeps it and queries never use it.
#[derive(Default)]
pub(crate) struct StemMemo(Interner);

pub(crate) struct StemMemoGuard<'a>(&'a mut StemMemo);

impl StemMemo {
    pub(crate) fn install(&mut self) -> StemMemoGuard<'_> {
        let table = std::mem::take(&mut self.0);
        ACTIVE_STEM_MEMO.with(|active| *active.borrow_mut() = Some(table));
        StemMemoGuard(self)
    }
}

impl Drop for StemMemoGuard<'_> {
    fn drop(&mut self) {
        if let Some(table) = ACTIVE_STEM_MEMO.with(|active| active.borrow_mut().take()) {
            self.0 .0 = table;
        }
    }
}

/// The batch interners a store keeps between its document batches, so a corpus loaded in
/// consecutive batches stems each distinct word once rather than once per batch and rayon
/// split. While a batch runs, each split takes an idle interner (the one that has seen the
/// most words) and returns it when done, so a worker's later splits reuse what its earlier
/// ones stemmed. When the batch ends ([`StemMemoPool::trim`]) only the largest is kept,
/// bounded by [`STEM_MEMO_CAPACITY`] words. With keeping turned off
/// ([`StemMemoPool::set_keep`]) each split's interner is freed with the split.
#[derive(Default)]
pub(crate) struct StemMemoPool {
    idle: crate::core::sync::Mutex<Vec<StemMemo>>,
    release: std::sync::atomic::AtomicBool,
}

impl StemMemoPool {
    /// The idle interner that has seen the most words, or a new one; it goes back to the pool
    /// when the result drops.
    pub(crate) fn take(&self) -> PooledStemMemo<'_> {
        let memo = {
            let mut idle = self.idle.lock();
            let largest = (0..idle.len()).max_by_key(|&i| idle[i].0.words.len());
            largest.map(|i| idle.swap_remove(i))
        };
        PooledStemMemo {
            pool: self,
            memo: memo.unwrap_or_default(),
        }
    }

    /// Keeps the idle interners that have seen the most words, together at most
    /// [`STEM_MEMO_CAPACITY`] words (always the largest one); the others are freed.
    pub(crate) fn trim(&self) {
        let mut idle = std::mem::take(&mut *self.idle.lock());
        idle.sort_unstable_by_key(|memo| std::cmp::Reverse(memo.0.words.len()));
        let mut words = 0;
        let keep = idle
            .iter()
            .position(|memo| {
                words += memo.0.words.len();
                words > STEM_MEMO_CAPACITY
            })
            .unwrap_or(idle.len())
            .max(1)
            .min(idle.len());
        let freed = idle.split_off(keep);
        self.idle.lock().append(&mut idle);
        // The freed interners drop here, with the lock released.
        drop(freed);
    }

    /// Whether interners outlive their batch; turning it off frees the idle ones now.
    pub(crate) fn set_keep(&self, keep: bool) {
        self.release
            .store(!keep, std::sync::atomic::Ordering::Relaxed);
        if !keep {
            let freed = std::mem::take(&mut *self.idle.lock());
            drop(freed);
        }
    }
}

pub(crate) struct PooledStemMemo<'a> {
    pool: &'a StemMemoPool,
    memo: StemMemo,
}

impl PooledStemMemo<'_> {
    pub(crate) fn install(&mut self) -> StemMemoGuard<'_> {
        self.memo.install()
    }
}

impl Drop for PooledStemMemo<'_> {
    fn drop(&mut self) {
        let memo = std::mem::take(&mut self.memo);
        if self.pool.release.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        self.pool.idle.lock().push(memo);
    }
}

/// [`Interner::chunk_terms`] through the batch interner installed on this thread, or a
/// one-off one.
#[cfg(test)]
fn chunk_terms(text: &str, title_weight: u32, stem: bool) -> (StoredTerms, usize) {
    analyze_chunk(text, title_weight, stem, None)
}

/// A chunk's term table through the batch interner installed on this thread, or a one-off
/// one, with each term's word positions when `positions` is `Some(whole_document)`.
fn analyze_chunk(
    text: &str,
    title_weight: u32,
    stem: bool,
    positions: Option<bool>,
) -> (StoredTerms, usize) {
    ACTIVE_STEM_MEMO.with(|active| match active.borrow_mut().as_mut() {
        Some(interner) => interner.chunk_terms(text, title_weight, stem, positions),
        None => Interner::default().chunk_terms(text, title_weight, stem, positions),
    })
}

pub(crate) fn prepare(
    input: DocumentInput,
    dimensions: usize,
    embedding_dimensions: Option<usize>,
    encoder: TextEncoder,
) -> Result<StoredDocument> {
    prepare_with_options(
        input,
        dimensions,
        embedding_dimensions,
        None,
        encoder,
        &IngestOptions::default(),
    )
}

/// [`prepare`] with an optional lexical key per chunk (for example the chunk prefixed with
/// facts extracted from it; stored text and byte offsets stay the chunk's own) and the chunk vectors encoded now, later or never (`mode`). The
/// lexical side of the document is the same in every mode.
pub(crate) fn prepare_with_options(
    input: DocumentInput,
    dimensions: usize,
    embedding_dimensions: Option<usize>,
    keys: Option<&[String]>,
    encoder: TextEncoder,
    options: &IngestOptions,
) -> Result<StoredDocument> {
    check_mode(&input, encoder, options.encode)?;
    let spans = chunk_spans_with(&input, options)?;
    prepare_spans(
        input,
        spans,
        dimensions,
        embedding_dimensions,
        keys,
        encoder,
        options,
    )
}

fn check_mode(input: &DocumentInput, encoder: TextEncoder, mode: EncodeMode) -> Result<()> {
    ensure!(
        mode != EncodeMode::Deferred || !encoder.is_embedding(),
        "deferred encoding applies to text encoders; {} vectors come from embeddings",
        encoder.name()
    );
    ensure!(
        mode != EncodeMode::Deferred || input.store_text.unwrap_or(true),
        "deferred encoding needs the chunk text; set storeText or encode eagerly"
    );
    Ok(())
}

/// [`prepare_with_options`] of an input whose chunk spans ([`chunk_spans`]) are known.
pub(crate) fn prepare_spans(
    input: DocumentInput,
    chunks: Vec<(u32, u32)>,
    dimensions: usize,
    embedding_dimensions: Option<usize>,
    keys: Option<&[String]>,
    encoder: TextEncoder,
    options: &IngestOptions,
) -> Result<StoredDocument> {
    let mode = options.encode;
    check_mode(&input, encoder, mode)?;
    ensure!(
        keys.is_none_or(|k| k.len() == chunks.len()),
        "provide one key per chunk"
    );
    ensure!(
        input.metadata.as_ref().is_none_or(Value::is_object),
        "metadata must be a JSON object"
    );
    ensure!(
        serialized_size(&input.metadata, 64 * 1024).is_ok(),
        "metadata exceeds 64 KiB"
    );
    ensure!(
        input.source_uri.as_ref().is_none_or(|s| s.len() <= 8192),
        "sourceUri exceeds 8192 bytes"
    );
    ensure!(
        input
            .version
            .as_ref()
            .is_none_or(|s| !s.is_empty() && s.len() <= 1024),
        "version requires 1..=1024 bytes"
    );
    if let Some(embeddings) = &input.embeddings {
        ensure!(
            embedding_dimensions.is_some() && embeddings.len() == chunks.len(),
            "provide one embedding per chunk and configure the store embedding space"
        );
        ensure!(
            embeddings.iter().map(Vec::len).sum::<usize>() <= 1_048_576,
            "document exceeds 1048576 embedding values; split the document"
        );
    }
    let mut stored = Vec::with_capacity(chunks.len());
    let title_weight = input.title_weight.unwrap_or(1);
    ensure!(
        (1..=16).contains(&title_weight),
        "titleWeight must be between 1 and 16"
    );
    for (i, &(start_byte, end_byte)) in chunks.iter().enumerate() {
        let text = &input.text[start_byte as usize..end_byte as usize];
        let source = keys.map_or(text, |k| k[i].as_str());
        ensure!(
            !options.positions || keys.is_none(),
            "positions require the original chunk text as the lexical key"
        );
        let (terms, word_count) = analyze_chunk(
            source,
            if i == 0 { title_weight } else { 1 },
            !options.unstemmed,
            options.positions.then_some(options.whole_document),
        );
        let embedding = input
            .embeddings
            .as_ref()
            .map(|all| {
                normalize_embedding(&all[i], embedding_dimensions.unwrap_or(0))
                    .map(|values| Embedding::from_f64(&values))
            })
            .transpose()?;
        let (vector, vector_state) = match (mode, encoder.is_embedding(), &embedding) {
            (EncodeMode::Lexical, _, _) => (
                EntangledHVec::from_indices(Vec::new(), dimensions),
                VectorState::Lexical,
            ),
            (EncodeMode::Deferred, _, _) => (
                EntangledHVec::from_indices(Vec::new(), dimensions),
                VectorState::Pending,
            ),
            (EncodeMode::Eager, false, _) => {
                (encoder.encode(text, dimensions), VectorState::Encoded)
            }
            (EncodeMode::Eager, true, Some(e)) => (
                EntangledHVec::from_dense(e, dimensions),
                VectorState::Encoded,
            ),
            (EncodeMode::Eager, true, None) => bail!(
                "the {} text encoder needs a dense embedding for every chunk; install a text embedder",
                encoder.name()
            ),
        };
        stored.push(StoredChunk {
            vector,
            vector_state,
            text: input.store_text.unwrap_or(true).then(|| text.to_owned()),
            text_at: None,
            id: chunk_id(&input.id, i),
            start_byte,
            end_byte,
            embedding,
            terms,
            word_count,
            stemmed: !options.unstemmed,
        });
    }
    let mut document = StoredDocument {
        id: input.id,
        source_uri: input.source_uri,
        version: input.version.unwrap_or_else(|| "1".into()),
        metadata: input.metadata.unwrap_or_else(|| serde_json::json!({})),
        chunks: stored,
    };
    compact_string(&mut document.id);
    compact_string(&mut document.version);
    if let Some(uri) = &mut document.source_uri {
        compact_string(uri);
    }
    compact_metadata(&mut document.metadata);
    compact_chunks(&mut document.chunks);
    Ok(document)
}

/// Identity of the content deferred encoding reads: the version, byte spans and text of every
/// chunk, mixed with the crate's splitmix64 chain. `None` when a chunk's text is unavailable.
/// `Mutation::EncodeDocument` carries it so vectors computed for one revision of a document
/// are never installed on another that reused its id, version and chunk count.
pub(crate) fn document_fingerprint(
    store: &DocumentStore,
    document: &StoredDocument,
) -> Option<u64> {
    let mut h = super::encoding::hash_str_seed(&document.version, 0x5EED_D0C5_F1A6_0001);
    for chunk in &document.chunks {
        let text = store.chunk_text(chunk)?;
        h = super::entangled::hash_u64(h, u64::from(chunk.start_byte));
        h = super::entangled::hash_u64(h, u64::from(chunk.end_byte));
        h = super::encoding::hash_str_seed(&text, h);
    }
    Some(h)
}

/// Whether every query in `prepared` ranks from the checkpoint under `store` and the store's
/// own documents (see `search::overlay_serves`).
pub(crate) fn overlay_serves(store: &DocumentStore, prepared: &[PreparedSearch]) -> bool {
    prepared
        .iter()
        .all(|query| search::overlay_serves(query, store))
}

/// [`capture_search`] over the checkpoint under `store` and the store's own documents, for
/// queries [`overlay_serves`] accepts.
pub(crate) fn capture_overlaid_search(
    store: &DocumentStore,
    prepared: Vec<PreparedSearch>,
) -> Result<CapturedSearch> {
    search::capture_overlaid_search(store, prepared)
}

/// The checkpoint of `store` for the generation whose name has CRC32 `generation`; `None` when
/// a chunk is not lexical-only or the store would not fit the layout.
pub(crate) fn encode_checkpoint(
    store: &DocumentStore,
    generation: u32,
    dimensions: usize,
) -> Result<Option<Vec<u8>>> {
    image::encode(store, generation, dimensions)
}
#[cfg(test)]
pub(crate) fn search(
    store: &DocumentStore,
    text: &str,
    options: &SearchOptions,
    embedding_dimensions: Option<usize>,
) -> Result<Vec<DocumentResult>> {
    let prepared = prepare_search(text, options, embedding_dimensions)?;
    let captured = capture_search(store, vec![prepared]);
    Ok(execute_search(&captured, 0))
}

#[cfg(test)]
fn eligible_chunks<'a>(
    documents: impl Iterator<Item = &'a StoredDocument>,
    options: &SearchOptions,
) -> Vec<(&'a StoredDocument, &'a StoredChunk)> {
    documents
        .filter(|document| matches_document(document, options))
        .flat_map(|d| d.chunks.iter().map(move |chunk| (d, chunk)))
        .collect()
}

#[cfg(test)]
fn matches_document(document: &StoredDocument, options: &SearchOptions) -> bool {
    options
        .document_ids
        .as_ref()
        .is_none_or(|ids| ids.contains(&document.id))
        && matches_document_metadata(document, options)
}

fn matches_document_metadata(document: &StoredDocument, options: &SearchOptions) -> bool {
    options
        .source_uri
        .as_ref()
        .is_none_or(|uri| document.source_uri.as_ref() == Some(uri))
        && options
            .filter
            .as_ref()
            .and_then(Value::as_object)
            .is_none_or(|filter| {
                filter
                    .iter()
                    .all(|(key, value)| document.metadata.get(key) == Some(value))
            })
}

fn top_ranked<T>(
    values: &mut Vec<T>,
    limit: usize,
    mut compare: impl FnMut(&T, &T) -> std::cmp::Ordering,
) {
    if values.len() > limit {
        values.select_nth_unstable_by(limit - 1, &mut compare);
        values.truncate(limit);
    }
    values.sort_unstable_by(compare);
}

#[cfg(test)]
fn search_impl(
    store: &DocumentStore,
    text: &str,
    options: &SearchOptions,
    embedding_dimensions: Option<usize>,
    indexed: bool,
) -> Result<Vec<DocumentResult>> {
    if indexed {
        return search(store, text, options, embedding_dimensions);
    }
    ensure!(text.len() <= MAX_DOCUMENT_BYTES, "query exceeds 8 MiB");
    let k = options.k.unwrap_or(10) as usize;
    let limit = options.candidate_limit.unwrap_or(100.max(k as u32)) as usize;
    ensure!(
        (1..=1000).contains(&k) && (k..=4096).contains(&limit),
        "require 1 <= k <= 1000 and k <= candidateLimit <= 4096"
    );
    ensure!(
        options.filter.as_ref().is_none_or(Value::is_object),
        "filter must be a JSON object of exact metadata matches"
    );
    let lexical_weight = options.lexical_weight.unwrap_or(1.0);
    let semantic_weight = options.semantic_weight.unwrap_or(1.0);
    ensure!(
        [lexical_weight, semantic_weight]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
            && lexical_weight + semantic_weight > 0.0,
        "search weights must be finite, nonnegative and not both zero"
    );
    let bm25 = Bm25::from_options(options)?;
    let minimum = options.min_semantic_score.unwrap_or(0.0);
    ensure!(
        minimum.is_finite() && (-1.0..=1.0).contains(&minimum),
        "minSemanticScore must be in [-1,1]"
    );
    let rrf = match options.fusion.as_deref() {
        None | Some("blend") => false,
        Some("rrf") => true,
        Some(other) => anyhow::bail!("fusion must be \"blend\" or \"rrf\", not {other:?}"),
    };
    let query_embedding = options
        .embedding
        .as_ref()
        .map(|v| {
            ensure!(
                embedding_dimensions.is_some(),
                "configure an embedding space before semantic search"
            );
            normalize_embedding(v, embedding_dimensions.unwrap_or(0))
        })
        .transpose()?;
    // Unstemmed and stemmed query forms: a chunk is scored with the form it was stored in.
    let query_terms = [query_words(text, false), query_words(text, true)];
    let candidates = if indexed {
        store.candidate_ids(options)
    } else {
        None
    };
    let eligible = match candidates {
        Some(ids) => eligible_chunks(
            ids.into_iter().map(|id| {
                store
                    .documents
                    .get(id)
                    .expect("document candidate is absent from store")
                    .as_ref()
            }),
            options,
        ),
        None => eligible_chunks(store.values(), options),
    };
    if eligible.is_empty() {
        return Ok(Vec::new());
    }
    let n = eligible.len() as f64;
    let average = eligible.iter().map(|(_, c)| c.word_count).sum::<usize>() as f64 / n;
    let posting_work: usize = query_terms
        .iter()
        .enumerate()
        .map(|(form, terms)| {
            terms
                .iter()
                .map(|word| store.postings[form].get(word).map_or(0, |p| p.len()))
                .sum::<usize>()
        })
        .sum();
    let scan_work = eligible.len().saturating_mul(
        query_terms
            .iter()
            .map(std::collections::BTreeSet::len)
            .sum(),
    );
    let use_postings = indexed && posting_work.saturating_add(store.slot_count) < scan_work;
    let mut positions = Vec::new();
    if use_postings {
        positions.resize(store.slot_count, usize::MAX);
        let mut previous = "";
        let mut chunk = 0;
        for (i, (document, _)) in eligible.iter().enumerate() {
            if document.id != previous {
                previous = &document.id;
                chunk = 0;
            }
            positions[store.slots[&document.id][chunk]] = i;
            chunk += 1;
        }
    }
    let idf: [BTreeMap<&String, f64>; 2] = std::array::from_fn(|form| {
        let stemmed = form == 1;
        query_terms[form]
            .iter()
            .map(|word| {
                let df = if use_postings {
                    store.postings[form].get(word).map_or(0, |posting| {
                        posting
                            .iter()
                            .filter(|&(slot, _)| positions[slot] != usize::MAX)
                            .count()
                    })
                } else {
                    eligible
                        .iter()
                        .filter(|(_, c)| c.stemmed == stemmed && c.terms.contains_key(word))
                        .count()
                } as f64;
                (word, Bm25::idf(n, df))
            })
            .collect()
    });
    let mut scores: Vec<(f64, Option<f64>)> = eligible
        .iter()
        .map(|(_, c)| {
            let lexical = if use_postings {
                0.0
            } else {
                idf[usize::from(c.stemmed)]
                    .iter()
                    .map(|(word, idf)| {
                        let tf = c.terms.get(word).copied().unwrap_or(0) as f64;
                        bm25.weight(*idf, tf, bm25.norm(c.word_count as f64, average.max(1.0)))
                    })
                    .sum()
            };
            let semantic = query_embedding
                .as_ref()
                .zip(c.embedding.as_ref())
                .map(|(q, v)| v.cosine(q));
            (lexical, semantic)
        })
        .collect();
    if use_postings {
        for (form, weights) in idf.iter().enumerate() {
            for (word, idf) in weights {
                if let Some(posting) = store.postings[form].get(word) {
                    for (slot, frequency) in posting.iter() {
                        let i = positions[slot];
                        if i == usize::MAX {
                            continue;
                        }
                        let c = eligible[i].1;
                        let tf = f64::from(frequency);
                        scores[i].0 +=
                            bm25.weight(*idf, tf, bm25.norm(c.word_count as f64, average.max(1.0)));
                    }
                }
            }
        }
    }
    let mut lexical: Vec<usize> = (0..scores.len())
        .filter(|&i| scores[i].0 > 0.0 && lexical_weight > 0.0)
        .collect();
    let mut semantic: Vec<usize> = (0..scores.len())
        .filter(|&i| semantic_weight > 0.0 && scores[i].1.is_some_and(|s| s >= minimum))
        .collect();
    top_ranked(&mut lexical, limit, |&a, &b| {
        scores[b].0.total_cmp(&scores[a].0).then(a.cmp(&b))
    });
    top_ranked(&mut semantic, limit, |&a, &b| {
        scores[b]
            .1
            .unwrap_or(-1.0)
            .total_cmp(&scores[a].1.unwrap_or(-1.0))
            .then(a.cmp(&b))
    });
    let mut fused = BTreeMap::<usize, f64>::new();
    for (list, (indices, weight)) in [(&lexical, lexical_weight), (&semantic, semantic_weight)]
        .into_iter()
        .enumerate()
    {
        let top = &indices[..indices.len().min(limit)];
        if rrf {
            for (rank, &i) in top.iter().enumerate() {
                *fused.entry(i).or_default() += weight / (60.0 + rank as f64 + 1.0);
            }
            continue;
        }
        // Min-max over the list's own top candidates; a flat list contributes its weight.
        let value = |i: usize| {
            if list == 0 {
                scores[i].0
            } else {
                scores[i].1.unwrap_or(-1.0)
            }
        };
        let (hi, lo) = match (top.first(), top.last()) {
            (Some(&f), Some(&l)) => (value(f), value(l)),
            _ => continue,
        };
        for &i in top {
            let norm = if hi > lo {
                (value(i) - lo) / (hi - lo)
            } else {
                1.0
            };
            *fused.entry(i).or_default() += weight * norm;
        }
    }
    let mut ranked: Vec<_> = fused.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(ranked
        .into_iter()
        .take(k)
        .map(|(i, score)| {
            let (d, c) = eligible[i];
            DocumentResult {
                id: c.id.clone(),
                document_id: d.id.clone(),
                source_uri: d.source_uri.clone(),
                version: d.version.clone(),
                metadata: d.metadata.clone(),
                text: c.text.clone(),
                start_byte: c.start_byte,
                end_byte: c.end_byte,
                score,
                lexical_score: scores[i].0,
                semantic_score: scores[i].1,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The chunk id keeps the journal's text form, and recognising it agrees with comparing
    /// against it, near misses included.
    #[test]
    fn chunk_ids_match_their_text_form() {
        for document in ["d", "a:b", "x:1:2", "ünï", ""] {
            for index in [0, 9, 10, 99, 100, 4095, usize::MAX] {
                let id = chunk_id(document, index);
                let text = format!("{CHUNK_PREFIX}{}:{document}:{index}", document.len());
                assert_eq!(id, text);
                assert_eq!(id.capacity(), id.len());
                assert!(is_chunk_id(&id, document, index));
                for other in [
                    format!("{id}0"),
                    id[..id.len() - 1].to_owned(),
                    chunk_id(document, index.wrapping_add(1)),
                    chunk_id(&format!("{document}x"), index),
                    id.replacen(CHUNK_PREFIX, "", 1),
                ] {
                    assert_eq!(
                        is_chunk_id(&other, document, index),
                        other == id,
                        "{other:?}"
                    );
                }
            }
        }
    }

    fn document(i: usize, revision: usize) -> StoredDocument {
        let input: DocumentInput = serde_json::from_value(json!({
            "id":format!("d{i:03}"), "version":revision.to_string(),
            "text":format!("running common topic{} rare{i} shared birds fly distant place revision{revision}",i%7),
            "sourceUri":format!("source{}",i%3), "metadata":{"group":i%5,"empty":null,"flag":i.is_multiple_of(2),"text":"n","fraction":-0.0,"nested":{"key":i%5}},
            "chunkWords":5,"overlapWords":1,
            "embeddings":[[1.0,0.1,0.0],[0.0,1.0,0.1],[0.1,0.0,1.0]]
        })).unwrap();
        let mut doc = prepare(input, 1024, Some(3), TextEncoder::default()).unwrap();
        if i.is_multiple_of(2) {
            for chunk in &mut doc.chunks {
                chunk.terms = terms(chunk.text.as_deref().unwrap(), false).into();
                chunk.stemmed = false;
            }
        }
        doc
    }

    #[test]
    fn bm25_options_change_scores_only_when_set() {
        let mut store = DocumentStore::default();
        for i in 0..12 {
            let doc = document(i, 1);
            store.insert(doc.id.clone(), doc);
        }
        let lexical = |k1: Option<f64>, b: Option<f64>| {
            let options = SearchOptions {
                k: Some(5),
                lexical_weight: Some(1.0),
                semantic_weight: Some(0.0),
                bm25_k1: k1,
                bm25_b: b,
                ..Default::default()
            };
            search_impl(&store, "rare3 shared", &options, Some(3), true).map(|hits| {
                hits.into_iter()
                    .map(|h| (h.id, h.lexical_score))
                    .collect::<Vec<_>>()
            })
        };
        let default = lexical(None, None).unwrap();
        assert_eq!(default, lexical(Some(1.2), Some(0.75)).unwrap());
        // k1 = 0 saturates every term at its idf, so the top hit scores the sum of its idfs.
        let flat = lexical(Some(0.0), Some(0.5)).unwrap();
        assert_ne!(flat, default);
        let n = store.chunk_count as f64;
        let idf = |word: &str| {
            let df = store.postings[1].get(word).map_or(0, |list| list.len()) as f64;
            (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
        };
        let stem = |word: &str| terms(word, true).into_keys().next().unwrap();
        assert!((flat[0].1 - (idf(&stem("rare3")) + idf(&stem("shared")))).abs() < 1e-12);
        assert!(lexical(Some(-1.0), None).is_err());
        assert!(lexical(None, Some(1.5)).is_err());
        assert!(lexical(Some(f64::NAN), None).is_err());
        assert_scan_parity(&store);
    }

    fn assert_scan_parity(store: &DocumentStore) {
        for query in [
            "running rare3 missing absent unknown extra terms for selective postings",
            "running rare3",
            "birds distant",
            "common",
            "missing token",
            "",
        ] {
            for fusion in ["blend", "rrf"] {
                for filter in [
                    None,
                    Some(json!({"group":3})),
                    Some(json!({"group":99})),
                    Some(json!({"empty":null,"flag":true,"text":"n"})),
                    Some(json!({"fraction":0.0,"nested":{"key":3}})),
                ] {
                    for limit in [9, 10, 11, 100] {
                        for embedding in [None, Some(vec![1., 0., 0.])] {
                            let opts = SearchOptions {
                                k: Some(9),
                                candidate_limit: Some(limit),
                                filter: filter.clone(),
                                fusion: Some(fusion.into()),
                                embedding,
                                min_semantic_score: Some(-1.),
                                ..Default::default()
                            };
                            let indexed = search_impl(store, query, &opts, Some(3), true).unwrap();
                            let scanned = search_impl(store, query, &opts, Some(3), false).unwrap();
                            assert_eq!(
                                serde_json::to_value(indexed).unwrap(),
                                serde_json::to_value(scanned).unwrap()
                            );
                        }
                    }
                }
            }
        }
        let opts = SearchOptions {
            source_uri: Some("source1".into()),
            document_ids: Some(vec![
                "d010".into(),
                "d003".into(),
                "d010".into(),
                "absent".into(),
            ]),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(search_impl(store, "running", &opts, Some(3), true).unwrap())
                .unwrap(),
            serde_json::to_value(search_impl(store, "running", &opts, Some(3), false).unwrap())
                .unwrap()
        );
    }

    #[test]
    fn postings_preserve_filtered_mixed_tokenization_scores_after_slot_reuse() {
        let mut store = DocumentStore::default();
        for i in 0..40 {
            let doc = document(i, 1);
            store.insert(doc.id.clone(), doc);
        }
        let query = "running rare3 missing absent unknown extra terms for selective postings";
        let forms = [query_words(query, false), query_words(query, true)];
        let posting_work: usize = forms
            .iter()
            .enumerate()
            .map(|(form, terms)| {
                terms
                    .iter()
                    .map(|word| store.postings[form].get(word).map_or(0, |p| p.len()))
                    .sum::<usize>()
            })
            .sum();
        let eligible = store
            .values()
            .filter(|d| d.metadata["group"] == 3)
            .map(|d| d.chunks.len())
            .sum::<usize>();
        assert!(
            posting_work + store.slot_count
                < eligible
                    * forms
                        .iter()
                        .map(std::collections::BTreeSet::len)
                        .sum::<usize>()
        );
        assert_scan_parity(&store);
        for i in (0..40).step_by(2) {
            store.remove(&format!("d{i:03}"));
        }
        for i in 0..20 {
            let mut doc = document(i, 2);
            if i == 3 {
                doc.metadata["group"] = json!(22);
                doc.source_uri = Some("replacement-source".into());
            }
            store.insert(doc.id.clone(), doc);
        }
        assert_scan_parity(&store);
        let ids: Vec<_> = store.values().map(|doc| doc.id.clone()).collect();
        for id in ids {
            store.remove(&id);
        }
        assert_eq!(store.slot_count, 0);
        assert!(store.postings.iter().all(|p| p.term_count() == 0));
        let doc = document(0, 3);
        store.insert(doc.id.clone(), doc);
        assert_scan_parity(&store);
    }

    #[test]
    fn deleting_low_slots_reclaims_postings_with_high_slot_survivors() {
        for n in [1023, 1024, 1025] {
            let mut store = DocumentStore::default();
            for i in 0..n {
                let mut doc = document(i, 1);
                doc.chunks.truncate(1);
                store.insert(doc.id.clone(), doc);
            }
            for i in 0..n - 1 {
                store.remove(&format!("d{i:03}"));
            }
            assert!(store.slot_count <= 1024);
            assert_eq!(store.values().count(), 1);
            for form in &store.postings {
                form.assert_valid();
            }
            assert_scan_parity(&store);
        }
    }

    #[test]
    fn restored_zero_frequency_terms_do_not_become_lexical_candidates() {
        let mut store = DocumentStore::default();
        let mut document = document(0, 1);
        for chunk in &mut document.chunks {
            chunk.terms.insert("ghost".into(), 0);
        }
        store.insert(document.id.clone(), document);
        for fusion in ["blend", "rrf"] {
            let options = SearchOptions {
                fusion: Some(fusion.into()),
                ..Default::default()
            };
            assert!(search_impl(&store, "ghost", &options, None, true)
                .unwrap()
                .is_empty());
            for query in ["ghost common", "ghost rare0"] {
                assert_eq!(
                    serde_json::to_value(search_impl(&store, query, &options, None, true).unwrap())
                        .unwrap(),
                    serde_json::to_value(
                        search_impl(&store, query, &options, None, false).unwrap()
                    )
                    .unwrap()
                );
            }
        }
        // Counts agree: a zero-frequency posting is no match, through the multichunk bitmap
        // and through the one-chunk-per-document shortcut.
        assert_eq!(count_matches(&store, "ghost").unwrap(), 0);
        assert_eq!(count_matches(&store, "ghost common").unwrap(), 1);
        let mut single = DocumentStore::default();
        // `document` names the moved local here; the helper is reached by path.
        let mut document = self::document(1, 1);
        document.chunks.truncate(1);
        document.chunks[0].terms.insert("ghost".into(), 0);
        single.insert(document.id.clone(), document);
        assert_eq!(count_matches(&single, "ghost").unwrap(), 0);
        assert_eq!(count_matches(&single, "ghost common").unwrap(), 1);
    }

    #[test]
    fn embeddings_round_to_f32_and_read_legacy_f64_records() {
        let legacy = "[0.1, 0.2, 0.30000000000000004, -0.7071067811865476]";
        let read: Embedding = serde_json::from_str(legacy).unwrap();
        assert_eq!(
            read,
            Embedding::from_f32(vec![0.1, 0.2, 0.3, -std::f32::consts::FRAC_1_SQRT_2])
        );
        let written = serde_json::to_string(&read).unwrap();
        assert_eq!(written, "[0.1,0.2,0.3,-0.70710677]");
        let again: Embedding = serde_json::from_str(&written).unwrap();
        assert_eq!(again, read);
        assert_eq!(
            read.to_f64(),
            read.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
        );

        let stored = document(5, 1);
        let embedded = stored.chunks[0].embedding.as_ref().unwrap();
        let reference = normalize_embedding(&[1.0, 0.1, 0.0], 3).unwrap();
        assert_eq!(
            &embedded[..],
            &reference.iter().map(|&v| v as f32).collect::<Vec<_>>()[..]
        );
        let query = normalize_embedding(&[0.3, -0.2, 0.9], 3).unwrap();
        let exact: f64 = query.iter().zip(&reference).map(|(q, v)| q * v).sum();
        assert!((embedded.cosine(&query) - exact).abs() < 1e-6);
        let widened: f64 = query
            .iter()
            .zip(embedded.iter())
            .map(|(q, &v)| q * f64::from(v))
            .sum();
        assert_eq!(embedded.cosine(&query), widened.clamp(-1.0, 1.0));

        let json = serde_json::to_value(&stored).unwrap();
        assert!(json["chunks"][0]["embedding"].is_array());
        let parsed: StoredDocument = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.chunks[0].embedding, stored.chunks[0].embedding);
        let without: StoredChunk = {
            let mut chunk = stored.chunks[0].clone();
            chunk.embedding = None;
            chunk
        };
        assert_eq!(
            serde_json::to_value(&without).unwrap()["embedding"],
            Value::Null
        );
    }

    #[test]
    fn replacements_preserve_frequency_filters_embeddings_and_retained_views() {
        for chunks in [1, 2, 3] {
            for change_tokenization in [false, true] {
                let mut store = DocumentStore::default();
                for i in 0..20 {
                    let mut doc = document(i, 1);
                    if i == 3 {
                        doc.chunks.truncate(2);
                    }
                    store.insert(doc.id.clone(), doc);
                }
                store.remove("d000");
                let options = SearchOptions {
                    k: Some(100),
                    candidate_limit: Some(100),
                    embedding: Some(vec![1.0, 0.0, 0.0]),
                    ..Default::default()
                };
                let prepared = prepare_search("running common ghost", &options, Some(3)).unwrap();
                let retained = capture_search(&store, vec![prepared]);
                let old = serde_json::to_value(execute_search(&retained, 0)).unwrap();
                let mut replacement = document(3, 2);
                replacement.chunks.truncate(chunks);
                replacement.metadata = json!({"group":99, "flag":false});
                replacement.source_uri = Some("replacement-source".into());
                for chunk in &mut replacement.chunks {
                    chunk.terms.insert("common".into(), (chunks - 1) as u32);
                    chunk.terms.insert("ghost".into(), 2);
                    chunk.terms.remove("run");
                    chunk.word_count += 1;
                    chunk.embedding = None;
                    if change_tokenization {
                        chunk.stemmed = !chunk.stemmed;
                    }
                }
                store.insert(replacement.id.clone(), replacement);
                assert_scan_parity(&store);
                let captured = capture_search(
                    &store,
                    vec![prepare_search("running common ghost", &options, Some(3)).unwrap()],
                );
                assert_eq!(
                    serde_json::to_value(execute_search(&captured, 0)).unwrap(),
                    serde_json::to_value(
                        search_impl(&store, "running common ghost", &options, Some(3), false)
                            .unwrap()
                    )
                    .unwrap()
                );
                assert_eq!(
                    serde_json::to_value(execute_search(&retained, 0)).unwrap(),
                    old
                );
                assert_eq!(
                    store.embedded_count(),
                    store
                        .values()
                        .flat_map(|d| &d.chunks)
                        .filter(|c| c.embedding.is_some())
                        .count()
                );
                assert_eq!(
                    store.word_count,
                    store
                        .values()
                        .flat_map(|d| &d.chunks)
                        .map(|c| c.word_count)
                        .sum::<usize>()
                );
                assert_eq!(
                    store.chunk_count,
                    store.values().map(|d| d.chunks.len()).sum::<usize>()
                );
                for form in &store.postings {
                    form.assert_valid();
                }
                let selected = search_impl(
                    &store,
                    "ghost",
                    &SearchOptions {
                        source_uri: Some("replacement-source".into()),
                        filter: Some(json!({"group":99})),
                        k: Some(100),
                        candidate_limit: Some(100),
                        ..Default::default()
                    },
                    Some(3),
                    true,
                )
                .unwrap();
                assert_eq!(selected.len(), chunks);
                assert!(selected.iter().all(|r| r.document_id == "d003"
                    && r.version == "2"
                    && r.semantic_score.is_none()));
                let mut embedded = document(3, 3);
                embedded.chunks.truncate(chunks);
                store.insert(embedded.id.clone(), embedded);
                assert_scan_parity(&store);
                assert!(store.embedded_documents.contains("d003"));
                assert_eq!(
                    store.embedded_count(),
                    store
                        .values()
                        .flat_map(|d| &d.chunks)
                        .filter(|c| c.embedding.is_some())
                        .count()
                );
                store.remove("d003");
                assert_scan_parity(&store);
                assert_eq!(
                    serde_json::to_value(execute_search(&retained, 0)).unwrap(),
                    old
                );
            }
        }
    }
}
