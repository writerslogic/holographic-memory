// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::{encoding::encode_text_internal, entangled::EntangledHVec};

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

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct StoredChunk {
    pub id: String,
    pub text: Option<String>,
    pub start_byte: u32,
    pub end_byte: u32,
    pub vector: EntangledHVec,
    pub embedding: Option<Vec<f64>>,
    pub terms: BTreeMap<String, u32>,
    pub word_count: usize,
    /// Whether `terms` were stemmed at ingest; chunks stored before stemming existed match
    /// the unstemmed query terms.
    #[serde(default)]
    pub stemmed: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct StoredDocument {
    pub id: String,
    pub source_uri: Option<String>,
    pub version: String,
    pub metadata: Value,
    pub chunks: Vec<StoredChunk>,
}

pub fn chunk_document(input: &DocumentInput) -> Result<Vec<DocumentChunk>> {
    ensure!(
        !input.id.is_empty() && input.id.len() <= 4096 && !input.id.contains('\0'),
        "document id must contain 1..=4096 bytes"
    );
    ensure!(
        input.text.len() <= MAX_DOCUMENT_BYTES,
        "document exceeds 8 MiB"
    );
    let size = input.chunk_words.unwrap_or(256) as usize;
    let overlap = input
        .overlap_words
        .unwrap_or(32.min(size.saturating_sub(1)) as u32) as usize;
    ensure!(
        (1..=4096).contains(&size) && overlap < size,
        "chunkWords must be 1..=4096 and overlapWords smaller than chunkWords"
    );
    let mut words = Vec::new();
    let mut start = None;
    for (i, c) in input.text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                words.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
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
        chunks.push(DocumentChunk {
            id: format!(
                "{CHUNK_PREFIX}{}:{}:{}",
                input.id.len(),
                input.id,
                chunks.len()
            ),
            text: input.text[start_byte..end_byte].into(),
            start_byte: start_byte as u32,
            end_byte: end_byte as u32,
        });
        if end_word == words.len() {
            break;
        }
        word += size - overlap;
    }
    Ok(chunks)
}

pub(crate) fn normalize_embedding(values: &[f64], dimensions: usize) -> Result<Vec<f64>> {
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
pub(crate) fn terms(text: &str, stem: bool) -> BTreeMap<String, u32> {
    let mut result = BTreeMap::new();
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
    {
        let lower = word.to_lowercase();
        let key = if stem {
            stemmer().stem(&lower).into_owned()
        } else {
            lower
        };
        *result.entry(key).or_default() += 1;
    }
    result
}

pub(crate) fn prepare(
    input: DocumentInput,
    dimensions: usize,
    embedding_dimensions: Option<usize>,
) -> Result<StoredDocument> {
    prepare_with_keys(input, dimensions, embedding_dimensions, None)
}

/// `keys` replaces each chunk's text as its lexical key (for example the chunk prefixed with
/// facts extracted from it); the stored text and byte offsets stay the chunk's own.
pub(crate) fn prepare_with_keys(
    input: DocumentInput,
    dimensions: usize,
    embedding_dimensions: Option<usize>,
    keys: Option<&[String]>,
) -> Result<StoredDocument> {
    let chunks = chunk_document(&input)?;
    ensure!(
        keys.is_none_or(|k| k.len() == chunks.len()),
        "provide one key per chunk"
    );
    ensure!(
        input.metadata.as_ref().is_none_or(Value::is_object),
        "metadata must be a JSON object"
    );
    ensure!(
        serde_json::to_vec(&input.metadata)?.len() <= 64 * 1024,
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
    for (i, chunk) in chunks.into_iter().enumerate() {
        let terms = terms(keys.map_or(chunk.text.as_str(), |k| k[i].as_str()), true);
        let word_count = terms.values().map(|&n| n as usize).sum();
        let embedding = input
            .embeddings
            .as_ref()
            .map(|all| normalize_embedding(&all[i], embedding_dimensions.unwrap_or(0)))
            .transpose()?;
        stored.push(StoredChunk {
            vector: encode_text_internal(&chunk.text, dimensions),
            text: input.store_text.unwrap_or(true).then_some(chunk.text),
            id: chunk.id,
            start_byte: chunk.start_byte,
            end_byte: chunk.end_byte,
            embedding,
            terms,
            word_count,
            stemmed: true,
        });
    }
    Ok(StoredDocument {
        id: input.id,
        source_uri: input.source_uri,
        version: input.version.unwrap_or_else(|| "1".into()),
        metadata: input.metadata.unwrap_or_else(|| serde_json::json!({})),
        chunks: stored,
    })
}

pub(crate) fn search(
    documents: &BTreeMap<String, StoredDocument>,
    text: &str,
    options: &SearchOptions,
    embedding_dimensions: Option<usize>,
) -> Result<Vec<DocumentResult>> {
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
    let query_terms = [terms(text, false), terms(text, true)];
    let eligible: Vec<_> = documents
        .values()
        .filter(|d| {
            options
                .source_uri
                .as_ref()
                .is_none_or(|uri| d.source_uri.as_ref() == Some(uri))
                && options
                    .document_ids
                    .as_ref()
                    .is_none_or(|ids| ids.contains(&d.id))
                && options
                    .filter
                    .as_ref()
                    .and_then(Value::as_object)
                    .is_none_or(|filter| {
                        filter
                            .iter()
                            .all(|(key, value)| d.metadata.get(key) == Some(value))
                    })
        })
        .flat_map(|d| d.chunks.iter().map(move |chunk| (d, chunk)))
        .collect();
    if eligible.is_empty() {
        return Ok(Vec::new());
    }
    let n = eligible.len() as f64;
    let average = eligible.iter().map(|(_, c)| c.word_count).sum::<usize>() as f64 / n;
    let idf: [BTreeMap<&String, f64>; 2] = std::array::from_fn(|form| {
        let stemmed = form == 1;
        query_terms[form]
            .keys()
            .map(|word| {
                let df = eligible
                    .iter()
                    .filter(|(_, c)| c.stemmed == stemmed && c.terms.contains_key(word))
                    .count() as f64;
                (word, (1.0 + (n - df + 0.5) / (df + 0.5)).ln())
            })
            .collect()
    });
    let scores: Vec<(f64, Option<f64>)> = eligible
        .iter()
        .map(|(_, c)| {
            let lexical = idf[usize::from(c.stemmed)]
                .iter()
                .map(|(word, idf)| {
                    let tf = c.terms.get(*word).copied().unwrap_or(0) as f64;
                    idf * tf * 2.2
                        / (tf + 1.2 * (0.25 + 0.75 * c.word_count as f64 / average.max(1.0)))
                })
                .sum();
            let semantic = query_embedding
                .as_ref()
                .zip(c.embedding.as_ref())
                .map(|(q, v)| {
                    q.iter()
                        .zip(v)
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        .clamp(-1.0, 1.0)
                });
            (lexical, semantic)
        })
        .collect();
    let mut lexical: Vec<usize> = (0..scores.len())
        .filter(|&i| scores[i].0 > 0.0 && lexical_weight > 0.0)
        .collect();
    let mut semantic: Vec<usize> = (0..scores.len())
        .filter(|&i| semantic_weight > 0.0 && scores[i].1.is_some_and(|s| s >= minimum))
        .collect();
    lexical.sort_unstable_by(|&a, &b| scores[b].0.total_cmp(&scores[a].0).then(a.cmp(&b)));
    semantic.sort_unstable_by(|&a, &b| {
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
