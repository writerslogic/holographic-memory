// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Document endpoints: ingest (single, batch, NDJSON stream), delete, list, stats, query and
//! index maintenance.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::rejection::QueryRejection;
use axum::extract::{Path as UrlPath, Query, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use holographic_memory::core::documents::{
    GraphUnavailable, IngestOptions, VectorIndexMode, VectorQueryOptions, MAX_EF,
};
use holographic_memory::{chunk_document, DocumentInput, SearchOptions};
use http_body_util::BodyExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::auth::Scope;
use crate::config::*;
use crate::error::{ApiError, ApiResult};
use crate::request::*;
use crate::tenants::{AppState, CoreGuard};
use crate::types::*;

/// One document. `vector` has exactly `--input-dim` finite numbers (omit it only when the server
/// embeds text on-device).
#[derive(Deserialize, JsonSchema)]
pub struct DocIn {
    pub id: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub vector: Option<Vec<f64>>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

pub struct ValidDoc {
    pub input: DocumentInput,
}

pub fn dense(values: &[f64], input_dim: usize, what: &str) -> Result<(), ApiError> {
    if values.len() != input_dim {
        return Err(ApiError::bad(format!(
            "{what} must have exactly {input_dim} values (got {})",
            values.len()
        )));
    }
    if !values.iter().all(|v| v.is_finite()) {
        return Err(ApiError::bad(format!("{what} values must be finite")));
    }
    if values.iter().all(|&v| v == 0.0) {
        return Err(ApiError::bad(format!("{what} must not be all zeros")));
    }
    Ok(())
}

pub fn validate_doc(doc: DocIn, cfg: &Config, server_embeds: bool) -> Result<ValidDoc, ApiError> {
    if doc.id.is_empty()
        || doc.id.len() > MAX_ID_BYTES
        || doc.id.chars().any(char::is_control)
        || doc.id.starts_with("hms:")
    {
        return Err(ApiError::bad(format!(
            "id must be 1..={MAX_ID_BYTES} bytes, without control characters or the reserved \"hms:\" prefix"
        )));
    }
    let text = doc.text.unwrap_or_default();
    if text.len() > MAX_TEXT_BYTES {
        return Err(ApiError::bad(format!(
            "text exceeds {MAX_TEXT_BYTES} bytes"
        )));
    }
    let metadata = match doc.metadata {
        None | Some(Value::Null) => json!({}),
        Some(m @ Value::Object(_)) => m,
        Some(_) => return Err(ApiError::bad("metadata must be a JSON object")),
    };
    if serde_json::to_vec(&metadata).map_or(true, |v| v.len() > MAX_METADATA_BYTES) {
        return Err(ApiError::bad(format!(
            "metadata exceeds {MAX_METADATA_BYTES} bytes"
        )));
    }
    // With an on-device embedder a document may omit its vector; the server embeds its text.
    let embeddings = match doc.vector {
        None if server_embeds && text.split_whitespace().next().is_some() => None,
        vector => {
            let values = vector.unwrap_or_default();
            dense(&values, cfg.input_dim, "vector")?;
            Some(vec![values])
        }
    };
    // The document store indexes text for lexical search and requires at least one
    // word. Queries here are vector-only, so a vector-only document is indexed under a
    // placeholder word that is never stored or returned.
    let has_text = text.split_whitespace().next().is_some();
    let input = DocumentInput {
        id: doc.id,
        text: if has_text { text } else { "_".into() },
        source_uri: None,
        version: None,
        metadata: Some(metadata),
        chunk_words: Some(CHUNK_WORDS),
        overlap_words: Some(0),
        store_text: Some(has_text),
        embeddings,
        title_weight: None,
    };
    match chunk_document(&input) {
        Ok(chunks) if chunks.len() == 1 => Ok(ValidDoc { input }),
        Ok(_) => Err(ApiError::bad(format!(
            "text must fit one chunk of at most {CHUNK_WORDS} words; split the document and send one vector per part"
        ))),
        Err(e) => Err(ApiError::bad(format!("invalid document: {e}"))),
    }
}

fn store_docs(state: &AppState, tenant: &str, docs: Vec<ValidDoc>) -> Result<usize, ApiError> {
    let n = docs.len();
    let mut positions = HashMap::with_capacity(n);
    let mut final_docs: Vec<Option<ValidDoc>> = Vec::with_capacity(n);
    for doc in docs {
        if let Some(previous) = positions.insert(doc.input.id.clone(), final_docs.len()) {
            final_docs[previous] = None;
        }
        final_docs.push(Some(doc));
    }
    let inputs: Vec<DocumentInput> = final_docs.into_iter().flatten().map(|d| d.input).collect();
    let core = state.core(tenant)?;
    let quota = state.cfg.max_documents_per_tenant;
    if quota > 0 {
        let fresh = inputs.iter().filter(|d| !core.has_document(&d.id)).count() as u64;
        if core.document_count() as u64 + fresh > quota {
            state
                .metrics
                .quota_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "quota_exceeded",
                format!("tenant document quota of {quota} would be exceeded"),
            )
            .retry_after(60));
        }
    }
    // Validated at startup (`Config::validate`); parsed per batch so the engine's error text
    // is the one reported should the two ever disagree.
    let options = IngestOptions {
        encode: state
            .cfg
            .ingest_encode
            .parse()
            .map_err(ApiError::internal)?,
    };
    core.memorize_documents_with(inputs, &options)
        .map_err(ApiError::internal)?;
    core.bump();
    state
        .metrics
        .ingest_documents
        .fetch_add(n as u64, Ordering::Relaxed);
    Ok(n)
}

pub fn validate_batch(docs: &[ValidDoc], input_dim: usize) -> Result<(), ApiError> {
    let mut source_bytes = 0usize;
    let mut metadata_bytes = 0usize;
    let mut embedding_values = 0usize;
    for doc in docs {
        source_bytes += doc.input.id.len() + doc.input.text.len();
        metadata_bytes += serde_json::to_vec(&doc.input.metadata)
            .map_err(ApiError::internal)?
            .len();
        embedding_values += doc
            .input
            .embeddings
            .as_ref()
            .map_or(input_dim, |embeddings| {
                embeddings.iter().map(Vec::len).sum()
            });
        if source_bytes > MAX_BATCH_SOURCE_BYTES
            || metadata_bytes > MAX_BATCH_METADATA_BYTES
            || embedding_values > MAX_BATCH_EMBEDDING_VALUES
        {
            return Err(ApiError::bad(
                "batch exceeds 8 MiB ids/text, 8 MiB metadata or 1048576 embedding values",
            ));
        }
    }
    Ok(())
}

pub async fn add_batch(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    json_write(state, req, Scope::Write, |state, tenant, bytes| {
        let docs: Vec<DocIn> = parse(&bytes)?;
        if docs.is_empty() || docs.len() > state.cfg.max_batch {
            return Err(ApiError::bad(format!(
                "batch must contain 1..={} documents",
                state.cfg.max_batch
            )));
        }
        let valid = docs
            .into_iter()
            .map(|d| validate_doc(d, &state.cfg, state.server_embeds()))
            .collect::<Result<Vec<_>, _>>()?;
        validate_batch(&valid, state.cfg.input_dim)?;
        let n = store_docs(state, tenant, valid)?;
        Ok((StatusCode::OK, body(&Added { added: n })))
    })
    .await
}

pub async fn add_one(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    json_write(state, req, Scope::Write, |state, tenant, bytes| {
        let doc: DocIn = parse(&bytes)?;
        let valid = validate_doc(doc, &state.cfg, state.server_embeds())?;
        let n = store_docs(state, tenant, vec![valid])?;
        Ok((StatusCode::OK, body(&Added { added: n })))
    })
    .await
}

pub fn is_ndjson(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        })
        .is_some_and(|t| {
            matches!(
                t.as_str(),
                "application/x-ndjson" | "application/ndjson" | "application/jsonl"
            )
        })
}

const MAX_REPORTED_LINE_ERRORS: usize = 100;

struct StreamTally {
    added: u64,
    rejected: u64,
    errors: Vec<LineError>,
}

impl StreamTally {
    fn reject(&mut self, line: u64, message: String) {
        self.rejected += 1;
        if self.errors.len() < MAX_REPORTED_LINE_ERRORS {
            self.errors.push(LineError { line, message });
        }
    }
}

type Committed = (usize, Vec<(u64, String)>);

impl StreamTally {
    fn apply(&mut self, (added, rejected): Committed) {
        self.added += added as u64;
        for (line, message) in rejected {
            self.reject(line, message);
        }
    }
}

/// Each group is committed on a blocking thread, one at a time, under the stream's single permit.
async fn commit_group(
    state: &Arc<AppState>,
    tenant: &str,
    permit: &Arc<tokio::sync::OwnedSemaphorePermit>,
    lines: Vec<(u64, Vec<u8>)>,
) -> Result<Committed, ApiError> {
    let (state, tenant, permit) = (state.clone(), tenant.to_string(), permit.clone());
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        commit_lines(&state, &tenant, lines)
    })
    .await
    .map_err(ApiError::internal)?
}

/// Parse, validate and commit one group of NDJSON lines as one transaction.
fn commit_lines(
    state: &AppState,
    tenant: &str,
    lines: Vec<(u64, Vec<u8>)>,
) -> Result<Committed, ApiError> {
    let mut valid = Vec::with_capacity(lines.len());
    let mut rejected = Vec::new();
    for (number, bytes) in lines {
        let outcome =
            parse::<DocIn>(&bytes).and_then(|d| validate_doc(d, &state.cfg, state.server_embeds()));
        match outcome {
            Ok(doc) => valid.push(doc),
            Err(e) => rejected.push((number, e.message)),
        }
    }
    validate_batch(&valid, state.cfg.input_dim)?;
    let added = if valid.is_empty() {
        0
    } else {
        store_docs(state, tenant, valid)?
    };
    Ok((added, rejected))
}

/// `POST /api/v1/documents/stream`: newline-delimited documents, committed every
/// `--stream-batch` lines. Valid lines are kept even when others are rejected; the response
/// reports counts and the first rejected lines. Not atomic across batches.
pub async fn ingest_stream(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Write)?;
    if !is_ndjson(req.headers()) {
        return Err(ApiError::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
            "Content-Type must be application/x-ndjson",
        ));
    }
    let permit = acquire(&state).await?;
    let mut stream_body = req.into_body();
    let idle = Duration::from_secs(state.cfg.body_timeout_seconds);
    let (max_line, max_total) = (state.cfg.max_stream_line_bytes, state.cfg.max_stream_bytes);
    let mut tally = StreamTally {
        added: 0,
        rejected: 0,
        errors: Vec::new(),
    };
    let mut pending: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut buffer: Vec<u8> = Vec::new();
    let (mut total, mut line_no) = (0u64, 0u64);
    let mut ended = false;
    let permit = Arc::new(permit);
    while !ended {
        let frame = tokio::time::timeout(idle, stream_body.frame())
            .await
            .map_err(|_| {
                ApiError::new(
                    StatusCode::REQUEST_TIMEOUT,
                    "request_timeout",
                    "request body timed out",
                )
            })?;
        match frame {
            None => ended = true,
            Some(Err(_)) => {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    "request body could not be read",
                ))
            }
            Some(Ok(frame)) => {
                let Ok(data) = frame.into_data() else {
                    continue;
                };
                total += data.len() as u64;
                if total > max_total {
                    return Err(too_large(format!("stream exceeds {max_total} bytes")));
                }
                buffer.extend_from_slice(&data);
            }
        }
        let mut start = 0;
        while let Some(offset) = buffer[start..].iter().position(|b| *b == b'\n') {
            let end = start + offset;
            line_no += 1;
            if end - start > max_line {
                return Err(too_large(format!(
                    "line {line_no} exceeds {max_line} bytes"
                )));
            }
            let line = buffer[start..end].trim_ascii();
            if !line.is_empty() {
                pending.push((line_no, line.to_vec()));
            }
            start = end + 1;
        }
        buffer.drain(..start);
        if buffer.len() > max_line {
            return Err(too_large(format!(
                "line {} exceeds {max_line} bytes",
                line_no + 1
            )));
        }
        if pending.len() >= state.cfg.stream_batch || (ended && !pending.is_empty()) {
            tally
                .apply(commit_group(&state, &tenant, &permit, std::mem::take(&mut pending)).await?);
        }
    }
    let last = buffer.trim_ascii();
    if !last.is_empty() {
        line_no += 1;
        tally.apply(commit_group(&state, &tenant, &permit, vec![(line_no, last.to_vec())]).await?);
    }
    Ok(json_response(
        StatusCode::OK,
        &body(&StreamResult {
            added: tally.added,
            rejected: tally.rejected,
            errors_truncated: tally.rejected as usize > tally.errors.len(),
            errors: tally.errors,
        }),
    ))
}

fn too_large(message: String) -> ApiError {
    ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large", message)
}

pub async fn delete_doc(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<String>,
    req: Request,
) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Write)?;
    // `?purge=true` also rewrites the journal and scrubs what it replaced; without it the text
    // segment is scrubbed but the deleted frame stays in the journal until compaction.
    let purge = req
        .uri()
        .query()
        .is_some_and(|q| q.split('&').any(|pair| pair == "purge=true"));
    let permit = acquire(&state).await?;
    blocking(permit, move || {
        if id.is_empty() || id.len() > MAX_ID_BYTES || id.starts_with("hms:") {
            return Err(ApiError::bad("invalid id"));
        }
        let core = state.core(&tenant)?;
        if core.delete_document(&id).map_err(ApiError::internal)? {
            if purge {
                core.compact_scrubbing().map_err(ApiError::internal)?;
            }
            core.bump();
            Ok(())
        } else {
            Err(ApiError::not_found("document not found"))
        }
    })
    .await?;
    Ok(json_response(
        StatusCode::OK,
        &body(&Deleted { deleted: true }),
    ))
}

/// A lexical text query over the tenant's documents (memory facts included).
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchIn {
    /// 1..=8192 bytes with at least one word.
    pub query: String,
    /// 1..=`--max-top-k`, default 10.
    #[serde(default)]
    pub top_k: Option<i64>,
    /// Extra fields per hit: any of `"text"` and `"metadata"`. Default: ids and scores only.
    #[serde(default)]
    pub include: Vec<String>,
}

/// Text ranking (BM25; the model stages when configured) of the tenant's documents by `query`: ids and scores, best first, one hit per
/// document. The response is serialized from the engine's results without an intermediate
/// JSON tree, and stored text or metadata is copied into it only when requested.
pub async fn search(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Read)?;
    let permit = acquire(&state).await?;
    let bytes = read_body(&state, req.into_body()).await?;
    let out = blocking(permit, move || {
        let q: SearchIn = parse(&bytes)?;
        if q.query.len() > 8192 || q.query.split_whitespace().next().is_none() {
            return Err(ApiError::bad("query requires 1..=8192 bytes with a word"));
        }
        let top_k = q.top_k.unwrap_or(10);
        if top_k < 1 || top_k > i64::from(state.cfg.max_top_k) {
            return Err(ApiError::bad(format!(
                "top_k must be 1..={}",
                state.cfg.max_top_k
            )));
        }
        let (mut text, mut metadata) = (false, false);
        for field in &q.include {
            match field.as_str() {
                "text" => text = true,
                "metadata" => metadata = true,
                _ => return Err(ApiError::bad("include accepts \"text\" and \"metadata\"")),
            }
        }
        let options = SearchOptions {
            k: Some(top_k as u32),
            candidate_limit: Some(top_k as u32),
            lexical_weight: Some(1.0),
            semantic_weight: Some(0.0),
            ..SearchOptions::default()
        };
        let results = state
            .core(&tenant)?
            .search_documents(&q.query, &options)
            .map_err(ApiError::internal)?;
        // A document longer than one chunk can rank several chunks; its best one stands.
        let mut seen = std::collections::HashSet::with_capacity(results.len());
        let first: Vec<bool> = results
            .iter()
            .map(|hit| seen.insert(hit.document_id.as_str()))
            .collect();
        let hits: Vec<SearchHit> = results
            .into_iter()
            .zip(first)
            .filter(|(_, first)| *first)
            .map(|(hit, _)| SearchHit {
                id: hit.document_id,
                score: hit.score,
                text: text.then(|| hit.text.unwrap_or_default()),
                metadata: metadata.then_some(hit.metadata),
            })
            .collect();
        serde_json::to_vec(&SearchHits { hits }).map_err(ApiError::internal)
    })
    .await?;
    Ok(json_bytes_response(StatusCode::OK, out))
}

#[derive(Deserialize, JsonSchema)]
pub struct QueryIn {
    /// Query embedding: exactly `--input-dim` finite numbers.
    #[serde(default)]
    pub query_vector: Option<Vec<f64>>,
    /// Text query; requires the on-device model stages (`local-models`).
    #[serde(default)]
    pub query: Option<String>,
    /// Date relative times in `query` resolve against (query rewriting).
    #[serde(default)]
    #[cfg_attr(not(feature = "local-models"), allow(dead_code))]
    pub today: Option<String>,
    /// 1..=`--max-top-k`, default 3.
    #[serde(default)]
    pub top_k: Option<i64>,
    /// Keep documents whose metadata contains every key with an equal JSON value.
    #[serde(default)]
    pub filter: Option<Value>,
    /// Route of an unfiltered vector query: `auto` (the graph index when built, else the
    /// exact scan), `exact` or `graph` (409 `index_unavailable` until maintenance built one).
    /// Default: the server's `--query-index`.
    #[serde(default)]
    pub index: Option<String>,
    /// Beam width of a graph-index query, 1..=4096 (recall knob). Default: `--query-ef`.
    #[serde(default)]
    pub ef: Option<u32>,
}

fn query_options(
    cfg: &Config,
    index: Option<&str>,
    ef: Option<u32>,
) -> Result<VectorQueryOptions, ApiError> {
    let mode = match index {
        None => VectorIndexMode::parse(&cfg.query_index).unwrap_or_default(),
        Some(name) => VectorIndexMode::parse(name)
            .ok_or_else(|| ApiError::bad("index must be \"auto\", \"exact\" or \"graph\""))?,
    };
    let ef = ef.unwrap_or(cfg.query_ef);
    if !(1..=MAX_EF as u32).contains(&ef) {
        return Err(ApiError::bad(format!("ef must be 1..={MAX_EF}")));
    }
    Ok(VectorQueryOptions {
        mode,
        ef: ef as usize,
        min_score: -1.0,
    })
}

pub async fn query(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Read)?;
    let permit = acquire(&state).await?;
    let bytes = read_body(&state, req.into_body()).await?;
    let matches = blocking(permit, move || {
        let q: QueryIn = parse(&bytes)?;
        let top_k = q.top_k.unwrap_or(3);
        if top_k < 1 || top_k > i64::from(state.cfg.max_top_k) {
            return Err(ApiError::bad(format!(
                "top_k must be 1..={}",
                state.cfg.max_top_k
            )));
        }
        let filter = match q.filter {
            None | Some(Value::Null) => None,
            Some(Value::Object(m)) if m.is_empty() => None,
            Some(Value::Object(m)) => Some(m),
            Some(_) => return Err(ApiError::bad("filter must be a JSON object")),
        };
        let core = state.core(&tenant)?;
        let options = query_options(&state.cfg, q.index.as_deref(), q.ef)?;
        let hits = match (q.query_vector, q.query) {
            (Some(v), None) => {
                dense(&v, state.cfg.input_dim, "query_vector")?;
                vector_search(&core, v, top_k as u32, filter, &options)?
            }
            #[cfg(feature = "local-models")]
            (v, Some(text)) if state.stages.is_some() => {
                if text.trim().is_empty() || text.len() > 8192 {
                    return Err(ApiError::bad("query requires 1..=8192 bytes"));
                }
                if let Some(v) = &v {
                    dense(v, state.cfg.input_dim, "query_vector")?;
                }
                let options = SearchOptions {
                    k: Some(top_k as u32),
                    candidate_limit: Some((top_k as u32).max(100)),
                    filter: filter.map(Value::Object),
                    embedding: v,
                    ..SearchOptions::default()
                };
                let stages = state.stages.as_ref().expect("guarded");
                core.search_documents_with_models(
                    &text,
                    q.today.as_deref().unwrap_or(""),
                    &options,
                    stages,
                )
                .map_err(ApiError::internal)?
            }
            (_, Some(_)) => {
                return Err(ApiError::bad(
                    "text queries need the server's on-device model stages; send query_vector",
                ))
            }
            (None, None) => return Err(ApiError::bad("query_vector is required")),
        };
        Ok(hits
            .into_iter()
            .map(|hit| Match {
                id: hit.document_id,
                text: hit.text.unwrap_or_default(),
                metadata: hit.metadata,
                score: hit.score,
            })
            .collect::<Vec<_>>())
    })
    .await?;
    Ok(json_response(StatusCode::OK, &body(&Matches { matches })))
}

/// An unfiltered query goes to the engine's vector dispatch (snapshot: graph or exact scan,
/// generic exact path without one); a filtered query takes the generic path, which is the
/// only one that applies metadata filters.
fn vector_search(
    core: &CoreGuard,
    query_vector: Vec<f64>,
    top_k: u32,
    filter: Option<serde_json::Map<String, Value>>,
    route: &VectorQueryOptions,
) -> Result<Vec<holographic_memory::DocumentResult>, ApiError> {
    let Some(filter) = filter else {
        return core
            .search_vectors(&query_vector, top_k as usize, route)
            .map_err(|e| {
                if e.downcast_ref::<GraphUnavailable>().is_some() {
                    ApiError::new(StatusCode::CONFLICT, "index_unavailable", e.to_string())
                } else {
                    ApiError::internal(e)
                }
            });
    };
    if route.mode == VectorIndexMode::Graph {
        return Err(ApiError::bad(
            "index \"graph\" does not apply to filtered queries; omit index or send \"exact\"",
        ));
    }
    let options = SearchOptions {
        k: Some(top_k),
        candidate_limit: Some(top_k.max(100)),
        filter: Some(Value::Object(filter)),
        embedding: Some(query_vector),
        lexical_weight: Some(0.0),
        semantic_weight: Some(1.0),
        min_semantic_score: Some(-1.0),
        ..SearchOptions::default()
    };
    let hits = core
        .search_documents("", &options)
        .map_err(ApiError::internal)?;
    // The response score of a vector query is the cosine similarity.
    Ok(hits
        .into_iter()
        .map(|mut hit| {
            hit.score = hit.semantic_score.unwrap_or(0.0);
            hit
        })
        .collect())
}

pub async fn maintain_index(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Write)?;
    let permit = acquire(&state).await?;
    let bytes = read_body(&state, req.into_body()).await?;
    let trained = blocking(permit, move || {
        if !bytes.is_empty() && parse::<Value>(&bytes)? != json!({}) {
            return Err(ApiError::bad(
                "index maintenance accepts an empty body or {}",
            ));
        }
        let core = state.core(&tenant)?;
        // The vector snapshot (graph index above --graph-min-rows) first, so the bounding-box
        // index below sees it current and does not build it a second time.
        core.rebuild_vector_index(state.cfg.graph_min_rows)
            .map_err(ApiError::internal)?;
        core.train_document_index().map_err(ApiError::internal)?;
        Ok((
            core.document_index_trained(),
            VectorIndexBody::from(core.vector_index_status()),
        ))
    })
    .await?;
    let (trained, vector_index) = trained;
    Ok(json_response(
        StatusCode::OK,
        &body(&Maintained {
            document_index_trained: trained,
            vector_index,
        }),
    ))
}

#[derive(Deserialize, JsonSchema)]
pub struct ListParams {
    /// Page size, 1..=1000 (default 100).
    #[serde(default)]
    pub limit: Option<usize>,
    /// The `next_cursor` of the previous page; ids are listed in ascending byte order.
    #[serde(default)]
    pub after: Option<String>,
}

fn etag(state: &AppState, core: &CoreGuard) -> String {
    format!(
        "W/\"{:x}-{}\"",
        state.epoch,
        core.tenant.generation.load(Ordering::Acquire)
    )
}

fn not_modified(headers: &HeaderMap, etag: &str) -> bool {
    headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|t| t.trim() == etag || t.trim() == "*"))
}

fn with_etag(mut response: Response, etag: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(etag) {
        response.headers_mut().insert(header::ETAG, value);
    }
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

/// `GET /api/v1/documents`: ids and metadata, paginated by id cursor, with the total count.
pub async fn list_documents(
    State(state): State<Arc<AppState>>,
    params: Result<Query<ListParams>, QueryRejection>,
    req: Request,
) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Read)?;
    let Query(params) = params.map_err(|e| ApiError::bad(format!("invalid query string: {e}")))?;
    let limit = params.limit.unwrap_or(100);
    if !(1..=1000).contains(&limit) {
        return Err(ApiError::bad("limit must be 1..=1000"));
    }
    let headers = req.headers().clone();
    let permit = acquire(&state).await?;
    let state2 = state.clone();
    let (status, body, tag) = blocking(permit, move || {
        let core = state2.core(&tenant)?;
        let tag = etag(&state2, &core);
        if not_modified(&headers, &tag) {
            return Ok((StatusCode::NOT_MODIFIED, Value::Null, tag));
        }
        let from = params.after.as_deref().map(|after| format!("{after}\0"));
        let mut page = core.list_documents(from.as_deref(), limit + 1);
        let more = page.len() > limit;
        page.truncate(limit);
        let next = more
            .then(|| page.last().map(|record| record.id.clone()))
            .flatten();
        let documents = page
            .into_iter()
            .map(|record| ListedDocument {
                id: record.id,
                metadata: record.metadata,
            })
            .collect();
        Ok((
            StatusCode::OK,
            body(&DocumentPage {
                documents,
                next_cursor: next,
                count: core.document_count(),
            }),
            tag,
        ))
    })
    .await?;
    let response = if status == StatusCode::NOT_MODIFIED {
        status.into_response()
    } else {
        json_response(status, &body)
    };
    Ok(with_etag(response, &tag))
}

/// `GET /api/v1/stats`: document and vector counts and storage use for the tenant.
pub async fn stats(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Read)?;
    let headers = req.headers().clone();
    let permit = acquire(&state).await?;
    let state2 = state.clone();
    let (status, body, tag) = blocking(permit, move || {
        let core = state2.core(&tenant)?;
        let tag = etag(&state2, &core);
        if not_modified(&headers, &tag) {
            return Ok((StatusCode::NOT_MODIFIED, Value::Null, tag));
        }
        let health = core.storage_health();
        Ok((
            StatusCode::OK,
            body(&Stats {
                tenant,
                documents: core.document_count(),
                vectors: core.document_embedding_count(),
                document_index_trained: core.document_index_trained(),
                vector_index: VectorIndexBody::from(core.vector_index_status()),
                used_bytes: health.used_bytes,
                capacity_bytes: health.capacity_bytes,
            }),
            tag,
        ))
    })
    .await?;
    let response = if status == StatusCode::NOT_MODIFIED {
        status.into_response()
    } else {
        json_response(status, &body)
    };
    Ok(with_etag(response, &tag))
}
