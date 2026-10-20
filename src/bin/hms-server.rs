// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! HTTP service for the `holographic-sdk` Python client.
//!
//! Endpoints: `POST /api/v1/documents/batch`, `POST /api/v1/documents`,
//! `DELETE /api/v1/documents/{id}`, `POST /api/v1/query`, and the memory API under
//! `POST /api/v1/memory/{remember,recall,explain,forget,get,history,applyForgetting,consolidate}`.
//! See docs/SERVER.md and docs/memory-api.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use axum::body::{to_bytes, Body};
use axum::extract::{Path as UrlPath, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, post};
use axum::{Json, Router};
use clap::Parser;
use holographic_memory::core::memory_api::{Memory, MemoryConfig, MemoryError, OPERATIONS};
use holographic_memory::core::HmsConfig;
use holographic_memory::{chunk_document, DocumentInput, EmbeddingSpace, HmsCore, SearchOptions};
use parking_lot::{Condvar, Mutex};
use serde::Deserialize;
use serde_json::{json, Value};
use subtle::ConstantTimeEq;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const MAX_ID_BYTES: usize = 256;
/// One client vector describes one stored chunk, so a document must fit in one
/// chunk: the engine's per-chunk limits are 4096 words and 64 KiB.
const MAX_TEXT_BYTES: usize = 64 * 1024;
const CHUNK_WORDS: u32 = 4096;
const MAX_METADATA_BYTES: usize = 64 * 1024;
const MAX_TENANT_BYTES: usize = 64;
const MAX_BATCH_DOCUMENTS: usize = 4096;
const MAX_BATCH_SOURCE_BYTES: usize = 8 * 1024 * 1024;
const MAX_BATCH_METADATA_BYTES: usize = 8 * 1024 * 1024;
const MAX_BATCH_EMBEDDING_VALUES: usize = 1_048_576;
const MAX_CONCURRENT_WORK: usize = 256;

#[derive(Parser)]
#[command(
    name = "hms-server",
    about = "HTTP service for the holographic-sdk client. The API key is read from HMS_API_KEY."
)]
struct Cli {
    /// Listen address. Defaults to loopback only.
    #[arg(long, env = "HMS_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    /// Directory holding one store per tenant.
    #[arg(long, env = "HMS_DATA_DIR", default_value = "./hms-data")]
    data_dir: PathBuf,
    /// Hypervector dimensions of each store (>= 256).
    #[arg(long, env = "HMS_DIM", default_value_t = 4096)]
    dim: u32,
    /// Required length of client embeddings.
    #[arg(long, env = "HMS_INPUT_DIM", default_value_t = 384)]
    input_dim: usize,
    /// Embedding model identifier recorded in each store. Reopening a store with
    /// a different model, revision or input dimension is refused.
    #[arg(long, env = "HMS_EMBEDDING_MODEL", default_value = "client-supplied")]
    embedding_model: String,
    /// Embedding model revision recorded in each store.
    #[arg(long, env = "HMS_EMBEDDING_REVISION", default_value = "unversioned")]
    embedding_revision: String,
    /// Maximum request body in bytes.
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    max_body_bytes: usize,
    /// Maximum time to read an admitted request body.
    #[arg(long, env = "HMS_BODY_TIMEOUT_SECONDS", default_value_t = 30)]
    body_timeout_seconds: u64,
    /// Maximum documents per batch request.
    #[arg(long, default_value_t = 1000)]
    max_batch: usize,
    /// Maximum top_k.
    #[arg(long, default_value_t = 100)]
    max_top_k: u32,
    /// Maximum number of tenant stores open at once.
    #[arg(long, default_value_t = 64)]
    max_tenants: usize,
    /// JSON file holding the memory API configuration (policies and defaults) applied to
    /// every tenant; see docs/memory-api. Absent: the conservative defaults.
    #[arg(long, env = "HMS_MEMORY_CONFIG")]
    memory_config: Option<PathBuf>,
    /// Maximum admitted requests reading bodies or executing blocking work.
    #[arg(long, env = "HMS_MAX_CONCURRENT_WORK", default_value_t = 4)]
    max_concurrent_work: usize,
    #[cfg(feature = "local-models")]
    #[command(flatten)]
    models: ModelArgs,
}

/// On-device model stages (feature `local-models`). Each directory holds safetensors and
/// tokenizer.json and must record the given revision; nothing is downloaded.
#[cfg(feature = "local-models")]
#[derive(clap::Args)]
struct ModelArgs {
    /// Qwen3-Embedding directory: documents without a vector and text queries are embedded here.
    #[arg(long, env = "HMS_EMBED_MODEL", requires = "embed_revision")]
    embed_model: Option<PathBuf>,
    #[arg(long, env = "HMS_EMBED_REVISION")]
    embed_revision: Option<String>,
    /// Qwen3-Reranker directory: re-ranks the fused head of text queries.
    #[arg(long, env = "HMS_RERANK_MODEL", requires = "rerank_revision")]
    rerank_model: Option<PathBuf>,
    #[arg(long, env = "HMS_RERANK_REVISION")]
    rerank_revision: Option<String>,
    /// Qwen3 instruct LLM directory for fact extraction and query rewriting.
    #[arg(long, env = "HMS_LLM_MODEL", requires = "llm_revision")]
    llm_model: Option<PathBuf>,
    #[arg(long, env = "HMS_LLM_REVISION")]
    llm_revision: Option<String>,
    /// Extract facts at ingest with the LLM and index them with each document.
    #[arg(long, env = "HMS_EXTRACT_FACTS", requires = "llm_model")]
    extract_facts: bool,
    /// Rewrite text queries with the LLM and search both forms.
    #[arg(long, env = "HMS_REWRITE_QUERIES", requires = "llm_model")]
    rewrite_queries: bool,
}

#[cfg(feature = "local-models")]
fn load_stages(
    args: &ModelArgs,
) -> Result<Option<Arc<holographic_memory::core::models::ModelStages>>> {
    use holographic_memory::core::models::{
        default_device, Embedder, Generator, ModelSource, ModelStages, Reranker,
    };
    let source = |dir: &Option<PathBuf>, rev: &Option<String>| {
        dir.as_ref()
            .map(|d| ModelSource::new(d, rev.clone().unwrap_or_default()))
    };
    let device = default_device()?;
    let stages = ModelStages {
        embedder: source(&args.embed_model, &args.embed_revision)
            .map(|s| Embedder::load(&s, &device))
            .transpose()?,
        reranker: source(&args.rerank_model, &args.rerank_revision)
            .map(|s| Reranker::load(&s, &device))
            .transpose()?,
        generator: source(&args.llm_model, &args.llm_revision)
            .map(|s| Generator::load(&s, &device))
            .transpose()?,
        extract_facts: args.extract_facts,
        rewrite_queries: args.rewrite_queries,
        params: Default::default(),
    };
    let any = stages.embedder.is_some() || stages.reranker.is_some() || stages.generator.is_some();
    Ok(any.then(|| Arc::new(stages)))
}

fn load_memory_config(path: Option<&std::path::Path>) -> Result<MemoryConfig> {
    let Some(path) = path else {
        return Ok(MemoryConfig::default());
    };
    let config: MemoryConfig = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?,
    )
    .with_context(|| format!("invalid memory config {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

struct Tenant {
    core: Arc<HmsCore>,
    memory: Memory,
}

enum TenantStatus {
    Opening,
    Ready(Arc<Tenant>),
    Failed(ApiError),
}

struct TenantSlot {
    status: Mutex<TenantStatus>,
    ready: Condvar,
}

impl TenantSlot {
    fn wait(&self) -> Result<Arc<Tenant>, ApiError> {
        let mut status = self.status.lock();
        loop {
            match &*status {
                TenantStatus::Opening => self.ready.wait(&mut status),
                TenantStatus::Ready(tenant) => return Ok(tenant.clone()),
                TenantStatus::Failed(error) => return Err(error.clone()),
            }
        }
    }
}

struct TenantInitialization<'a> {
    state: &'a AppState,
    name: &'a str,
    slot: Arc<TenantSlot>,
    completed: bool,
}

impl TenantInitialization<'_> {
    fn publish(&self, result: &Result<Arc<Tenant>, ApiError>) {
        *self.slot.status.lock() = match result {
            Ok(tenant) => TenantStatus::Ready(tenant.clone()),
            Err(error) => TenantStatus::Failed(error.clone()),
        };
        self.slot.ready.notify_all();
        if result.is_err() {
            let mut tenants = self.state.tenants.lock();
            if tenants
                .get(self.name)
                .is_some_and(|slot| Arc::ptr_eq(slot, &self.slot))
            {
                tenants.remove(self.name);
            }
        }
    }

    fn finish(mut self, result: Result<Arc<Tenant>, ApiError>) -> Result<Arc<Tenant>, ApiError> {
        self.publish(&result);
        self.completed = true;
        result
    }
}

impl Drop for TenantInitialization<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.publish(&Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "internal server error",
            )));
        }
    }
}

struct WorkLimit {
    slots: Arc<Semaphore>,
    accepting: AtomicBool,
}

impl WorkLimit {
    fn new(limit: usize) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(limit)),
            accepting: AtomicBool::new(true),
        }
    }

    fn acquire(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        let closed = || {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "shutting_down",
                "server is shutting down",
            )
        };
        if !self.accepting.load(Ordering::Acquire) {
            return Err(closed());
        }
        let permit = self.slots.clone().try_acquire_owned().map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_busy",
                "server work capacity is full; retry later",
            )
        })?;
        if !self.accepting.load(Ordering::Acquire) {
            return Err(closed());
        }
        Ok(permit)
    }

    fn stop(&self) {
        self.accepting.store(false, Ordering::Release);
    }
}

struct AppState {
    cli: Cli,
    api_key: Option<Vec<u8>>,
    tenants: Mutex<HashMap<String, Arc<TenantSlot>>>,
    work: WorkLimit,
    #[cfg(feature = "local-models")]
    stages: Option<Arc<holographic_memory::core::models::ModelStages>>,
}

impl AppState {
    /// Whether this server embeds document and query text itself.
    fn server_embeds(&self) -> bool {
        #[cfg(feature = "local-models")]
        if let Some(s) = &self.stages {
            return s.embedder.is_some();
        }
        false
    }

    fn tenant(&self, name: &str) -> Result<Arc<Tenant>, ApiError> {
        let (slot, initialize) = {
            let mut tenants = self.tenants.lock();
            if let Some(slot) = tenants.get(name) {
                (slot.clone(), false)
            } else {
                if tenants.len() >= self.cli.max_tenants {
                    return Err(ApiError::new(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "tenant_limit",
                        "too many tenants are open on this server",
                    ));
                }
                let slot = Arc::new(TenantSlot {
                    status: Mutex::new(TenantStatus::Opening),
                    ready: Condvar::new(),
                });
                tenants.insert(name.to_string(), slot.clone());
                (slot, true)
            }
        };
        if !initialize {
            return slot.wait();
        }
        let reservation = TenantInitialization {
            state: self,
            name,
            slot,
            completed: false,
        };
        reservation.finish(
            self.open_tenant(name)
                .map(Arc::new)
                .map_err(ApiError::internal),
        )
    }

    fn open_tenant(&self, name: &str) -> Result<Tenant> {
        // Hex keeps the directory name safe and distinct on case-insensitive filesystems.
        let hex: String = name.bytes().map(|b| format!("{b:02x}")).collect();
        let tenants_root = self.cli.data_dir.join("tenants");
        let open = || -> Result<Tenant> {
            std::fs::create_dir_all(&tenants_root)?;
            // The hex name cannot contain a separator or `..`; the containment check on the
            // canonical root states that invariant where the path is built.
            let root = tenants_root.canonicalize()?;
            let dir = root.join(&hex);
            anyhow::ensure!(
                dir.starts_with(&root) && dir.parent() == Some(root.as_path()),
                "tenant directory escapes the tenants root"
            );
            std::fs::create_dir_all(&dir)?;
            let config = HmsConfig {
                embedding_space: Some(EmbeddingSpace {
                    model: self.cli.embedding_model.clone(),
                    revision: self.cli.embedding_revision.clone(),
                    dimensions: self.cli.input_dim,
                    normalization: "l2".into(),
                    metric: "cosine".into(),
                }),
                ..HmsConfig::default()
            };
            let core = HmsCore::new(self.cli.dim, Some(dir.display().to_string()), Some(config))?;
            #[cfg(feature = "local-models")]
            core.set_model_stages(self.stages.clone())?;
            let core = Arc::new(core);
            let memory = Memory::with_config(
                core.clone(),
                load_memory_config(self.cli.memory_config.as_deref())?,
            )?;
            Ok(Tenant { core, memory })
        };
        open()
    }
}

#[derive(Clone, Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn bad(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message)
    }

    /// Detail goes to stderr only; the client sees a fixed message.
    fn internal(e: impl std::fmt::Display) -> Self {
        eprintln!("internal error: {e:#}");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "internal server error",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let retry = matches!(self.code, "server_busy" | "shutting_down");
        let mut response = (
            self.status,
            Json(json!({"error": {"code": self.code, "message": self.message}})),
        )
            .into_response();
        if retry {
            response.headers_mut().insert(
                header::RETRY_AFTER,
                "1".parse().expect("static retry header"),
            );
        }
        response
    }
}

type ApiResult = Result<Response, ApiError>;

async fn blocking<T: Send + 'static>(
    permit: OwnedSemaphorePermit,
    f: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        f()
    })
    .await
    .map_err(ApiError::internal)?
}

fn valid_tenant(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_TENANT_BYTES
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn tenant_name(headers: &HeaderMap) -> Result<String, ApiError> {
    let Some(value) = headers.get("x-tenant-id") else {
        return Ok("default".into());
    };
    match value.to_str() {
        Ok(v) if valid_tenant(v) => Ok(v.to_string()),
        _ => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_tenant",
            "X-Tenant-ID must be 1..=64 characters of [A-Za-z0-9_-]",
        )),
    }
}

async fn auth(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    if let Some(key) = &state.api_key {
        let presented = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        let ok: bool = presented.as_bytes().ct_eq(key).into();
        if !ok {
            let mut resp = ApiError::new(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing or invalid API key",
            )
            .into_response();
            resp.headers_mut()
                .insert(header::WWW_AUTHENTICATE, "Bearer".parse().unwrap());
            return resp;
        }
    }
    next.run(req).await
}

async fn read_body(state: &AppState, body: Body) -> Result<Vec<u8>, ApiError> {
    tokio::time::timeout(
        Duration::from_secs(state.cli.body_timeout_seconds),
        to_bytes(body, state.cli.max_body_bytes),
    )
    .await
    .map_err(|_| {
        ApiError::new(
            StatusCode::REQUEST_TIMEOUT,
            "request_timeout",
            "request body timed out",
        )
    })?
    .map(|b| b.to_vec())
    .map_err(|_| {
        ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            format!("request body exceeds {} bytes", state.cli.max_body_bytes),
        )
    })
}

fn parse<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(bytes).map_err(|e| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_json",
            format!("malformed request body: {e}"),
        )
    })
}

#[derive(Deserialize)]
struct DocIn {
    id: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    vector: Option<Vec<f64>>,
    #[serde(default)]
    metadata: Option<Value>,
}

struct ValidDoc {
    input: DocumentInput,
}

fn dense(values: &[f64], input_dim: usize, what: &str) -> Result<(), ApiError> {
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

fn validate_doc(doc: DocIn, cli: &Cli, server_embeds: bool) -> Result<ValidDoc, ApiError> {
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
            dense(&values, cli.input_dim, "vector")?;
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

fn store_docs(tenant: &Tenant, docs: Vec<ValidDoc>) -> Result<usize, ApiError> {
    let n = docs.len();
    let mut positions = HashMap::with_capacity(n);
    let mut final_docs: Vec<Option<ValidDoc>> = Vec::with_capacity(n);
    for doc in docs {
        if let Some(previous) = positions.insert(doc.input.id.clone(), final_docs.len()) {
            final_docs[previous] = None;
        }
        final_docs.push(Some(doc));
    }
    tenant
        .core
        .memorize_documents(
            final_docs
                .into_iter()
                .flatten()
                .map(|doc| doc.input)
                .collect(),
        )
        .map_err(ApiError::internal)?;
    Ok(n)
}

fn validate_batch(docs: &[ValidDoc], input_dim: usize) -> Result<(), ApiError> {
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

async fn add_batch(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let name = tenant_name(req.headers())?;
    let permit = state.work.acquire()?;
    let bytes = read_body(&state, req.into_body()).await?;
    let n = blocking(permit, move || {
        let docs: Vec<DocIn> = parse(&bytes)?;
        if docs.is_empty() || docs.len() > state.cli.max_batch {
            return Err(ApiError::bad(format!(
                "batch must contain 1..={} documents",
                state.cli.max_batch
            )));
        }
        let valid = docs
            .into_iter()
            .map(|d| validate_doc(d, &state.cli, state.server_embeds()))
            .collect::<Result<Vec<_>, _>>()?;
        validate_batch(&valid, state.cli.input_dim)?;
        let tenant = state.tenant(&name)?;
        store_docs(&tenant, valid)
    })
    .await?;
    Ok(Json(json!({"added": n})).into_response())
}

async fn add_one(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let name = tenant_name(req.headers())?;
    let permit = state.work.acquire()?;
    let bytes = read_body(&state, req.into_body()).await?;
    let n = blocking(permit, move || {
        let doc: DocIn = parse(&bytes)?;
        let valid = validate_doc(doc, &state.cli, state.server_embeds())?;
        let tenant = state.tenant(&name)?;
        store_docs(&tenant, vec![valid])
    })
    .await?;
    Ok(Json(json!({"added": n})).into_response())
}

async fn delete_doc(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<String>,
    req: Request,
) -> ApiResult {
    let name = tenant_name(req.headers())?;
    let permit = state.work.acquire()?;
    blocking(permit, move || {
        if id.is_empty() || id.len() > MAX_ID_BYTES || id.starts_with("hms:") {
            return Err(ApiError::bad("invalid id"));
        }
        let tenant = state.tenant(&name)?;
        if tenant
            .core
            .delete_document(&id)
            .map_err(ApiError::internal)?
        {
            Ok(())
        } else {
            Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found",
                "document not found",
            ))
        }
    })
    .await?;
    Ok(Json(json!({"deleted": true})).into_response())
}

#[derive(Deserialize)]
struct QueryIn {
    #[serde(default)]
    query_vector: Option<Vec<f64>>,
    /// Text query; requires the on-device model stages (`local-models`).
    #[serde(default)]
    query: Option<String>,
    /// Date relative times in `query` resolve against (query rewriting).
    #[serde(default)]
    #[cfg_attr(not(feature = "local-models"), allow(dead_code))]
    today: Option<String>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    filter: Option<Value>,
}

async fn query(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let name = tenant_name(req.headers())?;
    let permit = state.work.acquire()?;
    let bytes = read_body(&state, req.into_body()).await?;
    let matches = blocking(permit, move || {
        let q: QueryIn = parse(&bytes)?;
        let top_k = q.top_k.unwrap_or(3);
        if top_k < 1 || top_k > i64::from(state.cli.max_top_k) {
            return Err(ApiError::bad(format!(
                "top_k must be 1..={}",
                state.cli.max_top_k
            )));
        }
        let filter = match q.filter {
            None | Some(Value::Null) => None,
            Some(Value::Object(m)) if m.is_empty() => None,
            Some(Value::Object(m)) => Some(m),
            Some(_) => return Err(ApiError::bad("filter must be a JSON object")),
        };
        let tenant = state.tenant(&name)?;
        let hits = match (q.query_vector, q.query) {
            (Some(v), None) => {
                dense(&v, state.cli.input_dim, "query_vector")?;
                vector_search(&tenant, v, top_k as u32, filter)?
            }
            #[cfg(feature = "local-models")]
            (v, Some(text)) if state.stages.is_some() => {
                if text.trim().is_empty() || text.len() > 8192 {
                    return Err(ApiError::bad("query requires 1..=8192 bytes"));
                }
                if let Some(v) = &v {
                    dense(v, state.cli.input_dim, "query_vector")?;
                }
                let options = SearchOptions {
                    k: Some(top_k as u32),
                    candidate_limit: Some((top_k as u32).max(100)),
                    filter: filter.map(Value::Object),
                    embedding: v,
                    ..SearchOptions::default()
                };
                let stages = state.stages.as_ref().expect("guarded");
                tenant
                    .core
                    .search_documents_with_models(
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
            .map(|hit| {
                json!({
                    "id": hit.document_id,
                    "text": hit.text.unwrap_or_default(),
                    "metadata": hit.metadata,
                    "score": hit.score,
                })
            })
            .collect::<Vec<_>>())
    })
    .await?;
    Ok(Json(json!({"matches": matches})).into_response())
}

async fn maintain_index(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let name = tenant_name(req.headers())?;
    let permit = state.work.acquire()?;
    let bytes = read_body(&state, req.into_body()).await?;
    let trained = blocking(permit, move || {
        if !bytes.is_empty() && parse::<Value>(&bytes)? != json!({}) {
            return Err(ApiError::bad(
                "index maintenance accepts an empty body or {}",
            ));
        }
        let tenant = state.tenant(&name)?;
        tenant
            .core
            .train_document_index()
            .map_err(ApiError::internal)?;
        Ok(tenant.core.document_index_trained())
    })
    .await?;
    Ok(Json(json!({"documentIndexTrained": trained})).into_response())
}

fn vector_search(
    tenant: &Tenant,
    query_vector: Vec<f64>,
    top_k: u32,
    filter: Option<serde_json::Map<String, Value>>,
) -> Result<Vec<holographic_memory::DocumentResult>, ApiError> {
    let options = SearchOptions {
        k: Some(top_k),
        candidate_limit: Some(top_k.max(100)),
        filter: filter.map(Value::Object),
        embedding: Some(query_vector),
        lexical_weight: Some(0.0),
        semantic_weight: Some(1.0),
        min_semantic_score: Some(-1.0),
        ..SearchOptions::default()
    };
    let hits = tenant
        .core
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

/// Limits the memory routes share with the document routes, applied to the raw request before
/// the memory API sees it.
fn validate_memory_request(
    op: &str,
    request: &mut Value,
    state: &AppState,
) -> Result<(), ApiError> {
    let cli = &state.cli;
    let vector = |value: &Value, what: &str| -> Result<(), ApiError> {
        match value {
            Value::Null => Ok(()),
            Value::Array(items) => {
                let values: Option<Vec<f64>> = items.iter().map(Value::as_f64).collect();
                dense(
                    &values.ok_or_else(|| ApiError::bad(format!("{what} must be numbers")))?,
                    cli.input_dim,
                    what,
                )
            }
            _ => Err(ApiError::bad(format!("{what} must be an array of numbers"))),
        }
    };
    let fact = |fact: &Value| -> Result<(), ApiError> {
        let Value::Object(map) = fact else {
            return Err(ApiError::bad("a fact must be a JSON object"));
        };
        if let Some(Value::String(id)) = map.get("id") {
            if id.len() > MAX_ID_BYTES {
                return Err(ApiError::bad(format!("id exceeds {MAX_ID_BYTES} bytes")));
            }
        }
        if let Some(Value::String(text)) = map.get("text") {
            if text.len() > MAX_TEXT_BYTES {
                return Err(ApiError::bad(format!(
                    "text exceeds {MAX_TEXT_BYTES} bytes"
                )));
            }
        }
        if let Some(metadata) = map.get("metadata") {
            if serde_json::to_vec(metadata).map_or(true, |v| v.len() > MAX_METADATA_BYTES) {
                return Err(ApiError::bad(format!(
                    "metadata exceeds {MAX_METADATA_BYTES} bytes"
                )));
            }
        }
        map.get("embedding")
            .map_or(Ok(()), |v| vector(v, "embedding"))
    };
    match op {
        "remember" => match request.get("facts") {
            Some(Value::Array(facts)) => {
                if facts.is_empty() || facts.len() > cli.max_batch {
                    return Err(ApiError::bad(format!(
                        "facts must contain 1..={} entries",
                        cli.max_batch
                    )));
                }
                facts.iter().try_for_each(fact)
            }
            Some(_) => Err(ApiError::bad("facts must be an array")),
            None => fact(request),
        },
        "recall" | "explain" => {
            if let Some(Value::String(query)) = request.get("query") {
                if query.len() > 8192 {
                    return Err(ApiError::bad("query requires at most 8192 bytes"));
                }
            }
            if let Some(k) = request.get("k").and_then(Value::as_i64) {
                if k < 1 || k > i64::from(cli.max_top_k) {
                    return Err(ApiError::bad(format!("k must be 1..={}", cli.max_top_k)));
                }
            }
            request
                .get("embedding")
                .map_or(Ok(()), |v| vector(v, "embedding"))
        }
        _ => Ok(()),
    }
}

fn memory_error(error: anyhow::Error) -> ApiError {
    match error.downcast_ref::<MemoryError>() {
        Some(MemoryError::Invalid(message)) => ApiError::bad(message.clone()),
        Some(MemoryError::NotFound(message)) => {
            ApiError::new(StatusCode::NOT_FOUND, "not_found", message.clone())
        }
        Some(MemoryError::Conflict(message)) => {
            ApiError::new(StatusCode::CONFLICT, "conflict", message.clone())
        }
        None => ApiError::internal(error),
    }
}

async fn memory_op(
    State(state): State<Arc<AppState>>,
    UrlPath(op): UrlPath<String>,
    req: Request,
) -> ApiResult {
    if !OPERATIONS.contains(&op.as_str()) {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "no such memory operation",
        ));
    }
    let name = tenant_name(req.headers())?;
    let permit = state.work.acquire()?;
    let bytes = read_body(&state, req.into_body()).await?;
    let response = blocking(permit, move || {
        let mut request: Value = if bytes.is_empty() {
            json!({})
        } else {
            parse(&bytes)?
        };
        validate_memory_request(&op, &mut request, &state)?;
        let tenant = state.tenant(&name)?;
        tenant.memory.call(&op, request).map_err(memory_error)
    })
    .await?;
    Ok(Json(response).into_response())
}

async fn not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "not_found", "no such route")
}

async fn method_not_allowed() -> ApiError {
    ApiError::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        "method not allowed",
    )
}

fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/documents/batch", post(add_batch))
        .route("/api/v1/documents", post(add_one))
        .route("/api/v1/documents/{id}", delete(delete_doc))
        .route("/api/v1/query", post(query))
        .route("/api/v1/index/maintain", post(maintain_index))
        .route("/api/v1/memory/{op}", post(memory_op))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state)
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.dim < 256 || cli.input_dim == 0 || cli.input_dim > 65536 {
        bail!("--dim must be >= 256 and --input-dim 1..=65536");
    }
    if !(1..=MAX_CONCURRENT_WORK).contains(&cli.max_concurrent_work) {
        bail!("--max-concurrent-work must be 1..=256");
    }
    if cli.max_batch > MAX_BATCH_DOCUMENTS {
        bail!("--max-batch must not exceed the native 4096-document transaction limit");
    }
    if !(1..=300).contains(&cli.body_timeout_seconds) {
        bail!("--body-timeout-seconds must be 1..=300");
    }
    let api_key = match std::env::var("HMS_API_KEY") {
        Ok(k) if k.is_empty() => bail!("HMS_API_KEY is set but empty"),
        Ok(k) => Some(k.into_bytes()),
        Err(_) => None,
    };
    if api_key.is_none() && !cli.bind.ip().is_loopback() {
        eprintln!("warning: listening on a non-loopback address without HMS_API_KEY");
    }
    std::fs::create_dir_all(&cli.data_dir).context("cannot create --data-dir")?;
    let bind = cli.bind;
    #[cfg(feature = "local-models")]
    let stages = load_stages(&cli.models)?;
    let work = WorkLimit::new(cli.max_concurrent_work);
    load_memory_config(cli.memory_config.as_deref())?;
    let state = Arc::new(AppState {
        cli,
        api_key,
        tenants: Mutex::new(HashMap::new()),
        work,
        #[cfg(feature = "local-models")]
        stages,
    });
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("cannot bind {bind}"))?;
    println!("hms-server listening on {}", listener.local_addr()?);
    let shutdown_state = state.clone();
    let served = axum::serve(listener, router(state.clone()))
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            shutdown_state.work.stop();
        })
        .await;
    state.work.stop();
    let drained = state
        .work
        .slots
        .clone()
        .acquire_many_owned(state.cli.max_concurrent_work as u32)
        .await
        .context("cannot drain server work")?;
    tokio::task::spawn_blocking(move || -> Result<()> {
        let _drained = drained;
        let tenants: Vec<_> = state.tenants.lock().drain().collect();
        let mut failed = false;
        for (name, slot) in tenants {
            let tenant = match &*slot.status.lock() {
                TenantStatus::Ready(tenant) => tenant.clone(),
                _ => bail!("tenant initialization remained incomplete at shutdown"),
            };
            if let Err(error) = tenant.core.flush() {
                eprintln!("flush failed for tenant {name}: {error:#}");
                failed = true;
            }
        }
        if failed {
            bail!("shutdown flush failed");
        }
        Ok(())
    })
    .await
    .context("server shutdown worker failed")??;
    served?;
    Ok(())
}

#[cfg(test)]
#[path = "hms-server/concurrency_tests.rs"]
mod concurrency_tests;
