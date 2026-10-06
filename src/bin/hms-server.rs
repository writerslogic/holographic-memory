// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! HTTP service for the `holographic-sdk` Python client.
//!
//! Endpoints: `POST /api/v1/documents/batch`, `POST /api/v1/documents`,
//! `DELETE /api/v1/documents/{id}`, `POST /api/v1/query`. See docs/SERVER.md.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::body::{to_bytes, Body};
use axum::extract::{Path as UrlPath, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, post};
use axum::{Json, Router};
use clap::Parser;
use holographic_memory::core::HmsConfig;
use holographic_memory::{chunk_document, DocumentInput, EmbeddingSpace, HmsCore, SearchOptions};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};
use subtle::ConstantTimeEq;

const MAX_ID_BYTES: usize = 256;
/// One client vector describes one stored chunk, so a document must fit in one
/// chunk: the engine's per-chunk limits are 4096 words and 64 KiB.
const MAX_TEXT_BYTES: usize = 64 * 1024;
const CHUNK_WORDS: u32 = 4096;
const MAX_METADATA_BYTES: usize = 64 * 1024;
const MAX_TENANT_BYTES: usize = 64;

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
    /// Maximum documents per batch request.
    #[arg(long, default_value_t = 1000)]
    max_batch: usize,
    /// Maximum top_k.
    #[arg(long, default_value_t = 100)]
    max_top_k: u32,
    /// Maximum number of tenant stores open at once.
    #[arg(long, default_value_t = 64)]
    max_tenants: usize,
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

struct Tenant {
    core: HmsCore,
}

struct AppState {
    cli: Cli,
    api_key: Option<Vec<u8>>,
    tenants: Mutex<HashMap<String, Arc<Tenant>>>,
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
        let mut tenants = self.tenants.lock();
        if let Some(t) = tenants.get(name) {
            return Ok(t.clone());
        }
        if tenants.len() >= self.cli.max_tenants {
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "tenant_limit",
                "too many tenants are open on this server",
            ));
        }
        // Hex keeps the directory name safe and distinct on case-insensitive filesystems.
        let hex: String = name.bytes().map(|b| format!("{b:02x}")).collect();
        let dir = self.cli.data_dir.join("tenants").join(hex);
        let open = || -> Result<Tenant> {
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
            Ok(Tenant { core })
        };
        let tenant = Arc::new(open().map_err(ApiError::internal)?);
        tenants.insert(name.to_string(), tenant.clone());
        Ok(tenant)
    }
}

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
        (
            self.status,
            Json(json!({"error": {"code": self.code, "message": self.message}})),
        )
            .into_response()
    }
}

type ApiResult = Result<Response, ApiError>;

async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(f)
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
    to_bytes(body, state.cli.max_body_bytes)
        .await
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

fn dense(values: &[f64], input_dim: usize, what: &str) -> Result<Vec<f32>, ApiError> {
    if values.len() != input_dim {
        return Err(ApiError::bad(format!(
            "{what} must have exactly {input_dim} values (got {})",
            values.len()
        )));
    }
    let out: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    if !out.iter().all(|v| v.is_finite()) {
        return Err(ApiError::bad(format!("{what} values must be finite")));
    }
    if out.iter().all(|&v| v == 0.0) {
        return Err(ApiError::bad(format!("{what} must not be all zeros")));
    }
    Ok(out)
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
    for doc in docs {
        tenant
            .core
            .memorize_document(doc.input)
            .map_err(ApiError::internal)?;
    }
    Ok(n)
}

async fn add_batch(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let name = tenant_name(req.headers())?;
    let bytes = read_body(&state, req.into_body()).await?;
    let docs: Vec<DocIn> = parse(&bytes)?;
    if docs.is_empty() || docs.len() > state.cli.max_batch {
        return Err(ApiError::bad(format!(
            "batch must contain 1..={} documents",
            state.cli.max_batch
        )));
    }
    let n = blocking(move || {
        let valid = docs
            .into_iter()
            .map(|d| validate_doc(d, &state.cli, state.server_embeds()))
            .collect::<Result<Vec<_>, _>>()?;
        let tenant = state.tenant(&name)?;
        store_docs(&tenant, valid)
    })
    .await?;
    Ok(Json(json!({"added": n})).into_response())
}

async fn add_one(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let name = tenant_name(req.headers())?;
    let bytes = read_body(&state, req.into_body()).await?;
    let doc: DocIn = parse(&bytes)?;
    let n = blocking(move || {
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
    blocking(move || {
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
    today: Option<String>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    filter: Option<Value>,
}

async fn query(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let name = tenant_name(req.headers())?;
    let bytes = read_body(&state, req.into_body()).await?;
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
    let matches = blocking(move || {
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

fn vector_search(
    tenant: &Tenant,
    query_vector: Vec<f64>,
    top_k: u32,
    filter: Option<serde_json::Map<String, Value>>,
) -> Result<Vec<holographic_memory::DocumentResult>, ApiError> {
    // Exact cosine over every stored embedding (a linear scan); no lexical term.
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
    let state = Arc::new(AppState {
        cli,
        api_key,
        tenants: Mutex::new(HashMap::new()),
        #[cfg(feature = "local-models")]
        stages,
    });
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("cannot bind {bind}"))?;
    println!("hms-server listening on {}", listener.local_addr()?);
    axum::serve(listener, router(state.clone()))
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    let tenants: Vec<_> = state.tenants.lock().drain().collect();
    let mut failed = false;
    for (name, tenant) in tenants {
        if let Err(e) = tenant.core.flush() {
            eprintln!("flush failed for tenant {name}: {e:#}");
            failed = true;
        }
    }
    if failed {
        bail!("shutdown flush failed");
    }
    Ok(())
}
