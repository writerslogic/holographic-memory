// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! HTTP service for the `holographic-sdk` Python client.
//!
//! Endpoints: `POST /api/v1/documents/batch`, `POST /api/v1/documents`,
//! `DELETE /api/v1/documents/{id}`, `POST /api/v1/query`. See docs/SERVER.md.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
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
use holographic_memory::{EntangledHVec, HmsCore};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use subtle::ConstantTimeEq;

const MAX_ID_BYTES: usize = 256;
const MAX_TEXT_BYTES: usize = 1 << 20;
const MAX_METADATA_BYTES: usize = 64 * 1024;
const MAX_TENANT_BYTES: usize = 64;
/// Engine query ceiling; filtered queries rank this many candidates first.
const FILTER_WINDOW: u32 = 10_000;

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
}

#[derive(Clone, Serialize, Deserialize)]
struct Entry {
    text: Option<String>,
    metadata: Value,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum LogLine {
    Put {
        id: String,
        text: Option<String>,
        metadata: Value,
    },
    Del {
        id: String,
    },
}

/// Text and metadata sidecar: an append-only JSONL log replayed (and compacted) on open.
struct MetaStore {
    entries: HashMap<String, Entry>,
    log: BufWriter<File>,
}

impl MetaStore {
    fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("meta.jsonl");
        let mut entries = HashMap::new();
        if path.exists() {
            for line in BufReader::new(File::open(&path)?).lines() {
                let line = line?;
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<LogLine>(&line) {
                    Ok(LogLine::Put { id, text, metadata }) => {
                        entries.insert(id, Entry { text, metadata });
                    }
                    Ok(LogLine::Del { id }) => {
                        entries.remove(&id);
                    }
                    // A torn final line from a crash; earlier lines are intact.
                    Err(_) => break,
                }
            }
        }
        let tmp = dir.join("meta.jsonl.tmp");
        {
            let mut out = BufWriter::new(File::create(&tmp)?);
            for (id, e) in &entries {
                serde_json::to_writer(
                    &mut out,
                    &LogLine::Put {
                        id: id.clone(),
                        text: e.text.clone(),
                        metadata: e.metadata.clone(),
                    },
                )?;
                out.write_all(b"\n")?;
            }
            out.flush()?;
            out.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        let log = BufWriter::new(OpenOptions::new().append(true).open(&path)?);
        Ok(Self { entries, log })
    }

    fn append(&mut self, line: &LogLine) -> Result<()> {
        serde_json::to_writer(&mut self.log, line)?;
        self.log.write_all(b"\n")?;
        self.log.flush()?;
        Ok(())
    }

    fn put(&mut self, id: String, entry: Entry) -> Result<()> {
        self.append(&LogLine::Put {
            id: id.clone(),
            text: entry.text.clone(),
            metadata: entry.metadata.clone(),
        })?;
        self.entries.insert(id, entry);
        Ok(())
    }

    fn del(&mut self, id: &str) -> Result<bool> {
        if !self.entries.contains_key(id) {
            return Ok(false);
        }
        self.append(&LogLine::Del { id: id.into() })?;
        self.entries.remove(id);
        Ok(true)
    }
}

struct Tenant {
    core: HmsCore,
    meta: Mutex<MetaStore>,
}

struct AppState {
    cli: Cli,
    api_key: Option<Vec<u8>>,
    tenants: Mutex<HashMap<String, Arc<Tenant>>>,
}

impl AppState {
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
            let core = HmsCore::new(self.cli.dim, Some(dir.display().to_string()), None)?;
            let meta = Mutex::new(MetaStore::open(&dir)?);
            Ok(Tenant { core, meta })
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
    id: String,
    vector: EntangledHVec,
    entry: Entry,
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

fn validate_doc(doc: DocIn, cli: &Cli) -> Result<ValidDoc, ApiError> {
    if doc.id.is_empty()
        || doc.id.len() > MAX_ID_BYTES
        || doc.id.chars().any(char::is_control)
        || doc.id.starts_with("hms:")
    {
        return Err(ApiError::bad(format!(
            "id must be 1..={MAX_ID_BYTES} bytes, without control characters or the reserved \"hms:\" prefix"
        )));
    }
    if doc.text.as_ref().is_some_and(|t| t.len() > MAX_TEXT_BYTES) {
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
    let values = doc.vector.unwrap_or_default();
    let dense = dense(&values, cli.input_dim, "vector")?;
    Ok(ValidDoc {
        id: doc.id,
        vector: EntangledHVec::from_dense(&dense, cli.dim as usize),
        entry: Entry {
            text: doc.text,
            metadata,
        },
    })
}

fn store_docs(tenant: &Tenant, docs: Vec<ValidDoc>) -> Result<usize, ApiError> {
    let n = docs.len();
    let mut meta = tenant.meta.lock();
    for doc in docs {
        meta.put(doc.id.clone(), doc.entry)
            .map_err(ApiError::internal)?;
        tenant
            .core
            .memorize(doc.id, doc.vector)
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
            .map(|d| validate_doc(d, &state.cli))
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
        let valid = validate_doc(doc, &state.cli)?;
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
        let mut meta = tenant.meta.lock();
        let in_engine = tenant.core.delete(&id).map_err(ApiError::internal)?;
        let in_meta = meta.del(&id).map_err(ApiError::internal)?;
        if in_engine || in_meta {
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
    query_vector: Vec<f64>,
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
        let dense = dense(&q.query_vector, state.cli.input_dim, "query_vector")?;
        let tenant = state.tenant(&name)?;
        let vector = EntangledHVec::from_dense(&dense, state.cli.dim as usize);
        let fetch = if filter.is_some() {
            FILTER_WINDOW
        } else {
            top_k as u32
        };
        let hits = tenant.core.query(&vector, fetch);
        let meta = tenant.meta.lock();
        let mut out = Vec::new();
        for hit in hits {
            let entry = meta.entries.get(&hit.id);
            if let Some(f) = &filter {
                let ok = entry.is_some_and(|e| f.iter().all(|(k, v)| e.metadata.get(k) == Some(v)));
                if !ok {
                    continue;
                }
            }
            let score = if hit.similarity.is_finite() {
                hit.similarity
            } else {
                0.0
            };
            out.push(json!({
                "id": hit.id,
                "text": entry.and_then(|e| e.text.clone()).unwrap_or_default(),
                "metadata": entry.map_or_else(|| json!({}), |e| e.metadata.clone()),
                "score": score,
            }));
            if out.len() == top_k as usize {
                break;
            }
        }
        Ok(out)
    })
    .await?;
    Ok(Json(json!({"matches": matches})).into_response())
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
    let state = Arc::new(AppState {
        cli,
        api_key,
        tenants: Mutex::new(HashMap::new()),
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
        if let Err(e) = tenant
            .meta
            .lock()
            .log
            .get_ref()
            .sync_all()
            .map_err(anyhow::Error::from)
        {
            eprintln!("metadata sync failed for tenant {name}: {e:#}");
            failed = true;
        }
    }
    if failed {
        bail!("shutdown flush failed");
    }
    Ok(())
}
