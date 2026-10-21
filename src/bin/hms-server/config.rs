// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Every runtime knob of `hms-server`: one flag, one `HMS_*` environment variable, one default.
//! `docs/server/configuration.md` lists them all; a test keeps the three in step.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};

pub const MAX_ID_BYTES: usize = 256;
/// One client vector describes one stored chunk, so a document must fit in one
/// chunk: the engine's per-chunk limits are 4096 words and 64 KiB.
pub const MAX_TEXT_BYTES: usize = 64 * 1024;
pub const CHUNK_WORDS: u32 = 4096;
pub const MAX_METADATA_BYTES: usize = 64 * 1024;
pub const MAX_TENANT_BYTES: usize = 64;
pub const MAX_BATCH_DOCUMENTS: usize = 4096;
pub const MAX_BATCH_SOURCE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_BATCH_METADATA_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_BATCH_EMBEDDING_VALUES: usize = 1_048_576;
pub const MAX_CONCURRENT_WORK: usize = 256;

#[derive(Parser)]
#[command(
    name = "hms-server",
    version,
    about = "HTTP service for the holographic-sdk client. See docs/SERVER.md and docs/server/.",
    long_about = "HTTP service for the holographic-sdk client.\n\nAuthentication is required unless --no-auth is given (loopback only). Credentials: HMS_API_KEY (one all-tenant key, environment only) and/or --keys-file (per-tenant hashed keys with scopes; create entries with `hms-server keygen`)."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
    #[command(flatten)]
    pub config: Config,
}

#[derive(Subcommand)]
pub enum Command {
    /// Print the OpenAPI 3.1 document for this build and exit.
    Openapi,
    /// Generate an API key. The token is printed once on stdout; only its SHA-256 digest is stored.
    Keygen(crate::auth::KeygenArgs),
    /// Probe a running server and exit 0 when it is ready (for container HEALTHCHECK, where the
    /// image has no shell or curl). Plain TCP or unix socket; with TLS it only checks the port.
    Healthcheck(HealthArgs),
}

#[derive(clap::Args)]
pub struct HealthArgs {
    /// Address of the server; an unspecified address (0.0.0.0) is probed as loopback.
    #[arg(long, env = "HMS_BIND", default_value = "127.0.0.1:8080")]
    pub bind: SocketAddr,
    #[arg(long, env = "HMS_UNIX_SOCKET")]
    pub unix_socket: Option<PathBuf>,
    /// Treat the server as TLS: only verify that the port accepts connections.
    #[arg(long, env = "HMS_TLS_CERT")]
    pub tls_cert: Option<PathBuf>,
    #[arg(long, default_value = "/readyz")]
    pub path: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    Json,
    Text,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

#[derive(clap::Args, Clone)]
pub struct Config {
    /// Listen address. Defaults to loopback only.
    #[arg(long, env = "HMS_BIND", default_value = "127.0.0.1:8080")]
    pub bind: SocketAddr,
    /// Listen on this unix domain socket instead of TCP (mode 0600). Takes precedence over --bind.
    #[arg(long, env = "HMS_UNIX_SOCKET")]
    pub unix_socket: Option<PathBuf>,
    /// PEM certificate chain: serve HTTPS (rustls, TLS 1.2 and 1.3). Reloaded on SIGHUP.
    #[arg(long, env = "HMS_TLS_CERT", requires = "tls_key")]
    pub tls_cert: Option<PathBuf>,
    /// PEM private key for --tls-cert.
    #[arg(long, env = "HMS_TLS_KEY", requires = "tls_cert")]
    pub tls_key: Option<PathBuf>,
    /// JSON file with the memory API configuration (policies and defaults) applied to every
    /// tenant; see docs/memory-api. Absent: the conservative defaults.
    #[arg(long, env = "HMS_MEMORY_CONFIG")]
    pub memory_config: Option<PathBuf>,
    /// Directory holding one store per tenant.
    #[arg(long, env = "HMS_DATA_DIR", default_value = "./hms-data")]
    pub data_dir: PathBuf,
    /// Hypervector dimensions of each store (>= 256).
    #[arg(long, env = "HMS_DIM", default_value_t = 4096)]
    pub dim: u32,
    /// Required length of client embeddings.
    #[arg(long, env = "HMS_INPUT_DIM", default_value_t = 384)]
    pub input_dim: usize,
    /// Embedding model identifier recorded in each store. Reopening a store with
    /// a different model, revision or input dimension is refused.
    #[arg(long, env = "HMS_EMBEDDING_MODEL", default_value = "client-supplied")]
    pub embedding_model: String,
    /// Embedding model revision recorded in each store.
    #[arg(long, env = "HMS_EMBEDDING_REVISION", default_value = "unversioned")]
    pub embedding_revision: String,

    /// Serve without authentication. Refused unless the listener is loopback or a unix socket.
    #[arg(long, env = "HMS_NO_AUTH")]
    pub no_auth: bool,
    /// JSON file of hashed per-tenant API keys. Reloaded on SIGHUP and `POST /api/v1/admin/keys/reload`.
    #[arg(long, env = "HMS_KEYS_FILE")]
    pub keys_file: Option<PathBuf>,
    /// Serve /metrics without authentication (otherwise it needs the admin scope).
    #[arg(long, env = "HMS_METRICS_PUBLIC")]
    pub metrics_public: bool,
    /// Comma-separated origins allowed by CORS (`*` for any). Empty disables CORS headers.
    #[arg(long, env = "HMS_CORS_ORIGINS", value_delimiter = ',')]
    pub cors_origins: Vec<String>,
    /// Seconds browsers may cache a CORS preflight.
    #[arg(long, env = "HMS_CORS_MAX_AGE_SECONDS", default_value_t = 600)]
    pub cors_max_age_seconds: u64,

    /// Maximum request body in bytes (endpoints other than ingest streaming and restore upload).
    #[arg(long, env = "HMS_MAX_BODY_BYTES", default_value_t = 16 * 1024 * 1024)]
    pub max_body_bytes: usize,
    /// Maximum time to read an admitted request body (also the idle gap allowed while streaming).
    #[arg(long, env = "HMS_BODY_TIMEOUT_SECONDS", default_value_t = 30)]
    pub body_timeout_seconds: u64,
    /// Maximum time to receive a request's headers before the connection is dropped.
    #[arg(long, env = "HMS_HEADER_READ_TIMEOUT_SECONDS", default_value_t = 10)]
    pub header_read_timeout_seconds: u64,
    /// Maximum simultaneous connections; further connections wait in the kernel backlog.
    #[arg(long, env = "HMS_MAX_CONNECTIONS", default_value_t = 1024)]
    pub max_connections: usize,
    /// Maximum documents per batch request.
    #[arg(long, env = "HMS_MAX_BATCH", default_value_t = 1000)]
    pub max_batch: usize,
    /// Maximum top_k.
    #[arg(long, env = "HMS_MAX_TOP_K", default_value_t = 100)]
    pub max_top_k: u32,
    /// Route of unfiltered vector queries: `auto` (the graph index once maintenance built one,
    /// the exact scan otherwise), `exact` (always the exact scan) or `graph` (409 until a graph
    /// index is built). A request's `index` field overrides it.
    #[arg(long, env = "HMS_QUERY_INDEX", default_value = "auto")]
    pub query_index: String,
    /// Beam width of graph-index queries, 1..=4096: the recall knob. A request's `ef` field
    /// overrides it.
    #[arg(long, env = "HMS_QUERY_EF", default_value_t = 128)]
    pub query_ef: u32,
    /// Fewest embedded chunks for which index maintenance builds the graph index; smaller
    /// stores answer from the exact scan, which is faster at that size.
    #[arg(long, env = "HMS_GRAPH_MIN_ROWS", default_value_t = 4096)]
    pub graph_min_rows: usize,
    /// When ingest computes chunk hypervectors (the text-vector channel of `HmsCore::query`):
    /// `eager` (at ingest, as before this flag), `deferred` (on the first vector query or
    /// index maintenance; needs stored text and a text encoder) or `lexical` (never: an
    /// embedding-only workload skips the channel's encoding time and memory). Text search and
    /// `query_vector` search see every document in every mode.
    #[arg(long, env = "HMS_INGEST_ENCODE", default_value = "eager")]
    pub ingest_encode: String,
    /// Maximum number of tenant stores open at once.
    #[arg(long, env = "HMS_MAX_TENANTS", default_value_t = 64)]
    pub max_tenants: usize,
    /// Close a tenant store idle for this many seconds (flushed, reopened on demand). 0 keeps stores open.
    #[arg(long, env = "HMS_TENANT_IDLE_SECONDS", default_value_t = 900)]
    pub tenant_idle_seconds: u64,
    /// Admitted requests reading bodies or executing blocking work.
    #[arg(long, env = "HMS_MAX_CONCURRENT_WORK", default_value_t = 4)]
    pub max_concurrent_work: usize,
    /// Requests allowed to wait for a work slot. 0 rejects immediately with 503.
    #[arg(long, env = "HMS_MAX_QUEUE", default_value_t = 0)]
    pub max_queue: usize,
    /// Longest a queued request waits for a work slot.
    #[arg(long, env = "HMS_QUEUE_TIMEOUT_MS", default_value_t = 1000)]
    pub queue_timeout_ms: u64,
    /// After SIGTERM/SIGINT, keep serving this long with /readyz failing so load balancers can
    /// drain the instance before the listener closes. 0 closes immediately.
    #[arg(long, env = "HMS_SHUTDOWN_DELAY_SECONDS", default_value_t = 0)]
    pub shutdown_delay_seconds: u64,
    /// Seconds in-flight requests may take to finish after SIGTERM/SIGINT before connections are cut.
    #[arg(long, env = "HMS_SHUTDOWN_GRACE_SECONDS", default_value_t = 30)]
    pub shutdown_grace_seconds: u64,

    /// Per-tenant sustained request rate (requests/second). 0 disables rate limiting.
    #[arg(long, env = "HMS_RATE_LIMIT_RPS", default_value_t = 0.0)]
    pub rate_limit_rps: f64,
    /// Per-tenant burst size. 0 means one second of the sustained rate.
    #[arg(long, env = "HMS_RATE_LIMIT_BURST", default_value_t = 0)]
    pub rate_limit_burst: u32,
    /// Per-tenant document quota; writes that would exceed it get 429. 0 disables it.
    #[arg(long, env = "HMS_MAX_DOCUMENTS_PER_TENANT", default_value_t = 0)]
    pub max_documents_per_tenant: u64,
    /// How long an Idempotency-Key replays its recorded response.
    #[arg(long, env = "HMS_IDEMPOTENCY_TTL_SECONDS", default_value_t = 600)]
    pub idempotency_ttl_seconds: u64,
    /// Idempotency records kept in memory (oldest evicted first).
    #[arg(long, env = "HMS_IDEMPOTENCY_CAPACITY", default_value_t = 10_000)]
    pub idempotency_capacity: usize,

    /// Largest NDJSON ingest stream.
    #[arg(long, env = "HMS_MAX_STREAM_BYTES", default_value_t = 1u64 << 30)]
    pub max_stream_bytes: u64,
    /// Documents committed per transaction while streaming NDJSON.
    #[arg(long, env = "HMS_STREAM_BATCH", default_value_t = 500)]
    pub stream_batch: usize,
    /// Largest line of an NDJSON stream.
    #[arg(long, env = "HMS_MAX_STREAM_LINE_BYTES", default_value_t = 4 * 1024 * 1024)]
    pub max_stream_line_bytes: usize,
    /// Largest snapshot archive accepted by restore upload.
    #[arg(long, env = "HMS_MAX_RESTORE_BYTES", default_value_t = 8u64 << 30)]
    pub max_restore_bytes: u64,
    /// Compact every open tenant store this often. 0 disables scheduled compaction.
    #[arg(long, env = "HMS_COMPACTION_INTERVAL_SECONDS", default_value_t = 0)]
    pub compaction_interval_seconds: u64,

    /// Log line format on stderr.
    #[arg(long, env = "HMS_LOG_FORMAT", value_enum, default_value_t = LogFormat::Json)]
    pub log_format: LogFormat,
    /// Minimum log level.
    #[arg(long, env = "HMS_LOG_LEVEL", value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,

    #[cfg(feature = "local-models")]
    #[command(flatten)]
    pub models: ModelArgs,
}

/// On-device model stages (feature `local-models`). Each directory holds safetensors and
/// tokenizer.json and must record the given revision; nothing is downloaded.
#[cfg(feature = "local-models")]
#[derive(clap::Args, Clone)]
pub struct ModelArgs {
    /// Qwen3-Embedding directory: documents without a vector and text queries are embedded here.
    #[arg(long, env = "HMS_EMBED_MODEL", requires = "embed_revision")]
    pub embed_model: Option<PathBuf>,
    #[arg(long, env = "HMS_EMBED_REVISION")]
    pub embed_revision: Option<String>,
    /// Qwen3-Reranker directory: re-ranks the fused head of text queries.
    #[arg(long, env = "HMS_RERANK_MODEL", requires = "rerank_revision")]
    pub rerank_model: Option<PathBuf>,
    #[arg(long, env = "HMS_RERANK_REVISION")]
    pub rerank_revision: Option<String>,
    /// Qwen3 instruct LLM directory for fact extraction and query rewriting.
    #[arg(long, env = "HMS_LLM_MODEL", requires = "llm_revision")]
    pub llm_model: Option<PathBuf>,
    #[arg(long, env = "HMS_LLM_REVISION")]
    pub llm_revision: Option<String>,
    /// Extract facts at ingest with the LLM and index them with each document.
    #[arg(long, env = "HMS_EXTRACT_FACTS", requires = "llm_model")]
    pub extract_facts: bool,
    /// Rewrite text queries with the LLM and search both forms.
    #[arg(long, env = "HMS_REWRITE_QUERIES", requires = "llm_model")]
    pub rewrite_queries: bool,
}

impl Config {
    pub fn validate(&self, credentials: bool) -> Result<()> {
        if self.dim < 256 || self.input_dim == 0 || self.input_dim > 65536 {
            bail!("--dim must be >= 256 and --input-dim 1..=65536");
        }
        if !(1..=MAX_CONCURRENT_WORK).contains(&self.max_concurrent_work) {
            bail!("--max-concurrent-work must be 1..=256");
        }
        if holographic_memory::core::documents::VectorIndexMode::parse(&self.query_index).is_none()
        {
            bail!("--query-index must be auto, exact or graph");
        }
        if !(1..=holographic_memory::core::documents::MAX_EF as u32).contains(&self.query_ef) {
            bail!("--query-ef must be 1..=4096");
        }
        if !(holographic_memory::core::documents::SNAPSHOT_MIN_ROWS..=(1 << 22))
            .contains(&self.graph_min_rows)
        {
            bail!("--graph-min-rows must be 128..=4194304");
        }
        if self
            .ingest_encode
            .parse::<holographic_memory::core::documents::EncodeMode>()
            .is_err()
        {
            bail!("--ingest-encode must be eager, deferred or lexical");
        }
        if self.max_batch == 0 || self.max_batch > MAX_BATCH_DOCUMENTS {
            bail!("--max-batch must be 1..=4096 (the native transaction limit)");
        }
        if !(1..=300).contains(&self.body_timeout_seconds) {
            bail!("--body-timeout-seconds must be 1..=300");
        }
        if !(1..=300).contains(&self.header_read_timeout_seconds) {
            bail!("--header-read-timeout-seconds must be 1..=300");
        }
        if self.max_connections == 0 || self.max_connections > 1_000_000 {
            bail!("--max-connections must be 1..=1000000");
        }
        if self.shutdown_delay_seconds > 300 || self.shutdown_grace_seconds > 3600 {
            bail!("--shutdown-delay-seconds must be <= 300 and --shutdown-grace-seconds <= 3600");
        }
        if self.max_queue > 100_000 || self.queue_timeout_ms > 600_000 {
            bail!("--max-queue must be <= 100000 and --queue-timeout-ms <= 600000");
        }
        if self.max_tenants == 0 || self.max_top_k == 0 || self.max_body_bytes == 0 {
            bail!("--max-tenants, --max-top-k and --max-body-bytes must be positive");
        }
        if !self.rate_limit_rps.is_finite() || self.rate_limit_rps < 0.0 {
            bail!("--rate-limit-rps must be a finite number >= 0");
        }
        if self.stream_batch == 0 || self.stream_batch > MAX_BATCH_DOCUMENTS {
            bail!("--stream-batch must be 1..=4096");
        }
        if self.max_stream_line_bytes < 1024 || self.max_stream_line_bytes > 64 * 1024 * 1024 {
            bail!("--max-stream-line-bytes must be 1 KiB..=64 MiB");
        }
        if self.idempotency_capacity == 0 {
            bail!("--idempotency-capacity must be positive");
        }
        if self
            .cors_origins
            .iter()
            .any(|o| o.is_empty() || o.contains(char::is_whitespace))
        {
            bail!("--cors-origins entries must be non-empty origins without spaces");
        }
        if !credentials {
            if !self.no_auth {
                bail!("no credentials configured: set HMS_API_KEY or --keys-file, or pass --no-auth for unauthenticated loopback development");
            }
            if self.unix_socket.is_none() && !self.bind.ip().is_loopback() {
                bail!("--no-auth is only allowed on a loopback address or a unix socket");
            }
        } else if self.no_auth {
            bail!("--no-auth conflicts with configured credentials (HMS_API_KEY or --keys-file)");
        }
        Ok(())
    }
}
