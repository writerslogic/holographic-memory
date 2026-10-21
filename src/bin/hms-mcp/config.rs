// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Command line and environment configuration. Every policy has a default and a knob.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Transport {
    Stdio,
    Http,
}

/// How a call finds its store when the client does not choose one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum NamespaceStrategy {
    /// One store named by `--namespace` (default `default`).
    Global,
    /// One store per project directory (`--project`, default the working directory's git root).
    Project,
}

#[derive(Parser, Debug)]
#[command(
    name = "hms-mcp",
    version,
    about = "MCP server for Holographic Memory: remember, recall, search, forget, list, stats and export over stdio or streamable HTTP"
)]
pub struct Cli {
    /// Transport: `stdio` (one client per process) or `http` (streamable HTTP, shared by many clients).
    #[arg(long, env = "HMS_MCP_TRANSPORT", value_enum, default_value = "stdio")]
    pub transport: Transport,
    /// Listen address for `--transport http`. Loopback unless a token is set and
    /// `--allow-remote` is given.
    #[arg(long, env = "HMS_MCP_BIND", default_value = "127.0.0.1:8765")]
    pub bind: SocketAddr,
    /// URL path of the MCP endpoint.
    #[arg(long, env = "HMS_MCP_PATH", default_value = "/mcp")]
    pub path: String,
    /// Extra `Host` values the HTTP endpoint accepts (default: loopback names only).
    #[arg(
        long = "allow-host",
        env = "HMS_MCP_ALLOW_HOSTS",
        value_delimiter = ','
    )]
    pub allow_hosts: Vec<String>,
    /// Browser origins the HTTP endpoint accepts, for example `https://app.example.com:443`.
    /// Requests without an `Origin` header (non-browser clients) always pass; a present
    /// `Origin` must match this list, which is empty by default.
    #[arg(
        long = "allow-origin",
        env = "HMS_MCP_ALLOW_ORIGINS",
        value_delimiter = ','
    )]
    pub allow_origins: Vec<String>,
    /// Permit binding a non-loopback address (requires `HMS_MCP_TOKEN`).
    #[arg(long, env = "HMS_MCP_ALLOW_REMOTE")]
    pub allow_remote: bool,
    /// Largest HTTP request body.
    #[arg(long, env = "HMS_MCP_MAX_REQUEST_BYTES", default_value_t = 1 << 20)]
    pub max_request_bytes: usize,

    /// Root directory for stores. Default: the platform data directory, `hms-mcp` inside it.
    #[arg(long, env = "HMS_MCP_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
    /// Use exactly this directory as the single store (no namespacing).
    #[arg(long, env = "HMS_MCP_STORE", conflicts_with_all = ["data_dir", "namespace_arg"])]
    pub store: Option<PathBuf>,
    /// How the default namespace is chosen.
    #[arg(
        long,
        env = "HMS_MCP_NAMESPACE_STRATEGY",
        value_enum,
        default_value = "global"
    )]
    pub namespace_strategy: NamespaceStrategy,
    /// Namespace name for the `global` strategy: 1 to 64 characters of `[a-z0-9_-]`.
    #[arg(long, env = "HMS_MCP_NAMESPACE", default_value = "default")]
    pub namespace: String,
    /// Project directory for the `project` strategy (its git root is used when it has one).
    #[arg(long, env = "HMS_MCP_PROJECT")]
    pub project: Option<PathBuf>,
    /// Let each tool call pass a `namespace` argument selecting another store.
    #[arg(long, env = "HMS_MCP_NAMESPACE_ARG")]
    pub namespace_arg: bool,
    /// Stores held open at once (each holds its lock while open).
    #[arg(long, env = "HMS_MCP_MAX_STORES", default_value_t = 16)]
    pub max_stores: usize,
    /// Hypervector dimensions per store (>= 256). Fixed once a store exists.
    #[arg(long, env = "HMS_MCP_DIM", default_value_t = 4096)]
    pub dim: u32,

    /// Encrypt stores at rest (needs a build with the `security` feature).
    #[arg(long, env = "HMS_MCP_ENCRYPT")]
    pub encrypt: bool,
    /// Name of the environment variable holding the encryption passphrase.
    #[arg(
        long,
        env = "HMS_MCP_PASSPHRASE_ENV",
        default_value = "HMS_MCP_PASSPHRASE"
    )]
    pub passphrase_env: String,

    /// Tools to expose, comma separated. Default: all.
    #[arg(long, env = "HMS_MCP_TOOLS", value_delimiter = ',')]
    pub tools: Vec<String>,
    /// Expose only tools that do not modify the store.
    #[arg(long, env = "HMS_MCP_READ_ONLY")]
    pub read_only: bool,
    /// Do not expose memories as MCP resources.
    #[arg(long, env = "HMS_MCP_NO_RESOURCES")]
    pub no_resources: bool,
    /// Do not expose MCP prompts.
    #[arg(long, env = "HMS_MCP_NO_PROMPTS")]
    pub no_prompts: bool,

    /// Largest memory text in bytes (at most 65536; also at most 4096 words).
    #[arg(long, env = "HMS_MCP_MAX_TEXT_BYTES", default_value_t = 32 * 1024)]
    pub max_text_bytes: usize,
    /// Largest metadata object in bytes.
    #[arg(long, env = "HMS_MCP_MAX_METADATA_BYTES", default_value_t = 16 * 1024)]
    pub max_metadata_bytes: usize,
    /// Most tags per memory.
    #[arg(long, env = "HMS_MCP_MAX_TAGS", default_value_t = 32)]
    pub max_tags: usize,
    /// Upper bound for `k` in recall and search.
    #[arg(long, env = "HMS_MCP_MAX_TOP_K", default_value_t = 50)]
    pub max_top_k: usize,
    /// Upper bound for a page of `list`.
    #[arg(long, env = "HMS_MCP_MAX_PAGE", default_value_t = 200)]
    pub max_page: usize,
    /// Largest export payload in one call, in bytes; larger exports page.
    #[arg(long, env = "HMS_MCP_MAX_EXPORT_BYTES", default_value_t = 4 << 20)]
    pub max_export_bytes: usize,
    /// Most memories one `forget` call may remove.
    #[arg(long, env = "HMS_MCP_MAX_FORGET", default_value_t = 1000)]
    pub max_forget: usize,
    /// Tool calls running at once; further calls wait.
    #[arg(long, env = "HMS_MCP_MAX_CONCURRENT", default_value_t = 4)]
    pub max_concurrent: usize,
    /// Log level on stderr: error, warn, info or debug. Logs never contain memory text.
    #[arg(long, env = "HMS_MCP_LOG", default_value = "warn")]
    pub log: String,

    #[cfg(feature = "local-models")]
    #[command(flatten)]
    pub models: ModelArgs,
}

#[cfg(feature = "local-models")]
#[derive(clap::Args, Debug)]
pub struct ModelArgs {
    /// Qwen3-Embedding directory: memories and queries are embedded here, which makes recall semantic.
    #[arg(long, env = "HMS_EMBED_MODEL", requires = "embed_revision")]
    pub embed_model: Option<PathBuf>,
    #[arg(long, env = "HMS_EMBED_REVISION")]
    pub embed_revision: Option<String>,
}

pub const HARD_MAX_TEXT_BYTES: usize = 64 * 1024;

pub fn valid_namespace(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

impl Cli {
    pub fn validate(&self) -> Result<()> {
        if !valid_namespace(&self.namespace) {
            bail!("--namespace must be 1 to 64 characters of [a-z0-9_-]");
        }
        if !self.path.starts_with('/') || self.path.len() < 2 || self.path.contains("//") {
            bail!("--path must start with / and name a sub-path such as /mcp");
        }
        if !(256..=1_000_000).contains(&self.dim) {
            bail!("--dim must be 256..=1000000");
        }
        if !(1..=HARD_MAX_TEXT_BYTES).contains(&self.max_text_bytes) {
            bail!("--max-text-bytes must be 1..={HARD_MAX_TEXT_BYTES}");
        }
        for (flag, value, max) in [
            ("--max-top-k", self.max_top_k, 1000),
            ("--max-page", self.max_page, 10_000),
            ("--max-stores", self.max_stores, 1024),
            ("--max-concurrent", self.max_concurrent, 256),
            ("--max-tags", self.max_tags, 1024),
            ("--max-forget", self.max_forget, 100_000),
        ] {
            if !(1..=max).contains(&value) {
                bail!("{flag} must be 1..={max}");
            }
        }
        if self.max_metadata_bytes == 0 || self.max_export_bytes < 1024 {
            bail!("--max-metadata-bytes must be positive and --max-export-bytes at least 1024");
        }
        if self.max_request_bytes < 1024 {
            bail!("--max-request-bytes must be at least 1024");
        }
        if self.encrypt {
            if !cfg!(feature = "security") {
                bail!("--encrypt needs a build with the `security` feature (build with --features mcp,security)");
            }
            let present = std::env::var(&self.passphrase_env).is_ok_and(|v| !v.is_empty());
            if !present {
                bail!(
                    "--encrypt needs the passphrase in the environment variable {}",
                    self.passphrase_env
                );
            }
        }
        if self.transport == Transport::Http {
            let loopback = self.bind.ip().is_loopback();
            let token = http_token()?.is_some();
            if !(loopback || (self.allow_remote && token)) {
                bail!(
                    "binding {} needs --allow-remote and HMS_MCP_TOKEN set",
                    self.bind
                );
            }
        }
        Ok(())
    }

    /// The directory that holds all stores.
    pub fn resolved_data_dir(&self) -> Result<PathBuf> {
        if let Some(dir) = &self.data_dir {
            return Ok(dir.clone());
        }
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let base = if cfg!(target_os = "macos") {
            home.map(|h| h.join("Library/Application Support"))
        } else if cfg!(windows) {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| home.map(|h| h.join(".local/share")))
        };
        Ok(base
            .context("cannot locate a data directory; set --data-dir or HMS_MCP_DATA_DIR")?
            .join("hms-mcp"))
    }
}

/// The optional bearer token for the HTTP transport. Environment only, never a flag.
pub fn http_token() -> Result<Option<String>> {
    match std::env::var("HMS_MCP_TOKEN") {
        Ok(t) if t.is_empty() => bail!("HMS_MCP_TOKEN is set but empty"),
        Ok(t) if t.len() < 16 => bail!("HMS_MCP_TOKEN must be at least 16 characters"),
        Ok(t) => Ok(Some(t)),
        Err(_) => Ok(None),
    }
}
