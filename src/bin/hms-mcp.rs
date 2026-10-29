// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! `hms-mcp`: a Model Context Protocol server for Holographic Memory.
//!
//! Tools: remember, recall, search, get, forget, list, stats, export. Resources expose stored
//! memories; prompts wrap recall and review. Transports: stdio (default) and streamable HTTP.

#[path = "hms-mcp/config.rs"]
mod config;
#[path = "hms-mcp/http.rs"]
mod http_server;
#[path = "hms-mcp/log.rs"]
mod log;
#[path = "hms-mcp/memory.rs"]
mod memory;
#[path = "hms-mcp/server.rs"]
mod server;
#[path = "hms-mcp/session.rs"]
mod session;
#[path = "hms-mcp/stdio.rs"]
mod stdio_server;
#[path = "hms-mcp/store.rs"]
mod store;
#[path = "hms-mcp/tools.rs"]
mod tools;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use clap::Parser;
use rmcp::ServiceExt;
use subtle::ConstantTimeEq;

use config::{Cli, NamespaceStrategy, Transport};
use memory::Limits;
use store::{HmsBackend, StoreOptions, StoreRoot};

/// Nearest ancestor containing `.git`, else the directory itself.
fn project_root(start: &Path) -> PathBuf {
    let start = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    start
        .ancestors()
        .find(|dir| dir.join(".git").exists())
        .map_or(start.clone(), Path::to_path_buf)
}

/// A stable namespace for a project directory: a readable slug plus a hash of its path.
fn project_namespace(root: &Path) -> String {
    let name = root
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let slug: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .chars()
        .take(40)
        .collect();
    let slug = if slug.is_empty() {
        "project".into()
    } else {
        slug
    };
    format!(
        "{slug}-{:08x}",
        path_hash(root.to_string_lossy().as_bytes()) as u32
    )
}

/// 64-bit Fx hash of the path bytes followed by the 0xff terminator `str::hash` appends.
/// Namespaces already on disk carry this value, so the algorithm must not change.
fn path_hash(bytes: &[u8]) -> u64 {
    const SEED: u64 = 0x517c_c1b7_2722_0a95;
    let mix = |hash: u64, word: u64| (hash.rotate_left(5) ^ word).wrapping_mul(SEED);

    let (words, rest) = bytes.as_chunks::<8>();
    let mut hash = 0;
    for word in words {
        hash = mix(hash, u64::from_ne_bytes(*word));
    }
    let mut rest = rest;
    if let Some((word, tail)) = rest.split_first_chunk::<4>() {
        hash = mix(hash, u64::from(u32::from_ne_bytes(*word)));
        rest = tail;
    }
    for &byte in rest {
        hash = mix(hash, u64::from(byte));
    }
    mix(hash, 0xff)
}

fn default_namespace(cli: &Cli) -> Result<String> {
    Ok(match cli.namespace_strategy {
        NamespaceStrategy::Global => cli.namespace.clone(),
        NamespaceStrategy::Project => {
            let start = match &cli.project {
                Some(p) => p.clone(),
                None => std::env::current_dir().context("cannot read the working directory")?,
            };
            if !start.is_dir() {
                bail!("--project {} is not a directory", start.display());
            }
            project_namespace(&project_root(&start))
        }
    })
}

#[cfg(feature = "local-models")]
fn load_stages(
    cli: &Cli,
) -> Result<(
    Option<Arc<holographic_memory::core::models::ModelStages>>,
    Option<usize>,
)> {
    use holographic_memory::core::models::{default_device, Embedder, ModelSource, ModelStages};
    let (Some(dir), Some(rev)) = (&cli.models.embed_model, &cli.models.embed_revision) else {
        return Ok((None, None));
    };
    let device = default_device()?;
    let embedder = Embedder::load(&ModelSource::new(dir, rev.clone()), &device)?;
    let dim = embedder.dimensions();
    let stages = ModelStages {
        embedder: Some(embedder),
        reranker: None,
        generator: None,
        extract_facts: false,
        rewrite_queries: false,
        params: Default::default(),
    };
    Ok((Some(Arc::new(stages)), Some(dim)))
}

fn limits(cli: &Cli) -> Limits {
    Limits {
        max_text_bytes: cli.max_text_bytes,
        max_metadata_bytes: cli.max_metadata_bytes,
        max_tags: cli.max_tags,
        max_top_k: cli.max_top_k,
        max_page: cli.max_page,
        max_export_bytes: cli.max_export_bytes,
        max_forget: cli.max_forget,
    }
}

async fn bearer(State(token): State<Arc<String>>, req: Request, next: Next) -> Response {
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    // Constant-time over the bytes; the token length is not treated as secret.
    let ok =
        presented.len() == token.len() && bool::from(presented.as_bytes().ct_eq(token.as_bytes()));
    if ok {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "unauthorized",
        )
            .into_response()
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
}

async fn serve_http(cli: &Cli, handler: server::HmsMcp) -> Result<()> {
    http_server::serve(cli, handler).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    log::set_level(&cli.log).map_err(anyhow::Error::msg)?;
    cli.validate()?;
    let exposure = tools::Exposure::new(
        &cli.tools,
        cli.read_only,
        cli.namespace_arg,
        default_namespace(&cli)?,
    )
    .map_err(anyhow::Error::msg)?;
    let root = match &cli.store {
        Some(dir) => StoreRoot::Single(dir.clone()),
        None => StoreRoot::Namespaced(cli.resolved_data_dir()?),
    };
    #[cfg(feature = "local-models")]
    let (stages, embedding_dim) = load_stages(&cli)?;
    let backend = Arc::new(HmsBackend::new(
        StoreOptions {
            root,
            dim: cli.dim,
            max_stores: cli.max_stores,
            encrypt: cli.encrypt.then(|| cli.passphrase_env.clone()),
            #[cfg(feature = "local-models")]
            stages,
            #[cfg(feature = "local-models")]
            embedding_dim,
        },
        limits(&cli),
    ));
    // Open the default store now so a held lock or a wrong passphrase fails at startup.
    backend
        .preopen(&exposure.default_namespace)
        .map_err(|f| anyhow::anyhow!("{}: {}", f.code, f.message))?;
    log::info(&format!(
        "namespace {} ready; {} tools enabled",
        exposure.default_namespace,
        exposure.enabled.len()
    ));
    let handler = server::HmsMcp::new(
        backend.clone(),
        exposure,
        limits(&cli),
        !cli.no_resources,
        !cli.no_prompts,
        server::WorkLimits {
            concurrent: cli.max_concurrent,
            pending: cli.max_pending,
            queue_timeout: std::time::Duration::from_millis(cli.queue_timeout_ms),
        },
    );
    let result = match cli.transport {
        Transport::Stdio => {
            let (read, write) = rmcp::transport::stdio();
            let running = handler
                .serve(stdio_server::bounded(read, write, cli.max_request_bytes))
                .await
                .context("stdio session failed to start")?;
            tokio::select! {
                r = running.waiting() => r.map(|_| ()).map_err(anyhow::Error::from),
                () = shutdown_signal() => Ok(()),
            }
        }
        Transport::Http => serve_http(&cli, handler).await,
    };
    // Every admitted request has finished (graceful shutdown), so flushing is safe.
    backend.flush_all();
    result
}

#[cfg(test)]
mod namespace_tests {
    use super::project_namespace;
    use std::path::Path;

    #[test]
    fn project_namespace_keeps_existing_hash_suffix() {
        assert_eq!(
            project_namespace(Path::new("/Users/x/proj")),
            "proj-3b82364d"
        );
    }
}
