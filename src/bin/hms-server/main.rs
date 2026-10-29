// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! HTTP service for the `holographic-memory-sdk` Python client and any HTTP client.
//!
//! The API contract is in `docs/server/openapi.json` (also served at `/openapi.json`); operation
//! and configuration docs are in `docs/SERVER.md` and `docs/server/`.

mod admin;
mod auth;
mod config;
mod documents;
mod error;
#[cfg(test)]
mod import_tests;
mod limits;
mod log;
mod memory;
mod metrics;
mod middleware;
mod ops;
mod request;
mod restore;
mod routes;
mod serve;
mod tenants;
mod types;
mod vindex;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use config::{Cli, Command, Config};
use tenants::AppState;

#[cfg(feature = "local-models")]
fn load_stages(
    args: &config::ModelArgs,
) -> Result<Option<Arc<holographic_memory::core::models::ModelStages>>> {
    use holographic_memory::core::models::{
        default_device, Embedder, Generator, ModelSource, ModelStages, Reranker,
    };
    use std::path::PathBuf;
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

/// SIGHUP re-reads the key file and the TLS certificate.
#[cfg(unix)]
fn spawn_reload_handler(state: Arc<AppState>, tls: Option<Arc<serve::Tls>>) {
    use tokio::signal::unix::{signal, SignalKind};
    let Ok(mut hup) = signal(SignalKind::hangup()) else {
        return;
    };
    tokio::spawn(async move {
        while hup.recv().await.is_some() {
            if state.cfg.keys_file.is_some() {
                match state.auth.reload() {
                    Ok(keys) => log_event!(Info, "keys_reloaded", "keys" => keys),
                    Err(e) => log_event!(Error, "keys_reload_failed", "detail" => format!("{e:#}")),
                }
            }
            if let Some(tls) = &tls {
                match tls.reload() {
                    Ok(()) => log_event!(Info, "tls_reloaded"),
                    Err(e) => log_event!(Error, "tls_reload_failed", "detail" => format!("{e:#}")),
                }
            }
        }
    });
}

fn spawn_background(state: &Arc<AppState>) -> Vec<tokio::task::JoinHandle<()>> {
    let mut tasks = Vec::new();
    if state.cfg.tenant_idle_seconds > 0 {
        let state = state.clone();
        let every = Duration::from_secs((state.cfg.tenant_idle_seconds / 4).clamp(1, 30));
        tasks.push(tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let state = state.clone();
                let _ = tokio::task::spawn_blocking(move || state.evict_idle()).await;
            }
        }));
    }
    if state.cfg.compaction_interval_seconds > 0 {
        let state = state.clone();
        let every = Duration::from_secs(state.cfg.compaction_interval_seconds);
        tasks.push(tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let state = state.clone();
                let _ = tokio::task::spawn_blocking(move || state.compact_changed()).await;
            }
        }));
    }
    tasks
}

fn run_server(config: Config) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(serve_main(config))
}

async fn serve_main(config: Config) -> Result<()> {
    let legacy_key = match std::env::var("HMS_API_KEY") {
        Ok(k) if k.is_empty() => anyhow::bail!("HMS_API_KEY is set but empty"),
        Ok(k) => Some(k),
        Err(_) => None,
    };
    let authenticator = auth::Authenticator::new(
        config.no_auth,
        legacy_key.as_deref(),
        config.keys_file.clone(),
    )?;
    config.validate(authenticator.has_credentials())?;
    log::init(config.log_format, config.log_level);
    std::fs::create_dir_all(&config.data_dir).context("cannot create --data-dir")?;
    #[cfg(feature = "local-models")]
    let stages = load_stages(&config.models)?;
    tenants::load_memory_config(config.memory_config.as_deref())?;
    let tls = serve::Tls::new(&config)?;
    let public = config.unix_socket.is_none() && !config.bind.ip().is_loopback();
    let state = Arc::new(AppState::new(
        config,
        authenticator,
        routes::route_templates(),
        #[cfg(feature = "local-models")]
        stages,
    ));
    if let Some(tls) = &tls {
        let _ = state.tls.set(tls.clone());
    }
    let bound = serve::bind(&state.cfg).await?;
    let address = bound.describe()?;
    // The first stdout line is the machine-readable listen address (scripts and tests read it).
    println!("hms-server listening on {address}");
    log_event!(Info, "listening",
        "address" => address,
        "tls" => tls.is_some(),
        "auth" => if state.auth.is_open() { "none" } else { "keys" },
        "version" => env!("CARGO_PKG_VERSION"));
    if state.auth.is_open() {
        log_event!(Warn, "authentication_disabled", "detail" => "--no-auth: every caller has every scope");
    }
    if public && tls.is_none() {
        log_event!(Warn, "plaintext_on_public_address", "detail" => "bearer tokens cross the network unencrypted; pass --tls-cert/--tls-key or terminate TLS in front");
    }
    #[cfg(unix)]
    spawn_reload_handler(state.clone(), tls.clone());
    let background = spawn_background(&state);
    state.ready.store(true, Ordering::Release);

    let app = routes::router(state.clone());
    let shutdown_state = state.clone();
    let served = serve::serve(state.clone(), bound, tls, app, async move {
        shutdown_signal().await;
        shutdown_state.ready.store(false, Ordering::Release);
        log_event!(Info, "shutdown_signal", "delay_seconds" => shutdown_state.cfg.shutdown_delay_seconds);
        tokio::time::sleep(Duration::from_secs(shutdown_state.cfg.shutdown_delay_seconds)).await;
        shutdown_state.work.stop();
        log_event!(Info, "shutdown_started");
    })
    .await;
    for task in background {
        task.abort();
    }
    state.ready.store(false, Ordering::Release);
    state.work.stop();
    // Admitted work, including blocking jobs whose clients disconnected, finishes before the flush.
    let drained = state
        .work
        .slots
        .clone()
        .acquire_many_owned(state.cfg.max_concurrent_work as u32)
        .await
        .context("cannot drain server work")?;
    tokio::task::spawn_blocking(move || {
        let _drained = drained;
        state.flush_all()
    })
    .await
    .context("server shutdown worker failed")??;
    log_event!(Info, "shutdown_complete");
    served
}

/// Exit 0 only when the server answers 200.
fn healthcheck(args: &config::HealthArgs) -> Result<()> {
    use std::io::{Read, Write};
    use std::net::{IpAddr, SocketAddr, TcpStream};
    let timeout = Duration::from_secs(3);
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        args.path
    );
    let mut response = Vec::new();
    if let Some(path) = &args.unix_socket {
        #[cfg(unix)]
        {
            let mut stream = std::os::unix::net::UnixStream::connect(path)?;
            stream.set_read_timeout(Some(timeout))?;
            stream.write_all(request.as_bytes())?;
            stream.take(4096).read_to_end(&mut response)?;
        }
        #[cfg(not(unix))]
        anyhow::bail!("unix sockets are not supported here: {}", path.display());
    } else {
        let mut target = args.bind;
        if target.ip().is_unspecified() {
            let loopback: IpAddr = if target.is_ipv6() {
                std::net::Ipv6Addr::LOCALHOST.into()
            } else {
                std::net::Ipv4Addr::LOCALHOST.into()
            };
            target = SocketAddr::new(loopback, target.port());
        }
        let mut stream = TcpStream::connect_timeout(&target, timeout)?;
        if args.tls_cert.is_some() {
            return Ok(());
        }
        stream.set_read_timeout(Some(timeout))?;
        stream.write_all(request.as_bytes())?;
        stream.take(4096).read_to_end(&mut response)?;
    }
    let head = String::from_utf8_lossy(&response);
    anyhow::ensure!(
        head.starts_with("HTTP/1.1 200"),
        "unhealthy: {}",
        head.lines().next().unwrap_or("no response")
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Openapi) => {
            println!("{}", serde_json::to_string_pretty(routes::openapi())?);
            Ok(())
        }
        Some(Command::Keygen(args)) => auth::keygen(&args),
        Some(Command::Healthcheck(args)) => healthcheck(&args),
        None => run_server(cli.config),
    }
}

#[cfg(test)]
mod tests;
