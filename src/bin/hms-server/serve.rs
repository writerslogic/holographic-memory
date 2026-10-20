// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The accept loop: TCP or unix socket, optional rustls, a connection cap, a header-read
//! deadline against slow clients, and graceful drain.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use axum::Router;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto;
use hyper_util::server::graceful::GracefulShutdown;
use parking_lot::RwLock;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::Semaphore;
use tokio_rustls::TlsAcceptor;
use tower_service::Service;

use crate::config::Config;
use crate::tenants::AppState;

trait Io: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> Io for T {}

pub enum Bound {
    Tcp(TcpListener),
    Unix(UnixListener, PathBuf),
}

impl Bound {
    pub fn describe(&self) -> Result<String> {
        Ok(match self {
            Self::Tcp(l) => l.local_addr()?.to_string(),
            Self::Unix(_, path) => format!("unix:{}", path.display()),
        })
    }

    async fn accept(&self) -> std::io::Result<Box<dyn Io>> {
        match self {
            Self::Tcp(l) => {
                let (stream, _) = l.accept().await?;
                let _ = stream.set_nodelay(true);
                Ok(Box::new(stream))
            }
            Self::Unix(l, _) => Ok(Box::new(l.accept().await?.0)),
        }
    }
}

impl Drop for Bound {
    fn drop(&mut self) {
        if let Self::Unix(_, path) = self {
            let _ = std::fs::remove_file(path);
        }
    }
}

pub async fn bind(cfg: &Config) -> Result<Bound> {
    let Some(path) = &cfg.unix_socket else {
        let listener = TcpListener::bind(cfg.bind)
            .await
            .with_context(|| format!("cannot bind {}", cfg.bind))?;
        return Ok(Bound::Tcp(listener));
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::{FileTypeExt, PermissionsExt};
        if let Ok(meta) = std::fs::symlink_metadata(path) {
            if !meta.file_type().is_socket() {
                bail!("{} exists and is not a socket", path.display());
            }
            if tokio::net::UnixStream::connect(path).await.is_ok() {
                bail!("{} is already in use by a running server", path.display());
            }
            std::fs::remove_file(path).context("cannot remove the stale socket")?;
        }
        let listener =
            UnixListener::bind(path).with_context(|| format!("cannot bind {}", path.display()))?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        Ok(Bound::Unix(listener, path.clone()))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        bail!("--unix-socket is only supported on unix platforms")
    }
}

/// The certificate in use, replaceable without a restart.
pub struct Tls {
    acceptor: RwLock<TlsAcceptor>,
    cert: PathBuf,
    key: PathBuf,
}

fn load_tls(cert: &Path, key: &Path) -> Result<TlsAcceptor> {
    let chain = CertificateDer::pem_file_iter(cert)
        .with_context(|| format!("cannot read {}", cert.display()))?
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("invalid certificate in {}", cert.display()))?;
    if chain.is_empty() {
        bail!("{} contains no certificate", cert.display());
    }
    let key = PrivateKeyDer::from_pem_file(key)
        .with_context(|| format!("cannot read a private key from {}", key.display()))?;
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_no_client_auth()
    .with_single_cert(chain, key)
    .context("certificate and key do not match")?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(TlsAcceptor::from(Arc::new(config)))
}

impl Tls {
    pub fn new(cfg: &Config) -> Result<Option<Arc<Self>>> {
        let (Some(cert), Some(key)) = (&cfg.tls_cert, &cfg.tls_key) else {
            return Ok(None);
        };
        Ok(Some(Arc::new(Self {
            acceptor: RwLock::new(load_tls(cert, key)?),
            cert: cert.clone(),
            key: key.clone(),
        })))
    }

    /// Re-read the certificate and key. A bad pair leaves the current one in use.
    pub fn reload(&self) -> Result<()> {
        *self.acceptor.write() = load_tls(&self.cert, &self.key)?;
        Ok(())
    }

    fn acceptor(&self) -> TlsAcceptor {
        self.acceptor.read().clone()
    }
}

struct OpenConnection<'a>(&'a AppState);

impl<'a> OpenConnection<'a> {
    fn new(state: &'a AppState) -> Self {
        state
            .metrics
            .connections_open
            .fetch_add(1, Ordering::Relaxed);
        state
            .metrics
            .connections_total
            .fetch_add(1, Ordering::Relaxed);
        Self(state)
    }
}

impl Drop for OpenConnection<'_> {
    fn drop(&mut self) {
        self.0
            .metrics
            .connections_open
            .fetch_sub(1, Ordering::Relaxed);
    }
}

/// Serve until `shutdown` resolves, then stop accepting and give open connections
/// `--shutdown-grace-seconds` to finish their in-flight requests.
pub async fn serve(
    state: Arc<AppState>,
    bound: Bound,
    tls: Option<Arc<Tls>>,
    app: Router,
    shutdown: impl Future<Output = ()>,
) -> Result<()> {
    let graceful = GracefulShutdown::new();
    let slots = Arc::new(Semaphore::new(state.cfg.max_connections));
    let header_timeout = Duration::from_secs(state.cfg.header_read_timeout_seconds);
    tokio::pin!(shutdown);
    loop {
        let permit = tokio::select! {
            permit = slots.clone().acquire_owned() => permit.context("connection limiter closed")?,
            () = &mut shutdown => break,
        };
        let io = tokio::select! {
            accepted = bound.accept() => accepted,
            () = &mut shutdown => break,
        };
        let io = match io {
            Ok(io) => io,
            Err(error) => {
                // Typically fd exhaustion or a reset before accept; back off instead of spinning.
                crate::log_event!(Warn, "accept_failed", "detail" => error.to_string());
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
        };
        let (state, app, tls) = (state.clone(), app.clone(), tls.clone());
        let watcher = graceful.watcher();
        tokio::spawn(async move {
            let _permit = permit;
            let _open = OpenConnection::new(&state);
            let io: Pin<Box<dyn Io>> = match tls {
                None => Box::into_pin(io),
                Some(tls) => {
                    match tokio::time::timeout(header_timeout, tls.acceptor().accept(io)).await {
                        Ok(Ok(stream)) => Box::pin(stream),
                        _ => {
                            state
                                .metrics
                                .tls_handshake_failures
                                .fetch_add(1, Ordering::Relaxed);
                            return;
                        }
                    }
                }
            };
            let service = service_fn(move |request: hyper::Request<Incoming>| {
                let mut app = app.clone();
                async move { app.call(request).await }
            });
            let mut builder = auto::Builder::new(TokioExecutor::new());
            builder
                .http1()
                .timer(TokioTimer::new())
                .header_read_timeout(header_timeout);
            builder.http2().timer(TokioTimer::new());
            let connection = builder.serve_connection(TokioIo::new(io), service);
            if let Err(error) = watcher.watch(connection).await {
                crate::log_event!(Debug, "connection_error", "detail" => error.to_string());
            }
        });
    }
    // Stop listening first so new connections are refused while open ones drain.
    drop(bound);
    let grace = Duration::from_secs(state.cfg.shutdown_grace_seconds);
    tokio::select! {
        () = graceful.shutdown() => {}
        () = tokio::time::sleep(grace) => {
            crate::log_event!(Warn, "shutdown_grace_elapsed",
                "open_connections" => state.metrics.connections_open.load(Ordering::Relaxed));
        }
    }
    Ok(())
}
