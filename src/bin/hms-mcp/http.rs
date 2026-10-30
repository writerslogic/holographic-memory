// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::{bail, Context as _, Result};
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use hyper::body::{Body as HttpBody, Frame, Incoming};
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio::net::TcpListener;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tower_service::Service;

use crate::{config, log, server, session};

#[derive(clap::Args, Clone, Debug)]
pub struct HttpLimits {
    /// Maximum accepted HTTP connections, including idle keep-alive connections.
    #[arg(long, env = "HMS_MCP_MAX_HTTP_CONNECTIONS", default_value_t = 64)]
    pub max_http_connections: usize,
    /// Maximum HTTP requests/response streams; excess requests fail immediately.
    #[arg(long, env = "HMS_MCP_MAX_HTTP_INFLIGHT", default_value_t = 32)]
    pub max_http_inflight: usize,
    /// Deadline for receiving HTTP headers.
    #[arg(long, env = "HMS_MCP_HTTP_HEADER_TIMEOUT_MS", default_value_t = 5000)]
    pub http_header_timeout_ms: u64,
    /// Deadline from HTTP request admission until response headers, including body reads.
    #[arg(long, env = "HMS_MCP_HTTP_REQUEST_TIMEOUT_MS", default_value_t = 30000)]
    pub http_request_timeout_ms: u64,
    /// Response-body production deadline; TCP lifetime also bounds stalled writers.
    #[arg(long, env = "HMS_MCP_HTTP_STREAM_TIMEOUT_MS", default_value_t = 300000)]
    pub http_stream_timeout_ms: u64,
    /// Maximum lifetime of a TCP connection, including stalled writers.
    #[arg(
        long,
        env = "HMS_MCP_HTTP_CONNECTION_TIMEOUT_MS",
        default_value_t = 600000
    )]
    pub http_connection_timeout_ms: u64,
    /// Maximum encoded bytes in one HTTP response or SSE stream.
    #[arg(long, env = "HMS_MCP_MAX_HTTP_RESPONSE_BYTES", default_value_t = 32 << 20)]
    pub max_http_response_bytes: usize,
    /// Legacy protocol sessions retained in memory. Current stateless requests use HTTP limits.
    #[arg(long, env = "HMS_MCP_MAX_HTTP_SESSIONS", default_value_t = 16)]
    pub max_http_sessions: usize,
    /// Retire an idle legacy session and remove its replay caches after this duration.
    #[arg(long, env = "HMS_MCP_HTTP_SESSION_IDLE_MS", default_value_t = 300000)]
    pub http_session_idle_ms: u64,
    /// Maximum request/notification/stream operations per legacy session before renewal.
    #[arg(
        long,
        env = "HMS_MCP_MAX_HTTP_SESSION_REQUESTS",
        default_value_t = 4096
    )]
    pub max_http_session_requests: usize,
    /// Cumulative serialized outbound bytes retained during a legacy session.
    #[arg(long, env = "HMS_MCP_MAX_HTTP_SESSION_BYTES", default_value_t = 64 << 20)]
    pub max_http_session_bytes: usize,
    /// Grace period for HTTP shutdown before remaining connection tasks are aborted.
    #[arg(long, env = "HMS_MCP_HTTP_SHUTDOWN_GRACE_MS", default_value_t = 10000)]
    pub http_shutdown_grace_ms: u64,
}

impl HttpLimits {
    pub fn validate(&self) -> Result<()> {
        for (name, value, maximum) in [
            ("connections", self.max_http_connections, 4096),
            ("inflight", self.max_http_inflight, 4096),
            ("sessions", self.max_http_sessions, 1024),
            (
                "session requests",
                self.max_http_session_requests,
                1_000_000,
            ),
        ] {
            if !(1..=maximum).contains(&value) {
                bail!("HTTP {name} must be 1..={maximum}");
            }
        }
        for value in [
            self.http_header_timeout_ms,
            self.http_request_timeout_ms,
            self.http_stream_timeout_ms,
            self.http_connection_timeout_ms,
            self.http_session_idle_ms,
            self.http_shutdown_grace_ms,
        ] {
            if !(1..=3_600_000).contains(&value) {
                bail!("HTTP deadlines must be 1..=3600000 milliseconds");
            }
        }
        if !(1024..=256 << 20).contains(&self.max_http_response_bytes)
            || !(self.max_http_response_bytes..=1 << 30).contains(&self.max_http_session_bytes)
        {
            bail!("HTTP response bytes must be 1024..=268435456 and session bytes at least that limit and at most 1073741824");
        }
        Ok(())
    }

    fn session_limits(&self) -> session::Limits {
        session::Limits {
            sessions: self.max_http_sessions,
            idle: Duration::from_millis(self.http_session_idle_ms),
            initialize: Duration::from_millis(self.http_request_timeout_ms),
            requests: self.max_http_session_requests,
            bytes: self.max_http_session_bytes,
            message_bytes: self.max_http_response_bytes,
        }
    }
}

#[derive(Clone)]
struct Admission {
    slots: Arc<Semaphore>,
    limits: HttpLimits,
    cancel: CancellationToken,
}

async fn admit(State(state): State<Admission>, request: Request, next: Next) -> Response {
    let Ok(permit) = state.slots.clone().try_acquire_owned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "HTTP request capacity reached",
        )
            .into_response();
    };
    if state.cancel.is_cancelled() {
        return (StatusCode::SERVICE_UNAVAILABLE, "server shutting down").into_response();
    }
    let response = tokio::select! {
        () = state.cancel.cancelled() => {
            return (StatusCode::SERVICE_UNAVAILABLE, "server shutting down").into_response();
        }
        response = tokio::time::timeout(
            Duration::from_millis(state.limits.http_request_timeout_ms),
            next.run(request),
        ) => response,
    };
    match response {
        Ok(response) => {
            let (parts, body) = response.into_parts();
            Response::from_parts(
                parts,
                Body::new(LeaseBody {
                    inner: body,
                    remaining: state.limits.max_http_response_bytes,
                    permit: Some(permit),
                    deadline: Box::pin(tokio::time::sleep(Duration::from_millis(
                        state.limits.http_stream_timeout_ms,
                    ))),
                    cancelled: Box::pin(state.cancel.cancelled_owned()),
                }),
            )
        }
        Err(_) => (
            StatusCode::REQUEST_TIMEOUT,
            "HTTP request deadline exceeded",
        )
            .into_response(),
    }
}

struct LeaseBody {
    inner: Body,
    remaining: usize,
    permit: Option<OwnedSemaphorePermit>,
    deadline: Pin<Box<tokio::time::Sleep>>,
    cancelled: Pin<Box<dyn Future<Output = ()> + Send + Sync>>,
}

impl HttpBody for LeaseBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        if self.deadline.as_mut().poll(cx).is_ready() || self.cancelled.as_mut().poll(cx).is_ready()
        {
            self.permit.take();
            self.inner = Body::empty();
            return Poll::Ready(None);
        }
        let frame = Pin::new(&mut self.inner).poll_frame(cx);
        match &frame {
            Poll::Ready(Some(Ok(frame))) => {
                let len = frame.data_ref().map_or(0, Bytes::len);
                if len > self.remaining {
                    self.permit.take();
                    self.inner = Body::empty();
                    return Poll::Ready(Some(Err(axum::Error::new(std::io::Error::other(
                        "HTTP response byte budget exceeded",
                    )))));
                }
                self.remaining -= len;
            }
            Poll::Ready(None | Some(Err(_))) => {
                self.permit.take();
            }
            Poll::Pending => {}
        }
        frame
    }
}

fn app<S: rmcp::ServerHandler + Clone + Send + 'static>(
    cli: &config::Cli,
    handler: S,
    sessions: Arc<session::BoundedSessions>,
    cancel: CancellationToken,
    token: Option<Arc<String>>,
    oauth: Option<Arc<crate::oauth::Authority>>,
) -> Router {
    let mut hosts: Vec<String> = ["localhost", "127.0.0.1", "::1"].map(String::from).into();
    hosts.extend(cli.allow_hosts.iter().cloned());
    let transport = StreamableHttpServerConfig::default()
        .with_allowed_hosts(hosts)
        .with_allowed_origins(cli.allow_origins.clone())
        .enforce_origin_validation()
        .with_legacy_session_mode(oauth.is_none())
        .with_max_request_body_bytes(cli.max_request_bytes)
        .with_cancellation_token(cancel.clone());
    let service = StreamableHttpService::new(move || Ok(handler.clone()), sessions, transport);
    let mut mcp = Router::new().nest_service(&cli.path, service);
    if let Some(token) = token {
        mcp = mcp.layer(middleware::from_fn_with_state(token, crate::bearer));
    }
    if let Some(authority) = oauth {
        mcp = mcp.layer(middleware::from_fn_with_state(
            authority.clone(),
            crate::oauth::authorize,
        ));
        let path = authority.metadata_path.clone();
        mcp = mcp.route(
            &path,
            get(move || {
                let metadata = authority.metadata();
                async move {
                    (
                        [
                            ("cache-control", "no-store"),
                            ("access-control-allow-origin", "*"),
                        ],
                        axum::Json(metadata),
                    )
                }
            }),
        );
    }
    mcp.route("/health", get(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(
            Admission {
                slots: Arc::new(Semaphore::new(cli.http.max_http_inflight)),
                limits: cli.http.clone(),
                cancel,
            },
            admit,
        ))
}

pub async fn serve(cli: &config::Cli, handler: server::HmsMcp) -> Result<()> {
    let token = config::http_token()?.map(Arc::new);
    let oauth = crate::oauth::Authority::from_cli(cli)?;
    if token.is_none() && oauth.is_none() {
        log::warn("HTTP transport without HMS_MCP_TOKEN: any local process can use the store");
    }
    let cancel = CancellationToken::new();
    let sessions = session::BoundedSessions::new(cli.http.session_limits());
    let app = app(cli, handler, sessions.clone(), cancel.clone(), token, oauth);
    let listener = TcpListener::bind(cli.bind)
        .await
        .with_context(|| format!("cannot bind {}", cli.bind))?;
    let address = listener.local_addr()?;
    println!("hms-mcp listening on http://{address}{}", cli.path);
    let reaper = tokio::spawn({
        let sessions = sessions.clone();
        let cancel = cancel.clone();
        async move { sessions.reap(cancel).await }
    });
    let result = serve_bound(
        listener,
        app,
        cli.http.clone(),
        cancel.clone(),
        crate::shutdown_signal(),
    )
    .await;
    cancel.cancel();
    sessions.shutdown().await;
    reaper.await.context("session reaper failed")?;
    result
}

async fn serve_bound(
    listener: TcpListener,
    app: Router,
    limits: HttpLimits,
    cancel: CancellationToken,
    shutdown: impl Future<Output = ()>,
) -> Result<()> {
    let slots = Arc::new(Semaphore::new(limits.max_http_connections));
    let graceful = GracefulShutdown::new();
    let mut tasks = JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        while let Some(result) = tasks.try_join_next() {
            if let Err(error) = result {
                log::error(&format!("HTTP connection task failed: {error}"));
            }
        }
        let accepted = tokio::select! {
            () = &mut shutdown => break,
            finished = tasks.join_next(), if !tasks.is_empty() => {
                if let Some(Err(error)) = finished { log::error(&format!("HTTP connection task failed: {error}")); }
                continue;
            }
            accepted = listener.accept() => accepted,
        };
        let (stream, _) = match accepted {
            Ok(accepted) => accepted,
            Err(error) => {
                log::warn(&format!("HTTP accept failed: {error}"));
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
        };
        let Ok(permit) = slots.clone().try_acquire_owned() else {
            drop(stream);
            continue;
        };
        let _ = stream.set_nodelay(true);
        let app = app.clone();
        let watcher = graceful.watcher();
        let limits = limits.clone();
        tasks.spawn(async move {
            let _permit = permit;
            let service = service_fn(move |request: hyper::Request<Incoming>| {
                let mut app = app.clone();
                async move { app.call(request).await }
            });
            let mut builder = hyper::server::conn::http1::Builder::new();
            builder
                .timer(TokioTimer::new())
                .header_read_timeout(Duration::from_millis(limits.http_header_timeout_ms))
                .max_buf_size(16 * 1024)
                .max_headers(64);
            let connection = builder.serve_connection(TokioIo::new(stream), service);
            match tokio::time::timeout(
                Duration::from_millis(limits.http_connection_timeout_ms),
                watcher.watch(connection),
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(error)) => log::debug(&format!("HTTP connection ended: {error}")),
                Err(_) => log::debug("HTTP connection lifetime exceeded"),
            }
        });
    }
    drop(listener);
    cancel.cancel();
    if tokio::time::timeout(
        Duration::from_millis(limits.http_shutdown_grace_ms),
        graceful.shutdown(),
    )
    .await
    .is_err()
    {
        log::warn("HTTP shutdown grace elapsed; closing remaining connections");
    }
    tasks.abort_all();
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result {
            if !error.is_cancelled() {
                log::error(&format!("HTTP connection shutdown failed: {error}"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    fn limits() -> HttpLimits {
        config::Cli::parse_from(["hms-mcp"]).http
    }

    async fn start(
        app: Router,
        limits: HttpLimits,
        cancel: CancellationToken,
    ) -> (
        std::net::SocketAddr,
        CancellationToken,
        tokio::task::JoinHandle<Result<()>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let stop = CancellationToken::new();
        let shutdown = stop.clone().cancelled_owned();
        let task = tokio::spawn(serve_bound(listener, app, limits, cancel, shutdown));
        (address, stop, task)
    }

    fn admitted(app: Router, limits: HttpLimits, cancel: CancellationToken) -> Router {
        app.layer(middleware::from_fn_with_state(
            Admission {
                slots: Arc::new(Semaphore::new(limits.max_http_inflight)),
                limits,
                cancel,
            },
            admit,
        ))
    }

    async fn request(address: std::net::SocketAddr, path: &str) -> Vec<u8> {
        let mut connection = TcpStream::connect(address).await.unwrap();
        connection
            .write_all(
                format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            )
            .await
            .unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(
            Duration::from_secs(2),
            connection.read_to_end(&mut response),
        )
        .await
        .unwrap()
        .unwrap();
        response
    }

    async fn stop(stop: CancellationToken, task: tokio::task::JoinHandle<Result<()>>) {
        stop.cancel();
        tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn slow_headers_and_connection_overflow_are_bounded() {
        let mut limits = limits();
        limits.max_http_connections = 1;
        limits.http_header_timeout_ms = 50;
        let (address, shutdown, task) = start(
            Router::new().route("/ok", get(|| async { "ok" })),
            limits,
            CancellationToken::new(),
        )
        .await;
        let mut slow = TcpStream::connect(address).await.unwrap();
        slow.write_all(b"GET /ok HTTP/1.1\r\nHost:").await.unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), slow.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(response.is_empty() || response.starts_with(b"HTTP/1.1 408"));
        let mut first = TcpStream::connect(address).await.unwrap();
        first
            .write_all(b"GET /ok HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut received = vec![0; 1024];
        let n = first.read(&mut received).await.unwrap();
        assert!(received[..n].starts_with(b"HTTP/1.1 200"));
        let mut overflow = TcpStream::connect(address).await.unwrap();
        let n = tokio::time::timeout(Duration::from_secs(1), overflow.read(&mut received))
            .await
            .unwrap()
            .unwrap_or(0);
        assert_eq!(n, 0);
        drop(first);
        stop(shutdown, task).await;
    }

    #[tokio::test]
    async fn http_admission_lives_through_streaming_and_shutdown_closes_streams() {
        let mut limits = limits();
        limits.max_http_inflight = 1;
        limits.http_stream_timeout_ms = 100;
        limits.http_shutdown_grace_ms = 100;
        let cancel = CancellationToken::new();
        let app = Router::new()
            .route(
                "/slow",
                get(|| async {
                    Body::from_stream(
                        futures_util::stream::pending::<Result<Bytes, std::io::Error>>(),
                    )
                }),
            )
            .route("/ok", get(|| async { "ok" }));
        let app = admitted(app, limits.clone(), cancel.clone());
        let (address, shutdown, task) = start(app, limits, cancel).await;
        let mut slow = TcpStream::connect(address).await.unwrap();
        slow.write_all(b"GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut received = vec![0; 1024];
        let n = tokio::time::timeout(Duration::from_secs(1), slow.read(&mut received))
            .await
            .unwrap()
            .unwrap();
        assert!(received[..n].starts_with(b"HTTP/1.1 200"));
        assert!(request(address, "/ok").await.starts_with(b"HTTP/1.1 503"));
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(request(address, "/ok").await.starts_with(b"HTTP/1.1 200"));
        stop(shutdown, task).await;
        let mut rest = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), slow.read_to_end(&mut rest))
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn response_byte_boundaries_and_slow_request_bodies_use_real_http() {
        for bytes in [1023, 1024, 1025] {
            let mut limits = limits();
            limits.max_http_response_bytes = 1024;
            limits.http_request_timeout_ms = 50;
            let cancel = CancellationToken::new();
            let app = Router::new()
                .route("/data", get(move || async move { "x".repeat(bytes) }))
                .route(
                    "/body",
                    axum::routing::post(|body: Bytes| async move { body.len().to_string() }),
                );
            let app = admitted(app, limits.clone(), cancel.clone());
            let (address, shutdown, task) = start(app, limits, cancel).await;
            let response = request(address, "/data").await;
            assert_eq!(
                response
                    .windows(bytes)
                    .any(|window| window.iter().all(|b| *b == b'x')),
                bytes <= 1024
            );
            let mut slow = TcpStream::connect(address).await.unwrap();
            slow.write_all(b"POST /body HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\nConnection: close\r\n\r\nx").await.unwrap();
            let mut response = Vec::new();
            tokio::time::timeout(Duration::from_secs(1), slow.read_to_end(&mut response))
                .await
                .unwrap()
                .unwrap();
            assert!(response.starts_with(b"HTTP/1.1 408"));
            stop(shutdown, task).await;
        }
    }

    #[tokio::test]
    async fn shutdown_cancels_a_request_waiting_for_its_body() {
        let mut limits = limits();
        limits.http_shutdown_grace_ms = 1000;
        let cancel = CancellationToken::new();
        let arrived = Arc::new(tokio::sync::Notify::new());
        let observed = arrived.clone();
        let app = admitted(
            Router::new()
                .route(
                    "/body",
                    axum::routing::post(|body: Bytes| async move { body.len().to_string() }),
                )
                .layer(middleware::from_fn(move |request: Request, next: Next| {
                    let observed = observed.clone();
                    async move {
                        observed.notify_one();
                        next.run(request).await
                    }
                })),
            limits.clone(),
            cancel.clone(),
        );
        let (address, shutdown, task) = start(app, limits, cancel).await;
        let mut slow = TcpStream::connect(address).await.unwrap();
        slow.write_all(b"POST /body HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\nConnection: close\r\n\r\nx").await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), arrived.notified())
            .await
            .unwrap();
        shutdown.cancel();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_millis(500), slow.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(response.starts_with(b"HTTP/1.1 503"));
        stop(shutdown, task).await;
    }

    #[derive(Clone)]
    struct Ping;

    impl rmcp::ServerHandler for Ping {}

    async fn mcp_request(
        address: std::net::SocketAddr,
        method: &str,
        session: Option<&str>,
        body: &str,
    ) -> String {
        let session = session.map_or_else(String::new, |id| format!("Mcp-Session-Id: {id}\r\n"));
        let request = format!(
            "{method} /mcp HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nAccept: application/json, text/event-stream\r\nContent-Type: application/json\r\nMcp-Protocol-Version: 2025-06-18\r\n{session}Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        String::from_utf8(response).unwrap()
    }

    #[tokio::test]
    async fn real_http_session_capacity_and_delete_reclaim_the_slot() {
        let mut cli = config::Cli::parse_from(["hms-mcp"]);
        cli.http.max_http_sessions = 1;
        let sessions = session::BoundedSessions::new(cli.http.session_limits());
        let cancel = CancellationToken::new();
        let app = app(&cli, Ping, sessions.clone(), cancel.clone(), None, None);
        let (address, shutdown, task) = start(app, cli.http, cancel).await;
        let initialize = r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#;
        let first = mcp_request(address, "POST", None, initialize).await;
        assert!(first.starts_with("HTTP/1.1 200"), "{first}");
        let id = first
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(name, _)| name.eq_ignore_ascii_case("mcp-session-id"))
            .map(|(_, id)| id.trim().to_owned())
            .unwrap();
        let overflow = mcp_request(address, "POST", None, initialize).await;
        assert!(overflow.starts_with("HTTP/1.1 500"), "{overflow}");
        let deleted = mcp_request(address, "DELETE", Some(&id), "").await;
        assert!(deleted.starts_with("HTTP/1.1 202"), "{deleted}");
        let renewed = mcp_request(address, "POST", None, initialize).await;
        assert!(renewed.starts_with("HTTP/1.1 200"), "{renewed}");
        stop(shutdown, task).await;
        sessions.shutdown().await;
    }

    #[tokio::test]
    async fn oauth_discovery_scopes_revocation_and_session_isolation_use_real_http() {
        use serde_json::json;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let claims = Arc::new(std::sync::Mutex::new(json!({
            "active":true,"iss":"https://issuer.example","aud":"https://memory.example/mcp",
            "sub":"alice","exp":now+600,"token_type":"Bearer","scope":"hms:read","hms_namespace":"team"
        })));
        let calls = Arc::new(AtomicUsize::new(0));
        let authority_app = Router::new().route(
            "/introspect",
            axum::routing::post({
                let claims = claims.clone();
                let calls = calls.clone();
                move |headers: axum::http::HeaderMap, body: String| {
                    let claims = claims.clone();
                    let calls = calls.clone();
                    async move {
                        assert_eq!(headers["authorization"], "Bearer resource-credential");
                        assert!(body.contains("token=access-token"));
                        calls.fetch_add(1, Ordering::SeqCst);
                        axum::Json(claims.lock().unwrap().clone())
                    }
                }
            }),
        );
        let (as_address, as_stop, as_task) =
            start(authority_app, limits(), CancellationToken::new()).await;
        let authority =
            crate::oauth::Authority::for_test(format!("http://{as_address}/introspect"));
        let cli = config::Cli::parse_from(["hms-mcp"]);
        let sessions = session::BoundedSessions::new(cli.http.session_limits());
        let cancel = CancellationToken::new();
        let application = app(
            &cli,
            Ping,
            sessions.clone(),
            cancel.clone(),
            None,
            Some(authority),
        );
        let (address, shutdown, task) = start(application, cli.http, cancel).await;
        let discovery =
            String::from_utf8(request(address, "/.well-known/oauth-protected-resource/mcp").await)
                .unwrap();
        assert!(discovery.starts_with("HTTP/1.1 200"));
        assert!(discovery.contains("https://memory.example/mcp"));
        assert!(discovery.contains("https://issuer.example"));
        let missing = String::from_utf8(request(address, "/mcp").await).unwrap();
        assert!(missing.starts_with("HTTP/1.1 401"));
        assert!(missing.contains(
            "resource_metadata=\"https://memory.example/.well-known/oauth-protected-resource/mcp\""
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        async fn send(address: std::net::SocketAddr, body: &str, extra: &str) -> String {
            let request = format!("POST /mcp HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nAuthorization: Bearer access-token\r\nAccept: application/json, text/event-stream\r\nContent-Type: application/json\r\nMcp-Protocol-Version: 2025-06-18\r\n{extra}Content-Length: {}\r\n\r\n{body}",body.len());
            let mut stream = TcpStream::connect(address).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            let mut response = Vec::new();
            tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
                .await
                .unwrap()
                .unwrap();
            String::from_utf8(response).unwrap()
        }
        let initialize = r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#;
        let success = send(address, initialize, "").await;
        assert!(success.starts_with("HTTP/1.1 200"), "{success}");
        assert!(!success.to_ascii_lowercase().contains("mcp-session-id:"));
        let mutation = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"remember","arguments":{}}}"#;
        let denied = send(address, mutation, "").await;
        assert!(denied.starts_with("HTTP/1.1 403"), "{denied}");
        assert!(denied.contains("insufficient_scope"));
        assert!(send(address, initialize, "Mcp-Session-Id: stolen\r\n")
            .await
            .starts_with("HTTP/1.1 400"));
        claims.lock().unwrap()["active"] = json!(false);
        assert!(send(address, initialize, "")
            .await
            .starts_with("HTTP/1.1 401"));
        claims.lock().unwrap()["active"] = json!(true);
        claims.lock().unwrap()["hms_namespace"] = json!("another-team");
        assert!(send(address, initialize, "")
            .await
            .starts_with("HTTP/1.1 401"));
        assert_eq!(calls.load(Ordering::SeqCst), 5);
        claims.lock().unwrap()["padding"] = json!("x".repeat(16 * 1024));
        assert!(send(address, initialize, "")
            .await
            .starts_with("HTTP/1.1 503"));
        stop(shutdown, task).await;
        sessions.shutdown().await;
        stop(as_stop, as_task).await;
    }
}
