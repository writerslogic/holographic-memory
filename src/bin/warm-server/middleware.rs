// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-cutting layers, outermost first: observe (request id, metrics, access log), CORS, auth.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Request, State};
use axum::http::{header, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::config::LogLevel;
use crate::error::ApiError;
use crate::log::REQUEST_ID;
use crate::routes;
use crate::tenants::{valid_tenant, AppState};

/// Who the request was served for, carried on the response for the access log.
#[derive(Clone)]
struct Served {
    key_id: String,
}

fn valid_request_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

fn method_label(m: &Method) -> &'static str {
    match *m {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::OPTIONS => "OPTIONS",
        Method::HEAD => "HEAD",
        _ => "OTHER",
    }
}

/// Counts a request in flight until dropped, so a request whose future is dropped (client gone,
/// connection cut at the end of the shutdown grace) leaves the gauge.
struct InFlight<'a>(&'a AppState);

impl<'a> InFlight<'a> {
    fn new(state: &'a AppState) -> Self {
        state.metrics.in_flight.fetch_add(1, Ordering::Relaxed);
        Self(state)
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.metrics.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

pub async fn observe(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| valid_request_id(v))
        .map_or_else(|| format!("{:016x}", rand::random::<u64>()), str::to_string);
    let method = method_label(req.method());
    let route = routes::route_label(req.uri().path());
    let tenant = req
        .headers()
        .get("x-tenant-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| valid_tenant(v))
        .map(str::to_string);
    let bytes_in = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let started = Instant::now();
    let in_flight = InFlight::new(&state);
    let mut response = REQUEST_ID
        .scope(id.clone(), async { next.run(req).await })
        .await;
    drop(in_flight);
    let elapsed = started.elapsed();
    let status = response.status().as_u16();
    state
        .metrics
        .observe_request(route, method, status, elapsed);
    if let Ok(v) = HeaderValue::from_str(&id) {
        response.headers_mut().insert("x-request-id", v);
    }
    let level = if matches!(route, "/healthz" | "/readyz" | "/metrics") && status < 500 {
        LogLevel::Debug
    } else if status >= 500 && status != 503 {
        LogLevel::Error
    } else if status >= 400 {
        LogLevel::Warn
    } else {
        LogLevel::Info
    };
    let key = response
        .extensions()
        .get::<Served>()
        .map(|s| s.key_id.clone());
    REQUEST_ID
        .scope(id, async {
            crate::log::emit(
                level,
                "request",
                &[
                    ("method", method.into()),
                    ("route", route.into()),
                    ("status", status.into()),
                    ("latency_ms", (elapsed.as_secs_f64() * 1000.0).into()),
                    ("tenant", tenant.into()),
                    ("key_id", key.into()),
                    ("bytes_in", bytes_in.into()),
                ],
            );
        })
        .await;
    response
}

const ALLOWED_HEADERS: &str =
    "authorization, content-type, x-tenant-id, idempotency-key, if-none-match, x-request-id";
const EXPOSED_HEADERS: &str =
    "etag, retry-after, x-request-id, idempotent-replayed, www-authenticate";

pub async fn cors(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let origins = &state.cfg.cors_origins;
    if origins.is_empty() {
        return next.run(req).await;
    }
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let allowed = origin
        .as_deref()
        .filter(|o| origins.iter().any(|a| a == "*" || a == o));
    let preflight = req.method() == Method::OPTIONS
        && req.headers().contains_key("access-control-request-method");
    let mut response = if preflight {
        let mut r = StatusCode::NO_CONTENT.into_response();
        if allowed.is_some() {
            let h = r.headers_mut();
            h.insert(
                header::ACCESS_CONTROL_ALLOW_METHODS,
                HeaderValue::from_static("GET, POST, PUT, DELETE, OPTIONS"),
            );
            h.insert(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                HeaderValue::from_static(ALLOWED_HEADERS),
            );
            h.insert(
                header::ACCESS_CONTROL_MAX_AGE,
                HeaderValue::from(state.cfg.cors_max_age_seconds),
            );
        }
        r
    } else {
        next.run(req).await
    };
    let h = response.headers_mut();
    h.append(header::VARY, HeaderValue::from_static("Origin"));
    if let Some(origin) = allowed {
        let value = if origins.iter().any(|a| a == "*") {
            HeaderValue::from_static("*")
        } else {
            HeaderValue::from_str(origin).unwrap_or_else(|_| HeaderValue::from_static("null"))
        };
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
        h.insert(
            HeaderName::from_static("access-control-expose-headers"),
            HeaderValue::from_static(EXPOSED_HEADERS),
        );
    }
    response
}

/// Without authentication every caller holds every scope, so a web page the developer visits
/// must not reach the server. A browser marks its requests with `Origin` (simple POSTs need no
/// preflight, so CORS alone does not stop their side effects) and, after DNS rebinding, with
/// the attacker's name in `Host`. Clients that are not browsers send neither.
fn open_request_allowed(state: &AppState, req: &Request) -> bool {
    let origins = &state.cfg.cors_origins;
    if let Some(origin) = req.headers().get(header::ORIGIN) {
        let listed = origin
            .to_str()
            .is_ok_and(|o| origins.iter().any(|a| a == "*" || a == o));
        if !listed {
            return false;
        }
    }
    if state.cfg.unix_socket.is_some() {
        return true;
    }
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .or_else(|| req.uri().authority().map(|a| a.as_str()));
    host.is_none_or(loopback_host)
}

/// `localhost`, a `*.localhost` name or a loopback address, with an optional port.
fn loopback_host(host: &str) -> bool {
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split_once(']').map_or(rest, |(ip, _)| ip),
        None => host.rsplit_once(':').map_or(host, |(name, _)| name),
    };
    let name = name.trim_end_matches('.').to_ascii_lowercase();
    name == "localhost"
        || name.ends_with(".localhost")
        || name
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

pub async fn auth(State(state): State<Arc<AppState>>, mut req: Request, next: Next) -> Response {
    let path = req.uri().path();
    let public =
        routes::is_public(req.method(), path) || (state.cfg.metrics_public && path == "/metrics");
    if public {
        return next.run(req).await;
    }
    if state.auth.is_open() && !open_request_allowed(&state, &req) {
        state.metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
        return ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "this server runs without authentication and serves only same-machine clients: \
             the request's Origin is not in --cors-origins or its Host is not a loopback name",
        )
        .into_response();
    }
    match state.auth.authenticate(req.headers()) {
        Ok(principal) => {
            let served = Served {
                key_id: principal.key_id.clone(),
            };
            req.extensions_mut().insert(Arc::new(principal));
            let mut response = next.run(req).await;
            response.extensions_mut().insert(served);
            response
        }
        Err(error) => {
            state.metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
            error.into_response()
        }
    }
}

pub async fn not_found() -> ApiError {
    ApiError::not_found("no such route")
}

pub async fn method_not_allowed() -> ApiError {
    ApiError::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        "method not allowed",
    )
}
