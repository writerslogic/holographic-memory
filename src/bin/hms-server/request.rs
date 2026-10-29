// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The request pipeline shared by every handler: principal, scope, tenant, rate limit, work
//! admission, bounded body read, blocking execution and idempotent replay.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::OwnedSemaphorePermit;

use crate::auth::{Principal, Scope};
use crate::error::{ApiError, ApiResult};
use crate::limits::{fingerprint, Begin, StoredResponse};
use crate::tenants::AppState;

pub fn principal(req: &Request) -> Result<Arc<Principal>, ApiError> {
    req.extensions()
        .get::<Arc<Principal>>()
        .cloned()
        .ok_or_else(|| {
            // The auth middleware runs before every API handler; reaching here is a wiring bug.
            ApiError::internal("request reached a handler without a principal")
        })
}

/// Authorize `req` for `scope`, resolve its tenant and charge the tenant's rate limit.
pub fn admit(
    state: &AppState,
    req: &Request,
    scope: Scope,
) -> Result<(Arc<Principal>, String), ApiError> {
    let principal = principal(req)?;
    if let Err(error) = principal.require(scope) {
        state.metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
        return Err(error);
    }
    let tenant = match principal.tenant(req.headers()) {
        Ok(t) => t,
        Err(error) => {
            if error.code == "tenant_forbidden" {
                state.metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
            }
            return Err(error);
        }
    };
    if let Some(limiter) = &state.rate {
        if let Err(seconds) = limiter.check(&tenant) {
            state.metrics.rate_limited.fetch_add(1, Ordering::Relaxed);
            return Err(ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "tenant request rate exceeded; retry after the indicated delay",
            )
            .retry_after(seconds));
        }
    }
    Ok((principal, tenant))
}

pub async fn acquire(state: &AppState) -> Result<OwnedSemaphorePermit, ApiError> {
    state.work.acquire().await.inspect_err(|e| {
        if e.code == "server_busy" {
            state
                .metrics
                .busy_rejections
                .fetch_add(1, Ordering::Relaxed);
        }
    })
}

pub async fn read_body(state: &AppState, body: Body) -> Result<Vec<u8>, ApiError> {
    tokio::time::timeout(
        Duration::from_secs(state.cfg.body_timeout_seconds),
        to_bytes(body, state.cfg.max_body_bytes),
    )
    .await
    .map_err(|_| {
        ApiError::new(
            StatusCode::REQUEST_TIMEOUT,
            "request_timeout",
            "request body timed out",
        )
    })?
    .map(Vec::from)
    .map_err(|_| {
        ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            format!("request body exceeds {} bytes", state.cfg.max_body_bytes),
        )
    })
}

pub async fn blocking<T: Send + 'static>(
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

pub fn parse<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(bytes).map_err(|e| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_json",
            format!("malformed request body: {e}"),
        )
    })
}

/// A response from already serialized JSON.
pub fn json_bytes_response(status: StatusCode, body: Vec<u8>) -> Response {
    (
        status,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        body,
    )
        .into_response()
}

pub fn json_response(status: StatusCode, body: &Value) -> Response {
    (
        status,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        body.to_string(),
    )
        .into_response()
}

fn replay(stored: &StoredResponse) -> Response {
    let mut response = if stored.status.is_success() {
        json_response(stored.status, &stored.body)
    } else {
        let mut r = (stored.status, stored.body.to_string()).into_response();
        r.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/problem+json"),
        );
        r
    };
    response
        .headers_mut()
        .insert("idempotent-replayed", HeaderValue::from_static("true"));
    response
}

pub fn idempotency_key(headers: &HeaderMap) -> Result<Option<String>, ApiError> {
    let Some(value) = headers.get("idempotency-key") else {
        return Ok(None);
    };
    match value.to_str() {
        Ok(k) if (1..=128).contains(&k.len()) && k.bytes().all(|b| (0x21..=0x7e).contains(&b)) => {
            Ok(Some(k.to_string()))
        }
        _ => Err(ApiError::bad(
            "Idempotency-Key must be 1..=128 visible ASCII characters",
        )),
    }
}

/// A JSON-bodied write: admission, bounded body read, optional idempotent replay, then `work`
/// on a blocking thread. The idempotency record is completed inside the blocking task, so a
/// client that disconnects mid-request still leaves a replayable outcome.
pub async fn json_write(
    state: Arc<AppState>,
    req: Request,
    scope: Scope,
    work: impl FnOnce(&AppState, &str, Vec<u8>) -> Result<(StatusCode, Value), ApiError>
        + Send
        + 'static,
) -> ApiResult {
    let (_, tenant) = admit(&state, &req, scope)?;
    let key = idempotency_key(req.headers())?;
    let (method, path) = (req.method().to_string(), req.uri().path().to_string());
    let permit = acquire(&state).await?;
    let bytes = read_body(&state, req.into_body()).await?;
    let guard = match key {
        None => None,
        Some(key) => {
            match state
                .idem
                .begin(&tenant, &key, fingerprint(&method, &path, &tenant, &bytes))?
            {
                Begin::Replay(stored) => {
                    state
                        .metrics
                        .idempotent_replays
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(replay(&stored));
                }
                Begin::Proceed(guard) => Some(guard),
            }
        }
    };
    let worker = state.clone();
    let result = blocking(permit, move || {
        // The stored replay shares the response body instead of deep-copying it.
        let result = work(&worker, &tenant, bytes).map(|(status, body)| (status, Arc::new(body)));
        if let Some(guard) = guard {
            guard.complete(&match &result {
                Ok((status, body)) => StoredResponse {
                    status: *status,
                    body: Arc::clone(body),
                },
                Err(e) => StoredResponse {
                    status: e.status,
                    body: Arc::new(e.problem()),
                },
            });
        }
        result
    })
    .await?;
    Ok(json_response(result.0, &result.1))
}

#[cfg(test)]
mod committed_tests {
    use super::*;
    use axum::body::Body;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn idempotent_retry_replays_a_committed_failure_without_repeating_work() {
        let dir = tempfile::tempdir().unwrap();
        let state = crate::tests::state_with(dir.path(), 2, &[]);
        let executions = Arc::new(AtomicUsize::new(0));
        for attempt in 0..2 {
            let mut request = Request::builder()
                .method("POST")
                .uri("/api/v1/documents")
                .header("Authorization", "Bearer test-key")
                .header("Idempotency-Key", "committed-failure")
                .body(Body::from("{}"))
                .unwrap();
            let principal = state.auth.authenticate(request.headers()).unwrap();
            request.extensions_mut().insert(Arc::new(principal));
            let count = executions.clone();
            let result = json_write(
                state.clone(),
                request,
                Scope::Write,
                move |state, tenant, _| {
                    count.fetch_add(1, Ordering::SeqCst);
                    let input = serde_json::from_value(
                        serde_json::json!({"id":"durable", "text":"committed once"}),
                    )
                    .unwrap();
                    state
                        .core(tenant)?
                        .memorize_document(input)
                        .map_err(ApiError::from_core)?;
                    Err(ApiError::committed(anyhow::anyhow!(
                        "injected post-commit effect failure"
                    )))
                },
            )
            .await;
            if attempt == 0 {
                assert_eq!(result.unwrap_err().code, "mutation_committed");
            } else {
                let response = result.unwrap();
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(response.headers()["idempotent-replayed"], "true");
            }
        }
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        assert!(state.core("default").unwrap().has_document("durable"));
    }
}
