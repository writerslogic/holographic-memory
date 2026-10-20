// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Operational endpoints: liveness, readiness, metrics and the OpenAPI document.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::IntoResponse;

use crate::auth::Scope;
use crate::error::{ApiError, ApiResult};
use crate::request::{json_response, principal};
use crate::routes;
use crate::tenants::AppState;
use crate::types::{body, Health};

pub async fn healthz() -> ApiResult {
    Ok(json_response(
        StatusCode::OK,
        &body(&Health {
            status: "ok".into(),
        }),
    ))
}

/// Ready once the listener is bound; not ready again as soon as shutdown begins.
pub async fn readyz(State(state): State<Arc<AppState>>) -> ApiResult {
    if state.ready.load(Ordering::Acquire) {
        Ok(json_response(
            StatusCode::OK,
            &body(&Health {
                status: "ready".into(),
            }),
        ))
    } else {
        Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_ready",
            "server is starting or shutting down",
        ))
    }
}

pub async fn metrics(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    if !state.cfg.metrics_public {
        principal(&req)?.require(Scope::Admin)?;
    }
    let text = state.metrics.render(&state.saturation());
    Ok((
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        text,
    )
        .into_response())
}

pub async fn openapi() -> ApiResult {
    Ok(json_response(StatusCode::OK, routes::openapi()))
}
