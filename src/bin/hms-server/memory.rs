// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The memory API (`POST /api/v1/memory/{remember,recall,explain,forget,get,history,
//! applyForgetting,consolidate}`; docs/memory-api) behind the server's auth, limits, tenant
//! lifecycle and idempotency.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use holographic_memory::core::memory_api::MemoryError;
use serde_json::{json, Value};

use crate::auth::Scope;
use crate::config::{MAX_ID_BYTES, MAX_TEXT_BYTES};
use crate::documents::{check_metadata_size, dense, enforce_quota};
use crate::error::{ApiError, ApiResult};
use crate::request::{json_write, parse};
use crate::tenants::AppState;

/// Limits the memory routes share with the document routes, applied to the raw request before
/// the memory API sees it.
fn validate_memory_request(op: &str, request: &Value, state: &AppState) -> Result<(), ApiError> {
    let cfg = &state.cfg;
    let vector = |value: &Value, what: &str| -> Result<(), ApiError> {
        match value {
            Value::Null => Ok(()),
            Value::Array(items) => {
                let values: Option<Vec<f64>> = items.iter().map(Value::as_f64).collect();
                dense(
                    &values.ok_or_else(|| ApiError::bad(format!("{what} must be numbers")))?,
                    cfg.input_dim,
                    what,
                )
            }
            _ => Err(ApiError::bad(format!("{what} must be an array of numbers"))),
        }
    };
    let fact = |fact: &Value| -> Result<(), ApiError> {
        let Value::Object(map) = fact else {
            return Err(ApiError::bad("a fact must be a JSON object"));
        };
        if let Some(Value::String(id)) = map.get("id") {
            if id.len() > MAX_ID_BYTES {
                return Err(ApiError::bad(format!("id exceeds {MAX_ID_BYTES} bytes")));
            }
        }
        if let Some(Value::String(text)) = map.get("text") {
            if text.len() > MAX_TEXT_BYTES {
                return Err(ApiError::bad(format!(
                    "text exceeds {MAX_TEXT_BYTES} bytes"
                )));
            }
        }
        if let Some(metadata) = map.get("metadata") {
            check_metadata_size(metadata)?;
        }
        map.get("embedding")
            .map_or(Ok(()), |v| vector(v, "embedding"))
    };
    match op {
        "remember" => match request.get("facts") {
            Some(Value::Array(facts)) => {
                if facts.is_empty() || facts.len() > cfg.max_batch {
                    return Err(ApiError::bad(format!(
                        "facts must contain 1..={} entries",
                        cfg.max_batch
                    )));
                }
                facts.iter().try_for_each(fact)
            }
            Some(_) => Err(ApiError::bad("facts must be an array")),
            None => fact(request),
        },
        "recall" | "explain" => {
            if let Some(Value::String(query)) = request.get("query") {
                if query.len() > 8192 {
                    return Err(ApiError::bad("query requires at most 8192 bytes"));
                }
            }
            if let Some(k) = request.get("k").and_then(Value::as_i64) {
                if k < 1 || k > i64::from(cfg.max_top_k) {
                    return Err(ApiError::bad(format!("k must be 1..={}", cfg.max_top_k)));
                }
            }
            request
                .get("embedding")
                .map_or(Ok(()), |v| vector(v, "embedding"))
        }
        _ => Ok(()),
    }
}

fn memory_error(error: anyhow::Error) -> ApiError {
    match error.downcast_ref::<MemoryError>() {
        Some(MemoryError::Invalid(message)) => ApiError::bad(message.clone()),
        Some(MemoryError::NotFound(message)) => ApiError::not_found(message.clone()),
        Some(MemoryError::Conflict(message)) => ApiError::conflict(message.clone()),
        Some(MemoryError::Concurrent(message)) => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "concurrent_mutation",
            message.clone(),
        )
        .retry_after(1),
        None => ApiError::from_core(error),
    }
}

/// Facts a `remember` request would add, for the tenant quota. A fact with an id that already
/// exists replaces it and does not grow the store.
fn new_facts(request: &Value, exists: impl Fn(&str) -> bool) -> u64 {
    let one = |fact: &Value| !fact.get("id").and_then(Value::as_str).is_some_and(&exists);
    match request.get("facts").and_then(Value::as_array) {
        Some(facts) => facts.iter().filter(|f| one(f)).count() as u64,
        None => u64::from(one(request)),
    }
}

async fn run(op: &'static str, scope: Scope, state: Arc<AppState>, req: Request) -> ApiResult {
    json_write(state, req, scope, move |state, tenant, bytes| {
        let request: Value = if bytes.is_empty() {
            json!({})
        } else {
            parse(&bytes)?
        };
        validate_memory_request(op, &request, state)?;
        let core = state.core(tenant)?;
        let _mutation = (scope == Scope::Write).then(|| core.quota_lease());
        if op == "remember" {
            enforce_quota(state, &core, || {
                new_facts(&request, |id| core.has_document(id))
            })?;
        }
        let response = core.memory().call(op, request).map_err(memory_error)?;
        if scope == Scope::Write {
            core.bump();
        }
        Ok((StatusCode::OK, response))
    })
    .await
}

macro_rules! memory_handler {
    ($name:ident, $op:literal, $scope:expr) => {
        pub async fn $name(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
            run($op, $scope, state, req).await
        }
    };
}

memory_handler!(remember, "remember", Scope::Write);
memory_handler!(recall, "recall", Scope::Read);
memory_handler!(explain, "explain", Scope::Read);
memory_handler!(forget, "forget", Scope::Write);
memory_handler!(get, "get", Scope::Read);
memory_handler!(history, "history", Scope::Read);
memory_handler!(apply_forgetting, "applyForgetting", Scope::Write);
memory_handler!(consolidate, "consolidate", Scope::Write);
