// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Error model: RFC 9457 `application/problem+json`.
//!
//! The v1 body also carries the legacy `error: {code, message}` member the published SDK reads.
//! It is deprecated but stays for the whole of `/api/v1`; see docs/server/versioning.md.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use writerslogic_warm::core::query_limits::QueryLimitExceeded;

pub const PROBLEM_BASE: &str =
    "https://github.com/writerslogic/warm/blob/main/docs/server/errors.md#";

/// Every `code` the server emits: (code, status, title). Documented in docs/server/errors.md.
pub const CODES: &[(&str, u16, &str)] = &[
    ("invalid_json", 400, "Malformed JSON"),
    ("invalid_body", 400, "Unreadable request body"),
    ("invalid_tenant", 400, "Invalid tenant id"),
    ("invalid_request", 422, "Invalid request"),
    ("unauthorized", 401, "Authentication required"),
    ("forbidden", 403, "Insufficient scope"),
    ("tenant_forbidden", 403, "Tenant not permitted for this key"),
    ("not_found", 404, "Not found"),
    ("method_not_allowed", 405, "Method not allowed"),
    ("request_timeout", 408, "Request timed out"),
    ("conflict", 409, "Conflict"),
    ("index_unavailable", 409, "Graph index unavailable"),
    (
        "idempotency_in_flight",
        409,
        "Idempotent request still running",
    ),
    (
        "idempotency_reconciliation_required",
        409,
        "Idempotent request requires reconciliation",
    ),
    (
        "idempotency_key_reused",
        422,
        "Idempotency key reused with a different request",
    ),
    ("payload_too_large", 413, "Payload too large"),
    ("unsupported_media_type", 415, "Unsupported media type"),
    ("rate_limited", 429, "Rate limit exceeded"),
    ("quota_exceeded", 429, "Tenant quota exceeded"),
    (
        QueryLimitExceeded::CODE,
        429,
        "Query resource limit exceeded",
    ),
    ("internal", 500, "Internal server error"),
    (
        "mutation_committed",
        500,
        "Mutation committed; reconciliation required",
    ),
    ("not_ready", 503, "Server not ready"),
    ("server_busy", 503, "Server busy"),
    (
        "concurrent_mutation",
        503,
        "Concurrent mutation; retry required",
    ),
    ("tenant_limit", 503, "Too many open tenants"),
    (
        "tenant_busy",
        503,
        "Tenant store is being snapshotted or restored",
    ),
    ("shutting_down", 503, "Server shutting down"),
    ("not_implemented", 501, "Not implemented"),
];

#[derive(Clone, Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    pub retry_after: Option<u64>,
    pub www_authenticate: bool,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            retry_after: matches!(code, "server_busy" | "shutting_down" | "tenant_busy")
                .then_some(1),
            www_authenticate: false,
        }
    }

    pub fn invalid_body() -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_body",
            "request body could not be read",
        )
    }

    pub fn bad(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", message)
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, "conflict", message)
    }

    pub fn retry_after(mut self, seconds: u64) -> Self {
        self.retry_after = Some(seconds.max(1));
        self
    }

    pub fn from_core(error: anyhow::Error) -> Self {
        if error.is::<writerslogic_warm::CommittedMutationError>() {
            Self::committed(error)
        } else if let Some(limit) = error.downcast_ref::<QueryLimitExceeded>() {
            Self::new(
                StatusCode::TOO_MANY_REQUESTS,
                QueryLimitExceeded::CODE,
                limit.to_string(),
            )
        } else {
            Self::internal(error)
        }
    }

    pub fn committed(error: anyhow::Error) -> Self {
        crate::log_event!(Error, "committed_effect_failed", "detail" => format!("{error:#}"));
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "mutation_committed",
            "data mutation is durable; reopen or reconcile before retrying",
        )
    }

    /// Detail goes to the log only; the client sees a fixed message.
    pub fn internal(e: impl std::fmt::Display) -> Self {
        crate::log_event!(Error, "internal_error", "detail" => format!("{e:#}"));
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "internal server error",
        )
    }

    pub fn problem(&self) -> Value {
        let mut body = json!({
            "type": format!("{PROBLEM_BASE}{}", self.code),
            "title": CODES
                .iter()
                .find(|(code, _, _)| *code == self.code)
                .map_or_else(|| self.status.canonical_reason().unwrap_or("Error"), |c| c.2),
            "status": self.status.as_u16(),
            "detail": self.message,
            "code": self.code,
            // Deprecated: kept for SDK clients of v1.
            "error": {"code": self.code, "message": self.message},
        });
        if let Some(id) = crate::log::request_id() {
            body["instance"] = format!("urn:hms:request:{id}").into();
            body["request_id"] = id.into();
        }
        body
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, self.problem().to_string()).into_response();
        let headers = response.headers_mut();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/problem+json"),
        );
        if let Some(seconds) = self.retry_after {
            headers.insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        if self.www_authenticate {
            headers.insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

pub type ApiResult = Result<Response, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_exhaustion_is_a_resource_problem_without_private_context() {
        for resource in ["work", "bytes", "frontier"] {
            let error = anyhow::Error::new(QueryLimitExceeded { resource, limit: 7 })
                .context("private context");
            let error = ApiError::from_core(error);
            let problem = error.problem();
            assert_eq!(problem["status"], 429);
            assert_eq!(problem["code"], "resource_limit");
            assert_eq!(problem["error"]["code"], "resource_limit");
            assert_eq!(
                problem["detail"],
                format!("query {resource} limit 7 exceeded")
            );
            assert!(problem["type"]
                .as_str()
                .unwrap()
                .ends_with("#resource_limit"));
            assert_eq!(error.retry_after, None);
        }
        assert_eq!(
            ApiError::from_core(anyhow::anyhow!("query work limit 7 exceeded")).code,
            "internal"
        );
        let committed = writerslogic_warm::CommittedMutationError {
            effect: "publication",
            source: QueryLimitExceeded {
                resource: "work",
                limit: 7,
            }
            .into(),
        };
        assert_eq!(
            ApiError::from_core(committed.into()).code,
            "mutation_committed"
        );
    }

    #[test]
    fn problem_carries_rfc9457_members_and_the_legacy_error_object() {
        let problem = ApiError::bad("top_k must be 1..=100").problem();
        assert_eq!(problem["status"], 422);
        assert_eq!(problem["code"], "invalid_request");
        assert_eq!(problem["detail"], "top_k must be 1..=100");
        assert!(problem["type"]
            .as_str()
            .unwrap()
            .ends_with("#invalid_request"));
        assert_eq!(problem["error"]["code"], "invalid_request");
        assert_eq!(problem["error"]["message"], "top_k must be 1..=100");
    }

    #[test]
    fn every_code_has_a_section_in_the_error_reference() {
        let reference = include_str!("../../../docs/server/errors.md");
        for (code, _, _) in CODES {
            assert!(
                reference.lines().any(|l| l == format!("### {code}")),
                "docs/server/errors.md lacks a section for {code}"
            );
        }
    }

    #[test]
    fn code_table_is_unique_and_statuses_are_valid() {
        let mut seen = std::collections::HashSet::new();
        for (code, status, _) in CODES {
            assert!(seen.insert(*code), "duplicate code {code}");
            assert!(StatusCode::from_u16(*status).is_ok());
        }
    }
}
