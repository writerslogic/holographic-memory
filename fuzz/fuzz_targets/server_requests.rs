// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![allow(dead_code, unused_imports)]

//! hms-server request surface, through the server's own parsing and validation: the single,
//! batch, NDJSON and query bodies, the tenant, idempotency and content-type headers, and
//! snapshot ids. The first input byte picks the surface. The server modules are compiled into
//! this crate from `src/bin/hms-server/`, so the code under test is the shipped code.

#[path = "../../src/bin/hms-server/admin.rs"]
mod admin;
#[path = "../../src/bin/hms-server/auth.rs"]
mod auth;
#[path = "../../src/bin/hms-server/config.rs"]
mod config;
#[path = "../../src/bin/hms-server/documents.rs"]
mod documents;
#[path = "../../src/bin/hms-server/error.rs"]
mod error;
#[path = "../../src/bin/hms-server/limits.rs"]
mod limits;
#[path = "../../src/bin/hms-server/log.rs"]
mod log;
#[path = "../../src/bin/hms-server/memory.rs"]
mod memory;
#[path = "../../src/bin/hms-server/metrics.rs"]
mod metrics;
#[path = "../../src/bin/hms-server/middleware.rs"]
mod middleware;
#[path = "../../src/bin/hms-server/ops.rs"]
mod ops;
#[path = "../../src/bin/hms-server/request.rs"]
mod request;
#[path = "../../src/bin/hms-server/restore.rs"]
mod restore;
#[path = "../../src/bin/hms-server/routes.rs"]
mod routes;
#[path = "../../src/bin/hms-server/serve.rs"]
mod serve;
#[path = "../../src/bin/hms-server/tenants.rs"]
mod tenants;
#[path = "../../src/bin/hms-server/vindex.rs"]
mod vindex;
#[path = "../../src/bin/hms-server/types.rs"]
mod types;

use std::collections::HashSet;
use std::sync::OnceLock;

use axum::http::{header, HeaderMap, HeaderValue};
use clap::Parser;
use libfuzzer_sys::fuzz_target;

use auth::{Principal, ScopeSet, TenantAccess};
use documents::{is_ndjson, validate_batch, validate_doc, DocIn, QueryIn};

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    config: config::Config,
}

fn config() -> &'static config::Config {
    static CONFIG: OnceLock<config::Config> = OnceLock::new();
    CONFIG.get_or_init(|| Args::parse_from(["hms-server"]).config)
}

fn principal(tenants: TenantAccess) -> Principal {
    Principal {
        key_id: "fuzz".into(),
        tenants,
        scopes: ScopeSet::from_names(&["read".into(), "write".into(), "admin".into()])
            .expect("known scope names"),
    }
}

fuzz_target!(|data: &[u8]| {
    let cfg = config();
    let Some((&mode, body)) = data.split_first() else {
        return;
    };
    match mode % 7 {
        0 => {
            if let Ok(doc) = request::parse::<DocIn>(body) {
                let _ = validate_doc(doc, cfg, mode & 0x80 != 0);
            }
        }
        1 => {
            if let Ok(docs) = request::parse::<Vec<DocIn>>(body) {
                if docs.is_empty() || docs.len() > cfg.max_batch {
                    return;
                }
                let valid: Result<Vec<_>, _> = docs
                    .into_iter()
                    .map(|doc| validate_doc(doc, cfg, false))
                    .collect();
                if let Ok(valid) = valid {
                    let _ = validate_batch(&valid, cfg.input_dim);
                }
            }
        }
        2 => {
            if let Ok(query) = request::parse::<QueryIn>(body) {
                if let Some(vector) = query.query_vector {
                    let _ = documents::dense(&vector, cfg.input_dim, "query_vector");
                }
            }
        }
        3 => {
            let mut headers = HeaderMap::new();
            if let Ok(value) = HeaderValue::from_bytes(body) {
                headers.insert("x-tenant-id", value);
            }
            let only = TenantAccess::Only(HashSet::from(["default".to_string()]));
            for access in [TenantAccess::Any, only] {
                let _ = principal(access).tenant(&headers);
            }
        }
        4 => {
            let mut headers = HeaderMap::new();
            if let Ok(value) = HeaderValue::from_bytes(body) {
                headers.insert("idempotency-key", value);
            }
            let _ = request::idempotency_key(&headers);
        }
        5 => {
            // NDJSON: a content type from the first line, then one document per remaining line.
            let mut lines = body.split(|&b| b == b'\n');
            let mut headers = HeaderMap::new();
            if let Some(Ok(value)) = lines.next().map(HeaderValue::from_bytes) {
                headers.insert(header::CONTENT_TYPE, value);
            }
            if is_ndjson(&headers) {
                for line in lines.take(cfg.max_batch) {
                    if let Ok(doc) = request::parse::<DocIn>(line) {
                        let _ = validate_doc(doc, cfg, false);
                    }
                }
            }
        }
        _ => {
            if let Ok(text) = std::str::from_utf8(body) {
                let _ = tenants::valid_snapshot_id(text);
                let _ = tenants::valid_tenant(text);
            }
        }
    }
});
