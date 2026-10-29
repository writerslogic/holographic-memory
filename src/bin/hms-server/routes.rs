// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The route table. One `Op` per operation carries the handler, the required scope and the
//! documentation; both the router and the OpenAPI 3.1 document are built from `OPS`, so a route
//! cannot exist without being documented and secured. Schemas come from the request and response
//! types themselves.

use std::sync::{Arc, OnceLock};

use axum::http::Method;
use axum::middleware::from_fn_with_state;
use axum::routing::{delete, get, post, put, MethodRouter};
use axum::Router;
use holographic_memory::core::memory_api::{
    contract as memory_contract, ConsolidateOptions, ConsolidationReport, ForgetOptions,
    ForgetReport,
};
use schemars::generate::SchemaSettings;
use schemars::{JsonSchema, SchemaGenerator};
use serde_json::{json, Map, Value};

use crate::auth::Scope;
use crate::tenants::AppState;
use crate::{admin, documents, memory, middleware, ops, types};

type SchemaFn = fn(&mut SchemaGenerator) -> Value;

fn schema<T: JsonSchema>(g: &mut SchemaGenerator) -> Value {
    serde_json::to_value(g.subschema_for::<T>()).expect("schema serializes")
}

fn maintenance_request<T: JsonSchema>(g: &mut SchemaGenerator) -> Value {
    let mut schema = serde_json::to_value(T::json_schema(g)).expect("schema serializes");
    schema["properties"]["dryRun"]["default"] = true.into();
    schema
}

fn doc_array(g: &mut SchemaGenerator) -> Value {
    serde_json::to_value(g.subschema_for::<Vec<documents::DocIn>>()).expect("schema serializes")
}

#[derive(Clone, Copy)]
pub enum Request {
    None,
    Json(SchemaFn),
    OptionalJson(SchemaFn),
    /// An optional `{}` body.
    OptionalEmptyObject,
    Raw(&'static str, &'static str),
}

#[derive(Clone, Copy)]
pub enum Reply {
    Json(SchemaFn),
    Raw(&'static str, &'static str),
}

pub struct Param {
    pub name: &'static str,
    pub location: &'static str,
    pub description: &'static str,
    pub kind: &'static str,
}

pub struct Op {
    pub method: &'static str,
    pub path: &'static str,
    pub id: &'static str,
    pub tag: &'static str,
    pub summary: &'static str,
    /// `None` marks an operation anyone may call without credentials.
    pub scope: Option<Scope>,
    /// Whether `X-Tenant-ID` selects the store the operation acts on.
    pub tenant: bool,
    pub request: Request,
    pub status: u16,
    pub reply: Reply,
    pub params: &'static [Param],
    pub handler: fn() -> MethodRouter<Arc<AppState>>,
}

const LIST_PARAMS: &[Param] = &[
    Param {
        name: "limit",
        location: "query",
        description: "Page size, 1..=1000 (default 100).",
        kind: "integer",
    },
    Param {
        name: "after",
        location: "query",
        description: "The next_cursor of the previous page.",
        kind: "string",
    },
];
const IDEMPOTENCY: &[Param] = &[Param {
    name: "Idempotency-Key",
    location: "header",
    description:
        "Replays the recorded response for a retried request (1..=128 visible ASCII characters).",
    kind: "string",
}];
const PURGE: &[Param] = &[
    Param {
        name: "purge",
        location: "query",
        description:
            "true also rewrites the journal and scrubs what the deleted document left behind.",
        kind: "boolean",
    },
    Param {
        name: "import_identity",
        location: "query",
        description:
            "Required matching lowercase SHA-256 owner digest when deleting an imported document.",
        kind: "string",
    },
];

const CONDITIONAL: &[Param] = &[Param {
    name: "If-None-Match",
    location: "header",
    description: "Return 304 when the ETag still matches.",
    kind: "string",
}];

pub static OPS: &[Op] = &[
    Op { method: "get", path: "/healthz", id: "healthz", tag: "operations", summary: "Liveness: the process is serving.", scope: None, tenant: false, request: Request::None, status: 200, reply: Reply::Json(schema::<types::Health>), params: &[], handler: || get(ops::healthz) },
    Op { method: "get", path: "/readyz", id: "readyz", tag: "operations", summary: "Readiness: 200 while accepting work, 503 while starting or draining.", scope: None, tenant: false, request: Request::None, status: 200, reply: Reply::Json(schema::<types::Health>), params: &[], handler: || get(ops::readyz) },
    Op { method: "get", path: "/metrics", id: "metrics", tag: "operations", summary: "Prometheus metrics. Needs the admin scope unless --metrics-public is set.", scope: Some(Scope::Admin), tenant: false, request: Request::None, status: 200, reply: Reply::Raw("text/plain", "Prometheus text exposition format 0.0.4"), params: &[], handler: || get(ops::metrics) },
    Op { method: "get", path: "/openapi.json", id: "openapi", tag: "operations", summary: "This document.", scope: None, tenant: false, request: Request::None, status: 200, reply: Reply::Raw("application/json", "OpenAPI 3.1 document"), params: &[], handler: || get(ops::openapi) },
    Op { method: "post", path: "/api/v1/documents", id: "addDocument", tag: "documents", summary: "Store or replace one document.", scope: Some(Scope::Write), tenant: true, request: Request::Json(schema::<documents::DocIn>), status: 200, reply: Reply::Json(schema::<types::Added>), params: IDEMPOTENCY, handler: || post(documents::add_one) },
    Op { method: "get", path: "/api/v1/documents", id: "listDocuments", tag: "documents", summary: "List document ids and metadata, paginated by id cursor.", scope: Some(Scope::Read), tenant: true, request: Request::None, status: 200, reply: Reply::Json(schema::<types::DocumentPage>), params: LIST_PARAMS, handler: || get(documents::list_documents) },
    Op { method: "post", path: "/api/v1/documents/batch", id: "addDocuments", tag: "documents", summary: "Store or replace up to --max-batch documents in one atomic commit.", scope: Some(Scope::Write), tenant: true, request: Request::Json(doc_array), status: 200, reply: Reply::Json(schema::<types::Added>), params: IDEMPOTENCY, handler: || post(documents::add_batch) },
    Op { method: "post", path: "/api/v1/documents/stream", id: "streamDocuments", tag: "documents", summary: "Bulk ingest newline-delimited documents, committed in batches; not atomic across batches.", scope: Some(Scope::Write), tenant: true, request: Request::Raw("application/x-ndjson", "One document object per line"), status: 200, reply: Reply::Json(schema::<types::StreamResult>), params: &[], handler: || post(documents::ingest_stream) },
    Op { method: "delete", path: "/api/v1/documents/{id}", id: "deleteDocument", tag: "documents", summary: "Delete a document by id (percent-encoded).", scope: Some(Scope::Write), tenant: true, request: Request::None, status: 200, reply: Reply::Json(schema::<types::Deleted>), params: PURGE, handler: || delete(documents::delete_doc) },
    Op { method: "post", path: "/api/v1/query", id: "query", tag: "documents", summary: "Nearest documents to a vector, best first, with exact cosine scores.", scope: Some(Scope::Read), tenant: true, request: Request::Json(schema::<documents::QueryIn>), status: 200, reply: Reply::Json(schema::<types::Matches>), params: &[], handler: || post(documents::query) },
    Op { method: "post", path: "/api/v1/search", id: "search", tag: "documents", summary: "Text search: document ids and scores, best first (BM25, or the on-device model stages when configured); stored text and metadata on request.", scope: Some(Scope::Read), tenant: true, request: Request::Json(schema::<documents::SearchIn>), status: 200, reply: Reply::Json(schema::<types::SearchHits>), params: &[], handler: || post(documents::search) },
    Op { method: "get", path: "/api/v1/stats", id: "stats", tag: "documents", summary: "Document and vector counts and storage use for the tenant.", scope: Some(Scope::Read), tenant: true, request: Request::None, status: 200, reply: Reply::Json(schema::<types::Stats>), params: CONDITIONAL, handler: || get(documents::stats) },
    Op { method: "post", path: "/api/v1/memory/remember", id: "memoryRemember", tag: "memory", summary: "Store one fact or an array of facts with validity time, source and supersession. See docs/memory-api.", scope: Some(Scope::Write), tenant: true, request: Request::Json(schema::<memory_contract::RememberRequest>), status: 200, reply: Reply::Json(schema::<memory_contract::RememberReply>), params: IDEMPOTENCY, handler: || post(memory::remember) },
    Op { method: "post", path: "/api/v1/memory/recall", id: "memoryRecall", tag: "memory", summary: "Retrieve facts for a query, honoring validity time and supersession. See docs/memory-api.", scope: Some(Scope::Read), tenant: true, request: Request::Json(schema::<memory_contract::RecallRequest>), status: 200, reply: Reply::Json(schema::<memory_contract::RecallReply>), params: IDEMPOTENCY, handler: || post(memory::recall) },
    Op { method: "post", path: "/api/v1/memory/explain", id: "memoryExplain", tag: "memory", summary: "Recall with the evidence behind each result and its supersession chain. See docs/memory-api.", scope: Some(Scope::Read), tenant: true, request: Request::Json(schema::<memory_contract::ExplainRequest>), status: 200, reply: Reply::Json(schema::<memory_contract::ExplainReply>), params: IDEMPOTENCY, handler: || post(memory::explain) },
    Op { method: "post", path: "/api/v1/memory/forget", id: "memoryForget", tag: "memory", summary: "Forget facts by id, source or scope. See docs/memory-api.", scope: Some(Scope::Write), tenant: true, request: Request::Json(schema::<memory_contract::ForgetRequest>), status: 200, reply: Reply::Json(schema::<ForgetReport>), params: IDEMPOTENCY, handler: || post(memory::forget) },
    Op { method: "post", path: "/api/v1/memory/get", id: "memoryGet", tag: "memory", summary: "Fetch one fact by id. See docs/memory-api.", scope: Some(Scope::Read), tenant: true, request: Request::Json(schema::<memory_contract::IdRequest>), status: 200, reply: Reply::Json(schema::<memory_contract::GetReply>), params: IDEMPOTENCY, handler: || post(memory::get) },
    Op { method: "post", path: "/api/v1/memory/history", id: "memoryHistory", tag: "memory", summary: "A fact's version and supersession history. See docs/memory-api.", scope: Some(Scope::Read), tenant: true, request: Request::Json(schema::<memory_contract::IdRequest>), status: 200, reply: Reply::Json(schema::<memory_contract::HistoryReply>), params: IDEMPOTENCY, handler: || post(memory::history) },
    Op { method: "post", path: "/api/v1/memory/applyForgetting", id: "memoryApplyForgetting", tag: "memory", summary: "Apply the configured forgetting policy. See docs/memory-api.", scope: Some(Scope::Write), tenant: true, request: Request::OptionalJson(maintenance_request::<ForgetOptions>), status: 200, reply: Reply::Json(schema::<ForgetReport>), params: IDEMPOTENCY, handler: || post(memory::apply_forgetting) },
    Op { method: "post", path: "/api/v1/memory/consolidate", id: "memoryConsolidate", tag: "memory", summary: "Run the configured consolidation policy. See docs/memory-api.", scope: Some(Scope::Write), tenant: true, request: Request::OptionalJson(maintenance_request::<ConsolidateOptions>), status: 200, reply: Reply::Json(schema::<ConsolidationReport>), params: IDEMPOTENCY, handler: || post(memory::consolidate) },
    Op { method: "post", path: "/api/v1/index/maintain", id: "maintainIndex", tag: "index", summary: "Train the tenant's document index over stored embeddings.", scope: Some(Scope::Write), tenant: true, request: Request::OptionalEmptyObject, status: 200, reply: Reply::Json(schema::<types::Maintained>), params: &[], handler: || post(documents::maintain_index) },
    Op { method: "get", path: "/api/v1/compaction", id: "compactionStatus", tag: "index", summary: "Compaction status and schedule for the tenant.", scope: Some(Scope::Read), tenant: true, request: Request::None, status: 200, reply: Reply::Json(schema::<types::CompactionStatusBody>), params: &[], handler: || get(admin::compaction_status) },
    Op { method: "post", path: "/api/v1/compaction", id: "runCompaction", tag: "index", summary: "Compact the tenant's store now.", scope: Some(Scope::Admin), tenant: true, request: Request::None, status: 200, reply: Reply::Json(schema::<types::CompactionRun>), params: &[], handler: || post(admin::run_compaction) },
    Op { method: "post", path: "/api/v1/snapshots", id: "createSnapshot", tag: "snapshots", summary: "Snapshot the tenant's store on the server (the tenant pauses while it is copied).", scope: Some(Scope::Admin), tenant: true, request: Request::None, status: 201, reply: Reply::Json(schema::<types::Snapshot>), params: &[], handler: || post(admin::create_snapshot) },
    Op { method: "get", path: "/api/v1/snapshots", id: "listSnapshots", tag: "snapshots", summary: "List the tenant's snapshots.", scope: Some(Scope::Admin), tenant: true, request: Request::None, status: 200, reply: Reply::Json(schema::<types::Snapshots>), params: &[], handler: || get(admin::list_snapshots) },
    Op { method: "delete", path: "/api/v1/snapshots/{id}", id: "deleteSnapshot", tag: "snapshots", summary: "Delete a snapshot.", scope: Some(Scope::Admin), tenant: true, request: Request::None, status: 200, reply: Reply::Json(schema::<types::Deleted>), params: &[], handler: || delete(admin::delete_snapshot) },
    Op { method: "get", path: "/api/v1/snapshots/{id}/archive", id: "downloadSnapshot", tag: "snapshots", summary: "Download a snapshot as a tar stream (the off-box backup format).", scope: Some(Scope::Admin), tenant: true, request: Request::None, status: 200, reply: Reply::Raw("application/x-tar", "Uncompressed tar of the store directory"), params: &[], handler: || get(admin::download_snapshot) },
    Op { method: "post", path: "/api/v1/snapshots/{id}/restore", id: "restoreSnapshot", tag: "snapshots", summary: "Replace the tenant's store with a snapshot; the replaced store is kept as a snapshot.", scope: Some(Scope::Admin), tenant: true, request: Request::None, status: 200, reply: Reply::Json(schema::<types::Restored>), params: &[], handler: || post(admin::restore_snapshot) },
    Op { method: "put", path: "/api/v1/restore", id: "restoreUpload", tag: "snapshots", summary: "Replace the tenant's store with an uploaded snapshot archive after verifying it.", scope: Some(Scope::Admin), tenant: true, request: Request::Raw("application/x-tar", "Tar produced by the archive endpoint"), status: 200, reply: Reply::Json(schema::<types::Restored>), params: &[], handler: || put(admin::restore_upload) },
    Op { method: "post", path: "/api/v1/admin/keys/reload", id: "reloadKeys", tag: "admin", summary: "Re-read --keys-file. Needs an all-tenant admin key.", scope: Some(Scope::Admin), tenant: false, request: Request::None, status: 200, reply: Reply::Json(schema::<types::KeysReloaded>), params: &[], handler: || post(admin::reload_keys) },
    Op { method: "post", path: "/api/v1/admin/tls/reload", id: "reloadTls", tag: "admin", summary: "Re-read --tls-cert and --tls-key. Needs an all-tenant admin key.", scope: Some(Scope::Admin), tenant: false, request: Request::None, status: 200, reply: Reply::Json(schema::<types::TlsReloaded>), params: &[], handler: || post(admin::reload_tls) },
    Op { method: "get", path: "/api/v1/admin/tenants", id: "listTenants", tag: "admin", summary: "Tenants with a store on disk. Needs an all-tenant admin key.", scope: Some(Scope::Admin), tenant: false, request: Request::None, status: 200, reply: Reply::Json(schema::<types::Tenants>), params: &[], handler: || get(admin::list_tenants) },
];

fn find_op(method: &str, path: &str) -> Option<&'static Op> {
    OPS.iter()
        .filter(|op| op.method.eq_ignore_ascii_case(method) && template_matches(op.path, path))
        .min_by_key(|op| op.path.matches('{').count())
}

fn template_matches(template: &str, path: &str) -> bool {
    let (mut t, mut p) = (template.split('/'), path.split('/'));
    loop {
        match (t.next(), p.next()) {
            (None, None) => return true,
            (Some(t), Some(p)) if t.starts_with('{') => {
                if p.is_empty() {
                    return false;
                }
            }
            (Some(t), Some(p)) if t == p => {}
            _ => return false,
        }
    }
}

/// The route template for metrics and logs; `"unmatched"` keeps label cardinality bounded.
pub fn route_label(path: &str) -> &'static str {
    OPS.iter()
        .filter(|op| template_matches(op.path, path))
        .min_by_key(|op| op.path.matches('{').count())
        .map_or("unmatched", |op| op.path)
}

pub fn is_public(method: &Method, path: &str) -> bool {
    find_op(method.as_str(), path).is_some_and(|op| op.scope.is_none())
}

pub fn route_templates() -> impl Iterator<Item = &'static str> {
    let mut seen = std::collections::BTreeSet::new();
    OPS.iter()
        .map(|op| op.path)
        .filter(move |p| seen.insert(*p))
}

pub fn router(state: Arc<AppState>) -> Router {
    let mut router = Router::new();
    for op in OPS {
        router = router.route(op.path, (op.handler)());
    }
    router
        .fallback(middleware::not_found)
        .method_not_allowed_fallback(middleware::method_not_allowed)
        .layer(from_fn_with_state(state.clone(), middleware::auth))
        .layer(from_fn_with_state(state.clone(), middleware::cors))
        .layer(from_fn_with_state(state.clone(), middleware::observe))
        .with_state(state)
}

fn json_content(content_type: &str, schema: Value) -> Value {
    json!({ content_type: { "schema": schema } })
}

pub fn openapi() -> &'static Value {
    static SPEC: OnceLock<Value> = OnceLock::new();
    SPEC.get_or_init(build_openapi)
}

fn build_openapi() -> Value {
    let mut settings = SchemaSettings::draft2020_12();
    settings.definitions_path = "/components/schemas".into();
    settings.meta_schema = None;
    let mut generator = SchemaGenerator::new(settings);
    let problem = schema::<types::Problem>(&mut generator);
    let mut paths: Map<String, Value> = Map::new();
    for op in OPS {
        let mut parameters = Vec::new();
        if op.tenant {
            parameters.push(json!({
                "name": "X-Tenant-ID", "in": "header", "required": false,
                "description": "Tenant whose store is used: 1..=64 of [A-Za-z0-9_-]. Defaults to the key's only tenant, else \"default\".",
                "schema": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"}
            }));
        }
        if op.path.contains("{id}") {
            parameters.push(json!({
                "name": "id", "in": "path", "required": true,
                "description": if op.path.contains("snapshots") { "Snapshot id." } else { "Document id, percent-encoded." },
                "schema": {"type": "string"}
            }));
        }
        for p in op.params {
            parameters.push(json!({
                "name": p.name, "in": p.location, "required": false,
                "description": p.description, "schema": {"type": p.kind}
            }));
        }
        let mut operation = Map::new();
        operation.insert("operationId".into(), op.id.into());
        operation.insert("summary".into(), op.summary.into());
        operation.insert("tags".into(), json!([op.tag]));
        operation.insert("parameters".into(), Value::Array(parameters));
        match op.request {
            Request::None => {}
            Request::Json(f) | Request::OptionalJson(f) => {
                operation.insert(
                    "requestBody".into(),
                    json!({"required": matches!(op.request, Request::Json(_)), "content": json_content("application/json", f(&mut generator))}),
                );
            }
            Request::OptionalEmptyObject => {
                operation.insert(
                    "requestBody".into(),
                    json!({"required": false, "content": json_content("application/json", json!({"type": "object", "maxProperties": 0}))}),
                );
            }
            Request::Raw(content_type, description) => {
                operation.insert(
                    "requestBody".into(),
                    json!({"required": true, "description": description, "content": json_content(content_type, json!({"type": "string", "format": "binary"}))}),
                );
            }
        }
        let success = match op.reply {
            Reply::Json(f) => {
                json!({"description": "Success", "content": json_content("application/json", f(&mut generator))})
            }
            Reply::Raw(content_type, description) => {
                let schema = if content_type == "application/json" {
                    json!({"type": "object"})
                } else {
                    json!({"type": "string"})
                };
                json!({"description": description, "content": json_content(content_type, schema)})
            }
        };
        let mut responses = Map::new();
        responses.insert(op.status.to_string(), success);
        if matches!(op.id, "stats" | "listDocuments") {
            responses.insert(
                "304".into(),
                json!({"description": "ETag matched If-None-Match"}),
            );
        }
        responses.insert(
            "default".into(),
            json!({"description": "Error (RFC 9457 problem details)", "content": json_content("application/problem+json", problem.clone())}),
        );
        operation.insert("responses".into(), Value::Object(responses));
        operation.insert(
            "security".into(),
            match op.scope {
                None => json!([]),
                Some(scope) => json!([{ "bearerAuth": [scope.name()] }]),
            },
        );
        paths
            .entry(op.path.to_string())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("path item is an object")
            .insert(op.method.into(), Value::Object(operation));
    }
    let mut schemas = generator.take_definitions(true);
    schemas.remove("$schema");
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "hms-server",
            "version": env!("CARGO_PKG_VERSION"),
            "summary": "Holographic Memory HTTP service",
            "description": "One engine store per tenant. The /api/v1 contract is stable: see docs/server/versioning.md. Errors are RFC 9457 problem details.",
            "license": {"name": "Apache-2.0", "identifier": "Apache-2.0"}
        },
        "servers": [{"url": "/"}],
        "tags": [
            {"name": "documents"}, {"name": "memory"}, {"name": "index"}, {"name": "snapshots"},
            {"name": "admin"}, {"name": "operations"}
        ],
        "paths": paths,
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http", "scheme": "bearer",
                    "description": "API key. Scopes: read, write, admin (admin includes read and write). Keys are bound to tenants."
                }
            }
        }
    })
}
