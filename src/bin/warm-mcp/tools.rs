// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Tool definitions and handlers. Handlers parse strictly (unknown fields are errors), validate
//! against `Limits`, and call only the `WarmBackend` methods.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::memory::{
    preview, Fail, ForgetReq, Limits, ListReq, Memory, RememberReq, Res, SearchReq,
};
use crate::store::WarmBackend;

pub const ALL_TOOLS: [&str; 8] = [
    "remember", "recall", "search", "get", "forget", "list", "stats", "export",
];
const WRITE_TOOLS: [&str; 2] = ["remember", "forget"];

pub fn requires_write(name: &str) -> bool {
    WRITE_TOOLS.contains(&name)
}
const LIST_PREVIEW_CHARS: usize = 240;

/// Which tools are exposed and how namespaces are chosen.
#[derive(Clone)]
pub struct Exposure {
    pub enabled: BTreeSet<&'static str>,
    pub namespace_arg: bool,
    pub default_namespace: String,
}

impl Exposure {
    pub fn new(
        requested: &[String],
        read_only: bool,
        namespace_arg: bool,
        default_namespace: String,
    ) -> Result<Self, String> {
        let mut enabled: BTreeSet<&'static str> = if requested.is_empty() {
            ALL_TOOLS.into_iter().collect()
        } else {
            let mut set = BTreeSet::new();
            for name in requested {
                let known = ALL_TOOLS
                    .iter()
                    .find(|t| **t == name.trim())
                    .ok_or_else(|| {
                        format!(
                            "unknown tool {name:?}; known tools: {}",
                            ALL_TOOLS.join(", ")
                        )
                    })?;
                set.insert(*known);
            }
            set
        };
        if read_only {
            enabled.retain(|t| !WRITE_TOOLS.contains(t));
        }
        if enabled.is_empty() {
            return Err("no tools remain enabled".into());
        }
        Ok(Self {
            enabled,
            namespace_arg,
            default_namespace,
        })
    }

    pub fn resolve_namespace(&self, requested: Option<&str>) -> Res<String> {
        match requested {
            None => Ok(self.default_namespace.clone()),
            Some(ns) if !self.namespace_arg => Err(Fail::new(
                "forbidden",
                "this server does not accept a namespace argument",
            )
            .with(json!({"namespace": ns}))),
            Some(ns) if crate::config::valid_namespace(ns) => Ok(ns.to_string()),
            Some(_) => Err(Fail::invalid(
                "namespace is 1 to 64 characters of [a-z0-9_-]",
            )),
        }
    }
}

fn schema(exposure: &Exposure, mut properties: Value, required: &[&str]) -> Value {
    if exposure.namespace_arg {
        properties["namespace"] = json!({
            "type": "string",
            "description": "Store to use; omit for the server's default. 1 to 64 characters of [a-z0-9_-].",
            "pattern": "^[a-z0-9_-]{1,64}$"
        });
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

fn output_schema(properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

fn memory_schema() -> Value {
    output_schema(
        json!({
            "id": {"type": "string", "description": "Stable memory id."},
            "text": {"type": "string", "description": "Full content, or a preview when `text_truncated` is true."},
            "text_truncated": {"type": "boolean", "description": "Whether text is a preview."},
            "bytes": {"type": "integer", "description": "Length of the full text in UTF-8 bytes."},
            "title": {"type": "string", "description": "Optional display title."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Labels attached to the memory."},
            "metadata": {"type": "object", "description": "Stored user metadata.", "additionalProperties": true},
            "created_at": {"type": "integer", "description": "Creation time in Unix milliseconds."},
            "updated_at": {"type": "integer", "description": "Last update time in Unix milliseconds."},
            "version": {"type": "string", "description": "Pass to `remember.if_version` for a conditional update."}
        }),
        &[
            "id",
            "text",
            "bytes",
            "tags",
            "metadata",
            "created_at",
            "updated_at",
            "version",
        ],
    )
}

fn hit_schema() -> Value {
    output_schema(
        json!({
            "memory": memory_schema(),
            "score": {"type": "number", "description": "Relative combined rank score, not a probability."},
            "lexical_score": {"type": "number", "description": "Lexical ranking contribution."},
            "semantic_score": {"type": "number", "description": "Semantic ranking contribution when a model is configured."}
        }),
        &["memory", "score", "lexical_score"],
    )
}

/// Tool descriptors, serialized as MCP `Tool` objects.
pub fn definitions(exposure: &Exposure, limits: &Limits) -> Vec<Value> {
    let tags = json!({"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 64},
        "maxItems": limits.max_tags, "description": "Labels; filters match memories carrying every listed tag."});
    let filter = json!({"type": "object", "description": "Exact-match filter on top-level metadata keys.",
        "additionalProperties": true});
    let all = vec![
        json!({
            "name": "remember",
            "title": "Store memory",
            "description": "Store a durable fact, decision or preference; use `recall` to find one instead. Omit `id` to create a memory; an existing `id` replaces its text, title, tags and metadata, clearing optional fields you omit. Use `if_version` with a known id to reject concurrent changes, or `create_only` to reject overwrites.",
            "inputSchema": schema(exposure, json!({
                "text": {"type": "string", "maxLength": limits.max_text_bytes, "description": "UTF-8 memory content; must be nonblank unless `allow_empty` is true. The server also enforces a byte limit and a 4,096-word limit."},
                "id": {"type": "string", "minLength": 1, "maxLength": 256, "description": "Stable id; replaces an existing memory with this id."},
                "title": {"type": "string", "maxLength": 256, "description": "Optional display title; does not change the id."},
                "tags": tags,
                "metadata": {"type": "object", "description": format!("Arbitrary JSON object; keys starting with {:?} are reserved.", crate::memory::RESERVED), "additionalProperties": true},
                "if_version": {"type": "string", "description": "Version from `remember` or `get`; write only if the stored version still matches. Cannot combine with `create_only`."},
                "create_only": {"type": "boolean", "default": false, "description": "Reject the write if the chosen id already exists; cannot combine with `if_version`."},
                "allow_empty": {"type": "boolean", "default": false, "description": "Permit blank text for file-like use; otherwise blank text is rejected."}
            }), &["text"]),
            "outputSchema": output_schema(json!({
                "id": {"type": "string", "description": "Id of the stored memory."},
                "version": {"type": "string", "description": "New version for a later conditional update."},
                "created": {"type": "boolean", "description": "True when this call created an id."},
                "bytes": {"type": "integer", "description": "Stored text size in UTF-8 bytes."},
                "namespace": {"type": "string", "description": "Store namespace used for the write."}
            }), &["id", "version", "created", "bytes", "namespace"]),
            "annotations": {"title": "Store memory", "readOnlyHint": false, "destructiveHint": true, "idempotentHint": false, "openWorldHint": false}
        }),
        json!({
            "name": "recall",
            "title": "Recall matching memories",
            "description": "Retrieve full memories ranked for a natural-language question. `tags` require every listed label, and `min_score` drops low relative-score matches; use `search` for exact metadata-key filters, custom ranking weights or fusion, or text previews instead of full memories. Read-only; ranking is lexical by default, with semantic ranking when a model is configured.",
            "inputSchema": schema(exposure, json!({
                "query": {"type": "string", "minLength": 1, "maxLength": 4096, "description": "Natural-language question or phrase to match against stored memories; nonblank, at most 4,096 UTF-8 bytes."},
                "k": {"type": "integer", "minimum": 1, "maximum": limits.max_top_k, "default": 5, "description": "Maximum number of ranked memories to return."},
                "tags": tags,
                "min_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Drop hits below this relative rank score (0 to 1); this is not a probability."}
            }), &["query"]),
            "outputSchema": output_schema(json!({
                "count": {"type": "integer", "description": "Number of returned hits."},
                "results": {"type": "array", "items": hit_schema(), "description": "Ranked hits with full memory text."}
            }), &["count", "results"]),
            "annotations": {"title": "Recall matching memories", "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "search",
            "title": "Search memory with filters",
            "description": "Find memories with metadata filters or ranking controls; use `recall` for simpler full-text answers. Read-only. Results contain text previews and relative ranking scores, not probabilities; use `get` to read a result in full. Semantic scores require a configured model.",
            "inputSchema": schema(exposure, json!({
                "query": {"type": "string", "minLength": 1, "maxLength": 4096, "description": "Natural-language phrase to match against stored memories; nonblank, at most 4,096 UTF-8 bytes."},
                "k": {"type": "integer", "minimum": 1, "maximum": limits.max_top_k, "default": 10, "description": "Maximum number of ranked previews to return."},
                "tags": tags,
                "filter": filter,
                "min_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Drop hits below this relative rank score (0 to 1); this is not a probability."},
                "lexical_weight": {"type": "number", "minimum": 0, "maximum": 100, "description": "Weight for lexical BM25 ranking before fusion, from 0 to 100."},
                "semantic_weight": {"type": "number", "minimum": 0, "maximum": 100, "description": "Weight for model-based semantic ranking before fusion, from 0 to 100; requires a configured model."},
                "fusion": {"type": "string", "enum": ["blend", "rrf"], "description": "How to combine lexical and semantic ranks: weighted score blend or reciprocal rank fusion."}
            }), &["query"]),
            "outputSchema": output_schema(json!({
                "count": {"type": "integer", "description": "Number of returned hits."},
                "results": {"type": "array", "items": hit_schema(), "description": "Ranked hits with text previews and score details."}
            }), &["count", "results"]),
            "annotations": {"title": "Search memory with filters", "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "get",
            "title": "Get memory",
            "description": "Read one known memory id in full, including tags, metadata and version. Use an id from `list` or `search`; use `recall` when only a natural-language question is known. Read-only; an unknown id returns an error.",
            "inputSchema": schema(exposure, json!({"id": {"type": "string", "minLength": 1, "maxLength": 256, "description": "Exact id of the memory to read, as returned by `remember`, `list` or `search`."}}), &["id"]),
            "outputSchema": output_schema(json!({
                "memory": memory_schema()
            }), &["memory"]),
            "annotations": {"title": "Get memory", "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "forget",
            "title": "Delete memories",
            "description": "Delete memories from the active store by exact id(s) or id prefix; use `remember` to update content instead. Deletion has no undo: inspect prefix matches with `list` first and set `confirm: true` for prefix deletion. Exact ids and a prefix cannot be combined, and a batch failure may leave earlier ids deleted.",
            "inputSchema": schema(exposure, json!({
                "id": {"type": "string", "minLength": 1, "maxLength": 256, "description": "One exact memory id to delete; cannot combine with `prefix`."},
                "ids": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 256}, "maxItems": limits.max_forget, "description": "Exact memory ids to delete in one call; cannot combine with `prefix`."},
                "prefix": {"type": "string", "minLength": 1, "maxLength": 256, "description": "Delete every id starting with this prefix; use `list` with the same prefix first."},
                "confirm": {"type": "boolean", "description": "Set true to authorize prefix deletion; exact ids do not need it."}
            }), &[]),
            "outputSchema": output_schema(json!({
                "deleted": {"type": "integer", "description": "Number of memories deleted."},
                "missing": {"type": "array", "items": {"type": "string"}, "description": "Requested ids that were not present."}
            }), &["deleted", "missing"]),
            "annotations": {"title": "Delete memories", "readOnlyHint": false, "destructiveHint": true, "idempotentHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "list",
            "title": "List memories",
            "description": "Browse memories in id order, optionally limited by id prefix or tag; use `search` for relevance ranking. Read-only. Returns text previews unless `full` is true; pass `next_cursor` as `cursor` to continue.",
            "inputSchema": schema(exposure, json!({
                "prefix": {"type": "string", "maxLength": 256, "description": "Include only memory ids starting with this prefix."},
                "tag": {"type": "string", "minLength": 1, "maxLength": 64, "description": "Include only memories carrying this tag."},
                "cursor": {"type": "string", "maxLength": 256, "description": "`next_cursor` of the previous page."},
                "limit": {"type": "integer", "minimum": 1, "maximum": limits.max_page, "default": 50, "description": "Maximum memories in this page."},
                "full": {"type": "boolean", "default": false, "description": "Return full text instead of a 240-character preview."}
            }), &[]),
            "outputSchema": output_schema(json!({
                "count": {"type": "integer", "description": "Number of memories in this page."},
                "memories": {"type": "array", "items": memory_schema(), "description": "Memories in id order, with full text or previews."},
                "next_cursor": {"type": ["string", "null"], "description": "Pass a string value as `cursor` to continue; null ends the listing."}
            }), &["count", "memories", "next_cursor"]),
            "annotations": {"title": "List memories", "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "stats",
            "title": "Store statistics",
            "description": "Inspect store counts, recall mode, encryption status, storage size and active limits without reading memory contents. Read-only; use `list` to inspect stored memories.",
            "inputSchema": schema(exposure, json!({}), &[]),
            "outputSchema": output_schema(json!({
                "namespace": {"type": "string", "description": "Store namespace being inspected."},
                "store_path": {"type": "string", "description": "Local filesystem path of the store."},
                "memories": {"type": "integer", "description": "Number of stored memories."},
                "embedded_memories": {"type": "integer", "description": "Memories with a semantic embedding."},
                "recall": {"type": "string", "description": "Active lexical or hybrid recall mode."},
                "encryption_at_rest": {"type": "boolean", "description": "Whether the store is encrypted at rest."},
                "storage_bytes": {"type": "integer", "description": "Approximate local store size in bytes."},
                "dimensions": {"type": "integer", "description": "Configured vector dimensions."},
                "limits": {"type": "object", "description": "Active text, metadata, tag, result and export caps.", "additionalProperties": {"type": "integer"}},
                "server_version": {"type": "string", "description": "Version of the core server binary."}
            }), &["namespace", "store_path", "memories", "embedded_memories", "recall", "encryption_at_rest", "storage_bytes", "dimensions", "limits", "server_version"]),
            "annotations": {"title": "Store statistics", "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "export",
            "title": "Export memory contents",
            "description": "Export a portable copy of many memories as JSON, JSONL or Markdown. Use `get` for one known memory, `recall` or `search` to find matches, and `list` for short previews. The text payload is capped by the export byte limit; pass `next_cursor` as `cursor` for more. Pages read the live store, so concurrent writes can change later pages.",
            "inputSchema": schema(exposure, json!({
                "format": {"type": "string", "enum": ["json", "jsonl", "markdown"], "default": "json", "description": "Serialization of the returned text payload."},
                "prefix": {"type": "string", "maxLength": 256, "description": "Include only memory ids starting with this prefix."},
                "tag": {"type": "string", "minLength": 1, "maxLength": 64, "description": "Include only memories carrying this tag."},
                "cursor": {"type": "string", "maxLength": 256, "description": "`next_cursor` from the previous export segment."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 1000, "description": "Maximum memories in this export segment, also bounded by the server's byte limit."}
            }), &[]),
            "outputSchema": output_schema(json!({
                "format": {"type": "string", "description": "Serialization used for the text payload."},
                "count": {"type": "integer", "description": "Number of exported memories in this segment."},
                "bytes": {"type": "integer", "description": "Size of the returned text payload in bytes."},
                "next_cursor": {"type": ["string", "null"], "description": "Pass a string value as `cursor` for the next segment; null ends the export."}
            }), &["format", "count", "bytes", "next_cursor"]),
            "annotations": {"title": "Export memory contents", "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
        }),
    ];
    all.into_iter()
        .filter(|t| {
            t["name"]
                .as_str()
                .is_some_and(|n| exposure.enabled.contains(n))
        })
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RememberArgs {
    text: String,
    id: Option<String>,
    title: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    metadata: Map<String, Value>,
    if_version: Option<String>,
    #[serde(default)]
    create_only: bool,
    #[serde(default)]
    allow_empty: bool,
    namespace: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecallArgs {
    query: String,
    k: Option<usize>,
    #[serde(default)]
    tags: Vec<String>,
    min_score: Option<f64>,
    namespace: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    query: String,
    k: Option<usize>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    filter: Map<String, Value>,
    min_score: Option<f64>,
    lexical_weight: Option<f64>,
    semantic_weight: Option<f64>,
    fusion: Option<String>,
    namespace: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GetArgs {
    id: String,
    namespace: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForgetArgs {
    id: Option<String>,
    #[serde(default)]
    ids: Vec<String>,
    prefix: Option<String>,
    #[serde(default)]
    confirm: bool,
    namespace: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListArgs {
    prefix: Option<String>,
    tag: Option<String>,
    cursor: Option<String>,
    limit: Option<usize>,
    #[serde(default)]
    full: bool,
    namespace: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StatsArgs {
    namespace: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportArgs {
    format: Option<String>,
    prefix: Option<String>,
    tag: Option<String>,
    cursor: Option<String>,
    limit: Option<usize>,
    namespace: Option<String>,
}

/// The result of a successful call: structured data plus optional text for the client to show.
pub struct Output {
    pub structured: Value,
    pub text: Option<String>,
}

impl Output {
    fn data(structured: Value) -> Self {
        Self {
            structured,
            text: None,
        }
    }
}

fn parse<T: for<'de> Deserialize<'de>>(arguments: Option<Map<String, Value>>) -> Res<T> {
    serde_json::from_value(Value::Object(arguments.unwrap_or_default()))
        .map_err(|e| Fail::invalid(format!("invalid arguments: {e}")))
}

pub fn memory_json(memory: &Memory) -> Value {
    serde_json::to_value(memory).unwrap_or(Value::Null)
}

/// Runs one tool call. Blocking: callers run it on a blocking thread.
pub fn call(
    backend: &WarmBackend,
    exposure: &Exposure,
    limits: &Limits,
    name: &str,
    arguments: Option<Map<String, Value>>,
) -> Res<Output> {
    if !exposure.enabled.contains(name) {
        return Err(Fail::new(
            "unknown_tool",
            format!("tool {name:?} is not available on this server"),
        ));
    }
    match name {
        "remember" => {
            let a: RememberArgs = parse(arguments)?;
            let ns = exposure.resolve_namespace(a.namespace.as_deref())?;
            let req = RememberReq {
                text: a.text,
                id: a.id,
                title: a.title,
                tags: a.tags,
                metadata: a.metadata,
                if_version: a.if_version,
                create_only: a.create_only,
                allow_empty: a.allow_empty,
            };
            req.validate(limits)?;
            let out = backend.remember(&ns, req)?;
            Ok(Output::data(serde_json::to_value(out).unwrap_or_default()))
        }
        "recall" | "search" => {
            let is_recall = name == "recall";
            let (ns, req) = if is_recall {
                let a: RecallArgs = parse(arguments)?;
                (
                    exposure.resolve_namespace(a.namespace.as_deref())?,
                    SearchReq {
                        query: a.query,
                        k: a.k.unwrap_or(5),
                        tags: a.tags,
                        filter: Map::new(),
                        min_score: a.min_score,
                        lexical_weight: None,
                        semantic_weight: None,
                        fusion: None,
                    },
                )
            } else {
                let a: SearchArgs = parse(arguments)?;
                (
                    exposure.resolve_namespace(a.namespace.as_deref())?,
                    SearchReq {
                        query: a.query,
                        k: a.k.unwrap_or(10),
                        tags: a.tags,
                        filter: a.filter,
                        min_score: a.min_score,
                        lexical_weight: a.lexical_weight,
                        semantic_weight: a.semantic_weight,
                        fusion: a.fusion,
                    },
                )
            };
            req.validate(limits)?;
            let hits = backend.search(&ns, req)?;
            let results: Vec<Value> = hits
                .iter()
                .map(|h| {
                    let mut value = serde_json::to_value(h).unwrap_or_default();
                    if !is_recall {
                        let (text, truncated) = preview(&h.memory.text, LIST_PREVIEW_CHARS);
                        value["memory"]["text"] = json!(text);
                        value["memory"]["text_truncated"] = json!(truncated);
                    }
                    value
                })
                .collect();
            Ok(Output::data(
                json!({"count": results.len(), "results": results}),
            ))
        }
        "get" => {
            let a: GetArgs = parse(arguments)?;
            crate::memory::validate_id(&a.id)?;
            let ns = exposure.resolve_namespace(a.namespace.as_deref())?;
            let memory = backend
                .get(&ns, &a.id)?
                .ok_or_else(|| Fail::not_found(&a.id))?;
            Ok(Output::data(json!({"memory": memory_json(&memory)})))
        }
        "forget" => {
            let a: ForgetArgs = parse(arguments)?;
            let ns = exposure.resolve_namespace(a.namespace.as_deref())?;
            let mut ids = a.ids;
            ids.extend(a.id);
            if a.prefix.is_some() && !a.confirm {
                return Err(Fail::invalid(
                    "deleting by prefix needs confirm: true; call `list` with the prefix first to see what matches",
                ));
            }
            let req = ForgetReq {
                ids,
                prefix: a.prefix,
            };
            req.validate(limits)?;
            let out = backend.forget(&ns, req)?;
            Ok(Output::data(serde_json::to_value(out).unwrap_or_default()))
        }
        "list" => {
            let a: ListArgs = parse(arguments)?;
            let ns = exposure.resolve_namespace(a.namespace.as_deref())?;
            let req = ListReq {
                prefix: a.prefix,
                tag: a.tag,
                cursor: a.cursor,
                limit: a.limit.unwrap_or(50),
            };
            req.validate(limits)?;
            let page = backend.list(&ns, req)?;
            let memories: Vec<Value> = page
                .memories
                .iter()
                .map(|m| {
                    let mut value = memory_json(m);
                    if !a.full {
                        let (text, truncated) = preview(&m.text, LIST_PREVIEW_CHARS);
                        value["text"] = json!(text);
                        value["text_truncated"] = json!(truncated);
                    }
                    value
                })
                .collect();
            Ok(Output::data(json!({
                "count": memories.len(),
                "memories": memories,
                "next_cursor": page.next_cursor,
            })))
        }
        "stats" => {
            let a: StatsArgs = parse(arguments)?;
            let ns = exposure.resolve_namespace(a.namespace.as_deref())?;
            Ok(Output::data(backend.stats(&ns)?))
        }
        "export" => {
            let a: ExportArgs = parse(arguments)?;
            let ns = exposure.resolve_namespace(a.namespace.as_deref())?;
            export(backend, limits, &ns, a)
        }
        other => Err(Fail::new(
            "unknown_tool",
            format!("tool {other:?} is not available on this server"),
        )),
    }
}

fn export(backend: &WarmBackend, limits: &Limits, ns: &str, a: ExportArgs) -> Res<Output> {
    let format = a.format.as_deref().unwrap_or("json");
    if !["json", "jsonl", "markdown"].contains(&format) {
        return Err(Fail::invalid("format must be json, jsonl or markdown"));
    }
    let wanted = a.limit.unwrap_or(1000);
    if !(1..=10_000).contains(&wanted) {
        return Err(Fail::invalid("limit must be 1..=10000"));
    }
    let mut cursor = a.cursor;
    let mut taken: Vec<Memory> = Vec::new();
    let mut rendered: Vec<String> = Vec::new();
    let mut bytes = 0usize;
    let mut next_cursor = None;
    'pages: loop {
        let req = ListReq {
            prefix: a.prefix.clone(),
            tag: a.tag.clone(),
            cursor: cursor.clone(),
            limit: limits.max_page.min(wanted - taken.len()).max(1),
        };
        req.validate(limits)?;
        let page = backend.list(ns, req)?;
        for memory in page.memories {
            let piece = match format {
                "markdown" => markdown_entry(&memory),
                _ => serde_json::to_string(&memory_json(&memory)).map_err(Fail::internal)?,
            };
            if !taken.is_empty() && bytes + piece.len() + 2 > limits.max_export_bytes {
                next_cursor = taken.last().map(|m| m.id.clone());
                break 'pages;
            }
            bytes += piece.len() + 2;
            rendered.push(piece);
            taken.push(memory);
            if taken.len() == wanted {
                // A full export reports a cursor only when more memories follow.
                let last = taken.last().map(|m| m.id.clone());
                let more = backend.list(
                    ns,
                    ListReq {
                        prefix: a.prefix.clone(),
                        tag: a.tag.clone(),
                        cursor: last.clone(),
                        limit: 1,
                    },
                )?;
                next_cursor = (!more.memories.is_empty()).then_some(last).flatten();
                break 'pages;
            }
        }
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    let payload = match format {
        "json" => format!("[{}]", rendered.join(",")),
        "jsonl" => rendered.join("\n"),
        _ => rendered.join("\n"),
    };
    Ok(Output {
        structured: json!({
            "format": format,
            "count": taken.len(),
            "bytes": payload.len(),
            "next_cursor": next_cursor,
        }),
        text: Some(payload),
    })
}

fn markdown_entry(m: &Memory) -> String {
    let heading = m.title.as_deref().unwrap_or(&m.id);
    let mut out = format!(
        "## {heading}\n\n- id: `{}`\n- version: `{}`\n",
        m.id, m.version
    );
    if !m.tags.is_empty() {
        let _ = writeln!(out, "- tags: {}", m.tags.join(", "));
    }
    if !m.metadata.is_empty() {
        let _ = writeln!(
            out,
            "- metadata: `{}`",
            serde_json::to_string(&m.metadata).unwrap_or_default()
        );
    }
    out.push('\n');
    out.push_str(&m.text);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::test_limits;

    fn exposure(tools: &[&str], read_only: bool, namespace_arg: bool) -> Result<Exposure, String> {
        let requested: Vec<String> = tools.iter().map(|t| (*t).to_string()).collect();
        Exposure::new(&requested, read_only, namespace_arg, "default".into())
    }

    #[test]
    fn exposure_defaults_read_only_and_unknown_names() {
        assert_eq!(exposure(&[], false, false).unwrap().enabled.len(), 8);
        let ro = exposure(&[], true, false).unwrap();
        assert!(!ro.enabled.contains("remember") && !ro.enabled.contains("forget"));
        assert_eq!(ro.enabled.len(), 6);
        assert!(exposure(&["recall", "nope"], false, false).is_err());
        assert!(
            exposure(&["remember"], true, false).is_err(),
            "nothing left to expose"
        );
    }

    #[test]
    fn definitions_follow_exposure_and_namespace_argument() {
        let limits = test_limits();
        let all = definitions(&exposure(&[], false, false).unwrap(), &limits);
        assert_eq!(all.len(), 8);
        for tool in &all {
            let schema = &tool["inputSchema"];
            assert_eq!(schema["type"], "object");
            assert_eq!(schema["additionalProperties"], false);
            assert!(schema["properties"].get("namespace").is_none());
            for property in schema["properties"].as_object().unwrap().values() {
                assert!(property["description"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty()));
            }
            assert!(tool["outputSchema"]["properties"].is_object());
            let _: rmcp::model::Tool = serde_json::from_value(tool.clone()).unwrap();
        }
        let remember = all.iter().find(|t| t["name"] == "remember").unwrap();
        assert_eq!(
            remember["inputSchema"]["properties"]["text"]["maxLength"],
            100
        );
        let with_ns = definitions(&exposure(&["stats"], false, true).unwrap(), &limits);
        assert_eq!(with_ns.len(), 1);
        assert!(with_ns[0]["inputSchema"]["properties"]["namespace"]["pattern"].is_string());
    }

    #[test]
    fn namespace_resolution_is_strict() {
        let closed = exposure(&[], false, false).unwrap();
        assert_eq!(closed.resolve_namespace(None).unwrap(), "default");
        assert_eq!(
            closed.resolve_namespace(Some("x")).unwrap_err().code,
            "forbidden"
        );
        let open = exposure(&[], false, true).unwrap();
        assert_eq!(
            open.resolve_namespace(Some("team-1_a")).unwrap(),
            "team-1_a"
        );
        for bad in ["", "UP", "a/b", "..", "a b", &"x".repeat(65)] {
            assert_eq!(
                open.resolve_namespace(Some(bad)).unwrap_err().code,
                "invalid_argument",
                "{bad:?}"
            );
        }
    }
}
