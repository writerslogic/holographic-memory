// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Tool definitions and handlers. Handlers parse strictly (unknown fields are errors), validate
//! against `Limits`, and call only the `HmsBackend` methods.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::memory::{
    preview, Fail, ForgetReq, Limits, ListReq, Memory, RememberReq, Res, SearchReq,
};
use crate::store::HmsBackend;

pub const ALL_TOOLS: [&str; 8] = [
    "remember", "recall", "search", "get", "forget", "list", "stats", "export",
];
const WRITE_TOOLS: [&str; 2] = ["remember", "forget"];
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

/// Tool descriptors, serialized as MCP `Tool` objects.
pub fn definitions(exposure: &Exposure, limits: &Limits) -> Vec<Value> {
    let tags = json!({"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 64},
        "maxItems": limits.max_tags, "description": "Labels; filters match memories carrying every listed tag."});
    let filter = json!({"type": "object", "description": "Exact-match filter on top-level metadata keys.",
        "additionalProperties": true});
    let all = vec![
        json!({
            "name": "remember",
            "title": "Remember",
            "description": "Store or replace a memory. Without `id` a new id is generated; with an existing `id` the memory is replaced. Use `if_version` for compare-and-set and `create_only` to refuse overwrites.",
            "inputSchema": schema(exposure, json!({
                "text": {"type": "string", "maxLength": limits.max_text_bytes, "description": "The memory."},
                "id": {"type": "string", "minLength": 1, "maxLength": 256, "description": "Stable id; replaces an existing memory with this id."},
                "title": {"type": "string", "maxLength": 256},
                "tags": tags,
                "metadata": {"type": "object", "description": "Arbitrary JSON object; keys starting with \"hms.\" are reserved.", "additionalProperties": true},
                "if_version": {"type": "string", "description": "Write only if the stored memory still has this version."},
                "create_only": {"type": "boolean", "default": false},
                "allow_empty": {"type": "boolean", "default": false, "description": "Permit blank text (file-like use)."}
            }), &["text"]),
            "annotations": {"title": "Remember", "readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": false}
        }),
        json!({
            "name": "recall",
            "title": "Recall",
            "description": "Retrieve the memories most relevant to a natural-language query, best first, with their full text.",
            "inputSchema": schema(exposure, json!({
                "query": {"type": "string", "minLength": 1, "maxLength": 4096},
                "k": {"type": "integer", "minimum": 1, "maximum": limits.max_top_k, "default": 5},
                "tags": tags,
                "min_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Drop hits ranked below this relative score."}
            }), &["query"]),
            "annotations": {"title": "Recall", "readOnlyHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "search",
            "title": "Search",
            "description": "Search with explicit control: metadata filter, tags, lexical and semantic weights, rank fusion. Returns previews and the score breakdown; use `get` for full text.",
            "inputSchema": schema(exposure, json!({
                "query": {"type": "string", "minLength": 1, "maxLength": 4096},
                "k": {"type": "integer", "minimum": 1, "maximum": limits.max_top_k, "default": 10},
                "tags": tags,
                "filter": filter,
                "min_score": {"type": "number", "minimum": 0, "maximum": 1},
                "lexical_weight": {"type": "number", "minimum": 0, "maximum": 100},
                "semantic_weight": {"type": "number", "minimum": 0, "maximum": 100},
                "fusion": {"type": "string", "enum": ["blend", "rrf"]}
            }), &["query"]),
            "annotations": {"title": "Search", "readOnlyHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "get",
            "title": "Get memory",
            "description": "Read one memory by id, with its full text, tags, metadata and version.",
            "inputSchema": schema(exposure, json!({"id": {"type": "string", "minLength": 1, "maxLength": 256}}), &["id"]),
            "annotations": {"title": "Get memory", "readOnlyHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "forget",
            "title": "Forget",
            "description": "Delete memories by id, or every memory whose id starts with `prefix` (requires `confirm: true`).",
            "inputSchema": schema(exposure, json!({
                "id": {"type": "string", "minLength": 1, "maxLength": 256},
                "ids": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 256}, "maxItems": limits.max_forget},
                "prefix": {"type": "string", "minLength": 1, "maxLength": 256},
                "confirm": {"type": "boolean", "description": "Required with `prefix`."}
            }), &[]),
            "annotations": {"title": "Forget", "readOnlyHint": false, "destructiveHint": true, "idempotentHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "list",
            "title": "List memories",
            "description": "Page through memories in id order, optionally by id prefix or tag. Returns text previews unless `full` is true; continue with `next_cursor`.",
            "inputSchema": schema(exposure, json!({
                "prefix": {"type": "string", "maxLength": 256},
                "tag": {"type": "string", "minLength": 1, "maxLength": 64},
                "cursor": {"type": "string", "maxLength": 256, "description": "`next_cursor` of the previous page."},
                "limit": {"type": "integer", "minimum": 1, "maximum": limits.max_page, "default": 50},
                "full": {"type": "boolean", "default": false}
            }), &[]),
            "annotations": {"title": "List memories", "readOnlyHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "stats",
            "title": "Store statistics",
            "description": "Counts, recall mode, encryption status, storage size and the active limits of the store.",
            "inputSchema": schema(exposure, json!({}), &[]),
            "annotations": {"title": "Store statistics", "readOnlyHint": true, "openWorldHint": false}
        }),
        json!({
            "name": "export",
            "title": "Export memories",
            "description": "Export memories as json, jsonl or markdown. Output is size-bounded; when `next_cursor` is returned, call again with it to continue.",
            "inputSchema": schema(exposure, json!({
                "format": {"type": "string", "enum": ["json", "jsonl", "markdown"], "default": "json"},
                "prefix": {"type": "string", "maxLength": 256},
                "tag": {"type": "string", "minLength": 1, "maxLength": 64},
                "cursor": {"type": "string", "maxLength": 256},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 1000}
            }), &[]),
            "annotations": {"title": "Export memories", "readOnlyHint": true, "openWorldHint": false}
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
    backend: &HmsBackend,
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

fn export(backend: &HmsBackend, limits: &Limits, ns: &str, a: ExportArgs) -> Res<Output> {
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
