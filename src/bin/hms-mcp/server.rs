// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! The rmcp `ServerHandler`: tools, resources and prompts over an `HmsBackend`.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    GetPromptRequestParams, GetPromptResponse, ListPromptsResult, ListResourceTemplatesResult,
    ListResourcesResult, ListToolsResult, PaginatedRequestParams, ReadResourceRequestParams,
    ReadResourceResponse, ServerCapabilities, ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler};
use serde::de::DeserializeOwned;
use serde_json::{json, Map, Value};
use tokio::sync::Semaphore;

use crate::memory::{preview, Fail, Limits, ListReq, SearchReq};
use crate::store::HmsBackend;
use crate::tools::{self, Exposure};

const INSTRUCTIONS: &str = "Persistent memory backed by Holographic Memory, stored on this machine. \
Use `recall` before answering questions that earlier conversations may have settled, and `remember` \
for durable facts, decisions and preferences (one fact or topic per memory). `search` gives explicit \
control over filters and ranking; `list`, `get`, `forget`, `stats` and `export` manage the store. \
This is the document-memory profile: it supports version-conditional writes and document pagination. \
It does not expose the temporal Memory API's asOf, knownAt or supersession operations.";

const RESOURCE_PAGE: usize = 100;
const MEMORY_URI: &str = "hms://memory/";
const STATS_URI: &str = "hms://stats";

pub struct Inner {
    pub backend: Arc<HmsBackend>,
    pub exposure: Exposure,
    pub limits: Limits,
    pub resources: bool,
    pub prompts: bool,
    pub gate: Arc<Semaphore>,
    pending: Arc<Semaphore>,
    queue_timeout: Duration,
}

pub struct WorkLimits {
    pub concurrent: usize,
    pub pending: usize,
    pub queue_timeout: Duration,
}

#[derive(Clone)]
pub struct HmsMcp {
    pub inner: Arc<Inner>,
}

impl HmsMcp {
    pub fn new(
        backend: Arc<HmsBackend>,
        exposure: Exposure,
        limits: Limits,
        resources: bool,
        prompts: bool,
        work: WorkLimits,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                backend,
                exposure,
                limits,
                resources,
                prompts,
                gate: Arc::new(Semaphore::new(work.concurrent)),
                pending: Arc::new(Semaphore::new(work.pending)),
                queue_timeout: work.queue_timeout,
            }),
        }
    }

    /// Run blocking store work on a worker thread, at most `max_concurrent` at a time.
    async fn run<T: Send + 'static>(
        &self,
        work: impl FnOnce(&Inner) -> T + Send + 'static,
    ) -> Result<T, ErrorData> {
        let busy = || {
            ErrorData::internal_error(
                "server busy; retry the operation",
                Some(json!({"code":"server_busy", "retryable":true})),
            )
        };
        let permit = match self.inner.gate.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                let _pending = self
                    .inner
                    .pending
                    .clone()
                    .try_acquire_owned()
                    .map_err(|_| busy())?;
                tokio::time::timeout(
                    self.inner.queue_timeout,
                    self.inner.gate.clone().acquire_owned(),
                )
                .await
                .map_err(|_| busy())?
                .map_err(|_| busy())?
            }
        };
        let inner = Arc::clone(&self.inner);
        let out = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work(&inner)
        })
        .await;
        out.map_err(|e| {
            crate::log::error(&format!("worker failed: {e}"));
            ErrorData::internal_error("internal error; see the server log", None)
        })
    }
}

fn from_json<T: DeserializeOwned>(value: Value) -> Result<T, ErrorData> {
    serde_json::from_value(value).map_err(|e| {
        crate::log::error(&format!("protocol type mismatch: {e}"));
        ErrorData::internal_error("internal error; see the server log", None)
    })
}

fn fail_result(fail: &Fail) -> Result<CallToolResponse, ErrorData> {
    let structured = json!({"error": fail});
    let text = serde_json::to_string(&structured).unwrap_or_default();
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.is_error = Some(true);
    result.structured_content = Some(structured);
    Ok(result.into())
}

fn percent_encode(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for b in id.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = text.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn memory_uri(namespace: &str, default_namespace: &str, id: &str) -> String {
    if namespace == default_namespace {
        format!("{MEMORY_URI}{}", percent_encode(id))
    } else {
        format!("hms://namespace/{namespace}/memory/{}", percent_encode(id))
    }
}

/// Splits a memory URI into (namespace, id).
fn parse_memory_uri(uri: &str, exposure: &Exposure) -> Option<(String, String)> {
    if let Some(rest) = uri.strip_prefix(MEMORY_URI) {
        return Some((exposure.default_namespace.clone(), percent_decode(rest)?));
    }
    let rest = uri.strip_prefix("hms://namespace/")?;
    let (ns, id) = rest.split_once("/memory/")?;
    Some((ns.to_string(), percent_decode(id)?))
}

fn recall_text(hits: &[crate::memory::Hit]) -> String {
    if hits.is_empty() {
        return "(no matching memories)".into();
    }
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            let label = h.memory.title.as_deref().unwrap_or(&h.memory.id);
            format!(
                "[{}] {} (id {})\n{}",
                i + 1,
                label,
                h.memory.id,
                h.memory.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

impl ServerHandler for HmsMcp {
    fn get_info(&self) -> ServerConfig {
        let mut caps = serde_json::Map::new();
        caps.insert("tools".into(), json!({}));
        if self.inner.resources {
            caps.insert("resources".into(), json!({}));
        }
        if self.inner.prompts {
            caps.insert("prompts".into(), json!({}));
        }
        let caps: ServerCapabilities =
            serde_json::from_value(Value::Object(caps)).unwrap_or_default();
        let mut info = ServerConfig::new(caps).with_instructions(INSTRUCTIONS);
        info.server_info =
            rmcp::model::Implementation::new("holographic-memory", env!("CARGO_PKG_VERSION"))
                .with_title("Holographic Memory");
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let definitions = tools::definitions(&self.inner.exposure, &self.inner.limits);
        let list: Vec<rmcp::model::Tool> = from_json(Value::Array(definitions))?;
        Ok(ListToolsResult::with_all_items(list))
    }

    fn get_tool(&self, name: &str) -> Option<rmcp::model::Tool> {
        tools::definitions(&self.inner.exposure, &self.inner.limits)
            .into_iter()
            .find(|t| t["name"] == name)
            .and_then(|t| serde_json::from_value(t).ok())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let name = request.name.to_string();
        if !self.inner.exposure.enabled.contains(name.as_str()) {
            return Err(ErrorData::invalid_params(
                format!("unknown tool {name:?}"),
                None,
            ));
        }
        crate::log::debug(&format!("tool {name} called"));
        let arguments = request.arguments;
        let started = Instant::now();
        let tool = name.clone();
        let outcome = self
            .run(move |inner| {
                tools::call(
                    &inner.backend,
                    &inner.exposure,
                    &inner.limits,
                    &tool,
                    arguments,
                )
            })
            .await?;
        let elapsed = started.elapsed().as_millis();
        match outcome {
            Ok(out) => {
                crate::log::info(&format!("tool {name} ok in {elapsed} ms"));
                let text = out
                    .text
                    .unwrap_or_else(|| serde_json::to_string(&out.structured).unwrap_or_default());
                let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
                result.structured_content = Some(out.structured);
                Ok(result.into())
            }
            Err(fail) => {
                crate::log::info(&format!("tool {name} {} in {elapsed} ms", fail.code));
                fail_result(&fail)
            }
        }
    }

    async fn list_resources(
        &self,
        request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        if !self.inner.resources {
            return Ok(ListResourcesResult::default());
        }
        let cursor = request.and_then(|r| r.cursor);
        let first_page = cursor.is_none();
        let page = self
            .run(move |inner| {
                let ns = inner.exposure.default_namespace.clone();
                inner.backend.list(
                    &ns,
                    ListReq {
                        cursor: cursor.clone(),
                        limit: RESOURCE_PAGE.min(inner.limits.max_page),
                        ..ListReq::default()
                    },
                )
            })
            .await?
            .map_err(|f| ErrorData::internal_error(f.message, None))?;
        let ns = self.inner.exposure.default_namespace.clone();
        let mut items: Vec<Value> = Vec::with_capacity(page.memories.len() + 1);
        if first_page {
            items.push(json!({
                "uri": STATS_URI, "name": "stats", "title": "Memory store statistics",
                "description": "Counts, recall mode, encryption status and limits.",
                "mimeType": "application/json"
            }));
        }
        for m in &page.memories {
            let (snippet, _) = preview(&m.text, 160);
            items.push(json!({
                "uri": memory_uri(&ns, &ns, &m.id),
                "name": m.id,
                "title": m.title.clone().unwrap_or_else(|| m.id.clone()),
                "description": snippet,
                "mimeType": "text/plain",
                "size": m.text.len(),
            }));
        }
        let mut result = ListResourcesResult::with_all_items(from_json(Value::Array(items))?);
        result.next_cursor = page.next_cursor;
        Ok(result)
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        if !self.inner.resources {
            return Ok(ListResourceTemplatesResult::default());
        }
        let mut templates = vec![json!({
            "uriTemplate": "hms://memory/{id}", "name": "memory", "title": "Memory by id",
            "description": "One memory's text; the id is percent-encoded.", "mimeType": "text/plain"
        })];
        if self.inner.exposure.namespace_arg {
            templates.push(json!({
                "uriTemplate": "hms://namespace/{namespace}/memory/{id}", "name": "namespaced-memory",
                "title": "Memory by namespace and id", "mimeType": "text/plain"
            }));
        }
        Ok(ListResourceTemplatesResult::with_all_items(from_json(
            Value::Array(templates),
        )?))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        if !self.inner.resources {
            return Err(ErrorData::resource_not_found(
                "resources are disabled",
                None,
            ));
        }
        let uri = request.uri;
        let not_found = |uri: &str| {
            ErrorData::resource_not_found(
                format!("no such resource {uri}"),
                Some(json!({"uri": uri})),
            )
        };
        if uri == STATS_URI {
            let stats = self
                .run(|inner| inner.backend.stats(&inner.exposure.default_namespace))
                .await?
                .map_err(|f| ErrorData::internal_error(f.message, None))?;
            return Ok(from_json::<rmcp::model::ReadResourceResult>(json!({
                "contents": [{"uri": uri, "mimeType": "application/json", "text": stats.to_string()}]
            }))?
            .into());
        }
        let Some((ns, id)) = parse_memory_uri(&uri, &self.inner.exposure) else {
            return Err(not_found(&uri));
        };
        let exposure = self.inner.exposure.clone();
        let lookup = {
            let (ns, id) = (ns.clone(), id.clone());
            self.run(move |inner| {
                let ns = exposure.resolve_namespace(
                    Some(&ns)
                        .filter(|n| **n != exposure.default_namespace)
                        .map(String::as_str),
                )?;
                crate::memory::validate_id(&id)?;
                inner.backend.get(&ns, &id)
            })
            .await?
        };
        match lookup {
            Ok(Some(memory)) => Ok(from_json::<rmcp::model::ReadResourceResult>(json!({
                "contents": [{
                    "uri": uri, "mimeType": "text/plain", "text": memory.text,
                    "_meta": {"id": memory.id, "title": memory.title, "tags": memory.tags,
                              "metadata": memory.metadata, "version": memory.version,
                              "createdAt": memory.created_at, "updatedAt": memory.updated_at}
                }]
            }))?
            .into()),
            Ok(None) => Err(not_found(&uri)),
            Err(f) if f.code == "invalid_argument" || f.code == "forbidden" => {
                Err(ErrorData::invalid_params(f.message, None))
            }
            Err(f) => Err(ErrorData::internal_error(f.message, None)),
        }
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, ErrorData> {
        let enabled = &self.inner.exposure.enabled;
        let mut prompts = Vec::new();
        if self.inner.prompts && enabled.contains("recall") {
            prompts.push(json!({
                "name": "recall-context", "title": "Recall context",
                "description": "Retrieve the memories relevant to a topic and ask the model to use them.",
                "arguments": [
                    {"name": "topic", "description": "What to recall.", "required": true},
                    {"name": "k", "description": "How many memories (default 5).", "required": false}
                ]
            }));
        }
        if self.inner.prompts && enabled.contains("remember") {
            prompts.push(json!({
                "name": "save-memory", "title": "Save a memory",
                "description": "Ask the model to store a fact or decision with the remember tool.",
                "arguments": [
                    {"name": "content", "description": "What to remember.", "required": true},
                    {"name": "tags", "description": "Comma-separated tags.", "required": false}
                ]
            }));
        }
        if self.inner.prompts && enabled.contains("list") {
            prompts.push(json!({
                "name": "review-memories", "title": "Review memories",
                "description": "Show stored memories and ask the model to propose consolidation.",
                "arguments": [{"name": "tag", "description": "Only memories with this tag.", "required": false}]
            }));
        }
        Ok(ListPromptsResult::with_all_items(from_json(Value::Array(
            prompts,
        ))?))
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResponse, ErrorData> {
        let args: Map<String, Value> = request.arguments.unwrap_or_default();
        let arg = |name: &str| args.get(name).and_then(Value::as_str).map(str::to_string);
        let enabled = &self.inner.exposure.enabled;
        let wrong = |what: &str| ErrorData::invalid_params(what.to_string(), None);
        let text = match request.name.as_str() {
            "recall-context" if self.inner.prompts && enabled.contains("recall") => {
                let topic = arg("topic").ok_or_else(|| wrong("argument `topic` is required"))?;
                let k = match arg("k") {
                    None => 5,
                    Some(k) => k
                        .parse::<usize>()
                        .map_err(|_| wrong("`k` must be an integer"))?,
                };
                let req = SearchReq {
                    query: topic.clone(),
                    k,
                    tags: vec![],
                    filter: Map::new(),
                    min_score: None,
                    lexical_weight: None,
                    semantic_weight: None,
                    fusion: None,
                };
                req.validate(&self.inner.limits)
                    .map_err(|f| wrong(&f.message))?;
                let hits = self
                    .run(move |inner| inner.backend.search(&inner.exposure.default_namespace, req))
                    .await?
                    .map_err(|f| ErrorData::internal_error(f.message, None))?;
                format!(
                    "Relevant memories for \"{topic}\":\n\n{}\n\nUse them where they apply; say so when a memory looks out of date.",
                    recall_text(&hits)
                )
            }
            "save-memory" if self.inner.prompts && enabled.contains("remember") => {
                let content =
                    arg("content").ok_or_else(|| wrong("argument `content` is required"))?;
                let tags = arg("tags")
                    .map(|t| format!(" with tags [{t}]"))
                    .unwrap_or_default();
                format!("Store this durable fact with the `remember` tool{tags}, as one memory:\n\n{content}")
            }
            "review-memories" if self.inner.prompts && enabled.contains("list") => {
                let tag = arg("tag");
                let limit = self.inner.limits.max_page.min(100);
                let tag_for_list = tag.clone();
                let page = self
                    .run(move |inner| {
                        inner.backend.list(
                            &inner.exposure.default_namespace,
                            ListReq {
                                tag: tag_for_list,
                                limit,
                                ..ListReq::default()
                            },
                        )
                    })
                    .await?
                    .map_err(|f| wrong(&f.message))?;
                let body = page
                    .memories
                    .iter()
                    .map(|m| format!("- {} ({}): {}", m.id, m.version, preview(&m.text, 200).0))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "These memories are stored{}:\n\n{}\n\nPropose duplicates to merge, outdated entries to delete and gaps to fill. Make no changes until I agree.",
                    tag.map(|t| format!(" under tag \"{t}\"")).unwrap_or_default(),
                    if body.is_empty() { "(none)".into() } else { body }
                )
            }
            other => return Err(wrong(&format!("unknown prompt {other:?}"))),
        };
        Ok(from_json::<rmcp::model::GetPromptResult>(json!({
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}]
        }))?
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::test_limits;
    use crate::store::{StoreOptions, StoreRoot};

    fn handler(dir: &std::path::Path, work: WorkLimits) -> HmsMcp {
        let limits = test_limits();
        HmsMcp::new(
            Arc::new(HmsBackend::new(
                StoreOptions {
                    root: StoreRoot::Namespaced(dir.to_path_buf()),
                    dim: 256,
                    max_stores: 1,
                    encrypt: None,
                    #[cfg(feature = "local-models")]
                    stages: None,
                    #[cfg(feature = "local-models")]
                    embedding_dim: None,
                },
                limits.clone(),
            )),
            Exposure::new(&[], false, false, "default".into()).unwrap(),
            limits,
            false,
            false,
            work,
        )
    }

    #[tokio::test]
    async fn queue_is_bounded_times_out_and_releases_cancelled_waiters() {
        use std::future::{poll_fn, Future};
        use std::task::Poll;

        let dir = tempfile::tempdir().unwrap();
        let handler = handler(
            dir.path(),
            WorkLimits {
                concurrent: 1,
                pending: 1,
                queue_timeout: Duration::from_millis(20),
            },
        );
        let executing = handler.inner.gate.clone().acquire_owned().await.unwrap();
        let mut pending = Box::pin(handler.run(|_| ()));
        poll_fn(|cx| {
            assert!(pending.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
        assert_eq!(handler.inner.pending.available_permits(), 0);
        let overloaded = handler
            .run(|_| panic!("overloaded work ran"))
            .await
            .unwrap_err();
        assert_eq!(overloaded.data.unwrap()["code"], "server_busy");
        drop(pending);
        assert_eq!(handler.inner.pending.available_permits(), 1);
        let expired = handler
            .run(|_| panic!("expired work ran"))
            .await
            .unwrap_err();
        assert_eq!(expired.data.unwrap()["code"], "server_busy");
        assert_eq!(handler.inner.pending.available_permits(), 1);
        drop(executing);
        assert_eq!(handler.run(|_| 42).await.unwrap(), 42);
    }

    #[tokio::test]
    async fn cancelled_request_keeps_work_slot_until_worker_finishes() {
        let dir = tempfile::tempdir().unwrap();
        let handler = handler(
            dir.path(),
            WorkLimits {
                concurrent: 1,
                pending: 1,
                queue_timeout: Duration::from_secs(5),
            },
        );
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker_handler = handler.clone();
        let request = tokio::spawn(async move {
            worker_handler
                .run(move |_| {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                })
                .await
        });
        started_rx.await.unwrap();
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        let slot_retained = handler.inner.gate.available_permits() == 0;
        release_tx.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), handler.run(|_| ()))
            .await
            .unwrap()
            .unwrap();
        assert!(
            slot_retained,
            "cancelled request released a running worker's slot"
        );
    }
}
