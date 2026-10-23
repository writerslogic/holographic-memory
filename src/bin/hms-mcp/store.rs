// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! `MemoryBackend` over the HMS document store: one engine store per namespace.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use holographic_memory::core::HmsConfig;
use holographic_memory::{chunk_document, DocumentInput, DocumentRecord, HmsCore, SearchOptions};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use crate::memory::{
    Fail, ForgetOut, ForgetReq, Hit, Limits, ListReq, Memory, Page, RememberOut, RememberReq, Res,
    SearchReq, RESERVED,
};

const K_CREATED: &str = "hms.created";
const K_UPDATED: &str = "hms.updated";
const K_TITLE: &str = "hms.title";
const K_TAGS: &str = "hms.tags";
const K_LEAD: &str = "hms.lead";
const K_TRAIL: &str = "hms.trail";
const K_BLANK: &str = "hms.blank";
const TAG_PREFIX: &str = "hms.tag.";
/// Placeholder word for a blank memory, which the document store would otherwise reject.
const BLANK_WORD: &str = "_";
const SCAN_PAGE: usize = 256;

pub struct StoreOptions {
    pub root: StoreRoot,
    pub dim: u32,
    pub max_stores: usize,
    pub encrypt: Option<String>,
    #[cfg(feature = "local-models")]
    pub stages: Option<Arc<holographic_memory::core::models::ModelStages>>,
    #[cfg(feature = "local-models")]
    pub embedding_dim: Option<usize>,
}

pub enum StoreRoot {
    /// Exactly this directory, for every namespace.
    Single(PathBuf),
    /// `<dir>/stores/<namespace>`.
    Namespaced(PathBuf),
}

struct Store {
    core: HmsCore,
    /// Serializes read-modify-write sequences (compare-and-set, create-only, prefix deletes).
    gate: Mutex<()>,
    dir: PathBuf,
}

pub struct HmsBackend {
    options: StoreOptions,
    limits: Limits,
    stores: Mutex<HashMap<String, Arc<Store>>>,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

fn fresh_version() -> String {
    format!("{:016x}", rand::random::<u64>())
}

pub fn generated_id() -> String {
    format!("mem-{:011x}-{:08x}", now_ms(), rand::random::<u32>())
}

impl HmsBackend {
    pub fn new(options: StoreOptions, limits: Limits) -> Self {
        Self {
            options,
            limits,
            stores: Mutex::new(HashMap::new()),
        }
    }

    fn store(&self, namespace: &str) -> Res<Arc<Store>> {
        let mut stores = self.stores.lock();
        if let Some(store) = stores.get(namespace) {
            return Ok(Arc::clone(store));
        }
        if stores.len() >= self.options.max_stores {
            return Err(Fail::new(
                "too_many_stores",
                format!("{} namespaces are already open", stores.len()),
            ));
        }
        let dir = match &self.options.root {
            StoreRoot::Single(dir) => dir.clone(),
            StoreRoot::Namespaced(root) => root.join("stores").join(namespace),
        };
        let core = self.open(&dir)?;
        let store = Arc::new(Store {
            core,
            gate: Mutex::new(()),
            dir,
        });
        stores.insert(namespace.to_string(), Arc::clone(&store));
        Ok(store)
    }

    fn open(&self, dir: &Path) -> Res<HmsCore> {
        let mut config = HmsConfig::default();
        #[cfg(feature = "local-models")]
        if let Some(dim) = self.options.embedding_dim {
            config.embedding_space = Some(holographic_memory::EmbeddingSpace {
                model: "qwen3-embedding".into(),
                revision: "local".into(),
                dimensions: dim,
                normalization: "l2".into(),
                metric: "cosine".into(),
            });
        }
        if let Some(env) = &self.options.encrypt {
            config.security.encryption_enabled = true;
            config.security.encryption_passphrase_env = Some(env.clone());
        }
        let core = HmsCore::new(
            self.options.dim,
            Some(dir.display().to_string()),
            Some(config),
        )
        .map_err(|e| {
            let text = format!("{e:#}");
            if text.contains("already open") {
                Fail::new(
                    "store_locked",
                    "the store is open in another process; run one hms-mcp with \
                         --transport http and point every client at it",
                )
                .with(json!({"store": dir.display().to_string()}))
            } else if text.contains("passphrase") || text.contains("decrypt") {
                Fail::new(
                    "store_unavailable",
                    "the store could not be opened with the configured encryption setting",
                )
            } else {
                Fail::internal(text)
            }
        })?;
        #[cfg(feature = "local-models")]
        core.set_model_stages(self.options.stages.clone())
            .map_err(Fail::internal)?;
        Ok(core)
    }

    /// Open the configured default store now, so locking and decryption problems surface at
    /// startup rather than on the first call.
    pub fn preopen(&self, namespace: &str) -> Res<()> {
        self.store(namespace).map(|_| ())
    }

    fn semantic(&self) -> bool {
        #[cfg(feature = "local-models")]
        {
            self.options.embedding_dim.is_some()
        }
        #[cfg(not(feature = "local-models"))]
        {
            false
        }
    }
}

fn tag_filter(tags: &[String], filter: &Map<String, Value>) -> Value {
    let mut map = filter.clone();
    for tag in tags {
        map.insert(format!("{TAG_PREFIX}{tag}"), Value::Bool(true));
    }
    Value::Object(map)
}

/// The text stored in the document index and the whitespace needed to restore the original.
fn encode_text(req: &RememberReq) -> (String, bool, Map<String, Value>) {
    let mut extra = Map::new();
    let trimmed = req.text.trim();
    if trimmed.is_empty() {
        extra.insert(K_BLANK.into(), Value::String(req.text.clone()));
        return (BLANK_WORD.into(), false, extra);
    }
    let lead = &req.text[..req.text.len() - req.text.trim_start().len()];
    let trail = &req.text[req.text.trim_end().len()..];
    if !lead.is_empty() {
        extra.insert(K_LEAD.into(), Value::String(lead.into()));
    }
    if !trail.is_empty() {
        extra.insert(K_TRAIL.into(), Value::String(trail.into()));
    }
    (trimmed.to_string(), true, extra)
}

fn ms_of(meta: &Map<String, Value>, key: &str) -> i64 {
    // f64 -> i64 `as` saturates and maps NaN to 0, which is the intended clamp here.
    meta.get(key)
        .and_then(Value::as_f64)
        .map_or(0, |v| v as i64)
}

fn memory_from(id: &str, text: Option<&str>, metadata: &Value, version: &str) -> Memory {
    let empty = Map::new();
    let meta = metadata.as_object().unwrap_or(&empty);
    let text = match meta.get(K_BLANK).and_then(Value::as_str) {
        Some(blank) => blank.to_string(),
        None => format!(
            "{}{}{}",
            meta.get(K_LEAD).and_then(Value::as_str).unwrap_or(""),
            text.unwrap_or(""),
            meta.get(K_TRAIL).and_then(Value::as_str).unwrap_or("")
        ),
    };
    Memory {
        id: id.to_string(),
        bytes: text.len(),
        text,
        title: meta
            .get(K_TITLE)
            .and_then(Value::as_str)
            .map(str::to_string),
        tags: meta
            .get(K_TAGS)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        metadata: meta
            .iter()
            .filter(|(k, _)| !k.starts_with(RESERVED))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        created_at: ms_of(meta, K_CREATED),
        updated_at: ms_of(meta, K_UPDATED),
        version: version.to_string(),
    }
}

fn record_memory(record: &DocumentRecord) -> Memory {
    memory_from(
        &record.id,
        record.text.as_deref(),
        &record.metadata,
        &record.version,
    )
}

fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    let mut visited = 0usize;
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            visited += 1;
            if visited > 100_000 {
                return total;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
}

impl HmsBackend {
    /// Flush every open store; called once at shutdown after in-flight work has finished.
    pub fn flush_all(&self) {
        for (namespace, store) in self.stores.lock().iter() {
            if let Err(e) = store.core.flush() {
                crate::log::error(&format!("flush of namespace {namespace} failed: {e:#}"));
            }
        }
    }

    pub fn remember(&self, namespace: &str, req: RememberReq) -> Res<RememberOut> {
        let store = self.store(namespace)?;
        let id = req.id.clone().unwrap_or_else(generated_id);
        let _write = store.gate.lock();
        let existing = store.core.get_document(&id);
        if req.create_only && existing.is_some() {
            return Err(Fail::new("already_exists", format!("memory {id:?} exists"))
                .with(json!({"id": id})));
        }
        if let Some(expected) = &req.if_version {
            let current = existing.as_ref().map(|d| d.version.as_str());
            if current != Some(expected.as_str()) {
                return Err(Fail::new(
                    "version_conflict",
                    "the memory changed since it was read; read it again and retry",
                )
                .with(json!({"id": id, "current_version": current})));
            }
        }
        let now = now_ms();
        let created = existing
            .as_ref()
            .and_then(|d| d.metadata.as_object())
            .map_or(now, |m| ms_of(m, K_CREATED));
        let (text, store_text, mut meta) = encode_text(&req);
        meta.extend(req.metadata.clone());
        // Millisecond timestamps are stored as floats: the store indexes every integer
        // metadata value for filtering, and a unique timestamp per memory would bloat it.
        meta.insert(K_CREATED.into(), json!(created as f64));
        meta.insert(K_UPDATED.into(), json!(now as f64));
        if let Some(title) = &req.title {
            meta.insert(K_TITLE.into(), json!(title));
        }
        let mut tags = req.tags.clone();
        tags.sort();
        tags.dedup();
        for tag in &tags {
            meta.insert(format!("{TAG_PREFIX}{tag}"), Value::Bool(true));
        }
        meta.insert(K_TAGS.into(), json!(tags));
        let version = fresh_version();
        let input = DocumentInput {
            id: id.clone(),
            text,
            source_uri: None,
            version: Some(version.clone()),
            metadata: Some(Value::Object(meta)),
            chunk_words: Some(4096),
            overlap_words: Some(0),
            store_text: Some(store_text),
            embeddings: None,
            title_weight: None,
        };
        match chunk_document(&input) {
            Ok(chunks) if chunks.len() == 1 => {}
            Ok(_) => {
                return Err(Fail::new(
                    "too_large",
                    "text does not fit one storage chunk (4096 words, 64 KiB); split it",
                ))
            }
            Err(e) => return Err(Fail::invalid(format!("invalid memory: {e}"))),
        }
        store
            .core
            .memorize_document(input)
            .map_err(Fail::internal)?;
        Ok(RememberOut {
            id,
            version,
            created: existing.is_none(),
            bytes: req.text.len(),
            namespace: namespace.to_string(),
        })
    }

    pub fn get(&self, namespace: &str, id: &str) -> Res<Option<Memory>> {
        let store = self.store(namespace)?;
        Ok(store.core.get_document(id).map(|r| record_memory(&r)))
    }

    pub fn search(&self, namespace: &str, req: SearchReq) -> Res<Vec<Hit>> {
        let store = self.store(namespace)?;
        let filter = tag_filter(&req.tags, &req.filter);
        let options = SearchOptions {
            k: Some(req.k as u32),
            candidate_limit: Some((req.k as u32 * 5).clamp(100, 4096)),
            filter: filter
                .as_object()
                .filter(|m| !m.is_empty())
                .map(|_| filter.clone()),
            lexical_weight: req.lexical_weight,
            semantic_weight: req.semantic_weight,
            fusion: req.fusion.clone(),
            ..SearchOptions::default()
        };
        let results = store
            .core
            .search_documents(&req.query, &options)
            .map_err(|e| Fail::invalid(format!("search failed: {e:#}")))?;
        Ok(results
            .into_iter()
            .filter(|r| req.min_score.is_none_or(|min| r.score >= min))
            .map(|r| Hit {
                memory: memory_from(&r.document_id, r.text.as_deref(), &r.metadata, &r.version),
                score: r.score,
                lexical_score: r.lexical_score,
                semantic_score: r.semantic_score,
            })
            .collect())
    }

    pub fn list(&self, namespace: &str, req: ListReq) -> Res<Page> {
        let store = self.store(namespace)?;
        let tag_key = req.tag.as_ref().map(|t| format!("{TAG_PREFIX}{t}"));
        let prefix = req.prefix.as_deref().unwrap_or("");
        // Start at the later of the prefix and the successor of the cursor.
        let mut from = match req.cursor.as_deref() {
            Some(cursor) if cursor >= prefix => format!("{cursor}\0"),
            _ => prefix.to_string(),
        };
        let mut memories = Vec::new();
        loop {
            let batch = store.core.list_documents(Some(&from), SCAN_PAGE);
            let exhausted = batch.len() < SCAN_PAGE;
            for record in &batch {
                if !record.id.starts_with(prefix) {
                    // Ids are ordered, so the prefix range has ended.
                    return Ok(Page {
                        memories,
                        next_cursor: None,
                    });
                }
                let tagged = tag_key
                    .as_ref()
                    .is_none_or(|k| record.metadata.get(k).and_then(Value::as_bool) == Some(true));
                if !tagged {
                    continue;
                }
                if memories.len() == req.limit {
                    let last = memories.last().map(|m: &Memory| m.id.clone());
                    return Ok(Page {
                        memories,
                        next_cursor: last,
                    });
                }
                memories.push(record_memory(record));
            }
            match batch.last() {
                Some(last) if !exhausted => from = format!("{}\0", last.id),
                _ => {
                    return Ok(Page {
                        memories,
                        next_cursor: None,
                    })
                }
            }
        }
    }

    pub fn forget(&self, namespace: &str, req: ForgetReq) -> Res<ForgetOut> {
        let store = self.store(namespace)?;
        let _write = store.gate.lock();
        let mut ids = req.ids.clone();
        if let Some(prefix) = &req.prefix {
            let mut from = prefix.clone();
            'scan: loop {
                let batch = store.core.list_documents(Some(&from), SCAN_PAGE);
                for record in &batch {
                    if !record.id.starts_with(prefix.as_str()) {
                        break 'scan;
                    }
                    if ids.len() >= self.limits.max_forget {
                        return Err(Fail::new(
                            "too_many",
                            format!(
                                "more than {} memories match the prefix; narrow it or raise --max-forget",
                                self.limits.max_forget
                            ),
                        ));
                    }
                    ids.push(record.id.clone());
                }
                match batch.last() {
                    Some(last) if batch.len() == SCAN_PAGE => from = format!("{}\0", last.id),
                    _ => break,
                }
            }
        }
        let mut deleted = 0;
        let mut missing = Vec::new();
        for id in ids {
            if store.core.delete_document(&id).map_err(Fail::internal)? {
                deleted += 1;
            } else {
                missing.push(id);
            }
        }
        Ok(ForgetOut { deleted, missing })
    }

    pub fn stats(&self, namespace: &str) -> Res<Value> {
        let store = self.store(namespace)?;
        let security = store.core.security_status();
        Ok(json!({
            "namespace": namespace,
            "store_path": store.dir.display().to_string(),
            "memories": store.core.document_count(),
            "embedded_memories": store.core.document_embedding_count(),
            "recall": if self.semantic() { "hybrid lexical and semantic" } else { "lexical (BM25) with metadata filters" },
            "encryption_at_rest": security["encryptionActive"],
            "storage_bytes": dir_size(&store.dir),
            "dimensions": store.core.dimensions(),
            "limits": {
                "max_text_bytes": self.limits.max_text_bytes,
                "max_metadata_bytes": self.limits.max_metadata_bytes,
                "max_tags": self.limits.max_tags,
                "max_top_k": self.limits.max_top_k,
                "max_page": self.limits.max_page,
                "max_export_bytes": self.limits.max_export_bytes,
            },
            "server_version": env!("CARGO_PKG_VERSION"),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::test_limits;

    fn backend(dir: &Path) -> HmsBackend {
        HmsBackend::new(
            StoreOptions {
                root: StoreRoot::Namespaced(dir.to_path_buf()),
                dim: 256,
                max_stores: 2,
                encrypt: None,
                #[cfg(feature = "local-models")]
                stages: None,
                #[cfg(feature = "local-models")]
                embedding_dim: None,
            },
            test_limits(),
        )
    }

    fn put(b: &HmsBackend, id: &str, text: &str, tags: &[&str]) -> RememberOut {
        b.remember(
            "t",
            RememberReq {
                text: text.into(),
                id: Some(id.into()),
                title: None,
                tags: tags.iter().map(|t| (*t).to_string()).collect(),
                metadata: Map::new(),
                if_version: None,
                create_only: false,
                allow_empty: true,
            },
        )
        .unwrap()
    }

    fn ids(page: &Page) -> Vec<&str> {
        page.memories.iter().map(|m| m.id.as_str()).collect()
    }

    #[test]
    fn text_round_trips_exactly_including_whitespace_and_blank() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(dir.path());
        for text in [
            "plain",
            "\n  lead and trail\n\n",
            "inner   gaps\tkept",
            "  \n",
            "",
        ] {
            put(&b, "x", text, &[]);
            assert_eq!(b.get("t", "x").unwrap().unwrap().text, text, "{text:?}");
        }
    }

    #[test]
    fn replace_keeps_created_at_and_changes_version() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(dir.path());
        let first = put(&b, "x", "one", &[]);
        let created = b.get("t", "x").unwrap().unwrap().created_at;
        let second = put(&b, "x", "two", &[]);
        assert!(first.created && !second.created);
        assert_ne!(first.version, second.version);
        let m = b.get("t", "x").unwrap().unwrap();
        assert_eq!((m.created_at, m.text.as_str()), (created, "two"));
        assert!(m.updated_at >= created);
    }

    #[test]
    fn compare_and_set_and_create_only() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(dir.path());
        let v1 = put(&b, "x", "one", &[]).version;
        let req = |text: &str, if_version: Option<&str>, create_only: bool| RememberReq {
            text: text.into(),
            id: Some("x".into()),
            title: None,
            tags: vec![],
            metadata: Map::new(),
            if_version: if_version.map(str::to_string),
            create_only,
            allow_empty: false,
        };
        assert_eq!(
            b.remember("t", req("n", None, true)).unwrap_err().code,
            "already_exists"
        );
        let v2 = b
            .remember("t", req("two", Some(&v1), false))
            .unwrap()
            .version;
        let stale = b.remember("t", req("three", Some(&v1), false)).unwrap_err();
        assert_eq!(stale.code, "version_conflict");
        assert_eq!(stale.details.unwrap()["current_version"], v2);
        assert_eq!(b.get("t", "x").unwrap().unwrap().text, "two");
    }

    #[test]
    fn list_pages_by_prefix_and_tag_without_gaps() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(dir.path());
        for i in 0..7 {
            put(
                &b,
                &format!("a/{i}"),
                "alpha",
                if i % 2 == 0 { &["even"] } else { &[] },
            );
        }
        put(&b, "b/0", "beta", &["even"]);
        let list = |prefix: Option<&str>, tag: Option<&str>, cursor: Option<&str>, limit| {
            b.list(
                "t",
                ListReq {
                    prefix: prefix.map(str::to_string),
                    tag: tag.map(str::to_string),
                    cursor: cursor.map(str::to_string),
                    limit,
                },
            )
            .unwrap()
        };
        let p1 = list(Some("a/"), None, None, 3);
        assert_eq!(ids(&p1), ["a/0", "a/1", "a/2"]);
        let p2 = list(Some("a/"), None, p1.next_cursor.as_deref(), 3);
        assert_eq!(ids(&p2), ["a/3", "a/4", "a/5"]);
        let p3 = list(Some("a/"), None, p2.next_cursor.as_deref(), 3);
        assert_eq!(ids(&p3), ["a/6"]);
        assert!(p3.next_cursor.is_none());
        let exact = list(Some("a/"), None, None, 7);
        assert_eq!(exact.memories.len(), 7);
        assert!(
            exact.next_cursor.is_none(),
            "limit == matches must not invent a next page"
        );
        let even = list(None, Some("even"), None, 10);
        assert_eq!(ids(&even), ["a/0", "a/2", "a/4", "a/6", "b/0"]);
        // A cursor before the prefix does not rewind past it.
        assert_eq!(ids(&list(Some("b/"), None, Some("a/3"), 10)), ["b/0"]);
    }

    #[test]
    fn forget_by_ids_and_prefix_and_search_by_tag() {
        let dir = tempfile::tempdir().unwrap();
        let b = backend(dir.path());
        put(&b, "n/1", "red apples grow in orchards", &["fruit"]);
        put(&b, "n/2", "green apples are tart", &["fruit", "green"]);
        put(&b, "m/1", "apples in the music store", &[]);
        let search = |tags: &[&str]| {
            b.search(
                "t",
                SearchReq {
                    query: "apples".into(),
                    k: 5,
                    tags: tags.iter().map(|t| (*t).to_string()).collect(),
                    filter: Map::new(),
                    min_score: None,
                    lexical_weight: None,
                    semantic_weight: None,
                    fusion: None,
                },
            )
            .unwrap()
        };
        assert_eq!(search(&[]).len(), 3);
        assert_eq!(search(&["fruit"]).len(), 2);
        assert_eq!(search(&["fruit", "green"]).len(), 1);
        let out = b
            .forget(
                "t",
                ForgetReq {
                    ids: vec!["m/1".into(), "m/9".into()],
                    prefix: None,
                },
            )
            .unwrap();
        assert_eq!((out.deleted, out.missing), (1, vec!["m/9".to_string()]));
        let out = b
            .forget(
                "t",
                ForgetReq {
                    ids: vec![],
                    prefix: Some("n/".into()),
                },
            )
            .unwrap();
        assert_eq!(out.deleted, 2);
        assert!(search(&[]).is_empty());
        assert_eq!(b.stats("t").unwrap()["memories"], 0);
    }

    #[test]
    fn locked_store_and_store_cap_are_structured_errors() {
        let dir = tempfile::tempdir().unwrap();
        let first = backend(dir.path());
        first.preopen("one").unwrap();
        let second = backend(dir.path());
        assert_eq!(second.preopen("one").unwrap_err().code, "store_locked");
        first.preopen("two").unwrap();
        assert_eq!(first.preopen("three").unwrap_err().code, "too_many_stores");
    }

    #[test]
    fn data_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let b = backend(dir.path());
            put(&b, "keep", "durable fact about glaciers", &["geo"]);
            b.flush_all();
        }
        let b = backend(dir.path());
        let m = b.get("t", "keep").unwrap().unwrap();
        assert_eq!(
            (m.text.as_str(), m.tags.clone()),
            ("durable fact about glaciers", vec!["geo".to_string()])
        );
    }
}
