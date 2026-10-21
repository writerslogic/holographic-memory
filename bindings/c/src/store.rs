// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Store lifecycle, ingest, search, delete, statistics and snapshot entry points.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use holographic_memory::core::admin::inspect_store;
use holographic_memory::core::{EmbeddingSpace, HmsConfig};
use holographic_memory::{
    chunk_document, DocumentChunk, DocumentInput, DocumentResult, HmsCore, SearchOptions,
};
use parking_lot::{Mutex, RwLock};

use crate::error::*;
use crate::types::*;

/// An open store. Opaque to C.
#[allow(non_camel_case_types)]
pub struct hms_store_t {
    core: HmsCore,
    /// Mutations hold the read side; a snapshot holds the write side so the copied files are
    /// a single consistent point in time.
    barrier: RwLock<()>,
    path: PathBuf,
}

/// A search result set. Opaque to C; read with `hms1_results_count` and `hms1_results_get`.
#[allow(non_camel_case_types)]
pub struct hms_results_t {
    hits: Vec<DocumentResult>,
    metadata: Vec<String>,
}

/// A chunking result. Opaque to C; read with `hms1_chunks_count` and `hms1_chunks_get`.
#[allow(non_camel_case_types)]
pub struct hms_chunks_t {
    chunks: Vec<DocumentChunk>,
}

/// Live store handles, keyed by an opaque token. Looking one up clones the `Arc`, so closing a handle while another thread is inside a call on it is safe and a stale
/// or forged handle is rejected instead of dereferenced.
fn registry() -> &'static Mutex<HashMap<usize, Arc<hms_store_t>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Arc<hms_store_t>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lookup(handle: *const hms_store_t) -> ApiResult<Arc<hms_store_t>> {
    if handle.is_null() {
        return Err(ApiError::new(HMS_ERR_NULL_POINTER, "store handle is NULL"));
    }
    registry()
        .lock()
        .get(&(handle as usize))
        .cloned()
        .ok_or_else(|| {
            ApiError::new(
                HMS_ERR_INVALID_HANDLE,
                "store handle is closed or was never issued",
            )
        })
}

fn bool_i32(value: bool) -> i32 {
    i32::from(value)
}

fn tri_state(value: i32, what: &str) -> ApiResult<Option<bool>> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(true)),
        -1 => Ok(Some(false)),
        other => Err(ApiError::new(
            HMS_ERR_INVALID_ARGUMENT,
            format!("{what} must be -1, 0 or 1 (got {other})"),
        )),
    }
}

/// Fill `options` with the library defaults. Callers that build the struct field by field must
/// still set `struct_size`.
///
/// # Safety
/// `options` must be NULL or point to a writable `hms_open_options_t`.
#[no_mangle]
pub unsafe extern "C" fn hms1_open_options_init(options: *mut hms_open_options_t) {
    if let Some(options) = options.as_mut() {
        *options = hms_open_options_t::DEFAULT;
    }
}

/// Fill `options` with the library defaults (all flags clear).
///
/// # Safety
/// `options` must be NULL or point to a writable `hms_search_options_t`.
#[no_mangle]
pub unsafe extern "C" fn hms1_search_options_init(options: *mut hms_search_options_t) {
    if let Some(options) = options.as_mut() {
        *options = hms_search_options_t::DEFAULT;
    }
}

/// Open or create the store in directory `path` with vectors of `dimensions` (1..=1000000).
///
/// A store directory is held by exactly one open handle at a time, across processes
/// (`HMS_ERR_LOCKED` otherwise). `options` may be NULL for defaults. On success `*out`
/// receives a handle that must be released with `hms1_store_close`; on failure `*out` is NULL.
///
/// # Safety
/// `path` and every `hms_str_t` in `options` must be valid for their lengths; `out` must be
/// writable.
#[no_mangle]
pub unsafe extern "C" fn hms1_store_open(
    path: hms_str_t,
    dimensions: u32,
    options: *const hms_open_options_t,
    out: *mut *mut hms_store_t,
) -> hms_status_t {
    if let Some(out) = out.as_mut() {
        *out = std::ptr::null_mut();
    }
    guard(|| {
        let out = require_out(out, "out")?;
        let path = utf8(path, "path")?;
        if path.is_empty() || path.contains('\0') {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                "path must be non-empty and contain no NUL",
            ));
        }
        if !(1..=1_000_000).contains(&dimensions) {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                format!("dimensions must be between 1 and 1000000 (got {dimensions})"),
            ));
        }
        let options = overlay(options, hms_open_options_t::DEFAULT, "options")?;
        if options.flags != 0 {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                "options.flags is reserved and must be zero",
            ));
        }
        let mut config = HmsConfig::default();
        if let Some(key) = opt_utf8(options.encryption_key, "options.encryption_key")? {
            if key.is_empty() {
                return Err(ApiError::new(
                    HMS_ERR_INVALID_ARGUMENT,
                    "encryption key must not be empty",
                ));
            }
            config.security.encryption_enabled = true;
            config.security.encryption_passphrase = Some(key.to_owned());
        }
        if let Some(on) = tri_state(options.signing, "options.signing")? {
            config.security.signing_enabled = on;
        }
        if let Some(on) = tri_state(options.audit, "options.audit")? {
            config.security.audit_enabled = on;
        }
        let model = opt_utf8(options.embedding_model, "options.embedding_model")?;
        let revision = opt_utf8(options.embedding_revision, "options.embedding_revision")?;
        match (model, revision, options.embedding_dimensions) {
            (None, None, 0) => {}
            (Some(model), Some(revision), dims) if dims != 0 => {
                config.embedding_space = Some(EmbeddingSpace {
                    model: model.to_owned(),
                    revision: revision.to_owned(),
                    dimensions: dims as usize,
                    normalization: "l2".into(),
                    metric: "cosine".into(),
                });
            }
            _ => {
                return Err(ApiError::new(
                    HMS_ERR_INVALID_ARGUMENT,
                    "embedding_model, embedding_revision and embedding_dimensions go together",
                ))
            }
        }
        let core = HmsCore::new(dimensions, Some(path.to_owned()), Some(config))
            .map_err(|e| ApiError::from_engine(&e, HMS_ERR_IO))?;
        let store = Arc::new(hms_store_t {
            core,
            barrier: RwLock::new(()),
            path: PathBuf::from(path),
        });
        // The handle is a never-reused token, not an address: a stale handle can never alias a
        // store opened later, and it is never dereferenced.
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        let token = NEXT.fetch_add(1, Ordering::Relaxed) << 4;
        registry().lock().insert(token, store);
        *out = token as *mut hms_store_t;
        Ok(())
    })
}

/// Flush and release a store. Calls already running on it complete first; afterwards the
/// handle is invalid. NULL is ignored. Closing an unknown or already closed handle returns
/// `HMS_ERR_INVALID_HANDLE`. If the final flush fails the handle is still released and the
/// failure is reported.
#[no_mangle]
pub extern "C" fn hms1_store_close(store: *mut hms_store_t) -> hms_status_t {
    guard(|| {
        if store.is_null() {
            return Ok(());
        }
        let removed = registry().lock().remove(&(store as usize)).ok_or_else(|| {
            ApiError::new(HMS_ERR_INVALID_HANDLE, "store handle is closed or unknown")
        })?;
        let flushed = {
            let _mutations = removed.barrier.read();
            removed.core.flush()
        };
        drop(removed);
        flushed.map_err(|e| ApiError::from_engine(&e, HMS_ERR_IO))
    })
}

fn document_input(doc: &hms_document_t) -> ApiResult<DocumentInput> {
    let id = utf8(doc.id, "document.id")?.to_owned();
    let text = utf8(doc.text, "document.text")?.to_owned();
    let source_uri = opt_utf8(doc.source_uri, "document.source_uri")?.map(str::to_owned);
    let version = opt_utf8(doc.version, "document.version")?.map(str::to_owned);
    let metadata = match opt_utf8(doc.metadata_json, "document.metadata_json")? {
        None => None,
        Some(json) => Some(serde_json::from_str(json).map_err(|e| {
            ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                format!("document.metadata_json is not JSON: {e}"),
            )
        })?),
    };
    let store_text = match doc.store_text {
        0 => None,
        1 => Some(true),
        -1 => Some(false),
        other => {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                format!("document.store_text must be -1, 0 or 1 (got {other})"),
            ))
        }
    };
    let embeddings = if doc.embedding_rows == 0 && doc.embedding_dim == 0 {
        if !doc.embeddings.is_null() {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                "document.embeddings is set but its shape is 0 x 0",
            ));
        }
        None
    } else {
        if doc.embedding_rows == 0 || doc.embedding_dim == 0 {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                "document embedding rows and dimension must both be nonzero",
            ));
        }
        let total = doc
            .embedding_rows
            .checked_mul(doc.embedding_dim)
            .ok_or_else(|| ApiError::new(HMS_ERR_LIMIT_EXCEEDED, "embedding shape overflows"))?;
        let flat = slice(doc.embeddings, total, "document.embeddings")?;
        Some(
            flat.chunks_exact(doc.embedding_dim)
                .map(<[f64]>::to_vec)
                .collect(),
        )
    };
    Ok(DocumentInput {
        id,
        text,
        source_uri,
        version,
        metadata,
        chunk_words: opt_u32(doc.chunk_words),
        overlap_words: (doc.overlap_words != 0).then_some(doc.overlap_words),
        store_text,
        embeddings,
        title_weight: opt_u32(doc.title_weight),
    })
}

/// Ingest `count` documents as one atomic batch: either all become visible or none do.
/// `chunk_counts`, if not NULL, receives the number of chunks of each document and must hold
/// `count` entries. An empty batch succeeds and does nothing. At most 16777216 documents per
/// call; each document is bounded by the engine (8 MiB of text).
///
/// # Safety
/// `documents` must point to `count` valid `hms_document_t` whose strings and embeddings are
/// valid for their lengths for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn hms1_store_ingest(
    store: *mut hms_store_t,
    documents: *const hms_document_t,
    count: usize,
    chunk_counts: *mut u32,
) -> hms_status_t {
    guard(|| {
        let store = lookup(store)?;
        let docs = slice(documents, count, "documents")?;
        let inputs = docs
            .iter()
            .map(document_input)
            .collect::<ApiResult<Vec<_>>>()?;
        if inputs.is_empty() {
            return Ok(());
        }
        let counts = {
            let _mutation = store.barrier.read();
            store
                .core
                .memorize_documents(inputs)
                .map_err(|e| ApiError::from_engine(&e, HMS_ERR_INVALID_ARGUMENT))?
        };
        if !chunk_counts.is_null() {
            // SAFETY: the caller promised room for `count` entries; `counts` has `count`.
            unsafe { std::ptr::copy_nonoverlapping(counts.as_ptr(), chunk_counts, counts.len()) };
        }
        Ok(())
    })
}

fn search_options(options: *const hms_search_options_t) -> ApiResult<SearchOptions> {
    let o = overlay(options, hms_search_options_t::DEFAULT, "options")?;
    let document_ids = if o.document_id_count == 0 {
        None
    } else {
        let ids = slice(o.document_ids, o.document_id_count, "options.document_ids")?;
        Some(
            ids.iter()
                .map(|id| utf8(*id, "options.document_ids[]").map(str::to_owned))
                .collect::<ApiResult<Vec<_>>>()?,
        )
    };
    let embedding = if o.embedding_len == 0 {
        None
    } else {
        Some(slice(o.embedding, o.embedding_len, "options.embedding")?.to_vec())
    };
    let filter = match opt_utf8(o.filter_json, "options.filter_json")? {
        None => None,
        Some(json) => Some(serde_json::from_str(json).map_err(|e| {
            ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                format!("options.filter_json is not JSON: {e}"),
            )
        })?),
    };
    Ok(SearchOptions {
        k: opt_u32(o.k),
        candidate_limit: opt_u32(o.candidate_limit),
        filter,
        source_uri: opt_utf8(o.source_uri, "options.source_uri")?.map(str::to_owned),
        document_ids,
        embedding,
        lexical_weight: flagged(o.flags, HMS_SEARCH_SET_LEXICAL_WEIGHT, o.lexical_weight),
        semantic_weight: flagged(o.flags, HMS_SEARCH_SET_SEMANTIC_WEIGHT, o.semantic_weight),
        min_semantic_score: flagged(
            o.flags,
            HMS_SEARCH_SET_MIN_SEMANTIC_SCORE,
            o.min_semantic_score,
        ),
        fusion: opt_utf8(o.fusion, "options.fusion")?.map(str::to_owned),
        bm25_k1: flagged(o.flags, HMS_SEARCH_SET_BM25_K1, o.bm25_k1),
        bm25_b: flagged(o.flags, HMS_SEARCH_SET_BM25_B, o.bm25_b),
    })
}

/// Search the store for `query` (hybrid lexical BM25 plus, with an embedding, semantic).
/// `options` may be NULL for defaults. On success `*out` receives a result set to release with
/// `hms1_results_free`; it is NULL on failure. Concurrent searches are safe.
///
/// # Safety
/// `query`, `options` and the strings it points to must be valid for their lengths; `out`
/// must be writable.
#[no_mangle]
pub unsafe extern "C" fn hms1_store_search(
    store: *mut hms_store_t,
    query: hms_str_t,
    options: *const hms_search_options_t,
    out: *mut *mut hms_results_t,
) -> hms_status_t {
    if let Some(out) = out.as_mut() {
        *out = std::ptr::null_mut();
    }
    guard(|| {
        let out = require_out(out, "out")?;
        let store = lookup(store)?;
        let query = utf8(query, "query")?;
        let options = search_options(options)?;
        let hits = store
            .core
            .search_documents(query, &options)
            .map_err(|e| ApiError::from_engine(&e, HMS_ERR_INVALID_ARGUMENT))?;
        let metadata = hits.iter().map(|h| h.metadata.to_string()).collect();
        *out = Box::into_raw(Box::new(hms_results_t { hits, metadata }));
        Ok(())
    })
}

/// Number of hits. NULL counts as zero.
///
/// # Safety
/// `results` must be NULL or a live result set.
#[no_mangle]
pub unsafe extern "C" fn hms1_results_count(results: *const hms_results_t) -> usize {
    // SAFETY: NULL is handled; a non-NULL pointer is a live result set by contract.
    unsafe { results.as_ref() }.map_or(0, |r| r.hits.len())
}

/// Read hit `index` into `out`. The strings borrow from `results` until it is freed.
///
/// # Safety
/// `results` must be a live result set and `out` writable.
#[no_mangle]
pub unsafe extern "C" fn hms1_results_get(
    results: *const hms_results_t,
    index: usize,
    out: *mut hms_result_t,
) -> hms_status_t {
    guard(|| {
        let results = require_ptr(results, "results")?;
        let out = require_out(out, "out")?;
        let hit = results.hits.get(index).ok_or_else(|| {
            ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                format!(
                    "index {index} is out of range for {} hits",
                    results.hits.len()
                ),
            )
        })?;
        let optional =
            |value: &Option<String>| value.as_deref().map_or(hms_str_t::ABSENT, hms_str_t::of);
        *out = hms_result_t {
            id: hms_str_t::of(&hit.id),
            document_id: hms_str_t::of(&hit.document_id),
            source_uri: optional(&hit.source_uri),
            version: hms_str_t::of(&hit.version),
            metadata_json: hms_str_t::of(&results.metadata[index]),
            text: optional(&hit.text),
            start_byte: hit.start_byte,
            end_byte: hit.end_byte,
            score: hit.score,
            lexical_score: hit.lexical_score,
            semantic_score: hit.semantic_score.unwrap_or(f64::NAN),
        };
        Ok(())
    })
}

/// Release a result set and every string borrowed from it. NULL is ignored.
///
/// # Safety
/// `results` must be NULL or a result set not yet freed.
#[no_mangle]
pub unsafe extern "C" fn hms1_results_free(results: *mut hms_results_t) {
    if !results.is_null() {
        // Dropping Vec<String> cannot panic, so no guard is needed on this void function.
        drop(Box::from_raw(results));
    }
}

/// Delete a document and all its chunks. `*deleted` (optional) receives 1 if it existed.
/// Deleting an absent id succeeds with `*deleted == 0`.
///
/// # Safety
/// `id` must be valid for its length; `deleted` NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn hms1_store_delete(
    store: *mut hms_store_t,
    id: hms_str_t,
    deleted: *mut i32,
) -> hms_status_t {
    guard(|| {
        let store = lookup(store)?;
        let id = utf8(id, "id")?;
        let existed = {
            let _mutation = store.barrier.read();
            store
                .core
                .delete_document(id)
                .map_err(|e| ApiError::from_engine(&e, HMS_ERR_INVALID_ARGUMENT))?
        };
        if let Some(deleted) = deleted.as_mut() {
            *deleted = bool_i32(existed);
        }
        Ok(())
    })
}

/// Read store statistics into `out`; set `out->struct_size` first.
///
/// # Safety
/// `out` must be writable for `out->struct_size` bytes (at most `sizeof(hms_stats_t)` are
/// written).
#[no_mangle]
pub unsafe extern "C" fn hms1_store_stats(
    store: *mut hms_store_t,
    out: *mut hms_stats_t,
) -> hms_status_t {
    guard(|| {
        let store = lookup(store)?;
        let index = store.core.index_status();
        let health = store.core.storage_health();
        let stats = hms_stats_t {
            struct_size: std::mem::size_of::<hms_stats_t>() as u32,
            dimensions: store.core.dimensions() as u32,
            vector_count: store.core.vector_count(),
            nsg_trained: bool_i32(index.nsg_trained),
            ivf_trained: bool_i32(index.ivf_trained),
            nsg_training_recommended: bool_i32(index.nsg_training_recommended),
            ivf_training_recommended: bool_i32(index.ivf_training_recommended),
            document_index_trained: bool_i32(store.core.document_index_trained()),
            storage_format_version: health.format_version,
            storage_segment_count: health.segment_count,
            document_embedding_count: store.core.document_embedding_count() as u64,
            storage_used_bytes: health.used_bytes as u64,
            storage_capacity_bytes: health.capacity_bytes as u64,
        };
        write_out(out, stats, "out")
    })
}

/// Read the store's security status into `out`; set `out->struct_size` first. Encryption is
/// reported as active only for a store that was opened with an encryption key.
///
/// # Safety
/// As `hms1_store_stats`.
#[no_mangle]
pub unsafe extern "C" fn hms1_store_security_status(
    store: *mut hms_store_t,
    out: *mut hms_security_status_t,
) -> hms_status_t {
    guard(|| {
        let store = lookup(store)?;
        let status = store.core.security_status();
        let flag = |name: &str| bool_i32(status[name].as_bool().unwrap_or(false));
        let value = hms_security_status_t {
            struct_size: std::mem::size_of::<hms_security_status_t>() as u32,
            security_compiled: flag("securityCompiled"),
            encryption_active: flag("encryptionActive"),
            signing_active: flag("signingActive"),
            audit_active: flag("auditActive"),
            differential_privacy_active: flag("differentialPrivacyActive"),
            epsilon: status["epsilon"].as_f64().unwrap_or(f64::NAN),
        };
        write_out(out, value, "out")
    })
}

/// Flush all state to durable storage.
#[no_mangle]
pub extern "C" fn hms1_store_flush(store: *mut hms_store_t) -> hms_status_t {
    guard(|| {
        let store = lookup(store)?;
        let _mutation = store.barrier.read();
        store
            .core
            .flush()
            .map_err(|e| ApiError::from_engine(&e, HMS_ERR_IO))
    })
}

/// Rewrite the transaction log without superseded and deleted records, reclaiming space.
/// Safe alongside concurrent searches and ingests.
#[no_mangle]
pub extern "C" fn hms1_store_compact(store: *mut hms_store_t) -> hms_status_t {
    guard(|| {
        let store = lookup(store)?;
        let _mutation = store.barrier.read();
        store
            .core
            .compact()
            .map_err(|e| ApiError::from_engine(&e, HMS_ERR_IO))
    })
}

/// Train any index whose configured threshold has been reached. Idempotent.
#[no_mangle]
pub extern "C" fn hms1_store_maintain(store: *mut hms_store_t) -> hms_status_t {
    guard(|| {
        let store = lookup(store)?;
        let _mutation = store.barrier.read();
        store
            .core
            .maintain_indices()
            .map(|_| ())
            .map_err(|e| ApiError::from_engine(&e, HMS_ERR_INTERNAL))
    })
}

fn copy_tree(source: &Path, destination: &Path) -> ApiResult<()> {
    let io = |e: std::io::Error, path: &Path| {
        ApiError::new(HMS_ERR_IO, format!("{}: {e}", path.display()))
    };
    std::fs::create_dir(destination).map_err(|e| io(e, destination))?;
    for entry in std::fs::read_dir(source).map_err(|e| io(e, source))? {
        let entry = entry.map_err(|e| io(e, source))?;
        let kind = entry.file_type().map_err(|e| io(e, &entry.path()))?;
        if entry.file_name() == ".hms.lock" {
            continue;
        }
        let target = destination.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), &target).map_err(|e| io(e, &entry.path()))?;
        } else {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                format!(
                    "snapshot refuses link or special file {}",
                    entry.path().display()
                ),
            ));
        }
    }
    Ok(())
}

/// Write a verified point-in-time copy of the store to the new directory `destination`,
/// which must not exist and must not lie inside the store. Mutations wait for the copy.
/// The copy appears atomically only after its framing and checksums verify, and is an
/// ordinary store: open it with the same encryption key. Reads are not blocked.
///
/// # Safety
/// `destination` must be valid for its length.
#[no_mangle]
pub unsafe extern "C" fn hms1_store_snapshot(
    store: *mut hms_store_t,
    destination: hms_str_t,
) -> hms_status_t {
    guard(|| {
        let store = lookup(store)?;
        let destination = PathBuf::from(utf8(destination, "destination")?);
        if destination.as_os_str().is_empty() {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                "destination is empty",
            ));
        }
        if destination.exists() {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                format!(
                    "snapshot destination {} already exists",
                    destination.display()
                ),
            ));
        }
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)
            .map_err(|e| ApiError::new(HMS_ERR_IO, format!("{}: {e}", parent.display())))?;
        let source = store
            .path
            .canonicalize()
            .map_err(|e| ApiError::new(HMS_ERR_IO, format!("{}: {e}", store.path.display())))?;
        let parent = parent
            .canonicalize()
            .map_err(|e| ApiError::new(HMS_ERR_IO, format!("{}: {e}", parent.display())))?;
        if parent.starts_with(&source) {
            return Err(ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                "snapshot destination cannot be inside the store",
            ));
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let temporary = parent.join(format!(".hms-snapshot-{}-{nonce}", std::process::id()));
        let result = (|| {
            {
                let _quiesce = store.barrier.write();
                store
                    .core
                    .flush()
                    .map_err(|e| ApiError::from_engine(&e, HMS_ERR_IO))?;
                copy_tree(&source, &temporary)?;
            }
            inspect_store(&temporary, true)
                .map_err(|e| ApiError::from_engine(&e, HMS_ERR_CORRUPT))?;
            let name = destination.file_name().ok_or_else(|| {
                ApiError::new(HMS_ERR_INVALID_ARGUMENT, "destination has no name")
            })?;
            std::fs::rename(&temporary, parent.join(name))
                .map_err(|e| ApiError::new(HMS_ERR_IO, format!("publishing snapshot: {e}")))
        })();
        if result.is_err() && temporary.exists() {
            let _ = std::fs::remove_dir_all(&temporary);
        }
        result
    })
}

/// Split a document into the chunks ingest would store, without touching any store.
/// `document` supplies `id`, `text`, `chunk_words` and `overlap_words`. On success `*out`
/// receives a chunk set to release with `hms1_chunks_free`.
///
/// # Safety
/// `document` must point to a valid `hms_document_t`; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn hms1_chunk_document(
    document: *const hms_document_t,
    out: *mut *mut hms_chunks_t,
) -> hms_status_t {
    if let Some(out) = out.as_mut() {
        *out = std::ptr::null_mut();
    }
    guard(|| {
        let out = require_out(out, "out")?;
        let document = require_ptr(document, "document")?;
        let input = document_input(document)?;
        let chunks = chunk_document(&input)
            .map_err(|e| ApiError::from_engine(&e, HMS_ERR_INVALID_ARGUMENT))?;
        *out = Box::into_raw(Box::new(hms_chunks_t { chunks }));
        Ok(())
    })
}

/// Number of chunks. NULL counts as zero.
///
/// # Safety
/// `chunks` must be NULL or a live chunk set.
#[no_mangle]
pub unsafe extern "C" fn hms1_chunks_count(chunks: *const hms_chunks_t) -> usize {
    // SAFETY: NULL is handled; a non-NULL pointer is a live chunk set by contract.
    unsafe { chunks.as_ref() }.map_or(0, |c| c.chunks.len())
}

/// Read chunk `index` into `out`; strings borrow from `chunks` until it is freed.
///
/// # Safety
/// `chunks` must be a live chunk set and `out` writable.
#[no_mangle]
pub unsafe extern "C" fn hms1_chunks_get(
    chunks: *const hms_chunks_t,
    index: usize,
    out: *mut hms_chunk_t,
) -> hms_status_t {
    guard(|| {
        let chunks = require_ptr(chunks, "chunks")?;
        let out = require_out(out, "out")?;
        let chunk = chunks.chunks.get(index).ok_or_else(|| {
            ApiError::new(
                HMS_ERR_INVALID_ARGUMENT,
                format!(
                    "index {index} is out of range for {} chunks",
                    chunks.chunks.len()
                ),
            )
        })?;
        *out = hms_chunk_t {
            id: hms_str_t::of(&chunk.id),
            text: hms_str_t::of(&chunk.text),
            start_byte: chunk.start_byte,
            end_byte: chunk.end_byte,
        };
        Ok(())
    })
}

/// Release a chunk set. NULL is ignored.
///
/// # Safety
/// `chunks` must be NULL or a chunk set not yet freed.
#[no_mangle]
pub unsafe extern "C" fn hms1_chunks_free(chunks: *mut hms_chunks_t) {
    if !chunks.is_null() {
        drop(Box::from_raw(chunks));
    }
}
