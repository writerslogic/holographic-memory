// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `repr(C)` types and the checked readers for caller-supplied memory.

use crate::error::*;

/// A length-prefixed UTF-8 string. It is not NUL terminated and may contain any valid UTF-8,
/// including NUL. As an input, `ptr == NULL` with `len == 0` means "absent" (use the default).
/// As an output it borrows from the handle that produced it.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct hms_str_t {
    pub ptr: *const u8,
    pub len: usize,
}

impl hms_str_t {
    pub const ABSENT: Self = Self {
        ptr: std::ptr::null(),
        len: 0,
    };

    pub fn of(value: &str) -> Self {
        Self {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }
}

/// Store open options. Set `struct_size = sizeof(hms_open_options_t)`; fields past the
/// caller's `struct_size` take their defaults, so a newer library accepts an older caller.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct hms_open_options_t {
    /// `sizeof` of this struct as the caller compiled it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub flags: u32,
    /// Encryption passphrase (UTF-8). Absent leaves encryption off. A store created with a
    /// key can only be opened with the same key.
    pub encryption_key: hms_str_t,
    /// 0 default (off), 1 on, -1 off. Signs mutations with an Ed25519 key kept in the store.
    pub signing: i32,
    /// 0 default (off), 1 on, -1 off.
    pub audit: i32,
    /// Embedding space for caller-supplied embeddings. All three or none; the model and
    /// revision are identifiers the store pins, the dimension is 1..=65536.
    pub embedding_model: hms_str_t,
    pub embedding_revision: hms_str_t,
    pub embedding_dimensions: u32,
}

impl hms_open_options_t {
    pub const DEFAULT: Self = Self {
        struct_size: std::mem::size_of::<Self>() as u32,
        flags: 0,
        encryption_key: hms_str_t::ABSENT,
        signing: 0,
        audit: 0,
        embedding_model: hms_str_t::ABSENT,
        embedding_revision: hms_str_t::ABSENT,
        embedding_dimensions: 0,
    };
}

/// One document to ingest. Zero, NULL and absent-string fields take the engine defaults.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct hms_document_t {
    /// Unique non-empty identifier, at most 4096 bytes. Re-ingesting an id replaces it.
    pub id: hms_str_t,
    pub text: hms_str_t,
    pub source_uri: hms_str_t,
    pub version: hms_str_t,
    /// A JSON value (object recommended) stored with the document and returned with hits.
    pub metadata_json: hms_str_t,
    /// Words per chunk; 0 is the engine default.
    pub chunk_words: u32,
    /// Overlap words between chunks; 0 is the engine default.
    pub overlap_words: u32,
    /// Counts the first line's terms this many times in the first chunk (BM25 title boost);
    /// 0 or 1 leaves the document unchanged.
    pub title_weight: u32,
    /// 0 default (keep), 1 keep the text for retrieval, -1 index without keeping the text.
    pub store_text: i32,
    /// Optional row-major embeddings, `embedding_rows` x `embedding_dim` doubles, one row per
    /// chunk; requires the store to be opened with an embedding space of that dimension.
    pub embeddings: *const f64,
    pub embedding_rows: usize,
    pub embedding_dim: usize,
}

/// Search options. Set `struct_size = sizeof(hms_search_options_t)`; see
/// `hms_open_options_t` for the compatibility rule. Initialise it with `hms1_search_options_init`;
/// a zero-initialised struct (with `struct_size` set) selects every engine default.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct hms_search_options_t {
    pub struct_size: u32,
    /// Bitwise OR of `HMS_SEARCH_SET_*`; a numeric field below is honoured only when its bit
    /// is set (zero is a meaningful weight), so a zero-initialised struct means all defaults.
    pub flags: u32,
    /// Hits to return; 0 is the engine default.
    pub k: u32,
    /// Candidates gathered per list before fusion; 0 is the engine default.
    pub candidate_limit: u32,
    /// JSON metadata filter; absent matches everything.
    pub filter_json: hms_str_t,
    /// Restrict to one source URI.
    pub source_uri: hms_str_t,
    /// Restrict to these document ids (`document_id_count` entries).
    pub document_ids: *const hms_str_t,
    pub document_id_count: usize,
    /// Query embedding for semantic or hybrid search (`embedding_len` doubles).
    pub embedding: *const f64,
    pub embedding_len: usize,
    /// Honoured with `HMS_SEARCH_SET_LEXICAL_WEIGHT`.
    pub lexical_weight: f64,
    /// Honoured with `HMS_SEARCH_SET_SEMANTIC_WEIGHT`.
    pub semantic_weight: f64,
    /// Honoured with `HMS_SEARCH_SET_MIN_SEMANTIC_SCORE`.
    pub min_semantic_score: f64,
    /// BM25 term-frequency saturation k1 (default 1.2, finite and nonnegative); honoured with
    /// `HMS_SEARCH_SET_BM25_K1`.
    pub bm25_k1: f64,
    /// BM25 length normalization b (default 0.75, in [0, 1]); honoured with
    /// `HMS_SEARCH_SET_BM25_B`.
    pub bm25_b: f64,
    /// `"blend"` (default) or `"rrf"`; absent is the default.
    pub fusion: hms_str_t,
}

impl hms_search_options_t {
    pub const DEFAULT: Self = Self {
        struct_size: std::mem::size_of::<Self>() as u32,
        flags: 0,
        k: 0,
        candidate_limit: 0,
        filter_json: hms_str_t::ABSENT,
        source_uri: hms_str_t::ABSENT,
        document_ids: std::ptr::null(),
        document_id_count: 0,
        embedding: std::ptr::null(),
        embedding_len: 0,
        lexical_weight: 0.0,
        semantic_weight: 0.0,
        min_semantic_score: 0.0,
        bm25_k1: 0.0,
        bm25_b: 0.0,
        fusion: hms_str_t::ABSENT,
    };
}

pub const HMS_SEARCH_SET_LEXICAL_WEIGHT: u32 = 1 << 0;
pub const HMS_SEARCH_SET_SEMANTIC_WEIGHT: u32 = 1 << 1;
pub const HMS_SEARCH_SET_MIN_SEMANTIC_SCORE: u32 = 1 << 2;
pub const HMS_SEARCH_SET_BM25_K1: u32 = 1 << 3;
pub const HMS_SEARCH_SET_BM25_B: u32 = 1 << 4;

/// One search hit. Strings borrow from the `hms_results_t` that produced it and stay valid
/// until `hms1_results_free`.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct hms_result_t {
    /// Chunk id.
    pub id: hms_str_t,
    pub document_id: hms_str_t,
    /// `ptr == NULL` when the document has no source URI.
    pub source_uri: hms_str_t,
    pub version: hms_str_t,
    /// The document metadata as JSON text.
    pub metadata_json: hms_str_t,
    /// The chunk text; `ptr == NULL` when the document was stored without text.
    pub text: hms_str_t,
    pub start_byte: u32,
    pub end_byte: u32,
    pub score: f64,
    pub lexical_score: f64,
    /// NaN when the hit has no semantic score.
    pub semantic_score: f64,
}

/// One chunk of a document as `hms1_chunk_document` would split it.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct hms_chunk_t {
    pub id: hms_str_t,
    pub text: hms_str_t,
    pub start_byte: u32,
    pub end_byte: u32,
}

/// Store statistics. Set `struct_size` before the call; the library fills the fields the
/// caller's struct covers.
#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(non_camel_case_types)]
pub struct hms_stats_t {
    pub struct_size: u32,
    pub dimensions: u32,
    /// Vectors in the store (document chunks count as vectors).
    pub vector_count: u64,
    pub nsg_trained: i32,
    pub ivf_trained: i32,
    pub nsg_training_recommended: i32,
    pub ivf_training_recommended: i32,
    pub document_index_trained: i32,
    pub storage_format_version: u32,
    pub storage_segment_count: u32,
    pub document_embedding_count: u64,
    pub storage_used_bytes: u64,
    pub storage_capacity_bytes: u64,
}

/// Security status. Set `struct_size` before the call.
#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(non_camel_case_types)]
pub struct hms_security_status_t {
    pub struct_size: u32,
    /// The library was built with encryption and signing support.
    pub security_compiled: i32,
    /// Data at rest is encrypted with AES-256-GCM under an Argon2-derived key.
    pub encryption_active: i32,
    pub signing_active: i32,
    pub audit_active: i32,
    pub differential_privacy_active: i32,
    pub epsilon: f64,
}

pub(crate) fn require_ptr<'a, T>(ptr: *const T, what: &str) -> ApiResult<&'a T> {
    // SAFETY: the caller promises a valid pointer or NULL; NULL is rejected here.
    unsafe { ptr.as_ref() }
        .ok_or_else(|| ApiError::new(HMS_ERR_NULL_POINTER, format!("{what} is NULL")))
}

pub(crate) fn require_out<'a, T>(ptr: *mut T, what: &str) -> ApiResult<&'a mut T> {
    // SAFETY: as `require_ptr`; the caller promises exclusive access for the call.
    unsafe { ptr.as_mut() }
        .ok_or_else(|| ApiError::new(HMS_ERR_NULL_POINTER, format!("{what} is NULL")))
}

/// Read `len` elements at `ptr`, rejecting NULL with a nonzero length and any length beyond
/// the documented limits. Nothing is allocated from a caller-supplied length.
pub(crate) fn slice<'a, T>(ptr: *const T, len: usize, what: &str) -> ApiResult<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(ApiError::new(
            HMS_ERR_NULL_POINTER,
            format!("{what} is NULL with length {len}"),
        ));
    }
    let bytes = len.checked_mul(std::mem::size_of::<T>());
    let limit = if std::mem::size_of::<T>() == 1 {
        HMS_MAX_BUFFER_BYTES
    } else {
        HMS_MAX_ARRAY_LEN.saturating_mul(std::mem::size_of::<T>().max(1))
    };
    match bytes {
        Some(bytes) if bytes <= limit => {}
        _ => {
            return Err(ApiError::new(
                HMS_ERR_LIMIT_EXCEEDED,
                format!("{what} length {len} exceeds the limit"),
            ))
        }
    }
    // SAFETY: non-null, bounded; the caller promises `len` readable elements for the call.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

pub(crate) fn utf8<'a>(value: hms_str_t, what: &str) -> ApiResult<&'a str> {
    let bytes = slice(value.ptr, value.len, what)?;
    std::str::from_utf8(bytes)
        .map_err(|e| ApiError::new(HMS_ERR_INVALID_UTF8, format!("{what} is not UTF-8: {e}")))
}

pub(crate) fn opt_utf8<'a>(value: hms_str_t, what: &str) -> ApiResult<Option<&'a str>> {
    if value.ptr.is_null() && value.len == 0 {
        Ok(None)
    } else {
        utf8(value, what).map(Some)
    }
}

pub(crate) fn opt_u32(value: u32) -> Option<u32> {
    (value != 0).then_some(value)
}

pub(crate) fn flagged(flags: u32, bit: u32, value: f64) -> Option<f64> {
    (flags & bit != 0).then_some(value)
}

/// The `struct_size` a caller declared in the `u32` at offset 0 of its non-null struct.
fn declared_size(caller: *const u8, what: &str) -> ApiResult<usize> {
    // SAFETY: the first field of every caller struct is a `u32` struct_size at offset 0.
    let declared = unsafe { std::ptr::read_unaligned(caller.cast::<u32>()) } as usize;
    if declared < std::mem::size_of::<u32>() {
        return Err(ApiError::new(
            HMS_ERR_ABI_MISMATCH,
            format!("{what}.struct_size {declared} is smaller than 4"),
        ));
    }
    Ok(declared)
}

/// Copy a caller struct whose first field is its own `struct_size` onto `defaults`, taking
/// only the bytes the caller's struct covers so older callers get defaults for newer fields.
pub(crate) fn overlay<T: Copy>(caller: *const T, defaults: T, what: &str) -> ApiResult<T> {
    let caller = caller.cast::<u8>();
    if caller.is_null() {
        return Ok(defaults);
    }
    let declared = declared_size(caller, what)?;
    let mut merged = defaults;
    let take = declared.min(std::mem::size_of::<T>());
    // SAFETY: `take` bytes are readable by the caller's struct_size contract and `merged` is
    // a plain-data value of at least `take` bytes.
    unsafe { std::ptr::copy_nonoverlapping(caller, (&mut merged as *mut T).cast::<u8>(), take) };
    Ok(merged)
}

/// Write `value` to the caller's output struct, covering only the bytes its `struct_size`
/// declares. `struct_size` is the field at offset 0 of both.
pub(crate) fn write_out<T: Copy>(out: *mut T, mut value: T, what: &str) -> ApiResult<()> {
    let out = out.cast::<u8>();
    if out.is_null() {
        return Err(ApiError::new(
            HMS_ERR_NULL_POINTER,
            format!("{what} is NULL"),
        ));
    }
    let declared = declared_size(out, what)?;
    let take = declared.min(std::mem::size_of::<T>());
    // SAFETY: the first field of `T` is the u32 struct_size; echo what the caller declared.
    unsafe { std::ptr::write_unaligned((&mut value as *mut T).cast::<u32>(), declared as u32) };
    // SAFETY: the caller guarantees `declared` writable bytes; `take` never exceeds that.
    unsafe { std::ptr::copy_nonoverlapping((&value as *const T).cast::<u8>(), out, take) };
    Ok(())
}
