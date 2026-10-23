// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Pure readers for the persisted artefacts: byte slices in, findings out. None of them
//! panics, none allocates more than a bounded multiple of its input, and none touches the
//! filesystem, so `fuzz/fuzz_targets/format_readers.rs` can drive every one with arbitrary
//! bytes. The store-level verifier ([`super::verify`]) is built from them.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::{
    AUDIT_ENTRY_LEN, BINARY_TRANSACTION_MAGIC, END_MARK, FRAME_HEADER_LEN, MAX_FRAME_PAYLOAD,
    TEXT_SEGMENT_HEADER_LEN, TEXT_SEGMENT_MAGIC, TRANSACTION_MAGIC, TRANSACTION_MAGIC_PREFIX,
};

/// Location and header fields of one journal frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRef {
    /// Byte offset inside the segment file.
    pub offset: usize,
    /// Decompressed payload length.
    pub raw_len: u32,
    /// Stored payload length; equal to `raw_len` when the payload is not compressed.
    pub comp_len: u32,
    /// Writer sequence number (header word at +12). Advisory: see `docs/file-format/JOURNAL.md`.
    pub seq: u32,
    /// CRC32 of the decompressed payload.
    pub crc: u32,
}

impl FrameRef {
    /// Offset of the first byte after the frame.
    pub fn end(&self) -> usize {
        self.offset + FRAME_HEADER_LEN + self.comp_len as usize
    }
}

/// Why the walk over a segment stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentEnd {
    /// A 16-byte end-of-log marker follows the last frame.
    Marker,
    /// An all-zero header follows the last frame (stores written before the marker).
    ZeroHeader,
    /// Fewer than 16 bytes remain after the last frame.
    Exhausted,
    /// A frame fails its checks and nothing committed follows it: an append the crash
    /// interrupted. The engine truncates the log there on open (final segment only).
    TornTail { reason: &'static str },
    /// A frame fails its checks and committed data follows it (or it sits in a segment that
    /// was complete): damage inside committed data. The engine refuses to open.
    Corrupt {
        reason: &'static str,
        /// Offset of the committed data found after the damage, if any.
        committed_at: Option<usize>,
    },
}

/// Result of walking one segment.
#[derive(Debug, Clone)]
pub struct SegmentScan {
    pub frames: Vec<FrameRef>,
    /// Offset where the walk stopped: the end of the last good frame.
    pub end_offset: usize,
    pub end: SegmentEnd,
}

fn word(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

/// True if the 16 bytes at `at` are an end-of-log marker or the all-zero legacy header.
pub fn is_log_end(data: &[u8], at: usize) -> bool {
    at.checked_add(FRAME_HEADER_LEN)
        .is_some_and(|end| end <= data.len())
        && word(data, at + 4) == 0
        && word(data, at + 8) == 0
        && ((word(data, at) == 0 && word(data, at + 12) == END_MARK)
            || data[at..at + FRAME_HEADER_LEN].iter().all(|&b| b == 0))
}

fn is_end_marker(data: &[u8], at: usize) -> bool {
    is_log_end(data, at) && word(data, at + 12) == END_MARK
}

/// Decodes the frame at `offset`: header sanity, payload bounds, LZ4, CRC32. Returns the
/// frame and its decompressed payload. The checks are the engine's (`frame_checks_out`),
/// plus a ratio bound that keeps a hostile `raw_len` from forcing a large allocation: an LZ4
/// block never expands by more than 255 times.
pub fn decode_frame(data: &[u8], offset: usize) -> Result<(FrameRef, Vec<u8>), &'static str> {
    let mut payload = Vec::new();
    let frame = decode_frame_into(data, offset, &mut payload)?;
    Ok((frame, payload))
}

/// [`decode_frame`] into `out`, which holds exactly the decoded payload on success (and
/// anything on failure) and keeps its capacity for the next call.
fn decode_frame_into(
    data: &[u8],
    offset: usize,
    out: &mut Vec<u8>,
) -> Result<FrameRef, &'static str> {
    let header_end = offset
        .checked_add(FRAME_HEADER_LEN)
        .filter(|&end| end <= data.len())
        .ok_or("truncated-header")?;
    let (crc, raw_len, comp_len, seq) = (
        word(data, offset),
        word(data, offset + 4),
        word(data, offset + 8),
        word(data, offset + 12),
    );
    if raw_len == 0 || comp_len == 0 || raw_len as usize > MAX_FRAME_PAYLOAD || comp_len > raw_len {
        return Err("header-lengths");
    }
    let end = header_end
        .checked_add(comp_len as usize)
        .filter(|&end| end <= data.len())
        .ok_or("truncated-payload")?;
    let payload = &data[header_end..end];
    if comp_len < raw_len {
        if u64::from(raw_len) > u64::from(comp_len) * 255 {
            return Err("implausible-ratio");
        }
        // The output bound `lz4_flex::decompress(payload, raw_len)` gives; `out` is only
        // zero-filled where it grows.
        let raw_len = raw_len as usize;
        if out.len() < raw_len {
            out.resize(raw_len, 0);
        }
        let written =
            lz4_flex::block::decompress_into(payload, &mut out[..raw_len]).map_err(|_| "lz4")?;
        out.truncate(written);
    } else {
        out.clear();
        out.extend_from_slice(payload);
    }
    if crc32fast::hash(out) != crc {
        return Err("crc32");
    }
    Ok(FrameRef {
        offset,
        raw_len,
        comp_len,
        seq,
        crc,
    })
}

/// The first offset at or after `from` holding committed data: a frame that decodes, or an
/// end marker. Mirrors the engine's rule for telling a torn append from corruption: the
/// writer puts the end marker in the same flush as the frame, so anything valid after a bad
/// frame was committed after it.
fn committed_after(data: &[u8], from: usize) -> Option<usize> {
    let last = data.iter().rposition(|&b| b != 0)?;
    // One output buffer for every candidate offset of a damaged tail.
    let mut scratch = Vec::new();
    (from..=last.saturating_sub(FRAME_HEADER_LEN - 1)).find(|&at| {
        at + FRAME_HEADER_LEN <= data.len()
            && (is_end_marker(data, at) || decode_frame_into(data, at, &mut scratch).is_ok())
    })
}

/// Walks the frames of one segment file. `is_final` selects the engine's rule for the active
/// segment: a bad frame with nothing committed after it is a torn tail, not corruption.
/// `visit` receives each good frame and its decompressed payload.
pub fn scan_segment(
    data: &[u8],
    is_final: bool,
    mut visit: impl FnMut(&FrameRef, &[u8]),
) -> SegmentScan {
    let mut frames = Vec::new();
    let mut offset = 0usize;
    let finish = |frames: Vec<FrameRef>, offset: usize, end: SegmentEnd| SegmentScan {
        frames,
        end_offset: offset,
        end,
    };
    while offset + FRAME_HEADER_LEN <= data.len() {
        if word(data, offset + 4) == 0 && word(data, offset + 8) == 0 {
            if is_log_end(data, offset) {
                let end = if is_end_marker(data, offset) {
                    SegmentEnd::Marker
                } else {
                    SegmentEnd::ZeroHeader
                };
                return finish(frames, offset, end);
            }
            return fault(data, is_final, frames, offset, "invalid-end-header");
        }
        match decode_frame(data, offset) {
            Ok((frame, payload)) => {
                visit(&frame, &payload);
                offset = frame.end();
                frames.push(frame);
            }
            Err(reason) => return fault(data, is_final, frames, offset, reason),
        }
    }
    finish(frames, offset, SegmentEnd::Exhausted)
}

fn fault(
    data: &[u8],
    is_final: bool,
    frames: Vec<FrameRef>,
    offset: usize,
    reason: &'static str,
) -> SegmentScan {
    let committed_at = committed_after(data, offset + 1);
    let end = if is_final && committed_at.is_none() {
        SegmentEnd::TornTail { reason }
    } else {
        SegmentEnd::Corrupt {
            reason,
            committed_at,
        }
    };
    SegmentScan {
        frames,
        end_offset: offset,
        end,
    }
}

/// What a frame payload is, before any of it is decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadKind<'a> {
    /// `HMS-TXN-2\0` followed by the JSON mutation array.
    Transaction(&'a [u8]),
    /// `HMS-TXN-3\0` followed by the binary mutations; [`transaction_json`] reads them as the
    /// JSON array an `HMS-TXN-2` frame of the same mutations holds.
    BinaryTransaction(&'a [u8]),
    /// `HMS-TXN-<n>\0` with an `n` this build does not read.
    OtherTransactionVersion(String),
    /// No journal magic: damaged, foreign, or still encrypted.
    Unrecognized,
}

pub fn classify_payload(payload: &[u8]) -> PayloadKind<'_> {
    if let Some(json) = payload.strip_prefix(TRANSACTION_MAGIC) {
        return PayloadKind::Transaction(json);
    }
    if let Some(body) = payload.strip_prefix(BINARY_TRANSACTION_MAGIC) {
        return PayloadKind::BinaryTransaction(body);
    }
    if let Some(rest) = payload.strip_prefix(TRANSACTION_MAGIC_PREFIX) {
        let digits = rest.iter().take_while(|b| b.is_ascii_digit()).count();
        if digits > 0 && digits <= 9 && rest.get(digits) == Some(&0) {
            return PayloadKind::OtherTransactionVersion(
                String::from_utf8_lossy(&rest[..digits]).into_owned(),
            );
        }
    }
    PayloadKind::Unrecognized
}

/// Mutation variants the journal may carry (`docs/file-format/MUTATIONS.md`).
pub const MUTATION_VARIANTS: [&str; 14] = [
    "Vector",
    "Atom",
    "Composite",
    "StoredTriple",
    "Document",
    "EncodeDocument",
    "Derived",
    "DeleteDocument",
    "Delete",
    "Triplet",
    "Relation",
    "RemoveRelation",
    "RelationType",
    "Rule",
];

/// Limits the engine enforces on replay (`validate_mutation`).
const MAX_TRANSACTION_OPS: usize = 4096;
const MAX_ID_BYTES: usize = u16::MAX as usize;
const MAX_DOCUMENT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_CHUNK_BYTES: u64 = 64 * 1024;
const MAX_CHUNKS: usize = 4096;
/// The `$hms_positions` version of whole-document postings.
const WHOLE_DOCUMENT_POSITIONS: u64 = 2;

/// The longest byte range one stored chunk may cover. Whole-document postings cover the single
/// lexical chunk of their document, from byte 0, up to the document limit; any other chunk is
/// bounded by the chunk limit. `None` when whole-document postings break those conditions.
/// Replay (`validate_mutation`) and this verifier both apply this rule.
pub(crate) fn chunk_byte_limit(
    whole_document: bool,
    chunk_count: usize,
    start_byte: u64,
    lexical: bool,
) -> Option<u64> {
    if !whole_document {
        return Some(MAX_CHUNK_BYTES);
    }
    (lexical && chunk_count == 1 && start_byte == 0).then_some(MAX_DOCUMENT_BYTES)
}

/// Live state implied by replaying transactions, computed without the engine. Used for the
/// summary figures and for the deep cross-check against an opened store.
#[derive(Debug, Clone, Default)]
pub struct Replay {
    pub documents: BTreeMap<String, DocumentSummary>,
    pub vectors: BTreeSet<String>,
    pub derived: BTreeMap<String, Vec<String>>,
    pub relations: BTreeSet<(String, String, String)>,
    pub mutations: usize,
}

#[derive(Debug, Clone)]
pub struct DocumentSummary {
    pub chunks: usize,
    /// Chunks whose vector is in the vector index: those without a `vector_state` (encoded
    /// at ingest) plus pending ones an `EncodeDocument` installed.
    pub indexed: usize,
    pub version: String,
    /// Carries a memory-API stamp: a `mem1:` version or a `_mem` metadata key.
    pub stamped: bool,
}

impl Replay {
    /// Vectors the engine holds: standalone vectors plus one per indexed document chunk.
    pub fn vector_total(&self) -> usize {
        self.vectors.len() + self.documents.values().map(|d| d.indexed).sum::<usize>()
    }

    fn drop_derived(&mut self, id: &str) {
        if let Some(children) = self.derived.remove(id) {
            for child in children {
                self.vectors.remove(&child);
            }
        }
    }

    fn apply(&mut self, variant: &str, body: &Value) {
        let id = body.get("id").and_then(Value::as_str).unwrap_or_default();
        match variant {
            "Vector" | "Composite" => {
                if variant == "Vector" {
                    self.drop_derived(id);
                }
                self.vectors.insert(id.to_owned());
            }
            "Triplet" => {
                self.drop_derived(id);
                self.vectors.insert(id.to_owned());
            }
            "Document" => {
                let version = body
                    .get("version")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let stamped = version.starts_with("mem1:")
                    || body
                        .get("metadata")
                        .and_then(Value::as_object)
                        .is_some_and(|m| m.contains_key("_mem"));
                let chunks = body.get("chunks").and_then(Value::as_array);
                self.documents.insert(
                    id.to_owned(),
                    DocumentSummary {
                        chunks: chunks.map_or(0, Vec::len),
                        indexed: chunks.map_or(0, |chunks| {
                            chunks
                                .iter()
                                .filter(|chunk| chunk.get("vector_state").is_none())
                                .count()
                        }),
                        version: version.to_owned(),
                        stamped,
                    },
                );
            }
            "EncodeDocument" => {
                // The engine also requires the content fingerprint to match; the scanner has
                // no chunk text, so it assumes the common case (the document is unchanged).
                let version = body
                    .get("version")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let vectors = body
                    .get("vectors")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                if let Some(document) = self.documents.get_mut(id) {
                    if document.version == version && document.chunks == vectors {
                        document.indexed = document.chunks;
                    }
                }
            }
            "DeleteDocument" => {
                self.documents.remove(id);
            }
            "Delete" => {
                self.drop_derived(id);
                self.vectors.remove(id);
            }
            "Derived" => {
                let children = body
                    .get("children")
                    .and_then(Value::as_array)
                    .map(|c| {
                        c.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default();
                self.derived.insert(id.to_owned(), children);
            }
            "Relation" => {
                self.relations
                    .insert(relation_key(body, "sourceId", "relationType", "targetId"));
            }
            "RemoveRelation" => {
                self.relations
                    .remove(&relation_key(body, "source", "relation", "target"));
            }
            _ => {}
        }
    }
}

fn relation_key(body: &Value, a: &str, b: &str, c: &str) -> (String, String, String) {
    let get = |k: &str| {
        body.get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    (get(a), get(b), get(c))
}

fn valid_id(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|id| !id.is_empty() && id.len() <= MAX_ID_BYTES && !id.contains('\0'))
}

fn valid_vector(value: Option<&Value>, dimensions: Option<u64>) -> Result<(), String> {
    let vector = value
        .and_then(Value::as_object)
        .ok_or("vector is not an object")?;
    let dim = vector
        .get("dim")
        .and_then(Value::as_u64)
        .ok_or("vector has no integer dim")?;
    if dimensions.is_some_and(|d| d != dim) {
        return Err(format!("vector dim {dim} differs from the store's"));
    }
    let indices = vector
        .get("indices")
        .and_then(Value::as_array)
        .ok_or("vector has no indices array")?;
    let mut previous: Option<u64> = None;
    for index in indices {
        let index = index.as_u64().ok_or("vector index is not an integer")?;
        if index >= dim || previous.is_some_and(|p| p >= index) {
            return Err("vector indices are not strictly ascending below dim".into());
        }
        previous = Some(index);
    }
    Ok(())
}

fn check_document(body: &Value, dimensions: Option<u64>) -> Result<(), String> {
    if !valid_id(body.get("id")) {
        return Err("document id is not 1..=65535 bytes without NUL".into());
    }
    let chunks = body
        .get("chunks")
        .and_then(Value::as_array)
        .ok_or("document has no chunks array")?;
    if chunks.is_empty() || chunks.len() > MAX_CHUNKS {
        return Err(format!(
            "document has {} chunks, expected 1..=4096",
            chunks.len()
        ));
    }
    for chunk in chunks {
        if !valid_id(chunk.get("id")) {
            return Err("chunk id is not 1..=65535 bytes without NUL".into());
        }
        valid_vector(chunk.get("vector"), dimensions)?;
        let start = chunk.get("start_byte").and_then(Value::as_u64);
        let end = chunk.get("end_byte").and_then(Value::as_u64);
        let (Some(start), Some(end)) = (start, end) else {
            return Err("chunk byte range is missing".into());
        };
        let whole_document = chunk
            .get("terms")
            .and_then(|terms| terms.get("$hms_positions"))
            .and_then(|positions| positions.get("version"))
            .and_then(Value::as_u64)
            == Some(WHOLE_DOCUMENT_POSITIONS);
        let lexical = chunk.get("vector_state").and_then(Value::as_str) == Some("lexical");
        let Some(limit) = chunk_byte_limit(whole_document, chunks.len(), start, lexical) else {
            return Err("whole-document postings require one lexical chunk from byte 0".into());
        };
        if start >= end || end > MAX_DOCUMENT_BYTES || end - start > limit {
            return Err("chunk byte range is empty or over the limits".into());
        }
        if let Some(text) = chunk.get("text").and_then(Value::as_str) {
            if text.len() as u64 != end - start {
                return Err("chunk text length differs from its byte range".into());
            }
        }
        if let Some(state) = chunk.get("vector_state") {
            let known = matches!(state.as_str(), Some("pending" | "lexical"));
            let empty = chunk
                .get("vector")
                .and_then(|v| v.get("indices"))
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty);
            if !known || !empty {
                return Err(
                    "chunk vector_state must be pending or lexical with an empty vector".into(),
                );
            }
        }
    }
    Ok(())
}

/// Checks an `EncodeDocument` body: a valid id, a version of 1..=1024 bytes and 1..=4096
/// well-formed vectors.
fn check_encode_document(body: &Value, dimensions: Option<u64>) -> Result<(), String> {
    if !valid_id(body.get("id")) {
        return Err("document id is not 1..=65535 bytes without NUL".into());
    }
    let version_ok = body
        .get("version")
        .and_then(Value::as_str)
        .is_some_and(|v| !v.is_empty() && v.len() <= 1024);
    if !version_ok {
        return Err("deferred encoding version is not 1..=1024 bytes".into());
    }
    let vectors = body
        .get("vectors")
        .and_then(Value::as_array)
        .ok_or("deferred encoding has no vectors array")?;
    if vectors.is_empty() || vectors.len() > MAX_CHUNKS {
        return Err(format!(
            "deferred encoding has {} vectors, expected 1..=4096",
            vectors.len()
        ));
    }
    for vector in vectors {
        valid_vector(Some(vector), dimensions)?;
    }
    Ok(())
}

/// Checks one mutation against what replay requires and applies it to `replay`.
fn check_mutation(
    value: &Value,
    dimensions: Option<u64>,
    replay: &mut Replay,
) -> Result<(), String> {
    let object = value.as_object().ok_or("mutation is not an object")?;
    if object.len() != 1 {
        return Err("mutation must have exactly one variant key".into());
    }
    let (variant, body) = object.iter().next().ok_or("empty mutation")?;
    if !MUTATION_VARIANTS.contains(&variant.as_str()) {
        return Err(format!("unknown mutation variant {variant:?}"));
    }
    match variant.as_str() {
        "Vector" | "Atom" | "Composite" => {
            if !valid_id(body.get("id")) {
                return Err("vector id is not 1..=65535 bytes without NUL".into());
            }
            valid_vector(body.get("vector"), dimensions)?;
        }
        "Document" => check_document(body, dimensions)?,
        "EncodeDocument" => check_encode_document(body, dimensions)?,
        "Delete" | "DeleteDocument" => {
            if !valid_id(body.get("id")) {
                return Err("id is not 1..=65535 bytes without NUL".into());
            }
        }
        "Derived" => {
            if !valid_id(body.get("id")) || !body.get("children").is_some_and(Value::is_array) {
                return Err("derived record needs an id and children".into());
            }
        }
        "Triplet" => {
            for key in ["id", "subject", "relation", "object"] {
                if !valid_id(body.get(key)) {
                    return Err(format!("triplet {key} is not a valid id"));
                }
            }
        }
        "Relation" => {
            for key in ["sourceId", "relationType", "targetId"] {
                if !valid_id(body.get(key)) {
                    return Err(format!("relation {key} is not a valid id"));
                }
            }
        }
        "RemoveRelation" => {
            for key in ["source", "relation", "target"] {
                if !valid_id(body.get(key)) {
                    return Err(format!("relation removal {key} is not a valid id"));
                }
            }
        }
        _ => {
            if !body.is_object() {
                return Err(format!("{variant} body is not an object"));
            }
        }
    }
    replay.apply(variant, body);
    replay.mutations += 1;
    Ok(())
}

/// The JSON mutation array of a transaction payload of either journal format (`None` for
/// any other payload); a malformed binary payload is the problem as text.
pub fn transaction_json(payload: &[u8]) -> Option<Result<std::borrow::Cow<'_, [u8]>, String>> {
    crate::core::engine::journal::payload_json(payload)
        .map(|json| json.map_err(|error| format!("{error:#}")))
}

/// Decodes a transaction's JSON and checks every mutation, applying them to `replay`.
/// Returns the first problem as text.
pub fn check_transaction(
    json: &[u8],
    dimensions: Option<u64>,
    replay: &mut Replay,
) -> Result<usize, String> {
    let value: Value = serde_json::from_slice(json).map_err(|e| format!("not JSON: {e}"))?;
    let mutations = value.as_array().ok_or("transaction is not an array")?;
    if mutations.is_empty() || mutations.len() > MAX_TRANSACTION_OPS {
        return Err(format!(
            "transaction has {} operations, expected 1..=4096",
            mutations.len()
        ));
    }
    for (n, mutation) in mutations.iter().enumerate() {
        check_mutation(mutation, dimensions, replay).map_err(|e| format!("mutation {n}: {e}"))?;
    }
    Ok(mutations.len())
}

/// The header of `documents.text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextHeader {
    /// `HMST`, a version, and the flag byte (bit 0: records are encrypted).
    Versioned { version: u8, flags: u8 },
    /// No `HMST` magic: a version-1 segment (plaintext, no header) or a foreign file.
    Headerless,
}

pub fn parse_text_header(bytes: &[u8]) -> TextHeader {
    if bytes.len() >= TEXT_SEGMENT_HEADER_LEN && &bytes[..4] == TEXT_SEGMENT_MAGIC {
        TextHeader::Versioned {
            version: bytes[4],
            flags: bytes[5],
        }
    } else {
        TextHeader::Headerless
    }
}

/// Result of reading `audit.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditScan {
    pub entries: usize,
    /// Bytes after the last whole entry: an append the crash interrupted.
    pub trailing_bytes: usize,
    /// Entries whose operation byte is not 1..=3; the reader skips them.
    pub unknown_ops: usize,
}

pub fn scan_audit(bytes: &[u8]) -> AuditScan {
    let entries = bytes.len() / AUDIT_ENTRY_LEN;
    let unknown_ops = (0..entries)
        .filter(|i| !(1..=3).contains(&bytes[i * AUDIT_ENTRY_LEN + 8]))
        .count();
    AuditScan {
        entries,
        trailing_bytes: bytes.len() % AUDIT_ENTRY_LEN,
        unknown_ops,
    }
}

/// `indices.json`: the checkpoint that binds the cache files to the journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndicesMeta {
    /// Arena generation directory name and journal length in bytes it was written for.
    pub checkpoint: (String, u64),
    pub shard_count: u64,
    pub shards: Vec<IndexShardMeta>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexShardMeta {
    pub nsg: bool,
    pub ivf: bool,
    pub nsg_crc: Option<u32>,
    pub ivf_crc: Option<u32>,
}

pub fn parse_indices(bytes: &[u8]) -> Result<IndicesMeta, String> {
    let value: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let checkpoint = value
        .get("checkpoint")
        .and_then(Value::as_array)
        .filter(|c| c.len() == 2)
        .and_then(|c| Some((c[0].as_str()?.to_owned(), c[1].as_u64()?)))
        .ok_or("checkpoint is not [string, integer]")?;
    let shard_count = value
        .get("shardCount")
        .and_then(Value::as_u64)
        .ok_or("shardCount is not an integer")?;
    let shards = value
        .get("shards")
        .and_then(Value::as_array)
        .ok_or("shards is not an array")?
        .iter()
        .map(|s| IndexShardMeta {
            nsg: s.get("nsg").and_then(Value::as_bool).unwrap_or(false),
            ivf: s.get("ivf").and_then(Value::as_bool).unwrap_or(false),
            nsg_crc: s
                .get("nsgCrc")
                .and_then(Value::as_u64)
                .and_then(|c| u32::try_from(c).ok()),
            ivf_crc: s
                .get("ivfCrc")
                .and_then(Value::as_u64)
                .and_then(|c| u32::try_from(c).ok()),
        })
        .collect();
    Ok(IndicesMeta {
        checkpoint,
        shard_count,
        shards,
    })
}

/// Runs every reader on arbitrary bytes. For the fuzz target: it must return, never panic,
/// and allocate no more than a small multiple of `bytes.len()`.
pub fn probe(bytes: &[u8]) -> usize {
    let mut touched = 0usize;
    let mut replay = Replay::default();
    for is_final in [true, false] {
        let scan = scan_segment(bytes, is_final, |_, payload| {
            match classify_payload(payload) {
                PayloadKind::Transaction(json) => {
                    touched += check_transaction(json, None, &mut replay).unwrap_or(0);
                }
                _ => touched += 1,
            }
        });
        touched += scan.frames.len();
    }
    if let PayloadKind::Transaction(json) = classify_payload(bytes) {
        touched += check_transaction(json, Some(1024), &mut replay).unwrap_or(0);
    }
    touched += usize::from(parse_text_header(bytes) != TextHeader::Headerless);
    touched += scan_audit(bytes).entries;
    touched += parse_indices(bytes).map_or(0, |m| m.shards.len());
    touched += serde_json::from_slice::<super::super::storage::FormatManifest>(bytes)
        .map_or(0, |m| m.version as usize);
    #[cfg(feature = "experimental")]
    {
        touched += usize::from(crate::core::qgraph::ivf::QuantizedIvf::from_bytes(bytes).is_ok());
    }
    touched
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(payload: &[u8], seq: u32) -> Vec<u8> {
        let compressed = lz4_flex::compress(payload);
        let (body, comp_len) = if compressed.len() < payload.len() {
            (compressed.clone(), compressed.len())
        } else {
            (payload.to_vec(), payload.len())
        };
        let mut out = Vec::new();
        out.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(comp_len as u32).to_le_bytes());
        out.extend_from_slice(&seq.to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn marker() -> [u8; 16] {
        let mut m = [0u8; 16];
        m[12..].copy_from_slice(&END_MARK.to_le_bytes());
        m
    }

    #[test]
    fn walks_frames_and_classifies_the_end() {
        let mut segment = frame(&[b'a'; 300], 0);
        segment.extend(frame(b"short", 1));
        segment.extend(marker());
        segment.resize(4096, 0);
        let scan = scan_segment(&segment, true, |_, _| {});
        assert_eq!(scan.frames.len(), 2);
        assert_eq!(scan.end, SegmentEnd::Marker);
        assert_eq!(scan.frames[1].end(), scan.end_offset);
        assert!(scan.frames[0].comp_len < scan.frames[0].raw_len);

        segment[scan.end_offset..scan.end_offset + 16].fill(0);
        assert_eq!(
            scan_segment(&segment, true, |_, _| {}).end,
            SegmentEnd::ZeroHeader
        );
    }

    /// A damaged frame is a torn tail only when nothing committed follows it, and only in the
    /// final segment.
    #[test]
    fn tells_a_torn_tail_from_corruption() {
        let first = frame(b"first frame payload", 0);
        let second = frame(b"second frame payload", 1);
        let mut log = first.clone();
        log.extend(&second);
        log.extend(marker());
        log.resize(2048, 0);

        let mut torn = log.clone();
        let cut = first.len() + 20;
        torn[cut..].fill(0);
        let scan = scan_segment(&torn, true, |_, _| {});
        assert!(
            matches!(scan.end, SegmentEnd::TornTail { .. }),
            "{:?}",
            scan.end
        );
        assert_eq!(scan.end_offset, first.len());
        assert!(matches!(
            scan_segment(&torn, false, |_, _| {}).end,
            SegmentEnd::Corrupt { .. }
        ));

        let mut flipped = log.clone();
        flipped[16 + 3] ^= 0xff;
        let scan = scan_segment(&flipped, true, |_, _| {});
        assert!(matches!(
            scan.end,
            SegmentEnd::Corrupt {
                committed_at: Some(_),
                ..
            }
        ));
        assert!(scan.frames.is_empty());

        let mut last = log;
        last[first.len() + 16 + 2] ^= 0xff;
        assert!(matches!(
            scan_segment(&last, true, |_, _| {}).end,
            SegmentEnd::Corrupt {
                committed_at: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn rejects_hostile_lengths_without_allocating_them() {
        let mut data = vec![0u8; 64];
        data[4..8].copy_from_slice(&(MAX_FRAME_PAYLOAD as u32).to_le_bytes());
        data[8..12].copy_from_slice(&16u32.to_le_bytes());
        assert_eq!(decode_frame(&data, 0).unwrap_err(), "implausible-ratio");
        data[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode_frame(&data, 0).unwrap_err(), "header-lengths");
        data[4..8].copy_from_slice(&8u32.to_le_bytes());
        data[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode_frame(&data, 0).unwrap_err(), "header-lengths");
    }

    #[test]
    fn classifies_journal_versions() {
        assert!(matches!(
            classify_payload(b"HMS-TXN-2\0[]"),
            PayloadKind::Transaction(b"[]")
        ));
        assert_eq!(
            classify_payload(b"HMS-TXN-3\0[]"),
            PayloadKind::BinaryTransaction(b"[]")
        );
        assert_eq!(
            classify_payload(b"HMS-TXN-4\0"),
            PayloadKind::OtherTransactionVersion("4".into())
        );
        assert_eq!(classify_payload(b"HMS-TXN-\0"), PayloadKind::Unrecognized);
        assert_eq!(classify_payload(b"HMS-TXN-2"), PayloadKind::Unrecognized);
    }

    #[test]
    fn replays_the_live_state() {
        let json = br#"[
            {"Document":{"id":"a","version":"mem1:{}","metadata":{},"chunks":[
                {"id":"hms:chunk:a:0","start_byte":0,"end_byte":3,"text":"abc",
                 "vector":{"dim":8,"indices":[1,4]}}]}},
            {"Vector":{"id":"v","vector":{"dim":8,"indices":[]}}},
            {"Document":{"id":"b","version":"1","metadata":{"_mem":{}},"chunks":[
                {"id":"hms:chunk:b:0","start_byte":0,"end_byte":1,"vector":{"dim":8,"indices":[2]}}]}},
            {"DeleteDocument":{"id":"b"}}
        ]"#;
        let mut replay = Replay::default();
        assert_eq!(check_transaction(json, Some(8), &mut replay), Ok(4));
        assert_eq!(replay.documents.len(), 1);
        assert!(replay.documents["a"].stamped);
        assert_eq!(replay.vector_total(), 2);
    }

    #[test]
    fn rejects_what_replay_rejects() {
        let bad = |json: &str| check_transaction(json.as_bytes(), Some(8), &mut Replay::default());
        assert!(bad("[]").is_err());
        assert!(bad(r#"[{"Mystery":{}}]"#).is_err());
        assert!(bad(r#"[{"Delete":{"id":""}}]"#).is_err());
        assert!(bad(r#"[{"Delete":{"id":"a\u0000b"}}]"#).is_err());
        assert!(bad(r#"[{"Vector":{"id":"v","vector":{"dim":8,"indices":[3,3]}}}]"#).is_err());
        assert!(bad(r#"[{"Vector":{"id":"v","vector":{"dim":8,"indices":[8]}}}]"#).is_err());
        assert!(bad(r#"[{"Vector":{"id":"v","vector":{"dim":9,"indices":[1]}}}]"#).is_err());
        assert!(bad(r#"[{"Vector":{"id":"v","vector":{"dim":8,"indices":[7]}}}]"#).is_ok());
    }

    #[test]
    fn whole_document_chunks_reach_the_document_limit() {
        let document = |chunks: &str| {
            format!(r#"[{{"Document":{{"id":"w","version":"1","chunks":[{chunks}]}}}}]"#)
        };
        let chunk = |start: u64, end: u64, state: &str, version: u8| {
            format!(
                r#"{{"id":"hms:chunk:w:{start}","start_byte":{start},"end_byte":{end},"vector":{{"dim":8,"indices":[]}},"vector_state":"{state}","terms":{{"$hms_positions":{{"version":{version},"offsets":[0,1],"data":[0]}},"red":1}}}}"#
            )
        };
        let check = |json: String| {
            check_transaction(json.as_bytes(), Some(8), &mut Replay::default()).is_ok()
        };
        let over = MAX_CHUNK_BYTES + 1;
        assert!(check(document(&chunk(0, over, "lexical", 2))));
        assert!(check(document(&chunk(0, MAX_DOCUMENT_BYTES, "lexical", 2))));
        assert!(!check(document(&chunk(
            0,
            MAX_DOCUMENT_BYTES + 1,
            "lexical",
            2
        ))));
        assert!(!check(document(&chunk(1, over, "lexical", 2))));
        assert!(!check(document(&chunk(0, over, "pending", 2))));
        assert!(!check(document(&chunk(0, over, "lexical", 1))));
        assert!(check(document(&chunk(0, MAX_CHUNK_BYTES, "lexical", 1))));
        let pair = format!(
            "{},{}",
            chunk(0, 2, "lexical", 2),
            chunk(2, 3, "lexical", 2)
        );
        assert!(!check(document(&pair)));
    }

    #[test]
    fn probe_survives_garbage() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for len in [0usize, 1, 15, 16, 17, 100, 5000] {
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 33) as u8
                })
                .collect();
            probe(&bytes);
        }
    }
}
