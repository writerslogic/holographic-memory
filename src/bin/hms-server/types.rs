// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Response bodies. The handlers serialize these types and the OpenAPI document is generated
//! from the same definitions, so the published schema cannot drift from what is sent.

use schemars::JsonSchema;
use serde::Serialize;
use serde_json::Value;

use crate::tenants::{CompactionStatus, SnapshotInfo};

pub fn body<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("response types serialize")
}

#[derive(Serialize, JsonSchema)]
pub struct Added {
    /// Incoming documents stored (a repeated id counts each occurrence).
    pub added: usize,
}

#[derive(Serialize, JsonSchema)]
pub struct Deleted {
    pub deleted: bool,
}

#[derive(Serialize, JsonSchema)]
pub struct Match {
    pub id: String,
    /// Stored text, empty when the document was stored without text.
    pub text: String,
    pub metadata: Value,
    /// Exact cosine similarity for vector queries; fused rank score for text queries.
    pub score: f64,
}

#[derive(Serialize, JsonSchema)]
pub struct Matches {
    pub matches: Vec<Match>,
}

/// One text search hit. `text` and `metadata` are present only when named in `include`.
#[derive(Serialize, JsonSchema)]
pub struct SearchHit {
    pub id: String,
    /// BM25 score.
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Serialize, JsonSchema)]
pub struct SearchHits {
    pub hits: Vec<SearchHit>,
}

#[derive(Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Maintained {
    pub document_index_trained: bool,
    pub vector_index: VectorIndexBody,
}

/// The vector snapshot that answers unfiltered vector queries (see `--query-index`).
#[derive(Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct VectorIndexBody {
    /// Embedded chunks the snapshot covers; 0 until maintenance builds one.
    pub rows: usize,
    /// Whether a graph index is built over the snapshot (`auto` queries use it).
    pub graph: bool,
    /// Whether the snapshot is the live store. A stale snapshot still answers correctly:
    /// deleted and replaced documents are dropped and new ones scored exactly.
    pub current: bool,
    /// Inserts, replaces and deletes since the snapshot was captured.
    pub stale_mutations: u64,
    /// Whether the next index maintenance or compaction will capture a fresh snapshot.
    pub rebuild_recommended: bool,
}

impl From<holographic_memory::core::documents::VectorIndexStatus> for VectorIndexBody {
    fn from(status: holographic_memory::core::documents::VectorIndexStatus) -> Self {
        Self {
            rows: status.rows,
            graph: status.graph,
            current: status.current,
            stale_mutations: status.stale_mutations,
            rebuild_recommended: status.rebuild_recommended,
        }
    }
}

#[derive(Serialize, JsonSchema)]
pub struct LineError {
    /// 1-based line number in the stream.
    pub line: u64,
    pub message: String,
}

#[derive(Serialize, JsonSchema)]
pub struct StreamResult {
    pub added: u64,
    pub rejected: u64,
    /// The first rejected lines (at most 100).
    pub errors: Vec<LineError>,
    pub errors_truncated: bool,
}

#[derive(Serialize, JsonSchema)]
pub struct ListedDocument {
    pub id: String,
    pub metadata: Value,
}

#[derive(Serialize, JsonSchema)]
pub struct DocumentPage {
    pub documents: Vec<ListedDocument>,
    /// Pass as `after` for the next page; null on the last page.
    pub next_cursor: Option<String>,
    /// Total documents in the tenant.
    pub count: usize,
}

#[derive(Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub tenant: String,
    pub documents: usize,
    pub vectors: usize,
    pub document_index_trained: bool,
    pub vector_index: VectorIndexBody,
    pub used_bytes: f64,
    pub capacity_bytes: f64,
}

#[derive(Serialize, JsonSchema)]
pub struct Snapshot {
    pub id: String,
    pub bytes: u64,
    pub created_at: String,
}

impl From<&SnapshotInfo> for Snapshot {
    fn from(s: &SnapshotInfo) -> Self {
        Self {
            id: s.id.clone(),
            bytes: s.bytes,
            created_at: s.created_at.clone(),
        }
    }
}

#[derive(Serialize, JsonSchema)]
pub struct Snapshots {
    pub snapshots: Vec<Snapshot>,
}

#[derive(Serialize, JsonSchema)]
pub struct Restored {
    pub restored: bool,
    /// The replaced store, kept as a snapshot with this id.
    pub previous_snapshot: String,
}

#[derive(Serialize, JsonSchema)]
pub struct CompactionView {
    pub running: bool,
    pub runs: u64,
    pub failures: u64,
    pub last_trigger: Option<String>,
    pub last_started_at: Option<String>,
    pub last_duration_ms: Option<u64>,
    pub last_error: Option<String>,
}

impl From<&CompactionStatus> for CompactionView {
    fn from(c: &CompactionStatus) -> Self {
        Self {
            running: c.running,
            runs: c.runs,
            failures: c.failures,
            last_trigger: c.last_trigger.map(str::to_string),
            last_started_at: c.last_started_at.clone(),
            last_duration_ms: c.last_duration_ms,
            last_error: c.last_error.map(str::to_string),
        }
    }
}

#[derive(Serialize, JsonSchema)]
pub struct CompactionStatusBody {
    /// 0 when scheduled compaction is off.
    pub scheduled_interval_seconds: u64,
    pub status: CompactionView,
}

#[derive(Serialize, JsonSchema)]
pub struct CompactionRun {
    pub status: CompactionView,
}

#[derive(Serialize, JsonSchema)]
pub struct KeysReloaded {
    pub keys: usize,
}

#[derive(Serialize, JsonSchema)]
pub struct TlsReloaded {
    pub reloaded: bool,
}

#[derive(Serialize, JsonSchema)]
pub struct Tenants {
    /// Tenants with a store on disk.
    pub tenants: Vec<String>,
    /// Stores currently open.
    pub open: usize,
}

#[derive(Serialize, JsonSchema)]
pub struct Health {
    pub status: String,
}

/// RFC 9457 problem details, plus the deprecated `error` object kept for v1 SDK clients.
#[derive(Serialize, JsonSchema)]
pub struct Problem {
    #[serde(rename = "type")]
    pub kind: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
    pub code: String,
    pub instance: Option<String>,
    pub request_id: Option<String>,
    pub error: LegacyError,
}

#[derive(Serialize, JsonSchema)]
pub struct LegacyError {
    pub code: String,
    pub message: String,
}
