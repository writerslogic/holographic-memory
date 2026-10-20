// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! UniFFI surface of the Holographic Memory System document store, for Swift and Kotlin.
//!
//! The capability set matches the Node binding: ingest (single and batch, with chunking,
//! caller embeddings and `titleWeight`), search (BM25 options, filters, hybrid fusion),
//! delete, stats, and snapshot/restore. Every operation that touches the disk has a
//! blocking form and an async form that runs on a worker thread; the async forms become
//! Swift `async` functions and Kotlin `suspend` functions.

mod error;
mod snapshot;
mod types;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use holographic_memory::HmsCore;
use parking_lot::RwLock;

pub use error::HmsError;
pub use types::{
    DocumentChunk, DocumentInput, Fusion, SearchHit, SearchOptions, SnapshotInfo, StoreOptions,
    StoreStats,
};

uniffi::setup_scaffolding!();

/// Version of this binding crate.
#[uniffi::export]
pub fn binding_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// Splits a document into the chunks the store would index, without storing anything.
#[uniffi::export]
pub fn chunk_document(input: DocumentInput) -> Result<Vec<DocumentChunk>, HmsError> {
    guarded(|| {
        let chunks = holographic_memory::chunk_document(&input.into_core()?)?;
        Ok(chunks.into_iter().map(DocumentChunk::from).collect())
    })
}

struct Inner {
    core: RwLock<Option<Arc<HmsCore>>>,
    /// Mutations hold this shared; a snapshot holds it exclusively while it copies files.
    gate: RwLock<()>,
    path: PathBuf,
    dimensions: u32,
    encrypted: bool,
}

/// An open document store. Thread-safe: share one handle across threads and coroutines. A
/// directory can be open once per process; opening it again fails with `StoreLocked`.
#[derive(uniffi::Object)]
pub struct HmsStore {
    inner: Arc<Inner>,
}

impl Inner {
    fn open(path: &str, options: StoreOptions) -> Result<Arc<Self>, HmsError> {
        if path.is_empty() {
            return Err(HmsError::invalid("path must not be empty"));
        }
        let config = options.to_config()?;
        let core = guarded(|| {
            Ok(HmsCore::new(
                options.dimensions,
                Some(path.to_owned()),
                Some(config),
            )?)
        })?;
        Ok(Arc::new(Self {
            core: RwLock::new(Some(Arc::new(core))),
            gate: RwLock::new(()),
            path: PathBuf::from(path),
            dimensions: options.dimensions,
            encrypted: options.encryption_key.is_some(),
        }))
    }

    fn core(&self) -> Result<Arc<HmsCore>, HmsError> {
        self.core.read().clone().ok_or(HmsError::Closed)
    }

    fn ingest(&self, input: DocumentInput) -> Result<u32, HmsError> {
        let input = input.into_core()?;
        let core = self.core()?;
        let _writer = self.gate.read();
        guarded(|| Ok(core.memorize_document(input)?))
    }

    fn ingest_batch(&self, inputs: Vec<DocumentInput>) -> Result<Vec<u32>, HmsError> {
        let inputs = inputs
            .into_iter()
            .map(DocumentInput::into_core)
            .collect::<Result<Vec<_>, _>>()?;
        let core = self.core()?;
        let _writer = self.gate.read();
        guarded(|| Ok(core.memorize_documents(inputs)?))
    }

    fn search(
        &self,
        query: &str,
        options: Option<SearchOptions>,
    ) -> Result<Vec<SearchHit>, HmsError> {
        let options = options.unwrap_or_default().into_core()?;
        let core = self.core()?;
        guarded(|| {
            Ok(core
                .search_documents(query, &options)?
                .into_iter()
                .map(SearchHit::from)
                .collect())
        })
    }

    fn delete_document(&self, id: &str) -> Result<bool, HmsError> {
        let core = self.core()?;
        let _writer = self.gate.read();
        guarded(|| Ok(core.delete_document(id)?))
    }

    fn compact(&self) -> Result<(), HmsError> {
        let core = self.core()?;
        let _writer = self.gate.read();
        guarded(|| Ok(core.compact()?))
    }

    fn train_index(&self) -> Result<(), HmsError> {
        let core = self.core()?;
        let _writer = self.gate.read();
        guarded(|| Ok(core.train_document_index()?))
    }

    fn stats(&self) -> Result<StoreStats, HmsError> {
        let core = self.core()?;
        let health = core.storage_health();
        let disk = directory_bytes(&self.path)?;
        Ok(StoreStats {
            document_count: core.document_count() as u64,
            chunk_count: core.vector_count(),
            dimensions: self.dimensions,
            embedded_chunk_count: core.document_embedding_count() as u64,
            dense_index_trained: core.document_index_trained(),
            encrypted: self.encrypted,
            log_used_bytes: health.used_bytes as u64,
            log_capacity_bytes: health.capacity_bytes as u64,
            disk_bytes: disk.0,
            disk_allocated_bytes: disk.1,
        })
    }

    fn snapshot(&self, destination: &str) -> Result<SnapshotInfo, HmsError> {
        if destination.is_empty() {
            return Err(HmsError::invalid("destination must not be empty"));
        }
        let core = self.core()?;
        let destination = Path::new(destination);
        if destination.starts_with(&self.path) {
            return Err(HmsError::invalid(
                "the snapshot cannot live inside the store",
            ));
        }
        // Writers wait here; searches keep running against the engine's own snapshot.
        let _exclusive = self.gate.write();
        guarded(|| {
            core.compact()?;
            core.flush()?;
            Ok(())
        })?;
        let written = snapshot::write(
            &self.path,
            destination,
            self.dimensions,
            core.document_count() as u64,
        )?;
        Ok(SnapshotInfo {
            path: destination.display().to_string(),
            file_count: written.file_count,
            total_bytes: written.total_bytes,
            document_count: written.document_count,
        })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl HmsStore {
    /// Opens or creates the store in the directory `path`. Replaying a large store takes
    /// time, so prefer this async form on a UI thread.
    #[uniffi::constructor(default(options = None))]
    pub async fn open(path: String, options: Option<StoreOptions>) -> Result<Arc<Self>, HmsError> {
        offload(move || Self::open_blocking(path, options)).await
    }

    /// Blocking form of `open`.
    #[uniffi::constructor(default(options = None))]
    pub fn open_blocking(
        path: String,
        options: Option<StoreOptions>,
    ) -> Result<Arc<Self>, HmsError> {
        let inner = Inner::open(&path, options.unwrap_or_default())?;
        Ok(Arc::new(Self { inner }))
    }

    /// Restores the snapshot in `snapshot_path` into the new directory `path` after verifying
    /// every file against the snapshot manifest, then opens it. `options` must match those the
    /// snapshotted store was created with (dimensions, encryption key, embedding space).
    #[uniffi::constructor(default(options = None))]
    pub async fn restore(
        snapshot_path: String,
        path: String,
        options: Option<StoreOptions>,
    ) -> Result<Arc<Self>, HmsError> {
        offload(move || Self::restore_blocking(snapshot_path, path, options)).await
    }

    /// Blocking form of `restore`.
    #[uniffi::constructor(default(options = None))]
    pub fn restore_blocking(
        snapshot_path: String,
        path: String,
        options: Option<StoreOptions>,
    ) -> Result<Arc<Self>, HmsError> {
        if path.is_empty() || snapshot_path.is_empty() {
            return Err(HmsError::invalid("paths must not be empty"));
        }
        snapshot::restore(Path::new(&snapshot_path), Path::new(&path))?;
        match Inner::open(&path, options.unwrap_or_default()) {
            Ok(inner) => Ok(Arc::new(Self { inner })),
            Err(error) => {
                // A restore that cannot be opened must not leave a half-trusted store behind.
                let _ = std::fs::remove_dir_all(&path);
                Err(error)
            }
        }
    }

    /// Stores a document, replacing any document with the same id. Returns its chunk count.
    pub async fn ingest(self: Arc<Self>, input: DocumentInput) -> Result<u32, HmsError> {
        offload(move || self.inner.ingest(input)).await
    }

    /// Blocking form of `ingest`.
    pub fn ingest_blocking(&self, input: DocumentInput) -> Result<u32, HmsError> {
        self.inner.ingest(input)
    }

    /// Stores documents as one commit: all or none. Returns each document's chunk count.
    pub async fn ingest_batch(
        self: Arc<Self>,
        inputs: Vec<DocumentInput>,
    ) -> Result<Vec<u32>, HmsError> {
        offload(move || self.inner.ingest_batch(inputs)).await
    }

    /// Blocking form of `ingest_batch`.
    pub fn ingest_batch_blocking(&self, inputs: Vec<DocumentInput>) -> Result<Vec<u32>, HmsError> {
        self.inner.ingest_batch(inputs)
    }

    /// Ranks chunks for `query`, best first.
    #[uniffi::method(default(options = None))]
    pub async fn search(
        self: Arc<Self>,
        query: String,
        options: Option<SearchOptions>,
    ) -> Result<Vec<SearchHit>, HmsError> {
        offload(move || self.inner.search(&query, options)).await
    }

    /// Blocking form of `search`.
    #[uniffi::method(default(options = None))]
    pub fn search_blocking(
        &self,
        query: String,
        options: Option<SearchOptions>,
    ) -> Result<Vec<SearchHit>, HmsError> {
        self.inner.search(&query, options)
    }

    /// Removes a document. Returns whether it existed.
    pub async fn delete_document(self: Arc<Self>, id: String) -> Result<bool, HmsError> {
        offload(move || self.inner.delete_document(&id)).await
    }

    /// Blocking form of `delete_document`.
    pub fn delete_document_blocking(&self, id: String) -> Result<bool, HmsError> {
        self.inner.delete_document(&id)
    }

    /// Whether a document with this id is stored.
    pub fn has_document(&self, id: String) -> Result<bool, HmsError> {
        Ok(self.inner.core()?.has_document(&id))
    }

    /// Counts and sizes of the open store.
    pub fn stats(&self) -> Result<StoreStats, HmsError> {
        self.inner.stats()
    }

    /// Rewrites the journal to its live content. Optional: the store stays correct without it.
    pub async fn compact(self: Arc<Self>) -> Result<(), HmsError> {
        offload(move || self.inner.compact()).await
    }

    /// Blocking form of `compact`.
    pub fn compact_blocking(&self) -> Result<(), HmsError> {
        self.inner.compact()
    }

    /// Trains the dense index over caller embeddings, when the store has enough of them.
    pub async fn train_index(self: Arc<Self>) -> Result<(), HmsError> {
        offload(move || self.inner.train_index()).await
    }

    /// Blocking form of `train_index`.
    pub fn train_index_blocking(&self) -> Result<(), HmsError> {
        self.inner.train_index()
    }

    /// Writes a verified copy of the store into the new directory `destination`. Searches
    /// continue during the copy; ingests and deletes wait for it.
    pub async fn snapshot(self: Arc<Self>, destination: String) -> Result<SnapshotInfo, HmsError> {
        offload(move || self.inner.snapshot(&destination)).await
    }

    /// Blocking form of `snapshot`.
    pub fn snapshot_blocking(&self, destination: String) -> Result<SnapshotInfo, HmsError> {
        self.inner.snapshot(&destination)
    }

    /// Releases the store directory now, so it can be opened again. Later calls on this handle
    /// fail with `Closed`. Dropping the last reference does the same; call this where the
    /// handle's lifetime is not deterministic (Swift) or the directory is reopened at once.
    pub fn shutdown(&self) {
        let _exclusive = self.inner.gate.write();
        self.inner.core.write().take();
    }
}

/// Runs blocking engine work off the caller's thread.
async fn offload<T, F>(work: F) -> Result<T, HmsError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, HmsError> + Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| HmsError::Internal {
            details: e.to_string(),
        })?
}

/// Converts an engine panic into `HmsError::Internal` instead of unwinding into foreign code.
fn guarded<T>(work: impl FnOnce() -> Result<T, HmsError>) -> Result<T, HmsError> {
    catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|payload| {
        let details = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "engine panic".to_owned());
        Err(HmsError::Internal { details })
    })
}

/// Logical and allocated bytes of every file under `path`.
fn directory_bytes(path: &Path) -> Result<(u64, u64), HmsError> {
    let (mut logical, mut allocated) = (0, 0);
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            let (l, a) = directory_bytes(&entry.path())?;
            logical += l;
            allocated += a;
        } else if kind.is_file() {
            let metadata = entry.metadata()?;
            logical += metadata.len();
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                allocated += metadata.blocks() * 512;
            }
            #[cfg(not(unix))]
            {
                allocated += metadata.len();
            }
        }
    }
    Ok((logical, allocated))
}
