// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared server state and the tenant store lifecycle: open once, serve, compact, snapshot,
//! restore, close when idle.
//!
//! Each tenant's engine and memory API live behind an `RwLock<Option<Open>>`. Requests hold the read side
//! while they work. Snapshot, restore and idle eviction take the write side, which is how they
//! quiesce the store and release the engine's store lock without racing a request. `None` means
//! the store was closed; the next request finds the slot gone and opens a fresh one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use axum::http::StatusCode;
use holographic_memory::core::memory_api::{Memory, MemoryConfig};
use holographic_memory::core::HmsConfig;
use holographic_memory::{EmbeddingSpace, HmsCore};
use parking_lot::lock_api::ArcRwLockReadGuard;
use parking_lot::{Condvar, Mutex, RawRwLock, RwLock};
use serde::Serialize;

use crate::auth::Authenticator;
use crate::config::{Config, MAX_TENANT_BYTES};
use crate::error::ApiError;
use crate::limits::{Idempotency, RateLimiter, WorkLimit};
use crate::metrics::{Metrics, Saturation, TenantGauge};
use crate::vindex::{sync_dir, VectorIndex};

/// The memory API configuration applied to every tenant; absent means the conservative defaults.
pub fn load_memory_config(path: Option<&Path>) -> Result<MemoryConfig> {
    let Some(path) = path else {
        return Ok(MemoryConfig::default());
    };
    let config: MemoryConfig = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?,
    )
    .with_context(|| format!("invalid memory config {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

pub fn valid_tenant(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_TENANT_BYTES
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Hex keeps the directory name safe and distinct on case-insensitive filesystems.
pub fn tenant_hex(name: &str) -> String {
    name.bytes().map(|b| format!("{b:02x}")).collect()
}

/// The tenant a store directory name encodes, if `name` is exactly `tenant_hex` of a valid one.
pub fn tenant_from_hex(name: &str) -> Option<String> {
    let tenant = String::from_utf8(crate::auth::unhex(name)?).ok()?;
    (valid_tenant(&tenant) && tenant_hex(&tenant) == name).then_some(tenant)
}

#[derive(Default, Clone, Serialize)]
pub struct CompactionStatus {
    pub running: bool,
    pub runs: u64,
    pub failures: u64,
    pub last_trigger: Option<&'static str>,
    pub last_started_at: Option<String>,
    pub last_duration_ms: Option<u64>,
    pub last_error: Option<&'static str>,
    #[serde(skip)]
    pub total_seconds: f64,
    #[serde(skip)]
    pub generation_at_last_run: Option<u64>,
}

pub struct Tenant {
    pub name: String,
    pub dir: PathBuf,
    core: Arc<RwLock<Option<Open>>>,
    last_used_ms: AtomicU64,
    /// Advanced by every committed change made through this server; feeds ETags and compaction.
    /// Drawn from `clock`, shared by every tenant this process opens, so a tenant reopened after
    /// eviction never repeats a generation an earlier open of it handed out.
    pub generation: AtomicU64,
    clock: Arc<AtomicU64>,
    /// Set while a snapshot or restore holds the store exclusively; requests are turned away
    /// at once instead of occupying work slots while they wait.
    quiesced: AtomicBool,
    pub compaction: Mutex<CompactionStatus>,
}

/// An open store: the engine, the memory API over it and the persisted vector index next to
/// it. Dropping it releases the store lock.
pub struct Open {
    core: Arc<HmsCore>,
    memory: Memory,
    /// Absent when its files could not be opened; the engine then answers every vector query.
    vectors: Option<Arc<VectorIndex>>,
}

impl Tenant {
    fn advance(&self) {
        let tick = self.clock.fetch_add(1, Ordering::AcqRel) + 1;
        self.generation.fetch_max(tick, Ordering::AcqRel);
    }
}

impl Open {
    /// Flushes the engine and marks the vector index cleanly closed, so the next open trusts
    /// its log without checking the engine.
    pub fn settle(&self) -> Result<()> {
        self.core.flush()?;
        if let Some(vectors) = &self.vectors {
            vectors.close()?;
        }
        Ok(())
    }
}

impl std::ops::Deref for Open {
    type Target = HmsCore;
    fn deref(&self) -> &HmsCore {
        &self.core
    }
}

/// Exclusive access to a tenant's store. While held, new requests for the tenant get 503
/// `tenant_busy` immediately; requests already running finish first.
pub struct Quiesced {
    pub tenant: Arc<Tenant>,
    guard: parking_lot::lock_api::ArcRwLockWriteGuard<RawRwLock, Option<Open>>,
}

impl Quiesced {
    pub fn open(&self) -> Option<&Open> {
        self.guard.as_ref()
    }

    /// Close the engine, releasing the store lock; `reopen` puts a new one in place.
    pub fn close(&mut self) {
        drop(self.guard.take());
    }

    pub fn reopen(&mut self, open: Open) {
        *self.guard = Some(open);
    }
}

impl Drop for Quiesced {
    fn drop(&mut self) {
        self.tenant.quiesced.store(false, Ordering::Release);
    }
}

pub struct CoreGuard {
    guard: ArcRwLockReadGuard<RawRwLock, Option<Open>>,
    pub tenant: Arc<Tenant>,
}

impl std::ops::Deref for CoreGuard {
    type Target = HmsCore;
    fn deref(&self) -> &HmsCore {
        self.open()
    }
}

impl CoreGuard {
    fn open(&self) -> &Open {
        self.guard
            .as_ref()
            .expect("checked when the guard was built")
    }

    pub fn memory(&self) -> &Memory {
        &self.open().memory
    }

    pub fn vectors(&self) -> Option<&VectorIndex> {
        self.open().vectors.as_deref()
    }

    /// Whether the persisted vector index holds a graph over every embedded chunk, so it
    /// answers the unfiltered graph queries.
    pub fn vectors_cover(&self) -> bool {
        self.vectors().is_some_and(|vectors| vectors.covers(self))
    }

    /// Maintains the persisted vector index, then the engine's snapshot; while the persisted
    /// graph covers the store the engine keeps only its exact scan. `force` rebuilds after any
    /// mutation; otherwise after the rebuild threshold.
    pub fn maintain_vector_indexes(&self, graph_min_rows: usize, force: bool) -> Result<()> {
        if let Some(vectors) = self.vectors() {
            vectors.maintain(graph_min_rows, force)?;
        }
        let engine_rows = if self.vectors_cover() {
            usize::MAX
        } else {
            graph_min_rows
        };
        if force {
            self.open().core.rebuild_vector_index(engine_rows)?;
        } else {
            self.open().core.maintain_vector_index(engine_rows)?;
        }
        Ok(())
    }

    pub fn bump(&self) {
        self.tenant.advance();
    }
}

enum TenantStatus {
    Opening,
    Ready(Arc<Tenant>),
    Failed(ApiError),
}

pub struct TenantSlot {
    status: Mutex<TenantStatus>,
    ready: Condvar,
}

impl TenantSlot {
    fn wait(&self) -> Result<Arc<Tenant>, ApiError> {
        let mut status = self.status.lock();
        loop {
            match &*status {
                TenantStatus::Opening => self.ready.wait(&mut status),
                TenantStatus::Ready(tenant) => return Ok(tenant.clone()),
                TenantStatus::Failed(error) => return Err(error.clone()),
            }
        }
    }
}

struct TenantInitialization<'a> {
    state: &'a AppState,
    name: &'a str,
    slot: Arc<TenantSlot>,
    completed: bool,
}

impl TenantInitialization<'_> {
    fn publish(&self, result: &Result<Arc<Tenant>, ApiError>) {
        *self.slot.status.lock() = match result {
            Ok(tenant) => TenantStatus::Ready(tenant.clone()),
            Err(error) => TenantStatus::Failed(error.clone()),
        };
        self.slot.ready.notify_all();
        if result.is_err() {
            self.state.forget(self.name, &self.slot);
        }
    }

    fn finish(mut self, result: Result<Arc<Tenant>, ApiError>) -> Result<Arc<Tenant>, ApiError> {
        self.publish(&result);
        self.completed = true;
        result
    }
}

impl Drop for TenantInitialization<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.publish(&Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "internal server error",
            )));
        }
    }
}

pub struct AppState {
    pub cfg: Config,
    pub auth: Authenticator,
    pub work: WorkLimit,
    pub metrics: Metrics,
    pub rate: Option<RateLimiter>,
    pub idem: Idempotency,
    pub ready: AtomicBool,
    pub started: Instant,
    /// Random per process; makes ETags from a previous run never match this one.
    pub epoch: u64,
    tenants: Mutex<HashMap<String, Arc<TenantSlot>>>,
    generations: Arc<AtomicU64>,
    compaction_lock: Mutex<()>,
    /// Set once at startup when TLS is on, so the reload endpoint can reach the certificate.
    pub tls: std::sync::OnceLock<Arc<crate::serve::Tls>>,
    #[cfg(feature = "local-models")]
    pub stages: Option<Arc<holographic_memory::core::models::ModelStages>>,
}

pub struct SnapshotInfo {
    pub id: String,
    pub bytes: u64,
    pub created_at: String,
}

impl AppState {
    pub fn new(
        cfg: Config,
        auth: Authenticator,
        routes: impl IntoIterator<Item = &'static str>,
        #[cfg(feature = "local-models")] stages: Option<
            Arc<holographic_memory::core::models::ModelStages>,
        >,
    ) -> Self {
        Self {
            work: WorkLimit::new(
                cfg.max_concurrent_work,
                cfg.max_queue,
                Duration::from_millis(cfg.queue_timeout_ms),
            ),
            metrics: Metrics::new(routes),
            rate: RateLimiter::new(cfg.rate_limit_rps, cfg.rate_limit_burst),
            idem: Idempotency::new(
                Duration::from_secs(cfg.idempotency_ttl_seconds),
                cfg.idempotency_capacity,
            ),
            ready: AtomicBool::new(false),
            started: Instant::now(),
            epoch: rand::random(),
            tenants: Mutex::new(HashMap::new()),
            generations: Arc::new(AtomicU64::new(0)),
            compaction_lock: Mutex::new(()),
            tls: std::sync::OnceLock::new(),
            cfg,
            auth,
            #[cfg(feature = "local-models")]
            stages,
        }
    }

    /// Whether this server embeds document and query text itself.
    pub fn server_embeds(&self) -> bool {
        #[cfg(feature = "local-models")]
        if let Some(s) = &self.stages {
            return s.embedder.is_some();
        }
        false
    }

    pub fn snapshot_root(&self, name: &str) -> PathBuf {
        self.cfg.data_dir.join("snapshots").join(tenant_hex(name))
    }

    pub fn staging_root(&self) -> PathBuf {
        self.cfg.data_dir.join("staging")
    }

    fn forget(&self, name: &str, slot: &Arc<TenantSlot>) {
        let mut tenants = self.tenants.lock();
        if tenants.get(name).is_some_and(|s| Arc::ptr_eq(s, slot)) {
            tenants.remove(name);
        }
    }

    fn slot(&self, name: &str) -> Result<Arc<Tenant>, ApiError> {
        let (slot, initialize) = {
            let mut tenants = self.tenants.lock();
            if let Some(slot) = tenants.get(name) {
                (slot.clone(), false)
            } else {
                if tenants.len() >= self.cfg.max_tenants {
                    return Err(ApiError::new(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "tenant_limit",
                        "too many tenants are open on this server",
                    ));
                }
                let slot = Arc::new(TenantSlot {
                    status: Mutex::new(TenantStatus::Opening),
                    ready: Condvar::new(),
                });
                tenants.insert(name.to_string(), slot.clone());
                (slot, true)
            }
        };
        if !initialize {
            return slot.wait();
        }
        let reservation = TenantInitialization {
            state: self,
            name,
            slot,
            completed: false,
        };
        reservation.finish(
            self.open_tenant(name)
                .map(Arc::new)
                .map_err(ApiError::internal),
        )
    }

    fn open_core(&self, dir: &Path) -> Result<Open> {
        let config = HmsConfig {
            embedding_space: Some(EmbeddingSpace {
                model: self.cfg.embedding_model.clone(),
                revision: self.cfg.embedding_revision.clone(),
                dimensions: self.cfg.input_dim,
                normalization: "l2".into(),
                metric: "cosine".into(),
            }),
            ..HmsConfig::default()
        };
        let core = HmsCore::new(self.cfg.dim, Some(dir.display().to_string()), Some(config))?;
        // One interner per open tenant would add up to tens of MB each; free it per batch.
        core.set_keep_ingest_interner(false);
        #[cfg(feature = "local-models")]
        core.set_model_stages(self.stages.clone())?;
        let core = Arc::new(core);
        let memory = Memory::with_config(
            core.clone(),
            load_memory_config(self.cfg.memory_config.as_deref())?,
        )?;
        let vectors = match VectorIndex::open(&dir.join("vindex"), self.cfg.input_dim, &core) {
            Ok(index) => Some(Arc::new(index)),
            Err(error) => {
                crate::log_event!(Warn, "vector_index_open_failed", "path" => dir.display().to_string(), "detail" => format!("{error:#}"));
                None
            }
        };
        Ok(Open {
            core,
            memory,
            vectors,
        })
    }

    fn open_tenant(&self, name: &str) -> Result<Tenant> {
        let tenants_root = self.cfg.data_dir.join("tenants");
        std::fs::create_dir_all(&tenants_root)?;
        // The hex name cannot contain a separator or `..`; the containment check on the
        // canonical root states that invariant where the path is built.
        let root = tenants_root.canonicalize()?;
        let dir = root.join(tenant_hex(name));
        anyhow::ensure!(
            dir.starts_with(&root) && dir.parent() == Some(root.as_path()),
            "tenant directory escapes the tenants root"
        );
        std::fs::create_dir_all(&dir)?;
        let core = self.open_core(&dir)?;
        let generation = self.generations.fetch_add(1, Ordering::AcqRel) + 1;
        Ok(Tenant {
            name: name.to_string(),
            dir,
            core: Arc::new(RwLock::new(Some(core))),
            last_used_ms: AtomicU64::new(self.now_ms()),
            generation: AtomicU64::new(generation),
            clock: self.generations.clone(),
            quiesced: AtomicBool::new(false),
            // Nothing changed since the open, so there is nothing to compact yet.
            compaction: Mutex::new(CompactionStatus {
                generation_at_last_run: Some(generation),
                ..CompactionStatus::default()
            }),
        })
    }

    fn now_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// The open engine for `name`, opening it if needed, held for reading until dropped.
    pub fn core(&self, name: &str) -> Result<CoreGuard, ApiError> {
        for _ in 0..16 {
            let tenant = self.slot(name)?;
            if tenant.quiesced.load(Ordering::Acquire) {
                return Err(ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "tenant_busy",
                    "the tenant's store is being snapshotted or restored; retry shortly",
                )
                .retry_after(1));
            }
            let guard = tenant.core.read_arc();
            if guard.is_some() {
                tenant.last_used_ms.store(self.now_ms(), Ordering::Relaxed);
                return Ok(CoreGuard { guard, tenant });
            }
            // Closed by eviction or a failed restore between lookup and lock.
            drop(guard);
            self.forget_if_closed(name, &tenant);
            std::thread::yield_now();
        }
        Err(ApiError::internal(
            "tenant store kept closing while opening",
        ))
    }

    /// Take the tenant's store exclusively, opening it first if needed.
    pub fn quiesce(&self, name: &str) -> Result<Quiesced, ApiError> {
        let tenant = self.core(name)?.tenant.clone();
        if tenant.quiesced.swap(true, Ordering::AcqRel) {
            return Err(ApiError::conflict(
                "another snapshot or restore of this tenant is running",
            ));
        }
        let guard = tenant.core.write_arc();
        let quiesced = Quiesced { tenant, guard };
        // Eviction can close the store between the lookup and the write lock; the slot then
        // belongs to whichever open comes next, and moving files under it would race that one.
        if quiesced.open().is_none() {
            return Err(ApiError::conflict("tenant store was closed; retry"));
        }
        Ok(quiesced)
    }

    fn forget_if_closed(&self, name: &str, tenant: &Arc<Tenant>) {
        let mut tenants = self.tenants.lock();
        if let Some(slot) = tenants.get(name) {
            if matches!(&*slot.status.lock(), TenantStatus::Ready(t) if Arc::ptr_eq(t, tenant)) {
                tenants.remove(name);
            }
        }
    }

    fn ready_tenants(&self) -> Vec<Arc<Tenant>> {
        self.tenants
            .lock()
            .values()
            .filter_map(|slot| match &*slot.status.lock() {
                TenantStatus::Ready(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn open_tenant_count(&self) -> usize {
        self.tenants.lock().len()
    }

    /// Flush and close tenants idle longer than `--tenant-idle-seconds`.
    pub fn evict_idle(&self) -> usize {
        let idle = self.cfg.tenant_idle_seconds.saturating_mul(1000);
        if idle == 0 {
            return 0;
        }
        let now = self.now_ms();
        let mut closed = 0;
        for tenant in self.ready_tenants() {
            if now.saturating_sub(tenant.last_used_ms.load(Ordering::Relaxed)) < idle
                || tenant.quiesced.load(Ordering::Acquire)
            {
                continue;
            }
            // A request or maintenance task holding the read side means it is not idle.
            let Some(mut write) = tenant.core.try_write_arc() else {
                continue;
            };
            // A snapshot or restore waiting for the write side owns the store next.
            if now.saturating_sub(tenant.last_used_ms.load(Ordering::Relaxed)) < idle
                || tenant.quiesced.load(Ordering::Acquire)
            {
                continue;
            }
            if let Some(core) = write.as_ref() {
                if let Err(error) = core.settle() {
                    crate::log_event!(Error, "evict_flush_failed", "tenant" => tenant.name, "detail" => format!("{error:#}"));
                    continue;
                }
            }
            // Dropping the engine releases the store lock; only then may the slot disappear.
            drop(write.take());
            let mut tenants = self.tenants.lock();
            if let Some(slot) = tenants.get(&tenant.name) {
                if matches!(&*slot.status.lock(), TenantStatus::Ready(t) if Arc::ptr_eq(t, &tenant))
                {
                    tenants.remove(&tenant.name);
                }
            }
            drop(tenants);
            drop(write);
            self.metrics
                .tenant_evictions
                .fetch_add(1, Ordering::Relaxed);
            crate::log_event!(Info, "tenant_evicted", "tenant" => tenant.name);
            closed += 1;
        }
        closed
    }

    /// Flush every open store; used at shutdown after work has drained.
    pub fn flush_all(&self) -> Result<()> {
        let tenants: Vec<_> = self.tenants.lock().drain().collect();
        let mut failed = false;
        for (name, slot) in tenants {
            // The map is already drained: stopping here would leave every later tenant unflushed.
            let tenant = match &*slot.status.lock() {
                TenantStatus::Ready(tenant) => tenant.clone(),
                TenantStatus::Opening => {
                    crate::log_event!(Error, "shutdown_flush_failed", "tenant" => name, "detail" => "initialization incomplete at shutdown");
                    failed = true;
                    continue;
                }
                TenantStatus::Failed(_) => continue,
            };
            let guard = tenant.core.write();
            if let Some(core) = guard.as_ref() {
                if let Err(error) = core.settle() {
                    crate::log_event!(Error, "shutdown_flush_failed", "tenant" => name, "detail" => format!("{error:#}"));
                    failed = true;
                }
            }
        }
        anyhow::ensure!(!failed, "shutdown flush failed");
        Ok(())
    }

    pub fn saturation(&self) -> Saturation {
        let mut tenants: Vec<TenantGauge> = self
            .ready_tenants()
            .into_iter()
            .filter_map(|t| {
                // Never block a scrape behind a snapshot or restore.
                let guard = t.core.try_read()?;
                let core = guard.as_ref()?;
                let c = t.compaction.lock().clone();
                Some(TenantGauge {
                    name: t.name.clone(),
                    documents: core.document_count() as u64,
                    vectors: core.document_embedding_count() as u64,
                    used_bytes: core.storage_health().used_bytes,
                    index_trained: core.document_index_trained(),
                    compaction_runs: c.runs,
                    compaction_failures: c.failures,
                    compaction_seconds: c.total_seconds,
                })
            })
            .collect();
        tenants.sort_by(|a, b| a.name.cmp(&b.name));
        Saturation {
            work_in_use: self.work.in_use(),
            work_capacity: self.work.capacity,
            queue_depth: self.work.queue_depth(),
            tenants,
            idempotency_records: self.idem.len(),
            ready: self.ready.load(Ordering::Acquire),
        }
    }

    /// Compact one tenant's store and record the outcome. Compactions run one at a time.
    pub fn compact(&self, name: &str, trigger: &'static str) -> Result<CompactionStatus, ApiError> {
        // Queue for the serial slot before holding the store: a compaction waiting its turn must
        // not pin the read side, which would keep a snapshot of this tenant (and with it every
        // request, turned away as busy) waiting on another tenant's compaction.
        let _serial = self.compaction_lock.lock();
        let guard = self.core(name)?;
        let generation = guard.tenant.generation.load(Ordering::Acquire);
        {
            let mut status = guard.tenant.compaction.lock();
            status.running = true;
            status.last_trigger = Some(trigger);
            status.last_started_at = Some(crate::log::rfc3339(SystemTime::now()));
        }
        let started = Instant::now();
        let result = guard.compact();
        // Scheduled maintenance of the vector indexes rides on compaction: a rebuild happens
        // only past the mutation threshold (or with no snapshot yet), and its duration counts
        // into this compaction's.
        if result.is_ok() {
            if let Err(error) = guard.maintain_vector_indexes(self.cfg.graph_min_rows, false) {
                crate::log_event!(Warn, "vector_index_rebuild_failed", "tenant" => name, "detail" => format!("{error:#}"));
            }
        }
        let mut status = guard.tenant.compaction.lock();
        status.running = false;
        status.last_duration_ms =
            Some(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        status.total_seconds += started.elapsed().as_secs_f64();
        match result {
            Ok(()) => {
                status.runs += 1;
                status.last_error = None;
                status.generation_at_last_run = Some(generation);
                crate::log_event!(Info, "compaction_done", "tenant" => name, "trigger" => trigger, "ms" => status.last_duration_ms);
                Ok(status.clone())
            }
            Err(error) => {
                status.failures += 1;
                status.last_error = Some("compaction failed; see the server log");
                drop(status);
                Err(ApiError::internal(error))
            }
        }
    }

    /// One scheduled pass: compact every open tenant that changed since its last compaction.
    pub fn compact_changed(&self) {
        for tenant in self.ready_tenants() {
            let generation = tenant.generation.load(Ordering::Acquire);
            let (running, last) = {
                let s = tenant.compaction.lock();
                (s.running, s.generation_at_last_run)
            };
            if running || last == Some(generation) {
                continue;
            }
            if let Err(error) = self.compact(&tenant.name, "scheduled") {
                crate::log_event!(Warn, "compaction_failed", "tenant" => tenant.name, "code" => error.code);
            }
        }
    }

    pub fn compaction_status(&self, name: &str) -> Result<CompactionStatus, ApiError> {
        let guard = self.core(name)?;
        let status = guard.tenant.compaction.lock().clone();
        Ok(status)
    }

    pub fn create_snapshot(&self, name: &str) -> Result<SnapshotInfo, ApiError> {
        let quiesced = self.quiesce(name)?;
        let tenant = quiesced.tenant.clone();
        let id = format!(
            "{}-{:04x}",
            crate::log::rfc3339(SystemTime::now())
                .replace(['-', ':'], "")
                .replace('.', "")
                .to_lowercase(),
            rand::random::<u16>()
        );
        let root = self.snapshot_root(name);
        let target = root.join(&id);
        // Requests for this tenant are turned away while the store is copied, so the copy is
        // one consistent state.
        let Some(core) = quiesced.open() else {
            return Err(ApiError::conflict("tenant store was closed; retry"));
        };
        let result = (|| -> Result<u64> {
            core.settle()?;
            create_dir_durably(&root)?;
            let partial = root.join(format!(".partial-{id}"));
            let bytes = copy_store(&tenant.dir, &partial)?;
            std::fs::rename(&partial, &target)?;
            if let Err(error) = sync_dir(&root) {
                // Not reported as created, so not left behind as one either.
                let _ = std::fs::remove_dir_all(&target);
                return Err(error.into());
            }
            Ok(bytes)
        })();
        drop(quiesced);
        match result {
            Ok(bytes) => {
                crate::log_event!(Info, "snapshot_created", "tenant" => name, "snapshot" => id, "bytes" => bytes);
                Ok(SnapshotInfo {
                    id,
                    bytes,
                    created_at: crate::log::rfc3339(SystemTime::now()),
                })
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(root.join(format!(".partial-{id}")));
                Err(ApiError::internal(error))
            }
        }
    }

    pub fn list_snapshots(&self, name: &str) -> Result<Vec<SnapshotInfo>, ApiError> {
        let root = self.snapshot_root(name);
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&root) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(ApiError::internal(e)),
        };
        for entry in entries.flatten() {
            let id = entry.file_name().to_string_lossy().into_owned();
            if id.starts_with('.') || !entry.path().is_dir() {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            out.push(SnapshotInfo {
                bytes: dir_size(&entry.path()),
                created_at: crate::log::rfc3339(modified),
                id,
            });
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    pub fn snapshot_path(&self, name: &str, id: &str) -> Result<PathBuf, ApiError> {
        if !valid_snapshot_id(id) {
            return Err(ApiError::bad("invalid snapshot id"));
        }
        let path = self.snapshot_root(name).join(id);
        if path.is_dir() {
            Ok(path)
        } else {
            Err(ApiError::not_found("no such snapshot"))
        }
    }

    pub fn delete_snapshot(&self, name: &str, id: &str) -> Result<(), ApiError> {
        let path = self.snapshot_path(name, id)?;
        std::fs::remove_dir_all(path).map_err(ApiError::internal)
    }

    /// Replace the tenant's store with the verified store in `staged` (a directory under
    /// `staging_root`). The previous store is kept as snapshot `pre-restore-<time>`. On any
    /// failure the previous store is back in place and serving.
    pub fn restore_staged(&self, name: &str, staged: &Path) -> Result<String, ApiError> {
        let mut write = self.quiesce(name)?;
        let tenant = write.tenant.clone();
        let outcome = (|| -> Result<String> {
            if let Some(core) = write.open() {
                core.settle()?;
            }
            write.close();
            let backup_id = format!(
                "pre-restore-{}",
                crate::log::rfc3339(SystemTime::now())
                    .replace(['-', ':', '.'], "")
                    .to_lowercase()
            );
            let root = self.snapshot_root(name);
            create_dir_durably(&root)?;
            let backup = root.join(&backup_id);
            let tenants_root = parent(&tenant.dir)?;
            let moved = (|| -> Result<()> {
                std::fs::rename(&tenant.dir, &backup)
                    .context("cannot set the current store aside")?;
                sync_dir(&root)?;
                sync_dir(tenants_root)?;
                std::fs::rename(staged, &tenant.dir)
                    .context("cannot move the restored store into place")?;
                sync_dir(tenants_root)?;
                sync_dir(parent(staged)?)?;
                Ok(())
            })();
            let reopened = moved.and_then(|()| self.open_core(&tenant.dir));
            match reopened {
                Ok(core) => {
                    write.reopen(core);
                    Ok(backup_id)
                }
                Err(error) => {
                    // Undo: discard whatever now sits at the tenant path and bring the old store back.
                    if backup.is_dir() {
                        let _ = std::fs::remove_dir_all(&tenant.dir);
                        std::fs::rename(&backup, &tenant.dir).context("rollback failed")?;
                        sync_dir(tenants_root).context("rollback failed")?;
                        sync_dir(&root).context("rollback failed")?;
                    }
                    write.reopen(
                        self.open_core(&tenant.dir)
                            .context("rollback reopen failed")?,
                    );
                    Err(error)
                }
            }
        })();
        drop(write);
        match outcome {
            Ok(backup) => {
                tenant.advance();
                *tenant.compaction.lock() = CompactionStatus::default();
                crate::log_event!(Info, "snapshot_restored", "tenant" => name, "previous_store" => backup);
                Ok(backup)
            }
            Err(error) => Err(ApiError::internal(error)),
        }
    }
}

pub fn valid_snapshot_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && !id.starts_with('.')
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

fn parent(path: &Path) -> Result<&Path> {
    path.parent()
        .with_context(|| format!("{} has no parent directory", path.display()))
}

/// `create_dir_all`, with every directory it created synced into its parent.
fn create_dir_durably(dir: &Path) -> Result<()> {
    let existing = dir.ancestors().find(|a| a.is_dir()).map(Path::to_path_buf);
    if existing.as_deref() == Some(dir) {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    for created in dir.ancestors().skip(1) {
        if created.as_os_str().is_empty() {
            break;
        }
        sync_dir(created)?;
        if Some(created) == existing.as_deref() {
            break;
        }
    }
    Ok(())
}

fn skip_entry(name: &str) -> bool {
    name == ".hms.lock" || name.starts_with(".compact-")
}

/// Copy a store directory durably, leaving out the process lock and compaction scratch.
pub fn copy_store(src: &Path, dst: &Path) -> Result<u64> {
    std::fs::create_dir_all(dst)?;
    let mut bytes = 0;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if skip_entry(&name.to_string_lossy()) {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        let kind = entry.file_type()?;
        if kind.is_dir() {
            bytes += copy_store(&from, &to)?;
        } else if kind.is_file() {
            bytes += std::fs::copy(&from, &to)?;
            std::fs::File::open(&to)?.sync_all()?;
        }
    }
    sync_dir(dst)?;
    Ok(bytes)
}

pub fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn dirty(state: &AppState, name: &str) -> bool {
        state
            .cfg
            .data_dir
            .join("tenants")
            .join(tenant_hex(name))
            .join("vindex")
            .join("dirty")
            .exists()
    }

    /// An ETag issued before an idle eviction never matches the reopened store's content.
    #[test]
    fn generation_keeps_rising_across_eviction_and_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::tests::state_with(directory.path(), 2, &["--tenant-idle-seconds", "1"]);
        let core = state.core("a").unwrap();
        core.bump();
        let before = core.tenant.generation.load(Ordering::Acquire);
        drop(core);
        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(state.evict_idle(), 1);
        let reopened = state.core("a").unwrap();
        assert!(reopened.tenant.generation.load(Ordering::Acquire) > before);
        let opened = reopened.tenant.generation.load(Ordering::Acquire);
        drop(reopened);
        state.compact_changed();
        assert_eq!(
            state.compaction_status("a").unwrap().runs,
            0,
            "an unchanged reopened store is not compacted"
        );
        assert_eq!(
            state
                .core("a")
                .unwrap()
                .tenant
                .generation
                .load(Ordering::Acquire),
            opened
        );
    }

    /// Eviction leaves a tenant alone while a snapshot or restore is waiting to take it.
    #[test]
    fn eviction_skips_a_quiesced_tenant() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::tests::state_with(directory.path(), 2, &["--tenant-idle-seconds", "1"]);
        let tenant = state.core("a").unwrap().tenant.clone();
        tenant.quiesced.store(true, Ordering::Release);
        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(state.evict_idle(), 0);
        assert!(tenant.core.read().is_some());
        tenant.quiesced.store(false, Ordering::Release);
        assert_eq!(state.evict_idle(), 1);
    }

    /// Shutdown settles every ready tenant even when another one never finished opening.
    #[test]
    fn flush_all_settles_every_tenant_past_an_incomplete_one() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::tests::state_with(directory.path(), 2, &[]);
        let names: Vec<String> = (0..16).map(|i| format!("t{i}")).collect();
        for name in &names {
            let core = state.core(name).unwrap();
            let row = crate::vindex::normalize(&[1.0, 0.0]).unwrap();
            core.vectors()
                .unwrap()
                .stage(&[("d".into(), 1, row)])
                .unwrap();
            assert!(dirty(&state, name));
        }
        state.tenants.lock().insert(
            "opening".into(),
            Arc::new(TenantSlot {
                status: Mutex::new(TenantStatus::Opening),
                ready: Condvar::new(),
            }),
        );
        assert!(
            state.flush_all().is_err(),
            "the incomplete tenant is reported"
        );
        for name in &names {
            assert!(!dirty(&state, name), "{name} was not settled");
        }
    }

    /// A compaction queued behind another tenant's does not hold its own tenant's store, so a
    /// snapshot of that tenant proceeds.
    #[test]
    fn queued_compaction_does_not_hold_the_tenant() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::tests::state_with(directory.path(), 2, &[]);
        drop(state.core("a").unwrap());
        let serial = state.compaction_lock.lock();
        let compacting = {
            let state = state.clone();
            std::thread::spawn(move || state.compact("a", "manual").map(|_| ()))
        };
        std::thread::sleep(Duration::from_millis(200));
        let (sent, received) = mpsc::channel();
        {
            let state = state.clone();
            std::thread::spawn(move || {
                let quiesced = state.quiesce("a").map(drop);
                sent.send(quiesced.is_ok()).unwrap();
            });
        }
        let quiesced = received.recv_timeout(Duration::from_secs(5));
        drop(serial);
        compacting.join().unwrap().unwrap();
        assert_eq!(quiesced, Ok(true), "snapshot waited on another compaction");
    }
}
