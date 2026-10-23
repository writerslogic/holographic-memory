// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Durable single-writer vector updates; binary contracts are in
//! `docs/file-format/VECTOR-UPDATES.md`.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::core::sync::RwLock;

use super::{
    kernels::dot, vertex::SearchScratch, BuildParams, Graph, Metric, PreparedRows, Residency,
    VGraph, VSearchParams,
};
use crate::core::durable_file::{atomic_write, sync_directory, sync_mapped_file};
use crate::core::error::HmsError;

mod delta;
mod diagnostics;
mod format;
use delta::Delta;
use format::{BaseRows, BaseRowsWriter};

const MAX_DIM: usize = 1 << 20;
const MAX_PAYLOAD: usize = 64 << 20;
const MAX_OPS: usize = 65536;
/// Row counts below this fit 3-byte neighbour ids (`0xFF_FFFF` is the padding id).
const THREE_BYTE_IDS_BELOW: usize = (1 << 24) - 1;

/// Failures at the update and persistence boundary.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    #[error("unknown vector id {0}")]
    UnknownId(u32),
    #[error("vector id {0} already exists")]
    AlreadyExists(u32),
    #[error("id {0} occurs twice in one batch")]
    DuplicateIdInBatch(u32),
    #[error("expected dimension {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("invalid vector: {0}")]
    InvalidVector(&'static str),
    #[error("id u32::MAX is reserved")]
    IdOutOfRange,
    #[error("vector index allocation or batch limit exceeded")]
    Capacity,
    #[error("corrupt vector log at byte {offset}: {reason}")]
    CorruptLog { offset: u64, reason: String },
    #[error("corrupt vector base: {0}")]
    CorruptBase(String),
    #[error("unsupported vector format: {0}")]
    UnsupportedFormat(&'static str),
    #[error("another writer is active")]
    WriterBusy,
    #[error("index is already open by a writer")]
    Locked,
    #[error("write outcome uncertain; reopen the index before further writes")]
    RecoveryRequired,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Persistence(#[from] anyhow::Error),
    #[error(transparent)]
    Search(#[from] HmsError),
}

/// Advisory thresholds; callers schedule consolidation outside their commit latency.
#[derive(Clone, Copy, Debug)]
pub struct ConsolidationPolicy {
    pub delta_ratio: f64,
    pub tombstone_ratio: f64,
    pub min_delta: usize,
}
impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self {
            delta_ratio: 0.1,
            tombstone_ratio: 0.2,
            min_delta: 1024,
        }
    }
}

#[derive(Clone)]
pub(super) struct Encoding {
    pub(super) bits: u8,
    pub(super) residual_bits: u8,
    pub(super) align: bool,
    pub(super) reorder: bool,
    pub(super) id_bytes: u8,
    pub(super) screen: bool,
    pub(super) block_bits: u8,
}
impl Default for Encoding {
    fn default() -> Self {
        Self {
            bits: 8,
            residual_bits: 8,
            align: true,
            reorder: false,
            id_bytes: 4,
            screen: false,
            block_bits: 0,
        }
    }
}
#[derive(Clone)]
struct Manifest {
    identity: [u8; 16],
    dim: usize,
    sequence: u64,
    build: BuildParams,
    encoding: Encoding,
}
impl Manifest {
    fn vertex_path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("base-{}.vertex", self.sequence))
    }
    fn rows_path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("rows-{}.bin", self.sequence))
    }
}

#[derive(Clone)]
enum Operation {
    Insert(u32, Vec<f32>),
    Delete(u32),
    Replace(u32, Vec<f32>),
}
impl Operation {
    fn id(&self) -> u32 {
        match self {
            Self::Insert(id, _) | Self::Delete(id) | Self::Replace(id, _) => *id,
        }
    }
    fn tag(&self) -> u8 {
        match self {
            Self::Insert(..) => 1,
            Self::Delete(..) => 2,
            Self::Replace(..) => 3,
        }
    }
    fn vector(&self) -> Option<&[f32]> {
        match self {
            Self::Insert(_, v) | Self::Replace(_, v) => Some(v),
            Self::Delete(_) => None,
        }
    }
}
struct Base {
    graph: Option<VGraph>,
    rows: BaseRows,
    dense_ids: bool,
}
impl Base {
    fn open(dir: &Path, m: &Manifest) -> Result<Self, UpdateError> {
        let rows = BaseRows::open(&m.rows_path(dir), m)?;
        let graph = if rows.ids.is_empty() {
            None
        } else {
            let graph = VGraph::open(&m.vertex_path(dir), Residency::Mapped)?;
            if graph.len() != rows.ids.len()
                || graph.dim() != m.dim
                || graph.metric() != m.build.metric.into()
            {
                return Err(UpdateError::CorruptBase(
                    "vertex/rows shape mismatch".into(),
                ));
            }
            Some(graph)
        };
        let dense_ids = rows.ids.iter().enumerate().all(|(i, &id)| i == id as usize);
        Ok(Self {
            graph,
            rows,
            dense_ids,
        })
    }
}
#[derive(Clone)]
struct State {
    base: Arc<Base>,
    tombstones: Vec<u64>,
    deleted: usize,
    delta: Delta,
    pending: Vec<Operation>,
    touched: HashSet<u32>,
    pending_bytes: usize,
    manifest: Manifest,
    sequence: u64,
    log_end: u64,
}
impl State {
    fn new(base: Base, manifest: Manifest) -> Self {
        Self {
            tombstones: vec![0; base.rows.ids.len().div_ceil(64)],
            base: Arc::new(base),
            deleted: 0,
            delta: Delta::default(),
            pending: Vec::new(),
            touched: HashSet::new(),
            pending_bytes: 0,
            sequence: manifest.sequence,
            manifest,
            log_end: format::LOG_HEADER,
        }
    }
    fn dead(&self, slot: usize) -> bool {
        self.tombstones[slot / 64] & (1 << (slot % 64)) != 0
    }
    fn base_slot(&self, id: u32) -> Option<usize> {
        self.base
            .rows
            .ids
            .binary_search(&id)
            .ok()
            .filter(|&s| !self.dead(s))
    }
    fn contains(&self, id: u32) -> bool {
        self.delta.live.contains_key(&id) || self.base_slot(id).is_some()
    }
    fn len(&self) -> usize {
        self.base.rows.ids.len() - self.deleted + self.delta.live.len()
    }
    fn validate_op(&self, op: &Operation) -> Result<(), UpdateError> {
        if op.id() == u32::MAX {
            return Err(UpdateError::IdOutOfRange);
        }
        if let Some(v) = op.vector() {
            validate_vector(v, self.manifest.dim, self.manifest.build.metric)?;
        }
        match op {
            Operation::Insert(id, _) if self.contains(*id) => Err(UpdateError::AlreadyExists(*id)),
            Operation::Delete(id) | Operation::Replace(id, _) if !self.contains(*id) => {
                Err(UpdateError::UnknownId(*id))
            }
            _ => Ok(()),
        }
    }
    fn apply(&mut self, op: &Operation) {
        let id = op.id();
        if !matches!(op, Operation::Insert(..)) {
            self.delta.remove(id);
            if let Some(slot) = self.base_slot(id) {
                self.tombstones[slot / 64] |= 1 << (slot % 64);
                self.deleted += 1;
            }
        }
        if let Some(row) = op.vector() {
            let mut row = row.to_vec();
            super::prepare_rows(&mut row, self.manifest.dim, self.manifest.build.metric);
            self.delta.insert(id, &row, &self.manifest.build);
        }
    }
    fn replay(&mut self, ops: Vec<Operation>) -> Result<(), UpdateError> {
        for op in &ops {
            self.validate_op(op)?;
        }
        for op in &ops {
            self.apply(op);
        }
        Ok(())
    }
}

thread_local! { static SCRATCH: RefCell<Option<SearchScratch>> = const { RefCell::new(None) }; }

/// A mapped base plus a mutable graph delta, with atomic durable batches.
/// `insert`, `delete`, and `replace` publish immediately; `commit` makes the batch durable.
/// Only one process may open a directory, while any number of threads may search it.
pub struct UpdatableVGraph {
    dir: PathBuf,
    _lock: IndexLock,
    writer: AtomicU8,
    state: RwLock<State>,
    diagnostics: diagnostics::Diagnostics,
}
struct Writer<'a>(&'a AtomicU8);
impl Writer<'_> {
    fn uncertain(&self) {
        self.0.store(2, Ordering::Release);
    }
}
impl Drop for Writer<'_> {
    fn drop(&mut self) {
        let _ = self
            .0
            .compare_exchange(1, 0, Ordering::Release, Ordering::Relaxed);
    }
}
struct IndexLock(File);
impl Drop for IndexLock {
    fn drop(&mut self) {
        // Explicit unlock also releases descriptors inherited by a concurrent fork.
        if let Err(error) = self.0.unlock() {
            tracing::warn!(%error, "failed to release vector index lock");
        }
    }
}
fn lock(dir: &Path) -> Result<IndexLock, UpdateError> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join(".lock"))?;
    file.try_lock().map_err(|e| match e {
        TryLockError::WouldBlock => UpdateError::Locked,
        TryLockError::Error(e) => UpdateError::Io(e),
    })?;
    Ok(IndexLock(file))
}
fn validate_vector(row: &[f32], dim: usize, metric: Metric) -> Result<(), UpdateError> {
    if row.len() != dim {
        return Err(UpdateError::DimensionMismatch {
            expected: dim,
            actual: row.len(),
        });
    }
    if row.iter().any(|v| !v.is_finite()) {
        return Err(UpdateError::InvalidVector("non-finite coordinate"));
    }
    if !dot(row, row).is_finite() {
        return Err(UpdateError::InvalidVector("squared norm overflows f32"));
    }
    if metric == Metric::Angular && row.iter().all(|&v| v == 0.0) {
        return Err(UpdateError::InvalidVector("zero angular vector"));
    }
    Ok(())
}
fn distance(q: &[f32], row: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Angular => -dot(q, row),
        Metric::Euclidean => dot(row, row) * 0.5 - dot(q, row),
    }
}

fn widened_width(requested: usize, allocated: usize, live: usize) -> usize {
    ((requested.min(allocated) as u128 * allocated as u128).div_ceil(live as u128))
        .min(allocated as u128) as usize
}

impl UpdatableVGraph {
    /// Creates a new directory from row-major vectors with ids `0..n`.
    pub fn create(
        path: &Path,
        data: &[f32],
        dim: usize,
        params: &BuildParams,
    ) -> Result<Self, UpdateError> {
        Self::create_with_encoding(path, data, dim, params, Encoding::default())
    }
    fn create_with_encoding(
        path: &Path,
        data: &[f32],
        dim: usize,
        params: &BuildParams,
        encoding: Encoding,
    ) -> Result<Self, UpdateError> {
        if dim == 0
            || dim > MAX_DIM
            || !data.len().is_multiple_of(dim)
            || data.len() / dim >= u32::MAX as usize
        {
            return Err(UpdateError::Capacity);
        }
        let manifest = Manifest {
            dim,
            identity: rand::random(),
            sequence: 0,
            build: params.clone(),
            encoding,
        };
        Manifest::decode(&manifest.encode())?;
        for row in data.chunks_exact(dim) {
            validate_vector(row, dim, params.metric)?;
        }
        std::fs::create_dir(path)?;
        let file = lock(path)?;
        let mut vectors = data.to_vec();
        super::prepare_rows(&mut vectors, dim, params.metric);
        let ids: Vec<u32> = (0..(data.len() / dim) as u32).collect();
        write_base(path, &manifest, &ids, vectors)?;
        format::rotate(&path.join("updates.log"), &manifest)?;
        atomic_write(&path.join("CURRENT"), &manifest.encode())?;
        sync_directory(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new(".")),
        )?;
        let base = Base::open(path, &manifest)?;
        Ok(Self {
            dir: path.to_path_buf(),
            _lock: file,
            writer: AtomicU8::new(0),
            state: RwLock::new(State::new(base, manifest)),
            diagnostics: diagnostics::Diagnostics::from_env(),
        })
    }
    /// Saves a graph and its matching vertex encoding without rebuilding the graph.
    pub fn create_from_graph(
        path: &Path,
        graph: &Graph,
        vertex: &VGraph,
    ) -> Result<Self, UpdateError> {
        if graph.n != vertex.len()
            || graph.dim != vertex.dim()
            || vertex.metric() != graph.metric.into()
        {
            return Err(UpdateError::CorruptBase(
                "graph and vertex encoding differ".into(),
            ));
        }
        let manifest = Manifest {
            identity: rand::random(),
            dim: graph.dim,
            sequence: 0,
            build: graph.build.clone(),
            encoding: vertex.update_encoding(),
        };
        Manifest::decode(&manifest.encode())?;
        for row in graph.vectors.chunks_exact(graph.dim) {
            validate_vector(row, graph.dim, graph.metric)?;
        }
        std::fs::create_dir(path)?;
        let file = lock(path)?;
        vertex.save(&manifest.vertex_path(path))?;
        let ids: Vec<_> = (0..graph.n as u32).collect();
        BaseRows::write(&manifest.rows_path(path), &manifest, &ids, &graph.vectors)?;
        format::rotate(&path.join("updates.log"), &manifest)?;
        atomic_write(&path.join("CURRENT"), &manifest.encode())?;
        sync_directory(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new(".")),
        )?;
        let base = Base::open(path, &manifest)?;
        Ok(Self {
            dir: path.to_path_buf(),
            _lock: file,
            writer: AtomicU8::new(0),
            state: RwLock::new(State::new(base, manifest)),
            diagnostics: diagnostics::Diagnostics::from_env(),
        })
    }
    /// Opens and replays committed batches; truncates only an incomplete final frame.
    pub fn open(path: &Path) -> Result<Self, UpdateError> {
        let file = lock(path)?;
        let manifest = Manifest::read(&path.join("CURRENT"))?;
        let base = Base::open(path, &manifest)?;
        let mut state = State::new(base, manifest.clone());
        let log = path.join("updates.log");
        let (sequence, end) = format::replay(&log, &manifest, |ops| state.replay(ops))?;
        let log_file = OpenOptions::new().write(true).open(&log)?;
        if log_file.metadata()?.len() != end {
            log_file.set_len(end)?;
            log_file.sync_all()?;
            sync_mapped_file(&log_file)?;
        }
        state.sequence = sequence;
        state.log_end = end;
        cleanup_generations(path, state.manifest.sequence)?;
        Ok(Self {
            dir: path.to_path_buf(),
            _lock: file,
            writer: AtomicU8::new(0),
            state: RwLock::new(state),
            diagnostics: diagnostics::Diagnostics::from_env(),
        })
    }
    fn writer(&self) -> Result<Writer<'_>, UpdateError> {
        self.writer
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|v| {
                if v == 2 {
                    UpdateError::RecoveryRequired
                } else {
                    UpdateError::WriterBusy
                }
            })?;
        Ok(Writer(&self.writer))
    }
    pub fn len(&self) -> usize {
        self.state.read().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dim(&self) -> usize {
        self.state.read().manifest.dim
    }
    /// Stages a new id in the current batch.
    pub fn insert(&self, id: u32, vector: &[f32]) -> Result<(), UpdateError> {
        self.stage(id, Some(vector), 1)
    }
    /// Stages deletion of an existing id.
    pub fn delete(&self, id: u32) -> Result<(), UpdateError> {
        self.stage(id, None, 2)
    }
    /// Atomically invalidates the previous vector and inserts the replacement.
    pub fn replace(&self, id: u32, vector: &[f32]) -> Result<(), UpdateError> {
        self.stage(id, Some(vector), 3)
    }
    fn stage(&self, id: u32, vector: Option<&[f32]>, tag: u8) -> Result<(), UpdateError> {
        let _writer = self.writer()?;
        let mut s = self.state.write();
        if id == u32::MAX {
            return Err(UpdateError::IdOutOfRange);
        }
        if s.touched.contains(&id) {
            return Err(UpdateError::DuplicateIdInBatch(id));
        }
        if let Some(v) = vector {
            validate_vector(v, s.manifest.dim, s.manifest.build.metric)?;
        }
        let size = 5 + vector.map_or(0, |v| v.len() * 4);
        if s.pending.len() >= MAX_OPS
            || s.pending_bytes + size > MAX_PAYLOAD
            || s.delta.len() >= (u32::MAX - 1) as usize
        {
            return Err(UpdateError::Capacity);
        }
        let op = match tag {
            1 => Operation::Insert(id, vector.expect("insert vector").to_vec()),
            2 => Operation::Delete(id),
            _ => Operation::Replace(id, vector.expect("replace vector").to_vec()),
        };
        s.validate_op(&op)?;
        s.apply(&op);
        s.touched.insert(id);
        s.pending_bytes += size;
        s.pending.push(op);
        Ok(())
    }
    /// Most operations one batch can stage before `commit` for vectors of this dimension.
    pub fn batch_capacity(&self) -> usize {
        MAX_OPS.min(MAX_PAYLOAD / (5 + 4 * self.dim()))
    }
    /// Stages new ids for row-major `vectors`, all or none, linking them into the delta in
    /// parallel. Up to `batch_capacity` operations fit in one batch.
    pub fn insert_batch(&self, ids: &[u32], vectors: &[f32]) -> Result<(), UpdateError> {
        let _writer = self.writer()?;
        let mut s = self.state.write();
        let dim = s.manifest.dim;
        if vectors.len() != ids.len() * dim {
            return Err(UpdateError::DimensionMismatch {
                expected: ids.len() * dim,
                actual: vectors.len(),
            });
        }
        let size = (5 + 4 * dim) * ids.len();
        if s.pending.len() + ids.len() > MAX_OPS
            || s.pending_bytes + size > MAX_PAYLOAD
            || s.delta.len() + ids.len() >= (u32::MAX - 1) as usize
        {
            return Err(UpdateError::Capacity);
        }
        let mut batch = HashSet::with_capacity(ids.len());
        for (&id, row) in ids.iter().zip(vectors.chunks_exact(dim)) {
            if id == u32::MAX {
                return Err(UpdateError::IdOutOfRange);
            }
            if s.touched.contains(&id) || !batch.insert(id) {
                return Err(UpdateError::DuplicateIdInBatch(id));
            }
            if s.contains(id) {
                return Err(UpdateError::AlreadyExists(id));
            }
            validate_vector(row, dim, s.manifest.build.metric)?;
        }
        let mut rows = vectors.to_vec();
        super::prepare_rows(&mut rows, dim, s.manifest.build.metric);
        let build = s.manifest.build.clone();
        s.delta.insert_batch(ids, &rows, &build);
        for (&id, row) in ids.iter().zip(vectors.chunks_exact(dim)) {
            s.pending.push(Operation::Insert(id, row.to_vec()));
            s.touched.insert(id);
        }
        s.pending_bytes += size;
        Ok(())
    }
    /// Appends the whole pending batch and completes its durable sync.
    pub fn commit(&self) -> Result<(), UpdateError> {
        let writer = self.writer()?;
        self.commit_with(&writer)
    }
    fn commit_with(&self, writer: &Writer<'_>) -> Result<(), UpdateError> {
        self.commit_with_checkpoints(writer, |_| Ok(()))
    }
    fn commit_with_checkpoints(
        &self,
        writer: &Writer<'_>,
        mut checkpoint: impl FnMut(u8) -> Result<(), UpdateError>,
    ) -> Result<(), UpdateError> {
        let (bytes, end, sequence) = {
            let s = self.state.read();
            if s.pending.is_empty() {
                return Ok(());
            }
            let sequence = s.sequence.checked_add(1).ok_or(UpdateError::Capacity)?;
            (
                format::frame(&s.pending, sequence, s.manifest.dim)?,
                s.log_end,
                sequence,
            )
        };
        if let Err(e) = format::append(&self.dir.join("updates.log"), end, &bytes, &mut checkpoint)
        {
            writer.uncertain();
            return Err(e);
        }
        let mut s = self.state.write();
        s.sequence = sequence;
        s.log_end += bytes.len() as u64;
        s.pending.clear();
        s.touched.clear();
        s.pending_bytes = 0;
        drop(s);
        if let Err(error) = checkpoint(3) {
            writer.uncertain();
            return Err(error);
        }
        Ok(())
    }
    /// Searches a consistent in-memory view. `k` is clamped to the live count.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        params: VSearchParams,
    ) -> Result<Vec<u32>, UpdateError> {
        let mut out = Vec::new();
        self.search_into(query, k, params, &mut out)?;
        Ok(out)
    }
    /// Searches with a reusable result buffer, avoiding allocation on clean-base queries.
    pub fn search_into(
        &self,
        query: &[f32],
        k: usize,
        params: VSearchParams,
        out: &mut Vec<u32>,
    ) -> Result<(), UpdateError> {
        let s = self.state.read();
        validate_vector(query, s.manifest.dim, s.manifest.build.metric)?;
        let k = k.min(s.len());
        out.clear();
        if k == 0 {
            return Ok(());
        }
        self.diagnostics.query();
        if s.delta.len() == 0 && s.deleted == 0 {
            if let Some(graph) = &s.base.graph {
                SCRATCH.with(|cache| {
                    graph.search_reusing(&mut cache.borrow_mut(), query, k, params, out)
                })?;
                if !s.base.dense_ids {
                    for slot in out {
                        *slot = s.base.rows.ids[*slot as usize];
                    }
                }
            }
            return Ok(());
        }
        if let Some(graph) = s
            .base
            .graph
            .as_ref()
            .filter(|_| s.deleted < s.base.rows.ids.len())
        {
            let live = graph.len() - s.deleted;
            let mut take = widened_width(params.ef.max(params.rerank).max(k), graph.len(), live);
            loop {
                let mut widened = params;
                widened.ef = take;
                widened.rerank = 0;
                widened.patience = 0;
                SCRATCH.with(|cache| {
                    graph.search_reusing(&mut cache.borrow_mut(), query, take, widened, out)
                })?;
                if take == graph.len()
                    || out.iter().filter(|&&slot| !s.dead(slot as usize)).count() >= k.min(live)
                {
                    break;
                }
                take = take.saturating_mul(2).min(graph.len());
            }
        }
        let metric = s.manifest.build.metric;
        let mut candidates =
            Vec::with_capacity(out.len() + s.delta.live.len().min(params.ef.max(k)));
        for &slot in out.iter() {
            let slot = slot as usize;
            if !s.dead(slot) {
                candidates.push((
                    distance(query, s.base.rows.row(slot)?, metric),
                    s.base.rows.ids[slot],
                ));
            }
        }
        if !s.delta.live.is_empty() {
            let mut ef = widened_width(
                params.ef.max(params.rerank).max(k),
                s.delta.len(),
                s.delta.live.len(),
            );
            loop {
                let found = if self.diagnostics.enabled {
                    let (found, visits) = s.delta.search_profiled(query, ef, metric);
                    self.diagnostics.delta_search(visits, found.len());
                    found
                } else {
                    s.delta.search(query, ef, metric)
                };
                if ef == s.delta.len() || found.len() >= k.min(s.delta.live.len()) {
                    candidates.extend(found);
                    break;
                }
                ef = ef.saturating_mul(2).min(s.delta.len());
            }
        }
        candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        candidates.truncate(k);
        out.clear();
        out.extend(candidates.into_iter().map(|(_, id)| id));
        Ok(())
    }
    /// Returns streaming counters and delta topology when `HMS_STREAM_DIAGNOSTICS=1`
    /// was set at open/create. Topology scans run only on this explicit call. Base topology
    /// is unavailable through the immutable graph interface and is marked accordingly.
    pub fn streaming_diagnostics(&self) -> Option<std::collections::BTreeMap<String, f64>> {
        self.diagnostics.snapshot(&self.state.read())
    }
    pub fn needs_consolidation(&self, policy: ConsolidationPolicy) -> Result<bool, UpdateError> {
        if !policy.delta_ratio.is_finite()
            || !policy.tombstone_ratio.is_finite()
            || !(0.0..=1.0).contains(&policy.delta_ratio)
            || !(0.0..=1.0).contains(&policy.tombstone_ratio)
        {
            return Err(UpdateError::InvalidVector(
                "consolidation ratios must be in 0..=1",
            ));
        }
        let s = self.state.read();
        let base = s.base.rows.ids.len().max(1) as f64;
        Ok(
            (s.delta.len() >= policy.min_delta
                && s.delta.len() as f64 / base >= policy.delta_ratio)
                || (s.deleted > 0 && s.deleted as f64 / base >= policy.tombstone_ratio),
        )
    }
    /// Commits pending work and publishes a new immutable generation while readers run.
    pub fn consolidate(&self) -> Result<(), UpdateError> {
        self.consolidate_inner(|_| Ok(()))
    }
    fn consolidate_inner(
        &self,
        mut checkpoint: impl FnMut(u8) -> Result<(), UpdateError>,
    ) -> Result<(), UpdateError> {
        let writer = self.writer()?;
        self.commit_with(&writer)?;
        checkpoint(0)?;
        // The writer flag excludes every other mutation, so a read guard sees the same state
        // a clone would, without copying the delta.
        let snapshot = self.state.read();
        if snapshot.delta.len() == 0 && snapshot.deleted == 0 {
            return Ok(());
        }
        let mut manifest = snapshot.manifest.clone();
        manifest.sequence = snapshot.sequence;
        let mut ids: Vec<_> = snapshot
            .base
            .rows
            .ids
            .iter()
            .enumerate()
            .filter(|(slot, _)| !snapshot.dead(*slot))
            .map(|(_, id)| *id)
            .chain(snapshot.delta.live.keys().copied())
            .collect();
        ids.sort_unstable();
        // Fixed-width neighbour ids take the narrowest width this generation's count allows,
        // as the static index does: 3 bytes save `degree` bytes per row below 2^24 - 1 rows.
        if matches!(manifest.encoding.id_bytes, 3 | 4) {
            manifest.encoding.id_bytes = if ids.len() < THREE_BYTE_IDS_BELOW {
                3
            } else {
                4
            };
        }
        let base = Arc::clone(&snapshot.base);
        drop(snapshot);
        if !manifest.encoding.screen && manifest.encoding.block_bits == 0 {
            self.write_mapped_base(&manifest, &ids, &base, &mut checkpoint)?;
        } else {
            let snapshot = self.state.read();
            let mut rows = Vec::new();
            rows.try_reserve_exact(
                ids.len()
                    .checked_mul(manifest.dim)
                    .ok_or(UpdateError::Capacity)?,
            )
            .map_err(|_| UpdateError::Capacity)?;
            for &id in &ids {
                if let Some(&slot) = snapshot.delta.live.get(&id) {
                    rows.extend_from_slice(snapshot.delta.row(slot, manifest.dim));
                } else {
                    rows.extend_from_slice(
                        snapshot
                            .base
                            .rows
                            .row(snapshot.base_slot(id).expect("live base id"))?,
                    );
                }
            }
            drop(snapshot);
            write_base_with_checkpoints(&self.dir, &manifest, &ids, rows, &mut checkpoint)?;
        }
        drop(base);
        checkpoint(1)?;
        let base = Base::open(&self.dir, &manifest)?;
        let publication = (|| {
            atomic_write(&self.dir.join("CURRENT"), &manifest.encode())?;
            checkpoint(2)?;
            checkpoint(3)?;
            format::rotate(&self.dir.join("updates.log"), &manifest)?;
            checkpoint(4)?;
            Ok::<_, UpdateError>(())
        })();
        if let Err(e) = publication {
            writer.uncertain();
            return Err(e);
        }
        *self.state.write() = State::new(base, manifest.clone());
        if let Err(error) = checkpoint(5) {
            writer.uncertain();
            return Err(error);
        }
        cleanup_generations(&self.dir, manifest.sequence)?;
        Ok(())
    }
    fn write_mapped_base(
        &self,
        manifest: &Manifest,
        ids: &[u32],
        base: &Base,
        mut checkpoint: impl FnMut(u8) -> Result<(), UpdateError>,
    ) -> Result<(), UpdateError> {
        checkpoint(6)?;
        let mut output = BaseRowsWriter::new(&manifest.rows_path(&self.dir), manifest, ids)?;
        let mut prepared = PreparedRows::new(ids.len(), manifest.dim)?;
        let chunk_rows = ((4 << 20) / 4 / manifest.dim).max(1);
        let mut rows = Vec::new();
        rows.try_reserve_exact(
            chunk_rows
                .min(ids.len())
                .checked_mul(manifest.dim)
                .ok_or(UpdateError::Capacity)?,
        )
        .map_err(|_| UpdateError::Capacity)?;
        if let Some(graph) = &base.graph {
            graph.release_pages()?;
        }
        for chunk in ids.chunks(chunk_rows) {
            rows.clear();
            let snapshot = self.state.read();
            for &id in chunk {
                if let Some(&slot) = snapshot.delta.live.get(&id) {
                    rows.extend_from_slice(snapshot.delta.row(slot, manifest.dim));
                } else {
                    rows.extend_from_slice(
                        base.rows
                            .row(snapshot.base_slot(id).expect("live base id"))?,
                    );
                }
            }
            drop(snapshot);
            output.append(&rows)?;
            prepared.append(&rows)?;
            base.rows.release_pages()?;
        }
        output.finish()?;
        checkpoint(7)?;
        if let Some(graph) = &base.graph {
            graph.release_pages()?;
        }
        if !ids.is_empty() {
            let graph = prepared.build(&manifest.build)?;
            checkpoint(8)?;
            let e = &manifest.encoding;
            VGraph::save_prepared(
                &graph,
                &manifest.vertex_path(&self.dir),
                e.bits,
                e.residual_bits,
                e.align,
                e.reorder,
                e.id_bytes,
            )?;
            checkpoint(9)?;
        }
        sync_directory(&self.dir)?;
        checkpoint(10)?;
        Ok(())
    }
}
fn cleanup_generations(dir: &Path, active: u64) -> Result<(), UpdateError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let sequence = name
            .strip_prefix("base-")
            .and_then(|s| s.strip_suffix(".vertex"))
            .or_else(|| {
                name.strip_prefix("rows-")
                    .and_then(|s| s.strip_suffix(".bin"))
            })
            .and_then(|s| s.parse::<u64>().ok());
        if sequence.is_some_and(|s| s != active) {
            std::fs::remove_file(entry.path())?;
        }
    }
    sync_directory(dir)?;
    Ok(())
}

/// Writes the rows file, then builds the graph from the same buffer (`rows` are already
/// prepared, so the build's in-place preparation leaves them unchanged).
fn write_base(
    dir: &Path,
    manifest: &Manifest,
    ids: &[u32],
    rows: Vec<f32>,
) -> Result<(), UpdateError> {
    write_base_with_checkpoints(dir, manifest, ids, rows, |_| Ok(()))
}

fn write_base_with_checkpoints(
    dir: &Path,
    manifest: &Manifest,
    ids: &[u32],
    rows: Vec<f32>,
    mut checkpoint: impl FnMut(u8) -> Result<(), UpdateError>,
) -> Result<(), UpdateError> {
    checkpoint(6)?;
    BaseRows::write(&manifest.rows_path(dir), manifest, ids, &rows)?;
    checkpoint(7)?;
    if !ids.is_empty() {
        let graph = Graph::build_owned(rows, manifest.dim, &manifest.build)?;
        let e = &manifest.encoding;
        checkpoint(8)?;
        if !e.screen && e.block_bits == 0 {
            VGraph::save_from_graph(
                &graph,
                &manifest.vertex_path(dir),
                e.bits,
                e.residual_bits,
                e.align,
                e.reorder,
                e.id_bytes,
            )?;
        } else {
            let mut vertex = VGraph::from_graph_with(
                &graph,
                e.bits,
                e.residual_bits,
                e.align,
                e.reorder,
                e.id_bytes,
                e.screen,
            )?;
            if e.block_bits != 0 {
                vertex = vertex.with_blocks(&graph, e.block_bits)?;
            }
            vertex.save(&manifest.vertex_path(dir))?;
        }
        checkpoint(9)?;
    }
    sync_directory(dir)?;
    checkpoint(10)?;
    Ok(())
}

#[cfg(test)]
mod tests;
