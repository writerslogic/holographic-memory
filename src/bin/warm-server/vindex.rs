// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The persisted vector index of a tenant: the graph that answers unfiltered vector queries,
//! kept on disk so a restart opens it instead of rebuilding it.
//!
//! Layout under `<tenant>/vindex/`:
//! - `rows.log`: every vector ingested through the documents API, appended and synced before
//!   the engine commits the document (one CRC-checked frame per insert, replace or delete). A
//!   torn tail is cut at open.
//! - `CURRENT`, `graph-<epoch>.vertex`, `rows-<epoch>.bin`, `ids-<epoch>.bin`: the snapshot
//!   maintenance built: the vertex graph (`VGraph::save`), its l2-normalised `f32` rows and
//!   the document id of every vertex. Each is written to a temporary file, synced and renamed,
//!   and checked at open; a snapshot that does not read back is ignored, the log alone answers
//!   (exactly, in memory) and the next maintenance writes a fresh one. The log is rotated
//!   when a snapshot is published, keeping only the frames the snapshot does not cover.
//! - `dirty`: present from the first mutation after open until a clean close. Found at open
//!   it means the previous process stopped mid-way: every indexed document is then checked
//!   against the engine, and one the engine lacks, or holds at another version, leaves the
//!   index.
//!
//! Rows the snapshot covers are scored through the mapped rows file; rows appended since are
//! scored exactly in memory. Hits are validated against the engine by the caller, so a
//! document deleted since the capture never reaches a response. The index covers the
//! documents ingested through the documents API with a vector; `covers` tells whether that is
//! every embedded chunk of the store, which is when queries may be answered from it.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use writerslogic_warm::core::documents::MAX_EF;
use writerslogic_warm::core::qgraph::{BuildParams, Graph, Residency, VGraph, VSearchParams};
use writerslogic_warm::WarmCore;

const LOG_MAGIC: &[u8; 8] = b"HMSSVLG1";
const ROWS_MAGIC: &[u8; 8] = b"HMSSVRW1";
const IDS_MAGIC: &[u8; 8] = b"HMSSVID1";
/// Log header: magic, dimensions (`u32`), epoch (`u64`).
const LOG_HEADER: u64 = 8 + 4 + 8;
/// Rows header: magic, row count (`u64`), dimensions (`u64`).
const ROWS_HEADER: usize = 8 + 8 + 8;
const UPSERT: u8 = 1;
const DELETE: u8 = 2;
/// Mutations since the snapshot after which scheduled maintenance rebuilds it (the engine's
/// own threshold for its snapshot).
pub const REBUILD_AFTER_MUTATIONS: usize = 64;
/// Prefix of the document version the server stamps on an indexed document; the memory API's
/// stamps use another prefix.
const VERSION_PREFIX: &str = "hv1:";

/// The document version recorded for an indexed vector: lets a restart after an unclean stop
/// tell whether the engine holds the document revision the index holds.
pub fn version_stamp(token: u64) -> String {
    format!("{VERSION_PREFIX}{token:016x}")
}

/// l2-normalises `values` the way the engine does before storing an embedding, as `f32`.
pub fn normalize(values: &[f64]) -> Result<Vec<f32>> {
    let mut scale = 0.0_f64;
    for &value in values {
        ensure!(value.is_finite(), "vector must be finite");
        scale = scale.max(value.abs());
    }
    ensure!(scale > 0.0, "vector must not be all zeros");
    let norm = values
        .iter()
        .map(|x| (x / scale) * (x / scale))
        .sum::<f64>()
        .sqrt();
    let magnitude = scale * norm;
    if magnitude.is_normal() {
        Ok(values.iter().map(|x| (x / magnitude) as f32).collect())
    } else {
        Ok(values.iter().map(|x| ((x / scale) / norm) as f32).collect())
    }
}

/// Dot product with eight independent accumulators so the loop vectorises.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (wide_a, tail_a) = a.as_chunks::<8>();
    let (wide_b, tail_b) = b.as_chunks::<8>();
    for (x, y) in wide_a.iter().zip(wide_b) {
        for ((lane, x), y) in lanes.iter_mut().zip(x).zip(y) {
            *lane += x * y;
        }
    }
    let mut sum = lanes.iter().sum::<f32>();
    for (x, y) in tail_a.iter().zip(tail_b) {
        sum += x * y;
    }
    sum
}

/// `fsync` without the drive-cache flush: the durability class of the engine's per-commit
/// `msync`, which the index shares (a vector is never more durable than its document).
fn fsync_plain(file: &File) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        loop {
            // SAFETY: `file` is open for the duration of the call; fsync takes no buffer.
            if unsafe { libc::fsync(file.as_raw_fd()) } == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
    #[cfg(not(unix))]
    file.sync_data()
}

pub fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Writes `bytes` to `path` through a temporary file next to it, synced before the rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut file = File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Clone, Serialize, Deserialize)]
struct Manifest {
    epoch: u64,
    /// Epoch of the log the snapshot was captured from, and the log offset the capture
    /// covers: frames at or past it, or in a log of another epoch, are not in the snapshot.
    log_epoch: u64,
    covered: u64,
    rows: u64,
    dim: u64,
    #[serde(default)]
    rows_crc: Option<u32>,
}

/// The snapshot: the vertex graph over `n` rows and the mapped rows file.
struct Base {
    /// `None` when the graph file did not read back: the rows still answer, exactly, until
    /// maintenance rebuilds it.
    graph: Option<VGraph>,
    rows: memmap2::Mmap,
    n: usize,
}

impl Base {
    fn row_into(&self, dim: usize, slot: usize, out: &mut Vec<f32>) {
        let at = ROWS_HEADER + slot * dim * 4;
        out.clear();
        out.extend(
            self.rows[at..at + dim * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        );
    }
}

/// Slots `0..base_len` are the snapshot's vertices; later slots are rows appended since, in
/// log order, with their vectors in `delta`. A slot whose id is `None` was deleted or
/// replaced (its vector stays in place until the next snapshot).
struct State {
    dim: usize,
    base: Option<Base>,
    base_len: usize,
    ids: Vec<Option<Arc<str>>>,
    tokens: Vec<u64>,
    by_id: HashMap<Arc<str>, u32>,
    delta: Vec<f32>,
    live: usize,
    base_dead: usize,
    mutations: usize,
}

impl State {
    fn empty(dim: usize) -> Self {
        Self {
            dim,
            base: None,
            base_len: 0,
            ids: Vec::new(),
            tokens: Vec::new(),
            by_id: HashMap::new(),
            delta: Vec::new(),
            live: 0,
            base_dead: 0,
            mutations: 0,
        }
    }

    fn tombstone(&mut self, slot: u32) {
        let slot = slot as usize;
        if self.ids[slot].take().is_some() {
            self.live -= 1;
            if slot < self.base_len {
                self.base_dead += 1;
            }
        }
    }

    fn upsert(&mut self, id: Arc<str>, token: u64, vector: &[f32]) {
        if let Some(previous) = self.by_id.get(&id).copied() {
            self.tombstone(previous);
        }
        let slot = self.ids.len() as u32;
        self.ids.push(Some(id.clone()));
        self.tokens.push(token);
        self.delta.extend_from_slice(vector);
        self.by_id.insert(id, slot);
        self.live += 1;
        self.mutations += 1;
    }

    fn delete(&mut self, id: &str) -> bool {
        match self.by_id.remove(id) {
            Some(slot) => {
                self.tombstone(slot);
                self.mutations += 1;
                true
            }
            None => false,
        }
    }

    fn row_into(&self, slot: usize, out: &mut Vec<f32>) {
        if slot < self.base_len {
            self.base
                .as_ref()
                .expect("base slots exist only with a base")
                .row_into(self.dim, slot, out);
        } else {
            let at = (slot - self.base_len) * self.dim;
            out.clear();
            out.extend_from_slice(&self.delta[at..at + self.dim]);
        }
    }
}

struct Log {
    file: File,
    len: u64,
    epoch: u64,
}

fn log_header(dim: usize, epoch: u64) -> Vec<u8> {
    let mut out = LOG_MAGIC.to_vec();
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.extend_from_slice(&epoch.to_le_bytes());
    out
}

fn push_frame(out: &mut Vec<u8>, kind: u8, id: &str, token: u64, vector: &[f32]) {
    let start = out.len();
    out.push(kind);
    out.extend_from_slice(&(id.len() as u16).to_le_bytes());
    out.extend_from_slice(id.as_bytes());
    out.extend_from_slice(&token.to_le_bytes());
    for value in vector {
        out.extend_from_slice(&value.to_le_bytes());
    }
    let crc = crc32fast::hash(&out[start..]);
    out.extend_from_slice(&crc.to_le_bytes());
}

struct Frame<'a> {
    kind: u8,
    id: &'a str,
    token: u64,
    vector: &'a [u8],
    len: usize,
}

/// One complete, checksummed frame at the start of `bytes`, or `None` at a torn or damaged
/// tail.
fn read_frame(bytes: &[u8], dim: usize) -> Option<Frame<'_>> {
    if bytes.len() < 3 {
        return None;
    }
    let kind = bytes[0];
    let id_len = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
    let vector_len = match kind {
        UPSERT => dim * 4,
        DELETE => 0,
        _ => return None,
    };
    let body = 3 + id_len + 8 + vector_len;
    let len = body + 4;
    if bytes.len() < len {
        return None;
    }
    let crc = u32::from_le_bytes(bytes[body..len].try_into().ok()?);
    if crc32fast::hash(&bytes[..body]) != crc {
        return None;
    }
    let id = std::str::from_utf8(&bytes[3..3 + id_len]).ok()?;
    let token = u64::from_le_bytes(bytes[3 + id_len..3 + id_len + 8].try_into().ok()?);
    Some(Frame {
        kind,
        id,
        token,
        vector: &bytes[3 + id_len + 8..body],
        len,
    })
}

fn apply_frames(state: &mut State, bytes: &[u8], dim: usize) -> usize {
    let mut at = 0;
    let mut scratch = Vec::with_capacity(dim);
    while let Some(frame) = read_frame(&bytes[at..], dim) {
        match frame.kind {
            UPSERT => {
                scratch.clear();
                scratch.extend(
                    frame
                        .vector
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b)),
                );
                state.upsert(Arc::from(frame.id), frame.token, &scratch);
            }
            _ => {
                state.delete(frame.id);
            }
        }
        at += frame.len;
    }
    at
}

/// What the stats and maintenance responses report for the index.
pub struct Status {
    pub rows: usize,
    pub graph: bool,
    pub mutations: usize,
}

pub struct VectorIndex {
    dir: PathBuf,
    dim: usize,
    state: RwLock<State>,
    log: Mutex<Log>,
    build: Mutex<()>,
    manifest: Mutex<Option<Manifest>>,
    dirty: AtomicBool,
    token: AtomicU64,
}

impl VectorIndex {
    /// Opens (or creates) the index under `dir`, cutting a torn log tail, ignoring a
    /// snapshot that does not read back and, after an unclean stop, dropping every indexed
    /// document `core` does not hold at the indexed version.
    pub fn open(dir: &Path, dim: usize, core: &WarmCore) -> Result<Self> {
        ensure!(dim > 0, "vector index needs at least one dimension");
        std::fs::create_dir_all(dir)?;
        let manifest = match std::fs::read(dir.join("CURRENT")) {
            Ok(bytes) => serde_json::from_slice::<Manifest>(&bytes).ok(),
            Err(_) => None,
        };
        let mut state = State::empty(dim);
        let mut manifest = manifest.filter(|m| m.dim == dim as u64);
        if let Some(loaded) = manifest.as_ref().map(|m| load_base(dir, m, dim)) {
            match loaded {
                Ok((base, ids, tokens)) => {
                    state.base_len = base.n;
                    state.live = base.n;
                    state.base = Some(base);
                    state.by_id = ids
                        .iter()
                        .enumerate()
                        .map(|(slot, id)| (id.clone(), slot as u32))
                        .collect();
                    state.ids = ids.into_iter().map(Some).collect();
                    state.tokens = tokens;
                }
                Err(error) => {
                    crate::log_event!(Warn, "vector_index_snapshot_unreadable", "path" => dir.display().to_string(), "detail" => format!("{error:#}"));
                    manifest = None;
                }
            }
        }
        let (log, bytes) = open_log(dir, dim, manifest.as_ref().map_or(0, |m| m.epoch))?;
        let skip = match &manifest {
            Some(m) if m.log_epoch == log.epoch => {
                (m.covered.max(LOG_HEADER) as usize).min(bytes.len())
            }
            _ => LOG_HEADER as usize,
        };
        let applied = apply_frames(&mut state, &bytes[skip..], dim);
        let mut log = log;
        let end = (skip + applied) as u64;
        if end < log.len {
            crate::log_event!(Warn, "vector_index_log_truncated", "path" => dir.display().to_string(), "dropped_bytes" => log.len - end);
            log.file.set_len(end)?;
            fsync_plain(&log.file)?;
            log.len = end;
        }
        // The snapshot's own rows are not mutations; replayed frames are counted as such so
        // maintenance knows the snapshot is behind.
        state.mutations = state.ids.len() - state.base_len + state.base_dead;
        let index = Self {
            dir: dir.to_path_buf(),
            dim,
            state: RwLock::new(state),
            log: Mutex::new(log),
            build: Mutex::new(()),
            manifest: Mutex::new(manifest),
            dirty: AtomicBool::new(false),
            token: AtomicU64::new(rand::random()),
        };
        if dir.join("dirty").exists() {
            index.reconcile(core)?;
        }
        Ok(index)
    }

    /// After an unclean stop: drop every indexed document the engine does not hold at the
    /// version the index recorded (the request may have stopped between the index append
    /// and the engine commit), then clear the marker.
    fn reconcile(&self, core: &WarmCore) -> Result<()> {
        let stale: Vec<Arc<str>> = {
            let state = self.state.read();
            state
                .ids
                .iter()
                .zip(&state.tokens)
                .filter_map(|(id, token)| {
                    let id = id.as_ref()?;
                    let keep = core
                        .get_document(id)
                        .is_some_and(|record| record.version == version_stamp(*token));
                    (!keep).then(|| id.clone())
                })
                .collect()
        };
        let ids: Vec<&str> = stale.iter().map(AsRef::as_ref).collect();
        if !ids.is_empty() {
            self.remove(&ids)?;
        }
        crate::log_event!(Info, "vector_index_reconciled", "path" => self.dir.display().to_string(), "dropped" => ids.len());
        self.clear_dirty()
    }

    fn mark_dirty(&self) -> Result<()> {
        if self.dirty.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let result = (|| -> Result<()> {
            let file = File::create(self.dir.join("dirty"))?;
            fsync_plain(&file)?;
            sync_dir(&self.dir)?;
            Ok(())
        })();
        if result.is_err() {
            self.dirty.store(false, Ordering::Release);
        }
        result
    }

    fn clear_dirty(&self) -> Result<()> {
        match std::fs::remove_file(self.dir.join("dirty")) {
            Ok(()) => sync_dir(&self.dir)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        self.dirty.store(false, Ordering::Release);
        Ok(())
    }

    /// Marks the index cleanly closed; the next open trusts the log without checking the
    /// engine. Call after the engine flushed.
    pub fn close(&self) -> Result<()> {
        let log = self.log.lock();
        fsync_plain(&log.file)?;
        drop(log);
        self.clear_dirty()
    }

    pub fn next_token(&self) -> u64 {
        self.token.fetch_add(1, Ordering::Relaxed)
    }

    pub fn status(&self) -> Status {
        let state = self.state.read();
        Status {
            rows: state.base_len,
            graph: state.base.as_ref().is_some_and(|b| b.graph.is_some()),
            mutations: state.mutations,
        }
    }

    /// Indexed documents (snapshot rows plus those appended since, minus the deleted).
    #[cfg(test)]
    pub fn live(&self) -> usize {
        self.state.read().live
    }

    /// Whether the index holds a graph over every embedded chunk of the store, so an
    /// unfiltered graph query can be answered from it.
    pub fn covers(&self, core: &WarmCore) -> bool {
        let state = self.state.read();
        state.base.as_ref().is_some_and(|b| b.graph.is_some())
            && state.live == core.document_embedding_count()
    }

    /// Retain the returned guard through state publication so maintenance captures both together.
    fn append(&self, bytes: &[u8]) -> Result<parking_lot::MutexGuard<'_, Log>> {
        let mut log = self.log.lock();
        self.mark_dirty()?;
        let end = log.len;
        log.file.seek(SeekFrom::Start(end))?;
        log.file.write_all(bytes)?;
        fsync_plain(&log.file)?;
        log.len += bytes.len() as u64;
        Ok(log)
    }

    /// Records `docs` (id, version token, normalised vector) durably and makes them
    /// searchable; an id already indexed is replaced. Call before the engine commits them.
    pub fn stage(&self, docs: &[(String, u64, Vec<f32>)]) -> Result<()> {
        let mut bytes = Vec::with_capacity(docs.len() * (16 + self.dim * 4));
        for (id, token, vector) in docs {
            ensure!(
                vector.len() == self.dim,
                "vector has {} dimensions, the index {}",
                vector.len(),
                self.dim
            );
            ensure!(
                id.len() <= u16::MAX as usize,
                "document id too long for the vector index"
            );
            push_frame(&mut bytes, UPSERT, id, *token, vector);
        }
        let _log = self.append(&bytes)?;
        let mut state = self.state.write();
        for (id, token, vector) in docs {
            state.upsert(Arc::from(id.as_str()), *token, vector);
        }
        Ok(())
    }

    /// Drops `ids` from the index (unknown ids are ignored).
    pub fn remove(&self, ids: &[&str]) -> Result<()> {
        let known: Vec<&str> = {
            let state = self.state.read();
            ids.iter()
                .copied()
                .filter(|id| state.by_id.contains_key(*id))
                .collect()
        };
        if known.is_empty() {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(known.len() * 32);
        for id in &known {
            push_frame(&mut bytes, DELETE, id, 0, &[]);
        }
        let _log = self.append(&bytes)?;
        let mut state = self.state.write();
        for id in known {
            state.delete(id);
        }
        Ok(())
    }

    /// The `k` nearest indexed documents to the l2-normalised `query` by cosine, best first:
    /// the graph's candidates re-scored exactly, merged with an exact pass over the rows the
    /// snapshot does not cover.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(Arc<str>, f32)>> {
        ensure!(
            query.len() == self.dim,
            "query has {} dimensions, the index {}",
            query.len(),
            self.dim
        );
        let state = self.state.read();
        let mut candidates: Vec<(f32, u32)> = Vec::new();
        let mut row = Vec::with_capacity(self.dim);
        match &state.base {
            Some(Base {
                graph: Some(graph),
                n,
                ..
            }) => {
                let live = n - state.base_dead;
                let mut take = if live == 0 {
                    0
                } else if state.base_dead > 0 {
                    k.saturating_mul(4).min(*n)
                } else {
                    k.min(*n)
                };
                while take > 0 {
                    let ef = ef.max(take).min(MAX_EF.max(take));
                    let params = VSearchParams {
                        ef,
                        rerank: ef,
                        ..VSearchParams::default()
                    };
                    let mut found = Vec::with_capacity(take);
                    graph
                        .searcher()
                        .search(query, take, params, &mut found)
                        .map_err(|e| anyhow::anyhow!("graph search: {e}"))?;
                    let enough = found
                        .iter()
                        .filter(|&&v| state.ids.get(v as usize).is_some_and(Option::is_some))
                        .count()
                        >= k.min(live);
                    if !enough && take < *n {
                        take = take.saturating_mul(2).min(*n);
                        continue;
                    }
                    for vertex in found {
                        let slot = vertex as usize;
                        if slot < state.base_len && state.ids[slot].is_some() {
                            state.row_into(slot, &mut row);
                            candidates.push((dot(query, &row), vertex));
                        }
                    }
                    break;
                }
            }
            Some(_) => {
                for slot in (0..state.base_len).filter(|&slot| state.ids[slot].is_some()) {
                    state.row_into(slot, &mut row);
                    candidates.push((dot(query, &row), slot as u32));
                }
            }
            None => {}
        }
        for (i, vector) in state.delta.chunks_exact(self.dim).enumerate() {
            let slot = state.base_len + i;
            if state.ids[slot].is_some() {
                candidates.push((dot(query, vector), slot as u32));
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        candidates.dedup_by(|a, b| a.1 == b.1);
        candidates.truncate(k);
        Ok(candidates
            .into_iter()
            .filter_map(|(score, slot)| {
                state.ids[slot as usize]
                    .clone()
                    .map(|id| (id, score.clamp(-1.0, 1.0)))
            })
            .collect())
    }

    /// Builds and publishes a snapshot over the live rows when there are at least
    /// `graph_min_rows` of them and something changed since the current one: always with
    /// `force`, otherwise after `REBUILD_AFTER_MUTATIONS` mutations. Another build in progress
    /// is left to finish.
    pub fn maintain(&self, graph_min_rows: usize, force: bool) -> Result<()> {
        let Some(_build) = self.build.try_lock() else {
            return Ok(());
        };
        let checksummed = self
            .manifest
            .lock()
            .as_ref()
            .is_some_and(|m| m.rows_crc.is_some());
        let (matrix, ids, tokens, covered, log_epoch) = {
            let log = self.log.lock();
            let state = self.state.read();
            if state.live < graph_min_rows.max(1) {
                return Ok(());
            }
            let has_graph = checksummed && state.base.as_ref().is_some_and(|b| b.graph.is_some());
            if has_graph
                && (state.mutations == 0 || (!force && state.mutations < REBUILD_AFTER_MUTATIONS))
            {
                return Ok(());
            }
            let mut matrix = Vec::new();
            matrix
                .try_reserve_exact(state.live * self.dim)
                .context("cannot allocate the graph build matrix")?;
            let mut ids = Vec::with_capacity(state.live);
            let mut tokens = Vec::with_capacity(state.live);
            let mut row = Vec::with_capacity(self.dim);
            for slot in 0..state.ids.len() {
                let Some(id) = &state.ids[slot] else {
                    continue;
                };
                state.row_into(slot, &mut row);
                matrix.extend_from_slice(&row);
                ids.push(id.clone());
                tokens.push(state.tokens[slot]);
            }
            (matrix, ids, tokens, log.len, log.epoch)
        };
        let n = ids.len();
        let epoch = self.manifest.lock().as_ref().map_or(0, |m| m.epoch) + 1;
        let rows_path = self.dir.join(format!("rows-{epoch}.bin"));
        let ids_path = self.dir.join(format!("ids-{epoch}.bin"));
        let graph_path = self.dir.join(format!("graph-{epoch}.vertex"));
        let mut rows_bytes = Vec::with_capacity(ROWS_HEADER + matrix.len() * 4);
        rows_bytes.extend_from_slice(ROWS_MAGIC);
        rows_bytes.extend_from_slice(&(n as u64).to_le_bytes());
        rows_bytes.extend_from_slice(&(self.dim as u64).to_le_bytes());
        for value in &matrix {
            rows_bytes.extend_from_slice(&value.to_le_bytes());
        }
        write_atomic(&rows_path, &rows_bytes)?;
        let rows_crc = Some(crc32fast::hash(&rows_bytes));
        drop(rows_bytes);
        let mut ids_bytes = Vec::with_capacity(16 + n * 24);
        ids_bytes.extend_from_slice(IDS_MAGIC);
        ids_bytes.extend_from_slice(&(n as u64).to_le_bytes());
        for (id, token) in ids.iter().zip(&tokens) {
            ids_bytes.extend_from_slice(&(id.len() as u16).to_le_bytes());
            ids_bytes.extend_from_slice(id.as_bytes());
            ids_bytes.extend_from_slice(&token.to_le_bytes());
        }
        let crc = crc32fast::hash(&ids_bytes);
        ids_bytes.extend_from_slice(&crc.to_le_bytes());
        write_atomic(&ids_path, &ids_bytes)?;
        let built = Graph::build_owned(matrix, self.dim, &BuildParams::default())
            .map_err(|e| anyhow::anyhow!("graph build: {e}"))?;
        let graph =
            VGraph::from_graph(&built, true).map_err(|e| anyhow::anyhow!("graph encode: {e}"))?;
        drop(built);
        ensure!(
            graph.len() == n,
            "graph has {} vertices for {n} rows",
            graph.len()
        );
        graph.save(&graph_path)?;
        sync_dir(&self.dir)?;
        let manifest = Manifest {
            epoch,
            log_epoch,
            covered,
            rows: n as u64,
            dim: self.dim as u64,
            rows_crc,
        };
        write_atomic(&self.dir.join("CURRENT"), &serde_json::to_vec(&manifest)?)?;
        sync_dir(&self.dir)?;
        let rows_file = File::open(&rows_path)?;
        // SAFETY: the file is replaced only by renaming a new one over it, never modified or
        // truncated in place, so the mapping's bytes are stable while it is open.
        let rows = unsafe { memmap2::Mmap::map(&rows_file)? };
        let mut fresh = State::empty(self.dim);
        fresh.base_len = n;
        fresh.live = n;
        fresh.by_id = ids
            .iter()
            .enumerate()
            .map(|(slot, id)| (id.clone(), slot as u32))
            .collect();
        fresh.ids = ids.into_iter().map(Some).collect();
        fresh.tokens = tokens;
        fresh.base = Some(Base {
            graph: Some(graph),
            rows,
            n,
        });
        {
            let mut log = self.log.lock();
            // Frames appended since the capture are the new delta; the rotated log keeps them.
            let mut tail = vec![0u8; (log.len - covered) as usize];
            log.file.seek(SeekFrom::Start(covered))?;
            log.file.read_exact(&mut tail)?;
            let applied = apply_frames(&mut fresh, &tail, self.dim);
            ensure!(applied == tail.len(), "vector log tail does not replay");
            fresh.mutations = fresh.ids.len() - n + fresh.base_dead;
            let mut rotated = log_header(self.dim, epoch);
            rotated.extend_from_slice(&tail);
            let path = self.dir.join("rows.log");
            write_atomic(&path, &rotated)?;
            sync_dir(&self.dir)?;
            log.file = OpenOptions::new().read(true).write(true).open(&path)?;
            log.len = rotated.len() as u64;
            log.epoch = epoch;
            let mut state = self.state.write();
            *state = fresh;
            *self.manifest.lock() = Some(manifest);
        }
        if epoch > 1 {
            for name in [
                format!("rows-{}.bin", epoch - 1),
                format!("ids-{}.bin", epoch - 1),
                format!("graph-{}.vertex", epoch - 1),
            ] {
                let _ = std::fs::remove_file(self.dir.join(name));
            }
        }
        Ok(())
    }
}

fn open_log(dir: &Path, dim: usize, epoch: u64) -> Result<(Log, Vec<u8>)> {
    let path = dir.join("rows.log");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let header_ok = bytes.len() >= LOG_HEADER as usize
        && &bytes[..8] == LOG_MAGIC
        && u32::from_le_bytes(bytes[8..12].try_into().expect("four bytes")) == dim as u32;
    if !header_ok {
        if !bytes.is_empty() {
            crate::log_event!(Warn, "vector_index_log_unreadable", "path" => path.display().to_string(), "bytes" => bytes.len());
            let aside = dir.join(format!("rows.log.unreadable-{}", std::process::id()));
            std::fs::rename(&path, &aside)?;
            file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)?;
        }
        let header = log_header(dim, epoch);
        file.write_all(&header)?;
        fsync_plain(&file)?;
        sync_dir(dir)?;
        return Ok((
            Log {
                file,
                len: LOG_HEADER,
                epoch,
            },
            header,
        ));
    }
    let log_epoch = u64::from_le_bytes(bytes[12..20].try_into().expect("eight bytes"));
    let len = bytes.len() as u64;
    Ok((
        Log {
            file,
            len,
            epoch: log_epoch,
        },
        bytes,
    ))
}

fn load_base(dir: &Path, m: &Manifest, dim: usize) -> Result<(Base, Vec<Arc<str>>, Vec<u64>)> {
    let n = usize::try_from(m.rows).context("row count")?;
    let rows_file = File::open(dir.join(format!("rows-{}.bin", m.epoch)))?;
    let expected = ROWS_HEADER as u64 + (n as u64) * (dim as u64) * 4;
    ensure!(
        rows_file.metadata()?.len() == expected,
        "rows file is not {expected} bytes"
    );
    // SAFETY: see `maintain`; the file is only ever replaced by rename.
    let rows = unsafe { memmap2::Mmap::map(&rows_file)? };
    if let Some(expected) = m.rows_crc {
        ensure!(
            crc32fast::hash(&rows) == expected,
            "rows file checksum mismatch"
        );
    }
    ensure!(
        &rows[..8] == ROWS_MAGIC
            && u64::from_le_bytes(rows[8..16].try_into().expect("eight bytes")) == n as u64
            && u64::from_le_bytes(rows[16..24].try_into().expect("eight bytes")) == dim as u64,
        "rows file header does not match the manifest"
    );
    let ids_bytes = std::fs::read(dir.join(format!("ids-{}.bin", m.epoch)))?;
    ensure!(
        ids_bytes.len() >= 20 && &ids_bytes[..8] == IDS_MAGIC,
        "ids file header"
    );
    let (body, crc) = ids_bytes.split_at(ids_bytes.len() - 4);
    ensure!(
        crc32fast::hash(body) == u32::from_le_bytes(crc.try_into().expect("four bytes")),
        "ids file checksum"
    );
    ensure!(
        u64::from_le_bytes(body[8..16].try_into().expect("eight bytes")) == n as u64,
        "ids file row count"
    );
    let mut ids = Vec::with_capacity(n);
    let mut tokens = Vec::with_capacity(n);
    let mut at = 16;
    for _ in 0..n {
        ensure!(body.len() >= at + 2, "ids file is short");
        let len = u16::from_le_bytes([body[at], body[at + 1]]) as usize;
        at += 2;
        ensure!(body.len() >= at + len + 8, "ids file is short");
        let id = std::str::from_utf8(&body[at..at + len]).context("ids file id")?;
        ids.push(Arc::<str>::from(id));
        at += len;
        tokens.push(u64::from_le_bytes(
            body[at..at + 8].try_into().expect("eight bytes"),
        ));
        at += 8;
    }
    ensure!(at == body.len(), "ids file has trailing bytes");
    let graph_path = dir.join(format!("graph-{}.vertex", m.epoch));
    let graph = match VGraph::open(&graph_path, Residency::Prefaulted) {
        Ok(graph) if graph.len() == n && graph.dim() == dim => Some(graph),
        Ok(_) => {
            crate::log_event!(Warn, "vector_index_graph_unreadable", "path" => graph_path.display().to_string(), "detail" => "does not match the manifest");
            None
        }
        Err(error) => {
            crate::log_event!(Warn, "vector_index_graph_unreadable", "path" => graph_path.display().to_string(), "detail" => error.to_string());
            None
        }
    };
    Ok((Base { graph, rows, n }, ids, tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalize_handles_extreme_finite_values() {
        let row = normalize(&[f64::MAX, f64::MAX]).unwrap();
        assert!((dot(&row, &row) - 1.0).abs() < 1e-6, "row={row:?}");
    }

    #[test]
    fn legacy_snapshot_gains_row_checksum_during_maintenance() {
        let dir = tempfile::tempdir().unwrap();
        let core = core(dir.path(), 8);
        let vdir = dir.path().join("vindex");
        let index = VectorIndex::open(&vdir, 8, &core).unwrap();
        ingest(&index, &core, 0, 2, 8);
        index.maintain(1, true).unwrap();
        index.close().unwrap();
        drop(index);
        let path = vdir.join("CURRENT");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest.as_object_mut().unwrap().remove("rows_crc");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let index = VectorIndex::open(&vdir, 8, &core).unwrap();
        assert!(index.covers(&core));
        index.maintain(1, false).unwrap();
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(manifest.rows_crc.is_some());
        assert_eq!(top(&index, 0, 8)[0], "d0");
    }

    #[test]
    fn append_marks_dirty_under_log_lock() {
        let dir = tempfile::tempdir().unwrap();
        let core = core(dir.path(), 8);
        let vdir = dir.path().join("vindex");
        let index = VectorIndex::open(&vdir, 8, &core).unwrap();
        ingest(&index, &core, 0, 2, 8);
        let row = normalize(&vector(2, 8)).unwrap();
        let token = index.next_token();
        let mut bytes = Vec::new();
        push_frame(&mut bytes, UPSERT, "d2", token, &row);
        std::thread::scope(|scope| {
            let log = index.append(&bytes).unwrap();
            let maintenance = scope.spawn(|| index.maintain(1, true));
            assert!(index.log.try_lock().is_none());
            index.state.write().upsert(Arc::from("d2"), token, &row);
            drop(log);
            maintenance.join().unwrap().unwrap();
        });
        core.memorize_documents(vec![writerslogic_warm::DocumentInput {
            id: "d2".into(),
            text: "_".into(),
            source_uri: None,
            version: Some(version_stamp(token)),
            metadata: None,
            chunk_words: None,
            overlap_words: None,
            store_text: Some(false),
            embeddings: Some(vec![vector(2, 8)]),
            title_weight: None,
        }])
        .unwrap();
        core.flush().unwrap();
        assert_eq!(index.live(), 3);
        index.close().unwrap();
        drop(index);
        let reopened = VectorIndex::open(&vdir, 8, &core).unwrap();
        assert_eq!(
            reopened.live(),
            3,
            "snapshot covered a frame absent from its captured state"
        );
    }

    #[test]
    fn corrupted_rows_file_disables_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let core = core(dir.path(), 8);
        let vdir = dir.path().join("vindex");
        let index = VectorIndex::open(&vdir, 8, &core).unwrap();
        ingest(&index, &core, 0, 2, 8);
        index.maintain(1, true).unwrap();
        index.close().unwrap();
        drop(index);
        let path = vdir.join("rows-1.bin");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[ROWS_HEADER..ROWS_HEADER + 8 * 4].fill(0);
        std::fs::write(path, bytes).unwrap();
        let reopened = VectorIndex::open(&vdir, 8, &core).unwrap();
        assert!(
            !reopened.covers(&core),
            "corrupt rows remain eligible for authoritative graph routing"
        );
    }

    #[test]
    fn search_fills_k_past_tombstones() {
        let dir = tempfile::tempdir().unwrap();
        let core = core(dir.path(), 8);
        let vdir = dir.path().join("vindex");
        let index = VectorIndex::open(&vdir, 8, &core).unwrap();
        ingest(&index, &core, 0, 20, 8);
        index.maintain(1, true).unwrap();
        let query = normalize(&vector(0, 8)).unwrap();
        let nearest = index.search(&query, 4, 64).unwrap();
        let ids: Vec<&str> = nearest.iter().map(|(id, _)| id.as_ref()).collect();
        for id in &ids {
            assert!(core.delete_document(id).unwrap());
        }
        index.remove(&ids).unwrap();
        assert_eq!(index.live(), 16);
        assert!(index.covers(&core));
        assert_eq!(
            index.search(&query, 1, 64).unwrap().len(),
            1,
            "four tombstones exhaust fixed overfetch"
        );
    }

    fn core(dir: &Path, dim: usize) -> Arc<WarmCore> {
        let config = writerslogic_warm::core::WarmConfig {
            embedding_space: Some(writerslogic_warm::EmbeddingSpace {
                model: "test".into(),
                revision: "1".into(),
                dimensions: dim,
                normalization: "l2".into(),
                metric: "cosine".into(),
            }),
            ..writerslogic_warm::core::WarmConfig::default()
        };
        Arc::new(
            WarmCore::new(
                256,
                Some(dir.join("store").display().to_string()),
                Some(config),
            )
            .unwrap(),
        )
    }

    /// Well-separated pseudo-random vectors: one mixed 64-bit hash per coordinate.
    fn vector(seed: usize, dim: usize) -> Vec<f64> {
        (0..dim)
            .map(|i| {
                let mut x = (seed as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
                x ^= x >> 31;
                x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x ^= x >> 29;
                ((x >> 40) as f64 / (1u64 << 24) as f64) - 0.5
            })
            .collect()
    }

    fn ingest(index: &VectorIndex, core: &WarmCore, from: usize, to: usize, dim: usize) {
        let staged: Vec<(String, u64, Vec<f32>)> = (from..to)
            .map(|i| {
                (
                    format!("d{i}"),
                    index.next_token(),
                    normalize(&vector(i, dim)).unwrap(),
                )
            })
            .collect();
        index.stage(&staged).unwrap();
        let inputs = staged
            .iter()
            .enumerate()
            .map(|(k, (id, token, _))| writerslogic_warm::DocumentInput {
                id: id.clone(),
                text: "_".into(),
                source_uri: None,
                version: Some(version_stamp(*token)),
                metadata: None,
                chunk_words: None,
                overlap_words: None,
                store_text: Some(false),
                embeddings: Some(vec![vector(from + k, dim)]),
                title_weight: None,
            })
            .collect();
        core.memorize_documents(inputs).unwrap();
    }

    fn top(index: &VectorIndex, seed: usize, dim: usize) -> Vec<String> {
        index
            .search(&normalize(&vector(seed, dim)).unwrap(), 3, 64)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id.to_string())
            .collect()
    }

    #[test]
    fn snapshot_reopens_without_a_rebuild_and_a_torn_snapshot_falls_back() {
        let dim = 8;
        let dir = tempfile::tempdir().unwrap();
        let vdir = dir.path().join("vindex");
        let core = core(dir.path(), dim);
        let index = VectorIndex::open(&vdir, dim, &core).unwrap();
        ingest(&index, &core, 0, 300, dim);
        assert!(!index.status().graph);
        index.maintain(1, true).unwrap();
        let status = index.status();
        assert!(status.graph && status.rows == 300 && status.mutations == 0);
        assert!(index.covers(&core));
        assert_eq!(top(&index, 7, dim)[0], "d7");
        // Rows after the snapshot are searchable and survive a clean close.
        ingest(&index, &core, 300, 310, dim);
        assert_eq!(top(&index, 305, dim)[0], "d305");
        index.close().unwrap();
        drop(index);
        core.flush().unwrap();

        let reopened = VectorIndex::open(&vdir, dim, &core).unwrap();
        let status = reopened.status();
        assert!(status.graph, "the snapshot opened");
        assert_eq!(
            (status.rows, reopened.live(), status.mutations),
            (300, 310, 10)
        );
        assert!(reopened.covers(&core));
        assert_eq!(top(&reopened, 7, dim)[0], "d7");
        assert_eq!(top(&reopened, 305, dim)[0], "d305");
        // A forced maintenance with nothing new since the delta is a rebuild over 310 rows;
        // one with no change at all is a no-op.
        reopened.maintain(1, true).unwrap();
        assert_eq!(reopened.status().rows, 310);
        let epoch_files = std::fs::read_dir(&vdir).unwrap().count();
        reopened.maintain(1, true).unwrap();
        assert_eq!(std::fs::read_dir(&vdir).unwrap().count(), epoch_files);
        reopened.close().unwrap();
        drop(reopened);

        // Truncate the graph file: open ignores the snapshot, answers from the log, and the
        // next maintenance writes a fresh one.
        let graph = std::fs::read_dir(&vdir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "vertex"))
            .unwrap();
        let bytes = std::fs::read(&graph).unwrap();
        std::fs::write(&graph, &bytes[..bytes.len() / 2]).unwrap();
        let fallback = VectorIndex::open(&vdir, dim, &core).unwrap();
        let status = fallback.status();
        assert!(!status.graph, "a torn graph is ignored");
        assert_eq!(fallback.live(), 310, "its rows still answer");
        assert!(!fallback.covers(&core));
        assert_eq!(top(&fallback, 7, dim)[0], "d7");
        fallback.maintain(1, true).unwrap();
        let status = fallback.status();
        assert!(status.graph && status.rows == 310, "rebuilt");
        assert!(fallback.covers(&core));
        assert_eq!(top(&fallback, 305, dim)[0], "d305");
    }

    #[test]
    fn torn_log_tail_is_cut_and_an_unclean_stop_reconciles_with_the_engine() {
        let dim = 8;
        let dir = tempfile::tempdir().unwrap();
        let vdir = dir.path().join("vindex");
        let core = core(dir.path(), dim);
        let index = VectorIndex::open(&vdir, dim, &core).unwrap();
        ingest(&index, &core, 0, 20, dim);
        // A frame the engine never committed (the request stopped after the index append).
        index
            .stage(&[(
                "ghost".into(),
                index.next_token(),
                normalize(&vector(99, dim)).unwrap(),
            )])
            .unwrap();
        // A frame whose document the engine holds at another version.
        let replaced = index.next_token();
        index
            .stage(&[("d3".into(), replaced, normalize(&vector(3, dim)).unwrap())])
            .unwrap();
        drop(index);
        // A torn tail: half a frame.
        let log = vdir.join("rows.log");
        let mut bytes = std::fs::read(&log).unwrap();
        let mut torn = Vec::new();
        push_frame(
            &mut torn,
            UPSERT,
            "torn",
            1,
            &normalize(&vector(5, dim)).unwrap(),
        );
        bytes.extend_from_slice(&torn[..torn.len() / 2]);
        std::fs::write(&log, &bytes).unwrap();
        assert!(vdir.join("dirty").exists(), "unclean stop");

        let reopened = VectorIndex::open(&vdir, dim, &core).unwrap();
        assert!(!vdir.join("dirty").exists(), "reconciled");
        assert_eq!(
            reopened.live(),
            19,
            "ghost and the re-versioned d3 left the index"
        );
        assert!(
            !reopened.covers(&core),
            "d3 is embedded in the engine but not indexed"
        );
        assert!(top(&reopened, 99, dim).iter().all(|id| id != "ghost"));
        assert!(top(&reopened, 3, dim).iter().all(|id| id != "d3"));
        let body = std::fs::read(&log).unwrap();
        let mut check = State::empty(dim);
        assert_eq!(
            apply_frames(&mut check, &body[LOG_HEADER as usize..], dim),
            body.len() - LOG_HEADER as usize,
            "no torn frame remains"
        );
    }
}
