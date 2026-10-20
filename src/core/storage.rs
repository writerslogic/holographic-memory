// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{anyhow, ensure, Context, Result};
use crc32fast::Hasher;
use memmap2::{Mmap, MmapMut};
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

mod staging;
pub(crate) use staging::StagedArena;

pub(crate) const FORMAT_MANIFEST: &str = "format.json";
pub(crate) const FORMAT_MAGIC: &str = "HMS_ARENA";
pub const STORAGE_FORMAT_VERSION: u32 = 4;

pub(crate) const SEGMENT_SIZE: usize = 64 * 1024 * 1024;
pub(crate) const LEGACY_SEGMENT_SIZE: usize = 1024 * 1024 * 1024;
// Windows requires unmapping all views before extending a mapped file.
const INCREMENTAL_ALLOCATION: bool = cfg!(unix);
/// Frame header: [CRC32: u32][RawLen: u32][CompLen: u32][Version: u32]
pub(crate) const HEADER_SIZE: usize = 16;
/// Maximum decompressed frame payload (50 MB). Used consistently in
/// `discover_offset` and `read_frame` to reject corrupt/malicious data.
pub(crate) const MAX_RAW_FRAME_SIZE: usize = 50 * 1024 * 1024;
/// Version word of the end-of-log marker: a zero-length header written after every frame in
/// the frame's own flush range, so the committed end of the log is recorded at no extra cost.
/// A frame that fails its checks with the marker (or a valid frame) after it was committed
/// and is corrupt; one with nothing after it is a crash-truncated append. A plain zero header
/// (stores written before the marker) still means end of log.
pub(crate) const END_MARK: u32 = 0x454E_444C;

/// True if the 16 bytes at `at` are an end-of-log marker or the legacy zero header.
#[inline]
pub(crate) fn is_log_end(header: &[u8]) -> bool {
    let word = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().expect("4 bytes"));
    header.len() >= HEADER_SIZE
        && word(4) == 0
        && word(8) == 0
        && (header[..HEADER_SIZE].iter().all(|&b| b == 0) || (word(0) == 0 && word(12) == END_MARK))
}

fn end_marker_bytes() -> [u8; HEADER_SIZE] {
    let mut bytes = [0u8; HEADER_SIZE];
    bytes[12..16].copy_from_slice(&END_MARK.to_le_bytes());
    bytes
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct FormatManifest {
    pub magic: String,
    pub version: u32,
    pub segment_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ArenaStats {
    pub format_version: u32,
    pub segment_count: usize,
    pub used_bytes: usize,
    pub capacity_bytes: usize,
}

/// RwLock-guarded segmented mmap arena with LZ4 compression and CRC32 framing.
/// Writers acquire a write lock; readers acquire a read lock.
/// Note: `version_counter` resets to 0 on restart (no WAL recovery).
/// Every entry is framed: [CRC32: u32][RawLen: u32][CompLen: u32][Version: u32][Data: bytes]
pub struct PersistentArena {
    segment_size: usize,
    format_version: u32,
    base_path: PathBuf,
    data_path: RwLock<PathBuf>,
    read_segments: RwLock<Vec<Arc<Mmap>>>,
    active_segment: Arc<RwLock<MmapMut>>,
    active_id: AtomicUsize,
    write_offset: AtomicUsize,
    active_capacity: AtomicUsize,
    version_counter: AtomicUsize,
}

pub(crate) struct PreparedCompaction {
    arena: PersistentArena,
}

impl PreparedCompaction {
    pub(crate) fn append_frame(&mut self, data: &[u8]) -> Result<()> {
        let offset = self.arena.write_frame(data, false)?;
        let (written, _) = self.arena.read_frame(offset)?;
        anyhow::ensure!(
            written == data,
            "compaction replay frame changed after writing"
        );
        Ok(())
    }
}

impl PersistentArena {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base)?;
        let manifest = Self::validate_or_create_manifest(&base)?;
        let data_path = super::durable_file::arena_generation(&base)?;
        if let Some(store) = base.parent() {
            super::durable_file::verify_manifests(store, &base)?;
        }

        let mut id = 0;
        loop {
            let p = data_path.join(format!("seg_{}.bin", id));
            if !p.exists() {
                break;
            }
            id += 1;
        }

        let active_id = if id > 0 { id - 1 } else { 0 };
        let mut segments = Vec::new();

        for i in 0..active_id {
            let p = data_path.join(format!("seg_{}.bin", i));
            let file = File::open(&p)?;
            let mmap = unsafe { Mmap::map(&file)? };
            segments.push(Arc::new(mmap));
        }

        let active_path = data_path.join(format!("seg_{}.bin", active_id));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&active_path)?;
        let active_capacity = if manifest.version >= 4 && INCREMENTAL_ALLOCATION {
            let length = file.metadata()?.len();
            ensure!(
                length <= manifest.segment_size as u64,
                "active segment exceeds its declared size"
            );
            usize::try_from(length)
                .context("arena capacity exceeds address space")?
                .max(SEGMENT_SIZE)
                .div_ceil(SEGMENT_SIZE)
                * SEGMENT_SIZE
        } else {
            manifest.segment_size
        };
        file.set_len(active_capacity as u64)?;
        file.sync_all()?;
        let mut mut_map = unsafe { MmapMut::map_mut(&file)? };

        let (recovered_offset, max_version) = Self::discover_offset(&mut_map)
            .with_context(|| format!("arena segment {}", active_path.display()))?;
        // A partially written tail may contain bytes from an older frame. Persist the end
        // marker at the recovered boundary so future restarts never interpret stale tail
        // bytes as a valid frame.
        if recovered_offset + HEADER_SIZE <= mut_map.len() {
            mut_map[recovered_offset..recovered_offset + HEADER_SIZE]
                .copy_from_slice(&end_marker_bytes());
            mut_map.flush_range(recovered_offset, HEADER_SIZE)?;
        }

        Ok(Self {
            segment_size: manifest.segment_size,
            format_version: manifest.version,
            base_path: base,
            data_path: RwLock::new(data_path),
            read_segments: RwLock::new(segments),
            active_segment: Arc::new(RwLock::new(mut_map)),
            active_id: AtomicUsize::new(active_id),
            write_offset: AtomicUsize::new(recovered_offset),
            active_capacity: AtomicUsize::new(active_capacity),
            version_counter: AtomicUsize::new(if recovered_offset > 0 {
                max_version as usize + 1
            } else {
                0
            }),
        })
    }

    fn validate_or_create_manifest(base: &Path) -> Result<FormatManifest> {
        let path = base.join(FORMAT_MANIFEST);
        let source = if path.exists() {
            path.clone()
        } else {
            super::durable_file::arena_generation(base)?.join(FORMAT_MANIFEST)
        };
        if source.exists() {
            let bytes = std::fs::read(&source)?;
            let manifest: FormatManifest = serde_json::from_slice(&bytes)
                .map_err(|e| anyhow!("Invalid HMS storage manifest {}: {}", source.display(), e))?;
            if manifest.magic != FORMAT_MAGIC {
                return Err(anyhow!(
                    "Storage manifest has unknown magic: {}",
                    manifest.magic
                ));
            }
            if manifest.version > STORAGE_FORMAT_VERSION {
                return Err(anyhow!(
                    "Storage format {} is newer than supported format {}",
                    manifest.version,
                    STORAGE_FORMAT_VERSION
                ));
            }
            let expected = if manifest.version == 3 {
                SEGMENT_SIZE
            } else {
                LEGACY_SEGMENT_SIZE
            };
            if manifest.segment_size != expected {
                return Err(anyhow!(
                    "Storage segment size mismatch: manifest={}, runtime={}",
                    manifest.segment_size,
                    expected
                ));
            }
            if source != path {
                super::durable_file::atomic_write(&path, &bytes)?;
            }
            return Ok(manifest);
        }

        // Pre-manifest stores are format v1. Creating the manifest is a backward-compatible
        // migration and does not rewrite their framed data.
        let legacy = source.with_file_name("seg_0.bin").exists();
        let manifest = FormatManifest {
            magic: FORMAT_MAGIC.to_string(),
            version: if legacy { 2 } else { STORAGE_FORMAT_VERSION },
            segment_size: LEGACY_SEGMENT_SIZE,
        };
        super::durable_file::atomic_write(&path, &serde_json::to_vec_pretty(&manifest)?)?;
        Ok(manifest)
    }

    pub fn stats(&self) -> ArenaStats {
        let segment_count = self.active_id.load(Ordering::Acquire) + 1;
        let used_bytes = self.active_id.load(Ordering::Acquire) * self.segment_size
            + self.write_offset.load(Ordering::Acquire);
        ArenaStats {
            format_version: self.format_version.max(2),
            segment_count,
            used_bytes,
            capacity_bytes: (segment_count - 1) * self.segment_size
                + self.active_capacity.load(Ordering::Acquire),
        }
    }

    /// Durably flush all pending writes in the active segment.
    pub fn flush(&self) -> Result<()> {
        self.active_segment
            .write()
            .flush()
            .map_err(|e| anyhow!("mmap flush failed: {}", e))
    }

    /// Walk CRC32-framed entries in the active segment to find the first free offset
    /// and the highest version number present. A frame that fails its checks ends the walk
    /// only when nothing valid follows it (a crash-truncated append); a bad frame with a
    /// valid frame after it is corruption inside committed data and an error.
    fn discover_offset(mmap: &MmapMut) -> Result<(usize, u32)> {
        let mut offset = 0;
        let mut max_version = 0;
        let data = &mmap[..];
        let fail = |offset: usize, max_version: u32| -> Result<(usize, u32)> {
            match Self::committed_after(data, offset + 1) {
                Some(next) => Err(anyhow!(
                    "frame at offset {offset} fails its checks but committed data follows at \
                     offset {next}: the segment is corrupt, not a torn tail; restore a \
                     snapshot or re-encode the sources"
                )),
                None => Ok((offset, max_version)),
            }
        };
        while offset + HEADER_SIZE <= data.len() {
            // Unwraps are safe: the loop guard ensures at least HEADER_SIZE (16) bytes
            // remain, so each 4-byte slice is within bounds.
            let raw_len = u32::from_le_bytes(
                data[offset + 4..offset + 8]
                    .try_into()
                    .expect("4-byte slice"),
            ) as usize;
            let comp_len = u32::from_le_bytes(
                data[offset + 8..offset + 12]
                    .try_into()
                    .expect("4-byte slice"),
            ) as usize;
            let version = u32::from_le_bytes(
                data[offset + 12..offset + 16]
                    .try_into()
                    .expect("4-byte slice"),
            );

            // A zero header or the end marker means no more frames.
            if raw_len == 0 && comp_len == 0 {
                if is_log_end(&data[offset..offset + HEADER_SIZE]) {
                    break;
                }
                return fail(offset, max_version);
            }
            // Sanity: reject impossibly large or zero-progress frames
            if comp_len > mmap.len() || raw_len > MAX_RAW_FRAME_SIZE || comp_len == 0 {
                return fail(offset, max_version);
            }

            let frame_size = HEADER_SIZE + comp_len;
            if offset + frame_size > data.len() {
                return fail(offset, max_version);
            }

            if !Self::frame_checks_out(data, offset) {
                return fail(offset, max_version);
            }

            if version > max_version {
                max_version = version;
            }
            offset += frame_size;
        }
        Ok((offset, max_version))
    }

    /// True if the frame at `offset` has plausible lengths, decompresses and matches its CRC32.
    fn frame_checks_out(data: &[u8], offset: usize) -> bool {
        if offset + HEADER_SIZE > data.len() {
            return false;
        }
        let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().expect("4 bytes"));
        let (expected_crc, raw_len, comp_len) = (
            word(offset),
            word(offset + 4) as usize,
            word(offset + 8) as usize,
        );
        if comp_len == 0 || raw_len == 0 || raw_len > MAX_RAW_FRAME_SIZE || comp_len > raw_len {
            return false;
        }
        let end = match offset.checked_add(HEADER_SIZE + comp_len) {
            Some(end) if end <= data.len() => end,
            _ => return false,
        };
        let payload = &data[offset + HEADER_SIZE..end];
        let decompressed = if comp_len < raw_len {
            match lz4_flex::decompress(payload, raw_len) {
                Ok(d) => d,
                Err(_) => return false,
            }
        } else {
            payload.to_vec()
        };
        let mut hasher = Hasher::new();
        hasher.update(&decompressed);
        hasher.finalize() == expected_crc
    }

    /// The first offset at or after `from` holding committed data: a valid frame or the end
    /// marker, scanning every byte up to the last non-zero byte of the segment (bytes beyond
    /// the write frontier are zero).
    fn committed_after(data: &[u8], from: usize) -> Option<usize> {
        let last = data.iter().rposition(|&b| b != 0)?;
        (from..=last.saturating_sub(HEADER_SIZE - 1)).find(|&at| {
            at + HEADER_SIZE <= data.len()
                && (Self::frame_checks_out(data, at)
                    || (is_log_end(&data[at..at + HEADER_SIZE])
                        && data[at + 12..at + 16] == END_MARK.to_le_bytes()))
        })
    }

    pub fn read_slice(&self, global_offset: usize, len: usize) -> Result<Vec<u8>> {
        // Rotation and publication take active before segments; retain both to select one mapping.
        let active = self.active_segment.read();
        let reader = self.read_segments.read();
        let seg_idx = global_offset / self.segment_size;
        if seg_idx > self.active_id.load(Ordering::Acquire) {
            return Err(anyhow!("read beyond final arena segment"));
        }
        let local_offset = global_offset % self.segment_size;
        let end = local_offset
            .checked_add(len)
            .ok_or_else(|| anyhow!("arena read length overflow"))?;
        if end > self.segment_size {
            return Err(anyhow!("read exceeds arena segment"));
        }
        if seg_idx < reader.len() {
            let seg = &reader[seg_idx];
            if end <= seg.len() {
                return Ok(seg[local_offset..end].to_vec());
            }
            return Err(anyhow!("read exceeds arena segment"));
        }
        if end <= active.len() {
            Ok(active[local_offset..end].to_vec())
        } else {
            Err(anyhow!(
                "Read out of bounds: offset {}, len {}",
                global_offset,
                len
            ))
        }
    }

    pub fn read_frame(&self, global_offset: usize) -> Result<(Vec<u8>, u32)> {
        let header_bytes = self.read_slice(global_offset, HEADER_SIZE)?;
        if header_bytes.len() < HEADER_SIZE {
            return Err(anyhow!("Truncated header at offset {}", global_offset));
        }

        let expected_crc = u32::from_le_bytes(header_bytes[0..4].try_into().expect("4-byte slice"));
        let raw_len =
            u32::from_le_bytes(header_bytes[4..8].try_into().expect("4-byte slice")) as usize;
        let comp_len =
            u32::from_le_bytes(header_bytes[8..12].try_into().expect("4-byte slice")) as usize;
        let version = u32::from_le_bytes(header_bytes[12..16].try_into().expect("4-byte slice"));

        if raw_len == 0 {
            return Err(anyhow!("Empty frame at offset {}", global_offset));
        }
        if raw_len > MAX_RAW_FRAME_SIZE {
            return Err(anyhow!(
                "Suspiciously large raw frame length: {} at {}",
                raw_len,
                global_offset
            ));
        }
        // Compressed data cannot exceed a full segment; reject to prevent OOM.
        if comp_len > self.segment_size {
            return Err(anyhow!(
                "Suspiciously large compressed frame length: {} at {}",
                comp_len,
                global_offset
            ));
        }

        let payload = self.read_slice(global_offset + HEADER_SIZE, comp_len)?;
        if payload.len() < comp_len {
            return Err(anyhow!(
                "Truncated payload at offset {}",
                global_offset + HEADER_SIZE
            ));
        }

        let data = if comp_len < raw_len {
            lz4_flex::decompress(&payload, raw_len)
                .map_err(|e| anyhow!("LZ4 decompression failed: {}", e))?
        } else {
            payload
        };

        let mut hasher = Hasher::new();
        hasher.update(&data);
        let actual_crc = hasher.finalize();

        if actual_crc != expected_crc {
            return Err(anyhow!(
                "CRC32 mismatch at offset {}: expected {:x}, got {:x}",
                global_offset,
                expected_crc,
                actual_crc
            ));
        }

        Ok((data, version))
    }

    pub fn next_offset(&self, global_offset: usize) -> Result<usize> {
        let header_bytes = self.read_slice(global_offset, HEADER_SIZE)?;
        if header_bytes.len() < HEADER_SIZE {
            return Err(anyhow!("Truncated header at offset {}", global_offset));
        }

        let comp_len =
            u32::from_le_bytes(header_bytes[8..12].try_into().expect("4-byte slice")) as usize;
        if comp_len == 0 {
            return Err(anyhow!("Empty frame at offset {}", global_offset));
        }
        let next = global_offset + HEADER_SIZE + comp_len;
        let segment = global_offset / self.segment_size;
        if segment < self.active_id.load(Ordering::Acquire) {
            if next % self.segment_size + HEADER_SIZE > self.segment_size {
                return Ok((segment + 1) * self.segment_size);
            }
            let header = self.read_slice(next, HEADER_SIZE)?;
            if is_log_end(&header) {
                return Ok((segment + 1) * self.segment_size);
            }
        }
        Ok(next)
    }

    pub fn write_slice(&self, data: &[u8]) -> Result<usize> {
        self.write_frame(data, true)
    }

    fn write_frame(&self, data: &[u8], durable: bool) -> Result<usize> {
        let raw_len = data.len();
        if raw_len == 0 || raw_len > MAX_RAW_FRAME_SIZE {
            return Err(anyhow!(
                "arena payload must contain 1..={MAX_RAW_FRAME_SIZE} bytes"
            ));
        }
        let version = self.version_counter.fetch_add(1, Ordering::SeqCst) as u32;

        let mut hasher = Hasher::new();
        hasher.update(data);
        let crc = hasher.finalize();

        let compressed = lz4_flex::compress(data);
        let (write_data, comp_len) = if compressed.len() < raw_len {
            let clen = compressed.len();
            (compressed, clen)
        } else {
            (data.to_vec(), raw_len)
        };

        let total_frame_len = HEADER_SIZE + comp_len;

        // Guard against raw_len truncation when cast to u32 in the header.
        // This is defensive; MAX_RAW_FRAME_SIZE (50 MB) << u32::MAX (4 GB).
        if raw_len > u32::MAX as usize || comp_len > u32::MAX as usize {
            return Err(anyhow!(
                "Frame payload exceeds u32 header capacity: raw={}, comp={}",
                raw_len,
                comp_len
            ));
        }

        // Acquire write lock for the entire reserve+write sequence.
        // This prevents data races from concurrent writers sharing the
        // same mmap and eliminates the TOCTOU race on segment rotation.
        let mut active = self.active_segment.write();

        // Check-then-reserve under write lock to avoid wasting space at segment boundaries.
        let mut offset = self.write_offset.load(Ordering::Acquire);
        if offset + total_frame_len > self.segment_size {
            self.rotate_segment_locked(&mut active)?;
            offset = 0;
        }
        if offset + total_frame_len > active.len() {
            self.grow_segment_locked(&mut active, offset + total_frame_len)?;
        }
        self.write_offset
            .store(offset + total_frame_len, Ordering::Release);

        // SAFETY: we hold the exclusive write lock, so no concurrent writers.
        // Growth and rotation ensure offset + total_frame_len fits the guarded mapping.
        // Each copy_nonoverlapping targets a distinct, non-overlapping range:
        //   [offset..+4], [offset+4..+8], [offset+8..+12], [offset+12..+16], [offset+16..+16+comp_len]
        debug_assert!(offset + total_frame_len <= self.segment_size);
        debug_assert!(offset + total_frame_len <= active.len());
        debug_assert!(total_frame_len == HEADER_SIZE + comp_len);
        unsafe {
            let ptr = active.as_ptr() as *mut u8;
            let target = ptr.add(offset);
            // CRC32 at +0
            std::ptr::copy_nonoverlapping(crc.to_le_bytes().as_ptr(), target, 4);
            // RawLen at +4
            std::ptr::copy_nonoverlapping(
                (raw_len as u32).to_le_bytes().as_ptr(),
                target.add(4),
                4,
            );
            // CompLen at +8
            std::ptr::copy_nonoverlapping(
                (comp_len as u32).to_le_bytes().as_ptr(),
                target.add(8),
                4,
            );
            // Version at +12
            std::ptr::copy_nonoverlapping(version.to_le_bytes().as_ptr(), target.add(12), 4);
            // Data at +HEADER_SIZE
            std::ptr::copy_nonoverlapping(write_data.as_ptr(), target.add(HEADER_SIZE), comp_len);
        }
        let frame_end = offset + total_frame_len;
        // The end marker rides in the frame's flush range: it records the committed end.
        let sentinel_len = if frame_end + HEADER_SIZE <= active.len() {
            active[frame_end..frame_end + HEADER_SIZE].copy_from_slice(&end_marker_bytes());
            HEADER_SIZE
        } else {
            0
        };
        if durable {
            active
                .flush_range(offset, total_frame_len + sentinel_len)
                .map_err(|e| anyhow!("mmap flush_range failed: {}", e))?;
        }
        let global = self.active_id.load(Ordering::SeqCst) * self.segment_size + offset;
        Ok(global)
    }

    fn grow_segment_locked(
        &self,
        active: &mut parking_lot::RwLockWriteGuard<'_, MmapMut>,
        required: usize,
    ) -> Result<()> {
        ensure!(
            self.format_version >= 4
                && INCREMENTAL_ALLOCATION
                && required <= self.segment_size
                && required > active.len(),
            "invalid arena growth request"
        );
        let capacity = required.div_ceil(SEGMENT_SIZE) * SEGMENT_SIZE;
        let path = self.data_path.read().join(format!(
            "seg_{}.bin",
            self.active_id.load(Ordering::Acquire)
        ));
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        active.flush()?;
        file.set_len(capacity as u64)?;
        file.sync_all()?;
        // SAFETY: the writer guard excludes all readers and the file has its new durable length.
        let mapping = unsafe { MmapMut::map_mut(&file)? };
        **active = mapping;
        self.active_capacity.store(capacity, Ordering::Release);
        Ok(())
    }

    /// Rotate to a new segment. Caller must hold the write lock on active_segment
    /// and pass the guard so we can swap the mmap in place.
    fn rotate_segment_locked(
        &self,
        active: &mut parking_lot::RwLockWriteGuard<'_, MmapMut>,
    ) -> Result<()> {
        let current_id = self.active_id.load(Ordering::SeqCst);
        let path = self
            .data_path
            .read()
            .join(format!("seg_{}.bin", current_id));
        let file = File::open(&path)?;
        active.flush()?;
        let mmap = unsafe { Mmap::map(&file)? };
        self.read_segments.write().push(Arc::new(mmap));

        let next_id = current_id + 1;
        let next_path = self.data_path.read().join(format!("seg_{}.bin", next_id));
        let next_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&next_path)?;
        let capacity = if self.format_version >= 4 && INCREMENTAL_ALLOCATION {
            SEGMENT_SIZE
        } else {
            self.segment_size
        };
        next_file.set_len(capacity as u64)?;
        next_file.sync_all()?;
        super::durable_file::sync_directory(&self.data_path.read())?;
        let next_map = unsafe { MmapMut::map_mut(&next_file)? };

        **active = next_map;
        self.active_capacity.store(capacity, Ordering::Release);
        self.active_id.store(next_id, Ordering::SeqCst);
        self.write_offset.store(0, Ordering::SeqCst);
        Ok(())
    }

    pub(crate) fn replay_compaction_tail(
        &self,
        prepared: &mut PreparedCompaction,
        cursor: &mut usize,
        end: usize,
    ) -> Result<()> {
        ensure!(
            *cursor <= end,
            "compaction journal checkpoint moved backwards"
        );
        while *cursor < end {
            let segment_end = (*cursor / self.segment_size)
                .checked_add(1)
                .and_then(|segment| segment.checked_mul(self.segment_size))
                .context("compaction segment boundary overflow")?;
            let header_end = cursor
                .checked_add(HEADER_SIZE)
                .context("compaction frame header overflow")?;
            if header_end > segment_end {
                ensure!(segment_end <= end, "truncated compaction replay header");
                *cursor = segment_end;
                continue;
            }
            ensure!(header_end <= end, "truncated compaction replay header");
            let header = self.read_slice(*cursor, HEADER_SIZE)?;
            if is_log_end(&header) {
                ensure!(segment_end <= end, "empty compaction replay frame");
                *cursor = segment_end;
                continue;
            }
            let raw = u32::from_le_bytes(header[4..8].try_into()?) as usize;
            let compressed = u32::from_le_bytes(header[8..12].try_into()?) as usize;
            ensure!(
                (1..=MAX_RAW_FRAME_SIZE).contains(&raw) && (1..=raw).contains(&compressed),
                "invalid compaction replay frame lengths"
            );
            let next = header_end
                .checked_add(compressed)
                .context("compaction replay frame length overflow")?;
            ensure!(
                next <= end && next <= segment_end,
                "compaction replay frame exceeds its captured boundary"
            );
            let (data, _) = self
                .read_frame(*cursor)
                .with_context(|| format!("cannot replay compaction frame at {}", *cursor))?;
            prepared.append_frame(&data)?;
            *cursor = next;
        }
        Ok(())
    }

    pub(crate) fn prepare_compacted(self) -> Result<PreparedCompaction> {
        let staged = self;
        let temp_base = staged.base_path.clone();
        for i in 0..staged.stats().segment_count {
            let path = temp_base.join(format!("seg_{i}.bin"));
            let file = File::open(&path)?;
            // SAFETY: the staged arena has no concurrent writer.
            let map = unsafe { Mmap::map(&file)? };
            let (_, bytes) = super::admin::inspect_segment(&map, &path, true)?;
            if i == staged.active_id.load(Ordering::Acquire) {
                anyhow::ensure!(
                    bytes == staged.write_offset.load(Ordering::Acquire),
                    "staged arena length changed before publication"
                );
            }
        }
        staged.flush()?;
        super::durable_file::sync_directory(&temp_base)?;
        Ok(PreparedCompaction { arena: staged })
    }

    pub(crate) fn replace_with_compacted(&self, prepared: PreparedCompaction) -> Result<()> {
        let staged = prepared.arena;
        ensure!(
            staged.segment_size == self.segment_size
                && staged.format_version == self.format_version,
            "compaction storage geometry differs from the live arena"
        );
        staged.flush()?;
        let temp_base = staged.base_path.clone();
        super::durable_file::sync_directory(&temp_base)?;
        let generation = format!(
            "gen-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        let new_path = self.base_path.join(&generation);
        std::fs::rename(&temp_base, &new_path)?;
        super::durable_file::sync_directory(&self.base_path)?;
        let replacement = staged;
        #[cfg(test)]
        compaction_crash_point("staged");
        let mut active = self.active_segment.write();
        let mut segments = self.read_segments.write();
        let mut data_path = self.data_path.write();
        active.flush()?;
        // IMPORTANT: old mappings/files stay valid until the durable pointer commits.
        let publication = super::durable_file::atomic_write(
            &self.base_path.join("CURRENT"),
            generation.as_bytes(),
        );
        if publication.is_err()
            && super::durable_file::arena_generation(&self.base_path)? != new_path
        {
            return publication;
        }
        #[cfg(test)]
        compaction_crash_point("published");
        std::mem::swap(&mut *active, &mut *replacement.active_segment.write());
        std::mem::swap(&mut *segments, &mut *replacement.read_segments.write());
        let old_path = std::mem::replace(&mut *data_path, new_path);
        self.active_capacity.store(
            replacement.active_capacity.load(Ordering::Acquire),
            Ordering::Release,
        );
        self.active_id.store(
            replacement.active_id.load(Ordering::Acquire),
            Ordering::Release,
        );
        self.write_offset.store(
            replacement.write_offset.load(Ordering::Acquire),
            Ordering::Release,
        );
        self.version_counter.store(
            replacement.version_counter.load(Ordering::Acquire),
            Ordering::Release,
        );
        drop(replacement);
        #[cfg(test)]
        compaction_crash_point("mapped");
        publication?;
        if old_path != self.base_path {
            if let Err(error) = std::fs::remove_dir_all(old_path) {
                tracing::warn!(%error, "old arena generation could not be removed");
            }
        } else {
            for entry in std::fs::read_dir(&self.base_path)? {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("seg_") && name.ends_with(".bin") {
                    if let Err(error) = std::fs::remove_file(entry.path()) {
                        tracing::warn!(%error, "legacy arena segment cleanup failed");
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn checkpoint(&self) -> (String, usize) {
        (
            self.data_path
                .read()
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            self.stats().used_bytes,
        )
    }
}

impl Drop for PersistentArena {
    fn drop(&mut self) {
        let _ = self.active_segment.write().flush();
    }
}

#[cfg(test)]
fn compaction_crash_point(phase: &str) {
    if std::env::var("HMS_TEST_COMPACTION_PHASE").as_deref() == Ok(phase) {
        std::process::exit(73);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn reads_remain_on_the_requested_segment_during_rotation() -> Result<()> {
        let root = tempdir()?;
        let arena = Arc::new(PersistentArena::new(root.path())?);
        let old = arena.write_slice(b"original segment")?;
        let mut active = arena.active_segment.write();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let reader_arena = Arc::clone(&arena);
        let reader = std::thread::spawn(move || -> Result<()> {
            started_tx.send(())?;
            for _ in 0..128 {
                anyhow::ensure!(reader_arena.read_frame(old)?.0 == b"original segment");
            }
            Ok(())
        });
        started_rx.recv()?;
        arena.rotate_segment_locked(&mut active)?;
        drop(active);
        let new = arena.write_slice(b"replacement segment")?;
        reader.join().unwrap()?;
        assert_eq!(arena.read_frame(new)?.0, b"replacement segment");
        for length in [1, 2, 3] {
            assert_eq!(
                arena.read_slice(new + SEGMENT_SIZE - 2, length).is_ok(),
                length <= 2
            );
        }
        assert!(arena.read_slice(new + 1, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn compaction_crash_child() -> Result<()> {
        let Ok(root) = std::env::var("HMS_TEST_COMPACTION_ROOT") else {
            return Ok(());
        };
        let root = PathBuf::from(root);
        let arena = PersistentArena::new(root.join("arena"))?;
        arena.write_slice(b"old generation")?;
        let staged = PersistentArena::new(root.join("stage"))?;
        staged.write_slice(b"new generation")?;
        arena.replace_with_compacted(staged.prepare_compacted()?)?;
        Ok(())
    }

    #[test]
    fn compaction_tail_replay_crosses_rotated_padding_without_splitting_frames() -> Result<()> {
        let root = tempdir()?;
        let arena = PersistentArena::new(root.path().join("live"))?;
        arena.write_slice(b"original segment")?;
        let mut cursor = arena.stats().used_bytes;
        {
            let mut active = arena.active_segment.write();
            arena.rotate_segment_locked(&mut active)?;
        }
        arena.write_slice(b"first replayed transaction")?;
        arena.write_slice(b"second replayed transaction")?;
        let end = arena.stats().used_bytes;
        let staged = PersistentArena::new(root.path().join("stage"))?;
        staged.write_slice(b"original segment")?;
        let mut prepared = staged.prepare_compacted()?;
        arena.replay_compaction_tail(&mut prepared, &mut cursor, end)?;
        assert_eq!(cursor, end);
        arena.replace_with_compacted(prepared)?;
        drop(arena);
        let restored = PersistentArena::new(root.path().join("live"))?;
        let mut offset = 0;
        for payload in [
            b"original segment".as_slice(),
            b"first replayed transaction",
            b"second replayed transaction",
        ] {
            assert_eq!(restored.read_frame(offset)?.0, payload);
            offset = restored.next_offset(offset)?;
        }
        assert_eq!(offset, restored.stats().used_bytes);
        Ok(())
    }

    #[test]
    fn compaction_replay_rejects_payload_length_mismatch_despite_valid_crc() -> Result<()> {
        for delta in [-1_isize, 0, 1] {
            let root = tempdir()?;
            let arena = PersistentArena::new(root.path().join("live"))?;
            arena.write_slice(b"distinct replay payload")?;
            let end = arena.stats().used_bytes;
            {
                let mut map = arena.active_segment.write();
                let raw = u32::from_le_bytes(map[4..8].try_into()?);
                let compressed = u32::from_le_bytes(map[8..12].try_into()?);
                assert_eq!(raw, compressed);
                let declared = (raw as usize).checked_add_signed(delta).unwrap() as u32;
                map[4..8].copy_from_slice(&declared.to_le_bytes());
                map.flush()?;
            }
            let staged = PersistentArena::new(root.path().join("stage"))?;
            let mut prepared = staged.prepare_compacted()?;
            let mut cursor = 0;
            let result = arena.replay_compaction_tail(&mut prepared, &mut cursor, end);
            assert_eq!(result.is_ok(), delta == 0);
            assert!(!root.path().join("live/CURRENT").exists());
        }
        Ok(())
    }

    #[test]
    fn compaction_replay_enforces_raw_frame_limit_boundaries() -> Result<()> {
        for length in [
            MAX_RAW_FRAME_SIZE - 1,
            MAX_RAW_FRAME_SIZE,
            MAX_RAW_FRAME_SIZE + 1,
        ] {
            let root = tempdir()?;
            let arena = PersistentArena::new(root.path().join("live"))?;
            let payload = vec![7; length.min(MAX_RAW_FRAME_SIZE)];
            arena.write_slice(&payload)?;
            if length > MAX_RAW_FRAME_SIZE {
                let mut map = arena.active_segment.write();
                map[4..8].copy_from_slice(&(length as u32).to_le_bytes());
                map.flush()?;
            }
            let staged = PersistentArena::new(root.path().join("stage"))?;
            let mut prepared = staged.prepare_compacted()?;
            let mut cursor = 0;
            let result =
                arena.replay_compaction_tail(&mut prepared, &mut cursor, arena.stats().used_bytes);
            if length <= MAX_RAW_FRAME_SIZE {
                result?;
                assert_eq!(prepared.arena.read_frame(0)?.0, payload);
            } else {
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "invalid compaction replay frame lengths"
                );
                assert_eq!(prepared.arena.stats().used_bytes, 0);
            }
        }
        Ok(())
    }

    #[test]
    fn interrupted_compaction_recovers_a_complete_generation() -> Result<()> {
        for phase in ["staged", "published", "mapped"] {
            let root = tempdir()?;
            let output = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "core::storage::tests::compaction_crash_child"])
                .env("HMS_TEST_COMPACTION_ROOT", root.path())
                .env("HMS_TEST_COMPACTION_PHASE", phase)
                .output()?;
            assert_eq!(output.status.code(), Some(73));
            let arena = PersistentArena::new(root.path().join("arena"))?;
            let expected: &[u8] = if phase == "staged" {
                b"old generation"
            } else {
                b"new generation"
            };
            assert_eq!(arena.read_frame(0)?.0, expected);
            arena.write_slice(b"after recovery")?;
        }
        Ok(())
    }

    #[test]
    fn compaction_rejects_corrupt_or_shortened_staging_without_publishing() -> Result<()> {
        for frame in 0..3 {
            for truncate in [false, true] {
                let root = tempdir()?;
                let arena = PersistentArena::new(root.path().join("live"))?;
                arena.write_slice(b"acknowledged generation")?;
                let staged = PersistentArena::new(root.path().join("stage"))?;
                let mut offsets = Vec::new();
                for payload in [b"first".as_slice(), b"middle", b"last"] {
                    offsets.push(staged.write_slice(payload)?);
                }
                {
                    let mut map = staged.active_segment.write();
                    let offset = offsets[frame];
                    if truncate {
                        map[offset..offset + HEADER_SIZE].fill(0);
                    } else {
                        map[offset + HEADER_SIZE] ^= 1;
                    }
                    map.flush()?;
                }
                assert!(staged
                    .prepare_compacted()
                    .and_then(|prepared| arena.replace_with_compacted(prepared))
                    .is_err());
                assert_eq!(arena.read_frame(0)?.0, b"acknowledged generation");
                drop(arena);
                let restored = PersistentArena::new(root.path().join("live"))?;
                assert_eq!(restored.read_frame(0)?.0, b"acknowledged generation");
            }
        }
        Ok(())
    }

    #[test]
    fn test_persistent_arena_persistence() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().to_path_buf();

        // 1. Create and write some data
        {
            let arena = PersistentArena::new(&path)?;
            let offset1 = arena.write_slice(b"hello world")?;
            let _offset2 = arena.write_slice(b"foo bar")?;
            assert_eq!(offset1, 0);
        }

        // 2. Re-open and verify data + no new segments
        {
            let arena = PersistentArena::new(&path)?;
            let (data1, version1) = arena.read_frame(0)?;
            assert_eq!(data1, b"hello world");
            assert_eq!(version1, 0);

            // Check that we only have seg_0.bin
            let mut count = 0;
            for entry in std::fs::read_dir(&path)? {
                let entry = entry?;
                if entry.path().extension().is_some_and(|ext| ext == "bin") {
                    count += 1;
                }
            }
            assert_eq!(count, 1);

            // 3. Write more and verify version continuity
            let offset3 = arena.write_slice(b"baz")?;
            let (data3, version3) = arena.read_frame(offset3)?;
            assert_eq!(data3, b"baz");
            assert_eq!(version3, 2); // 0 (hello world), 1 (foo bar), 2 (baz)
        }

        Ok(())
    }

    #[test]
    fn rejects_future_storage_format() -> Result<()> {
        let dir = tempdir()?;
        let manifest = FormatManifest {
            magic: FORMAT_MAGIC.to_string(),
            version: STORAGE_FORMAT_VERSION + 1,
            segment_size: SEGMENT_SIZE,
        };
        std::fs::write(
            dir.path().join(FORMAT_MANIFEST),
            serde_json::to_vec(&manifest)?,
        )?;
        let error = PersistentArena::new(dir.path()).err().expect("must reject");
        assert!(error.to_string().contains("newer than supported"));
        Ok(())
    }

    #[test]
    fn recovers_from_a_torn_tail_and_reuses_its_offset() -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};

        let dir = tempdir()?;
        let second_offset;
        {
            let arena = PersistentArena::new(dir.path())?;
            arena.write_slice(b"valid frame")?;
            second_offset = arena.write_slice(b"frame that will be torn")?;
        }

        // A crash-truncated append: the frame's payload stops short and the end marker
        // that would have followed it was never written (the file is zero beyond it).
        let segment = dir.path().join("seg_0.bin");
        let mut file = OpenOptions::new().write(true).open(segment)?;
        file.seek(SeekFrom::Start((second_offset + HEADER_SIZE + 4) as u64))?;
        file.write_all(&[0u8; 64])?;
        file.sync_all()?;

        {
            let arena = PersistentArena::new(dir.path())?;
            assert_eq!(arena.stats().used_bytes, second_offset);
            let replacement = arena.write_slice(b"replacement")?;
            assert_eq!(replacement, second_offset);
        }
        let arena = PersistentArena::new(dir.path())?;
        assert_eq!(arena.read_frame(second_offset)?.0, b"replacement");
        Ok(())
    }

    /// A flipped byte inside a committed frame is corruption, not a torn tail: committed data
    /// (the next frame, or the end marker after the last frame) follows it, so open fails
    /// instead of serving the log truncated at the damage.
    #[test]
    fn open_fails_on_corruption_inside_committed_data() -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        for damage_last in [false, true] {
            let dir = tempdir()?;
            let (first, last);
            {
                let arena = PersistentArena::new(dir.path())?;
                first = arena.write_slice(b"first committed frame")?;
                arena.write_slice(b"second committed frame")?;
                last = arena.write_slice(b"third committed frame")?;
            }
            let segment = dir.path().join("seg_0.bin");
            let mut file = OpenOptions::new().write(true).open(&segment)?;
            let target = if damage_last { last } else { first };
            file.seek(SeekFrom::Start((target + HEADER_SIZE + 2) as u64))?;
            file.write_all(&[0xff])?;
            file.sync_all()?;
            let error = match PersistentArena::new(dir.path()) {
                Ok(_) => panic!("corrupt frame at {target} opened (damage_last {damage_last})"),
                Err(e) => format!("{e:#}"),
            };
            assert!(error.contains("corrupt"), "{error}");
            // The damage is left in place for diagnosis: no marker was written over it.
            let bytes = std::fs::read(&segment)?;
            assert_eq!(bytes[target + HEADER_SIZE + 2], 0xff);
        }
        Ok(())
    }

    /// A manifest whose sidecar checksum does not match its bytes fails open; manifests
    /// without a sidecar (older stores) still open.
    #[test]
    fn manifest_checksum_mismatch_fails_open() -> Result<()> {
        let store = tempdir()?;
        let arena_dir = store.path().join("vectors_data.bin");
        {
            let arena = PersistentArena::new(&arena_dir)?;
            arena.write_slice(b"frame")?;
        }
        let schema = store.path().join("store.json");
        super::super::durable_file::atomic_write(&schema, br#"{"version":2,"dimensions":1024}"#)?;
        assert!(store.path().join("store.json.crc32").exists());
        PersistentArena::new(&arena_dir)?;
        std::fs::write(&schema, br#"{"version":2,"dimensions":1025}"#)?;
        let error = match PersistentArena::new(&arena_dir) {
            Ok(_) => panic!("flipped manifest opened"),
            Err(e) => format!("{e:#}"),
        };
        assert!(error.contains("checksum"), "{error}");
        std::fs::remove_file(store.path().join("store.json.crc32"))?;
        PersistentArena::new(&arena_dir)?;
        Ok(())
    }
}

#[cfg(test)]
mod layout_tests;

#[cfg(all(test, unix))]
mod growth_tests;
