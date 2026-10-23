// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::sync::RwLock;
use anyhow::{anyhow, ensure, Context, Result};
use crc32fast::Hasher;
use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(test)]
#[path = "storage/recovery_tests.rs"]
mod recovery_tests;
mod staging;
pub(crate) use staging::StagedArena;

pub(crate) const FORMAT_MANIFEST: &str = "format.json";
pub(crate) const FORMAT_MAGIC: &str = "HMS_ARENA";
pub(crate) const STORAGE_FORMAT_VERSION: u32 = 4;

/// The fixed segment size of format 3, and the step by which a format-4 active segment's
/// file grows toward its declared size.
pub(crate) const SEGMENT_SIZE: usize = 64 * 1024 * 1024;
/// The segment size formats 1, 2 and 4 declare in their manifest (format 4 allocates it in
/// [`SEGMENT_SIZE`] steps); only format 3 declared [`SEGMENT_SIZE`].
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

/// The header words `[crc, raw_len, comp_len, version]` of the frame at the start of `bytes`,
/// or `None` if fewer than [`HEADER_SIZE`] bytes remain.
#[inline]
#[must_use]
fn header_words(bytes: &[u8]) -> Option<[u32; 4]> {
    let (words, _) = bytes.first_chunk::<HEADER_SIZE>()?.as_chunks::<4>();
    Some(std::array::from_fn(|i| u32::from_le_bytes(words[i])))
}

/// True if `header` starts with an end-of-log marker or the legacy zero header.
#[inline]
#[must_use]
pub(crate) fn is_log_end(header: &[u8]) -> bool {
    header_words(header).is_some_and(|[crc, raw_len, comp_len, version]| {
        crc == 0 && raw_len == 0 && comp_len == 0 && (version == 0 || version == END_MARK)
    })
}

fn end_marker_bytes() -> [u8; HEADER_SIZE] {
    let mut bytes = [0u8; HEADER_SIZE];
    bytes[12..16].copy_from_slice(&END_MARK.to_le_bytes());
    bytes
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct FormatManifest {
    pub(crate) magic: String,
    pub(crate) version: u32,
    pub(crate) segment_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ArenaStats {
    pub(crate) format_version: u32,
    pub(crate) segment_count: usize,
    pub(crate) used_bytes: usize,
    pub(crate) capacity_bytes: usize,
}

/// The writable segment: its mapping and the open file behind it, replaced together, so a
/// commit can complete the frame's durability on the file the frame was written to.
pub(crate) struct ActiveSegment {
    map: MmapMut,
    file: File,
}

impl ActiveSegment {
    fn new(file: File) -> Result<Self> {
        // SAFETY: the arena owns the segment file; writers hold the guard around this value.
        let map = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { map, file })
    }

    /// Writes back the mapping's dirty pages (`msync`) without the platform's full sync:
    /// for intermediate flushes whose durability a later [`Self::sync`] completes.
    fn write_back(&self) -> Result<()> {
        self.map
            .flush()
            .map_err(|e| anyhow!("mmap flush failed: {}", e))
    }

    /// Makes every written byte of the segment durable.
    fn sync(&self) -> Result<()> {
        self.write_back()?;
        super::durable_file::sync_mapped_file(&self.file).context("segment sync failed")
    }

    /// Makes `[offset, offset + len)` durable.
    fn sync_range(&self, offset: usize, len: usize) -> Result<()> {
        self.map
            .flush_range(offset, len)
            .map_err(|e| anyhow!("mmap flush_range failed: {}", e))?;
        #[cfg(test)]
        tests::fail_point("commit sync")?;
        super::durable_file::sync_mapped_file(&self.file).context("segment sync failed")
    }
}

/// RwLock-guarded segmented mmap arena with LZ4 compression and CRC32 framing.
/// Every entry is framed: [CRC32: u32][RawLen: u32][CompLen: u32][Version: u32][Data: bytes]
pub(crate) struct PersistentArena {
    segment_size: usize,
    format_version: u32,
    base_path: PathBuf,
    data_path: RwLock<PathBuf>,
    read_segments: RwLock<Vec<Arc<Mmap>>>,
    active_segment: RwLock<ActiveSegment>,
    active_id: AtomicUsize,
    write_offset: AtomicUsize,
    active_capacity: AtomicUsize,
    version_counter: AtomicUsize,
    /// Set once a rotation has created the next segment's file and cleared when the rotation
    /// completes: an open treats every segment but the last as sealed, so after a failed
    /// rotation the next append must rotate rather than write into the current segment.
    rotation_pending: AtomicBool,
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
    pub(crate) fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
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
            // SAFETY: segments before the active one are sealed and never written again; the
            // store lock keeps other engines out of the directory.
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
        // A reopened segment already has its durable length (growth syncs it); only a new or
        // resized file needs the length change and the full sync.
        let length = file.metadata()?.len();
        if length != active_capacity as u64 {
            file.set_len(active_capacity as u64)?;
            file.sync_all()?;
        }
        let mut active = ActiveSegment::new(file)?;
        let mut_map = &mut active.map;

        let (recovered_offset, max_version) = Self::discover_offset(mut_map)
            .with_context(|| format!("arena segment {}", active_path.display()))?;
        // A partially written tail may contain bytes from an older frame. Persist the end
        // marker at the recovered boundary so future restarts never interpret stale tail
        // bytes as a valid frame. (On macOS the next commit's full sync covers it.) A segment
        // created empty just now has no tail: it reads back as zeros, which recover to the
        // same empty log, and the first commit's flush range starts at the marker.
        if recovered_offset + HEADER_SIZE <= mut_map.len()
            && mut_map[recovered_offset..recovered_offset + HEADER_SIZE] != end_marker_bytes()
        {
            mut_map[recovered_offset..recovered_offset + HEADER_SIZE]
                .copy_from_slice(&end_marker_bytes());
            if length != 0 {
                mut_map.flush_range(recovered_offset, HEADER_SIZE)?;
            }
        }
        // `discover_offset` touched every written page and the zero tail up to the segment's
        // capacity; none of it needs to stay resident (replay copies each frame out).
        if super::os_memory::release_journal_pages_enabled() {
            super::os_memory::release_file_pages(mut_map.as_ptr(), mut_map.len());
        }

        Ok(Self {
            segment_size: manifest.segment_size,
            format_version: manifest.version,
            base_path: base,
            data_path: RwLock::new(data_path),
            read_segments: RwLock::new(segments),
            active_segment: RwLock::new(active),
            active_id: AtomicUsize::new(active_id),
            write_offset: AtomicUsize::new(recovered_offset),
            active_capacity: AtomicUsize::new(active_capacity),
            version_counter: AtomicUsize::new(if recovered_offset > 0 {
                max_version as usize + 1
            } else {
                0
            }),
            rotation_pending: AtomicBool::new(false),
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

    pub(crate) fn stats(&self) -> ArenaStats {
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
    pub(crate) fn flush(&self) -> Result<()> {
        self.active_segment.write().sync()
    }

    /// Writes back the active segment's dirty pages without the platform's full sync; a later
    /// [`Self::flush`] makes them durable.
    pub(crate) fn write_back(&self) -> Result<()> {
        self.active_segment.write().write_back()
    }

    /// Walk CRC32-framed entries in the active segment to find the first free offset
    /// and the highest version number present. A frame that fails its checks ends the walk
    /// only when nothing valid follows it (a crash-truncated append); a bad frame with a
    /// valid frame after it is corruption inside committed data and an error.
    fn discover_offset(mmap: &MmapMut) -> Result<(usize, u32)> {
        let mut offset = 0;
        let mut max_version = 0;
        let mut scratch = Vec::new();
        let data = &mmap[..];
        let fail = |offset: usize, max_version: u32| -> Result<(usize, u32)> {
            match Self::committed_after(data, offset + 1, true) {
                Some(next) => Err(anyhow!(
                    "frame at offset {offset} fails its checks but committed data follows at \
                     offset {next}: the segment is corrupt, not a torn tail; restore a \
                     snapshot or re-encode the sources"
                )),
                None => Ok((offset, max_version)),
            }
        };
        // `offset` never passes `data.len()`: it only advances past a frame that fits.
        while let Some([_, raw_len, comp_len, version]) = header_words(&data[offset..]) {
            let (raw_len, comp_len) = (raw_len as usize, comp_len as usize);

            // A zero header or the end marker means no more frames.
            if raw_len == 0 && comp_len == 0 {
                if is_log_end(&data[offset..offset + HEADER_SIZE]) {
                    // A torn first append can persist its end marker without its header.
                    if version == 0
                        && Self::committed_after(data, offset + HEADER_SIZE, false).is_some()
                    {
                        return fail(offset, max_version);
                    }
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

            if !Self::frame_checks_out(data, offset, &mut scratch) {
                return fail(offset, max_version);
            }

            if version > max_version {
                max_version = version;
            }
            offset += frame_size;
        }
        Ok((offset, max_version))
    }

    /// The directory of the live generation (the arena directory itself for a legacy layout).
    pub(crate) fn data_dir(&self) -> PathBuf {
        self.data_path.read().clone()
    }

    /// True if the frame at `offset` has plausible lengths, decompresses and matches its CRC32.
    /// Decompresses into `scratch`, which keeps its capacity for the next call: recovery tests
    /// every offset of a damaged tail, and a fresh output buffer of up to
    /// [`MAX_RAW_FRAME_SIZE`] bytes per candidate costs more than the test.
    fn frame_checks_out(data: &[u8], offset: usize, scratch: &mut Vec<u8>) -> bool {
        let Some([expected_crc, raw_len, comp_len, _]) = data.get(offset..).and_then(header_words)
        else {
            return false;
        };
        let (raw_len, comp_len) = (raw_len as usize, comp_len as usize);
        if comp_len == 0 || raw_len == 0 || raw_len > MAX_RAW_FRAME_SIZE || comp_len > raw_len {
            return false;
        }
        let end = match offset.checked_add(HEADER_SIZE + comp_len) {
            Some(end) if end <= data.len() => end,
            _ => return false,
        };
        let payload = &data[offset + HEADER_SIZE..end];
        let mut hasher = Hasher::new();
        if comp_len < raw_len {
            // The output bound `decompress(payload, raw_len)` gives: `raw_len` bytes.
            if scratch.len() < raw_len {
                scratch.resize(raw_len, 0);
            }
            match lz4_flex::block::decompress_into(payload, &mut scratch[..raw_len]) {
                Ok(written) if written == raw_len => hasher.update(&scratch[..written]),
                Ok(_) => return false,
                Err(_) => return false,
            }
        } else {
            hasher.update(payload);
        }
        hasher.finalize() == expected_crc
    }

    /// The first offset at or after `from` holding committed data: a valid frame or the end
    /// marker when `include_end_marker` is true. Bytes beyond the write frontier are zero.
    fn committed_after(data: &[u8], from: usize, include_end_marker: bool) -> Option<usize> {
        let last = super::format::scan::last_nonzero(data)?;
        let mut scratch = Vec::new();
        (from..=last.saturating_sub(HEADER_SIZE - 1)).find(|&at| {
            at + HEADER_SIZE <= data.len()
                && (Self::frame_checks_out(data, at, &mut scratch)
                    || (include_end_marker
                        && is_log_end(&data[at..at + HEADER_SIZE])
                        && data[at + 12..at + 16] == END_MARK.to_le_bytes()))
        })
    }

    pub(crate) fn read_slice(&self, global_offset: usize, len: usize) -> Result<Vec<u8>> {
        self.with_slice(global_offset, len, <[u8]>::to_vec)
    }

    /// Runs `read` on the mapped bytes `[global_offset, global_offset + len)` under one
    /// acquisition of the mapping locks.
    fn with_slice<R>(
        &self,
        global_offset: usize,
        len: usize,
        read: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        self.with_mapping(global_offset, |rest| {
            rest.get(..len)
                .map(read)
                .ok_or_else(|| anyhow!("Read out of bounds: offset {global_offset}, len {len}"))
        })
    }

    /// Runs `read` on the mapped bytes of the segment holding `global_offset`, from that
    /// offset to the end of the mapping (and never past the segment), under one acquisition
    /// of the mapping locks.
    fn with_mapping<R>(
        &self,
        global_offset: usize,
        read: impl FnOnce(&[u8]) -> Result<R>,
    ) -> Result<R> {
        // Rotation and publication take active before segments; retain both to select one mapping.
        let active = self.active_segment.read();
        let reader = self.read_segments.read();
        let seg_idx = global_offset / self.segment_size;
        if seg_idx > self.active_id.load(Ordering::Acquire) {
            return Err(anyhow!("read beyond final arena segment"));
        }
        let local_offset = global_offset % self.segment_size;
        let mapping: &[u8] = if seg_idx < reader.len() {
            &reader[seg_idx]
        } else {
            &active.map
        };
        let end = mapping.len().min(self.segment_size);
        let rest = mapping
            .get(local_offset..end)
            .ok_or_else(|| anyhow!("read exceeds arena segment"))?;
        read(rest)
    }

    pub(crate) fn read_frame(&self, global_offset: usize) -> Result<(Vec<u8>, u32)> {
        // Header and payload come from one mapping under one lock acquisition; the payload is
        // decompressed (or copied) straight from it.
        let (data, expected_crc, version) = self.with_mapping(global_offset, |rest| {
            let [expected_crc, raw_len, comp_len, version] = header_words(rest)
                .ok_or_else(|| anyhow!("Truncated header at offset {}", global_offset))?;
            let (raw_len, comp_len) = (raw_len as usize, comp_len as usize);
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
            if comp_len == 0 || comp_len > raw_len {
                return Err(anyhow!(
                    "Suspiciously large compressed frame length: {} at {}",
                    comp_len,
                    global_offset
                ));
            }
            let payload = rest
                .get(HEADER_SIZE..HEADER_SIZE + comp_len)
                .ok_or_else(|| {
                    anyhow!(
                        "Truncated payload at offset {}",
                        global_offset + HEADER_SIZE
                    )
                })?;
            let data = if comp_len < raw_len {
                lz4_flex::decompress(payload, raw_len)
                    .map_err(|e| anyhow!("LZ4 decompression failed: {}", e))?
            } else {
                payload.to_vec()
            };
            ensure!(
                data.len() == raw_len,
                "frame payload length differs from its header"
            );
            Ok((data, expected_crc, version))
        })?;

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

    pub(crate) fn next_offset(&self, global_offset: usize) -> Result<usize> {
        let header_bytes = self.read_slice(global_offset, HEADER_SIZE)?;
        let [_, _, comp_len, _] = header_words(&header_bytes)
            .ok_or_else(|| anyhow!("Truncated header at offset {}", global_offset))?;
        let comp_len = comp_len as usize;
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

    pub(crate) fn write_slice(&self, data: &[u8]) -> Result<usize> {
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

        // The header stores both lengths as u32.
        if raw_len > u32::MAX as usize || comp_len > u32::MAX as usize {
            return Err(anyhow!(
                "Frame payload exceeds u32 header capacity: raw={}, comp={}",
                raw_len,
                comp_len
            ));
        }

        // One write guard spans reserve and write, so concurrent writers cannot share a range
        // and rotation cannot interleave between the check and the copy.
        let mut active = self.active_segment.write();

        // Check-then-reserve under write lock to avoid wasting space at segment boundaries.
        let mut offset = self.write_offset.load(Ordering::Acquire);
        if offset + total_frame_len > self.segment_size
            || self.rotation_pending.load(Ordering::Acquire)
        {
            self.rotate_segment_locked(&mut active)?;
            offset = 0;
        }
        if offset + total_frame_len > active.map.len() {
            self.grow_segment_locked(&mut active, offset + total_frame_len)?;
        }
        self.write_offset
            .store(offset + total_frame_len, Ordering::Release);

        // Growth and rotation made the guarded mapping hold the whole frame.
        debug_assert!(offset + total_frame_len <= self.segment_size);
        let frame = &mut active.map[offset..offset + total_frame_len];
        frame[0..4].copy_from_slice(&crc.to_le_bytes());
        frame[4..8].copy_from_slice(&(raw_len as u32).to_le_bytes());
        frame[8..12].copy_from_slice(&(comp_len as u32).to_le_bytes());
        frame[12..HEADER_SIZE].copy_from_slice(&version.to_le_bytes());
        frame[HEADER_SIZE..].copy_from_slice(&write_data);
        let frame_end = offset + total_frame_len;
        // The end marker rides in the frame's flush range: it records the committed end.
        let sentinel_len = if frame_end + HEADER_SIZE <= active.map.len() {
            active.map[frame_end..frame_end + HEADER_SIZE].copy_from_slice(&end_marker_bytes());
            HEADER_SIZE
        } else {
            0
        };
        if durable {
            if let Err(error) = active.sync_range(offset, total_frame_len + sentinel_len) {
                // The caller treats a failed commit as not committed and does not apply it:
                // take the frame back out of the log, so a restart does not replay it and the
                // next append overwrites its range (and syncs that range again). The frame's
                // bytes and end marker are zeroed: torn-tail recovery treats a stale end
                // marker after a damaged frame as committed data.
                active.map[offset..offset + HEADER_SIZE].copy_from_slice(&end_marker_bytes());
                active.map[offset + HEADER_SIZE..frame_end + sentinel_len].fill(0);
                self.write_offset.store(offset, Ordering::Release);
                // Best effort: the original error is the one to report. If this sync fails
                // too, a power loss before the next commit can still bring the frame back.
                let _ = active.sync_range(offset, total_frame_len + sentinel_len);
                return Err(error);
            }
            // The frame is on disk and applied in memory by the caller; its pages are only
            // needed again by replay or compaction, which read them back from the file.
            if super::os_memory::release_journal_pages_enabled() {
                let released = total_frame_len + sentinel_len;
                super::os_memory::release_file_pages(
                    active.map[offset..offset + released].as_ptr(),
                    released,
                );
            }
        }
        let global = self.active_id.load(Ordering::SeqCst) * self.segment_size + offset;
        Ok(global)
    }

    /// Drops the resident pages of `[global_offset, global_offset + len)` once their bytes
    /// are copied out (replay) or committed; the next read refaults them from the file. A
    /// hint, so an out-of-range span is ignored. Off with `HMS_RELEASE_JOURNAL_PAGES=0`.
    pub(crate) fn release_pages(&self, global_offset: usize, len: usize) {
        if len == 0 || !super::os_memory::release_journal_pages_enabled() {
            return;
        }
        let active = self.active_segment.read();
        let reader = self.read_segments.read();
        let seg_idx = global_offset / self.segment_size;
        let local = global_offset % self.segment_size;
        let Some(end) = local.checked_add(len) else {
            return;
        };
        let bytes: &[u8] = if seg_idx < reader.len() {
            &reader[seg_idx][..]
        } else if seg_idx == self.active_id.load(Ordering::Acquire) {
            &active.map[..]
        } else {
            return;
        };
        if end <= bytes.len() {
            super::os_memory::release_file_pages(bytes[local..end].as_ptr(), len);
        }
    }

    /// [`Self::release_pages`] over every mapping of the arena: after a compacted
    /// generation is published, its freshly written pages are all resident.
    fn release_all_pages(active: &MmapMut, segments: &[Arc<Mmap>]) {
        if !super::os_memory::release_journal_pages_enabled() {
            return;
        }
        super::os_memory::release_file_pages(active.as_ptr(), active.len());
        for segment in segments {
            super::os_memory::release_file_pages(segment.as_ptr(), segment.len());
        }
    }

    fn grow_segment_locked(
        &self,
        active: &mut crate::core::sync::RwLockWriteGuard<'_, ActiveSegment>,
        required: usize,
    ) -> Result<()> {
        ensure!(
            self.format_version >= 4
                && INCREMENTAL_ALLOCATION
                && required <= self.segment_size
                && required > active.map.len(),
            "invalid arena growth request"
        );
        let capacity = required.div_ceil(SEGMENT_SIZE) * SEGMENT_SIZE;
        let path = self.data_path.read().join(format!(
            "seg_{}.bin",
            self.active_id.load(Ordering::Acquire)
        ));
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        active.write_back()?;
        file.set_len(capacity as u64)?;
        file.sync_all()?;
        // SAFETY: the writer guard excludes all readers and the file has its new durable length.
        **active = ActiveSegment::new(file)?;
        self.active_capacity.store(capacity, Ordering::Release);
        Ok(())
    }

    /// Rotate to a new segment. Caller must hold the write lock on active_segment
    /// and pass the guard so we can swap the mmap in place.
    fn rotate_segment_locked(
        &self,
        active: &mut crate::core::sync::RwLockWriteGuard<'_, ActiveSegment>,
    ) -> Result<()> {
        let current_id = self.active_id.load(Ordering::SeqCst);
        let path = self
            .data_path
            .read()
            .join(format!("seg_{}.bin", current_id));
        let file = File::open(&path)?;
        active.sync()?;
        // SAFETY: the segment is sealed by this rotation and never written again.
        let sealed = unsafe { Mmap::map(&file)? };

        let next_id = current_id + 1;
        let next_path = self.data_path.read().join(format!("seg_{}.bin", next_id));
        self.rotation_pending.store(true, Ordering::Release);
        let next_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&next_path)?;
        #[cfg(test)]
        tests::fail_point("rotation after create")?;
        let capacity = if self.format_version >= 4 && INCREMENTAL_ALLOCATION {
            SEGMENT_SIZE
        } else {
            self.segment_size
        };
        next_file.set_len(capacity as u64)?;
        next_file.sync_all()?;
        super::durable_file::sync_directory(&self.data_path.read())?;
        let next = ActiveSegment::new(next_file)?;
        // Every fallible step is behind us: a failure above leaves the arena on its current
        // segment with `read_segments.len() == active_id`, so a retry starts from scratch.
        self.read_segments.write().push(Arc::new(sealed));
        **active = next;
        self.active_capacity.store(capacity, Ordering::Release);
        self.active_id.store(next_id, Ordering::SeqCst);
        self.write_offset.store(0, Ordering::SeqCst);
        self.rotation_pending.store(false, Ordering::Release);
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
            let [_, raw, compressed, _] =
                header_words(&header).context("truncated compaction replay header")?;
            let (raw, compressed) = (raw as usize, compressed as usize);
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

    fn generation_path(&self) -> Result<(String, PathBuf)> {
        let name = format!(
            "gen-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        let path = self.base_path.join(&name);
        Ok((name, path))
    }

    /// Creates the empty directory a compaction stages into, already named as the generation
    /// it will become, so publishing it never renames a directory holding mapped files.
    pub(crate) fn new_generation_dir(&self) -> Result<PathBuf> {
        let (_, path) = self.generation_path()?;
        std::fs::create_dir(&path)?;
        Ok(path)
    }

    /// Publishes the compacted generation in `prepared` and removes the replaced one. With
    /// `scrub`, the replaced generation is overwritten with zeros and synced before unlinking,
    /// and so is every unreferenced generation or legacy segment left by an interrupted
    /// compaction.
    pub(crate) fn replace_with_compacted(
        &self,
        prepared: PreparedCompaction,
        scrub: bool,
    ) -> Result<()> {
        let staged = prepared.arena;
        ensure!(
            staged.segment_size == self.segment_size
                && staged.format_version == self.format_version,
            "compaction storage geometry differs from the live arena"
        );
        staged.flush()?;
        let temp_base = staged.base_path.clone();
        super::durable_file::sync_directory(&temp_base)?;
        // A generation is staged in place (see `new_generation_dir`): Windows refuses to rename
        // a directory whose files are memory mapped, so publication never moves one.
        let generation = temp_base
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|n| n.starts_with("gen-"))
            .map(str::to_owned);
        let (generation, new_path) = match generation {
            Some(name) if temp_base.parent() == Some(self.base_path.as_path()) => {
                (name, temp_base.clone())
            }
            _ => anyhow::bail!("compaction must be staged in a directory from new_generation_dir"),
        };
        super::durable_file::sync_directory(&self.base_path)?;
        let replacement = staged;
        #[cfg(test)]
        compaction_crash_point("staged");
        let mut active = self.active_segment.write();
        let mut segments = self.read_segments.write();
        let mut data_path = self.data_path.write();
        active.write_back()?;
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
        Self::release_all_pages(&active.map, &segments);
        let live_path = new_path.clone();
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
        self.rotation_pending.store(
            replacement.rotation_pending.load(Ordering::Acquire),
            Ordering::Release,
        );
        drop(replacement);
        #[cfg(test)]
        compaction_crash_point("mapped");
        publication?;
        if old_path != self.base_path {
            let removed = if scrub {
                super::durable_file::scrub_remove_dir(&old_path)
            } else {
                std::fs::remove_dir_all(&old_path).map_err(Into::into)
            };
            if let Err(error) = removed {
                if scrub {
                    return Err(error.context("old arena generation could not be scrubbed"));
                }
                tracing::warn!(%error, "old arena generation could not be removed");
            }
        } else {
            for entry in std::fs::read_dir(&self.base_path)? {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("seg_") && name.ends_with(".bin") {
                    let removed = if scrub {
                        super::durable_file::scrub_remove_file(&entry.path())
                    } else {
                        std::fs::remove_file(entry.path()).map_err(Into::into)
                    };
                    if let Err(error) = removed {
                        if scrub {
                            return Err(error.context("legacy arena segment could not be scrubbed"));
                        }
                        tracing::warn!(%error, "legacy arena segment cleanup failed");
                    }
                }
            }
        }
        if scrub {
            self.scrub_unreferenced(&live_path)?;
        }
        Ok(())
    }

    /// Scrubs and removes generations and legacy segments that the live pointer does not
    /// reference: leftovers of a compaction that died before publishing. The caller holds
    /// the compaction lock, so no staging generation is in progress.
    fn scrub_unreferenced(&self, live: &Path) -> Result<()> {
        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let meta = entry.file_type()?;
            if meta.is_dir() && name.starts_with("gen-") && path != live {
                super::durable_file::scrub_remove_dir(&path)?;
            } else if meta.is_file()
                && live != self.base_path.as_path()
                && name.starts_with("seg_")
                && name.ends_with(".bin")
            {
                super::durable_file::scrub_remove_file(&path)?;
            }
        }
        super::durable_file::sync_directory(&self.base_path)
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
        // Drop cannot report a failure, and nothing depends on this write-back: every
        // acknowledged frame was already synced by its commit or by `flush`.
        let _ = self.active_segment.write().write_back();
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

    thread_local! {
        /// The named point at which the calling thread's next pass fails, once.
        static FAIL_AT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
    }

    /// Fails if `point` is armed on this thread: "commit sync" fails a commit's sync after its
    /// bytes were written back (as a failed `fsync` does); "rotation after create" fails a
    /// rotation once the next segment's file exists.
    pub(super) fn fail_point(point: &str) -> Result<()> {
        if FAIL_AT.with(|armed| armed.get() == Some(point) && armed.take().is_some()) {
            return Err(anyhow!("injected failure at {point}"));
        }
        Ok(())
    }

    /// Runs `body` with `point` armed, disarming it afterwards if `body` did not reach it.
    fn failing_at<R>(point: &'static str, body: impl FnOnce() -> R) -> R {
        FAIL_AT.set(Some(point));
        let result = body();
        FAIL_AT.set(None);
        result
    }

    /// After a rotation fails once the next segment's file exists, an open would treat the
    /// current segment as sealed and skip its torn-tail recovery: no later append may go into
    /// it. Crash test: a torn append after the failed rotation still recovers.
    #[test]
    fn failed_rotation_after_creating_the_next_segment_seals_the_current_one() -> Result<()> {
        let dir = tempdir()?;
        {
            let arena = PersistentArena::new(dir.path())?;
            arena.write_slice(b"before the failed rotation")?;
            assert!(failing_at("rotation after create", || {
                arena.rotate_segment_locked(&mut arena.active_segment.write())
            })
            .is_err());
            assert!(dir.path().join("seg_1.bin").exists());
            let after = arena.write_slice(b"after the failed rotation")?;
            assert_eq!(after, arena.segment_size);
            // A torn append: header and part of the payload reached the file, nothing else.
            let torn = arena.stats().used_bytes - arena.segment_size;
            let mut active = arena.active_segment.write();
            active.map[torn..torn + 4].copy_from_slice(&1u32.to_le_bytes());
            active.map[torn + 4..torn + 8].copy_from_slice(&4096u32.to_le_bytes());
            active.map[torn + 8..torn + 12].copy_from_slice(&4096u32.to_le_bytes());
            active.map[torn + 12..torn + 16].copy_from_slice(&9u32.to_le_bytes());
            active.map[torn + HEADER_SIZE..torn + HEADER_SIZE + 64].fill(0xa5);
            active.sync()?;
        }
        let restored = PersistentArena::new(dir.path())?;
        assert_eq!(
            replayed(&restored)?,
            [
                b"before the failed rotation".to_vec(),
                b"after the failed rotation".to_vec()
            ]
        );
        Ok(())
    }

    /// The payloads of every frame in log order, as replay walks them.
    fn replayed(arena: &PersistentArena) -> Result<Vec<Vec<u8>>> {
        let mut frames = Vec::new();
        let mut offset = 0;
        while offset < arena.stats().used_bytes {
            frames.push(arena.read_frame(offset)?.0);
            offset = arena.next_offset(offset)?;
        }
        Ok(frames)
    }

    /// A commit whose sync fails is reported as not committed and is not applied by the
    /// caller, so it must leave the log as it was: not replayed after a restart, and its
    /// range reused by the next append.
    #[test]
    fn failed_commit_sync_leaves_the_log_unchanged() -> Result<()> {
        for retry in [false, true] {
            let dir = tempdir()?;
            {
                let arena = PersistentArena::new(dir.path())?;
                arena.write_slice(b"committed")?;
                let checkpoint = arena.checkpoint();
                let failed = failing_at("commit sync", || {
                    arena.write_slice(b"commit whose sync failed")
                });
                assert!(failed.is_err());
                assert_eq!(arena.checkpoint(), checkpoint);
                if retry {
                    assert_eq!(arena.write_slice(b"next commit")?, checkpoint.1);
                }
            }
            let expected: &[&[u8]] = if retry {
                &[b"committed", b"next commit"]
            } else {
                &[b"committed"]
            };
            assert_eq!(replayed(&PersistentArena::new(dir.path())?)?, expected);
        }
        Ok(())
    }

    /// A rotation that fails part way leaves the arena on its old segment; retrying it must
    /// not map the old segment a second time, which would serve the new segment's reads.
    #[test]
    fn failed_rotation_can_be_retried_without_aliasing_segments() -> Result<()> {
        let dir = tempdir()?;
        let arena = PersistentArena::new(dir.path())?;
        arena.write_slice(b"first segment")?;
        let blocker = dir.path().join("seg_1.bin");
        std::fs::create_dir(&blocker)?;
        assert!(arena
            .rotate_segment_locked(&mut arena.active_segment.write())
            .is_err());
        assert_eq!(arena.stats().segment_count, 1);
        assert_eq!(arena.read_segments.read().len(), 0);
        std::fs::remove_dir(&blocker)?;
        arena.rotate_segment_locked(&mut arena.active_segment.write())?;
        assert_eq!(arena.read_segments.read().len(), 1);
        let second = arena.write_slice(b"second segment")?;
        assert_eq!(second, arena.segment_size);
        assert_eq!(arena.read_frame(second)?.0, b"second segment");
        assert_eq!(arena.read_frame(0)?.0, b"first segment");
        drop(arena);
        let restored = PersistentArena::new(dir.path())?;
        assert_eq!(
            replayed(&restored)?,
            [b"first segment".to_vec(), b"second segment".to_vec()]
        );
        Ok(())
    }

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
                arena
                    .read_slice(
                        new + arena.active_capacity.load(Ordering::Acquire) - 2,
                        length
                    )
                    .is_ok(),
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
        let staged = PersistentArena::new(arena.new_generation_dir()?)?;
        staged.write_slice(b"new generation")?;
        arena.replace_with_compacted(staged.prepare_compacted()?, false)?;
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
        let staged = PersistentArena::new(arena.new_generation_dir()?)?;
        staged.write_slice(b"original segment")?;
        let mut prepared = staged.prepare_compacted()?;
        arena.replay_compaction_tail(&mut prepared, &mut cursor, end)?;
        assert_eq!(cursor, end);
        arena.replace_with_compacted(prepared, false)?;
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
                let mut active = arena.active_segment.write();
                let map = &mut active.map;
                let raw = u32::from_le_bytes(map[4..8].try_into()?);
                let compressed = u32::from_le_bytes(map[8..12].try_into()?);
                assert_eq!(raw, compressed);
                let declared = (raw as usize).checked_add_signed(delta).unwrap() as u32;
                map[4..8].copy_from_slice(&declared.to_le_bytes());
                map.flush()?;
            }
            let staged = PersistentArena::new(arena.new_generation_dir()?)?;
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
                let mut active = arena.active_segment.write();
                let map = &mut active.map;
                map[4..8].copy_from_slice(&(length as u32).to_le_bytes());
                map.flush()?;
            }
            let staged = PersistentArena::new(arena.new_generation_dir()?)?;
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
                let staged = PersistentArena::new(arena.new_generation_dir()?)?;
                let mut offsets = Vec::new();
                for payload in [b"first".as_slice(), b"middle", b"last"] {
                    offsets.push(staged.write_slice(payload)?);
                }
                {
                    let mut active = staged.active_segment.write();
                    let map = &mut active.map;
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
                    .and_then(|prepared| arena.replace_with_compacted(prepared, false))
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

    /// A store's first open skips flushing the end marker of its new, empty segment. A crash
    /// before the first commit leaves that segment all zeros, header included (or without even
    /// its length); recovery opens it as the same empty log and appends from offset 0.
    #[test]
    fn recovers_a_new_segment_whose_end_marker_never_reached_disk() -> Result<()> {
        for keep_length in [true, false] {
            let dir = tempdir()?;
            drop(PersistentArena::new(dir.path())?);
            let segment = dir.path().join("seg_0.bin");
            let file = OpenOptions::new().write(true).open(&segment)?;
            let length = file.metadata()?.len();
            assert!(length >= HEADER_SIZE as u64);
            file.set_len(0)?;
            if keep_length {
                file.set_len(length)?;
            }
            file.sync_all()?;
            drop(file);

            let second = {
                let arena = PersistentArena::new(dir.path())?;
                assert_eq!(arena.stats().used_bytes, 0, "keep_length {keep_length}");
                assert_eq!(arena.write_slice(b"first after the crash")?, 0);
                arena.write_slice(b"second")?
            };
            let arena = PersistentArena::new(dir.path())?;
            assert_eq!(arena.read_frame(0)?.0, b"first after the crash");
            assert_eq!(arena.read_frame(second)?.0, b"second");
        }
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
