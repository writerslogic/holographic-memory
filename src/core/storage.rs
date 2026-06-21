// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use crc32fast::Hasher;
use memmap2::{Mmap, MmapMut};
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Fixed segment size for mmap arena (1 GB).
const SEGMENT_SIZE: usize = 1024 * 1024 * 1024;
/// Frame header: [CRC32: u32][RawLen: u32][CompLen: u32][Version: u32]
const HEADER_SIZE: usize = 16;
/// Maximum decompressed frame payload (50 MB). Used consistently in
/// `discover_offset` and `read_frame` to reject corrupt/malicious data.
const MAX_RAW_FRAME_SIZE: usize = 50 * 1024 * 1024;

/// RwLock-guarded segmented mmap arena with LZ4 compression and CRC32 framing.
/// Writers acquire a write lock; readers acquire a read lock.
/// Note: `version_counter` resets to 0 on restart (no WAL recovery).
/// Every entry is framed: [CRC32: u32][RawLen: u32][CompLen: u32][Version: u32][Data: bytes]
pub struct PersistentArena {
    base_path: PathBuf,
    read_segments: RwLock<Vec<Arc<Mmap>>>,
    active_segment: Arc<RwLock<MmapMut>>,
    active_id: AtomicUsize,
    write_offset: AtomicUsize,
    version_counter: AtomicUsize,
}

impl PersistentArena {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base)?;

        let mut id = 0;
        loop {
            let p = base.join(format!("seg_{}.bin", id));
            if !p.exists() {
                break;
            }
            id += 1;
        }

        let active_id = if id > 0 { id - 1 } else { 0 };
        let mut segments = Vec::new();

        for i in 0..active_id {
            let p = base.join(format!("seg_{}.bin", i));
            let file = File::open(&p)?;
            let mmap = unsafe { Mmap::map(&file)? };
            segments.push(Arc::new(mmap));
        }

        let active_path = base.join(format!("seg_{}.bin", active_id));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&active_path)?;
        file.set_len(SEGMENT_SIZE as u64)?;
        file.sync_all()?;
        let mut_map = unsafe { MmapMut::map_mut(&file)? };

        let (recovered_offset, max_version) = Self::discover_offset(&mut_map);

        Ok(Self {
            base_path: base,
            read_segments: RwLock::new(segments),
            active_segment: Arc::new(RwLock::new(mut_map)),
            active_id: AtomicUsize::new(active_id),
            write_offset: AtomicUsize::new(recovered_offset),
            version_counter: AtomicUsize::new(if recovered_offset > 0 {
                max_version as usize + 1
            } else {
                0
            }),
        })
    }

    /// Walk CRC32-framed entries in the active segment to find the first free offset
    /// and the highest version number present.
    fn discover_offset(mmap: &MmapMut) -> (usize, u32) {
        let mut offset = 0;
        let mut max_version = 0;
        let data = &mmap[..];
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

            // Zero header means no more frames
            if raw_len == 0 && comp_len == 0 {
                break;
            }
            // Sanity: reject impossibly large or zero-progress frames
            if comp_len > SEGMENT_SIZE || raw_len > MAX_RAW_FRAME_SIZE || comp_len == 0 {
                break;
            }

            let frame_size = HEADER_SIZE + comp_len;
            if offset + frame_size > data.len() {
                break;
            }

            // Verify CRC to make sure this is a valid frame
            let expected_crc =
                u32::from_le_bytes(data[offset..offset + 4].try_into().expect("4-byte slice"));
            let payload = &data[offset + HEADER_SIZE..offset + frame_size];
            let decompressed = if comp_len < raw_len {
                match lz4_flex::decompress(payload, raw_len) {
                    Ok(d) => d,
                    Err(_) => break,
                }
            } else {
                payload.to_vec()
            };

            let mut hasher = Hasher::new();
            hasher.update(&decompressed);
            if hasher.finalize() != expected_crc {
                break;
            }

            if version > max_version {
                max_version = version;
            }
            offset += frame_size;
        }
        (offset, max_version)
    }

    pub fn read_slice(&self, global_offset: usize, len: usize) -> Result<Vec<u8>> {
        let seg_idx = global_offset / SEGMENT_SIZE;
        let local_offset = global_offset % SEGMENT_SIZE;

        let reader = self.read_segments.read();
        if seg_idx < reader.len() {
            let seg = &reader[seg_idx];
            if local_offset + len <= seg.len() {
                return Ok(seg[local_offset..local_offset + len].to_vec());
            }
        }
        drop(reader);

        let active = self.active_segment.read();
        if local_offset + len <= active.len() {
            Ok(active[local_offset..local_offset + len].to_vec())
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
        if comp_len > SEGMENT_SIZE {
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
        Ok(global_offset + HEADER_SIZE + comp_len)
    }

    pub fn write_slice(&self, data: &[u8]) -> Result<usize> {
        let raw_len = data.len();
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
        if offset + total_frame_len > SEGMENT_SIZE {
            self.rotate_segment_locked(&mut active)?;
            offset = 0;
        }
        self.write_offset
            .store(offset + total_frame_len, Ordering::Release);

        // SAFETY: we hold the exclusive write lock, so no concurrent writers.
        // offset + total_frame_len <= SEGMENT_SIZE was verified above.
        // Each copy_nonoverlapping targets a distinct, non-overlapping range:
        //   [offset..+4], [offset+4..+8], [offset+8..+12], [offset+12..+16], [offset+16..+16+comp_len]
        debug_assert!(offset + total_frame_len <= SEGMENT_SIZE);
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
        active
            .flush_range(offset, total_frame_len)
            .map_err(|e| anyhow!("mmap flush_range failed: {}", e))?;
        let global = self.active_id.load(Ordering::SeqCst) * SEGMENT_SIZE + offset;
        Ok(global)
    }

    /// Rotate to a new segment. Caller must hold the write lock on active_segment
    /// and pass the guard so we can swap the mmap in place.
    fn rotate_segment_locked(
        &self,
        active: &mut parking_lot::RwLockWriteGuard<'_, MmapMut>,
    ) -> Result<()> {
        let current_id = self.active_id.load(Ordering::SeqCst);
        let path = self.base_path.join(format!("seg_{}.bin", current_id));
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        self.read_segments.write().push(Arc::new(mmap));

        let next_id = current_id + 1;
        let next_path = self.base_path.join(format!("seg_{}.bin", next_id));
        let next_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&next_path)?;
        next_file.set_len(SEGMENT_SIZE as u64)?;
        let next_map = unsafe { MmapMut::map_mut(&next_file)? };

        **active = next_map;
        self.active_id.store(next_id, Ordering::SeqCst);
        self.write_offset.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Atomically replace the arena contents with a compacted version.
    /// The caller must have already written the compacted data to `temp_base`
    /// using a separate PersistentArena instance (which must be dropped before
    /// calling this method so its file handles are released).
    pub fn replace_with_compacted(&self, temp_base: &Path) -> Result<()> {
        let mut segments = self.read_segments.write();
        let mut active = self.active_segment.write();

        // 1. Release all mmaps by replacing with a dummy
        segments.clear();
        let dummy_path = self.base_path.join(".dummy_mmap");
        let dummy_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&dummy_path)?;
        dummy_file.set_len(1)?;
        *active = unsafe { MmapMut::map_mut(&dummy_file)? };

        // 2. Delete all current segment files
        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "bin") {
                std::fs::remove_file(path)?;
            }
        }
        let _ = std::fs::remove_file(&dummy_path);

        // 3. Move compacted files into base path
        for entry in std::fs::read_dir(temp_base)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "bin") {
                let dest = self.base_path.join(entry.file_name());
                std::fs::rename(&path, &dest)?;
            }
        }

        // 4. Re-open segments from the compacted files
        let mut id = 0;
        loop {
            let p = self.base_path.join(format!("seg_{}.bin", id));
            if !p.exists() {
                break;
            }
            id += 1;
        }

        let active_id = if id > 0 { id - 1 } else { 0 };
        for i in 0..active_id {
            let p = self.base_path.join(format!("seg_{}.bin", i));
            let file = File::open(&p)?;
            let mmap = unsafe { Mmap::map(&file)? };
            segments.push(Arc::new(mmap));
        }

        let active_path = self.base_path.join(format!("seg_{}.bin", active_id));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&active_path)?;
        file.set_len(SEGMENT_SIZE as u64)?;
        let new_mmap = unsafe { MmapMut::map_mut(&file)? };

        let (recovered_offset, max_version) = Self::discover_offset(&new_mmap);

        *active = new_mmap;
        self.active_id.store(active_id, Ordering::SeqCst);
        self.write_offset.store(recovered_offset, Ordering::SeqCst);
        self.version_counter.store(
            if recovered_offset > 0 {
                max_version as usize + 1
            } else {
                0
            },
            Ordering::SeqCst,
        );

        // 5. Clean up temp directory
        let _ = std::fs::remove_dir_all(temp_base);

        Ok(())
    }
}

impl Drop for PersistentArena {
    fn drop(&mut self) {
        let _ = self.active_segment.write().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
}
