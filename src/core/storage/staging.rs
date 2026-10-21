// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use std::path::Path;

use super::{PersistentArena, HEADER_SIZE};

const MAX_PENDING_BYTES: usize = 1024 * 1024;
const MAX_PENDING_FRAMES: usize = 1024;

pub(crate) struct StagedArena {
    arena: PersistentArena,
    pending_bytes: usize,
    pending_frames: usize,
}

impl StagedArena {
    pub(crate) fn new(path: &Path, format: Option<&PersistentArena>) -> Result<Self> {
        ensure!(
            std::fs::read_dir(path)?.next().transpose()?.is_none(),
            "staging requires an empty unpublished directory"
        );
        if let Some(format) = format {
            let manifest = super::FormatManifest {
                magic: super::FORMAT_MAGIC.into(),
                version: format.format_version,
                segment_size: format.segment_size,
            };
            crate::core::durable_file::atomic_write(
                &path.join(super::FORMAT_MANIFEST),
                &serde_json::to_vec_pretty(&manifest)?,
            )?;
        }
        Ok(Self {
            arena: PersistentArena::new(path)?,
            pending_bytes: 0,
            pending_frames: 0,
        })
    }

    pub(crate) fn write_slice(&mut self, data: &[u8]) -> Result<usize> {
        let offset = self.arena.write_frame(data, false)?;
        self.pending_bytes = self
            .pending_bytes
            .checked_add(data.len())
            .and_then(|bytes| bytes.checked_add(HEADER_SIZE))
            .context("staged frame byte count overflow")?;
        self.pending_frames += 1;
        if self.pending_bytes >= MAX_PENDING_BYTES || self.pending_frames >= MAX_PENDING_FRAMES {
            // Bounds the dirty pages; `finish` makes the arena durable.
            self.arena.write_back()?;
            self.pending_bytes = 0;
            self.pending_frames = 0;
        }
        Ok(offset)
    }

    pub(crate) fn finish(self) -> Result<PersistentArena> {
        self.arena.flush()?;
        Ok(self.arena)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};

    fn round_trip(payloads: &[Vec<u8>]) -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut arena = StagedArena::new(directory.path(), None)?;
        for payload in payloads {
            arena.write_slice(payload)?;
        }
        arena.finish()?;
        let restored = PersistentArena::new(directory.path())?;
        let mut offset = 0;
        for payload in payloads {
            assert_eq!(restored.read_frame(offset)?.0, *payload);
            offset = restored.next_offset(offset)?;
        }
        assert_eq!(offset, restored.stats().used_bytes);
        Ok(())
    }

    #[test]
    fn staged_flush_count_boundaries_preserve_individual_frames_after_restart() -> Result<()> {
        for count in [
            0,
            MAX_PENDING_FRAMES - 1,
            MAX_PENDING_FRAMES,
            MAX_PENDING_FRAMES + 1,
        ] {
            let payloads: Vec<_> = (0..count).map(|i| i.to_le_bytes().to_vec()).collect();
            round_trip(&payloads)?;
        }
        Ok(())
    }

    #[test]
    fn staged_flush_byte_boundaries_preserve_individual_frames_after_restart() -> Result<()> {
        for bytes in [
            MAX_PENDING_BYTES - 1,
            MAX_PENDING_BYTES,
            MAX_PENDING_BYTES + 1,
        ] {
            round_trip(&[vec![17; bytes - HEADER_SIZE], b"after boundary".to_vec()])?;
        }
        Ok(())
    }

    #[test]
    fn staged_arena_cannot_open_an_existing_store() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let arena = PersistentArena::new(directory.path())?;
        arena.write_slice(b"acknowledged")?;
        assert!(StagedArena::new(directory.path(), None).is_err());
        assert_eq!(arena.read_frame(0)?.0, b"acknowledged");
        Ok(())
    }

    /// Staged frames carry the same end marker as live writes, so damage inside a staged
    /// segment is corruption (committed data follows it), never a recoverable torn tail.
    #[test]
    fn staged_corrupt_frame_fails_open_like_a_live_frame() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut arena = StagedArena::new(directory.path(), None)?;
        arena.write_slice(b"acknowledged prefix")?;
        let bad = arena.write_slice(b"corrupted record")?;
        arena.write_slice(b"later frame")?;
        arena.finish()?;
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(directory.path().join("seg_0.bin"))?;
        file.seek(SeekFrom::Start((bad + HEADER_SIZE) as u64))?;
        let mut byte = [0];
        file.read_exact(&mut byte)?;
        file.seek(SeekFrom::Start((bad + HEADER_SIZE) as u64))?;
        file.write_all(&[byte[0] ^ 1])?;
        file.sync_all()?;
        drop(file);
        let error = match PersistentArena::new(directory.path()) {
            Ok(_) => panic!("corrupt staged frame opened"),
            Err(e) => format!("{e:#}"),
        };
        assert!(error.contains("corrupt"), "{error}");
        Ok(())
    }
}
