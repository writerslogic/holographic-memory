// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn segment_gaps_are_rejected_without_changing_the_arena() -> Result<()> {
    use std::io::Read;

    for missing_zero in [false, true] {
        let dir = tempfile::tempdir()?;
        let arena = PersistentArena::new(dir.path())?;
        arena.write_slice(b"acknowledged history")?;
        drop(arena);
        let original = dir.path().join("seg_0.bin");
        if missing_zero {
            std::fs::rename(&original, dir.path().join("seg_1.bin"))?;
        } else {
            std::fs::write(dir.path().join("seg_2.bin"), b"later history")?;
        }
        let manifest = std::fs::read(dir.path().join(FORMAT_MANIFEST))?;
        let ids = if missing_zero { vec![1] } else { vec![0, 2] };
        let mut before = Vec::new();
        for id in &ids {
            let path = dir.path().join(format!("seg_{id}.bin"));
            let mut file = File::open(&path)?;
            let mut prefix = Vec::new();
            (&mut file).take(64).read_to_end(&mut prefix)?;
            before.push((path, file.metadata()?.len(), prefix));
        }

        let error = PersistentArena::new(dir.path()).err().expect("segment gap");
        assert!(error.to_string().contains("not contiguous"), "{error:#}");
        assert_eq!(std::fs::read(dir.path().join(FORMAT_MANIFEST))?, manifest);
        if missing_zero {
            assert!(!original.exists());
        } else {
            assert!(!dir.path().join("seg_1.bin").exists());
        }
        for (path, length, prefix) in before {
            let mut file = File::open(path)?;
            let mut actual = Vec::new();
            (&mut file).take(64).read_to_end(&mut actual)?;
            assert_eq!(file.metadata()?.len(), length);
            assert_eq!(actual, prefix);
        }
    }
    Ok(())
}

#[test]
fn noncanonical_segment_names_cannot_create_a_new_active_segment() -> Result<()> {
    for alias in ["seg_00.bin", "seg_+0.bin"] {
        let dir = tempfile::tempdir()?;
        let arena = PersistentArena::new(dir.path())?;
        let offset = arena.write_slice(b"acknowledged history")?;
        drop(arena);
        let canonical = dir.path().join("seg_0.bin");
        let renamed = dir.path().join(alias);
        std::fs::rename(&canonical, &renamed)?;

        assert!(PersistentArena::new(dir.path()).is_err());
        assert!(!canonical.exists());
        std::fs::rename(renamed, canonical)?;
        let restored = PersistentArena::new(dir.path())?;
        assert_eq!(restored.read_frame(offset)?.0, b"acknowledged history");
    }
    Ok(())
}

#[test]
fn active_segment_size_bounds_preserve_oversized_files() -> Result<()> {
    use std::io::{Read, Seek, SeekFrom, Write};

    for (version, length) in [
        (3, SEGMENT_SIZE - 1),
        (3, SEGMENT_SIZE),
        (3, SEGMENT_SIZE + 1),
        (2, LEGACY_SEGMENT_SIZE + 1),
        (4, LEGACY_SEGMENT_SIZE + 1),
    ] {
        let dir = tempfile::tempdir()?;
        let segment_size = if version == 3 {
            SEGMENT_SIZE
        } else {
            LEGACY_SEGMENT_SIZE
        };
        let manifest = FormatManifest {
            magic: FORMAT_MAGIC.into(),
            version,
            segment_size,
        };
        super::super::durable_file::atomic_write(
            &dir.path().join(FORMAT_MANIFEST),
            &serde_json::to_vec(&manifest)?,
        )?;
        let path = dir.path().join("seg_0.bin");
        let mut file = File::create(&path)?;
        file.set_len(length as u64)?;
        if length > segment_size {
            file.seek(SeekFrom::End(-1))?;
            file.write_all(&[0xa5])?;
        }
        drop(file);

        let opened = PersistentArena::new(dir.path());
        if length <= segment_size {
            assert_eq!(opened?.stats().capacity_bytes, segment_size);
        } else {
            let error = opened.err().expect("oversized active segment");
            assert!(
                error.to_string().contains("exceeds its declared size"),
                "{error:#}"
            );
            let mut file = File::open(path)?;
            assert_eq!(file.metadata()?.len(), length as u64);
            file.seek(SeekFrom::End(-1))?;
            let mut sentinel = [0];
            file.read_exact(&mut sentinel)?;
            assert_eq!(sentinel, [0xa5]);
        }
    }
    Ok(())
}

#[test]
fn torn_append_with_only_a_landed_end_marker_is_recoverable() -> Result<()> {
    use crate::core::format::scan::{scan_segment, SegmentEnd};
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let payload: Vec<u8> = (0..70_000).map(|_| rng.random()).collect();
    for end in [0, END_MARK] {
        let dir = tempfile::tempdir()?;
        let arena = PersistentArena::new(dir.path())?;
        arena.write_slice(&payload)?;
        assert!(arena.write_offset.load(Ordering::Acquire) > 4096);
        {
            let mut active = arena.active_segment.write();
            active.map[..HEADER_SIZE].fill(0);
            active.map[12..16].copy_from_slice(&end.to_le_bytes());
            active.sync()?;
            let scanned = scan_segment(&active.map, true, |_, _| {});
            assert!(scanned.frames.is_empty());
            assert_eq!(scanned.end_offset, 0);
            assert!(matches!(
                scanned.end,
                SegmentEnd::ZeroHeader | SegmentEnd::Marker
            ));
        }
        drop(arena);
        let reopened = PersistentArena::new(dir.path())?;
        assert_eq!(reopened.write_offset.load(Ordering::Acquire), 0);
        let offset = reopened.write_slice(b"replacement")?;
        assert_eq!(reopened.read_frame(offset)?.0, b"replacement");
    }
    Ok(())
}

#[test]
fn log_end_before_committed_frames_is_corruption() -> Result<()> {
    use crate::core::format::scan::{scan_segment, SegmentEnd};
    for end in [0, END_MARK] {
        let dir = tempfile::tempdir()?;
        let arena = PersistentArena::new(dir.path())?;
        arena.write_slice(b"first committed record")?;
        arena.write_slice(b"second committed record")?;
        {
            let mut active = arena.active_segment.write();
            active.map[..HEADER_SIZE].fill(0);
            active.map[12..16].copy_from_slice(&end.to_le_bytes());
            active.sync()?;
            assert!(matches!(
                scan_segment(&active.map, true, |_, _| {}).end,
                SegmentEnd::Corrupt { .. }
            ));
        }
        drop(arena);
        assert!(
            PersistentArena::new(dir.path()).is_err(),
            "committed data after a log end must not be discarded"
        );
    }
    Ok(())
}

#[test]
fn raw_length_mismatch_is_rejected() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let arena = PersistentArena::new(dir.path())?;
    let data = vec![7; 1024];
    arena.write_slice(&data)?;
    {
        let mut active = arena.active_segment.write();
        active.map[4..8].copy_from_slice(&1025u32.to_le_bytes());
        active.sync()?;
    }
    drop(arena);
    assert!(
        PersistentArena::new(dir.path()).is_err(),
        "declared raw length disagrees with decompressed length"
    );
    Ok(())
}
