// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
