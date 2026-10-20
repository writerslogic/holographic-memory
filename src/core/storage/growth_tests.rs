// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;
use tempfile::tempdir;

fn incompressible_payload(length: usize) -> Vec<u8> {
    let mut state = 0x2859_c7a1_9b38_6de4_u64;
    let mut bytes = vec![0; length];
    for byte in &mut bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state as u8;
    }
    bytes
}

#[test]
fn current_growth_preserves_complete_frames_at_mapping_boundaries() -> Result<()> {
    let mut state = 0x2859_c7a1_9b38_6de4_u64;
    let mut payload = vec![0; SEGMENT_SIZE / 2];
    for byte in &mut payload {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state as u8;
    }
    for delta in [-1_isize, 0, 1] {
        let root = tempdir()?;
        let arena = PersistentArena::new(root.path())?;
        let first = arena.write_slice(&payload)?;
        let second_len = (SEGMENT_SIZE - payload.len() - 2 * HEADER_SIZE)
            .checked_add_signed(delta)
            .unwrap();
        let second = arena.write_slice(&payload[..second_len])?;
        assert_eq!(second, HEADER_SIZE + payload.len());
        let last = arena.write_slice(b"last")?;
        let used = arena.stats().used_bytes;
        assert_eq!(arena.stats().segment_count, 1);
        assert_eq!(arena.stats().capacity_bytes, 2 * SEGMENT_SIZE);
        drop(arena);
        let restored = PersistentArena::new(root.path())?;
        assert_eq!(restored.stats().used_bytes, used);
        assert_eq!(restored.read_frame(first)?, (payload.clone(), 0));
        assert_eq!(
            restored.read_frame(second)?,
            (payload[..second_len].to_vec(), 1)
        );
        assert_eq!(restored.read_frame(last)?, (b"last".to_vec(), 2));
        assert_eq!(restored.next_offset(first)?, second);
        assert_eq!(restored.next_offset(second)?, last);
        assert_eq!(restored.next_offset(last)?, used);
    }
    Ok(())
}

#[test]
fn failed_growth_preserves_complete_frames_and_allows_identical_payload_retry() -> Result<()> {
    let root = tempdir()?;
    let arena = PersistentArena::new(root.path())?;
    let payload = incompressible_payload(SEGMENT_SIZE / 2);
    let first = arena.write_slice(&payload)?;
    let second = arena.write_slice(&payload[..payload.len() - 2 * HEADER_SIZE])?;
    assert_eq!(arena.stats().used_bytes, SEGMENT_SIZE);
    let checkpoint = arena.checkpoint();
    std::fs::rename(
        root.path().join("seg_0.bin"),
        root.path().join("temporarily-missing.bin"),
    )?;
    assert!(arena.write_slice(b"retry the same payload").is_err());
    assert_eq!(arena.checkpoint(), checkpoint);
    assert_eq!(arena.stats().capacity_bytes, SEGMENT_SIZE);
    assert_eq!(arena.read_frame(first)?.0, payload);
    std::fs::rename(
        root.path().join("temporarily-missing.bin"),
        root.path().join("seg_0.bin"),
    )?;
    let offset = arena.write_slice(b"retry the same payload")?;
    assert_eq!(offset, SEGMENT_SIZE);
    assert_eq!(arena.stats().capacity_bytes, 2 * SEGMENT_SIZE);
    drop(arena);
    let restored = PersistentArena::new(root.path())?;
    assert_eq!(restored.read_frame(first)?.0, payload);
    assert_eq!(
        restored.read_frame(second)?.0,
        payload[..payload.len() - 2 * HEADER_SIZE]
    );
    assert_eq!(restored.read_frame(offset)?.0, b"retry the same payload");
    Ok(())
}

#[test]
fn unfinished_growth_keeps_the_acknowledged_prefix_and_capacity_is_bounded() -> Result<()> {
    for length in [
        SEGMENT_SIZE + 1,
        2 * SEGMENT_SIZE - 1,
        2 * SEGMENT_SIZE,
        LEGACY_SEGMENT_SIZE - 1,
        LEGACY_SEGMENT_SIZE,
        LEGACY_SEGMENT_SIZE + 1,
    ] {
        let root = tempdir()?;
        let arena = PersistentArena::new(root.path())?;
        arena.write_slice(b"acknowledged prefix")?;
        let used = arena.stats().used_bytes;
        let expected = arena.read_slice(0, used)?;
        drop(arena);
        let path = root.path().join("seg_0.bin");
        let file = OpenOptions::new().write(true).open(&path)?;
        file.set_len(length as u64)?;
        file.sync_all()?;
        drop(file);
        let reopened = PersistentArena::new(root.path());
        if length > LEGACY_SEGMENT_SIZE {
            assert!(reopened.is_err());
            assert_eq!(std::fs::metadata(path)?.len(), length as u64);
        } else {
            let restored = reopened?;
            assert_eq!(restored.stats().used_bytes, used);
            assert_eq!(restored.read_slice(0, used)?, expected);
            assert_eq!(
                restored.stats().capacity_bytes,
                length.div_ceil(SEGMENT_SIZE) * SEGMENT_SIZE
            );
            restored.write_slice(b"after allocation recovery")?;
        }
    }
    Ok(())
}

#[test]
fn readers_wait_for_growth_and_resume_on_the_same_complete_frames() -> Result<()> {
    let root = tempdir()?;
    let arena = Arc::new(PersistentArena::new(root.path())?);
    let offset = arena.write_slice(b"original frame")?;
    let mut active = arena.active_segment.write();
    let (tx, rx) = std::sync::mpsc::channel();
    let other = Arc::clone(&arena);
    let reader = std::thread::spawn(move || -> Result<()> {
        tx.send(())?;
        for _ in 0..128 {
            ensure!(other.read_frame(offset)?.0 == b"original frame");
        }
        Ok(())
    });
    rx.recv()?;
    arena.grow_segment_locked(&mut active, SEGMENT_SIZE + 1)?;
    drop(active);
    reader.join().unwrap()?;
    assert_eq!(arena.stats().capacity_bytes, 2 * SEGMENT_SIZE);
    arena.write_slice(b"after growth")?;
    drop(arena);
    let restored = PersistentArena::new(root.path())?;
    assert_eq!(restored.read_frame(offset)?.0, b"original frame");
    Ok(())
}
