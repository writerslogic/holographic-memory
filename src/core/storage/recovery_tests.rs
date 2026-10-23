// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn zeroed_interior_header_is_corruption() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let arena = PersistentArena::new(dir.path())?;
    arena.write_slice(b"first committed record")?;
    arena.write_slice(b"second committed record")?;
    {
        let mut active = arena.active_segment.write();
        active.map[..HEADER_SIZE].fill(0);
        active.sync()?;
    }
    drop(arena);
    assert!(
        PersistentArena::new(dir.path()).is_err(),
        "committed data after a zeroed header must not be discarded"
    );
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
