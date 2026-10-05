// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{bail, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    std::fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    temp.persist(path)?;
    sync_directory(parent)
}

pub(crate) fn arena_generation(base: &Path) -> Result<PathBuf> {
    let pointer = base.join("CURRENT");
    if !pointer.exists() {
        return Ok(base.to_path_buf());
    }
    let name = std::fs::read_to_string(pointer)?;
    if !name.starts_with("gen-") || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
        bail!("invalid arena generation pointer");
    }
    let path = base.join(name);
    if !path.is_dir() || path.symlink_metadata()?.file_type().is_symlink() {
        bail!("arena generation is missing or is a symbolic link");
    }
    Ok(path)
}
