// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use fs2::FileExt;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const LOCK_FILE: &str = ".hms.lock";

/// Process-scoped exclusive ownership of a writable HMS store.
///
/// HMS does not currently expose a read-only mode, so every engine instance is
/// a potential writer. Holding this lock for the engine lifetime prevents two
/// processes (or two instances in one process) from mutating the same store.
pub struct StoreLock {
    file: File,
    path: PathBuf,
}

impl StoreLock {
    pub fn acquire(store_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(store_path)
            .with_context(|| format!("failed to create store {}", store_path.display()))?;
        let path = store_path.join(LOCK_FILE);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("failed to open store lock {}", path.display()))?;

        file.try_lock_exclusive().with_context(|| {
            format!(
                "HMS store {} is already open by another engine; concurrent writers are not supported",
                store_path.display()
            )
        })?;

        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        writeln!(file, "pid={}", std::process::id())?;
        file.sync_all()?;

        Ok(Self { file, path })
    }
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        if let Err(error) = self.file.unlock() {
            tracing::warn!(path = %self.path.display(), %error, "failed to release HMS store lock");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_is_exclusive_and_released_on_drop() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let first = StoreLock::acquire(dir.path())?;
        let error = StoreLock::acquire(dir.path())
            .err()
            .expect("second lock must fail");
        assert!(error.to_string().contains("already open"));
        drop(first);
        let reacquired = StoreLock::acquire(dir.path())?;
        drop(reacquired);
        Ok(())
    }
}
