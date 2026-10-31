// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Context, Result};
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub(crate) const LOCK_FILE: &str = ".hms.lock";

/// Process-scoped exclusive ownership of a writable WARM store.
///
/// WARM does not currently expose a read-only mode, so every engine instance is
/// a potential writer. Holding this lock for the engine lifetime prevents two
/// processes (or two instances in one process) from mutating the same store.
pub(crate) struct StoreLock {
    file: File,
    path: PathBuf,
}

impl StoreLock {
    pub(crate) fn acquire(store_path: &Path) -> Result<Self> {
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

        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => bail!(
                "WARM store {} is already open by another engine; concurrent writers are not supported",
                store_path.display()
            ),
            Err(TryLockError::Error(error)) => {
                return Err(error)
                    .with_context(|| format!("failed to lock store {}", store_path.display()));
            }
        }

        // The pid is for a person inspecting the store; the lock is the `flock` above, which
        // nothing on disk records, so the write is not synced.
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        writeln!(file, "pid={}", std::process::id())?;

        Ok(Self { file, path })
    }
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        if let Err(error) = self.file.unlock() {
            tracing::warn!(path = %self.path.display(), %error, "failed to release WARM store lock");
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
