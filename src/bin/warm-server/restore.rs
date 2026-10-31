// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::vindex::sync_dir;

const INTENT: &str = ".restore-intent.json";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intent {
    version: u32,
    backup: String,
    staged: String,
}

pub struct Restore {
    live: PathBuf,
    snapshots: PathBuf,
    staging: PathBuf,
    intent: Intent,
}

pub struct TenantLock(File);

impl Drop for TenantLock {
    fn drop(&mut self) {
        // Closing alone can leave a concurrent fork's descriptor holding the lock.
        if let Err(error) = self.0.unlock() {
            crate::log::emit(
                crate::config::LogLevel::Warn,
                "tenant_lock_release_failed",
                &[("error", error.to_string().into())],
            );
        }
    }
}

pub fn lock_tenant(root: &Path, name: &str) -> Result<TenantLock> {
    let locks = root.join("tenant-locks");
    std::fs::create_dir_all(&locks)?;
    ensure!(directory(&locks)?, "missing tenant ownership directory");
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(locks.join(format!("{name}.owner")))?;
    file.try_lock()
        .context("tenant is owned by another server")?;
    Ok(TenantLock(file))
}

fn directory(path: &Path) -> Result<bool> {
    match path.symlink_metadata() {
        Ok(meta) => {
            ensure!(
                meta.is_dir() && !meta.file_type().is_symlink(),
                "restore path is not a real directory"
            );
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn name(path: &Path) -> Result<String> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("invalid restore directory name")?;
    ensure!(safe_name(name), "unsafe restore directory name");
    Ok(name.into())
}

fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

impl Restore {
    pub fn begin(
        live: &Path,
        snapshots: &Path,
        staging: &Path,
        staged: &Path,
        backup: &str,
    ) -> Result<Self> {
        ensure!(
            crate::tenants::valid_snapshot_id(backup),
            "invalid restore backup name"
        );
        ensure!(
            directory(live)? && directory(staged)?,
            "missing restore source"
        );
        ensure!(
            staged.parent().context("staged parent")?.canonicalize()? == staging.canonicalize()?,
            "staged restore escapes staging root"
        );
        let restore = Self {
            live: live.into(),
            snapshots: snapshots.into(),
            staging: staging.into(),
            intent: Intent {
                version: 1,
                backup: backup.into(),
                staged: name(staged)?,
            },
        };
        ensure!(
            !restore.backup().try_exists()? && !snapshots.join(INTENT).try_exists()?,
            "restore already pending"
        );
        restore.persist()?;
        crash_point("intent");
        Ok(restore)
    }

    fn backup(&self) -> PathBuf {
        self.snapshots.join(&self.intent.backup)
    }
    fn staged(&self) -> PathBuf {
        self.staging.join(&self.intent.staged)
    }

    fn persist(&self) -> Result<()> {
        let mut file = tempfile::NamedTempFile::new_in(&self.snapshots)?;
        file.write_all(&serde_json::to_vec(&self.intent)?)?;
        file.as_file().sync_all()?;
        file.persist(self.snapshots.join(INTENT))?;
        sync_dir(&self.snapshots)?;
        Ok(())
    }

    pub fn publish(&self) -> Result<()> {
        std::fs::rename(&self.live, self.backup()).context("cannot set current store aside")?;
        sync_dir(&self.snapshots)?;
        sync_dir(self.live.parent().context("live parent")?)?;
        crash_point("old-moved");
        std::fs::rename(self.staged(), &self.live).context("cannot publish restored store")?;
        sync_dir(self.live.parent().context("live parent")?)?;
        sync_dir(&self.staging)?;
        crash_point("new-moved");
        Ok(())
    }

    pub fn finish(&self) -> Result<()> {
        std::fs::remove_file(self.snapshots.join(INTENT))?;
        sync_dir(&self.snapshots)?;
        crash_point("committed");
        Ok(())
    }

    pub fn rollback(&self) -> Result<()> {
        self.persist()?;
        if directory(&self.backup())? {
            if directory(&self.live)? {
                std::fs::remove_dir_all(&self.live)?;
            }
            std::fs::rename(self.backup(), &self.live)?;
            sync_dir(self.live.parent().context("live parent")?)?;
            sync_dir(&self.snapshots)?;
        }
        ensure!(
            directory(&self.live)?,
            "restore recovery has neither live nor backup store"
        );
        if directory(&self.staged())? {
            std::fs::remove_dir_all(self.staged())?;
            sync_dir(&self.staging)?;
        }
        self.finish()
    }
}

pub fn recover(live: &Path, snapshots: &Path, staging: &Path) -> Result<()> {
    let path = snapshots.join(INTENT);
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let mut bytes = Vec::new();
    file.take(16 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 16 * 1024,
        "restore intent exceeds size limit"
    );
    let intent: Intent = serde_json::from_slice(&bytes).context("invalid restore intent")?;
    ensure!(
        intent.version == 1
            && crate::tenants::valid_snapshot_id(&intent.backup)
            && safe_name(&intent.staged),
        "unsupported or unsafe restore intent"
    );
    Restore {
        live: live.into(),
        snapshots: snapshots.into(),
        staging: staging.into(),
        intent,
    }
    .rollback()
}

fn crash_point(phase: &str) {
    #[cfg(test)]
    if std::env::var("WARM_RESTORE_CRASH_AT").ok().as_deref() == Some(phase) {
        std::process::exit(83);
    }
    let _ = phase;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_crash_child() -> Result<()> {
        let Some(root) = std::env::var_os("WARM_RESTORE_CRASH_ROOT") else {
            return Ok(());
        };
        let root = PathBuf::from(root);
        let operation = Restore::begin(
            &root.join("live"),
            &root.join("snapshots"),
            &root.join("staging"),
            &root.join("staging/new"),
            "backup",
        )?;
        operation.publish()?;
        operation.finish()
    }

    #[test]
    fn interrupted_restore_recovers_old_or_committed_new_without_empty_store() -> Result<()> {
        for phase in ["intent", "old-moved", "new-moved", "committed"] {
            let root = tempfile::tempdir()?;
            for dir in ["live", "snapshots", "staging/new"] {
                std::fs::create_dir_all(root.path().join(dir))?;
            }
            std::fs::write(root.path().join("live/value"), "old")?;
            std::fs::write(root.path().join("staging/new/value"), "new")?;
            let status = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "restore::tests::restore_crash_child",
                    "--nocapture",
                ])
                .env("WARM_RESTORE_CRASH_ROOT", root.path())
                .env("WARM_RESTORE_CRASH_AT", phase)
                .status()?;
            assert_eq!(status.code(), Some(83), "{phase}");
            recover(
                &root.path().join("live"),
                &root.path().join("snapshots"),
                &root.path().join("staging"),
            )?;
            assert_eq!(
                std::fs::read_to_string(root.path().join("live/value"))?,
                if phase == "committed" { "new" } else { "old" }
            );
            recover(
                &root.path().join("live"),
                &root.path().join("snapshots"),
                &root.path().join("staging"),
            )?;
            assert!(!root.path().join("snapshots").join(INTENT).exists());
        }
        Ok(())
    }

    #[test]
    fn tenant_ownership_survives_moving_the_live_directory() -> Result<()> {
        let root = tempfile::tempdir()?;
        std::fs::create_dir(root.path().join("live"))?;
        let first = lock_tenant(root.path(), "tenant")?;
        std::fs::rename(root.path().join("live"), root.path().join("backup"))?;
        assert!(lock_tenant(root.path(), "tenant").is_err());
        let inherited = first.0.try_clone()?;
        drop(first);
        lock_tenant(root.path(), "tenant")?;
        drop(inherited);
        Ok(())
    }
}
