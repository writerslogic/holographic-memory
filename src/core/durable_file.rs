// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

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

/// Completes the durability of a range of a `MAP_SHARED` mapping of `file` whose `msync`
/// (memmap2 `flush_range`/`flush`) has just returned.
///
/// - macOS: `msync(MS_SYNC)` and `fsync` hand the pages to the drive, whose write cache can
///   still lose them on power failure. `fcntl(F_FULLFSYNC)` also flushes that cache. A
///   filesystem that refuses it (some network and FAT volumes) gets `fsync`, as SQLite does,
///   which is the strongest request such a volume accepts.
/// - Linux: `msync(MS_SYNC)` on a shared file mapping runs `vfs_fsync_range` with `datasync`,
///   the same durability point as `fdatasync` of that range (data written, drive cache flushed
///   on filesystems with write barriers). The arena makes the file length durable when a
///   segment is created or grown, so a data sync suffices; nothing further is done.
/// - Windows: memmap2's flush is `FlushViewOfFile` followed by `FlushFileBuffers`; nothing
///   further is done.
pub(crate) fn sync_mapped_file(file: &std::fs::File) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        use std::os::raw::c_int;
        extern "C" {
            fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
            fn fsync(fd: c_int) -> c_int;
        }
        // <sys/fcntl.h>: "fsync + ask the drive to flush to the media".
        const F_FULLFSYNC: c_int = 51;
        let fd = file.as_raw_fd();
        loop {
            // SAFETY: `fd` is open for the lifetime of `file`; F_FULLFSYNC takes no argument.
            if unsafe { fcntl(fd, F_FULLFSYNC) } == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                tracing::debug!(%error, "F_FULLFSYNC refused; falling back to fsync");
                break;
            }
        }
        loop {
            // SAFETY: as above.
            if unsafe { fsync(fd) } == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = file;
        Ok(())
    }
}

/// Overwrites a file with zeros, syncs it, then unlinks it, so the bytes are not left in the
/// freed blocks of a filesystem that does not zero them. Copy-on-write and wear-levelled
/// media can still keep prior copies, which no file API controls.
pub(crate) fn scrub_remove_file(path: &Path) -> Result<()> {
    use std::io::{Seek, SeekFrom};
    if path.symlink_metadata()?.file_type().is_symlink() {
        // Never write through a link; the target is not ours to overwrite.
        return Ok(std::fs::remove_file(path)?);
    }
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    let mut remaining = file.metadata()?.len();
    let zeros = vec![0u8; 1 << 20];
    file.seek(SeekFrom::Start(0))?;
    while remaining > 0 {
        let n = remaining.min(zeros.len() as u64) as usize;
        file.write_all(&zeros[..n])?;
        remaining -= n as u64;
    }
    file.sync_all()?;
    drop(file);
    std::fs::remove_file(path)?;
    Ok(())
}

/// [`scrub_remove_file`] for every file under `dir`, then removes the directory tree.
pub(crate) fn scrub_remove_dir(dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            scrub_remove_dir(&entry.path())?;
        } else {
            scrub_remove_file(&entry.path())?;
        }
    }
    std::fs::remove_dir(dir)?;
    sync_directory(dir.parent().unwrap_or_else(|| Path::new(".")))
}

/// Manifests that carry a `<name>.crc32` sidecar holding the CRC32 of their bytes, so a flip
/// inside a value that still parses is caught at open and by `hms-admin verify`. Both are
/// written once; `indices.json`, which every flush rewrites, is not on the list: a crash
/// between a rewrite's sidecar and its file would otherwise make an intact store refuse to
/// open, and the file is a cache that already binds itself with a checkpoint and per-file
/// checksums. Stores written before this change may carry an `indices.json.crc32`; it is
/// ignored.
pub(crate) const CHECKSUMMED_MANIFESTS: [&str; 2] = ["store.json", "format.json"];

fn sidecar(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    CHECKSUMMED_MANIFESTS
        .contains(&name)
        .then(|| path.with_file_name(format!("{name}.crc32")))
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if let Some(side) = sidecar(path) {
        // The sidecar lands first: a crash between the two writes leaves a sidecar that does
        // not match the old manifest, which reads as a mismatch rather than a silent accept.
        write_file(
            &side,
            format!("{:08x}\n", crc32fast::hash(bytes)).as_bytes(),
        )?;
    }
    write_file(path, bytes)?;
    sync_directory(parent)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    temp.persist(path)?;
    Ok(())
}

/// Reads a checksummed manifest. A sidecar that does not match the bytes is an error; a
/// manifest without a sidecar (written before sidecars existed) is returned unverified.
pub(crate) fn read_manifest(path: &Path) -> Result<(Vec<u8>, bool)> {
    let bytes = std::fs::read(path)?;
    let Some(side) = sidecar(path) else {
        return Ok((bytes, false));
    };
    if !side.exists() {
        return Ok((bytes, false));
    }
    let recorded = std::fs::read_to_string(&side)?;
    let recorded = u32::from_str_radix(recorded.trim(), 16)
        .map_err(|e| anyhow::anyhow!("invalid checksum sidecar {}: {e}", side.display()))?;
    let actual = crc32fast::hash(&bytes);
    if recorded != actual {
        bail!(
            "manifest {} fails its checksum (recorded {recorded:08x}, actual {actual:08x}): the store is corrupt",
            path.display()
        );
    }
    Ok((bytes, true))
}

/// Verifies every checksummed manifest present under `store` (and `format.json` under its
/// arena directory); returns the names verified.
pub(crate) fn verify_manifests(store: &Path, arena: &Path) -> Result<Vec<String>> {
    let mut verified = Vec::new();
    for (dir, name) in [(store, "store.json"), (arena, "format.json")] {
        let path = dir.join(name);
        if path.exists() && read_manifest(&path)?.1 {
            verified.push(name.to_string());
        }
    }
    Ok(verified)
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
