// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Directory snapshots of a store: a verified copy of every file the store needs to reopen,
//! plus a manifest of sizes and SHA-256 digests that restore checks before it opens anything.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::HmsError;

pub(crate) const MANIFEST: &str = "hms-snapshot.json";
const LOCK_FILE: &str = ".hms.lock";
const COMPACT_PREFIX: &str = ".compact-";
const FORMAT: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: u32,
    created_unix_seconds: u64,
    dimensions: u32,
    document_count: u64,
    files: Vec<Entry>,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    path: String,
    bytes: u64,
    sha256: String,
}

pub(crate) struct Written {
    pub file_count: u32,
    pub total_bytes: u64,
    pub document_count: u64,
}

/// Copies `store` into the new directory `dest`. The caller holds off writers.
pub(crate) fn write(
    store: &Path,
    dest: &Path,
    dimensions: u32,
    document_count: u64,
) -> Result<Written, HmsError> {
    let staging = Staging::create(dest)?;
    let mut files = Vec::new();
    copy_tree(store, staging.path(), Path::new(""), true, &mut files)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let total_bytes = files.iter().map(|f| f.bytes).sum();
    let file_count = u32::try_from(files.len()).map_err(|_| HmsError::corrupt("too many files"))?;
    let manifest = Manifest {
        format: FORMAT,
        created_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        dimensions,
        document_count,
        files,
    };
    let mut file = File::create(staging.path().join(MANIFEST))?;
    file.write_all(
        &serde_json::to_vec_pretty(&manifest).map_err(|e| HmsError::corrupt(e.to_string()))?,
    )?;
    file.sync_all()?;
    staging.publish()?;
    Ok(Written {
        file_count,
        total_bytes,
        document_count,
    })
}

/// Verifies `snapshot` against its manifest while copying it into the new directory `dest`.
pub(crate) fn restore(snapshot: &Path, dest: &Path) -> Result<(), HmsError> {
    let manifest_path = snapshot.join(MANIFEST);
    let bytes = fs::read(&manifest_path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => HmsError::NotFound {
            details: format!("{} has no {MANIFEST}", snapshot.display()),
        },
        _ => HmsError::from(e),
    })?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| HmsError::corrupt(format!("unreadable snapshot manifest: {e}")))?;
    if manifest.format != FORMAT {
        return Err(HmsError::corrupt(format!(
            "unsupported snapshot format {}",
            manifest.format
        )));
    }
    let staging = Staging::create(dest)?;
    for entry in &manifest.files {
        let relative = checked_relative(&entry.path)?;
        let target = staging.path().join(&relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let (bytes, digest) = copy_file(&snapshot.join(&relative), &target)?;
        if bytes != entry.bytes || digest != entry.sha256 {
            return Err(HmsError::corrupt(format!(
                "{} does not match the snapshot manifest",
                entry.path
            )));
        }
    }
    staging.publish()
}

/// A sibling directory that becomes `dest` only when complete, so an interrupted copy never
/// leaves a store that looks usable.
struct Staging {
    dir: PathBuf,
    dest: PathBuf,
    published: bool,
}

impl Staging {
    fn create(dest: &Path) -> Result<Self, HmsError> {
        if dest.exists() {
            let empty = dest.is_dir() && fs::read_dir(dest)?.next().is_none();
            if !empty {
                return Err(HmsError::AlreadyExists {
                    details: format!("{} exists and is not an empty directory", dest.display()),
                });
            }
        }
        let name = dest
            .file_name()
            .ok_or_else(|| HmsError::invalid("destination has no file name"))?
            .to_string_lossy()
            .into_owned();
        let parent = match dest.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        fs::create_dir_all(&parent)?;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = parent.join(format!(".{name}.partial-{}-{nanos}", std::process::id()));
        fs::create_dir(&dir)?;
        Ok(Self {
            dir,
            dest: dest.to_path_buf(),
            published: false,
        })
    }

    fn path(&self) -> &Path {
        &self.dir
    }

    fn publish(mut self) -> Result<(), HmsError> {
        if self.dest.is_dir() {
            fs::remove_dir(&self.dest)?;
        }
        fs::rename(&self.dir, &self.dest)?;
        self.published = true;
        if let Some(parent) = self.dest.parent().filter(|p| !p.as_os_str().is_empty()) {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

fn copy_tree(
    root: &Path,
    out_root: &Path,
    relative: &Path,
    top: bool,
    files: &mut Vec<Entry>,
) -> Result<(), HmsError> {
    for entry in fs::read_dir(root.join(relative))? {
        let entry = entry?;
        let name = entry.file_name();
        let name_text = name.to_string_lossy();
        if top && (name_text == LOCK_FILE || name_text.starts_with(COMPACT_PREFIX)) {
            continue;
        }
        let child = relative.join(&name);
        let kind = entry.file_type()?;
        if kind.is_dir() {
            fs::create_dir_all(out_root.join(&child))?;
            copy_tree(root, out_root, &child, false, files)?;
        } else if kind.is_file() {
            let (bytes, sha256) = copy_file(&root.join(&child), &out_root.join(&child))?;
            files.push(Entry {
                path: child
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/"),
                bytes,
                sha256,
            });
        }
    }
    Ok(())
}

/// Copies one file, returning its length and SHA-256 hex digest, and syncs the copy. Blocks of
/// zeros become holes, so the store's preallocated log segment costs only its used bytes on
/// file systems with sparse files; the digest still covers the full logical content.
fn copy_file(from: &Path, to: &Path) -> Result<(u64, String), HmsError> {
    let mut input = File::open(from)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(to)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 256 * 1024];
    let mut total = 0u64;
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        if buffer[..n].iter().all(|&b| b == 0) {
            output.seek(SeekFrom::Current(n as i64))?;
        } else {
            output.write_all(&buffer[..n])?;
        }
        total += n as u64;
    }
    output.set_len(total)?;
    output.sync_all()?;
    let digest = hasher.finalize();
    Ok((total, digest.iter().map(|b| format!("{b:02x}")).collect()))
}

/// A manifest path is relative and stays inside the snapshot; anything else is a tampered
/// manifest.
fn checked_relative(path: &str) -> Result<PathBuf, HmsError> {
    let relative = PathBuf::from(path);
    let clean = !path.is_empty()
        && relative
            .components()
            .all(|c| matches!(c, Component::Normal(_)));
    if !clean || path == MANIFEST {
        return Err(HmsError::corrupt(format!(
            "snapshot manifest names an unsafe path: {path}"
        )));
    }
    Ok(relative)
}
