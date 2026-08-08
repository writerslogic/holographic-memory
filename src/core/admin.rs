// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, bail, Context, Result};
use crc32fast::Hasher;
use memmap2::Mmap;
use serde::Serialize;
use std::fs::File;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::storage::{
    FormatManifest, FORMAT_MAGIC, FORMAT_MANIFEST, HEADER_SIZE, MAX_RAW_FRAME_SIZE, SEGMENT_SIZE,
    STORAGE_FORMAT_VERSION,
};
use super::store_lock::{StoreLock, LOCK_FILE};

const ARENA_DIRECTORY: &str = "vectors_data.bin";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreInspection {
    pub store_path: String,
    pub format_version: u32,
    pub segment_count: usize,
    pub frame_count: usize,
    pub logical_bytes: u64,
    pub allocated_bytes: u64,
    pub checksums_verified: bool,
}

/// Inspect storage framing without opening or changing the HMS engine.
pub fn inspect_store(path: impl AsRef<Path>, verify_checksums: bool) -> Result<StoreInspection> {
    let store_path = path.as_ref();
    let arena_path = store_path.join(ARENA_DIRECTORY);
    let manifest_path = arena_path.join(FORMAT_MANIFEST);
    let manifest: FormatManifest = serde_json::from_slice(
        &std::fs::read(&manifest_path)
            .with_context(|| format!("missing storage manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("invalid storage manifest {}", manifest_path.display()))?;

    if manifest.magic != FORMAT_MAGIC {
        bail!("unknown storage magic {}", manifest.magic);
    }
    if manifest.version > STORAGE_FORMAT_VERSION {
        bail!(
            "storage format {} is newer than supported format {}",
            manifest.version,
            STORAGE_FORMAT_VERSION
        );
    }
    if manifest.segment_size != SEGMENT_SIZE {
        bail!(
            "storage segment size mismatch: manifest={}, runtime={}",
            manifest.segment_size,
            SEGMENT_SIZE
        );
    }

    let mut segment_count = 0usize;
    let mut frame_count = 0usize;
    let mut logical_bytes = 0u64;
    loop {
        let segment_path = arena_path.join(format!("seg_{segment_count}.bin"));
        if !segment_path.exists() {
            break;
        }
        let file = File::open(&segment_path)?;
        let metadata = file.metadata()?;
        if metadata.len() != SEGMENT_SIZE as u64 {
            bail!(
                "segment {} has size {}, expected {}",
                segment_path.display(),
                metadata.len(),
                SEGMENT_SIZE
            );
        }
        let mmap = unsafe { Mmap::map(&file)? };
        let (frames, bytes) = inspect_segment(&mmap, &segment_path, verify_checksums)?;
        frame_count += frames;
        logical_bytes += bytes as u64;
        segment_count += 1;
    }
    if segment_count == 0 {
        bail!("store contains no arena segments");
    }
    for entry in std::fs::read_dir(&arena_path)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if let Some(id) = name
            .strip_prefix("seg_")
            .and_then(|value| value.strip_suffix(".bin"))
            .and_then(|value| value.parse::<usize>().ok())
        {
            if id >= segment_count {
                bail!("arena segments are not contiguous (unexpected seg_{id}.bin)");
            }
        }
    }

    Ok(StoreInspection {
        store_path: store_path.display().to_string(),
        format_version: manifest.version,
        segment_count,
        frame_count,
        logical_bytes,
        allocated_bytes: segment_count as u64 * SEGMENT_SIZE as u64,
        checksums_verified: verify_checksums,
    })
}

fn inspect_segment(data: &[u8], path: &Path, verify_checksums: bool) -> Result<(usize, usize)> {
    let mut offset = 0usize;
    let mut frames = 0usize;
    while offset + HEADER_SIZE <= data.len() {
        let expected_crc = u32::from_le_bytes(data[offset..offset + 4].try_into()?);
        let raw_len = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
        let comp_len = u32::from_le_bytes(data[offset + 8..offset + 12].try_into()?) as usize;
        if raw_len == 0 && comp_len == 0 {
            return Ok((frames, offset));
        }
        if raw_len == 0 || raw_len > MAX_RAW_FRAME_SIZE || comp_len == 0 || comp_len > SEGMENT_SIZE
        {
            bail!(
                "invalid frame header in {} at byte {}",
                path.display(),
                offset
            );
        }
        let frame_end = offset
            .checked_add(HEADER_SIZE + comp_len)
            .ok_or_else(|| anyhow!("frame length overflow in {}", path.display()))?;
        if frame_end > data.len() {
            bail!("truncated frame in {} at byte {}", path.display(), offset);
        }
        if verify_checksums {
            let payload = &data[offset + HEADER_SIZE..frame_end];
            let decoded = if comp_len < raw_len {
                lz4_flex::decompress(payload, raw_len).with_context(|| {
                    format!("invalid LZ4 frame in {} at byte {}", path.display(), offset)
                })?
            } else {
                payload.to_vec()
            };
            let mut hasher = Hasher::new();
            hasher.update(&decoded);
            if hasher.finalize() != expected_crc {
                bail!("checksum mismatch in {} at byte {}", path.display(), offset);
            }
        }
        frames += 1;
        offset = frame_end;
    }
    Ok((frames, offset))
}

/// Create a verified, point-in-time copy of a current-format store.
/// The source is exclusively locked for the full copy and verification. The
/// destination appears only after verification succeeds.
pub fn migrate_store(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<StoreInspection> {
    let source = source.as_ref();
    let destination = destination.as_ref();
    if destination.exists() {
        bail!(
            "migration destination already exists: {}",
            destination.display()
        );
    }
    let source_canonical = source.canonicalize()?;
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let parent_canonical = parent.canonicalize()?;
    if parent_canonical.starts_with(&source_canonical) {
        bail!("migration destination cannot be inside the source store");
    }

    let _source_lock = StoreLock::acquire(source)?;
    // Verify after acquiring the lock so the exact snapshot being copied is
    // the snapshot that passed integrity validation.
    inspect_store(source, true)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temp = parent.join(format!(".hms-migrate-{}-{nonce}", std::process::id()));
    let result = (|| -> Result<StoreInspection> {
        copy_tree(source, &temp)?;
        let inspection = inspect_store(&temp, true)?;
        std::fs::rename(&temp, destination).with_context(|| {
            format!(
                "failed to atomically publish migration {} -> {}",
                temp.display(),
                destination.display()
            )
        })?;
        Ok(StoreInspection {
            store_path: destination.display().to_string(),
            ..inspection
        })
    })();
    if result.is_err() && temp.exists() {
        let _ = std::fs::remove_dir_all(&temp);
    }
    result
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    std::fs::create_dir(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!("migration refuses symbolic link {}", entry.path().display());
        }
        if entry.file_name() == LOCK_FILE {
            continue;
        }
        let target = destination.join(entry.file_name());
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_file() {
            std::fs::copy(entry.path(), &target)?;
            std::fs::set_permissions(&target, entry.metadata()?.permissions())?;
        } else {
            bail!("migration refuses special file {}", entry.path().display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::entangled::EntangledHVec;
    use crate::core::HmsCore;

    #[test]
    fn inspect_verify_and_migrate_roundtrip() -> Result<()> {
        let root = tempfile::tempdir()?;
        let source = root.path().join("source");
        {
            let hms = HmsCore::new(256, Some(source.display().to_string()), None)?;
            hms.memorize("one".into(), EntangledHVec::new_deterministic(256, 1))?;
        }

        let inspected = inspect_store(&source, false)?;
        assert_eq!(inspected.frame_count, 1);
        assert!(!inspected.checksums_verified);
        let verified = inspect_store(&source, true)?;
        assert!(verified.checksums_verified);
        let destination = root.path().join("destination");
        let migrated = migrate_store(&source, &destination)?;
        assert_eq!(migrated.frame_count, 1);
        let reopened = HmsCore::new(256, Some(destination.display().to_string()), None)?;
        assert_eq!(reopened.index_status().vector_count, 1);
        Ok(())
    }

    #[test]
    fn verification_rejects_corruption() -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::{Seek, SeekFrom, Write};
        let root = tempfile::tempdir()?;
        let source = root.path().join("source");
        {
            let hms = HmsCore::new(256, Some(source.display().to_string()), None)?;
            hms.memorize("one".into(), EntangledHVec::new_deterministic(256, 1))?;
        }
        let segment = source.join(ARENA_DIRECTORY).join("seg_0.bin");
        let mut file = OpenOptions::new().write(true).open(segment)?;
        file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        file.write_all(&[0xff])?;
        file.sync_all()?;
        assert!(inspect_store(&source, true).is_err());
        Ok(())
    }
}
