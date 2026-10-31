// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, bail, ensure, Context, Result};
use crc32fast::Hasher;
use memmap2::Mmap;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::storage::{
    is_log_end, list_segments, FormatManifest, FORMAT_MAGIC, FORMAT_MANIFEST, HEADER_SIZE,
    LEGACY_SEGMENT_SIZE, MAX_RAW_FRAME_SIZE, SEGMENT_SIZE, STORAGE_FORMAT_VERSION,
};
use super::store_lock::{StoreLock, LOCK_FILE};
use super::{WarmConfig, WarmCore};

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
    /// Manifests whose checksum sidecar was verified (`verify` only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub manifests_verified: Vec<String>,
    /// Document cross-check against the opened store (`verify` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documents: Option<DocumentCheck>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentCheck {
    /// False when the store was not opened; `reason` says why.
    pub checked: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Documents the transaction log says are live; absent when the frames are encrypted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<usize>,
    pub present: usize,
    pub missing: Vec<String>,
}

/// Inspect storage framing without opening or changing the WARM engine.
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
    let expected = if manifest.version == 3 {
        SEGMENT_SIZE
    } else {
        LEGACY_SEGMENT_SIZE
    };
    if manifest.segment_size != expected {
        bail!(
            "storage segment size mismatch: manifest={}, runtime={}",
            manifest.segment_size,
            expected
        );
    }

    let arena_path = super::durable_file::arena_generation(&arena_path)?;
    let ids = list_segments(&arena_path)?;
    ensure!(!ids.is_empty(), "store contains no arena segments");
    ensure!(
        ids.iter().enumerate().all(|(expected, &id)| expected == id),
        "arena segments are not contiguous from 0: {ids:?}"
    );
    let segment_count = ids.len();
    let mut frame_count = 0usize;
    let mut logical_bytes = 0u64;
    let mut allocated_bytes = 0u64;
    for id in ids {
        let segment_path = arena_path.join(format!("seg_{id}.bin"));
        let file = File::open(&segment_path)?;
        let metadata = file.metadata()?;
        let final_segment = id + 1 == segment_count;
        let valid_size = if manifest.version >= 4 && final_segment {
            (SEGMENT_SIZE as u64..=manifest.segment_size as u64).contains(&metadata.len())
                && metadata.len() % SEGMENT_SIZE as u64 == 0
        } else {
            metadata.len() == manifest.segment_size as u64
        };
        if !valid_size {
            bail!(
                "segment {} has size {}, expected {}",
                segment_path.display(),
                metadata.len(),
                manifest.segment_size
            );
        }
        allocated_bytes = allocated_bytes
            .checked_add(metadata.len())
            .context("arena allocated byte count overflow")?;
        let mmap = unsafe { Mmap::map(&file)? };
        let (frames, bytes) = inspect_segment(&mmap, &segment_path, verify_checksums)?;
        frame_count += frames;
        logical_bytes += bytes as u64;
    }

    Ok(StoreInspection {
        store_path: store_path.display().to_string(),
        format_version: manifest.version,
        segment_count,
        frame_count,
        logical_bytes,
        allocated_bytes,
        checksums_verified: verify_checksums,
        manifests_verified: Vec::new(),
        documents: None,
    })
}

/// `inspect_store(path, true)` plus the manifest checksums and a document cross-check: the
/// store is opened through the engine, which authenticates and replays every frame, and
/// every document the log says is live must resolve there. Corruption fails the verify
/// instead of being served as a shorter store.
pub fn verify_store(
    path: impl AsRef<Path>,
    encryption_passphrase_env: Option<String>,
) -> Result<StoreInspection> {
    let store = path.as_ref();
    let mut inspection = inspect_store(store, true)?;
    let arena_path = store.join(ARENA_DIRECTORY);
    inspection.manifests_verified = super::durable_file::verify_manifests(store, &arena_path)?;
    let schema_path = store.join("store.json");
    let schema: Value = serde_json::from_slice(
        &std::fs::read(&schema_path)
            .with_context(|| format!("missing store schema {}", schema_path.display()))?,
    )
    .with_context(|| format!("invalid store schema {}", schema_path.display()))?;
    let dimensions = u32::try_from(
        schema["dimensions"]
            .as_u64()
            .context("store.json lacks dimensions")?,
    )?;
    let mut config = WarmConfig::default();
    if let Some(name) = schema["text_encoder"].as_str() {
        config.text_encoder = name
            .parse()
            .map_err(|e| anyhow!("store.json text encoder {name}: {e}"))?;
    }
    config.meaning.enabled = schema["meaning_enabled"].as_bool().unwrap_or(false);
    config.embedding_space = serde_json::from_value(schema["embedding_space"].clone())
        .context("store.json embedding_space")?;
    let expected = if schema["encrypted"].as_bool().unwrap_or(false) {
        let Some(env) = encryption_passphrase_env else {
            inspection.documents = Some(DocumentCheck {
                checked: false,
                reason: Some(
                    "encrypted store: pass --encryption-passphrase-env to open it and \
                     authenticate every frame"
                        .into(),
                ),
                expected: None,
                present: 0,
                missing: Vec::new(),
            });
            return Ok(inspection);
        };
        config.security.encryption_enabled = true;
        config.security.encryption_passphrase_env = Some(env);
        None
    } else {
        Some(live_document_ids(&arena_path)?)
    };
    let core = WarmCore::new(dimensions, Some(store.display().to_string()), Some(config))
        .context("opening the store for the document cross-check")?;
    inspection.documents = Some(match expected {
        Some(ids) => {
            let missing = missing_documents(&ids, &core);
            DocumentCheck {
                checked: true,
                reason: None,
                expected: Some(ids.len()),
                present: ids.len() - missing.len(),
                missing,
            }
        }
        None => DocumentCheck {
            checked: true,
            reason: Some(
                "every frame authenticated on open; the presence cross-check needs plaintext \
                 frames"
                    .into(),
            ),
            expected: None,
            present: 0,
            missing: Vec::new(),
        },
    });
    Ok(inspection)
}

/// Document ids the plaintext transaction log leaves live: every `Document` id minus later
/// `DeleteDocument`s. Frame bounds and checksums were validated by `inspect_store` first.
fn live_document_ids(arena_path: &Path) -> Result<BTreeSet<String>> {
    let generation = super::durable_file::arena_generation(arena_path)?;
    let mut live = BTreeSet::new();
    for segment in 0usize.. {
        let segment_path = generation.join(format!("seg_{segment}.bin"));
        if !segment_path.exists() {
            break;
        }
        let mmap = unsafe { Mmap::map(&File::open(&segment_path)?)? };
        let mut offset = 0usize;
        while offset + HEADER_SIZE <= mmap.len() {
            let header = &mmap[offset..offset + HEADER_SIZE];
            if is_log_end(header) {
                break;
            }
            let raw_len = u32::from_le_bytes(header[4..8].try_into()?) as usize;
            let comp_len = u32::from_le_bytes(header[8..12].try_into()?) as usize;
            let frame_end = offset + HEADER_SIZE + comp_len;
            ensure!(
                frame_end <= mmap.len(),
                "truncated frame in {} at byte {offset}",
                segment_path.display()
            );
            let payload = &mmap[offset + HEADER_SIZE..frame_end];
            let decoded = if comp_len < raw_len {
                lz4_flex::decompress(payload, raw_len)?
            } else {
                payload.to_vec()
            };
            let json = super::engine::journal::payload_json(&decoded)
                .with_context(|| {
                    format!(
                        "frame at byte {offset} of {} is not a transaction frame",
                        segment_path.display()
                    )
                })?
                .with_context(|| {
                    format!(
                        "frame at byte {offset} of {} is not a transaction",
                        segment_path.display()
                    )
                })?;
            let mutations: Vec<Value> = serde_json::from_slice(&json).with_context(|| {
                format!(
                    "frame at byte {offset} of {} is not a transaction",
                    segment_path.display()
                )
            })?;
            for mutation in &mutations {
                let id_of = |variant: &str| {
                    mutation
                        .get(variant)
                        .and_then(|m| m.get("id"))
                        .and_then(Value::as_str)
                };
                if let Some(id) = id_of("Document") {
                    live.insert(id.to_string());
                } else if let Some(id) = id_of("DeleteDocument") {
                    live.remove(id);
                }
            }
            offset = frame_end;
        }
    }
    Ok(live)
}

fn missing_documents(expected: &BTreeSet<String>, core: &WarmCore) -> Vec<String> {
    expected
        .iter()
        .filter(|id| !core.has_document(id))
        .cloned()
        .collect()
}

pub(crate) fn inspect_segment(
    data: &[u8],
    path: &Path,
    verify_checksums: bool,
) -> Result<(usize, usize)> {
    let mut offset = 0usize;
    let mut frames = 0usize;
    while offset + HEADER_SIZE <= data.len() {
        let expected_crc = u32::from_le_bytes(data[offset..offset + 4].try_into()?);
        let raw_len = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
        let comp_len = u32::from_le_bytes(data[offset + 8..offset + 12].try_into()?) as usize;
        if raw_len == 0 && comp_len == 0 {
            if is_log_end(&data[offset..offset + HEADER_SIZE]) {
                return Ok((frames, offset));
            }
            bail!(
                "invalid end-of-log header in {} at byte {}",
                path.display(),
                offset
            );
        }
        if raw_len == 0 || raw_len > MAX_RAW_FRAME_SIZE || comp_len == 0 || comp_len > data.len() {
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
    let temp = parent.join(format!(".warm-migrate-{}-{nonce}", std::process::id()));
    let result = (|| -> Result<StoreInspection> {
        copy_tree(source, &temp)?;
        let inspection = inspect_store(&temp, true)?;
        // Copied files are flushed; supported platforms also sync directory entries.
        std::fs::rename(&temp, destination).with_context(|| {
            format!(
                "failed to atomically publish migration {} -> {}",
                temp.display(),
                destination.display()
            )
        })?;
        super::durable_file::sync_directory(parent)?;
        Ok(StoreInspection {
            store_path: destination.display().to_string(),
            ..inspection
        })
    })();
    if result.is_err() && temp.exists() {
        // Best effort: the migration error is what the caller needs, not a cleanup failure.
        let _ = std::fs::remove_dir_all(&temp);
    }
    result
}

pub(crate) fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
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
            super::durable_file::copy_file_synced(&entry.path(), &target)?;
        } else {
            bail!("migration refuses special file {}", entry.path().display());
        }
    }
    super::durable_file::sync_directory(destination)
}

/// Re-encode original JSONL source documents into a new, atomically published store.
pub fn reencode_documents(
    input: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    dimensions: u32,
    config: super::WarmConfig,
) -> Result<StoreInspection> {
    use std::io::{BufRead, BufReader, Read};
    let destination = destination.as_ref();
    if destination.exists() {
        bail!("re-encoding destination already exists");
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temporary = tempfile::Builder::new()
        .prefix(".warm-reencode-")
        .tempdir_in(parent)?;
    let mut source = BufReader::new(File::open(input.as_ref())?);
    {
        let warm = super::WarmCore::new(
            dimensions,
            Some(temporary.path().display().to_string()),
            Some(config),
        )?;
        let mut line_number = 0;
        loop {
            let mut line = String::new();
            let bytes = source
                .by_ref()
                .take(16 * 1024 * 1024 + 1)
                .read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            line_number += 1;
            if bytes > 16 * 1024 * 1024 {
                bail!("JSONL line {line_number} exceeds 16 MiB");
            }
            if line.trim().is_empty() {
                continue;
            }
            let document: super::documents::DocumentInput = serde_json::from_str(&line)
                .with_context(|| format!("invalid source document on line {line_number}"))?;
            warm.memorize_document(document)
                .with_context(|| format!("cannot re-encode line {line_number}"))?;
        }
        warm.flush()?;
    }
    let inspection = inspect_store(temporary.path(), true)?;
    super::durable_file::sync_directory(temporary.path())?;
    std::fs::rename(temporary.path(), destination)?;
    super::durable_file::sync_directory(parent)?;
    Ok(StoreInspection {
        store_path: destination.display().to_string(),
        ..inspection
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::documents::DocumentInput;
    use crate::core::entangled::EntangledHVec;
    use crate::core::WarmCore;

    /// `verify` cross-checks the opened store against the log: a healthy store reports every
    /// live document present, and a logged document the store cannot resolve is named.
    #[test]
    fn verify_cross_checks_logged_documents() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = root.path().join("store");
        {
            let warm = WarmCore::new(256, Some(store.display().to_string()), None)?;
            for n in 0..3 {
                warm.memorize_document(DocumentInput {
                    id: format!("doc{n}"),
                    text: format!("document {n} about harbors and ledgers"),
                    source_uri: None,
                    version: None,
                    metadata: None,
                    chunk_words: None,
                    overlap_words: None,
                    store_text: None,
                    embeddings: None,
                    title_weight: None,
                })?;
            }
            assert!(warm.delete_document("doc1")?);
            warm.flush()?;
        }
        let verified = verify_store(&store, None)?;
        assert!(verified
            .manifests_verified
            .iter()
            .any(|m| m == "store.json"));
        assert!(verified
            .manifests_verified
            .iter()
            .any(|m| m == "format.json"));
        let documents = verified
            .documents
            .expect("plaintext stores are cross-checked");
        assert_eq!(
            (documents.checked, documents.expected, documents.present),
            (true, Some(2), 2)
        );
        assert!(documents.missing.is_empty());

        let core = WarmCore::new(256, Some(store.display().to_string()), None)?;
        let expected: BTreeSet<String> = ["doc0", "doc2", "doc9"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            missing_documents(&expected, &core),
            vec!["doc9".to_string()]
        );
        Ok(())
    }

    #[test]
    fn migration_copy_preserves_read_only_files_and_excludes_the_lock() -> Result<()> {
        let root = tempfile::tempdir()?;
        let source = root.path().join("source");
        let target = root.path().join("target");
        std::fs::create_dir_all(source.join("nested"))?;
        let original = source.join("nested/data");
        std::fs::write(&original, b"migration data")?;
        #[cfg(windows)]
        std::fs::write(source.join("nested/data:metadata"), b"alternate stream")?;
        std::fs::write(source.join(LOCK_FILE), b"process lock")?;
        let writable = std::fs::metadata(&original)?.permissions();
        let mut read_only = writable.clone();
        read_only.set_readonly(true);
        std::fs::set_permissions(&original, read_only)?;

        let result = copy_tree(&source, &target);
        let copied = target.join("nested/data");
        let source_read_only = std::fs::metadata(&original)?.permissions().readonly();
        let copied_read_only = std::fs::metadata(&copied).map(|m| m.permissions().readonly());
        std::fs::set_permissions(&original, writable.clone())?;
        if copied.exists() {
            std::fs::set_permissions(&copied, writable)?;
        }

        result?;
        assert_eq!(std::fs::read(copied)?, b"migration data");
        #[cfg(windows)]
        assert_eq!(
            std::fs::read(target.join("nested/data:metadata"))?,
            b"alternate stream"
        );
        assert!(source_read_only && copied_read_only?);
        assert!(!target.join(LOCK_FILE).exists());
        Ok(())
    }

    #[test]
    fn inspect_verify_and_migrate_roundtrip() -> Result<()> {
        let root = tempfile::tempdir()?;
        let source = root.path().join("source");
        {
            let warm = WarmCore::new(256, Some(source.display().to_string()), None)?;
            warm.memorize("one".into(), EntangledHVec::new_deterministic(256, 1))?;
        }

        let inspected = inspect_store(&source, false)?;
        assert_eq!(inspected.frame_count, 1);
        assert!(!inspected.checksums_verified);
        let verified = inspect_store(&source, true)?;
        assert!(verified.checksums_verified);
        let destination = root.path().join("destination");
        let migrated = migrate_store(&source, &destination)?;
        assert_eq!(migrated.frame_count, 1);
        let reopened = WarmCore::new(256, Some(destination.display().to_string()), None)?;
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
            let warm = WarmCore::new(256, Some(source.display().to_string()), None)?;
            warm.memorize("one".into(), EntangledHVec::new_deterministic(256, 1))?;
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
