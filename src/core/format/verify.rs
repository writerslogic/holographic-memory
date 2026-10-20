// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Offline verification of a store directory. Read-only: it never opens the engine unless
//! `deep` is set, and then only on a private copy, because opening a store writes to it
//! (end marker, `format.json`, `documents.text`, the lock file).

use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde_json::Value;

use super::scan::{self, PayloadKind, Replay, SegmentEnd, TextHeader};
use super::{
    Report, Severity, TenantReport, CURRENT, SALT_LEN, STORAGE_FORMAT_CURRENT,
    STORAGE_FORMAT_OLDEST, STORE_SCHEMA_VERSION,
};
use crate::core::storage::{FormatManifest, FORMAT_MAGIC, LEGACY_SEGMENT_SIZE, SEGMENT_SIZE};

/// Directory of the arena inside a store.
const ARENA_DIRECTORY: &str = "vectors_data.bin";

/// What `verify_path` does beyond the default offline checks.
#[derive(Debug, Clone, Default)]
pub struct VerifyOptions {
    /// Environment variable holding the passphrase of an encrypted store. Without it the
    /// frames of an encrypted store are checked for framing and CRC32 only.
    pub passphrase_env: Option<String>,
    /// Also open a private copy of the store with the engine and compare what it loaded
    /// with what the journal says. Needs the disk space of one copy.
    pub deep: bool,
}

/// Verifies a store directory, or a server data directory (`tenants/<hex name>/` stores).
/// Damage is reported as findings; only an unreadable path is an `Err`.
pub fn verify_path(path: &Path, options: &VerifyOptions) -> Result<Report> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
    if !metadata.is_dir() {
        bail!("{} is not a directory", path.display());
    }
    let mut report = Report::new(path);
    let is_store = path.join("store.json").exists() || path.join(ARENA_DIRECTORY).exists();
    let tenants = path.join("tenants");
    if !is_store && tenants.is_dir() {
        let mut names: Vec<_> = std::fs::read_dir(&tenants)?
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for dir in names {
            let name = decode_tenant(&dir);
            if name.is_none() {
                report.add(
                    Severity::Info,
                    "tenant-name",
                    format!("tenants/{dir}"),
                    None,
                    "directory name is not the hex encoding of a UTF-8 tenant name",
                );
            }
            let sub = verify_store(&tenants.join(&dir), options)?;
            report.tenants.push(TenantReport {
                name: name.unwrap_or(dir),
                report: sub,
            });
        }
        if report.tenants.is_empty() {
            report.add(
                Severity::Info,
                "no-tenants",
                "tenants",
                None,
                "no tenant stores",
            );
        }
        report.checks.push("data-directory");
        report.finish();
        return Ok(report);
    }
    if !is_store {
        report.add(
            Severity::Error,
            "not-a-store",
            ".",
            None,
            "neither store.json, vectors_data.bin nor a tenants directory is present",
        );
        report.finish();
        return Ok(report);
    }
    verify_store(path, options)
}

fn decode_tenant(dir: &str) -> Option<String> {
    if dir.is_empty() || !dir.len().is_multiple_of(2) || !dir.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return None;
    }
    let bytes: Option<Vec<u8>> = (0..dir.len() / 2)
        .map(|i| u8::from_str_radix(&dir[2 * i..2 * i + 2], 16).ok())
        .collect();
    String::from_utf8(bytes?).ok()
}

/// What the verifier needs to know about the store after reading `store.json`.
struct Schema {
    dimensions: Option<u64>,
    encrypted: bool,
    meaning_enabled: bool,
    raw: Value,
}

fn verify_store(store: &Path, options: &VerifyOptions) -> Result<Report> {
    let mut report = Report::new(store);
    let schema = check_schema(store, &mut report);
    let arena = check_arena(store, &schema, options, &mut report)?;
    check_text_segment(store, &schema, &mut report);
    check_indices(store, &arena, &mut report);
    check_ancillary(store, &schema, &mut report);
    if options.deep && report.findings.iter().all(|f| f.severity < Severity::Error) {
        if let Some(replay) = &arena.replay {
            if let Err(error) = deep_check(store, &schema, replay, options, &mut report) {
                report.add(
                    Severity::Error,
                    "deep-open",
                    ".",
                    None,
                    format!("engine cross-check failed: {error:#}"),
                );
            }
        } else {
            report.add(
                Severity::Info,
                "deep-skipped",
                ".",
                None,
                "payloads were not authenticated, so there is no journal state to cross-check",
            );
        }
    }
    report.finish();
    Ok(report)
}

fn check_schema(store: &Path, report: &mut Report) -> Schema {
    let mut schema = Schema {
        dimensions: None,
        encrypted: false,
        meaning_enabled: false,
        raw: Value::Null,
    };
    let path = store.join("store.json");
    if !path.exists() {
        if store.join(ARENA_DIRECTORY).exists() {
            report.add(
                Severity::Unsupported,
                "pre-schema-store",
                "store.json",
                None,
                "no store.json: written before stores recorded their embedding schema. The \
                 engine refuses to open it; rebuild with `hms-admin reencode` from the source \
                 documents",
            );
        } else {
            report.add(
                Severity::Error,
                "missing-file",
                "store.json",
                None,
                "store.json is missing",
            );
        }
        return schema;
    }
    let bytes = match crate::core::durable_file::read_manifest(&path) {
        Ok((bytes, verified)) => {
            if verified {
                report.checks.push("store.json checksum");
            } else {
                report.add(
                    Severity::Info,
                    "no-checksum-sidecar",
                    "store.json",
                    None,
                    "no store.json.crc32 (written by a release before checksum sidecars)",
                );
            }
            bytes
        }
        Err(error) => {
            report.add(
                Severity::Error,
                "manifest-checksum",
                "store.json",
                None,
                format!("{error:#}"),
            );
            match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(_) => return schema,
            }
        }
    };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            report.add(
                Severity::Error,
                "schema-invalid",
                "store.json",
                None,
                error.to_string(),
            );
            return schema;
        }
    };
    let version = value.get("version").and_then(Value::as_u64);
    report.versions.store_schema = version.map(|v| v as u32);
    match version {
        Some(v) if v == u64::from(STORE_SCHEMA_VERSION) => {}
        Some(v) if v > u64::from(STORE_SCHEMA_VERSION) => report.add(
            Severity::Unsupported,
            "schema-newer",
            "store.json",
            None,
            format!(
                "store.json version {v} is newer than the supported version {STORE_SCHEMA_VERSION}"
            ),
        ),
        other => report.add(
            Severity::Unsupported,
            "schema-version",
            "store.json",
            None,
            format!("store.json version {other:?} is not a version this build reads"),
        ),
    }
    schema.dimensions = value.get("dimensions").and_then(Value::as_u64);
    if schema.dimensions.is_none_or(|d| d == 0 || d > 1_000_000) {
        report.add(
            Severity::Error,
            "schema-invalid",
            "store.json",
            None,
            "dimensions is missing or outside 1..=1000000",
        );
    }
    report.versions.dimensions = schema.dimensions;
    schema.encrypted = value
        .get("encrypted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    schema.meaning_enabled = value
        .get("meaning_enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    report.versions.encrypted = Some(schema.encrypted);
    if let Some(name) = value.get("text_encoder").and_then(Value::as_str) {
        report.versions.text_encoder = Some(name.to_owned());
        if name.parse::<crate::core::encoding::TextEncoder>().is_err() {
            report.add(
                Severity::Unsupported,
                "unknown-encoder",
                "store.json",
                None,
                format!("text encoder {name:?} is not known to this build"),
            );
        }
    }
    if let Some(name) = value.get("vector_encoder").and_then(Value::as_str) {
        report.versions.vector_encoder = Some(name.to_owned());
        if name != "signed-projection-v2" {
            report.add(
                Severity::Unsupported,
                "unknown-encoder",
                "store.json",
                None,
                format!("vector encoder {name:?} is not known to this build"),
            );
        }
    }
    if value.get("metric").and_then(Value::as_str) != Some("jaccard") {
        report.add(
            Severity::Unsupported,
            "unknown-metric",
            "store.json",
            None,
            "metric is not jaccard",
        );
    }
    schema.raw = value;
    schema
}

/// What the arena check learned that later checks need.
struct ArenaState {
    /// Name of the generation directory and journal length: the engine's checkpoint.
    checkpoint: Option<(String, u64)>,
    replay: Option<Replay>,
}

fn list_segments(dir: &Path) -> Result<Vec<usize>> {
    let mut ids = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        if let Some(id) = name
            .to_string_lossy()
            .strip_prefix("seg_")
            .and_then(|v| v.strip_suffix(".bin"))
            .and_then(|v| v.parse::<usize>().ok())
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

/// The directory holding the live segments of the arena at `arena` (the one `CURRENT` names).
pub fn resolve_generation(arena: &Path) -> Result<PathBuf> {
    crate::core::durable_file::arena_generation(arena)
}

#[cfg(feature = "security")]
type Cipher = crate::core::security::EncryptionManager;
#[cfg(not(feature = "security"))]
type Cipher = std::convert::Infallible;

fn open_cipher(
    store: &Path,
    schema: &Schema,
    options: &VerifyOptions,
    report: &mut Report,
) -> Option<Cipher> {
    if !schema.encrypted {
        return None;
    }
    let salt = store.join("encryption.salt");
    match std::fs::metadata(&salt) {
        Ok(m) if m.len() == SALT_LEN as u64 => {}
        Ok(m) => {
            report.add(
                Severity::Error,
                "salt-invalid",
                "encryption.salt",
                None,
                format!("expected {SALT_LEN} bytes, found {}", m.len()),
            );
            return None;
        }
        Err(_) => {
            report.add(
                Severity::Error,
                "missing-file",
                "encryption.salt",
                None,
                "the store is encrypted but encryption.salt is missing: the key cannot be derived",
            );
            return None;
        }
    }
    let Some(variable) = &options.passphrase_env else {
        report.add(
            Severity::Info,
            "payloads-not-authenticated",
            ARENA_DIRECTORY,
            None,
            "encrypted store: frames were checked for framing and CRC32 only; pass \
             --passphrase-env to authenticate and decode every payload",
        );
        return None;
    };
    #[cfg(feature = "security")]
    {
        let Ok(passphrase) = std::env::var(variable) else {
            report.add(
                Severity::Info,
                "payloads-not-authenticated",
                ARENA_DIRECTORY,
                None,
                format!("environment variable {variable} is not set"),
            );
            return None;
        };
        match Cipher::new(&passphrase, store) {
            Ok(cipher) => Some(cipher),
            Err(error) => {
                report.add(
                    Severity::Error,
                    "salt-invalid",
                    "encryption.salt",
                    None,
                    format!("{error:#}"),
                );
                None
            }
        }
    }
    #[cfg(not(feature = "security"))]
    {
        let _ = variable;
        report.add(
            Severity::Info,
            "payloads-not-authenticated",
            ARENA_DIRECTORY,
            None,
            "this build lacks the `security` feature, so encrypted payloads cannot be decrypted",
        );
        None
    }
}

fn check_arena(
    store: &Path,
    schema: &Schema,
    options: &VerifyOptions,
    report: &mut Report,
) -> Result<ArenaState> {
    let mut state = ArenaState {
        checkpoint: None,
        replay: None,
    };
    let base = store.join(ARENA_DIRECTORY);
    if !base.is_dir() {
        if schema.raw != Value::Null {
            report.add(
                Severity::Error,
                "missing-file",
                ARENA_DIRECTORY,
                None,
                "arena directory is missing",
            );
        }
        return Ok(state);
    }
    let generation = match crate::core::durable_file::arena_generation(&base) {
        Ok(path) => path,
        Err(error) => {
            report.add(
                Severity::Error,
                "pointer-invalid",
                "vectors_data.bin/CURRENT",
                None,
                format!("{error:#}"),
            );
            return Ok(state);
        }
    };
    let generation_name = generation
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    // Compaction staging directories and superseded generations are leftovers, not state.
    for entry in std::fs::read_dir(&base)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("gen-") && entry.path() != generation {
            report.add(
                Severity::Info,
                "orphan-generation",
                format!("{ARENA_DIRECTORY}/{name}"),
                None,
                "a generation directory CURRENT does not name (an interrupted or superseded \
                 compaction); safe to delete",
            );
        }
    }

    let manifest = check_manifest(&base, &generation, report);
    let Some(manifest) = manifest else {
        return Ok(state);
    };
    if report.versions.storage_format != Some(1) {
        report.versions.storage_format = Some(manifest.version);
    }
    let segment_size = if manifest.version == 3 {
        SEGMENT_SIZE
    } else {
        LEGACY_SEGMENT_SIZE
    };
    if manifest.segment_size != segment_size {
        report.add(
            Severity::Error,
            "segment-size",
            format!("{ARENA_DIRECTORY}/format.json"),
            None,
            format!(
                "segment_size {} is not the {segment_size} that format {} requires",
                manifest.segment_size, manifest.version
            ),
        );
        return Ok(state);
    }

    let ids = list_segments(&generation)?;
    if ids.is_empty() {
        report.add(
            Severity::Error,
            "segment-gap",
            ARENA_DIRECTORY,
            None,
            "the arena has no segments",
        );
        return Ok(state);
    }
    if ids.iter().enumerate().any(|(i, &id)| i != id) {
        report.add(
            Severity::Error,
            "segment-gap",
            ARENA_DIRECTORY,
            None,
            format!("segments are not numbered contiguously from 0: {ids:?}"),
        );
        return Ok(state);
    }
    report.segments = ids.len();

    let cipher = open_cipher(store, schema, options, report);
    let authenticated = !schema.encrypted || cipher.is_some();
    let mut replay = Replay::default();
    let mut transaction_formats = BTreeSet::new();
    let mut payload_problems = 0usize;
    let mut final_end = 0usize;
    let mut zero_terminated = false;
    for (n, &id) in ids.iter().enumerate() {
        let is_final = n + 1 == ids.len();
        let relative = format!("{ARENA_DIRECTORY}/seg_{id}.bin");
        let path = generation.join(format!("seg_{id}.bin"));
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let length = file.metadata()?.len();
        check_segment_size(&relative, length, is_final, &manifest, report);
        if length == 0 {
            report.add(
                Severity::Error,
                "segment-size",
                &relative,
                None,
                "segment file is empty",
            );
            continue;
        }
        // SAFETY: read-only mapping of a file this process does not modify; the verifier
        // holds no reference into it past this loop iteration.
        let map = unsafe { Mmap::map(&file)? };
        let scan = scan::scan_segment(&map, is_final, |frame, payload| {
            if payload_problems >= 20 {
                return;
            }
            let offset = frame.offset as u64;
            #[cfg(feature = "security")]
            let plain: Vec<u8>;
            let payload = if schema.encrypted {
                #[cfg(feature = "security")]
                match cipher.as_ref().map(|c| c.decrypt(payload)) {
                    Some(Ok(bytes)) => {
                        plain = bytes;
                        &plain[..]
                    }
                    Some(Err(error)) => {
                        payload_problems += 1;
                        report.add(
                            Severity::Error,
                            "auth-failed",
                            &relative,
                            Some(offset),
                            format!("{error:#} (wrong passphrase, or the frame was altered)"),
                        );
                        return;
                    }
                    None => return,
                }
                #[cfg(not(feature = "security"))]
                {
                    let _ = &cipher;
                    return;
                }
            } else {
                payload
            };
            match scan::classify_payload(payload) {
                PayloadKind::Transaction(json) => {
                    transaction_formats.insert("HMS-TXN-2".to_owned());
                    if schema.encrypted && !authenticated {
                        return;
                    }
                    match scan::check_transaction(json, schema.dimensions, &mut replay) {
                        Ok(_) => {}
                        Err(message) => {
                            payload_problems += 1;
                            report.add(
                                Severity::Error,
                                "payload-structure",
                                &relative,
                                Some(offset),
                                message,
                            );
                        }
                    }
                }
                PayloadKind::OtherTransactionVersion(version) => {
                    payload_problems += 1;
                    let newer = version.parse::<u32>().map_or(true, |v| v > 2);
                    transaction_formats.insert(format!("HMS-TXN-{version}"));
                    report.add(
                        Severity::Unsupported,
                        if newer {
                            "transaction-newer"
                        } else {
                            "transaction-older"
                        },
                        &relative,
                        Some(offset),
                        format!(
                            "journal frame format HMS-TXN-{version} is {} than the supported \
                             HMS-TXN-2",
                            if newer { "newer" } else { "older" }
                        ),
                    );
                }
                PayloadKind::Unrecognized => {
                    payload_problems += 1;
                    report.add(
                        Severity::Error,
                        "payload-unrecognized",
                        &relative,
                        Some(offset),
                        if schema.encrypted {
                            "decrypted payload has no journal magic"
                        } else {
                            "payload has no journal magic (an encrypted frame in a store marked \
                             unencrypted, or damage)"
                        },
                    );
                }
            }
        });
        report.frames += scan.frames.len();
        match &scan.end {
            SegmentEnd::Marker | SegmentEnd::Exhausted => {}
            SegmentEnd::ZeroHeader => zero_terminated = true,
            SegmentEnd::TornTail { reason } => report.add(
                Severity::Warning,
                "torn-tail",
                &relative,
                Some(scan.end_offset as u64),
                format!(
                    "the last append did not complete ({reason}); the engine truncates the log \
                     to offset {} on open",
                    scan.end_offset
                ),
            ),
            SegmentEnd::Corrupt {
                reason,
                committed_at,
            } => report.add(
                Severity::Error,
                "frame-corrupt",
                &relative,
                Some(scan.end_offset as u64),
                match committed_at {
                    Some(next) => format!(
                        "frame fails its checks ({reason}) but committed data follows at \
                         offset {next}: damage inside committed data"
                    ),
                    None => format!(
                        "frame fails its checks ({reason}) inside a segment that was complete"
                    ),
                },
            ),
        }
        if is_final {
            final_end = scan.end_offset;
        }
    }
    if zero_terminated {
        report.add(
            Severity::Info,
            "legacy-zero-terminator",
            ARENA_DIRECTORY,
            None,
            "a segment ends in an all-zero header, the form written before end markers; the \
             engine rewrites it as an end marker on open",
        );
    }
    report.versions.transaction_formats = transaction_formats.into_iter().collect();
    report.checks.push("frames");
    state.checkpoint = Some((
        generation_name,
        ((ids.len() - 1) as u64) * manifest.segment_size as u64 + final_end as u64,
    ));
    report.payloads_authenticated = authenticated;
    if authenticated {
        report.mutations = replay.mutations;
        report.live_documents = Some(replay.documents.len());
        report.live_vectors = Some(replay.vector_total());
        report.memory_stamped_documents =
            Some(replay.documents.values().filter(|d| d.stamped).count());
        if payload_problems == 0 {
            report.checks.push("payloads");
        }
        state.replay = Some(replay);
    }
    Ok(state)
}

fn check_manifest(base: &Path, generation: &Path, report: &mut Report) -> Option<FormatManifest> {
    let top = base.join("format.json");
    let path = if top.exists() {
        top
    } else {
        generation.join("format.json")
    };
    let relative = format!("{ARENA_DIRECTORY}/format.json");
    if !path.exists() {
        if generation.join("seg_0.bin").exists() {
            report.versions.storage_format = Some(1);
            report.add(
                Severity::Info,
                "pre-manifest-store",
                &relative,
                None,
                "no format.json: a format-1 store; the engine reads it as format 2 and writes \
                 the manifest on open",
            );
            return Some(FormatManifest {
                magic: FORMAT_MAGIC.to_owned(),
                version: 2,
                segment_size: LEGACY_SEGMENT_SIZE,
            });
        }
        report.add(
            Severity::Error,
            "missing-file",
            &relative,
            None,
            "format.json is missing",
        );
        return None;
    }
    let bytes = match crate::core::durable_file::read_manifest(&path) {
        Ok((bytes, verified)) => {
            if verified {
                report.checks.push("format.json checksum");
            } else {
                report.add(
                    Severity::Info,
                    "no-checksum-sidecar",
                    &relative,
                    None,
                    "no format.json.crc32 (written by a release before checksum sidecars)",
                );
            }
            bytes
        }
        Err(error) => {
            report.add(
                Severity::Error,
                "manifest-checksum",
                &relative,
                None,
                format!("{error:#}"),
            );
            std::fs::read(&path).ok()?
        }
    };
    let manifest: FormatManifest = match serde_json::from_slice(&bytes) {
        Ok(manifest) => manifest,
        Err(error) => {
            report.add(
                Severity::Error,
                "manifest-invalid",
                &relative,
                None,
                error.to_string(),
            );
            return None;
        }
    };
    if manifest.magic != FORMAT_MAGIC {
        report.add(
            Severity::Unsupported,
            "unknown-magic",
            &relative,
            None,
            format!(
                "manifest magic {:?} is not {FORMAT_MAGIC:?}",
                manifest.magic
            ),
        );
        return None;
    }
    if manifest.version > STORAGE_FORMAT_CURRENT {
        report.versions.storage_format = Some(manifest.version);
        report.add(
            Severity::Unsupported,
            "storage-format-newer",
            &relative,
            None,
            format!(
                "storage format {} is newer than the supported format {STORAGE_FORMAT_CURRENT}",
                manifest.version
            ),
        );
        return None;
    }
    if manifest.version < STORAGE_FORMAT_OLDEST {
        report.versions.storage_format = Some(manifest.version);
        report.add(
            Severity::Unsupported,
            "storage-format-unknown",
            &relative,
            None,
            format!(
                "storage format {} was never written by a release",
                manifest.version
            ),
        );
        return None;
    }
    Some(manifest)
}

fn check_segment_size(
    relative: &str,
    length: u64,
    is_final: bool,
    manifest: &FormatManifest,
    report: &mut Report,
) {
    let declared = manifest.segment_size as u64;
    if length > declared {
        report.add(
            Severity::Error,
            "segment-size",
            relative,
            None,
            format!("segment is {length} bytes, longer than the declared {declared}"),
        );
    } else if manifest.version >= 4 && is_final {
        if !length.is_multiple_of(SEGMENT_SIZE as u64) || length < SEGMENT_SIZE as u64 {
            report.add(
                Severity::Warning,
                "segment-size-unaligned",
                relative,
                None,
                format!(
                    "the active segment of a format-{} store is allocated in {SEGMENT_SIZE}-byte \
                     steps; this one is {length} bytes (a copy that was trimmed? the engine \
                     extends it with zeros on open)",
                    manifest.version
                ),
            );
        }
    } else if length != declared {
        let severity = if is_final {
            Severity::Warning
        } else {
            Severity::Error
        };
        report.add(
            severity,
            "segment-size",
            relative,
            None,
            format!(
                "segment is {length} bytes, the declared size is {declared}{}",
                if is_final {
                    " (the engine extends the active segment with zeros on open)"
                } else {
                    ""
                }
            ),
        );
    }
}

fn check_text_segment(store: &Path, schema: &Schema, report: &mut Report) {
    let path = store.join("documents.text");
    let Ok(mut file) = File::open(&path) else {
        return;
    };
    let mut header = [0u8; super::TEXT_SEGMENT_HEADER_LEN];
    let read = std::io::Read::read(&mut file, &mut header).unwrap_or(0);
    match scan::parse_text_header(&header[..read]) {
        TextHeader::Versioned { version, flags } => {
            report.versions.text_segment = Some(version);
            let encrypted = flags & 1 == 1;
            let note = if version > CURRENT.text_segment {
                "newer than this build writes"
            } else if encrypted != schema.encrypted {
                "its encryption flag differs from the store's"
            } else {
                "a cache the engine rebuilds from the journal on every open"
            };
            report.add(
                Severity::Info,
                "text-segment-cache",
                "documents.text",
                None,
                format!("version {version}, flags {flags:#04x}: {note}"),
            );
        }
        TextHeader::Headerless => report.add(
            Severity::Info,
            "text-segment-cache",
            "documents.text",
            None,
            "no HMST header (a format-1 text segment); the engine truncates it on open",
        ),
    }
}

fn check_indices(store: &Path, arena: &ArenaState, report: &mut Report) {
    let path = store.join("indices.json");
    if !path.exists() {
        return;
    }
    let bytes = match crate::core::durable_file::read_manifest(&path) {
        Ok((bytes, verified)) => {
            if verified {
                report.checks.push("indices.json checksum");
            }
            bytes
        }
        Err(error) => {
            report.add(
                Severity::Error,
                "manifest-checksum",
                "indices.json",
                None,
                format!("{error:#}"),
            );
            return;
        }
    };
    let meta = match scan::parse_indices(&bytes) {
        Ok(meta) => meta,
        Err(error) => {
            report.add(
                Severity::Warning,
                "index-cache-invalid",
                "indices.json",
                None,
                format!("{error}; the engine ignores it and rebuilds the indices"),
            );
            return;
        }
    };
    let fresh = arena.checkpoint.as_ref() == Some(&meta.checkpoint);
    report.add(
        Severity::Info,
        if fresh {
            "index-cache-fresh"
        } else {
            "index-cache-stale"
        },
        "indices.json",
        None,
        if fresh {
            "bound to the current journal end".to_owned()
        } else {
            format!(
                "bound to {:?} at byte {}, not the current journal end; the engine rebuilds \
                 the indices",
                meta.checkpoint.0, meta.checkpoint.1
            )
        },
    );
    let multi = meta.shard_count > 1;
    for (i, shard) in meta.shards.iter().enumerate() {
        for (flag, crc, stem) in [
            (shard.nsg, shard.nsg_crc, "nsg_index"),
            (shard.ivf, shard.ivf_crc, "ivf_index"),
        ] {
            let name = if multi {
                format!("{stem}_{i}.bin")
            } else {
                format!("{stem}.bin")
            };
            let file = store.join(&name);
            if !flag || !file.exists() {
                continue;
            }
            match (std::fs::read(&file), crc) {
                (Ok(bytes), Some(expected)) if crc32fast::hash(&bytes) == expected => {
                    report.checks.push("index cache checksum");
                }
                (Ok(_), _) => report.add(
                    Severity::Warning,
                    "index-cache-invalid",
                    &name,
                    None,
                    "checksum differs from indices.json; the engine discards this cache",
                ),
                (Err(error), _) => report.add(
                    Severity::Warning,
                    "index-cache-invalid",
                    &name,
                    None,
                    error.to_string(),
                ),
            }
        }
    }
}

fn check_ancillary(store: &Path, schema: &Schema, report: &mut Report) {
    const KNOWN: [&str; 17] = [
        "store.json",
        "store.json.crc32",
        "indices.json",
        "indices.json.crc32",
        "documents.text",
        "encryption.salt",
        "shard_meta.json",
        "audit.bin",
        "hms_signing.key",
        "hms_provenance.key",
        "provenance_log.jsonl",
        "provenance_log.head",
        "provenance_log.statuslist",
        "kel.json",
        "connection_graph.log",
        ".hms.lock",
        ARENA_DIRECTORY,
    ];
    if let Ok(bytes) = std::fs::read(store.join("audit.bin")) {
        let audit = scan::scan_audit(&bytes);
        report.checks.push("audit.bin");
        if audit.trailing_bytes != 0 {
            report.add(
                Severity::Warning,
                "audit-torn-tail",
                "audit.bin",
                Some((audit.entries * super::AUDIT_ENTRY_LEN) as u64),
                format!(
                    "{} trailing bytes after {} whole entries (an interrupted append)",
                    audit.trailing_bytes, audit.entries
                ),
            );
        }
        if audit.unknown_ops != 0 {
            report.add(
                Severity::Warning,
                "audit-unknown-op",
                "audit.bin",
                None,
                format!(
                    "{} entries have an unknown operation byte; readers skip them",
                    audit.unknown_ops
                ),
            );
        }
    }
    if let Ok(bytes) = std::fs::read(store.join("shard_meta.json")) {
        let count = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|v| v.get("shard_count").and_then(Value::as_u64));
        if count.is_none() {
            report.add(
                Severity::Warning,
                "shard-meta-invalid",
                "shard_meta.json",
                None,
                "shard_count is missing; the engine starts with a single shard",
            );
        }
    }
    if let Ok(metadata) = std::fs::metadata(store.join("hms_signing.key")) {
        if metadata.len() != 32 {
            report.add(
                Severity::Error,
                "key-invalid",
                "hms_signing.key",
                None,
                format!("expected 32 bytes, found {}", metadata.len()),
            );
        }
    }
    if !schema.encrypted && store.join("encryption.salt").exists() {
        report.add(
            Severity::Info,
            "salt-unused",
            "encryption.salt",
            None,
            "present but store.json says the store is not encrypted",
        );
    }
    if let Ok(entries) = std::fs::read_dir(store) {
        let mut unknown = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let cache = (name.starts_with("nsg_index") || name.starts_with("ivf_index"))
                && name.ends_with(".bin");
            if !cache && !KNOWN.contains(&name.as_str()) {
                unknown.push(name);
            }
        }
        unknown.sort();
        for name in unknown {
            report.add(
                Severity::Info,
                "unrecognized-file",
                name,
                None,
                "not a file this format version defines; ignored by the engine",
            );
        }
    }
}

fn copy_store(source: &Path, destination: &Path) -> Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == crate::core::store_lock::LOCK_FILE {
            continue;
        }
        let target: PathBuf = destination.join(&name);
        let kind = entry.file_type()?;
        if kind.is_dir() {
            copy_store(&entry.path(), &target)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Opens a private copy with the engine and compares what it loaded with the journal replay
/// computed here, independently of the engine.
fn deep_check(
    store: &Path,
    schema: &Schema,
    replay: &Replay,
    options: &VerifyOptions,
    report: &mut Report,
) -> Result<()> {
    let dimensions = u32::try_from(schema.dimensions.context("store.json lacks dimensions")?)?;
    let mut config = crate::core::HmsConfig::default();
    if let Some(name) = schema.raw["text_encoder"].as_str() {
        config.text_encoder = name.parse().map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    config.meaning.enabled = schema.meaning_enabled;
    config.embedding_space = serde_json::from_value(schema.raw["embedding_space"].clone())
        .context("store.json embedding_space")?;
    if schema.encrypted {
        config.security.encryption_enabled = true;
        config.security.encryption_passphrase_env = options.passphrase_env.clone();
    }
    let scratch = tempfile::tempdir()?;
    let copy = scratch.path().join("store");
    copy_store(store, &copy)?;
    let core =
        crate::core::HmsCore::new(dimensions, Some(copy.display().to_string()), Some(config))?;
    let missing: Vec<&String> = replay
        .documents
        .keys()
        .filter(|id| !core.has_document(id))
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "documents in the journal that the engine did not load: {missing:?}"
    );
    if !schema.meaning_enabled {
        anyhow::ensure!(
            core.vector_count() as usize == replay.vector_total(),
            "engine holds {} vectors, the journal replays to {}",
            core.vector_count(),
            replay.vector_total()
        );
        anyhow::ensure!(
            core.relation_count() == replay.relations.len(),
            "engine holds {} relations, the journal replays to {}",
            core.relation_count(),
            replay.relations.len()
        );
    }
    report.checks.push("engine open");
    Ok(())
}
