// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Operator endpoints: snapshots and restore, compaction, key reload, tenant listing.
//! All need the `admin` scope; the key and tenant-listing endpoints also need an all-tenant key.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path as UrlPath, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use http_body_util::BodyExt;
use tokio::io::AsyncWriteExt;

use crate::auth::Scope;
use crate::error::{ApiError, ApiResult};
use crate::request::*;
use crate::tenants::{copy_store, tenant_from_hex, AppState};
use crate::types::*;

pub async fn create_snapshot(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Admin)?;
    let permit = acquire(&state).await?;
    let info = blocking(permit, move || state.create_snapshot(&tenant)).await?;
    Ok(json_response(
        StatusCode::CREATED,
        &body(&Snapshot::from(&info)),
    ))
}

pub async fn list_snapshots(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Admin)?;
    let permit = acquire(&state).await?;
    let list = blocking(permit, move || state.list_snapshots(&tenant)).await?;
    let snapshots = list.iter().map(Snapshot::from).collect();
    Ok(json_response(
        StatusCode::OK,
        &body(&Snapshots { snapshots }),
    ))
}

pub async fn delete_snapshot(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<String>,
    req: Request,
) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Admin)?;
    let permit = acquire(&state).await?;
    blocking(permit, move || state.delete_snapshot(&tenant, &id)).await?;
    Ok(json_response(
        StatusCode::OK,
        &body(&Deleted { deleted: true }),
    ))
}

struct ChannelWriter(tokio::sync::mpsc::Sender<std::io::Result<Bytes>>);

impl Write for ChannelWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .blocking_send(Ok(Bytes::copy_from_slice(buf)))
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "client went away"))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn append_snapshot<W: Write>(
    builder: &mut tar::Builder<W>,
    root: &Path,
    path: &Path,
    sizes: &BTreeMap<PathBuf, u64>,
) -> std::io::Result<()> {
    let mut pending = vec![path.to_path_buf()];
    while let Some(path) = pending.pop() {
        let relative = path.strip_prefix(root).map_err(std::io::Error::other)?;
        let name = if relative.as_os_str().is_empty() {
            Path::new(".")
        } else {
            relative
        };
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.is_dir() {
            builder.append_dir(name, &path)?;
            for entry in std::fs::read_dir(&path)? {
                pending.push(entry?.path());
            }
        } else if metadata.is_file() {
            let size = sizes.get(&path).copied().unwrap_or(metadata.len());
            let file = std::fs::File::open(&path)?;
            if file.metadata()?.len() < size {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "snapshot file shrank",
                ));
            }
            let mut header = tar::Header::new_gnu();
            header.set_metadata_in_mode(&metadata, tar::HeaderMode::Deterministic);
            header.set_size(size);
            builder.append_data(&mut header, name, file.take(size))?;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "snapshot contains a non-regular entry",
            ));
        }
    }
    Ok(())
}

/// `GET /api/v1/snapshots/{id}/archive`: the snapshot as an uncompressed tar stream, the
/// off-box backup format accepted by `PUT /api/v1/restore`.
pub async fn download_snapshot(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<String>,
    req: Request,
) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Admin)?;
    let permit = acquire(&state).await?;
    let dir = state.snapshot_path(&tenant, &id)?;
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let result = (|| -> std::io::Result<()> {
            let sizes = writerslogic_warm::core::admin::snapshot_archive_sizes(&dir)
                .map_err(std::io::Error::other)?;
            let mut builder = tar::Builder::new(std::io::BufWriter::with_capacity(
                256 * 1024,
                ChannelWriter(tx.clone()),
            ));
            builder.mode(tar::HeaderMode::Deterministic);
            builder.follow_symlinks(false);
            append_snapshot(&mut builder, &dir, &dir, &sizes)?;
            builder.into_inner()?.flush()
        })();
        if let Err(error) = result {
            // The stream is already underway; make the client see a failed transfer.
            let _ = tx.blocking_send(Err(error));
        }
    });
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    });
    let mut response = Response::new(Body::from_stream(stream));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-tar"),
    );
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{id}.tar\"")) {
        headers.insert(header::CONTENT_DISPOSITION, v);
    }
    Ok(response)
}

fn verify(dir: &Path) -> Result<(), ApiError> {
    writerslogic_warm::core::admin::verify_store(dir, None)
        .map(|_| ())
        .map_err(|e| {
            crate::log_event!(Warn, "restore_verification_failed", "detail" => format!("{e:#}"));
            ApiError::bad(
                "the snapshot is not a valid, intact store; the current store is unchanged",
            )
        })
}

/// Make a fresh staging directory next to the data, removed on drop unless taken.
struct Staging(Option<PathBuf>);

impl Staging {
    fn new(state: &AppState) -> Result<Self, ApiError> {
        let root = state.staging_root();
        std::fs::create_dir_all(&root).map_err(ApiError::internal)?;
        let dir = root.join(format!("{:016x}", rand::random::<u64>()));
        Ok(Self(Some(dir)))
    }

    fn path(&self) -> &Path {
        self.0.as_deref().expect("not taken")
    }

    fn take(mut self) -> PathBuf {
        self.0.take().expect("not taken")
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if let Some(dir) = &self.0 {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// The spooled restore upload, removed on drop: a client that goes away mid-upload drops the
/// handler, and the partial archive must not stay behind in the staging directory.
struct Upload(PathBuf);

impl Drop for Upload {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Make every file and directory under `dir` durable, so the store renamed into place after a
/// crash holds the unpacked bytes rather than whatever reached the disk.
fn sync_tree(dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            sync_tree(&entry.path())?;
        } else if kind.is_file() {
            crate::tenants::sync_staged_file(&entry.path())?;
        }
    }
    crate::vindex::sync_dir(dir)
}

fn restore_stage(tenant: &str, stage: &str) {
    crate::log_event!(Info, "restore_stage", "tenant" => tenant, "stage" => stage);
}

/// Verify the staged store and swap it in for the tenant's; the staging directory is removed
/// on failure. Returns the snapshot taken of the replaced store.
fn install_staged(state: &AppState, tenant: &str, staging: Staging) -> Result<String, ApiError> {
    restore_stage(tenant, "verify");
    verify(staging.path())?;
    let staged = staging.take();
    restore_stage(tenant, "install");
    let result = state.restore_staged(tenant, &staged);
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staged);
    }
    result
}

/// `POST /api/v1/snapshots/{id}/restore`: replace the tenant's store with a snapshot. The
/// store being replaced is kept as snapshot `pre-restore-<time>`.
pub async fn restore_snapshot(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<String>,
    req: Request,
) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Admin)?;
    let permit = acquire(&state).await?;
    let previous = blocking(permit, move || {
        let source = state.snapshot_path(&tenant, &id)?;
        let staging = Staging::new(&state)?;
        copy_store(&source, staging.path()).map_err(ApiError::internal)?;
        install_staged(&state, &tenant, staging)
    })
    .await?;
    Ok(json_response(
        StatusCode::OK,
        &body(&Restored {
            restored: true,
            previous_snapshot: previous,
        }),
    ))
}

const MAX_ARCHIVE_METADATA_BYTES: u64 = 64 * 1024;

// The normal tar iterator buffers extension records before yielding an entry. Inspect raw
// headers first, seeking over file payloads and retaining only bounded PAX metadata.
fn validate_archive_metadata(file: &mut std::fs::File, max_bytes: u64) -> Result<(), ApiError> {
    let invalid = |why: &str| ApiError::bad(format!("invalid snapshot archive: {why}"));
    let length = file.metadata().map_err(ApiError::internal)?.len();
    let (mut offset, mut total, mut count) = (0u64, 0u64, 0usize);
    let mut pending_size = None;
    let mut extensions = 0u8;
    loop {
        file.seek(SeekFrom::Start(offset))
            .map_err(ApiError::internal)?;
        let mut archive = tar::Archive::new(&mut *file);
        let mut entries = archive
            .entries()
            .map_err(|_| invalid("not a tar file"))?
            .raw(true);
        let Some(entry) = entries.next() else {
            if extensions != 0 {
                return Err(invalid("extension has no following entry"));
            }
            break;
        };
        let mut entry = entry.map_err(|_| invalid("unreadable entry"))?;
        let kind = entry.header().entry_type();
        let extension = if entry.header().as_gnu().is_some() || entry.header().as_ustar().is_some()
        {
            if kind.is_gnu_longname() {
                1
            } else if kind.is_gnu_longlink() {
                2
            } else if kind.is_pax_local_extensions() {
                4
            } else {
                0
            }
        } else {
            0
        };
        let size = if extension != 0 {
            if entry.size() > MAX_ARCHIVE_METADATA_BYTES {
                return Err(invalid("extended metadata exceeds 64 KiB"));
            }
            if extensions & extension != 0 {
                return Err(invalid("duplicate extension for one entry"));
            }
            extensions |= extension;
            if kind.is_pax_local_extensions() {
                if let Some(fields) = entry
                    .pax_extensions()
                    .map_err(|_| invalid("bad PAX metadata"))?
                {
                    for field in fields {
                        let field = field.map_err(|_| invalid("bad PAX metadata"))?;
                        if field.key_bytes() == b"size" {
                            pending_size = Some(
                                field
                                    .value()
                                    .ok()
                                    .and_then(|value| value.parse::<u64>().ok())
                                    .ok_or_else(|| invalid("bad PAX size"))?,
                            );
                            break;
                        }
                    }
                }
            }
            entry.size()
        } else {
            if !(kind.is_file() || kind.is_dir()) {
                return Err(invalid("only regular files and directories are allowed"));
            }
            extensions = 0;
            let size = pending_size.take().unwrap_or(entry.size());
            count += 1;
            total = total.saturating_add(size);
            if count > 100_000 || total > max_bytes {
                return Err(invalid("archive is larger than this server accepts"));
            }
            size
        };
        offset = size
            .checked_add(511)
            .and_then(|size| offset.checked_add(512)?.checked_add(size & !511))
            .filter(|end| *end <= length)
            .ok_or_else(|| invalid("truncated or oversized entry"))?;
    }
    file.rewind().map_err(ApiError::internal)?;
    Ok(())
}

/// Unpack a tar of regular files and directories into `dst`, refusing links, devices, absolute
/// or parent-relative paths, and oversized archives.
fn unpack_archive(mut file: std::fs::File, dst: &Path, max_bytes: u64) -> Result<(), ApiError> {
    let invalid = |why: &str| ApiError::bad(format!("invalid snapshot archive: {why}"));
    validate_archive_metadata(&mut file, max_bytes)?;
    std::fs::create_dir_all(dst).map_err(ApiError::internal)?;
    let mut archive = tar::Archive::new(file);
    let (mut total, mut count) = (0u64, 0usize);
    for entry in archive.entries().map_err(|_| invalid("not a tar file"))? {
        let mut entry = entry.map_err(|_| invalid("unreadable entry"))?;
        let kind = entry.header().entry_type();
        if !(kind.is_file() || kind.is_dir()) {
            return Err(invalid("only regular files and directories are allowed"));
        }
        let path = entry.path().map_err(|_| invalid("bad path"))?.into_owned();
        if path
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
        {
            return Err(invalid("paths must be relative without \"..\""));
        }
        count += 1;
        total = total.saturating_add(entry.size());
        if count > 100_000 || total > max_bytes {
            return Err(invalid("archive is larger than this server accepts"));
        }
        if !entry
            .unpack_in(dst)
            .map_err(|_| invalid("cannot unpack entry"))?
        {
            return Err(invalid("entry escapes the destination"));
        }
    }
    Ok(())
}

/// `PUT /api/v1/restore` with a tar body (as produced by the archive endpoint): replace the
/// tenant's store. The upload is size-limited by `--max-restore-bytes`, verified, then swapped in.
pub async fn restore_upload(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Admin)?;
    let permit = acquire(&state).await?;
    let staging = Staging::new(&state)?;
    let upload = Upload(staging.path().with_extension("tar"));
    let max = state.cfg.max_restore_bytes;
    let idle = Duration::from_secs(state.cfg.body_timeout_seconds);
    restore_stage(&tenant, "receive");
    let spooled: Result<(), ApiError> = async {
        if let Some(parent) = upload.0.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(ApiError::internal)?;
        }
        let mut file = tokio::fs::File::create(&upload.0)
            .await
            .map_err(ApiError::internal)?;
        let mut body = req.into_body();
        let mut total = 0u64;
        loop {
            let frame = tokio::time::timeout(idle, body.frame())
                .await
                .map_err(|_| {
                    ApiError::new(
                        StatusCode::REQUEST_TIMEOUT,
                        "request_timeout",
                        "upload timed out",
                    )
                })?;
            match frame {
                None => break,
                Some(Err(_)) => return Err(ApiError::invalid_body()),
                Some(Ok(frame)) => {
                    if let Ok(data) = frame.into_data() {
                        total += data.len() as u64;
                        if total > max {
                            return Err(ApiError::new(
                                StatusCode::PAYLOAD_TOO_LARGE,
                                "payload_too_large",
                                format!("archive exceeds {max} bytes"),
                            ));
                        }
                        file.write_all(&data).await.map_err(ApiError::internal)?;
                    }
                }
            }
        }
        crate::log_event!(Info, "restore_stage", "tenant" => tenant, "stage" => "sync_upload", "bytes" => total);
        file.sync_all().await.map_err(ApiError::internal)?;
        Ok(())
    }
    .await;
    spooled?;
    // The blocking task owns the upload from here, so a client leaving mid-restore cannot
    // remove the archive while it is being unpacked.
    let previous = blocking(permit, move || {
        let file = std::fs::File::open(&upload.0).map_err(ApiError::internal)?;
        restore_stage(&tenant, "unpack");
        unpack_archive(file, staging.path(), max)?;
        drop(upload);
        restore_stage(&tenant, "sync_store");
        sync_tree(staging.path()).map_err(ApiError::internal)?;
        install_staged(&state, &tenant, staging)
    })
    .await?;
    Ok(json_response(
        StatusCode::OK,
        &body(&Restored {
            restored: true,
            previous_snapshot: previous,
        }),
    ))
}

pub async fn compaction_status(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Read)?;
    let permit = acquire(&state).await?;
    let interval = state.cfg.compaction_interval_seconds;
    let status = blocking(permit, move || state.compaction_status(&tenant)).await?;
    Ok(json_response(
        StatusCode::OK,
        &body(&CompactionStatusBody {
            scheduled_interval_seconds: interval,
            status: CompactionView::from(&status),
        }),
    ))
}

pub async fn run_compaction(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let (_, tenant) = admit(&state, &req, Scope::Admin)?;
    let permit = acquire(&state).await?;
    let status = blocking(permit, move || state.compact(&tenant, "manual")).await?;
    Ok(json_response(
        StatusCode::OK,
        &body(&CompactionRun {
            status: CompactionView::from(&status),
        }),
    ))
}

pub async fn reload_keys(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let principal = principal(&req)?;
    principal.require(Scope::Admin)?;
    principal.require_global()?;
    let keys = state.auth.reload().map_err(|e| {
        ApiError::bad(format!(
            "keys not reloaded, previous keys stay in force: {e:#}"
        ))
    })?;
    crate::log_event!(Info, "keys_reloaded", "keys" => keys);
    Ok(json_response(StatusCode::OK, &body(&KeysReloaded { keys })))
}

pub async fn reload_tls(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let principal = principal(&req)?;
    principal.require(Scope::Admin)?;
    principal.require_global()?;
    let Some(tls) = state.tls.get() else {
        return Err(ApiError::bad("TLS is not enabled on this server"));
    };
    tls.reload().map_err(|e| {
        ApiError::bad(format!(
            "certificate not reloaded, the previous one stays in use: {e:#}"
        ))
    })?;
    crate::log_event!(Info, "tls_reloaded");
    Ok(json_response(
        StatusCode::OK,
        &body(&TlsReloaded { reloaded: true }),
    ))
}

pub async fn list_tenants(State(state): State<Arc<AppState>>, req: Request) -> ApiResult {
    let principal = principal(&req)?;
    principal.require(Scope::Admin)?;
    principal.require_global()?;
    let root = state.cfg.data_dir.join("tenants");
    let mut tenants = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&root) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            tenants.extend(tenant_from_hex(&name));
        }
    }
    tenants.sort();
    Ok(json_response(
        StatusCode::OK,
        &body(&Tenants {
            open: state.open_tenant_count(),
            tenants,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_archive_omits_only_unused_capacity_and_restores_vectors() -> anyhow::Result<()> {
        use writerslogic_warm::{EntangledHVec, WarmCore};

        let root = tempfile::tempdir()?;
        let source = root.path().join("snapshot");
        let vector = EntangledHVec::new_deterministic(256, 7);
        {
            let core = WarmCore::new(256, Some(source.display().to_string()), None)?;
            core.memorize("kept".into(), vector.clone())?;
            core.flush()?;
        }
        std::fs::remove_file(source.join(".hms.lock"))?;
        let segment = source.join("vectors_data.bin/seg_0.bin");
        let file = std::fs::OpenOptions::new().write(true).open(&segment)?;
        file.set_len(128 * 1024 * 1024)?;
        file.sync_all()?;
        drop(file);
        let mut auxiliary = std::fs::File::create(source.join("sparse-auxiliary.bin"))?;
        auxiliary.set_len(512 * 1024)?;
        auxiliary.seek(SeekFrom::End(-1))?;
        auxiliary.write_all(&[0x7f])?;
        auxiliary.sync_all()?;
        drop(auxiliary);
        let sizes = writerslogic_warm::core::admin::snapshot_archive_sizes(&source)?;
        assert_eq!(sizes.get(&segment), Some(&(64 * 1024 * 1024)));
        let archive = root.path().join("snapshot.tar");
        {
            let mut builder = tar::Builder::new(std::fs::File::create(&archive)?);
            builder.mode(tar::HeaderMode::Deterministic);
            builder.follow_symlinks(false);
            append_snapshot(&mut builder, &source, &source, &sizes)?;
            builder.finish()?;
        }
        assert!(std::fs::metadata(&archive)?.len() < 65 * 1024 * 1024);
        assert_eq!(std::fs::metadata(&segment)?.len(), 128 * 1024 * 1024);
        for entry in tar::Archive::new(std::fs::File::open(&archive)?).entries()? {
            let kind = entry?.header().entry_type();
            assert!(kind.is_file() || kind.is_dir());
        }
        let restored = root.path().join("restored");
        unpack_archive(std::fs::File::open(archive)?, &restored, 256 * 1024 * 1024).unwrap();
        let auxiliary = std::fs::read(restored.join("sparse-auxiliary.bin"))?;
        assert_eq!(auxiliary.len(), 512 * 1024);
        assert_eq!(auxiliary.last(), Some(&0x7f));
        assert_eq!(
            std::fs::metadata(restored.join("vectors_data.bin/seg_0.bin"))?.len(),
            64 * 1024 * 1024
        );
        assert_eq!(
            writerslogic_warm::core::admin::inspect_store(&restored, true)?.frame_count,
            1
        );
        let core = WarmCore::new(256, Some(restored.display().to_string()), None)?;
        assert_eq!(core.query(&vector, 1)?[0].id, "kept");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_archive_rejects_links() -> std::io::Result<()> {
        let root = tempfile::tempdir()?;
        let file = root.path().join("data");
        let link = root.path().join("link");
        std::fs::write(&file, b"private")?;
        std::os::unix::fs::symlink(file, &link)?;
        let mut builder = tar::Builder::new(Vec::new());
        let error =
            append_snapshot(&mut builder, root.path(), &link, &BTreeMap::new()).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        Ok(())
    }

    fn archive_with_pax_comment(path: &Path, size: usize) {
        let mut builder = tar::Builder::new(std::fs::File::create(path).unwrap());
        let prefix = format!("{size} comment=");
        let payload = format!("{prefix}{}\n", "x".repeat(size - prefix.len() - 1));
        let mut header = tar::Header::new_ustar();
        header.set_entry_type(tar::EntryType::new(b'x'));
        header.set_size(payload.len() as u64);
        header.set_cksum();
        builder
            .append_data(&mut header, "pax", payload.as_bytes())
            .unwrap();
        let mut header = tar::Header::new_ustar();
        header.set_size(3);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "data", b"abc".as_slice())
            .unwrap();
        builder.finish().unwrap();
    }

    #[test]
    fn archive_extended_metadata_is_bounded_before_extraction() {
        let dir = tempfile::tempdir().unwrap();
        for size in [65535, 65536, 65537] {
            let path = dir.path().join(format!("{size}.tar"));
            let dst = dir.path().join(size.to_string());
            archive_with_pax_comment(&path, size);
            let result = unpack_archive(std::fs::File::open(path).unwrap(), &dst, 1024 * 1024);
            if size <= 65536 {
                result.unwrap();
                assert_eq!(std::fs::read(dst.join("data")).unwrap(), b"abc");
            } else {
                assert_eq!(result.unwrap_err().status, StatusCode::UNPROCESSABLE_ENTITY);
                assert!(!dst.join("data").exists());
            }
        }
    }

    fn raw_entry(file: &mut std::fs::File, kind: u8, name: &str, declared: u64, bytes: &[u8]) {
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::new(kind));
        header.set_path(name).unwrap();
        header.set_mode(0o644);
        header.set_size(declared);
        header.set_cksum();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(bytes).unwrap();
        let padding = (512 - bytes.len() % 512) % 512;
        file.write_all(&vec![0; padding]).unwrap();
    }

    #[test]
    fn archive_preflight_preserves_pax_sizes_across_gnu_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.tar");
        let mut file = std::fs::File::create(&path).unwrap();
        raw_entry(&mut file, b'x', "pax", 10, b"10 size=3\n");
        let long_name = "a".repeat(140);
        let encoded_name = format!("{long_name}\0");
        raw_entry(
            &mut file,
            b'L',
            "././@LongLink",
            encoded_name.len() as u64,
            encoded_name.as_bytes(),
        );
        raw_entry(&mut file, b'0', "placeholder", 0, b"abc");
        raw_entry(&mut file, b'0', "next", 2, b"de");
        file.write_all(&[0; 1024]).unwrap();
        drop(file);
        let staged = dir.path().join("staged");
        unpack_archive(std::fs::File::open(&path).unwrap(), &staged, 5).unwrap();
        assert_eq!(std::fs::read(staged.join(long_name)).unwrap(), b"abc");
        assert_eq!(std::fs::read(staged.join("next")).unwrap(), b"de");
        let rejected = dir.path().join("too-small");
        assert!(unpack_archive(std::fs::File::open(path).unwrap(), &rejected, 4).is_err());
        assert!(!rejected.exists());
    }

    #[test]
    fn oversized_extension_headers_are_rejected_without_reading_their_payload() {
        let dir = tempfile::tempdir().unwrap();
        for kind in *b"xLK" {
            let path = dir.path().join(format!("{kind}.tar"));
            let mut file = std::fs::File::create(&path).unwrap();
            raw_entry(&mut file, kind, "metadata", 65537, &[]);
            drop(file);
            let error = unpack_archive(
                std::fs::File::open(path).unwrap(),
                &dir.path().join("out"),
                u64::MAX,
            )
            .unwrap_err();
            assert!(
                error.message.contains("extended metadata exceeds 64 KiB"),
                "{}",
                error.message
            );
        }
    }

    #[test]
    fn archive_preflight_rejects_malformed_metadata_and_sparse_entries() {
        let dir = tempfile::tempdir().unwrap();
        for case in [
            "duplicate",
            "dangling",
            "bad-size",
            "overflow",
            "truncated",
            "sparse",
        ] {
            let path = dir.path().join(format!("{case}.tar"));
            let mut file = std::fs::File::create(&path).unwrap();
            match case {
                "duplicate" => {
                    raw_entry(&mut file, b'x', "pax", 10, b"10 size=3\n");
                    raw_entry(&mut file, b'x', "pax", 10, b"10 size=3\n");
                    raw_entry(&mut file, b'0', "data", 3, b"abc");
                }
                "dangling" => raw_entry(&mut file, b'x', "pax", 10, b"10 size=3\n"),
                "bad-size" => {
                    raw_entry(&mut file, b'x', "pax", 10, b"10 size=x\n");
                    raw_entry(&mut file, b'0', "data", 3, b"abc");
                }
                "overflow" => raw_entry(&mut file, b'0', "data", u64::MAX, &[]),
                "truncated" => raw_entry(&mut file, b'0', "data", 1, &[]),
                "sparse" => raw_entry(&mut file, b'S', "data", 0, &[]),
                _ => unreachable!(),
            }
            drop(file);
            let dst = dir.path().join(case);
            let error =
                unpack_archive(std::fs::File::open(path).unwrap(), &dst, u64::MAX).unwrap_err();
            assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY, "{case}");
            assert!(!dst.exists(), "{case}");
        }
    }

    #[test]
    fn uploaded_readonly_files_are_synced_without_changing_attributes() {
        let dir = tempfile::tempdir().unwrap();
        let archive_path = dir.path().join("snapshot.tar");
        let mut archive = tar::Builder::new(std::fs::File::create(&archive_path).unwrap());
        for (name, mode, content) in [
            ("nested/deployment.json", 0o444, b"readonly".as_slice()),
            ("nested/writable.json", 0o644, b"writable".as_slice()),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(mode);
            header.set_cksum();
            archive.append_data(&mut header, name, content).unwrap();
        }
        archive.finish().unwrap();
        drop(archive);
        let staged = dir.path().join("staged");
        unpack_archive(std::fs::File::open(archive_path).unwrap(), &staged, 1024).unwrap();
        let readonly = staged.join("nested/deployment.json");
        let permissions = std::fs::metadata(&readonly).unwrap().permissions();
        assert!(permissions.readonly());
        #[cfg(windows)]
        assert_eq!(
            std::fs::OpenOptions::new()
                .write(true)
                .open(&readonly)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::PermissionDenied,
        );
        let result = sync_tree(&staged);
        let after = std::fs::metadata(&readonly).unwrap().permissions();
        #[cfg(windows)]
        {
            let mut writable = permissions.clone();
            writable.set_readonly(false);
            std::fs::set_permissions(&readonly, writable).unwrap();
        }
        result.unwrap();
        assert_eq!(permissions, after);
        assert_eq!(std::fs::read(&readonly).unwrap(), b"readonly");
        assert_eq!(
            std::fs::read(staged.join("nested/writable.json")).unwrap(),
            b"writable"
        );
    }
}
