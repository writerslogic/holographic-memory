// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Operator endpoints: snapshots and restore, compaction, key reload, tenant listing.
//! All need the `admin` scope; the key and tenant-listing endpoints also need an all-tenant key.

use std::io::Write;
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
            let mut builder = tar::Builder::new(std::io::BufWriter::with_capacity(
                256 * 1024,
                ChannelWriter(tx.clone()),
            ));
            builder.mode(tar::HeaderMode::Deterministic);
            builder.follow_symlinks(false);
            builder.append_dir_all(".", &dir)?;
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
    holographic_memory::core::admin::verify_store(dir, None)
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
            // Unix fsyncs through any handle (an archive may unpack read-only files); Windows
            // flushes file buffers only through a write handle.
            std::fs::OpenOptions::new()
                .read(cfg!(unix))
                .write(!cfg!(unix))
                .open(entry.path())?
                .sync_all()?;
        }
    }
    crate::vindex::sync_dir(dir)
}

/// Verify the staged store and swap it in for the tenant's; the staging directory is removed
/// on failure. Returns the snapshot taken of the replaced store.
fn install_staged(state: &AppState, tenant: &str, staging: Staging) -> Result<String, ApiError> {
    verify(staging.path())?;
    let staged = staging.take();
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

/// Unpack a tar of regular files and directories into `dst`, refusing links, devices, absolute
/// or parent-relative paths, and oversized archives.
fn unpack_archive(file: std::fs::File, dst: &Path, max_bytes: u64) -> Result<(), ApiError> {
    let invalid = |why: &str| ApiError::bad(format!("invalid snapshot archive: {why}"));
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
        total = total.saturating_add(entry.header().size().unwrap_or(0));
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
                Some(Err(_)) => return Err(ApiError::bad("request body could not be read")),
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
        file.sync_all().await.map_err(ApiError::internal)?;
        Ok(())
    }
    .await;
    spooled?;
    // The blocking task owns the upload from here, so a client leaving mid-restore cannot
    // remove the archive while it is being unpacked.
    let previous = blocking(permit, move || {
        let file = std::fs::File::open(&upload.0).map_err(ApiError::internal)?;
        unpack_archive(file, staging.path(), max)?;
        drop(upload);
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
