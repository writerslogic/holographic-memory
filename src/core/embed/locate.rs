// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Finding the pack for a model. Order for a registry model: `HMS_EMBED_PACK`, the model
//! directory (option or `HMS_EMBED_MODEL_DIR`), the cache, then a pinned download (only with
//! feature `local-embed-download` and never when offline).

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use super::registry::{self, ModelSpec};
use super::EmbedOptions;

/// A pack file to load directly for a registry model.
pub const ENV_PACK: &str = "HMS_EMBED_PACK";
/// A directory searched for ready packs before the cache (air-gapped installs, companion wheels).
pub const ENV_MODEL_DIR: &str = "HMS_EMBED_MODEL_DIR";
/// The download cache directory.
pub const ENV_CACHE: &str = "HMS_EMBED_CACHE";
/// `1`, `true` or `yes` forbids any network access.
pub const ENV_OFFLINE: &str = "HMS_EMBED_OFFLINE";

#[derive(Debug)]
pub(crate) struct Located {
    pub(crate) path: PathBuf,
    /// The hash the bytes must have (the registry pin, or one the caller supplied).
    pub(crate) sha256: Option<String>,
    pub(crate) from_cache: bool,
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

pub(crate) fn offline(options: &EmbedOptions) -> bool {
    options.offline.unwrap_or_else(|| {
        std::env::var(ENV_OFFLINE)
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

pub(crate) fn cache_dir(options: &EmbedOptions) -> Result<PathBuf> {
    if let Some(d) = options.cache_dir.clone().or_else(|| env_path(ENV_CACHE)) {
        return Ok(d);
    }
    let home = || env_path("HOME").or_else(|| env_path("USERPROFILE"));
    let base = if cfg!(target_os = "macos") {
        home().map(|h| h.join("Library/Caches"))
    } else if cfg!(windows) {
        env_path("LOCALAPPDATA").or_else(|| home().map(|h| h.join("AppData/Local")))
    } else {
        env_path("XDG_CACHE_HOME").or_else(|| home().map(|h| h.join(".cache")))
    };
    match base {
        Some(b) => Ok(b.join("holographic-memory").join("embed")),
        None => bail!("no cache directory: set {ENV_CACHE} to a writable directory"),
    }
}

pub(crate) fn is_path(model: &str) -> bool {
    model.contains(['/', '\\']) || model.ends_with(".hmsemb") || Path::new(model).is_file()
}

pub(crate) fn resolve(options: &EmbedOptions) -> Result<Located> {
    if is_path(&options.model) {
        let path = PathBuf::from(&options.model);
        if !path.is_file() {
            bail!("embedding pack {} does not exist", path.display());
        }
        return Ok(Located {
            path,
            sha256: options.sha256.clone(),
            from_cache: false,
        });
    }
    let spec = registry::find(&options.model)?;
    let pin = Some(spec.pack_sha256.to_string());
    if let Some(p) = env_path(ENV_PACK) {
        if !p.is_file() {
            bail!("{ENV_PACK}={} does not exist", p.display());
        }
        return Ok(Located {
            path: p,
            sha256: pin,
            from_cache: false,
        });
    }
    if let Some(dir) = options
        .model_dir
        .clone()
        .or_else(|| env_path(ENV_MODEL_DIR))
    {
        let p = dir.join(spec.pack_name());
        if p.is_file() {
            return Ok(Located {
                path: p,
                sha256: pin,
                from_cache: false,
            });
        }
    }
    let cache = cache_dir(options)?;
    let cached = cache.join(spec.pack_name());
    if cached.is_file() {
        return Ok(Located {
            path: cached,
            sha256: pin,
            from_cache: true,
        });
    }
    fetch_or_explain(spec, &cache, options)
}

#[cfg(feature = "local-embed-download")]
fn fetch_or_explain(spec: &ModelSpec, cache: &Path, options: &EmbedOptions) -> Result<Located> {
    if offline(options) {
        bail!("{}", missing_message(spec, cache, "offline mode is on"));
    }
    let path = super::fetch::download_pack(spec, cache)?;
    Ok(Located {
        path,
        sha256: Some(spec.pack_sha256.to_string()),
        from_cache: true,
    })
}

#[cfg(not(feature = "local-embed-download"))]
fn fetch_or_explain(spec: &ModelSpec, cache: &Path, options: &EmbedOptions) -> Result<Located> {
    let why = if offline(options) {
        "offline mode is on"
    } else {
        "this build has no downloader (feature local-embed-download)"
    };
    bail!("{}", missing_message(spec, cache, why))
}

pub(crate) fn missing_message(spec: &ModelSpec, cache: &Path, why: &str) -> String {
    format!(
        "embedding model {id:?} is not installed and {why}. Provide the pack {name} (sha256 {sha}) in one of these ways: \
         set {ENV_PACK} to its path; put it in a directory named by {ENV_MODEL_DIR}; or copy it to {cache}. \
         On a connected machine, `hms-embed fetch {id} --out DIR` creates it. \
         Unset {ENV_OFFLINE} to allow the one-time pinned download.",
        id = spec.id,
        name = spec.pack_name(),
        sha = spec.pack_sha256,
        cache = cache.display(),
    )
}
