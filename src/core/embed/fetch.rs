// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The one place this module touches the network (feature `local-embed-download`): fetches the
//! pinned upstream files of a registry model, checks each against its sha256 and size, converts
//! them to a pack and checks the pack against its pinned sha256 before an atomic rename into the
//! cache. Nothing unverified is ever left under the final name.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};

use super::pack::{self, hex};
use super::registry::ModelSpec;

/// Replaces the Hugging Face host, for mirrors (the path layout must match).
pub const ENV_MIRROR: &str = "HMS_EMBED_MIRROR";

fn download_file(url: &str, dest: &Path, want_sha: &str, max: u64) -> Result<()> {
    let mut resp = ureq::get(url)
        .call()
        .with_context(|| format!("download {url}"))?;
    let mut reader = resp.body_mut().with_config().limit(max + 1).reader();
    let mut file =
        std::fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("read {url}"))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        ensure!(total <= max, "{url} is larger than the pinned {max} bytes");
        hash.update(&buf[..n]);
        file.write_all(&buf[..n])?;
    }
    file.sync_all()?;
    let got = hex(&hash.finalize());
    ensure!(
        got.eq_ignore_ascii_case(want_sha),
        "sha256 mismatch for {url}: expected {want_sha}, downloaded {got}; refusing to use it"
    );
    Ok(())
}

pub fn download_pack(spec: &ModelSpec, cache: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(cache)
        .with_context(|| format!("create cache directory {}", cache.display()))?;
    let work = cache.join(format!(".fetch-{}-{}", spec.id, std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work)?;
    let result = (|| -> Result<PathBuf> {
        let base = std::env::var(ENV_MIRROR)
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "https://huggingface.co".into());
        let base = base.trim_end_matches('/');
        for f in spec.files {
            let url = format!("{base}/{}/resolve/{}/{}", spec.repo, spec.revision, f.name);
            download_file(&url, &work.join(f.name), f.sha256, f.size)?;
        }
        let bytes = spec.convert(&work)?;
        let got = pack::sha256_hex(&bytes);
        if !got.eq_ignore_ascii_case(spec.pack_sha256) {
            bail!(
                "converted pack for {} has sha256 {got}, the registry pins {}; the upstream files verified, so this build's converter differs from the one that produced the pin",
                spec.id,
                spec.pack_sha256
            );
        }
        let tmp = cache.join(format!(".{}.{}.tmp", spec.pack_name(), std::process::id()));
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        let dest = cache.join(spec.pack_name());
        std::fs::rename(&tmp, &dest).with_context(|| format!("install {}", dest.display()))?;
        Ok(dest)
    })();
    let _ = std::fs::remove_dir_all(&work);
    result
}
