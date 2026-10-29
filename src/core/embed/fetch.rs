// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The one place this module touches the network (feature `local-embed-download`): fetches the
//! pinned upstream files of a registry model, checks each against its sha256 and size, converts
//! them to a pack and checks the pack against its pinned sha256 before an atomic rename into the
//! cache. Nothing unverified is ever left under the final name.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
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

/// Installs `bytes` as `cache/name`, staged in `work` (a directory of `cache`, so the rename
/// stays on one filesystem) that the caller removes either way: a failed write or rename
/// leaves nothing in the cache.
fn install(work: &Path, cache: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
    let tmp = work.join(format!(".{name}.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    let dest = cache.join(name);
    std::fs::rename(&tmp, &dest).with_context(|| format!("install {}", dest.display()))?;
    crate::core::durable_file::sync_directory(cache)?;
    Ok(dest)
}

struct FetchCoordinator {
    active: std::sync::Mutex<std::collections::HashSet<PathBuf>>,
    ready: std::sync::Condvar,
}

fn coordinator() -> &'static FetchCoordinator {
    static STATE: std::sync::OnceLock<FetchCoordinator> = std::sync::OnceLock::new();
    STATE.get_or_init(|| FetchCoordinator {
        active: std::sync::Mutex::new(Default::default()),
        ready: std::sync::Condvar::new(),
    })
}

struct FetchLease(PathBuf);

impl FetchLease {
    fn acquire(path: PathBuf) -> Self {
        let state = coordinator();
        let mut active = state.active.lock().unwrap_or_else(|e| e.into_inner());
        while active.contains(&path) {
            active = state.ready.wait(active).unwrap_or_else(|e| e.into_inner());
        }
        active.insert(path.clone());
        Self(path)
    }
}

impl Drop for FetchLease {
    fn drop(&mut self) {
        let state = coordinator();
        state
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.0);
        state.ready.notify_all();
    }
}

fn build_pack(
    spec: &ModelSpec,
    cache: &Path,
    build: impl FnOnce(&Path) -> Result<Vec<u8>>,
) -> Result<PathBuf> {
    ensure!(
        !spec.id.is_empty()
            && spec
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b)),
        "invalid model cache id"
    );
    std::fs::create_dir_all(cache)
        .with_context(|| format!("create cache directory {}", cache.display()))?;
    let cache = cache.canonicalize()?;
    let dest = cache.join(spec.pack_name());
    let _lease = FetchLease::acquire(dest.clone());
    if dest.is_file() && pack::sha256_file(&dest)?.eq_ignore_ascii_case(spec.pack_sha256) {
        return Ok(dest);
    }
    let work = tempfile::Builder::new()
        .prefix(".hms-embed-fetch-")
        .tempdir_in(&cache)?;
    let bytes = build(work.path())?;
    let got = pack::sha256_hex(&bytes);
    ensure!(
        got.eq_ignore_ascii_case(spec.pack_sha256),
        "converted pack for {} has sha256 {got}, expected {}; refusing to install it",
        spec.id,
        spec.pack_sha256
    );
    install(work.path(), &cache, &spec.pack_name(), &bytes)
}

pub fn download_pack(spec: &ModelSpec, cache: &Path) -> Result<PathBuf> {
    build_pack(spec, cache, |work| {
        let base = std::env::var(ENV_MIRROR)
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "https://huggingface.co".into());
        let base = base.trim_end_matches('/');
        for f in spec.files {
            let url = format!("{base}/{}/resolve/{}/{}", spec.repo, spec.revision, f.name);
            download_file(&url, &work.join(f.name), f.sha256, f.size)?;
        }
        spec.convert(work)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_fetches_build_once_and_failures_release_the_reservation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let cache = tempfile::tempdir().unwrap();
        let bytes = super::super::testing::tiny_pack_bytes(
            super::super::Quantization::Int8,
            super::super::Pooling::Mean,
        );
        let sha = pack::sha256_hex(&bytes);
        let source = &super::super::MODELS[0];
        let spec = ModelSpec {
            id: "test",
            repo: "test",
            revision: "test",
            license: "MIT",
            pooling: source.pooling,
            query_prefix: "",
            passage_prefix: "",
            max_length: 64,
            quantization: source.quantization,
            int8_group: 16,
            files: &[],
            pack_sha256: Box::leak(sha.into_boxed_str()),
            pack_bytes: bytes.len() as u64,
        };
        assert!(build_pack(&spec, cache.path(), |_| anyhow::bail!("injected failure")).is_err());
        let calls = AtomicUsize::new(0);
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        build_pack(&spec, cache.path(), |_| {
                            calls.fetch_add(1, Ordering::SeqCst);
                            Ok(bytes.clone())
                        })
                        .unwrap()
                    })
                })
                .collect();
            for handle in handles {
                assert_eq!(
                    pack::sha256_file(&handle.join().unwrap()).unwrap(),
                    spec.pack_sha256
                );
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_failed_install_leaves_nothing_in_the_cache() {
        let cache = tempfile::tempdir().unwrap();
        let work = cache.path().join(".fetch-test");
        std::fs::create_dir(&work).unwrap();
        // A non-empty directory under the pack's name makes the rename fail.
        std::fs::create_dir_all(cache.path().join("m.pack/x")).unwrap();
        assert!(install(&work, cache.path(), "m.pack", b"pack").is_err());
        std::fs::remove_dir_all(&work).unwrap();
        let mut left: Vec<_> = std::fs::read_dir(cache.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        left.sort();
        assert_eq!(left, ["m.pack"]);

        std::fs::remove_dir_all(cache.path().join("m.pack")).unwrap();
        std::fs::create_dir(&work).unwrap();
        let dest = install(&work, cache.path(), "m.pack", b"pack").unwrap();
        assert_eq!(std::fs::read(dest).unwrap(), b"pack");
    }
}
