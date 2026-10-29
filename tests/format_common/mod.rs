// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Shared by `tests/format_*.rs`: locating, unpacking, packing and checking golden stores.
//!
//! A golden store is a directory under `tests/golden/<name>/` holding the store's files as
//! the writing code left them, except that every arena segment is truncated after its last
//! non-zero byte (segments are preallocated to 64 MiB or 1 GiB of zeros) and the lock file is
//! left out. `golden.json` records how it was made, what it declares, and what the current
//! code must find in it. Unpacking restores the zero tail from `padTo`.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use holographic_memory::core::{EmbeddingSpace, HmsConfig};
use holographic_memory::{EntangledHVec, HmsCore, SearchOptions};
use serde_json::{json, Value};

pub const PASSPHRASE: &str = "golden-passphrase-1";

/// Pretty JSON with a trailing newline.
pub fn json_file(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn golden_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

pub struct Golden {
    pub name: String,
    pub dir: PathBuf,
    pub meta: Value,
}

/// Every golden store committed under `tests/golden/`.
pub fn goldens() -> Result<Vec<Golden>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(golden_root())? {
        let dir = entry?.path();
        let meta_path = dir.join("golden.json");
        if dir.is_dir() && meta_path.exists() {
            found.push(Golden {
                name: dir.file_name().unwrap().to_string_lossy().into_owned(),
                meta: serde_json::from_slice(&std::fs::read(&meta_path)?)
                    .with_context(|| format!("parsing {}", meta_path.display()))?,
                dir,
            });
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(found)
}

impl Golden {
    pub fn str(&self, key: &str) -> Option<&str> {
        self.meta[key].as_str()
    }

    pub fn needs(&self, feature: &str) -> bool {
        self.meta["features"]
            .as_array()
            .is_some_and(|f| f.iter().any(|x| x == feature))
    }

    /// True if this build can create the golden's configuration (and so open the store).
    pub fn openable_here(&self) -> bool {
        (!self.needs("security") || cfg!(feature = "security"))
            && (!self.needs("experimental") || cfg!(feature = "experimental"))
    }

    /// Copies the golden into `dest` and restores the zero tail of every trimmed segment.
    pub fn unpack(&self, dest: &Path) -> Result<()> {
        copy_tree(&self.dir, dest, &["golden.json"])?;
        if let Some(pad) = self.meta["padTo"].as_object() {
            for (relative, length) in pad {
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(dest.join(relative))
                    .with_context(|| format!("padding {relative}"))?;
                file.set_len(length.as_u64().context("padTo length")?)?;
            }
        }
        Ok(())
    }

    /// The configuration the golden was written with.
    pub fn config(&self) -> Result<HmsConfig> {
        let mut config = HmsConfig::default();
        if let Some(name) = self.str("textEncoder") {
            config.text_encoder = name.parse()?;
        }
        if !self.meta["embeddingSpace"].is_null() {
            config.embedding_space = Some(serde_json::from_value::<EmbeddingSpace>(
                self.meta["embeddingSpace"].clone(),
            )?);
        }
        if self.meta["encrypted"] == true {
            config.security.encryption_enabled = true;
            config.security.encryption_passphrase = Some(PASSPHRASE.into());
        }
        config.meaning.enabled = self.meta["meaningEnabled"] == true;
        if self.meta["audit"] == true {
            config.security.audit_enabled = true;
        }
        Ok(config)
    }

    pub fn dimensions(&self) -> u32 {
        self.meta["dimensions"].as_u64().expect("dimensions") as u32
    }
}

/// Recursively copies `source` into `destination`, skipping the lock file and `skip` names at
/// the top level.
pub fn copy_tree(source: &Path, destination: &Path, skip: &[&str]) -> Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == ".hms.lock" || skip.contains(&name.as_str()) {
            continue;
        }
        let target = destination.join(&name);
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target, &[])?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Copies a store written by real code into a golden directory: segments are trimmed after
/// their last non-zero byte, and the original lengths go into `meta["padTo"]`.
pub fn pack(store: &Path, destination: &Path, mut meta: Value) -> Result<()> {
    ensure!(!destination.exists(), "{} exists", destination.display());
    let mut pad = serde_json::Map::new();
    pack_dir(store, destination, store, &mut pad)?;
    meta["padTo"] = Value::Object(pad);
    std::fs::write(destination.join("golden.json"), json_file(&meta)?)?;
    Ok(())
}

fn pack_dir(
    source: &Path,
    destination: &Path,
    root: &Path,
    pad: &mut serde_json::Map<String, Value>,
) -> Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == ".hms.lock" {
            continue;
        }
        let target = destination.join(&name);
        if entry.file_type()?.is_dir() {
            pack_dir(&entry.path(), &target, root, pad)?;
        } else if name.starts_with("seg_") && name.ends_with(".bin") {
            let bytes = std::fs::read(entry.path())?;
            let keep = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
            std::fs::write(&target, &bytes[..keep])?;
            let relative = entry
                .path()
                .strip_prefix(root)?
                .to_string_lossy()
                .into_owned();
            pad.insert(relative, json!(bytes.len()));
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// `(length, crc32)` of every file under `dir`, keyed by relative path.
pub fn file_fingerprints(dir: &Path) -> Result<BTreeMap<String, (u64, u32)>> {
    fn walk(dir: &Path, base: &Path, out: &mut BTreeMap<String, (u64, u32)>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                walk(&entry.path(), base, out)?;
            } else {
                let bytes = std::fs::read(entry.path())?;
                out.insert(
                    entry
                        .path()
                        .strip_prefix(base)?
                        .to_string_lossy()
                        .replace('\\', "/"),
                    (bytes.len() as u64, crc32fast::hash(&bytes)),
                );
            }
        }
        Ok(())
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out)?;
    Ok(out)
}

/// Checks a store opened by the current code against a golden's `expect` object.
pub fn check_expectations(core: &HmsCore, expect: &Value, dimensions: u32) -> Result<()> {
    if let Some(meaning) = expect.get("meaning") {
        ensure!(
            core.meaning_atom_count() as u64 == meaning["atoms"].as_u64().unwrap(),
            "atom count differs"
        );
        ensure!(
            core.meaning_composite_count() as u64 == meaning["composites"].as_u64().unwrap(),
            "composite count differs"
        );
        ensure!(
            core.meaning_triple_count() as u64 == meaning["triples"].as_u64().unwrap(),
            "triple count differs"
        );
    }
    for document in expect["documents"].as_array().into_iter().flatten() {
        let id = document["id"].as_str().context("document id")?;
        ensure!(core.has_document(id), "document {id} is missing");
        let text = document["text"].as_str().context("document text")?;
        let hits = core.search_documents(
            text,
            &SearchOptions {
                k: Some(64),
                candidate_limit: Some(64),
                document_ids: Some(vec![id.to_owned()]),
                ..Default::default()
            },
        )?;
        ensure!(
            !hits.is_empty(),
            "document {id} has no search hits for its own text"
        );
        for hit in &hits {
            ensure!(hit.document_id == id, "filter leaked {}", hit.document_id);
            ensure!(
                hit.version == document["version"].as_str().unwrap_or(""),
                "{id}: version {:?}",
                hit.version
            );
            ensure!(
                hit.source_uri.as_deref() == document["sourceUri"].as_str(),
                "{id}: source {:?}",
                hit.source_uri
            );
            ensure!(
                hit.metadata == document["metadata"],
                "{id}: metadata {}",
                hit.metadata
            );
            let chunk = hit.text.as_deref().context("chunk text")?;
            ensure!(
                text.get(hit.start_byte as usize..hit.end_byte as usize) == Some(chunk),
                "{id}: chunk text differs from the source bytes"
            );
        }
        if let Some(chunks) = document["chunks"].as_u64() {
            ensure!(
                hits.len() as u64 == chunks,
                "{id}: {} chunks found, {chunks} expected",
                hits.len()
            );
        }
    }
    for id in expect["absentDocuments"].as_array().into_iter().flatten() {
        let id = id.as_str().context("id")?;
        ensure!(!core.has_document(id), "deleted document {id} is present");
    }
    for vector in expect["vectors"].as_array().into_iter().flatten() {
        let id = vector["id"].as_str().context("vector id")?;
        let seed = vector["seed"].as_u64().context("seed")?;
        let top = core.query(
            &EntangledHVec::new_deterministic(dimensions as usize, seed),
            1,
        );
        ensure!(
            top.first().map(|r| r.id.as_str()) == Some(id),
            "vector {id}: top hit {:?}",
            top.first().map(|r| &r.id)
        );
    }
    if let Some(count) = expect["vectorCount"].as_u64() {
        ensure!(
            core.vector_count() == count,
            "vector count {}",
            core.vector_count()
        );
    }
    if let Some(count) = expect["relations"].as_u64() {
        ensure!(
            core.relation_count() as u64 == count,
            "relations {}",
            core.relation_count()
        );
    }
    for search in expect["searches"].as_array().into_iter().flatten() {
        let query = search["query"].as_str().unwrap_or("");
        let options = SearchOptions {
            k: Some(1),
            embedding: search["embedding"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_f64).collect()),
            ..Default::default()
        };
        let hits = core.search_documents(query, &options)?;
        ensure!(
            hits.first().map(|h| h.document_id.as_str()) == search["top"].as_str(),
            "search {query:?}: top hit {:?}, expected {:?}",
            hits.first().map(|h| &h.document_id),
            search["top"]
        );
    }
    Ok(())
}
