// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! A purge leaves a store indistinguishable from one that never held the purged document and
//! was purged at the same point: the same results and score bits in every search mode, the
//! same status apart from the opaque counters, and the same bytes in every file apart from the
//! generation directory's name (which carries the compaction time, like a file's mtime).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use holographic_memory::core::documents::{VectorIndexMode, VectorQueryOptions};
use holographic_memory::core::memory_api::{
    ForgetOptions, ForgetScope, ManualClock, MaxFacts, Memory, RecallOptions, RememberInput,
};
use holographic_memory::core::HmsConfig;
use holographic_memory::{
    DocumentInput, EmbeddingSpace, HmsCore, MemorizeBatchItem, SearchOptions,
};
use serde_json::{json, Value};

const DIM: usize = 8;
const TARGET: usize = 999;

fn config() -> HmsConfig {
    HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "fixture".into(),
            revision: "1".into(),
            dimensions: DIM,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    }
}

fn embedding(id: usize) -> Vec<f64> {
    (0..DIM)
        .map(|c| ((id * 7 + c * 13) as f64 * 0.3).sin())
        .collect()
}

fn document(id: usize) -> DocumentInput {
    serde_json::from_value(json!({
        "id": format!("d{id:04}"),
        "text": format!("common topic{} rare{id} shared words here", id % 7),
        "metadata": {"group": id % 3},
        "embeddings": [embedding(id)],
    }))
    .unwrap()
}

fn raw_items(ids: impl Iterator<Item = usize>) -> Vec<MemorizeBatchItem> {
    ids.map(|i| MemorizeBatchItem {
        id: format!("v{i:04}"),
        text: format!("vector item {i}"),
    })
    .collect()
}

fn open(dir: &Path) -> Result<HmsCore> {
    HmsCore::new(256, Some(dir.display().to_string()), Some(config()))
}

/// Raw vectors, then documents 0..240 in three transactions, with `extra` committed as its
/// own transaction after the first; then the vector snapshot (with its graph) and the dense
/// index.
fn build(dir: &Path, extra: Option<usize>) -> Result<HmsCore> {
    let core = open(dir)?;
    core.memorize_batch(&raw_items(0..32))?;
    for (n, range) in [0..80, 80..160, 160..240].into_iter().enumerate() {
        core.memorize_documents(range.map(document).collect())?;
        if let Some(x) = extra.filter(|_| n == 0) {
            core.memorize_document(document(x))?;
        }
    }
    core.rebuild_vector_index(1)?;
    core.train_document_index()?;
    Ok(core)
}

fn observe(core: &HmsCore) -> Result<Value> {
    let target = embedding(TARGET);
    let mut out = Vec::new();
    for text in ["rare999 common", "topic5 shared", "rare17", "words"] {
        for (lexical, semantic) in [(1.0, 0.0), (1.0, 1.0)] {
            let options = SearchOptions {
                k: Some(20),
                embedding: Some(target.clone()),
                lexical_weight: Some(lexical),
                semantic_weight: Some(semantic),
                min_semantic_score: Some(-1.0),
                ..Default::default()
            };
            out.push(serde_json::to_value(
                core.search_documents(text, &options)?,
            )?);
        }
    }
    for mode in [VectorIndexMode::Graph, VectorIndexMode::Exact] {
        for k in [5, 50] {
            let options = VectorQueryOptions {
                mode,
                ..Default::default()
            };
            out.push(serde_json::to_value(
                core.search_vectors(&target, k, &options)?,
            )?);
        }
    }
    let mut status = serde_json::to_value(core.vector_index_status())?;
    status.as_object_mut().unwrap().remove("generation");
    out.push(status);
    out.push(json!(core.document_index_trained()));
    out.push(serde_json::to_value(core.index_status())?);
    out.push(serde_json::to_value(core.list_documents(None, 1000))?);
    Ok(Value::Array(out))
}

/// Every file's bytes by path, the lock left out and generation directory names masked in
/// paths and in the small files that name them.
fn files(dir: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(&path, out)?;
            } else {
                out.push(path);
            }
        }
        Ok(())
    }
    let mut paths = Vec::new();
    walk(dir, &mut paths)?;
    let names: Vec<String> = paths
        .iter()
        .flat_map(|p| p.strip_prefix(dir).unwrap().components())
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .filter(|c| c.starts_with("gen-"))
        .collect();
    let mut map = BTreeMap::new();
    for path in paths {
        let mut key = path.strip_prefix(dir)?.display().to_string();
        if key == ".hms.lock" {
            continue;
        }
        let mut bytes = std::fs::read(&path)?;
        for name in &names {
            key = key.replace(name.as_str(), "gen-*");
            if bytes.len() < 1 << 16 {
                bytes = String::from_utf8_lossy(&bytes)
                    .replace(name.as_str(), "gen-*")
                    .into_bytes();
            }
        }
        map.insert(key, bytes);
    }
    Ok(map)
}

fn differing_files(a: &Path, b: &Path) -> Result<Vec<String>> {
    let (a, b) = (files(a)?, files(b)?);
    let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
    keys.dedup();
    keys.sort();
    keys.dedup();
    Ok(keys
        .into_iter()
        .filter(|k| a.get(*k) != b.get(*k))
        .cloned()
        .collect())
}

#[test]
fn purge_leaves_the_store_of_a_document_never_stored() -> Result<()> {
    let (a_dir, b_dir) = (tempfile::tempdir()?, tempfile::tempdir()?);
    let a = build(a_dir.path(), Some(TARGET))?;
    let b = build(b_dir.path(), None)?;
    assert_ne!(observe(&a)?, observe(&b)?, "the document is visible");
    assert!(a.delete_document(&format!("d{TARGET:04}"))?);
    a.compact_scrubbing()?;
    b.compact_scrubbing()?;
    assert_eq!(observe(&a)?, observe(&b)?);
    assert_eq!(
        differing_files(a_dir.path(), b_dir.path())?,
        Vec::<String>::new()
    );
    // Later admissions in the same process land in the same slots and segment offsets.
    for core in [&a, &b] {
        core.memorize_documents((240..250).map(document).collect())?;
        core.rebuild_vector_index(1)?;
        core.train_document_index()?;
    }
    assert_eq!(observe(&a)?, observe(&b)?);
    assert_eq!(
        differing_files(a_dir.path(), b_dir.path())?,
        Vec::<String>::new()
    );
    drop((a, b));
    let (a, b) = (open(a_dir.path())?, open(b_dir.path())?);
    for core in [&a, &b] {
        core.rebuild_vector_index(1)?;
        core.train_document_index()?;
    }
    assert_eq!(observe(&a)?, observe(&b)?);
    assert_eq!(
        differing_files(a_dir.path(), b_dir.path())?,
        Vec::<String>::new()
    );
    Ok(())
}

#[test]
fn open_scrubs_a_canonical_segment_left_by_an_interrupted_purge() -> Result<()> {
    let dir = tempfile::tempdir()?;
    drop(build(dir.path(), None)?);
    let staging = dir.path().join("documents.text.canonical");
    std::fs::write(&staging, "zyxleftovermarker text of a purged document")?;
    let core = open(dir.path())?;
    assert!(!staging.exists());
    assert_eq!(core.document_count(), 240);
    Ok(())
}

/// The purge re-seals every surviving record at its new segment offset (the offset is the
/// record's AAD); surviving text must read back in the same process and after a reopen.
#[cfg(feature = "security")]
#[test]
fn encrypted_purge_keeps_surviving_text_readable() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut config = HmsConfig::default();
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some("forget-equivalence".into());
    let open = || {
        HmsCore::new(
            256,
            Some(dir.path().display().to_string()),
            Some(config.clone()),
        )
    };
    let text = |core: &HmsCore| -> Result<Option<String>> {
        let options = SearchOptions {
            k: Some(1),
            ..Default::default()
        };
        Ok(core
            .search_documents("survivor", &options)?
            .into_iter()
            .next()
            .and_then(|hit| hit.text))
    };
    let core = open()?;
    for (id, body) in [("gone", "purged text"), ("kept", "survivor text kept")] {
        core.memorize_document(serde_json::from_value(json!({"id": id, "text": body}))?)?;
    }
    assert!(core.delete_document("gone")?);
    core.compact_scrubbing()?;
    assert_eq!(text(&core)?.as_deref(), Some("survivor text kept"));
    drop(core);
    let core = open()?;
    assert_eq!(text(&core)?.as_deref(), Some("survivor text kept"));
    Ok(())
}

#[test]
fn compaction_output_does_not_depend_on_vector_arrival_order() -> Result<()> {
    let (forward, backward) = (tempfile::tempdir()?, tempfile::tempdir()?);
    for (dir, reverse) in [(forward.path(), false), (backward.path(), true)] {
        let core = HmsCore::new(256, Some(dir.display().to_string()), None)?;
        let mut items = raw_items(0..3000);
        if reverse {
            items.reverse();
        }
        for batch in items.chunks(1000) {
            core.memorize_batch(batch)?;
        }
        core.compact()?;
    }
    assert_eq!(
        differing_files(forward.path(), backward.path())?,
        Vec::<String>::new()
    );
    Ok(())
}

/// Facts `f0..f8` remembered at times 10, 20, ... and, with `target`, fact `x` at time 35.
fn facts(dir: &Path, limit: usize, target: bool) -> Result<Memory> {
    let clock = Arc::new(ManualClock::new(0));
    let core = Arc::new(HmsCore::new(256, Some(dir.display().to_string()), None)?);
    let memory = Memory::new(core)
        .with_clock(clock.clone())
        .with_forgetting(Arc::new(MaxFacts { limit }));
    for (id, time) in (0..8)
        .map(|i| (format!("f{i}"), 10 * (i + 1)))
        .chain(target.then(|| ("x".to_string(), 35)))
    {
        clock.set(time as i64);
        memory.remember(RememberInput {
            id: Some(id.clone()),
            text: format!("fact {id} about the shared topic"),
            ..RememberInput::default()
        })?;
    }
    clock.set(1000);
    Ok(memory)
}

fn fact_view(memory: &Memory) -> Result<String> {
    let mut view = format!(
        "{:?}",
        memory.recall("fact shared topic", &RecallOptions::default())?
    );
    for id in ["f0", "f1", "f2", "f7", "x"] {
        view.push_str(&format!("{:?}", memory.get(id)?));
    }
    Ok(view)
}

#[test]
fn truncation_after_a_purge_matches_the_never_stored_store() -> Result<()> {
    let purge = ForgetOptions {
        purge: true,
        ..ForgetOptions::default()
    };
    let x = ForgetScope::Ids {
        ids: vec!["x".into()],
    };
    // A fact budget enforced after the purge selects from the live facts alone.
    let (a_dir, b_dir) = (tempfile::tempdir()?, tempfile::tempdir()?);
    let (a, b) = (
        facts(a_dir.path(), 6, true)?,
        facts(b_dir.path(), 6, false)?,
    );
    assert_eq!(a.forget(&x, &purge)?.forgotten, vec!["x".to_string()]);
    b.core().compact_scrubbing()?;
    a.apply_forgetting(&purge)?;
    b.apply_forgetting(&purge)?;
    assert_eq!(fact_view(&a)?, fact_view(&b)?);
    // One enforced while the fact was stored evicted a fact the never-stored store keeps; the
    // purge cannot bring it back, so equivalence holds only from the last such decision on.
    let (a_dir, b_dir) = (tempfile::tempdir()?, tempfile::tempdir()?);
    let (a, b) = (
        facts(a_dir.path(), 8, true)?,
        facts(b_dir.path(), 8, false)?,
    );
    a.apply_forgetting(&purge)?;
    b.apply_forgetting(&purge)?;
    a.forget(&x, &purge)?;
    b.core().compact_scrubbing()?;
    assert_ne!(fact_view(&a)?, fact_view(&b)?);
    Ok(())
}

#[test]
fn revision_counts_transactions_without_revealing_how_many() -> Result<()> {
    let (one, two) = (tempfile::tempdir()?, tempfile::tempdir()?);
    let (first, second) = (open(one.path())?, open(two.path())?);
    let before = first.revision();
    first.memorize_document(document(1))?;
    assert_eq!(first.revision().wrapping_sub(before), 1);
    second.memorize_document(document(1))?;
    assert_ne!(
        first.revision(),
        second.revision(),
        "two stores with one transaction each start from independent random values"
    );
    Ok(())
}
