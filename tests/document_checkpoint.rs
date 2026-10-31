// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Journal checkpoints (`documents.ckpt`): a store whose journal generation holds a checkpoint
//! in place of the frames before it answers every query exactly as the same store without
//! checkpoints (full replay) does, before and after writes, and keeps nothing a purge removed.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::json;
use writerslogic_warm::core::documents::{EncodeMode, IngestOptions};
use writerslogic_warm::{core::WarmConfig, DocumentInput, DocumentResult, SearchOptions, WarmCore};

const DIM: u32 = 256;
const CHECKPOINT: &str = "documents.ckpt";

fn config(checkpoints: bool) -> WarmConfig {
    let mut config = WarmConfig::default();
    config.document_checkpoint.enabled = checkpoints;
    config.document_checkpoint.min_journal_bytes = 0;
    config
}

fn open(path: &Path, config: WarmConfig) -> Result<WarmCore> {
    WarmCore::new(DIM, Some(path.display().to_string()), Some(config))
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn words(state: &mut u64, count: usize) -> String {
    const STEMS: [&str; 12] = [
        "run",
        "running",
        "runs",
        "index",
        "indexes",
        "memory",
        "graph",
        "holographic",
        "vectors",
        "binding",
        "zyzzogeton",
        "café",
    ];
    (0..count)
        .map(|_| match next(state) % 5 {
            0 => STEMS[(next(state) % STEMS.len() as u64) as usize].to_string(),
            _ => format!("w{}", next(state) % 300),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn lexical() -> IngestOptions {
    IngestOptions {
        encode: EncodeMode::Lexical,
        ..Default::default()
    }
}

/// Multi-chunk documents with metadata, stored and unstored text, a chunk without terms, replaced and
/// deleted documents (so free and reused slots), written in several batches.
fn build(core: &WarmCore, seed: u64, batches: usize) -> Result<()> {
    let mut state = seed;
    for batch in 0..batches {
        let inputs = (0..60)
            .map(|i| {
                let id = format!("doc-{:03}", (batch * 47 + i * 7) % 211);
                let count = (next(&mut state) % 90) as usize;
                serde_json::from_value::<DocumentInput>(json!({
                    "id": id,
                    "text": if i == 5 { "!!! ... ???".to_string() } else { words(&mut state, 10 + count) },
                    "metadata": {"batch": batch, "weight": 0.1 + i as f64 / 3.0},
                    "chunk_words": 24,
                    "overlap_words": 4,
                    "store_text": i % 3 != 0,
                    "source_uri": (i % 4 == 0).then(|| format!("file:///{i}")),
                }))
            })
            .collect::<serde_json::Result<Vec<_>>>()?;
        let mut seen = std::collections::HashSet::new();
        let inputs: Vec<_> = inputs
            .into_iter()
            .filter(|d| seen.insert(d.id.clone()))
            .collect();
        core.memorize_documents_with(inputs, &lexical())?;
        for i in 0..5 {
            core.delete_document(&format!("doc-{:03}", (batch * 31 + i * 13) % 211))?;
        }
    }
    Ok(())
}

/// One replace, one delete and one new document, as the ops benchmark does.
fn write_some(core: &WarmCore, round: usize) -> Result<()> {
    let mut state = 0xC0FFEE + round as u64;
    core.memorize_documents_with(
        vec![serde_json::from_value(json!({
            "id": format!("doc-{:03}", (round * 17) % 211),
            "version": format!("r{round}"),
            "text": words(&mut state, 40),
        }))?],
        &lexical(),
    )?;
    core.delete_document(&format!("doc-{:03}", (round * 29 + 3) % 211))?;
    core.memorize_documents_with(
        vec![serde_json::from_value(json!({
            "id": format!("new-{round}"),
            "text": words(&mut state, 30),
        }))?],
        &lexical(),
    )?;
    Ok(())
}

fn queries() -> Vec<(String, SearchOptions)> {
    let mut state = 7u64;
    let mut out = Vec::new();
    for i in 0..40 {
        let text = words(&mut state, 1 + i % 4);
        let options = match i % 4 {
            0 => SearchOptions::default(),
            1 => SearchOptions {
                k: Some(3),
                ..Default::default()
            },
            2 => SearchOptions {
                k: Some(25),
                bm25_k1: Some(0.9),
                bm25_b: Some(0.3),
                ..Default::default()
            },
            _ => SearchOptions {
                k: Some(1000),
                ..Default::default()
            },
        };
        out.push((text, options));
    }
    out.push(("nothing-matches-this".into(), SearchOptions::default()));
    out
}

/// Every field of every hit (scores as bits), then every stored record.
fn answers(core: &WarmCore) -> Result<Vec<String>> {
    let render = |r: &DocumentResult| {
        format!(
            "{}|{}|{:?}|{}|{}|{:?}|{}..{}|{:x}|{:x}|{:?}",
            r.id,
            r.document_id,
            r.source_uri,
            r.version,
            r.metadata,
            r.text,
            r.start_byte,
            r.end_byte,
            r.score.to_bits(),
            r.lexical_score.to_bits(),
            r.semantic_score.map(f64::to_bits)
        )
    };
    let mut out = Vec::new();
    for (text, options) in queries() {
        out.extend(core.search_documents(&text, &options)?.iter().map(render));
        out.push(String::new());
    }
    Ok(out)
}

fn records(core: &WarmCore) -> Vec<String> {
    core.list_documents(None, 10_000)
        .iter()
        .map(|record| format!("{record:?}"))
        .collect()
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).expect("store dir").flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

fn checkpoints(dir: &Path) -> Vec<PathBuf> {
    walk(dir)
        .into_iter()
        .filter(|path| path.file_name().is_some_and(|name| name == CHECKPOINT))
        .collect()
}

/// The invariant: opened from its checkpoint, after a rebuild, after writes, and reopened with
/// frames after the checkpoint, a store answers exactly as its twin without checkpoints.
#[test]
fn checkpointed_store_answers_exactly_as_full_replay_through_writes_and_reopens() -> Result<()> {
    let (with, without) = (tempfile::tempdir()?, tempfile::tempdir()?);
    for (dir, checkpoints) in [(&with, true), (&without, false)] {
        build(
            &open(dir.path(), config(checkpoints))?,
            0x9E37_79B9_7F4A_7C15,
            4,
        )?;
    }
    assert_eq!(
        checkpoints(with.path()).len(),
        1,
        "closing writes one checkpoint"
    );
    assert!(checkpoints(without.path()).is_empty());

    let twin = open(without.path(), config(false))?;
    let core = open(with.path(), config(true))?;
    assert!(
        core.serves_from_checkpoint(),
        "no frames after the checkpoint"
    );
    let expected = answers(&twin)?;
    assert!(expected.iter().filter(|line| !line.is_empty()).count() > 100);
    assert_eq!(answers(&core)?, expected);
    assert!(
        core.serves_from_checkpoint(),
        "lexical queries never rebuild"
    );
    assert_eq!(records(&core), records(&twin), "a read rebuilds the store");
    assert!(!core.serves_from_checkpoint());
    assert_eq!(answers(&core)?, expected);

    for round in 0..3 {
        write_some(&core, round)?;
        write_some(&twin, round)?;
        assert_eq!(answers(&core)?, answers(&twin)?, "round {round}");
    }
    drop(core);

    // Frames after the checkpoint: the open rebuilds from it and replays only those.
    let mut keep_tail = config(true);
    keep_tail.document_checkpoint.min_journal_bytes = usize::MAX;
    let core = open(with.path(), keep_tail.clone())?;
    assert!(core.serves_from_checkpoint(), "closing folded the writes");
    assert_eq!(answers(&core)?, answers(&twin)?);
    write_some(&core, 7)?;
    write_some(&twin, 7)?;
    drop(core);
    let core = open(with.path(), keep_tail)?;
    assert!(
        core.serves_from_checkpoint(),
        "frames after the checkpoint replay over it"
    );
    assert_eq!(answers(&core)?, answers(&twin)?);
    assert_eq!(records(&core), records(&twin));
    drop(core);

    // Closing with frames folds them into a new checkpoint in a new generation.
    drop(open(with.path(), config(true))?);
    assert_eq!(checkpoints(with.path()).len(), 1);
    let core = open(with.path(), config(true))?;
    assert!(core.serves_from_checkpoint());
    assert_eq!(answers(&core)?, answers(&twin)?);
    assert_eq!(records(&core), records(&twin));
    Ok(())
}

fn ids(core: &WarmCore) -> Vec<String> {
    core.list_documents(None, 10_000)
        .into_iter()
        .map(|record| record.id)
        .collect()
}

/// Writes to a store served from its checkpoint go to a store over it that marks the
/// checkpoint documents they replace or delete dead. Through random replaces, inserts, deletes,
/// re-inserts of deleted checkpoint ids, a reopen replaying the writes, and deleting everything,
/// every answer (scores as bits) equals the twin's without checkpoints, and nothing rebuilds the
/// store until a read needs it.
#[test]
fn writes_over_a_checkpoint_score_exactly_as_the_store_without_one() -> Result<()> {
    for seed in [0x5EED_u64, 0xDEAD_BEEF, 0x0123_4567_89AB_CDEF] {
        let (with, without) = (tempfile::tempdir()?, tempfile::tempdir()?);
        for (dir, checkpoints) in [(&with, true), (&without, false)] {
            build(&open(dir.path(), config(checkpoints))?, seed, 3)?;
        }
        let mut keep_tail = config(true);
        keep_tail.document_checkpoint.min_journal_bytes = usize::MAX;
        let twin = open(without.path(), config(false))?;
        let mut core = open(with.path(), keep_tail.clone())?;
        let original = ids(&twin);
        let mut state = seed;
        for step in 0..48 {
            let pick = next(&mut state);
            let id = if pick.is_multiple_of(4) {
                format!("fresh-{}", pick % 9)
            } else {
                original[(pick / 4) as usize % original.len()].clone()
            };
            if next(&mut state).is_multiple_of(3) {
                assert_eq!(core.delete_document(&id)?, twin.delete_document(&id)?);
            } else {
                let input = json!({
                    "id": id,
                    "version": format!("s{step}"),
                    "text": words(&mut state, 5 + (pick % 70) as usize),
                    "chunk_words": 24,
                    "overlap_words": 4,
                });
                for store in [&core, &twin] {
                    store.memorize_documents_with(
                        vec![serde_json::from_value(input.clone())?],
                        &lexical(),
                    )?;
                }
            }
            if step == 24 {
                drop(core);
                core = open(with.path(), keep_tail.clone())?;
            }
            assert!(core.serves_from_checkpoint(), "seed {seed:x} step {step}");
            assert_eq!(
                answers(&core)?,
                answers(&twin)?,
                "seed {seed:x} step {step}"
            );
        }
        for id in ids(&twin) {
            assert!(core.delete_document(&id)? && twin.delete_document(&id)?);
        }
        assert!(core.serves_from_checkpoint());
        assert_eq!(answers(&core)?, answers(&twin)?, "seed {seed:x} emptied");
        for (i, id) in original.iter().take(5).enumerate() {
            let input = json!({"id": id, "text": words(&mut state, 12 + i)});
            for store in [&core, &twin] {
                store.memorize_documents_with(
                    vec![serde_json::from_value(input.clone())?],
                    &lexical(),
                )?;
            }
        }
        assert!(core.serves_from_checkpoint());
        assert_eq!(answers(&core)?, answers(&twin)?, "seed {seed:x} refilled");
        assert_eq!(records(&core), records(&twin), "a read rebuilds the store");
        assert!(!core.serves_from_checkpoint());
        assert_eq!(answers(&core)?, answers(&twin)?, "seed {seed:x} rebuilt");
    }
    Ok(())
}

/// A checkpoint is the store's data: a damaged one fails the open instead of being skipped.
const EMBED: usize = 8;

fn embedded(checkpoints: bool, keep_tail: bool) -> WarmConfig {
    let mut config = config(checkpoints);
    if keep_tail {
        config.document_checkpoint.min_journal_bytes = usize::MAX;
    }
    config.embedding_space = Some(writerslogic_warm::EmbeddingSpace {
        model: "fixture".into(),
        revision: "1".into(),
        dimensions: EMBED,
        normalization: "l2".into(),
        metric: "cosine".into(),
    });
    config
}

fn embedding(seed: u64) -> Vec<f64> {
    (0..EMBED)
        .map(|i| ((seed % 1009) as f64 * 0.37 + i as f64 * 1.3).sin())
        .collect()
}

/// Chunk-vector hits (`WarmCore::query`) of a few probes: ids and similarity bits, in a
/// canonical order.
fn vector_hits(core: &WarmCore) -> Vec<String> {
    let mut state = 11u64;
    let mut out = Vec::new();
    for _ in 0..4 {
        let probe = core.encode_text(&words(&mut state, 12));
        let mut hits: Vec<_> = core
            .query(&probe, 4096)
            .into_iter()
            .map(|hit| (hit.similarity.to_bits(), hit.id))
            .collect();
        hits.sort();
        out.extend(hits.into_iter().map(|(bits, id)| format!("{id}|{bits:x}")));
        out.push(String::new());
    }
    out
}

/// Hybrid (BM25 plus query embedding) and exact embedding-only answers, scores as bits.
fn embedding_answers(core: &WarmCore) -> Result<Vec<String>> {
    use writerslogic_warm::core::documents::{VectorIndexMode, VectorQueryOptions};
    let render = |r: &DocumentResult| {
        format!(
            "{}|{:x}|{:x}|{:?}",
            r.id,
            r.score.to_bits(),
            r.lexical_score.to_bits(),
            r.semantic_score.map(f64::to_bits)
        )
    };
    let mut state = 23u64;
    let mut out = Vec::new();
    for i in 0..6u64 {
        let options = SearchOptions {
            k: Some(50),
            embedding: Some(embedding(i * 31 + 5)),
            fusion: (i % 2 == 1).then(|| "rrf".into()),
            ..Default::default()
        };
        out.extend(
            core.search_documents(&words(&mut state, 3), &options)?
                .iter()
                .map(render),
        );
        out.push(String::new());
        let exact = VectorQueryOptions {
            mode: VectorIndexMode::Exact,
            ..VectorQueryOptions::default()
        };
        out.extend(
            core.search_vectors(&embedding(i * 31 + 5), 20, &exact)?
                .iter()
                .map(render),
        );
        out.push(String::new());
    }
    Ok(out)
}

/// Eager, deferred and lexical writes (some with caller embeddings) and deletes over a store
/// served from its checkpoint stay off the checkpoint: no write, deferred encoding or
/// chunk-vector query rebuilds it, and lexical, chunk-vector, hybrid and embedding answers
/// (scores as bits) equal those of the twin without checkpoints, through a reopen that replays
/// the writes over the checkpoint.
#[test]
fn encoded_writes_over_a_checkpoint_answer_as_the_store_without_one() -> Result<()> {
    for seed in [0xA11CE_u64, 0xB0B5_EED5] {
        let (with, without) = (tempfile::tempdir()?, tempfile::tempdir()?);
        for (dir, checkpoints) in [(&with, true), (&without, false)] {
            build(&open(dir.path(), embedded(checkpoints, false))?, seed, 3)?;
        }
        assert_eq!(checkpoints(with.path()).len(), 1);
        let twin = open(without.path(), embedded(false, false))?;
        let mut core = open(with.path(), embedded(true, true))?;
        assert!(core.serves_from_checkpoint());
        let original = ids(&twin);
        let mut state = seed;
        let (mut hybrid_semantic, mut chunk_hits) = (0, 0);
        for step in 0..40 {
            let pick = next(&mut state);
            let id = if pick.is_multiple_of(4) {
                format!("fresh-{}", pick % 9)
            } else {
                original[(pick / 4) as usize % original.len()].clone()
            };
            if next(&mut state).is_multiple_of(4) {
                assert_eq!(core.delete_document(&id)?, twin.delete_document(&id)?);
            } else {
                let encode = match next(&mut state) % 3 {
                    0 => EncodeMode::Eager,
                    1 => EncodeMode::Deferred,
                    _ => EncodeMode::Lexical,
                };
                let input = if pick.is_multiple_of(3) {
                    json!({
                        "id": id,
                        "version": format!("s{step}"),
                        "text": words(&mut state, 5 + (pick % 15) as usize),
                        "embeddings": [embedding(pick)],
                    })
                } else {
                    json!({
                        "id": id,
                        "version": format!("s{step}"),
                        "text": words(&mut state, 5 + (pick % 70) as usize),
                        "chunk_words": 24,
                        "overlap_words": 4,
                    })
                };
                for store in [&core, &twin] {
                    store.memorize_documents_with(
                        vec![serde_json::from_value(input.clone())?],
                        &IngestOptions {
                            encode,
                            ..Default::default()
                        },
                    )?;
                }
            }
            assert!(core.serves_from_checkpoint(), "seed {seed:x} step {step}");
            assert_eq!(
                answers(&core)?,
                answers(&twin)?,
                "seed {seed:x} step {step}"
            );
            if step % 5 == 4 {
                let hits = vector_hits(&twin);
                chunk_hits += hits.iter().filter(|line| !line.is_empty()).count();
                assert_eq!(vector_hits(&core), hits, "seed {seed:x} step {step}");
                assert!(core.serves_from_checkpoint(), "seed {seed:x} step {step}");
            }
            if step == 24 {
                let expected = embedding_answers(&twin)?;
                hybrid_semantic += expected.iter().filter(|l| l.contains("Some(")).count();
                assert_eq!(embedding_answers(&core)?, expected, "seed {seed:x}");
                drop(core);
                core = open(with.path(), embedded(true, true))?;
                assert!(core.serves_from_checkpoint(), "seed {seed:x} reopened");
                assert_eq!(
                    embedding_answers(&core)?,
                    expected,
                    "seed {seed:x} reopened"
                );
                drop(core);
                core = open(with.path(), embedded(true, true))?;
            }
        }
        assert!(chunk_hits > 0 && hybrid_semantic > 0, "seed {seed:x}");
        assert_eq!(vector_hits(&core), vector_hits(&twin), "seed {seed:x} end");
        assert_eq!(
            embedding_answers(&core)?,
            embedding_answers(&twin)?,
            "seed {seed:x} end"
        );
        assert_eq!(records(&core), records(&twin), "seed {seed:x}");
    }
    Ok(())
}

#[test]
fn damaged_checkpoint_fails_the_open() -> Result<()> {
    let dir = tempfile::tempdir()?;
    build(&open(dir.path(), config(true))?, 11, 2)?;
    let path = checkpoints(dir.path()).pop().expect("a checkpoint");
    let mut bytes = std::fs::read(&path)?;
    let last = bytes.len() - 1;
    bytes[last] ^= 0x40;
    std::fs::write(&path, bytes)?;
    let error = open(dir.path(), config(true))
        .err()
        .expect("damage detected");
    assert!(format!("{error:#}").contains("checkpoint"), "{error:#}");
    Ok(())
}

/// Stores with vector state or encoded chunks keep plain frames: a checkpoint holds lexical
/// documents only.
#[test]
fn stores_with_encoded_chunks_get_no_checkpoint() -> Result<()> {
    let dir = tempfile::tempdir()?;
    {
        let core = open(dir.path(), config(true))?;
        core.memorize_document(serde_json::from_value(
            json!({"id": "a", "text": "graph memory"}),
        )?)?;
    }
    assert!(checkpoints(dir.path()).is_empty());
    let core = open(dir.path(), config(true))?;
    assert!(core.has_document("a"));
    Ok(())
}

/// A purge returns only when no checkpoint, frame or text in any file holds the document.
#[test]
fn purge_leaves_no_trace_in_any_checkpoint() -> Result<()> {
    const MARKER: &str = "quokkapurgemarker";
    let dir = tempfile::tempdir()?;
    {
        let core = open(dir.path(), config(true))?;
        build(&core, 5, 1)?;
        core.memorize_documents_with(
            vec![serde_json::from_value(json!({
                "id": "secret-doc", "text": format!("{MARKER} lives here"), "store_text": true,
            }))?],
            &lexical(),
        )?;
    }
    let leaks = |dir: &Path| -> Vec<PathBuf> {
        walk(dir)
            .into_iter()
            .filter(|path| {
                let bytes = std::fs::read(path).unwrap_or_default();
                bytes.windows(MARKER.len()).any(|w| w == MARKER.as_bytes())
                    || bytes.windows(10).any(|w| w == b"secret-doc")
            })
            .collect()
    };
    assert!(
        !leaks(dir.path()).is_empty(),
        "the checkpoint holds it before the purge"
    );
    let core = open(dir.path(), config(true))?;
    assert!(core.serves_from_checkpoint());
    core.delete_document("secret-doc")?;
    core.compact_scrubbing()?;
    assert_eq!(leaks(dir.path()), Vec::<PathBuf>::new());
    assert_eq!(checkpoints(dir.path()).len(), 1);
    drop(core);
    let core = open(dir.path(), config(true))?;
    assert!(!core.has_document("secret-doc"));
    assert!(core
        .search_documents(MARKER, &SearchOptions::default())?
        .is_empty());
    Ok(())
}

#[cfg(feature = "security")]
#[test]
fn encrypted_checkpoints_are_sealed_and_answer_as_replay() -> Result<()> {
    let sealed = |checkpoints: bool| {
        let mut config = config(checkpoints);
        config.security.encryption_enabled = true;
        config.security.encryption_passphrase = Some("checkpoint-fixture".into());
        config
    };
    let (with, without) = (tempfile::tempdir()?, tempfile::tempdir()?);
    for (dir, checkpoints) in [(&with, true), (&without, false)] {
        build(&open(dir.path(), sealed(checkpoints))?, 3, 2)?;
    }
    let path = checkpoints(with.path()).pop().expect("a checkpoint");
    let bytes = std::fs::read(&path)?;
    assert!(bytes.starts_with(b"HMSCKPE\x01"));
    for marker in [&b"zyzzogeton"[..], b"holographic", b"doc-0"] {
        for file in walk(with.path()) {
            let bytes = std::fs::read(&file)?;
            assert!(
                !bytes.windows(marker.len()).any(|w| w == marker),
                "{} holds plaintext",
                file.display()
            );
        }
    }
    let core = open(with.path(), sealed(true))?;
    assert!(core.serves_from_checkpoint());
    let twin = open(without.path(), sealed(false))?;
    assert_eq!(answers(&core)?, answers(&twin)?);
    write_some(&core, 1)?;
    write_some(&twin, 1)?;
    assert_eq!(answers(&core)?, answers(&twin)?);
    drop(core);
    let mut wrong = sealed(true);
    wrong.security.encryption_passphrase = Some("wrong".into());
    assert!(open(with.path(), wrong).is_err());
    Ok(())
}

/// Compacting a store still served from its checkpoint rebuilds it before taking any lock
/// the rebuild needs (it once deadlocked on the shards lock) and keeps every answer.
#[test]
fn compaction_right_after_opening_from_a_checkpoint_completes() -> Result<()> {
    let dir = tempfile::tempdir()?;
    build(&open(dir.path(), config(true))?, 21, 2)?;
    let core = std::sync::Arc::new(open(dir.path(), config(true))?);
    assert!(core.serves_from_checkpoint());
    let before = answers(&core)?;
    let (done, finished) = std::sync::mpsc::channel();
    let compactor = std::sync::Arc::clone(&core);
    std::thread::spawn(move || done.send(compactor.compact().map_err(|e| format!("{e:#}"))));
    let outcome = finished
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("compaction finished within a minute");
    assert_eq!(outcome, Ok(()));
    assert_eq!(answers(&core)?, before);
    Ok(())
}

#[test]
fn deep_verification_checks_checkpoint_records_after_fast_count() -> Result<()> {
    use writerslogic_warm::core::format::{verify_path, Verdict, VerifyOptions};
    let dir = tempfile::tempdir()?;
    build(&open(dir.path(), config(true))?, 11, 1)?;
    let path = checkpoints(dir.path()).pop().expect("a checkpoint");
    let mut bytes = std::fs::read(&path)?;
    // Slot section offset: header word 14 + 2 * section 5.
    let slot_at = usize::try_from(u64::from_le_bytes(bytes[200..208].try_into()?))?;
    let slots_len = usize::try_from(u64::from_le_bytes(bytes[208..216].try_into()?))?;
    let occupied = (slot_at..slot_at + slots_len)
        .step_by(12)
        .find(|&at| bytes[at + 4..at + 8] != u32::MAX.to_le_bytes())
        .expect("occupied slot");
    bytes[occupied + 8..occupied + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    let crc = crc32fast::hash(&bytes[16..]);
    bytes[8..16].copy_from_slice(&u64::from(crc).to_le_bytes());
    std::fs::write(path, bytes)?;
    let report = verify_path(
        dir.path(),
        &VerifyOptions {
            deep: true,
            ..Default::default()
        },
    )?;
    assert_ne!(report.verdict, Verdict::Ok);
    assert!(serde_json::to_string(&report)?.contains("checkpoint chunk slots out of order"));
    Ok(())
}

#[test]
fn metadata_pages_merge_checkpoint_replacements_and_deletions() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let input = |id: &str, version: &str| {
        serde_json::from_value::<DocumentInput>(json!({
            "id": id, "version": version, "text": "stored text", "metadata": {"version": version}
        }))
    };
    let core = open(dir.path(), config(true))?;
    for id in ["a", "a-b", "b", "café", "z"] {
        core.memorize_document_with(input(id, "v1")?, &lexical())?;
    }
    drop(core);
    assert!(!checkpoints(dir.path()).is_empty());
    let mut keep_tail = config(true);
    keep_tail.document_checkpoint.min_journal_bytes = usize::MAX;
    let mut core = open(dir.path(), keep_tail.clone())?;
    core.delete_document("a-b")?;
    core.memorize_document_with(input("b", "v2")?, &lexical())?;
    core.memorize_document_with(input("a-c", "v1")?, &lexical())?;
    for reopen in [false, true] {
        if reopen {
            drop(core);
            core = open(dir.path(), keep_tail.clone())?;
        }
        assert_eq!(core.document_count(), 5);
        assert!(core.get_document_info("a-b").is_none());
        let replacement = core.get_document_info("b").unwrap();
        assert_eq!(replacement.version, "v2");
        assert_eq!(replacement.metadata, json!({"version": "v2"}));
        for (cursor, expected) in [
            (None, vec!["a", "a-c", "b", "café", "z"]),
            (Some(""), vec!["a", "a-c", "b", "café", "z"]),
            (Some("a-b"), vec!["a-c", "b", "café", "z"]),
            (Some("b"), vec!["b", "café", "z"]),
            (Some("c"), vec!["café", "z"]),
            (Some("z"), vec!["z"]),
            (Some("zz"), vec![]),
        ] {
            for limit in [0, 1, 2, 5, 6] {
                let page: Vec<_> = core
                    .list_document_info(cursor, limit)
                    .into_iter()
                    .map(|record| record.id)
                    .collect();
                assert_eq!(
                    page,
                    expected.iter().take(limit).copied().collect::<Vec<_>>()
                );
            }
        }
    }
    assert_eq!(
        core.get_document("b").unwrap().text.as_deref(),
        Some("stored text")
    );
    Ok(())
}
