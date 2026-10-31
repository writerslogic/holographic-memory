// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::path::Path;

use super::{Mutation, WarmConfig, WarmCore};
use crate::core::documents::{self, EncodeMode, IngestOptions, VectorState};
use crate::core::error::CommittedMutationError;

const TEXT: &str = "needle alpha beta";

fn pending(directory: &Path) -> Result<WarmCore> {
    let mut config = WarmConfig::default();
    config.document_checkpoint.enabled = false;
    let core = WarmCore::new(256, Some(directory.display().to_string()), Some(config))?;
    core.memorize_document_with(
        serde_json::from_value(serde_json::json!({"id":"a", "version":"1", "text":TEXT}))?,
        &IngestOptions {
            encode: EncodeMode::Deferred,
            ..Default::default()
        },
    )?;
    assert_eq!(core.documents.read().pending_vector_count(), 1);
    Ok(core)
}

fn damage(directory: &Path, truncate: bool) -> Result<Vec<u8>> {
    let path = directory.join("documents.text");
    let original = std::fs::read(&path)?;
    assert_eq!(&original[8..], TEXT.as_bytes());
    let mut damaged = original.clone();
    if truncate {
        damaged.pop();
    } else {
        damaged[8] ^= 1;
    }
    std::fs::write(path, damaged)?;
    Ok(original)
}

#[test]
fn deferred_cache_failure_before_commit_keeps_encoding_pending() -> Result<()> {
    let mut panics = Vec::new();
    for truncate in [false, true] {
        let directory = tempfile::tempdir()?;
        let core = pending(directory.path())?;
        let revision = core.revision();
        let journal = core.arena.checkpoint();
        let original = damage(directory.path(), truncate)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            core.encode_pending_documents()
        }));
        std::fs::write(directory.path().join("documents.text"), original)?;
        match result {
            Ok(result) => {
                let error = result.expect_err("damaged cache must not encode");
                assert!(!error.is::<CommittedMutationError>());
            }
            Err(_) => panics.push(truncate),
        }
        assert_eq!(core.revision(), revision);
        assert_eq!(core.arena.checkpoint(), journal);
        assert_eq!(core.documents.read().pending_vector_count(), 1);
        assert_eq!(core.encode_pending_documents()?, 1);
        assert_eq!(core.encode_pending_documents()?, 0);
        assert_eq!(core.documents.read().pending_vector_count(), 0);
    }
    assert!(
        panics.is_empty(),
        "encoding panicked for truncate={panics:?}"
    );
    Ok(())
}

#[test]
fn deferred_cache_failure_after_commit_reports_durability_and_replays() -> Result<()> {
    let mut panics = Vec::new();
    for truncate in [false, true] {
        let directory = tempfile::tempdir()?;
        let core = pending(directory.path())?;
        let config = core.config.clone();
        let fingerprint = {
            let store = core.documents.read();
            documents::document_fingerprint(&store, store.get("a").unwrap())?
                .expect("retained text")
        };
        let mutation = Mutation::EncodeDocument {
            id: "a".into(),
            version: "1".into(),
            fingerprint,
            vectors: vec![core.config.text_encoder.encode(TEXT, 256)],
        };
        let revision = core.revision();
        let journal = core.arena.checkpoint();
        damage(directory.path(), truncate)?;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| core.commit(&[mutation])));
        match result {
            Ok(result) => {
                let error = result.expect_err("damaged live cache must reject installation");
                assert!(error.is::<CommittedMutationError>(), "{error:#}");
                assert_eq!(core.revision(), revision + 1);
                assert_ne!(core.arena.checkpoint(), journal);
            }
            Err(_) => panics.push(truncate),
        }
        assert_eq!(core.documents.read().pending_vector_count(), 1);
        drop(core);
        let reopened = WarmCore::new(
            256,
            Some(directory.path().display().to_string()),
            Some(config),
        )?;
        assert_eq!(reopened.documents.read().pending_vector_count(), 0);
        assert_eq!(
            reopened.documents.read().get("a").unwrap().chunks[0].vector_state,
            VectorState::Encoded
        );
        assert_eq!(
            reopened
                .read_document_chunk("a", 0)?
                .unwrap()
                .text
                .as_deref(),
            Some(TEXT)
        );
        assert_eq!(reopened.encode_pending_documents()?, 0);
    }
    assert!(
        panics.is_empty(),
        "committed installation panicked for truncate={panics:?}"
    );
    Ok(())
}

#[test]
fn checked_queries_reject_partial_results_when_deferred_encoding_fails() -> Result<()> {
    let mut accepted = Vec::new();
    for truncate in [false, true] {
        let directory = tempfile::tempdir()?;
        let core = pending(directory.path())?;
        let query = core.try_encode_text(TEXT)?;
        core.memorize("healthy".into(), query.clone())?;
        let revision = core.revision();
        let journal = core.arena.checkpoint();
        let original = damage(directory.path(), truncate)?;
        let query_paths = || {
            [
                ("raw", core.query(&query, 10)),
                (
                    "batch",
                    core.query_batch(std::slice::from_ref(&query), 10)
                        .map(|groups| groups.into_iter().flatten().collect()),
                ),
                ("components", core.analyze_components(&query)),
                ("dense", core.query_vector(&[1.0, 0.0], 10)),
                (
                    "triplet",
                    core.query_triplet("head".into(), "relation".into(), 10),
                ),
                ("analogy", core.try_find_analogy("a", "b", "c", 10)),
                ("sequence", core.query_sequence(&["needle".into()], 10)),
                ("federated", core.federated_query(&[], &query, 10)),
            ]
        };
        let outcomes = query_paths();
        std::fs::write(directory.path().join("documents.text"), original)?;
        for (path, outcome) in outcomes {
            if outcome.is_ok() {
                accepted.push((truncate, path));
            }
        }
        assert_eq!(core.revision(), revision);
        assert_eq!(core.arena.checkpoint(), journal);
        assert_eq!(core.documents.read().pending_vector_count(), 1);
        assert_eq!(core.encode_pending_documents()?, 1);
        assert_eq!(core.query_vector(&[1.0, 0.0], 10)?.len(), 2);
        for (path, outcome) in query_paths() {
            assert!(outcome.is_ok(), "repaired {path} query failed: {outcome:?}");
        }
    }
    assert!(
        accepted.is_empty(),
        "queries accepted incomplete results: {accepted:?}"
    );
    Ok(())
}
