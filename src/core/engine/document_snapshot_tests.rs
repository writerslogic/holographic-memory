// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{mpsc, Arc};
use std::time::Duration;

use anyhow::{bail, Result};
use serde_json::json;

use super::WarmCore;
use crate::core::documents::{self, DocumentInput, SearchOptions};

fn input(id: &str, version: &str, text: &str, group: &str) -> Result<DocumentInput> {
    Ok(serde_json::from_value(json!({
        "id": id, "version": version, "text": text,
        "sourceUri": format!("source:{version}"), "metadata": {"group": group},
        "embeddings": [[1.0, 0.0]],
    }))?)
}

#[test]
fn captured_queries_survive_durable_mutation_compaction_and_restart() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().display().to_string();
    let config = super::fixture_config(2);
    let core = Arc::new(WarmCore::new(
        1024,
        Some(path.clone()),
        Some(config.clone()),
    )?);
    core.memorize_document(input("a", "old", "needleold first", "before")?)?;
    core.memorize_document(input("b", "old", "needleold needleoldremoved", "before")?)?;
    let options = SearchOptions {
        filter: Some(json!({"group": "before"})),
        embedding: Some(vec![1.0, 0.0]),
        semantic_weight: Some(0.0),
        ..Default::default()
    };
    let captured = {
        let prepared = ["needleold", "needleoldremoved"]
            .into_iter()
            .map(|query| documents::prepare_search(query, &options, Some(2)))
            .collect::<Result<Vec<_>>>()?;
        let _transaction = core.mutation_gate.read();
        documents::capture_search(&core.documents.read(), prepared)
    };
    let (done_tx, done_rx) = mpsc::channel();
    let writer_core = Arc::clone(&core);
    let writer = std::thread::spawn(move || -> Result<()> {
        writer_core.memorize_document(input("a", "new", "replacement", "after")?)?;
        writer_core.delete_document("b")?;
        writer_core.memorize_document(input("c", "new", "needleold", "before")?)?;
        done_tx.send(())?;
        writer_core.compact()
    });
    if done_rx.recv_timeout(Duration::from_secs(5)).is_err() {
        drop(captured);
        writer.join().unwrap()?;
        bail!("retained search snapshot blocked durable mutations");
    }
    writer.join().unwrap()?;
    drop(core);
    let old = documents::execute_search(&captured, 0)?;
    assert_eq!(
        old.iter()
            .map(|r| r.document_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert!(old.iter().all(|r| r.version == "old"
        && r.source_uri.as_deref() == Some("source:old")
        && r.metadata["group"] == "before"
        && r.semantic_score == Some(1.0)));
    let second = documents::execute_search(&captured, 1)?;
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].document_id, "b");
    // The delete zeroed the text on disk, so a query captured before it reads none.
    assert_eq!(second[0].text.as_deref(), Some(""));
    let restored = WarmCore::new(1024, Some(path), Some(config))?;
    let current = restored.search_documents("needleold", &options)?;
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].document_id, "c");
    assert_eq!(current[0].version, "new");
    assert!(restored
        .search_documents("needleoldremoved", &options)?
        .is_empty());
    Ok(())
}

/// A replacement between a search's capture and its text read scrubs the captured text; the
/// read must notice instead of pairing the old version with empty text.
#[test]
fn ranked_hits_are_not_read_after_a_replacement_scrubs_their_text() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let core = WarmCore::new(
        1024,
        Some(directory.path().display().to_string()),
        Some(super::fixture_config(2)),
    )?;
    core.memorize_document(input("a", "1", "current revision1", "g")?)?;
    let capture = || -> Result<_> {
        let prepared = documents::prepare_search("current", &SearchOptions::default(), Some(2))?;
        let _transaction = core.mutation_gate.read();
        Ok(documents::capture_search(
            &core.documents.read(),
            vec![prepared],
        ))
    };
    let captured = capture()?;
    let ranked = documents::rank_search(&captured, 0);
    let read = documents::read_ranked(&captured, 0, ranked)?.expect("nothing scrubbed");
    assert_eq!(read[0].text.as_deref(), Some("current revision1"));
    let ranked = documents::rank_search(&captured, 0);
    core.memorize_document(input("a", "2", "current revision2", "g")?)?;
    assert!(documents::read_ranked(&captured, 0, ranked)?.is_none());
    let found = core.search_documents("current", &SearchOptions::default())?;
    assert_eq!(found[0].version, "2");
    assert_eq!(found[0].text.as_deref(), Some("current revision2"));
    Ok(())
}

/// The vector snapshot path reads hit text the same way: a replacement between the ranking
/// and the read is noticed, and the query falls back to a consistent answer.
#[test]
fn vector_hits_are_not_read_after_a_replacement_scrubs_their_text() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let core = WarmCore::new(
        1024,
        Some(directory.path().display().to_string()),
        Some(super::fixture_config(2)),
    )?;
    core.memorize_document(input("a", "1", "current revision1", "g")?)?;
    assert!(core.rebuild_vector_index(usize::MAX)?);
    let snapshot = core.vector_index.get().expect("published snapshot");
    let options = documents::VectorQueryOptions::default();
    let rank = || {
        documents::rank_vector_snapshot(
            &snapshot,
            &core.documents,
            &[1.0, 0.0],
            &[1.0, 0.0],
            1,
            &options,
        )
    };
    let ranked = rank()?.expect("current snapshot");
    let read =
        documents::read_vector_hits(ranked, &core.documents.read())?.expect("nothing scrubbed");
    assert_eq!(read[0].text.as_deref(), Some("current revision1"));
    let ranked = rank()?.expect("current snapshot");
    core.memorize_document(input("a", "2", "current revision2", "g")?)?;
    assert!(documents::read_vector_hits(ranked, &core.documents.read())?.is_none());
    let found = core.search_vectors(&[1.0, 0.0], 1, &options)?;
    assert_eq!(found[0].version, "2");
    assert_eq!(found[0].text.as_deref(), Some("current revision2"));
    Ok(())
}

#[test]
fn stale_document_training_cannot_publish_after_a_durable_replacement() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let config = super::fixture_config(2);
    let core = WarmCore::new(
        1024,
        Some(directory.path().display().to_string()),
        Some(config),
    )?;
    core.memorize_documents(
        (0..128)
            .map(|id| input(&id.to_string(), "old", "needle", "before"))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let snapshot = {
        let _transaction = core.mutation_gate.read();
        documents::capture_dense_index(&core.documents.read(), 2)?.unwrap()
    };
    let trained = documents::build_dense_index(snapshot)?.unwrap();
    core.memorize_document(input("0", "new", "replacement", "after")?)?;
    let published = {
        let _transaction = core.mutation_gate.read();
        documents::publish_dense_index(&mut core.documents.write(), &trained)
    };
    assert!(!published);
    assert!(!core.document_index_trained());
    let expected = serde_json::to_value(core.search_documents("needle", &Default::default())?)?;
    core.train_document_index()?;
    assert_eq!(
        serde_json::to_value(core.search_documents("needle", &Default::default())?)?,
        expected
    );
    Ok(())
}

#[test]
fn filtered_hybrid_delta_survives_compaction_and_reopen() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().display().to_string();
    let config = super::fixture_config(2);
    let core = WarmCore::new(1024, Some(path.clone()), Some(config.clone()))?;
    core.memorize_documents(
        (0..256)
            .map(|id| {
                let mut document = input(&format!("d{id:03}"), "old", "needle common", "before")?;
                document.embeddings = Some(vec![vec![0.0, 1.0]]);
                Ok(document)
            })
            .collect::<Result<Vec<_>>>()?,
    )?;
    core.train_dense_index()?;
    core.memorize_document(input(
        "d000",
        "new",
        "needle replacement longer text",
        "after",
    )?)?;
    core.delete_document("d001")?;
    core.memorize_document(input("d999", "new", "needle fresh", "before")?)?;
    let options = SearchOptions {
        filter: Some(json!({"group": "before"})),
        embedding: Some(vec![1.0, 0.0]),
        candidate_limit: Some(11),
        k: Some(7),
        fusion: Some("rrf".into()),
        ..Default::default()
    };
    let expected = core.search_documents("needle fresh", &options)?;
    assert!(!expected.is_empty());
    assert!(expected
        .iter()
        .all(|hit| hit.document_id != "d000" && hit.document_id != "d001"));
    assert!(expected.iter().any(|hit| hit.document_id == "d999"));
    for hit in &expected {
        assert_eq!(
            hit.text.as_ref().unwrap().len(),
            (hit.end_byte - hit.start_byte) as usize
        );
    }
    let expected = serde_json::to_value(expected)?;
    core.compact()?;
    assert_eq!(
        serde_json::to_value(core.search_documents("needle fresh", &options)?)?,
        expected
    );
    drop(core);
    let reopened = WarmCore::new(1024, Some(path), Some(config))?;
    assert_eq!(
        serde_json::to_value(reopened.search_documents("needle fresh", &options)?)?,
        expected
    );
    reopened.train_dense_index()?;
    assert_eq!(
        serde_json::to_value(reopened.search_documents("needle fresh", &options)?)?,
        expected
    );
    Ok(())
}

#[test]
fn dense_training_reports_busy_and_releases_build_capacity() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let core = Arc::new(WarmCore::new(
        1024,
        Some(directory.path().display().to_string()),
        Some(super::fixture_config(2)),
    )?);
    core.memorize_documents(
        (0..128)
            .map(|id| input(&id.to_string(), "old", "needle", "before"))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let held = core.vector_index.try_build().unwrap();
    let documents = core.documents.write();
    let (sender, receiver) = mpsc::channel();
    let contender = std::thread::spawn({
        let core = Arc::clone(&core);
        move || sender.send(core.train_dense_index()).unwrap()
    });
    let immediate = receiver.recv_timeout(Duration::from_secs(2));
    drop(documents);
    contender.join().unwrap();
    assert!(immediate
        .expect("busy maintenance must reject before capturing or locking documents")
        .unwrap_err()
        .to_string()
        .contains("maintenance already in progress"));
    assert!(!core.document_index_trained());
    drop(held);
    core.train_dense_index()?;
    assert!(core.document_index_trained());
    assert!(core.vector_index.try_build().is_some());
    Ok(())
}

/// A canonical rebuild whose rename fails leaves the store on its old text segment: the retry
/// stages over the staging file, not over the segment the store still reads.
#[test]
fn failed_canonical_rebuild_keeps_the_live_text_segment() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().display().to_string();
    let core = WarmCore::new(1024, Some(path), Some(super::fixture_config(2)))?;
    core.memorize_document(input("a", "v1", "kept alpha text", "g")?)?;
    core.memorize_document(input("b", "v1", "purged beta text", "g")?)?;
    core.delete_document("b")?;
    let text = directory.path().join("documents.text");
    let aside = directory.path().join("documents.text.aside");
    std::fs::rename(&text, &aside)?;
    std::fs::create_dir(&text)?;
    std::fs::write(text.join("occupied"), b"x")?;
    assert!(core.compact_scrubbing().is_err());
    std::fs::remove_file(text.join("occupied"))?;
    std::fs::remove_dir(&text)?;
    core.compact_scrubbing()?;
    let record = core.get_document("a").expect("document a");
    assert_eq!(record.text.as_deref(), Some("kept alpha text"));
    Ok(())
}

fn check_query_cache_errors(query: impl Fn(&WarmCore) -> Result<usize>) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let core = WarmCore::new(
        256,
        Some(directory.path().display().to_string()),
        Some(super::fixture_config(2)),
    )?;
    core.memorize_document(input("a", "1", "needle alpha beta", "g")?)?;
    core.memorize_document(input("b", "1", "needle alpha beta", "g")?)?;
    assert!(core.rebuild_vector_index(usize::MAX)?);
    assert_eq!(query(&core)?, 2);
    let path = directory.path().join("documents.text");
    let original = std::fs::read(&path)?;
    let damaged_at = core.documents.read().get("b").unwrap().chunks[0]
        .text_at
        .unwrap()
        .0 as usize;
    assert_eq!(&original[damaged_at..], b"needle alpha beta");
    let mut failures = Vec::new();
    for truncate in [false, true] {
        let mut damaged = original.clone();
        if truncate {
            damaged.pop();
        } else {
            damaged[damaged_at] ^= 1;
        }
        std::fs::write(&path, damaged)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| query(&core)));
        std::fs::write(&path, &original)?;
        let expected = if truncate {
            "reading chunk text"
        } else {
            "checksum mismatch"
        };
        match result {
            Ok(Err(error)) if format!("{error:#}").contains(expected) => {}
            Ok(other) => failures.push(format!(
                "truncate={truncate}: expected cache error, got {other:?}"
            )),
            Err(_) => failures.push(format!("truncate={truncate}: query panicked")),
        }
        assert_eq!(query(&core)?, 2);
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
    Ok(())
}

#[test]
fn document_queries_report_cache_errors_without_panicking() -> Result<()> {
    check_query_cache_errors(|core| {
        core.search_documents("needle", &SearchOptions::default())
            .map(|hits| hits.len())
    })
}

#[test]
fn vector_queries_report_cache_errors_without_panicking() -> Result<()> {
    check_query_cache_errors(|core| {
        core.search_vectors(&[1.0, 0.0], 2, &documents::VectorQueryOptions::default())
            .map(|hits| hits.len())
    })
}

#[test]
fn phrase_queries_report_cache_errors_without_panicking() -> Result<()> {
    check_query_cache_errors(|core| {
        core.search_lexical(
            &documents::LexicalQuery::parse("\"alpha beta\"")?,
            &documents::LexicalOptions::default(),
        )
        .map(|result| result.hits.len())
    })
}

#[test]
fn captured_queries_report_cache_errors_without_panicking() -> Result<()> {
    check_query_cache_errors(|core| {
        documents::search(
            &core.documents.read(),
            "needle",
            &SearchOptions::default(),
            Some(2),
        )
        .map(|hits| hits.len())
    })
}
