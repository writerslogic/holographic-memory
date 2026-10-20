// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::sync::{mpsc, Arc};
use std::time::Duration;

use anyhow::{bail, Result};
use serde_json::json;

use super::HmsCore;
use crate::core::documents::{self, DocumentInput, SearchOptions};
use crate::core::{EmbeddingSpace, HmsConfig};

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
    let config = HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "fixture".into(),
            revision: "1".into(),
            dimensions: 2,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    let core = Arc::new(HmsCore::new(
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
    let old = documents::execute_search(&captured, 0);
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
    let second = documents::execute_search(&captured, 1);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].document_id, "b");
    assert_eq!(
        second[0].text.as_deref(),
        Some("needleold needleoldremoved")
    );
    let restored = HmsCore::new(1024, Some(path), Some(config))?;
    let current = restored.search_documents("needleold", &options)?;
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].document_id, "c");
    assert_eq!(current[0].version, "new");
    assert!(restored
        .search_documents("needleoldremoved", &options)?
        .is_empty());
    Ok(())
}

#[test]
fn stale_document_training_cannot_publish_after_a_durable_replacement() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let config = HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "fixture".into(),
            revision: "1".into(),
            dimensions: 2,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    let core = HmsCore::new(
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
