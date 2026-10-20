// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::io::{self, Write};

use anyhow::{ensure, Result};
use serde::Serialize;

use super::mutation::Mutation;
use crate::core::documents::{chunk_document, DocumentInput, MAX_DOCUMENT_BYTES};

const MAX_ITEMS: usize = 4096;
const MAX_EMBEDDING_VALUES: usize = 1_048_576;
const MAX_TRANSACTION_BYTES: usize = 32 * 1024 * 1024;

struct SizeLimit {
    bytes: usize,
    limit: usize,
}

impl Write for SizeLimit {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self
            .bytes
            .checked_add(bytes.len())
            .filter(|&n| n <= self.limit);
        match next {
            Some(next) => {
                self.bytes = next;
                Ok(bytes.len())
            }
            None => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "document batch size limit exceeded",
            )),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialized_size(value: &impl Serialize, limit: usize) -> Result<usize> {
    let mut sink = SizeLimit { bytes: 0, limit };
    serde_json::to_writer(&mut sink, value)?;
    Ok(sink.bytes)
}

fn add_size(total: &mut usize, size: usize, limit: usize, message: &str) -> Result<()> {
    *total = total
        .checked_add(size)
        .filter(|&n| n <= limit)
        .ok_or_else(|| anyhow::anyhow!("{message}"))?;
    Ok(())
}

pub(super) fn validate_inputs(
    inputs: &[DocumentInput],
    inferred_dimensions: Option<usize>,
) -> Result<()> {
    ensure!(
        !inputs.is_empty() && inputs.len() <= MAX_ITEMS,
        "document batch requires 1..=4096 items"
    );
    let mut ids = HashSet::with_capacity(inputs.len());
    let mut source_bytes = 0;
    let mut metadata_bytes = 0;
    let mut embedding_values = 0;
    let mut chunks = 0;
    for input in inputs {
        ensure!(
            ids.insert(input.id.as_str()),
            "document batch IDs must be unique"
        );
        for size in [
            input.id.len(),
            input.text.len(),
            input.source_uri.as_ref().map_or(0, String::len),
            input.version.as_ref().map_or(0, String::len),
        ] {
            add_size(
                &mut source_bytes,
                size,
                MAX_DOCUMENT_BYTES,
                "document batch source exceeds 8 MiB",
            )?;
        }
        if let Some(metadata) = &input.metadata {
            add_size(
                &mut metadata_bytes,
                serialized_size(metadata, MAX_DOCUMENT_BYTES)?,
                MAX_DOCUMENT_BYTES,
                "document batch metadata exceeds 8 MiB",
            )?;
        }
        let chunk_count = chunk_document(input)?.len();
        if let Some(embeddings) = &input.embeddings {
            for embedding in embeddings {
                add_size(
                    &mut embedding_values,
                    embedding.len(),
                    MAX_EMBEDDING_VALUES,
                    "document batch exceeds 1048576 embedding values",
                )?;
            }
        } else if let Some(dimensions) = inferred_dimensions {
            let values = chunk_count
                .checked_mul(dimensions)
                .ok_or_else(|| anyhow::anyhow!("document batch embedding allocation overflow"))?;
            add_size(
                &mut embedding_values,
                values,
                MAX_EMBEDDING_VALUES,
                "document batch exceeds 1048576 embedding values",
            )?;
        }
        add_size(
            &mut chunks,
            chunk_count,
            MAX_ITEMS,
            "document batch exceeds 4096 chunks",
        )?;
    }
    Ok(())
}

pub(super) fn validate_prepared(mutations: &[Mutation]) -> Result<()> {
    serialized_size(&mutations, MAX_TRANSACTION_BYTES)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn input(id: &str) -> DocumentInput {
        serde_json::from_value(json!({"id": id, "text": "needle"})).unwrap()
    }

    #[test]
    fn batch_item_and_chunk_boundaries_are_independent() {
        for count in [4095, 4096, 4097] {
            let inputs: Vec<_> = (0..count).map(|id| input(&id.to_string())).collect();
            assert_eq!(validate_inputs(&inputs, None).is_ok(), count <= MAX_ITEMS);
        }
        assert!(validate_inputs(&[], None).is_err());
        for count in [4095, 4096, 4097] {
            let mut inputs: Vec<_> = (0..count - 2).map(|id| input(&id.to_string())).collect();
            let mut last = input("last");
            last.chunk_words = Some(1);
            last.overlap_words = Some(0);
            last.text = "two chunks".into();
            inputs.push(last);
            assert_eq!(validate_inputs(&inputs, None).is_ok(), count <= MAX_ITEMS);
        }
    }

    #[test]
    fn aggregate_source_and_metadata_budget_boundaries() {
        let mut sources: Vec<_> = (0..1024)
            .map(|id| {
                let mut document = input(&id.to_string());
                document.source_uri = Some("s".repeat(8184));
                document
            })
            .collect();
        let prefix: usize = sources
            .iter()
            .take(1023)
            .map(|document| {
                document.id.len()
                    + document.text.len()
                    + document.source_uri.as_ref().unwrap().len()
            })
            .sum();
        let last = sources.last_mut().unwrap();
        let remaining = MAX_DOCUMENT_BYTES - prefix - last.id.len() - last.text.len();
        assert!(remaining < 8192);
        for count in [remaining - 1, remaining, remaining + 1] {
            sources.last_mut().unwrap().source_uri = Some("s".repeat(count));
            assert_eq!(validate_inputs(&sources, None).is_ok(), count <= remaining);
        }
        let mut metadata: Vec<_> = (0..129)
            .map(|id| {
                let mut document = input(&id.to_string());
                document.metadata = Some(json!({"x": "m".repeat(65030)}));
                document
            })
            .collect();
        let prefix: usize = metadata
            .iter()
            .take(128)
            .map(|document| {
                serialized_size(document.metadata.as_ref().unwrap(), MAX_DOCUMENT_BYTES).unwrap()
            })
            .sum();
        let remaining = MAX_DOCUMENT_BYTES
            - prefix
            - serialized_size(&json!({"x": ""}), MAX_DOCUMENT_BYTES).unwrap();
        assert!(remaining < 65030);
        for count in [remaining - 1, remaining, remaining + 1] {
            metadata.last_mut().unwrap().metadata = Some(json!({"x": "m".repeat(count)}));
            assert_eq!(validate_inputs(&metadata, None).is_ok(), count <= remaining);
        }
    }

    #[test]
    fn document_batch_is_one_recoverable_transaction_and_invalid_updates_are_atomic() -> Result<()>
    {
        use crate::core::{EmbeddingSpace, HmsConfig};
        let directory = tempfile::tempdir()?;
        let path = directory.path().display().to_string();
        let config = HmsConfig {
            embedding_space: Some(EmbeddingSpace {
                model: "fixture".into(),
                revision: "1".into(),
                dimensions: 3,
                normalization: "l2".into(),
                metric: "cosine".into(),
            }),
            ..Default::default()
        };
        let core = super::super::HmsCore::new(1024, Some(path.clone()), Some(config.clone()))?;
        let mut first = input("a");
        first.embeddings = Some(vec![vec![1.0, 0.0, 0.0]]);
        core.memorize_document(first.clone())?;
        let revision = core.revision.load(std::sync::atomic::Ordering::Acquire);
        first.version = Some("2".into());
        let mut second = input("b");
        second.text = "needle three four".into();
        second.chunk_words = Some(1);
        second.overlap_words = Some(0);
        second.embeddings = Some(vec![vec![0.0, 1.0, 0.0]; 3]);
        assert_eq!(
            core.memorize_documents(vec![first.clone(), second])?,
            [1, 3]
        );
        assert_eq!(
            core.revision.load(std::sync::atomic::Ordering::Acquire),
            revision + 1
        );
        let expected = serde_json::to_value(core.search_documents("needle", &Default::default())?)?;
        let revision = core.revision.load(std::sync::atomic::Ordering::Acquire);
        first.version = Some("invalid".into());
        let mut invalid = input("c");
        invalid.embeddings = Some(vec![vec![0.0, 0.0, 0.0]]);
        assert!(core
            .memorize_documents(vec![first, input("new"), invalid])
            .is_err());
        assert_eq!(
            core.revision.load(std::sync::atomic::Ordering::Acquire),
            revision
        );
        assert_eq!(
            serde_json::to_value(core.search_documents("needle", &Default::default())?)?,
            expected
        );
        assert_eq!(core.vector_count(), 4);
        core.compact()?;
        drop(core);
        let core = super::super::HmsCore::new(1024, Some(path), Some(config))?;
        assert_eq!(core.vector_count(), 4);
        assert_eq!(
            serde_json::to_value(core.search_documents("needle", &Default::default())?)?,
            expected
        );
        assert!(core.delete_document("b")?);
        assert_eq!(core.vector_count(), 1);
        Ok(())
    }

    #[test]
    fn readers_observe_whole_document_batches() -> Result<()> {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        let directory = tempfile::tempdir()?;
        let core = Arc::new(super::super::HmsCore::new(
            1024,
            Some(directory.path().display().to_string()),
            None,
        )?);
        let done = Arc::new(AtomicBool::new(false));
        let writer_core = Arc::clone(&core);
        let writer_done = Arc::clone(&done);
        let writer = std::thread::spawn(move || {
            let result =
                writer_core.memorize_documents((0..300).map(|id| input(&id.to_string())).collect());
            writer_done.store(true, Ordering::Release);
            result
        });
        let options = crate::SearchOptions {
            k: Some(300),
            candidate_limit: Some(300),
            ..Default::default()
        };
        let mut reads = 0;
        while !done.load(Ordering::Acquire) || reads < 32 {
            let results = core.search_documents("needle", &options)?;
            assert!(results.is_empty() || results.len() == 300);
            reads += 1;
        }
        assert_eq!(writer.join().unwrap()?.len(), 300);
        assert_eq!(core.search_documents("needle", &options)?.len(), 300);
        Ok(())
    }

    #[test]
    fn embedding_budget_and_duplicate_identity_reject_before_preparation() {
        for count in [
            MAX_EMBEDDING_VALUES - 1,
            MAX_EMBEDDING_VALUES,
            MAX_EMBEDDING_VALUES + 1,
        ] {
            let mut document = input("a");
            document.embeddings = Some(vec![vec![1.0; count]]);
            assert_eq!(
                validate_inputs(&[document], None).is_ok(),
                count <= MAX_EMBEDDING_VALUES
            );
        }
        assert!(validate_inputs(&[input("same"), input("same")], None).is_err());
        for count in [1023, 1024, 1025] {
            let inputs: Vec<_> = (0..count).map(|id| input(&id.to_string())).collect();
            assert_eq!(validate_inputs(&inputs, Some(1024)).is_ok(), count <= 1024);
        }
        assert!(validate_inputs(&[input("large")], Some(usize::MAX)).is_err());
    }

    #[test]
    fn serialized_budget_handles_limit_neighbors_without_allocating_output() {
        let value = "source";
        let size = serde_json::to_vec(value).unwrap().len();
        for limit in [size - 1, size, size + 1] {
            assert_eq!(serialized_size(&value, limit).is_ok(), limit >= size);
        }
        let mut total = usize::MAX;
        assert!(add_size(&mut total, 1, usize::MAX, "overflow").is_err());
    }
}
