// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::{ensure, Result};
use rayon::prelude::*;

use super::mutation::{Mutation, TRANSACTION_MAGIC};
use super::HmsCore;
use crate::core::documents::{
    chunk_spans_with, serialized_size, BatchTooLarge, DocumentInput, IngestOptions,
    MAX_BATCH_ITEMS as MAX_ITEMS, MAX_DOCUMENT_BYTES,
};

const MAX_EMBEDDING_VALUES: usize = 1_048_576;
const MAX_TRANSACTION_BYTES: usize = 32 * 1024 * 1024;

fn add_size(total: &mut usize, size: usize, limit: usize, message: &'static str) -> Result<()> {
    *total = total
        .checked_add(size)
        .filter(|&n| n <= limit)
        .ok_or(BatchTooLarge(message))?;
    Ok(())
}

/// Per input: its serialized metadata size, if it has metadata, and its chunk spans.
type Measured = (Option<Result<usize>>, Result<Vec<(u32, u32)>>);

fn source_sizes(input: &DocumentInput) -> [usize; 4] {
    [
        input.id.len(),
        input.text.len(),
        input.source_uri.as_ref().map_or(0, String::len),
        input.version.as_ref().map_or(0, String::len),
    ]
}

/// Checks a batch against every limit and returns each input's chunk spans
/// ([`chunk_spans_with`]), which preparation then reuses.
pub(super) fn validate_inputs_with(
    inputs: &[DocumentInput],
    inferred_dimensions: Option<usize>,
    options: &IngestOptions,
) -> Result<Vec<Vec<(u32, u32)>>> {
    const ITEMS: &str = "document batch requires 1..=4096 items";
    ensure!(!inputs.is_empty(), ITEMS);
    if inputs.len() > MAX_ITEMS {
        return Err(BatchTooLarge(ITEMS).into());
    }
    // The fold below stops at the input whose source bytes overflow the budget, if any, so no
    // input after it can be the one reported: only the inputs up to it are measured.
    let mut total = 0usize;
    let measured_len = inputs
        .iter()
        .position(|input| {
            source_sizes(input).into_iter().any(|size| {
                total = total.saturating_add(size);
                total > MAX_DOCUMENT_BYTES
            })
        })
        .map_or(inputs.len(), |at| at + 1);
    // The per-input measures (metadata size, chunk spans) are computed in parallel; the
    // budgets then fold in input order, so the first input over a limit is the one reported,
    // as it is when everything runs serially.
    let measured: Vec<Measured> = inputs[..measured_len]
        .par_iter()
        .map(|input| {
            let metadata = input
                .metadata
                .as_ref()
                .map(|metadata| serialized_size(metadata, MAX_DOCUMENT_BYTES));
            (metadata, chunk_spans_with(input, options))
        })
        .collect();
    let mut ids = HashSet::with_capacity(inputs.len());
    let mut source_bytes = 0;
    let mut metadata_bytes = 0;
    let mut embedding_values = 0;
    let mut chunks = 0;
    let mut spans = Vec::with_capacity(inputs.len());
    for (input, (metadata_size, chunk_spans)) in inputs.iter().zip(measured) {
        ensure!(
            ids.insert(input.id.as_str()),
            "document batch IDs must be unique"
        );
        for size in source_sizes(input) {
            add_size(
                &mut source_bytes,
                size,
                MAX_DOCUMENT_BYTES,
                "document batch source exceeds 8 MiB",
            )?;
        }
        if let Some(metadata_size) = metadata_size {
            add_size(
                &mut metadata_bytes,
                metadata_size?,
                MAX_DOCUMENT_BYTES,
                "document batch metadata exceeds 8 MiB",
            )?;
        }
        let chunk_spans = chunk_spans?;
        let chunk_count = chunk_spans.len();
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
        spans.push(chunk_spans);
    }
    Ok(spans)
}

#[cfg(test)]
fn validate_inputs(
    inputs: &[DocumentInput],
    inferred_dimensions: Option<usize>,
) -> Result<Vec<Vec<(u32, u32)>>> {
    validate_inputs_with(inputs, inferred_dimensions, &IngestOptions::default())
}

/// The journal payload of a prepared document batch in the store's format, serialized once
/// and checked against the transaction limit; `HmsCore::commit_serialized` appends it as is.
pub(super) fn serialize_transaction(core: &HmsCore, mutations: &[Mutation]) -> Result<Vec<u8>> {
    let payload = core.transaction_payload(mutations)?;
    // Both magics are the same length.
    if payload.len() - TRANSACTION_MAGIC.len() > MAX_TRANSACTION_BYTES {
        return Err(BatchTooLarge("document batch size limit exceeded").into());
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn input(id: &str) -> DocumentInput {
        serde_json::from_value(json!({"id": id, "text": "needle"})).unwrap()
    }

    /// Accepted when `within`; otherwise rejected as a retryable size limit
    /// ([`BatchTooLarge`]), not as an invalid batch.
    fn assert_limit<T>(result: Result<T>, within: bool, case: &str) {
        match result {
            Ok(_) => assert!(within, "{case}: accepted beyond the limit"),
            Err(error) => assert!(!within && error.is::<BatchTooLarge>(), "{case}: {error:#}"),
        }
    }

    #[test]
    fn batch_item_and_chunk_boundaries_are_independent() {
        for count in [4095, 4096, 4097] {
            let inputs: Vec<_> = (0..count).map(|id| input(&id.to_string())).collect();
            assert_limit(
                validate_inputs(&inputs, None),
                count <= MAX_ITEMS,
                &format!("{count} items"),
            );
        }
        let empty = validate_inputs(&[], None).unwrap_err();
        assert!(
            !empty.is::<BatchTooLarge>(),
            "an empty batch is not a size limit"
        );
        for count in [4095, 4096, 4097] {
            let mut inputs: Vec<_> = (0..count - 2).map(|id| input(&id.to_string())).collect();
            let mut last = input("last");
            last.chunk_words = Some(1);
            last.overlap_words = Some(0);
            last.text = "two chunks".into();
            inputs.push(last);
            assert_limit(
                validate_inputs(&inputs, None),
                count <= MAX_ITEMS,
                &format!("{count} chunks"),
            );
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
            assert_limit(
                validate_inputs(&sources, None),
                count <= remaining,
                &format!("source {count}"),
            );
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
            assert_limit(
                validate_inputs(&metadata, None),
                count <= remaining,
                &format!("metadata {count}"),
            );
        }
    }

    #[test]
    fn document_batch_is_one_recoverable_transaction_and_invalid_updates_are_atomic() -> Result<()>
    {
        let directory = tempfile::tempdir()?;
        let path = directory.path().display().to_string();
        let config = super::super::fixture_config(3);
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
            assert_limit(
                validate_inputs(&[document], None),
                count <= MAX_EMBEDDING_VALUES,
                &format!("{count} embedding values"),
            );
        }
        let duplicate = validate_inputs(&[input("same"), input("same")], None).unwrap_err();
        assert!(
            !duplicate.is::<BatchTooLarge>(),
            "a duplicate id is not a size limit"
        );
        for count in [1023, 1024, 1025] {
            let inputs: Vec<_> = (0..count).map(|id| input(&id.to_string())).collect();
            assert_limit(
                validate_inputs(&inputs, Some(1024)),
                count <= 1024,
                &format!("{count} inferred embeddings"),
            );
        }
        assert_limit(
            validate_inputs(&[input("large")], Some(usize::MAX)),
            false,
            "usize::MAX dimensions",
        );
    }

    #[test]
    fn serialized_budget_handles_limit_neighbors_without_allocating_output() {
        let value = "source";
        let size = serde_json::to_vec(value).unwrap().len();
        for limit in [size - 1, size, size + 1] {
            assert_eq!(serialized_size(&value, limit).is_ok(), limit >= size);
        }
        let mut total = usize::MAX;
        assert_limit(
            add_size(&mut total, 1, usize::MAX, "overflow"),
            false,
            "overflow",
        );
    }

    /// The journal transaction of each batch is byte-identical whatever the thread count and
    /// whether the store keeps its batch interners: they are pure caches, so neither which
    /// split stems a word first nor what earlier batches left in the pool shows in the terms.
    #[test]
    fn batch_payloads_do_not_depend_on_threads_or_kept_interners() -> Result<()> {
        let vocabulary = [
            "running",
            "runners",
            "RUN",
            "Quickly",
            "the",
            "ΟΔΟΣ",
            "Οδός",
            "naïve",
            "Café",
            "straße",
            "İstanbul",
            "x2",
            "42",
            "abcdefghijklmnopq",
            "generalizations",
            "relational",
            "conditional",
            "hopefulness",
            "agreed",
            "flies",
        ];
        let batches: Vec<Vec<DocumentInput>> = (0..3)
            .map(|batch| {
                (0..64)
                    .map(|doc| {
                        let text: Vec<String> = (0..90)
                            .map(|w| {
                                let i = (doc * 7 + w * 13 + batch * 3) % 41;
                                match vocabulary.get(i) {
                                    Some(word) => (*word).to_owned(),
                                    None => format!("w{}x{}", batch * 1000 + doc, i),
                                }
                            })
                            .collect();
                        serde_json::from_value(json!({
                            "id": format!("d{batch}-{doc}"),
                            "text": text.join(" "),
                            "chunkWords": 24,
                            "overlapWords": 4,
                        }))
                        .unwrap()
                    })
                    .collect()
            })
            .collect();
        let options = IngestOptions {
            encode: crate::core::documents::EncodeMode::Lexical,
            ..Default::default()
        };
        let payloads = |threads: usize, keep: bool| -> Result<Vec<Vec<u8>>> {
            let directory = tempfile::tempdir()?;
            let core = HmsCore::new(1024, Some(directory.path().display().to_string()), None)?;
            core.set_keep_ingest_interner(keep);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?;
            pool.install(|| {
                batches
                    .iter()
                    .map(|batch| {
                        let (mutations, payload) = core.prepare_batch(batch.clone(), &options)?;
                        core.commit_serialized(&mutations, &payload)?;
                        Ok(payload)
                    })
                    .collect()
            })
        };
        let expected = payloads(1, true)?;
        for (threads, keep) in [(1, false), (2, true), (8, true), (8, false)] {
            assert!(
                payloads(threads, keep)? == expected,
                "threads {threads}, keep {keep}"
            );
        }
        Ok(())
    }
}
