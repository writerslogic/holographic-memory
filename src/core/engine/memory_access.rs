// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Read and delete access for the memory API (`core::memory_api`). Memory facts are ordinary
//! documents, so nothing here adds a persisted format; it only exposes what the API needs
//! from the document store without widening its visibility.

use anyhow::Result;
use serde_json::Value;

use super::{mutation::Mutation, HmsCore};

/// Most deletions one journal transaction carries.
pub(crate) const DELETE_BATCH: usize = 4096;

/// Everything needed to store a document again unchanged: its first chunk's text and
/// embedding, its metadata, source and version.
pub(crate) struct DocumentParts {
    pub metadata: Value,
    pub text: Option<String>,
    pub source_uri: Option<String>,
    pub version: String,
    pub embedding: Option<Vec<f64>>,
}

/// A document as the memory API sees it: its metadata and its first chunk's text. Memory
/// facts are stored as a single chunk.
pub(crate) struct DocumentView<'a> {
    pub id: &'a str,
    pub version: &'a str,
    pub metadata: &'a Value,
    text: &'a dyn Fn() -> Option<String>,
}

impl DocumentView<'_> {
    pub(crate) fn text(&self) -> Option<String> {
        (self.text)()
    }
}

impl HmsCore {
    /// The number of write transactions this process has committed and replayed so far. It
    /// changes whenever the store does, so equal values bracket an unchanged store.
    pub fn revision(&self) -> u64 {
        self.revision.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Visits every document under one consistent snapshot and returns that snapshot's
    /// revision.
    pub(crate) fn scan_documents(&self, mut visit: impl FnMut(DocumentView<'_>)) -> u64 {
        let _transaction = self.mutation_gate.read();
        let revision = self.revision();
        let store = self.documents.read();
        for document in store.shared_values() {
            let text = || {
                document
                    .chunks
                    .first()
                    .and_then(|chunk| store.chunk_text(chunk))
            };
            visit(DocumentView {
                id: &document.id,
                version: &document.version,
                metadata: &document.metadata,
                text: &text,
            });
        }
        revision
    }

    /// Metadata and text of one document.
    pub(crate) fn document_snapshot(&self, id: &str) -> Option<(String, Value, Option<String>)> {
        self.document_parts(id)
            .map(|parts| (parts.version, parts.metadata, parts.text))
    }

    pub(crate) fn document_parts(&self, id: &str) -> Option<DocumentParts> {
        let _transaction = self.mutation_gate.read();
        let store = self.documents.read();
        let document = store.get(id)?;
        let chunk = document.chunks.first();
        Some(DocumentParts {
            metadata: document.metadata.clone(),
            text: chunk.and_then(|chunk| store.chunk_text(chunk)),
            source_uri: document.source_uri.clone(),
            version: document.version.clone(),
            embedding: chunk.and_then(|chunk| chunk.embedding.as_ref().map(|e| e.to_f64())),
        })
    }

    /// Deletes documents in transactions of at most [`DELETE_BATCH`]; returns how many
    /// existed and how many transactions were committed. Deleting an absent id is a no-op,
    /// so a retry after a crash between transactions completes the rest.
    pub(crate) fn delete_documents(&self, ids: &[String]) -> Result<(usize, u64)> {
        let mut deleted = 0;
        let mut commits = 0;
        for batch in ids.chunks(DELETE_BATCH) {
            let existing: Vec<&String> = {
                let store = self.documents.read();
                batch.iter().filter(|id| store.contains_key(id)).collect()
            };
            if existing.is_empty() {
                continue;
            }
            let mutations: Vec<Mutation> = existing
                .iter()
                .map(|id| Mutation::DeleteDocument { id: (*id).clone() })
                .collect();
            self.commit(&mutations)?;
            deleted += existing.len();
            commits += 1;
        }
        Ok((deleted, commits))
    }
}
