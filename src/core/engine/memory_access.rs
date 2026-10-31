// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Read and delete access for the memory API (`core::memory_api`). Memory facts are ordinary
//! documents, so nothing here adds a persisted format; it only exposes what the API needs
//! from the document store without widening its visibility.

use anyhow::Result;
use serde_json::Value;

use super::{mutation::Mutation, WarmCore};

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

impl WarmCore {
    pub(crate) fn validate_document_search(
        &self,
        text: &str,
        options: &crate::SearchOptions,
    ) -> Result<()> {
        crate::core::documents::validate_search(
            text,
            options,
            self.config
                .embedding_space
                .as_ref()
                .map(|space| space.dimensions),
        )
        .map(drop)
    }

    /// Advances by one per write transaction this process commits or replays, so equal values
    /// bracket an unchanged store and the difference of two readings counts the transactions
    /// between them. It starts at a random value per open, so one reading does not reveal how
    /// many transactions (inserts, deletes) the store has seen.
    pub fn revision(&self) -> u64 {
        self.revision.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Visits every document under one consistent snapshot and returns that snapshot's
    /// revision.
    pub(crate) fn scan_documents(&self, mut visit: impl FnMut(DocumentView<'_>)) -> u64 {
        self.try_scan_documents(None, None, |document| {
            visit(document);
            Ok(())
        })
        .expect("infallible document visitor")
    }

    pub(crate) fn try_scan_documents(
        &self,
        ids: Option<&std::collections::BTreeSet<String>>,
        filter: Option<&Value>,
        mut visit: impl FnMut(DocumentView<'_>) -> Result<()>,
    ) -> Result<u64> {
        let _transaction = self.mutation_gate.read();
        let revision = self.revision();
        self.hydrate_documents()?;
        let store = self.documents.read();
        let mut visit_document =
            |document: std::sync::Arc<crate::core::documents::StoredDocument>| {
                if ids.is_some_and(|ids| !ids.contains(&document.id))
                    || !crate::core::documents::matches_metadata(&document.metadata, filter)
                {
                    return Ok(());
                }
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
                })
            };
        let candidates = match (ids, store.metadata_candidates(filter)) {
            (Some(ids), Some(metadata)) if metadata.len() < ids.len() => Some(metadata),
            (Some(ids), _) => Some(ids),
            (None, metadata) => metadata,
        };
        if let Some(ids) = candidates {
            for id in ids {
                if let Some(document) = store.get(id) {
                    visit_document(std::sync::Arc::clone(document))?;
                }
            }
        } else {
            for document in store.shared_values() {
                visit_document(document)?;
            }
        }
        Ok(revision)
    }

    /// Metadata and text of one document.
    pub(crate) fn document_snapshot(&self, id: &str) -> Option<(String, Value, Option<String>)> {
        self.document_parts(id)
            .map(|parts| (parts.version, parts.metadata, parts.text))
    }

    pub(crate) fn document_parts(&self, id: &str) -> Option<DocumentParts> {
        let _transaction = self.mutation_gate.read();
        let store = self.documents();
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

    pub(crate) fn memory_revision_matches(&self, revision: u64) -> bool {
        let _writer = self.mutation_gate.maintenance();
        self.revision() == revision
    }

    pub(crate) fn change_memory_documents(
        &self,
        inputs: Vec<crate::DocumentInput>,
        deleted: &[String],
        revision: u64,
    ) -> Result<bool> {
        anyhow::ensure!(
            inputs
                .len()
                .checked_add(deleted.len())
                .is_some_and(|n| n <= DELETE_BATCH),
            "memory transaction requires at most 4096 document changes"
        );
        let mut mutations = if inputs.is_empty() {
            Vec::new()
        } else {
            self.prepare_batch(inputs, &Default::default())?.0
        };
        mutations.extend(
            deleted
                .iter()
                .cloned()
                .map(|id| Mutation::DeleteDocument { id }),
        );
        if mutations.is_empty() {
            return Ok(self.memory_revision_matches(revision));
        }
        let payload = super::document_batch::serialize_transaction(self, &mutations)?;
        self.commit_serialized_if_revision(&mutations, &payload, revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_scan_intersects_scope_and_checks_unindexed_values() -> Result<()> {
        use serde_json::json;
        use std::collections::BTreeSet;

        let directory = tempfile::tempdir()?;
        let core = WarmCore::new(256, Some(directory.path().display().to_string()), None)?;
        for (id, group, selected) in [
            ("a", 1, false),
            ("b", 1, true),
            ("c", 2, true),
            ("d", 2, false),
        ] {
            core.memorize_document(serde_json::from_value(json!({
                "id": id, "text": "evidence", "metadata": {"group": group, "nested": {"selected": selected}}
            }))?)?;
        }
        for (scope, filter, expected) in [
            (
                Some(BTreeSet::from(["a".into(), "b".into(), "c".into()])),
                json!({"group": 2}),
                vec!["c"],
            ),
            (
                Some(BTreeSet::from(["b".into()])),
                json!({"group": 1}),
                vec!["b"],
            ),
            (
                None,
                json!({"group": 2, "nested": {"selected": true}}),
                vec!["c"],
            ),
            (None, json!({"nested": {"selected": true}}), vec!["b", "c"]),
            (None, json!({"group": 3}), vec![]),
        ] {
            let mut visited = Vec::new();
            let revision = core.try_scan_documents(scope.as_ref(), Some(&filter), |document| {
                visited.push(document.id.to_owned());
                Ok(())
            })?;
            assert_eq!(revision, core.revision());
            assert_eq!(visited, expected);
        }
        Ok(())
    }

    #[test]
    fn fallible_scoped_scan_stops_at_the_first_error() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let core = WarmCore::new(256, Some(directory.path().display().to_string()), None)?;
        for id in ["a", "b", "c"] {
            core.memorize_document(serde_json::from_value(
                serde_json::json!({"id": id, "text": "evidence"}),
            )?)?;
        }
        let scope = std::collections::BTreeSet::from(["b".into(), "c".into()]);
        let mut visited = Vec::new();
        let result = core.try_scan_documents(Some(&scope), None, |document| {
            visited.push(document.id.to_owned());
            anyhow::bail!("budget exhausted")
        });
        assert!(result.is_err());
        assert_eq!(visited, ["b"]);
        Ok(())
    }
}
