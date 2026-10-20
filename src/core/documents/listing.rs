// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Read access to stored documents by id and in id order, for callers that must enumerate or
//! export a store (the search path only returns documents that match a query).

use serde::Serialize;
use serde_json::Value;

use super::index::text_of;
use super::{DocumentStore, StoredDocument};

/// A stored document as the store holds it: first chunk's text, metadata and version.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentRecord {
    pub id: String,
    pub source_uri: Option<String>,
    pub version: String,
    pub metadata: Value,
    /// Text of the first chunk; `None` for a document stored without text. A document that
    /// fit one chunk when stored reads back exactly (modulo leading and trailing whitespace).
    pub text: Option<String>,
    pub chunk_count: usize,
}

fn record(store: &DocumentStore, document: &StoredDocument) -> DocumentRecord {
    let segment = store.text_segment();
    DocumentRecord {
        id: document.id.clone(),
        source_uri: document.source_uri.clone(),
        version: document.version.clone(),
        metadata: document.metadata.clone(),
        text: document
            .chunks
            .first()
            .and_then(|chunk| text_of(chunk, segment.as_deref())),
        chunk_count: document.chunks.len(),
    }
}

pub(crate) fn get_record(store: &DocumentStore, id: &str) -> Option<DocumentRecord> {
    store.get(id).map(|document| record(store, document))
}

/// Up to `limit` documents with ids at or after `from`, in id order.
pub(crate) fn list_records(
    store: &DocumentStore,
    from: Option<&str>,
    limit: usize,
) -> Vec<DocumentRecord> {
    use std::ops::Bound::{Included, Unbounded};
    let start = from.map_or(Unbounded, Included);
    store
        .documents
        .range::<str, _>((start, Unbounded))
        .take(limit)
        .map(|(_, document)| record(store, document))
        .collect()
}

pub(crate) fn count(store: &DocumentStore) -> usize {
    store.documents.len()
}

#[cfg(test)]
mod tests {
    use super::super::{prepare, DocumentInput, DocumentStore};
    use super::*;

    fn store_with(ids: &[&str]) -> DocumentStore {
        let mut store = DocumentStore::default();
        for id in ids {
            let input = DocumentInput {
                id: (*id).into(),
                text: format!("body of {id}"),
                source_uri: None,
                version: Some("v1".into()),
                metadata: Some(serde_json::json!({"k": 1})),
                chunk_words: None,
                overlap_words: None,
                store_text: Some(true),
                embeddings: None,
                title_weight: None,
            };
            let prepared = prepare(input, 256, None, Default::default()).unwrap();
            store.insert(prepared.id.clone(), prepared);
        }
        store
    }

    #[test]
    fn lists_in_id_order_and_resumes_after_a_cursor() {
        let store = store_with(&["b", "a", "d", "c"]);
        assert_eq!(count(&store), 4);
        let first = list_records(&store, None, 2);
        let ids =
            |records: &[DocumentRecord]| records.iter().map(|r| r.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&first), ["a", "b"]);
        let resume = format!("{}\0", first[1].id);
        assert_eq!(ids(&list_records(&store, Some(&resume), 10)), ["c", "d"]);
        assert!(list_records(&store, Some("e"), 10).is_empty());
        assert_eq!(list_records(&store, Some("c"), 1)[0].id, "c");
    }

    #[test]
    fn get_returns_text_metadata_and_version() {
        let store = store_with(&["x"]);
        let record = get_record(&store, "x").unwrap();
        assert_eq!(record.text.as_deref(), Some("body of x"));
        assert_eq!(record.version, "v1");
        assert_eq!(record.metadata["k"], 1);
        assert_eq!(record.chunk_count, 1);
        assert!(get_record(&store, "missing").is_none());
    }
}
