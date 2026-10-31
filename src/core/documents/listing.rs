// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Read access to stored documents by id and in id order, for callers that must enumerate or
//! export a store (the search path only returns documents that match a query).

use serde::Serialize;
use serde_json::Value;

use std::sync::Arc;

use anyhow::{ensure, Context, Result};

use super::index::text_of;
use super::text::TextSegment;
use super::{DocumentStore, StoredDocument};

/// Document identity, provenance and metadata without copying stored chunk text.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct DocumentInfo {
    pub id: String,
    pub source_uri: Option<String>,
    pub version: String,
    pub metadata: Value,
    pub chunk_count: usize,
}

/// One retained chunk, with its source byte span and document metadata.
/// `text` is absent when text retention was disabled; spans do not restore omitted whitespace.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct DocumentChunkRecord {
    /// Opaque store revision captured with this chunk; compare across reads, do not interpret numerically.
    pub revision: String,
    pub document: DocumentInfo,
    pub index: usize,
    pub id: String,
    pub start_byte: u32,
    pub end_byte: u32,
    pub text: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("document changed repeatedly while reading; retry")]
pub struct DocumentReadChanged;

pub(crate) struct CapturedChunkRead {
    record: DocumentChunkRecord,
    segment: Option<(Arc<TextSegment>, u64, u32, u32)>,
    document: Option<Arc<StoredDocument>>,
}

pub(crate) struct ChunkReadIdentity {
    document: Option<Arc<StoredDocument>>,
    segment: Option<Arc<TextSegment>>,
}

impl ChunkReadIdentity {
    pub(crate) fn is_current(&self, store: &DocumentStore) -> bool {
        let Some(document) = &self.document else {
            return true;
        };
        store
            .get(&document.id)
            .is_some_and(|current| Arc::ptr_eq(document, current))
            && match (&self.segment, store.text_segment()) {
                (Some(old), Some(current)) => Arc::ptr_eq(old, &current),
                (None, None) => true,
                _ => false,
            }
    }
}

impl CapturedChunkRead {
    pub(crate) fn read(mut self) -> (ChunkReadIdentity, Result<DocumentChunkRecord>) {
        let identity = ChunkReadIdentity {
            document: self.document,
            segment: self
                .segment
                .as_ref()
                .map(|(segment, _, _, _)| Arc::clone(segment)),
        };
        let result = (|| {
            if let Some((segment, at, len, checksum)) = self.segment {
                let text = segment.read(at, len, checksum)?;
                ensure!(
                    text.len() <= super::MAX_DOCUMENT_BYTES,
                    "stored chunk exceeds read limit"
                );
                self.record.text = Some(text);
            }
            ensure!(
                self.record.text.as_ref().is_none_or(|text| self
                    .record
                    .end_byte
                    .checked_sub(self.record.start_byte)
                    .is_some_and(|length| text.len() == length as usize)),
                "stored chunk text length disagrees with its byte span"
            );
            Ok(self.record)
        })();
        (identity, result)
    }
}

pub(crate) fn capture_chunk(
    store: &DocumentStore,
    id: &str,
    index: usize,
    revision: u64,
) -> Result<Option<CapturedChunkRead>> {
    if let Some(document) = store.get(id) {
        let Some(chunk) = document.chunks.get(index) else {
            return Ok(None);
        };
        ensure!(
            chunk
                .text
                .as_ref()
                .is_none_or(|text| text.len() <= super::MAX_DOCUMENT_BYTES),
            "stored chunk exceeds read limit"
        );
        let segment = if chunk.text.is_none() {
            chunk
                .text_at
                .map(|(at, len, checksum)| {
                    ensure!(
                        len as usize <= super::MAX_DOCUMENT_BYTES + 12 + 16,
                        "stored chunk exceeds read limit"
                    );
                    Ok((
                        store
                            .text_segment()
                            .context("stored chunk has no text segment")?,
                        at,
                        len,
                        checksum,
                    ))
                })
                .transpose()?
        } else {
            None
        };
        return Ok(Some(CapturedChunkRead {
            record: DocumentChunkRecord {
                revision: revision.to_string(),
                document: info(document),
                index,
                id: chunk.id.clone(),
                start_byte: chunk.start_byte,
                end_byte: chunk.end_byte,
                text: chunk.text.clone(),
            },
            document: segment.as_ref().map(|_| Arc::clone(document)),
            segment,
        }));
    }
    let Some(base) = &store.base else {
        return Ok(None);
    };
    let Some(ordinal) = base.live(id) else {
        return Ok(None);
    };
    Ok(base
        .image
        .document_chunk(ordinal, index, revision)?
        .map(|record| CapturedChunkRead {
            record,
            segment: None,
            document: None,
        }))
}

fn info(document: &StoredDocument) -> DocumentInfo {
    DocumentInfo {
        id: document.id.clone(),
        source_uri: document.source_uri.clone(),
        version: document.version.clone(),
        metadata: document.metadata.clone(),
        chunk_count: document.chunks.len(),
    }
}

pub(crate) fn get_info(store: &DocumentStore, id: &str) -> Option<DocumentInfo> {
    if let Some(document) = store.get(id) {
        return Some(info(document));
    }
    let base = store.base.as_ref()?;
    base.live(id).map(|ordinal| {
        base.image
            .document_info(ordinal)
            .expect("decoding checkpoint document metadata")
    })
}

pub(crate) fn list_info(
    store: &DocumentStore,
    from: Option<&str>,
    limit: usize,
) -> Vec<DocumentInfo> {
    use std::cmp::Ordering;
    use std::ops::Bound::{Included, Unbounded};
    let start = from.map_or(Unbounded, Included);
    let mut live = store
        .documents
        .range::<str, _>((start, Unbounded))
        .peekable();
    let Some(base) = &store.base else {
        return live
            .take(limit)
            .map(|(_, document)| info(document))
            .collect();
    };
    let mut checkpoint = base.live_ordinals_from(from).peekable();
    let mut result = Vec::new();
    while result.len() < limit {
        let take_live = match (checkpoint.peek(), live.peek()) {
            (None, None) => break,
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (Some(&ordinal), Some((id, _))) => match base.image.id(ordinal).cmp(id.as_str()) {
                Ordering::Less => false,
                Ordering::Greater => true,
                Ordering::Equal => {
                    panic!("checkpoint and live store contain the same active document")
                }
            },
        };
        if take_live {
            result.push(info(live.next().expect("peeked document").1));
        } else {
            result.push(
                base.image
                    .document_info(checkpoint.next().expect("peeked checkpoint ordinal"))
                    .expect("decoding checkpoint document metadata"),
            );
        }
    }
    result
}

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

fn record(document: &StoredDocument, segment: Option<&TextSegment>) -> DocumentRecord {
    DocumentRecord {
        id: document.id.clone(),
        source_uri: document.source_uri.clone(),
        version: document.version.clone(),
        metadata: document.metadata.clone(),
        text: document
            .chunks
            .first()
            .and_then(|chunk| text_of(chunk, segment)),
        chunk_count: document.chunks.len(),
    }
}

pub(crate) fn get_record(store: &DocumentStore, id: &str) -> Option<DocumentRecord> {
    store.get(id).map(|document| {
        let segment = store.text_segment();
        record(document, segment.as_deref())
    })
}

/// Up to `limit` documents with ids at or after `from`, in id order.
pub(crate) fn list_records(
    store: &DocumentStore,
    from: Option<&str>,
    limit: usize,
) -> Vec<DocumentRecord> {
    use std::ops::Bound::{Included, Unbounded};
    let start = from.map_or(Unbounded, Included);
    let segment = store.text_segment();
    store
        .documents
        .range::<str, _>((start, Unbounded))
        .take(limit)
        .map(|(_, document)| record(document, segment.as_deref()))
        .collect()
}

pub(crate) fn count(store: &DocumentStore) -> usize {
    store.documents.len() + store.base.as_ref().map_or(0, |base| base.live_documents())
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
