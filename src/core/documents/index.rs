// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::dense::{DenseIndex, MAX_ROWS, MIN_ROWS};
use super::postings::PostingLists;
use super::text::TextSegment;
use super::{SearchOptions, StoredChunk, StoredDocument};

#[derive(Default)]
pub(crate) struct DocumentStore {
    pub(super) documents: BTreeMap<String, Arc<StoredDocument>>,
    pub(super) postings: [PostingLists; 2],
    pub(super) slots: BTreeMap<String, Vec<usize>>,
    pub(super) slot_count: usize,
    pub(super) chunk_count: usize,
    pub(super) word_count: usize,
    pub(super) generation: u64,
    pub(super) dense_index: Option<Arc<DenseIndex>>,
    pub(super) embedded_documents: BTreeSet<String>,
    embedded_count: usize,
    owners: Vec<Option<(Arc<StoredDocument>, usize)>>,
    /// Word count of the chunk in each slot, so lexical scoring never touches the document.
    word_counts: Vec<u32>,
    free: BTreeSet<usize>,
    sources: BTreeMap<String, BTreeSet<String>>,
    metadata: BTreeMap<(String, String), BTreeSet<String>>,
    /// Chunk text lives here, not in the documents, once attached.
    text: Option<Arc<TextSegment>>,
}

/// A chunk's word count is bounded by the 8 MiB document limit, so it fits a `u32`.
pub(super) fn slot_words(count: usize) -> u32 {
    u32::try_from(count).expect("chunk word count over u32")
}

fn scalar_key(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => Some("n".into()),
        serde_json::Value::Bool(v) => Some(format!("b{v}")),
        serde_json::Value::String(v) => Some(format!("s{v}")),
        serde_json::Value::Number(v) if v.is_i64() || v.is_u64() => Some(format!("i{v}")),
        _ => None,
    }
}

/// A stored document serialized exactly as the derived form of [`StoredDocument`] with every
/// chunk's text inline, resolving text moved to a segment; the journal and compaction frames
/// carry documents this way, so a restart sees the same bytes as a fresh commit.
pub(crate) struct DocumentWithText<'a> {
    pub(crate) document: &'a StoredDocument,
    pub(crate) segment: Option<&'a TextSegment>,
}

impl serde::Serialize for DocumentWithText<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let d = self.document;
        let mut s = serializer.serialize_struct("StoredDocument", 5)?;
        s.serialize_field("id", &d.id)?;
        s.serialize_field("source_uri", &d.source_uri)?;
        s.serialize_field("version", &d.version)?;
        s.serialize_field("metadata", &d.metadata)?;
        let chunks: Vec<ChunkWithText<'_>> = d
            .chunks
            .iter()
            .map(|chunk| ChunkWithText {
                chunk,
                segment: self.segment,
            })
            .collect();
        s.serialize_field("chunks", &chunks)?;
        s.end()
    }
}

struct ChunkWithText<'a> {
    chunk: &'a StoredChunk,
    segment: Option<&'a TextSegment>,
}

impl serde::Serialize for ChunkWithText<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let c = self.chunk;
        let mut s = serializer.serialize_struct("StoredChunk", 9)?;
        s.serialize_field("id", &c.id)?;
        s.serialize_field("text", &text_of(c, self.segment))?;
        s.serialize_field("start_byte", &c.start_byte)?;
        s.serialize_field("end_byte", &c.end_byte)?;
        s.serialize_field("vector", &c.vector)?;
        s.serialize_field("embedding", &c.embedding)?;
        s.serialize_field("terms", &c.terms)?;
        s.serialize_field("word_count", &c.word_count)?;
        s.serialize_field("stemmed", &c.stemmed)?;
        s.end()
    }
}

/// A chunk's text from its inline copy or the segment holding it.
pub(super) fn text_of(chunk: &StoredChunk, segment: Option<&TextSegment>) -> Option<String> {
    match (&chunk.text, chunk.text_at) {
        (Some(text), _) => Some(text.clone()),
        (None, Some((at, len))) => Some(
            segment
                .expect("chunk text moved to a segment the store no longer has")
                .read(at, len)
                .expect("chunk text read"),
        ),
        (None, None) => None,
    }
}

fn remove_secondary<K: Ord>(map: &mut BTreeMap<K, BTreeSet<String>>, key: &K, id: &str) {
    let posting = map.get_mut(key).expect("missing document filter posting");
    assert!(posting.remove(id), "missing document from filter posting");
    if posting.is_empty() {
        map.remove(key);
    }
}

fn replace_postings(
    postings: &mut [PostingLists; 2],
    old: &StoredChunk,
    new: &StoredChunk,
    slot: usize,
) {
    for term in old.terms.keys() {
        if let Some(&frequency) = new.terms.get(term).filter(|_| old.stemmed == new.stemmed) {
            postings[usize::from(old.stemmed)].set_frequency(term, slot, frequency);
        } else {
            postings[usize::from(old.stemmed)].remove(term, slot);
        }
    }
    for (term, &frequency) in &new.terms {
        if old.stemmed != new.stemmed || !old.terms.contains_key(term) {
            postings[usize::from(new.stemmed)].insert(term, slot, frequency);
        }
    }
}

impl DocumentStore {
    pub(crate) fn embedded_count(&self) -> usize {
        self.embedded_count
    }

    pub(crate) fn dense_index_trained(&self) -> bool {
        self.dense_index
            .as_ref()
            .is_some_and(|index| index.generation == self.generation)
    }

    pub(crate) fn dense_index_recommended(&self) -> bool {
        !self.dense_index_trained() && (MIN_ROWS..=MAX_ROWS).contains(&self.embedded_count)
    }

    fn invalidate_dense_index(&mut self) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("document generation overflow");
        self.dense_index = None;
    }

    #[cfg(test)]
    pub(crate) fn values(&self) -> impl Iterator<Item = &StoredDocument> {
        self.documents.values().map(Arc::as_ref)
    }

    pub(crate) fn shared_values(&self) -> impl Iterator<Item = Arc<StoredDocument>> + '_ {
        self.documents.values().cloned()
    }

    pub(crate) fn contains_key(&self, id: &str) -> bool {
        self.documents.contains_key(id)
    }

    pub(crate) fn get(&self, id: &str) -> Option<&Arc<StoredDocument>> {
        self.documents.get(id)
    }

    pub(super) fn candidate_ids<'a>(
        &'a self,
        options: &'a SearchOptions,
    ) -> Option<Vec<&'a String>> {
        let mut smallest: Option<&BTreeSet<String>> = None;
        if let Some(uri) = &options.source_uri {
            let Some(ids) = self.sources.get(uri) else {
                return Some(Vec::new());
            };
            smallest = Some(ids);
        }
        if let Some(filter) = options
            .filter
            .as_ref()
            .and_then(serde_json::Value::as_object)
        {
            for (key, value) in filter {
                if let Some(value) = scalar_key(value) {
                    let Some(ids) = self.metadata.get(&(key.clone(), value)) else {
                        return Some(Vec::new());
                    };
                    if smallest.is_none_or(|best| ids.len() < best.len()) {
                        smallest = Some(ids);
                    }
                }
            }
        }
        if let Some(ids) = options
            .document_ids
            .as_ref()
            .filter(|ids| smallest.is_none_or(|best| ids.len() < best.len()))
        {
            let ids: BTreeSet<_> = ids.iter().collect();
            return Some(
                ids.into_iter()
                    .filter(|id| self.documents.contains_key(*id))
                    .collect(),
            );
        }
        smallest.map(|ids| ids.iter().collect())
    }

    pub(crate) fn insert(&mut self, id: String, document: impl Into<Arc<StoredDocument>>) {
        self.insert_shared(id, document.into());
    }

    /// Word count of the chunk occupying `slot`.
    pub(super) fn slot_word_count(&self, slot: usize) -> u32 {
        self.word_counts[slot]
    }

    pub(super) fn owned_chunk(&self, slot: usize) -> (Arc<StoredDocument>, usize) {
        let (document, chunk) = self.owners[slot]
            .as_ref()
            .expect("unoccupied document slot");
        (Arc::clone(document), *chunk)
    }

    /// Keeps chunk text in the segment at `path` (created or truncated now) instead of in
    /// memory; the caller replays the journal afterwards, which refills it.
    pub(crate) fn attach_text_segment(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        assert!(
            self.documents.is_empty(),
            "text segment attached to a populated store"
        );
        self.text = Some(Arc::new(TextSegment::create(path)?));
        Ok(())
    }

    /// The text segment, for readers that outlive the store lock.
    pub(crate) fn text_segment(&self) -> Option<Arc<TextSegment>> {
        self.text.clone()
    }

    /// Moves inline chunk text into the segment, when one is attached.
    fn stripped(&mut self, document: Arc<StoredDocument>) -> Arc<StoredDocument> {
        let Some(segment) = self.text.as_deref() else {
            return document;
        };
        if document.chunks.iter().all(|c| c.text.is_none()) {
            return document;
        }
        let mut owned = StoredDocument::clone(&document);
        for chunk in &mut owned.chunks {
            if let Some(text) = chunk.text.take() {
                chunk.text_at = Some(segment.append(&text).expect("text segment append"));
            }
        }
        Arc::new(owned)
    }

    fn insert_shared(&mut self, id: String, document: Arc<StoredDocument>) {
        assert_eq!(id, document.id, "document key differs from document ID");
        let document = self.stripped(document);
        if let Some(previous) = self
            .documents
            .get(&id)
            .filter(|old| old.chunks.len() == document.chunks.len())
            .cloned()
        {
            self.replace_shared(id, previous, document);
            return;
        }
        self.remove(&id);
        let mut slots = Vec::with_capacity(document.chunks.len());
        let mut embedded = false;
        for (chunk_index, chunk) in document.chunks.iter().enumerate() {
            if chunk.embedding.is_some() {
                self.embedded_count = self
                    .embedded_count
                    .checked_add(1)
                    .expect("document embedded count overflow");
                embedded = true;
            }
            let slot = self.free.pop_first().unwrap_or_else(|| {
                let slot = self.slot_count;
                self.slot_count = self
                    .slot_count
                    .checked_add(1)
                    .expect("document slot overflow");
                self.owners.push(None);
                self.word_counts.push(0);
                slot
            });
            assert!(
                self.owners[slot]
                    .replace((Arc::clone(&document), chunk_index))
                    .is_none(),
                "document slot already owned"
            );
            self.word_counts[slot] = slot_words(chunk.word_count);
            self.chunk_count = self
                .chunk_count
                .checked_add(1)
                .expect("document chunk count overflow");
            self.word_count = self
                .word_count
                .checked_add(chunk.word_count)
                .expect("document word count overflow");
            for (term, &frequency) in &chunk.terms {
                self.postings[usize::from(chunk.stemmed)].insert(term, slot, frequency);
            }
            slots.push(slot);
        }
        self.slots.insert(id.clone(), slots);
        if embedded {
            assert!(
                self.embedded_documents.insert(id.clone()),
                "duplicate embedded document"
            );
        }
        self.insert_filters(&document);
        self.invalidate_dense_index();
        self.documents.insert(id, document);
    }

    fn insert_filters(&mut self, document: &StoredDocument) {
        if let Some(uri) = &document.source_uri {
            self.sources
                .entry(uri.clone())
                .or_default()
                .insert(document.id.clone());
        }
        if let Some(metadata) = document.metadata.as_object() {
            for (key, value) in metadata {
                if let Some(value) = scalar_key(value) {
                    self.metadata
                        .entry((key.clone(), value))
                        .or_default()
                        .insert(document.id.clone());
                }
            }
        }
    }

    fn remove_filters(&mut self, document: &StoredDocument) {
        if let Some(uri) = &document.source_uri {
            remove_secondary(&mut self.sources, uri, &document.id);
        }
        if let Some(metadata) = document.metadata.as_object() {
            for (key, value) in metadata {
                if let Some(value) = scalar_key(value) {
                    remove_secondary(&mut self.metadata, &(key.clone(), value), &document.id);
                }
            }
        }
    }

    fn replace_shared(
        &mut self,
        id: String,
        previous: Arc<StoredDocument>,
        document: Arc<StoredDocument>,
    ) {
        self.remove_filters(&previous);
        self.insert_filters(&document);
        let mut old_embedded = false;
        let mut new_embedded = false;
        let slots = self.slots.get(&id).expect("document missing slot map");
        assert_eq!(
            slots.len(),
            document.chunks.len(),
            "document slot count mismatch"
        );
        for (index, ((old, new), &slot)) in previous
            .chunks
            .iter()
            .zip(&document.chunks)
            .zip(slots)
            .enumerate()
        {
            let owner = self.owners[slot]
                .as_mut()
                .expect("document slot has no owner");
            assert!(
                Arc::ptr_eq(&owner.0, &previous),
                "document slot owner mismatch"
            );
            assert_eq!(owner.1, index, "document slot chunk mismatch");
            *owner = (Arc::clone(&document), index);
            self.word_counts[slot] = slot_words(new.word_count);
            self.word_count = self
                .word_count
                .checked_sub(old.word_count)
                .and_then(|count| count.checked_add(new.word_count))
                .expect("document word count overflow");
            self.embedded_count = self
                .embedded_count
                .checked_sub(usize::from(old.embedding.is_some()))
                .and_then(|count| count.checked_add(usize::from(new.embedding.is_some())))
                .expect("document embedded count overflow");
            old_embedded |= old.embedding.is_some();
            new_embedded |= new.embedding.is_some();
            replace_postings(&mut self.postings, old, new, slot);
        }
        if old_embedded && !new_embedded {
            assert!(
                self.embedded_documents.remove(&id),
                "missing embedded document"
            );
        } else if !old_embedded && new_embedded {
            assert!(
                self.embedded_documents.insert(id.clone()),
                "duplicate embedded document"
            );
        }
        self.invalidate_dense_index();
        self.documents.insert(id, document);
    }

    pub(crate) fn remove(&mut self, id: &str) -> Option<Arc<StoredDocument>> {
        let document = self.documents.remove(id)?;
        self.invalidate_dense_index();
        if document
            .chunks
            .iter()
            .any(|chunk| chunk.embedding.is_some())
        {
            assert!(
                self.embedded_documents.remove(id),
                "missing embedded document"
            );
        }
        self.remove_filters(&document);
        let slots = self.slots.remove(id).expect("document missing slot map");
        assert_eq!(
            slots.len(),
            document.chunks.len(),
            "document slot count mismatch"
        );
        for (index, (chunk, slot)) in document.chunks.iter().zip(slots).enumerate() {
            if chunk.embedding.is_some() {
                self.embedded_count = self
                    .embedded_count
                    .checked_sub(1)
                    .expect("document embedded count underflow");
            }
            let owner = self.owners[slot]
                .take()
                .expect("document slot has no owner");
            assert_eq!(owner.0.id, id, "document slot owner mismatch");
            assert_eq!(owner.1, index, "document slot chunk mismatch");
            self.chunk_count = self
                .chunk_count
                .checked_sub(1)
                .expect("document chunk count underflow");
            self.word_count = self
                .word_count
                .checked_sub(chunk.word_count)
                .expect("document word count underflow");
            let postings = &mut self.postings[usize::from(chunk.stemmed)];
            for term in chunk.terms.keys() {
                postings.remove(term, slot);
            }
            assert!(self.free.insert(slot), "document slot already free");
        }
        while self.slot_count > 0 && self.free.remove(&(self.slot_count - 1)) {
            self.slot_count -= 1;
        }
        self.owners.truncate(self.slot_count);
        self.word_counts.truncate(self.slot_count);
        if self.slot_count > 1024 && self.free.len() > self.slot_count / 2 {
            let generation = self.generation;
            let documents = std::mem::take(&mut self.documents);
            *self = Self::default();
            self.generation = generation;
            for (id, document) in documents {
                self.insert_shared(id, document);
            }
        }
        Some(document)
    }
}
