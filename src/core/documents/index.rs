// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::dense::{DenseIndex, MAX_ROWS, MIN_ROWS};
use super::postings::PostingLists;
use super::text::TextSegment;
use super::{SearchOptions, StoredChunk, StoredDocument, VectorState};

/// Tells store instances apart for caches keyed by `generation`, which counts per instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StoreIdentity(u64);

impl Default for StoreIdentity {
    fn default() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Default)]
pub(crate) struct DocumentStore {
    /// With `generation`, names one state of the lexical index (every mutation advances
    /// `generation`).
    pub(super) identity: StoreIdentity,
    pub(super) documents: BTreeMap<String, Arc<StoredDocument>>,
    pub(super) postings: [PostingLists; 2],
    pub(super) slots: BTreeMap<String, Vec<usize>>,
    pub(super) slot_count: usize,
    pub(super) chunk_count: usize,
    pub(super) word_count: usize,
    pub(super) nonpositional_chunks: usize,
    pub(super) multichunk_documents: BTreeSet<String>,
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
    /// Chunks memorized with deferred encoding whose vectors are not installed yet.
    pending_chunks: usize,
    /// Chunks memorized lexical-only (never in the vector index).
    lexical_chunks: usize,
    /// Serializes the store's lazy encoders, so a backlog is encoded once, not per caller.
    encode_gate: Arc<parking_lot::Mutex<()>>,
}

/// A random starting value for a process-local mutation counter, below 2^53 so increments
/// cannot overflow. A counter that starts here still advances by one per mutation for the
/// caches and handles comparing two readings, but one reading says nothing about how many
/// mutations (inserts and deletes since open) came before it.
pub(crate) fn opaque_counter_start() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    // 53 random bits: values stay exact as JSON numbers read by JavaScript.
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
        >> 11
}

/// Where [`DocumentStore::rebuild_canonical`] stages the segment that replaces `path`. A
/// leftover from an interrupted purge holds live text in plaintext; open scrubs it.
pub(crate) fn canonical_staging_path(path: &std::path::Path) -> std::path::PathBuf {
    path.with_extension("text.canonical")
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
        if !c.vector_state.is_encoded() {
            s.serialize_field("vector_state", &c.vector_state)?;
        }
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
    for (term, &old_frequency) in &old.terms {
        if let Some(&frequency) = new.terms.get(term).filter(|_| old.stemmed == new.stemmed) {
            // Unchanged terms (a deferred encoding landing, a re-memorized document) leave
            // the list alone; rewriting it would decode and encode the whole list.
            if new.word_count < old.word_count {
                postings[usize::from(old.stemmed)].invalidate_length(term, slot);
            }
            if frequency != old_frequency {
                postings[usize::from(old.stemmed)].set_frequency(term, slot, frequency);
            }
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

    /// Chunks whose deferred encoding has not been installed yet.
    pub(crate) fn pending_vector_count(&self) -> usize {
        self.pending_chunks
    }

    /// Chunks memorized lexical-only.
    pub(crate) fn lexical_vector_count(&self) -> usize {
        self.lexical_chunks
    }

    /// The lock lazy encoders of this store take for the duration of a backlog.
    pub(crate) fn encode_gate(&self) -> Arc<parking_lot::Mutex<()>> {
        Arc::clone(&self.encode_gate)
    }

    fn count_vector_state(&mut self, state: VectorState, added: bool) {
        let counter = match state {
            VectorState::Encoded => return,
            VectorState::Pending => &mut self.pending_chunks,
            VectorState::Lexical => &mut self.lexical_chunks,
        };
        let next = if added {
            counter.checked_add(1)
        } else {
            counter.checked_sub(1)
        };
        *counter = next.expect("document vector state count out of range");
    }

    /// Applies the net change of pending and lexical chunk counts of a replacement.
    fn shift_vector_states(&mut self, pending: isize, lexical: isize) {
        self.pending_chunks = self
            .pending_chunks
            .checked_add_signed(pending)
            .expect("document vector state count out of range");
        self.lexical_chunks = self
            .lexical_chunks
            .checked_add_signed(lexical)
            .expect("document vector state count out of range");
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

    /// A chunk's text from its inline copy or this store's text segment.
    pub(crate) fn chunk_text(&self, chunk: &StoredChunk) -> Option<String> {
        text_of(chunk, self.text.as_deref())
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
        let document = document.into();
        assert_eq!(id, document.id, "document key differs from document ID");
        self.insert_batch(vec![document]);
    }

    /// Inserts `documents` in order, reaching the state the same inserts one at a time
    /// reach: slots are assigned in document and chunk order, chunk text goes to the segment
    /// in that order, and a document that replaces a stored one (or repeats an id earlier in
    /// the batch) takes the replacement path at its turn. The maximal runs of new, distinct
    /// documents between those are indexed in bulk (`postings::index_chunks`).
    pub(crate) fn insert_batch(&mut self, documents: Vec<Arc<StoredDocument>>) {
        let documents: Vec<Arc<StoredDocument>> = documents
            .into_iter()
            .map(|document| self.stripped(document))
            .collect();
        let mut group: Vec<Arc<StoredDocument>> = Vec::with_capacity(documents.len());
        let mut grouped: std::collections::HashSet<String> = std::collections::HashSet::new();
        for document in documents {
            if !grouped.insert(document.id.clone()) {
                self.install_new(std::mem::take(&mut group));
                grouped.clear();
                grouped.insert(document.id.clone());
            }
            if self.documents.contains_key(&document.id) {
                self.install_new(std::mem::take(&mut group));
                self.replace(document);
            } else {
                group.push(document);
            }
        }
        self.install_new(group);
    }

    /// Word count of the chunk occupying `slot`.
    pub(super) fn slot_word_count(&self, slot: usize) -> u32 {
        self.word_counts[slot]
    }

    /// The `(document id, chunk index)` ranking ties are broken by, for the chunk in `slot`.
    pub(super) fn slot_order(&self, slot: usize) -> (&str, usize) {
        let (document, chunk) = self.owners[slot]
            .as_ref()
            .expect("unoccupied document slot");
        (&document.id, *chunk)
    }

    pub(super) fn borrowed_chunk(&self, slot: usize) -> (&StoredDocument, usize) {
        let (document, index) = self.owners[slot]
            .as_ref()
            .expect("unoccupied document slot");
        (document.as_ref(), *index)
    }

    pub(super) fn owned_chunk(&self, slot: usize) -> (Arc<StoredDocument>, usize) {
        let (document, chunk) = self.owners[slot]
            .as_ref()
            .expect("unoccupied document slot");
        (Arc::clone(document), *chunk)
    }

    /// Keeps chunk text in the segment at `path` (created or truncated now) instead of in
    /// memory; the caller replays the journal afterwards, which refills it.
    pub(crate) fn attach_text_segment(
        &mut self,
        path: &std::path::Path,
        #[cfg(feature = "security")] cipher: Option<Arc<crate::core::security::EncryptionManager>>,
    ) -> anyhow::Result<()> {
        assert!(
            self.documents.is_empty(),
            "text segment attached to a populated store"
        );
        self.text = Some(Arc::new(TextSegment::create(
            path,
            #[cfg(feature = "security")]
            cipher,
        )?));
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

    /// Installs `document` (text already stripped) over the stored document with its id: in
    /// place when the chunk count matches, otherwise by removal and a fresh insert.
    fn replace(&mut self, document: Arc<StoredDocument>) {
        let id = document.id.clone();
        if let Some(previous) = self
            .documents
            .get(&id)
            .filter(|old| old.chunks.len() == document.chunks.len())
            .cloned()
        {
            self.replace_shared(id, previous, document);
            return;
        }
        if let Some(old) = self.remove(&id) {
            self.scrub_replaced(&old, &document);
        }
        self.install_new(vec![document]);
    }

    /// Indexes documents absent from the store, with distinct ids and text already stripped,
    /// in order: slots are assigned chunk by chunk exactly as single inserts assign them (the
    /// smallest free slots, then new ones), so the group's chunks are in ascending slot order
    /// and their postings build in one bulk pass.
    fn install_new(&mut self, group: Vec<Arc<StoredDocument>>) {
        if group.is_empty() {
            return;
        }
        let total = group.iter().map(|document| document.chunks.len()).sum();
        let mut indexed: Vec<(usize, &StoredChunk)> = Vec::with_capacity(total);
        for document in &group {
            debug_assert!(
                !self.documents.contains_key(&document.id),
                "bulk insert over a stored document"
            );
            let mut slots = Vec::with_capacity(document.chunks.len());
            let mut embedded = false;
            for (chunk_index, chunk) in document.chunks.iter().enumerate() {
                self.nonpositional_chunks += usize::from(chunk.terms.positions.is_none());
                self.count_vector_state(chunk.vector_state, true);
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
                        .replace((Arc::clone(document), chunk_index))
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
                indexed.push((slot, chunk));
                slots.push(slot);
            }
            if document.chunks.len() > 1 {
                self.multichunk_documents.insert(document.id.clone());
            }
            self.slots.insert(document.id.clone(), slots);
            if embedded {
                assert!(
                    self.embedded_documents.insert(document.id.clone()),
                    "duplicate embedded document"
                );
            }
            self.insert_filters(document);
        }
        super::postings::index_chunks(&mut self.postings, &indexed, &self.word_counts);
        // One generation step per document, as single inserts take.
        self.generation = self
            .generation
            .checked_add(group.len() as u64)
            .expect("document generation overflow");
        self.dense_index = None;
        for document in group {
            self.documents.insert(document.id.clone(), document);
        }
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
        let mut pending_delta: isize = 0;
        let mut lexical_delta: isize = 0;
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
            self.nonpositional_chunks = self
                .nonpositional_chunks
                .checked_sub(usize::from(old.terms.positions.is_none()))
                .and_then(|n| n.checked_add(usize::from(new.terms.positions.is_none())))
                .expect("non-positional chunk count overflow");
            for (state, delta) in [(old.vector_state, -1), (new.vector_state, 1)] {
                match state {
                    VectorState::Encoded => {}
                    VectorState::Pending => pending_delta += delta,
                    VectorState::Lexical => lexical_delta += delta,
                }
            }
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
        self.shift_vector_states(pending_delta, lexical_delta);
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
        self.scrub_replaced(&previous, &document);
        self.documents.insert(id, document);
    }

    /// Whether `document` is no longer a live document: absent from the store, or with chunk
    /// text that a delete has zeroed since the caller captured it.
    pub(crate) fn is_deleted(&self, document: &StoredDocument) -> bool {
        !self.documents.contains_key(&document.id)
            || self.text.as_deref().is_some_and(|segment| {
                document
                    .chunks
                    .iter()
                    .filter_map(|c| c.text_at)
                    .any(|(at, len)| segment.is_scrubbed(at, len))
            })
    }

    /// Zeroes the text records of `old` that its replacement `new` no longer uses, so updated
    /// text does not outlive the update in the segment.
    fn scrub_replaced(&self, old: &StoredDocument, new: &StoredDocument) {
        let Some(segment) = self.text.as_deref() else {
            return;
        };
        for span in old.chunks.iter().filter_map(|c| c.text_at) {
            if new.chunks.iter().all(|c| c.text_at != Some(span)) {
                segment.scrub(span.0, span.1).expect("text segment scrub");
            }
        }
    }

    /// Removes a document and zeroes its chunk text in the segment (see
    /// [`TextSegment::scrub`]); the caller syncs the segment once per transaction.
    pub(crate) fn remove_scrubbed(
        &mut self,
        id: &str,
    ) -> anyhow::Result<Option<Arc<StoredDocument>>> {
        let removed = self.remove(id);
        if let (Some(document), Some(segment)) = (&removed, self.text.as_deref()) {
            for (at, len) in document.chunks.iter().filter_map(|c| c.text_at) {
                segment.scrub(at, len)?;
            }
        }
        Ok(removed)
    }

    /// An empty store whose generation starts at [`opaque_counter_start`], so the generation a
    /// status reports does not count the mutations before it.
    pub(crate) fn opened() -> Self {
        Self {
            generation: opaque_counter_start(),
            ..Self::default()
        }
    }

    /// Rebuilds the store from its documents in id order over a fresh text segment staged
    /// beside `path` and renamed over it. Slots, postings, filters and the segment bytes become
    /// what a reopen of the compacted journal produces (it replays documents in id order), so
    /// none of them reflects a deleted document or the order documents arrived in. Readers
    /// holding the old segment keep reading the unlinked file; deleted records in it are
    /// already zero. No-op without a segment.
    pub(crate) fn rebuild_canonical(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let Some(old) = self.text.clone() else {
            return Ok(());
        };
        let staging = canonical_staging_path(path);
        let segment = Arc::new(old.recreate(&staging)?);
        let documents = self
            .documents
            .values()
            .map(|document| {
                let mut owned = StoredDocument::clone(document);
                for chunk in &mut owned.chunks {
                    if let Some((at, len)) = chunk.text_at.take() {
                        chunk.text = Some(old.read(at, len)?);
                    }
                }
                Ok(Arc::new(owned))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let generation = self.generation;
        let encode_gate = Arc::clone(&self.encode_gate);
        *self = Self::default();
        self.generation = generation;
        self.encode_gate = encode_gate;
        self.text = Some(segment);
        self.insert_batch(documents);
        std::fs::rename(&staging, path)?;
        if let Some(parent) = path.parent() {
            crate::core::durable_file::sync_directory(parent)?;
        }
        Ok(())
    }

    pub(crate) fn remove(&mut self, id: &str) -> Option<Arc<StoredDocument>> {
        let document = self.documents.remove(id)?;
        self.multichunk_documents.remove(id);
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
            self.nonpositional_chunks = self
                .nonpositional_chunks
                .checked_sub(usize::from(chunk.terms.positions.is_none()))
                .expect("non-positional chunk count underflow");
            self.count_vector_state(chunk.vector_state, false);
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
            let text = self.text.take();
            let encode_gate = Arc::clone(&self.encode_gate);
            let documents = std::mem::take(&mut self.documents);
            *self = Self::default();
            self.generation = generation;
            self.text = text;
            self.encode_gate = encode_gate;
            self.insert_batch(documents.into_values().collect());
        }
        Some(document)
    }
}
