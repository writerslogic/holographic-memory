// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::dense::{DenseChange, DenseIndex, MAX_ROWS, MIN_ROWS};
use super::positions::Positions;
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
    pub(super) dense_changes: BTreeMap<String, DenseChange>,
    pub(super) embedded_documents: BTreeSet<String>,
    embedded_count: usize,
    owners: Vec<Option<(Arc<StoredDocument>, usize)>>,
    /// Word count of the chunk in each slot, so lexical scoring never touches the document.
    word_counts: Vec<u32>,
    /// Shared immutable position records by slot; empty while no positional chunks are live.
    position_sources: Vec<Option<Arc<Positions>>>,
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
    encode_gate: Arc<crate::core::sync::Mutex<()>>,
    /// The checkpoint this store is served from, whose live documents count as the store's
    /// (`overlay`): the store holds only documents written since.
    pub(super) base: Option<super::overlay::Base>,
}

#[cfg(test)]
impl DocumentStore {
    pub(crate) fn memory_census(&self) -> super::census::MemoryCensus {
        use std::mem::{size_of, size_of_val};
        let mut census = super::census::MemoryCensus::default();
        for postings in &self.postings {
            postings.record_memory(&mut census);
        }
        census.count("documents", self.documents.len());
        census.count("chunks", self.chunk_count);
        census.bytes(
            "slots.owners.capacity",
            self.owners.capacity() * size_of::<Option<(Arc<StoredDocument>, usize)>>(),
        );
        census.bytes(
            "slots.word_counts.capacity",
            self.word_counts.capacity() * size_of::<u32>(),
        );
        census.bytes(
            "slots.position_sources.capacity",
            self.position_sources.capacity() * size_of::<Option<Arc<Positions>>>(),
        );
        for source in self.position_sources.iter().flatten() {
            source.record_memory(&mut census);
        }
        census.bytes(
            "documents.map.payload",
            self.documents
                .iter()
                .map(|(id, document)| size_of_val(id) + size_of_val(document) + id.capacity())
                .sum(),
        );
        census.bytes(
            "slots.map.payload",
            self.slots
                .iter()
                .map(|(id, slots)| {
                    size_of_val(id)
                        + size_of_val(slots)
                        + id.capacity()
                        + slots.capacity() * size_of::<usize>()
                })
                .sum(),
        );
        census.bytes("slots.free.payload", self.free.len() * size_of::<usize>());
        for document in self.documents.values() {
            census.bytes(
                "documents.arc_records",
                size_of::<StoredDocument>() + 2 * size_of::<usize>(),
            );
            census.bytes(
                "documents.strings",
                document.id.capacity()
                    + document.version.capacity()
                    + document.source_uri.as_ref().map_or(0, String::capacity),
            );
            census.count(
                "documents.with_metadata",
                usize::from(!document.metadata.is_null()),
            );
            census.bytes(
                "chunks.capacity",
                document.chunks.capacity() * size_of::<StoredChunk>(),
            );
            for chunk in &document.chunks {
                census.bytes(
                    "chunks.strings",
                    chunk.id.capacity() + chunk.text.as_ref().map_or(0, String::capacity),
                );
                census.bytes(
                    "chunks.vector_indices.capacity",
                    chunk.vector.indices.capacity() * size_of::<u32>(),
                );
                census.count(
                    "chunks.with_embedding",
                    usize::from(chunk.embedding.is_some()),
                );
                chunk.terms.record_memory(&mut census);
            }
        }
        census.count("unmeasured.metadata_entries", self.metadata.len());
        census.count("unmeasured.source_entries", self.sources.len());
        census.count(
            "unmeasured.multichunk_documents",
            self.multichunk_documents.len(),
        );
        census.count(
            "unmeasured.embedded_documents",
            self.embedded_documents.len(),
        );
        census.count(
            "unmeasured.dense_index",
            usize::from(self.dense_index.is_some()),
        );
        census
    }
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
/// [`DocumentStore::overlay`]: a checkpoint, its dead document ids, the documents written since.
pub(crate) type Overlay = (
    Arc<super::ServingImage>,
    Vec<String>,
    Vec<Arc<StoredDocument>>,
);

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

/// Whether the segment record holding a chunk's text was scrubbed (its document replaced or
/// deleted since the chunk was captured).
pub(super) fn text_scrubbed(chunk: &StoredChunk, segment: Option<&TextSegment>) -> bool {
    chunk.text.is_none()
        && chunk
            .text_at
            .zip(segment)
            .is_some_and(|((at, len), segment)| segment.is_scrubbed(at, len))
}

fn remove_secondary<K: Ord>(map: &mut BTreeMap<K, BTreeSet<String>>, key: &K, id: &str) {
    let posting = map.get_mut(key).expect("missing document filter posting");
    assert!(posting.remove(id), "missing document from filter posting");
    if posting.is_empty() {
        map.remove(key);
    }
}

fn set_position_source(
    sources: &mut Vec<Option<Arc<Positions>>>,
    slot_count: usize,
    slot: usize,
    positions: &Option<Arc<Positions>>,
) {
    if positions.is_none() && sources.is_empty() {
        return;
    }
    sources.resize_with(slot_count, || None);
    sources[slot] = positions.clone();
}

/// Moves `slot`'s postings from `old` to `new`; `words` already holds the new chunk's length.
fn replace_postings(
    postings: &mut [PostingLists; 2],
    old: &StoredChunk,
    new: &StoredChunk,
    slot: usize,
    words: &[u32],
) {
    for term in old.terms.keys() {
        if old.stemmed != new.stemmed || !new.terms.contains_key(term) {
            postings[usize::from(old.stemmed)].remove(term, slot, words);
        }
    }
    let mut old_terms = (&old.terms).into_iter().enumerate().peekable();
    for (index, (term, &frequency)) in (&new.terms).into_iter().enumerate() {
        while old_terms.peek().is_some_and(|(_, (old, _))| *old < term) {
            old_terms.next();
        }
        let ordinal = if new.terms.positions.is_some() {
            index as u32 + 1
        } else {
            0
        };
        let list = &mut postings[usize::from(new.stemmed)];
        let previous = old_terms
            .peek()
            .filter(|(_, (old_term, _))| old.stemmed == new.stemmed && *old_term == term);
        if let Some(&(old_index, (_, &old_frequency))) = previous {
            if new.word_count < old.word_count {
                list.invalidate_length(term, slot);
            }
            let old_ordinal = if old.terms.positions.is_some() {
                old_index as u32 + 1
            } else {
                0
            };
            if frequency != old_frequency || ordinal != old_ordinal {
                list.set_entry(term, slot, frequency, Some(ordinal), words);
            }
        } else {
            list.insert_with_ordinal(term, slot, frequency, ordinal, words);
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
    pub(crate) fn encode_gate(&self) -> Arc<crate::core::sync::Mutex<()>> {
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
        self.dense_index.is_none() && (MIN_ROWS..=MAX_ROWS).contains(&self.embedded_count)
    }

    pub(crate) fn discard_stale_dense_index(&mut self) {
        if !self.dense_index_trained() {
            self.dense_index = None;
            self.dense_changes.clear();
        }
    }

    fn advance_generation(&mut self) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("document generation overflow");
    }

    fn track_dense_change(&mut self, id: &str, chunks: usize, previous: Option<&StoredDocument>) {
        use super::dense::{MAX_DELTA_CHUNKS, MAX_DELTA_DOCUMENTS, MAX_DELTA_ID_BYTES};
        if self.dense_index.is_none() {
            return;
        }
        let documents =
            self.dense_changes.len() + usize::from(!self.dense_changes.contains_key(id));
        let rows = self
            .dense_changes
            .values()
            .map(|change| change.chunks)
            .sum::<usize>()
            - self.dense_changes.get(id).map_or(0, |change| change.chunks);
        let bytes = self.dense_changes.keys().map(String::len).sum::<usize>()
            + if self.dense_changes.contains_key(id) {
                0
            } else {
                id.len()
            };
        if documents > MAX_DELTA_DOCUMENTS
            || chunks > MAX_DELTA_CHUNKS.saturating_sub(rows)
            || bytes > MAX_DELTA_ID_BYTES
        {
            self.dense_index = None;
            self.dense_changes.clear();
        } else if let Some(change) = self.dense_changes.get_mut(id) {
            change.chunks = chunks;
        } else {
            // Keep the first replaced owner: every captured index pins its indexed allocations.
            self.dense_changes.insert(
                id.to_owned(),
                DenseChange {
                    chunks,
                    previous_owner: previous
                        .map(|document| document as *const StoredDocument as usize),
                },
            );
        }
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

    /// Whether document `id` is stored, here or live in the checkpoint under the store.
    pub(crate) fn contains_key(&self, id: &str) -> bool {
        self.documents.contains_key(id)
            || self
                .base
                .as_ref()
                .is_some_and(|base| base.live(id).is_some())
    }

    /// A store of no documents of its own over the checkpoint `base`.
    pub(crate) fn over(base: super::overlay::Base) -> Self {
        Self {
            base: Some(base),
            ..Self::opened()
        }
    }

    /// Whether the store is served from a checkpoint (see [`Self::over`]).
    pub(crate) fn has_base(&self) -> bool {
        self.base.is_some()
    }

    /// For a store served from a checkpoint: the checkpoint, the ids of its dead documents, and
    /// the documents written since, in id order.
    pub(crate) fn overlay(&self) -> Option<Overlay> {
        let base = self.base.as_ref()?;
        let dead = base
            .dead_ordinals()
            .map(|ordinal| base.image.id(ordinal).to_owned())
            .collect();
        Some((
            Arc::clone(&base.image),
            dead,
            self.documents.values().cloned().collect(),
        ))
    }

    /// Marks checkpoint document `id` dead if it is live there; true if it was.
    fn kill_base(&mut self, id: &str) -> bool {
        let Some(base) = self.base.as_mut() else {
            return false;
        };
        let Some(ordinal) = base.live(id) else {
            return false;
        };
        base.kill(ordinal);
        self.generation = self
            .generation
            .checked_add(1)
            .expect("document generation overflow");
        true
    }

    pub(crate) fn get(&self, id: &str) -> Option<&Arc<StoredDocument>> {
        self.documents.get(id)
    }

    pub(crate) fn metadata_candidates(
        &self,
        filter: Option<&serde_json::Value>,
    ) -> Option<&BTreeSet<String>> {
        static EMPTY: BTreeSet<String> = BTreeSet::new();
        let mut smallest: Option<&BTreeSet<String>> = None;
        for (key, value) in filter
            .and_then(serde_json::Value::as_object)
            .into_iter()
            .flatten()
        {
            if let Some(value) = scalar_key(value) {
                let Some(ids) = self.metadata.get(&(key.clone(), value)) else {
                    return Some(&EMPTY);
                };
                if smallest.is_none_or(|best| ids.len() < best.len()) {
                    smallest = Some(ids);
                }
            }
        }
        smallest
    }

    pub(super) fn candidate_ids<'a>(
        &'a self,
        options: &'a SearchOptions,
    ) -> Option<Vec<&'a String>> {
        let mut smallest = self.metadata_candidates(options.filter.as_ref());
        if let Some(uri) = &options.source_uri {
            let Some(ids) = self.sources.get(uri) else {
                return Some(Vec::new());
            };
            if smallest.is_none_or(|best| ids.len() < best.len()) {
                smallest = Some(ids);
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
    pub(crate) fn insert_batch(&mut self, mut documents: Vec<Arc<StoredDocument>>) {
        for group in documents.chunks_mut(super::MAX_BATCH_ITEMS) {
            if group
                .iter()
                .all(|d| d.chunks.iter().all(|c| c.terms.is_whole_document()))
            {
                super::packed_terms::StoredTerms::share_documents(group);
            }
        }
        let documents: Vec<Arc<StoredDocument>> = documents
            .into_iter()
            .map(|document| {
                self.kill_base(&document.id);
                self.stripped(document)
            })
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

    /// The store holding `documents` (id order, text already moved to `text` when attached),
    /// each chunk in the slot its list gives, with `postings` already built for exactly those
    /// chunks: the state a checkpoint records, holes and all. Slots no chunk occupies are free.
    pub(super) fn restore(
        documents: Vec<(Arc<StoredDocument>, Vec<usize>)>,
        slot_count: usize,
        postings: [PostingLists; 2],
        text: Option<Arc<TextSegment>>,
    ) -> Self {
        let mut store = Self {
            postings,
            slot_count,
            text,
            owners: vec![None; slot_count],
            word_counts: vec![0; slot_count],
            ..Self::opened()
        };
        for (document, slots) in documents {
            for (index, (chunk, &slot)) in document.chunks.iter().zip(&slots).enumerate() {
                store.nonpositional_chunks += usize::from(chunk.terms.positions.is_none());
                store.count_vector_state(chunk.vector_state, true);
                assert!(
                    store.owners[slot]
                        .replace((Arc::clone(&document), index))
                        .is_none(),
                    "document slot already owned"
                );
                store.word_counts[slot] = slot_words(chunk.word_count);
                set_position_source(
                    &mut store.position_sources,
                    store.slot_count,
                    slot,
                    &chunk.terms.positions,
                );
                store.chunk_count += 1;
                store.word_count += chunk.word_count;
            }
            if document.chunks.len() > 1 {
                store.multichunk_documents.insert(document.id.clone());
            }
            store.insert_filters(&document);
            store.slots.insert(document.id.clone(), slots);
            store.generation += 1;
            store.documents.insert(document.id.clone(), document);
        }
        store.free = (0..slot_count)
            .filter(|&slot| store.owners[slot].is_none())
            .collect();
        store
    }

    /// Word count of the chunk occupying `slot`.
    pub(super) fn slot_word_count(&self, slot: usize) -> u32 {
        self.word_counts[slot]
    }

    #[cfg(test)]
    pub(super) fn slot_positions(&self, slot: usize) -> &Positions {
        self.position_sources[slot]
            .as_deref()
            .expect("positional slot has no source")
    }

    fn trim_position_sources(&mut self) {
        if self.nonpositional_chunks == self.chunk_count {
            self.position_sources = Vec::new();
        } else {
            self.position_sources.truncate(self.slot_count);
        }
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
                set_position_source(
                    &mut self.position_sources,
                    self.slot_count,
                    slot,
                    &chunk.terms.positions,
                );
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
        for document in group {
            self.track_dense_change(&document.id, document.chunks.len(), None);
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
            set_position_source(
                &mut self.position_sources,
                self.slot_count,
                slot,
                &new.terms.positions,
            );
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
            replace_postings(&mut self.postings, old, new, slot, &self.word_counts);
        }
        self.shift_vector_states(pending_delta, lexical_delta);
        self.trim_position_sources();
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
        self.advance_generation();
        self.track_dense_change(&id, document.chunks.len(), Some(&previous));
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
    #[cfg(test)]
    pub(crate) fn remove_scrubbed(
        &mut self,
        id: &str,
    ) -> anyhow::Result<Option<Arc<StoredDocument>>> {
        let removed = self.remove(id);
        self.scrub_removed(removed)
    }

    /// Zeroes the segment text of `removed`, returning it.
    pub(crate) fn scrub_removed(
        &self,
        removed: Option<Arc<StoredDocument>>,
    ) -> anyhow::Result<Option<Arc<StoredDocument>>> {
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
        anyhow::ensure!(
            self.base.is_none(),
            "canonical rebuild of a store still served from its checkpoint"
        );
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
        // Built aside and installed only once the staged segment holds `path`: a failed rename
        // leaves the store reading its old segment, and a retry restages over the staging file.
        let mut rebuilt = Self {
            generation: self.generation,
            encode_gate: Arc::clone(&self.encode_gate),
            text: Some(segment),
            ..Self::default()
        };
        rebuilt.insert_batch(documents);
        std::fs::rename(&staging, path)?;
        *self = rebuilt;
        if let Some(parent) = path.parent() {
            crate::core::durable_file::sync_directory(parent)?;
        }
        Ok(())
    }

    /// Removes document `id`, returning it when it was one of the store's own; a live
    /// checkpoint document is marked dead instead (and `None` returned).
    pub(crate) fn remove(&mut self, id: &str) -> Option<Arc<StoredDocument>> {
        self.remove_with(id, true)
    }

    /// Removes and scrubs (see [`Self::scrub_removed`]) a document
    /// `ServingImage::hydrate_excluding` installed without postings: everything but the posting
    /// lists, which never held it.
    pub(crate) fn remove_unindexed_scrubbed(
        &mut self,
        id: &str,
    ) -> anyhow::Result<Option<Arc<StoredDocument>>> {
        let removed = self.remove_with(id, false);
        self.scrub_removed(removed)
    }

    fn remove_with(&mut self, id: &str, indexed: bool) -> Option<Arc<StoredDocument>> {
        if !self.documents.contains_key(id) {
            self.kill_base(id);
            return None;
        }
        let document = self.documents.remove(id)?;
        self.multichunk_documents.remove(id);
        self.advance_generation();
        self.track_dense_change(id, 0, Some(&document));
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
            if let Some(source) = self.position_sources.get_mut(slot) {
                *source = None;
            }
            self.chunk_count = self
                .chunk_count
                .checked_sub(1)
                .expect("document chunk count underflow");
            self.word_count = self
                .word_count
                .checked_sub(chunk.word_count)
                .expect("document word count underflow");
            if indexed {
                let postings = &mut self.postings[usize::from(chunk.stemmed)];
                for term in chunk.terms.keys() {
                    postings.remove(term, slot, &self.word_counts);
                }
            }
            assert!(self.free.insert(slot), "document slot already free");
        }
        while self.slot_count > 0 && self.free.remove(&(self.slot_count - 1)) {
            self.slot_count -= 1;
        }
        self.owners.truncate(self.slot_count);
        self.word_counts.truncate(self.slot_count);
        self.trim_position_sources();
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

#[cfg(test)]
mod position_source_tests {
    use super::*;
    use crate::core::documents::{self, DocumentInput, EncodeMode, IngestOptions};
    use crate::core::encoding::TextEncoder;

    fn document(
        id: &str,
        text: &str,
        positional: bool,
        unstemmed: bool,
        words: usize,
    ) -> Arc<StoredDocument> {
        let input: DocumentInput = serde_json::from_value(serde_json::json!({
            "id": id, "text": text, "chunkWords": words, "overlapWords": 0, "storeText": false,
        }))
        .unwrap();
        Arc::new(
            documents::prepare_with_options(
                input,
                64,
                None,
                None,
                TextEncoder::default(),
                &IngestOptions {
                    encode: EncodeMode::Lexical,
                    positions: positional,
                    unstemmed,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    }

    fn check(store: &DocumentStore) {
        if store.nonpositional_chunks == store.chunk_count {
            assert_eq!(store.position_sources.capacity(), 0);
            return;
        }
        assert_eq!(store.position_sources.len(), store.slot_count);
        for (owner, source) in store.owners.iter().zip(&store.position_sources) {
            let expected = owner
                .as_ref()
                .and_then(|(document, index)| document.chunks[*index].terms.positions.as_ref());
            match (expected, source) {
                (Some(expected), Some(source)) => assert!(Arc::ptr_eq(expected, source)),
                (None, None) => {}
                _ => panic!("slot position source differs from its owner"),
            }
        }
        for lists in &store.postings {
            lists.assert_valid();
        }
    }

    #[test]
    fn shared_position_sources_follow_replacement_holes_and_snapshot_lifetimes() {
        let mut store = DocumentStore::default();
        store.insert(
            "plain".into(),
            document("plain", "red blue", false, true, 4096),
        );
        check(&store);
        let held = document("live", "red blue red", true, true, 4096);
        let serialized = serde_json::to_vec(&held).unwrap();
        let old_source = Arc::downgrade(held.chunks[0].terms.positions.as_ref().unwrap());
        store.insert("live".into(), Arc::clone(&held));
        check(&store);
        assert_eq!(old_source.strong_count(), 2);
        store.remove("plain");
        check(&store);
        assert!(store.position_sources[0].is_none());
        let captured = documents::capture_search(
            &store,
            vec![documents::prepare_search("red", &SearchOptions::default(), None).unwrap()],
        );
        let captured_result =
            serde_json::to_value(documents::execute_search(&captured, 0)).unwrap();

        store.insert(
            "live".into(),
            document("live", "running foxes red", true, false, 4096),
        );
        check(&store);
        assert_eq!(old_source.strong_count(), 1);
        let changed_source = Arc::downgrade(store.position_sources[1].as_ref().unwrap());
        assert!(!std::ptr::eq(
            store.slot_positions(1),
            held.chunks[0].terms.positions.as_deref().unwrap()
        ));
        store.insert(
            "live".into(),
            document("live", "running foxes red", false, false, 4096),
        );
        check(&store);
        assert!(changed_source.upgrade().is_none());

        store.insert(
            "live".into(),
            document("live", "red blue red blue", true, true, 2),
        );
        check(&store);
        assert_eq!(store.slots["live"], vec![0, 1]);
        let tail_source = Arc::downgrade(store.position_sources[1].as_ref().unwrap());
        store.insert(
            "live".into(),
            document("live", "blue red", true, true, 4096),
        );
        check(&store);
        assert_eq!(store.slot_count, 1);
        assert!(tail_source.upgrade().is_none());
        store.insert(
            "reused".into(),
            document("reused", "red blue", true, false, 4096),
        );
        check(&store);
        store.remove("live");
        check(&store);
        assert!(store.position_sources[0].is_none());
        let removed_source = Arc::downgrade(store.position_sources[1].as_ref().unwrap());
        store.remove("reused");
        check(&store);
        assert!(removed_source.upgrade().is_none());
        assert_eq!(serde_json::to_vec(&held).unwrap(), serialized);
        assert_eq!(
            serde_json::to_value(documents::execute_search(&captured, 0)).unwrap(),
            captured_result
        );
        drop(held);
        assert!(old_source.upgrade().is_some());
        drop(captured);
        assert!(old_source.upgrade().is_none());
    }

    #[test]
    fn shared_position_sources_restore_reindex_and_count_one_payload() {
        for count in [1023, 1024, 1025] {
            let template = document("template", "red blue red", true, true, 4096);
            let source = template.chunks[0].terms.positions.as_ref().unwrap();
            let mut store = DocumentStore::default();
            store.insert_batch(
                (0..count)
                    .map(|index| {
                        let mut copied = (*template).clone();
                        copied.id = format!("d{index:04}");
                        Arc::new(copied)
                    })
                    .collect(),
            );
            check(&store);
            let census = store.memory_census();
            assert_eq!(census.positions.len(), 1);
            assert_eq!(
                census.bytes["positions.records"],
                std::mem::size_of::<Positions>()
            );
            assert_eq!(
                census.bytes["positions.arc_headers"],
                2 * std::mem::size_of::<usize>()
            );
            assert_eq!(census.bytes["positions.data"], source.data.len());
            assert_eq!(
                census.bytes["slots.position_sources.capacity"],
                store.position_sources.capacity() * std::mem::size_of::<Option<Arc<Positions>>>()
            );

            store.remove("d0001");
            let documents = store
                .documents
                .iter()
                .map(|(id, document)| (Arc::clone(document), store.slots[id].clone()))
                .collect();
            let restored = DocumentStore::restore(
                documents,
                store.slot_count,
                std::mem::take(&mut store.postings),
                None,
            );
            drop(store);
            let mut store = restored;
            check(&store);
            assert!(store.position_sources[1].is_none());
            for index in 0..=count / 2 {
                store.remove(&format!("d{index:04}"));
            }
            check(&store);
            assert_eq!(
                store.slot_count,
                if count > 1024 {
                    store.chunk_count
                } else {
                    count
                }
            );
            assert_eq!(store.memory_census().positions.len(), 1);
            let query = documents::LexicalQuery::parse("\"red blue\"").unwrap();
            let expected = documents::execute_lexical(
                &store,
                &query,
                &documents::LexicalOptions {
                    k: 10,
                    count: true,
                    execution: documents::LexicalExecution::Exhaustive,
                    ..Default::default()
                },
            )
            .unwrap();
            let owners = Arc::strong_count(source);
            assert_eq!(
                documents::execute_lexical(
                    &store,
                    &query,
                    &documents::LexicalOptions {
                        k: 10,
                        count: true,
                        ..Default::default()
                    }
                )
                .unwrap(),
                expected
            );
            assert_eq!(Arc::strong_count(source), owners);
            assert_eq!(expected.count, Some(store.documents.len()));
        }
    }
}
