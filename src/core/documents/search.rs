// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::postings::{PostingList, Postings};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::core::hash::{FastMap, FastSet};
use anyhow::{ensure, Result};
use serde_json::Value;

use super::dense::DenseIndex;
use super::image::ServingImage;
use super::overlay::Overlaid;
use super::{
    matches_document_metadata, normalize_embedding, query_words, terms, top_ranked, Bm25,
    DocumentResult, DocumentStore, SearchOptions, StoredChunk, StoredDocument, MAX_DOCUMENT_BYTES,
};

pub(crate) struct PreparedSearch {
    filters: SearchOptions,
    /// The query's distinct words per form; BM25 here ignores query term frequency.
    query_terms: [BTreeSet<String>; 2],
    embedding: Option<Vec<f64>>,
    k: usize,
    limit: usize,
    lexical_weight: f64,
    semantic_weight: f64,
    minimum: f64,
    rrf: bool,
    lexical_only: bool,
    unfiltered: bool,
    bm25: Bm25,
}

impl PreparedSearch {
    fn matches(&self, document: &StoredDocument) -> bool {
        self.filters
            .document_ids
            .as_ref()
            .is_none_or(|ids| ids.binary_search(&document.id).is_ok())
            && matches_document_metadata(document, &self.filters)
    }
}

pub(crate) struct CapturedSearch {
    queries: Vec<CapturedQuery>,
}

#[derive(Clone)]
struct CapturedChunk {
    document: Arc<StoredDocument>,
    index: usize,
    word_count: u32,
    semantic_cache: Option<f64>,
}

impl CapturedChunk {
    fn chunk(&self) -> &StoredChunk {
        &self.document.chunks[self.index]
    }

    fn order(&self) -> (&str, usize) {
        (&self.document.id, self.index)
    }
}

struct CapturedPosting {
    chunk: u32,
    frequency: u32,
}

/// Slot to captured-chunk position: one `u32` per store slot while that table is small enough
/// to fill per query, a hash map beyond.
enum SlotPositions {
    Dense(Vec<u32>),
    Sparse(FastMap<usize, u32>),
}

const DENSE_SLOT_LIMIT: usize = 1 << 19;

impl SlotPositions {
    fn new(slot_count: usize) -> Self {
        if slot_count <= DENSE_SLOT_LIMIT {
            Self::Dense(vec![u32::MAX; slot_count])
        } else {
            Self::Sparse(FastMap::default())
        }
    }

    fn get(&self, slot: usize) -> Option<u32> {
        match self {
            Self::Dense(table) => table.get(slot).copied().filter(|&p| p != u32::MAX),
            Self::Sparse(map) => map.get(&slot).copied(),
        }
    }

    fn insert(&mut self, slot: usize, position: u32) {
        match self {
            Self::Dense(table) => table[slot] = position,
            Self::Sparse(map) => {
                map.insert(slot, position);
            }
        }
    }
}

struct CapturedQuery {
    prepared: PreparedSearch,
    chunks: Vec<CapturedChunk>,
    postings: [BTreeMap<String, Vec<CapturedPosting>>; 2],
    count: usize,
    words: usize,
    use_postings: bool,
    dense: Option<Arc<DenseIndex>>,
    eligible: Option<FastSet<usize>>,
    excluded_owners: FastSet<usize>,
    dfs: Option<[BTreeMap<String, usize>; 2]>,
    /// The store's text segment, read for result text after the store lock is released (under
    /// the mutation gate for a live search, see [`read_ranked`]).
    text: Option<Arc<super::text::TextSegment>>,
    /// Set when the capture already scored the query (an unfiltered lexical query): `chunks`
    /// are then its top `limit` lexical chunks, best first, and these their BM25 scores.
    lexical_scores: Option<Vec<f64>>,
}

pub(crate) fn prepare_search(
    text: &str,
    options: &SearchOptions,
    embedding_dimensions: Option<usize>,
) -> Result<PreparedSearch> {
    let mut prepared = validate_search(text, options, embedding_dimensions)?;
    prepared.query_terms = [query_words(text, false), query_words(text, true)];
    Ok(prepared)
}

pub(crate) fn validate_search(
    text: &str,
    options: &SearchOptions,
    embedding_dimensions: Option<usize>,
) -> Result<PreparedSearch> {
    ensure!(text.len() <= MAX_DOCUMENT_BYTES, "query exceeds 8 MiB");
    let k = options.k.unwrap_or(10) as usize;
    let limit = options.candidate_limit.unwrap_or(100.max(k as u32)) as usize;
    ensure!(
        (1..=1000).contains(&k) && (k..=4096).contains(&limit),
        "require 1 <= k <= 1000 and k <= candidateLimit <= 4096"
    );
    ensure!(
        options.filter.as_ref().is_none_or(Value::is_object),
        "filter must be a JSON object of exact metadata matches"
    );
    let lexical_weight = options.lexical_weight.unwrap_or(1.0);
    let semantic_weight = options.semantic_weight.unwrap_or(1.0);
    let total_weight = lexical_weight + semantic_weight;
    ensure!(
        [lexical_weight, semantic_weight]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
            && total_weight.is_finite()
            && total_weight > 0.0,
        "search weights must be finite, nonnegative and not both zero"
    );
    let minimum = options.min_semantic_score.unwrap_or(0.0);
    ensure!(
        minimum.is_finite() && (-1.0..=1.0).contains(&minimum),
        "minSemanticScore must be in [-1,1]"
    );
    let rrf = match options.fusion.as_deref() {
        None | Some("blend") => false,
        Some("rrf") => true,
        Some(other) => anyhow::bail!("fusion must be \"blend\" or \"rrf\", not {other:?}"),
    };
    let embedding = options
        .embedding
        .as_ref()
        .map(|values| {
            ensure!(
                embedding_dimensions.is_some(),
                "configure an embedding space before semantic search"
            );
            normalize_embedding(values, embedding_dimensions.unwrap_or(0))
        })
        .transpose()?;
    let unfiltered = options.source_uri.is_none()
        && options.document_ids.is_none()
        && options
            .filter
            .as_ref()
            .and_then(Value::as_object)
            .is_none_or(|filter| filter.is_empty());
    let lexical_only = unfiltered && (embedding.is_none() || semantic_weight == 0.0);
    let bm25 = Bm25::from_options(options)?;
    let document_ids = options.document_ids.as_ref().map(|ids| {
        let mut ids = ids.clone();
        ids.sort_unstable();
        ids.dedup();
        ids
    });
    Ok(PreparedSearch {
        filters: SearchOptions {
            filter: options.filter.clone(),
            source_uri: options.source_uri.clone(),
            document_ids,
            ..Default::default()
        },
        query_terms: Default::default(),
        embedding,
        k,
        limit,
        lexical_weight,
        semantic_weight,
        minimum,
        rrf,
        lexical_only,
        unfiltered,
        bm25,
    })
}

/// The number of documents with at least one chunk containing a term of `text` at a positive
/// frequency, in either stored form: the documents an unfiltered lexical search scores above
/// zero.
pub(crate) fn count_matches(store: &DocumentStore, text: &str) -> Result<usize> {
    ensure!(text.len() <= MAX_DOCUMENT_BYTES, "query exceeds 8 MiB");
    let terms = [terms(text, false), terms(text, true)];
    Ok(super::lexical::execution::count_terms(
        store,
        std::array::from_fn(|form| terms[form].keys().map(String::as_str).collect()),
    ))
}

pub(crate) fn capture_search(
    store: &DocumentStore,
    prepared: Vec<PreparedSearch>,
) -> CapturedSearch {
    CapturedSearch {
        queries: prepared
            .into_iter()
            .map(|prepared| capture_query(store, prepared))
            .collect(),
    }
}

fn capture_query(store: &DocumentStore, prepared: PreparedSearch) -> CapturedQuery {
    if prepared.semantic_weight > 0.0 {
        if let Some(index) = store.dense_index.as_ref().filter(|index| {
            prepared
                .embedding
                .as_ref()
                .is_some_and(|query| query.len() == index.dimensions)
        }) {
            return capture_planned_query(store, prepared, Some(Arc::clone(index)));
        }
    }
    if prepared.lexical_only {
        return capture_ranked_lexical(store, prepared);
    }
    if prepared.embedding.is_none() || prepared.semantic_weight == 0.0 {
        return capture_planned_query(store, prepared, None);
    }
    capture_postings(store, prepared)
}

/// Captures the chunks a query may rank and, where postings are cheaper than a scan of the
/// candidates, their postings, for [`execute_chunks`] to score.
fn capture_postings(store: &DocumentStore, prepared: PreparedSearch) -> CapturedQuery {
    let mut chunks = Vec::new();
    let mut positions = SlotPositions::new(store.slot_count);
    let (count, words, use_postings) = if prepared.lexical_only {
        (store.chunk_count, store.word_count, true)
    } else {
        let candidates = store.candidate_ids(&prepared.filters);
        let documents: Vec<_> = match candidates {
            Some(ids) => ids.into_iter().map(|id| &store.documents[id]).collect(),
            None => store.documents.values().collect(),
        };
        let mut words = 0;
        for document in documents
            .into_iter()
            .filter(|document| prepared.matches(document))
        {
            for (index, (&slot, chunk)) in store.slots[&document.id]
                .iter()
                .zip(&document.chunks)
                .enumerate()
            {
                positions.insert(slot, chunks.len() as u32);
                words += chunk.word_count;
                chunks.push(CapturedChunk {
                    document: Arc::clone(document),
                    index,
                    word_count: super::index::slot_words(chunk.word_count),
                    semantic_cache: None,
                });
            }
        }
        let posting_work: usize = prepared
            .query_terms
            .iter()
            .enumerate()
            .map(|(form, terms)| {
                terms
                    .iter()
                    .map(|word| store.postings[form].get(word).map_or(0, PostingList::len))
                    .sum::<usize>()
            })
            .sum();
        let scan_work = chunks
            .len()
            .saturating_mul(prepared.query_terms.iter().map(BTreeSet::len).sum());
        (
            chunks.len(),
            words,
            posting_work.saturating_add(store.slot_count) < scan_work,
        )
    };
    let mut postings: [BTreeMap<String, Vec<CapturedPosting>>; 2] = Default::default();
    if use_postings && (prepared.lexical_weight > 0.0 || !prepared.lexical_only) {
        for (form, terms) in prepared.query_terms.iter().enumerate() {
            for word in terms {
                let Some(source) = store.postings[form].get(word) else {
                    continue;
                };
                let mut captured = Vec::with_capacity(source.len().min(count));
                source.for_each(|slot, frequency| {
                    let index = match positions.get(slot) {
                        Some(index) => index,
                        None if prepared.lexical_only => {
                            let (document, index) = store.owned_chunk(slot);
                            let position = chunks.len() as u32;
                            chunks.push(CapturedChunk {
                                document,
                                index,
                                word_count: store.slot_word_count(slot),
                                semantic_cache: None,
                            });
                            positions.insert(slot, position);
                            position
                        }
                        None => return,
                    };
                    captured.push(CapturedPosting {
                        chunk: index,
                        frequency,
                    });
                });
                postings[form].insert(word.clone(), captured);
            }
        }
    }
    CapturedQuery {
        prepared,
        chunks,
        postings,
        count,
        words,
        use_postings,
        dense: None,
        eligible: None,
        excluded_owners: FastSet::default(),
        dfs: None,
        text: store.text_segment(),
        lexical_scores: None,
    }
}

/// Captures an unfiltered lexical query already ranked: the postings of its terms are scored
/// into this thread's slot table exactly as [`execute_chunks`] scores captured postings, and
/// only the top `limit` chunks are captured, so the capture holds `limit` documents rather than
/// one per matching chunk.
fn capture_ranked_lexical(store: &DocumentStore, prepared: PreparedSearch) -> CapturedQuery {
    let mut chunks = Vec::new();
    let mut scores = Vec::new();
    if prepared.lexical_weight > 0.0 {
        let ranked = LEXICAL_SCRATCH.with(|scratch| scratch.borrow_mut().rank(store, &prepared));
        chunks.reserve_exact(ranked.len());
        scores.reserve_exact(ranked.len());
        for (score, slot) in ranked {
            let slot = slot as usize;
            let (document, index) = store.owned_chunk(slot);
            chunks.push(CapturedChunk {
                document,
                index,
                word_count: store.slot_word_count(slot),
                semantic_cache: None,
            });
            scores.push(score);
        }
    }
    CapturedQuery {
        prepared,
        chunks,
        postings: Default::default(),
        count: store.chunk_count,
        words: store.word_count,
        use_postings: true,
        dense: None,
        eligible: None,
        excluded_owners: FastSet::default(),
        dfs: None,
        text: store.text_segment(),
        lexical_scores: Some(scores),
    }
}

/// What [`LexicalScratch::rank`] reads: the live store, or a mapped checkpoint of one.
pub(super) trait LexicalSource {
    type List<'a>: PostingSet
    where
        Self: 'a;
    fn slot_count(&self) -> usize;
    fn chunk_count(&self) -> usize;
    fn word_count(&self) -> usize;
    /// The instance and generation cached norms are valid for.
    fn state(&self) -> (super::index::StoreIdentity, u64);
    fn postings(&self, form: usize, term: &str) -> Option<Self::List<'_>>;
    fn slot_word_count(&self, slot: usize) -> u32;
    /// `(document id, chunk index)` of the chunk in `slot`, the tie order of equal scores.
    fn slot_order(&self, slot: usize) -> (&str, usize);
}

/// A term's postings as [`LexicalScratch::rank`] reads them.
pub(super) trait PostingSet {
    fn len(&self) -> usize;
    fn for_each(&self, f: impl FnMut(usize, u32));
}

impl PostingSet for Postings<'_> {
    fn len(&self) -> usize {
        Postings::len(self)
    }

    fn for_each(&self, f: impl FnMut(usize, u32)) {
        Postings::for_each(self, f);
    }
}

impl PostingSet for &PostingList {
    fn len(&self) -> usize {
        PostingList::len(self)
    }

    fn for_each(&self, f: impl FnMut(usize, u32)) {
        PostingList::for_each(self, f);
    }
}

impl LexicalSource for DocumentStore {
    type List<'a> = &'a PostingList;

    fn slot_count(&self) -> usize {
        self.slot_count
    }

    fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    fn word_count(&self) -> usize {
        self.word_count
    }

    fn state(&self) -> (super::index::StoreIdentity, u64) {
        (self.identity, self.generation)
    }

    fn postings(&self, form: usize, term: &str) -> Option<&PostingList> {
        self.postings[form].get(term)
    }

    fn slot_word_count(&self, slot: usize) -> u32 {
        DocumentStore::slot_word_count(self, slot)
    }

    fn slot_order(&self, slot: usize) -> (&str, usize) {
        DocumentStore::slot_order(self, slot)
    }
}

/// [`capture_ranked_lexical`] over a serving image, for queries [`overlay_serves`] accepts.
fn capture_image(image: &ServingImage, prepared: PreparedSearch) -> anyhow::Result<CapturedQuery> {
    let mut chunks = Vec::new();
    let mut scores = Vec::new();
    if prepared.lexical_weight > 0.0 {
        let ranked = LEXICAL_SCRATCH.with(|scratch| scratch.borrow_mut().rank(image, &prepared));
        chunks.reserve_exact(ranked.len());
        scores.reserve_exact(ranked.len());
        for (score, slot) in ranked {
            let slot = slot as usize;
            let (document, index) = image.chunk(slot)?;
            chunks.push(CapturedChunk {
                document,
                index,
                word_count: image.slot_word_count(slot),
                semantic_cache: None,
            });
            scores.push(score);
        }
    }
    Ok(CapturedQuery {
        prepared,
        chunks,
        postings: Default::default(),
        count: image.chunk_count(),
        words: image.word_count(),
        use_postings: true,
        dense: None,
        eligible: None,
        excluded_owners: FastSet::default(),
        dfs: None,
        text: None,
        lexical_scores: Some(scores),
    })
}

/// Whether `prepared` ranks from the checkpoint under `store` together with the store's own
/// documents as [`capture_ranked_lexical`] would from the store rebuilt from them: unfiltered,
/// lexical only, without a query embedding (so no chunk embedding is consulted), on a store
/// small enough for the per-slot ranking.
pub(super) fn overlay_serves(prepared: &PreparedSearch, store: &DocumentStore) -> bool {
    store.base.as_ref().is_some_and(|base| {
        prepared.lexical_only
            && prepared.embedding.is_none()
            && base.image.slot_total() + store.slot_count <= DENSE_SLOT_LIMIT
    })
}

/// [`capture_image`] over the checkpoint under `store` and the store's own documents.
fn capture_overlaid(
    store: &DocumentStore,
    prepared: PreparedSearch,
) -> anyhow::Result<CapturedQuery> {
    let base = store.base.as_ref().expect("checked by overlay_serves");
    if base.intact() && store.documents.is_empty() {
        return capture_image(&base.image, prepared);
    }
    let source = Overlaid { base, store };
    let split = source.split();
    let mut chunks = Vec::new();
    let mut scores = Vec::new();
    if prepared.lexical_weight > 0.0 {
        let ranked = LEXICAL_SCRATCH.with(|scratch| scratch.borrow_mut().rank(&source, &prepared));
        chunks.reserve_exact(ranked.len());
        scores.reserve_exact(ranked.len());
        for (score, slot) in ranked {
            let slot = slot as usize;
            let (document, index) = match slot.checked_sub(split) {
                None => base.image.chunk(slot)?,
                Some(own) => store.owned_chunk(own),
            };
            chunks.push(CapturedChunk {
                document,
                index,
                word_count: source.slot_word_count(slot),
                semantic_cache: None,
            });
            scores.push(score);
        }
    }
    Ok(CapturedQuery {
        prepared,
        chunks,
        postings: Default::default(),
        count: source.chunk_count(),
        words: source.word_count(),
        use_postings: true,
        dense: None,
        eligible: None,
        excluded_owners: FastSet::default(),
        dfs: None,
        text: store.text_segment(),
        lexical_scores: Some(scores),
    })
}

pub(super) fn capture_overlaid_search(
    store: &DocumentStore,
    prepared: Vec<PreparedSearch>,
) -> anyhow::Result<CapturedSearch> {
    Ok(CapturedSearch {
        queries: prepared
            .into_iter()
            .map(|prepared| capture_overlaid(store, prepared))
            .collect::<anyhow::Result<_>>()?,
    })
}

/// The store state and BM25 constants a thread's cached length normalizations belong to.
#[derive(Clone, Copy, PartialEq)]
struct NormKey {
    store: super::index::StoreIdentity,
    generation: u64,
    k1: u64,
    one_minus_b: u64,
    b: u64,
}

/// Per-thread working memory of [`capture_ranked_lexical`], reused across queries so a query
/// neither allocates nor clears a table per store slot. It grows to the largest store a thread
/// queries (20 bytes per slot plus 4 per posting of the longest query: no more than the
/// store's own owner and word-count tables per slot) and is not shrunk.
#[derive(Default)]
struct LexicalScratch {
    /// BM25 sum per slot; every entry is 0 between queries.
    scores: Vec<f64>,
    /// Length normalization per slot under `key`, valid where `norm_epochs` holds `epoch`.
    norms: Vec<f64>,
    norm_epochs: Vec<u32>,
    epoch: u32,
    key: Option<NormKey>,
    /// Whether every slot's norm is valid under `key`.
    complete: bool,
    /// Slots a sparse query reached, with repeats; read back once each by the selection.
    touched: Vec<u32>,
    /// Selection candidates and the bounded heap of the best scores (as `f64` bits) seen.
    candidates: Vec<(f64, u32)>,
    best: std::collections::BinaryHeap<std::cmp::Reverse<u64>>,
    /// Set while a query runs; a query that finds it set (an earlier one panicked midway)
    /// zeroes every score first.
    dirty: bool,
}

thread_local! {
    static LEXICAL_SCRATCH: RefCell<LexicalScratch> = RefCell::new(LexicalScratch::default());
}

impl LexicalScratch {
    /// The query's top `limit` chunks with a positive BM25 score as `(score, slot)`, ordered by
    /// score descending, then document id and chunk index ascending. Scores are the sums
    /// `execute_chunks` computes from captured postings: the same terms in the same order
    /// (form 0 terms, then form 1 terms, each sorted), the same expressions, from 0.
    fn rank<S: LexicalSource>(&mut self, store: &S, prepared: &PreparedSearch) -> Vec<(f64, u32)> {
        let slots = store.slot_count();
        if self.dirty {
            self.scores.fill(0.0);
        }
        if self.scores.len() < slots {
            self.scores.resize(slots, 0.0);
            self.norms.resize(slots, 0.0);
            self.norm_epochs.resize(slots, 0);
            self.complete = false;
        }
        self.dirty = true;
        let n = store.chunk_count() as f64;
        let average = (store.word_count() as f64 / n).max(1.0);
        let bm25 = prepared.bm25;
        let mut lists = Vec::new();
        let mut postings = 0usize;
        for (form, terms) in prepared.query_terms.iter().enumerate() {
            for word in terms {
                if let Some(list) = store.postings(form, word) {
                    let df = list.len() as f64;
                    postings = postings.saturating_add(list.len());
                    lists.push((Bm25::idf(n, df), list));
                }
            }
        }
        // A query whose postings cover a good part of the store reads every slot's norm and
        // score; a sparser one keeps its norms lazy and its selection to the slots it reached.
        let dense = postings >= slots / 2;
        let (identity, generation) = store.state();
        let key = NormKey {
            store: identity,
            generation,
            k1: bm25.k1.to_bits(),
            one_minus_b: bm25.one_minus_b.to_bits(),
            b: bm25.b.to_bits(),
        };
        if self.key != Some(key) {
            self.key = Some(key);
            self.complete = false;
            self.epoch = match self.epoch.checked_add(1) {
                Some(epoch) => epoch,
                None => {
                    self.norm_epochs.fill(0);
                    1
                }
            };
        }
        if !self.complete && dense {
            for slot in 0..slots {
                if self.norm_epochs[slot] != self.epoch {
                    self.norm_epochs[slot] = self.epoch;
                    self.norms[slot] = bm25.norm(f64::from(store.slot_word_count(slot)), average);
                }
            }
            self.complete = true;
        }
        let scores = &mut self.scores[..slots];
        self.touched.clear();
        if dense {
            let norms = &self.norms[..slots];
            for (idf, list) in &lists {
                let idf = *idf;
                list.for_each(|slot, frequency| {
                    scores[slot] += bm25.weight(idf, f64::from(frequency), norms[slot]);
                });
            }
        } else {
            self.touched.resize(postings, 0);
            let (norms, norm_epochs, touched) =
                (&mut self.norms, &mut self.norm_epochs, &mut self.touched);
            let (epoch, complete) = (self.epoch, self.complete);
            let mut reached = 0;
            for (idf, list) in &lists {
                let idf = *idf;
                list.for_each(|slot, frequency| {
                    if !complete && norm_epochs[slot] != epoch {
                        norm_epochs[slot] = epoch;
                        norms[slot] = bm25.norm(f64::from(store.slot_word_count(slot)), average);
                    }
                    // Weights are never -0, so a slot reads +0 until its first nonzero weight;
                    // a slot recorded twice reads 0 the second time in the selection.
                    let score = scores[slot];
                    touched[reached] = slot as u32;
                    reached += usize::from(score == 0.0);
                    scores[slot] = score + bm25.weight(idf, f64::from(frequency), norms[slot]);
                });
            }
            self.touched.truncate(reached);
        }
        // Selection: a bounded heap holds the best `limit` scores read so far (as `f64` bits,
        // which order positive values); once it is full its smallest is the floor a later
        // score must reach. Every score at or above the floor when read is kept, so all chunks
        // scoring at least the final floor (the `limit`-th best score) are among the kept
        // ones, ties included, and the exact order decides between them.
        let limit = prepared.limit;
        let (candidates, best) = (&mut self.candidates, &mut self.best);
        candidates.clear();
        best.clear();
        // The least admissible score as bits: any positive one until the heap is full, then the
        // floor. One unsigned range test admits positive scores (infinity included) at or
        // above it and turns away zero, NaN and the scores below it without branching on each.
        const INFINITY: u64 = 0x7FF0_0000_0000_0000;
        let mut floor = 1u64;
        let mut consider = |slot: u32, score: f64| {
            let bits = score.to_bits();
            if bits.wrapping_sub(floor) > INFINITY - floor {
                return;
            }
            candidates.push((score, slot));
            if best.len() < limit {
                best.push(std::cmp::Reverse(bits));
            } else if bits > floor {
                *best.peek_mut().expect("full heap") = std::cmp::Reverse(bits);
            } else {
                return;
            }
            if best.len() == limit {
                floor = best.peek().expect("full heap").0;
            }
        };
        if dense {
            for (slot, score) in scores.iter_mut().enumerate() {
                let value = std::mem::replace(score, 0.0);
                consider(slot as u32, value);
            }
        } else {
            for &slot in &self.touched {
                consider(slot, std::mem::replace(&mut scores[slot as usize], 0.0));
            }
        }
        let bar = if best.len() < limit {
            0
        } else {
            best.peek().expect("full heap").0
        };
        candidates.retain(|&(score, _)| score.to_bits() >= bar);
        top_ranked(candidates, limit, |a, b| {
            b.0.total_cmp(&a.0).then_with(|| {
                store
                    .slot_order(a.1 as usize)
                    .cmp(&store.slot_order(b.1 as usize))
            })
        });
        self.dirty = false;
        candidates.clone()
    }
}

struct FilteredPostings<'a> {
    list: &'a PostingList,
    slots: &'a [bool],
    count: usize,
}

impl PostingSet for FilteredPostings<'_> {
    fn len(&self) -> usize {
        self.count
    }

    fn for_each(&self, mut f: impl FnMut(usize, u32)) {
        self.list.for_each(|slot, frequency| {
            if self.slots[slot] {
                f(slot, frequency);
            }
        });
    }
}

struct FilteredLexical<'a> {
    store: &'a DocumentStore,
    slots: &'a [bool],
    dfs: &'a [BTreeMap<String, usize>; 2],
    count: usize,
    words: usize,
    identity: super::index::StoreIdentity,
}

impl LexicalSource for FilteredLexical<'_> {
    type List<'a>
        = FilteredPostings<'a>
    where
        Self: 'a;

    fn slot_count(&self) -> usize {
        self.slots.len()
    }
    fn chunk_count(&self) -> usize {
        self.count
    }
    fn word_count(&self) -> usize {
        self.words
    }
    fn state(&self) -> (super::index::StoreIdentity, u64) {
        (self.identity, self.store.generation)
    }
    fn slot_word_count(&self, slot: usize) -> u32 {
        self.store.slot_word_count(slot)
    }
    fn slot_order(&self, slot: usize) -> (&str, usize) {
        self.store.slot_order(slot)
    }

    fn postings(&self, form: usize, term: &str) -> Option<Self::List<'_>> {
        Some(FilteredPostings {
            list: self.store.postings[form].get(term)?,
            slots: self.slots,
            count: self.dfs[form][term],
        })
    }
}

fn capture_planned_query(
    store: &DocumentStore,
    prepared: PreparedSearch,
    dense: Option<Arc<DenseIndex>>,
) -> CapturedQuery {
    let mut eligible = None;
    let (count, words, dfs, ranked) = if prepared.unfiltered {
        let dfs = std::array::from_fn(|form| {
            prepared.query_terms[form]
                .iter()
                .map(|word| {
                    (
                        word.clone(),
                        store.postings[form].get(word).map_or(0, PostingList::len),
                    )
                })
                .collect()
        });
        let ranked = if prepared.lexical_weight > 0.0 {
            LEXICAL_SCRATCH
                .with(|scratch| scratch.borrow_mut().rank(store, &prepared))
                .into_iter()
                .map(|(score, slot)| (score, slot as usize))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        (store.chunk_count, store.word_count, dfs, ranked)
    } else {
        let candidates = store.candidate_ids(&prepared.filters);
        let mut documents: Vec<_> = match candidates {
            Some(ids) => ids.into_iter().map(|id| &store.documents[id]).collect(),
            None => store.documents.values().collect(),
        };
        documents.retain(|document| prepared.matches(document));
        let count: usize = documents.iter().map(|document| document.chunks.len()).sum();
        if count < super::dense::MIN_ROWS {
            return capture_postings(store, prepared);
        }
        let posting_work = prepared
            .query_terms
            .iter()
            .enumerate()
            .flat_map(|(form, terms)| {
                terms
                    .iter()
                    .map(move |word| store.postings[form].get(word).map_or(0, PostingList::len))
            })
            .fold(0usize, usize::saturating_add);
        let scan_work = count.saturating_mul(prepared.query_terms.iter().map(BTreeSet::len).sum());
        let slots = (store.slot_count <= DENSE_SLOT_LIMIT
            && posting_work.saturating_add(store.slot_count) < scan_work)
            .then(|| {
                let mut slots = vec![false; store.slot_count];
                for document in &documents {
                    for &slot in &store.slots[&document.id] {
                        slots[slot] = true;
                    }
                }
                slots
            });
        if slots.is_some() && dense.is_some() && documents.len() <= DENSE_SLOT_LIMIT {
            // The tree pins its owners, so current-owner membership also excludes replaced rows.
            eligible = Some(
                documents
                    .iter()
                    .map(|document| Arc::as_ptr(document) as usize)
                    .collect(),
            );
        }
        let words = documents
            .iter()
            .flat_map(|document| &document.chunks)
            .map(|chunk| chunk.word_count)
            .sum();
        let dfs: [BTreeMap<String, usize>; 2] = std::array::from_fn(|form| {
            prepared.query_terms[form]
                .iter()
                .map(|word| {
                    let df = if let Some(slots) = &slots {
                        let mut count = 0;
                        if let Some(list) = store.postings[form].get(word) {
                            list.for_each(|slot, _| count += usize::from(slots[slot]));
                        }
                        count
                    } else {
                        documents
                            .iter()
                            .flat_map(|document| &document.chunks)
                            .filter(|chunk| {
                                chunk.stemmed == (form == 1) && chunk.terms.contains_key(word)
                            })
                            .count()
                    };
                    (word.clone(), df)
                })
                .collect()
        });
        let mut ranked = Vec::new();
        if let Some(slots) = slots.as_ref().filter(|_| prepared.lexical_weight > 0.0) {
            let source = FilteredLexical {
                store,
                slots,
                dfs: &dfs,
                count,
                words,
                identity: super::index::StoreIdentity::default(),
            };
            ranked = LEXICAL_SCRATCH
                .with(|scratch| scratch.borrow_mut().rank(&source, &prepared))
                .into_iter()
                .map(|(score, slot)| (score, slot as usize))
                .collect();
        } else if prepared.lexical_weight > 0.0 {
            let idfs: [Vec<_>; 2] = std::array::from_fn(|form| {
                dfs[form]
                    .iter()
                    .map(|(word, &df)| (word, Bm25::idf(count as f64, df as f64)))
                    .collect()
            });
            let average = (words as f64 / count as f64).max(1.0);
            let compare = |a: &(f64, usize), b: &(f64, usize)| {
                b.0.total_cmp(&a.0)
                    .then_with(|| store.slot_order(a.1).cmp(&store.slot_order(b.1)))
            };
            for document in documents {
                for (&slot, chunk) in store.slots[&document.id].iter().zip(&document.chunks) {
                    let norm = prepared.bm25.norm(chunk.word_count as f64, average);
                    let score: f64 = idfs[usize::from(chunk.stemmed)]
                        .iter()
                        .map(|(word, idf)| {
                            prepared.bm25.weight(
                                *idf,
                                f64::from(chunk.terms.get(word).copied().unwrap_or(0)),
                                norm,
                            )
                        })
                        .sum();
                    if score > 0.0 {
                        ranked.push((score, slot));
                        if ranked.len() >= 2 * prepared.limit {
                            top_ranked(&mut ranked, prepared.limit, compare);
                        }
                    }
                }
            }
            top_ranked(&mut ranked, prepared.limit, compare);
        }
        (count, words, dfs, ranked)
    };
    let mut chunks: Vec<_> = ranked
        .into_iter()
        .map(|(_, slot)| {
            let (document, index) = store.owned_chunk(slot);
            CapturedChunk {
                document,
                index,
                word_count: store.slot_word_count(slot),
                semantic_cache: None,
            }
        })
        .collect();
    if dense.is_some() && !store.dense_changes.is_empty() {
        let mut captured_slots: FastSet<_> = chunks
            .iter()
            .map(|chunk| store.slots[&chunk.document.id][chunk.index])
            .collect();
        for id in store.dense_changes.keys() {
            let Some(document) = store
                .documents
                .get(id)
                .filter(|document| prepared.matches(document))
            else {
                continue;
            };
            for (index, (&slot, chunk)) in store.slots[id].iter().zip(&document.chunks).enumerate()
            {
                if chunk.embedding.is_some() && captured_slots.insert(slot) {
                    chunks.push(CapturedChunk {
                        document: Arc::clone(document),
                        index,
                        word_count: store.slot_word_count(slot),
                        semantic_cache: None,
                    });
                }
            }
        }
    }
    let excluded_owners = if dense.is_some() && eligible.is_none() {
        store
            .dense_changes
            .values()
            .filter_map(|change| change.previous_owner)
            .collect()
    } else {
        FastSet::default()
    };
    CapturedQuery {
        prepared,
        chunks,
        postings: Default::default(),
        count,
        words,
        use_postings: false,
        dense,
        eligible,
        excluded_owners,
        dfs: Some(dfs),
        text: store.text_segment(),
        lexical_scores: None,
    }
}

impl CapturedSearch {
    pub(crate) fn len(&self) -> usize {
        self.queries.len()
    }

    pub(crate) fn into_prepared(self) -> Vec<PreparedSearch> {
        self.queries
            .into_iter()
            .map(|query| query.prepared)
            .collect()
    }
}

/// A ranked result whose chunk text is not read yet (see [`read_ranked`]).
pub(crate) struct RankedHit {
    result: DocumentResult,
    document: Arc<StoredDocument>,
    chunk: usize,
}

impl RankedHit {
    fn read(mut self, segment: Option<&super::text::TextSegment>) -> DocumentResult {
        self.result.text = super::index::text_of(&self.document.chunks[self.chunk], segment);
        self.result
    }
}

/// Ranks and reads a capture. A replace or delete scrubs the text it supersedes, so a capture
/// that outlived the store's locks reads such text as empty (see [`read_ranked`]).
pub(crate) fn execute_search(captured: &CapturedSearch, query: usize) -> Vec<DocumentResult> {
    let segment = captured.queries[query].text.as_deref();
    rank_search(captured, query)
        .into_iter()
        .map(|hit| hit.read(segment))
        .collect()
}

/// [`execute_search`] without reading the hits' text, so ranking needs no lock.
pub(crate) fn rank_search(captured: &CapturedSearch, query: usize) -> Vec<RankedHit> {
    execute_query(&captured.queries[query])
}

/// The hits with their text, or `None` when the text of one was scrubbed since the capture
/// (its document replaced or deleted). The caller holds the mutation gate's read side, which
/// excludes the writes that scrub, so a read never overlaps one.
pub(crate) fn read_ranked(
    captured: &CapturedSearch,
    query: usize,
    hits: Vec<RankedHit>,
) -> Option<Vec<DocumentResult>> {
    let segment = captured.queries[query].text.as_deref();
    hits.into_iter()
        .map(|hit| {
            let scrubbed = super::index::text_scrubbed(&hit.document.chunks[hit.chunk], segment);
            (!scrubbed).then(|| hit.read(segment))
        })
        .collect()
}

fn execute_query(captured: &CapturedQuery) -> Vec<RankedHit> {
    if let Some(index) = &captured.dense {
        let query = captured
            .prepared
            .embedding
            .as_ref()
            .expect("indexed search lost its query embedding");
        let (hits, _) = if captured.prepared.unfiltered && captured.excluded_owners.is_empty() {
            index.search(query, captured.prepared.limit, captured.prepared.minimum)
        } else {
            index.search_where(
                query,
                captured.prepared.limit,
                captured.prepared.minimum,
                |document| {
                    captured.eligible.as_ref().map_or_else(
                        || {
                            !captured
                                .excluded_owners
                                .contains(&(document as *const StoredDocument as usize))
                                && captured.prepared.matches(document)
                        },
                        |eligible| eligible.contains(&(document as *const StoredDocument as usize)),
                    )
                },
            )
        };
        let mut chunks = captured.chunks.clone();
        let mut positions: FastMap<_, _> = captured
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| (chunk.order(), i))
            .collect();
        for hit in &hits {
            let key = (hit.document.id.as_str(), hit.chunk);
            if let Some(&i) = positions.get(&key) {
                chunks[i].semantic_cache = Some(hit.score);
            } else {
                positions.insert(key, chunks.len());
                chunks.push(CapturedChunk {
                    document: Arc::clone(&hit.document),
                    index: hit.chunk,
                    word_count: super::index::slot_words(hit.document.chunks[hit.chunk].word_count),
                    semantic_cache: Some(hit.score),
                });
            }
        }
        return execute_chunks(captured, &chunks);
    }
    execute_chunks(captured, &captured.chunks)
}

fn execute_chunks(captured: &CapturedQuery, chunks: &[CapturedChunk]) -> Vec<RankedHit> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let prepared = &captured.prepared;
    let semantic = |captured_chunk: &CapturedChunk| {
        captured_chunk.semantic_cache.or_else(|| {
            prepared.embedding.as_ref().and_then(|q| {
                captured_chunk
                    .chunk()
                    .embedding
                    .as_ref()
                    .map(|values| values.cosine(q))
            })
        })
    };
    let scores: Vec<(f64, Option<f64>)> = match &captured.lexical_scores {
        Some(lexical) => chunks
            .iter()
            .zip(lexical)
            .map(|(chunk, &score)| (score, semantic(chunk)))
            .collect(),
        None => score_chunks(captured, chunks, semantic),
    };
    let order = |a: usize, b: usize| chunks[a].order().cmp(&chunks[b].order());
    // Scores travel with their indices so the selection compares without indirection.
    let mut lexical: Vec<(f64, usize)> = if prepared.lexical_weight > 0.0 {
        scores
            .iter()
            .enumerate()
            .filter(|(_, score)| score.0 > 0.0)
            .map(|(i, score)| (score.0, i))
            .collect()
    } else {
        Vec::new()
    };
    let mut semantic: Vec<(f64, usize)> = if prepared.semantic_weight > 0.0 {
        scores
            .iter()
            .enumerate()
            .filter_map(|(i, score)| {
                score
                    .1
                    .filter(|&value| value >= prepared.minimum)
                    .map(|value| (value, i))
            })
            .collect()
    } else {
        Vec::new()
    };
    let by_score =
        |a: &(f64, usize), b: &(f64, usize)| b.0.total_cmp(&a.0).then_with(|| order(a.1, b.1));
    top_ranked(&mut lexical, prepared.limit, by_score);
    top_ranked(&mut semantic, prepared.limit, by_score);
    // Each list's contribution per chunk, lexical first; a stable sort by chunk then sums them
    // in that order, starting from 0.
    let mut contributions = Vec::with_capacity(lexical.len() + semantic.len());
    for (indices, weight) in [
        (&lexical, prepared.lexical_weight),
        (&semantic, prepared.semantic_weight),
    ] {
        if prepared.rrf {
            for (rank, &(_, i)) in indices.iter().enumerate() {
                contributions.push((i, weight / (60.0 + rank as f64 + 1.0)));
            }
            continue;
        }
        let (high, low) = match (indices.first(), indices.last()) {
            (Some(&(first, _)), Some(&(last, _))) => (first, last),
            _ => continue,
        };
        for &(value, i) in indices {
            let normalized = if high > low {
                (value - low) / (high - low)
            } else {
                1.0
            };
            contributions.push((i, weight * normalized));
        }
    }
    // With one list contributing, each chunk appears once and the final selection orders
    // them, so only a fusion of both lists needs grouping by chunk.
    if !lexical.is_empty() && !semantic.is_empty() {
        contributions.sort_by_key(|&(i, _)| i);
    }
    let mut ranked: Vec<(usize, f64)> = Vec::with_capacity(contributions.len());
    for (i, contribution) in contributions {
        match ranked.last_mut() {
            Some(fused) if fused.0 == i => fused.1 += contribution,
            _ => ranked.push((i, 0.0 + contribution)),
        }
    }
    top_ranked(&mut ranked, prepared.k, |a, b| {
        b.1.total_cmp(&a.1).then_with(|| order(a.0, b.0))
    });
    ranked
        .into_iter()
        .map(|(i, score)| {
            let document = &chunks[i].document;
            let chunk = chunks[i].chunk();
            RankedHit {
                result: DocumentResult {
                    id: chunk.id.clone(),
                    document_id: document.id.clone(),
                    source_uri: document.source_uri.clone(),
                    version: document.version.clone(),
                    metadata: document.metadata.clone(),
                    text: None,
                    start_byte: chunk.start_byte,
                    end_byte: chunk.end_byte,
                    score,
                    lexical_score: scores[i].0,
                    semantic_score: scores[i].1,
                },
                document: Arc::clone(document),
                chunk: chunks[i].index,
            }
        })
        .collect()
}

/// Lexical (BM25) and semantic scores of captured `chunks` whose lexical scores the capture
/// did not compute.
fn score_chunks(
    captured: &CapturedQuery,
    chunks: &[CapturedChunk],
    semantic: impl Fn(&CapturedChunk) -> Option<f64>,
) -> Vec<(f64, Option<f64>)> {
    let prepared = &captured.prepared;
    let n = captured.count as f64;
    let average = (captured.words as f64 / n).max(1.0);
    let idf: [BTreeMap<&String, f64>; 2] = std::array::from_fn(|form| {
        prepared.query_terms[form]
            .iter()
            .map(|word| {
                let df = if let Some(dfs) = &captured.dfs {
                    dfs[form][word]
                } else if captured.use_postings {
                    captured.postings[form].get(word).map_or(0, Vec::len)
                } else {
                    chunks
                        .iter()
                        .filter(|chunk| {
                            chunk.chunk().stemmed == (form == 1)
                                && chunk.chunk().terms.contains_key(word)
                        })
                        .count()
                } as f64;
                (word, Bm25::idf(n, df))
            })
            .collect()
    });
    let mut norms = Vec::with_capacity(if captured.use_postings {
        chunks.len()
    } else {
        0
    });
    let mut scores: Vec<(f64, Option<f64>)> = chunks
        .iter()
        .map(|captured_chunk| {
            let norm = prepared
                .bm25
                .norm(captured_chunk.word_count as f64, average);
            let lexical = if captured.use_postings {
                norms.push(norm);
                0.0
            } else {
                let chunk = captured_chunk.chunk();
                idf[usize::from(chunk.stemmed)]
                    .iter()
                    .map(|(word, idf)| {
                        let tf = f64::from(chunk.terms.get(word).copied().unwrap_or(0));
                        prepared.bm25.weight(*idf, tf, norm)
                    })
                    .sum()
            };
            (lexical, semantic(captured_chunk))
        })
        .collect();
    if captured.use_postings {
        for (form, weights) in idf.iter().enumerate() {
            for (word, idf) in weights {
                if let Some(posting) = captured.postings[form].get(*word) {
                    for posting in posting {
                        let chunk = posting.chunk as usize;
                        let tf = f64::from(posting.frequency);
                        scores[chunk].0 += prepared.bm25.weight(*idf, tf, norms[chunk]);
                    }
                }
            }
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::encoding::TextEncoder;
    use crate::core::sync::RwLock;
    use serde_json::json;
    use std::sync::Barrier;

    #[test]
    fn planned_filtered_lexical_and_hybrid_match_oracle_under_term_skew() {
        let mut store = store(512);
        let mut long = document(7, 1);
        long.chunks[0].word_count = 300;
        store.insert(long.id.clone(), long);
        for trained in [false, true] {
            if trained {
                let index = super::super::build_dense_index(
                    super::super::capture_dense_index(&store, 3)
                        .unwrap()
                        .unwrap(),
                )
                .unwrap()
                .unwrap();
                assert!(super::super::publish_dense_index(&mut store, &index));
            }
            for group in [1, 0, 1] {
                for query in [
                    "rare7 rare9 ghost absent",
                    "common",
                    "common rare7 rare9 ghost absent",
                ] {
                    for embedding in [None, Some(vec![1.0, 0.0, 0.0])] {
                        for k1 in [0.0, 1.2] {
                            let options = SearchOptions {
                                k: Some(3),
                                candidate_limit: Some(5),
                                filter: Some(json!({"group": group})),
                                embedding: embedding.clone(),
                                bm25_k1: Some(k1),
                                min_semantic_score: Some(-1.0),
                                ..Default::default()
                            };
                            let captured = capture_search(
                                &store,
                                vec![prepare_search(query, &options, Some(3)).unwrap()],
                            );
                            if trained || embedding.is_none() {
                                assert!(captured.queries[0].chunks.len() <= 5);
                                assert_eq!(captured.queries[0].count, 512);
                            }
                            let expected =
                                super::super::search_impl(&store, query, &options, Some(3), false)
                                    .unwrap();
                            assert_eq!(
                                serde_json::to_value(execute_search(&captured, 0)).unwrap(),
                                serde_json::to_value(expected).unwrap()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn delta_capture_bounds_owners_and_survives_later_replacement() {
        let mut store = store(512);
        let index = super::super::build_dense_index(
            super::super::capture_dense_index(&store, 3)
                .unwrap()
                .unwrap(),
        )
        .unwrap()
        .unwrap();
        assert!(super::super::publish_dense_index(&mut store, &index));
        let base_owner = Arc::clone(&store.documents["d007"]);
        store.insert("d007".into(), document(7, 2));
        let options = SearchOptions {
            k: Some(3),
            candidate_limit: Some(5),
            filter: Some(json!({"group": 1})),
            embedding: Some(vec![1.0, 0.0, 0.0]),
            min_semantic_score: Some(-1.0),
            ..Default::default()
        };
        let captured = capture_search(
            &store,
            vec![prepare_search("common rare7", &options, Some(3)).unwrap()],
        );
        assert!(captured.queries[0].dense.is_some());
        assert!(captured.queries[0].chunks.len() <= 5 + store.documents["d007"].chunks.len());
        if let Some(eligible) = &captured.queries[0].eligible {
            assert!(!eligible.contains(&(Arc::as_ptr(&base_owner) as usize)));
            assert!(eligible.contains(&(Arc::as_ptr(&store.documents["d007"]) as usize)));
            assert!(captured.queries[0].excluded_owners.is_empty());
        } else {
            assert!(captured.queries[0]
                .excluded_owners
                .contains(&(Arc::as_ptr(&base_owner) as usize)));
        }
        let expected =
            super::super::search_impl(&store, "common rare7", &options, Some(3), false).unwrap();
        store.insert("d007".into(), document(7, 3));
        assert_eq!(
            serde_json::to_value(execute_search(&captured, 0)).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }

    #[test]
    fn audit_regression_filtered_hybrid_capture_bounds_owners_and_reuses_exact_index() {
        let mut store = store(512);
        let trained = super::super::build_dense_index(
            super::super::capture_dense_index(&store, 3)
                .unwrap()
                .unwrap(),
        )
        .unwrap()
        .unwrap();
        assert!(super::super::publish_dense_index(&mut store, &trained));
        for filter in [
            None,
            Some(json!({"group": 1})),
            Some(json!({"nested": {"flag": 0}})),
        ] {
            for fusion in ["blend", "rrf"] {
                let options = SearchOptions {
                    k: Some(7),
                    candidate_limit: Some(11),
                    embedding: Some(vec![1.0, 0.0, 0.0]),
                    filter: filter.clone(),
                    fusion: Some(fusion.into()),
                    min_semantic_score: Some(-1.0),
                    ..Default::default()
                };
                let captured = capture_search(
                    &store,
                    vec![prepare_search("common rare7 missing", &options, Some(3)).unwrap()],
                );
                assert!(captured.queries[0].dense.is_some());
                assert!(captured.queries[0].chunks.len() <= 11);
                let expected = super::super::search_impl(
                    &store,
                    "common rare7 missing",
                    &options,
                    Some(3),
                    false,
                )
                .unwrap();
                assert_eq!(
                    serde_json::to_value(execute_search(&captured, 0)).unwrap(),
                    serde_json::to_value(expected).unwrap()
                );
            }
        }
    }

    #[test]
    fn audit_regression_zero_bm25_saturation_ignores_absent_terms() {
        let store = store(128);
        for filter in [None, Some(json!({"group": 1}))] {
            let options = SearchOptions {
                k: Some(7),
                bm25_k1: Some(0.0),
                filter,
                source_uri: Some("source1".into()),
                ..Default::default()
            };
            let actual = super::super::search(&store, "common missing", &options, Some(3)).unwrap();
            let expected = super::super::search(&store, "common", &options, Some(3)).unwrap();
            assert!(!expected.is_empty());
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        }
    }

    #[test]
    fn trained_semantic_capture_pins_one_index_and_only_lexical_posting_owners() {
        let mut store = store(128);
        let trained = super::super::build_dense_index(
            super::super::capture_dense_index(&store, 3)
                .unwrap()
                .unwrap(),
        )
        .unwrap()
        .unwrap();
        assert!(super::super::publish_dense_index(&mut store, &trained));
        let options = SearchOptions {
            embedding: Some(vec![1.0, 0.0, 0.0]),
            lexical_weight: Some(0.0),
            min_semantic_score: Some(-1.0),
            ..Default::default()
        };
        let captured = capture_search(&store, vec![prepare_search("", &options, Some(3)).unwrap()]);
        assert!(captured.queries[0].chunks.is_empty());
        assert!(Arc::ptr_eq(
            captured.queries[0].dense.as_ref().unwrap(),
            store.dense_index.as_ref().unwrap()
        ));
        let expected = serde_json::to_value(
            super::super::search_impl(&store, "", &options, Some(3), false).unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(execute_search(&captured, 0)).unwrap(),
            expected
        );
        let options = SearchOptions {
            lexical_weight: Some(1.0),
            ..options
        };
        let captured = capture_search(
            &store,
            vec![prepare_search("rare7 ghost", &options, Some(3)).unwrap()],
        );
        assert_eq!(captured.queries[0].chunks.len(), 1);
        assert_eq!(captured.queries[0].chunks[0].document.id, "d007");
        assert!(Arc::ptr_eq(
            &captured.queries[0].chunks[0].document,
            &store.documents["d007"]
        ));
        let expected = serde_json::to_value(
            super::super::search_impl(&store, "rare7 ghost", &options, Some(3), false).unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(execute_search(&captured, 0)).unwrap(),
            expected
        );
    }

    fn document(id: usize, revision: usize) -> StoredDocument {
        let input = serde_json::from_value(json!({
            "id": format!("d{id:03}"), "version": revision.to_string(),
            "text": format!("running common rare{id} generation{revision} old semantic"),
            "sourceUri": format!("source{}", id % 3),
            "metadata": {"group": id % 2, "nested": {"flag": id % 2}},
            "chunkWords": 3, "overlapWords": 0,
            "embeddings": [[1.0, id as f64 / 10.0, 0.0], [0.0, 1.0, revision as f64 / 100.0]],
        }))
        .unwrap();
        let mut document =
            super::super::prepare(input, 1024, Some(3), TextEncoder::default()).unwrap();
        for chunk in &mut document.chunks {
            if id.is_multiple_of(2) {
                chunk.terms = terms(chunk.text.as_deref().unwrap(), false).into();
                chunk.stemmed = false;
            }
            chunk.terms.insert("ghost".into(), 0);
        }
        document
    }

    fn store(count: usize) -> DocumentStore {
        let mut store = DocumentStore::default();
        for id in (0..count).rev() {
            let document = document(id, 1);
            store.insert(document.id.clone(), document);
        }
        store
    }

    fn variants() -> Vec<(String, SearchOptions)> {
        let mut variants = Vec::new();
        for fusion in ["blend", "rrf"] {
            for filter in [
                None,
                Some(json!({"group": 1})),
                Some(json!({"nested": {"flag": 0}})),
                Some(json!({"group": 99})),
            ] {
                for embedding in [None, Some(vec![1.0, 0.0, 0.0])] {
                    for limit in [9, 10, 11] {
                        variants.push((
                            "running semantic ghost rare3".into(),
                            SearchOptions {
                                k: Some(9),
                                candidate_limit: Some(limit),
                                fusion: Some(fusion.into()),
                                filter: filter.clone(),
                                embedding: embedding.clone(),
                                min_semantic_score: Some(-1.0),
                                ..Default::default()
                            },
                        ));
                    }
                }
            }
        }
        variants.push((
            "running".into(),
            SearchOptions {
                source_uri: Some("source1".into()),
                document_ids: Some(vec!["d001".into(), "d004".into(), "d001".into()]),
                ..Default::default()
            },
        ));
        variants
    }

    fn capture_variants(
        store: &DocumentStore,
        variants: &[(String, SearchOptions)],
    ) -> CapturedSearch {
        capture_search(
            store,
            variants
                .iter()
                .map(|(text, options)| prepare_search(text, options, Some(3)).unwrap())
                .collect(),
        )
    }

    fn scanned_variants(store: &DocumentStore, variants: &[(String, SearchOptions)]) -> Vec<Value> {
        variants
            .iter()
            .map(|(text, options)| {
                serde_json::to_value(
                    super::super::search_impl(store, text, options, Some(3), false).unwrap(),
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn captured_view_survives_replace_delete_slot_reuse_and_store_drop() {
        let mut store = store(12);
        let variants = variants();
        let expected = scanned_variants(&store, &variants);
        let captured = capture_variants(&store, &variants);
        let old = Arc::downgrade(&store.documents["d000"]);
        let captured_document = captured.queries[0]
            .chunks
            .iter()
            .find(|chunk| chunk.document.id == "d000")
            .unwrap();
        assert!(Arc::ptr_eq(
            &captured_document.document,
            &store.documents["d000"]
        ));
        assert_eq!(
            captured_document
                .chunk()
                .embedding
                .as_ref()
                .unwrap()
                .as_ptr(),
            store.documents["d000"].chunks[captured_document.index]
                .embedding
                .as_ref()
                .unwrap()
                .as_ptr()
        );
        let replacement = document(0, 2);
        store.insert(replacement.id.clone(), replacement);
        assert!(store.remove("d002").is_some());
        let replacement = document(20, 2);
        store.insert(replacement.id.clone(), replacement);
        let ids: Vec<_> = store.documents.keys().cloned().collect();
        for id in ids {
            store.remove(&id);
        }
        assert_eq!(store.slot_count, 0);
        drop(store);
        assert!(old.upgrade().is_some());
        for (i, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                serde_json::to_value(execute_search(&captured, i)).unwrap(),
                expected
            );
        }
        drop(captured);
        assert!(old.upgrade().is_none());
    }

    #[test]
    fn lexical_capture_holds_only_ranked_candidates_and_shares_embeddings() {
        let store = store(120);
        for (text, limit, held) in [("rare7", 100, 1), ("common", 100, 100), ("common", 7, 7)] {
            let options = SearchOptions {
                k: Some(limit),
                candidate_limit: Some(limit),
                ..Default::default()
            };
            let captured = capture_search(
                &store,
                vec![prepare_search(text, &options, Some(3)).unwrap()],
            );
            let query = &captured.queries[0];
            assert_eq!(query.count, 240);
            assert_eq!(query.chunks.len(), held);
            assert_eq!(query.lexical_scores.as_ref().map(Vec::len), Some(held));
            assert!(query.postings.iter().all(BTreeMap::is_empty));
            for chunk in &query.chunks {
                assert!(Arc::ptr_eq(
                    &chunk.document,
                    &store.documents[&chunk.document.id]
                ));
            }
            assert_eq!(
                serde_json::to_value(execute_search(&captured, 0)).unwrap(),
                serde_json::to_value(
                    super::super::search_impl(&store, text, &options, Some(3), false).unwrap()
                )
                .unwrap()
            );
        }
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A document of 1 to 12 words from a small vocabulary (so that scores tie), chunked 1 to
    /// 4 words apiece; a third keep unstemmed terms and a quarter carry a zero-frequency term.
    fn random_document(id: usize, state: &mut u64) -> StoredDocument {
        const WORDS: [&str; 9] = [
            "alpha", "beta", "gamma", "delta", "running", "runs", "ran", "shared", "Shared",
        ];
        let template = id.is_multiple_of(5);
        let mut seed = if template { 7 } else { xorshift(state) };
        let count = 1 + (xorshift(&mut seed) % 12) as usize;
        let text: Vec<&str> = (0..count)
            .map(|_| WORDS[(xorshift(&mut seed) % WORDS.len() as u64) as usize])
            .collect();
        let input = serde_json::from_value(json!({
            "id": format!("d{id:04}"), "text": text.join(" "),
            "chunkWords": 1 + xorshift(state) % 4, "overlapWords": 0,
        }))
        .unwrap();
        let mut document =
            super::super::prepare(input, 1024, None, TextEncoder::default()).unwrap();
        for chunk in &mut document.chunks {
            if id.is_multiple_of(3) {
                chunk.terms = terms(chunk.text.as_deref().unwrap(), false).into();
                chunk.stemmed = false;
            }
            if id % 4 == 1 {
                chunk.terms.insert("ghost".into(), 0);
            }
        }
        document
    }

    /// Lexical hits of the ranked capture and of the exhaustive posting capture, as JSON.
    fn ranked_and_exhaustive(
        store: &DocumentStore,
        text: &str,
        options: &SearchOptions,
    ) -> (Value, Value) {
        let ranked = capture_search(store, vec![prepare_search(text, options, None).unwrap()]);
        assert!(ranked.queries[0].lexical_scores.is_some());
        let exhaustive = CapturedSearch {
            queries: vec![capture_postings(
                store,
                prepare_search(text, options, None).unwrap(),
            )],
        };
        (
            serde_json::to_value(execute_search(&ranked, 0)).unwrap(),
            serde_json::to_value(execute_search(&exhaustive, 0)).unwrap(),
        )
    }

    fn assert_ranked_matches_exhaustive(store: &DocumentStore) {
        let bm25 = [
            (None, None),
            (Some(0.0), Some(0.0)),
            (Some(0.0), Some(1.0)),
            (Some(1.0), Some(1.0)),
            (Some(2.5), Some(0.3)),
        ];
        for text in [
            "gamma",
            "alpha beta",
            "ran shared ghost",
            "ghost",
            "running runs alpha alpha delta",
            "missing",
            "",
        ] {
            for (k1, b) in bm25 {
                let all = SearchOptions {
                    k: Some(1000),
                    candidate_limit: Some(4096),
                    bm25_k1: k1,
                    bm25_b: b,
                    ..Default::default()
                };
                let (ranked, exhaustive) = ranked_and_exhaustive(store, text, &all);
                assert_eq!(ranked, exhaustive, "{text:?} k1 {k1:?} b {b:?}");
                let matched = exhaustive.as_array().unwrap().len() as u32;
                for limit in [1, 2, 5, matched.saturating_sub(1), matched, matched + 1] {
                    let limit = limit.max(1);
                    for (k, fusion) in [(limit.min(1000), "blend"), (limit.min(3), "rrf")] {
                        let options = SearchOptions {
                            k: Some(k),
                            candidate_limit: Some(limit),
                            fusion: Some(fusion.into()),
                            ..all.clone()
                        };
                        let (ranked, exhaustive) = ranked_and_exhaustive(store, text, &options);
                        assert_eq!(ranked, exhaustive, "{text:?} limit {limit} {fusion}");
                    }
                }
            }
        }
    }

    /// The ranked lexical capture returns exactly what the exhaustive posting capture returns
    /// (hits, order, bit-identical scores) through ties, zero-frequency terms, mixed
    /// tokenization, multi-chunk documents, deletes, replacements with slot reuse, a registry
    /// rebuild, stores of equal generation, BM25 constants at their bounds, limits around the
    /// match count, and the scratch's recovery and epoch wrap.
    #[test]
    fn ranked_lexical_capture_matches_the_exhaustive_capture() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut store = DocumentStore::default();
        for id in 0..60 {
            let document = random_document(id, &mut state);
            store.insert(document.id.clone(), document);
        }
        assert_ranked_matches_exhaustive(&store);
        for id in (0..60).step_by(3) {
            assert!(store.remove(&format!("d{id:04}")).is_some());
        }
        assert_ranked_matches_exhaustive(&store);
        // Replacements: same chunk count (in place) and a different one (fresh slots), then
        // new documents in the freed slots.
        for id in [1, 2, 4, 5, 7] {
            let mut document = random_document(id + 100, &mut state);
            document.id = format!("d{id:04}");
            for (index, chunk) in document.chunks.iter_mut().enumerate() {
                chunk.id = format!("{}{}:d{id:04}:{index}", super::super::CHUNK_PREFIX, 5);
            }
            store.insert(document.id.clone(), document);
        }
        for id in 60..80 {
            let document = random_document(id, &mut state);
            store.insert(document.id.clone(), document);
        }
        assert_ranked_matches_exhaustive(&store);

        // A query that panicked midway leaves the scratch dirty; the next query resets it.
        LEXICAL_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.scores.fill(3.5);
            scratch.dirty = true;
        });
        assert_ranked_matches_exhaustive(&store);
        // The norm epoch wraps (every change of store state or BM25 constants advances it):
        // norms of earlier states must not survive the wrap.
        LEXICAL_SCRATCH.with(|scratch| scratch.borrow_mut().epoch = u32::MAX - 1);
        let document = random_document(80, &mut state);
        store.insert(document.id.clone(), document);
        assert_ranked_matches_exhaustive(&store);
        assert!(LEXICAL_SCRATCH.with(|scratch| scratch.borrow().epoch) < 1000);
        assert!(store.remove("d0080").is_some());
        assert_ranked_matches_exhaustive(&store);

        // Two stores at the same generation, queried alternately, keep their own norms.
        let mut other = DocumentStore::default();
        let mut long_state = 0x9E37_79B9_7F4A_7C15u64;
        while other.generation < store.generation {
            let id = other.generation as usize + 1000;
            let mut document = random_document(id, &mut long_state);
            for chunk in &mut document.chunks {
                chunk.word_count += 7;
            }
            other.insert(document.id.clone(), document);
        }
        assert_eq!(other.generation, store.generation);
        for _ in 0..2 {
            assert_ranked_matches_exhaustive(&other);
            assert_ranked_matches_exhaustive(&store);
        }

        // Deleting most of a large store rebuilds its registry (new slots, new identity).
        let mut large = DocumentStore::default();
        for id in 0..1100 {
            let mut document = random_document(id, &mut state);
            document.chunks.truncate(1);
            large.insert(document.id.clone(), document);
        }
        assert_ranked_matches_exhaustive(&large);
        let identity = large.identity;
        for id in 0..700 {
            large.remove(&format!("d{id:04}"));
        }
        assert_ne!(large.identity, identity, "registry rebuilt");
        assert_ranked_matches_exhaustive(&large);
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(32))]
        /// The ranked capture returns what the exhaustive capture returns, bit for bit, and
        /// counts agree with the chunks, at limits far below the posting count, before and
        /// after deletes, replacements and slot reuse.
        #[test]
        fn ranking_and_counts_match_exhaustive_under_churn(
            seed in proptest::prelude::any::<u64>(),
            documents in 20usize..90,
            limit in 1u32..9,
            churn in 2usize..9,
        ) {
            let mut state = seed | 1;
            let mut store = DocumentStore::default();
            for id in 0..documents {
                let document = random_document(id, &mut state);
                store.insert(document.id.clone(), document);
            }
            let check = |store: &DocumentStore| {
                for text in ["gamma", "alpha beta", "ran shared ghost", "running runs alpha delta"] {
                    let options = SearchOptions {
                        k: Some(limit.min(3)),
                        candidate_limit: Some(limit),
                        ..Default::default()
                    };
                    let (ranked, exhaustive) = ranked_and_exhaustive(store, text, &options);
                    assert_eq!(ranked, exhaustive, "{text:?} limit {limit}");
                    let matched = matching_documents(store, text);
                    assert_eq!(count_matches(store, text).unwrap(), matched, "{text:?}");
                }
            };
            check(&store);
            for id in (0..documents).step_by(churn) {
                store.remove(&format!("d{id:04}"));
            }
            check(&store);
            for id in (1..documents).step_by(churn + 1) {
                let mut document = random_document(id + documents, &mut state);
                document.id = format!("d{id:04}");
                for (index, chunk) in document.chunks.iter_mut().enumerate() {
                    chunk.id = format!("{}{}:d{id:04}:{index}", super::super::CHUNK_PREFIX, 5);
                }
                store.insert(document.id.clone(), document);
            }
            check(&store);
        }
    }

    /// A checkpoint of a store with zero-frequency terms, free slots and both posting forms
    /// ranks from its mapping, and rebuilds into a store whose block bounds are a fresh
    /// build's, exactly as the store that wrote it ranks.
    #[test]
    fn checkpoint_ranks_and_rebuilds_as_the_store_that_wrote_it() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut store = DocumentStore::default();
        for id in 0..300 {
            let mut document = random_document(id, &mut state);
            for chunk in &mut document.chunks {
                chunk.vector_state = super::super::VectorState::Lexical;
                chunk.vector =
                    crate::core::entangled::EntangledHVec::from_indices(Vec::new(), 1024);
            }
            store.insert(document.id.clone(), document);
        }
        for id in (0..300).step_by(7) {
            store.remove(&format!("d{id:04}"));
        }
        let bytes = super::super::image::encode(&store, 1, 1024)
            .unwrap()
            .unwrap();
        let image = ServingImage::new(super::super::Backing::Owned(bytes)).unwrap();
        let rebuilt = image.hydrate(None).unwrap();
        let words: Vec<u32> = (0..rebuilt.slot_count)
            .map(|slot| rebuilt.slot_word_count(slot))
            .collect();
        for postings in &rebuilt.postings {
            postings.assert_valid();
        }
        for text in [
            "gamma",
            "alpha beta",
            "ran shared ghost",
            "running runs alpha delta",
        ] {
            for (form, terms) in [terms(text, false), terms(text, true)].iter().enumerate() {
                for word in terms.keys() {
                    if let Some(list) = rebuilt.postings[form].get(word) {
                        assert!(list.bounds_are_tight(&words), "{word} bounds");
                    }
                }
            }
            let options = SearchOptions {
                k: Some(10),
                ..Default::default()
            };
            let expected = ranked_and_exhaustive(&store, text, &options).0;
            assert_eq!(ranked_and_exhaustive(&rebuilt, text, &options).0, expected);
            let mapped = CapturedSearch {
                queries: vec![
                    capture_image(&image, prepare_search(text, &options, None).unwrap()).unwrap(),
                ],
            };
            assert_eq!(
                serde_json::to_value(execute_search(&mapped, 0)).unwrap(),
                expected,
                "{text:?}"
            );
        }
    }

    /// Rebuilding a checkpoint without the postings of its dead documents and then removing
    /// them leaves the store byte for byte as rebuilding everything and removing them does,
    /// before and after more writes, below and above the slot count at which a removal re-slots
    /// the whole store.
    #[test]
    fn hydrating_without_dead_postings_matches_removing_them() {
        for (documents, dead_every, keep_every) in [(300, 7, 1), (1500, 1, 3)] {
            let mut state = 0x2545_f491_4f6c_dd1du64;
            let mut store = DocumentStore::default();
            for id in 0..documents {
                let mut document = random_document(id, &mut state);
                for chunk in &mut document.chunks {
                    chunk.vector_state = super::super::VectorState::Lexical;
                    chunk.vector =
                        crate::core::entangled::EntangledHVec::from_indices(Vec::new(), 1024);
                }
                store.insert(document.id.clone(), document);
            }
            let bytes = super::super::image::encode(&store, 1, 1024)
                .unwrap()
                .unwrap();
            let image = ServingImage::new(super::super::Backing::Owned(bytes)).unwrap();
            // Every `dead_every`-th document, sparing every `keep_every`-th of those, in id order
            // as the overlay lists them.
            let dead: Vec<String> = (0..documents)
                .step_by(dead_every)
                .enumerate()
                .filter(|(i, _)| keep_every == 1 || i % keep_every != 0)
                .map(|(_, id)| format!("d{id:04}"))
                .collect();
            let mut removed = image.hydrate(None).unwrap();
            for id in &dead {
                removed.remove_scrubbed(id).unwrap();
            }
            let mut excluded = image.hydrate_excluding(None, &dead).unwrap();
            for id in &dead {
                excluded.remove_unindexed_scrubbed(id).unwrap();
            }
            let encode = |store: &DocumentStore| {
                super::super::image::encode(store, 1, 1024)
                    .unwrap()
                    .expect("lexical-only store")
            };
            assert_eq!(encode(&excluded), encode(&removed));
            for id in documents..documents + 40 {
                let document = random_document(id, &mut state);
                for store in [&mut removed, &mut excluded] {
                    let mut document = document.clone();
                    for chunk in &mut document.chunks {
                        chunk.vector_state = super::super::VectorState::Lexical;
                        chunk.vector =
                            crate::core::entangled::EntangledHVec::from_indices(Vec::new(), 1024);
                    }
                    store.insert(document.id.clone(), document);
                }
            }
            for postings in &excluded.postings {
                postings.assert_valid();
            }
            assert_eq!(encode(&excluded), encode(&removed));
        }
    }

    /// Documents with a chunk holding a term of `text` at a positive frequency, read from the
    /// chunks themselves.
    fn matching_documents(store: &DocumentStore, text: &str) -> usize {
        let terms = [terms(text, false), terms(text, true)];
        store
            .documents
            .values()
            .filter(|document| {
                document.chunks.iter().any(|chunk| {
                    terms[usize::from(chunk.stemmed)]
                        .keys()
                        .any(|word| chunk.terms.get(word).is_some_and(|&f| f > 0))
                })
            })
            .count()
    }

    #[test]
    fn captured_view_survives_registry_rebuild_without_payload_copy() {
        for count in [1023, 1024, 1025] {
            let mut store = DocumentStore::default();
            for id in 0..count {
                let mut document = document(id, 1);
                document.chunks.truncate(1);
                store.insert(document.id.clone(), document);
            }
            let text = format!("rare{}", count - 1);
            let options = SearchOptions::default();
            let captured = capture_search(
                &store,
                vec![prepare_search(&text, &options, Some(3)).unwrap()],
            );
            let expected = serde_json::to_value(execute_search(&captured, 0)).unwrap();
            for id in 0..count - 1 {
                store.remove(&format!("d{id:03}"));
            }
            assert!(store.slot_count <= 1024);
            assert!(Arc::ptr_eq(
                &captured.queries[0].chunks[0].document,
                &store.documents[&format!("d{:03}", count - 1)]
            ));
            assert_eq!(
                serde_json::to_value(execute_search(&captured, 0)).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn captured_variants_remain_coherent_during_concurrent_publication() {
        for (count, indexed) in [(16, false), (128, true)] {
            let mut store = store(count);
            if indexed {
                let trained = super::super::build_dense_index(
                    super::super::capture_dense_index(&store, 3)
                        .unwrap()
                        .unwrap(),
                )
                .unwrap()
                .unwrap();
                assert!(super::super::publish_dense_index(&mut store, &trained));
            }
            let store = RwLock::new(store);
            let variants = variants();
            let expected = scanned_variants(&store.read(), &variants);
            let captured = capture_variants(&store.read(), &variants);
            let barrier = Barrier::new(2);
            std::thread::scope(|scope| {
                let writer = scope.spawn(|| {
                    barrier.wait();
                    for revision in 2..=9 {
                        for id in 0..count {
                            let document = document(id, revision);
                            let mut store = store.write();
                            store.remove(&document.id);
                            store.insert(document.id.clone(), document);
                        }
                    }
                });
                barrier.wait();
                for _ in 0..8 {
                    for (i, expected) in expected.iter().enumerate() {
                        assert_eq!(
                            serde_json::to_value(execute_search(&captured, i)).unwrap(),
                            *expected
                        );
                    }
                }
                writer.join().unwrap();
            });
            let current = capture_search(
                &store.read(),
                vec![prepare_search("generation9", &SearchOptions::default(), Some(3)).unwrap()],
            );
            assert!(execute_search(&current, 0)
                .iter()
                .all(|result| result.version == "9"));
            assert!(execute_search(&captured, 0)
                .iter()
                .all(|result| result.version == "1"));
            assert_eq!(
                captured.queries.iter().any(|query| query.dense.is_some()),
                indexed
            );
        }
    }

    #[test]
    fn combined_weight_boundary_is_finite_and_overflow_is_rejected() {
        let half = f64::MAX / 2.0;
        let below = f64::from_bits(half.to_bits() - 1);
        let above = f64::from_bits(half.to_bits() + 1);
        let store = store(12);
        for lexical_weight in [below, half] {
            let options = SearchOptions {
                lexical_weight: Some(lexical_weight),
                semantic_weight: Some(half),
                embedding: Some(vec![1.0, 0.0, 0.0]),
                ..Default::default()
            };
            let captured = capture_search(
                &store,
                vec![prepare_search("running", &options, Some(3)).unwrap()],
            );
            assert!(execute_search(&captured, 0)
                .iter()
                .all(|result| result.score.is_finite()));
        }
        for (lexical_weight, semantic_weight) in [(above, half), (f64::MAX, f64::MAX)] {
            let options = SearchOptions {
                lexical_weight: Some(lexical_weight),
                semantic_weight: Some(semantic_weight),
                ..Default::default()
            };
            assert!(prepare_search("running", &options, Some(3)).is_err());
        }
        assert!(capture_search(&store, Vec::new()).queries.is_empty());
    }
}

/// Lexical latency microbenchmark on an NFCorpus-shaped synthetic store (3633 one-chunk
/// documents, Zipf vocabulary, 324 queries); run ignored with `--nocapture`, and set
/// `WARM_LEX_BENCH_OUT` to write the phase timings as JSON.
#[cfg(test)]
mod latency_bench {
    use super::*;
    use crate::core::encoding::TextEncoder;
    use serde_json::json;
    use std::time::Instant;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn zipf_table(size: usize, exponent: f64) -> Vec<f64> {
        let mut cumulative = Vec::with_capacity(size);
        let mut total = 0.0;
        for i in 1..=size {
            total += 1.0 / (i as f64).powf(exponent);
            cumulative.push(total);
        }
        for value in &mut cumulative {
            *value /= total;
        }
        cumulative
    }

    fn sample(table: &[f64], rng: &mut Rng) -> usize {
        let u = rng.unit();
        table.partition_point(|&c| c < u).min(table.len() - 1)
    }

    fn word(i: usize) -> String {
        format!("w{i}x")
    }

    fn percentile(sorted: &[u64], p: f64) -> u64 {
        sorted[((sorted.len() - 1) as f64 * p).round() as usize]
    }

    fn zipf_store(documents: usize, vocabulary: usize, rng: &mut Rng) -> DocumentStore {
        let body = zipf_table(vocabulary, 0.9);
        let mut store = DocumentStore::default();
        for id in 0..documents {
            let words: Vec<String> = (0..200).map(|_| word(sample(&body, rng))).collect();
            let input = serde_json::from_value(json!({
                "id": format!("MED-{id}"), "text": words.join(" "), "storeText": false,
                "metadata": {"title": format!("title {id}"), "year": 2000 + id % 20},
                "chunkWords": 4096, "overlapWords": 0,
            }))
            .unwrap();
            let document =
                super::super::prepare(input, 1024, None, TextEncoder::default()).unwrap();
            store.insert(document.id.clone(), document);
        }
        store
    }

    fn zipf_queries(count: usize, vocabulary: usize, rng: &mut Rng) -> Vec<String> {
        let table = zipf_table(vocabulary, 1.0);
        (0..count)
            .map(|_| {
                let n = 3 + (rng.next() % 5) as usize;
                (0..n)
                    .map(|_| word(sample(&table, rng)))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    #[test]
    #[ignore]
    fn lexical_latency_microbench() {
        let documents = 3633;
        let vocabulary = 25_000;
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (store, queries) = match crate::compatibility::var("WARM_LEX_BENCH_INPUT") {
            Ok(path) => {
                let input: Value =
                    serde_json::from_slice(&std::fs::read(path).expect("bench input")).unwrap();
                let mut store = DocumentStore::default();
                for row in input["documents"].as_array().unwrap() {
                    let input = serde_json::from_value(json!({
                        "id": row["id"], "text": row["text"], "storeText": false,
                        "chunkWords": 4096, "overlapWords": 0,
                    }))
                    .unwrap();
                    let document =
                        super::super::prepare(input, 1024, None, TextEncoder::default()).unwrap();
                    store.insert(document.id.clone(), document);
                }
                let queries = input["queries"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|q| q["text"].as_str().unwrap().to_owned())
                    .collect();
                (store, queries)
            }
            Err(_) => (
                zipf_store(documents, vocabulary, &mut rng),
                zipf_queries(324, vocabulary, &mut rng),
            ),
        };
        {
            let k: u32 = crate::compatibility::var("WARM_LEX_BENCH_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100);
            let options = SearchOptions {
                k: Some(k),
                candidate_limit: Some(k.max(100)),
                lexical_weight: Some(1.0),
                semantic_weight: Some(0.0),
                ..Default::default()
            };
            let passes = 8;
            let mut prepare_ns = Vec::new();
            let mut capture_ns = Vec::new();
            let mut execute_ns = Vec::new();
            let mut total_ns = Vec::new();
            let mut postings = 0usize;
            let mut results = 0usize;
            for pass in 0..passes {
                for text in &queries {
                    let t0 = Instant::now();
                    let prepared = prepare_search(text, &options, None).unwrap();
                    let t1 = Instant::now();
                    let captured = capture_search(&store, vec![prepared]);
                    let t2 = Instant::now();
                    let hits = execute_search(&captured, 0);
                    let t3 = Instant::now();
                    if pass == 0 {
                        let query = &captured.queries[0].prepared;
                        postings += (0..2)
                            .map(|form| {
                                query.query_terms[form]
                                    .iter()
                                    .map(|word| {
                                        store.postings[form].get(word).map_or(0, PostingList::len)
                                    })
                                    .sum::<usize>()
                            })
                            .sum::<usize>();
                        results += hits.len();
                    }
                    prepare_ns.push((t1 - t0).as_nanos() as u64);
                    capture_ns.push((t2 - t1).as_nanos() as u64);
                    execute_ns.push((t3 - t2).as_nanos() as u64);
                    total_ns.push((t3 - t0).as_nanos() as u64);
                }
            }
            let summary = |name: &str, values: &mut Vec<u64>| {
                values.sort_unstable();
                let mean = values.iter().sum::<u64>() / values.len() as u64;
                json!({"phase": name, "mean_ns": mean, "p50_ns": percentile(values, 0.5),
                   "p99_ns": percentile(values, 0.99)})
            };
            let report = json!({
                "store_chunks": store.chunk_count,
                "documents": documents, "vocabulary": vocabulary, "queries": queries.len(),
                "passes": passes, "postings_per_query": postings as f64 / queries.len() as f64,
                "results_per_query": results as f64 / queries.len() as f64,
                "phases": [summary("prepare", &mut prepare_ns), summary("capture", &mut capture_ns),
                           summary("execute", &mut execute_ns), summary("total", &mut total_ns)],
            });
            println!("{report}");
            if let Ok(path) = crate::compatibility::var("WARM_LEX_BENCH_OUT") {
                std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
            }
        }
    }
}
