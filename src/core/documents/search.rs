// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::postings::PostingList;
use std::{collections::BTreeMap, sync::Arc};

use anyhow::{ensure, Result};
use fxhash::FxHashMap;
use serde_json::Value;

use super::dense::DenseIndex;
use super::{
    matches_document, normalize_embedding, terms, top_ranked, Bm25, DocumentResult, DocumentStore,
    SearchOptions, StoredChunk, StoredDocument, MAX_DOCUMENT_BYTES,
};

pub(crate) struct PreparedSearch {
    filters: SearchOptions,
    query_terms: [BTreeMap<String, u32>; 2],
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
    Sparse(FxHashMap<usize, u32>),
}

const DENSE_SLOT_LIMIT: usize = 1 << 19;

impl SlotPositions {
    fn new(slot_count: usize) -> Self {
        if slot_count <= DENSE_SLOT_LIMIT {
            Self::Dense(vec![u32::MAX; slot_count])
        } else {
            Self::Sparse(FxHashMap::default())
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
    dfs: Option<[BTreeMap<String, usize>; 2]>,
    /// The store's text segment, read for result text after the store lock is released.
    text: Option<Arc<super::text::TextSegment>>,
}

pub(super) fn prepare_search(
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
    Ok(PreparedSearch {
        filters: SearchOptions {
            filter: options.filter.clone(),
            source_uri: options.source_uri.clone(),
            document_ids: options.document_ids.clone(),
            ..Default::default()
        },
        query_terms: [terms(text, false), terms(text, true)],
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

pub(super) fn capture_search(
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
    if prepared.unfiltered && prepared.semantic_weight > 0.0 && store.dense_index_trained() {
        if let Some(index) = store.dense_index.as_ref().filter(|index| {
            prepared
                .embedding
                .as_ref()
                .is_some_and(|query| query.len() == index.dimensions)
        }) {
            return capture_indexed_query(store, prepared, Arc::clone(index));
        }
    }
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
            .filter(|document| matches_document(document, &prepared.filters))
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
                    .keys()
                    .map(|word| store.postings[form].get(word).map_or(0, PostingList::len))
                    .sum::<usize>()
            })
            .sum();
        let scan_work = chunks
            .len()
            .saturating_mul(prepared.query_terms.iter().map(BTreeMap::len).sum());
        (
            chunks.len(),
            words,
            posting_work.saturating_add(store.slot_count) < scan_work,
        )
    };
    let mut postings: [BTreeMap<String, Vec<CapturedPosting>>; 2] = Default::default();
    if use_postings && (prepared.lexical_weight > 0.0 || !prepared.lexical_only) {
        for (form, terms) in prepared.query_terms.iter().enumerate() {
            for word in terms.keys() {
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
        dfs: None,
        text: store.text_segment(),
    }
}

fn capture_indexed_query(
    store: &DocumentStore,
    prepared: PreparedSearch,
    dense: Arc<DenseIndex>,
) -> CapturedQuery {
    let mut chunks = Vec::new();
    let mut positions = SlotPositions::new(store.slot_count);
    let dfs = std::array::from_fn(|form| {
        prepared.query_terms[form]
            .keys()
            .map(|word| {
                let source = store.postings[form].get(word);
                if prepared.lexical_weight > 0.0 {
                    if let Some(source) = source {
                        source.for_each(|slot, frequency| {
                            if frequency > 0 && positions.get(slot).is_none() {
                                let (document, index) = store.owned_chunk(slot);
                                let position = chunks.len() as u32;
                                chunks.push(CapturedChunk {
                                    document,
                                    index,
                                    word_count: store.slot_word_count(slot),
                                    semantic_cache: None,
                                });
                                positions.insert(slot, position);
                            }
                        });
                    }
                }
                (word.clone(), source.map_or(0, PostingList::len))
            })
            .collect()
    });
    CapturedQuery {
        prepared,
        chunks,
        postings: Default::default(),
        count: store.chunk_count,
        words: store.word_count,
        use_postings: false,
        dense: Some(dense),
        dfs: Some(dfs),
        text: store.text_segment(),
    }
}

pub(super) fn execute_search(captured: &CapturedSearch, query: usize) -> Vec<DocumentResult> {
    execute_query(&captured.queries[query])
}

fn execute_query(captured: &CapturedQuery) -> Vec<DocumentResult> {
    if let Some(index) = &captured.dense {
        let query = captured
            .prepared
            .embedding
            .as_ref()
            .expect("indexed search lost its query embedding");
        let (hits, _) = index.search(query, captured.prepared.limit, captured.prepared.minimum);
        let mut chunks = captured.chunks.clone();
        let mut positions: FxHashMap<_, _> = captured
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

fn execute_chunks(captured: &CapturedQuery, chunks: &[CapturedChunk]) -> Vec<DocumentResult> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let prepared = &captured.prepared;
    let n = captured.count as f64;
    let average = (captured.words as f64 / n).max(1.0);
    let idf: [BTreeMap<&String, f64>; 2] = std::array::from_fn(|form| {
        prepared.query_terms[form]
            .keys()
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
                (word, (1.0 + (n - df + 0.5) / (df + 0.5)).ln())
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
            let semantic = captured_chunk.semantic_cache.or_else(|| {
                prepared.embedding.as_ref().and_then(|q| {
                    captured_chunk.chunk().embedding.as_ref().map(|values| {
                        q.iter()
                            .zip(values)
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            .clamp(-1.0, 1.0)
                    })
                })
            });
            (lexical, semantic)
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
    let mut fused = BTreeMap::<usize, f64>::new();
    for (indices, weight) in [
        (&lexical, prepared.lexical_weight),
        (&semantic, prepared.semantic_weight),
    ] {
        if prepared.rrf {
            for (rank, &(_, i)) in indices.iter().enumerate() {
                *fused.entry(i).or_default() += weight / (60.0 + rank as f64 + 1.0);
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
            *fused.entry(i).or_default() += weight * normalized;
        }
    }
    let mut ranked: Vec<_> = fused.into_iter().collect();
    top_ranked(&mut ranked, prepared.k, |a, b| {
        b.1.total_cmp(&a.1).then_with(|| order(a.0, b.0))
    });
    ranked
        .into_iter()
        .map(|(i, score)| {
            let document = &chunks[i].document;
            let chunk = chunks[i].chunk();
            DocumentResult {
                id: chunk.id.clone(),
                document_id: document.id.clone(),
                source_uri: document.source_uri.clone(),
                version: document.version.clone(),
                metadata: document.metadata.clone(),
                text: super::index::text_of(chunk, captured.text.as_deref()),
                start_byte: chunk.start_byte,
                end_byte: chunk.end_byte,
                score,
                lexical_score: scores[i].0,
                semantic_score: scores[i].1,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::encoding::TextEncoder;
    use parking_lot::RwLock;
    use serde_json::json;
    use std::sync::Barrier;

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
    fn lexical_capture_is_limited_to_posting_candidates_and_shares_embeddings() {
        let store = store(120);
        let options = SearchOptions::default();
        let captured = capture_search(
            &store,
            vec![prepare_search("rare7", &options, Some(3)).unwrap()],
        );
        assert_eq!(captured.queries[0].count, 240);
        assert_eq!(captured.queries[0].chunks.len(), 1);
        assert!(captured.queries[0].postings[0].is_empty());
        assert_eq!(captured.queries[0].postings[1].len(), 1);
        assert!(Arc::ptr_eq(
            &captured.queries[0].chunks[0].document,
            &store.documents["d007"]
        ));
        assert_eq!(
            serde_json::to_value(execute_search(&captured, 0)).unwrap(),
            serde_json::to_value(
                super::super::search_impl(&store, "rare7", &options, Some(3), false).unwrap()
            )
            .unwrap()
        );
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
/// `HMS_LEX_BENCH_OUT` to write the phase timings as JSON.
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
        let (store, queries) = match std::env::var("HMS_LEX_BENCH_INPUT") {
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
            let k: u32 = std::env::var("HMS_LEX_BENCH_K")
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
                        postings += captured.queries[0]
                            .postings
                            .iter()
                            .map(|form| form.values().map(Vec::len).sum::<usize>())
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
            if let Ok(path) = std::env::var("HMS_LEX_BENCH_OUT") {
                std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
            }
        }
    }
}
