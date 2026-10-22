// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::core::documents::postings::PostingCursor;
use std::collections::BTreeMap;

struct TermState<'a> {
    cursor: PostingCursor<'a>,
    df: usize,
    live_count: usize,
}

pub(super) struct TopK {
    values: Vec<(f64, usize)>,
    k: usize,
    floor: f64,
}

impl TopK {
    pub(super) fn new(k: usize) -> Self {
        Self {
            values: Vec::new(),
            k,
            floor: 0.0,
        }
    }
    pub(super) fn floor(&self) -> f64 {
        self.floor
    }
    fn trim(&mut self, store: &DocumentStore) {
        super::super::top_ranked(&mut self.values, self.k, |a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| store.slot_order(a.1).cmp(&store.slot_order(b.1)))
        });
        if self.values.len() == self.k {
            self.floor = self.values.last().expect("nonempty top-k").0;
        }
    }
    pub(super) fn consider(&mut self, store: &DocumentStore, score: f64, slot: usize) {
        if self.k == 0 || score < self.floor {
            return;
        }
        self.values.push((score, slot));
        if self.values.len() == self.k * 2 {
            self.trim(store);
        }
    }
    pub(super) fn finish(mut self, store: &DocumentStore) -> Vec<LexicalHit> {
        if self.k == 0 {
            return Vec::new();
        }
        self.trim(store);
        self.values
            .into_iter()
            .map(|(score, slot)| {
                let (document, chunk_index) = store.slot_order(slot);
                LexicalHit {
                    document_id: document.to_owned(),
                    chunk_index,
                    score,
                }
            })
            .collect()
    }
}

fn conjunction(states: &mut [TermState<'_>], required: &[usize]) -> Option<usize> {
    let (&lead, others) = required.split_first().expect("nonempty conjunction");
    'align: loop {
        let target = states[lead].cursor.peek()?.0;
        for &index in others {
            let found = states[index].cursor.advance(target)?.0;
            if found > target {
                states[lead].cursor.advance(found)?;
                continue 'align;
            }
        }
        return Some(target);
    }
}

pub(super) fn document_at_a_time(
    store: &DocumentStore,
    query: &LexicalQuery,
    options: &LexicalOptions,
    bm25: Bm25,
) -> Result<LexicalResults> {
    let mut top = TopK::new(options.k);
    let mut count = 0;
    let mut seen = if options.count && store.chunk_count != store.documents.len() {
        vec![0u64; store.slot_count.div_ceil(64)]
    } else {
        Vec::new()
    };
    if !query.has_positive() {
        return Ok(LexicalResults {
            hits: Vec::new(),
            count: options.count.then_some(0),
        });
    }
    let idfs = if options.k > 0 {
        clause_idfs(store, query)
    } else {
        Vec::new()
    };
    let average = (store.word_count as f64 / (store.chunk_count as f64).max(1.0)).max(1.0);
    let need_document = query.clauses.iter().any(|c| c.phrase)
        || (options.count && !store.multichunk_documents.is_empty());
    let single_phrase = query.clauses.len() == 1 && query.clauses[0].phrase;
    for form in 0..2 {
        let mut by_word = BTreeMap::new();
        let mut states = Vec::new();
        for clause in &query.clauses {
            for word in &clause.words[form] {
                if by_word.contains_key(word.as_str()) {
                    continue;
                }
                let index = store.postings[form].get(word).map(|list| {
                    let index = states.len();
                    states.push(TermState {
                        cursor: list.iter(),
                        df: list.len(),
                        live_count: list.positive_len(),
                    });
                    index
                });
                by_word.insert(word.as_str(), index);
            }
        }
        let indices: Vec<Vec<Option<usize>>> = query
            .clauses
            .iter()
            .map(|clause| {
                clause.words[form]
                    .iter()
                    .map(|w| by_word[w.as_str()])
                    .collect()
            })
            .collect();
        let mut required = Vec::new();
        let mut positive = BTreeSet::new();
        let mut missing_required = false;
        for (clause, indices) in query.clauses.iter().zip(&indices) {
            for &index in indices {
                if clause.occur == Occur::Must || single_phrase {
                    match index {
                        Some(index) => required.push(index),
                        None => missing_required = true,
                    }
                }
                if clause.occur != Occur::MustNot {
                    if let Some(index) = index {
                        positive.insert(index);
                    }
                }
            }
        }
        if missing_required {
            continue;
        }
        required.sort_unstable();
        required.dedup();
        required.sort_unstable_by_key(|&i| states[i].df);
        let single_required_term = !single_phrase
            && required.len() == 1
            && query
                .clauses
                .iter()
                .filter(|c| c.occur == Occur::Must)
                .all(|c| !c.phrase)
            && query.clauses.iter().all(|c| c.occur != Occur::MustNot);
        if options.k == 0
            && options.count
            && store.chunk_count == store.documents.len()
            && (single_required_term
                || (query.clauses.len() == 1
                    && !query.clauses[0].phrase
                    && query.clauses[0].occur != Occur::MustNot))
        {
            count += if single_required_term {
                states[required[0]].live_count
            } else {
                positive
                    .iter()
                    .map(|&i| states[i].live_count)
                    .sum::<usize>()
            };
            continue;
        }
        let mut frequencies = vec![0; query.clauses.len()];
        loop {
            let candidate = if required.is_empty() {
                positive
                    .iter()
                    .filter_map(|&i| states[i].cursor.peek().map(|p| p.0))
                    .min()
            } else {
                conjunction(&mut states, &required)
            };
            let Some(slot) = candidate else {
                break;
            };
            let document_chunk = need_document.then(|| store.borrowed_chunk(slot));
            for ((clause, indices), frequency_value) in
                query.clauses.iter().zip(&indices).zip(&mut frequencies)
            {
                if options.k == 0
                    && !required.is_empty()
                    && !single_phrase
                    && clause.occur == Occur::Should
                {
                    *frequency_value = 0;
                    continue;
                }
                if clause.phrase {
                    let possible = indices.iter().all(|&i| {
                        i.is_some_and(|i| {
                            states[i]
                                .cursor
                                .advance(slot)
                                .is_some_and(|p| p.0 == slot && p.1 > 0)
                        })
                    });
                    *frequency_value = if possible {
                        let (document, chunk_index) = document_chunk.expect("phrase owner");
                        frequency(clause, &document.chunks[chunk_index])
                    } else {
                        0
                    };
                } else {
                    *frequency_value = indices[0]
                        .and_then(|i| states[i].cursor.advance(slot))
                        .filter(|p| p.0 == slot)
                        .map_or(0, |p| p.1);
                }
            }
            if matches(query, &frequencies) {
                if options.count {
                    if let Some((document, _)) = document_chunk.filter(|(d, _)| d.chunks.len() > 1)
                    {
                        let first = store.slots[&document.id][0];
                        let bit = 1 << (first % 64);
                        if seen[first / 64] & bit == 0 {
                            count += 1;
                            seen[first / 64] |= bit;
                        }
                    } else {
                        count += 1;
                    }
                }
                if options.k > 0 {
                    top.consider(
                        store,
                        score(
                            query,
                            &frequencies,
                            &idfs,
                            form,
                            bm25.norm(f64::from(store.slot_word_count(slot)), average),
                            bm25,
                        )?,
                        slot,
                    );
                }
            }
            if let Some(&lead) = required.first() {
                states[lead].cursor.next();
            } else {
                for &index in &positive {
                    if states[index].cursor.peek().is_some_and(|p| p.0 == slot) {
                        states[index].cursor.next();
                    }
                }
            }
        }
    }
    Ok(LexicalResults {
        hits: top.finish(store),
        count: options.count.then_some(count),
    })
}

pub(super) fn count_union(store: &DocumentStore, query: &LexicalQuery) -> usize {
    if query.clauses.len() == 1 && store.chunk_count == store.documents.len() {
        return (0..2)
            .map(|form| {
                store.postings[form]
                    .get(&query.clauses[0].words[form][0])
                    .map_or(0, |l| l.positive_len())
            })
            .sum();
    }
    let mut matched = vec![0u64; store.slot_count.div_ceil(64)];
    for form in 0..2 {
        let words: BTreeSet<_> = query.clauses.iter().map(|c| &c.words[form][0]).collect();
        for word in words {
            if let Some(list) = store.postings[form].get(word) {
                list.mark_slots(&mut matched);
            }
        }
    }
    let mut count: usize = matched.iter().map(|word| word.count_ones() as usize).sum();
    for id in &store.multichunk_documents {
        let chunks = store.slots[id]
            .iter()
            .filter(|&&slot| matched[slot / 64] & (1 << (slot % 64)) != 0)
            .count();
        count = count
            .checked_sub(chunks.saturating_sub(1))
            .expect("matching chunk count underflow");
    }
    count
}
