// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::wand::{sum_bound, up, Bound};
use super::*;
use crate::core::documents::postings::PostingCursor;
use std::collections::BTreeMap;

struct TermState<'a> {
    cursor: PostingCursor<'a>,
    df: usize,
    live_count: usize,
    /// The largest frequency and least length bound over the whole list.
    maxima: (u32, u32),
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

fn count_match(store: &DocumentStore, slot: usize, seen: &mut [u64], count: &mut usize) {
    if !store.multichunk_documents.is_empty() {
        let (document, _) = store.borrowed_chunk(slot);
        if document.chunks.len() > 1 {
            let first = store.slots[&document.id][0];
            let bit = 1 << (first % 64);
            if seen[first / 64] & bit != 0 {
                return;
            }
            seen[first / 64] |= bit;
        }
    }
    *count += 1;
}

/// The weight bounds of one positive clause of a form.
struct ClauseBound {
    bound: Bound,
    /// A bound of the clause's weight in any chunk.
    ceiling: f64,
    /// The bound of the last block of a one-word clause, by block index.
    cached: (usize, f64),
}

/// Bounds of the positive clauses, or `None` where a bound cannot be proven.
struct Pruning {
    clauses: Vec<Option<ClauseBound>>,
}

impl Pruning {
    fn new(
        query: &LexicalQuery,
        states: &[TermState<'_>],
        indices: &[Vec<Option<usize>>],
        idfs: &[[f64; 2]],
        form: usize,
        average: f64,
        bm25: Bm25,
    ) -> Option<Self> {
        let mut clauses = Vec::with_capacity(query.clauses.len());
        let mut global = 0.0;
        for ((clause, indices), idf) in query.clauses.iter().zip(indices).zip(idfs) {
            if clause.occur == Occur::MustNot {
                clauses.push(None);
                continue;
            }
            let bound = Bound::new(idf[form], average, bm25)?;
            let ceiling = match indices
                .iter()
                .map(|index| index.map(|i| states[i].maxima))
                .collect::<Option<Vec<_>>>()
            {
                Some(maxima) => {
                    let (frequency, words) = combine(maxima.into_iter());
                    bound.at(frequency, words)
                }
                None => 0.0,
            };
            global = up(global + bound.global);
            clauses.push(Some(ClauseBound {
                bound,
                ceiling,
                cached: (usize::MAX, 0.0),
            }));
        }
        sum_bound(global, query.clauses.len())
            .is_finite()
            .then_some(Self { clauses })
    }

    /// A bound of the score of the chunk in `slot`, from the blocks holding it.
    fn at(
        &mut self,
        states: &mut [TermState<'_>],
        indices: &[Vec<Option<usize>>],
        slot: usize,
    ) -> f64 {
        let mut sum = 0.0;
        for (clause, indices) in self.clauses.iter_mut().zip(indices) {
            let Some(clause) = clause else { continue };
            let value = if let [Some(i)] = indices[..] {
                let state = &mut states[i];
                if state.cursor.locate(slot) {
                    let block = state.cursor.block_index();
                    if clause.cached.0 != block {
                        let (frequency, words) = state.cursor.block_maximum();
                        clause.cached = (block, clause.bound.at(frequency, words));
                    }
                    clause.cached.1
                } else {
                    0.0
                }
            } else {
                let mut maxima = Vec::with_capacity(indices.len());
                for index in indices {
                    match index {
                        Some(i) if states[*i].cursor.locate(slot) => {
                            maxima.push(states[*i].cursor.block_maximum());
                        }
                        _ => break,
                    }
                }
                if maxima.len() == indices.len() {
                    let (frequency, words) = combine(maxima.into_iter());
                    clause.bound.at(frequency, words)
                } else {
                    0.0
                }
            };
            sum = up(sum + value);
        }
        sum_bound(sum, self.clauses.len())
    }

    /// A bound of the score of any chunk in the block `lead` stands on, from that block's
    /// bounds and the bounds of the other lists' blocks over the same slots.
    fn over_lead_block(
        &self,
        states: &[TermState<'_>],
        indices: &[Vec<Option<usize>>],
        lead: usize,
    ) -> f64 {
        let cursor = &states[lead].cursor;
        let (Some((first, _)), Some(last)) = (cursor.peek(), cursor.block_end()) else {
            return 0.0;
        };
        let lead_maximum = cursor.block_maximum();
        let mut sum = 0.0;
        for (clause, indices) in self.clauses.iter().zip(indices) {
            let Some(clause) = clause else { continue };
            let maxima = indices.iter().map(|index| match index {
                Some(i) if *i == lead => lead_maximum,
                Some(i) => states[*i]
                    .cursor
                    .range_maximum(first, last)
                    .unwrap_or((0, u32::MAX)),
                None => (0, u32::MAX),
            });
            let (frequency, words) = combine(maxima);
            sum = up(sum + clause.bound.at(frequency, words));
        }
        sum_bound(sum, self.clauses.len())
    }
}

/// The frequency and length bounds a phrase of terms with the given bounds obeys: it occurs
/// at most as often as its rarest term in a chunk at least as long as every term's bound.
fn combine(maxima: impl Iterator<Item = (u32, u32)>) -> (u32, u32) {
    maxima.fold((u32::MAX, 0), |(frequency, words), (f, w)| {
        (frequency.min(f), words.max(w))
    })
}

fn positional_frequency(
    positions: &crate::core::documents::positions::Positions,
    indices: &[Option<usize>],
    states: &[TermState<'_>],
    exact: bool,
) -> u32 {
    let lead = indices
        .iter()
        .enumerate()
        .min_by_key(|(_, index)| {
            states[index.expect("present phrase term")]
                .cursor
                .peek()
                .expect("phrase candidate")
                .1
        })
        .expect("nonempty phrase")
        .0;
    let term_index = |index: Option<usize>| {
        states[index.expect("present phrase term")]
            .cursor
            .term_index()
    };
    let lead_positions = positions.iter(term_index(indices[lead]));
    if indices.len() == 2 {
        let mut cursors =
            [indices[0], indices[1]].map(|index| positions.iter(term_index(index)).peekable());
        return match_positions(lead_positions, lead, &mut cursors, exact);
    }
    if indices.len() == 3 {
        let mut cursors = [indices[0], indices[1], indices[2]]
            .map(|index| positions.iter(term_index(index)).peekable());
        return match_positions(lead_positions, lead, &mut cursors, exact);
    }
    if indices.len() == 4 {
        let mut cursors = [indices[0], indices[1], indices[2], indices[3]]
            .map(|index| positions.iter(term_index(index)).peekable());
        return match_positions(lead_positions, lead, &mut cursors, exact);
    }
    let mut cursors: Vec<_> = indices
        .iter()
        .map(|&index| positions.iter(term_index(index)).peekable())
        .collect();
    match_positions(lead_positions, lead, &mut cursors, exact)
}

fn match_positions<I: Iterator<Item = u32>>(
    lead_positions: I,
    lead: usize,
    cursors: &mut [std::iter::Peekable<I>],
    exact: bool,
) -> u32 {
    let mut count = 0;
    for position in lead_positions {
        let Some(start) = position.checked_sub(lead as u32) else {
            continue;
        };
        let matched = cursors.iter_mut().enumerate().all(|(offset, iter)| {
            let Some(target) = start.checked_add(offset as u32) else {
                return false;
            };
            while iter.peek().is_some_and(|&position| position < target) {
                iter.next();
            }
            iter.peek() == Some(&target)
        });
        count += u32::from(matched);
        if matched && !exact {
            break;
        }
    }
    count
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
    let single_phrase = query.clauses.len() == 1 && query.clauses[0].phrase;
    let phrases_ranked = !options.count && query.clauses.iter().any(|c| c.phrase);
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
                        maxima: list.maxima(),
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
        let term_plan =
            (!required.is_empty() && query.clauses.iter().all(|c| !c.phrase)).then(|| {
                let mut excluded = Vec::new();
                let mut scored = Vec::new();
                for (c, (clause, indices)) in query.clauses.iter().zip(&indices).enumerate() {
                    if let Some(index) = indices[0] {
                        if clause.occur == Occur::MustNot {
                            excluded.push(index);
                        } else if options.k > 0 {
                            scored.push((index, idfs[c][form]));
                        }
                    }
                }
                (excluded, scored)
            });
        // Ranked evaluation skips a chunk, or a whole block of the leading list, whose score
        // bound is below the floor of the top-k; counting still visits every match.
        let mut pruning = if options.k > 0 {
            Pruning::new(query, &states, &indices, &idfs, form, average, bm25)
        } else {
            None
        };
        let mut frequencies = vec![0; query.clauses.len()];
        let mut pending: Vec<Option<u32>> = vec![None; query.clauses.len()];
        // Optional words not yet required of every ranked candidate, by clause.
        let mut optional: Vec<(usize, usize)> = query
            .clauses
            .iter()
            .zip(&indices)
            .enumerate()
            .filter_map(|(c, (clause, indices))| match indices[..] {
                [Some(i)] if clause.occur == Occur::Should && !single_phrase => Some((c, i)),
                _ => None,
            })
            .collect();
        let mut checked_floor = 0.0;
        loop {
            // Once the floor exceeds every clause's ceiling but one optional word's, a chunk
            // without that word cannot rank: the word joins the conjunction (MaxScore's
            // essential lists), entering at the lead's position so no candidate repeats.
            if let (Some(pruning), Some(&lead), false) =
                (pruning.as_ref(), required.first(), options.count)
            {
                let floor = top.floor();
                if floor > checked_floor && !optional.is_empty() {
                    checked_floor = floor;
                    let mut promoted = false;
                    optional.retain(|&(c, i)| {
                        if required.contains(&i) {
                            return false;
                        }
                        let others = pruning
                            .clauses
                            .iter()
                            .enumerate()
                            .filter(|&(other, _)| other != c)
                            .filter_map(|(_, clause)| clause.as_ref())
                            .fold(0.0, |sum, clause| up(sum + clause.ceiling));
                        if sum_bound(others, query.clauses.len()) >= floor {
                            return true;
                        }
                        if let Some((at, _)) = states[lead].cursor.peek() {
                            states[i].cursor.advance(at);
                        }
                        required.push(i);
                        promoted = true;
                        false
                    });
                    if promoted {
                        required.sort_unstable_by_key(|&i| states[i].df);
                    }
                }
            }
            if let (Some(pruning), Some(&lead), false) =
                (pruning.as_ref(), required.first(), options.count)
            {
                while states[lead].cursor.at_block_start()
                    && pruning.over_lead_block(&states, &indices, lead) < top.floor()
                {
                    let end = states[lead].cursor.block_end().expect("block start");
                    let Some(next) = end.checked_add(1) else {
                        break;
                    };
                    states[lead].cursor.advance(next);
                }
            }
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
            if let Some((excluded, scored)) = &term_plan {
                let matched = required
                    .iter()
                    .all(|&i| states[i].cursor.peek().is_some_and(|p| p.1 > 0))
                    && !excluded.iter().any(|&i| {
                        states[i]
                            .cursor
                            .advance(slot)
                            .is_some_and(|p| p.0 == slot && p.1 > 0)
                    });
                if matched {
                    if options.count {
                        count_match(store, slot, &mut seen, &mut count);
                    }
                    if options.k > 0 {
                        let norm = bm25.norm(f64::from(store.slot_word_count(slot)), average);
                        let mut score = 0.0;
                        for &(i, idf) in scored {
                            if let Some((at, frequency)) = states[i].cursor.advance(slot) {
                                if at == slot && frequency > 0 {
                                    score += bm25.weight(idf, f64::from(frequency), norm);
                                }
                            }
                        }
                        ensure!(
                            score.is_finite(),
                            "lexical score is nonfinite; reduce BM25 parameters"
                        );
                        top.consider(store, score, slot);
                    }
                }
                states[required[0]].cursor.next();
                continue;
            }
            // A candidate's bound costs about what scoring its words costs, so it is checked
            // only where it can spare a phrase's position check.
            let skip = match pruning.as_mut() {
                Some(pruning) if phrases_ranked && top.floor() > 0.0 => {
                    pruning.at(&mut states, &indices, slot) < top.floor()
                }
                _ => false,
            };
            let mut scorable = true;
            if !skip {
                let mut phrases = false;
                for (((clause, indices), frequency_value), pending) in query
                    .clauses
                    .iter()
                    .zip(&indices)
                    .zip(&mut frequencies)
                    .zip(&mut pending)
                {
                    *pending = None;
                    if options.k == 0
                        && !required.is_empty()
                        && !single_phrase
                        && clause.occur == Occur::Should
                    {
                        *frequency_value = 0;
                        continue;
                    }
                    if clause.phrase {
                        let mut least = u32::MAX;
                        let possible = indices.iter().all(|&i| {
                            i.is_some_and(|i| {
                                states[i]
                                    .cursor
                                    .advance(slot)
                                    .filter(|p| p.0 == slot && p.1 > 0)
                                    .inspect(|p| least = least.min(p.1))
                                    .is_some()
                            })
                        });
                        *frequency_value = 0;
                        if possible {
                            *pending = Some(least);
                            phrases = true;
                        }
                    } else {
                        *frequency_value = indices[0]
                            .and_then(|i| states[i].cursor.advance(slot))
                            .filter(|p| p.0 == slot)
                            .map_or(0, |p| p.1);
                    }
                }
                if phrases {
                    let (document, chunk) = store.borrowed_chunk(slot);
                    let chunk = &document.chunks[chunk];
                    let positions = positions_for_chunk(store, chunk)?;
                    // The positions of a phrase are counted only where the count can
                    // change the ranking; otherwise its first occurrence settles the match.
                    let exact = options.k > 0
                        && match pruning.as_ref() {
                            Some(pruning) => {
                                let words = store.slot_word_count(slot);
                                let norm = bm25.norm(f64::from(words), average);
                                let mut sum = 0.0;
                                for ((&frequency, pending), bound) in
                                    frequencies.iter().zip(&pending).zip(&pruning.clauses)
                                {
                                    let Some(bound) = bound else { continue };
                                    let value = match pending {
                                        Some(least) => bound.bound.at(*least, words),
                                        None if frequency > 0 => {
                                            bm25.weight(bound.bound.idf, f64::from(frequency), norm)
                                        }
                                        None => 0.0,
                                    };
                                    sum = up(sum + value);
                                }
                                sum_bound(sum, query.clauses.len()) >= top.floor()
                            }
                            None => true,
                        };
                    scorable = exact;
                    // Below the floor and uncounted, the phrase's positions are not read.
                    let needed = exact || options.count;
                    for (((clause, frequency_value), pending), indices) in query
                        .clauses
                        .iter()
                        .zip(&mut frequencies)
                        .zip(&pending)
                        .zip(&indices)
                    {
                        if pending.is_some() && needed {
                            *frequency_value = if chunk.terms.positions.is_some() {
                                positional_frequency(
                                    &positions,
                                    indices,
                                    &states,
                                    exact && clause.occur != Occur::MustNot,
                                )
                            } else {
                                let mut ids = std::array::from_fn(|form| {
                                    clause.words[form]
                                        .iter()
                                        .map(|_| WordId::default())
                                        .collect()
                                });
                                frequency_with_positions(
                                    clause,
                                    chunk,
                                    exact && clause.occur != Occur::MustNot,
                                    &mut ids,
                                    Some(&positions),
                                )
                            };
                        }
                    }
                }
                if matches(query, &frequencies) {
                    if options.count {
                        count_match(store, slot, &mut seen, &mut count);
                    }
                    if options.k > 0 && scorable {
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
    count_terms(
        store,
        std::array::from_fn(|form| {
            let words: BTreeSet<&str> = query
                .clauses
                .iter()
                .map(|c| c.words[form][0].as_str())
                .collect();
            words.into_iter().collect()
        }),
    )
}

/// The number of documents with a chunk holding one of `words[form]` in that form's postings
/// at a positive frequency: the documents an unfiltered disjunction of the words scores.
pub(in crate::core::documents) fn count_terms(
    store: &DocumentStore,
    words: [Vec<&str>; 2],
) -> usize {
    if words.iter().all(|words| words.len() <= 1) && store.chunk_count == store.documents.len() {
        // One chunk per document and at most one list per form, whose slots are disjoint.
        return (0..2)
            .map(|form| {
                words[form]
                    .first()
                    .and_then(|word| store.postings[form].get(word))
                    .map_or(0, |list| list.positive_len())
            })
            .sum();
    }
    let mut matched = vec![0u64; store.slot_count.div_ceil(64)];
    for (form, words) in words.iter().enumerate() {
        for word in words {
            if let Some(list) = store.postings[form].get(word) {
                list.mark_slots(&mut matched);
            }
        }
    }
    let mut count: usize = matched.iter().map(|word| word.count_ones() as usize).sum();
    if !store.multichunk_documents.is_empty() {
        // A document counts once: each further matching chunk of one is taken back.
        // Sized once: at most one entry per matched chunk of a multi-chunk document.
        let mut seen = crate::core::hash::FastSet::with_capacity_and_hasher(
            count.min(store.multichunk_documents.len()),
            Default::default(),
        );
        for (index, &bits) in matched.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let slot = index * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (document, _) = store.borrowed_chunk(slot);
                if document.chunks.len() > 1 && !seen.insert(std::ptr::from_ref(document)) {
                    count -= 1;
                }
            }
        }
    }
    count
}
