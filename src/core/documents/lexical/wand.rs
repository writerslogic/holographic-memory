// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::execution::TopK;
use super::*;
use crate::core::documents::postings::{PostingCursor, PostingList, BLOCK_LEN, SUPER_LEN};

pub(super) fn up(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

/// Bounds of one clause's weights under its inverse document frequency.
#[derive(Clone, Copy)]
pub(super) struct Bound {
    pub(super) idf: f64,
    average: f64,
    bm25: Bm25,
    pub(super) global: f64,
}

impl Bound {
    pub(super) fn new(idf: f64, average: f64, bm25: Bm25) -> Option<Self> {
        if !idf.is_finite() || idf < 0.0 || !average.is_finite() || average < 1.0 {
            return None;
        }
        let maximum_norm = bm25.norm(usize::MAX as f64, average);
        let numerator = idf * f64::from(u32::MAX) * bm25.k1_plus_one;
        if !bm25.k1_plus_one.is_finite()
            || !maximum_norm.is_finite()
            || !(maximum_norm + f64::from(u32::MAX)).is_finite()
            || !numerator.is_finite()
            || (idf != 0.0
                && (!idf.is_normal() || !bm25.weight(idf, 1.0, maximum_norm).is_normal()))
        {
            return None;
        }
        let mut bound = Self {
            idf,
            average,
            bm25,
            global: 0.0,
        };
        bound.global = bound.at(u32::MAX, 0);
        bound.global.is_finite().then_some(bound)
    }

    /// A bound of the weight of any posting with at most `frequency` occurrences in a chunk
    /// of at least `words` words.
    pub(super) fn at(self, frequency: u32, words: u32) -> f64 {
        if frequency == 0 || self.idf == 0.0 {
            return 0.0;
        }
        // For normal operations this exceeds ((1+u)/(1-u))^4, u=2^-53:
        // both evaluation errors around the exact monotone tf/(tf+norm) are covered.
        up(self.bm25.weight(
            self.idf,
            f64::from(frequency),
            self.bm25.norm(f64::from(words), self.average),
        ) * (1.0 + 16.0 * f64::EPSILON))
    }
}

/// One term's cursor with the bounds of its weights.
pub(super) struct State<'a> {
    list: &'a PostingList,
    pub(super) cursor: PostingCursor<'a>,
    pub(super) bound: Bound,
    /// A bound of every weight of the list (its largest frequency at its least length).
    pub(super) ceiling: f64,
}

impl<'a> State<'a> {
    pub(super) fn new(list: &'a PostingList, bound: Bound) -> Self {
        let (frequency, words) = list.maxima();
        Self {
            list,
            cursor: list.iter(),
            bound,
            ceiling: bound.at(frequency, words),
        }
    }

    /// A bound of the weights in the cursor's block from its largest frequency and least
    /// length: one evaluation, looser than the frontier's.
    pub(super) fn loose_bound(&self) -> f64 {
        let (frequency, words) = self.cursor.block_maximum();
        self.bound.at(frequency, words)
    }

    /// The tightest bound of the weights in the cursor's block, from its frontier.
    #[cfg(test)]
    fn block_bound(&self) -> f64 {
        frontier_bound(self.bound, self.loose_bound(), self.cursor.block_impacts())
    }
}

/// The tightest bound of a block's weights from its frontier, or `loose` without one.
fn frontier_bound(bound: Bound, loose: f64, points: &[(u32, u32)]) -> f64 {
    if points.is_empty() {
        loose
    } else {
        points
            .iter()
            .map(|&(frequency, words)| bound.at(frequency, words))
            .fold(0.0, f64::max)
    }
}

/// Top-k of one list, strongest runs of blocks first. A long list's runs are ordered by the
/// bound of their summary frontier and visited until a run's bound is below the floor; within
/// a run, a block is decoded only when its own bounds reach the floor. Everything skipped is
/// below the floor, so the result is the exhaustive one.
fn single_list(
    store: &DocumentStore,
    state: &State<'_>,
    average: f64,
    bm25: Bm25,
    top: &mut TopK,
) -> Result<()> {
    let list = state.list;
    let mut decoded = [(0, 0); BLOCK_LEN];
    let mut visit = |index: usize, top: &mut TopK| -> Result<()> {
        let ((frequency, words), points) = list.block_bounds(index);
        let loose = state.bound.at(frequency, words);
        if loose < top.floor() || frontier_bound(state.bound, loose, points) < top.floor() {
            return Ok(());
        }
        let count = list.decode_block_into(index, &mut decoded);
        for &(slot, frequency) in &decoded[..count] {
            if frequency == 0 {
                continue;
            }
            let norm = bm25.norm(f64::from(store.slot_word_count(slot)), average);
            let score = bm25.weight(state.bound.idf, f64::from(frequency), norm);
            ensure!(
                score.is_finite(),
                "lexical score is nonfinite; reduce BM25 parameters"
            );
            top.consider(store, score, slot);
        }
        Ok(())
    };
    let Some(summary) = list.summary() else {
        for index in 0..list.block_count() {
            visit(index, top)?;
        }
        return Ok(());
    };
    let mut runs: Vec<(f64, usize)> = summary
        .iter()
        .enumerate()
        .map(|(run, points)| (frontier_bound(state.bound, 0.0, points), run))
        .collect();
    runs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for (bound, run) in runs {
        if bound < top.floor() {
            break;
        }
        let start = run * SUPER_LEN;
        for index in start..(start + SUPER_LEN).min(list.block_count()) {
            visit(index, top)?;
        }
    }
    Ok(())
}

pub(super) fn sum_bound(sum: f64, clauses: usize) -> f64 {
    up(sum * (1.0 + 2.0 * clauses as f64 * f64::EPSILON))
}

/// The states of one form's terms with a finite bound of any sum of their weights, or `None`
/// when a bound cannot be proven (the caller then evaluates exhaustively).
fn form_states<'a>(
    terms: impl Iterator<Item = (f64, &'a PostingList)>,
    average: f64,
    bm25: Bm25,
    clauses: usize,
) -> Option<Vec<State<'a>>> {
    let mut states = Vec::new();
    let mut global_sum = 0.0;
    for (idf, list) in terms {
        let bound = Bound::new(idf, average, bm25)?;
        global_sum = up(global_sum + bound.global);
        states.push(State::new(list, bound));
    }
    sum_bound(global_sum, clauses).is_finite().then_some(states)
}

/// Block-max MaxScore over the term states of one form into `top`. Lists whose ceilings
/// together stay below the floor are non-essential: only the other lists propose candidates,
/// and a candidate reads the non-essential lists (strongest first) only while a bound from
/// their blocks at it can still reach the floor. A candidate is skipped only when a bound of
/// its score is below the floor of `top`, and a scored one gets the sum in clause order that
/// exhaustive evaluation computes, so the result is the exhaustive one.
fn max_score(
    store: &DocumentStore,
    states: &mut [State<'_>],
    average: f64,
    bm25: Bm25,
    clauses: usize,
    top: &mut TopK,
) -> Result<()> {
    // Weakest ceiling first; `ceilings[i]` bounds the sum of the weights of lists `..=i`.
    let mut order: Vec<usize> = (0..states.len()).collect();
    order.sort_by(|&a, &b| states[a].ceiling.total_cmp(&states[b].ceiling));
    let mut ceilings = Vec::with_capacity(order.len());
    let mut sum = 0.0;
    for &i in &order {
        sum = up(sum + states[i].ceiling);
        ceilings.push(sum);
    }
    let mut weights = vec![None; states.len()];
    let mut blocks = vec![0.0; order.len()];
    loop {
        let floor = top.floor();
        let Some(essential) = ceilings
            .iter()
            .position(|&ceiling| sum_bound(ceiling, clauses) >= floor)
        else {
            break;
        };
        let Some(candidate) = order[essential..]
            .iter()
            .filter_map(|&i| states[i].cursor.peek().map(|p| p.0))
            .min()
        else {
            break;
        };
        let norm = bm25.norm(f64::from(store.slot_word_count(candidate)), average);
        let mut partial = 0.0;
        for &i in &order[essential..] {
            let state = &mut states[i];
            weights[i] = None;
            if let Some((slot, frequency)) = state.cursor.peek() {
                if slot == candidate {
                    if frequency > 0 {
                        let weight = bm25.weight(state.bound.idf, f64::from(frequency), norm);
                        partial = up(partial + weight);
                        weights[i] = Some(weight);
                    }
                    state.cursor.next();
                }
            }
        }
        // Block bounds of the non-essential lists at the candidate, read without decoding.
        let mut rest = 0.0;
        for (position, &i) in order[..essential].iter().enumerate() {
            let state = &mut states[i];
            weights[i] = None;
            let bound = if state.cursor.locate(candidate) {
                state.loose_bound()
            } else {
                0.0
            };
            rest = up(rest + bound);
            blocks[position] = rest;
        }
        let mut complete = true;
        for position in (0..essential).rev() {
            if sum_bound(up(partial + blocks[position]), clauses) < floor {
                complete = false;
                break;
            }
            let i = order[position];
            let state = &mut states[i];
            if let Some((slot, frequency)) = state.cursor.advance(candidate) {
                if slot == candidate && frequency > 0 {
                    let weight = bm25.weight(state.bound.idf, f64::from(frequency), norm);
                    partial = up(partial + weight);
                    weights[i] = Some(weight);
                }
            }
        }
        if !complete || sum_bound(partial, clauses) < floor {
            continue;
        }
        let mut score = 0.0;
        let mut matched = false;
        for weight in weights.iter().flatten() {
            matched = true;
            score += weight;
        }
        ensure!(
            score.is_finite(),
            "lexical score is nonfinite; reduce BM25 parameters"
        );
        if matched {
            top.consider(store, score, candidate);
        }
    }
    Ok(())
}

pub(super) fn execute(
    store: &DocumentStore,
    query: &LexicalQuery,
    options: &LexicalOptions,
    bm25: Bm25,
) -> Result<Option<LexicalResults>> {
    if query.clauses.is_empty()
        || query
            .clauses
            .iter()
            .any(|c| c.occur != Occur::Should || c.phrase)
    {
        return Ok(None);
    }
    let idfs = clause_idfs(store, query);
    let average = (store.word_count as f64 / (store.chunk_count as f64).max(1.0)).max(1.0);
    let mut forms = Vec::new();
    for form in 0..2 {
        let terms = query
            .clauses
            .iter()
            .zip(&idfs)
            .filter_map(|(clause, idfs)| {
                store.postings[form]
                    .get(&clause.words[form][0])
                    .map(|list| (idfs[form], list))
            });
        let Some(states) = form_states(terms, average, bm25, query.clauses.len()) else {
            return Ok(None);
        };
        forms.push(states);
    }
    let mut top = TopK::new(options.k);
    for mut states in forms {
        if let [state] = &states[..] {
            single_list(store, state, average, bm25, &mut top)?;
            continue;
        }
        max_score(
            store,
            &mut states,
            average,
            bm25,
            query.clauses.len(),
            &mut top,
        )?;
    }
    Ok(Some(LexicalResults {
        hits: top.finish(store),
        count: options
            .count
            .then(|| super::execution::count_union(store, query)),
    }))
}

#[cfg(test)]
mod tests {
    use super::super::tests::insert;
    use crate::core::documents::postings::SUPER_LEN;

    /// Enough one-word documents for a list of two summary runs.
    const SUMMARY_DOCUMENTS: usize = 2 * SUPER_LEN * BLOCK_LEN + 300;
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn bounds_reject_unproven_arithmetic_and_cover_zero_scores() {
        assert!(Bound::new(-1.0, 1.0, Bm25::DEFAULT).is_none());
        assert!(Bound::new(f64::from_bits(1), 1.0, Bm25::DEFAULT).is_none());
        assert!(Bound::new(1.0, f64::INFINITY, Bm25::DEFAULT).is_none());
        let extreme = Bm25::from_options(&SearchOptions {
            bm25_k1: Some(f64::MAX),
            ..Default::default()
        })
        .unwrap();
        assert!(Bound::new(1.0, 1.0, extreme).is_none());
        let zero = Bound::new(0.0, 1.0, Bm25::DEFAULT).unwrap();
        assert_eq!(zero.at(u32::MAX, 0), 0.0);
    }

    #[test]
    fn weak_blocks_are_below_the_threshold_without_losing_ties() {
        let mut store = DocumentStore::default();
        for i in 0..384 {
            let text = if i < 128 {
                "red ".repeat(100)
            } else {
                format!("red {}", "filler ".repeat(1000))
            };
            insert(&mut store, &format!("d{:04}", 384 - i), &text, true, true);
        }
        let query = LexicalQuery::parse("red").unwrap();
        let options = LexicalOptions {
            k: 10,
            count: true,
            execution: LexicalExecution::Exhaustive,
            ..Default::default()
        };
        let reference = super::super::execute(&store, &query, &options).unwrap();
        let average = store.word_count as f64 / store.chunk_count as f64;
        let idf = clause_idfs(&store, &query)[0][0];
        let list = store.postings[0].get("red").unwrap();
        let mut state = State::new(list, Bound::new(idf, average, Bm25::DEFAULT).unwrap());
        state.cursor.advance(128).unwrap();
        assert!(state.block_bound() < reference.hits.last().unwrap().score);
        let actual = super::super::execute(
            &store,
            &query,
            &LexicalOptions {
                execution: LexicalExecution::BlockMaxWand,
                ..options
            },
        )
        .unwrap();
        assert_eq!(actual, reference);
    }

    fn check_blocks(store: &DocumentStore) {
        for (k1, b) in [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.2, 0.75),
            (2.0, 1.0),
            (1e100, 0.5),
        ] {
            let bm25 = Bm25::from_options(&SearchOptions {
                bm25_k1: Some(k1),
                bm25_b: Some(b),
                ..Default::default()
            })
            .unwrap();
            for average in [1.0, 3.1, 1e10] {
                for word in ["red", "blue", "green"] {
                    let Some(list) = store.postings[0].get(word) else {
                        continue;
                    };
                    let bound = Bound::new(3.5, average, bm25).unwrap();
                    let mut state = State::new(list, bound);
                    while let Some((slot, frequency)) = state.cursor.peek() {
                        let score = bm25.weight(
                            3.5,
                            f64::from(frequency),
                            bm25.norm(f64::from(store.slot_word_count(slot)), average),
                        );
                        assert!(
                            state.block_bound() >= score && state.ceiling >= score,
                            "{slot}: {score} > {} or {}",
                            state.block_bound(),
                            state.ceiling
                        );
                        state.cursor.next();
                    }
                }
            }
        }
    }

    #[test]
    fn live_block_bounds_and_wand_survive_replacement_delete_and_slot_reuse() {
        let mut store = DocumentStore::default();
        for i in 0..513 {
            let text = if i < 20 {
                "red blue green ".repeat(12)
            } else {
                format!("red {} {}", "blue ".repeat(i % 5), "green ".repeat(i % 37))
            };
            insert(
                &mut store,
                &format!("d{:04}", 513 - i),
                &text,
                true,
                i % 7 != 0,
            );
        }
        for stage in 0..3 {
            check_blocks(&store);
            for text in [
                "red",
                "red blue green",
                "red absent blue",
                "running runs red blue",
                "\"red blue\"",
                "\"blue green\" red",
                "+red blue",
                "+red -green",
                "+\"red blue\" -\"blue blue\"",
            ] {
                let query = LexicalQuery::parse(text).unwrap();
                for (k, count) in [0, 1, 10, 127, 128, 129, 256, 1000]
                    .into_iter()
                    .flat_map(|k| [(k, true), (k, false)])
                {
                    for (k1, b) in [(0.0, 0.0), (1.2, 0.75), (2.0, 1.0), (1e100, 0.5)] {
                        let options = LexicalOptions {
                            k,
                            count,
                            bm25_k1: k1,
                            bm25_b: b,
                            execution: LexicalExecution::Exhaustive,
                        };
                        let reference = super::super::execute(&store, &query, &options).unwrap();
                        for execution in [
                            LexicalExecution::BlockMaxWand,
                            LexicalExecution::DocumentAtATime,
                            LexicalExecution::Auto,
                        ] {
                            let actual = super::super::execute(
                                &store,
                                &query,
                                &LexicalOptions {
                                    execution,
                                    ..options.clone()
                                },
                            )
                            .unwrap();
                            assert_eq!(
                                actual, reference,
                                "{text}, k={k}, stage={stage}, {execution:?}"
                            );
                        }
                    }
                }
            }
            if stage == 0 {
                for i in (1..=513).step_by(7) {
                    store.remove(&format!("d{i:04}"));
                }
                // Deletes (all of unstemmed documents) rewrite those lists with the length
                // bounds a fresh build gives them.
                let words: Vec<u32> = (0..store.slot_count)
                    .map(|slot| store.slot_word_count(slot))
                    .collect();
                for word in ["red", "blue", "green"] {
                    let list = store.postings[0].get(word).unwrap();
                    assert!(list.bounds_are_tight(&words), "{word} bounds loosened");
                }
            } else if stage == 1 {
                for i in (1..=513).step_by(11) {
                    insert(&mut store, &format!("d{i:04}"), "red blue", true, true);
                }
            }
        }
    }

    /// Unions past sixteen terms of mixed rarity, before and after deletes and replacements,
    /// ranked and counted: MaxScore and the single-list scan against exhaustive evaluation.
    #[test]
    fn many_term_unions_and_single_terms_match_exhaustive_through_churn() {
        let vocabulary: Vec<String> = (0..24).map(|i| format!("w{i}")).collect();
        let mut store = DocumentStore::default();
        let text = |i: usize| {
            let mut words = Vec::new();
            for (w, word) in vocabulary.iter().enumerate() {
                // Word w occurs in about one document of w + 1, more often in short ones.
                if (i * 7 + w * 13).is_multiple_of(w + 1) {
                    words.extend(std::iter::repeat_n(word.as_str(), 1 + (i + w) % 4));
                }
            }
            words.push("filler");
            words.extend(std::iter::repeat_n("pad", i % 50));
            words.join(" ")
        };
        for i in 0..700 {
            insert(&mut store, &format!("d{i:04}"), &text(i), true, true);
        }
        let all = vocabulary.join(" ");
        let rare = vocabulary[12..].join(" ");
        for stage in 0..3 {
            for text in [all.as_str(), rare.as_str(), "w0", "w5", "w23", "w0 w1 w23"] {
                let query = LexicalQuery::parse(text).unwrap();
                for k in [1, 10, 100, 1000] {
                    for count in [false, true] {
                        let options = LexicalOptions {
                            k,
                            count,
                            ..Default::default()
                        };
                        let reference = super::super::execute(
                            &store,
                            &query,
                            &LexicalOptions {
                                execution: LexicalExecution::Exhaustive,
                                ..options.clone()
                            },
                        )
                        .unwrap();
                        for execution in [LexicalExecution::Auto, LexicalExecution::BlockMaxWand] {
                            let actual = super::super::execute(
                                &store,
                                &query,
                                &LexicalOptions {
                                    execution,
                                    ..options.clone()
                                },
                            )
                            .unwrap();
                            assert_eq!(
                                actual, reference,
                                "{text}, k={k}, count={count}, stage={stage}"
                            );
                        }
                    }
                }
            }
            if stage == 0 {
                for i in (0..700).step_by(3) {
                    store.remove(&format!("d{i:04}"));
                }
            } else if stage == 1 {
                for i in (0..700).step_by(5) {
                    insert(
                        &mut store,
                        &format!("d{i:04}"),
                        &text(i * 3 + 1),
                        true,
                        true,
                    );
                }
            }
        }
    }

    /// A list long enough for run summaries, through deletes, length invalidations and
    /// replacements: the single-list scan against exhaustive evaluation.
    #[test]
    fn summarized_single_lists_match_exhaustive_through_churn() {
        let mut store = DocumentStore::default();
        let text = |i: usize| {
            let mut words = vec!["w"; 1 + (i * 7919) % 13];
            words.extend(std::iter::repeat_n("pad", (i * 104_729) % 97));
            words.join(" ")
        };
        let documents = SUMMARY_DOCUMENTS;
        for i in 0..documents {
            insert(&mut store, &format!("d{i:05}"), &text(i), false, true);
        }
        assert!(store.postings[0].get("w").unwrap().summary().is_some());
        let query = LexicalQuery::parse("w").unwrap();
        for stage in 0..3 {
            for k in [1, 10, 100] {
                let options = LexicalOptions {
                    k,
                    ..Default::default()
                };
                let reference = super::super::execute(
                    &store,
                    &query,
                    &LexicalOptions {
                        execution: LexicalExecution::Exhaustive,
                        ..options.clone()
                    },
                )
                .unwrap();
                let actual = super::super::execute(&store, &query, &options).unwrap();
                assert_eq!(actual, reference, "k={k}, stage={stage}");
            }
            if stage == 0 {
                for i in (0..documents).step_by(5) {
                    store.remove(&format!("d{i:05}"));
                }
            } else if stage == 1 {
                // Replacements reuse slots with new lengths and the strongest postings.
                for i in (0..documents).step_by(7) {
                    insert(&mut store, &format!("d{i:05}"), "w w w w w w", false, true);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn scalar_bound_dominates_each_frequency_and_length(
            maximum in 1u32..=u32::MAX,
            seed in any::<u32>(),
            minimum in any::<u32>(),
            extra in any::<u32>(),
            average in 1u32..=u32::MAX,
            k1 in 0.0f64..1e6,
            b in 0.0f64..=1.0,
            idf in 1e-12f64..64.0,
        ) {
            let bm25 = Bm25::from_options(&SearchOptions { bm25_k1: Some(k1), bm25_b: Some(b), ..Default::default() }).unwrap();
            let bound = Bound::new(idf, f64::from(average), bm25).unwrap();
            let frequency = seed % maximum + 1;
            let length = minimum.saturating_add(extra);
            let actual = bm25.weight(idf, f64::from(frequency), bm25.norm(f64::from(length), f64::from(average)));
            prop_assert!(bound.at(maximum, minimum) >= actual);
        }
    }
}
