// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::execution::TopK;
use super::*;
use crate::core::documents::postings::PostingCursor;

fn up(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

#[derive(Clone, Copy)]
struct Bound {
    idf: f64,
    average: f64,
    bm25: Bm25,
    global: f64,
}

impl Bound {
    fn new(idf: f64, average: f64, bm25: Bm25) -> Option<Self> {
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

    fn at(self, frequency: u32, words: u32) -> f64 {
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

struct State<'a> {
    cursor: PostingCursor<'a>,
    bound: Bound,
}

impl State<'_> {
    fn block_bound(&self) -> f64 {
        let points = self.cursor.block_impacts();
        if points.is_empty() {
            let (frequency, words) = self.cursor.block_maximum();
            self.bound.at(frequency, words)
        } else {
            points
                .iter()
                .map(|&(frequency, words)| self.bound.at(frequency, words))
                .fold(0.0, f64::max)
        }
    }
}

fn sum_bound(sum: f64, clauses: usize) -> f64 {
    up(sum * (1.0 + 2.0 * clauses as f64 * f64::EPSILON))
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
        let mut states = Vec::new();
        let mut global_sum = 0.0;
        for (clause, idfs) in query.clauses.iter().zip(&idfs) {
            if let Some(list) = store.postings[form].get(&clause.words[form][0]) {
                let Some(bound) = Bound::new(idfs[form], average, bm25) else {
                    return Ok(None);
                };
                global_sum = up(global_sum + bound.global);
                states.push(State {
                    cursor: list.iter(),
                    bound,
                });
            }
        }
        if !sum_bound(global_sum, query.clauses.len()).is_finite() {
            return Ok(None);
        }
        forms.push(states);
    }
    let mut top = TopK::new(options.k);
    for mut states in forms {
        let mut active = Vec::with_capacity(states.len());
        let mut window: Option<(usize, f64)> = None;
        loop {
            active.clear();
            active.extend((0..states.len()).filter(|&i| states[i].cursor.peek().is_some()));
            active.sort_unstable_by_key(|&i| states[i].cursor.peek().unwrap().0);
            let Some(&first) = active.first() else { break };
            let minimum = states[first].cursor.peek().unwrap().0;
            if window.is_none_or(|(end, _)| minimum > end) {
                let end = active
                    .iter()
                    .map(|&i| states[i].cursor.block_end().unwrap())
                    .min()
                    .unwrap();
                let mut bound = 0.0;
                for state in &states {
                    if state.cursor.peek().is_some_and(|p| p.0 <= end) {
                        bound = up(bound + state.block_bound());
                    }
                }
                window = Some((end, sum_bound(bound, query.clauses.len())));
            }
            let (end, bound) = window.unwrap();
            if bound < top.floor() {
                let Some(target) = end.checked_add(1) else {
                    break;
                };
                for state in &mut states {
                    if state.cursor.peek().is_some_and(|p| p.0 <= end) {
                        state.cursor.advance(target);
                    }
                }
                window = None;
                continue;
            }
            let mut bound = 0.0;
            let pivot = active.iter().find_map(|&i| {
                bound = up(bound + states[i].bound.global);
                (sum_bound(bound, query.clauses.len()) >= top.floor())
                    .then(|| states[i].cursor.peek().unwrap().0)
            });
            let Some(pivot) = pivot else { break };
            if minimum < pivot {
                states[first].cursor.advance(pivot);
                continue;
            }
            let norm = bm25.norm(f64::from(store.slot_word_count(minimum)), average);
            let mut score = 0.0;
            let mut matched = false;
            for state in &mut states {
                if let Some((slot, frequency)) = state.cursor.peek() {
                    if slot == minimum {
                        if frequency > 0 {
                            matched = true;
                            score += bm25.weight(state.bound.idf, f64::from(frequency), norm);
                        }
                        state.cursor.next();
                    }
                }
            }
            ensure!(
                score.is_finite(),
                "lexical score is nonfinite; reduce BM25 parameters"
            );
            if matched {
                top.consider(store, score, minimum);
            }
        }
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
        let mut state = State {
            cursor: list.iter(),
            bound: Bound::new(idf, average, Bm25::DEFAULT).unwrap(),
        };
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
                    let mut state = State {
                        cursor: list.iter(),
                        bound,
                    };
                    while let Some((slot, frequency)) = state.cursor.peek() {
                        let score = bm25.weight(
                            3.5,
                            f64::from(frequency),
                            bm25.norm(f64::from(store.slot_word_count(slot)), average),
                        );
                        assert!(
                            state.block_bound() >= score,
                            "{slot}: {score} > {}",
                            state.block_bound()
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
            ] {
                let query = LexicalQuery::parse(text).unwrap();
                for k in [0, 1, 10, 127, 128, 129, 256, 1000] {
                    for (k1, b) in [(0.0, 0.0), (1.2, 0.75), (2.0, 1.0), (1e100, 0.5)] {
                        let options = LexicalOptions {
                            k,
                            count: true,
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
            } else if stage == 1 {
                for i in (1..=513).step_by(11) {
                    insert(&mut store, &format!("d{i:04}"), "red blue", true, true);
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
