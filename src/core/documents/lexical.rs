// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::positions::tokens;
use super::{Bm25, DocumentStore, SearchOptions, StoredChunk, MAX_DOCUMENT_BYTES};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LexicalExecution {
    #[default]
    Auto,
    /// Unpruned posting-cursor execution.
    DocumentAtATime,
    /// Conservative block-max WAND for term disjunctions; other queries use cursors.
    BlockMaxWand,
    /// Full scan and full sort, useful for verifying another execution strategy.
    Exhaustive,
}

#[derive(Clone, Debug)]
pub struct LexicalOptions {
    pub k: usize,
    pub count: bool,
    pub bm25_k1: f64,
    pub bm25_b: f64,
    pub execution: LexicalExecution,
}

impl Default for LexicalOptions {
    fn default() -> Self {
        Self {
            k: 10,
            count: false,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            execution: LexicalExecution::Auto,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LexicalHit {
    pub document_id: String,
    pub chunk_index: usize,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LexicalResults {
    pub hits: Vec<LexicalHit>,
    pub count: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Occur {
    Must,
    Should,
    MustNot,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Clause {
    occur: Occur,
    words: [Vec<String>; 2],
    phrase: bool,
}

/// A flat Boolean query: bare optional clauses, +required clauses, -excluded clauses,
/// and quoted adjacent phrases. Purely negative queries match no documents.
#[derive(Clone, Debug)]
pub struct LexicalQuery {
    clauses: Vec<Clause>,
}

impl LexicalQuery {
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(text.len() <= MAX_DOCUMENT_BYTES, "query exceeds 8 MiB");
        let mut rest = text.trim();
        let mut clauses = Vec::new();
        while !rest.is_empty() {
            ensure!(clauses.len() < 1024, "lexical query exceeds 1024 clauses");
            let occur = match rest.as_bytes()[0] {
                b'+' => {
                    rest = &rest[1..];
                    Occur::Must
                }
                b'-' => {
                    rest = &rest[1..];
                    Occur::MustNot
                }
                _ => Occur::Should,
            };
            let phrase = rest.starts_with('"');
            let value;
            if phrase {
                rest = &rest[1..];
                let end = rest
                    .find('"')
                    .ok_or_else(|| anyhow::anyhow!("unterminated lexical phrase"))?;
                value = &rest[..end];
                rest = &rest[end + 1..];
                ensure!(
                    rest.is_empty() || rest.starts_with(char::is_whitespace),
                    "unsupported lexical phrase suffix (only zero slop is supported)"
                );
                ensure!(
                    value
                        .chars()
                        .all(|c| c.is_alphanumeric() || c.is_whitespace()),
                    "unsupported character in lexical phrase"
                );
            } else {
                let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
                value = &rest[..end];
                rest = &rest[end..];
                ensure!(
                    !value.is_empty() && value.chars().all(char::is_alphanumeric),
                    "unsupported lexical query syntax"
                );
            }
            let words = [tokens(value, false), tokens(value, true)];
            ensure!(!words[0].is_empty(), "lexical clause requires a token");
            ensure!(words[0].len() <= 1024, "lexical phrase exceeds 1024 tokens");
            clauses.push(Clause {
                occur,
                phrase: phrase && words[0].len() > 1,
                words,
            });
            rest = rest.trim_start();
        }
        clauses.sort_unstable();
        clauses.dedup();
        Ok(Self { clauses })
    }

    fn has_positive(&self) -> bool {
        self.clauses.iter().any(|c| c.occur != Occur::MustNot)
    }
}

fn frequency(clause: &Clause, chunk: &StoredChunk) -> u32 {
    let words = &clause.words[usize::from(chunk.stemmed)];
    if !clause.phrase {
        return chunk.terms.get(&words[0]).copied().unwrap_or(0);
    }
    let mut indices = Vec::with_capacity(words.len());
    let (mut lead, mut least) = (0, u32::MAX);
    for word in words {
        let Some(index) = chunk.terms.index(word) else {
            return 0;
        };
        let frequency = chunk.terms.frequency_at(index);
        if frequency == 0 {
            return 0;
        }
        if frequency < least {
            lead = indices.len();
            least = frequency;
        }
        indices.push(index);
    }
    let positions = chunk
        .terms
        .positions
        .as_ref()
        .expect("phrase capability checked");
    let mut cursors: Vec<_> = indices
        .iter()
        .map(|&i| positions.iter(i).peekable())
        .collect();
    let mut count = 0;
    for position in positions.iter(indices[lead]) {
        let Some(start) = position.checked_sub(lead as u32) else {
            continue;
        };
        let matched = cursors.iter_mut().enumerate().all(|(offset, iter)| {
            let Some(target) = start.checked_add(offset as u32) else {
                return false;
            };
            while iter.peek().is_some_and(|&p| p < target) {
                iter.next();
            }
            iter.peek() == Some(&target)
        });
        count += u32::from(matched);
    }
    count
}

fn prepare(store: &DocumentStore, query: &LexicalQuery, options: &LexicalOptions) -> Result<Bm25> {
    ensure!(options.k <= 1_000_000, "lexical k exceeds 1000000");
    let bm25 = Bm25::from_options(&SearchOptions {
        bm25_k1: Some(options.bm25_k1),
        bm25_b: Some(options.bm25_b),
        ..Default::default()
    })?;
    ensure!(!query.clauses.iter().any(|c| c.phrase) || store.nonpositional_chunks == 0,
        "unsupported index: lexical phrase queries require IngestOptions::positions for every live chunk");
    Ok(bm25)
}

fn clause_idfs(store: &DocumentStore, query: &LexicalQuery) -> Vec<[f64; 2]> {
    let n = store.chunk_count as f64;
    query
        .clauses
        .iter()
        .map(|clause| {
            std::array::from_fn(|form| {
                clause.words[form]
                    .iter()
                    .map(|word| {
                        let df = store.postings[form].get(word).map_or(0, |l| l.len()) as f64;
                        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
                    })
                    .sum()
            })
        })
        .collect()
}

fn matches(query: &LexicalQuery, frequencies: &[u32]) -> bool {
    let mut required = false;
    let mut optional = false;
    for (clause, &frequency) in query.clauses.iter().zip(frequencies) {
        match clause.occur {
            Occur::Must => {
                required = true;
                if frequency == 0 {
                    return false;
                }
            }
            Occur::MustNot if frequency != 0 => return false,
            Occur::Should => optional |= frequency != 0,
            _ => {}
        }
    }
    required || optional
}

fn score(
    query: &LexicalQuery,
    frequencies: &[u32],
    idfs: &[[f64; 2]],
    form: usize,
    norm: f64,
    bm25: Bm25,
) -> Result<f64> {
    let mut score = 0.0;
    for ((clause, &frequency), idf) in query.clauses.iter().zip(frequencies).zip(idfs) {
        if clause.occur != Occur::MustNot && frequency > 0 {
            score += bm25.weight(idf[form], f64::from(frequency), norm);
        }
    }
    ensure!(
        score.is_finite(),
        "lexical score is nonfinite; reduce BM25 parameters"
    );
    Ok(score)
}

fn order_hits(a: &LexicalHit, b: &LexicalHit) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.document_id.cmp(&b.document_id))
        .then_with(|| a.chunk_index.cmp(&b.chunk_index))
}

pub(crate) fn execute(
    store: &DocumentStore,
    query: &LexicalQuery,
    options: &LexicalOptions,
) -> Result<LexicalResults> {
    let bm25 = prepare(store, query, options)?;
    if options.execution != LexicalExecution::Exhaustive
        && options.k == 0
        && options.count
        && query
            .clauses
            .iter()
            .all(|clause| clause.occur == Occur::Should && !clause.phrase)
    {
        return Ok(LexicalResults {
            hits: Vec::new(),
            count: Some(execution::count_union(store, query)),
        });
    }
    match options.execution {
        LexicalExecution::Exhaustive => exhaustive(store, query, options, bm25),
        LexicalExecution::Auto | LexicalExecution::BlockMaxWand
            if options.k > 0
                && (options.execution == LexicalExecution::BlockMaxWand
                    || (query.clauses.len() <= 16 && options.k <= 1000)) =>
        {
            match wand::execute(store, query, options, bm25)? {
                Some(result) => Ok(result),
                None => document_at_a_time(store, query, options, bm25),
            }
        }
        _ => document_at_a_time(store, query, options, bm25),
    }
}

fn exhaustive(
    store: &DocumentStore,
    query: &LexicalQuery,
    options: &LexicalOptions,
    bm25: Bm25,
) -> Result<LexicalResults> {
    let mut hits = Vec::new();
    let mut matched = BTreeSet::new();
    if !query.has_positive() {
        return Ok(LexicalResults {
            hits,
            count: options.count.then_some(0),
        });
    }
    let idfs = if options.k > 0 {
        clause_idfs(store, query)
    } else {
        Vec::new()
    };
    let average = (store.word_count as f64 / (store.chunk_count as f64).max(1.0)).max(1.0);
    for (id, document) in &store.documents {
        for (chunk_index, chunk) in document.chunks.iter().enumerate() {
            let frequencies: Vec<_> = query
                .clauses
                .iter()
                .map(|clause| frequency(clause, chunk))
                .collect();
            if !matches(query, &frequencies) {
                continue;
            }
            if options.count {
                matched.insert(id);
            }
            if options.k > 0 {
                hits.push(LexicalHit {
                    document_id: id.clone(),
                    chunk_index,
                    score: score(
                        query,
                        &frequencies,
                        &idfs,
                        usize::from(chunk.stemmed),
                        bm25.norm(chunk.word_count as f64, average),
                        bm25,
                    )?,
                });
            }
        }
    }
    hits.sort_unstable_by(order_hits);
    hits.truncate(options.k);
    Ok(LexicalResults {
        hits,
        count: options.count.then_some(matched.len()),
    })
}

pub(super) mod execution;
mod wand;
use execution::document_at_a_time;

#[cfg(test)]
mod tests;
