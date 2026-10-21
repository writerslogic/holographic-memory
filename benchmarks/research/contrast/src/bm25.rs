// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer, stemmer and a small BM25 index over stem ids.
//!
//! # Tokenizer / stemmer
//! `stems` copies the root crate's `terms` (src/core/documents.rs:388-403):
//! split on non-alphanumeric chars, lowercase, English Snowball (Porter 2)
//! stem via `rust_stemmers` (src/core/documents.rs:381-384). The only change
//! is that the ordered sequence is kept instead of a `BTreeMap` of counts,
//! because the sketch needs order. The root function is `pub(crate)` and is
//! not callable from this crate.
//!
//! # BM25
//! Constants follow the root scorer's defaults (src/core/documents.rs:116-121):
//! `k1 = 1.2`, `b = 0.75`, weight `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg))`
//! (src/core/documents.rs:147-156). `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`
//! (Lucene/BEIR form; the root's idf was not read). Query terms are weighted by
//! their query frequency. This is an own implementation over the same stems:
//! the root's document search is only reachable through `HmsCore` and was not
//! used.
//!
//! # Data layout
//! Terms are the 64-bit atom seeds of stems (`hv::stem_seed`), so the BM25
//! index, the sketches and the pair tables share one id space. Postings are
//! `term -> Vec<(doc, tf)>`; per-document term maps support pairwise scoring
//! of an arbitrary candidate list.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::hv::stem_seed;

/// `k1` of the root scorer's default.
pub const K1: f64 = 1.2;
/// `b` of the root scorer's default.
pub const B: f64 = 0.75;

fn stemmer() -> &'static rust_stemmers::Stemmer {
    static STEMMER: OnceLock<rust_stemmers::Stemmer> = OnceLock::new();
    STEMMER.get_or_init(|| rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English))
}

/// Ordered lowercase stems of `text` (root tokenizer rules, order preserved).
pub fn stems(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|word| {
            let lower = word.to_lowercase();
            stemmer().stem(&lower).into_owned()
        })
        .collect()
}

/// Ordered stem ids (atom seeds) of `text`.
pub fn seeds(text: &str) -> Vec<u64> {
    stems(text).iter().map(|s| stem_seed(s)).collect()
}

/// Term frequencies of an id sequence.
pub fn counts(seq: &[u64]) -> HashMap<u64, u32> {
    let mut m: HashMap<u64, u32> = HashMap::with_capacity(seq.len());
    for &t in seq {
        *m.entry(t).or_default() += 1;
    }
    m
}

/// Multiset overlap `sum_t min(q_t, d_t)` divided by the query length (raw unigram overlap).
pub fn unigram_overlap(query: &HashMap<u64, u32>, query_len: u32, doc: &HashMap<u64, u32>) -> f64 {
    if query_len == 0 {
        return 0.0;
    }
    let shared: u32 = query
        .iter()
        .map(|(t, &qc)| doc.get(t).map_or(0, |&dc| qc.min(dc)))
        .sum();
    f64::from(shared) / f64::from(query_len)
}

/// BM25 index over stem ids.
pub struct Bm25Index {
    postings: HashMap<u64, Vec<(u32, u32)>>,
    doc_terms: Vec<HashMap<u64, u32>>,
    doc_len: Vec<u32>,
    avg_len: f64,
}

impl Bm25Index {
    /// Builds the index; `docs[i]` is document `i`'s ordered stem ids.
    pub fn build(docs: &[Vec<u64>]) -> Self {
        let mut postings: HashMap<u64, Vec<(u32, u32)>> = HashMap::new();
        let mut doc_terms = Vec::with_capacity(docs.len());
        let mut doc_len = Vec::with_capacity(docs.len());
        let mut total_len = 0u64;
        for (i, doc) in docs.iter().enumerate() {
            let c = counts(doc);
            for (&t, &tf) in &c {
                postings.entry(t).or_default().push((i as u32, tf));
            }
            doc_terms.push(c);
            doc_len.push(doc.len() as u32);
            total_len += doc.len() as u64;
        }
        for list in postings.values_mut() {
            list.sort_unstable();
        }
        let avg_len = if docs.is_empty() {
            1.0
        } else {
            (total_len as f64 / docs.len() as f64).max(1.0)
        };
        Self {
            postings,
            doc_terms,
            doc_len,
            avg_len,
        }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_len.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.doc_len.is_empty()
    }

    /// Inverse document frequency of a term.
    #[inline]
    pub fn idf(&self, term: u64) -> f64 {
        let n = self.len() as f64;
        let df = self.postings.get(&term).map_or(0.0, |l| l.len() as f64);
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    #[inline]
    fn norm(&self, doc: u32) -> f64 {
        K1 * (1.0 - B + B * f64::from(self.doc_len[doc as usize]) / self.avg_len)
    }

    /// BM25 of `query` (term -> query frequency) against one document.
    pub fn score_doc(&self, query: &HashMap<u64, u32>, doc: u32) -> f64 {
        let terms = &self.doc_terms[doc as usize];
        let norm = self.norm(doc);
        let mut s = 0.0;
        for (&t, &qf) in query {
            if let Some(&tf) = terms.get(&t) {
                let tf = f64::from(tf);
                s += f64::from(qf) * self.idf(t) * tf * (K1 + 1.0) / (tf + norm);
            }
        }
        s
    }

    /// BM25 of every document for `query`; documents sharing no term score 0.
    pub fn score_all(&self, query: &HashMap<u64, u32>) -> Vec<f64> {
        let mut scores = vec![0.0f64; self.len()];
        for (&t, &qf) in query {
            let Some(list) = self.postings.get(&t) else {
                continue;
            };
            let idf = self.idf(t);
            for &(doc, tf) in list {
                let tf = f64::from(tf);
                scores[doc as usize] += f64::from(qf) * idf * tf * (K1 + 1.0) / (tf + self.norm(doc));
            }
        }
        scores
    }

    /// Top-`k` documents by BM25 (score desc, doc id asc), excluding zero scores.
    pub fn top_k(&self, query: &HashMap<u64, u32>, k: usize) -> Vec<(u32, f64)> {
        let scores = self.score_all(query);
        let mut ranked: Vec<(u32, f64)> = scores
            .into_iter()
            .enumerate()
            .filter(|&(_, s)| s > 0.0)
            .map(|(d, s)| (d as u32, s))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k);
        ranked
    }

    /// Term counts of a document.
    pub fn doc_terms(&self, doc: u32) -> &HashMap<u64, u32> {
        &self.doc_terms[doc as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_follow_root_rules() {
        assert_eq!(stems("Statins, Running 1,000!"), vec!["statin", "run", "1", "000"]);
    }

    #[test]
    fn score_doc_matches_score_all() {
        let docs: Vec<Vec<u64>> = ["alice warned bob", "bob was warned by alice twice", "carol paid dan"]
            .iter()
            .map(|t| seeds(t))
            .collect();
        let idx = Bm25Index::build(&docs);
        let q = counts(&seeds("warned alice"));
        let all = idx.score_all(&q);
        for d in 0..docs.len() {
            assert!((all[d] - idx.score_doc(&q, d as u32)).abs() < 1e-12);
        }
        assert!(all[0] > all[2]);
    }
}
