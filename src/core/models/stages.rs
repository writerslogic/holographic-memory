// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{Embedder, FactExtractor, Generator, QueryRewriter, Reranker};
use crate::core::documents::DocumentResult;

/// Fusion parameters of the document API's model stages. Defaults are the tuned LongMemEval
/// turn-level configuration (`benchmarks/public/longmemeval_config.json`): fact-expanded keys,
/// original and rewritten query merged with equal weight, lexical 0.5 / semantic 1.0 inside each
/// list, re-rank of the fused top 20 added with weight 8, RRF constant 60.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct StageParams {
    pub k0: f64,
    pub orig_weight: f64,
    pub rewrite_weight: f64,
    pub lexical_weight: f64,
    pub semantic_weight: f64,
    pub rerank_weight: f64,
    pub rerank_n: usize,
}

impl Default for StageParams {
    fn default() -> Self {
        Self {
            k0: 60.0,
            orig_weight: 1.0,
            rewrite_weight: 1.0,
            lexical_weight: 0.5,
            semantic_weight: 1.0,
            rerank_weight: 8.0,
            rerank_n: 20,
        }
    }
}

/// The on-device stages a store uses; each is optional and absent stages leave the document
/// API's behaviour unchanged.
#[derive(Default)]
pub struct ModelStages {
    pub embedder: Option<Embedder>,
    pub reranker: Option<Reranker>,
    /// LLM for fact extraction at ingest and query rewriting at search.
    pub generator: Option<Generator>,
    pub extract_facts: bool,
    pub rewrite_queries: bool,
    pub params: StageParams,
}

impl ModelStages {
    /// Lexical/semantic key per chunk: the chunk's own facts, then its text (the tuned
    /// `turn_exp` key). `None` when fact extraction is off.
    pub fn fact_keys(&self, chunks: &[String]) -> Result<Option<Vec<String>>> {
        let Some(generator) = self.generator.as_ref().filter(|_| self.extract_facts) else {
            return Ok(None);
        };
        let (_, facts) = FactExtractor(generator).extract(chunks)?;
        Ok(Some(expand_with_facts(chunks, &facts)))
    }

    /// The query variants searched: the original, then the LLM rewrite when enabled.
    pub fn query_variants(&self, question: &str, today: &str) -> Result<Vec<(String, f64)>> {
        let mut out = vec![(question.to_string(), self.params.orig_weight)];
        if let Some(generator) = self.generator.as_ref().filter(|_| self.rewrite_queries) {
            if let Some(r) = QueryRewriter(generator).rewrite(today, question)?.1.rewrite {
                out.push((r, self.params.rewrite_weight));
            }
        }
        Ok(out)
    }

    /// Weighted RRF of the variants' ranked lists, then the re-ranker's ranks over the fused
    /// head added with `rerank_weight` (the reference `rank_question`). Truncated to `k`.
    pub fn fuse(
        &self,
        question: &str,
        lists: Vec<(f64, Vec<DocumentResult>)>,
        k: usize,
    ) -> Result<Vec<DocumentResult>> {
        let p = &self.params;
        let mut fused: HashMap<String, (f64, usize, DocumentResult)> = HashMap::new();
        for (weight, list) in lists {
            for (rank, r) in list.into_iter().enumerate() {
                let add = weight / (p.k0 + rank as f64 + 1.0);
                let next = fused.len();
                let e = fused.entry(r.id.clone()).or_insert_with(|| (0.0, next, r));
                e.0 += add;
            }
        }
        let mut all: Vec<(f64, usize, DocumentResult)> = fused.into_values().collect();
        let order = |a: &(f64, usize, DocumentResult), b: &(f64, usize, DocumentResult)| {
            b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
        };
        all.sort_by(order);
        if let Some(reranker) = self.reranker.as_ref().filter(|_| p.rerank_weight > 0.0) {
            let head: Vec<usize> = (0..all.len().min(p.rerank_n))
                .filter(|&i| all[i].2.text.is_some())
                .collect();
            let pairs: Vec<(&str, &str)> = head
                .iter()
                .map(|&i| (question, all[i].2.text.as_deref().unwrap_or_default()))
                .collect();
            let scores = reranker.score(&pairs)?;
            let mut by_score: Vec<usize> = (0..head.len()).collect();
            by_score.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            for (r, &j) in by_score.iter().enumerate() {
                all[head[j]].0 += p.rerank_weight / (p.k0 + r as f64 + 1.0);
            }
            all.sort_by(order);
        }
        Ok(all
            .into_iter()
            .take(k)
            .map(|(score, _, mut r)| {
                r.score = score;
                r
            })
            .collect())
    }
}

/// `turn_exp` keys: facts attributed to chunk i, then the chunk text, space-joined.
fn expand_with_facts(chunks: &[String], facts: &[(Option<usize>, String)]) -> Vec<String> {
    chunks
        .iter()
        .enumerate()
        .map(|(i, text)| {
            facts
                .iter()
                .filter(|(t, _)| *t == Some(i))
                .map(|(_, f)| f.as_str())
                .chain([text.as_str()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(id: &str) -> DocumentResult {
        DocumentResult {
            id: id.into(),
            document_id: id.into(),
            source_uri: None,
            version: "1".into(),
            metadata: serde_json::json!({}),
            text: Some(id.into()),
            start_byte: 0,
            end_byte: 1,
            score: 0.0,
            lexical_score: 0.0,
            semantic_score: None,
        }
    }

    #[test]
    fn facts_expand_their_own_chunk_only() {
        let chunks = vec!["a".to_string(), "b".to_string()];
        let facts = vec![
            (Some(1), "F1".to_string()),
            (None, "S".to_string()),
            (Some(1), "F2".to_string()),
        ];
        assert_eq!(expand_with_facts(&chunks, &facts), vec!["a", "F1 F2 b"]);
    }

    #[test]
    fn variant_lists_merge_by_weighted_rrf() {
        let stages = ModelStages::default();
        let lists = vec![
            (1.0, vec![result("x"), result("y")]),
            (1.0, vec![result("y"), result("z")]),
        ];
        let out = stages.fuse("q", lists, 2).unwrap();
        let ids: Vec<&str> = out.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["y", "x"]);
        let want = 1.0 / 62.0 + 1.0 / 61.0;
        assert!((out[0].score - want).abs() < 1e-12);
    }
}
