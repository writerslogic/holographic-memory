// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{hash_map::Entry, HashMap, HashSet};

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use super::{Embedder, FactExtractor, Generator, QueryRewriter, Reranker};
use crate::core::documents::DocumentResult;

const MAX_SEARCH_RESULTS: usize = 1000;

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

impl StageParams {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.k0,
                self.orig_weight,
                self.rewrite_weight,
                self.lexical_weight,
                self.semantic_weight,
                self.rerank_weight,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
            "model stage constants and weights must be finite and nonnegative"
        );
        let query_weight = self.orig_weight + self.rewrite_weight;
        ensure!(
            query_weight.is_finite() && query_weight > 0.0,
            "model query weights require a finite positive sum"
        );
        let retrieval_weight = self.lexical_weight + self.semantic_weight;
        ensure!(
            retrieval_weight.is_finite() && retrieval_weight > 0.0,
            "model retrieval weights require a finite positive sum"
        );
        ensure!(
            self.rerank_n <= MAX_SEARCH_RESULTS,
            "model rerank_n must not exceed the native 1000-result search limit"
        );
        Ok(())
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
        p.validate()?;
        ensure!(
            (1..=MAX_SEARCH_RESULTS).contains(&k),
            "model fusion requires 1 <= k <= 1000"
        );
        let mut fused: HashMap<String, (f64, usize, DocumentResult)> = HashMap::new();
        for (weight, list) in lists {
            ensure!(
                weight.is_finite() && weight >= 0.0,
                "model ranked-list weights must be finite and nonnegative"
            );
            ensure!(
                list.len() <= MAX_SEARCH_RESULTS,
                "model ranked lists must not exceed the native 1000-result search limit"
            );
            let mut seen = HashSet::with_capacity(list.len());
            for (rank, r) in list.into_iter().enumerate() {
                ensure!(
                    seen.insert(r.id.clone()),
                    "duplicate chunk identity in a model ranked list"
                );
                let next = fused.len();
                match fused.entry(r.id.clone()) {
                    Entry::Occupied(mut entry) => {
                        let (score, _, source) = entry.get_mut();
                        ensure!(
                            same_source(source, &r),
                            "conflicting source identity for a model-fused chunk"
                        );
                        *score = add_rank_score(*score, weight, p.k0, rank)?;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert((add_rank_score(0.0, weight, p.k0, rank)?, next, r));
                    }
                }
            }
        }
        let mut all: Vec<(f64, usize, DocumentResult)> = fused.into_values().collect();
        let order = |a: &(f64, usize, DocumentResult), b: &(f64, usize, DocumentResult)| {
            b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
        };
        all.sort_by(order);
        if let Some(reranker) = self
            .reranker
            .as_ref()
            .filter(|_| p.rerank_weight > 0.0 && p.rerank_n > 0)
        {
            let head: Vec<usize> = (0..all.len().min(p.rerank_n))
                .filter(|&i| all[i].2.text.is_some())
                .collect();
            let pairs: Vec<(&str, &str)> = head
                .iter()
                .map(|&i| (question, all[i].2.text.as_deref().unwrap_or_default()))
                .collect();
            let scores = reranker.score(&pairs)?;
            ensure!(
                scores.len() == head.len() && scores.iter().all(|score| score.is_finite()),
                "model reranker returned invalid scores"
            );
            let mut by_score: Vec<usize> = (0..head.len()).collect();
            by_score.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            for (r, &j) in by_score.iter().enumerate() {
                all[head[j]].0 = add_rank_score(all[head[j]].0, p.rerank_weight, p.k0, r)?;
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

fn same_source(a: &DocumentResult, b: &DocumentResult) -> bool {
    a.document_id == b.document_id
        && a.version == b.version
        && a.source_uri == b.source_uri
        && a.metadata == b.metadata
        && a.text == b.text
        && a.start_byte == b.start_byte
        && a.end_byte == b.end_byte
}

fn add_rank_score(score: f64, weight: f64, k0: f64, rank: usize) -> Result<f64> {
    let contribution = weight / (k0 + rank as f64 + 1.0);
    let fused = score + contribution;
    ensure!(
        contribution.is_finite() && fused.is_finite(),
        "model rank fusion produced a nonfinite score"
    );
    Ok(fused)
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
            end_byte: id.len() as u32,
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
        assert_eq!(out[0].score, want);
    }

    #[test]
    fn stage_parameters_reject_invalid_constants_and_weight_sums() {
        let setters: [fn(&mut StageParams, f64); 6] = [
            |p, value| p.k0 = value,
            |p, value| p.orig_weight = value,
            |p, value| p.rewrite_weight = value,
            |p, value| p.lexical_weight = value,
            |p, value| p.semantic_weight = value,
            |p, value| p.rerank_weight = value,
        ];
        for set in setters {
            for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut params = StageParams::default();
                set(&mut params, value);
                assert!(params.validate().is_err());
                let stages = ModelStages {
                    params,
                    ..Default::default()
                };
                assert!(stages.fuse("q", vec![(1.0, vec![result("x")])], 1).is_err());
            }
        }
        for weights in [(0.0, 0.0), (f64::MAX, f64::MAX)] {
            let params = StageParams {
                orig_weight: weights.0,
                rewrite_weight: weights.1,
                ..Default::default()
            };
            assert!(params.validate().is_err());
            let params = StageParams {
                lexical_weight: weights.0,
                semantic_weight: weights.1,
                ..Default::default()
            };
            assert!(params.validate().is_err());
        }
        StageParams {
            k0: 0.0,
            orig_weight: 0.0,
            lexical_weight: 0.0,
            rerank_weight: 0.0,
            rerank_n: 0,
            ..Default::default()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn native_result_count_boundaries_apply_to_params_and_fusion() {
        for count in [0, 1, 999, 1000] {
            let params = StageParams {
                rerank_n: count,
                ..Default::default()
            };
            params.validate().unwrap();
        }
        for count in [1001, usize::MAX] {
            let params = StageParams {
                rerank_n: count,
                ..Default::default()
            };
            assert!(params.validate().is_err());
        }
        let stages = ModelStages::default();
        for count in [1, 999, 1000] {
            let list: Vec<_> = (0..count).map(|i| result(&format!("chunk-{i}"))).collect();
            let out = stages.fuse("q", vec![(1.0, list)], count).unwrap();
            assert_eq!(out.len(), count);
            assert_eq!(out[0].id, "chunk-0");
            assert_eq!(out[count - 1].score, 1.0 / (60.0 + count as f64));
        }
        for k in [0, 1001, usize::MAX] {
            assert!(stages.fuse("q", vec![(1.0, vec![result("x")])], k).is_err());
        }
        let list: Vec<_> = (0..1001).map(|i| result(&format!("chunk-{i}"))).collect();
        assert!(stages.fuse("q", vec![(1.0, list)], 1000).is_err());
    }

    #[test]
    fn fusion_rejects_invalid_list_weights_and_duplicate_chunks() {
        let stages = ModelStages::default();
        for weight in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(stages
                .fuse("q", vec![(weight, vec![result("x")])], 1)
                .is_err());
        }
        for weight in [0.0, 1.0] {
            assert!(stages
                .fuse("q", vec![(weight, vec![result("x"), result("x")])], 1)
                .is_err());
        }
        let out = stages.fuse("q", vec![(0.0, vec![result("x")])], 1).unwrap();
        assert_eq!(out[0].score, 0.0);
    }

    #[test]
    fn fusion_rejects_conflicting_source_identity_across_lists() {
        let changes: [fn(&mut DocumentResult); 7] = [
            |r| r.document_id = "other-document".into(),
            |r| r.version = "2".into(),
            |r| r.source_uri = Some("urn:other-source".into()),
            |r| r.metadata = serde_json::json!({"different": true}),
            |r| r.text = Some("changed source text".into()),
            |r| r.start_byte = 1,
            |r| r.end_byte += 1,
        ];
        let stages = ModelStages::default();
        for change in changes {
            let original = result("x");
            let mut conflicting = original.clone();
            change(&mut conflicting);
            assert!(stages
                .fuse(
                    "q",
                    vec![(1.0, vec![original]), (1.0, vec![conflicting])],
                    1
                )
                .is_err());
        }
    }

    #[test]
    fn same_source_merges_despite_query_scores_and_ties_stay_deterministic() {
        let stages = ModelStages::default();
        let first = result("x");
        let mut second = first.clone();
        second.score = 99.0;
        second.lexical_score = 12.0;
        second.semantic_score = Some(0.75);
        let out = stages
            .fuse("q", vec![(1.0, vec![first]), (1.0, vec![second])], 1)
            .unwrap();
        assert_eq!(out[0].score, 1.0 / 61.0 + 1.0 / 61.0);
        assert_eq!(out[0].lexical_score, 0.0);
        assert_eq!(out[0].semantic_score, None);
        for _ in 0..3 {
            let out = stages
                .fuse(
                    "q",
                    vec![
                        (1.0, vec![result("x"), result("y")]),
                        (1.0, vec![result("y"), result("x")]),
                    ],
                    2,
                )
                .unwrap();
            assert_eq!(out[0].id, "x");
            assert_eq!(out[1].id, "y");
            assert_eq!(out[0].score, out[1].score);
        }
    }

    #[test]
    fn finite_weights_cannot_overflow_fused_scores() {
        let stages = ModelStages {
            params: StageParams {
                k0: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let out = stages
            .fuse("q", vec![(f64::MAX, vec![result("x")])], 1)
            .unwrap();
        assert_eq!(out[0].score, f64::MAX);
        assert!(stages
            .fuse(
                "q",
                vec![(f64::MAX, vec![result("x")]), (f64::MAX, vec![result("x")]),],
                1,
            )
            .is_err());
    }
}
