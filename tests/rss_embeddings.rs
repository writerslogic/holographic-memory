// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `f32` embedding storage: every semantic score is the exact `f64` cosine of the stored
//! `f32` vector (within 1e-6 of the cosine of the client's `f64` vector), the generic search
//! path and the vector snapshot report the same scores, the scores survive a reopen (the
//! journal carries the `f32` form), and the `lexical` ingest mode (no hypervector channel)
//! leaves vector search unchanged. With the `server` feature, `--ingest-encode lexical` is
//! exercised through `hms-server`.

use holographic_memory::core::documents::{
    EncodeMode, IngestOptions, VectorIndexMode, VectorQueryOptions,
};
use holographic_memory::core::HmsConfig;
use holographic_memory::{DocumentInput, DocumentResult, EmbeddingSpace, HmsCore, SearchOptions};
use serde_json::json;

#[cfg(feature = "server")]
mod server_support;

const DIM: usize = 64;
const DOCS: usize = 300;
const QUERIES: usize = 20;
const K: usize = 10;

/// 64-bit LCG; values uniform in (-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 52) as f64) - 1.0
    }

    fn vector(&mut self) -> Vec<f64> {
        (0..DIM).map(|_| self.next()).collect()
    }
}

/// The engine's normalisation (`documents::normalize_embedding`): scale by the largest
/// magnitude, then divide by the l2 norm, all in `f64`.
fn normalize(values: &[f64]) -> Vec<f64> {
    let scale = values.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let norm = values
        .iter()
        .map(|x| (x / scale).powi(2))
        .sum::<f64>()
        .sqrt();
    values.iter().map(|x| (x / scale) / norm).collect()
}

/// The cosine of the `f64` vectors and the score the store must report: the sequential `f64`
/// dot of the normalised query with the normalised document rounded to `f32` and widened.
fn reference(query: &[f64], document: &[f64]) -> (f64, f64) {
    let q = normalize(query);
    let d = normalize(document);
    let exact = q.iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
    let stored = q
        .iter()
        .zip(&d)
        .map(|(a, &b)| a * f64::from(b as f32))
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    (exact.clamp(-1.0, 1.0), stored)
}

fn corpus() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = Rng(0x05EE_DF32_0001);
    let documents = (0..DOCS).map(|_| rng.vector()).collect();
    let queries = (0..QUERIES).map(|_| rng.vector()).collect();
    (documents, queries)
}

fn inputs(documents: &[Vec<f64>]) -> Vec<DocumentInput> {
    documents
        .iter()
        .enumerate()
        .map(|(i, embedding)| {
            serde_json::from_value(json!({
                "id": format!("d{i:04}"),
                "text": format!("document {i} about topic {}", i % 7),
                "metadata": {"i": i},
                "embeddings": [embedding],
            }))
            .unwrap()
        })
        .collect()
}

fn open(dir: &std::path::Path) -> HmsCore {
    let config = HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "test".into(),
            revision: "1".into(),
            dimensions: DIM,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..HmsConfig::default()
    };
    HmsCore::new(1024, Some(dir.display().to_string()), Some(config)).unwrap()
}

fn generic(core: &HmsCore, query: &[f64]) -> Vec<DocumentResult> {
    let options = SearchOptions {
        k: Some(K as u32),
        candidate_limit: Some(100),
        embedding: Some(query.to_vec()),
        lexical_weight: Some(0.0),
        semantic_weight: Some(1.0),
        min_semantic_score: Some(-1.0),
        ..SearchOptions::default()
    };
    core.search_documents("", &options).unwrap()
}

fn snapshot(core: &HmsCore, query: &[f64]) -> Vec<DocumentResult> {
    let options = VectorQueryOptions {
        mode: VectorIndexMode::Exact,
        ..VectorQueryOptions::default()
    };
    core.search_vectors(query, K, &options).unwrap()
}

fn ids_and_scores(hits: &[DocumentResult], semantic: bool) -> Vec<(String, f64)> {
    hits.iter()
        .map(|hit| {
            let score = if semantic {
                hit.semantic_score.unwrap()
            } else {
                hit.score
            };
            (hit.document_id.clone(), score)
        })
        .collect()
}

/// The `K` best documents by the stored-`f32` reference score, with the `f64` cosine.
fn expected(documents: &[Vec<f64>], query: &[f64]) -> Vec<(String, f64, f64)> {
    let mut all: Vec<(String, f64, f64)> = documents
        .iter()
        .enumerate()
        .map(|(i, document)| {
            let (exact, stored) = reference(query, document);
            (format!("d{i:04}"), stored, exact)
        })
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(K);
    all
}

#[test]
fn f32_storage_scores_are_exact_for_the_stored_vector_and_within_1e6_of_f64() {
    let (documents, queries) = corpus();
    let dir = tempfile::tempdir().unwrap();
    let core = open(dir.path());
    core.memorize_documents(inputs(&documents)).unwrap();
    for query in &queries {
        let want = expected(&documents, query);
        let got_generic = ids_and_scores(&generic(&core, query), true);
        let got_snapshot = ids_and_scores(&snapshot(&core, query), false);
        assert_eq!(got_generic.len(), K);
        assert_eq!(
            got_generic, got_snapshot,
            "generic and snapshot paths differ"
        );
        for ((id, score), (want_id, stored, exact)) in got_generic.iter().zip(&want) {
            assert_eq!(id, want_id);
            assert!(
                (score - stored).abs() <= 1e-12,
                "{id}: {score} is not the f64 cosine of the stored f32 vector {stored}"
            );
            assert!(
                (score - exact).abs() < 1e-6,
                "{id}: {score} is more than 1e-6 from the f64 cosine {exact}"
            );
        }
    }
    for hit in snapshot(&core, &queries[0]) {
        assert_eq!(hit.semantic_score, Some(hit.score));
    }
    // A reopen replays the journal: the f32 form on disk gives the same scores.
    let before = ids_and_scores(&generic(&core, &queries[1]), true);
    drop(core);
    let reopened = open(dir.path());
    assert_eq!(
        ids_and_scores(&generic(&reopened, &queries[1]), true),
        before
    );
    assert_eq!(
        ids_and_scores(&snapshot(&reopened, &queries[1]), false),
        before
    );
}

#[test]
fn lexical_ingest_skips_the_hypervector_channel_without_changing_vector_search() {
    let (documents, queries) = corpus();
    let eager_dir = tempfile::tempdir().unwrap();
    let lexical_dir = tempfile::tempdir().unwrap();
    let eager = open(eager_dir.path());
    let lexical = open(lexical_dir.path());
    eager
        .memorize_documents_with(
            inputs(&documents),
            &IngestOptions {
                encode: EncodeMode::Eager,
                ..Default::default()
            },
        )
        .unwrap();
    lexical
        .memorize_documents_with(
            inputs(&documents),
            &IngestOptions {
                encode: EncodeMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap();
    for query in &queries {
        assert_eq!(
            ids_and_scores(&snapshot(&eager, query), false),
            ids_and_scores(&snapshot(&lexical, query), false)
        );
        assert_eq!(
            ids_and_scores(&generic(&eager, query), true),
            ids_and_scores(&generic(&lexical, query), true)
        );
    }
    // The lexical side is untouched: a text query finds the same documents either way.
    let text = SearchOptions {
        k: Some(5),
        ..SearchOptions::default()
    };
    assert_eq!(
        ids_and_scores(&eager.search_documents("topic 3", &text).unwrap(), false),
        ids_and_scores(&lexical.search_documents("topic 3", &text).unwrap(), false)
    );
}

#[cfg(feature = "server")]
#[test]
fn server_ingest_encode_lexical_serves_exact_vector_queries() {
    use server_support::*;
    const TENANT: &str = "rss";
    let s = Server::open(&["--ingest-encode", "lexical"]);
    let seeds = [11u64, 12, 13];
    let batch: Vec<serde_json::Value> = seeds
        .iter()
        .map(|seed| json!({"id": format!("s{seed}"), "text": format!("seed {seed}"), "vector": embedding(*seed), "metadata": {}}))
        .collect();
    let added = s
        .req("POST", "/api/v1/documents/batch")
        .tenant(TENANT)
        .json(&json!(batch))
        .send();
    assert_eq!(added.status, 200, "{}", added.text());
    let query = embedding(14);
    let queried = s
        .req("POST", "/api/v1/query")
        .tenant(TENANT)
        .json(&json!({"query_vector": query, "top_k": 3}))
        .send();
    assert_eq!(queried.status, 200, "{}", queried.text());
    let reply = queried.json();
    let mut want: Vec<(String, f64, f64)> = seeds
        .iter()
        .map(|seed| {
            let (exact, stored) = reference(&query, &embedding(*seed));
            (format!("s{seed}"), stored, exact)
        })
        .collect();
    want.sort_by(|a, b| b.1.total_cmp(&a.1));
    let matches = reply["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 3);
    for (hit, (id, stored, exact)) in matches.iter().zip(&want) {
        assert_eq!(hit["id"].as_str().unwrap(), id);
        let score = hit["score"].as_f64().unwrap();
        assert!((score - stored).abs() <= 1e-12, "{id}: {score} vs {stored}");
        assert!((score - exact).abs() < 1e-6, "{id}: {score} vs {exact}");
    }
    let stats = s.req("GET", "/api/v1/stats").tenant(TENANT).send();
    assert_eq!(stats.status, 200, "{}", stats.text());
    assert_eq!(stats.json()["vectors"].as_u64(), Some(3));
}
