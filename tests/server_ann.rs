// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Vector query routes of `hms-server`: the graph index and the exact scan agree on a fixed
//! synthetic corpus, deletes and replaces are visible before and after the next maintenance,
//! and the route options validate.

#![cfg(feature = "server")]

mod server_support;

use serde_json::{json, Value};
use server_support::*;

const DOCS: usize = 3000;
const CLUSTERS: usize = 24;
const QUERIES: usize = 40;
const K: usize = 10;
const TENANT: &str = "ann";

/// Deterministic generator (64-bit LCG, Box-Muller normals).
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(1e-12);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    fn vector(&mut self) -> Vec<f64> {
        (0..DIM).map(|_| self.normal()).collect()
    }
}

fn normalized(values: &[f64]) -> Vec<f64> {
    let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    values.iter().map(|v| v / norm).collect()
}

/// Gaussian clusters; queries are documents plus noise, so true neighbours exist.
fn corpus() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = Rng(0x5EED_2026);
    let centers: Vec<Vec<f64>> = (0..CLUSTERS).map(|_| rng.vector()).collect();
    let docs: Vec<Vec<f64>> = (0..DOCS)
        .map(|i| {
            let noise = rng.vector();
            let center = &centers[i % CLUSTERS];
            normalized(
                &center
                    .iter()
                    .zip(&noise)
                    .map(|(c, n)| c + 0.3 * n)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let queries: Vec<Vec<f64>> = (0..QUERIES)
        .map(|j| {
            let noise = rng.vector();
            let doc = &docs[(j * 73) % DOCS];
            normalized(
                &doc.iter()
                    .zip(&noise)
                    .map(|(d, n)| d + 0.05 * n)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    (docs, queries)
}

fn reference_top_k(docs: &[Vec<f64>], query: &[f64], k: usize) -> Vec<String> {
    let mut scored: Vec<(f64, usize)> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| (d.iter().zip(query).map(|(a, b)| a * b).sum::<f64>(), i))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored
        .iter()
        .take(k)
        .map(|(_, i)| format!("d{i}"))
        .collect()
}

fn document(id: &str, index: usize, vector: &[f64]) -> Value {
    json!({"id": id, "text": format!("document {index}"), "vector": vector, "metadata": {"i": index}})
}

fn ingest(server: &Server, docs: &[Vec<f64>]) {
    for (batch, chunk) in docs.chunks(1000).enumerate() {
        let body: Vec<Value> = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let index = batch * 1000 + i;
                document(&format!("d{index}"), index, v)
            })
            .collect();
        let resp = server
            .req("POST", "/api/v1/documents/batch")
            .tenant(TENANT)
            .json(&json!(body))
            .send();
        assert_eq!(resp.status, 200, "batch ingest: {}", resp.text());
    }
}

fn query(server: &Server, vector: &[f64], index: &str) -> (u16, Value) {
    let resp = server
        .req("POST", "/api/v1/query")
        .tenant(TENANT)
        .json(&json!({"query_vector": vector, "top_k": K, "index": index}))
        .send();
    (resp.status, resp.json())
}

fn ids(matches: &Value) -> Vec<String> {
    matches["matches"]
        .as_array()
        .unwrap_or_else(|| panic!("no matches in {matches}"))
        .iter()
        .map(|m| m["id"].as_str().expect("match id").to_string())
        .collect()
}

fn top_ids(server: &Server, vector: &[f64], index: &str) -> Vec<String> {
    let (status, body) = query(server, vector, index);
    assert_eq!(status, 200, "{index} query: {body}");
    ids(&body)
}

fn maintain(server: &Server) -> Value {
    let resp = server
        .req("POST", "/api/v1/index/maintain")
        .tenant(TENANT)
        .json(&json!({}))
        .send();
    assert_eq!(resp.status, 200, "maintain: {}", resp.text());
    resp.json()["vectorIndex"].clone()
}

fn vector_index_stats(server: &Server) -> Value {
    let resp = server.req("GET", "/api/v1/stats").tenant(TENANT).send();
    assert_eq!(resp.status, 200, "stats: {}", resp.text());
    resp.json()["vectorIndex"].clone()
}

fn overlap(a: &[String], b: &[String]) -> usize {
    a.iter().filter(|id| b.contains(id)).count()
}

#[test]
fn graph_and_exact_routes_agree_and_track_deletes_and_replaces() {
    let (docs, queries) = corpus();
    let server = Server::open(&["--graph-min-rows", "512"]);
    ingest(&server, &docs);

    // Before maintenance there is no snapshot: `graph` is refused, `exact` takes the generic
    // path and matches the local reference scan.
    let before = vector_index_stats(&server);
    assert_eq!(before["rows"], 0, "{before}");
    assert_eq!(before["graph"], false, "{before}");
    let (status, body) = query(&server, &queries[0], "graph");
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        top_ids(&server, &queries[0], "exact"),
        reference_top_k(&docs, &queries[0], K)
    );
    let (status, body) = query(&server, &queries[0], "bogus");
    assert_eq!(status, 422, "{body}");
    let resp = server
        .req("POST", "/api/v1/query")
        .tenant(TENANT)
        .json(&json!({"query_vector": queries[0], "top_k": K, "ef": 0}))
        .send();
    assert_eq!(resp.status, 422, "{}", resp.text());

    // Maintenance builds the snapshot with the graph (3000 rows >= --graph-min-rows 512).
    let built = maintain(&server);
    assert_eq!(built["rows"], DOCS, "{built}");
    assert_eq!(built["graph"], true, "{built}");
    assert_eq!(built["current"], true, "{built}");

    // The exact route over the snapshot equals the reference scan; the graph route recalls
    // at least 95% of it at the default ef.
    let mut found = 0;
    let mut exact_tops = Vec::with_capacity(QUERIES);
    for q in &queries {
        let exact = top_ids(&server, q, "exact");
        assert_eq!(exact, reference_top_k(&docs, q, K));
        let graph = top_ids(&server, q, "graph");
        assert_eq!(graph.len(), K);
        found += overlap(&graph, &exact);
        exact_tops.push(exact);
    }
    let recall = found as f64 / (QUERIES * K) as f64;
    assert!(recall >= 0.95, "graph recall@{K} {recall:.3} < 0.95");
    let auto = top_ids(&server, &queries[2], "auto");
    assert_eq!(auto.len(), K);
    let default_route = {
        let resp = server
            .req("POST", "/api/v1/query")
            .tenant(TENANT)
            .json(&json!({"query_vector": queries[2], "top_k": K}))
            .send();
        assert_eq!(resp.status, 200);
        ids(&resp.json())
    };
    assert_eq!(default_route, auto, "the default route is auto");

    // Delete the best match of query 0: both routes drop it at once (tombstone by identity).
    let victim = exact_tops[0][0].clone();
    let resp = server
        .req("DELETE", &format!("/api/v1/documents/{victim}"))
        .tenant(TENANT)
        .send();
    assert_eq!(resp.status, 200, "{}", resp.text());
    for route in ["graph", "exact", "auto"] {
        let got = top_ids(&server, &queries[0], route);
        assert!(!got.contains(&victim), "{route} still returns {victim}");
        assert_eq!(
            got.len(),
            K,
            "{route} returned {} after the delete",
            got.len()
        );
    }
    let stale = vector_index_stats(&server);
    assert_eq!(stale["current"], false, "{stale}");
    assert!(
        stale["staleMutations"].as_u64().unwrap_or(0) >= 1,
        "{stale}"
    );
    assert_eq!(stale["rows"], DOCS, "{stale}");

    // Replace d7 with query 1's vector: both routes find the new vector first (delta pass).
    let resp = server
        .req("POST", "/api/v1/documents")
        .tenant(TENANT)
        .json(&document("d7", 7, &queries[1]))
        .send();
    assert_eq!(resp.status, 200, "{}", resp.text());
    for route in ["graph", "exact", "auto"] {
        let (status, body) = query(&server, &queries[1], route);
        assert_eq!(status, 200, "{route}: {body}");
        let got = ids(&body);
        assert_eq!(got[0], "d7", "{route} top-1 after the replace: {body}");
        let score = body["matches"][0]["score"].as_f64().unwrap_or(0.0);
        assert!(
            score > 0.9999,
            "{route} score of the replaced vector: {score}"
        );
    }
    let reference = reference_top_k(
        &docs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if i == 7 {
                    queries[1].clone()
                } else {
                    d.clone()
                }
            })
            .collect::<Vec<_>>(),
        &queries[1],
        K + 1,
    );
    let reference: Vec<String> = reference
        .into_iter()
        .filter(|id| *id != victim)
        .take(K)
        .collect();
    assert_eq!(top_ids(&server, &queries[1], "exact"), reference);

    // Maintenance captures the mutated store; the fresh graph sees the same facts.
    let rebuilt = maintain(&server);
    assert_eq!(rebuilt["rows"], DOCS - 1, "{rebuilt}");
    assert_eq!(rebuilt["current"], true, "{rebuilt}");
    assert_eq!(rebuilt["staleMutations"], 0, "{rebuilt}");
    assert_eq!(rebuilt["graph"], true, "{rebuilt}");
    for route in ["graph", "exact"] {
        assert!(!top_ids(&server, &queries[0], route).contains(&victim));
        assert_eq!(top_ids(&server, &queries[1], route)[0], "d7");
    }
    assert_eq!(top_ids(&server, &queries[1], "exact"), reference);

    // A filtered vector query takes the generic path; `graph` is refused there explicitly.
    let resp = server
        .req("POST", "/api/v1/query")
        .tenant(TENANT)
        .json(&json!({"query_vector": queries[3], "top_k": K, "filter": {"i": 3}}))
        .send();
    assert_eq!(resp.status, 200, "{}", resp.text());
    assert_eq!(ids(&resp.json()), vec!["d3".to_string()]);
    let resp = server
        .req("POST", "/api/v1/query")
        .tenant(TENANT)
        .json(
            &json!({"query_vector": queries[3], "top_k": K, "filter": {"i": 3}, "index": "graph"}),
        )
        .send();
    assert_eq!(resp.status, 422, "{}", resp.text());
}
