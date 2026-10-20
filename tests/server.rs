// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "server")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};

const DIM: usize = 16;

struct Server {
    child: Child,
    addr: SocketAddr,
    _dir: Option<tempfile::TempDir>,
}

impl Server {
    fn start(extra: &[&str], api_key: Option<&str>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut server = Self::start_in(dir.path(), extra, api_key);
        server._dir = Some(dir);
        server
    }

    fn start_in(dir: &std::path::Path, extra: &[&str], api_key: Option<&str>) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_hms-server"));
        cmd.args(["--bind", "127.0.0.1:0", "--input-dim", &DIM.to_string()])
            .args(["--dim", "2048", "--data-dir"])
            .arg(dir)
            .args(extra)
            .env_remove("HMS_API_KEY")
            .stdout(Stdio::piped());
        match api_key {
            Some(key) => cmd.env("HMS_API_KEY", key),
            None => cmd.arg("--no-auth"),
        };
        let mut child = cmd.spawn().unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.as_mut().unwrap())
            .read_line(&mut line)
            .unwrap();
        let addr = line
            .trim()
            .rsplit(' ')
            .next()
            .and_then(|a| a.parse().ok())
            .unwrap_or_else(|| panic!("no listen address in {line:?}"));
        Self {
            child,
            addr,
            _dir: None,
        }
    }

    /// Returns (status, parsed JSON body or Null).
    fn request(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> (u16, Value) {
        let mut stream = TcpStream::connect(self.addr).unwrap();
        let mut req = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n",
            body.len()
        );
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        // The server may answer and close before reading an oversized body.
        let _ = stream.write_all(req.as_bytes());
        let _ = stream.write_all(body);
        let mut raw = Vec::new();
        let _ = stream.read_to_end(&mut raw);
        let text = String::from_utf8_lossy(&raw);
        let status = text
            .split(' ')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("no status in response {text:?}"));
        let body = text.split("\r\n\r\n").nth(1).unwrap_or("");
        (status, serde_json::from_str(body).unwrap_or(Value::Null))
    }

    fn json(&self, method: &str, path: &str, tenant: Option<&str>, body: &Value) -> (u16, Value) {
        let mut headers = vec![("Content-Type", "application/json")];
        if let Some(t) = tenant {
            headers.push(("X-Tenant-ID", t));
        }
        self.request(method, path, &headers, body.to_string().as_bytes())
    }

    fn stop(&mut self) -> bool {
        Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .unwrap();
        self.child.wait().unwrap().success()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Deterministic pseudo-random embedding for `seed`.
fn embedding(seed: u64) -> Vec<f64> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..DIM)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f64 / (1u64 << 30) as f64) - 1.0
        })
        .collect()
}

fn doc(id: &str, seed: u64, meta: Value) -> Value {
    json!({"id": id, "text": format!("text of {id}"), "vector": embedding(seed), "metadata": meta})
}

fn ids(resp: &Value) -> Vec<String> {
    resp["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn add_batch_query_filter_delete() {
    let s = Server::start(&[], None);
    let (st, _) = s.json(
        "POST",
        "/api/v1/documents",
        None,
        &doc("solo", 1, json!({"kind": "a"})),
    );
    assert_eq!(st, 200);
    let batch = json!([
        doc("b", 2, json!({"kind": "b"})),
        doc("c", 3, json!({"kind": "a"})),
        doc("d", 4, json!({"kind": "a", "n": 7})),
    ]);
    let (st, body) = s.json("POST", "/api/v1/documents/batch", None, &batch);
    assert_eq!((st, body["added"].as_u64()), (200, Some(3)));

    for (seed, id) in [(1, "solo"), (2, "b"), (3, "c"), (4, "d")] {
        let (st, r) = s.json(
            "POST",
            "/api/v1/query",
            None,
            &json!({"query_vector": embedding(seed), "top_k": 4}),
        );
        assert_eq!(st, 200);
        assert_eq!(ids(&r)[0], id, "nearest document must be first");
        let first = &r["matches"][0];
        assert_eq!(first["text"], format!("text of {id}"));
        let scores: Vec<f64> = r["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["score"].as_f64().unwrap())
            .collect();
        assert!(
            scores.windows(2).all(|w| w[0] >= w[1]),
            "best-first: {scores:?}"
        );
    }

    let q = json!({"query_vector": embedding(2), "top_k": 4, "filter": {"kind": "a"}});
    let (_, r) = s.json("POST", "/api/v1/query", None, &q);
    let got = ids(&r);
    assert!(!got.contains(&"b".to_string()) && got.len() == 3, "{got:?}");
    assert_eq!(r["matches"][0]["metadata"]["kind"], "a");

    let (st, _) = s.json("DELETE", "/api/v1/documents/solo", None, &json!(null));
    assert_eq!(st, 200);
    let (st, e) = s.json("DELETE", "/api/v1/documents/solo", None, &json!(null));
    assert_eq!((st, e["error"]["code"].as_str()), (404, Some("not_found")));
    let (_, r) = s.json(
        "POST",
        "/api/v1/query",
        None,
        &json!({"query_vector": embedding(1), "top_k": 10}),
    );
    assert!(!ids(&r).contains(&"solo".to_string()));
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn finite_f64_extremes_preserve_cosine_and_survive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::start_in(dir.path(), &[], None);
    let axis = |index: usize, magnitude: f64| {
        let mut values = vec![0.0; DIM];
        values[index] = magnitude;
        values
    };
    let batch = json!([
        {"id": "large", "vector": axis(0, f64::MAX)},
        {"id": "tiny", "vector": axis(1, f64::from_bits(1))}
    ]);
    assert_eq!(
        s.json("POST", "/api/v1/documents/batch", None, &batch).0,
        200
    );
    let mut expected = Vec::new();
    for magnitude in [f64::from_bits(1), 1e-300, 1e300, f64::MAX] {
        for (index, id) in [(0, "large"), (1, "tiny")] {
            let query = json!({"query_vector": axis(index, magnitude), "top_k": 2});
            let (status, response) = s.json("POST", "/api/v1/query", None, &query);
            assert_eq!(status, 200);
            assert_eq!(response["matches"][0]["id"], id);
            assert_eq!(response["matches"][0]["score"], 1.0);
            assert_eq!(response["matches"][1]["score"], 0.0);
            expected.push((query, response));
        }
    }
    assert!(s.stop());
    let reopened = Server::start_in(dir.path(), &[], None);
    for (query, response) in expected {
        assert_eq!(
            reopened.json("POST", "/api/v1/query", None, &query),
            (200, response)
        );
    }
}

#[test]
fn scores_are_exact_cosine_and_documents_must_fit_one_chunk() {
    let s = Server::start(&[], None);
    let batch = json!([
        doc("x", 11, json!({})),
        doc("y", 12, json!({})),
        doc("z", 13, json!({}))
    ]);
    let (st, _) = s.json("POST", "/api/v1/documents/batch", None, &batch);
    assert_eq!(st, 200);
    let query = embedding(14);
    let (st, r) = s.json(
        "POST",
        "/api/v1/query",
        None,
        &json!({"query_vector": query, "top_k": 3}),
    );
    assert_eq!(st, 200);
    let mut expected: Vec<(String, f64)> = [("x", 11), ("y", 12), ("z", 13)]
        .iter()
        .map(|(id, seed)| (id.to_string(), cosine(&query, &embedding(*seed))))
        .collect();
    expected.sort_by(|a, b| b.1.total_cmp(&a.1));
    let got: Vec<(String, f64)> = r["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["id"].as_str().unwrap().to_string(),
                m["score"].as_f64().unwrap(),
            )
        })
        .collect();
    assert_eq!(got.len(), 3);
    for ((gid, gs), (eid, es)) in got.iter().zip(&expected) {
        assert_eq!(gid, eid);
        assert!((gs - es).abs() < 1e-9, "{gid}: score {gs} vs cosine {es}");
    }

    let bare = json!({"id": "bare", "vector": embedding(16)});
    let (st, _) = s.json("POST", "/api/v1/documents", None, &bare);
    assert_eq!(st, 200);
    let (_, r) = s.json(
        "POST",
        "/api/v1/query",
        None,
        &json!({"query_vector": embedding(16), "top_k": 1}),
    );
    assert_eq!(
        (ids(&r)[0].as_str(), r["matches"][0]["text"].as_str()),
        ("bare", Some(""))
    );

    let words = "w ".repeat(4097);
    let long = json!({"id": "long", "text": words, "vector": embedding(15)});
    let (st, e) = s.json("POST", "/api/v1/documents", None, &long);
    assert_eq!(
        (st, e["error"]["code"].as_str()),
        (422, Some("invalid_request"))
    );
}

#[test]
fn upsert_replaces_text_and_metadata() {
    let s = Server::start(&[], None);
    s.json(
        "POST",
        "/api/v1/documents",
        None,
        &doc("x", 5, json!({"v": 1})),
    );
    let mut again = doc("x", 5, json!({"v": 2}));
    again["text"] = json!("new");
    s.json("POST", "/api/v1/documents", None, &again);
    let (_, r) = s.json(
        "POST",
        "/api/v1/query",
        None,
        &json!({"query_vector": embedding(5), "top_k": 5}),
    );
    assert_eq!(ids(&r), vec!["x"]);
    assert_eq!(
        (
            r["matches"][0]["text"].as_str(),
            r["matches"][0]["metadata"]["v"].as_i64()
        ),
        (Some("new"), Some(2))
    );
}

#[test]
fn auth_is_enforced_when_key_is_set() {
    let s = Server::start(&[], Some("s3cret-key"));
    let body = json!({"query_vector": embedding(1)});
    let (st, e) = s.json("POST", "/api/v1/query", None, &body);
    assert_eq!(
        (st, e["error"]["code"].as_str()),
        (401, Some("unauthorized"))
    );
    for bad in [
        "Bearer wrong",
        "Bearer s3cret-key-extra",
        "s3cret-key",
        "Bearer ",
    ] {
        let (st, _) = s.request(
            "POST",
            "/api/v1/query",
            &[("Authorization", bad)],
            body.to_string().as_bytes(),
        );
        assert_eq!(st, 401, "{bad}");
    }
    let (st, _) = s.request("DELETE", "/api/v1/documents/x", &[], b"");
    assert_eq!(st, 401);
    let (st, _) = s.request(
        "POST",
        "/api/v1/query",
        &[("Authorization", "Bearer s3cret-key")],
        body.to_string().as_bytes(),
    );
    assert_eq!(st, 200);
}

#[test]
fn oversized_body_is_rejected() {
    let s = Server::start(&["--max-body-bytes", "4096"], None);
    let big = vec![b' '; 6000];
    let (st, e) = s.request("POST", "/api/v1/documents/batch", &[], &big);
    assert_eq!(
        (st, e["error"]["code"].as_str()),
        (413, Some("payload_too_large"))
    );
}

#[test]
fn malformed_and_invalid_requests_get_json_errors() {
    let s = Server::start(&["--max-batch", "2"], None);
    for (path, body) in [
        ("/api/v1/query", &b"{not json"[..]),
        ("/api/v1/documents", b"[1,2]"),
        ("/api/v1/documents/batch", b"{\"id\":\"a\"}"),
        ("/api/v1/query", b""),
    ] {
        let (st, e) = s.request("POST", path, &[], body);
        assert_eq!(
            (st, e["error"]["code"].as_str()),
            (400, Some("invalid_json")),
            "{path}"
        );
    }

    let mut short = doc("a", 1, json!({}));
    short["vector"] = json!([0.5, 0.5]);
    let (st, e) = s.json("POST", "/api/v1/documents", None, &short);
    assert_eq!(
        (st, e["error"]["code"].as_str()),
        (422, Some("invalid_request"))
    );
    let (st, _) = s.json(
        "POST",
        "/api/v1/query",
        None,
        &json!({"query_vector": [1.0, 2.0]}),
    );
    assert_eq!(st, 422);

    let mut no_vec = doc("a", 1, json!({}));
    no_vec.as_object_mut().unwrap().remove("vector");
    assert_eq!(s.json("POST", "/api/v1/documents", None, &no_vec).0, 422);

    let (status, error) = s.request("POST", "/api/v1/query", &[], b"{\"query_vector\":[1e400]}");
    assert_eq!(
        (status, error["error"]["code"].as_str()),
        (400, Some("invalid_json"))
    );
    assert_eq!(
        s.json(
            "POST",
            "/api/v1/query",
            None,
            &json!({"query_vector": vec![0.0; DIM]})
        )
        .0,
        422
    );

    let q = |k: Value| {
        s.json(
            "POST",
            "/api/v1/query",
            None,
            &json!({"query_vector": embedding(1), "top_k": k}),
        )
        .0
    };
    assert_eq!(
        (q(json!(0)), q(json!(-1)), q(json!(101)), q(json!(100))),
        (422, 422, 422, 200)
    );
    assert_eq!(
        s.json(
            "POST",
            "/api/v1/query",
            None,
            &json!({"query_vector": embedding(1), "filter": [1]})
        )
        .0,
        422
    );

    let mut long_id = doc("a", 1, json!({}));
    long_id["id"] = json!("x".repeat(257));
    assert_eq!(s.json("POST", "/api/v1/documents", None, &long_id).0, 422);
    for bad in ["", "hms:chunk:1", "a\u{1}b"] {
        let mut d = doc("a", 1, json!({}));
        d["id"] = json!(bad);
        assert_eq!(
            s.json("POST", "/api/v1/documents", None, &d).0,
            422,
            "{bad:?}"
        );
    }
    let mut bad_meta = doc("a", 1, json!({}));
    bad_meta["metadata"] = json!([1]);
    assert_eq!(s.json("POST", "/api/v1/documents", None, &bad_meta).0, 422);

    assert_eq!(
        s.json("POST", "/api/v1/documents/batch", None, &json!([]))
            .0,
        422
    );
    let three = json!([
        doc("a", 1, json!({})),
        doc("b", 2, json!({})),
        doc("c", 3, json!({}))
    ]);
    assert_eq!(
        s.json("POST", "/api/v1/documents/batch", None, &three).0,
        422
    );
    // A batch with one bad document stores nothing.
    let mixed = json!([doc("good", 1, json!({})), short]);
    assert_eq!(
        s.json("POST", "/api/v1/documents/batch", None, &mixed).0,
        422
    );
    let (_, r) = s.json(
        "POST",
        "/api/v1/query",
        None,
        &json!({"query_vector": embedding(1)}),
    );
    assert!(ids(&r).is_empty());

    let (st, e) = s.request("GET", "/api/v1/query", &[], b"");
    assert_eq!(
        (st, e["error"]["code"].as_str()),
        (405, Some("method_not_allowed"))
    );
    assert_eq!(s.request("GET", "/nope", &[], b"").0, 404);
}

#[test]
fn tenants_are_isolated_and_validated() {
    let s = Server::start(&[], None);
    s.json(
        "POST",
        "/api/v1/documents",
        Some("alpha"),
        &doc("only-alpha", 1, json!({})),
    );
    s.json(
        "POST",
        "/api/v1/documents",
        Some("Alpha"),
        &doc("only-capital", 2, json!({})),
    );
    let q = |t: Option<&str>| {
        let (st, r) = s.json(
            "POST",
            "/api/v1/query",
            t,
            &json!({"query_vector": embedding(1), "top_k": 10}),
        );
        assert_eq!(st, 200);
        ids(&r)
    };
    assert_eq!(q(Some("alpha")), vec!["only-alpha"]);
    assert_eq!(q(Some("Alpha")), vec!["only-capital"]);
    assert!(q(Some("beta")).is_empty());
    assert!(q(None).is_empty());
    assert_eq!(
        s.json(
            "DELETE",
            "/api/v1/documents/only-alpha",
            Some("beta"),
            &json!(null)
        )
        .0,
        404
    );
    assert_eq!(q(Some("alpha")), vec!["only-alpha"]);

    for bad in ["../x", "a/b", "a b", "", &"t".repeat(65), "..", "a.b", "é"] {
        let (st, e) = s.request(
            "POST",
            "/api/v1/query",
            &[("X-Tenant-ID", bad)],
            json!({"query_vector": embedding(1)}).to_string().as_bytes(),
        );
        assert_eq!(
            (st, e["error"]["code"].as_str()),
            (400, Some("invalid_tenant")),
            "{bad:?}"
        );
    }
    let dirs = std::fs::read_dir(s._dir.as_ref().unwrap().path().join("tenants"))
        .unwrap()
        .count();
    assert_eq!(
        dirs, 4,
        "default, alpha, Alpha, beta; nothing created for invalid ids"
    );
}

#[test]
fn graceful_shutdown_persists_data() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::start_in(dir.path(), &[], None);
    s.json(
        "POST",
        "/api/v1/documents/batch",
        None,
        &json!([doc("keep", 1, json!({"k": "v"})), doc("drop", 2, json!({}))]),
    );
    s.json("DELETE", "/api/v1/documents/drop", None, &json!(null));
    assert!(s.stop(), "server must exit cleanly on SIGTERM");

    let s = Server::start_in(dir.path(), &[], None);
    let (_, r) = s.json(
        "POST",
        "/api/v1/query",
        None,
        &json!({"query_vector": embedding(1), "top_k": 5}),
    );
    assert_eq!(ids(&r), vec!["keep"]);
    assert_eq!(r["matches"][0]["metadata"]["k"], "v");
    assert_eq!(r["matches"][0]["text"], "text of keep");
}

#[test]
fn concurrent_first_access_opens_one_store_and_preserves_all_writes() {
    let s = Server::start(&["--max-tenants", "1", "--max-concurrent-work", "8"], None);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    std::thread::scope(|scope| {
        for index in 0..8 {
            let server = &s;
            let barrier = barrier.clone();
            scope.spawn(move || {
                barrier.wait();
                let (status, _) = server.json(
                    "POST",
                    "/api/v1/documents",
                    Some("first"),
                    &doc(&format!("document-{index}"), index, json!({})),
                );
                assert_eq!(status, 200);
            });
        }
    });
    let (status, response) = s.json(
        "POST",
        "/api/v1/query",
        Some("first"),
        &json!({"query_vector": embedding(0), "top_k": 10}),
    );
    assert_eq!(status, 200);
    assert_eq!(ids(&response).len(), 8);
    assert_eq!(
        std::fs::read_dir(s._dir.as_ref().unwrap().path().join("tenants"))
            .unwrap()
            .count(),
        1
    );
}

#[test]
fn concurrent_distinct_first_access_enforces_tenant_reservations_at_cap() {
    let s = Server::start(&["--max-tenants", "3", "--max-concurrent-work", "4"], None);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
    let results = std::thread::scope(|scope| {
        let mut tasks = Vec::new();
        for index in 0..4 {
            let server = &s;
            let barrier = barrier.clone();
            tasks.push(scope.spawn(move || {
                barrier.wait();
                server.json(
                    "POST",
                    "/api/v1/query",
                    Some(&format!("tenant-{index}")),
                    &json!({"query_vector": embedding(index)}),
                )
            }));
        }
        tasks
            .into_iter()
            .map(|task| task.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        results.iter().filter(|(status, _)| *status == 200).count(),
        3
    );
    let rejected = results.iter().find(|(status, _)| *status == 503).unwrap();
    assert_eq!(rejected.1["error"]["code"], "tenant_limit");
    assert_eq!(
        std::fs::read_dir(s._dir.as_ref().unwrap().path().join("tenants"))
            .unwrap()
            .count(),
        3
    );
}

#[test]
fn failed_tenant_initialization_releases_reservation_and_retries() {
    let directory = tempfile::tempdir().unwrap();
    let tenant_path = directory.path().join("tenants").join("626c6f636b6564");
    std::fs::create_dir_all(tenant_path.parent().unwrap()).unwrap();
    std::fs::write(&tenant_path, "not a directory").unwrap();
    let s = Server::start_in(directory.path(), &["--max-tenants", "1"], None);
    let query = json!({"query_vector": embedding(0)});
    let (status, error) = s.json("POST", "/api/v1/query", Some("blocked"), &query);
    assert_eq!(status, 500);
    assert_eq!(
        error["error"],
        json!({"code": "internal", "message": "internal server error"})
    );
    assert_eq!(
        error["detail"], "internal server error",
        "no internal detail leaks"
    );
    std::fs::remove_file(&tenant_path).unwrap();
    assert_eq!(
        s.json("POST", "/api/v1/query", Some("blocked"), &query).0,
        200
    );
    let (status, error) = s.json("POST", "/api/v1/query", Some("other"), &query);
    assert_eq!(status, 503);
    assert_eq!(error["error"]["code"], "tenant_limit");
}

#[test]
fn batch_last_write_wins_and_aggregate_rejection_preserve_atomicity() {
    let s = Server::start(&[], None);
    assert_eq!(
        s.json(
            "POST",
            "/api/v1/documents",
            None,
            &doc("keep", 1, json!({"version": "old"}))
        )
        .0,
        200
    );
    let mut last = doc("keep", 3, json!({"version": "later"}));
    last["text"] = json!("final source text");
    let duplicate = json!([doc("keep", 2, json!({"version": "new"})), last]);
    let (status, response) = s.json("POST", "/api/v1/documents/batch", None, &duplicate);
    assert_eq!(status, 200);
    assert_eq!(response["added"], 2);
    let mut invalid = doc("keep", 4, json!({"version": "invalid"}));
    invalid["vector"] = json!([0.5]);
    let batch = json!([invalid, doc("keep", 5, json!({"version": "discarded"}))]);
    assert_eq!(
        s.json("POST", "/api/v1/documents/batch", None, &batch).0,
        422
    );
    let mut large = vec![doc("keep", 2, json!({"version": "new"}))];
    for index in 0..132 {
        large.push(doc(
            &format!("extra-{index}"),
            index,
            json!({"payload": "x".repeat(64_000)}),
        ));
    }
    let (status, error) = s.json("POST", "/api/v1/documents/batch", None, &json!(large));
    assert_eq!(status, 422);
    assert_eq!(error["error"]["code"], "invalid_request");
    let (status, response) = s.json(
        "POST",
        "/api/v1/query",
        None,
        &json!({"query_vector": embedding(3), "top_k": 10}),
    );
    assert_eq!(status, 200);
    assert_eq!(ids(&response), vec!["keep"]);
    assert_eq!(response["matches"][0]["metadata"]["version"], "later");
    assert_eq!(response["matches"][0]["text"], "final source text");
    assert!((response["matches"][0]["score"].as_f64().unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn configured_work_and_batch_limits_reject_native_limit_overflow_at_startup() {
    for (flag, invalid) in [
        ("--max-concurrent-work", "0"),
        ("--max-concurrent-work", "257"),
        ("--max-batch", "4097"),
        ("--body-timeout-seconds", "0"),
        ("--body-timeout-seconds", "301"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_hms-server"))
            .args(["--no-auth", flag, invalid])
            .env_remove("HMS_API_KEY")
            .output()
            .unwrap();
        assert!(!output.status.success());
    }
    for limit in [255, 256] {
        let s = Server::start(&["--max-concurrent-work", &limit.to_string()], None);
        assert_eq!(
            s.json(
                "POST",
                "/api/v1/query",
                None,
                &json!({"query_vector": embedding(0)})
            )
            .0,
            200
        );
    }
    for limit in [4095, 4096] {
        let s = Server::start(&["--max-batch", &limit.to_string()], None);
        assert_eq!(
            s.json(
                "POST",
                "/api/v1/documents/batch",
                None,
                &json!([doc("one", 0, json!({}))])
            )
            .0,
            200
        );
    }
    for seconds in [299, 300] {
        let s = Server::start(&["--body-timeout-seconds", &seconds.to_string()], None);
        assert_eq!(
            s.json(
                "POST",
                "/api/v1/query",
                None,
                &json!({"query_vector": embedding(0)})
            )
            .0,
            200
        );
    }
}

#[test]
fn explicit_index_maintenance_preserves_exact_queries_and_tenant_isolation() {
    let s = Server::start(&[], None);
    let documents: Vec<_> = (0..128)
        .map(|index| {
            doc(
                &format!("vector-{index}"),
                index,
                json!({"parity": index % 2}),
            )
        })
        .collect();
    assert_eq!(
        s.json(
            "POST",
            "/api/v1/documents/batch",
            Some("indexed"),
            &json!(documents)
        )
        .0,
        200
    );
    let query = json!({"query_vector": embedding(12), "top_k": 10, "filter": {"parity": 0}});
    let (status, before) = s.json("POST", "/api/v1/query", Some("indexed"), &query);
    assert_eq!(status, 200);
    let (status, maintained) = s.json(
        "POST",
        "/api/v1/index/maintain",
        Some("indexed"),
        &json!({}),
    );
    assert_eq!(status, 200);
    assert_eq!(maintained, json!({"documentIndexTrained": true}));
    let (status, after) = s.json("POST", "/api/v1/query", Some("indexed"), &query);
    assert_eq!(status, 200);
    assert_eq!(before, after);
    assert_eq!(ids(&after)[0], "vector-12");
    let (status, empty) = s.json("POST", "/api/v1/index/maintain", Some("other"), &json!({}));
    assert_eq!(status, 200);
    assert_eq!(empty, json!({"documentIndexTrained": false}));
    assert_eq!(
        s.json(
            "POST",
            "/api/v1/index/maintain",
            Some("indexed"),
            &json!({"unsupported": true})
        )
        .0,
        422
    );
    let authenticated = Server::start(&[], Some("test-maintenance-key"));
    assert_eq!(
        authenticated
            .request("POST", "/api/v1/index/maintain", &[], b"")
            .0,
        401
    );
    assert_eq!(
        authenticated
            .request(
                "POST",
                "/api/v1/index/maintain",
                &[("Authorization", "Bearer test-maintenance-key")],
                b""
            )
            .0,
        200
    );
}
