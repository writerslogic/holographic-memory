// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

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
        if let Some(key) = api_key {
            cmd.env("HMS_API_KEY", key);
        }
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

    // 1e300 is finite as f64 but overflows f32.
    let mut huge = vec![0.0; DIM];
    huge[0] = 1e300;
    assert_eq!(
        s.json(
            "POST",
            "/api/v1/query",
            None,
            &json!({"query_vector": huge})
        )
        .0,
        422
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
