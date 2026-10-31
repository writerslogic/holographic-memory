// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "server")]

mod server_support;

use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use server_support::*;

fn wait_until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn refuses_to_start_without_credentials_or_explicit_dev_mode() {
    let stderr = Server::refuses(&[], &[]);
    assert!(stderr.contains("--no-auth"), "{stderr}");
    let stderr = Server::refuses(&["--no-auth", "--bind", "0.0.0.0:0"], &[]);
    assert!(stderr.contains("loopback"), "{stderr}");
    let stderr = Server::refuses(&["--no-auth"], &[("WARM_API_KEY", "k")]);
    assert!(stderr.contains("conflicts"), "{stderr}");
    assert!(Server::refuses(&[], &[("WARM_API_KEY", "")]).contains("empty"));
}

#[test]
fn keys_are_scoped_tenant_bound_and_rotate_without_restart() {
    let scratch = tempfile::tempdir().unwrap();
    let keys = scratch.path().join("keys.json");
    let reader = keygen(&keys, "reader", "acme", "read");
    let writer = keygen(&keys, "writer", "acme", "read,write");
    let admin = keygen(&keys, "root", "*", "admin");
    let keys_arg = keys.to_str().unwrap();
    let s = Server::start(&["--keys-file", keys_arg], &[]);

    // The key file stores digests, never tokens.
    let stored = std::fs::read_to_string(&keys).unwrap();
    assert!(!stored.contains(&reader) && !stored.contains("warm_reader."));
    assert!(stored.contains("sha256:"));

    let doc_body = doc("one", 1);
    // Scope enforcement.
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .bearer(&reader)
            .json(&doc_body)
            .send()
            .status,
        403
    );
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .bearer(&writer)
            .json(&doc_body)
            .send()
            .status,
        200
    );
    let query = json!({"query_vector": embedding(1)});
    let hit = s
        .req("POST", "/api/v1/query")
        .bearer(&reader)
        .json(&query)
        .send();
    assert_eq!(hit.status, 200);
    assert_eq!(hit.json()["matches"][0]["id"], "one");
    // A key's only tenant is its default; other tenants are refused.
    let other = s
        .req("POST", "/api/v1/query")
        .bearer(&reader)
        .tenant("beta")
        .json(&query)
        .send();
    assert_eq!(other.status, 403);
    assert_eq!(other.json()["code"], "tenant_forbidden");
    // Admin without tenant binding reaches every tenant; admin includes read and write.
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .bearer(&admin)
            .tenant("beta")
            .json(&doc("b", 2))
            .send()
            .status,
        200
    );
    // Bad tokens: unknown id, wrong secret, scheme, none.
    for auth in [
        "Bearer warm_nobody.00",
        "Bearer warm_reader.00",
        "Basic abc",
        "",
    ] {
        let r = s
            .req("POST", "/api/v1/query")
            .header("Authorization", auth)
            .json(&query)
            .send();
        assert_eq!(r.status, 401, "{auth}");
        assert_eq!(r.header("www-authenticate"), Some("Bearer"));
    }
    assert_eq!(
        s.req("POST", "/api/v1/query").json(&query).send().status,
        401
    );

    // Rotation: a new key is unknown until reload, then both keys work, then the old one is disabled.
    let rotated = keygen(&keys, "reader2", "acme", "read");
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .bearer(&rotated)
            .json(&query)
            .send()
            .status,
        401
    );
    assert_eq!(
        s.req("POST", "/api/v1/admin/keys/reload")
            .bearer(&reader)
            .send()
            .status,
        403
    );
    assert_eq!(
        s.req("POST", "/api/v1/admin/keys/reload")
            .bearer(&admin)
            .send()
            .json()["keys"],
        4
    );
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .bearer(&rotated)
            .json(&query)
            .send()
            .status,
        200
    );
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .bearer(&reader)
            .json(&query)
            .send()
            .status,
        200
    );
    let mut file: Value = serde_json::from_slice(&std::fs::read(&keys).unwrap()).unwrap();
    for key in file["keys"].as_array_mut().unwrap() {
        if key["id"] == "reader" {
            key["disabled"] = json!(true);
        }
    }
    std::fs::write(&keys, file.to_string()).unwrap();
    // SIGHUP reloads too.
    s.signal("HUP");
    wait_until("SIGHUP key reload", || {
        s.req("POST", "/api/v1/query")
            .bearer(&reader)
            .json(&query)
            .send()
            .status
            == 401
    });
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .bearer(&rotated)
            .json(&query)
            .send()
            .status,
        200
    );
    // A broken file is rejected and the keys stay in force.
    std::fs::write(&keys, b"{broken").unwrap();
    let r = s
        .req("POST", "/api/v1/admin/keys/reload")
        .bearer(&admin)
        .send();
    assert_eq!(r.status, 422);
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .bearer(&rotated)
            .json(&query)
            .send()
            .status,
        200
    );
}

#[test]
fn legacy_api_key_still_grants_everything() {
    let s = Server::with_key("legacy-secret", &[]);
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .bearer("legacy-secret")
            .tenant("any")
            .json(&doc("a", 1))
            .send()
            .status,
        200
    );
    assert_eq!(
        s.req("POST", "/api/v1/snapshots")
            .bearer("legacy-secret")
            .tenant("any")
            .send()
            .status,
        201
    );
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .bearer("legacy-secre")
            .json(&json!({}))
            .send()
            .status,
        401
    );
}

#[test]
fn errors_are_problem_json_with_request_ids_and_the_legacy_error_object() {
    let s = Server::open(&[]);
    let r = s
        .req("POST", "/api/v1/query")
        .header("X-Request-Id", "trace-123")
        .json(&json!({"query_vector": [1.0], "top_k": 5}))
        .send();
    assert_eq!(r.status, 422);
    assert_eq!(r.header("content-type"), Some("application/problem+json"));
    assert_eq!(r.header("x-request-id"), Some("trace-123"));
    let p = r.json();
    assert_eq!(p["status"], 422);
    assert_eq!(p["code"], "invalid_request");
    assert_eq!(p["request_id"], "trace-123");
    assert_eq!(p["instance"], "urn:hms:request:trace-123");
    assert!(p["type"].as_str().unwrap().ends_with("#invalid_request"));
    assert_eq!(p["error"]["code"], "invalid_request");
    assert_eq!(p["error"]["message"], p["detail"]);
    // A hostile request id is replaced, not echoed.
    let r = s
        .req("GET", "/healthz")
        .header("X-Request-Id", "bad id\"<script>")
        .send();
    assert!(r.header("x-request-id").unwrap().len() == 16);
    let r = s.req("GET", "/nope").send();
    assert_eq!(r.status, 404);
    assert_eq!(r.header("content-type"), Some("application/problem+json"));
    assert_eq!(s.req("DELETE", "/api/v1/query").send().status, 405);
}

#[test]
fn health_readiness_and_metrics() {
    let scratch = tempfile::tempdir().unwrap();
    let keys = scratch.path().join("keys.json");
    let reader = keygen(&keys, "r", "*", "read");
    let admin = keygen(&keys, "a", "*", "admin");
    let s = Server::start(&["--keys-file", keys.to_str().unwrap()], &[]);
    assert_eq!(s.req("GET", "/healthz").send().json()["status"], "ok");
    assert_eq!(s.req("GET", "/readyz").send().json()["status"], "ready");
    assert_eq!(s.req("GET", "/metrics").send().status, 401);
    assert_eq!(s.req("GET", "/metrics").bearer(&reader).send().status, 403);

    s.req("POST", "/api/v1/documents/batch")
        .bearer(&admin)
        .tenant("acme")
        .json(&json!([doc("a", 1), doc("b", 2)]))
        .send();
    s.req("POST", "/api/v1/query")
        .bearer(&admin)
        .tenant("acme")
        .json(&json!({"query_vector": embedding(1)}))
        .send();
    s.req("DELETE", "/api/v1/documents/nope%2Fx")
        .bearer(&admin)
        .tenant("acme")
        .send();
    let m = s.req("GET", "/metrics").bearer(&admin).send();
    assert_eq!(m.status, 200);
    assert!(m.header("content-type").unwrap().starts_with("text/plain"));
    let text = m.text();
    for needle in [
        "warm_build_info{version=",
        "warm_ready 1",
        "warm_work_slots_capacity 4",
        "warm_queue_depth 0",
        "warm_tenants_open 1",
        "warm_tenant_documents{tenant=\"acme\"} 2",
        "warm_tenant_vectors{tenant=\"acme\"} 2",
        "warm_http_requests_total{route=\"/api/v1/documents/batch\",method=\"POST\",status=\"200\"} 1",
        // Route templates, never raw ids, label the series.
        "warm_http_requests_total{route=\"/api/v1/documents/{id}\",method=\"DELETE\",status=\"404\"} 1",
        "warm_http_request_duration_seconds_bucket{route=\"/api/v1/query\",le=\"+Inf\"} 1",
        "warm_http_request_duration_seconds_count{route=\"/api/v1/query\"} 1",
        "warm_ingest_documents_total 2",
        "warm_auth_failures_total",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in\n{text}");
    }
    if cfg!(any(target_os = "linux", target_os = "macos")) {
        let rss: u64 = text
            .lines()
            .find_map(|l| l.strip_prefix("process_resident_memory_bytes "))
            .expect("rss gauge")
            .parse()
            .unwrap();
        assert!(rss > 1_000_000);
    }
    // --metrics-public drops the credential requirement for /metrics only.
    let p = Server::start(
        &["--keys-file", keys.to_str().unwrap(), "--metrics-public"],
        &[],
    );
    assert_eq!(p.req("GET", "/metrics").send().status, 200);
    assert_eq!(
        p.req("POST", "/api/v1/query")
            .json(&json!({}))
            .send()
            .status,
        401
    );
}

/// The committed spec is the build's spec; the server serves it; every operation in it is
/// secured with exactly the scope it declares.
#[test]
fn openapi_matches_the_committed_file_and_documented_security_is_enforced() {
    let out = Command::new(BIN).arg("openapi").output().unwrap();
    assert!(out.status.success());
    let generated: Value = serde_json::from_slice(&out.stdout).unwrap();
    let committed = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/server/openapi.json");
    if std::env::var_os("WARM_UPDATE_OPENAPI").is_some() {
        std::fs::write(committed, [out.stdout.as_slice(), b"\n"].concat()).unwrap();
    }
    let on_disk: Value = serde_json::from_slice(&std::fs::read(committed).unwrap())
        .expect("docs/server/openapi.json; regenerate with WARM_UPDATE_OPENAPI=1");
    assert_eq!(
        on_disk, generated,
        "docs/server/openapi.json is stale; run `WARM_UPDATE_OPENAPI=1 cargo test --features server --test server_api`"
    );
    assert_eq!(generated["openapi"], "3.1.0");

    let scratch = tempfile::tempdir().unwrap();
    let keys = scratch.path().join("keys.json");
    let reader = keygen(&keys, "r", "*", "read");
    let s = Server::start(&["--keys-file", keys.to_str().unwrap()], &[]);
    assert_eq!(s.req("GET", "/openapi.json").send().json(), generated);

    let mut operations = 0;
    for (path, item) in generated["paths"].as_object().unwrap() {
        for (method, op) in item.as_object().unwrap() {
            operations += 1;
            let url = path.replace("{id}", "x");
            let scope = op["security"][0]["bearerAuth"][0].as_str();
            let send = |token: Option<&str>| {
                let verb = method.to_uppercase();
                let mut r = s.req(&verb, &url);
                if let Some(t) = token {
                    r = r.bearer(t);
                }
                if op.get("requestBody").is_some() {
                    r = r.header("Content-Type", "application/json").body("{}");
                }
                r.send()
            };
            let anonymous = send(None);
            match scope {
                None => assert!(
                    ![401, 403].contains(&anonymous.status),
                    "{method} {path} should be public"
                ),
                Some(scope) => {
                    assert_eq!(anonymous.status, 401, "{method} {path}");
                    let as_reader = send(Some(&reader));
                    if scope == "read" {
                        assert_ne!(as_reader.status, 403, "{method} {path}");
                    } else {
                        assert_eq!(as_reader.status, 403, "{method} {path} needs {scope}");
                    }
                }
            }
            assert!(
                op["responses"].get("default").is_some(),
                "{method} {path} lacks the error model"
            );
        }
    }
    assert!(operations >= 20, "only {operations} operations");
    // The request schemas really constrain what the handlers accept.
    let doc_schema = &generated["components"]["schemas"]["DocIn"];
    assert_eq!(doc_schema["required"], json!(["id"]));
}

#[test]
fn list_stats_and_conditional_requests() {
    let s = Server::open(&["--max-batch", "50"]);
    let docs: Vec<Value> = (0..25).map(|i| doc(&format!("d{i:02}"), i)).collect();
    assert_eq!(
        s.req("POST", "/api/v1/documents/batch")
            .json(&Value::Array(docs))
            .send()
            .status,
        200
    );

    // Walk the cursor: every id exactly once, ascending, with the total on each page.
    let mut seen = Vec::new();
    let mut cursor: Option<String> = None;
    let mut pages = 0;
    loop {
        let path = match &cursor {
            None => "/api/v1/documents?limit=10".to_string(),
            Some(c) => format!("/api/v1/documents?limit=10&after={c}"),
        };
        let page = s.req("GET", &path).send();
        assert_eq!(page.status, 200);
        let body = page.json();
        assert_eq!(body["count"], 25);
        for d in body["documents"].as_array().unwrap() {
            assert!(d["metadata"]["seed"].as_u64().is_some());
            seen.push(d["id"].as_str().unwrap().to_string());
        }
        pages += 1;
        match body["next_cursor"].as_str() {
            Some(c) => cursor = Some(c.to_string()),
            None => break,
        }
    }
    assert_eq!(pages, 3);
    assert_eq!(seen.len(), 25);
    assert!(seen.windows(2).all(|w| w[0] < w[1]));
    // Page boundaries: limit-1, limit, limit+1 documents remaining.
    assert_eq!(
        s.req("GET", "/api/v1/documents?limit=25").send().json()["next_cursor"],
        Value::Null
    );
    assert_eq!(
        s.req("GET", "/api/v1/documents?limit=24").send().json()["next_cursor"],
        "d23"
    );
    assert_eq!(
        s.req("GET", "/api/v1/documents?after=d24").send().json()["documents"],
        json!([])
    );
    for bad in ["limit=0", "limit=1001", "limit=x", "limit=-1"] {
        assert_eq!(
            s.req("GET", &format!("/api/v1/documents?{bad}"))
                .send()
                .status,
            422,
            "{bad}"
        );
    }
    // Deleted documents leave the listing.
    assert_eq!(s.req("DELETE", "/api/v1/documents/d00").send().status, 200);
    assert_eq!(
        s.req("GET", "/api/v1/documents?limit=1").send().json()["count"],
        24
    );

    let stats = s.req("GET", "/api/v1/stats").send();
    let tag = stats.header("etag").unwrap().to_string();
    assert_eq!(stats.json()["documents"], 24);
    assert_eq!(stats.json()["vectors"], 24);
    let again = s
        .req("GET", "/api/v1/stats")
        .header("If-None-Match", &tag)
        .send();
    assert_eq!(again.status, 304);
    assert!(again.body.is_empty());
    assert_eq!(again.header("etag"), Some(tag.as_str()));
    assert_eq!(
        s.req("GET", "/api/v1/stats")
            .header("If-None-Match", "\"other\", *")
            .send()
            .status,
        304
    );
    // A write changes the validator; the list shares it.
    s.req("POST", "/api/v1/documents")
        .json(&doc("new", 99))
        .send();
    assert_eq!(
        s.req("GET", "/api/v1/stats")
            .header("If-None-Match", &tag)
            .send()
            .status,
        200
    );
    assert_ne!(
        s.req("GET", "/api/v1/documents").send().header("etag"),
        Some(tag.as_str())
    );
    // Tenants do not share counts.
    assert_eq!(
        s.req("GET", "/api/v1/stats").tenant("empty").send().json()["documents"],
        0
    );
}

#[test]
fn idempotency_keys_replay_conflict_and_scope_to_the_tenant() {
    let s = Server::open(&[]);
    let batch = json!([doc("a", 1), doc("b", 2)]);
    let first = s
        .req("POST", "/api/v1/documents/batch")
        .header("Idempotency-Key", "k-1")
        .json(&batch)
        .send();
    assert_eq!(first.status, 200);
    assert!(first.header("idempotent-replayed").is_none());
    // Remove a document, then replay: the recorded response comes back without re-running.
    assert_eq!(s.req("DELETE", "/api/v1/documents/a").send().status, 200);
    let replay = s
        .req("POST", "/api/v1/documents/batch")
        .header("Idempotency-Key", "k-1")
        .json(&batch)
        .send();
    assert_eq!(replay.status, 200);
    assert_eq!(replay.header("idempotent-replayed"), Some("true"));
    assert_eq!(replay.json(), first.json());
    assert_eq!(
        s.req("GET", "/api/v1/documents").send().json()["count"],
        1,
        "replay did not store again"
    );
    // Same key, different body.
    let reused = s
        .req("POST", "/api/v1/documents/batch")
        .header("Idempotency-Key", "k-1")
        .json(&json!([doc("c", 3)]))
        .send();
    assert_eq!(reused.status, 422);
    assert_eq!(reused.json()["code"], "idempotency_key_reused");
    // Another tenant may use the same key for its own request.
    let other = s
        .req("POST", "/api/v1/documents/batch")
        .tenant("t2")
        .header("Idempotency-Key", "k-1")
        .json(&batch)
        .send();
    assert_eq!(other.status, 200);
    assert!(other.header("idempotent-replayed").is_none());
    // Validation failures replay too; shed load and server errors do not.
    let bad = json!([{"id": ""}]);
    let a = s
        .req("POST", "/api/v1/documents/batch")
        .header("Idempotency-Key", "k-2")
        .json(&bad)
        .send();
    let b = s
        .req("POST", "/api/v1/documents/batch")
        .header("Idempotency-Key", "k-2")
        .json(&bad)
        .send();
    assert_eq!((a.status, b.status), (422, 422));
    assert_eq!(b.header("idempotent-replayed"), Some("true"));
    assert_eq!(b.header("content-type"), Some("application/problem+json"));
    // Header validation.
    let r = s
        .req("POST", "/api/v1/documents")
        .header("Idempotency-Key", &"x".repeat(129))
        .json(&doc("z", 1))
        .send();
    assert_eq!(r.status, 422);
}

#[test]
fn ndjson_stream_commits_in_batches_and_reports_rejected_lines() {
    let s = Server::open(&["--stream-batch", "3", "--max-stream-line-bytes", "4096"]);
    let mut body = String::new();
    for i in 0..10 {
        body.push_str(&format!("{}\n", doc(&format!("s{i}"), i)));
        if i == 4 {
            body.push_str("{not json}\n\n");
            body.push_str(&format!(
                "{}\n",
                json!({"id": "short", "vector": [1.0, 2.0]})
            ));
        }
    }
    // The final line has no newline.
    body.push_str(&doc("tail", 77).to_string());
    let r = s
        .req("POST", "/api/v1/documents/stream")
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .send();
    assert_eq!(r.status, 200, "{}", r.text());
    let out = r.json();
    assert_eq!(out["added"], 11);
    assert_eq!(out["rejected"], 2);
    assert_eq!(out["errors"][0]["line"], 6);
    assert_eq!(out["errors"][1]["line"], 8);
    assert_eq!(out["errors_truncated"], false);
    assert_eq!(s.req("GET", "/api/v1/documents").send().json()["count"], 11);
    let hit = s
        .req("POST", "/api/v1/query")
        .json(&json!({"query_vector": embedding(77), "top_k": 1}))
        .send();
    assert_eq!(hit.json()["matches"][0]["id"], "tail");

    let plain = s
        .req("POST", "/api/v1/documents/stream")
        .header("Content-Type", "application/json")
        .body("{}")
        .send();
    assert_eq!(plain.status, 415);
    let long = format!("{}\n", json!({"id": "x", "text": "y".repeat(5000)}));
    let r = s
        .req("POST", "/api/v1/documents/stream")
        .header("Content-Type", "application/x-ndjson")
        .body(long)
        .send();
    assert_eq!(r.status, 413);
    assert_eq!(r.json()["code"], "payload_too_large");
    // Without a newline the unterminated line is bounded as well.
    let r = s
        .req("POST", "/api/v1/documents/stream")
        .header("Content-Type", "application/x-ndjson")
        .body("y".repeat(5000))
        .send();
    assert_eq!(r.status, 413);
    // The stream limit is independent of --max-body-bytes.
    let small = Server::open(&["--max-body-bytes", "1024", "--max-stream-bytes", "100000"]);
    let many: String = (0..20)
        .map(|i| format!("{}\n", doc(&format!("m{i}"), i)))
        .collect();
    assert!(many.len() > 1024 && many.len() < 100_000);
    let r = small
        .req("POST", "/api/v1/documents/stream")
        .header("Content-Type", "application/x-ndjson")
        .body(many)
        .send();
    assert_eq!(r.json()["added"], 20);
    let capped = Server::open(&["--max-stream-bytes", "2000"]);
    let many: String = (0..20)
        .map(|i| format!("{}\n", doc(&format!("m{i}"), i)))
        .collect();
    assert_eq!(
        capped
            .req("POST", "/api/v1/documents/stream")
            .header("Content-Type", "application/x-ndjson")
            .body(many)
            .send()
            .status,
        413
    );
}

#[test]
fn per_tenant_rate_limit_and_document_quota() {
    let s = Server::open(&["--rate-limit-rps", "1", "--rate-limit-burst", "3"]);
    let q = json!({"query_vector": embedding(1)});
    let statuses: Vec<u16> = (0..5)
        .map(|_| {
            s.req("POST", "/api/v1/query")
                .tenant("noisy")
                .json(&q)
                .send()
                .status
        })
        .collect();
    assert_eq!(&statuses[..3], &[200, 200, 200]);
    assert_eq!(statuses[3], 429);
    let limited = s
        .req("POST", "/api/v1/query")
        .tenant("noisy")
        .json(&q)
        .send();
    assert_eq!(limited.json()["code"], "rate_limited");
    assert!(
        limited
            .header("retry-after")
            .unwrap()
            .parse::<u64>()
            .unwrap()
            >= 1
    );
    // Another tenant is unaffected; health stays reachable.
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .tenant("quiet")
            .json(&q)
            .send()
            .status,
        200
    );
    assert_eq!(s.req("GET", "/healthz").send().status, 200);
    std::thread::sleep(Duration::from_millis(1100));
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .tenant("noisy")
            .json(&q)
            .send()
            .status,
        200
    );

    let s = Server::open(&["--max-documents-per-tenant", "3"]);
    assert_eq!(
        s.req("POST", "/api/v1/documents/batch")
            .json(&json!([doc("a", 1), doc("b", 2), doc("c", 3)]))
            .send()
            .status,
        200
    );
    let over = s.req("POST", "/api/v1/documents").json(&doc("d", 4)).send();
    assert_eq!(over.status, 429);
    assert_eq!(over.json()["code"], "quota_exceeded");
    // Replacing an existing id is not growth; the limit applies per tenant.
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .json(&doc("a", 9))
            .send()
            .status,
        200
    );
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .tenant("other")
            .json(&doc("d", 4))
            .send()
            .status,
        200
    );
    // A batch that would cross the line is rejected whole.
    let r = s
        .req("POST", "/api/v1/documents/batch")
        .tenant("other")
        .json(&json!([doc("e", 5), doc("f", 6), doc("g", 7)]))
        .send();
    assert_eq!(r.status, 429);
    assert_eq!(
        s.req("GET", "/api/v1/stats").tenant("other").send().json()["documents"],
        1
    );
    // Deleting frees quota.
    s.req("DELETE", "/api/v1/documents/a").send();
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .json(&doc("d", 4))
            .send()
            .status,
        200
    );
}

#[test]
fn cors_is_off_by_default_and_exact_when_configured() {
    let off = Server::open(&[]);
    let r = off
        .req("GET", "/healthz")
        .header("Origin", "https://app.example")
        .send();
    assert!(r.header("access-control-allow-origin").is_none());

    let s = Server::open(&[
        "--cors-origins",
        "https://app.example,https://admin.example",
    ]);
    let pre = s
        .req("OPTIONS", "/api/v1/query")
        .header("Origin", "https://app.example")
        .header("Access-Control-Request-Method", "POST")
        .send();
    assert_eq!(pre.status, 204);
    assert_eq!(
        pre.header("access-control-allow-origin"),
        Some("https://app.example")
    );
    assert!(pre
        .header("access-control-allow-headers")
        .unwrap()
        .contains("idempotency-key"));
    assert_eq!(pre.header("access-control-max-age"), Some("600"));
    let r = s
        .req("GET", "/healthz")
        .header("Origin", "https://admin.example")
        .send();
    assert_eq!(
        r.header("access-control-allow-origin"),
        Some("https://admin.example")
    );
    assert!(r
        .header("access-control-expose-headers")
        .unwrap()
        .contains("etag"));
    assert!(r.header("vary").unwrap().contains("Origin"));
    let evil = s
        .req("GET", "/healthz")
        .header("Origin", "https://evil.example")
        .send();
    assert!(evil.header("access-control-allow-origin").is_none());
    let any = Server::open(&["--cors-origins", "*"]);
    assert_eq!(
        any.req("GET", "/healthz")
            .header("Origin", "https://x.example")
            .send()
            .header("access-control-allow-origin"),
        Some("*")
    );
}

#[test]
fn logs_are_structured_and_never_contain_secrets_or_content() {
    let secret_text = "ultra-private-sentence-4f9c";
    let secret_meta = "meta-secret-77ab";
    let secret_key = "very-secret-api-key-5521";
    let mut s = Server::with_key(secret_key, &[]);
    let body = json!({"id": "log-doc-id", "text": secret_text, "vector": embedding(1), "metadata": {"m": secret_meta}});
    s.req("POST", "/api/v1/documents")
        .bearer(secret_key)
        .tenant("logt")
        .json(&body)
        .send();
    s.req("POST", "/api/v1/query")
        .bearer(secret_key)
        .tenant("logt")
        .json(&json!({"query_vector": embedding(1), "filter": {"m": secret_meta}}))
        .send();
    s.req("POST", "/api/v1/query")
        .bearer("wrong-key-value-1")
        .json(&json!({}))
        .send();
    s.req("DELETE", "/api/v1/documents/log-doc-id")
        .bearer(secret_key)
        .tenant("logt")
        .send();
    assert!(s.stop());
    let lines = s.log_lines();
    let raw = lines.join("\n");
    for secret in [
        secret_text,
        secret_meta,
        secret_key,
        "wrong-key-value-1",
        "log-doc-id",
    ] {
        assert!(!raw.contains(secret), "log leaks {secret:?}");
    }
    let events = s.log_json();
    let requests: Vec<&Value> = events.iter().filter(|e| e["event"] == "request").collect();
    assert_eq!(requests.len(), 4);
    for r in &requests {
        assert!(r["request_id"].as_str().unwrap().len() >= 8);
        assert!(r["latency_ms"].as_f64().unwrap() >= 0.0);
        assert!(r["level"].is_string());
        assert!(r["ts"].as_str().unwrap().ends_with('Z'));
    }
    assert_eq!(requests[0]["route"], "/api/v1/documents");
    assert_eq!(requests[0]["tenant"], "logt");
    assert_eq!(requests[0]["key_id"], "legacy");
    assert_eq!(requests[2]["status"], 401);
    assert_eq!(requests[3]["route"], "/api/v1/documents/{id}");
    assert!(events.iter().any(|e| e["event"] == "shutdown_complete"));
}

#[test]
fn search_ranks_like_lexical_recall_and_returns_fields_only_on_request() {
    let s = Server::open(&[]);
    let words = [
        "apple", "river", "stone", "cloud", "ember", "maple", "quartz",
    ];
    let facts: Vec<Value> = (0..40)
        .map(|i| {
            let text: Vec<&str> = (0..6)
                .map(|j| words[(i * 3 + j * j) % words.len()])
                .collect();
            json!({"id": format!("f{i:02}"), "text": format!("{} note {i}", text.join(" "))})
        })
        .collect();
    let stored = s
        .req("POST", "/api/v1/memory/remember")
        .json(&json!({ "facts": facts }))
        .send();
    assert_eq!(stored.status, 200);

    for query in ["apple river", "quartz", "ember maple stone", "cloud note"] {
        let recall = s
            .req("POST", "/api/v1/memory/recall")
            .json(&json!({"query": query, "k": 10, "candidateLimit": 10, "lexicalWeight": 1, "semanticWeight": 0}))
            .send()
            .json();
        let expected: Vec<&str> = recall["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_str().unwrap())
            .collect();
        let found = s
            .req("POST", "/api/v1/search")
            .json(&json!({"query": query, "top_k": 10}))
            .send();
        assert_eq!(found.status, 200);
        let found = found.json();
        let hits = found["hits"].as_array().unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h["id"].as_str().unwrap()).collect();
        assert!(!ids.is_empty());
        assert_eq!(ids, expected, "query {query:?}");
        assert!(hits
            .iter()
            .all(|h| h.get("text").is_none() && h.get("metadata").is_none()));
        assert!(hits
            .windows(2)
            .all(|w| w[0]["score"].as_f64() >= w[1]["score"].as_f64()));
    }

    let full = s
        .req("POST", "/api/v1/search")
        .json(&json!({"query": "quartz", "top_k": 1, "include": ["text", "metadata"]}))
        .send()
        .json();
    let hit = &full["hits"][0];
    assert!(hit["text"].as_str().unwrap().contains("quartz"));
    assert!(hit["metadata"].is_object());
    assert_eq!(full["hits"].as_array().unwrap().len(), 1);

    for (bad, status) in [
        (json!({"query": "apple", "top_k": 0}), 422),
        (json!({"query": "  "}), 422),
        (json!({"query": "apple", "include": ["vector"]}), 422),
        (json!({"query": "apple", "k": 3}), 400),
    ] {
        assert_eq!(
            s.req("POST", "/api/v1/search").json(&bad).send().status,
            status,
            "{bad}"
        );
    }
}
