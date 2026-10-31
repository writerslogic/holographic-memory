// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Abuse and fuzz tests against the real binary: hostile bodies, paths and headers must produce
//! clean 4xx problem responses, never a 5xx, a hang, a crash, data loss or unbounded memory.

#![cfg(feature = "server")]

mod server_support;

use std::io::{Read, Write};
use std::time::Duration;

use serde_json::{json, Value};
use server_support::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

fn rss(s: &Server) -> u64 {
    s.req("GET", "/metrics")
        .bearer("k")
        .send()
        .text()
        .lines()
        .find_map(|l| l.strip_prefix("process_resident_memory_bytes "))
        .map_or(0, |v| v.parse().unwrap())
}

/// Seed corpus the mutator starts from.
fn corpus() -> Vec<Vec<u8>> {
    vec![
        doc("seed", 1).to_string().into_bytes(),
        json!([doc("a", 1), doc("b", 2)]).to_string().into_bytes(),
        json!({"query_vector": embedding(1), "top_k": 3, "filter": {"seed": 1}})
            .to_string()
            .into_bytes(),
        b"{}".to_vec(),
        b"[]".to_vec(),
        b"null".to_vec(),
    ]
}

fn mutate(rng: &mut Rng, seed: &[u8]) -> Vec<u8> {
    let mut data = seed.to_vec();
    for _ in 0..=rng.below(6) {
        if data.is_empty() {
            data.push(rng.next() as u8);
            continue;
        }
        let at = rng.below(data.len());
        match rng.below(6) {
            0 => data[at] ^= 1 << rng.below(8),
            1 => {
                data.remove(at);
            }
            2 => data.insert(at, rng.next() as u8),
            3 => data.truncate(at),
            4 => {
                let end = (at + rng.below(32)).min(data.len());
                let chunk = data[at..end].to_vec();
                data.splice(at..at, chunk);
            }
            _ => data[at] = b"{}[]\",:0-.eE\\ntfu"[rng.below(17)],
        }
    }
    data
}

fn hostile_bodies() -> Vec<(&'static str, Vec<u8>)> {
    let mut deep_array = vec![b'['; 100_000];
    deep_array.extend(vec![b']'; 100_000]);
    let deep_object = format!("{}1{}", "{\"a\":".repeat(50_000), "}".repeat(50_000)).into_bytes();
    let nan = format!("{{\"query_vector\":[{}]}}", vec!["NaN"; DIM].join(",")).into_bytes();
    let overflow = format!("{{\"query_vector\":[{}]}}", vec!["1e999"; DIM].join(",")).into_bytes();
    let huge_id = json!({"id": "x".repeat(2_000_000), "vector": embedding(1)})
        .to_string()
        .into_bytes();
    let many = format!(
        "{{\"query_vector\":[{}]}}",
        vec!["0.5"; 1_000_000].join(",")
    )
    .into_bytes();
    let dup = format!(
        "{{\"id\":\"a\",\"id\":\"b\",\"vector\":{}}}",
        json!(embedding(1))
    )
    .into_bytes();
    let mut bom = vec![0xEF, 0xBB, 0xBF];
    bom.extend(doc("bom", 1).to_string().into_bytes());
    vec![
        ("deep array", deep_array),
        ("deep object", deep_object),
        ("NaN", nan),
        ("1e999", overflow),
        ("2 MB id", huge_id),
        ("million-value vector", many),
        ("duplicate keys", dup),
        ("BOM", bom),
        (
            "lone surrogate",
            br#"{"id":"\ud800","vector":[1]}"#.to_vec(),
        ),
        (
            "invalid utf-8",
            b"{\"id\":\"\xff\xfe\",\"vector\":[1]}".to_vec(),
        ),
        (
            "nul in id",
            json!({"id": "a\u{0}b", "vector": embedding(1)})
                .to_string()
                .into_bytes(),
        ),
        (
            "control chars",
            json!({"id": "a\nb", "vector": embedding(1)})
                .to_string()
                .into_bytes(),
        ),
        (
            "reserved prefix",
            json!({"id": "hms:chunk", "vector": embedding(1)})
                .to_string()
                .into_bytes(),
        ),
        (
            "metadata array",
            json!({"id": "m", "vector": embedding(1), "metadata": [1, 2]})
                .to_string()
                .into_bytes(),
        ),
        (
            "negative top_k",
            json!({"query_vector": embedding(1), "top_k": -5})
                .to_string()
                .into_bytes(),
        ),
        (
            "huge top_k",
            json!({"query_vector": embedding(1), "top_k": 9_223_372_036_854_775_807i64})
                .to_string()
                .into_bytes(),
        ),
        (
            "float top_k",
            br#"{"query_vector":[1],"top_k":1.5}"#.to_vec(),
        ),
        (
            "filter scalar",
            json!({"query_vector": embedding(1), "filter": 7})
                .to_string()
                .into_bytes(),
        ),
        ("empty", Vec::new()),
        ("whitespace", b"   \n\t ".to_vec()),
        (
            "zero vector",
            json!({"id": "z", "vector": vec![0.0; DIM]})
                .to_string()
                .into_bytes(),
        ),
    ]
}

#[test]
fn hostile_bodies_get_clean_client_errors_and_the_store_survives() {
    let s = Server::with_key("k", &["--max-body-bytes", "16777216"]);
    let auth = |method: &str, path: &str| {
        let method = method.to_string();
        let path = path.to_string();
        move |s: &Server, body: Vec<u8>| {
            s.req(&method, &path)
                .bearer("k")
                .header("Content-Type", "application/json")
                .body(body)
                .send()
        }
    };
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .bearer("k")
            .json(&doc("keep", 7))
            .send()
            .status,
        200
    );
    let before = rss(&s);
    let endpoints = [
        auth("POST", "/api/v1/documents"),
        auth("POST", "/api/v1/documents/batch"),
        auth("POST", "/api/v1/query"),
        auth("POST", "/api/v1/index/maintain"),
    ];
    let mut checked = 0;
    for (name, body) in hostile_bodies() {
        for send in &endpoints {
            let r = send(&s, body.clone());
            assert!(
                (400..500).contains(&r.status) || r.status == 200,
                "{name}: {} {}",
                r.status,
                r.text().chars().take(200).collect::<String>()
            );
            if r.status >= 400 {
                assert_eq!(
                    r.header("content-type"),
                    Some("application/problem+json"),
                    "{name}"
                );
                let p = r.json();
                assert_eq!(p["status"], r.status, "{name}");
                assert!(p["code"].is_string(), "{name}");
            }
            checked += 1;
        }
    }
    // Seeded mutation fuzzing over every body-taking endpoint.
    let mut rng = Rng(0x00C0_FFEE_5EED_1234);
    let seeds = corpus();
    for i in 0..1500 {
        let body = mutate(&mut rng, &seeds[i % seeds.len()]);
        let send = &endpoints[i % endpoints.len()];
        let r = send(&s, body);
        assert!(
            r.status == 200 || (400..500).contains(&r.status),
            "iteration {i}: {}",
            r.status
        );
        checked += 1;
    }
    // Pure noise.
    for _ in 0..200 {
        let n = rng.below(4096);
        let body = rng.bytes(n);
        let r = endpoints[rng.below(4)](&s, body);
        assert!((400..500).contains(&r.status), "{}", r.status);
        checked += 1;
    }
    assert!(checked > 1700);
    // Still correct and still healthy afterwards.
    assert_eq!(s.req("GET", "/healthz").send().status, 200);
    let hit = s
        .req("POST", "/api/v1/query")
        .bearer("k")
        .json(&json!({"query_vector": embedding(7), "top_k": 1}))
        .send()
        .json();
    assert_eq!(hit["matches"][0]["id"], "keep");
    let growth = rss(&s).saturating_sub(before);
    assert!(
        growth < 400 * 1024 * 1024,
        "RSS grew by {} MiB during abuse",
        growth >> 20
    );
}

#[test]
fn hostile_paths_methods_and_headers_never_reach_a_5xx() {
    let s = Server::with_key("k", &[]);
    let mut rng = Rng(0xABAD_1DEA_0000_0001);
    let mut paths: Vec<String> = [
        "/",
        "//",
        "/api",
        "/api/v1",
        "/api/v1/",
        "/api/v1/documents/",
        "/api/v1/documents//x",
        "/api/v1/documents/%00",
        "/api/v1/documents/..",
        "/api/v1/documents/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/api/v1/documents/%FF",
        "/api/v1/documents/\u{1F600}",
        "/api/v1/snapshots/%2F",
        "/api/v1/snapshots/%2e%2e/archive",
        "/api/v1/query?x=%",
        "/api/v1/query?%00",
        "/api/v1/documents?limit=%00&after=%zz",
        "/healthz/x",
        "/metrics/",
        "/openapi.json/",
        "/api/v1/admin",
        "/api/v1/admin/keys",
        "/.hms.lock",
        "/tenants",
        "/api/v1/documents/a%2Fb",
    ]
    .iter()
    .map(|p| p.to_string())
    .collect();
    paths.push(format!("/api/v1/documents/{}", "a".repeat(10_000)));
    paths.push(format!("/{}", "%2e".repeat(3000)));
    for _ in 0..200 {
        let n = 1 + rng.below(40);
        let seg: String = (0..n)
            .map(|_| format!("%{:02X}", rng.next() as u8))
            .collect();
        paths.push(format!("/api/v1/documents/{seg}"));
    }
    let methods = [
        "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE",
    ];
    let mut statuses = std::collections::BTreeMap::new();
    for (i, path) in paths.iter().enumerate() {
        let method = methods[i % methods.len()];
        // Raw request lines: the harness would otherwise reject a malformed target itself.
        let raw = format!("{method} {path} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
        let bytes = s.exchange_raw(raw.as_bytes());
        if bytes.is_empty() {
            continue; // hyper may close on an unparsable target
        }
        let text = String::from_utf8_lossy(&bytes);
        let status: u16 = text
            .split(' ')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        assert!(status != 0 && status < 500, "{method} {path:.80}: {status}");
        *statuses.entry(status).or_insert(0) += 1;
    }
    assert!(
        statuses
            .keys()
            .any(|s| *s == 404 || *s == 405 || *s == 422 || *s == 400),
        "{statuses:?}"
    );

    // Header abuse.
    let headers: Vec<(&str, String)> = vec![
        ("X-Tenant-ID", "../../etc".into()),
        ("X-Tenant-ID", "a b".into()),
        ("X-Tenant-ID", "x".repeat(65)),
        ("X-Tenant-ID", String::new()),
        ("X-Tenant-ID", "é".into()),
        ("Idempotency-Key", "bad key with spaces".into()),
        ("Idempotency-Key", "k".repeat(500)),
        ("X-Request-Id", "x".repeat(500)),
        ("Authorization", format!("Bearer {}", "A".repeat(60_000))),
        ("Authorization", "Bearer warm_x.".into()),
        ("If-None-Match", "\"unterminated".into()),
        ("Content-Type", "application/x-ndjson; charset=\"".into()),
    ];
    for (name, value) in &headers {
        let raw = format!(
            "POST /api/v1/query HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\n{name}: {value}\r\nConnection: close\r\nContent-Length: 2\r\n\r\n{{}}"
        );
        let bytes = s.exchange_raw(raw.as_bytes());
        if bytes.is_empty() {
            continue;
        }
        let status: u16 = String::from_utf8_lossy(&bytes)
            .split(' ')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        assert!(status != 0 && status < 500, "{name}: {status}");
    }
    // A flood of the oversized header is bounded by the HTTP layer, not buffered forever.
    let giant = format!(
        "GET /healthz HTTP/1.1\r\nHost: x\r\nX-Big: {}\r\n\r\n",
        "a".repeat(2_000_000)
    );
    let bytes = s.exchange_raw(giant.as_bytes());
    assert!(!String::from_utf8_lossy(&bytes).contains("200 OK"));

    // Malformed framing: conflicting lengths, bad chunks, smuggling shapes.
    let framing: Vec<&[u8]> = vec![
        b"POST /api/v1/query HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nContent-Length: 40\r\nConnection: close\r\n\r\n{}",
        b"POST /api/v1/query HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nContent-Length: 2\r\nConnection: close\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
        b"POST /api/v1/query HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\nzz\r\n{}\r\n0\r\n\r\n",
        b"POST /api/v1/query HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\nConnection: close\r\n\r\n",
        b"GET /healthz HTTP/9.9\r\nHost: x\r\n\r\n",
        b"\x00\x01\x02garbage\r\n\r\n",
        b"GET /healthz HTTP/1.1\r\nBad Header Line\r\n\r\n",
    ];
    for raw in framing {
        let bytes = s.exchange_raw(raw);
        if !bytes.is_empty() {
            let status: u16 = String::from_utf8_lossy(&bytes)
                .split(' ')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            assert!(
                status < 500,
                "{:?} -> {status}",
                String::from_utf8_lossy(raw)
            );
        }
    }
    assert_eq!(s.req("GET", "/healthz").send().status, 200);
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .bearer("k")
            .json(&doc("alive", 3))
            .send()
            .status,
        200
    );
    // Nothing escaped the data directory.
    let tenants = s.data_dir.join("tenants");
    for entry in std::fs::read_dir(&tenants).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(name.bytes().all(|b| b.is_ascii_hexdigit()), "{name}");
    }
}

#[test]
fn abandoned_and_stalled_connections_do_not_leak_work_slots_or_memory() {
    let s = Server::with_key(
        "k",
        &["--max-concurrent-work", "2", "--body-timeout-seconds", "1"],
    );
    let warm: Value = s.req("GET", "/healthz").send().json();
    assert_eq!(warm["status"], "ok");
    let before = rss(&s);
    for round in 0..24 {
        // Promise a body, send a fragment, then vanish or stall.
        let mut c = s.connect();
        let _ = c.write_all(b"POST /api/v1/documents/batch HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\nContent-Length: 100000\r\n\r\n[{\"id\":");
        if round % 2 == 0 {
            drop(c);
        } else {
            let mut buf = [0u8; 256];
            let _ = c.read(&mut buf);
        }
    }
    // All slots come back and the server still works at full concurrency.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let metrics = s.req("GET", "/metrics").bearer("k").send().text();
        if metrics.contains("warm_work_slots_in_use 0") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "work slots leaked:\n{metrics}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .bearer("k")
            .json(&doc("ok", 1))
            .send()
            .status,
        200
    );
    let growth = rss(&s).saturating_sub(before);
    assert!(
        growth < 100 * 1024 * 1024,
        "RSS grew by {} MiB",
        growth >> 20
    );
}
