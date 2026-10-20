// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The memory API through the real `hms-server` process: HTTP contract, error classes,
//! limits, tenant isolation, configuration file and persistence across a restart.

#![cfg(feature = "server")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};

const DIM: usize = 8;

struct Server {
    child: Child,
    addr: SocketAddr,
}

impl Server {
    fn start(dir: &std::path::Path, extra: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_hms-server"))
            .args(["--bind", "127.0.0.1:0", "--input-dim", &DIM.to_string()])
            .args(["--dim", "2048", "--data-dir"])
            .arg(dir)
            .args(extra)
            .env_remove("HMS_API_KEY")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
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
        Self { child, addr }
    }

    fn post(&self, op: &str, tenant: &str, body: &Value) -> (u16, Value) {
        let body = body.to_string();
        let mut stream = TcpStream::connect(self.addr).unwrap();
        let request = format!(
            "POST /api/v1/memory/{op} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
             Content-Type: application/json\r\nX-Tenant-ID: {tenant}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes()).unwrap();
        let mut raw = Vec::new();
        let _ = stream.read_to_end(&mut raw);
        let text = String::from_utf8_lossy(&raw);
        let status = text.split(' ').nth(1).and_then(|s| s.parse().ok()).unwrap();
        let body = text.split("\r\n\r\n").nth(1).unwrap_or("");
        (status, serde_json::from_str(body).unwrap_or(Value::Null))
    }

    fn stop(&mut self) {
        Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(self.child.wait().unwrap().success());
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn unit(axis: usize) -> Vec<f64> {
    (0..DIM)
        .map(|i| if i == axis { 1.0 } else { 0.0 })
        .collect()
}

fn ids(response: &Value) -> Vec<&str> {
    response["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect()
}

#[test]
fn memory_round_trip_over_http_with_history_errors_and_isolation() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::start(dir.path(), &[]);

    let (status, out) = server.post(
        "remember",
        "a",
        &json!({"id": "c1", "text": "Alice lives in Paris", "subject": "alice/city",
                "validFrom": 100, "embedding": unit(0)}),
    );
    assert_eq!(status, 200, "{out}");
    assert_eq!(out["remembered"][0]["created"], json!(true));
    let (status, out) = server.post(
        "remember",
        "a",
        &json!({"facts": [{"id": "c2", "text": "Alice lives in Berlin", "validFrom": 200,
                           "supersedes": ["c1"], "embedding": unit(1)}]}),
    );
    assert_eq!(status, 200, "{out}");
    assert_eq!(out["remembered"][0]["supersedes"], json!(["c1"]));

    // Current only by default; history on request; as-of in the past; vectors rank.
    let (_, current) = server.post(
        "recall",
        "a",
        &json!({"query": "Alice lives", "asOf": 1000}),
    );
    assert_eq!(ids(&current), ["c2"]);
    let (_, all) = server.post(
        "recall",
        "a",
        &json!({"query": "Alice lives", "history": "all", "asOf": 1000}),
    );
    assert_eq!(all["results"].as_array().unwrap().len(), 2);
    let (_, early) = server.post("recall", "a", &json!({"query": "Alice lives", "asOf": 150}));
    assert_eq!(ids(&early), ["c1"]);
    let (_, semantic) = server.post(
        "recall",
        "a",
        &json!({"query": "", "embedding": unit(1), "history": "all", "asOf": 1000,
                "lexicalWeight": 0.0, "minSemanticScore": 0.5}),
    );
    assert_eq!(ids(&semantic), ["c2"]);

    // Explain, get and history.
    let (status, explained) = server.post(
        "explain",
        "a",
        &json!({"query": "Alice Berlin", "id": "c2", "asOf": 1000}),
    );
    assert_eq!(status, 200, "{explained}");
    assert_eq!(
        explained["explanation"]["matchedTerms"],
        json!(["alice", "berlin"])
    );
    assert_eq!(explained["explanation"]["chain"][0]["id"], json!("c1"));
    let (_, got) = server.post("get", "a", &json!({"id": "c1"}));
    assert_eq!(got["fact"]["supersededBy"], json!(["c2"]));
    let (_, history) = server.post("history", "a", &json!({"id": "c2"}));
    assert_eq!(history["history"].as_array().unwrap().len(), 2);

    // Tenants are isolated.
    let (_, other) = server.post("recall", "b", &json!({"query": "Alice", "history": "all"}));
    assert!(ids(&other).is_empty());

    // Error classes.
    assert_eq!(server.post("nope", "a", &json!({})).0, 404);
    assert_eq!(
        server.post("history", "a", &json!({"id": "missing"})).0,
        404
    );
    let (status, conflict) =
        server.post("remember", "a", &json!({"id": "c1", "text": "different"}));
    assert_eq!(
        (status, conflict["error"]["code"].as_str()),
        (409, Some("conflict"))
    );
    for bad in [
        json!({"id": "x", "text": "t", "unknownField": 1}),
        json!({"id": "x", "text": "t", "embedding": [1.0, 2.0]}),
        json!({"id": "hms:x", "text": "t"}),
        json!({"id": "x", "text": "t", "supersedes": ["missing"]}),
        json!({"facts": []}),
    ] {
        let (status, body) = server.post("remember", "a", &bad);
        assert_eq!(status, 422, "{bad} -> {body}");
        assert_eq!(body["error"]["code"], json!("invalid_request"));
    }
    assert_eq!(
        server.post("recall", "a", &json!({"query": "q", "k": 0})).0,
        422
    );
    assert_eq!(server.post("recall", "a", &json!({"k": 3})).0, 422);

    // Retrying is idempotent.
    let (_, again) = server.post(
        "remember",
        "a",
        &json!({"id": "c1", "text": "Alice lives in Paris", "subject": "alice/city",
                "validFrom": 100, "embedding": unit(0)}),
    );
    assert_eq!(again["remembered"][0]["created"], json!(false));

    // Forget: dry run, then for real, then repeated.
    let scope = json!({"kind": "subject", "subject": "alice/city"});
    let (_, dry) = server.post("forget", "a", &json!({"scope": scope, "dryRun": true}));
    assert_eq!(dry["forgotten"], json!(["c1"]));
    let (_, done) = server.post("forget", "a", &json!({"scope": scope}));
    assert_eq!(done["forgotten"], json!(["c1"]));
    let (_, repeat) = server.post("forget", "a", &json!({"scope": scope}));
    assert_eq!(repeat["forgotten"], json!([]));
    let (_, now) = server.post("recall", "a", &json!({"query": "Alice", "history": "all"}));
    assert_eq!(ids(&now), ["c2"]);
}

#[test]
fn memory_state_and_configuration_survive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("memory.json");
    std::fs::write(
        &config,
        json!({"supersession": "latest-by-subject", "defaultHistory": "current", "defaultK": 3})
            .to_string(),
    )
    .unwrap();
    let data = dir.path().join("data");
    let args = ["--memory-config", config.to_str().unwrap()];
    let mut server = Server::start(&data, &args);
    for (id, from) in [("v1", 100), ("v2", 200), ("v3", 300)] {
        let (status, out) = server.post(
            "remember",
            "t",
            &json!({"id": id, "text": format!("setting value {id}"), "subject": "setting",
                    "validFrom": from}),
        );
        assert_eq!(status, 200, "{out}");
    }
    server.stop();

    let server = Server::start(&data, &args);
    let (_, current) = server.post(
        "recall",
        "t",
        &json!({"query": "setting value", "asOf": 1000}),
    );
    assert_eq!(ids(&current), ["v3"]);
    let (_, early) = server.post(
        "recall",
        "t",
        &json!({"query": "setting value", "asOf": 250}),
    );
    assert_eq!(ids(&early), ["v2"]);
    let (_, history) = server.post("history", "t", &json!({"id": "v1"}));
    assert_eq!(history["history"].as_array().unwrap().len(), 3);
}

#[test]
fn invalid_memory_config_stops_the_server_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("memory.json");
    std::fs::write(&config, r#"{"defaultK": 0}"#).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_hms-server"))
        .args(["--bind", "127.0.0.1:0", "--data-dir"])
        .arg(dir.path().join("data"))
        .args(["--memory-config", config.to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(!status.success());
}
