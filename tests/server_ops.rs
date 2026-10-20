// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "server")]

mod server_support;

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use serde_json::json;
use server_support::*;

fn wait_until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn count(s: &Server, tenant: &str) -> u64 {
    s.req("GET", "/api/v1/stats").tenant(tenant).send().json()["documents"]
        .as_u64()
        .unwrap_or(u64::MAX)
}

fn tar_with(entries: &[(&str, &[u8], tar::EntryType)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, data, kind) in entries {
        let mut h = tar::Header::new_gnu();
        // Write the name bytes directly so a hostile path such as `../x` can be represented.
        h.as_gnu_mut().unwrap().name[..name.len()].copy_from_slice(name.as_bytes());
        h.set_entry_type(*kind);
        h.set_size(data.len() as u64);
        h.set_mode(0o644);
        if *kind == tar::EntryType::Symlink {
            h.as_gnu_mut().unwrap().linkname[..6].copy_from_slice(b"/etc/x");
        }
        h.set_cksum();
        out.extend_from_slice(h.as_bytes());
        out.extend_from_slice(data);
        out.extend_from_slice(&vec![0; (512 - data.len() % 512) % 512]);
    }
    out.extend_from_slice(&[0; 1024]);
    out
}

#[test]
fn snapshot_restore_and_off_box_backup_round_trip() {
    let a = Server::with_key("root", &[]);
    let ok = |r: server_support::Resp, status: u16| {
        assert_eq!(r.status, status, "{}", r.text());
        r.json()
    };
    ok(
        a.req("POST", "/api/v1/documents/batch")
            .bearer("root")
            .tenant("t")
            .json(&json!([doc("a", 1), doc("b", 2)]))
            .send(),
        200,
    );
    let snap = ok(
        a.req("POST", "/api/v1/snapshots")
            .bearer("root")
            .tenant("t")
            .send(),
        201,
    );
    let id = snap["id"].as_str().unwrap().to_string();
    assert!(snap["bytes"].as_u64().unwrap() > 0);
    ok(
        a.req("POST", "/api/v1/documents")
            .bearer("root")
            .tenant("t")
            .json(&doc("c", 3))
            .send(),
        200,
    );
    assert_eq!(
        a.req("GET", "/api/v1/stats")
            .bearer("root")
            .tenant("t")
            .send()
            .json()["documents"],
        3
    );
    let listed = ok(
        a.req("GET", "/api/v1/snapshots")
            .bearer("root")
            .tenant("t")
            .send(),
        200,
    );
    assert_eq!(listed["snapshots"][0]["id"], id.as_str());
    // Other tenants do not see it.
    assert_eq!(
        ok(
            a.req("GET", "/api/v1/snapshots")
                .bearer("root")
                .tenant("u")
                .send(),
            200
        )["snapshots"],
        json!([])
    );

    // Off-box copy: the tar stream, without the process lock.
    let archive = a
        .req("GET", &format!("/api/v1/snapshots/{id}/archive"))
        .bearer("root")
        .tenant("t")
        .send();
    assert_eq!(archive.status, 200);
    assert_eq!(archive.header("content-type"), Some("application/x-tar"));
    let names: Vec<String> = tar::Archive::new(archive.body.as_slice())
        .entries()
        .unwrap()
        .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(names.iter().any(|n| n.ends_with("store.json")), "{names:?}");
    assert!(!names.iter().any(|n| n.contains(".hms.lock")), "{names:?}");

    // Restore in place: the live store goes back to two documents and the replaced store is kept.
    let restored = ok(
        a.req("POST", &format!("/api/v1/snapshots/{id}/restore"))
            .bearer("root")
            .tenant("t")
            .send(),
        200,
    );
    assert_eq!(restored["restored"], true);
    assert_eq!(count_with(&a, "root", "t"), 2);
    let hit = a
        .req("POST", "/api/v1/query")
        .bearer("root")
        .tenant("t")
        .json(&json!({"query_vector": embedding(2), "top_k": 1}))
        .send()
        .json();
    assert_eq!(hit["matches"][0]["id"], "b");
    let after = ok(
        a.req("GET", "/api/v1/snapshots")
            .bearer("root")
            .tenant("t")
            .send(),
        200,
    );
    let previous = restored["previous_snapshot"].as_str().unwrap();
    assert!(previous.starts_with("pre-restore-"));
    assert!(after["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["id"] == previous));
    // The pre-restore copy has the third document: restoring it again undoes the restore.
    ok(
        a.req("POST", &format!("/api/v1/snapshots/{previous}/restore"))
            .bearer("root")
            .tenant("t")
            .send(),
        200,
    );
    assert_eq!(count_with(&a, "root", "t"), 3);
    // Writes after a restore work (the engine reopened cleanly).
    ok(
        a.req("POST", "/api/v1/documents")
            .bearer("root")
            .tenant("t")
            .json(&doc("d", 4))
            .send(),
        200,
    );

    // A different server restores the uploaded archive: a real backup and disaster-recovery path.
    let b = Server::with_key("root", &[]);
    let up = b
        .req("PUT", "/api/v1/restore")
        .bearer("root")
        .tenant("t2")
        .header("Content-Type", "application/x-tar")
        .body(archive.body.clone())
        .send();
    assert_eq!(up.status, 200, "{}", up.text());
    assert_eq!(count_with(&b, "root", "t2"), 2);
    let hit = b
        .req("POST", "/api/v1/query")
        .bearer("root")
        .tenant("t2")
        .json(&json!({"query_vector": embedding(1), "top_k": 1}))
        .send()
        .json();
    assert_eq!(hit["matches"][0]["id"], "a");
    assert!((hit["matches"][0]["score"].as_f64().unwrap() - 1.0).abs() < 1e-9);

    // Delete.
    ok(
        a.req("DELETE", &format!("/api/v1/snapshots/{id}"))
            .bearer("root")
            .tenant("t")
            .send(),
        200,
    );
    assert_eq!(
        a.req("DELETE", &format!("/api/v1/snapshots/{id}"))
            .bearer("root")
            .tenant("t")
            .send()
            .status,
        404
    );
}

fn count_with(s: &Server, key: &str, tenant: &str) -> u64 {
    s.req("GET", "/api/v1/stats")
        .bearer(key)
        .tenant(tenant)
        .send()
        .json()["documents"]
        .as_u64()
        .unwrap()
}

#[test]
fn restore_rejects_hostile_or_damaged_archives_and_leaves_the_store_alone() {
    let s = Server::with_key("root", &[]);
    s.req("POST", "/api/v1/documents/batch")
        .bearer("root")
        .tenant("t")
        .json(&json!([doc("a", 1), doc("b", 2)]))
        .send();
    let snap = s
        .req("POST", "/api/v1/snapshots")
        .bearer("root")
        .tenant("t")
        .send()
        .json();
    let id = snap["id"].as_str().unwrap();
    let good = s
        .req("GET", &format!("/api/v1/snapshots/{id}/archive"))
        .bearer("root")
        .tenant("t")
        .send()
        .body;

    let file = tar::EntryType::Regular;
    let mut damaged = good.clone();
    let mid = damaged.len() / 2;
    damaged.truncate(mid);
    let mut flipped = good.clone();
    for byte in flipped.iter_mut().skip(2048).step_by(997) {
        *byte ^= 0xFF;
    }
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("parent traversal", tar_with(&[("../evil", b"x", file)])),
        ("absolute path", tar_with(&[("/tmp/evil", b"x", file)])),
        (
            "symlink",
            tar_with(&[("link", b"", tar::EntryType::Symlink)]),
        ),
        ("not a tar", b"definitely not a tar archive".repeat(100)),
        ("empty tar", tar_with(&[])),
        ("truncated", damaged),
        ("corrupted payload", flipped),
        (
            "no store inside",
            tar_with(&[("readme.txt", b"hello", file)]),
        ),
    ];
    for (name, archive) in cases {
        let r = s
            .req("PUT", "/api/v1/restore")
            .bearer("root")
            .tenant("t")
            .body(archive)
            .send();
        assert!(
            matches!(r.status, 422 | 500),
            "{name}: {} {}",
            r.status,
            r.text()
        );
        assert_eq!(
            r.header("content-type"),
            Some("application/problem+json"),
            "{name}"
        );
        assert_eq!(count_with(&s, "root", "t"), 2, "{name} changed the store");
        // Reads and writes still work, so the engine is open and healthy after the failure.
        assert_eq!(
            s.req("POST", "/api/v1/documents")
                .bearer("root")
                .tenant("t")
                .json(&doc("a", 1))
                .send()
                .status,
            200,
            "{name}"
        );
    }
    // The escape did not happen and nothing is left behind in staging.
    let staging = s.data_dir.join("staging");
    let leftovers: Vec<_> = std::fs::read_dir(&staging)
        .map(|d| d.flatten().collect())
        .unwrap_or_default();
    assert!(leftovers.is_empty(), "staging not cleaned: {leftovers:?}");
    assert!(
        !s.data_dir.join("evil").exists() && !s.data_dir.parent().unwrap().join("evil").exists()
    );

    // Size limit.
    let small = Server::with_key("root", &["--max-restore-bytes", "4096"]);
    let r = small
        .req("PUT", "/api/v1/restore")
        .bearer("root")
        .body(good)
        .send();
    assert_eq!(r.status, 413);
    // Snapshot ids never reach the filesystem unvalidated.
    for bad in ["..", "..%2F..%2Ftenants", "%2E%2E", ".hidden"] {
        let r = s
            .req("GET", &format!("/api/v1/snapshots/{bad}/archive"))
            .bearer("root")
            .tenant("t")
            .send();
        assert!(matches!(r.status, 404 | 422), "{bad}: {}", r.status);
        let r = s
            .req("DELETE", &format!("/api/v1/snapshots/{bad}"))
            .bearer("root")
            .tenant("t")
            .send();
        assert!(matches!(r.status, 404 | 422), "{bad}: {}", r.status);
    }
    assert_eq!(count_with(&s, "root", "t"), 2);
}

#[test]
fn compaction_runs_on_schedule_only_when_changed_and_on_demand() {
    let s = Server::with_key("root", &["--compaction-interval-seconds", "1"]);
    let status = |tenant: &str| {
        s.req("GET", "/api/v1/compaction")
            .bearer("root")
            .tenant(tenant)
            .send()
            .json()
    };
    assert_eq!(status("t")["scheduled_interval_seconds"], 1);
    s.req("POST", "/api/v1/documents/batch")
        .bearer("root")
        .tenant("t")
        .json(&json!([doc("a", 1), doc("b", 2)]))
        .send();
    wait_until("a scheduled compaction", || {
        status("t")["status"]["runs"].as_u64().unwrap() >= 1
    });
    let st = status("t")["status"].clone();
    assert_eq!(st["last_trigger"], "scheduled");
    assert_eq!(st["failures"], 0);
    assert!(st["last_started_at"].as_str().unwrap().ends_with('Z'));
    // Unchanged since the last run: the schedule does not recompact.
    let runs = st["runs"].as_u64().unwrap();
    std::thread::sleep(Duration::from_millis(2500));
    assert_eq!(status("t")["status"]["runs"].as_u64().unwrap(), runs);
    // On demand, and the data survives.
    let manual = s
        .req("POST", "/api/v1/compaction")
        .bearer("root")
        .tenant("t")
        .send();
    assert_eq!(manual.status, 200, "{}", manual.text());
    assert_eq!(manual.json()["status"]["last_trigger"], "manual");
    assert_eq!(manual.json()["status"]["runs"].as_u64().unwrap(), runs + 1);
    let hit = s
        .req("POST", "/api/v1/query")
        .bearer("root")
        .tenant("t")
        .json(&json!({"query_vector": embedding(2), "top_k": 1}))
        .send()
        .json();
    assert_eq!(hit["matches"][0]["id"], "b");
    let metrics = s.req("GET", "/metrics").bearer("root").send().text();
    assert!(
        metrics.contains(&format!(
            "hms_tenant_compactions_total{{tenant=\"t\"}} {}",
            runs + 1
        )),
        "{metrics}"
    );
}

#[test]
fn idle_tenants_close_and_reopen_over_http() {
    let s = Server::open(&["--tenant-idle-seconds", "1", "--max-tenants", "1"]);
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .tenant("a")
            .json(&doc("x", 1))
            .send()
            .status,
        200
    );
    // The one slot is taken until the idle sweep frees it.
    wait_until("tenant b to open", || {
        s.req("POST", "/api/v1/documents")
            .tenant("b")
            .json(&doc("y", 2))
            .send()
            .status
            == 200
    });
    wait_until("tenant a to reopen", || count(&s, "a") == 1);
    let metrics = s.req("GET", "/metrics").send().text();
    assert!(metrics.contains("hms_tenant_evictions_total"), "{metrics}");
    assert!(
        !metrics.contains("hms_tenant_evictions_total 0\n"),
        "{metrics}"
    );
}

#[cfg(unix)]
#[test]
fn unix_socket_serves_with_private_permissions_and_replaces_a_stale_socket() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::Builder::new().prefix("h").tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let sock_arg = sock.to_str().unwrap();
    let mut s = Server::open(&["--unix-socket", sock_arg]);
    assert_eq!(
        std::fs::metadata(&sock).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(s.req("GET", "/healthz").send().json()["status"], "ok");
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .json(&doc("u", 1))
            .send()
            .status,
        200
    );
    // A second server refuses a live socket.
    let stderr = Server::refuses(&["--no-auth", "--unix-socket", sock_arg], &[]);
    assert!(stderr.contains("already in use"), "{stderr}");
    // Killed without cleanup: the stale file is replaced on the next start.
    let _ = s.child.kill();
    let _ = s.child.wait();
    assert!(sock.exists());
    let s2 = Server::open(&["--unix-socket", sock_arg]);
    assert_eq!(s2.req("GET", "/healthz").send().status, 200);
    drop(s2);
    assert!(!sock.exists(), "clean exit removes the socket");
    // A regular file is never clobbered.
    std::fs::write(&sock, b"precious").unwrap();
    assert!(
        Server::refuses(&["--no-auth", "--unix-socket", sock_arg], &[]).contains("not a socket")
    );
    assert_eq!(std::fs::read(&sock).unwrap(), b"precious");
}

fn self_signed(names: &[&str]) -> (String, String) {
    let key =
        rcgen::generate_simple_self_signed(names.iter().map(|n| n.to_string()).collect::<Vec<_>>())
            .unwrap();
    (key.cert.pem(), key.signing_key.serialize_pem())
}

#[test]
fn tls_serves_https_and_reloads_the_certificate_on_sighup() {
    let dir = tempfile::tempdir().unwrap();
    let (cert, key) = (dir.path().join("cert.pem"), dir.path().join("key.pem"));
    let (pem_a, key_a) = self_signed(&["localhost"]);
    std::fs::write(&cert, &pem_a).unwrap();
    std::fs::write(&key, key_a).unwrap();
    let mut s = Server::with_key(
        "tls-key",
        &[
            "--tls-cert",
            cert.to_str().unwrap(),
            "--tls-key",
            key.to_str().unwrap(),
        ],
    );
    assert_eq!(s.req("GET", "/healthz").send().status, 200);
    assert_eq!(
        s.req("POST", "/api/v1/documents")
            .bearer("tls-key")
            .json(&doc("t", 1))
            .send()
            .status,
        200
    );
    // Plain HTTP to the TLS port gets nothing usable.
    {
        let mut tcp = std::net::TcpStream::connect(s.addr()).unwrap();
        tcp.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let _ = tcp.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
        let mut buf = Vec::new();
        let _ = tcp.read_to_end(&mut buf);
        assert!(!String::from_utf8_lossy(&buf).contains("200 OK"));
    }
    wait_until("handshake failure metric", || {
        s.req("GET", "/metrics")
            .bearer("tls-key")
            .send()
            .text()
            .contains("hms_tls_handshake_failures_total 1")
    });
    // Rotate: new certificate on disk, SIGHUP, clients that trust only the new one connect.
    let (pem_b, key_b) = self_signed(&["localhost"]);
    std::fs::write(&cert, &pem_b).unwrap();
    std::fs::write(&key, key_b).unwrap();
    // A mismatched pair is rejected and the old certificate stays in use.
    std::fs::write(&key, self_signed(&["localhost"]).1).unwrap();
    s.signal("HUP");
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(
        s.req("GET", "/healthz").send().status,
        200,
        "old certificate still serves"
    );
    assert!(s
        .log_json()
        .iter()
        .any(|e| e["event"] == "tls_reload_failed"));
    let refused = s
        .req("POST", "/api/v1/admin/tls/reload")
        .bearer("tls-key")
        .send();
    assert_eq!(refused.status, 422, "{}", refused.text());
    assert_eq!(s.req("GET", "/healthz").send().status, 200);
    // Now a matching pair. The image has no shell for a signal, so reload is also an endpoint.
    let (pem_c, key_c) = self_signed(&["localhost"]);
    std::fs::write(&cert, &pem_c).unwrap();
    std::fs::write(&key, key_c).unwrap();
    let ok = s
        .req("POST", "/api/v1/admin/tls/reload")
        .bearer("tls-key")
        .send();
    assert_eq!(ok.json()["reloaded"], true, "{}", ok.text());
    s.tls_root = Some(pem_c.into_bytes());
    wait_until("new certificate", || {
        s.req("GET", "/healthz")
            .try_send()
            .is_some_and(|r| r.status == 200)
    });
    s.tls_root = Some(pem_a.into_bytes());
    assert!(
        s.req("GET", "/healthz").try_send().is_none(),
        "the old certificate is gone"
    );
    assert!(s.stop());
}

#[test]
fn slow_header_clients_are_dropped_and_the_connection_cap_holds() {
    let s = Server::open(&[
        "--header-read-timeout-seconds",
        "1",
        "--max-connections",
        "2",
    ]);
    // Slowloris: never finish the headers.
    let started = Instant::now();
    let mut slow = s.connect();
    slow.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\nX-Slow: ")
        .unwrap();
    let mut buf = [0u8; 64];
    let _ = slow.read(&mut buf);
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(900) && waited < Duration::from_secs(6),
        "{waited:?}"
    );
    assert_eq!(s.req("GET", "/healthz").send().status, 200);

    // Two idle connections hold both slots; a third request is not served until one closes.
    let mut idle_a = s.connect();
    let mut idle_b = s.connect();
    idle_a
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
        .unwrap();
    idle_b
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
        .unwrap();
    let mut buf = [0u8; 512];
    assert!(idle_a.read(&mut buf).unwrap() > 0);
    assert!(idle_b.read(&mut buf).unwrap() > 0);
    let third = std::thread::spawn({
        let addr = s.addr();
        move || {
            let mut c = std::net::TcpStream::connect(addr).unwrap();
            c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            c.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut raw = Vec::new();
            let _ = c.read_to_end(&mut raw);
            (
                Instant::now(),
                String::from_utf8_lossy(&raw).contains("200 OK"),
            )
        }
    });
    std::thread::sleep(Duration::from_millis(400));
    assert!(
        !third.is_finished(),
        "third connection served while the cap was full"
    );
    let released = Instant::now();
    drop(idle_a);
    let (served_at, ok) = third.join().unwrap();
    assert!(ok && served_at >= released);
}

#[test]
fn shutdown_drains_in_flight_requests_and_flushes_before_exiting() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let mut s = Server::start_in(&data, &["--no-auth"], &[]);
    let body = doc("drained", 5).to_string();
    let (head, tail) = body.split_at(body.len() / 2);
    let mut conn = s.connect();
    conn.write_all(
        format!("POST /api/v1/documents HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{head}", body.len())
            .as_bytes(),
    )
    .unwrap();
    wait_until("the request to be admitted", || {
        s.req("GET", "/metrics")
            .send()
            .text()
            .contains("hms_work_slots_in_use 1")
    });
    s.signal("TERM");
    // The listener closes at once and readiness drops while the request is still running.
    wait_until("listener to close", || {
        std::net::TcpStream::connect(s.addr()).is_err()
    });
    conn.write_all(tail.as_bytes()).unwrap();
    let mut raw = Vec::new();
    let _ = conn.read_to_end(&mut raw);
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 200"), "{text}");
    assert!(text.contains("{\"added\":1}"));
    assert!(s.child.wait().unwrap().success());
    let events = s.log_json();
    let names: Vec<&str> = events.iter().filter_map(|e| e["event"].as_str()).collect();
    let started = names.iter().position(|n| *n == "shutdown_started").unwrap();
    let complete = names
        .iter()
        .position(|n| *n == "shutdown_complete")
        .unwrap();
    assert!(started < complete);
    // The accepted write is durable.
    let s2 = Server::start_in(&data, &["--no-auth"], &[]);
    let hit = s2
        .req("POST", "/api/v1/query")
        .json(&json!({"query_vector": embedding(5), "top_k": 1}))
        .send()
        .json();
    assert_eq!(hit["matches"][0]["id"], "drained");
}

#[test]
fn readiness_fails_during_the_shutdown_delay_while_requests_are_still_served() {
    let mut s = Server::open(&["--shutdown-delay-seconds", "2"]);
    assert_eq!(s.req("GET", "/readyz").send().status, 200);
    let signalled = Instant::now();
    s.signal("TERM");
    wait_until("readiness to fail", || {
        s.req("GET", "/readyz")
            .try_send()
            .is_some_and(|r| r.status == 503)
    });
    let not_ready = s.req("GET", "/readyz").send();
    assert_eq!(not_ready.json()["code"], "not_ready");
    // Work is still accepted during the delay, so a balancer that has not noticed loses nothing.
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .json(&json!({"query_vector": embedding(1)}))
            .send()
            .status,
        200
    );
    assert!(s
        .req("GET", "/metrics")
        .send()
        .text()
        .contains("hms_ready 0"));
    assert!(s.child.wait().unwrap().success());
    assert!(
        signalled.elapsed() >= Duration::from_millis(1900),
        "exited before the delay: {:?}",
        signalled.elapsed()
    );
}

#[test]
fn healthcheck_subcommand_reflects_readiness_for_containers_without_curl() {
    let probe = |addr: &str| {
        std::process::Command::new(BIN)
            .args(["healthcheck", "--bind", addr])
            .env_remove("HMS_BIND")
            .output()
            .unwrap()
    };
    let mut s = Server::open(&["--shutdown-delay-seconds", "3"]);
    let addr = s.addr().to_string();
    assert!(probe(&addr).status.success());
    // An unspecified address is probed as loopback, as the image's HMS_BIND=0.0.0.0:8080 needs.
    let any = format!("0.0.0.0:{}", s.addr().port());
    assert!(probe(&any).status.success());
    s.signal("TERM");
    wait_until("readiness to fail", || !probe(&addr).status.success());
    let failed = probe(&addr);
    assert!(String::from_utf8_lossy(&failed.stderr).contains("503"));
    assert!(s.child.wait().unwrap().success());
    assert!(!probe(&addr).status.success(), "nothing listens after exit");
}

#[test]
fn a_snapshot_pauses_only_its_own_tenant_and_never_hangs_requests() {
    let s = std::sync::Arc::new(Server::open(&[
        "--max-concurrent-work",
        "4",
        "--max-queue",
        "16",
    ]));
    let docs: Vec<_> = (0..200).map(|i| doc(&format!("d{i}"), i)).collect();
    s.req("POST", "/api/v1/documents/batch")
        .tenant("big")
        .json(&json!(docs))
        .send();
    s.req("POST", "/api/v1/documents")
        .tenant("calm")
        .json(&doc("c", 1))
        .send();
    let query = json!({"query_vector": embedding(1)});
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hammer = |tenant: &'static str| {
        let (s, stop, query) = (s.clone(), stop.clone(), query.clone());
        std::thread::spawn(move || {
            let mut seen = Vec::new();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let started = Instant::now();
                let r = s
                    .req("POST", "/api/v1/query")
                    .tenant(tenant)
                    .json(&query)
                    .send();
                let code = r.json()["code"].as_str().unwrap_or("").to_string();
                seen.push((r.status, code, started.elapsed()));
            }
            seen
        })
    };
    let big = hammer("big");
    let calm = hammer("calm");
    for _ in 0..6 {
        let r = s.req("POST", "/api/v1/snapshots").tenant("big").send();
        assert_eq!(r.status, 201, "{}", r.text());
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let (big, calm) = (big.join().unwrap(), calm.join().unwrap());
    assert!(
        big.iter()
            .all(|(st, code, _)| *st == 200 || (*st == 503 && code == "tenant_busy")),
        "{big:?}"
    );
    assert!(
        calm.iter().all(|(st, _, _)| *st == 200),
        "another tenant was disturbed: {calm:?}"
    );
    assert!(
        big.iter()
            .chain(calm.iter())
            .all(|(_, _, t)| *t < Duration::from_secs(5)),
        "a request hung"
    );
    assert_eq!(
        s.req("POST", "/api/v1/query")
            .tenant("big")
            .json(&query)
            .send()
            .status,
        200
    );
}
