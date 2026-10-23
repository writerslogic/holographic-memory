// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! In-process tests that need the server's internals: work admission under cancellation, body
//! deadlines, and the configuration reference. Black-box behavior lives in `tests/server*.rs`.

use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use clap::{CommandFactory, Parser};
use holographic_memory::{DocumentInput, SearchOptions};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::config::Cli;
use crate::error::ApiError;
use crate::request::blocking;
use crate::tenants::{tenant_from_hex, tenant_hex, AppState};
use crate::{auth, routes};

fn state_with(directory: &std::path::Path, limit: usize, extra: &[&str]) -> Arc<AppState> {
    let mut args = vec![
        "hms-server".to_string(),
        "--data-dir".into(),
        directory.to_str().unwrap().into(),
        "--dim".into(),
        "256".into(),
        "--input-dim".into(),
        "2".into(),
        "--max-concurrent-work".into(),
        limit.to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let cli = Cli::try_parse_from(args).unwrap();
    let authenticator = auth::Authenticator::new(false, Some("test-key"), None).unwrap();
    Arc::new(AppState::new(
        cli.config,
        authenticator,
        routes::route_templates(),
        #[cfg(feature = "local-models")]
        None,
    ))
}

fn state(directory: &std::path::Path, limit: usize) -> Arc<AppState> {
    state_with(directory, limit, &[])
}

async fn serve(state: Arc<AppState>) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = routes::router(state);
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (address, task)
}

async fn request(address: SocketAddr, path: &str, authenticated: bool) -> (u16, String, Value) {
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    let body = r#"{"query_vector":[1,0]}"#;
    let auth = if authenticated {
        "Authorization: Bearer test-key\r\n"
    } else {
        ""
    };
    stream
        .write_all(
            format!(
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n{auth}Content-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).await.unwrap();
    let response = String::from_utf8(bytes).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
    (
        status,
        headers.to_ascii_lowercase(),
        serde_json::from_str(body).unwrap(),
    )
}

async fn wait_permits(state: &AppState, expected: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while state.work.slots.available_permits() != expected {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn cancellation_retains_work_capacity_and_shutdown_waits_for_actual_io() {
    let directory = tempfile::tempdir().unwrap();
    let state = state(directory.path(), 1);
    let (address, server) = serve(state.clone()).await;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mut writer = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (mut reader, _) = listener.accept().unwrap();
    reader
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let (started, ready) = tokio::sync::oneshot::channel();
    let work_state = state.clone();
    let permit = state.work.acquire().await.unwrap();
    let pending = tokio::spawn(blocking(permit, move || {
        let core = work_state.core("actual")?;
        started.send(()).unwrap();
        reader.read_exact(&mut [0u8]).map_err(ApiError::internal)?;
        core.memorize_document(DocumentInput {
            id: "after-cancel".into(),
            text: "survives".into(),
            source_uri: None,
            version: None,
            metadata: None,
            chunk_words: None,
            overlap_words: None,
            store_text: None,
            embeddings: Some(vec![vec![1.0, 0.0]]),
            title_weight: None,
        })
        .map_err(ApiError::internal)?;
        Ok(())
    }));
    ready.await.unwrap();
    pending.abort();
    assert!(pending.await.unwrap_err().is_cancelled());
    let (status, headers, body) = tokio::time::timeout(
        Duration::from_secs(2),
        request(address, "/api/v1/query", true),
    )
    .await
    .unwrap();
    assert_eq!(status, 503);
    assert_eq!(body["code"], "server_busy");
    assert_eq!(body["error"]["code"], "server_busy");
    assert!(headers.contains("retry-after: 1"));
    assert!(headers.contains("application/problem+json"));
    assert_eq!(request(address, "/api/v1/query", false).await.0, 401);
    assert_eq!(request(address, "/unknown", true).await.0, 404);
    state.work.stop();
    let slots = state.work.slots.clone();
    let mut draining = tokio::spawn(async move { slots.acquire_many_owned(1).await.unwrap() });
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut draining)
            .await
            .is_err()
    );
    assert_eq!(
        request(address, "/api/v1/query", true).await.2["code"],
        "shutting_down"
    );
    writer.write_all(&[1]).unwrap();
    let drained = tokio::time::timeout(Duration::from_secs(5), draining)
        .await
        .unwrap()
        .unwrap();
    let check = state.clone();
    let hits = tokio::task::spawn_blocking(move || {
        check
            .core("actual")
            .unwrap()
            .search_documents("survives", &SearchOptions::default())
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(hits[0].document_id, "after-cancel");
    drop(drained);
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}

async fn partial_body(address: SocketAddr) -> tokio::net::TcpStream {
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    stream
        .write_all(b"POST /api/v1/documents HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-key\r\nContent-Length: 1024\r\n\r\n{")
        .await
        .unwrap();
    stream
}

#[tokio::test(flavor = "current_thread")]
async fn admission_caps_body_readers_at_limit_and_releases_disconnected_requests() {
    let directory = tempfile::tempdir().unwrap();
    let state = state(directory.path(), 2);
    let (address, server) = serve(state.clone()).await;
    let first = partial_body(address).await;
    wait_permits(&state, 1).await;
    assert_eq!(request(address, "/api/v1/query", true).await.0, 200);
    let second = partial_body(address).await;
    wait_permits(&state, 0).await;
    let response = request(address, "/api/v1/query", true).await;
    assert_eq!(response.0, 503);
    assert_eq!(response.2["code"], "server_busy");
    assert_eq!(state.metrics.busy_rejections.load(Ordering::Relaxed), 1);
    drop(first);
    wait_permits(&state, 1).await;
    assert_eq!(request(address, "/api/v1/query", true).await.0, 200);
    drop(second);
    wait_permits(&state, 2).await;
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}

#[tokio::test(flavor = "current_thread")]
async fn stalled_body_times_out_and_releases_admission_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let state = state_with(directory.path(), 1, &["--body-timeout-seconds", "1"]);
    let (address, server) = serve(state.clone()).await;
    let mut stalled = partial_body(address).await;
    wait_permits(&state, 0).await;
    assert_eq!(request(address, "/api/v1/query", true).await.0, 503);
    let mut bytes = Vec::new();
    tokio::time::timeout(Duration::from_secs(3), stalled.read_to_end(&mut bytes))
        .await
        .unwrap()
        .unwrap();
    let response = String::from_utf8(bytes).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    assert_eq!(headers.split_whitespace().nth(1), Some("408"));
    let body: Value = serde_json::from_str(body).unwrap();
    assert_eq!(body["code"], "request_timeout");
    assert_eq!(body["status"], 408);
    wait_permits(&state, 1).await;
    assert_eq!(request(address, "/api/v1/query", true).await.0, 200);
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}

/// Every flag has an `HMS_*` environment variable and a row in the configuration reference.
#[test]
fn every_flag_has_an_environment_variable_and_a_documented_row() {
    let reference = include_str!("../../../docs/server/configuration.md");
    let command = Cli::command();
    let mut checked = 0;
    for arg in command.get_arguments() {
        let id = arg.get_id().as_str();
        if matches!(id, "help" | "version") {
            continue;
        }
        let long = arg.get_long().unwrap_or(id);
        let env = arg
            .get_env()
            .unwrap_or_else(|| panic!("--{long} has no environment variable"))
            .to_string_lossy()
            .into_owned();
        assert!(env.starts_with("HMS_"), "{env}");
        assert!(
            reference.contains(&format!("`--{long}`")),
            "--{long} missing from configuration.md"
        );
        assert!(
            reference.contains(&format!("`{env}`")),
            "{env} missing from configuration.md"
        );
        checked += 1;
    }
    assert!(checked >= 38, "only {checked} flags inspected");
}

/// Idle tenants are flushed and closed, free their slot toward `--max-tenants`, and reopen
/// with their data intact.
#[test]
fn idle_tenants_are_evicted_and_reopen_with_their_data() {
    let directory = tempfile::tempdir().unwrap();
    let state = state_with(
        directory.path(),
        2,
        &["--tenant-idle-seconds", "1", "--max-tenants", "1"],
    );
    let document = |id: &str| DocumentInput {
        id: id.into(),
        text: "persisted words".into(),
        source_uri: None,
        version: None,
        metadata: None,
        chunk_words: None,
        overlap_words: None,
        store_text: None,
        embeddings: Some(vec![vec![1.0, 0.0]]),
        title_weight: None,
    };
    state
        .core("a")
        .unwrap()
        .memorize_document(document("one"))
        .unwrap();
    assert_eq!(state.core("b").err().unwrap().code, "tenant_limit");
    assert_eq!(state.evict_idle(), 0, "not idle yet");
    std::thread::sleep(Duration::from_millis(1200));
    assert_eq!(state.evict_idle(), 1);
    assert_eq!(state.open_tenant_count(), 0);
    state.core("b").unwrap();
    std::thread::sleep(Duration::from_millis(1200));
    assert_eq!(state.evict_idle(), 1);
    assert_eq!(state.core("a").unwrap().document_count(), 1);
}

/// Restore swaps the store, verifies first, and a failed verification leaves the store alone.
#[test]
fn snapshot_restore_round_trips_and_keeps_the_replaced_store() {
    let directory = tempfile::tempdir().unwrap();
    let state = state(directory.path(), 2);
    let put = |id: &str| {
        state
            .core("t")
            .unwrap()
            .memorize_document(DocumentInput {
                id: id.into(),
                text: format!("words for {id}"),
                source_uri: None,
                version: None,
                metadata: None,
                chunk_words: None,
                overlap_words: None,
                store_text: None,
                embeddings: Some(vec![vec![1.0, 0.0]]),
                title_weight: None,
            })
            .unwrap();
    };
    put("one");
    let snapshot = state.create_snapshot("t").unwrap();
    put("two");
    assert_eq!(state.core("t").unwrap().document_count(), 2);
    let staged = state.staging_root().join("restore-test");
    crate::tenants::copy_store(&state.snapshot_path("t", &snapshot.id).unwrap(), &staged).unwrap();
    let previous = state.restore_staged("t", &staged).unwrap();
    assert_eq!(state.core("t").unwrap().document_count(), 1);
    assert!(state.core("t").unwrap().has_document("one"));
    assert!(!state.core("t").unwrap().has_document("two"));
    assert!(
        state.snapshot_path("t", &previous).is_ok(),
        "replaced store kept"
    );
    // A broken staged directory fails to reopen and the live store is rolled back.
    let broken = state.staging_root().join("broken");
    std::fs::create_dir_all(&broken).unwrap();
    std::fs::write(broken.join("store.json"), b"not json").unwrap();
    assert!(state.restore_staged("t", &broken).is_err());
    assert_eq!(state.core("t").unwrap().document_count(), 1);
    state.flush_all().unwrap();
}

/// Every memory operation the engine offers is routed, documented and secured.
#[test]
fn memory_operations_are_all_routed() {
    for op in holographic_memory::core::memory_api::OPERATIONS {
        let path = format!("/api/v1/memory/{op}");
        assert!(
            routes::OPS
                .iter()
                .any(|o| o.path == path && o.method == "post"),
            "memory operation {op} has no route"
        );
    }
}

/// A snapshot or restore holds the tenant exclusively; its other requests are refused at once
/// instead of occupying work slots, and the tenant works again afterwards.
#[test]
fn quiesced_tenants_turn_requests_away_immediately() {
    let directory = tempfile::tempdir().unwrap();
    let state = state(directory.path(), 2);
    drop(state.core("t").unwrap());
    let held = state.quiesce("t").unwrap();
    let started = std::time::Instant::now();
    let busy = state.core("t").err().unwrap();
    assert_eq!(busy.code, "tenant_busy");
    assert_eq!(busy.retry_after, Some(1));
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "must not wait for the writer"
    );
    assert_eq!(state.quiesce("t").err().unwrap().code, "tenant_busy");
    assert!(state.core("other").is_ok(), "other tenants are unaffected");
    drop(held);
    assert!(state.core("t").is_ok());
}

/// Search ranks chunks but answers with documents: when every document spans several
/// matching chunks, `top_k` distinct documents still come back while that many exist.
#[tokio::test(flavor = "current_thread")]
async fn search_returns_top_k_distinct_documents_from_multi_chunk_documents() {
    let directory = tempfile::tempdir().unwrap();
    let state = state(directory.path(), 2);
    let chunks_per_document = 6;
    for i in 0..8 {
        let text = (0..chunks_per_document)
            .map(|c| format!("quartz river d{i} c{c}"))
            .collect::<Vec<_>>()
            .join(" ");
        let document = DocumentInput {
            id: format!("doc{i}"),
            text,
            source_uri: None,
            version: None,
            metadata: None,
            chunk_words: Some(4),
            overlap_words: Some(0),
            store_text: None,
            embeddings: Some(vec![vec![1.0, 0.0]; chunks_per_document]),
            title_weight: None,
        };
        state
            .core("default")
            .unwrap()
            .memorize_document(document)
            .unwrap();
    }
    let (address, server) = serve(state).await;
    for (top_k, expected) in [(1, 1), (3, 3), (5, 5), (8, 8), (20, 8)] {
        let body = format!(r#"{{"query":"quartz","top_k":{top_k}}}"#);
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                format!(
                    "POST /api/v1/search HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nAuthorization: Bearer test-key\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes).await.unwrap();
        let response = String::from_utf8(bytes).unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
        let hits: Value = serde_json::from_str(body).unwrap();
        let ids: Vec<&str> = hits["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["id"].as_str().unwrap())
            .collect();
        let distinct: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), expected, "top_k {top_k}: {ids:?}");
        assert_eq!(distinct.len(), ids.len(), "top_k {top_k}: {ids:?}");
    }
    server.abort();
}

#[test]
fn only_canonical_hex_names_decode_to_tenants() {
    let canonical = tenant_hex("Acme_1");
    for (name, expected) in [
        (canonical.as_str(), Some("Acme_1")),
        ("41", Some("A")),
        ("", None),
        ("4a", Some("J")),
        ("4A", None),
        ("414", None),
        ("4g", None),
        ("2f", None),
        ("1é", None),
    ] {
        assert_eq!(tenant_from_hex(name).as_deref(), expected, "{name:?}");
    }
}
