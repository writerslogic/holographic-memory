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
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use writerslogic_warm::{DocumentInput, SearchOptions};

use crate::config::Cli;
use crate::error::ApiError;
use crate::request::blocking;
use crate::tenants::{tenant_from_hex, tenant_hex, AppState};
use crate::{auth, routes};

pub(crate) fn state_with(
    directory: &std::path::Path,
    limit: usize,
    extra: &[&str],
) -> Arc<AppState> {
    let mut args = vec![
        "warm-server".to_string(),
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

/// Every flag has an `WARM_*` environment variable and a row in the configuration reference.
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
        assert!(env.starts_with("WARM_"), "{env}");
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

#[test]
fn idempotency_configuration_rejects_disabled_protection_bounds() {
    for option in [
        "--idempotency-capacity",
        "--idempotency-max-bytes",
        "--idempotency-ttl-seconds",
    ] {
        for value in [0, 1, 2] {
            let config = Cli::try_parse_from(["warm-server", option, &value.to_string()])
                .unwrap()
                .config;
            let result = config.validate(true);
            if value == 0 {
                assert!(result.unwrap_err().to_string().contains(option));
            } else {
                result.unwrap();
            }
        }
    }
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
    for op in writerslogic_warm::core::memory_api::OPERATIONS {
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

#[tokio::test(flavor = "current_thread")]
async fn a_dropped_request_leaves_the_in_flight_gauge() {
    use axum::middleware::from_fn_with_state;
    use tower_service::Service;
    let directory = tempfile::tempdir().unwrap();
    let state = state(directory.path(), 1);
    let mut app = axum::Router::new()
        .route(
            "/hang",
            axum::routing::get(std::future::pending::<&'static str>),
        )
        .layer(from_fn_with_state(
            state.clone(),
            crate::middleware::observe,
        ));
    let request = axum::http::Request::builder()
        .uri("/hang")
        .body(axum::body::Body::empty())
        .unwrap();
    let call = app.call(request);
    assert!(tokio::time::timeout(Duration::from_millis(50), call)
        .await
        .is_err());
    assert_eq!(state.metrics.in_flight.load(Ordering::Relaxed), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn an_abandoned_restore_upload_leaves_no_archive_behind() {
    let directory = tempfile::tempdir().unwrap();
    let state = state(directory.path(), 1);
    let chunks = futures_util::StreamExt::chain(
        futures_util::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(&[0; 1024]))
        }),
        futures_util::stream::pending(),
    );
    let mut request = axum::http::Request::builder()
        .method("PUT")
        .uri("/api/v1/restore")
        .body(axum::body::Body::from_stream(chunks))
        .unwrap();
    request.extensions_mut().insert(Arc::new(auth::Principal {
        key_id: "test".into(),
        tenants: auth::TenantAccess::Any,
        scopes: auth::ScopeSet::ALL,
    }));
    let upload = crate::admin::restore_upload(axum::extract::State(state.clone()), request);
    assert!(tokio::time::timeout(Duration::from_millis(200), upload)
        .await
        .is_err());
    let left: Vec<_> = std::fs::read_dir(state.staging_root())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert!(left.is_empty(), "staging still holds {left:?}");
}

async fn stream_documents(state: Arc<AppState>, chunks: Vec<Vec<u8>>) -> (u16, Value) {
    use tower_service::Service;

    let mut app = routes::router(state);
    let chunks = futures_util::stream::iter(chunks.into_iter().map(Ok::<_, std::io::Error>));
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/api/v1/documents/stream")
        .header("Authorization", "Bearer test-key")
        .header("Content-Type", "application/x-ndjson")
        .body(axum::body::Body::from_stream(chunks))
        .unwrap();
    let response = app.call(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test(flavor = "current_thread")]
async fn stream_batch_commit_boundaries_are_independent_of_body_frames() {
    for (documents, final_newline, quota, committed) in [(3, true, "2", 2), (2, false, "1", 0)] {
        let mut payload = (0..documents)
            .map(|i| format!("{{\"id\":\"doc{i}\",\"vector\":[1,0]}}\n"))
            .collect::<String>()
            .into_bytes();
        if !final_newline {
            payload.pop();
        }
        for frame_size in [payload.len(), 1] {
            let directory = tempfile::tempdir().unwrap();
            let state = state_with(
                directory.path(),
                1,
                &["--stream-batch", "2", "--max-documents-per-tenant", quota],
            );
            let chunks = payload.chunks(frame_size).map(<[u8]>::to_vec).collect();
            let (status, body) = stream_documents(state.clone(), chunks).await;
            assert_eq!(status, 429, "frame size {frame_size}: {body}");
            assert_eq!(body["code"], "quota_exceeded");
            let core = state.core("default").unwrap();
            assert_eq!(core.document_count(), committed, "frame size {frame_size}");
            for i in 0..documents {
                assert_eq!(core.has_document(&format!("doc{i}")), i < committed);
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn stream_batch_preserves_line_errors_and_final_unterminated_document() {
    let payload = concat!(
        "\n{\"id\":\"a\",\"vector\":[1,0]}\n \r\n",
        "{bad}\n",
        "{\"id\":\"b\",\"vector\":[1,0]}\n",
        "{\"id\":\"c\",\"vector\":[1,0]}\n",
        "{\"id\":\"d\",\"vector\":[1,0]}"
    )
    .as_bytes();
    for frame_size in [payload.len(), 7] {
        let directory = tempfile::tempdir().unwrap();
        let state = state_with(directory.path(), 1, &["--stream-batch", "2"]);
        let chunks = payload.chunks(frame_size).map(<[u8]>::to_vec).collect();
        let (status, body) = stream_documents(state.clone(), chunks).await;
        assert_eq!(status, 200, "frame size {frame_size}: {body}");
        assert_eq!(body["added"], 4);
        assert_eq!(body["rejected"], 1);
        assert_eq!(body["errors_truncated"], false);
        assert_eq!(body["errors"].as_array().unwrap().len(), 1);
        assert_eq!(body["errors"][0]["line"], 4);
        let core = state.core("default").unwrap();
        assert_eq!(core.document_count(), 4);
        assert!(core.has_document("d"));
    }
}

#[tokio::test]
async fn body_failure_distinguishes_size_limits_from_transport_errors() {
    use axum::body::Body;
    let directory = tempfile::tempdir().unwrap();
    let state = state_with(directory.path(), 1, &["--max-body-bytes", "8"]);
    for size in [7, 8, 9] {
        let chunks = futures_util::stream::iter([
            Ok::<_, std::io::Error>(bytes::Bytes::from(vec![0; 4])),
            Ok(bytes::Bytes::from(vec![0; size - 4])),
        ]);
        let result = crate::request::read_body(&state, Body::from_stream(chunks)).await;
        if size <= 8 {
            assert_eq!(result.unwrap().len(), size);
        } else {
            let error = result.unwrap_err();
            assert_eq!(error.status.as_u16(), 413);
            assert_eq!(error.code, "payload_too_large");
        }
    }
    let chunks = futures_util::stream::iter([
        Ok(bytes::Bytes::from_static(b"{")),
        Err(std::io::Error::other("sensitive transport detail")),
    ]);
    let error = crate::request::read_body(&state, Body::from_stream(chunks))
        .await
        .unwrap_err();
    assert_eq!(error.status.as_u16(), 400);
    assert_eq!(error.code, "invalid_body");
    assert!(!error
        .problem()
        .to_string()
        .contains("sensitive transport detail"));
}

async fn failed_body_request(path: &str, method: &str, prefix: &'static [u8], committed: usize) {
    use tower_service::Service;
    let directory = tempfile::tempdir().unwrap();
    let state = state_with(directory.path(), 1, &["--stream-batch", "1"]);
    let chunks = futures_util::stream::iter([
        Ok(bytes::Bytes::from_static(prefix)),
        Err(std::io::Error::other("sensitive transport detail")),
    ]);
    let request = axum::http::Request::builder()
        .method(method)
        .uri(path)
        .header("Authorization", "Bearer test-key")
        .header(
            "Content-Type",
            if path.ends_with("stream") {
                "application/x-ndjson"
            } else {
                "application/json"
            },
        )
        .header("Idempotency-Key", "failed-upload")
        .body(axum::body::Body::from_stream(chunks))
        .unwrap();
    let mut app = routes::router(state.clone());
    let response = app.call(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let problem: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(state.work.slots.available_permits(), 1);
    assert_eq!(state.metrics.in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(state.idem.usage(), (0, 0, 0));
    if state.staging_root().exists() {
        assert_eq!(std::fs::read_dir(state.staging_root()).unwrap().count(), 0);
    }
    assert_eq!(state.core("default").unwrap().document_count(), committed);
    assert_eq!(status, 400);
    assert_eq!(problem["code"], "invalid_body");
    assert_eq!(problem["status"], 400);
    assert!(!problem.to_string().contains("sensitive transport detail"));
}

#[tokio::test]
async fn body_failure_buffered_write_does_not_commit_or_reserve_a_retry_key() {
    failed_body_request(
        "/api/v1/documents",
        "POST",
        br#"{"id":"partial","vector":[1,0]}"#,
        0,
    )
    .await;
}

#[tokio::test]
async fn body_failure_stream_preserves_prior_committed_batches() {
    failed_body_request(
        "/api/v1/documents/stream",
        "POST",
        b"{\"id\":\"partial\",\"vector\":[1,0]}\n",
        1,
    )
    .await;
}

#[tokio::test]
async fn body_failure_restore_cleans_up_the_incomplete_archive() {
    failed_body_request("/api/v1/restore", "PUT", &[0; 1024], 0).await;
}

#[tokio::test]
async fn explicit_graph_query_has_a_documented_unavailability_problem_and_exact_fallback() {
    use tower_service::Service;
    let directory = tempfile::tempdir().unwrap();
    let state = state(directory.path(), 1);
    let mut app = routes::router(state);
    for (mode, expected) in [("graph", 409), ("exact", 200)] {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/api/v1/query")
            .header("Authorization", "Bearer test-key")
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(
                serde_json::json!({"query_vector":[1,0],"index":mode}).to_string(),
            ))
            .unwrap();
        let response = app.call(request).await.unwrap();
        assert_eq!(response.status().as_u16(), expected);
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        if mode == "graph" {
            assert_eq!(body["code"], "index_unavailable");
            assert_eq!(body["title"], "Graph index unavailable");
            assert!(body["type"]
                .as_str()
                .unwrap()
                .ends_with("#index_unavailable"));
        } else {
            assert_eq!(body["matches"], serde_json::json!([]));
        }
    }
}
