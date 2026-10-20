use super::*;
use std::io::{Read, Write};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn state(directory: &std::path::Path, limit: usize) -> Arc<AppState> {
    let cli = Cli::try_parse_from([
        "hms-server",
        "--data-dir",
        directory.to_str().unwrap(),
        "--dim",
        "256",
        "--input-dim",
        "2",
        "--max-concurrent-work",
        &limit.to_string(),
    ])
    .unwrap();
    Arc::new(AppState {
        cli,
        api_key: Some(b"test-key".to_vec()),
        tenants: Mutex::new(HashMap::new()),
        work: WorkLimit::new(limit),
        #[cfg(feature = "local-models")]
        stages: None,
    })
}

async fn serve(state: Arc<AppState>) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, router(state)).await.unwrap();
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
    let permit = state.work.acquire().unwrap();
    let pending = tokio::spawn(blocking(permit, move || {
        let tenant = work_state.tenant("actual")?;
        started.send(()).unwrap();
        reader.read_exact(&mut [0u8]).map_err(ApiError::internal)?;
        tenant
            .core
            .memorize_document(DocumentInput {
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
    assert_eq!(body["error"]["code"], "server_busy");
    assert!(headers.contains("retry-after: 1"));
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
        request(address, "/api/v1/query", true).await.2["error"]["code"],
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
            .tenant("actual")
            .unwrap()
            .core
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
    assert_eq!(response.2["error"]["code"], "server_busy");
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
    let mut state = state(directory.path(), 1);
    Arc::get_mut(&mut state).unwrap().cli.body_timeout_seconds = 1;
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
    assert_eq!(
        serde_json::from_str::<Value>(body).unwrap(),
        json!({
            "error": {"code": "request_timeout", "message": "request body timed out"}
        })
    );
    wait_permits(&state, 1).await;
    assert_eq!(request(address, "/api/v1/query", true).await.0, 200);
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}
