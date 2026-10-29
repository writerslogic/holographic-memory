// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::{body::Body, http::Request};
use serde_json::{json, Value};
use tower_service::Service;

use crate::tenants::AppState;

async fn call(state: Arc<AppState>, method: &str, path: &str, value: Value) -> (u16, Value) {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("Authorization", "Bearer test-key")
        .header("Content-Type", "application/json")
        .body(Body::from(value.to_string()))
        .unwrap();
    let response = crate::routes::router(state).call(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn imported(id: &str, owner: char) -> Value {
    let identity = owner.to_string().repeat(64);
    json!({"id":id,"vector":[1,0],"text":"imported source", "import_identity":identity,
        "metadata":{"hms_import_identity":identity,"hms_source":"fixture", "hms_namespace":"ns", "hms_source_id":id,"hms_import_parts":1}})
}

#[tokio::test]
async fn committed_cleanup_failure_invalidates_cached_document_reads() {
    for memory in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let state = crate::tests::state_with(dir.path(), 2, &[]);
        let (path, input) = if memory {
            (
                "/api/v1/memory/remember",
                json!({"id":"doc", "text":"retained fact"}),
            )
        } else {
            (
                "/api/v1/documents",
                json!({"id":"doc", "text":"retained fact", "vector":[1,0]}),
            )
        };
        assert_eq!(call(state.clone(), "POST", path, input).await.0, 200);
        let request = Request::builder()
            .uri("/api/v1/documents")
            .header("Authorization", "Bearer test-key")
            .body(Body::empty())
            .unwrap();
        let response = crate::routes::router(state.clone())
            .call(request)
            .await
            .unwrap();
        let tag = response.headers()["etag"].clone();
        let blocked = dir.path().join("tenants/64656661756c74/indices.json");
        if blocked.is_file() {
            std::fs::remove_file(&blocked).unwrap();
        }
        std::fs::create_dir(&blocked).unwrap();
        let (status, problem) = if memory {
            call(
                state.clone(),
                "POST",
                "/api/v1/memory/forget",
                json!({
                    "scope":{"kind":"ids", "ids":["doc"]}, "purge":true, "dryRun":false
                }),
            )
            .await
        } else {
            call(
                state.clone(),
                "DELETE",
                "/api/v1/documents/doc?purge=true",
                Value::Null,
            )
            .await
        };
        assert_eq!(status, 500, "{problem}");
        assert_eq!(problem["code"], "mutation_committed");
        let request = Request::builder()
            .uri("/api/v1/documents")
            .header("Authorization", "Bearer test-key")
            .header("If-None-Match", tag)
            .body(Body::empty())
            .unwrap();
        let response = crate::routes::router(state.clone())
            .call(request)
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let body = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["documents"],
            json!([])
        );
        std::fs::remove_dir(&blocked).unwrap();
    }
}

#[tokio::test]
async fn import_ownership_is_checked_on_upsert_batch_and_conditional_delete() {
    let dir = tempfile::tempdir().unwrap();
    let state = crate::tests::state_with(dir.path(), 2, &[]);
    assert_eq!(
        call(
            state.clone(),
            "POST",
            "/api/v1/documents",
            imported("doc", 'a')
        )
        .await
        .0,
        200
    );
    for replacement in [imported("doc", 'b'), json!({"id":"doc","vector":[0,1]})] {
        assert_eq!(
            call(state.clone(), "POST", "/api/v1/documents", replacement)
                .await
                .0,
            409
        );
    }
    assert_eq!(
        call(
            state.clone(),
            "POST",
            "/api/v1/documents/batch",
            json!([imported("collision", 'a'), imported("collision", 'b')])
        )
        .await
        .0,
        409
    );
    assert!(!state.core("default").unwrap().has_document("collision"));
    assert_eq!(
        call(
            state.clone(),
            "DELETE",
            "/api/v1/documents/doc",
            Value::Null
        )
        .await
        .0,
        409
    );
    let wrong = format!("/api/v1/documents/doc?import_identity={}", "b".repeat(64));
    assert_eq!(
        call(state.clone(), "DELETE", &wrong, Value::Null).await.0,
        409
    );
    assert_eq!(
        call(
            state.clone(),
            "POST",
            "/api/v1/documents",
            imported("doc", 'a')
        )
        .await
        .0,
        200
    );
    let correct = format!("/api/v1/documents/doc?import_identity={}", "a".repeat(64));
    assert_eq!(
        call(state.clone(), "DELETE", &correct, Value::Null).await.0,
        200
    );
    assert!(!state.core("default").unwrap().has_document("doc"));
}

#[tokio::test]
async fn legacy_adoption_requires_explicit_exact_source_and_layout() {
    let dir = tempfile::tempdir().unwrap();
    let state = crate::tests::state_with(dir.path(), 2, &[]);
    let legacy = json!({"id":"doc","vector":[1,0],"metadata":{"hms_source":"fixture","hms_namespace":"ns","hms_source_id":"doc"}});
    assert_eq!(
        call(state.clone(), "POST", "/api/v1/documents", legacy)
            .await
            .0,
        200
    );
    assert_eq!(
        call(
            state.clone(),
            "POST",
            "/api/v1/documents",
            imported("doc", 'a')
        )
        .await
        .0,
        409
    );
    let mut adopt = imported("doc", 'a');
    adopt["import_adopt_legacy"] = json!(true);
    for (key, value) in [
        ("hms_source_id", json!("other")),
        ("hms_part", json!(0)),
        ("hms_import_parts", json!(2)),
    ] {
        let mut wrong = adopt.clone();
        wrong["metadata"][key] = value;
        assert_eq!(
            call(state.clone(), "POST", "/api/v1/documents", wrong)
                .await
                .0,
            409
        );
    }
    assert_eq!(
        call(state.clone(), "POST", "/api/v1/documents", adopt)
            .await
            .0,
        200
    );
}

#[tokio::test]
async fn identity_validation_and_explicit_id_query_boundaries_are_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let state = crate::tests::state_with(dir.path(), 2, &[]);
    for length in [63, 64, 65] {
        let identity = "a".repeat(length);
        let mut doc = imported(&format!("doc{length}"), 'a');
        doc["import_identity"] = json!(identity);
        doc["metadata"]["hms_import_identity"] = json!(identity);
        assert_eq!(
            call(state.clone(), "POST", "/api/v1/documents", doc)
                .await
                .0,
            if length == 64 { 200 } else { 422 }
        );
    }
    for count in [999, 1000, 1001] {
        let ids: Vec<_> = (0..count).map(|i| format!("id{i}")).collect();
        assert_eq!(
            call(
                state.clone(),
                "POST",
                "/api/v1/query",
                json!({"query_vector":[1,0],"document_ids":ids})
            )
            .await
            .0,
            if count <= 1000 { 200 } else { 422 }
        );
    }
    for ids in [
        json!([]),
        json!(["doc64", "doc64"]),
        json!(["hms:reserved"]),
    ] {
        assert_eq!(
            call(
                state.clone(),
                "POST",
                "/api/v1/query",
                json!({"query_vector":[1,0],"document_ids":ids})
            )
            .await
            .0,
            422
        );
    }
    let reply = call(
        state.clone(),
        "POST",
        "/api/v1/query",
        json!({"query_vector":[1,0],"index":"graph","document_ids":["doc64"]}),
    )
    .await;
    assert_eq!(reply.0, 200);
    assert_eq!(reply.1["matches"][0]["id"], "doc64");
    let filtered = call(
        state.clone(),
        "POST",
        "/api/v1/query",
        json!({"query_vector":[1,0],"document_ids":["doc64"],"filter":{"missing":"value"}}),
    )
    .await;
    assert_eq!(filtered.0, 200);
    assert_eq!(filtered.1["matches"], json!([]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_import_owners_cannot_both_claim_one_destination() {
    let dir = tempfile::tempdir().unwrap();
    let state = crate::tests::state_with(dir.path(), 2, &[]);
    let a = tokio::spawn(call(
        state.clone(),
        "POST",
        "/api/v1/documents",
        imported("shared", 'a'),
    ));
    let b = tokio::spawn(call(
        state.clone(),
        "POST",
        "/api/v1/documents",
        imported("shared", 'b'),
    ));
    let mut statuses = vec![a.await.unwrap().0, b.await.unwrap().0];
    statuses.sort();
    assert_eq!(statuses, vec![200, 409]);
}
