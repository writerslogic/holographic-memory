// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Injected only into an isolated source archive by probe.py.
use super::*;
use crate::core::documents::{DocumentInput, EncodeMode, IngestOptions};
use crate::core::encoding::TextEncoder;
use serde_json::json;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: Counting = Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

fn input(id: usize, version: usize) -> DocumentInput {
    let embedding: Vec<_> = (0..16)
        .map(|axis| ((id * 131 + axis * 23 + version * 7) as f64 * 0.17).sin())
        .collect();
    serde_json::from_value(json!({
        "id": format!("d{id:05}"), "version": version.to_string(),
        "text": format!("common shared running rare{id} revision{version}"),
        "sourceUri": format!("fixture:source/{id}/{version}"),
        "metadata": {"group": id % 2}, "embeddings": [embedding],
    }))
    .unwrap()
}

fn prepared_document(id: usize, version: usize) -> StoredDocument {
    super::super::prepare_with_options(
        input(id, version),
        1024,
        Some(16),
        None,
        TextEncoder::default(),
        &IngestOptions {
            encode: EncodeMode::Lexical,
            ..Default::default()
        },
    )
    .unwrap()
}

fn train(store: &mut DocumentStore) -> u64 {
    let start = Instant::now();
    if let Some(snapshot) = super::super::capture_dense_index(store, 16).unwrap() {
        let trained = super::super::build_dense_index(snapshot).unwrap().unwrap();
        assert!(super::super::publish_dense_index(store, &trained));
    }
    start.elapsed().as_nanos() as u64
}

fn query(index: usize, hot: bool) -> (String, SearchOptions) {
    let text = if hot {
        "common shared running".into()
    } else {
        (0..8)
            .map(|offset| format!("rare{}", 1 + 2 * (index * 67 + offset * 37) % 9998))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let embedding = (0..16)
        .map(|axis| ((index * 17 + axis * 7) as f64 * 0.11).cos())
        .collect();
    (
        text,
        SearchOptions {
            k: Some(10),
            candidate_limit: Some(32),
            embedding: Some(embedding),
            min_semantic_score: Some(-1.0),
            filter: Some(json!({"group": 1})),
            fusion: Some("rrf".into()),
            ..Default::default()
        },
    )
}

fn queries(store: &DocumentStore, phase: &str) -> Value {
    let mut samples = Vec::new();
    let mut checksum = crc32fast::Hasher::new();
    for hot in [false, true] {
        for index in 0..128 {
            let (text, options) = query(index, hot);
            let start_bytes = LIVE.load(Ordering::Relaxed);
            PEAK.store(start_bytes, Ordering::Relaxed);
            let start_calls = CALLS.load(Ordering::Relaxed);
            let start = Instant::now();
            let prepared = prepare_search(&text, &options, Some(16)).unwrap();
            let prepared_at = Instant::now();
            let captured = capture_search(store, vec![prepared]);
            let captured_at = Instant::now();
            let captured_chunks = captured.queries[0].chunks.len();
            let actual = execute_search(&captured, 0);
            let ended = Instant::now();
            let peak_bytes = PEAK.load(Ordering::Relaxed).saturating_sub(start_bytes);
            let allocations = CALLS.load(Ordering::Relaxed) - start_calls;
            let actual = serde_json::to_value(actual).unwrap();
            let expected =
                super::super::search_impl(store, &text, &options, Some(16), false).unwrap();
            assert_eq!(
                actual,
                serde_json::to_value(expected).unwrap(),
                "{phase} hot={hot} query={index}"
            );
            checksum.update(&serde_json::to_vec(&actual).unwrap());
            samples.push(json!({"hot":hot, "query":index,
                "prepare_ns": (prepared_at-start).as_nanos(),
                "capture_ns": (captured_at-prepared_at).as_nanos(),
                "execute_ns": (ended-captured_at).as_nanos(), "total_ns": (ended-start).as_nanos(),
                "captured_chunks":captured_chunks, "peak_requested_bytes":peak_bytes,
                "allocations":allocations}));
        }
    }
    json!({"phase":phase, "oracle_checks":256, "ordered_results_crc32":checksum.finalize(), "samples":samples})
}

fn engine_lifecycle() -> Value {
    use crate::core::WarmConfig;
    use crate::{EmbeddingSpace, WarmCore};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().display().to_string();
    let config = WarmConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "campaign-synthetic".into(),
            revision: "1".into(),
            dimensions: 16,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    let options = IngestOptions {
        encode: EncodeMode::Lexical,
        ..Default::default()
    };
    let start = Instant::now();
    let core = WarmCore::new(1024, Some(path.clone()), Some(config.clone())).unwrap();
    let opened = Instant::now();
    for first in (0..1000).step_by(250) {
        core.memorize_documents_with(
            (first..first + 250).map(|id| input(id, 1)).collect(),
            &options,
        )
        .unwrap();
    }
    let ingested = Instant::now();
    core.train_document_index().unwrap();
    let trained = Instant::now();
    let (text, options) = query(1, false);
    let expected = serde_json::to_value(core.search_documents(&text, &options).unwrap()).unwrap();
    let searched = Instant::now();
    core.flush().unwrap();
    drop(core);
    let closed = Instant::now();
    let core = WarmCore::new(1024, Some(path), Some(config)).unwrap();
    let reopened = Instant::now();
    assert_eq!(
        serde_json::to_value(core.search_documents(&text, &options).unwrap()).unwrap(),
        expected
    );
    let read = Instant::now();
    json!({"rows":1000, "create_ns":(opened-start).as_nanos(),
        "ingest_ns":(ingested-opened).as_nanos(), "train_ns":(trained-ingested).as_nanos(),
        "first_query_ns":(searched-trained).as_nanos(), "flush_close_ns":(closed-searched).as_nanos(),
        "reopen_ns":(reopened-closed).as_nanos(), "reopened_query_ns":(read-reopened).as_nanos(),
        "live_requested_bytes":LIVE.load(Ordering::Relaxed)})
}

#[test]
#[ignore]
fn campaign_retrieval_probe() {
    let start = Instant::now();
    let disable_delta = std::env::var_os("WARM_PROBE_DISABLE_DELTA").is_some();
    let mut store = DocumentStore::default();
    for id in 0..10000 {
        let document = prepared_document(id, 1);
        store.insert(document.id.clone(), document);
    }
    let ingest_ns = start.elapsed().as_nanos();
    let train_ns = train(&mut store);
    let initial = queries(&store, "initial");
    let mut mutations = Vec::new();
    let mut rebuilds = 0;
    for operation in 0..32 {
        let start = Instant::now();
        match operation % 4 {
            0 => {
                let d = prepared_document(operation, 2);
                store.insert(d.id.clone(), d);
            }
            1 => {
                store.remove(&format!("d{operation:05}"));
            }
            2 => {
                let d = prepared_document(10000 + operation, 2);
                store.insert(d.id.clone(), d);
            }
            _ => {
                let d = prepared_document(3, operation + 2);
                store.insert(d.id.clone(), d);
            }
        }
        if disable_delta {
            store.dense_index = None;
        }
        let update_ns = start.elapsed().as_nanos();
        let recommended = store.dense_index_recommended();
        let maintenance_ns = if recommended {
            rebuilds += 1;
            train(&mut store)
        } else {
            0
        };
        mutations.push(
            json!({"update_ns":update_ns,"maintenance_ns":maintenance_ns,"rebuilt":recommended}),
        );
    }
    let updated = queries(&store, "updated");
    let census = store.memory_census().report();
    drop(store);
    let lifecycle = engine_lifecycle();
    let report = json!({"variant":std::env::var("WARM_PROBE_VARIANT").unwrap(),
        "rows":10000,"dimensions":16,"ingest_ns":ingest_ns,"initial_train_ns":train_ns,
        "maintenance_rebuilds":rebuilds,"mutations":mutations,"queries":[initial,updated],
        "store_census":census,"engine_lifecycle":lifecycle,"elapsed_seconds":start.elapsed().as_secs_f64(),
        "memory_note":"Counting system allocator requested bytes; atomic instrumentation included identically in all variants. RSS measured by parent; not allocator usable size."});
    std::fs::write(
        std::env::var_os("WARM_PROBE_OUT").unwrap(),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
