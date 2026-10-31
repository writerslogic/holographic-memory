// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

use writerslogic_warm::core::format::scan;
use writerslogic_warm::{core::WarmConfig, WarmCore};

struct Counted;
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
fn record(bytes: usize) {
    if TRACK.try_with(Cell::get).unwrap_or(false) {
        BYTES.with(|v| v.set(v.get() + bytes));
        CALLS.with(|v| v.set(v.get() + 1));
    }
}
unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counted = Counted;

fn measured<T>(f: impl FnOnce() -> T) -> (T, usize, usize, u128) {
    BYTES.with(|v| v.set(0));
    CALLS.with(|v| v.set(0));
    TRACK.with(|v| v.set(true));
    let start = Instant::now();
    let result = f();
    let micros = start.elapsed().as_micros();
    TRACK.with(|v| v.set(false));
    (result, BYTES.with(Cell::get), CALLS.with(Cell::get), micros)
}

#[cfg(feature = "security")]
#[test]
fn malformed_identity_files_do_not_allocate_their_declared_length() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("key");
    let mut baseline = None;
    for size in [31, 33, 128 << 20] {
        std::fs::File::create(&path).unwrap().set_len(size).unwrap();
        let mut config = WarmConfig::default();
        config.security.signing_enabled = true;
        config.security.key_path = Some(path.display().to_string());
        let store = directory.path().join(format!("store-{size}"));
        let (result, bytes, _, _) =
            measured(|| WarmCore::new(256, Some(store.display().to_string()), Some(config)));
        assert!(result.is_err());
        let baseline = *baseline.get_or_insert(bytes);
        assert!(
            bytes <= baseline + 8192,
            "{size}-byte input allocated {bytes} bytes, small malformed input used {baseline}"
        );
        assert_eq!(std::fs::metadata(&path).unwrap().len(), size);
    }
}

#[test]
fn extreme_privacy_scale_fails_before_store_creation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("must-not-create");
    let mut config = WarmConfig::default();
    config.privacy.epsilon = 1e-308;
    let error = WarmCore::new(256, Some(path.display().to_string()), Some(config))
        .err()
        .unwrap();
    assert!(error.to_string().contains("privacy epsilon"));
    assert!(!path.exists());
}

#[test]
fn scanner_reuses_a_single_raw_buffer() {
    let raw = vec![7u8; 1024 * 1024];
    let compressed = lz4_flex::compress(&raw);
    let mut frame = crc32fast::hash(&raw).to_le_bytes().to_vec();
    frame.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    frame.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    frame.extend_from_slice(&0u32.to_le_bytes());
    frame.extend_from_slice(&compressed);
    for count in [8, 16, 32] {
        let segment = frame.repeat(count);
        let (scan, allocated, calls, micros) =
            measured(|| scan::scan_segment(&segment, true, |_, _| {}));
        assert_eq!(scan.frames.len(), count);
        assert!(allocated < raw.len() * 2);
        eprintln!("scan: frames={count} compressed_bytes={} allocated_bytes={allocated} allocation_calls={calls} elapsed_us={micros}", segment.len());
    }
}

#[test]
fn recent_audit_query_uses_bounded_scratch() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = WarmConfig::default();
    config.security.audit_enabled = true;
    let core = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config)).unwrap();
    let mut entry = [0u8; 105];
    entry[8] = 1;
    for count in [10_000, 100_000, 200_000] {
        std::fs::write(dir.path().join("audit.bin"), entry.repeat(count)).unwrap();
        let (result, allocated, calls, micros) = measured(|| core.audit_since(1));
        assert!(result.unwrap().is_empty());
        assert!(allocated < 128 * 1024);
        eprintln!("audit: entries={count} returned=0 allocated_bytes={allocated} allocation_calls={calls} elapsed_us={micros}");
    }
}

#[test]
fn audit_entries_match_across_read_buffer_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = WarmConfig::default();
    config.security.audit_enabled = true;
    let core = WarmCore::new(256, Some(dir.path().display().to_string()), Some(config)).unwrap();
    for count in [623u64, 624, 625] {
        let mut bytes = Vec::new();
        for timestamp in 0..count {
            let mut entry = [0u8; 105];
            entry[..8].copy_from_slice(&timestamp.to_le_bytes());
            entry[8] = 1;
            bytes.extend_from_slice(&entry);
        }
        bytes.extend_from_slice(&[0; 104]);
        std::fs::write(dir.path().join("audit.bin"), bytes).unwrap();
        let entries = core.audit_since(count - 2).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].timestamp_ms, count - 2);
        assert_eq!(entries[1].timestamp_ms, count - 1);
    }
}

#[test]
fn goal_relevance_matches_sorted_lookup_without_allocating() {
    use writerslogic_warm::core::agency::goals::{Goal, GoalStore};
    use writerslogic_warm::EntangledHVec;
    let store = GoalStore::new();
    let query = EntangledHVec::new_deterministic(16384, 7);
    for i in 0..1000 {
        store.add(Goal {
            name: i.to_string(),
            description: "target".repeat(10),
            vector: query.clone(),
            relevance: i as f64 / 1000.0,
            urgency: 1.0,
            cost: 0.0,
            active: true,
        });
    }
    let (old, old_bytes, _, old_us) = measured(|| store.prioritized()[0].vector.similarity(&query));
    let (new, bytes, _, micros) = measured(|| store.goal_relevance(&query));
    assert_eq!(old, new);
    assert_eq!(bytes, 0);
    eprintln!(
        "GOALS sorted_bytes={old_bytes} scan_bytes={bytes} sorted_us={old_us} scan_us={micros}"
    );
}

#[test]
fn metadata_listing_does_not_allocate_stored_text() {
    use writerslogic_warm::core::documents::{DocumentInput, EncodeMode, IngestOptions};

    let dir = tempfile::tempdir().unwrap();
    let mut core = WarmCore::new(256, Some(dir.path().display().to_string()), None).unwrap();
    for index in 0..18 {
        core.memorize_document_with(
            DocumentInput {
                id: format!("doc-{index:02}"),
                text: "x".repeat(64 * 1024),
                source_uri: Some(format!("local://source/{index}")),
                version: Some("v1".into()),
                metadata: Some(serde_json::json!({"index": index})),
                chunk_words: None,
                overlap_words: None,
                store_text: Some(true),
                embeddings: None,
                title_weight: None,
            },
            &IngestOptions {
                encode: EncodeMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap();
    }
    for phase in ["live", "compacted", "reopened"] {
        if phase == "compacted" {
            core.compact().unwrap();
        } else if phase == "reopened" {
            drop(core);
            core = WarmCore::new(256, Some(dir.path().display().to_string()), None).unwrap();
        }
        let (count, allocated, _, _) = measured(|| core.document_count());
        assert_eq!(count, 18);
        assert!(allocated < 4096);
        for limit in [15, 16, 17] {
            let (records, allocated, calls, micros) = measured(|| {
                core.list_document_info(Some("doc-01"), limit)
                    .into_iter()
                    .map(|record| {
                        (
                            record.id,
                            record.version,
                            record.source_uri,
                            record.metadata,
                            record.chunk_count,
                        )
                    })
                    .collect::<Vec<_>>()
            });
            assert_eq!(records.len(), limit);
            for (index, record) in records.iter().enumerate() {
                assert_eq!(record.0, format!("doc-{:02}", index + 1));
                assert_eq!(record.1, "v1");
                assert_eq!(
                    record.2.as_deref(),
                    Some(format!("local://source/{}", index + 1).as_str())
                );
                assert_eq!(record.3, serde_json::json!({"index": index + 1}));
                assert_eq!(record.4, 1);
            }
            eprintln!("metadata listing: phase={phase} rows={limit} allocated_bytes={allocated} allocation_calls={calls} elapsed_us={micros}");
            assert!(
                allocated < 64 * 1024,
                "metadata-only listing allocated {allocated} bytes for {limit} rows"
            );
        }
        let (record, allocated, _, _) = measured(|| core.get_document_info("doc-01"));
        assert_eq!(record.unwrap().metadata, serde_json::json!({"index": 1}));
        assert!(
            allocated < 4096,
            "document summary allocated {allocated} bytes"
        );
        assert!(core.get_document_info("absent").is_none());
        assert!(core.list_document_info(None, 0).is_empty());
        assert!(core.list_document_info(Some("z"), 1).is_empty());
    }
    assert!(core.delete_document("doc-08").unwrap());
    assert!(core.get_document_info("doc-08").is_none());
    assert_eq!(
        core.list_document_info(Some("doc-07"), 3)
            .into_iter()
            .map(|record| record.id)
            .collect::<Vec<_>>(),
        ["doc-07", "doc-09", "doc-10"]
    );
    assert_eq!(core.document_count(), 17);
    assert_eq!(
        core.get_document("doc-01").unwrap().text.unwrap().len(),
        64 * 1024
    );
}
