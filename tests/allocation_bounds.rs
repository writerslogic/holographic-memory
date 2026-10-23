// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

use holographic_memory::core::format::scan;
use holographic_memory::{core::HmsConfig, HmsCore};

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
    let mut config = HmsConfig::default();
    config.security.audit_enabled = true;
    let core = HmsCore::new(256, Some(dir.path().display().to_string()), Some(config)).unwrap();
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
    let mut config = HmsConfig::default();
    config.security.audit_enabled = true;
    let core = HmsCore::new(256, Some(dir.path().display().to_string()), Some(config)).unwrap();
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
    use holographic_memory::core::agency::goals::{Goal, GoalStore};
    use holographic_memory::EntangledHVec;
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
