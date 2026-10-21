// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Leak check with a counting allocator: the live heap after many open/ingest/search/close
//! cycles must not grow with the cycle count. Runs on every platform, unlike valgrind/leaks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use holographic_memory_c::*;

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY: delegates every call to `System`, only adding bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = System.realloc(ptr, layout, new_size);
        if !new.is_null() {
            LIVE.fetch_add(
                new_size as isize - layout.size() as isize,
                Ordering::Relaxed,
            );
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn s(value: &str) -> hms_str_t {
    hms_str_t::of(value)
}

fn document<'a>(id: &'a str, text: &'a str) -> hms_document_t {
    hms_document_t {
        id: s(id),
        text: s(text),
        source_uri: hms_str_t::ABSENT,
        version: hms_str_t::ABSENT,
        metadata_json: s("{\"k\":\"v\"}"),
        chunk_words: 0,
        overlap_words: 0,
        title_weight: 0,
        store_text: 0,
        embeddings: std::ptr::null(),
        embedding_rows: 0,
        embedding_dim: 0,
    }
}

fn cycle(path: &str) {
    unsafe {
        let mut store = std::ptr::null_mut();
        assert_eq!(
            hms1_store_open(s(path), 512, std::ptr::null(), &mut store),
            HMS_OK
        );
        let docs = [
            document("a", "alpha beta gamma delta epsilon"),
            document("b", "zeta eta theta iota kappa lambda"),
        ];
        assert_eq!(
            hms1_store_ingest(store, docs.as_ptr(), 2, std::ptr::null_mut()),
            HMS_OK
        );
        for _ in 0..4 {
            let mut results = std::ptr::null_mut();
            assert_eq!(
                hms1_store_search(store, s("beta theta"), std::ptr::null(), &mut results),
                HMS_OK
            );
            let mut hit = std::mem::zeroed::<hms_result_t>();
            assert_eq!(hms1_results_get(results, 0, &mut hit), HMS_OK);
            hms1_results_free(results);
            // Failing calls allocate error text; it must not accumulate either.
            assert_ne!(
                hms1_store_search(store, s("x"), std::ptr::null(), std::ptr::null_mut()),
                HMS_OK
            );
        }
        let mut deleted = 0;
        assert_eq!(hms1_store_delete(store, s("a"), &mut deleted), HMS_OK);
        assert_eq!(hms1_store_close(store), HMS_OK);
    }
}

#[test]
fn live_heap_is_flat_across_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap().to_owned();
    // Warm up thread pools, thread-locals and lazily initialised statics.
    for _ in 0..3 {
        cycle(&path);
    }
    let before = LIVE.load(Ordering::Relaxed);
    for _ in 0..25 {
        cycle(&path);
    }
    let growth = LIVE.load(Ordering::Relaxed) - before;
    // One leaked search result alone is far above this; 25 cycles of any leak would be
    // many times over it. The slack covers allocator and pool bookkeeping noise.
    assert!(
        growth < 64 * 1024,
        "live heap grew by {growth} bytes over 25 cycles"
    );
}
