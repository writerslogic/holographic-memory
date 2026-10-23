// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Heap allocations, bytes and mean time per vector query (`HmsCore::query` and
//! `HmsCore::query_hopfield`) on the `query_scaling` corpus of `hms_benchmark`: dimension
//! 10,000 and n = 100, 500, 1000 encoded sentences. The query vector is encoded once outside
//! the counted region, so the counts are the query path's own. A counting global allocator
//! wraps the system one in this bench binary only. Run with `RAYON_NUM_THREADS=1`; the times
//! are development figures, the allocation counts are deterministic.

use holographic_memory::HmsCore;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: every call forwards to the system allocator with the caller's layout unchanged; the
// counters are side effects only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size as u64, Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const ITERATIONS: u64 = 2_000;

fn measure<R>(mut op: impl FnMut() -> R) -> (f64, f64, f64) {
    black_box(op());
    let (allocs, bytes) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(op());
    }
    let micros = start.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64;
    let per = |now: u64, then: u64| (now - then) as f64 / ITERATIONS as f64;
    (
        per(ALLOCS.load(Relaxed), allocs),
        per(BYTES.load(Relaxed), bytes),
        micros,
    )
}

fn main() {
    println!("| operation | n | route | allocs/op | bytes/op | us/op |");
    println!("|---|---|---|---|---|---|");
    for n in [100, 500, 1000] {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().to_string_lossy().into_owned();
        let hms = HmsCore::new(10_000, Some(path), None).expect("store");
        for i in 0..n {
            let text = format!("Sentence about topic {} with variation {}", i % 20, i);
            hms.memorize(format!("id_{i}"), hms.encode_text(&text))
                .expect("memorize");
        }
        let query = hms.encode_text("topic about variation");
        let route = hms.explain_query(&query, 5).route;
        let (allocs, bytes, micros) = measure(|| hms.query(black_box(&query), 5));
        println!("| query k=5 | {n} | {route} | {allocs:.1} | {bytes:.0} | {micros:.2} |");
        let (allocs, bytes, micros) = measure(|| hms.query_hopfield(black_box(&query), 5));
        println!("| query_hopfield 5 | {n} | hopfield | {allocs:.1} | {bytes:.0} | {micros:.2} |");
    }
}
