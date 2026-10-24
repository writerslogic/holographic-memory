// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

#[path = "../../results/holography_resonator_v1/baseline.rs"]
mod baseline;
mod cases;
#[path = "../../../src/core/phase_hvec.rs"]
mod phase_hvec;
#[cfg(not(resonator_scratch))]
#[path = "../../../src/core/phase_resonator.rs"]
mod phase_resonator;
#[cfg(resonator_scratch)]
#[path = "../../results/holography_resonator_v1/packed_scratch_experiment.rs"]
mod phase_resonator;

use cases::{assert_equal, baseline_config, config, dataset, Scenario};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TRACK: AtomicBool = AtomicBool::new(true);

struct CountingAllocator;

fn added(bytes: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    ALLOCATED.fetch_add(bytes, Ordering::Relaxed);
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: Every allocation is forwarded unchanged to System, including layout
// and alignment. The atomics only observe byte counts in this serial harness.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid allocator layout.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            added(layout.size());
        }
        result
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid allocator layout.
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            added(layout.size());
        }
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK.load(Ordering::Relaxed) {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: This is the pointer and layout returned by the delegated allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The caller supplies a live allocation and a valid new size.
        let result = unsafe { System.realloc(ptr, layout, new_size) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            added(new_size);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct AllocationStats {
    calls: usize,
    allocated: usize,
    retained: usize,
    peak: usize,
}

fn measured<T>(run: impl FnOnce() -> T) -> (T, Duration, AllocationStats) {
    let before_calls = CALLS.load(Ordering::Relaxed);
    let before_bytes = ALLOCATED.load(Ordering::Relaxed);
    let before_live = LIVE.load(Ordering::Relaxed);
    PEAK.store(before_live, Ordering::Relaxed);
    let started = Instant::now();
    let result = run();
    let duration = started.elapsed();
    let stats = AllocationStats {
        calls: CALLS.load(Ordering::Relaxed) - before_calls,
        allocated: ALLOCATED.load(Ordering::Relaxed) - before_bytes,
        retained: LIVE.load(Ordering::Relaxed) - before_live,
        peak: PEAK.load(Ordering::Relaxed) - before_live,
    };
    (result, duration, stats)
}

fn report_build(
    name: &str,
    backend: &str,
    elapsed: Duration,
    stats: AllocationStats,
    stack: usize,
) {
    println!(
        "{{\"kind\":\"build\",\"case\":{name:?},\"backend\":{backend:?},\"elapsed_ns\":{},\"alloc_calls\":{},\"allocated_bytes\":{},\"retained_heap_bytes\":{},\"peak_heap_bytes\":{},\"index_stack_bytes\":{stack}}}",
        elapsed.as_nanos(), stats.calls, stats.allocated, stats.retained, stats.peak
    );
}

fn time_queries<T>(
    queries: &[phase_hvec::PhaseHVec],
    mut run: impl FnMut(&phase_hvec::PhaseHVec) -> T,
) -> u128 {
    // Each query's allocations are dropped within this scope; no index state
    // changes. Suspending counters avoids biasing allocation-heavy timings.
    TRACK.store(false, Ordering::Relaxed);
    let started = Instant::now();
    for query in queries {
        black_box(run(black_box(query)));
    }
    let elapsed = started.elapsed().as_nanos();
    TRACK.store(true, Ordering::Relaxed);
    elapsed
}

fn main() {
    let samples: usize = std::env::args()
        .nth(1)
        .map(|v| v.parse().expect("integer sample count"))
        .unwrap_or(7);
    assert!((1..=100).contains(&samples));
    let scenarios = [
        ("small_4bit", 16, 1024, 12, 40, 4),
        ("small_8bit", 256, 1024, 12, 40, 4),
        ("knee_4bit", 16, 1024, 48, 40, 3),
        ("knee_8bit", 256, 1024, 48, 40, 3),
        ("large_4bit", 16, 8192, 256, 6, 2),
        ("large_8bit", 256, 8192, 256, 6, 2),
        ("high_resolution", 65536, 1024, 12, 20, 2),
    ];
    for (name, n, dim, entries, max_iter, trials) in scenarios {
        let scenario = Scenario {
            dim,
            n,
            factors: 3,
            entries,
            max_iter,
            seed: 0x5EED,
        };
        let mut data = dataset(scenario, trials, false);
        let corrupted = dataset(scenario, 1, true);
        data.composites.extend(corrupted.composites);
        data.targets.extend(corrupted.targets);
        println!("{{\"kind\":\"scenario\",\"case\":{name:?},\"dim\":{dim},\"n\":{n},\"entries\":{entries},\"factors\":3,\"max_iter\":{max_iter},\"seed\":24301,\"clean_queries\":{trials},\"corrupted_queries\":1,\"samples\":{samples}}}");

        let (old, elapsed, stats) = measured(|| {
            baseline::PhaseResonator::with_config(&data.books, baseline_config(max_iter))
        });
        report_build(
            name,
            "baseline",
            elapsed,
            stats,
            std::mem::size_of_val(&old),
        );
        let (expanded, elapsed, stats) = measured(|| {
            phase_resonator::PhaseResonator::with_config(&data.books, config(max_iter))
        });
        report_build(
            name,
            "expanded",
            elapsed,
            stats,
            std::mem::size_of_val(&expanded),
        );
        let (packed, elapsed, stats) = measured(|| {
            phase_resonator::PhaseResonator::with_packed_codebooks(&data.books, config(max_iter))
        });
        report_build(
            name,
            "packed",
            elapsed,
            stats,
            std::mem::size_of_val(&packed),
        );

        let mut recovered = 0;
        let mut iterations = Vec::new();
        for (i, query) in data.composites.iter().enumerate() {
            let expected = old.factorize(query);
            assert_equal(&expected, &expanded.factorize(query));
            assert_equal(&expected, &packed.factorize(query));
            recovered += usize::from(
                i < trials
                    && expected
                        .iter()
                        .zip(&data.targets[i])
                        .all(|(result, &target)| result.codebook_entry == target),
            );
            iterations.push(expected[0].iterations);
        }
        println!("{{\"kind\":\"parity\",\"case\":{name:?},\"bitwise_equal\":true,\"clean_recovered\":{recovered},\"clean_queries\":{trials},\"iterations\":{iterations:?}}}");

        let mut times: [Vec<u128>; 3] = std::array::from_fn(|_| Vec::with_capacity(samples));
        for sample in 0..samples {
            for offset in 0..3 {
                let backend = (sample + offset) % 3;
                let elapsed = match backend {
                    0 => time_queries(&data.composites, |q| old.factorize(q)),
                    1 => time_queries(&data.composites, |q| expanded.factorize(q)),
                    _ => time_queries(&data.composites, |q| packed.factorize(q)),
                };
                times[backend].push(elapsed);
            }
        }
        for (backend, mut nanos) in times.into_iter().enumerate() {
            let (_, _, stats) = measured(|| match backend {
                0 => {
                    for q in &data.composites {
                        black_box(old.factorize(q));
                    }
                }
                1 => {
                    for q in &data.composites {
                        black_box(expanded.factorize(q));
                    }
                }
                _ => {
                    for q in &data.composites {
                        black_box(packed.factorize(q));
                    }
                }
            });
            let label = ["baseline", "expanded", "packed"][backend];
            let samples_ns = format!("{nanos:?}");
            nanos.sort_unstable();
            let median = nanos[nanos.len() / 2];
            let queries = data.composites.len();
            println!("{{\"kind\":\"query\",\"case\":{name:?},\"backend\":{label:?},\"queries\":{queries},\"samples_ns\":{samples_ns},\"median_ns\":{median},\"alloc_calls\":{},\"allocated_bytes\":{},\"peak_heap_bytes\":{}}}", stats.calls, stats.allocated, stats.peak);
        }
    }
}
