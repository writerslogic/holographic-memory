// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

struct Allocator;

static CALLS: AtomicU64 = AtomicU64::new(0);
static REQUESTED: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

fn allocated(size: usize) {
    CALLS.fetch_add(1, Relaxed);
    REQUESTED.fetch_add(size as u64, Relaxed);
    let live = LIVE.fetch_add(size as u64, Relaxed) + size as u64;
    PEAK.fetch_max(live, Relaxed);
}

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Relaxed);
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = System.realloc(ptr, layout, size);
        if !result.is_null() {
            LIVE.fetch_sub(layout.size() as u64, Relaxed);
            allocated(size);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

#[allow(dead_code)]
mod documents {
    struct StoredChunk {
        terms: std::collections::BTreeMap<String, u32>,
        stemmed: bool,
    }

    mod postings {
        include!(env!("HMS_POSTINGS_SOURCE"));

        pub(super) fn profile() {
            use crate::{CALLS, LIVE, PEAK, REQUESTED};
            use std::sync::atomic::Ordering::Relaxed;
            for length in [0, 1, 127, 128, 129, 256] {
                for batch in [1, 128] {
                    for skyline in [false, true] {
                        let run: Vec<Posting> = (0..length)
                            .map(|slot| (slot, if skyline { slot as u32 + 1 } else { 1 }))
                            .collect();
                        let calls = CALLS.load(Relaxed);
                        let requested = REQUESTED.load(Relaxed);
                        let live = LIVE.load(Relaxed);
                        PEAK.store(live, Relaxed);
                        let lists: Vec<_> = (0..1000)
                            .map(|_| {
                                let mut list = PostingList::default();
                                for part in run.chunks(batch) {
                                    list.append_run_with_lengths(part, &mut Vec::new(), |slot| {
                                        if skyline {
                                            slot as u32 + 1
                                        } else {
                                            1
                                        }
                                    });
                                }
                                list
                            })
                            .collect();
                        let calls = CALLS.load(Relaxed) - calls;
                        let requested = REQUESTED.load(Relaxed) - requested;
                        let retained = LIVE.load(Relaxed) - live;
                        let peak = PEAK.load(Relaxed) - live;
                        let block_capacity: usize = lists.iter().map(|l| l.blocks.capacity()).sum();
                        let block_length: usize = lists.iter().map(|l| l.blocks.len()).sum();
                        for list in &lists {
                            assert_eq!(list.decode(), run);
                        }
                        println!(
                            "{{\"lists\":1000,\"postings_per_list\":{length},\"batch\":{batch},\"skyline\":{skyline},\"block_size\":{},\"list_size\":{},\"block_length\":{block_length},\"block_capacity\":{block_capacity},\"allocation_calls\":{calls},\"cumulative_requested_bytes\":{requested},\"retained_requested_bytes\":{retained},\"peak_requested_bytes\":{peak}}}",
                            std::mem::size_of::<Block>(), std::mem::size_of::<PostingList>()
                        );
                    }
                }
            }
        }
    }

    pub fn profile() {
        postings::profile();
    }
}

fn main() {
    documents::profile();
}
