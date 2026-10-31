// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-call cost of the memory API's write side: `remember` one fact per call, `remember_many`
//! in batches, and `forget` one id per call, on a durable store. Prints JSON with the p50 and
//! p95 latency, heap allocations and allocated bytes per call (a counting global allocator
//! local to this example).
//!
//! `cargo run --release --example memory_write_bench -- [remembers] [forgets] [batch] [vocabulary]`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use writerslogic_warm::core::memory_api::{ForgetOptions, ForgetScope, RememberInput};
use writerslogic_warm::{Memory, WarmCore};

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to the system allocator unchanged; the counters are relaxed
// atomics with no effect on the allocation.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn counters() -> (u64, u64) {
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

struct Sample {
    micros: Vec<f64>,
    allocs: u64,
    bytes: u64,
}

impl Sample {
    fn new() -> Self {
        Self {
            micros: Vec::new(),
            allocs: 0,
            bytes: 0,
        }
    }

    fn time<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let (a0, b0) = counters();
        let t = Instant::now();
        let out = f();
        self.micros.push(t.elapsed().as_secs_f64() * 1e6);
        let (a1, b1) = counters();
        self.allocs += a1 - a0;
        self.bytes += b1 - b0;
        out
    }

    fn json(mut self, items_per_call: usize) -> serde_json::Value {
        self.micros.sort_by(f64::total_cmp);
        let n = self.micros.len();
        let at = |p: f64| self.micros[((n - 1) as f64 * p).round() as usize];
        serde_json::json!({
            "calls": n,
            "items_per_call": items_per_call,
            "p50_us": at(0.5),
            "p95_us": at(0.95),
            "allocs_per_call": self.allocs as f64 / n as f64,
            "bytes_per_call": self.bytes as f64 / n as f64,
        })
    }
}

/// A fact of 24 words drawn from `vocabulary` pseudo-words with a skewed (squared uniform)
/// rank distribution, so a few words are in most facts and most words in few, as in text.
fn fact(i: usize, vocabulary: usize) -> RememberInput {
    let mut state = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let words: Vec<String> = (0..24)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            format!("w{}", (u * u * vocabulary as f64) as usize)
        })
        .collect();
    RememberInput {
        id: Some(format!("fact-{i}")),
        text: words.join(" "),
        ..RememberInput::default()
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (remembers, forgets, batch, vocabulary) = (
        args.first().copied().unwrap_or(10_000),
        args.get(1).copied().unwrap_or(1_000),
        args.get(2).copied().unwrap_or(100),
        args.get(3).copied().unwrap_or(4096).max(1),
    );
    let dir = tempfile::tempdir()?;
    let core = Arc::new(WarmCore::new(
        4096,
        Some(dir.path().display().to_string()),
        None,
    )?);
    let memory = Memory::new(core);

    let mut single = Sample::new();
    for i in 0..remembers {
        single.time(|| memory.remember(fact(i, vocabulary)))?;
    }
    let mut batched = Sample::new();
    for start in (remembers..2 * remembers).step_by(batch) {
        let facts: Vec<RememberInput> = (start..(start + batch).min(2 * remembers))
            .map(|i| fact(i, vocabulary))
            .collect();
        batched.time(|| memory.remember_many(facts))?;
    }
    let mut forget = Sample::new();
    for i in 0..forgets.min(remembers) {
        let scope = ForgetScope::Ids {
            ids: vec![format!("fact-{i}")],
        };
        let report = forget.time(|| memory.forget(&scope, &ForgetOptions::default()))?;
        anyhow::ensure!(report.forgotten.len() == 1, "fact-{i} was not forgotten");
    }
    println!(
        "{}",
        serde_json::json!({
            "facts_stored": 2 * remembers,
            "vocabulary": vocabulary,
            "remember": single.json(1),
            "remember_many": batched.json(batch),
            "forget": forget.json(1),
        })
    );
    Ok(())
}
