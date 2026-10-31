// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Heap allocations, bytes and peak live bytes of each phase of the document store's write
//! side: batch ingest, closing (checkpoint write), reopening (checkpoint map or journal replay
//! plus the first query), compaction, replace and delete. A counting global allocator wraps
//! the system one in this bench binary only.
//!
//! `cargo bench --bench ingest_alloc` runs a synthetic corpus; `WARM_ALLOC_INPUT=<input.json>`
//! (the competitive text harness's `{documents, queries}` file) runs a real one, and
//! `WARM_ALLOC_DOCS=<n>` caps or sizes it. `WARM_ALLOC_POSITIONS=1` ingests whole-document
//! positional chunks (the phrase-query index). Run with `RAYON_NUM_THREADS=1` for counts that
//! do not depend on the thread count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
use writerslogic_warm::core::documents::{
    BatchTooLarge, EncodeMode, IngestOptions, MAX_BATCH_ITEMS,
};
use writerslogic_warm::core::WarmConfig;
use writerslogic_warm::{DocumentInput, SearchOptions, WarmCore};

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(size: usize) {
    ALLOCS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size as u64, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every call forwards to the system allocator with the caller's layout unchanged; the
// counters are side effects only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A realloc counts as one allocation of the new size, freeing the old.
        LIVE.fetch_sub(layout.size(), Relaxed);
        grow(new_size);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

struct Phase {
    allocs: u64,
    bytes: u64,
    live: usize,
    start: Instant,
}

fn begin() -> Phase {
    let live = LIVE.load(Relaxed);
    PEAK.store(live, Relaxed);
    Phase {
        allocs: ALLOCS.load(Relaxed),
        bytes: BYTES.load(Relaxed),
        live,
        start: Instant::now(),
    }
}

fn end(name: &str, phase: Phase, per: usize) {
    let ms = phase.start.elapsed().as_secs_f64() * 1e3;
    let allocs = ALLOCS.load(Relaxed) - phase.allocs;
    let bytes = BYTES.load(Relaxed) - phase.bytes;
    let peak = PEAK.load(Relaxed).saturating_sub(phase.live);
    let retained = LIVE.load(Relaxed) as i64 - phase.live as i64;
    let per = per.max(1) as f64;
    println!(
        "| {name} | {per} | {allocs} | {:.1} | {:.0} | {:.2} | {:.2} | {:.1} |",
        allocs as f64 / per,
        bytes as f64 / per,
        peak as f64 / (1 << 20) as f64,
        retained as f64 / (1 << 20) as f64,
        ms
    );
}

/// A deterministic corpus with a Zipf-like vocabulary: `count` documents of 60-260 words.
fn synthetic(count: usize) -> Vec<DocumentInput> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let vocabulary: Vec<String> = (0..30_000)
        .map(|i| {
            let mut word = String::new();
            let mut n = i as u64 * 2_654_435_761 + 7;
            for _ in 0..(3 + i % 9) {
                word.push((b'a' + (n % 26) as u8) as char);
                n /= 26;
                n = n.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            }
            word
        })
        .collect();
    (0..count)
        .map(|i| {
            let words = 60 + next() as usize % 200;
            let mut text = String::new();
            for w in 0..words {
                // Rank ~ 1/u: frequent words dominate, as in natural text.
                let u = (next() % 1_000_000) as f64 / 1e6;
                let rank = ((vocabulary.len() as f64).powf(u) as usize).min(vocabulary.len()) - 1;
                if w > 0 {
                    text.push(if w % 17 == 0 { '.' } else { ' ' });
                    if w % 17 == 0 {
                        text.push(' ');
                    }
                }
                text.push_str(&vocabulary[rank]);
            }
            input(format!("doc-{i}"), text)
        })
        .collect()
}

fn input(id: String, text: String) -> DocumentInput {
    serde_json::from_value(serde_json::json!({"id": id, "text": text, "storeText": false}))
        .expect("document input")
}

fn load(path: &str) -> Vec<DocumentInput> {
    #[derive(serde::Deserialize)]
    struct File {
        documents: Vec<DocumentInput>,
    }
    let file: File =
        serde_json::from_slice(&std::fs::read(path).expect("read input")).expect("parse input");
    file.documents
}

fn main() {
    let positions = std::env::var("WARM_ALLOC_POSITIONS").is_ok_and(|v| v == "1");
    let cap: Option<usize> = std::env::var("WARM_ALLOC_DOCS")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut documents = match std::env::var("WARM_ALLOC_INPUT") {
        Ok(path) => load(&path),
        Err(_) => synthetic(cap.unwrap_or(20_000)),
    };
    if let Some(cap) = cap {
        documents.truncate(cap);
    }
    if positions {
        // Whole-article chunks keep every word position, as the phrase-query index does.
        for document in &mut documents {
            document.store_text = Some(false);
            document.chunk_words = None;
            document.overlap_words = None;
        }
    }
    let n = documents.len();
    let batch_size: usize = std::env::var("WARM_ALLOC_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
        .min(MAX_BATCH_ITEMS);
    let options = IngestOptions {
        encode: EncodeMode::Lexical,
        positions,
        whole_document: positions,
        ..Default::default()
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let store = dir.path().join("store").display().to_string();
    let query = SearchOptions {
        semantic_weight: Some(0.0),
        ..Default::default()
    };
    let replaced: Vec<DocumentInput> = documents
        .iter()
        .take(100)
        .map(|document| {
            let mut changed = document.clone();
            changed.text.push_str(" benchmarkrevision");
            changed.version = Some("2".into());
            changed
        })
        .collect();
    let deleted: Vec<String> = documents
        .iter()
        .skip(100)
        .take(100)
        .map(|d| d.id.clone())
        .collect();
    // Batch boundaries as the harness finds them (start at `batch_size`, halve on a batch-limit
    // error), planned in a throwaway store outside the measured phases.
    let mut ends = Vec::new();
    {
        let plan = tempfile::tempdir().expect("temp dir");
        let core =
            WarmCore::new(1024, Some(plan.path().display().to_string()), None).expect("plan store");
        let (mut at, mut size) = (0, batch_size);
        while at < n {
            let end = (at + size).min(n);
            match core.memorize_documents_with(documents[at..end].to_vec(), &options) {
                Ok(_) => {
                    ends.push(end);
                    at = end;
                }
                Err(error) if size > 1 && error.is::<BatchTooLarge>() => size /= 2,
                Err(error) => panic!("ingest: {error}"),
            }
        }
    }
    let mut batches = Vec::with_capacity(ends.len());
    let mut start = 0;
    for &end in &ends {
        batches.push(documents[start..end].to_vec());
        start = end;
    }
    let replay_batches = batches.clone();
    drop(documents);
    println!(
        "documents {n}, batches {}, positions {positions}, threads {}",
        ends.len(),
        rayon::current_num_threads()
    );
    println!("| phase | per | allocs | allocs/per | bytes/per | peak MiB | retained MiB | ms |");
    println!("|---|---|---|---|---|---|---|---|");

    let phase = begin();
    let core = WarmCore::new(1024, Some(store.clone()), None).expect("create store");
    for batch in batches {
        core.memorize_documents_with(batch, &options)
            .expect("ingest");
    }
    end("ingest", phase, n);

    let phase = begin();
    drop(core);
    end("close (checkpoint)", phase, n);

    let phase = begin();
    let core = WarmCore::new(1024, Some(store.clone()), None).expect("open");
    end("open", phase, n);
    let phase = begin();
    core.search_documents("system study effect", &query)
        .expect("first query");
    end("first query", phase, 1);

    let phase = begin();
    for document in replaced {
        core.memorize_document_with(document, &options)
            .expect("replace");
    }
    end("replace (each)", phase, 100);

    let phase = begin();
    for id in &deleted {
        assert!(core.delete_document(id).expect("delete"));
    }
    end("delete (each)", phase, 100);

    let phase = begin();
    core.compact().expect("compact");
    end("compact", phase, n);
    drop(core);

    // The same corpus in a store that never checkpoints: closing leaves the journal frames
    // and opening replays them.
    let mut config = WarmConfig::default();
    config.document_checkpoint.enabled = false;
    let store = dir.path().join("replay").display().to_string();
    let core = WarmCore::new(1024, Some(store.clone()), Some(config.clone())).expect("create");
    for batch in replay_batches {
        core.memorize_documents_with(batch, &options)
            .expect("ingest");
    }
    drop(core);
    let phase = begin();
    let core = WarmCore::new(1024, Some(store), Some(config)).expect("open");
    end("open (journal replay)", phase, n);
    let phase = begin();
    core.search_documents("system study effect", &query)
        .expect("first query");
    end("first query (replayed)", phase, 1);
}
