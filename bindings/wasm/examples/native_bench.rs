// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Native baseline for the wasm benchmark: the same corpus, queries and k as
//! `npm/bench/bench-core.mjs`, run against the in-memory store (the code the wasm module runs)
//! or the full engine (`WarmCore`, journaled in a temporary directory).
//!
//!   cargo run --release --example native_bench -- --engine memory|warm --documents N --queries Q

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use writerslogic_warm::core::documents::{DocumentInput, MemoryDocumentStore, SearchOptions};
    use writerslogic_warm::WarmCore;
    use std::time::Instant;

    const VOCABULARY: f64 = 20_000.0;
    const WORDS_PER_DOC: usize = 100;
    const QUERY_TERMS: usize = 3;
    const SEED: u32 = 0x5eed1234;
    const DIMENSIONS: usize = 16384;
    const K: u32 = 10;
    const BATCH: usize = 500;
    const WARMUP: usize = 20;

    struct Mulberry(u32);

    impl Mulberry {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x6d2b79f5);
            let mut t = self.0;
            t = (t ^ (t >> 15)).wrapping_mul(t | 1);
            t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
            t ^ (t >> 14)
        }

        fn word(&mut self) -> u32 {
            let u = f64::from(self.next()) / 4294967296.0;
            (VOCABULARY * u * u * u).floor() as u32
        }
    }

    fn document_text(index: u32) -> String {
        let mut rng = Mulberry(SEED ^ index.wrapping_add(1).wrapping_mul(0x9e3779b1));
        (0..WORDS_PER_DOC)
            .map(|_| format!("t{}", rng.word()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn query_text(count: u32, query_index: u32) -> String {
        let mut rng = Mulberry(SEED ^ query_index.wrapping_add(7).wrapping_mul(0x85ebca6b));
        let doc = rng.next() % count;
        let text = document_text(doc);
        let words: Vec<&str> = text.split(' ').collect();
        (0..QUERY_TERMS)
            .map(|_| words[(rng.next() as usize) % WORDS_PER_DOC])
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn fnv(mut hash: u32, text: &str) -> u32 {
        for byte in text.bytes() {
            hash = (hash ^ u32::from(byte)).wrapping_mul(0x01000193);
        }
        hash
    }

    fn input(index: u32) -> DocumentInput {
        DocumentInput {
            id: format!("doc{index}"),
            text: document_text(index),
            source_uri: None,
            version: None,
            metadata: None,
            chunk_words: None,
            overlap_words: None,
            store_text: None,
            embeddings: None,
            title_weight: None,
        }
    }

    fn peak_rss_bytes() -> u64 {
        // SAFETY: getrusage fills a zeroed rusage struct for the calling process.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        assert_eq!(rc, 0, "getrusage failed");
        let raw = usage.ru_maxrss as u64;
        if cfg!(target_os = "macos") {
            raw
        } else {
            raw * 1024
        }
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        sorted[((p / 100.0 * sorted.len() as f64) as usize).min(sorted.len() - 1)]
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let option = |name: &str, default: &str| {
            args.iter()
                .position(|a| a == name)
                .and_then(|i| args.get(i + 1).cloned())
                .unwrap_or_else(|| default.to_string())
        };
        let engine = option("--engine", "memory");
        let documents: u32 = option("--documents", "10000").parse().expect("--documents");
        let queries: usize = option("--queries", "200").parse().expect("--queries");

        let dir = tempfile::tempdir().expect("temporary directory");
        let mut memory = None;
        let mut warm = None;
        match engine.as_str() {
            "memory" => memory = Some(MemoryDocumentStore::new(DIMENSIONS, None).expect("store")),
            "warm" => {
                warm = Some(
                    WarmCore::new(
                        DIMENSIONS as u32,
                        Some(dir.path().display().to_string()),
                        None,
                    )
                    .expect("engine"),
                )
            }
            other => panic!("unknown engine {other}"),
        }

        let t = Instant::now();
        let mut start = 0;
        while start < documents {
            let end = (start + BATCH as u32).min(documents);
            let batch: Vec<DocumentInput> = (start..end).map(input).collect();
            if let Some(store) = memory.as_mut() {
                store.memorize_many(batch).expect("ingest");
            } else {
                warm.as_ref()
                    .unwrap()
                    .memorize_documents(batch)
                    .expect("ingest");
            }
            start = end;
        }
        let ingest_ms = t.elapsed().as_secs_f64() * 1000.0;
        let rss_after_ingest = peak_rss_bytes();

        let options = SearchOptions {
            k: Some(K),
            ..Default::default()
        };
        let search = |text: &str| {
            if let Some(store) = memory.as_ref() {
                store.search(text, &options).expect("search")
            } else {
                warm.as_ref()
                    .unwrap()
                    .search_documents(text, &options)
                    .expect("search")
            }
        };
        let texts: Vec<String> = (0..queries + WARMUP)
            .map(|i| query_text(documents, i as u32))
            .collect();
        for text in &texts[..WARMUP] {
            search(text);
        }
        let mut latencies = Vec::with_capacity(queries);
        let mut digest = 0x811c9dc5u32;
        let mut score_sum = 0.0f64;
        let t = Instant::now();
        for text in &texts[WARMUP..] {
            let q = Instant::now();
            let hits = search(text);
            latencies.push(q.elapsed().as_secs_f64() * 1000.0);
            for hit in &hits {
                digest = fnv(digest, &hit.document_id);
                score_sum += hit.score;
            }
        }
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let (snapshot_bytes, export_ms, import_ms) = if let Some(store) = memory.as_ref() {
            let t = Instant::now();
            let bytes = store.export().expect("export");
            let export_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let restored = MemoryDocumentStore::import(&bytes).expect("import");
            let import_ms = t.elapsed().as_secs_f64() * 1000.0;
            drop(restored);
            (Some(bytes.len()), Some(export_ms), Some(import_ms))
        } else {
            (None, None, None)
        };

        let report = serde_json::json!({
            "engine": engine,
            "documents": documents,
            "queries": queries,
            "wordsPerDocument": WORDS_PER_DOC,
            "k": K,
            "dimensions": DIMENSIONS,
            "ingestMs": ingest_ms,
            "ingestDocsPerSecond": f64::from(documents) / (ingest_ms / 1000.0),
            "peakRssAfterIngestBytes": rss_after_ingest,
            "peakRssBytes": peak_rss_bytes(),
            "search": {
                "meanMs": total_ms / queries as f64,
                "p50Ms": percentile(&latencies, 50.0),
                "p95Ms": percentile(&latencies, 95.0),
                "p99Ms": percentile(&latencies, 99.0),
                "maxMs": latencies[latencies.len() - 1],
                "qps": queries as f64 / (total_ms / 1000.0),
            },
            "topKDigest": digest,
            "scoreSum": score_sum,
            "snapshotBytes": snapshot_bytes,
            "exportMs": export_ms,
            "importMs": import_ms,
        });
        println!("{report}");
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::run();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
