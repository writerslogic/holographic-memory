// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Ingest, search, snapshot and reopen costs of the mobile surface, through the same API the
//! Swift and Kotlin bindings call. Prints one JSON object. The corpus is synthetic: words
//! drawn from a Zipf(1.0) distribution over a fixed vocabulary, so term statistics resemble
//! prose but no ranking quality is measured here.
//!
//! `cargo run --release --example mobile_bench -- --docs 20000 --words 150 --dims 4096`

use std::time::Instant;

use warm_mobile::{DocumentInput, WarmStore, SearchOptions, StoreOptions};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct Zipf {
    cumulative: Vec<f64>,
}

impl Zipf {
    fn new(vocabulary: usize) -> Self {
        let mut total = 0.0;
        let cumulative = (1..=vocabulary)
            .map(|rank| {
                total += 1.0 / rank as f64;
                total
            })
            .collect::<Vec<_>>();
        Self {
            cumulative: cumulative.iter().map(|c| c / total).collect(),
        }
    }

    fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.unit();
        self.cumulative.partition_point(|&c| c < u)
    }
}

fn word(rank: usize) -> String {
    // Pronounceable, distinct, stemmer-stable tokens: consonant-vowel syllables.
    const C: &[u8] = b"bdfghklmnprstvz";
    const V: &[u8] = b"aeiou";
    let mut n = rank + 1;
    let mut out = String::new();
    while n > 0 {
        out.push(C[n % C.len()] as char);
        n /= C.len();
        out.push(V[n % V.len()] as char);
        n /= V.len();
    }
    out
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn rss_bytes() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0)
        * 1024
}

fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage writes a fully initialized rusage into the zeroed struct we own.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    assert_eq!(status, 0);
    let max = usage.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        max
    } else {
        max * 1024
    }
}

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map_or(default, |v| v.parse().expect("numeric argument"))
}

fn doc(id: usize, words: usize, zipf: &Zipf, rng: &mut Rng) -> DocumentInput {
    let mut text = String::new();
    for _ in 0..words {
        text.push_str(&word(zipf.sample(rng)));
        text.push(' ');
    }
    DocumentInput {
        id: format!("doc-{id}"),
        text,
        source_uri: None,
        version: None,
        metadata_json: None,
        chunk_words: None,
        overlap_words: None,
        store_text: None,
        embeddings: None,
        title_weight: None,
    }
}

fn main() {
    let docs = arg("--docs", 20_000);
    let words = arg("--words", 150);
    let dims = arg("--dims", 4096) as u32;
    let queries = arg("--queries", 500);
    let vocabulary = 30_000;
    let zipf = Zipf::new(vocabulary);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let dir = std::env::var("BENCH_DIR").unwrap_or_else(|_| "/tmp/warm-mobile-bench".into());
    let _ = std::fs::remove_dir_all(&dir);
    let path = format!("{dir}/store");
    let options = StoreOptions {
        dimensions: dims,
        ..Default::default()
    };

    let rss_start = rss_bytes();
    let t = Instant::now();
    let store = WarmStore::open_blocking(path.clone(), Some(options.clone())).unwrap();
    let open_empty_ms = t.elapsed().as_secs_f64() * 1e3;

    // Single-document ingest latency on the first 500, then batches of 100.
    let single = docs.min(500);
    let mut single_ms = Vec::new();
    for i in 0..single {
        let input = doc(i, words, &zipf, &mut rng);
        let t = Instant::now();
        store.ingest_blocking(input).unwrap();
        single_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let t = Instant::now();
    let mut batched = 0;
    let mut next = single;
    while next < docs {
        let end = (next + 100).min(docs);
        let batch = (next..end)
            .map(|i| doc(i, words, &zipf, &mut rng))
            .collect::<Vec<_>>();
        batched += batch.len();
        store.ingest_batch_blocking(batch).unwrap();
        next = end;
    }
    let batch_seconds = t.elapsed().as_secs_f64();
    single_ms.sort_by(f64::total_cmp);

    let mut query_ms = Vec::new();
    let mut hit_counts = 0usize;
    for _ in 0..queries {
        let q = (0..3)
            .map(|_| word(zipf.sample(&mut rng) + 50))
            .collect::<Vec<_>>()
            .join(" ");
        let t = Instant::now();
        let hits = store
            .search_blocking(
                q,
                Some(SearchOptions {
                    k: Some(10),
                    ..Default::default()
                }),
            )
            .unwrap();
        query_ms.push(t.elapsed().as_secs_f64() * 1e3);
        hit_counts += hits.len();
    }
    query_ms.sort_by(f64::total_cmp);

    let stats = store.stats().unwrap();
    let rss_loaded = rss_bytes();

    let t = Instant::now();
    store.compact_blocking().unwrap();
    let compact_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let info = store.snapshot_blocking(format!("{dir}/snap")).unwrap();
    let snapshot_ms = t.elapsed().as_secs_f64() * 1e3;
    let after_compact = store.stats().unwrap();
    store.shutdown();
    drop(store);

    let t = Instant::now();
    let reopened = WarmStore::open_blocking(path, Some(options.clone())).unwrap();
    let reopen_ms = t.elapsed().as_secs_f64() * 1e3;
    let rss_reopened = rss_bytes();
    drop(reopened);
    let t = Instant::now();
    let restored = WarmStore::restore_blocking(
        format!("{dir}/snap"),
        format!("{dir}/restored"),
        Some(options),
    )
    .unwrap();
    let restore_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(restored.stats().unwrap().document_count, docs as u64);

    let result = serde_json::json!({
        "docs": docs,
        "wordsPerDoc": words,
        "dimensions": dims,
        "vocabulary": vocabulary,
        "chunks": stats.chunk_count,
        "openEmptyMs": open_empty_ms,
        "ingestSingle": {
            "n": single,
            "p50Ms": percentile(&single_ms, 0.5),
            "p95Ms": percentile(&single_ms, 0.95),
            "p99Ms": percentile(&single_ms, 0.99),
        },
        "ingestBatch100": {
            "docs": batched,
            "docsPerSecond": batched as f64 / batch_seconds,
        },
        "search": {
            "queries": queries,
            "termsPerQuery": 3,
            "k": 10,
            "meanHits": hit_counts as f64 / queries as f64,
            "p50Ms": percentile(&query_ms, 0.5),
            "p95Ms": percentile(&query_ms, 0.95),
            "p99Ms": percentile(&query_ms, 0.99),
        },
        "diskBytesBeforeCompact": stats.disk_bytes,
        "diskBytesAfterCompact": after_compact.disk_bytes,
        "diskAllocatedBytesAfterCompact": after_compact.disk_allocated_bytes,
        "compactMs": compact_ms,
        "snapshotMs": snapshot_ms,
        "snapshotBytes": info.total_bytes,
        "reopenMs": reopen_ms,
        "restoreMs": restore_ms,
        "rssBytes": {
            "beforeOpen": rss_start,
            "loaded": rss_loaded,
            "afterReopen": rss_reopened,
            "peak": peak_rss_bytes(),
        },
    });
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
