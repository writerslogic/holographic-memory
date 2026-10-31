// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::hint::black_box;
use writerslogic_warm::core::documents::{
    EncodeMode, IngestOptions, LexicalExecution, LexicalOptions, LexicalQuery,
};
use writerslogic_warm::{DocumentInput, EntangledHVec, SearchOptions, WarmCore};

fn benchmark_entangled(c: &mut Criterion) {
    let dim = 16384;

    c.bench_function("EntangledHVec random D=16384", |b| {
        b.iter(|| EntangledHVec::new_deterministic(black_box(dim), black_box(42)))
    });

    let e1 = EntangledHVec::new_deterministic(dim, 1);
    let e2 = EntangledHVec::new_deterministic(dim, 2);

    c.bench_function("EntangledHVec hamming", |b| {
        b.iter(|| e1.hamming(black_box(&e2)))
    });

    c.bench_function("EntangledHVec similarity", |b| {
        b.iter(|| e1.similarity(black_box(&e2)))
    });

    c.bench_function("EntangledHVec bind", |b| b.iter(|| e1.bind(black_box(&e2))));

    c.bench_function("EntangledHVec permute", |b| {
        b.iter(|| e1.permute(black_box(7)))
    });

    let evecs: Vec<EntangledHVec> = (0..10)
        .map(|i| EntangledHVec::new_deterministic(dim, i))
        .collect();
    c.bench_function("EntangledHVec bundle 10", |b| {
        b.iter(|| EntangledHVec::bundle(black_box(&evecs)))
    });
}

fn benchmark_comparison(c: &mut Criterion) {
    let dim = 16384;

    let mut group = c.benchmark_group("bind");
    let e1 = EntangledHVec::new_deterministic(dim, 1);
    let e2 = EntangledHVec::new_deterministic(dim, 2);

    group.bench_function("EntangledHVec XOR", |b| b.iter(|| e1.bind(black_box(&e2))));
    group.finish();

    let mut group = c.benchmark_group("similarity");
    group.bench_function("EntangledHVec Jaccard", |b| {
        b.iter(|| e1.similarity(black_box(&e2)))
    });
    group.finish();

    let mut group = c.benchmark_group("bundle_10");
    let evecs: Vec<EntangledHVec> = (0..10)
        .map(|i| EntangledHVec::new_deterministic(dim, i))
        .collect();
    group.bench_function("EntangledHVec", |b| {
        b.iter(|| EntangledHVec::bundle(black_box(&evecs)))
    });
    group.finish();
}

fn benchmark_warm(c: &mut Criterion) {
    let dim = 10_000;
    let warm = WarmCore::new(dim as u32, None, None).unwrap();

    for i in 0..100 {
        let text = format!("This is a test sentence number {}", i);
        let vec = warm.encode_text(&text);
        warm.memorize(format!("id_{}", i), vec).unwrap();
    }

    c.bench_function("WarmCore query brute-force 100 items", |b| {
        b.iter(|| {
            let q_vec = warm.encode_text(black_box("test sentence"));
            warm.query(black_box(&q_vec), black_box(5))
        })
    });

    c.bench_function("WarmCore query_hopfield 100 items", |b| {
        b.iter(|| {
            let q_vec = warm.encode_text(black_box("test sentence"));
            warm.query_hopfield(black_box(&q_vec), black_box(5))
        })
    });

    c.bench_function("WarmCore encode_text short", |b| {
        b.iter(|| warm.encode_text(black_box("hello world")))
    });

    c.bench_function("WarmCore encode_text sentence", |b| {
        b.iter(|| {
            warm.encode_text(black_box(
                "The quick brown fox jumps over the lazy dog near the river",
            ))
        })
    });
}

fn benchmark_warm_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scaling");
    group.sample_size(20);

    for &n in &[100, 500, 1000] {
        let dim = 10_000;
        let warm = WarmCore::new(dim as u32, None, None).unwrap();
        for i in 0..n {
            let text = format!("Sentence about topic {} with variation {}", i % 20, i);
            let vec = warm.encode_text(&text);
            warm.memorize(format!("id_{}", i), vec).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("brute_force", n), &n, |b, _| {
            b.iter(|| {
                let q = warm.encode_text(black_box("topic about variation"));
                warm.query(black_box(&q), black_box(5))
            })
        });

        group.bench_with_input(BenchmarkId::new("hopfield", n), &n, |b, _| {
            b.iter(|| {
                let q = warm.encode_text(black_box("topic about variation"));
                warm.query_hopfield(black_box(&q), black_box(5))
            })
        });
    }
    group.finish();
}

/// Deterministic xorshift stream for the text corpora.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A word of a 20 000-word vocabulary drawn log-uniformly (Zipf-like, rank `r` with
    /// probability about 1/r).
    fn zipf_word(&mut self) -> String {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        format!("w{}", (20_000f64.powf(unit)) as u32)
    }
}

fn text_input(id: String, text: String, chunk_words: u32) -> DocumentInput {
    serde_json::from_value(serde_json::json!({
        "id": id, "text": text, "chunkWords": chunk_words, "overlapWords": 0, "storeText": false
    }))
    .unwrap()
}

fn zipf_corpus(seed: u64, documents: usize, words: usize, chunk_words: u32) -> Vec<DocumentInput> {
    let mut stream = Stream(seed);
    (0..documents)
        .map(|i| {
            let text: Vec<String> = (0..words).map(|_| stream.zipf_word()).collect();
            text_input(format!("doc-{i:06}"), text.join(" "), chunk_words)
        })
        .collect()
}

fn lexical_ingest() -> IngestOptions {
    IngestOptions {
        encode: EncodeMode::Lexical,
        ..Default::default()
    }
}

/// A store in its own directory (an engine without a path opens the working directory).
fn text_store() -> (tempfile::TempDir, WarmCore) {
    let dir = tempfile::tempdir().unwrap();
    let core = WarmCore::new(64, Some(dir.path().display().to_string()), None).unwrap();
    (dir, core)
}

/// A store of `corpus` memorized `batch` documents at a time (a batch holds at most 4096 chunks).
fn lexical_store(corpus: Vec<DocumentInput>, batch: usize) -> (tempfile::TempDir, WarmCore) {
    let (dir, core) = text_store();
    for batch in corpus.chunks(batch) {
        core.memorize_documents_with(batch.to_vec(), &lexical_ingest())
            .unwrap();
    }
    (dir, core)
}

/// Full-text engine paths: hybrid lexical ranking (fresh and churned), block-max WAND, match
/// counts, positional ingest and journal frame writes.
fn benchmark_text(c: &mut Criterion) {
    const QUERIES: [&str; 3] = ["w1 w700 w9000", "w2 w40", "w3 w5 w12000 w150"];
    let mut group = c.benchmark_group("text");
    group.sample_size(30);

    let (_store_dir, store) = lexical_store(zipf_corpus(7, 40_000, 60, 4096), 4_000);
    let search = SearchOptions::default();
    group.bench_function("hybrid_lexical_zipf", |b| {
        b.iter(|| {
            for query in QUERIES {
                black_box(store.search_documents(black_box(query), &search).unwrap());
            }
        })
    });
    // Common terms only: little for pruning to skip.
    group.bench_function("hybrid_lexical_common", |b| {
        b.iter(|| {
            for query in ["w1", "w1 w2"] {
                black_box(store.search_documents(black_box(query), &search).unwrap());
            }
        })
    });
    let wand = LexicalOptions {
        execution: LexicalExecution::BlockMaxWand,
        ..Default::default()
    };
    let parsed: Vec<_> = QUERIES
        .iter()
        .map(|q| LexicalQuery::parse(q).unwrap())
        .collect();
    group.bench_function("wand_zipf", |b| {
        b.iter(|| {
            for query in &parsed {
                black_box(store.search_lexical(query, &wand).unwrap());
            }
        })
    });

    // Churn: replace one document in ten and delete one in twenty.
    let replacements = zipf_corpus(11, 40_000, 60, 4096);
    for input in replacements.into_iter().step_by(10) {
        store
            .memorize_documents_with(vec![input], &lexical_ingest())
            .unwrap();
    }
    for i in (5..40_000).step_by(20) {
        store.delete_document(&format!("doc-{i:06}")).unwrap();
    }
    group.bench_function("hybrid_lexical_churned", |b| {
        b.iter(|| {
            for query in QUERIES {
                black_box(store.search_documents(black_box(query), &search).unwrap());
            }
        })
    });
    group.bench_function("wand_churned", |b| {
        b.iter(|| {
            for query in &parsed {
                black_box(store.search_lexical(query, &wand).unwrap());
            }
        })
    });
    let churn = zipf_corpus(13, 400, 60, 4096);
    group.bench_function("replace_document", |b| {
        let mut next = 0;
        b.iter(|| {
            let mut input = churn[next % churn.len()].clone();
            input.id = format!("doc-{:06}", (next * 97) % 40_000);
            next += 1;
            store
                .memorize_documents_with(vec![input], &lexical_ingest())
                .unwrap()
        })
    });
    drop(store);

    let (_chunked_dir, chunked) = lexical_store(zipf_corpus(17, 8_000, 120, 20), 600);
    group.bench_function("count_documents_multichunk", |b| {
        b.iter(|| {
            for query in QUERIES {
                black_box(chunked.count_documents(black_box(query)).unwrap());
            }
        })
    });
    drop(chunked);

    let positional = zipf_corpus(19, 2_000, 200, 4096);
    let with_positions = IngestOptions {
        positions: true,
        ..lexical_ingest()
    };
    group.bench_function("ingest_positions", |b| {
        b.iter_batched(
            || (text_store(), positional.clone()),
            |((dir, core), inputs)| {
                core.memorize_documents_with(inputs, &with_positions)
                    .unwrap();
                (dir, core)
            },
            BatchSize::PerIteration,
        )
    });

    // Incompressible text: LZ4 falls back to the raw payload on every frame.
    let mut stream = Stream(23);
    let noisy: Vec<DocumentInput> = (0..4)
        .map(|i| {
            let words: Vec<String> = (0..60_000)
                .map(|_| format!("{:x}", stream.next() >> 24))
                .collect();
            text_input(format!("noisy-{i}"), words.join(" "), 4096)
        })
        .collect();
    group.sample_size(20);
    group.bench_function("ingest_persistent_incompressible", |b| {
        b.iter_batched(
            || {
                let (dir, core) = text_store();
                (dir, core, noisy.clone())
            },
            |(dir, core, inputs)| {
                core.memorize_documents_with(inputs, &lexical_ingest())
                    .unwrap();
                (dir, core)
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_entangled,
    benchmark_comparison,
    benchmark_warm,
    benchmark_warm_scaling,
    benchmark_text,
);
criterion_main!(benches);
