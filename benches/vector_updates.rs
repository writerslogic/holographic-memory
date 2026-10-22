// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use holographic_memory::core::qgraph::{
    BuildParams, Graph, Metric, UpdatableVGraph, VGraph, VSearchParams,
};
use std::hint::black_box;

fn vectors() -> Vec<f32> {
    let mut seed = 42u64;
    (0..10000 * 32)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

fn search(c: &mut Criterion) {
    let data = vectors();
    let graph = Graph::build(
        &data,
        32,
        &BuildParams {
            metric: Metric::Euclidean,
            ..BuildParams::default()
        },
    )
    .unwrap();
    let index = VGraph::from_graph(&graph, true).unwrap();
    let mut searcher = index.searcher();
    let mut out = Vec::new();
    let params = VSearchParams {
        ef: 64,
        rerank: 64,
        ..VSearchParams::default()
    };
    let mut q = 0;
    c.bench_function("vertex_10k_32d", |b| {
        b.iter(|| {
            q = (q + 1) % 256;
            searcher
                .search(black_box(&data[q * 32..(q + 1) * 32]), 10, params, &mut out)
                .unwrap();
            black_box(&out);
        })
    });
    let directory = tempfile::tempdir().unwrap();
    let updated =
        UpdatableVGraph::create_from_graph(&directory.path().join("index"), &graph, &index)
            .unwrap();
    c.bench_function("updatable_10k_32d_clean", |b| {
        b.iter(|| {
            q = (q + 1) % 256;
            black_box(
                updated
                    .search(black_box(&data[q * 32..(q + 1) * 32]), 10, params)
                    .unwrap(),
            );
        })
    });
}

criterion_group! { name = benches; config = Criterion::default().sample_size(50); targets = search }
criterion_main!(benches);
