// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use holographic_memory::core::qgraph::{
    BuildParams, Graph, Metric, UpdatableVGraph, VGraph, VSearchParams,
};
use std::hint::black_box;
use std::time::Duration;

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
    c.bench_function("updatable_10k_32d_buffered", |b| {
        b.iter(|| {
            q = (q + 1) % 256;
            updated
                .search_into(black_box(&data[q * 32..(q + 1) * 32]), 10, params, &mut out)
                .unwrap();
            black_box(&out);
        })
    });
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

fn mixed_search(c: &mut Criterion) {
    let dim = 100;
    let mut seed = 42u64;
    let data: Vec<f32> = (0..12_000 * dim)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u32 << 24) as f32
        })
        .collect();
    let params = BuildParams {
        metric: Metric::Euclidean,
        degree: 32,
        build_ef: 64,
        ..Default::default()
    };
    let root = tempfile::tempdir().unwrap();
    let mixed = UpdatableVGraph::create(
        &root.path().join("mixed"),
        &data[..10_000 * dim],
        dim,
        &params,
    )
    .unwrap();
    let ids: Vec<_> = (10_000..12_000).collect();
    mixed.insert_batch(&ids, &data[10_000 * dim..]).unwrap();
    let delta = UpdatableVGraph::create(&root.path().join("delta"), &[], dim, &params).unwrap();
    delta.insert_batch(&ids, &data[10_000 * dim..]).unwrap();
    let mut group = c.benchmark_group("updated_100d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);
    for (name, index) in [("mixed_10k_2k", &mixed), ("delta_2k", &delta)] {
        for ef in [64, 1024, 2048] {
            let p = VSearchParams {
                ef,
                rerank: ef,
                ..Default::default()
            };
            let mut q = 0;
            let mut out = Vec::new();
            group.bench_with_input(BenchmarkId::new(name, ef), &ef, |b, _| {
                b.iter(|| {
                    q = (q + 1) % 256;
                    index
                        .search_into(black_box(&data[q * dim..(q + 1) * dim]), 10, p, &mut out)
                        .unwrap();
                    black_box(&out);
                })
            });
        }
    }
    group.finish();
}

criterion_group! { name = benches; config = Criterion::default().sample_size(50); targets = search, mixed_search }
criterion_main!(benches);
