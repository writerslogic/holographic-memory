// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::time::Duration;
use writerslogic_warm::core::qgraph::{
    BuildParams, Graph, Metric, QGraph, SearchKnobs, SearchParams, UpdatableVGraph, VGraph,
    VSearchParams,
};

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

/// `n * dim` seeded floats in `[-0.5, 0.5)`.
fn uniform(n: usize, dim: usize, mut seed: u64) -> Vec<f32> {
    (0..n * dim)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// Angular search over one 100k x 128 graph: the edge-code index and every vertex-index
/// path (plain, 1-bit screen, neighbour blocks, blocks with fusion, block screen), with
/// held-out queries.
fn index_search(c: &mut Criterion) {
    const N: usize = 100_000;
    const DIM: usize = 128;
    const QUERIES: usize = 256;
    let data = uniform(N, DIM, 7);
    let queries = uniform(QUERIES, DIM, 11);
    let params = BuildParams::default();
    let graph = Graph::build(&data, DIM, &params).unwrap();
    let query = |q: usize| &queries[q * DIM..(q + 1) * DIM];
    let mut group = c.benchmark_group("search_100k_128d");
    group.sample_size(30);
    let qgraph = QGraph::from_graph(&graph, &params).unwrap();
    let mut searcher = qgraph.searcher();
    let mut out = Vec::new();
    for ef in [64, 200] {
        let p = SearchParams { ef, max_exact: 0 };
        let mut q = 0;
        group.bench_with_input(BenchmarkId::new("qgraph", ef), &ef, |b, _| {
            b.iter(|| {
                q = (q + 1) % QUERIES;
                black_box(searcher.search(black_box(query(q)), 10, p, &mut out));
            })
        });
    }
    drop(searcher);
    drop(qgraph);
    let plain = VGraph::from_graph(&graph, true).unwrap();
    let screened = VGraph::from_graph_with(&graph, 8, 8, false, false, 4, true).unwrap();
    let blocks = VGraph::from_graph(&graph, true)
        .unwrap()
        .with_blocks(&graph, 4)
        .unwrap();
    let screen_knobs = SearchKnobs {
        block_screen: true,
        ..SearchKnobs::OFF
    };
    let cases = [
        ("vgraph_plain", &plain, false, SearchKnobs::OFF),
        ("vgraph_screen", &screened, false, SearchKnobs::OFF),
        ("vgraph_blocks", &blocks, false, SearchKnobs::OFF),
        ("vgraph_blocks_fuse", &blocks, true, SearchKnobs::OFF),
        ("vgraph_block_screen", &blocks, false, screen_knobs),
    ];
    for (name, index, fuse, knobs) in cases {
        let mut searcher = index.searcher_with(knobs);
        let mut ids = Vec::new();
        for ef in [64, 200] {
            let p = VSearchParams {
                ef,
                rerank: ef,
                fuse,
                ..VSearchParams::default()
            };
            let mut q = 0;
            group.bench_with_input(BenchmarkId::new(name, ef), &ef, |b, _| {
                b.iter(|| {
                    q = (q + 1) % QUERIES;
                    searcher
                        .search(black_box(query(q)), 10, p, &mut ids)
                        .unwrap();
                    black_box(&ids);
                })
            });
        }
    }
    group.finish();
}

/// Graph construction over 20k x 128 (angular), on the rayon pool as configured
/// (`RAYON_NUM_THREADS=1` for a single-thread figure).
fn graph_build(c: &mut Criterion) {
    const N: usize = 20_000;
    const DIM: usize = 128;
    let data = uniform(N, DIM, 13);
    let mut group = c.benchmark_group("build_20k_128d");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(20));
    group.bench_function("graph", |b| {
        b.iter(|| black_box(Graph::build(black_box(&data), DIM, &BuildParams::default()).unwrap()))
    });
    group.finish();
}

criterion_group! { name = benches; config = Criterion::default().sample_size(50); targets = search, mixed_search, index_search, graph_build }
criterion_main!(benches);
