// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! The search knobs of `core::qgraph` (`SearchKnobs`) that only change memory traffic keep the
//! results bit-identical, on a synthetic clustered set (50k x 100 in release builds; a smaller
//! one under debug assertions, where the graph build is slow): every prefetch depth, batched
//! estimation, a frontier of 1.0, and the block screen with an infinite margin (nothing
//! screened), which must reproduce the plain search over the same graph. The settings that
//! change results (a frontier below 1.0, the block screen at one sigma) may only prune: fewer
//! estimates in total, with recall close to the baseline's.
//!
//! Run in release: `cargo test --release --test qgraph_prefetch`.

use std::sync::OnceLock;

use holographic_memory::core::qgraph::{BuildParams, Graph, SearchKnobs, VGraph, VSearchParams};

const DIM: usize = 100;
const N: usize = if cfg!(debug_assertions) {
    4_000
} else {
    50_000
};
const CENTRES: usize = 64;
const QUERIES: usize = 200;
const K: usize = 10;

/// SplitMix64: a fixed seed gives the same set on every platform.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// `n` rows of `DIM` around `CENTRES` random centres with per-coordinate noise of half the
/// centre amplitude, clusters scattered over the ids.
fn clustered(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix(seed);
    let centres: Vec<f32> = (0..CENTRES * DIM).map(|_| rng.signed()).collect();
    let mut rows = Vec::with_capacity(n * DIM);
    for i in 0..n {
        let c = (i * 7919) % CENTRES;
        rows.extend(
            centres[c * DIM..(c + 1) * DIM]
                .iter()
                .map(|&v| v + 0.5 * rng.signed()),
        );
    }
    rows
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Exact cosine top-`K` of `q` over the unit rows.
fn top_k(q: &[f32], unit: &[f32]) -> Vec<u32> {
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(K + 1);
    for (i, row) in unit.as_chunks::<DIM>().0.iter().enumerate() {
        let s: f32 = q.iter().zip(row).map(|(a, b)| a * b).sum();
        if best.len() < K || s > best[K - 1].0 {
            let pos = best.partition_point(|t| t.0 >= s);
            best.insert(pos, (s, i as u32));
            best.truncate(K);
        }
    }
    best.into_iter().map(|t| t.1).collect()
}

struct Fixture {
    graph: Graph,
    queries: Vec<Vec<f32>>,
    truth: Vec<Vec<u32>>,
}

/// One graph per test process (nextest runs each test in its own): the build dominates the
/// test's time.
fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let rows = clustered(N, 0x5EED_0001);
        let mut unit = rows.clone();
        unit.as_chunks_mut::<DIM>()
            .0
            .iter_mut()
            .for_each(|row| normalize(row));
        let mut queries: Vec<Vec<f32>> = clustered(QUERIES, 0x5EED_0002)
            .as_chunks::<DIM>()
            .0
            .iter()
            .map(|row| row.to_vec())
            .collect();
        queries.iter_mut().for_each(|q| normalize(q));
        let truth = queries.iter().map(|q| top_k(q, &unit)).collect();
        let params = BuildParams {
            build_ef: 96,
            degree: 32,
            ..BuildParams::default()
        };
        let graph = Graph::build(&rows, DIM, &params).expect("the synthetic set builds");
        Fixture {
            graph,
            queries,
            truth,
        }
    })
}

struct Run {
    ids: Vec<Vec<u32>>,
    evals: Vec<usize>,
    screened: usize,
}

fn run(index: &VGraph, knobs: SearchKnobs, params: VSearchParams) -> Run {
    let f = fixture();
    let mut s = index.searcher_with(knobs);
    let mut out = Vec::new();
    let mut r = Run {
        ids: Vec::with_capacity(QUERIES),
        evals: Vec::with_capacity(QUERIES),
        screened: 0,
    };
    for q in &f.queries {
        let e = s
            .search(q, K, params, &mut out)
            .expect("unit queries of the index dimension are valid");
        r.ids.push(out.clone());
        r.evals.push(e);
        r.screened += s.last_screened();
    }
    r
}

fn recall(r: &Run) -> f64 {
    let f = fixture();
    let hits: usize = r
        .ids
        .iter()
        .zip(&f.truth)
        .map(|(got, t)| got.iter().filter(|g| t.contains(g)).count())
        .sum();
    hits as f64 / (K * QUERIES) as f64
}

fn total(r: &Run) -> usize {
    r.evals.iter().sum()
}

fn assert_identical(what: &str, a: &Run, b: &Run) {
    assert_eq!(a.ids, b.ids, "{what}: result ids differ");
    assert_eq!(a.evals, b.evals, "{what}: estimate counts differ");
}

fn params(ef: usize, rerank: usize) -> VSearchParams {
    VSearchParams {
        ef,
        rerank,
        patience: 0,
        screen_sigmas: 1.0,
        screen: true,
        fuse: false,
        neighbour_cap: 0,
    }
}

/// The measured glove configuration: 4-bit rows, 8-bit residual, aligned rows, BFS order,
/// 3-byte ids, no vertex screen.
fn plain(g: &Graph) -> VGraph {
    VGraph::from_graph_with(g, 4, 8, true, true, 3, false).expect("the graph encodes")
}

#[test]
fn prefetch_and_batched_estimation_keep_results_identical() {
    let f = fixture();
    let indexes = [
        (
            "4-bit, residual 8, aligned, bfs, 3-byte ids",
            plain(&f.graph),
        ),
        (
            "4-bit with the 1-bit vertex screen",
            VGraph::from_graph_with(&f.graph, 4, 8, true, true, 4, true).expect("encodes"),
        ),
        (
            "8-bit, unaligned, delta-coded lists",
            VGraph::from_graph_with(&f.graph, 8, 0, false, false, 0, false).expect("encodes"),
        ),
    ];
    let knobs = [
        (
            "prefetch 1",
            SearchKnobs {
                prefetch: 1,
                ..SearchKnobs::OFF
            },
        ),
        (
            "prefetch 2",
            SearchKnobs {
                prefetch: 2,
                ..SearchKnobs::OFF
            },
        ),
        (
            "prefetch 3",
            SearchKnobs {
                prefetch: 3,
                ..SearchKnobs::OFF
            },
        ),
        (
            "batch_est",
            SearchKnobs {
                batch_est: true,
                ..SearchKnobs::OFF
            },
        ),
        (
            "prefetch 3 + batch_est",
            SearchKnobs {
                prefetch: 3,
                batch_est: true,
                ..SearchKnobs::OFF
            },
        ),
        (
            "frontier 1.0",
            SearchKnobs {
                frontier: 1.0,
                ..SearchKnobs::OFF
            },
        ),
    ];
    for (name, index) in &indexes {
        for p in [params(96, 64), params(40, 0)] {
            let base = run(index, SearchKnobs::OFF, p);
            // The floor only rules out a broken search, against which identical results would
            // prove nothing (chance is K/N). Clustered rows with this much noise are hard: the
            // release fixture (50k rows) measures 0.63-0.66 at ef 96 with rerank and 0.39-0.46
            // at ef 40 ranked by the codes alone.
            let r = recall(&base);
            let floor = if p.rerank > 0 { 0.5 } else { 0.3 };
            assert!(
                r > floor,
                "{name}, ef {}, rerank {}: baseline recall {r}",
                p.ef,
                p.rerank
            );
            for (kname, k) in knobs {
                let what = format!("{name}, ef {}, {kname}", p.ef);
                assert_identical(&what, &base, &run(index, k, p));
            }
        }
    }
}

#[test]
fn frontier_below_one_only_prunes() {
    let f = fixture();
    let index = plain(&f.graph);
    let p = params(96, 64);
    let base = run(&index, SearchKnobs::OFF, p);
    let pruned = run(
        &index,
        SearchKnobs {
            frontier: 0.9,
            ..SearchKnobs::OFF
        },
        p,
    );
    // The frontier rule only cuts the same deterministic expansion sequence short.
    for (i, (a, b)) in base.evals.iter().zip(&pruned.evals).enumerate() {
        assert!(
            b <= a,
            "query {i}: {b} estimates with frontier 0.9, {a} without"
        );
    }
    let (t0, t1) = (total(&base), total(&pruned));
    assert!(
        t1 < t0,
        "frontier 0.9 never stopped early ({t1} against {t0} estimates)"
    );
    let (r0, r1) = (recall(&base), recall(&pruned));
    assert!(r1 >= 0.8 * r0, "recall {r1} with frontier 0.9 against {r0}");
}

#[test]
fn block_screen_with_an_infinite_margin_reproduces_the_plain_search() {
    let f = fixture();
    let plain_index = plain(&f.graph);
    let blocked = plain(&f.graph)
        .with_blocks(&f.graph, 1)
        .expect("1-bit neighbour blocks encode");
    assert!(blocked.has_blocks());
    let n = blocked.len();
    let per_node = blocked.block_bytes() / n;
    assert_eq!(blocked.block_bytes(), per_node * n, "blocks are per vertex");
    assert_eq!(
        blocked.index_bytes() - plain_index.index_bytes(),
        blocked.block_bytes(),
        "index_bytes reports the blocks"
    );
    eprintln!(
        "1-bit neighbour blocks: {per_node} bytes per node, {} bytes in all",
        blocked.block_bytes()
    );
    let on = SearchKnobs {
        block_screen: true,
        ..SearchKnobs::OFF
    };
    for ef in [40, 96] {
        let open = VSearchParams {
            screen_sigmas: f32::INFINITY,
            ..params(ef, 64)
        };
        let base = run(&plain_index, SearchKnobs::OFF, open);
        let same = run(&blocked, on, open);
        assert_eq!(
            same.screened, 0,
            "ef {ef}: an infinite margin screens nothing"
        );
        assert_identical(
            &format!("block screen, infinite margin, ef {ef}"),
            &base,
            &same,
        );
        let tight = run(&blocked, on, params(ef, 64));
        assert!(
            tight.screened > 0,
            "ef {ef}: the block screen skipped nothing at one sigma"
        );
        let (t0, t1) = (total(&base), total(&tight));
        assert!(
            t1 < t0,
            "ef {ef}: {t1} estimates screened against {t0} unscreened"
        );
        let (r0, r1) = (recall(&base), recall(&tight));
        assert!(r1 >= 0.8 * r0, "ef {ef}: recall {r1} screened against {r0}");
    }
}
