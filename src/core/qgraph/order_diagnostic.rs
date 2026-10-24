// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Locality of candidate vertex orders for the encoded index (experimental diagnostics).
//!
//! The vertex index renumbers vertices in breadth-first order so that vertices expanded
//! together sit close in memory and neighbour-id deltas stay small. This scores other orders
//! on the built graph without encoding it: bytes per edge of the sorted delta-varint lists the
//! `id_bytes = 0` layout stores, and the share of edges whose endpoints fall within 32 rows
//! (one 4 KiB page of 128-byte code rows) or 512 rows of each other, the Gorder-style window
//! objective. These are layout proxies; cache misses need hardware counters.

use std::time::Instant;

use super::{Graph, GraphView};

/// Scores of one order.
#[derive(Clone, Debug)]
pub struct OrderLocality {
    pub name: &'static str,
    /// Seconds to compute the order.
    pub order_secs: f64,
    /// Bytes per stored edge of the delta-varint lists.
    pub varint_bytes_per_edge: f64,
    /// Mean of `log2(|new(u) - new(v)| + 1)` over the edges `v -> u`.
    pub mean_log2_gap: f64,
    /// Share of edges with `|new(u) - new(v)| < 32`.
    pub within_32: f64,
    /// Share of edges with `|new(u) - new(v)| < 512`.
    pub within_512: f64,
    /// Mean number of distinct 32-row blocks (4 KiB pages of 128-byte code rows) the
    /// neighbours of one list fall in: the pages one expansion touches.
    pub pages_per_list: f64,
}

/// Scores the insertion order, the breadth-first order the encoder uses, its reverse (reverse
/// Cuthill-McKee on a graph whose lists all have the same length), a cluster order (vertices
/// grouped by their graph-nearest seed of the top routing layer by multi-source breadth-first
/// search, clusters in breadth-first order of their seeds, breadth-first within each cluster),
/// and Gorder with windows 5 and 32.
pub fn order_locality(g: &Graph) -> Vec<OrderLocality> {
    let view = g.view();
    let mut out = Vec::new();
    let mut score = |name, f: &dyn Fn() -> Vec<u32>| {
        let t = Instant::now();
        let order = f();
        let order_secs = t.elapsed().as_secs_f64();
        out.push(locality(view, name, order_secs, &order));
    };
    score("insertion", &|| (0..view.n as u32).collect());
    score("bfs", &|| super::vertex::bfs_order(view));
    score("reverse_bfs", &|| {
        let mut o = super::vertex::bfs_order(view);
        o.reverse();
        o
    });
    score("cluster_bfs", &|| cluster_bfs(view));
    score("gorder_w5", &|| gorder(view, 5));
    score("gorder_w32", &|| gorder(view, 32));
    out
}

fn locality(g: GraphView<'_>, name: &'static str, order_secs: f64, order: &[u32]) -> OrderLocality {
    let mut new_id = vec![0u32; g.n];
    for (v, &o) in order.iter().enumerate() {
        new_id[o as usize] = v as u32;
    }
    let (mut edges, mut bytes, mut log_gap, mut w32, mut w512) = (0u64, 0u64, 0f64, 0u64, 0u64);
    let (mut lists, mut pages) = (0u64, 0u64);
    let mut sorted = Vec::new();
    for (nv, &o) in order.iter().enumerate() {
        sorted.clear();
        sorted.extend(g.neighbors(o as usize).iter().map(|&u| new_id[u as usize]));
        sorted.sort_unstable();
        if !sorted.is_empty() {
            lists += 1;
            pages += 1 + sorted.windows(2).filter(|p| p[0] / 32 != p[1] / 32).count() as u64;
        }
        let mut prev = nv as i64;
        for (i, &u) in sorted.iter().enumerate() {
            let gap = (i64::from(u) - nv as i64).unsigned_abs();
            log_gap += ((gap + 1) as f64).log2();
            w32 += u64::from(gap < 32);
            w512 += u64::from(gap < 512);
            let delta = if i == 0 {
                let d = i64::from(u) - prev;
                ((d << 1) ^ (d >> 63)) as u64
            } else {
                (i64::from(u) - prev - 1) as u64
            };
            bytes += u64::from((64 - delta.leading_zeros()).max(1).div_ceil(7));
            prev = i64::from(u);
            edges += 1;
        }
    }
    let e = edges.max(1) as f64;
    OrderLocality {
        name,
        order_secs,
        varint_bytes_per_edge: bytes as f64 / e,
        mean_log2_gap: log_gap / e,
        within_32: w32 as f64 / e,
        within_512: w512 as f64 / e,
        pages_per_list: pages as f64 / lists.max(1) as f64,
    }
}

/// Gorder (Wei, Yu, Lu and Lin, SIGMOD 2016): places vertices one at a time, each time the
/// unplaced vertex with the highest score against the last `w` placed ones, where a pair
/// scores one per edge between them and one per shared in-neighbour (two vertices listed by
/// the same vertex are expanded together). The scores are kept in a bucket queue ("unit heap")
/// so each update is O(1); ties follow deterministic bucket insertion order. Cost is
/// about `2 n (d_out + d_in + d_in d_out)` updates.
fn gorder(g: GraphView<'_>, w: usize) -> Vec<u32> {
    const NONE: u32 = u32::MAX;
    let n = g.n;
    // In-neighbour lists, compressed.
    let mut off = vec![0usize; n + 1];
    for v in 0..n {
        for &u in g.neighbors(v) {
            off[u as usize + 1] += 1;
        }
    }
    for i in 0..n {
        off[i + 1] += off[i];
    }
    let mut fill = off[..n].to_vec();
    let mut ins = vec![0u32; off[n]];
    for v in 0..n {
        for &u in g.neighbors(v) {
            ins[fill[u as usize]] = v as u32;
            fill[u as usize] += 1;
        }
    }
    let in_of = |v: usize| &ins[off[v]..off[v + 1]];

    // Bucket queue over the unplaced vertices: `head[k]` starts the doubly linked list of the
    // vertices with score `k`; `top` bounds the highest non-empty bucket.
    let mut key = vec![0u32; n];
    let (mut prev, mut next) = (vec![NONE; n], vec![NONE; n]);
    let mut head: Vec<u32> = vec![NONE];
    let mut placed = vec![false; n];
    let bfs = super::vertex::bfs_order(g);
    for &v in bfs.iter().rev() {
        next[v as usize] = head[0];
        if head[0] != NONE {
            prev[head[0] as usize] = v;
        }
        head[0] = v;
    }
    let mut top = 0usize;
    let unlink = |v: usize, key: &[u32], prev: &mut [u32], next: &mut [u32], head: &mut [u32]| {
        let (p, q) = (prev[v], next[v]);
        if p == NONE {
            head[key[v] as usize] = q;
        } else {
            next[p as usize] = q;
        }
        if q != NONE {
            prev[q as usize] = p;
        }
        prev[v] = NONE;
        next[v] = NONE;
    };
    let mut order = Vec::with_capacity(n);
    let mut v = g.entry as usize;
    for i in 0..n {
        if i > 0 {
            while top > 0 && head[top] == NONE {
                top -= 1;
            }
            v = head[top] as usize;
        }
        unlink(v, &key, &mut prev, &mut next, &mut head);
        placed[v] = true;
        order.push(v as u32);
        // Score changes: the new vertex enters the window, the oldest one leaves it.
        let leaving = (i >= w).then(|| order[i - w] as usize);
        for (x, up) in std::iter::once((v, true)).chain(leaving.map(|x| (x, false))) {
            let related = g.neighbors(x).iter().chain(in_of(x)).copied().chain(
                in_of(x)
                    .iter()
                    .flat_map(|&p| g.neighbors(p as usize).iter().copied()),
            );
            for u in related {
                let u = u as usize;
                if placed[u] {
                    continue;
                }
                unlink(u, &key, &mut prev, &mut next, &mut head);
                if up {
                    key[u] += 1;
                } else {
                    key[u] -= 1;
                }
                let k = key[u] as usize;
                if k >= head.len() {
                    head.resize(k + 1, NONE);
                }
                next[u] = head[k];
                if head[k] != NONE {
                    prev[head[k] as usize] = u as u32;
                }
                head[k] = u as u32;
                top = top.max(k);
            }
        }
    }
    order
}

/// Multi-source breadth-first clusters around the top routing layer's vertices.
fn cluster_bfs(g: GraphView<'_>) -> Vec<u32> {
    const NONE: u32 = u32::MAX;
    // Seeds: the vertices of the smallest routing layer (or the entry alone), in BFS order.
    let seeds: Vec<u32> = match g.layers.last() {
        Some(top) => g.upper_ids[..top.len()].to_vec(),
        None => vec![g.entry],
    };
    let bfs = super::vertex::bfs_order(g);
    let mut rank = vec![0u32; g.n];
    for (i, &v) in bfs.iter().enumerate() {
        rank[v as usize] = i as u32;
    }
    let mut seeds = seeds;
    seeds.sort_by_key(|&s| rank[s as usize]);
    let mut owner = vec![NONE; g.n];
    let mut queue = Vec::with_capacity(g.n);
    for (c, &s) in seeds.iter().enumerate() {
        if owner[s as usize] == NONE {
            owner[s as usize] = c as u32;
            queue.push(s);
        }
    }
    let mut head = 0;
    while head < queue.len() {
        let v = queue[head] as usize;
        head += 1;
        for &u in g.neighbors(v) {
            if owner[u as usize] == NONE {
                owner[u as usize] = owner[v];
                queue.push(u);
            }
        }
    }
    // Unreached vertices join the last cluster. Then a breadth-first walk within each cluster
    // from its seed, in seed order, and from any vertex its cluster's walk missed.
    let last = seeds.len().saturating_sub(1) as u32;
    owner
        .iter_mut()
        .filter(|o| **o == NONE)
        .for_each(|o| *o = last);
    let mut placed = vec![false; g.n];
    let mut order = Vec::with_capacity(g.n);
    let starts = seeds.iter().chain(&bfs).map(|&v| v as usize);
    for start in starts {
        if placed[start] {
            continue;
        }
        let c = owner[start];
        placed[start] = true;
        let mut h = order.len();
        order.push(start as u32);
        while h < order.len() {
            let v = order[h] as usize;
            h += 1;
            for &u in g.neighbors(v) {
                if owner[u as usize] == c && !std::mem::replace(&mut placed[u as usize], true) {
                    order.push(u);
                }
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::qgraph::{BuildParams, Metric};

    #[test]
    fn gorder_selects_a_maximum_window_score() {
        for n in [4, 5, 6, 31, 32, 33] {
            let data: Vec<f32> = (0..n * 4)
                .map(|i| ((i * 17 + i / 4) % 101) as f32)
                .collect();
            let g = Graph::build(
                &data,
                4,
                &BuildParams {
                    metric: Metric::Euclidean,
                    degree: 4,
                    build_ef: 16,
                    ..BuildParams::default()
                },
            )
            .unwrap();
            let view = g.view();
            for window in [0, 1, 4, 5, 6, 31, 32, 33] {
                let order = gorder(view, window);
                assert_eq!(order, gorder(view, window));
                let mut remaining: std::collections::BTreeSet<_> = (0..n as u32).collect();
                for (i, &chosen) in order.iter().enumerate() {
                    let score = |candidate: u32| {
                        order[i.saturating_sub(window)..i]
                            .iter()
                            .map(|&prior| {
                                usize::from(view.neighbors(prior as usize).contains(&candidate))
                                    + usize::from(
                                        view.neighbors(candidate as usize).contains(&prior),
                                    )
                                    + (0..n)
                                        .filter(|&source| {
                                            let neighbors = view.neighbors(source);
                                            neighbors.contains(&prior)
                                                && neighbors.contains(&candidate)
                                        })
                                        .count()
                            })
                            .sum::<usize>()
                    };
                    assert_eq!(
                        score(chosen),
                        remaining.iter().copied().map(score).max().unwrap()
                    );
                    assert!(remaining.remove(&chosen));
                }
                assert!(remaining.is_empty());
            }
            let order: Vec<_> = (0..n as u32).collect();
            let nonempty: Vec<_> = (0..n)
                .map(|v| view.neighbors(v))
                .filter(|row| !row.is_empty())
                .collect();
            let expected_pages: usize = nonempty
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|v| v / 32)
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                })
                .sum();
            assert_eq!(
                locality(view, "insertion", 0.0, &order).pages_per_list,
                expected_pages as f64 / nonempty.len().max(1) as f64
            );
        }
    }

    #[test]
    fn every_order_is_a_permutation_with_the_same_edges() {
        let (n, dim) = (3000, 16);
        let mut s = 7u64;
        let data: Vec<f32> = (0..n * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let params = BuildParams {
            metric: Metric::Euclidean,
            degree: 16,
            build_ef: 40,
            ..BuildParams::default()
        };
        let g = Graph::build(&data, dim, &params).unwrap();
        let view = g.view();
        let mut orders = vec![
            super::super::vertex::bfs_order(view),
            cluster_bfs(view),
            gorder(view, 5),
        ];
        for o in &mut orders {
            o.sort_unstable();
            assert_eq!(*o, (0..n as u32).collect::<Vec<_>>());
        }
        let scores = order_locality(&g);
        assert_eq!(scores.len(), 6);
        // Reversal keeps every gap, so the window shares and the mean log gap are equal.
        assert_eq!(scores[1].within_32, scores[2].within_32);
        assert!((scores[1].mean_log2_gap - scores[2].mean_log2_gap).abs() < 1e-9);
    }
}
