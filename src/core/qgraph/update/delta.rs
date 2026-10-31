// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::HashMap;

use super::super::{
    adjacency::Adjacency, build_kernels::prefetch_row, kernels::dot, norms_opt, BuildParams,
    BuildVisited, Rows, NONE,
};
use super::Metric;

thread_local! { static SEEN: RefCell<BuildVisited> = RefCell::new(BuildVisited::default()); }

#[derive(Clone, Default)]
pub(super) struct Delta {
    pub(super) vectors: super::delta_rows::DeltaRows,
    norms: Vec<f32>,
    edges: Adjacency,
    first_child: Vec<u32>,
    next_sibling: Vec<u32>,
    ids: Vec<u32>,
    /// The slot of every live id. Changed only by `insert`, `insert_batch` and `remove`,
    /// which keep `alive` in step.
    pub live: HashMap<u32, usize>,
    /// Per slot: `live` maps its id to it, so a search checks liveness without a hash.
    alive: Vec<bool>,
}

impl Delta {
    #[cfg(test)]
    pub fn assert_invariants(&self, degree: usize) {
        for slot in 0..self.len() {
            let edges = &self.edges[slot];
            assert!(edges.len() <= degree);
            let mut unique = std::collections::HashSet::new();
            assert!(edges
                .iter()
                .all(|&u| (u as usize) < self.len() && u as usize != slot && unique.insert(u)));
        }
        assert!(self
            .live
            .iter()
            .all(|(id, &slot)| self.ids.get(slot) == Some(id)));
        assert_eq!(self.alive.len(), self.ids.len());
        for (slot, (id, &alive)) in self.ids.iter().zip(&self.alive).enumerate() {
            assert_eq!(alive, self.live.get(id) == Some(&slot), "slot {slot}");
        }
    }

    /// Marks `id` deleted; its slot stays allocated.
    pub fn remove(&mut self, id: u32) {
        if let Some(slot) = self.live.remove(&id) {
            self.alive[slot] = false;
        }
    }

    fn set_live(&mut self, id: u32, slot: usize) {
        if let Some(old) = self.live.insert(id, slot) {
            self.alive[old] = false;
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn row(&self, slot: usize, dim: usize) -> &[f32] {
        &self.vectors[slot * dim..(slot + 1) * dim]
    }

    fn score(&self, query: &[f32], slot: usize, metric: Metric) -> f32 {
        self.score_from(&self.vectors, query, slot, metric)
    }

    fn score_from(&self, vectors: &[f32], query: &[f32], slot: usize, metric: Metric) -> f32 {
        let at = slot * query.len();
        let row = &vectors[at..at + query.len()];
        match metric {
            Metric::Angular => -dot(query, row),
            Metric::Euclidean => self.norms[slot] * 0.5 - dot(query, row),
        }
    }

    pub fn insert(&mut self, id: u32, row: &[f32], params: &BuildParams) {
        let slot = self.ids.len();
        self.ids.push(id);
        self.alive.push(true);
        self.vectors.extend_from_slice(row);
        if params.metric == Metric::Euclidean {
            self.norms.push(dot(row, row));
        }
        self.edges.push(&[], params.degree);
        self.first_child.push(NONE);
        self.next_sibling.push(NONE);
        if slot > 0 {
            let rows = Rows {
                data: &self.vectors,
                dim: row.len(),
                norms: norms_opt(&self.norms),
                codes: None,
                quantized: false,
            };
            let candidates = SEEN.with(|seen| {
                rows.greedy(
                    &self.edges,
                    0,
                    slot as u32,
                    params.build_ef,
                    &mut seen.borrow_mut(),
                )
            });
            let neighbors = rows.prune(slot as u32, candidates, params.alpha, params.degree);
            // A rooted routing tree keeps every slot reachable when reciprocal pruning
            // removes its last incoming graph edge. Tree edges survive logical deletion.
            let parent = neighbors[0] as usize;
            self.next_sibling[slot] = self.first_child[parent];
            self.first_child[parent] = slot as u32;
            let mut d = Vec::new();
            for &u in &neighbors {
                // Four rows at a time; `dot4` is bit-identical to `dot`.
                let edges = &self.edges[u as usize];
                rows.d2_many(u, edges, &mut d);
                let mut candidates: Vec<_> = d.iter().copied().zip(edges.iter().copied()).collect();
                candidates.push((rows.d2(u, slot as u32), slot as u32));
                let pruned = rows.prune(u, candidates, params.alpha, params.degree);
                self.edges.set(u as usize, &pruned);
            }
            self.edges.set(slot, &neighbors);
        }
        self.set_live(id, slot);
    }

    /// Inserts prepared rows (row-major) under `ids`, in parallel: the build's batched
    /// insertion pass, whose batches each search a frozen graph, so the graph does not depend
    /// on thread scheduling.
    pub fn insert_batch(&mut self, ids: &[u32], rows: &[f32], params: &BuildParams) {
        let dim = rows.len() / ids.len().max(1);
        let first = self.ids.len();
        if ids.len() < 2 || dim == 0 {
            for (&id, row) in ids.iter().zip(rows.chunks_exact(dim.max(1))) {
                self.insert(id, row, params);
            }
            return;
        }
        self.ids.extend_from_slice(ids);
        self.alive.resize(self.ids.len(), true);
        self.vectors.extend_from_slice(rows);
        if params.metric == Metric::Euclidean {
            self.norms.extend(rows.chunks_exact(dim).map(|r| dot(r, r)));
        }
        for _ in ids {
            self.edges.push(&[], params.degree);
            self.first_child.push(NONE);
            self.next_sibling.push(NONE);
        }
        let order: Vec<u32> = (first as u32..self.ids.len() as u32).collect();
        let rows = Rows {
            data: &self.vectors,
            dim,
            norms: norms_opt(&self.norms),
            codes: None,
            quantized: false,
        };
        rows.insert_pass(
            &mut self.edges,
            &order,
            0,
            params.build_ef,
            params.alpha,
            params.degree,
        );
        for &slot in order.iter().filter(|&&s| s > 0) {
            // A parent inserted earlier keeps the routing tree acyclic and rooted at slot 0.
            let parent = self.edges[slot as usize]
                .iter()
                .copied()
                .find(|&u| u < slot)
                .unwrap_or(0) as usize;
            self.next_sibling[slot as usize] = self.first_child[parent];
            self.first_child[parent] = slot;
        }
        for (i, &id) in ids.iter().enumerate() {
            self.set_live(id, first + i);
        }
    }

    pub fn search(&self, query: &[f32], ef: usize, metric: Metric) -> Vec<(f32, u32)> {
        self.search_inner::<false>(query, ef, metric).0
    }

    pub fn search_profiled(
        &self,
        query: &[f32],
        ef: usize,
        metric: Metric,
    ) -> (Vec<(f32, u32)>, usize) {
        self.search_inner::<true>(query, ef, metric)
    }

    fn search_inner<const PROFILE: bool>(
        &self,
        query: &[f32],
        ef: usize,
        metric: Metric,
    ) -> (Vec<(f32, u32)>, usize) {
        if self.ids.is_empty() {
            return (Vec::new(), 0);
        }
        if ef >= self.len() {
            let mut all: Vec<_> = self
                .live
                .iter()
                .map(|(&id, &slot)| (self.score(query, slot, metric), id))
                .collect();
            all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            return (all, if PROFILE { self.live.len() } else { 0 });
        }
        SEEN.with(|seen| self.search_with::<PROFILE>(&mut seen.borrow_mut(), query, ef, metric))
    }

    fn search_with<const PROFILE: bool>(
        &self,
        seen: &mut BuildVisited,
        query: &[f32],
        ef: usize,
        metric: Metric,
    ) -> (Vec<(f32, u32)>, usize) {
        seen.reset(self.len());
        seen.insert(0);
        let vectors: &[f32] = &self.vectors;
        let score = |slot: usize| self.score_from(vectors, query, slot, metric);
        let mut pool = vec![(score(0), 0usize, false)];
        let mut visits = usize::from(PROFILE);
        let mut fresh = std::mem::take(&mut seen.fresh);
        // Every entry before `cur` is expanded: an insert at or before it moves it back.
        let mut cur = 0;
        loop {
            while cur < pool.len() && pool[cur].2 {
                cur += 1;
            }
            if cur == pool.len() {
                break;
            }
            pool[cur].2 = true;
            let slot = pool[cur].1;
            let children = std::iter::successors(
                (self.first_child[slot] != NONE).then_some(self.first_child[slot]),
                |&v| {
                    (self.next_sibling[v as usize] != NONE).then_some(self.next_sibling[v as usize])
                },
            );
            // Mark and prefetch every new neighbour first so their rows load concurrently.
            fresh.clear();
            for u in self.edges[slot].iter().copied().chain(children) {
                if seen.insert(u) {
                    let at = u as usize * query.len();
                    prefetch_row(&vectors[at..at + query.len()]);
                    fresh.push(u);
                }
            }
            if PROFILE {
                visits += fresh.len();
            }
            for &u in &fresh {
                let u = u as usize;
                let d = score(u);
                if pool.len() == ef {
                    let worst = pool.last().unwrap();
                    if worst.0.total_cmp(&d).then(worst.1.cmp(&u)).is_lt() {
                        continue;
                    }
                }
                let at = pool.partition_point(|p| p.0.total_cmp(&d).then(p.1.cmp(&u)).is_lt());
                if at < ef {
                    pool.insert(at, (d, u, false));
                    pool.truncate(ef);
                    cur = cur.min(at);
                }
            }
        }
        seen.fresh = fresh;
        let found = pool
            .into_iter()
            .filter_map(|(d, slot, _)| self.alive[slot].then_some((d, self.ids[slot])))
            .collect();
        (found, visits)
    }

    pub fn topology(&self) -> std::collections::BTreeMap<String, f64> {
        let n = self.len();
        let live: Vec<bool> = self
            .ids
            .iter()
            .enumerate()
            .map(|(slot, id)| self.live.get(id) == Some(&slot))
            .collect();
        let mut incoming = vec![0u8; n];
        let mut routing_incoming = vec![0u8; n];
        let (mut edges, mut stale) = (0usize, 0usize);
        for slot in 0..n {
            if !live[slot] {
                continue;
            }
            for &target in &self.edges[slot] {
                let target = target as usize;
                assert!(target < n, "delta edge outside allocated slots");
                edges += 1;
                if live[target] {
                    incoming[target] = incoming[target].saturating_add(1).min(2);
                } else {
                    stale += 1;
                }
            }
        }
        routing_incoming.copy_from_slice(&incoming);
        for slot in 0..n {
            if !live[slot] {
                continue;
            }
            let mut child = self.first_child[slot];
            while child != NONE {
                let target = child as usize;
                assert!(target < n && target > slot, "invalid delta routing tree");
                if live[target] && !self.edges[slot].contains(&child) {
                    routing_incoming[target] = routing_incoming[target].saturating_add(1).min(2);
                }
                child = self.next_sibling[target];
            }
        }
        let mut reached = vec![false; n];
        let mut stack = Vec::new();
        if n > 0 {
            reached[0] = true;
            stack.push(0);
        }
        while let Some(slot) = stack.pop() {
            let mut visit = |u: u32| {
                let u = u as usize;
                assert!(u < n, "delta edge outside allocated slots");
                if !reached[u] {
                    reached[u] = true;
                    stack.push(u);
                }
            };
            for &u in &self.edges[slot] {
                visit(u);
            }
            let mut child = self.first_child[slot];
            while child != NONE {
                visit(child);
                child = self.next_sibling[child as usize];
            }
        }
        let count = self.live.len();
        let low = incoming
            .iter()
            .zip(&live)
            .filter(|(degree, live)| **live && **degree <= 1)
            .count();
        let routing_low = routing_incoming
            .iter()
            .zip(&live)
            .filter(|(degree, live)| **live && **degree <= 1)
            .count();
        let reachable = reached
            .iter()
            .zip(&live)
            .filter(|(seen, live)| **seen && **live)
            .count();
        [
            ("delta_slots", n as f64),
            ("delta_live", count as f64),
            ("delta_dead_fraction", (n - count) as f64 / n.max(1) as f64),
            ("delta_edges_from_live", edges as f64),
            (
                "delta_stale_edge_fraction",
                stale as f64 / edges.max(1) as f64,
            ),
            ("delta_d_low", low as f64 / count.max(1) as f64),
            (
                "delta_d_low_with_tree",
                routing_low as f64 / count.max(1) as f64,
            ),
            (
                "delta_live_reachable_fraction",
                reachable as f64 / count.max(1) as f64,
            ),
            ("delta_entry_dead", usize::from(n > 0 && !live[0]) as f64),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::qgraph::{kernels::SplitMix, prepare_rows};

    fn search_by_sorting(
        delta: &Delta,
        query: &[f32],
        ef: usize,
        metric: Metric,
    ) -> (Vec<(f32, u32)>, usize) {
        let mut seen = vec![false; delta.len()];
        seen[0] = true;
        let mut visits = 1;
        let mut pool = vec![(delta.score(query, 0, metric), 0usize, false)];
        while let Some(at) = pool.iter().position(|p| !p.2) {
            pool[at].2 = true;
            let slot = pool[at].1;
            let mut neighbors = delta.edges[slot].to_vec();
            let mut child = delta.first_child[slot];
            while child != NONE {
                neighbors.push(child);
                child = delta.next_sibling[child as usize];
            }
            for slot in neighbors.into_iter().map(|s| s as usize) {
                if !seen[slot] {
                    seen[slot] = true;
                    visits += 1;
                    pool.push((delta.score(query, slot, metric), slot, false));
                }
            }
            pool.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            pool.truncate(ef);
        }
        (
            pool.into_iter()
                .filter_map(|(score, slot, _)| {
                    delta.alive[slot].then_some((score, delta.ids[slot]))
                })
                .collect(),
            visits,
        )
    }

    #[test]
    fn frontier_rejection_preserves_ties_and_deleted_routing() {
        for metric in [Metric::Angular, Metric::Euclidean] {
            for dim in [1, 4, 17] {
                let params = BuildParams {
                    metric,
                    degree: 8,
                    build_ef: 16,
                    ..Default::default()
                };
                let mut random = SplitMix(7654);
                let mut rows: Vec<f32> = (0..65 * dim)
                    .map(|_| (random.next_u64() % 5) as f32 - 2.0)
                    .collect();
                rows[..dim].fill(-0.0);
                if metric == Metric::Angular {
                    for row in rows.chunks_exact_mut(dim) {
                        row[0] = 1.0;
                    }
                }
                prepare_rows(&mut rows, dim, metric);
                let mut delta = Delta::default();
                delta.insert_batch(
                    &(0..64).rev().collect::<Vec<_>>(),
                    &rows[..64 * dim],
                    &params,
                );
                delta.insert(0, &rows[64 * dim..], &params);
                for id in [63, 7, 15, 23, 31] {
                    delta.remove(id);
                }
                for query in rows.chunks_exact(dim).take(8) {
                    for ef in [1, 2, 7, 8, 9, delta.len() - 1] {
                        let expected = search_by_sorting(&delta, query, ef, metric);
                        let found = delta.search_profiled(query, ef, metric);
                        let bits = |results: Vec<(f32, u32)>| {
                            results
                                .into_iter()
                                .map(|(d, id)| (d.to_bits(), id))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(found.1, expected.1, "{metric:?}, dim={dim}, ef={ef}");
                        assert_eq!(
                            bits(found.0),
                            bits(expected.0),
                            "{metric:?}, dim={dim}, ef={ef}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cached_norms_preserve_score_bits_and_delta_results() {
        use std::hash::{Hash, Hasher};
        let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
        for metric in [Metric::Angular, Metric::Euclidean] {
            for dim in [1, 15, 16, 17, 31, 32, 33, 99, 100, 101, 127, 128, 129] {
                let params = BuildParams {
                    metric,
                    degree: 8,
                    build_ef: 16,
                    ..Default::default()
                };
                let mut random = SplitMix(7654);
                let mut rows: Vec<_> = (0..33 * dim)
                    .map(|_| (random.next_u64() >> 40) as f32 / (1u32 << 24) as f32 - 0.5)
                    .collect();
                if metric == Metric::Euclidean {
                    rows[32 * dim..].fill(-0.0);
                }
                prepare_rows(&mut rows, dim, metric);
                let mut delta = Delta::default();
                let ids: Vec<_> = (0..32).rev().collect();
                delta.insert_batch(&ids, &rows[..32 * dim], &params);
                delta.insert(100, &rows[32 * dim..], &params);
                delta.insert(0, &rows[..dim], &params);
                delta.remove(1);
                let zero = vec![0.0; dim];
                let queries = rows
                    .chunks_exact(dim)
                    .take(4)
                    .chain((metric == Metric::Euclidean).then_some(zero.as_slice()));
                for query in queries {
                    for slot in 0..delta.len() {
                        assert_eq!(
                            delta.score(query, slot, metric).to_bits(),
                            super::super::distance(query, delta.row(slot, dim), metric).to_bits(),
                            "{metric:?}, dim={dim}, slot={slot}"
                        );
                    }
                    let mut expected: Vec<_> = delta
                        .live
                        .iter()
                        .map(|(&id, &slot)| {
                            (
                                super::super::distance(query, delta.row(slot, dim), metric),
                                id,
                            )
                        })
                        .collect();
                    expected.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    for ef in [1, 16, delta.len() - 1, delta.len(), delta.len() + 1] {
                        let found = delta.search(query, ef, metric);
                        for &(score, id) in &found {
                            (score.to_bits(), id).hash(&mut fingerprint);
                        }
                        assert_eq!(found, delta.search_profiled(query, ef, metric).0);
                        for &(score, id) in &found {
                            let slot = delta.live[&id];
                            assert_eq!(
                                score.to_bits(),
                                super::super::distance(query, delta.row(slot, dim), metric)
                                    .to_bits()
                            );
                        }
                        if ef >= delta.len() {
                            assert_eq!(found, expected);
                        }
                    }
                }
            }
        }
        println!("search_fingerprint={}", fingerprint.finish());
    }
    #[test]
    #[ignore = "development delta search timing measurement"]
    fn delta_search_live_set() {
        use rayon::prelude::*;
        use std::hash::{Hash, Hasher};
        use std::io::Read;

        let n = crate::compatibility::var("WARM_DELTA_ROWS")
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(100_000);
        assert!((1_000..=3_000_000).contains(&n));
        let dim = 100;
        let read_rows = |path: String, count: usize| {
            let mut file = std::fs::File::open(path).unwrap();
            let mut header = [0u8; 8];
            file.read_exact(&mut header).unwrap();
            assert!(u32::from_le_bytes(header[..4].try_into().unwrap()) as usize >= count);
            assert_eq!(
                u32::from_le_bytes(header[4..].try_into().unwrap()),
                dim as u32
            );
            let mut bytes = vec![0; count * dim * 4];
            file.read_exact(&mut bytes).unwrap();
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        let (vectors, queries) = if let Ok(path) = crate::compatibility::var("WARM_DELTA_DATA") {
            (
                read_rows(path, n),
                read_rows(
                    crate::compatibility::var("WARM_DELTA_QUERIES").unwrap(),
                    256,
                ),
            )
        } else {
            let mut rng = SplitMix(42);
            let mut rows: Vec<f32> = (0..(n + 256) * dim)
                .map(|_| (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32)
                .collect();
            let queries = rows.split_off(n * dim);
            (rows, queries)
        };
        let params = BuildParams {
            metric: Metric::Euclidean,
            degree: 32,
            build_ef: 64,
            ..Default::default()
        };
        let mut delta = Delta::default();
        delta.insert_batch(&(0..n as u32).collect::<Vec<_>>(), &vectors, &params);
        for threads in [1, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for ef in [256, 512] {
                let search = || {
                    pool.install(|| {
                        queries
                            .par_chunks(dim)
                            .map(|q| delta.search(q, ef, params.metric))
                            .collect::<Vec<_>>()
                    })
                };
                std::hint::black_box(search());
                for round in 0..5 {
                    let start = std::time::Instant::now();
                    let results = search();
                    let elapsed = start.elapsed().as_nanos();
                    let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
                    for result in results {
                        for (score, id) in result {
                            (score.to_bits(), id).hash(&mut fingerprint);
                        }
                    }
                    println!(
                        "query_round={round} rows={n} queries=256 threads={threads} ef={ef} elapsed_ns={elapsed} fingerprint={}",
                        fingerprint.finish()
                    );
                }
            }
        }
    }
}
