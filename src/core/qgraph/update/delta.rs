// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::HashMap;

use super::super::{
    adjacency::Adjacency, kernels::dot, norms_opt, BuildParams, BuildVisited, Rows, NONE,
};
use super::Metric;

thread_local! { static SEEN: RefCell<BuildVisited> = RefCell::new(BuildVisited::default()); }

#[derive(Clone, Default)]
pub(super) struct Delta {
    vectors: Vec<f32>,
    norms: Vec<f32>,
    edges: Adjacency,
    first_child: Vec<u32>,
    next_sibling: Vec<u32>,
    ids: Vec<u32>,
    pub live: HashMap<u32, usize>,
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
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn row(&self, slot: usize, dim: usize) -> &[f32] {
        &self.vectors[slot * dim..(slot + 1) * dim]
    }

    pub fn insert(&mut self, id: u32, row: &[f32], params: &BuildParams) {
        let slot = self.ids.len();
        self.ids.push(id);
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
            let candidates = rows.greedy(
                &self.edges,
                0,
                slot as u32,
                params.build_ef,
                &mut BuildVisited::default(),
            );
            let neighbors = rows.prune(slot as u32, candidates, params.alpha, params.degree);
            // A rooted routing tree keeps every slot reachable when reciprocal pruning
            // removes its last incoming graph edge. Tree edges survive logical deletion.
            let parent = neighbors[0] as usize;
            self.next_sibling[slot] = self.first_child[parent];
            self.first_child[parent] = slot as u32;
            for &u in &neighbors {
                let mut candidates: Vec<_> = self.edges[u as usize]
                    .iter()
                    .map(|&v| (rows.d2(u, v), v))
                    .collect();
                candidates.push((rows.d2(u, slot as u32), slot as u32));
                let pruned = rows.prune(u, candidates, params.alpha, params.degree);
                self.edges.set(u as usize, &pruned);
            }
            self.edges.set(slot, &neighbors);
        }
        self.live.insert(id, slot);
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
            self.live.insert(id, first + i);
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
                .map(|(&id, &slot)| {
                    (
                        super::distance(query, self.row(slot, query.len()), metric),
                        id,
                    )
                })
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
        let score = |slot: usize| super::distance(query, self.row(slot, query.len()), metric);
        let mut pool = vec![(score(0), 0usize, false)];
        let mut visits = usize::from(PROFILE);
        while let Some(at) = pool.iter().position(|p| !p.2) {
            pool[at].2 = true;
            let slot = pool[at].1;
            let children = std::iter::successors(
                (self.first_child[slot] != NONE).then_some(self.first_child[slot]),
                |&v| {
                    (self.next_sibling[v as usize] != NONE).then_some(self.next_sibling[v as usize])
                },
            );
            for u in self.edges[slot].iter().copied().chain(children) {
                if !seen.insert(u) {
                    continue;
                }
                let u = u as usize;
                if PROFILE {
                    visits += 1;
                }
                let d = score(u);
                let at = pool.partition_point(|p| p.0.total_cmp(&d).then(p.1.cmp(&u)).is_lt());
                if at < ef {
                    pool.insert(at, (d, u, false));
                    pool.truncate(ef);
                }
            }
        }
        let found = pool
            .into_iter()
            .filter_map(|(d, slot, _)| {
                let id = self.ids[slot];
                (self.live.get(&id) == Some(&slot)).then_some((d, id))
            })
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
