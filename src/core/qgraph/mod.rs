// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quantized graph index for approximate nearest-neighbour search on angular data.
//!
//! Vectors are normalized and rotated (random sign flips and fast Walsh-Hadamard transforms),
//! and a Vamana-style graph with alpha pruning and a fixed out-degree is built on exact inner
//! products, inserting in deterministic batches. Each vertex owns one contiguous block holding
//! its vector (f32, or i16/i8 with a per-vector scale; see [`Store`]) and its neighbours' ids.
//! With [`Codes::Edge`] the block also holds, for every edge, a 1-bit RaBitQ code of the edge
//! residual (neighbour minus vertex) with per-edge correction factors. With [`Codes::Vertex`]
//! each vertex instead has one code of its residual from the data centroid, kept in a separate
//! compact array that is gathered per neighbour.
//!
//! Search keeps a pool of `ef` candidates ordered by estimated distance. Expanding a vertex
//! scores it from the stored vector (this is the re-rank) and estimates its neighbours from
//! their codes against a 4-bit query code. The result is the top-k of the scored vertices.
//! `max_exact` caps the number of expansions (0 = no cap).
//!
//! Estimator (RaBitQ, Gao & Long 2024, applied per edge as in SymphonyQG, Gou et al. 2025):
//! with base point `v`, `w = u - v` and `s = sign(w)`,
//! `<q, u> ~= <q, v> + <v, w> + M (<s, q> - <s, v>)` where `M = |w|^2 / |w|_1`. The base point
//! is the expanded vertex (edge codes; `<q, v>` is its score) or the data centroid (vertex
//! codes; `<q, c>` is computed once per query). `<s, q>` comes from the bit planes.

mod kernels;
mod persist;

use std::cell::RefCell;

use rayon::prelude::*;

use kernels::{
    as_elems, as_elems_mut, as_f32, as_f32_mut, as_u32, as_u32_mut, dot, dot_elem, Elem, QueryCode,
    Rotation, SplitMix,
};

/// Default out-degree of every vertex; one batch of edge codes per expansion.
pub const DEFAULT_DEGREE: usize = 32;
const NONE: u32 = u32::MAX;
const UPPER_DEGREE: usize = 16;
const LAYER_RATIO: usize = 16;
const MIN_LAYER: usize = 64;

#[derive(Clone, Debug)]
pub struct BuildParams {
    /// Candidate list size of the construction-time greedy search.
    pub build_ef: usize,
    /// Out-degree of the bottom layer (even).
    pub degree: usize,
    /// Pruning parameter of the second pass (the first pass uses 1.0).
    pub alpha: f32,
    /// Length of the 1-bit edge codes; rounded up to a power of two of at least 64 and at
    /// least the dimension. 0 picks the smallest such length.
    pub code_bits: usize,
    pub seed: u64,
    /// Element type of the stored vectors.
    pub store: Store,
    /// Placement of the 1-bit codes.
    pub codes: Codes,
}

/// Element type of the stored vectors used for scoring expanded vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Store {
    F32,
    /// Signed 16-bit integers with one f32 scale per vector.
    I16,
    /// Signed 8-bit integers with one f32 scale per vector.
    I8,
}

/// Placement of the 1-bit RaBitQ codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codes {
    /// One code per edge (neighbour minus vertex), inside the vertex block.
    Edge,
    /// One code per vertex (vertex minus centroid), in a separate compact array.
    Vertex,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            build_ef: 128,
            degree: DEFAULT_DEGREE,
            alpha: 1.0,
            code_bits: 0,
            seed: 0x5EED,
            store: Store::F32,
            codes: Codes::Edge,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    /// Candidate pool size (beam width).
    pub ef: usize,
    /// Maximum exactly scored vertices per query; 0 means no cap.
    pub max_exact: usize,
}

/// Word offsets inside a vertex block.
struct Layout {
    vec_words: usize,
    codes_off: usize,
    edge_words: usize,
    factors_off: usize,
    ids_off: usize,
    stride: usize,
}

impl Layout {
    fn new(dim: usize, words: usize, degree: usize, store: Store, codes: Codes) -> Self {
        let (header, elem_bytes) = match store {
            Store::F32 => (0, 4),
            Store::I16 => (1, 2),
            Store::I8 => (1, 1),
        };
        let vec_words = header + (dim * elem_bytes).div_ceil(8);
        let codes_off = vec_words;
        let edge_words = match codes {
            Codes::Edge => degree * words,
            Codes::Vertex => 0,
        };
        let factors_off = codes_off + edge_words;
        let ids_off = factors_off + (3 * edge_words / words).div_ceil(2);
        Self {
            vec_words,
            codes_off,
            edge_words,
            factors_off,
            ids_off,
            stride: ids_off + degree.div_ceil(2),
        }
    }
}

pub struct QGraph {
    n: usize,
    dim: usize,
    degree: usize,
    words: usize,
    rotation: Rotation,
    store: Store,
    codes: Codes,
    /// Words of the stored vector at the start of each block (scale word first if integer).
    vec_words: usize,
    stride: usize,
    codes_off: usize,
    factors_off: usize,
    ids_off: usize,
    blocks: Vec<u64>,
    /// Vertex codes (`Codes::Vertex`): per vertex `words` code words, then `K, M, popcount`
    /// as f32 in two words. Empty for edge codes.
    vcodes: Vec<u64>,
    /// Rotated data centroid, the base point of the vertex codes.
    centroid: Vec<f32>,
    entry: u32,
    /// Global ids of the vertices in the upper layers, by local id.
    upper_ids: Vec<u32>,
    /// Upper-layer adjacency in local ids, largest layer first.
    layers: Vec<Vec<Vec<u32>>>,
}

#[derive(Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u32,
    done: bool,
}

struct Visited {
    marks: Vec<u32>,
    epoch: u32,
}

impl Visited {
    fn new(n: usize) -> Self {
        Self {
            marks: vec![0; n],
            epoch: 0,
        }
    }

    fn next(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// True if `id` was not yet visited in this epoch.
    fn insert(&mut self, id: u32) -> bool {
        let m = &mut self.marks[id as usize];
        let fresh = *m != self.epoch;
        *m = self.epoch;
        fresh
    }
}

thread_local! {
    static BUILD_VISITED: RefCell<Visited> = RefCell::new(Visited::new(0));
}

/// Squared Euclidean distance between unit vectors.
fn d2(a: &[f32], b: &[f32]) -> f32 {
    (2.0 - 2.0 * dot(a, b)).max(0.0)
}

struct Rows<'a> {
    data: &'a [f32],
    dim: usize,
}

impl Rows<'_> {
    fn row(&self, i: u32) -> &[f32] {
        let i = i as usize;
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    fn d2(&self, a: u32, b: u32) -> f32 {
        d2(self.row(a), self.row(b))
    }

    /// Greedy search from `start`; returns every expanded vertex with its distance to `target`.
    fn greedy(
        &self,
        graph: &[Vec<u32>],
        start: u32,
        target: &[f32],
        l: usize,
        visited: &mut Visited,
    ) -> Vec<(f32, u32)> {
        visited.next();
        visited.insert(start);
        let mut pool = vec![Candidate {
            dist: d2(target, self.row(start)),
            id: start,
            done: false,
        }];
        let mut expanded = Vec::new();
        let mut cur = 0;
        while cur < pool.len() {
            pool[cur].done = true;
            let v = pool[cur].id;
            expanded.push((pool[cur].dist, v));
            let mut best = usize::MAX;
            for &u in &graph[v as usize] {
                if !visited.insert(u) {
                    continue;
                }
                let du = d2(target, self.row(u));
                if pool.len() >= l && du >= pool[l - 1].dist {
                    continue;
                }
                let pos = pool.partition_point(|c| c.dist <= du);
                pool.insert(
                    pos,
                    Candidate {
                        dist: du,
                        id: u,
                        done: false,
                    },
                );
                pool.truncate(l);
                best = best.min(pos);
            }
            cur = best.min(cur + 1);
            while cur < pool.len() && pool[cur].done {
                cur += 1;
            }
        }
        expanded
    }

    /// Vamana robust prune: keep `c` unless an already kept `s` has `alpha^2 d2(s, c) <= d2(p, c)`.
    fn prune(&self, p: u32, mut cands: Vec<(f32, u32)>, alpha: f32, r: usize) -> Vec<u32> {
        cands.retain(|c| c.1 != p);
        cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        cands.dedup_by_key(|c| c.1);
        let a2 = alpha * alpha;
        let mut out: Vec<u32> = Vec::with_capacity(r);
        for &(dc, c) in &cands {
            if out.len() == r {
                break;
            }
            if out.iter().all(|&s| a2 * self.d2(s, c) > dc) {
                out.push(c);
            }
        }
        out
    }

    /// One insertion pass in batches of doubling size; each batch searches a frozen graph, so
    /// the result does not depend on thread scheduling.
    #[allow(clippy::too_many_arguments)]
    fn insert_pass(
        &self,
        graph: &mut [Vec<u32>],
        order: &[u32],
        entry: u32,
        l: usize,
        alpha: f32,
        r: usize,
    ) {
        let n = graph.len();
        let max_batch = (n / 50).max(64);
        let (mut start, mut size) = (0, 1);
        while start < order.len() {
            let end = (start + size).min(order.len());
            let batch = &order[start..end];
            let g: &[Vec<u32>] = graph;
            let outs: Vec<Vec<u32>> = batch
                .par_iter()
                .map(|&p| {
                    let mut cands = BUILD_VISITED.with(|cell| {
                        let mut vis = cell.borrow_mut();
                        if vis.marks.len() != n {
                            *vis = Visited::new(n);
                        }
                        self.greedy(g, entry, self.row(p), l, &mut vis)
                    });
                    cands.extend(g[p as usize].iter().map(|&u| (self.d2(p, u), u)));
                    self.prune(p, cands, alpha, r)
                })
                .collect();
            let mut rev: Vec<(u32, u32)> = batch
                .iter()
                .zip(&outs)
                .flat_map(|(&p, out)| out.iter().map(move |&u| (u, p)))
                .collect();
            for (&p, out) in batch.iter().zip(outs) {
                graph[p as usize] = out;
            }
            rev.sort_unstable();
            let groups: Vec<&[(u32, u32)]> = rev.chunk_by(|a, b| a.0 == b.0).collect();
            let g: &[Vec<u32>] = graph;
            let updates: Vec<(u32, Vec<u32>)> = groups
                .par_iter()
                .map(|grp| {
                    let t = grp[0].0;
                    let mut merged = g[t as usize].clone();
                    for &(_, s) in grp.iter() {
                        if !merged.contains(&s) {
                            merged.push(s);
                        }
                    }
                    if merged.len() > r {
                        let cands = merged.iter().map(|&u| (self.d2(t, u), u)).collect();
                        merged = self.prune(t, cands, alpha, r);
                    }
                    (t, merged)
                })
                .collect();
            for (t, m) in updates {
                graph[t as usize] = m;
            }
            start = end;
            size = (size * 2).min(max_batch);
        }
    }

    /// Fill every adjacency list up to `r` with the nearest two-hop neighbours; the
    /// search estimates a full batch per expansion anyway, so the extra edges are free.
    fn fill(&self, graph: &mut [Vec<u32>], r: usize) {
        let g: &[Vec<u32>] = graph;
        let extra: Vec<Vec<u32>> = (0..g.len())
            .into_par_iter()
            .map(|v| {
                let have = &g[v];
                if have.len() >= r {
                    return Vec::new();
                }
                let mut ids: Vec<u32> = have
                    .iter()
                    .flat_map(|&u| g[u as usize].iter().copied())
                    .filter(|&u| u as usize != v && !have.contains(&u))
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                let mut c: Vec<(f32, u32)> =
                    ids.iter().map(|&u| (self.d2(v as u32, u), u)).collect();
                c.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                c.into_iter().take(r - have.len()).map(|x| x.1).collect()
            })
            .collect();
        for (adj, e) in graph.iter_mut().zip(extra) {
            adj.extend(e);
        }
    }
}

/// Per-edge code and factors for the edge v -> u (rotated unit vectors).
/// Returns `(K, M, popcount)` with `K = <v, w> - M <s, v>`.
fn edge_factors(pv: &[f32], pu: &[f32], code: &mut [u64]) -> (f32, f32, f32) {
    code.fill(0);
    let (mut r2, mut l1, mut sv, mut vw) = (0f32, 0f32, 0f32, 0f32);
    for (i, (&a, &b)) in pv.iter().zip(pu).enumerate() {
        let w = b - a;
        r2 += w * w;
        l1 += w.abs();
        vw += a * w;
        if w > 0.0 {
            code[i / 64] |= 1 << (i % 64);
            sv += a;
        } else {
            sv -= a;
        }
    }
    let m = if l1 > 0.0 { r2 / l1 } else { 0.0 };
    let pop = code.iter().map(|c| c.count_ones()).sum::<u32>() as f32;
    (vw - m * sv, m, pop)
}

impl QGraph {
    /// Build over `data` (row-major, `dim` floats per vector). Rows are normalized; zero rows
    /// stay zero. Uses the rayon pool; the result depends only on the data and `params`.
    pub fn build(data: &[f32], dim: usize, params: &BuildParams) -> Self {
        assert!(dim > 0 && data.len().is_multiple_of(dim), "ragged data");
        let n = data.len() / dim;
        assert!(n > 0 && n < NONE as usize, "index size out of range");
        assert!(params.build_ef > 0, "build_ef must be positive");
        let r = params.degree;
        assert!(
            r > 0 && r.is_multiple_of(2),
            "degree must be positive and even"
        );
        let rotation = Rotation::new(dim, params.code_bits, params.seed);
        let words = rotation.padded() / 64;
        let mut unit = data.to_vec();
        unit.par_chunks_mut(dim).for_each(|x| {
            let norm = dot(x, x).sqrt();
            if norm > 0.0 {
                x.iter_mut().for_each(|o| *o /= norm);
            }
        });
        let rows = Rows { data: &unit, dim };

        let mut centroid = vec![0f32; dim];
        for x in unit.chunks_exact(dim) {
            centroid.iter_mut().zip(x).for_each(|(c, v)| *c += v);
        }
        let entry = (0..n as u32)
            .into_par_iter()
            .map(|i| (dot(rows.row(i), &centroid), i))
            .reduce(
                || (f32::NEG_INFINITY, NONE),
                |a, b| {
                    if b.0 > a.0 || (b.0 == a.0 && b.1 < a.1) {
                        b
                    } else {
                        a
                    }
                },
            )
            .1;

        let mut rng = SplitMix(params.seed ^ 0xA5A5_A5A5);
        let mut order: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let mut graph: Vec<Vec<u32>> = vec![Vec::new(); n];
        for alpha in [1.0, params.alpha] {
            rows.insert_pass(&mut graph, &order, entry, params.build_ef, alpha, r);
        }
        rows.fill(&mut graph, r);

        // Upper layers (HNSW-like) give each query a nearby entry vertex: layer j holds the
        // first n / 16^(j+1) vertices of `order`, so a vertex keeps its local id in every layer.
        let mut layers = Vec::new();
        let mut size = n / LAYER_RATIO;
        while size >= MIN_LAYER {
            let sub: Vec<f32> = order[..size]
                .iter()
                .flat_map(|&g| rows.row(g).iter().copied())
                .collect();
            let sub_rows = Rows { data: &sub, dim };
            let local: Vec<u32> = (0..size as u32).collect();
            let mut g: Vec<Vec<u32>> = vec![Vec::new(); size];
            sub_rows.insert_pass(
                &mut g,
                &local,
                0,
                params.build_ef,
                params.alpha,
                UPPER_DEGREE,
            );
            layers.push(g);
            size /= LAYER_RATIO;
        }
        let upper_ids = order[..n / LAYER_RATIO].to_vec();

        let Layout {
            vec_words,
            codes_off,
            edge_words,
            factors_off,
            ids_off,
            stride,
        } = Layout::new(dim, words, r, params.store, params.codes);
        let mut blocks = vec![0u64; n * stride];
        blocks
            .par_chunks_mut(stride)
            .zip(graph.par_iter())
            .enumerate()
            .for_each(|(v, (b, adj))| {
                let (vec_w, rest) = b.split_at_mut(codes_off);
                let (codes, rest) = rest.split_at_mut(edge_words);
                let (fac, ids) = rest.split_at_mut(ids_off - factors_off);
                let row = rows.row(v as u32);
                match params.store {
                    Store::F32 => store_vector::<f32>(row, vec_w),
                    Store::I16 => store_vector::<i16>(row, vec_w),
                    Store::I8 => store_vector::<i8>(row, vec_w),
                }
                let fac = as_f32_mut(fac);
                let ids = as_u32_mut(ids);
                ids.fill(NONE);
                ids[..adj.len()].copy_from_slice(adj);
                if params.codes == Codes::Vertex {
                    return;
                }
                let mut pv = vec![0f32; rotation.padded()];
                let mut pu = pv.clone();
                rotation.apply(row, &mut pv);
                for (j, &u) in adj.iter().enumerate() {
                    rotation.apply(rows.row(u), &mut pu);
                    let (k, m, pop) =
                        edge_factors(&pv, &pu, &mut codes[j * words..(j + 1) * words]);
                    fac[j] = k;
                    fac[r + j] = m;
                    fac[2 * r + j] = pop;
                }
            });

        let (vcodes, centroid) = match params.codes {
            Codes::Edge => (Vec::new(), Vec::new()),
            Codes::Vertex => {
                let mean: Vec<f32> = centroid.iter().map(|c| c / n as f32).collect();
                let mut pc = vec![0f32; rotation.padded()];
                rotation.apply(&mean, &mut pc);
                let vstride = words + 2;
                let mut vcodes = vec![0u64; n * vstride];
                vcodes
                    .par_chunks_mut(vstride)
                    .enumerate()
                    .for_each(|(v, rec)| {
                        let mut pu = vec![0f32; rotation.padded()];
                        rotation.apply(rows.row(v as u32), &mut pu);
                        let (code, fac) = rec.split_at_mut(words);
                        let (k, m, pop) = edge_factors(&pc, &pu, code);
                        as_f32_mut(fac)[..3].copy_from_slice(&[k, m, pop]);
                    });
                (vcodes, pc)
            }
        };
        Self {
            n,
            dim,
            degree: r,
            words,
            rotation,
            store: params.store,
            codes: params.codes,
            vec_words,
            stride,
            codes_off,
            factors_off,
            ids_off,
            blocks,
            vcodes,
            centroid,
            entry,
            upper_ids,
            layers,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bytes held by the index (vertex blocks, vertex codes and upper layers; the rotation is
    /// negligible).
    pub fn index_bytes(&self) -> usize {
        let upper: usize = self.layers.iter().flatten().map(|adj| adj.len() * 4).sum();
        (self.blocks.len() + self.vcodes.len()) * 8
            + (self.upper_ids.len() + self.centroid.len()) * 4
            + upper
    }

    /// Inner product of `q` (unrotated, length `dim`) with the vector stored in block `b`.
    #[inline]
    fn ip(&self, q: &[f32], b: &[u64]) -> f32 {
        let w = &b[..self.vec_words];
        match self.store {
            Store::F32 => dot(q, &as_f32(w)[..self.dim]),
            Store::I16 => as_f32(w)[0] * dot_elem(q, &as_elems::<i16>(&w[1..])[..self.dim]),
            Store::I8 => as_f32(w)[0] * dot_elem(q, &as_elems::<i8>(&w[1..])[..self.dim]),
        }
    }

    fn vcode(&self, v: u32) -> &[u64] {
        let s = self.words + 2;
        let v = v as usize;
        &self.vcodes[v * s..(v + 1) * s]
    }

    /// Greedy descent through the upper layers with exact distances; returns the bottom-layer
    /// entry vertex and the number of exact evaluations.
    fn descend(&self, pq: &[f32]) -> (u32, usize) {
        if self.layers.is_empty() {
            return (self.entry, 0);
        }
        let mut p = 0u32;
        let mut best = -self.ip(pq, self.block(self.upper_ids[0]));
        let mut evals = 1;
        for layer in self.layers.iter().rev() {
            loop {
                let mut moved = false;
                for &u in &layer[p as usize] {
                    let d = -self.ip(pq, self.block(self.upper_ids[u as usize]));
                    evals += 1;
                    if d < best {
                        best = d;
                        p = u;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        (self.upper_ids[p as usize], evals)
    }

    pub fn searcher(&self) -> Searcher<'_> {
        Searcher {
            index: self,
            visited: Visited::new(self.n),
            pq: vec![0.0; self.rotation.padded()],
            code: QueryCode::new(self.words),
            raw: vec![0; self.degree],
            est: vec![0.0; self.degree],
            fresh: Vec::with_capacity(self.degree),
            pool: Vec::new(),
            results: Vec::new(),
        }
    }

    fn block(&self, v: u32) -> &[u64] {
        let v = v as usize;
        &self.blocks[v * self.stride..(v + 1) * self.stride]
    }
}

/// Per-thread search state for one index.
pub struct Searcher<'a> {
    index: &'a QGraph,
    visited: Visited,
    pq: Vec<f32>,
    code: QueryCode,
    raw: Vec<u32>,
    est: Vec<f32>,
    /// Unvisited neighbours of the expanded vertex (vertex codes).
    fresh: Vec<u32>,
    pool: Vec<Candidate>,
    results: Vec<(f32, u32)>,
}

impl Searcher<'_> {
    /// Writes the ids of the (approximate) `k` nearest vectors by cosine into `out`, nearest
    /// first, and returns the number of exactly scored vertices.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        params: SearchParams,
        out: &mut Vec<u32>,
    ) -> usize {
        let idx = self.index;
        let ef = params.ef.max(1);
        idx.rotation.apply(query, &mut self.pq);
        self.code.encode(&self.pq);
        let (lo2, d2q, sq) = (2.0 * self.code.lo, 2.0 * self.code.delta, self.code.sum);
        let qc = match idx.codes {
            Codes::Edge => 0.0,
            Codes::Vertex => dot(&self.pq, &idx.centroid),
        };

        self.visited.next();
        let (entry, mut exact) = idx.descend(query);
        let mut expanded = 0;
        self.visited.insert(entry);
        self.pool.clear();
        self.results.clear();
        self.pool.push(Candidate {
            dist: 0.0,
            id: entry,
            done: false,
        });
        let mut cur = 0;
        while cur < self.pool.len() {
            if params.max_exact > 0 && expanded >= params.max_exact {
                break;
            }
            let v = self.pool.remove(cur).id;
            if let Some(next) = self.pool[cur..].iter().find(|c| !c.done) {
                kernels::prefetch(idx.block(next.id));
            }
            let b = idx.block(v);
            let ipv = idx.ip(query, b);
            exact += 1;
            expanded += 1;
            push_top(&mut self.results, k, -ipv, v);
            // The expanded vertex re-enters the pool with its exact distance, so an optimistic
            // estimate cannot hold a pool slot and stall the search.
            let pos = self.pool.partition_point(|c| c.dist <= -ipv);
            if pos < ef {
                self.pool.insert(
                    pos,
                    Candidate {
                        dist: -ipv,
                        id: v,
                        done: true,
                    },
                );
                self.pool.truncate(ef);
            }

            let ids = as_u32(&b[idx.ids_off..]);
            let mut best = usize::MAX;
            match idx.codes {
                Codes::Edge => {
                    kernels::raw_estimates(
                        &b[idx.codes_off..idx.factors_off],
                        &self.code.planes,
                        idx.words,
                        &mut self.raw,
                    );
                    let fac = as_f32(&b[idx.factors_off..idx.ids_off]);
                    let (kf, rest) = fac.split_at(idx.degree);
                    let (mf, pf) = rest.split_at(idx.degree);
                    for ((((e, &kj), &mj), &pj), &rj) in
                        self.est.iter_mut().zip(kf).zip(mf).zip(pf).zip(&self.raw)
                    {
                        *e = -(ipv + kj + mj * (lo2 * pj - sq + d2q * rj as f32));
                    }
                    for (&u, &d) in ids.iter().zip(&self.est) {
                        if u != NONE && self.visited.insert(u) {
                            offer(&mut self.pool, ef, d, u, &mut best);
                        }
                    }
                }
                Codes::Vertex => {
                    self.fresh.clear();
                    for &u in &ids[..idx.degree] {
                        if u != NONE && self.visited.insert(u) {
                            let rec = idx.vcode(u);
                            kernels::prefetch(&rec[..1]);
                            kernels::prefetch(&rec[rec.len() - 1..]);
                            self.fresh.push(u);
                        }
                    }
                    for &u in &self.fresh {
                        let (code, fac) = idx.vcode(u).split_at(idx.words);
                        kernels::raw_estimates(
                            code,
                            &self.code.planes,
                            idx.words,
                            &mut self.raw[..1],
                        );
                        let f = as_f32(fac);
                        let d = -(qc + f[0] + f[1] * (lo2 * f[2] - sq + d2q * self.raw[0] as f32));
                        offer(&mut self.pool, ef, d, u, &mut best);
                    }
                }
            }
            cur = best.min(cur);
            while cur < self.pool.len() && self.pool[cur].done {
                cur += 1;
            }
        }
        out.clear();
        out.extend(self.results.iter().map(|r| r.1));
        exact
    }
}

/// Inserts candidate `u` at estimated distance `d` into the sorted pool of at most `ef`, and
/// lowers `best` to its position.
#[inline]
fn offer(pool: &mut Vec<Candidate>, ef: usize, d: f32, u: u32, best: &mut usize) {
    if pool.len() >= ef && d >= pool[ef - 1].dist {
        return;
    }
    let pos = pool.partition_point(|c| c.dist <= d);
    pool.insert(
        pos,
        Candidate {
            dist: d,
            id: u,
            done: false,
        },
    );
    pool.truncate(ef);
    *best = (*best).min(pos);
}

/// Writes `x` into `w` as `T`: raw f32, or a scale word followed by the scaled integers.
fn store_vector<T: Elem>(x: &[f32], w: &mut [u64]) {
    if T::MAX == 0.0 {
        as_f32_mut(w)[..x.len()].copy_from_slice(x);
        return;
    }
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = if peak > 0.0 { peak / T::MAX } else { 1.0 };
    let (head, rest) = w.split_at_mut(1);
    as_f32_mut(head)[0] = scale;
    for (o, &v) in as_elems_mut::<T>(rest).iter_mut().zip(x) {
        *o = T::from_f32(v / scale);
    }
}

fn push_top(top: &mut Vec<(f32, u32)>, k: usize, dist: f32, id: u32) {
    if top.len() >= k && top.last().is_none_or(|l| dist >= l.0) {
        return;
    }
    let pos = top.partition_point(|c| c.0 <= dist);
    top.insert(pos, (dist, id));
    top.truncate(k);
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::raw_estimates;

    fn gaussian(rng: &mut SplitMix) -> f32 {
        let u = rng.next_f32().max(1e-7);
        let v = rng.next_f32();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }

    fn unit(mut x: Vec<f32>) -> Vec<f32> {
        let n = dot(&x, &x).sqrt();
        x.iter_mut().for_each(|v| *v /= n);
        x
    }

    /// Mean of the shipped per-edge estimator over rotation seeds must match `<q, u>`, for the
    /// automatic code length (128 bits at d = 100) and for 512-bit codes.
    /// The WHT-with-signs rotation is not Haar, so the bound is empirical: 400 seeds put the
    /// standard error near 0.003; tolerances are 0.02 (float query) and 0.03 (4-bit query).
    #[test]
    fn edge_estimator_is_unbiased_over_rotations() {
        let d = 100;
        let mut rng = SplitMix(42);
        for bits in [0, 0, 0, 512, 512, 512] {
            let q = unit((0..d).map(|_| gaussian(&mut rng)).collect());
            let v = unit((0..d).map(|_| gaussian(&mut rng)).collect());
            let u = unit(v.iter().map(|x| x + 0.15 * gaussian(&mut rng)).collect());
            let truth = dot(&q, &u);
            let (mut float_sum, mut quant_sum) = (0f64, 0f64);
            let seeds = 400;
            for seed in 0..seeds {
                let rot = Rotation::new(d, bits, seed);
                let p = rot.padded();
                let (mut pq, mut pv, mut pu) = (vec![0.0; p], vec![0.0; p], vec![0.0; p]);
                rot.apply(&q, &mut pq);
                rot.apply(&v, &mut pv);
                rot.apply(&u, &mut pu);
                let mut code = vec![0u64; p / 64];
                let (k, m, pop) = edge_factors(&pv, &pu, &mut code);
                let ipv = dot(&pq, &pv);
                let sq: f32 = pq
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        if code[i / 64] >> (i % 64) & 1 == 1 {
                            *x
                        } else {
                            -*x
                        }
                    })
                    .sum();
                float_sum += f64::from(ipv + k + m * sq);
                let mut qc = QueryCode::new(p / 64);
                qc.encode(&pq);
                let mut raw = [0u32];
                raw_estimates(&code, &qc.planes, p / 64, &mut raw);
                let est =
                    ipv + k + m * (2.0 * qc.lo * pop - qc.sum + 2.0 * qc.delta * raw[0] as f32);
                quant_sum += f64::from(est);
            }
            let float_mean = float_sum / f64::from(seeds as u32);
            let quant_mean = quant_sum / f64::from(seeds as u32);
            assert!(
                (float_mean - f64::from(truth)).abs() < 0.02,
                "{float_mean} vs {truth}"
            );
            assert!(
                (quant_mean - f64::from(truth)).abs() < 0.03,
                "{quant_mean} vs {truth}"
            );
        }
    }

    fn clustered(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix(seed);
        let centers: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..d).map(|_| gaussian(&mut rng)).collect())
            .collect();
        (0..n)
            .flat_map(|i| {
                let c = &centers[i % centers.len()];
                c.iter()
                    .map(|x| x + 0.4 * gaussian(&mut rng))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn brute_force(data: &[f32], d: usize, q: &[f32], k: usize) -> Vec<u32> {
        let mut s: Vec<(f32, u32)> = data
            .chunks_exact(d)
            .enumerate()
            .map(|(i, x)| (-dot(x, q) / dot(x, x).sqrt(), i as u32))
            .collect();
        s.sort_by(|a, b| a.0.total_cmp(&b.0));
        s.into_iter().take(k).map(|x| x.1).collect()
    }

    /// Regression guard for every store and code placement on a deliberately hard set (24
    /// dimensions of noise per cluster).
    #[test]
    fn recall_on_clustered_data() {
        let d = 24;
        let data = clustered(4000, d, 1);
        let queries = clustered(100, d, 2);
        for store in [Store::F32, Store::I16, Store::I8] {
            for codes in [Codes::Edge, Codes::Vertex] {
                let params = BuildParams {
                    store,
                    codes,
                    ..BuildParams::default()
                };
                let index = QGraph::build(&data, d, &params);
                let mut s = index.searcher();
                let mut out = Vec::new();
                let mut hits = 0;
                for q in queries.chunks_exact(d) {
                    let truth = brute_force(&data, d, q, 10);
                    s.search(
                        q,
                        10,
                        SearchParams {
                            ef: 64,
                            max_exact: 0,
                        },
                        &mut out,
                    );
                    hits += out.iter().filter(|x| truth.contains(x)).count();
                }
                let recall = hits as f64 / 1000.0;
                // Vertex codes are coded against the global centroid and so are noisier.
                let floor = if codes == Codes::Edge { 0.9 } else { 0.8 };
                assert!(recall >= floor, "{store:?} {codes:?} recall {recall}");
            }
        }
    }

    #[test]
    fn build_and_search_are_deterministic() {
        let d = 16;
        let data = clustered(3000, d, 3);
        for (store, codes) in [(Store::F32, Codes::Edge), (Store::I8, Codes::Vertex)] {
            let params = BuildParams {
                seed: 9,
                store,
                codes,
                ..BuildParams::default()
            };
            let a = QGraph::build(&data, d, &params);
            let b = QGraph::build(&data, d, &params);
            assert_eq!(a.entry, b.entry);
            assert!(a.blocks == b.blocks && a.vcodes == b.vcodes);
            let (mut sa, mut sb) = (a.searcher(), b.searcher());
            let (mut oa, mut ob) = (Vec::new(), Vec::new());
            let p = SearchParams {
                ef: 32,
                max_exact: 0,
            };
            for q in data.chunks_exact(d).take(50) {
                sa.search(q, 10, p, &mut oa);
                sb.search(q, 10, p, &mut ob);
                assert_eq!(oa, ob);
            }
        }
    }

    #[test]
    fn exact_budget_caps_work() {
        let d = 16;
        let data = clustered(2000, d, 4);
        let index = QGraph::build(&data, d, &BuildParams::default());
        let mut s = index.searcher();
        let mut out = Vec::new();
        let q = &data[..d];
        let capped = s.search(
            q,
            10,
            SearchParams {
                ef: 128,
                max_exact: 20,
            },
            &mut out,
        );
        assert_eq!(out.len(), 10);
        let free = s.search(
            q,
            10,
            SearchParams {
                ef: 128,
                max_exact: 0,
            },
            &mut out,
        );
        assert!(capped < free);
        assert_eq!(capped - index.descend(q).1, 20);
    }
}
