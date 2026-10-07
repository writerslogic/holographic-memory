// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quantized graph index for approximate nearest-neighbour search on angular data.
//!
//! Vectors are normalized and rotated (random sign flips and fast Walsh-Hadamard transforms),
//! and a Vamana-style graph with alpha pruning and a fixed out-degree is built on exact inner
//! products, inserting in deterministic batches. Each vertex owns one contiguous block holding
//! its rotated vector, its neighbours' ids, and for every edge a 1-bit RaBitQ code of the edge
//! residual (neighbour minus vertex) with per-edge correction factors.
//!
//! Search keeps a pool of `ef` candidates ordered by estimated distance. Expanding a vertex
//! scores it exactly (this is the re-rank) and estimates all of its neighbours in one batch from
//! the edge codes against a 4-bit query code. The result is the top-k of the exactly scored
//! vertices. `max_exact` caps the number of expansions (0 = no cap).
//!
//! Estimator (RaBitQ, Gao & Long 2024, applied per edge as in SymphonyQG, Gou et al. 2025):
//! with `w = u - v` and `s = sign(w)`, `<q, u> ~= <q, v> + <v, w> + M (<s, q> - <s, v>)` where
//! `M = |w|^2 / |w|_1`. `<q, v>` is exact at expansion time; `<s, q>` comes from the bit planes.

mod kernels;

use std::cell::RefCell;

use rayon::prelude::*;

use kernels::{
    as_f32, as_f32_mut, as_u32, as_u32_mut, as_u8, as_u8_mut, dot, fastscan, pack_batch, QueryCode,
    Rotation, SplitMix, BATCH,
};

/// Default out-degree of every vertex; one FastScan batch of edge codes per expansion.
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
    /// Coordinates of the rotated edge codes; rounded up to a power of two of at least 64 and
    /// at least the dimension. 0 picks the smallest such length.
    pub code_bits: usize,
    /// Bits per rotated coordinate of an edge code: 1 (RaBitQ) or 2 (extended RaBitQ, a
    /// tighter estimate for twice the code memory). An edge code takes
    /// `code_bits * edge_bits` bits.
    pub edge_bits: usize,
    pub seed: u64,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            build_ef: 128,
            degree: DEFAULT_DEGREE,
            alpha: 1.0,
            code_bits: 0,
            edge_bits: 1,
            seed: 0x5EED,
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

pub struct QGraph {
    n: usize,
    dim: usize,
    /// Edge slots per vertex: the degree rounded up to whole FastScan batches.
    degree: usize,
    /// Rotated dimension (coordinates per code plane).
    dims: usize,
    /// Code planes per edge (`BuildParams::edge_bits`).
    bits: usize,
    rotation: Rotation,
    stride: usize,
    codes_off: usize,
    factors_off: usize,
    ids_off: usize,
    blocks: Vec<u64>,
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

/// Per-edge code and factors for the edge v -> u (rotated unit vectors), with `bits` bits per
/// coordinate. The residual `w = u - v` is quantized to odd levels `l_i` in
/// `{-(2^bits - 1), .., -1, 1, .., 2^bits - 1}` stored as `c_i = (l_i + 2^bits - 1) / 2`,
/// bit plane `p` in `code[p * words..]`. For 1 bit, `l = sign(w)` (RaBitQ); for 2 bits the
/// `k` largest `|w_i|` get magnitude 3, with `k` maximizing the cosine between `l` and `w`
/// (extended RaBitQ; the optimum is a threshold on `|w_i|` by the rearrangement inequality).
///
/// The estimator is `<q, w> ~= <v, w> + G <l, q - v>` with `G = |w|^2 / <l, w>`.
/// Returns `(K, G, sum_i c_i)` with `K = <v, w> - G <l, v>`.
fn edge_factors(pv: &[f32], pu: &[f32], bits: usize, code: &mut [u64]) -> (f32, f32, f32) {
    let d = pv.len();
    let words = d / 64;
    debug_assert_eq!(code.len(), bits * words);
    code.fill(0);
    // Coordinates at or before `cut` in (|w| descending, index ascending) order get magnitude 3.
    let mut cut: Option<(f32, usize)> = None;
    if bits == 2 {
        let mut mag: Vec<(f32, usize)> = pv
            .iter()
            .zip(pu)
            .enumerate()
            .map(|(i, (a, b))| ((b - a).abs(), i))
            .collect();
        mag.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        let l1: f32 = mag.iter().map(|m| m.0).sum();
        let (mut best, mut best_k, mut top) = (l1 / (d as f32).sqrt(), 0, 0f32);
        for (k, m) in mag.iter().enumerate() {
            top += m.0;
            let cos = (l1 + 2.0 * top) / ((d + 8 * (k + 1)) as f32).sqrt();
            if cos > best {
                (best, best_k) = (cos, k + 1);
            }
        }
        cut = best_k.checked_sub(1).map(|k| mag[k]);
    }
    let offset = (1u32 << bits) - 1;
    let (mut r2, mut lw, mut lv, mut vw, mut sum) = (0f32, 0f32, 0f32, 0f32, 0u32);
    for (i, (&a, &b)) in pv.iter().zip(pu).enumerate() {
        let w = b - a;
        r2 += w * w;
        vw += a * w;
        let big = cut.is_some_and(|(m, j)| w.abs() > m || (w.abs() == m && i <= j));
        let level: i32 = if big { 3 } else { 1 };
        let level = if w > 0.0 { level } else { -level };
        let l = level as f32;
        lw += l * w;
        lv += l * a;
        let c = (level + offset as i32) as u32 / 2;
        sum += c;
        for p in 0..bits {
            code[p * words + i / 64] |= u64::from((c >> p) & 1) << (i % 64);
        }
    }
    let g = if lw > 0.0 { r2 / lw } else { 0.0 };
    (vw - g * lv, g, sum as f32)
}

fn normalized(data: &[f32], dim: usize) -> Vec<f32> {
    let mut unit = data.to_vec();
    unit.par_chunks_mut(dim).for_each(|x| {
        let norm = dot(x, x).sqrt();
        if norm > 0.0 {
            x.iter_mut().for_each(|o| *o /= norm);
        }
    });
    unit
}

/// FNV-1a over the bit patterns of `data`; identifies the data a topology was built on.
fn data_hash(data: &[f32]) -> u64 {
    data.iter().fold(0xCBF2_9CE4_8422_2325, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01B3)
    })
}

/// The graph of a [`QGraph`] without its codes: the expensive part of a build. Encoding a
/// topology with different code parameters is cheap, which is what makes code and layout
/// experiments affordable.
pub struct Topology {
    n: usize,
    dim: usize,
    build_ef: usize,
    degree: usize,
    alpha: f32,
    seed: u64,
    hash: u64,
    entry: u32,
    /// Insertion order; the first `n / 16` vertices form the upper layers.
    order: Vec<u32>,
    adj: Vec<Vec<u32>>,
    layers: Vec<Vec<Vec<u32>>>,
}

const TOPOLOGY_MAGIC: u64 = 0x4851_4754_4F50_0001;

impl Topology {
    /// Builds the graph over `data` (row-major, `dim` floats per vector); see [`QGraph::build`].
    pub fn build(data: &[f32], dim: usize, params: &BuildParams) -> Self {
        assert!(dim > 0 && data.len().is_multiple_of(dim), "ragged data");
        let unit = normalized(data, dim);
        Self::build_unit(&unit, dim, params, data_hash(data))
    }

    fn build_unit(unit: &[f32], dim: usize, params: &BuildParams, hash: u64) -> Self {
        let n = unit.len() / dim;
        assert!(n > 0 && n < NONE as usize, "index size out of range");
        assert!(params.build_ef > 0, "build_ef must be positive");
        let r = params.degree;
        assert!(
            r > 0 && r.is_multiple_of(2),
            "degree must be positive and even"
        );
        let rows = Rows { data: unit, dim };

        let mut centroid = vec![0f32; dim];
        for x in rows.data.chunks_exact(dim) {
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
        Self {
            n,
            dim,
            build_ef: params.build_ef,
            degree: r,
            alpha: params.alpha,
            seed: params.seed,
            hash,
            entry,
            order,
            adj: graph,
            layers,
        }
    }

    /// Panics unless this topology was built on `data` with the graph fields of `params`.
    fn check(&self, data: &[f32], dim: usize, params: &BuildParams) {
        assert_eq!(
            (self.dim, self.n * self.dim, self.hash),
            (dim, data.len(), data_hash(data)),
            "topology was built on other data"
        );
        assert!(
            self.build_ef == params.build_ef
                && self.degree == params.degree
                && self.alpha.to_bits() == params.alpha.to_bits()
                && self.seed == params.seed,
            "topology was built with other parameters"
        );
    }

    pub fn write(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        let mut put = |x: u64| w.write_all(&x.to_le_bytes());
        put(TOPOLOGY_MAGIC)?;
        for x in [self.n, self.dim, self.build_ef, self.degree] {
            put(x as u64)?;
        }
        put(u64::from(self.alpha.to_bits()))?;
        put(self.seed)?;
        put(self.hash)?;
        put(u64::from(self.entry))?;
        put(self.layers.len() as u64)?;
        let mut buf: Vec<u8> = Vec::new();
        let mut list = |l: &[u32]| {
            buf.extend_from_slice(&(l.len() as u32).to_le_bytes());
            l.iter()
                .for_each(|x| buf.extend_from_slice(&x.to_le_bytes()));
        };
        list(&self.order);
        self.adj.iter().for_each(|a| list(a));
        for layer in &self.layers {
            list(&[layer.len() as u32]);
            layer.iter().for_each(|a| list(a));
        }
        w.write_all(&buf)
    }

    pub fn read(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};
        let bad = |what: &str| Error::new(ErrorKind::InvalidData, format!("topology: {what}"));
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        let mut pos = 0usize;
        let mut take = |len: usize| -> std::io::Result<&[u8]> {
            let s = buf.get(pos..pos + len).ok_or_else(|| bad("truncated"))?;
            pos += len;
            Ok(s)
        };
        let mut u64s = [0u64; 10];
        for x in &mut u64s {
            *x = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        }
        let [magic, n, dim, build_ef, degree, alpha, seed, hash, entry, n_layers] = u64s;
        if magic != TOPOLOGY_MAGIC {
            return Err(bad("bad magic"));
        }
        let mut list = |limit: usize| -> std::io::Result<Vec<u32>> {
            let len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            if len > limit {
                return Err(bad("list too long"));
            }
            Ok(take(4 * len)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect())
        };
        let (n, degree) = (n as usize, degree as usize);
        let order = list(n)?;
        let adj = (0..n)
            .map(|_| list(degree))
            .collect::<Result<Vec<_>, _>>()?;
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let size = *list(1)?.first().ok_or_else(|| bad("layer size"))? as usize;
            layers.push(
                (0..size)
                    .map(|_| list(UPPER_DEGREE))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        let in_range = |l: &[u32], m: usize| l.iter().all(|&x| (x as usize) < m);
        let ok = order.len() == n
            && (entry as usize) < n
            && in_range(&order, n)
            && adj.iter().all(|a| in_range(a, n))
            && layers
                .iter()
                .all(|g| g.iter().all(|a| in_range(a, g.len())))
            && layers.first().is_none_or(|g| g.len() <= n / LAYER_RATIO);
        if !ok {
            return Err(bad("ids out of range"));
        }
        Ok(Self {
            n,
            dim: dim as usize,
            build_ef: build_ef as usize,
            degree,
            alpha: f32::from_bits(alpha as u32),
            seed,
            hash,
            entry: entry as u32,
            order,
            adj,
            layers,
        })
    }
}

impl QGraph {
    /// Build over `data` (row-major, `dim` floats per vector). Rows are normalized; zero rows
    /// stay zero. Uses the rayon pool; the result depends only on the data and `params`.
    pub fn build(data: &[f32], dim: usize, params: &BuildParams) -> Self {
        assert!(dim > 0 && data.len().is_multiple_of(dim), "ragged data");
        let unit = normalized(data, dim);
        let topo = Topology::build_unit(&unit, dim, params, data_hash(data));
        Self::encode(&unit, &topo, params)
    }

    /// Encodes a prebuilt topology; `data` and the graph fields of `params` must be the ones
    /// it was built with (checked).
    pub fn from_topology(data: &[f32], dim: usize, topo: &Topology, params: &BuildParams) -> Self {
        topo.check(data, dim, params);
        Self::encode(&normalized(data, dim), topo, params)
    }

    fn encode(unit: &[f32], topo: &Topology, params: &BuildParams) -> Self {
        let (n, dim) = (topo.n, topo.dim);
        let bits = params.edge_bits;
        assert!(bits == 1 || bits == 2, "edge_bits must be 1 or 2");
        // Edge slots are stored in FastScan batches of 32; unused slots have id NONE.
        let r = topo.degree.next_multiple_of(BATCH);
        let rows = Rows { data: unit, dim };
        let rotation = Rotation::new(dim, params.code_bits, params.seed);
        let dims = rotation.padded();
        let words = dims / 64;
        let mut rotated = vec![0f32; n * dims];
        rotated
            .par_chunks_mut(dims)
            .enumerate()
            .for_each(|(v, out)| rotation.apply(rows.row(v as u32), out));

        let codes_off = dim.div_ceil(2);
        let factors_off = codes_off + r * bits * words;
        let ids_off = factors_off + 3 * r / 2;
        let stride = ids_off + r / 2;
        let mut blocks = vec![0u64; n * stride];
        blocks
            .par_chunks_mut(stride)
            .zip(topo.adj.par_iter())
            .enumerate()
            .for_each(|(v, (b, adj))| {
                let (vec_w, rest) = b.split_at_mut(codes_off);
                let (codes, rest) = rest.split_at_mut(r * bits * words);
                let (fac, ids) = rest.split_at_mut(3 * r / 2);
                as_f32_mut(vec_w)[..dim].copy_from_slice(rows.row(v as u32));
                let fac = as_f32_mut(fac);
                let ids = as_u32_mut(ids);
                ids.fill(NONE);
                let pv = &rotated[v * dims..(v + 1) * dims];
                // Plain codes, `bits` planes of `words` per edge, packed per batch below.
                let mut plain = vec![0u64; r * bits * words];
                for (j, &u) in adj.iter().enumerate() {
                    ids[j] = u;
                    let pu = &rotated[u as usize * dims..(u as usize + 1) * dims];
                    let code = &mut plain[j * bits * words..(j + 1) * bits * words];
                    let (k, g, sum) = edge_factors(pv, pu, bits, code);
                    fac[j] = k;
                    fac[r + j] = g;
                    fac[2 * r + j] = sum;
                }
                let packed = as_u8_mut(codes);
                for (batch, out) in packed.chunks_exact_mut(BATCH * bits * dims / 8).enumerate() {
                    for (p, plane) in out.chunks_exact_mut(4 * dims).enumerate() {
                        let edges: Vec<&[u64]> = (batch * BATCH..(batch + 1) * BATCH)
                            .map(|j| &plain[(j * bits + p) * words..(j * bits + p + 1) * words])
                            .collect();
                        pack_batch(&edges, dims, plane);
                    }
                }
            });
        Self {
            n,
            dim,
            degree: r,
            dims,
            bits,
            rotation,
            stride,
            codes_off,
            factors_off,
            ids_off,
            blocks,
            entry: topo.entry,
            upper_ids: topo.order[..n / LAYER_RATIO].to_vec(),
            layers: topo.layers.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bytes held by the index (vertex blocks; the rotation is negligible).
    pub fn index_bytes(&self) -> usize {
        let upper: usize = self.layers.iter().flatten().map(|adj| adj.len() * 4).sum();
        self.blocks.len() * 8 + self.upper_ids.len() * 4 + upper
    }

    fn vector(&self, v: u32) -> &[f32] {
        &as_f32(&self.block(v)[..self.codes_off])[..self.dim]
    }

    /// Greedy descent through the upper layers with exact distances; returns the bottom-layer
    /// entry vertex and the number of exact evaluations.
    fn descend(&self, pq: &[f32]) -> (u32, usize) {
        if self.layers.is_empty() {
            return (self.entry, 0);
        }
        let mut p = 0u32;
        let mut best = -dot(pq, self.vector(self.upper_ids[0]));
        let mut evals = 1;
        for layer in self.layers.iter().rev() {
            loop {
                let mut moved = false;
                for &u in &layer[p as usize] {
                    let d = -dot(pq, self.vector(self.upper_ids[u as usize]));
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
            code: QueryCode::new(self.dims),
            raw: vec![0; self.degree],
            est: vec![0.0; self.degree],
            pool: Vec::new(),
            fresh: Vec::new(),
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
    pool: Vec<Candidate>,
    /// Accepted neighbours of the current expansion, sorted by estimated distance.
    fresh: Vec<Candidate>,
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
        let (lo2, d2q) = (2.0 * self.code.lo, 2.0 * self.code.delta);
        let sq = self.code.sum * ((1 << idx.bits) - 1) as f32;
        let plane_bytes = 4 * idx.dims;

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
            let v = self.pool[cur].id;
            if let Some(next) = self.pool[cur + 1..].iter().find(|c| !c.done) {
                kernels::prefetch(idx.block(next.id));
            }
            let b = idx.block(v);
            let ipv = dot(query, &as_f32(&b[..idx.codes_off])[..idx.dim]);
            exact += 1;
            expanded += 1;
            push_top(&mut self.results, k, -ipv, v);
            // The expanded vertex moves to its exact distance, so an optimistic estimate cannot
            // hold a pool slot and stall the search.
            reposition(&mut self.pool, cur, -ipv);

            let packed = as_u8(&b[idx.codes_off..idx.factors_off]);
            for (batch, raw) in packed
                .chunks_exact(idx.bits * plane_bytes)
                .zip(self.raw.as_chunks_mut::<BATCH>().0)
            {
                fastscan(batch, &self.code.lut, raw);
            }
            let fac = as_f32(&b[idx.factors_off..idx.ids_off]);
            let (kf, rest) = fac.split_at(idx.degree);
            let (mf, pf) = rest.split_at(idx.degree);
            for ((((e, &kj), &mj), &pj), &rj) in
                self.est.iter_mut().zip(kf).zip(mf).zip(pf).zip(&self.raw)
            {
                *e = -(ipv + kj + mj * (lo2 * pj - sq + d2q * rj as f32));
            }

            let bound = if self.pool.len() >= ef {
                self.pool[ef - 1].dist
            } else {
                f32::INFINITY
            };
            self.fresh.clear();
            for (&u, &d) in as_u32(&b[idx.ids_off..]).iter().zip(&self.est) {
                if u == NONE || !self.visited.insert(u) || d >= bound {
                    continue;
                }
                let pos = self.fresh.partition_point(|c| c.dist <= d);
                self.fresh.insert(
                    pos,
                    Candidate {
                        dist: d,
                        id: u,
                        done: false,
                    },
                );
            }
            cur = cur.min(merge(&mut self.pool, &self.fresh, ef));
            while cur < self.pool.len() && self.pool[cur].done {
                cur += 1;
            }
        }
        out.clear();
        out.extend(self.results.iter().map(|r| r.1));
        exact
    }
}

/// Moves the candidate at `cur` to the position its exact distance `dist` takes among the
/// others (after any equal distance) and marks it done.
fn reposition(pool: &mut [Candidate], cur: usize, dist: f32) {
    let v = pool[cur].id;
    let before = pool[..cur].partition_point(|c| c.dist <= dist);
    let pos = if before < cur {
        pool[before..=cur].rotate_right(1);
        before
    } else {
        let pos = cur + pool[cur + 1..].partition_point(|c| c.dist <= dist);
        pool[cur..=pos].rotate_left(1);
        pos
    };
    pool[pos] = Candidate {
        dist,
        id: v,
        done: true,
    };
}

/// Merges `fresh` (sorted) into the sorted `pool`, each after the pool entries of equal
/// distance, keeping the first `ef`. Returns the position of the first fresh entry
/// (`usize::MAX` if there is none); this equals inserting them one by one.
fn merge(pool: &mut Vec<Candidate>, fresh: &[Candidate], ef: usize) -> usize {
    let Some(first) = fresh.first() else {
        return usize::MAX;
    };
    let (mut i, mut j) = (pool.len(), fresh.len());
    pool.resize(i + j, *first);
    let mut k = i + j;
    while j > 0 {
        k -= 1;
        if i > 0 && pool[i - 1].dist > fresh[j - 1].dist {
            pool[k] = pool[i - 1];
            i -= 1;
        } else {
            pool[k] = fresh[j - 1];
            j -= 1;
        }
    }
    pool.truncate(ef);
    k
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

    /// The shipped per-edge estimator through the FastScan kernel, as `search` evaluates it.
    fn quantized_estimate(pq: &[f32], pv: &[f32], pu: &[f32], bits: usize) -> (f32, f32) {
        let p = pq.len();
        let mut code = vec![0u64; bits * p / 64];
        let (k, g, sum) = edge_factors(pv, pu, bits, &mut code);
        let ipv = dot(pq, pv);
        let offset = ((1 << bits) - 1) as f32;
        let lq: f32 = (0..p)
            .map(|i| {
                let c = (0..bits).map(|b| (code[b * p / 64 + i / 64] >> (i % 64) & 1) << b);
                (2.0 * c.sum::<u64>() as f32 - offset) * pq[i]
            })
            .sum();
        let mut qc = QueryCode::new(p);
        qc.encode(pq);
        let mut packed = vec![0u8; bits * 4 * p];
        for (plane, out) in code
            .chunks_exact(p / 64)
            .zip(packed.chunks_exact_mut(4 * p))
        {
            pack_batch(&[plane], p, out);
        }
        let mut out = [0u32; BATCH];
        fastscan(&packed, &qc.lut, &mut out);
        let raw = out[0];
        let quant =
            ipv + k + g * (2.0 * qc.lo * sum - offset * qc.sum + 2.0 * qc.delta * raw as f32);
        (ipv + k + g * lq, quant)
    }

    /// Mean of the shipped per-edge estimator over rotation seeds must match `<q, u>`, for the
    /// automatic code length (128 coordinates at d = 100) and for 512, with 1- and 2-bit codes;
    /// the 2-bit codes must have the smaller mean squared error.
    /// The WHT-with-signs rotation is not Haar, so the bound is empirical: 400 seeds put the
    /// standard error near 0.003; tolerances are 0.02 (float query) and 0.03 (4-bit query).
    #[test]
    fn edge_estimator_is_unbiased_over_rotations() {
        let d = 100;
        let mut rng = SplitMix(42);
        for code_bits in [0, 0, 0, 512, 512, 512] {
            let q = unit((0..d).map(|_| gaussian(&mut rng)).collect());
            let v = unit((0..d).map(|_| gaussian(&mut rng)).collect());
            let u = unit(v.iter().map(|x| x + 0.15 * gaussian(&mut rng)).collect());
            let truth = f64::from(dot(&q, &u));
            let mut mse = [0f64; 2];
            for bits in [1, 2] {
                let (mut float_sum, mut quant_sum) = (0f64, 0f64);
                let seeds = 400;
                for seed in 0..seeds {
                    let rot = Rotation::new(d, code_bits, seed);
                    let p = rot.padded();
                    let (mut pq, mut pv, mut pu) = (vec![0.0; p], vec![0.0; p], vec![0.0; p]);
                    rot.apply(&q, &mut pq);
                    rot.apply(&v, &mut pv);
                    rot.apply(&u, &mut pu);
                    let (float, quant) = quantized_estimate(&pq, &pv, &pu, bits);
                    float_sum += f64::from(float);
                    quant_sum += f64::from(quant);
                    mse[bits - 1] += (f64::from(quant) - truth).powi(2);
                }
                let float_mean = float_sum / f64::from(seeds as u32);
                let quant_mean = quant_sum / f64::from(seeds as u32);
                assert!(
                    (float_mean - truth).abs() < 0.02,
                    "{bits} bits: {float_mean} vs {truth}"
                );
                assert!(
                    (quant_mean - truth).abs() < 0.03,
                    "{bits} bits: {quant_mean} vs {truth}"
                );
            }
            assert!(mse[1] < mse[0], "2-bit mse {} >= 1-bit {}", mse[1], mse[0]);
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

    #[test]
    fn recall_on_clustered_data() {
        let d = 24;
        let data = clustered(4000, d, 1);
        let queries = clustered(100, d, 2);
        let index = QGraph::build(&data, d, &BuildParams::default());
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
        // Regression guard on a deliberately hard set (24 dimensions of noise per cluster).
        assert!(recall >= 0.9, "recall {recall}");
    }

    #[test]
    fn build_and_search_are_deterministic() {
        let d = 16;
        let data = clustered(3000, d, 3);
        let params = BuildParams {
            seed: 9,
            ..BuildParams::default()
        };
        let a = QGraph::build(&data, d, &params);
        let b = QGraph::build(&data, d, &params);
        assert_eq!(a.entry, b.entry);
        assert!(a.blocks == b.blocks);
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
