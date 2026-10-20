// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Second slice of the living-connection-graph direction, fixing the structural
// weakness of `plastic-graph`: that bin used INDEPENDENT relations, so it had no
// connections and no substructure sharing, and reduced to an LRU cache. Here
// relations are COMPOSITIONAL edges of a graph and queries traverse PATHS, so the
// holographic part can actually do something a cache cannot: generalize through
// shared substructure.
//
// Edge (s, rel, o) is represented compositionally: the sorted union of
// perm(subject, rel) and perm(object, rel). Two edges that share (subject, rel)
// -- e.g. (A, married, B) and (A, married, C) -- therefore SHARE the perm(A,
// married) half of their active indices. A path A->B->C is the pair of edges
// (A,r1,B),(B,r2,C); it is VALID iff both edges are present (min of edge scores).
//
// Two tests, two kill conditions:
//  1. PATH plasticity vs saturation. Traversing hot paths strengthens their
//     edges; decay forgets the rest. Does hot-path validity discrimination (valid
//     vs mis-path) hold at edge-loads where a static field collapses?
//  2. THE ANTI-CACHE TEST (the one a cache fails by construction). After
//     traversing hot paths, measure discrimination on UNTRAVERSED edges that
//     SHARE (subject, rel) substructure with a traversed edge. A cache gives them
//     zero lift. A holographic field lifts them through the shared perm(s, rel)
//     indices. KILL: if the generalization lift is ~0, this is a cache and the
//     holographic thesis is dead.
//
// Integer field => deterministic, replay-exact (verifiability demoed in
// plastic-graph; omitted here to keep the focus on the generalization test).
//
// Run: cargo run --release --bin path-plasticity

use holographic_memory::EntangledHVec;

const D: usize = 16_384;
const DENSITY_DENOM: usize = 256;
const N_ENTITIES: usize = 1024;
const N_RELATIONS: usize = 8;
const HOT_PATHS: usize = 24;
const WARMUP_ROUNDS: usize = 40;
const W_STORE: i64 = 1;
const W_STRENGTHEN: i64 = 2;
const DECAY_NUM: i64 = 7;
const DECAY_DEN: i64 = 8;
const SEEDS: u64 = 12;
const LOADS: &[usize] = &[64, 128, 256, 512, 1024, 2048]; // number of edges stored

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Per-relation permutation masks for subject and object roles.
fn subj_mask(rel: usize) -> u32 {
    (mix(rel as u64, 0x5B_1EC7) % D as u64) as u32 | 1
}
fn obj_mask(rel: usize) -> u32 {
    (mix(rel as u64, 0x0B_1EC7) % D as u64) as u32 | 1
}

fn permd(hv: &EntangledHVec, mask: u32) -> Vec<u32> {
    hv.indices().iter().map(|&i| i ^ mask).collect()
}

/// Compositional edge indices: union of perm(subject, rel) and perm(object, rel).
/// Edges sharing (subject, rel) share the subject half; this is what enables
/// generalization that a cache cannot do.
fn edge_indices(cb: &[EntangledHVec], s: usize, rel: usize, o: usize) -> Vec<u32> {
    let mut v = permd(&cb[s], subj_mask(rel));
    v.extend(permd(&cb[o], obj_mask(rel)));
    v.sort_unstable();
    v.dedup();
    v
}

fn score(field: &[i64], indices: &[u32], gmean: f64) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let s: i64 = indices.iter().map(|&i| field[i as usize]).sum();
    (s as f64 / indices.len() as f64) - gmean
}

fn gmean(field: &[i64]) -> f64 {
    field.iter().sum::<i64>() as f64 / field.len() as f64
}

fn auc(correct: &[f64], wrong: &[f64]) -> f64 {
    let mut wins = 0.0;
    for &c in correct {
        for &w in wrong {
            if c > w {
                wins += 1.0;
            } else if (c - w).abs() < 1e-9 {
                wins += 0.5;
            }
        }
    }
    wins / (correct.len() * wrong.len()).max(1) as f64
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, sd)
}

struct Edge {
    s: usize,
    rel: usize,
    o: usize,
    idx: Vec<u32>,
}

/// A path is two edges sharing the middle entity: (a,r1,b),(b,r2,c).
struct Path {
    e1: usize,
    e2: usize,
}

struct Graph {
    cb: Vec<EntangledHVec>,
    edges: Vec<Edge>,
    paths: Vec<Path>,
}

fn build_graph(n_edges: usize, seed: u64) -> Graph {
    let cb: Vec<EntangledHVec> = (0..N_ENTITIES)
        .map(|i| EntangledHVec::new_with_density(D, DENSITY_DENOM, mix(i as u64, seed)))
        .collect();
    // Random edges over a hub-y entity set so paths and shared substructure exist.
    let mut edges = Vec::with_capacity(n_edges);
    for i in 0..n_edges {
        let s = (mix(i as u64, seed ^ 0xE0) % N_ENTITIES as u64) as usize;
        let rel = (mix(i as u64, seed ^ 0xE1) % N_RELATIONS as u64) as usize;
        let o = (mix(i as u64, seed ^ 0xE2) % N_ENTITIES as u64) as usize;
        let idx = edge_indices(&cb, s, rel, o);
        edges.push(Edge { s, rel, o, idx });
    }
    // Form 2-hop paths by chaining edges whose object == next subject where we can,
    // else synthesize a shared middle node so hot paths genuinely share an edge end.
    let mut paths = Vec::new();
    for i in 0..n_edges {
        // pair edge i with an edge j whose subject is edge i's object, if any.
        if let Some(j) = (0..n_edges).find(|&j| j != i && edges[j].s == edges[i].o) {
            paths.push(Path { e1: i, e2: j });
        }
        if paths.len() >= HOT_PATHS * 4 {
            break;
        }
    }
    Graph { cb, edges, paths }
}

fn fold(field: &mut [i64], idx: &[u32], w: i64) {
    for &i in idx {
        field[i as usize] += w;
    }
}

fn decay(field: &mut [i64]) {
    for v in field.iter_mut() {
        *v = *v * DECAY_NUM / DECAY_DEN;
    }
}

/// A "mis-path" for a valid path (a,r1,b),(b,r2,c): keep the first edge but break
/// the connection -- replace the second edge's subject with a wrong entity, so the
/// composed 2-hop chain does not actually exist in the graph.
fn mispath_second_edge(g: &Graph, p: &Path, seed: u64) -> Vec<u32> {
    let e2 = &g.edges[p.e2];
    let wrong_s = (mix(p.e2 as u64, seed ^ 0x9999) % N_ENTITIES as u64) as usize;
    let wrong_s = if wrong_s == e2.s {
        (wrong_s + 1) % N_ENTITIES
    } else {
        wrong_s
    };
    edge_indices(&g.cb, wrong_s, e2.rel, e2.o)
}

struct Out {
    static_path_auc: f64,
    plastic_path_auc: f64,
    static_gen_auc: f64,
    plastic_gen_auc: f64,
}

fn run(n_edges: usize, seed: u64) -> Out {
    let g = build_graph(n_edges, seed);
    let n_hot = HOT_PATHS.min(g.paths.len());
    if n_hot == 0 {
        return Out {
            static_path_auc: 0.5,
            plastic_path_auc: 0.5,
            static_gen_auc: 0.5,
            plastic_gen_auc: 0.5,
        };
    }

    // Static field: every edge stored once.
    let mut sf = vec![0i64; D];
    for e in &g.edges {
        fold(&mut sf, &e.idx, W_STORE);
    }

    // Plastic field: store all, then traverse hot paths (strengthen both edges),
    // decaying the whole field once per pass.
    let mut pf = vec![0i64; D];
    for e in &g.edges {
        fold(&mut pf, &e.idx, W_STORE);
    }
    for _ in 0..WARMUP_ROUNDS {
        for p in g.paths.iter().take(n_hot) {
            fold(&mut pf, &g.edges[p.e1].idx, W_STRENGTHEN);
            fold(&mut pf, &g.edges[p.e2].idx, W_STRENGTHEN);
        }
        decay(&mut pf);
    }

    let (sm, pm) = (gmean(&sf), gmean(&pf));

    // Test 1: hot-path validity -- valid path score vs mis-path score.
    // Path score = min of its two edge scores (both must be present).
    let path_score = |field: &[i64], gm: f64, p: &Path| -> f64 {
        score(field, &g.edges[p.e1].idx, gm).min(score(field, &g.edges[p.e2].idx, gm))
    };
    let mut s_valid = Vec::new();
    let mut s_mis = Vec::new();
    let mut p_valid = Vec::new();
    let mut p_mis = Vec::new();
    for p in g.paths.iter().take(n_hot) {
        let mis2 = mispath_second_edge(&g, p, seed);
        // valid: min(edge1, edge2); mis: min(edge1, wrong-second-edge)
        s_valid.push(path_score(&sf, sm, p));
        s_mis.push(score(&sf, &g.edges[p.e1].idx, sm).min(score(&sf, &mis2, sm)));
        p_valid.push(path_score(&pf, pm, p));
        p_mis.push(score(&pf, &g.edges[p.e1].idx, pm).min(score(&pf, &mis2, pm)));
    }

    // Test 2 (anti-cache): UNTRAVERSED edges that SHARE (subject, rel) with a hot
    // edge. Build a novel edge (hot.s, hot.rel, random-o): it was never stored or
    // strengthened, but shares the perm(s, rel) subject half. Does the plastic
    // field score its shared substructure above a fully-unrelated novel edge?
    let mut s_share = Vec::new();
    let mut s_unrel = Vec::new();
    let mut p_share = Vec::new();
    let mut p_unrel = Vec::new();
    for p in g.paths.iter().take(n_hot) {
        let hot = &g.edges[p.e1];
        let novel_o = (mix(p.e1 as u64, seed ^ 0x7777) % N_ENTITIES as u64) as usize;
        let shared = edge_indices(&g.cb, hot.s, hot.rel, novel_o); // shares subject half
        let u_s = (mix(p.e1 as u64, seed ^ 0x8888) % N_ENTITIES as u64) as usize;
        let u_r = (mix(p.e1 as u64, seed ^ 0x8889) % N_RELATIONS as u64) as usize;
        let u_o = (mix(p.e1 as u64, seed ^ 0x888A) % N_ENTITIES as u64) as usize;
        let unrelated = edge_indices(&g.cb, u_s, u_r, u_o); // shares nothing with hot
        s_share.push(score(&sf, &shared, sm));
        s_unrel.push(score(&sf, &unrelated, sm));
        p_share.push(score(&pf, &shared, pm));
        p_unrel.push(score(&pf, &unrelated, pm));
    }

    Out {
        static_path_auc: auc(&s_valid, &s_mis),
        plastic_path_auc: auc(&p_valid, &p_mis),
        static_gen_auc: auc(&s_share, &s_unrel),
        plastic_gen_auc: auc(&p_share, &p_unrel),
    }
}

fn main() {
    println!("path-plasticity | D={D} entities={N_ENTITIES} rels={N_RELATIONS} hot_paths={HOT_PATHS} seeds={SEEDS}");
    println!("PATH AUC: valid 2-hop path vs broken-connection mis-path (chance 0.5).");
    println!("GEN AUC (anti-cache): untraversed edge sharing (subj,rel) with a hot edge vs a");
    println!("fully-unrelated novel edge. If plastic GEN >> 0.5 (and >> static), strengthening");
    println!("generalized through shared substructure -- a cache cannot do this.\n");
    println!(
        "{:>6}  {:>15}  {:>15}  {:>15}  {:>15}",
        "edges", "static PATH", "plastic PATH", "static GEN", "plastic GEN"
    );
    for &n in LOADS {
        let mut sp = Vec::new();
        let mut pp = Vec::new();
        let mut sg = Vec::new();
        let mut pg = Vec::new();
        for seed in 0..SEEDS {
            let o = run(n, seed);
            sp.push(o.static_path_auc);
            pp.push(o.plastic_path_auc);
            sg.push(o.static_gen_auc);
            pg.push(o.plastic_gen_auc);
        }
        println!(
            "{n:>6}  {:>9.3} ({:>4.2})  {:>9.3} ({:>4.2})  {:>9.3} ({:>4.2})  {:>9.3} ({:>4.2})",
            mean_sd(&sp).0,
            mean_sd(&sp).1,
            mean_sd(&pp).0,
            mean_sd(&pp).1,
            mean_sd(&sg).0,
            mean_sd(&sg).1,
            mean_sd(&pg).0,
            mean_sd(&pg).1,
        );
    }
    println!("\nKILL (cache verdict): if plastic GEN ~ 0.5 or ~ static GEN, strengthening did");
    println!(
        "NOT generalize through shared substructure -> it is a cache, holographic thesis dead."
    );
}
