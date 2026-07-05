// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// The decisive follow-up to relation-algebra: does the phase substrate RETRIEVE
// under superposition load, where §14 found the sparse-permutation substrate
// weak? Retrieval, not verification: store a functional graph -- many edges
// (a, r) -> b -- into ONE bundled memory, then query key (a, r) and reconstruct
// b by ranking the whole codebook. Rank-1 accuracy vs the number of bundled
// edges L (chance = 1/M).
//
// PHASE memory (deterministic, verifiability-preserving): each entity is a
// D-vector of quantized phases in Z_N; bind = phase add (mod N). Store edge as a
// "trace" phase t = (a + r + b) mod N and accumulate a per-(dim, phase)
// HISTOGRAM h[d][t_d] (integer -- no complex sum, so replay stays bit-exact).
// Retrieve for query (a, r): score candidate v by sum_d h[d][(a + r + v)_d]; the
// true b's trace matches itself on every dim (+D self-term) above a ~L/N
// coincidence floor, so it stays separable until L ~ D*N.
//
// SPARSE memory (the current substrate): edge = perm(a) ∪ perm(b) into a bloom
// union; retrieve by ranking v via containment of perm(a) ∪ perm(v). Binding does
// not isolate b for a given (a, r), which is why retrieval degrades.
//
// Run: cargo run --release --bin holographic-retrieval

// NOTE ON FAIRNESS: this is a matched-D comparison, which flatters the DENSE
// phase substrate (it carries ~D*log2(N) bits/entity vs the sparse ~K*log2(D)) --
// the same caveat as §14. Sparse density is set to HMS's real 1/256 so its bloom
// is not strawmanned by over-saturation. Read the retrieval gap as suggestive,
// not a clean iso-storage win; the clean, density-independent result is the
// relation-algebra experiment.
const N: u32 = 256;
const D: usize = 1024;
const SPARSE_K: usize = D / 256; // HMS's real sparse density (1/256)
const M: usize = 512; // codebook size (rank over these; chance = 1/512)
const QUERIES: usize = 200;
const SEEDS: u64 = 5;
const LOADS: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

fn phase_vec(id: u64, seed: u64) -> Vec<u32> {
    (0..D)
        .map(|d| (mix(id ^ (d as u64), seed) % N as u64) as u32)
        .collect()
}
fn rotation(rel: u64, seed: u64) -> Vec<u32> {
    (0..D)
        .map(|d| (mix(rel ^ (d as u64) ^ 0x0107_A70, seed) % N as u64) as u32)
        .collect()
}
fn sparse_entity(id: u64, seed: u64) -> Vec<u32> {
    let mut v: Vec<u32> = (0..SPARSE_K)
        .map(|k| (mix(id ^ (k as u64) ^ 0x5EED, seed) % D as u64) as u32)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}
fn masks(rel: u64, seed: u64) -> (u32, u32) {
    (
        (mix(rel, seed ^ 0x5B1) % D as u64) as u32 | 1,
        (mix(rel, seed ^ 0x0B1) % D as u64) as u32 | 1,
    )
}

struct Edge {
    a: usize,
    r: u64,
    b: usize,
}

/// A functional graph: each (a, r) maps to a single b. `n_rel` relation types.
fn build_edges(load: usize, seed: u64, n_rel: u64) -> Vec<Edge> {
    (0..load)
        .map(|i| {
            let a = (mix(i as u64, seed ^ 0xA) % M as u64) as usize;
            let r = mix(i as u64, seed ^ 0xC) % n_rel;
            let b = (mix(i as u64, seed ^ 0xB) % M as u64) as usize;
            Edge { a, r, b }
        })
        .collect()
}

fn rank1_phase(edges: &[Edge], seed: u64) -> f64 {
    // Precompute codebook + relation phases.
    let cb: Vec<Vec<u32>> = (0..M).map(|m| phase_vec(m as u64, seed)).collect();
    let rels: std::collections::HashMap<u64, Vec<u32>> =
        edges.iter().map(|e| (e.r, rotation(e.r, seed))).collect();

    // Store: per-(dim, phase) histogram of trace phases t = (a + r + b) mod N.
    let mut hist = vec![0u32; D * N as usize];
    for e in edges {
        let th = &rels[&e.r];
        for d in 0..D {
            let t = ((cb[e.a][d] + th[d] + cb[e.b][d]) % N) as usize;
            hist[d * N as usize + t] += 1;
        }
    }

    // Query the first QUERIES edges; reconstruct b by ranking the codebook.
    let mut hits = 0usize;
    let q = QUERIES.min(edges.len());
    for e in &edges[..q] {
        let th = &rels[&e.r];
        let mut best = usize::MAX;
        let mut best_s = i64::MIN;
        for (v, cbv) in cb.iter().enumerate() {
            let mut s = 0i64;
            for d in 0..D {
                let p = ((cb[e.a][d] + th[d] + cbv[d]) % N) as usize;
                s += hist[d * N as usize + p] as i64;
            }
            if s > best_s {
                best_s = s;
                best = v;
            }
        }
        if best == e.b {
            hits += 1;
        }
    }
    hits as f64 / q as f64
}

fn rank1_sparse(edges: &[Edge], seed: u64) -> f64 {
    let cb: Vec<Vec<u32>> = (0..M).map(|m| sparse_entity(m as u64, seed)).collect();
    let mut bundle = vec![false; D];
    for e in edges {
        let (ms, mo) = masks(e.r, seed);
        for &i in &cb[e.a] {
            bundle[(i ^ ms) as usize] = true;
        }
        for &i in &cb[e.b] {
            bundle[(i ^ mo) as usize] = true;
        }
    }
    let mut hits = 0usize;
    let q = QUERIES.min(edges.len());
    for e in &edges[..q] {
        let (ms, mo) = masks(e.r, seed);
        // fixed subject contribution
        let subj_present = cb[e.a]
            .iter()
            .filter(|&&i| bundle[(i ^ ms) as usize])
            .count();
        let mut best = usize::MAX;
        let mut best_s = -1.0f64;
        for (v, cbv) in cb.iter().enumerate() {
            let obj_present = cbv.iter().filter(|&&i| bundle[(i ^ mo) as usize]).count();
            let s = (subj_present + obj_present) as f64 / (cb[e.a].len() + cbv.len()) as f64;
            if s > best_s {
                best_s = s;
                best = v;
            }
        }
        if best == e.b {
            hits += 1;
        }
    }
    hits as f64 / q as f64
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn main() {
    let n_rel = 16u64;
    println!("holographic-retrieval | N={N} D={D} codebook M={M} (chance={:.4}) rels={n_rel} seeds={SEEDS}", 1.0 / M as f64);
    println!("Rank-1 accuracy reconstructing b from (a,r) as more edges L are bundled.\n");
    println!("{:>6}  {:>16}  {:>16}", "L", "phase acc", "sparse acc");
    for &l in LOADS {
        let mut p = Vec::new();
        let mut s = Vec::new();
        for seed in 0..SEEDS {
            let edges = build_edges(l, seed, n_rel);
            p.push(rank1_phase(&edges, seed));
            s.push(rank1_sparse(&edges, seed));
        }
        println!("{l:>6}  {:>16.3}  {:>16.3}", mean(&p), mean(&s));
    }
    println!("\nKill: if the phase memory does not retrieve markedly better than the sparse");
    println!("bundle under load, phase buys no retrieval capacity over the current substrate.");
}
