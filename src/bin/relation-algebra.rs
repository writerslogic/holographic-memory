// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Does phase-ROTATION binding buy relation algebra that the current permutation-
// union substrate provably lacks? Three relation patterns the connection graph
// cannot express (relations are independent mask pairs, no algebra between them):
//   symmetry     : (a,r,b) <=> (b,r,a)                 e.g. married_to
//   inversion    : (a,r1,b) <=> (b,r2,a), r2 = r1^-1   e.g. parent_of / child_of
//   composition  : (a,r1,b) & (b,r2,c) => (a,r3,c)     e.g. grandparent
//
// ROTATION substrate: each entity is a D-vector of quantized phases in Z_N; a
// relation is a rotation theta in Z_N; bind = (phase + theta) mod N (integer,
// exact, deterministic -- replay-safe). Group structure gives the patterns for
// free: inverse relation = -theta, composition = theta1+theta2, symmetric =
// theta in {0, N/2}. Entities on a stored chain are DERIVED by rotation.
//
// PERMUTATION substrate (the current connection-graph binding): entity = sparse
// index set; relation = role masks; edge = union of perm(subject) and
// perm(object); memory = bloom union; query = containment. Relations are
// independent, so a pattern-implied but unstored triple has no relationship to
// what was stored.
//
// For each pattern we STORE the base facts and QUERY the pattern-IMPLIED triple
// (never stored). Metric: AUC of (correct implied target) vs (a random
// distractor entity). Rotation should recover the patterns (AUC ~1); permutation
// cannot (AUC ~0.5). Swept over codebook size M to show it is scale-stable.
//
// Honest scope: this tests representational CAPACITY (clean, no superposition
// noise on the phase substrate itself). If rotation wins here, the next test is
// recovery under a noisy holographic bundle; if it does not win even clean, the
// phase direction dies cheaply. Run: cargo run --release --bin relation-algebra

const N: u32 = 256; // phase buckets (Z_N)
const D: usize = 1024; // dimensions
const SPARSE_K: usize = 64; // active indices per entity (permutation substrate)
const TRIALS: usize = 200; // pattern instances per condition
const LOADS: &[usize] = &[64, 256, 1024, 4096]; // codebook size M (distractor pool)

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

// ---- ROTATION substrate: quantized-phase vectors, rotation = mod-add ---------

fn phase_vec(id: u64, seed: u64) -> Vec<u32> {
    (0..D)
        .map(|d| (mix(id ^ (d as u64), seed) % N as u64) as u32)
        .collect()
}
fn rotation(rel: u64, seed: u64) -> Vec<u32> {
    (0..D)
        .map(|d| (mix(rel ^ (d as u64) ^ 0x0010_7A70, seed) % N as u64) as u32)
        .collect()
}
fn rotate(phi: &[u32], theta: &[u32]) -> Vec<u32> {
    phi.iter().zip(theta).map(|(&p, &t)| (p + t) % N).collect()
}
fn neg_rotation(theta: &[u32]) -> Vec<u32> {
    theta.iter().map(|&t| (N - t) % N).collect()
}
fn compose_rotation(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(&x, &y)| (x + y) % N).collect()
}
/// Fraction of dimensions where the rotated source phase equals the target phase.
fn rot_match(src: &[u32], theta: &[u32], target: &[u32]) -> f64 {
    let hits = src
        .iter()
        .zip(theta)
        .zip(target)
        .filter(|((&p, &t), &q)| (p + t) % N == q)
        .count();
    hits as f64 / D as f64
}

// ---- PERMUTATION substrate: sparse sets, role masks, bloom union -------------

fn sparse_entity(id: u64, seed: u64) -> Vec<u32> {
    let mut v: Vec<u32> = (0..SPARSE_K)
        .map(|k| (mix(id ^ (k as u64), seed) % D as u64) as u32)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}
fn role_masks(rel: u64, seed: u64) -> (u32, u32) {
    let s = (mix(rel, seed ^ 0x5B1) % D as u64) as u32 | 1;
    let o = (mix(rel, seed ^ 0x0B1) % D as u64) as u32 | 1;
    (s, o)
}
fn edge_set(a: &[u32], b: &[u32], masks: (u32, u32)) -> Vec<u32> {
    let mut v: Vec<u32> = a.iter().map(|&i| i ^ masks.0).collect();
    v.extend(b.iter().map(|&i| i ^ masks.1));
    v.sort_unstable();
    v.dedup();
    v
}
/// Containment of an edge in the bloom bundle (fraction of the edge present).
fn contained(edge: &[u32], bundle: &[bool]) -> f64 {
    if edge.is_empty() {
        return 0.0;
    }
    let hits = edge.iter().filter(|&&i| bundle[i as usize]).count();
    hits as f64 / edge.len() as f64
}

// ---- AUC (Mann-Whitney) -----------------------------------------------------

fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    let mut wins = 0.0;
    for &p in pos {
        for &n in neg {
            if p > n {
                wins += 1.0;
            } else if (p - n).abs() < 1e-12 {
                wins += 0.5;
            }
        }
    }
    wins / (pos.len() * neg.len()).max(1) as f64
}

#[derive(Clone, Copy)]
enum Pattern {
    Symmetry,
    Inversion,
    Composition,
}

/// Returns (rotation AUC, permutation AUC) for one pattern at codebook size m.
fn run_pattern(pat: Pattern, m: usize, seed: u64) -> (f64, f64) {
    // Relations (both substrates share the same abstract relation ids).
    let theta_r1 = match pat {
        // symmetric relation: theta in {0, N/2} per dim
        Pattern::Symmetry => (0..D)
            .map(|d| {
                if mix(d as u64, seed) & 1 == 0 {
                    0
                } else {
                    N / 2
                }
            })
            .collect::<Vec<u32>>(),
        _ => rotation(1, seed),
    };
    let theta_r2 = match pat {
        Pattern::Symmetry => theta_r1.clone(),
        Pattern::Inversion => neg_rotation(&theta_r1), // r2 = r1^-1
        Pattern::Composition => compose_rotation(&theta_r1, &theta_r1), // r3 = r1∘r1
    };
    let (masks1, masks2) = (role_masks(1, seed), role_masks(2, seed));

    let (mut rot_pos, mut rot_neg) = (Vec::new(), Vec::new());
    let (mut perm_pos, mut perm_neg) = (Vec::new(), Vec::new());

    for t in 0..TRIALS {
        let anchor = (t as u64) % m as u64;
        let distractor = ((t as u64).wrapping_mul(2654435761) % m as u64).max(1);

        // --- ROTATION: derive the chain by rotating, then query the implied triple.
        let a = phase_vec(anchor, seed);
        let b = rotate(&a, &theta_r1); // (a, r1, b) holds
        match pat {
            Pattern::Symmetry => {
                // query (b, r1, a): symmetric -> should hold
                rot_pos.push(rot_match(&b, &theta_r1, &a));
                rot_neg.push(rot_match(&b, &theta_r1, &phase_vec(distractor, seed)));
            }
            Pattern::Inversion => {
                // query (b, r2=r1^-1, a): should recover a
                rot_pos.push(rot_match(&b, &theta_r2, &a));
                rot_neg.push(rot_match(&b, &theta_r2, &phase_vec(distractor, seed)));
            }
            Pattern::Composition => {
                let c = rotate(&b, &theta_r1); // (b, r1, c)
                                               // query (a, r3 = r1∘r1, c): should recover c
                rot_pos.push(rot_match(&a, &theta_r2, &c));
                rot_neg.push(rot_match(&a, &theta_r2, &phase_vec(distractor, seed)));
            }
        }

        // --- PERMUTATION: store the base fact(s) in a bloom bundle, query implied.
        let mut bundle = vec![false; D];
        let ea = sparse_entity(anchor, seed);
        let eb = sparse_entity(anchor.wrapping_add(1_000_000), seed); // b is its own id
        for &i in &edge_set(&ea, &eb, masks1) {
            bundle[i as usize] = true;
        }
        let query_edge = match pat {
            Pattern::Symmetry => edge_set(&eb, &ea, masks1),
            Pattern::Inversion => edge_set(&eb, &ea, masks2),
            Pattern::Composition => {
                let ec = sparse_entity(anchor.wrapping_add(2_000_000), seed);
                for &i in &edge_set(&eb, &ec, masks1) {
                    bundle[i as usize] = true;
                }
                edge_set(&ea, &ec, masks2)
            }
        };
        let ed = sparse_entity(distractor.wrapping_add(9_000_000), seed);
        let neg_edge = edge_set(&ea, &ed, masks2);
        perm_pos.push(contained(&query_edge, &bundle));
        perm_neg.push(contained(&neg_edge, &bundle));
    }

    (auc(&rot_pos, &rot_neg), auc(&perm_pos, &perm_neg))
}

fn main() {
    println!("relation-algebra | N={N} D={D} trials={TRIALS}");
    println!("AUC: recover a pattern-implied UNSTORED triple vs a distractor (chance 0.5).");
    println!("Rotation binding has group structure (inverse/compose/symmetric); the");
    println!("permutation-union binding has independent relations and cannot.\n");
    println!(
        "{:>6}  {:>22}  {:>22}  {:>22}",
        "M", "symmetry rot|perm", "inversion rot|perm", "composition rot|perm"
    );
    for &m in LOADS {
        let (sr, sp) = run_pattern(Pattern::Symmetry, m, 1);
        let (ir, ip) = run_pattern(Pattern::Inversion, m, 1);
        let (cr, cp) = run_pattern(Pattern::Composition, m, 1);
        println!("{m:>6}  {sr:>9.3} | {sp:>8.3}  {ir:>9.3} | {ip:>8.3}  {cr:>9.3} | {cp:>8.3}");
    }
    println!("\nKill: if rotation does not clear permutation on inverse/composition, the");
    println!("phase/rotation direction buys no relation algebra and is not worth its cost.");
}
