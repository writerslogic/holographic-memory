// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Lane 1 continuation. The FPE fractional-encoding kernel is real but needs a
// COSINE (inner-product) readout; the integer phase-histogram substrate is
// exact-match, so it cannot host graceful similarity (a "nearest" test there
// passed only by coincidence). Cosine is a float op, which would break the
// bit-exact-replay verifiability we depend on.
//
// Resolution tested here: a FIXED-POINT COMPLEX substrate. A phase is turned into
// (cos, sin) via INTEGER lookup tables; similarity and superposition are integer
// inner products / sums. This is FHRR quantized to stay deterministic -- graceful
// similarity AND bit-exact replay at once.
//
//   Part 1 (readout): nearest-value recall via fixed-point cosine vs exact-match,
//     with queries landing BETWEEN grid points so only graceful similarity works.
//   Part 2 (superposition): store continuous-key -> symbol facts bundled into one
//     integer complex field; query a key between stored keys, recover the NEAREST
//     key's symbol, under increasing distractor load.
//
// Run: cargo run --release --bin fixed-point-holographic

const N: u32 = 256;
const D: usize = 1024;
const SCALE: i64 = 4096; // fixed-point unit for cos/sin

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

fn tables() -> (Vec<i64>, Vec<i64>) {
    let f = |p: u32, phase: f64| {
        (SCALE as f64 * (std::f64::consts::TAU * p as f64 / N as f64 + phase).cos()).round() as i64
    };
    let c = (0..N).map(|p| f(p, 0.0)).collect();
    let s = (0..N).map(|p| f(p, -std::f64::consts::FRAC_PI_2)).collect(); // sin(x)=cos(x-pi/2)
    (c, s)
}

fn base(seed: u64) -> Vec<i64> {
    (0..D)
        .map(|d| {
            let v = (mix(d as u64, seed) % 9) as i64 - 4;
            if v == 0 {
                1
            } else {
                v
            }
        })
        .collect()
}

/// Fractional-power phase vector for integer value `x`.
fn encode(x: i64, base: &[i64]) -> Vec<u32> {
    base.iter()
        .map(|&b| (x * b).rem_euclid(N as i64) as u32)
        .collect()
}

/// A random symbol phase vector (for the value side of a key->symbol fact).
fn symbol_phase(id: usize, seed: u64) -> Vec<u32> {
    (0..D)
        .map(|d| (mix(id as u64 ^ (d as u64), seed ^ 0x5111) % N as u64) as u32)
        .collect()
}

fn bind(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(&x, &y)| (x + y) % N).collect()
}

/// Fixed-point cosine similarity between two phase vectors (integer).
fn cos_sim(a: &[u32], b: &[u32], cos: &[i64]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&pa, &pb)| cos[((pa + N - pb) % N) as usize])
        .sum()
}

/// Exact-match count (the histogram substrate's readout).
fn exact_sim(a: &[u32], b: &[u32]) -> i64 {
    a.iter().zip(b).filter(|(&pa, &pb)| pa == pb).count() as i64
}

fn main() {
    let (cos, sin) = tables();
    let seeds = 5u64;
    let grid: Vec<i64> = (0..=120).step_by(10).collect();
    let queries: Vec<i64> = vec![7, 23, 38, 51, 64, 77, 92, 108];

    // --- Part 1: readout comparison (cosine vs exact) ---
    println!("fixed-point-holographic | N={N} D={D} SCALE={SCALE} seeds={seeds}");
    println!("Part 1 -- nearest-value recall (grid spacing 10, queries between points):");
    let (mut cos_hits, mut ex_hits, mut tot) = (0, 0, 0);
    for seed in 0..seeds {
        let b = base(seed);
        for &q in &queries {
            let truth = *grid.iter().min_by_key(|&&x| (x - q).abs()).unwrap();
            let qp = encode(q, &b);
            let cbest = *grid
                .iter()
                .max_by_key(|&&v| cos_sim(&qp, &encode(v, &b), &cos))
                .unwrap();
            let ebest = *grid
                .iter()
                .max_by_key(|&&v| exact_sim(&qp, &encode(v, &b)))
                .unwrap();
            cos_hits += (cbest == truth) as i64;
            ex_hits += (ebest == truth) as i64;
            tot += 1;
        }
    }
    println!(
        "  cosine (fixed-point): {:.0}%   exact-match (histogram): {:.0}%",
        100.0 * cos_hits as f64 / tot as f64,
        100.0 * ex_hits as f64 / tot as f64
    );

    // --- Part 2: superposition associative memory ---
    // Store (grid-key -> symbol_i) bundled; query a between-grid key, recover the
    // NEAREST key's symbol, as distractor facts are added.
    println!("\nPart 2 -- graceful associative retrieval under superposition load:");
    println!("{:>10}  {:>16}", "distractors", "recall %");
    for &extra in &[0usize, 25, 100, 400] {
        let (mut hits, mut total) = (0i64, 0i64);
        for seed in 0..seeds {
            let b = base(seed);
            let mut re = vec![0i64; D];
            let mut im = vec![0i64; D];
            let add = |phi: &[u32], re: &mut [i64], im: &mut [i64]| {
                for d in 0..D {
                    re[d] += cos[phi[d] as usize];
                    im[d] += sin[phi[d] as usize];
                }
            };
            // grid keys -> symbols 0..grid.len()
            for (si, &k) in grid.iter().enumerate() {
                add(
                    &bind(&encode(k, &b), &symbol_phase(si, seed)),
                    &mut re,
                    &mut im,
                );
            }
            // distractor facts (keys far outside the grid -> other symbols)
            for i in 0..extra {
                let dk = 500 + i as i64;
                add(
                    &bind(&encode(dk, &b), &symbol_phase(1000 + i, seed)),
                    &mut re,
                    &mut im,
                );
            }
            for &q in &queries {
                let truth_sym = grid
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &k)| (k - q).abs())
                    .unwrap()
                    .0;
                // recover: the symbol whose bind(query_key, symbol) best matches the field
                let qp = encode(q, &b);
                let best = (0..grid.len())
                    .max_by_key(|&si| {
                        let probe = bind(&qp, &symbol_phase(si, seed));
                        let mut s = 0i128;
                        for d in 0..D {
                            s += (cos[probe[d] as usize] as i128) * (re[d] as i128)
                                + (sin[probe[d] as usize] as i128) * (im[d] as i128);
                        }
                        s
                    })
                    .unwrap();
                hits += (best == truth_sym) as i64;
                total += 1;
            }
        }
        println!("{extra:>10}  {:>15.0}%", 100.0 * hits as f64 / total as f64);
    }

    println!("\nDeterminism: cos/sin are integer tables; store and query are i64/i128 sums --");
    println!("no float, so the field is a bit-exact fold of its inserts (replay-verifiable).");
    println!("Kill: if fixed-point cosine does not beat exact-match on graceful nearest, the");
    println!("complex substrate buys no continuous capability.");
}
