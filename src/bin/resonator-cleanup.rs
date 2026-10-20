// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Lane 1, §16. §15 left single-pass associative recall on the fixed-point complex
// substrate at 85% (zero load) -> 45% (400 distractors). The field superposes
// bind(key_i, symbol_i) facts; a continuous query fixes the key at encode(q),
// which is not a stored key, so the one-shot symbol readout carries interference
// from neighboring facts.
//
// This tests a two-factor RESONATOR: alternate hard cleanup between the stored-key
// codebook and the symbol codebook, snapping the continuous key estimate onto the
// nearest actual stored key and sharpening the symbol jointly. Kill can fire: a
// wrong key-snap propagates and can make the resonator worse than single-pass.
//
// Run: cargo run --release --bin resonator-cleanup

const N: u32 = 256;
const D: usize = 1024;
const SCALE: i64 = 4096;
const ITERS: usize = 4;

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
    let s = (0..N).map(|p| f(p, -std::f64::consts::FRAC_PI_2)).collect();
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

fn encode(x: i64, base: &[i64]) -> Vec<u32> {
    base.iter()
        .map(|&b| (x * b).rem_euclid(N as i64) as u32)
        .collect()
}

fn symbol_phase(id: usize, seed: u64) -> Vec<u32> {
    (0..D)
        .map(|d| (mix(id as u64 ^ (d as u64), seed ^ 0x5111) % N as u64) as u32)
        .collect()
}

fn bind(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(&x, &y)| (x + y) % N).collect()
}

/// Integer inner product of a probe phase vector against the complex field.
fn field_score(probe: &[u32], re: &[i64], im: &[i64], cos: &[i64], sin: &[i64]) -> i128 {
    let mut s = 0i128;
    for d in 0..D {
        s += (cos[probe[d] as usize] as i128) * (re[d] as i128)
            + (sin[probe[d] as usize] as i128) * (im[d] as i128);
    }
    s
}

/// argmax over a codebook of the field score of bind(fixed, book[j]).
fn best(
    book: &[Vec<u32>],
    fixed: &[u32],
    re: &[i64],
    im: &[i64],
    cos: &[i64],
    sin: &[i64],
) -> usize {
    (0..book.len())
        .max_by_key(|&j| field_score(&bind(fixed, &book[j]), re, im, cos, sin))
        .unwrap()
}

fn main() {
    let (cos, sin) = tables();
    let seeds = 5u64;
    let grid: Vec<i64> = (0..=120).step_by(10).collect();
    let queries: Vec<i64> = vec![7, 23, 38, 51, 64, 77, 92, 108];

    println!("resonator-cleanup | N={N} D={D} SCALE={SCALE} iters={ITERS} seeds={seeds}");
    println!("Symbol recall: single-pass vs two-factor resonator (chance ~7.7%)");
    println!(
        "{:>11}  {:>12}  {:>11}  {:>11}",
        "distractors", "single-pass", "resonator", "key-recovery"
    );

    for &extra in &[0usize, 25, 100, 400] {
        let (mut sp_hits, mut res_hits, mut key_hits, mut total) = (0i64, 0i64, 0i64, 0i64);
        for seed in 0..seeds {
            let b = base(seed);
            let keys: Vec<Vec<u32>> = grid.iter().map(|&k| encode(k, &b)).collect();
            let syms: Vec<Vec<u32>> = (0..grid.len()).map(|i| symbol_phase(i, seed)).collect();

            let mut re = vec![0i64; D];
            let mut im = vec![0i64; D];
            let add = |phi: &[u32], re: &mut [i64], im: &mut [i64]| {
                for d in 0..D {
                    re[d] += cos[phi[d] as usize];
                    im[d] += sin[phi[d] as usize];
                }
            };
            for si in 0..grid.len() {
                add(&bind(&keys[si], &syms[si]), &mut re, &mut im);
            }
            for i in 0..extra {
                let dk = 500 + i as i64;
                add(
                    &bind(&encode(dk, &b), &symbol_phase(1000 + i, seed)),
                    &mut re,
                    &mut im,
                );
            }

            for &q in &queries {
                let truth = grid
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &k)| (k - q).abs())
                    .unwrap()
                    .0;
                let qp = encode(q, &b);

                // single-pass: key fixed at the continuous query
                let sp = best(&syms, &qp, &re, &im, &cos, &sin);
                sp_hits += (sp == truth) as i64;

                // resonator: alternate symbol/key cleanup, key snaps onto grid
                let mut key_est = qp.clone();
                let mut sym_idx = sp;
                let mut key_idx = usize::MAX;
                for _ in 0..ITERS {
                    sym_idx = best(&syms, &key_est, &re, &im, &cos, &sin);
                    let k = best(&keys, &syms[sym_idx], &re, &im, &cos, &sin);
                    if k == key_idx {
                        break; // converged
                    }
                    key_idx = k;
                    key_est = keys[k].clone();
                }
                res_hits += (sym_idx == truth) as i64;
                key_hits += (key_idx == truth) as i64;
                total += 1;
            }
        }
        println!(
            "{extra:>11}  {:>11.0}%  {:>10.0}%  {:>10.0}%",
            100.0 * sp_hits as f64 / total as f64,
            100.0 * res_hits as f64 / total as f64,
            100.0 * key_hits as f64 / total as f64
        );
    }

    println!("\nKill: resonator <= single-pass at the load levels => alternating cleanup buys");
    println!("nothing here; ship the module single-pass. Determinism preserved (integer only).");
}
