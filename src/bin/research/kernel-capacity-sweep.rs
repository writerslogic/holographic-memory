// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Lane 1, §17. §15 recalls the nearest symbol only 85% of the time at 13 facts,
// zero distractors, D=1024 -- far below the capacity knee (~100 facts), so the
// zero-load gap is NOT capacity. This experiment discriminates three hypotheses
// with two knobs at zero load:
//   H1 confound  : the low-frequency FPE base makes the kernel too WIDE relative
//                  to grid spacing, so a between-grid query confuses neighbors.
//                  Narrowing the base-frequency spread W should lift recall.
//   H2 capacity  : raising D lifts recall, W does not.
//   H3 binding   : neither knob helps -> phase-add binding is intrinsically
//                  limited and a different bind (HRR conv) is needed.
//
// Secondary metric: exact-at-grid recall (query sits ON a grid key). Too-narrow
// kernels should keep exact recall high while between-grid recall collapses,
// exposing the width tradeoff.
//
// Run: cargo run --release --bin kernel-capacity-sweep

const N: u32 = 256;
const SCALE: i64 = 4096;
const SEEDS: u64 = 20;

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

/// Base multipliers. `w == 0` = fully random (multiplier uniform over 0..N, the
/// maximally-narrow non-graceful control); otherwise multipliers span ±(1..=w).
fn base(seed: u64, d: usize, w: u32) -> Vec<i64> {
    (0..d)
        .map(|i| {
            let raw = mix(i as u64, seed);
            if w == 0 {
                (raw % N as u64) as i64
            } else {
                let r = (raw % (2 * w as u64)) as i64;
                let m = r - w as i64;
                if m < 0 {
                    m
                } else {
                    m + 1
                }
            }
        })
        .collect()
}

fn encode(x: i64, base: &[i64]) -> Vec<u32> {
    base.iter()
        .map(|&b| (x * b).rem_euclid(N as i64) as u32)
        .collect()
}

fn symbol_phase(id: usize, seed: u64, d: usize) -> Vec<u32> {
    (0..d)
        .map(|i| (mix(id as u64 ^ (i as u64), seed ^ 0x5111) % N as u64) as u32)
        .collect()
}

fn bind(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(&x, &y)| (x + y) % N).collect()
}

/// Fraction of queries whose best-scoring symbol is the nearest grid key's symbol.
fn recall(d: usize, w: u32, queries: &[i64], cos: &[i64], sin: &[i64]) -> f64 {
    let grid: Vec<i64> = (0..=120).step_by(10).collect();
    let (mut hits, mut total) = (0i64, 0i64);
    for seed in 0..SEEDS {
        let b = base(seed, d, w);
        let syms: Vec<Vec<u32>> = (0..grid.len()).map(|i| symbol_phase(i, seed, d)).collect();
        let mut re = vec![0i64; d];
        let mut im = vec![0i64; d];
        for (si, &k) in grid.iter().enumerate() {
            let phi = bind(&encode(k, &b), &syms[si]);
            for j in 0..d {
                re[j] += cos[phi[j] as usize];
                im[j] += sin[phi[j] as usize];
            }
        }
        for &q in queries {
            let truth = grid
                .iter()
                .enumerate()
                .min_by_key(|(_, &k)| (k - q).abs())
                .unwrap()
                .0;
            let qp = encode(q, &b);
            let best = (0..grid.len())
                .max_by_key(|&si| {
                    let probe = bind(&qp, &syms[si]);
                    let mut s = 0i128;
                    for j in 0..d {
                        s += (cos[probe[j] as usize] as i128) * (re[j] as i128)
                            + (sin[probe[j] as usize] as i128) * (im[j] as i128);
                    }
                    s
                })
                .unwrap();
            hits += (best == truth) as i64;
            total += 1;
        }
    }
    100.0 * hits as f64 / total as f64
}

fn main() {
    let (cos, sin) = tables();
    let between: Vec<i64> = vec![7, 23, 38, 51, 64, 77, 92, 108];
    let on_grid: Vec<i64> = (0..=120).step_by(10).collect();
    let widths: [(u32, &str); 7] = [
        (2, "W=2"),
        (4, "W=4(base)"),
        (8, "W=8"),
        (16, "W=16"),
        (32, "W=32"),
        (64, "W=64"),
        (0, "random"),
    ];
    let dims = [512usize, 1024, 2048, 4096];

    println!("kernel-capacity-sweep | N={N} SCALE={SCALE} seeds={SEEDS} | 13 facts, zero load");
    println!("W=4 reproduces the §15 low-frequency base. chance ~7.7%.\n");

    for (label, queries) in [
        ("BETWEEN-grid recall", &between),
        ("EXACT-on-grid recall", &on_grid),
    ] {
        println!("== {label} (%) ==");
        print!("{:>10}", "");
        for d in dims {
            print!("  {:>7}", format!("D={d}"));
        }
        println!();
        for (w, name) in widths {
            print!("{name:>10}");
            for d in dims {
                print!("  {:>6.0}%", recall(d, w, queries, &cos, &sin));
            }
            println!();
        }
        println!();
    }

    println!("Discrimination: W lifts BETWEEN recall & D flat -> H1 confound (kernel width).");
    println!("D lifts it & W flat -> H2 capacity. Neither reaches ~100% -> H3 binding.");
}
