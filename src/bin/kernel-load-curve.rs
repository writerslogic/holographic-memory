// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Lane 1, §18. §17 fixed the ZERO-load recall ceiling by tuning the FPE kernel
// bandwidth (W=4->16 lifts 85%->100% at 13 facts). This tests the FULL associative
// curve under distractor load (the §15 Part-2 task): store 13 grid facts + `extra`
// distractors bundled into one fixed-point complex field, query between-grid keys,
// recover the nearest grid key's symbol.
//
//   (a) Does the tuned kernel (W=16) lift the whole curve, or only zero load?
//   (b) With the confound removed, does D behave as capacity theory predicts --
//       the lever for the distractor tail (recall ~ 1 - M/cD)?
//
// Run: cargo run --release --bin kernel-load-curve

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

fn base(seed: u64, d: usize, w: u32) -> Vec<i64> {
    (0..d)
        .map(|i| {
            let r = (mix(i as u64, seed) % (2 * w as u64)) as i64;
            let m = r - w as i64;
            if m < 0 {
                m
            } else {
                m + 1
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

fn recall(d: usize, w: u32, extra: usize, cos: &[i64], sin: &[i64]) -> f64 {
    let grid: Vec<i64> = (0..=120).step_by(10).collect();
    let queries: [i64; 8] = [7, 23, 38, 51, 64, 77, 92, 108];
    let (mut hits, mut total) = (0i64, 0i64);
    for seed in 0..SEEDS {
        let b = base(seed, d, w);
        let syms: Vec<Vec<u32>> = (0..grid.len()).map(|i| symbol_phase(i, seed, d)).collect();
        let mut re = vec![0i64; d];
        let mut im = vec![0i64; d];
        let add = |phi: &[u32], re: &mut [i64], im: &mut [i64]| {
            for j in 0..d {
                re[j] += cos[phi[j] as usize];
                im[j] += sin[phi[j] as usize];
            }
        };
        for (si, &k) in grid.iter().enumerate() {
            add(&bind(&encode(k, &b), &syms[si]), &mut re, &mut im);
        }
        for i in 0..extra {
            let dk = 500 + i as i64;
            add(
                &bind(&encode(dk, &b), &symbol_phase(1000 + i, seed, d)),
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
    let loads = [0usize, 25, 100, 400, 1000];
    let configs: [(u32, usize, &str); 5] = [
        (4, 1024, "W=4  D=1024 (§15)"),
        (16, 1024, "W=16 D=1024"),
        (16, 2048, "W=16 D=2048"),
        (16, 4096, "W=16 D=4096"),
        (8, 4096, "W=8  D=4096"),
    ];

    println!("kernel-load-curve | N={N} SCALE={SCALE} seeds={SEEDS} | 13 grid facts + distractors");
    println!("recall %, chance ~7.7%\n");
    print!("{:>20}", "config \\ distractors");
    for l in loads {
        print!("  {:>5}", l);
    }
    println!();
    for (w, d, label) in configs {
        print!("{label:>20}");
        for extra in loads {
            print!("  {:>4.0}%", recall(d, w, extra, &cos, &sin));
        }
        println!();
    }
    println!("\nKill: W=16 <= W=4 once load>0 -> kernel fix is zero-load only.");
    println!("D flat on the tail -> capacity-inefficient (revisit decorrelated bases).");
}
