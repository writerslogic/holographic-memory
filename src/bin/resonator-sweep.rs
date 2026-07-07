// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// §20 validation hardening. Extends the fixed resonator-factorize design along two
// controlling parameters WITHOUT changing the dynamics: phase resolution N (phase
// bits float/8/6/4/3/2 = N in {0,256,64,16,8,4}) and dimension D in {512,1024,2048}.
// More seeds (30 vs 24) tighten the error bars. The point is confidence in the §20
// result (quantization is free for resonator factorization), not a new claim.
//
// The dynamics (unbind + codebook cleanup + qFHRR snap) are copied verbatim from
// resonator-factorize.rs with D threaded as a parameter instead of a const, so the
// per-(seed,trial) computation is bit-identical at D=1024. The REPRO CHECK block
// below re-runs the exact frozen configuration (D=1024, 24 seeds, N in {0,256,16})
// and must match the table in docs/DETERMINISTIC-RESONATOR.md; if it does not, this
// reimplementation drifted and the extended numbers are not trustworthy.
//
// Self-contained (std only), so no cargo features are needed.
// Run: cargo run --release --bin resonator-sweep

use std::f64::consts::TAU;

const FACTORS: usize = 3;
const ITERS: usize = 40;
const TRIALS: usize = 32; // random factor tuples per seed

type Cx = (Vec<f64>, Vec<f64>);

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Complex phase vector for codebook item `id` on factor axis `axis`. `n==0` =
/// continuous phase (float FHRR); `n>0` = phase quantized to Z_N (qFHRR).
fn phase_vec(id: usize, axis: usize, seed: u64, n: u32, dim: usize) -> Cx {
    let mut re = vec![0.0; dim];
    let mut im = vec![0.0; dim];
    for d in 0..dim {
        let raw = mix((id as u64) << 20 | (axis as u64) << 40 | d as u64, seed);
        let phase = if n == 0 {
            (raw as f64 / u64::MAX as f64) * TAU
        } else {
            TAU * (raw % n as u64) as f64 / n as f64
        };
        re[d] = phase.cos();
        im[d] = phase.sin();
    }
    (re, im)
}

/// Snap each component to a unit phasor, quantizing phase to N levels (qFHRR recover).
fn g(v: &mut Cx, n: u32) {
    for d in 0..v.0.len() {
        let phase = v.1[d].atan2(v.0[d]);
        let p = if n == 0 {
            phase
        } else {
            let idx = (phase / TAU * n as f64).round() as i64;
            TAU * idx.rem_euclid(n as i64) as f64 / n as f64
        };
        v.0[d] = p.cos();
        v.1[d] = p.sin();
    }
}

fn cmul(a: &Cx, b: &Cx) -> Cx {
    let dim = a.0.len();
    let mut r = vec![0.0; dim];
    let mut i = vec![0.0; dim];
    for d in 0..dim {
        r[d] = a.0[d] * b.0[d] - a.1[d] * b.1[d];
        i[d] = a.0[d] * b.1[d] + a.1[d] * b.0[d];
    }
    (r, i)
}

fn conj(a: &Cx) -> Cx {
    (a.0.clone(), a.1.iter().map(|&x| -x).collect())
}

/// Real part of the Hermitian inner product Re<cb, v>.
fn sim(cb: &Cx, v: &Cx) -> f64 {
    let mut s = 0.0;
    for d in 0..cb.0.len() {
        s += cb.0[d] * v.0[d] + cb.1[d] * v.1[d];
    }
    s
}

/// Cleanup: project `v` onto the codebook (similarity-weighted superposition), snap.
fn cleanup(cb: &[Cx], v: &Cx, n: u32, dim: usize) -> Cx {
    let sims: Vec<f64> = cb.iter().map(|c| sim(c, v)).collect();
    let mut out: Cx = (vec![0.0; dim], vec![0.0; dim]);
    for (k, c) in cb.iter().enumerate() {
        let s = sims[k];
        for d in 0..dim {
            out.0[d] += s * c.0[d];
            out.1[d] += s * c.1[d];
        }
    }
    g(&mut out, n);
    out
}

fn argmax(cb: &[Cx], v: &Cx) -> usize {
    (0..cb.len())
        .map(|k| (k, sim(&cb[k], v)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

fn superpose_init(cb: &[Cx], n: u32, dim: usize) -> Cx {
    let mut acc: Cx = (vec![0.0; dim], vec![0.0; dim]);
    for c in cb {
        for d in 0..dim {
            acc.0[d] += c.0[d];
            acc.1[d] += c.1[d];
        }
    }
    g(&mut acc, n);
    acc
}

/// Mean and standard deviation (%) of per-seed 3-factor factorization accuracy over
/// `seeds` seeds. Identical dynamics to resonator-factorize; `dim` and `n` are the
/// swept parameters.
fn accuracy(f: usize, n: u32, dim: usize, seeds: u64) -> (f64, f64) {
    let mut per_seed = Vec::with_capacity(seeds as usize);
    for seed in 0..seeds {
        let books: Vec<Vec<Cx>> = (0..FACTORS)
            .map(|axis| (0..f).map(|i| phase_vec(i, axis, seed, n, dim)).collect())
            .collect();
        let mut ok = 0i64;
        for t in 0..TRIALS {
            let truth: Vec<usize> = (0..FACTORS)
                .map(|axis| (mix(t as u64, seed ^ (0x10 + axis as u64)) % f as u64) as usize)
                .collect();
            let mut comp = books[0][truth[0]].clone();
            for axis in 1..FACTORS {
                comp = cmul(&comp, &books[axis][truth[axis]]);
            }
            let mut est: Vec<Cx> = (0..FACTORS)
                .map(|axis| superpose_init(&books[axis], n, dim))
                .collect();
            let mut prev = vec![usize::MAX; FACTORS];
            let mut stable = 0;
            for _ in 0..ITERS {
                for axis in 0..FACTORS {
                    let mut ub = comp.clone();
                    for (j, e) in est.iter().enumerate() {
                        if j != axis {
                            ub = cmul(&ub, &conj(e));
                        }
                    }
                    est[axis] = cleanup(&books[axis], &ub, n, dim);
                }
                let cur: Vec<usize> = (0..FACTORS)
                    .map(|axis| argmax(&books[axis], &est[axis]))
                    .collect();
                if cur == prev {
                    stable += 1;
                    if stable >= 3 {
                        break;
                    }
                } else {
                    stable = 0;
                }
                prev = cur;
            }
            ok += (prev == truth) as i64;
        }
        per_seed.push(100.0 * ok as f64 / TRIALS as f64);
    }
    let mean = per_seed.iter().sum::<f64>() / per_seed.len() as f64;
    let var = per_seed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / per_seed.len() as f64;
    (mean, var.sqrt())
}

/// Print one accuracy table (rows = F, cols = the given N set) at fixed D and seeds.
fn table(fs: &[usize], ns: &[(u32, &str)], dim: usize, seeds: u64) {
    print!("{:>12}", "F(F^3)");
    for (_, label) in ns {
        print!("  {label:>11}");
    }
    println!();
    for &f in fs {
        print!("{:>12}", format!("{f}({})", f * f * f));
        for &(n, _) in ns {
            let (m, sd) = accuracy(f, n, dim, seeds);
            print!("  {:>7.0}±{:<3.0}", m, sd);
        }
        println!();
    }
}

fn main() {
    let fs = [16usize, 24, 32, 40, 48];

    // --- Integrity gate: reproduce the frozen resonator-factorize table exactly. ---
    println!("REPRO CHECK | D=1024 seeds=24 trials={TRIALS} factors={FACTORS}");
    println!("Must match docs/DETERMINISTIC-RESONATOR.md (float/N=256/N=16 columns).\n");
    let repro_ns: [(u32, &str); 3] = [(0, "float"), (256, "N=256"), (16, "N=16")];
    table(&fs, &repro_ns, 1024, 24);

    // --- Hardening sweep: phase bits {float,8,6,4,3,2} x D {512,1024,2048}, 30 seeds. ---
    let seeds = 30u64;
    let ns: [(u32, &str); 6] = [
        (0, "float"),
        (256, "N=256(8b)"),
        (64, "N=64(6b)"),
        (16, "N=16(4b)"),
        (8, "N=8(3b)"),
        (4, "N=4(2b)"),
    ];
    for dim in [512usize, 1024, 2048] {
        println!(
            "\n=== D={dim} | seeds={seeds} trials={TRIALS} factors={FACTORS} | chance=1/F^3 ==="
        );
        table(&fs, &ns, dim, seeds);
    }
    println!("\nRead: within each D, do the phase-bit columns overlap float within +-1 sigma");
    println!("across F? If yes at D=1024 (and it holds at 512/2048), the '4-bit is free'");
    println!("result is robust to resolution and dimension. Chance floor 1/F^3 is <0.03%.");
}
