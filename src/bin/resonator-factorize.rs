// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// §20. The quantized-phase-FHRR substrate (§15-18) is published prior art (qFHRR,
// arXiv 2604.25939) -- but qFHRR does NOT do resonators, and Frady/Kent resonator
// networks (SOTA for VSA factorization) are FLOAT. Open edge: does phase
// QUANTIZATION cost factorization capacity in a resonator?
//
// A resonator factors a composite c = bind(x_a, y_b, z_c) into its codebook factors
// by alternating unbind + codebook cleanup. THREE factors (search space F^3) is the
// honest benchmark -- 2 factors is trivially within capacity (100% everywhere,
// measures nothing). The state is quantized to N phase levels at each cleanup (the
// qFHRR bundle-recover step); N=0 is the float FHRR baseline. If quantized (N=256,
// even N=16-32) tracks float across the capacity knee, a DETERMINISTIC resonator is
// real -- integer-only, replay-exact factorization.
//
// Reproducibility. This is a fixed, seeded artifact. Seeds are the fixed set 0..SEEDS
// (SEEDS=24); each seed builds independent codebooks and TRIALS=32 random factor
// tuples via the splitmix `mix` hash, so the whole sweep is a pure function of the
// constants below -- no RNG state, no wall-clock, no threads. Reported mean/std are
// rounded to whole percent, so the output table is stable across runs and machines.
// The numbers this prints are the ones cited in docs/DETERMINISTIC-RESONATOR.md.
// Bump SEEDS/TRIALS or extend the F/N sweeps to tighten error bars; do not change the
// dynamics if you want the documented table to keep reproducing.
//
// Run (the bin requires the `experimental` feature):
//   cargo run --release --features experimental --bin resonator-factorize

use std::f64::consts::TAU;

const D: usize = 1024;
const FACTORS: usize = 3;
const ITERS: usize = 40;
const SEEDS: u64 = 24;
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
fn phase_vec(id: usize, axis: usize, seed: u64, n: u32) -> Cx {
    let mut re = vec![0.0; D];
    let mut im = vec![0.0; D];
    for d in 0..D {
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
    for d in 0..D {
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
    let mut r = vec![0.0; D];
    let mut i = vec![0.0; D];
    for d in 0..D {
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
    for d in 0..D {
        s += cb.0[d] * v.0[d] + cb.1[d] * v.1[d];
    }
    s
}

/// Cleanup: project `v` onto the codebook (similarity-weighted superposition), snap.
fn cleanup(cb: &[Cx], v: &Cx, n: u32) -> Cx {
    let sims: Vec<f64> = cb.iter().map(|c| sim(c, v)).collect();
    let mut out: Cx = (vec![0.0; D], vec![0.0; D]);
    for (k, c) in cb.iter().enumerate() {
        let s = sims[k];
        for d in 0..D {
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

fn superpose_init(cb: &[Cx], n: u32) -> Cx {
    let mut acc: Cx = (vec![0.0; D], vec![0.0; D]);
    for c in cb {
        for d in 0..D {
            acc.0[d] += c.0[d];
            acc.1[d] += c.1[d];
        }
    }
    g(&mut acc, n);
    acc
}

/// Mean and standard deviation (%) of per-seed factorization accuracy.
fn accuracy(f: usize, n: u32) -> (f64, f64) {
    let mut per_seed = Vec::with_capacity(SEEDS as usize);
    for seed in 0..SEEDS {
        let books: Vec<Vec<Cx>> = (0..FACTORS)
            .map(|axis| (0..f).map(|i| phase_vec(i, axis, seed, n)).collect())
            .collect();
        let mut ok = 0i64;
        for t in 0..TRIALS {
            let truth: Vec<usize> = (0..FACTORS)
                .map(|axis| (mix(t as u64, seed ^ (0x10 + axis as u64)) % f as u64) as usize)
                .collect();
            // composite = product of the true factor phasors
            let mut comp = books[0][truth[0]].clone();
            for axis in 1..FACTORS {
                comp = cmul(&comp, &books[axis][truth[axis]]);
            }
            let mut est: Vec<Cx> = (0..FACTORS)
                .map(|axis| superpose_init(&books[axis], n))
                .collect();
            let mut prev = vec![usize::MAX; FACTORS];
            let mut stable = 0;
            for _ in 0..ITERS {
                for axis in 0..FACTORS {
                    // unbind = composite ⊙ Π_{j≠axis} conj(est[j])
                    let mut ub = comp.clone();
                    for (j, e) in est.iter().enumerate() {
                        if j != axis {
                            ub = cmul(&ub, &conj(e));
                        }
                    }
                    est[axis] = cleanup(&books[axis], &ub, n);
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

fn main() {
    let fs = [16usize, 24, 32, 40, 48];
    let ns: [(u32, &str); 3] = [(0, "float"), (256, "N=256"), (16, "N=16(4bit)")];
    println!(
        "resonator-factorize | D={D} factors={FACTORS} iters={ITERS} seeds={SEEDS} trials={TRIALS}"
    );
    println!("3-factor factorization accuracy mean±std over seeds (%), chance = 1/F^3\n");
    print!("{:>10}", "F(F^3)");
    for (_, label) in ns {
        print!("  {label:>11}");
    }
    println!();
    for f in fs {
        print!("{:>10}", format!("{f}({})", f * f * f));
        for (n, _) in ns {
            let (m, sd) = accuracy(f, n);
            print!("  {:>7.0}±{:<3.0}", m, sd);
        }
        println!();
    }
    println!("\nStrong: N=256 (even N=16-32) tracks float across the knee -> quantization is");
    println!("free for factorization -> deterministic resonator is real (qFHRR + Frady/Kent).");
    println!("Kill: quantized knee shifts left of float -> quantization costs capacity.");
}
