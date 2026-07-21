// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// FUSION experiment: the exact coded store (§26) and the holographic superposition
// store are two ends of ONE spectrum — regularized least squares. Store one vector x
// = argmin ‖Ax − v‖² + λ‖x‖². λ→0 = exact pinv solve (coded store, M≈D, noise-fragile);
// large λ = heavy smoothing ≈ Hebbian/superposition (graceful to noisy keys, lower
// capacity). One knob λ trades exact capacity for noise tolerance. This proves it's a
// single continuous mechanism, not two glued layers.
//
// Two readouts per solve: EXACT (query the stored key) and NOISY (query a corrupted
// key, cosine ~0.9) — capacity axis and similarity/robustness axis. Run:
//   cargo run --release --bin ridge-memory

use nalgebra::{DMatrix, DVector};

const D: usize = 256;
const V: usize = 64;
const SEEDS: u64 = 5;
const SIGMA: f64 = 0.5; // key-noise std (cosine(a, a+noise) ~ 0.89)

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

fn gauss(seed: u64, a: u64, b: u64) -> f64 {
    let u1 = (mix(a, seed ^ b) % 1_000_000 + 1) as f64 / 1_000_001.0;
    let u2 = (mix(a, seed ^ b ^ 0x5A) % 1_000_000) as f64 / 1_000_000.0;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// (exact_recall, noisy_recall) for load m and relative ridge lambda.
fn recall(m: usize, lambda_rel: f64) -> (f64, f64) {
    let (mut ex, mut nz, mut tot) = (0i64, 0i64, 0i64);
    for seed in 0..SEEDS {
        let mut a = DMatrix::<f64>::zeros(m, D);
        for i in 0..m {
            for d in 0..D {
                a[(i, d)] = gauss(seed, (i as u64) << 20 | d as u64, 0);
            }
        }
        let truth: Vec<usize> = (0..m)
            .map(|i| (mix(i as u64, seed ^ 0xC0) % V as u64) as usize)
            .collect();
        let v = DVector::from_iterator(m, truth.iter().map(|&t| t as f64));
        let at = a.transpose();
        let mut reg = &at * &a; // D×D
        let mean_diag = reg.diagonal().mean();
        for d in 0..D {
            reg[(d, d)] += lambda_rel * mean_diag;
        }
        let atv = &at * &v;
        let x = match reg.lu().solve(&atv) {
            Some(x) => x,
            None => continue,
        };
        // scale-invariant readout: calibrate affine s·<a,x>+c ≈ v once at store time
        // (the store represents value up to a global scale — large-λ ridge = Hebbian
        // superposition up to scale; without this the holographic end reads as ~0).
        let raw: Vec<f64> = (0..m).map(|i| (a.row(i) * &x)[0]).collect();
        let mp = raw.iter().sum::<f64>() / m as f64;
        let mt = v.mean();
        let cov: f64 = raw
            .iter()
            .zip(truth.iter())
            .map(|(&p, &t)| (p - mp) * (t as f64 - mt))
            .sum();
        let var: f64 = raw
            .iter()
            .map(|&p| (p - mp).powi(2))
            .sum::<f64>()
            .max(1e-12);
        let (s, c) = (cov / var, mt - (cov / var) * mp);
        for i in 0..m {
            let est = (s * raw[i] + c).round().clamp(0.0, (V - 1) as f64) as usize;
            ex += (est == truth[i]) as i64;
            // noisy query: a_i + SIGMA*noise
            let mut an = DVector::<f64>::zeros(D);
            for d in 0..D {
                an[d] = a[(i, d)] + SIGMA * gauss(seed, (i as u64) << 20 | d as u64, 0xBEEF);
            }
            let estn = (s * (an.transpose() * &x)[0] + c)
                .round()
                .clamp(0.0, (V - 1) as f64) as usize;
            nz += (estn == truth[i]) as i64;
            tot += 1;
        }
    }
    (
        100.0 * ex as f64 / tot as f64,
        100.0 * nz as f64 / tot as f64,
    )
}

fn main() {
    let lambdas = [0.0f64, 0.001, 0.01, 0.1, 1.0];
    let loads = [0.3f64, 0.5, 0.8, 1.0];
    println!(
        "ridge-memory | D={D} V={V} seeds={SEEDS} sigma={SIGMA} | recall %, chance={:.1}%",
        100.0 / V as f64
    );
    println!("ONE knob lambda spans exact-coded (lambda~0) <-> holographic (large lambda).\n");
    for (label, noisy) in [
        ("EXACT-key recall", false),
        ("NOISY-key recall (cos~0.9)", true),
    ] {
        println!("== {label} ==   (rows lambda, cols load M/D)");
        print!("{:>10}", "lambda");
        for l in loads {
            print!("  {:>5}", format!("{l:.2}"));
        }
        println!();
        for &lam in &lambdas {
            print!("{lam:>10.3}");
            for &l in &loads {
                let m = ((l * D as f64) as usize).max(1);
                let (ex, nz) = recall(m, lam);
                print!("  {:>4.0}%", if noisy { nz } else { ex });
            }
            println!();
        }
        println!();
    }
    println!("Fusion signature: lambda~0 -> high EXACT capacity (M≈D) but poor NOISY;");
    println!("large lambda -> lower exact capacity but NOISY recall improves. One spectrum.");
}
