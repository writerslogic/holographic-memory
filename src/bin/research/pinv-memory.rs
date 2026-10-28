// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Open-exploration finding (unbiased wave): additive superposition (field = Σ
// bind(key,value)) is the WORST encoding. Instead of SUMMING the bound pairs, treat
// the stored vector x as unknown and SOLVE the linear system {⟨a_{k_i}, x⟩ = v_i}
// for x (least-norm / least-squares). For M ≤ D the solve is EXACT → ~100% recall
// to M ≈ D, far past the ~0.3·D matched-filter/AMP floor of superposition. This is
// the classic Kohonen/Personnaz optimal linear associator (projection rule) — which
// the VSA-superposition framing suppressed for the whole campaign.
//
// Reports recall vs load M/D, exact-solve and with x quantized to an integer grid
// (the deterministic-substrate cost). Run: cargo run --release --bin pinv-memory

use nalgebra::{DMatrix, DVector};

const D: usize = 256; // smaller D keeps the O(D^3) solves cheap; capacity law is M≈D
const V: usize = 64; // value alphabet (levels)
const SEEDS: u64 = 5;
const QLEVELS: f64 = 4096.0; // x quantized to this many levels over its range

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Deterministic Gaussian key vector for key id (Box-Muller from two hashes).
fn key_vec(id: usize, seed: u64) -> DVector<f64> {
    DVector::from_iterator(
        D,
        (0..D).map(|d| {
            let u1 = (mix((id as u64) << 20 | d as u64, seed) % 1_000_000 + 1) as f64 / 1_000_001.0;
            let u2 =
                (mix((id as u64) << 20 | d as u64, seed ^ 0x5A) % 1_000_000) as f64 / 1_000_000.0;
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }),
    )
}

#[allow(clippy::needless_range_loop)]
fn recall(m: usize) -> (f64, f64) {
    let (mut exact_ok, mut quant_ok, mut tot) = (0i64, 0i64, 0i64);
    for seed in 0..SEEDS {
        // A: M x D matrix of key vectors (rows)
        let mut a = DMatrix::<f64>::zeros(m, D);
        for i in 0..m {
            a.set_row(i, &key_vec(i, seed).transpose());
        }
        let truth: Vec<usize> = (0..m)
            .map(|i| (mix(i as u64, seed ^ 0xC0) % V as u64) as usize)
            .collect();
        let v = DVector::from_iterator(m, truth.iter().map(|&t| t as f64));
        // solve A x = v (least-norm if M<=D, least-squares if M>D) via pseudo-inverse
        let x = match a.clone().pseudo_inverse(1e-9) {
            Ok(pinv) => pinv * &v,
            Err(_) => continue,
        };
        // quantized x on the deterministic integer grid
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for &e in x.iter() {
            lo = lo.min(e);
            hi = hi.max(e);
        }
        let span = (hi - lo).max(1e-9);
        let xq = DVector::from_iterator(
            D,
            x.iter()
                .map(|&e| lo + ((e - lo) / span * QLEVELS).round() / QLEVELS * span),
        );
        for i in 0..m {
            let ai = a.row(i);
            let est = (ai * &x)[0].round().clamp(0.0, (V - 1) as f64) as usize;
            let estq = (ai * &xq)[0].round().clamp(0.0, (V - 1) as f64) as usize;
            exact_ok += (est == truth[i]) as i64;
            quant_ok += (estq == truth[i]) as i64;
            tot += 1;
        }
    }
    (
        100.0 * exact_ok as f64 / tot as f64,
        100.0 * quant_ok as f64 / tot as f64,
    )
}

fn main() {
    let loads = [0.3f64, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0];
    println!("pinv-memory | D={D} V={V} seeds={SEEDS} | recall % — SOLVE the system, don't sum it");
    println!("(vs additive superposition which floors at ~0.3·D)\n");
    println!("{:>8}   {:>10} {:>12}", "M/D", "exact", "quantized");
    for &l in &loads {
        let m = ((l * D as f64) as usize).max(1);
        let (ex, qz) = recall(m);
        println!("{l:>8.2}   {:>9.0}% {:>11.0}%", ex, qz);
    }
    println!("\nStrong: ~100% to M≈D (3-5x the superposition floor), graceful past D.");
    println!("Kill: recall <90% below M=D -> the linear-associator premise is wrong here.");
}
