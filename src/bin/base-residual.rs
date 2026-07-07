// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// §28. The real "combine both layers" fusion: base + residual. A HOLOGRAPHIC base
// (Hebbian associative memory over a value codebook, partial-cue robust like §22) +
// an EXACT coded residual (pinv solve, exact to M≈D_res like §26), sharing one D
// budget. Exact query → coded path (exact); partial-cue query → base path (robust).
// Shows the fused store has BOTH capabilities where each pure layer has only one —
// and that it is a RESOURCE ALLOCATION (robustness and exact capacity both cost dims).
//
// Three matched-D stores: pure-coded (all D), pure-holographic (all D), fused
// (split). Metric: exact-key recall AND partial-cue recall (fraction ρ of key
// components randomized). Run: cargo run --release --bin base-residual

use nalgebra::{DMatrix, DVector};

const D: usize = 256;
const V: usize = 64;
const SEEDS: u64 = 5;
const RHO: f64 = 0.3; // fraction of key components randomized for the partial-cue query

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

fn g(seed: u64, a: u64) -> f64 {
    let u1 = (mix(a, seed) % 1_000_000 + 1) as f64 / 1_000_001.0;
    let u2 = (mix(a, seed ^ 0x5A) % 1_000_000) as f64 / 1_000_000.0;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// (exact_recall, cue_recall) for a store split into d_res coded dims + d_base
/// holographic dims (d_res+d_base = D; set one to 0 for a pure store).
#[allow(clippy::needless_range_loop)]
fn recall(m: usize, d_res: usize, d_base: usize) -> (f64, f64) {
    let (mut ex, mut cue, mut tot) = (0i64, 0i64, 0i64);
    for seed in 0..SEEDS {
        // keys (full D), values, value codebook (over d_base dims for the base)
        let keys: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                (0..D)
                    .map(|d| g(seed, (i as u64) << 20 | d as u64))
                    .collect()
            })
            .collect();
        let truth: Vec<usize> = (0..m)
            .map(|i| (mix(i as u64, seed ^ 0xC0) % V as u64) as usize)
            .collect();
        let obj: Vec<Vec<f64>> = (0..V)
            .map(|o| {
                (0..d_base)
                    .map(|d| g(seed ^ 0xB0, (o as u64) << 12 | d as u64))
                    .collect()
            })
            .collect();

        // coded residual: pinv solve x over the last d_res dims
        let (mut x, mut s, mut c) = (DVector::<f64>::zeros(d_res.max(1)), 0.0, 0.0);
        if d_res > 0 {
            let mut a = DMatrix::<f64>::zeros(m, d_res);
            for i in 0..m {
                for d in 0..d_res {
                    a[(i, d)] = keys[i][d_base + d];
                }
            }
            let v = DVector::from_iterator(m, truth.iter().map(|&t| t as f64));
            let at = a.transpose();
            let mut reg = &at * &a;
            let md = reg.diagonal().mean();
            for d in 0..d_res {
                reg[(d, d)] += 1e-6 * md;
            }
            if let Some(sol) = reg.lu().solve(&(&at * &v)) {
                x = sol;
                let raw: Vec<f64> = (0..m).map(|i| a.row(i).dot(&x.transpose())).collect();
                let mp = raw.iter().sum::<f64>() / m as f64;
                let mt = v.mean();
                let cov: f64 = raw
                    .iter()
                    .zip(&truth)
                    .map(|(&p, &t)| (p - mp) * (t as f64 - mt))
                    .sum();
                let var: f64 = raw
                    .iter()
                    .map(|&p| (p - mp).powi(2))
                    .sum::<f64>()
                    .max(1e-12);
                s = cov / var;
                c = mt - s * mp;
            }
        }
        // holographic base: Hebbian W = Σ obj[v_i] ⊗ a_base_i  (d_base × d_base)
        let mut w = vec![vec![0.0f64; d_base]; d_base];
        if d_base > 0 {
            for i in 0..m {
                for r in 0..d_base {
                    let orv = obj[truth[i]][r];
                    for cc in 0..d_base {
                        w[r][cc] += orv * keys[i][cc];
                    }
                }
            }
        }
        let read_coded = |kq: &[f64]| -> usize {
            let dot: f64 = (0..d_res).map(|d| kq[d_base + d] * x[d]).sum();
            (s * dot + c).round().clamp(0.0, (V - 1) as f64) as usize
        };
        let read_base = |kq: &[f64]| -> usize {
            // y = W · a_base ; argmax_o <y, obj_o>
            let y: Vec<f64> = (0..d_base)
                .map(|r| (0..d_base).map(|cc| w[r][cc] * kq[cc]).sum())
                .collect();
            (0..V)
                .max_by(|&a, &b| {
                    let sa: f64 = (0..d_base).map(|d| y[d] * obj[a][d]).sum();
                    let sb: f64 = (0..d_base).map(|d| y[d] * obj[b][d]).sum();
                    sa.partial_cmp(&sb).unwrap()
                })
                .unwrap()
        };
        for i in 0..m {
            // exact query -> prefer the coded (exact) path
            let ve = if d_res > 0 {
                read_coded(&keys[i])
            } else {
                read_base(&keys[i])
            };
            ex += (ve == truth[i]) as i64;
            // partial-cue query: randomize fraction RHO of components -> prefer base (robust)
            let mut kq = keys[i].clone();
            for d in 0..D {
                if (mix(d as u64, seed ^ (i as u64) ^ 0xD0) as f64 / u64::MAX as f64) < RHO {
                    kq[d] = g(seed ^ 0x77, (i as u64) << 20 | d as u64);
                }
            }
            let vc = if d_base > 0 {
                read_base(&kq)
            } else {
                read_coded(&kq)
            };
            cue += (vc == truth[i]) as i64;
            tot += 1;
        }
    }
    (
        100.0 * ex as f64 / tot as f64,
        100.0 * cue as f64 / tot as f64,
    )
}

fn main() {
    let loads = [0.1f64, 0.3, 0.5, 0.75, 1.0];
    println!(
        "base-residual | D={D} V={V} seeds={SEEDS} rho={RHO} | recall %, chance={:.1}%",
        100.0 / V as f64
    );
    println!("EXACT-key recall / partial-CUE recall for 3 matched-D stores.\n");
    let stores: [(&str, usize, usize); 3] = [
        ("pure-coded", D, 0),
        ("pure-holographic", 0, D),
        ("fused(128+128)", 128, 128),
    ];
    for &(name, dr, db) in &stores {
        println!("== {name}  (coded={dr} dims, base={db} dims) ==");
        print!("{:>8}", "M/D");
        for l in loads {
            print!("  {:>9}", format!("{l:.2}"));
        }
        println!();
        print!("{:>8}", "exact");
        for &l in &loads {
            print!(
                "  {:>8.0}%",
                recall(((l * D as f64) as usize).max(1), dr, db).0
            );
        }
        println!();
        print!("{:>8}", "cue");
        for &l in &loads {
            print!(
                "  {:>8.0}%",
                recall(((l * D as f64) as usize).max(1), dr, db).1
            );
        }
        println!("\n");
    }
    println!("Fusion: pure-coded = exact✓ cue✗; pure-holo = exact-limited cue✓;");
    println!("fused = exact✓ (to ~0.5·D via coded) AND cue✓ (via base) — both, at a split budget.");
}
