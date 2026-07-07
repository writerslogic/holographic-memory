// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Capacity campaign, strongest cross-agent lead: the field Σ_i bind(key_i, obj_i)
// is a Sparse Superposition Code (SPARC) — M sections, one object active per
// section. Matched-filter decode is one linear pass (floors at ~0.12·D); AMP /
// soft interference-cancellation jointly estimates ALL facts and subtracts them
// from each other's residual, which single-fact cleanup (§23 Hopfield) cannot do.
// Tests: matched filter vs soft-IC (flat power) vs soft-IC (geometric power ladder,
// the SPARC capacity-achieving knob). Metric: fraction of M facts decoded, vs load.
//
// f64 linear complex field (the count-preserving substrate that makes CS/AMP work;
// phase-normalized would kill it). Run: cargo run --release --bin amp-decode

use std::f64::consts::TAU;

const D: usize = 1024;
const V: usize = 64; // object codebook
const SEEDS: u64 = 4;
const ITERS: usize = 14;
const BETA: f64 = 3.0;

type Cx = (Vec<f64>, Vec<f64>);

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

fn phasor(id: u64, salt: u64, seed: u64) -> Cx {
    let mut re = vec![0.0; D];
    let mut im = vec![0.0; D];
    for d in 0..D {
        let a = TAU * (mix(id.wrapping_mul(0x9E37) ^ (d as u64) ^ salt, seed) % 256) as f64 / 256.0;
        re[d] = a.cos();
        im[d] = a.sin();
    }
    (re, im)
}

/// bind = phase-add = complex product.
fn cmul(a: &Cx, b: &Cx) -> Cx {
    let mut r = vec![0.0; D];
    let mut i = vec![0.0; D];
    for d in 0..D {
        r[d] = a.0[d] * b.0[d] - a.1[d] * b.1[d];
        i[d] = a.0[d] * b.1[d] + a.1[d] * b.0[d];
    }
    (r, i)
}

/// Re<a, b> = Σ (a.re·b.re + a.im·b.im).
fn dot(a: &Cx, b: &Cx) -> f64 {
    let mut s = 0.0;
    for d in 0..D {
        s += a.0[d] * b.0[d] + a.1[d] * b.1[d];
    }
    s
}

struct Problem {
    atoms: Vec<Vec<Cx>>, // atoms[i][o] = bind(key_i, obj_o)
    amp: Vec<f64>,
    truth: Vec<usize>,
    field: Cx,
}

fn gen(m: usize, seed: u64, ladder: bool) -> Problem {
    let objs: Vec<Cx> = (0..V).map(|o| phasor(o as u64, 0xB0, seed)).collect();
    let atoms: Vec<Vec<Cx>> = (0..m)
        .map(|i| {
            let key = phasor(i as u64, 0xA0, seed);
            (0..V).map(|o| cmul(&key, &objs[o])).collect()
        })
        .collect();
    let truth: Vec<usize> = (0..m)
        .map(|i| (mix(i as u64, seed ^ 0xC0) % V as u64) as usize)
        .collect();
    // geometric power ladder (SPARC) or flat
    let amp: Vec<f64> = (0..m)
        .map(|i| {
            if ladder {
                2f64.powf(-(i as f64) / (m as f64 / 4.0).max(1.0))
            } else {
                1.0
            }
        })
        .collect();
    let mut field = (vec![0.0; D], vec![0.0; D]);
    for i in 0..m {
        let a = &atoms[i][truth[i]];
        for d in 0..D {
            field.0[d] += amp[i] * a.0[d];
            field.1[d] += amp[i] * a.1[d];
        }
    }
    Problem {
        atoms,
        amp,
        truth,
        field,
    }
}

/// Matched filter: per fact, argmax_o Re<field, atom_{i,o}>.
#[allow(clippy::needless_range_loop)]
fn matched(p: &Problem) -> usize {
    let mut ok = 0;
    for i in 0..p.truth.len() {
        let best = (0..V)
            .max_by(|&a, &b| {
                dot(&p.field, &p.atoms[i][a])
                    .partial_cmp(&dot(&p.field, &p.atoms[i][b]))
                    .unwrap()
            })
            .unwrap();
        ok += (best == p.truth[i]) as usize;
    }
    ok
}

/// Soft interference cancellation (AMP-lite): jointly estimate all facts, subtract
/// each other's soft reconstruction from the residual, re-score against residual+own.
#[allow(clippy::needless_range_loop)]
fn soft_ic(p: &Problem) -> usize {
    let m = p.truth.len();
    let mut w = vec![[0.0f64; V]; m]; // section soft weights
                                      // precompute nothing heavy; iterate
    for _ in 0..ITERS {
        // total soft reconstruction
        let mut recon = (vec![0.0; D], vec![0.0; D]);
        let mut contrib: Vec<Cx> = Vec::with_capacity(m);
        for i in 0..m {
            let mut ci = (vec![0.0; D], vec![0.0; D]);
            for o in 0..V {
                let wc = p.amp[i] * w[i][o];
                if wc != 0.0 {
                    let at = &p.atoms[i][o];
                    for d in 0..D {
                        ci.0[d] += wc * at.0[d];
                        ci.1[d] += wc * at.1[d];
                    }
                }
            }
            for d in 0..D {
                recon.0[d] += ci.0[d];
                recon.1[d] += ci.1[d];
            }
            contrib.push(ci);
        }
        let z = {
            let mut zr = vec![0.0; D];
            let mut zi = vec![0.0; D];
            for d in 0..D {
                zr[d] = p.field.0[d] - recon.0[d];
                zi[d] = p.field.1[d] - recon.1[d];
            }
            (zr, zi)
        };
        // re-score each section against residual + its own contribution
        for i in 0..m {
            let mut zi = (z.0.clone(), z.1.clone());
            for d in 0..D {
                zi.0[d] += contrib[i].0[d];
                zi.1[d] += contrib[i].1[d];
            }
            let mut sc = [0.0f64; V];
            for o in 0..V {
                sc[o] = p.amp[i] * dot(&zi, &p.atoms[i][o]);
            }
            let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
            let scale = (D as f64).sqrt();
            let mut zsum = 0.0;
            let mut ex = [0.0f64; V];
            for o in 0..V {
                ex[o] = (BETA * (sc[o] - mx) / scale).exp();
                zsum += ex[o];
            }
            for o in 0..V {
                w[i][o] = ex[o] / zsum;
            }
        }
    }
    let mut ok = 0;
    for i in 0..m {
        let best = (0..V)
            .max_by(|&a, &b| w[i][a].partial_cmp(&w[i][b]).unwrap())
            .unwrap();
        ok += (best == p.truth[i]) as usize;
    }
    ok
}

fn main() {
    let loads = [0.1f64, 0.2, 0.3, 0.5, 0.75, 1.0];
    println!("amp-decode | D={D} V={V} seeds={SEEDS} iters={ITERS} | fraction of M facts decoded, chance={:.1}%", 100.0 / V as f64);
    println!("field = Σ_i amp_i·bind(key_i,obj_i). Matched filter vs soft-IC joint decode.\n");
    println!(
        "{:>18}  {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "method \\ M/D", "0.10", "0.20", "0.30", "0.50", "0.75", "1.00"
    );
    let run = |name: &str, f: &dyn Fn(usize, u64) -> (usize, usize)| {
        print!("{name:>18}");
        for &l in &loads {
            let m = ((l * D as f64) as usize).max(1);
            let (mut ok, mut tot) = (0, 0);
            for s in 0..SEEDS {
                let (o, t) = f(m, s);
                ok += o;
                tot += t;
            }
            print!("  {:>4.0}%", 100.0 * ok as f64 / tot as f64);
        }
        println!();
    };
    run("matched-filter", &|m, s| {
        let p = gen(m, s, false);
        (matched(&p), m)
    });
    run("soft-IC (flat)", &|m, s| {
        let p = gen(m, s, false);
        (soft_ic(&p), m)
    });
    run("soft-IC (ladder)", &|m, s| {
        let p = gen(m, s, true);
        (soft_ic(&p), m)
    });
    println!("\nStrong: soft-IC pushes the reliable knee materially past matched filter (~0.3·D).");
    println!(
        "Kill: soft-IC <= matched filter at every load -> joint decode doesn't unlock capacity."
    );
}
