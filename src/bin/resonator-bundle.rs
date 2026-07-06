// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// §21. §20 factored a SINGLE clean composite; a memory holds MANY facts superposed.
// This tests whether the deterministic resonator recovers facts from a BUNDLE of B
// bound products (Σ bind(x,y,z)) by explaining-away -- factor the residual field,
// subtract the recovered product, repeat -- and whether phase QUANTIZATION costs
// more here (bundle interference compounding with quantization noise) than in the
// free single-composite case.
//
// Codebooks are quantized to N (N=0 = float baseline); the working field is a float
// complex accumulator (query-time state). Metric: set-recovery |recovered ∩ stored|/B.
// The hot loop is allocation-free (reusable scratch buffers) -- resonator inner
// steps otherwise churn D-length vectors and dominate runtime.
//
// Run: cargo run --release --bin resonator-bundle

use std::collections::HashSet;
use std::f64::consts::TAU;

const D: usize = 1024;
const FACTORS: usize = 3;
const F: usize = 16; // codebook size per axis
const ITERS: usize = 20;
const SEEDS: u64 = 3;
const TRIALS: usize = 2;

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Unit-phasor vector (re, im) for codebook item `id` on `axis`. `n==0` = float.
fn phase_vec(id: usize, axis: usize, seed: u64, n: u32) -> (Vec<f64>, Vec<f64>) {
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

/// Snap slices to unit phasors, quantizing phase to `n` levels (qFHRR recover).
fn snap(re: &mut [f64], im: &mut [f64], n: u32) {
    for d in 0..re.len() {
        let ang = im[d].atan2(re[d]);
        let p = if n == 0 {
            ang
        } else {
            TAU * ((ang / TAU * n as f64).round() as i64).rem_euclid(n as i64) as f64 / n as f64
        };
        re[d] = p.cos();
        im[d] = p.sin();
    }
}

fn dot(ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64]) -> f64 {
    let mut s = 0.0;
    for d in 0..ar.len() {
        s += ar[d] * br[d] + ai[d] * bi[d];
    }
    s
}

/// dst ⊙= conj(e), elementwise complex multiply-by-conjugate, in place.
fn mul_conj_inplace(dr: &mut [f64], di: &mut [f64], er: &[f64], ei: &[f64]) {
    for d in 0..dr.len() {
        let (r, i) = (dr[d], di[d]);
        dr[d] = r * er[d] + i * ei[d];
        di[d] = i * er[d] - r * ei[d];
    }
}

/// Reusable scratch to keep the resonator inner loop allocation-free.
struct Scratch {
    ubr: Vec<f64>,
    ubi: Vec<f64>,
    sims: Vec<f64>,
    est: Vec<(Vec<f64>, Vec<f64>)>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            ubr: vec![0.0; D],
            ubi: vec![0.0; D],
            sims: vec![0.0; F],
            est: (0..FACTORS).map(|_| (vec![0.0; D], vec![0.0; D])).collect(),
        }
    }
}

type Book = Vec<(Vec<f64>, Vec<f64>)>;

/// Factor a (possibly noisy) composite field into one index per axis.
fn resonate(fr: &[f64], fi: &[f64], books: &[Book], n: u32, s: &mut Scratch) -> Vec<usize> {
    // init each estimate = snapped superposition of its codebook
    for (f, cb) in books.iter().enumerate() {
        let (er, ei) = &mut s.est[f];
        er.iter_mut().for_each(|x| *x = 0.0);
        ei.iter_mut().for_each(|x| *x = 0.0);
        for c in cb {
            for d in 0..D {
                er[d] += c.0[d];
                ei[d] += c.1[d];
            }
        }
        snap(er, ei, n);
    }
    let mut prev = vec![usize::MAX; FACTORS];
    let mut stable = 0;
    for _ in 0..ITERS {
        for f in 0..FACTORS {
            s.ubr.copy_from_slice(fr);
            s.ubi.copy_from_slice(fi);
            for (j, (er, ei)) in s.est.iter().enumerate() {
                if j != f {
                    mul_conj_inplace(&mut s.ubr, &mut s.ubi, er, ei);
                }
            }
            // cleanup: est[f] = snap(Σ_k <cb_k, ub> cb_k)
            for (k, c) in books[f].iter().enumerate() {
                s.sims[k] = dot(&c.0, &c.1, &s.ubr, &s.ubi);
            }
            let (er, ei) = &mut s.est[f];
            er.iter_mut().for_each(|x| *x = 0.0);
            ei.iter_mut().for_each(|x| *x = 0.0);
            for (k, c) in books[f].iter().enumerate() {
                let w = s.sims[k];
                for d in 0..D {
                    er[d] += w * c.0[d];
                    ei[d] += w * c.1[d];
                }
            }
            snap(er, ei, n);
        }
        let cur: Vec<usize> = (0..FACTORS)
            .map(|f| {
                let (er, ei) = &s.est[f];
                let (mut best, mut best_s) = (0usize, f64::NEG_INFINITY);
                for (k, c) in books[f].iter().enumerate() {
                    let sc = dot(&c.0, &c.1, er, ei);
                    if sc > best_s {
                        best_s = sc;
                        best = k;
                    }
                }
                best
            })
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
    prev
}

/// Multiply factor phasors of `triple` into (dr, di) buffers.
fn product_into(books: &[Book], triple: &[usize], dr: &mut [f64], di: &mut [f64]) {
    dr.copy_from_slice(&books[0][triple[0]].0);
    di.copy_from_slice(&books[0][triple[0]].1);
    for axis in 1..FACTORS {
        let (cr, ci) = &books[axis][triple[axis]];
        for d in 0..D {
            let (r, i) = (dr[d], di[d]);
            dr[d] = r * cr[d] - i * ci[d];
            di[d] = r * ci[d] + i * cr[d];
        }
    }
}

fn recovery(bundle_size: usize, n: u32) -> f64 {
    let mut scratch = Scratch::new();
    let (mut pr, mut pi) = (vec![0.0; D], vec![0.0; D]);
    let mut rate_sum = 0.0;
    let mut count = 0;
    for seed in 0..SEEDS {
        let books: Vec<Book> = (0..FACTORS)
            .map(|axis| (0..F).map(|i| phase_vec(i, axis, seed, n)).collect())
            .collect();
        for t in 0..TRIALS {
            let mut stored: HashSet<Vec<usize>> = HashSet::new();
            while stored.len() < bundle_size {
                let key = (t as u64) << 32 | stored.len() as u64;
                let triple: Vec<usize> = (0..FACTORS)
                    .map(|axis| (mix(key, seed ^ (0x30 + axis as u64)) % F as u64) as usize)
                    .collect();
                stored.insert(triple);
            }
            let (mut field_r, mut field_i) = (vec![0.0; D], vec![0.0; D]);
            for fact in &stored {
                product_into(&books, fact, &mut pr, &mut pi);
                for d in 0..D {
                    field_r[d] += pr[d];
                    field_i[d] += pi[d];
                }
            }
            let mut matched: HashSet<Vec<usize>> = HashSet::new();
            for _ in 0..bundle_size {
                let rec = resonate(&field_r, &field_i, &books, n, &mut scratch);
                if stored.contains(&rec) {
                    matched.insert(rec.clone());
                }
                product_into(&books, &rec, &mut pr, &mut pi);
                for d in 0..D {
                    field_r[d] -= pr[d];
                    field_i[d] -= pi[d];
                }
            }
            rate_sum += matched.len() as f64 / bundle_size as f64;
            count += 1;
        }
    }
    100.0 * rate_sum / count as f64
}

fn main() {
    let bs = [1usize, 4, 8, 16];
    let ns: [(u32, &str); 3] = [(0, "float"), (256, "N=256"), (16, "N=16(4bit)")];
    println!("resonator-bundle | D={D} factors={FACTORS} F={F} iters={ITERS} seeds={SEEDS} trials={TRIALS}");
    println!("bundled 3-factor set-recovery % (|recovered ∩ stored|/B), explaining-away\n");
    print!("{:>8}", "B facts");
    for (_, label) in ns {
        print!("  {label:>11}");
    }
    println!();
    for b in bs {
        print!("{b:>8}");
        for (n, _) in ns {
            print!("  {:>10.0}%", recovery(b, n));
        }
        println!();
    }
    println!("\nStrong: quantized tracks float across B -> quantization free even under bundle");
    println!("load -> deterministic bundled factorization is real. Kill: quantized falls below");
    println!("float as B grows -> bundling is where quantization costs; gate the prod unbundle.");
}
