// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Capacity campaign — the nonlinear-escape lead (trigonometry lens). The linear
// matched filter is SNR-limited (~√(D/M)) and every LINEAR decoder is capped at the
// Donoho-Tanner wall (~0.4·D, universal). A SECOND-ORDER readout changes the
// measurement geometry: B[d,d'] = φ[d]·conj(φ[d']). Each fact's SELF-terms (i=j)
// contribute a coherent phase-difference signature Δ_i[d,d'] = (key_i[d]-key_i[d'])
// + (obj_i[d]-obj_i[d']) across P sampled pairs; cross-terms scatter. Matching B
// against the candidate object's signature gives SNR ~ √P/M, → D/M at P=D² — a √D
// improvement in the wall's exponent. Tests whether more pairs P push recall past
// the matched-filter floor at high load. Run: cargo run --release --bin bilinear-readout

use std::f64::consts::TAU;

const D: usize = 1024;
const N: u32 = 256;
const V: usize = 64;
const SEEDS: u64 = 3;
const QUERIES: usize = 24;

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

fn phase_vec(id: u64, salt: u64, seed: u64) -> Vec<u16> {
    (0..D)
        .map(|d| (mix(id.wrapping_mul(0x9E37) ^ (d as u64) ^ salt, seed) % N as u64) as u16)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run(seed: u64, m: usize, cos: &[f64], sin: &[f64]) -> (f64, [f64; 3]) {
    let nn = N as u16;
    let objs: Vec<Vec<u16>> = (0..V).map(|o| phase_vec(o as u64, 0xB0, seed)).collect();
    let keys: Vec<Vec<u16>> = (0..m).map(|i| phase_vec(i as u64, 0xA0, seed)).collect();
    let truth: Vec<usize> = (0..m)
        .map(|i| (mix(i as u64, seed ^ 0xC0) % V as u64) as usize)
        .collect();
    // linear field φ = Σ_i unitphasor(key_i + obj_i)
    let mut fre = vec![0.0; D];
    let mut fim = vec![0.0; D];
    for i in 0..m {
        for d in 0..D {
            let p = (keys[i][d] + objs[truth[i]][d]) % nn;
            fre[d] += cos[p as usize];
            fim[d] += sin[p as usize];
        }
    }
    // sampled pairs (fixed per seed)
    let p_counts = [D, D * 16, D * 128];
    let pmax = p_counts[2];
    let pairs: Vec<(usize, usize)> = (0..pmax)
        .map(|p| {
            let a = (mix(p as u64, seed ^ 0xE0) % D as u64) as usize;
            let b = (mix(p as u64 ^ 0x55, seed ^ 0xE1) % D as u64) as usize;
            (a, b.wrapping_add(1) % D) // ensure a != b mostly
        })
        .collect();
    // B[p] = φ[a]·conj(φ[b])
    let (mut bre, mut bim) = (vec![0.0; pmax], vec![0.0; pmax]);
    for p in 0..pmax {
        let (a, b) = pairs[p];
        bre[p] = fre[a] * fre[b] + fim[a] * fim[b];
        bim[p] = fim[a] * fre[b] - fre[a] * fim[b];
    }
    let step = (m / QUERIES).max(1);
    let (mut mf_ok, mut tot) = (0i64, 0i64);
    let mut bl_ok = [0i64; 3];
    for i in (0..m).step_by(step) {
        // matched filter: argmax_o Σ_d Re<φ, unitphasor(key_i+obj_o)>
        let mf = (0..V)
            .max_by(|&x, &y| {
                let sx = mf_score(&fre, &fim, &keys[i], &objs[x], nn, cos, sin);
                let sy = mf_score(&fre, &fim, &keys[i], &objs[y], nn, cos, sin);
                sx.partial_cmp(&sy).unwrap()
            })
            .unwrap();
        mf_ok += (mf == truth[i]) as i64;
        // bilinear at each P budget
        for (pi, &pc) in p_counts.iter().enumerate() {
            let best = (0..V)
                .max_by(|&x, &y| {
                    let sx = bl_score(&bre, &bim, &pairs, pc, &keys[i], &objs[x], nn, cos, sin);
                    let sy = bl_score(&bre, &bim, &pairs, pc, &keys[i], &objs[y], nn, cos, sin);
                    sx.partial_cmp(&sy).unwrap()
                })
                .unwrap();
            bl_ok[pi] += (best == truth[i]) as i64;
        }
        tot += 1;
    }
    let t = tot as f64;
    (
        100.0 * mf_ok as f64 / t,
        [
            100.0 * bl_ok[0] as f64 / t,
            100.0 * bl_ok[1] as f64 / t,
            100.0 * bl_ok[2] as f64 / t,
        ],
    )
}

fn mf_score(
    fre: &[f64],
    fim: &[f64],
    key: &[u16],
    obj: &[u16],
    n: u16,
    cos: &[f64],
    sin: &[f64],
) -> f64 {
    let mut s = 0.0;
    for d in 0..D {
        let p = (key[d] + obj[d]) % n;
        s += fre[d] * cos[p as usize] + fim[d] * sin[p as usize];
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn bl_score(
    bre: &[f64],
    bim: &[f64],
    pairs: &[(usize, usize)],
    pc: usize,
    key: &[u16],
    obj: &[u16],
    n: u16,
    cos: &[f64],
    sin: &[f64],
) -> f64 {
    let mut s = 0.0;
    for p in 0..pc {
        let (a, b) = pairs[p];
        // signature = (key[a]-key[b]) + (obj[a]-obj[b]) mod N
        let sig = ((key[a] as i32 - key[b] as i32 + obj[a] as i32 - obj[b] as i32)
            .rem_euclid(n as i32)) as usize;
        s += bre[p] * cos[sig] + bim[p] * sin[sig];
    }
    s
}

fn main() {
    let cos: Vec<f64> = (0..N).map(|p| (TAU * p as f64 / N as f64).cos()).collect();
    let sin: Vec<f64> = (0..N).map(|p| (TAU * p as f64 / N as f64).sin()).collect();
    let loads = [0.1f64, 0.2, 0.3, 0.5, 0.75, 1.0];
    println!(
        "bilinear-readout | D={D} N={N} V={V} seeds={SEEDS} | top-1 recall %, chance={:.1}%",
        100.0 / V as f64
    );
    println!(
        "matched-filter (linear) vs bilinear second-order readout at P pairs (D, 16D, 128D)\n"
    );
    println!(
        "{:>8}   {:>7} {:>8} {:>8} {:>8}",
        "M/D", "MF", "BL(D)", "BL(16D)", "BL(128D)"
    );
    for &l in &loads {
        let m = ((l * D as f64) as usize).max(1);
        let (mut mf, mut bl) = (0.0, [0.0; 3]);
        for s in 0..SEEDS {
            let (mfs, bls) = run(s, m, &cos, &sin);
            mf += mfs;
            for k in 0..3 {
                bl[k] += bls[k];
            }
        }
        let sd = SEEDS as f64;
        println!(
            "{l:>8.2}   {:>6.0}% {:>7.0}% {:>7.0}% {:>7.0}%",
            mf / sd,
            bl[0] / sd,
            bl[1] / sd,
            bl[2] / sd
        );
    }
    println!("\nStrong: bilinear recall rises with P and holds past 0.5·D where MF is dead ->");
    println!("nonlinear second-order readout beats the linear wall. Kill: BL tracks MF at all P.");
}
