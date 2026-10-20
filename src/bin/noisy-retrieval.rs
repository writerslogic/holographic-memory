// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// §22. A multi-model fusion pass concluded that 0.5·D well-posed retrieval from a
// single dense superposition is information-theoretically impossible, so capacity
// must come from sharding — which HMS already does. For EXACT (S,R) queries a plain
// hash-to-shard KV store suffices; the holographic/VSA layer earns its keep ONLY on
// NOISY / partial-cue queries. This is the sharpest test of HMS's premise: does VSA
// similarity retrieval beat exact hash once the query key is corrupted?
//
// Store M facts (random key ⊗ codebook object) superposed in one fixed-point complex
// field; query the true key with a fraction ρ of components randomized; recover the
// object by argmax over the codebook. Exact-hash baseline = 1.0 at ρ=0, 0 at ρ>0.
//
// Run: cargo run --release --bin noisy-retrieval

use std::f64::consts::TAU;

const D: usize = 1024;
const O: usize = 64; // object codebook size; chance = 1/O
const SEEDS: u64 = 6;
const QUERIES: usize = 40; // sampled facts queried per (config, seed)
const SCALE: i64 = 1 << 14;

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

fn luts(n: u32) -> (Vec<i64>, Vec<i64>) {
    let c = (0..n)
        .map(|k| (SCALE as f64 * (TAU * k as f64 / n as f64).cos()).round() as i64)
        .collect();
    let s = (0..n)
        .map(|k| (SCALE as f64 * (TAU * k as f64 / n as f64).sin()).round() as i64)
        .collect();
    (c, s)
}

/// Random phase vector in Z_N, deterministic in (id, salt, seed).
fn phasor(id: u64, salt: u64, seed: u64, n: u32) -> Vec<u16> {
    (0..D)
        .map(|d| (mix(id.wrapping_mul(0x9E37) ^ (d as u64) ^ salt, seed) % n as u64) as u16)
        .collect()
}

fn bind(a: &[u16], b: &[u16], n: u16) -> Vec<u16> {
    a.iter().zip(b).map(|(&x, &y)| (x + y) % n).collect()
}

/// VSA top-1 object recall over the (load, corruption) config.
fn vsa_recall(m: usize, rho: f64, n: u32, cos: &[i64], sin: &[i64]) -> f64 {
    let nn = n as u16;
    let (mut hits, mut total) = (0i64, 0i64);
    for seed in 0..SEEDS {
        // object codebook
        let objs: Vec<Vec<u16>> = (0..O).map(|o| phasor(o as u64, 0xB0, seed, n)).collect();
        // facts: random key phasor bound to a random object id
        let keys: Vec<Vec<u16>> = (0..m).map(|i| phasor(i as u64, 0xA0, seed, n)).collect();
        let fact_obj: Vec<usize> = (0..m)
            .map(|i| (mix(i as u64, seed ^ 0xC0) % O as u64) as usize)
            .collect();
        // superpose bind(key_i, obj) into one complex field
        let mut re = vec![0i64; D];
        let mut im = vec![0i64; D];
        for i in 0..m {
            let phi = bind(&keys[i], &objs[fact_obj[i]], nn);
            for d in 0..D {
                re[d] += cos[phi[d] as usize];
                im[d] += sin[phi[d] as usize];
            }
        }
        // query a sample of facts with partial-cue corruption
        let step = (m / QUERIES).max(1);
        for i in (0..m).step_by(step) {
            let mut q = keys[i].clone();
            for (d, qd) in q.iter_mut().enumerate() {
                if (mix(d as u64, seed ^ (i as u64) ^ 0xD0) as f64 / u64::MAX as f64) < rho {
                    *qd = (mix(d as u64 ^ 0x77, seed ^ (i as u64)) % n as u64) as u16;
                }
            }
            // argmax over codebook of Re<field, bind(q, obj_o)>
            let best = (0..O)
                .max_by_key(|&o| {
                    let probe = bind(&q, &objs[o], nn);
                    let mut sc = 0i128;
                    for d in 0..D {
                        sc += (cos[probe[d] as usize] as i128) * (re[d] as i128)
                            + (sin[probe[d] as usize] as i128) * (im[d] as i128);
                    }
                    sc
                })
                .unwrap();
            hits += (best == fact_obj[i]) as i64;
            total += 1;
        }
    }
    100.0 * hits as f64 / total as f64
}

fn main() {
    let loads = [0.05f64, 0.1, 0.2, 0.3];
    let rhos = [0.0f64, 0.1, 0.2, 0.4];
    println!(
        "noisy-retrieval | D={D} O={O} seeds={SEEDS} | top-1 object recall %, chance={:.1}%",
        100.0 / O as f64
    );
    println!("exact hash-KV baseline = 100% at rho=0, 0% at any rho>0.\n");
    for &n in &[256u32, 16] {
        let (cos, sin) = luts(n);
        println!("== N={n} ==   (rows = load M/D, cols = key corruption rho)");
        print!("{:>10}", "load\\rho");
        for r in rhos {
            print!("  {:>6}", format!("{r:.1}"));
        }
        println!();
        for &load in &loads {
            let m = ((load * D as f64) as usize).max(1);
            print!("{:>10}", format!("{load:.2}(M={m})"));
            for &rho in &rhos {
                print!("  {:>5.0}%", vsa_recall(m, rho, n, &cos, &sin));
            }
            println!();
        }
        println!();
    }
    println!("VSA earns its keep in the rho>0 region (where exact hash = 0). Kill: if VSA");
    println!("collapses to ~chance at rho=0.2, load<=0.2 -> no noise tolerance, HMS ~ sharded KV.");
}
