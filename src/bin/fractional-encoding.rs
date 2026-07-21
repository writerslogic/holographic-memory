// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Lane 1 (research): does fractional-power encoding (VFA/FPE) give the phasor
// substrate a notion of CONTINUOUS closeness -- the thing neither the sparse nor
// the discrete-relation phasor memory has? A writer's world is full of ordered,
// continuous quantities (dates, ages, chapter/scene order, distances); "close"
// must mean something.
//
// FPE: pick a random base phase vector `base in Z_N^D`. Encode a value `x` as
// `encode(x)[d] = (x * base[d]) mod N` -- raising the base phasor to the x-th
// power (phase scales with x). Then the holographic similarity between encodings
// is `sim(a,b) = mean_d cos(2*pi*(a-b)*base[d]/N)` -- a kernel that PEAKS at a=b
// and decays smoothly with |a-b|, so nearby values are similar and far ones are
// not. For integer-valued quantities the encoding is integer (deterministic).
//
// Two checks: (1) the kernel decays smoothly with value distance (random codes
// do not -- they have no ordering); (2) nearest-value retrieval works (query a
// value, recover the closest stored one). Run:
//   cargo run --release --bin fractional-encoding

const N: u32 = 256;
const D: usize = 2048;

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Base *frequencies* for the fractional encoder: small signed integers. The
/// kernel bandwidth is set by their magnitude -- values decorrelate over ~N/max
/// steps, so small frequencies give a wide, smooth kernel (uniform-over-[0,N)
/// frequencies would give a useless delta kernel). Here max |freq| = 4 gives a
/// kernel that decays over ~N/4 value units, matched to the demo's 0..100 range.
fn base(seed: u64) -> Vec<i64> {
    (0..D)
        .map(|d| {
            let v = (mix(d as u64, seed) % 9) as i64 - 4; // -4..=4
            if v == 0 {
                1
            } else {
                v
            }
        })
        .collect()
}

/// Fractional-power encoding of integer value `x`: phase scales with x.
fn encode_fpe(x: i64, base: &[i64]) -> Vec<u32> {
    base.iter()
        .map(|&b| (x * b).rem_euclid(N as i64) as u32)
        .collect()
}

/// A random (unordered) code for value `x` -- the baseline with no notion of near.
fn encode_rand(x: i64, seed: u64) -> Vec<u32> {
    (0..D)
        .map(|d| (mix((x as u64).wrapping_add(0xBEEF), seed ^ d as u64) % N as u64) as u32)
        .collect()
}

/// Holographic similarity: mean cosine of the per-dimension phase difference.
/// 1.0 for identical, ~0 for unrelated.
fn sim(a: &[u32], b: &[u32]) -> f64 {
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(&pa, &pb)| {
            let dphi = (pa as f64 - pb as f64) / N as f64;
            (std::f64::consts::TAU * dphi).cos()
        })
        .sum();
    s / D as f64
}

fn main() {
    let seeds = 8u64;

    // --- Kernel: similarity vs value distance ---
    println!("fractional-encoding | N={N} D={D} seeds={seeds}");
    println!("Kernel: similarity as value distance grows. FPE should decay smoothly;");
    println!("random codes stay ~0 (no notion of near).\n");
    println!("{:>6}  {:>14}  {:>14}", "|dx|", "FPE sim", "random sim");
    for dx in [0i64, 1, 2, 4, 8, 16, 32, 64] {
        let mut fpe = 0.0;
        let mut rnd = 0.0;
        for seed in 0..seeds {
            let b = base(seed);
            fpe += sim(&encode_fpe(0, &b), &encode_fpe(dx, &b));
            rnd += sim(&encode_rand(0, seed), &encode_rand(dx, seed));
        }
        println!(
            "{dx:>6}  {:>14.3}  {:>14.3}",
            fpe / seeds as f64,
            rnd / seeds as f64
        );
    }

    // --- Nearest-value retrieval: query a value, recover the closest stored one ---
    println!("\nNearest-value retrieval: stored values {{0,10,..,100}}; query -> closest.");
    let b = base(1);
    let stored: Vec<i64> = (0..=100).step_by(10).collect();
    let mut correct = 0;
    let queries = [7i64, 23, 48, 61, 94];
    for &q in &queries {
        let qv = encode_fpe(q, &b);
        let best = stored
            .iter()
            .copied()
            .max_by(|&x, &y| {
                sim(&encode_fpe(x, &b), &qv)
                    .partial_cmp(&sim(&encode_fpe(y, &b), &qv))
                    .unwrap()
            })
            .unwrap();
        let truth = *stored.iter().min_by_key(|&&x| (x - q).abs()).unwrap();
        println!("  query {q:>3} -> nearest stored {best:>3}  (true nearest {truth})");
        if best == truth {
            correct += 1;
        }
    }
    println!("nearest-value accuracy: {correct}/{}", queries.len());
    println!("\nKill: if the FPE kernel does not decay with distance, or nearest-value");
    println!("retrieval misses, fractional encoding buys no continuous structure.");
}
