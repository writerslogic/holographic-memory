// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Pre-registration step 1 (cheapest disconfirming test) from
// docs/PREREGISTRATION-binding-readout.md.
//
// Open question: is the self-inverse XOR bind the bottleneck for compositional
// (role-filler) discrimination as memory load grows?
//
// On this sparse substrate the binding operator and the readout are coupled:
// each binding has a matched query path, so the fair comparison is stack-vs-
// stack (each approach at its best), not one binding forced through the other's
// readout.
//   B0 = self-inverse XOR bind, NATIVE stack: role_hv XOR filler, majority-vote
//        bundle, query by XOR-unbind(role) then Jaccard vs candidate filler.
//   X  = non-self-inverse permutation, NATIVE stack: filler.hash_permute(role),
//        bloom-union bundle, query by density-corrected containment.
// Same D, same per-symbol density, shared filler codebook (real interference).
// We report d' AND bundle active-count. A kill is X failing to clear B0 near the
// knee; a win still needs the HRR/MAP baselines and density control in step 2.
//
// Metric: d' between correct-binding and mis-binding probe scores, swept over
// load N, averaged over seeds. Chance floor: d' = 0.
//
// Run: cargo run --release --bin binding-discriminator

use holographic_memory::EntangledHVec;

const DIM: usize = 16_384;
const DENSITY_DENOM: usize = 256; // active count per symbol ~ DIM/256 = 64
                                  // Large shared codebook: interference comes from bloom-bundle saturation
                                  // (superposition noise), while a mis-binding probe (role1, b) stays a genuine
                                  // non-member -- it is present only if some other fact coincidentally bound the
                                  // same symbol to role1, which is rare at this codebook size. Unique-per-fact
                                  // symbols remain banned (they would make the task trivial by construction).
const N_SYMBOLS: usize = 2048;
const ROLE1_XOR_SEED: u64 = 0x1111_1111;
const ROLE2_XOR_SEED: u64 = 0x2222_2222;
const ROLE1_PERM_SEED: u64 = 0xA5A5_A5A5;
const ROLE2_PERM_SEED: u64 = 0x5A5A_5A5A;
const SEEDS: u64 = 8;
const LOADS: &[usize] = &[5, 10, 20, 40, 80, 160, 320, 640];

/// A fact is two role-filler pairs over the SHARED codebook: (role1, a), (role2, b).
struct Fact {
    a: usize,
    b: usize,
}

fn build_facts(n: usize, seed: u64) -> Vec<Fact> {
    (0..n)
        .map(|i| {
            let a = (mix(i as u64, seed ^ 0xF11E) % N_SYMBOLS as u64) as usize;
            let mut b = (mix(i as u64, seed ^ 0xB0B0) % N_SYMBOLS as u64) as usize;
            if b == a {
                b = (b + 1) % N_SYMBOLS;
            }
            Fact { a, b }
        })
        .collect()
}

/// Cheap deterministic mixing hash for reproducible symbol selection.
fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Deterministic thinning to `k` active indices by pseudo-random hash rank.
/// Pure in (hv, k), so a stored pair and its query thin to the same set. Used to
/// density-match XOR-bound pairs (~2k active) down to the permutation density k.
fn thin(hv: &EntangledHVec, k: usize) -> EntangledHVec {
    let mut idx: Vec<u32> = hv.indices().to_vec();
    if idx.len() <= k {
        return hv.clone();
    }
    idx.sort_by_key(|&i| mix(i as u64, 0x7417_5A17));
    idx.truncate(k);
    idx.sort_unstable();
    EntangledHVec::from_indices(idx, DIM)
}

fn dprime(correct: &[f64], misbound: &[f64]) -> f64 {
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var = |v: &[f64], m: f64| v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    let mc = mean(correct);
    let mm = mean(misbound);
    let pooled = 0.5 * (var(correct, mc) + var(misbound, mm));
    if pooled <= f64::EPSILON {
        // Degenerate: identical-variance separation. Report a large but finite
        // sentinel rather than infinity so averages stay meaningful.
        return if (mc - mm).abs() <= f64::EPSILON {
            0.0
        } else {
            (mc - mm) / 1e-6
        };
    }
    (mc - mm) / pooled.sqrt()
}

/// One seed of the XOR-bind (B0) system at load n, using XOR's NATIVE stack:
/// majority-vote bundling + XOR-unbind + Jaccard readout. This is the fair
/// incumbent -- querying by unbinding the role from the bundle and comparing the
/// residue to the candidate filler, not by set containment (which suits
/// permutation, not XOR).
fn run_b0(n: usize, seed: u64, codebook: &[EntangledHVec]) -> f64 {
    let role1 = EntangledHVec::new_with_density(DIM, DENSITY_DENOM, ROLE1_XOR_SEED ^ seed);
    let role2 = EntangledHVec::new_with_density(DIM, DENSITY_DENOM, ROLE2_XOR_SEED ^ seed);
    let facts = build_facts(n, seed);

    let mut pairs = Vec::with_capacity(2 * n);
    for f in &facts {
        pairs.push(role1.bind(&codebook[f.a]));
        pairs.push(role2.bind(&codebook[f.b]));
    }
    let bundle = EntangledHVec::bundle(&pairs);

    let mut correct = Vec::with_capacity(n);
    let mut misbound = Vec::with_capacity(n);
    for f in &facts {
        // Unbind role1 from the bundle (XOR), then compare the residue to the
        // candidate filler. correct = a (truly bound to role1), mis = b.
        let residue = bundle.bind(&role1);
        correct.push(residue.similarity(&codebook[f.a]));
        misbound.push(residue.similarity(&codebook[f.b]));
    }
    dprime(&correct, &misbound)
}

/// Density-matched XOR control: XOR bind run through X's EXACT stack -- each
/// XOR-bound pair thinned to k active indices (matching permutation density),
/// bloom bundle, corrected-containment readout. The ONLY difference from `run_x`
/// is the bind operator (XOR-thinned vs permutation), at matched density and
/// matched readout. If X still clears this, the advantage is the bind operator,
/// not the density edge or the readout choice.
fn run_b0_thin(n: usize, seed: u64, codebook: &[EntangledHVec]) -> f64 {
    let role1 = EntangledHVec::new_with_density(DIM, DENSITY_DENOM, ROLE1_XOR_SEED ^ seed);
    let role2 = EntangledHVec::new_with_density(DIM, DENSITY_DENOM, ROLE2_XOR_SEED ^ seed);
    let facts = build_facts(n, seed);
    let k = DIM / DENSITY_DENOM;

    let mut pairs = Vec::with_capacity(2 * n);
    for f in &facts {
        pairs.push(thin(&role1.bind(&codebook[f.a]), k));
        pairs.push(thin(&role2.bind(&codebook[f.b]), k));
    }
    let bundle = EntangledHVec::bundle_bloom(&pairs);

    let mut correct = Vec::with_capacity(n);
    let mut misbound = Vec::with_capacity(n);
    for f in &facts {
        correct.push(thin(&role1.bind(&codebook[f.a]), k).corrected_containment(&bundle));
        misbound.push(thin(&role1.bind(&codebook[f.b]), k).corrected_containment(&bundle));
    }
    dprime(&correct, &misbound)
}

/// One seed of the permutation-bind (X) system at load n.
fn run_x(n: usize, seed: u64, codebook: &[EntangledHVec]) -> f64 {
    let r1 = ROLE1_PERM_SEED ^ seed;
    let r2 = ROLE2_PERM_SEED ^ seed;
    let facts = build_facts(n, seed);

    let mut pairs = Vec::with_capacity(2 * n);
    for f in &facts {
        pairs.push(codebook[f.a].hash_permute(r1));
        pairs.push(codebook[f.b].hash_permute(r2));
    }
    let bundle = EntangledHVec::bundle_bloom(&pairs);

    let mut correct = Vec::with_capacity(n);
    let mut misbound = Vec::with_capacity(n);
    for f in &facts {
        correct.push(
            codebook[f.a]
                .hash_permute(r1)
                .corrected_containment(&bundle),
        );
        misbound.push(
            codebook[f.b]
                .hash_permute(r1)
                .corrected_containment(&bundle),
        );
    }
    dprime(&correct, &misbound)
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, sd)
}

fn main() {
    println!(
        "binding-discriminator | D={DIM} density=1/{DENSITY_DENOM} symbols={N_SYMBOLS} seeds={SEEDS}"
    );
    println!("d' = discrimination of correct-binding vs mis-binding (chance floor 0).");
    println!("B0nat = XOR native stack (majority+unbind). B0thin = XOR thinned to k through");
    println!("X's bloom+containment stack (density- and readout-matched to X).\n");
    println!(
        "{:>5}  {:>16}  {:>16}  {:>16}",
        "N", "B0nat XOR d'(sd)", "B0thin XOR d'(sd)", "X perm  d'(sd)"
    );

    for &n in LOADS {
        // The codebook is shared across systems and seeds within a load so the
        // only difference between systems is the binding operator (and, for
        // B0nat, its native readout).
        let mut b0_d = Vec::with_capacity(SEEDS as usize);
        let mut b0t_d = Vec::with_capacity(SEEDS as usize);
        let mut x_d = Vec::with_capacity(SEEDS as usize);

        for seed in 0..SEEDS {
            let codebook: Vec<EntangledHVec> = (0..N_SYMBOLS)
                .map(|i| EntangledHVec::new_with_density(DIM, DENSITY_DENOM, mix(i as u64, seed)))
                .collect();
            b0_d.push(run_b0(n, seed, &codebook));
            b0t_d.push(run_b0_thin(n, seed, &codebook));
            x_d.push(run_x(n, seed, &codebook));
        }

        let (b0m, b0s) = mean_sd(&b0_d);
        let (b0tm, b0ts) = mean_sd(&b0t_d);
        let (xm, xs) = mean_sd(&x_d);
        println!(
            "{n:>5}  {:>10.2} ({:>4.2})  {:>10.2} ({:>4.2})  {:>10.2} ({:>4.2})",
            b0m, b0s, b0tm, b0ts, xm, xs
        );
    }

    println!("\nDensity-matched control: B0thin gives XOR X's exact density (k active/pair)");
    println!("and readout (bloom+containment). If X clears B0thin, the advantage is the bind");
    println!("operator itself, not density or readout. B0nat is XOR at its own native best.");
}
