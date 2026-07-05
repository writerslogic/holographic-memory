// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Pre-registration step 2 (strong baselines) from
// docs/PREREGISTRATION-binding-readout.md.
//
// The sparse permutation stack (P) survived steps 1 and the density/involution
// controls, beating HMS's set-XOR bind 3.5x-30x. But beating XOR only re-confirms
// settled "BSC is weak on sparse". The citable question is whether P is
// competitive with the STRONG incumbents the field actually uses: HRR (Plate,
// circular convolution) and MAP (Gayler, elementwise product). This harness runs
// all three on the identical mis-binding discriminator at matched dimensionality.
//
// All three use the SAME task and the SAME query TYPE -- membership: compose the
// query (role, filler) pair and score it against the bundle (correct pair is a
// member, mis-bound pair is not). This is the fair matched readout for a
// verification discriminator and the only readout P supports; HRR/MAP ALSO
// support retrieval (unbind->reconstruct the filler), a capability P lacks, but
// that is a qualitative advantage noted separately, not scored here.
//   P   = sparse permutation bind (idx^mask), bloom bundle, corrected containment.
//         The HMS-substrate candidate (~D/256 active indices per vector).
//   HRR = real dense N(0,1/D), circular-convolution bind, cosine-to-bundle.
//   MAP = bipolar dense {-1,+1}, elementwise-product bind, cosine-to-bundle.
//
// MATCHED-D is the VSA-literature convention, but it flatters the dense systems:
// at D they use far more bits/vector than the sparse stack (HRR: D f64; MAP: D
// signs; P: ~D/256 indices). This is a real caveat, printed with the result --
// a sparse win at matched D is a win at a fraction of the storage; a sparse loss
// is expected-given-more-bits and only interesting if small.
//
// Metric: d' between correct-binding and mis-binding probe scores vs load N.
// Run: cargo run --release --bin binding-baselines   (release: FFT-bound)

use holographic_memory::EntangledHVec;

const D: usize = 2048; // power of two (radix-2 FFT); matched across all systems
const SPARSE_DENOM: usize = 256; // sparse active count = D/256
const N_SYMBOLS: usize = 1024; // shared codebook (real interference, non-trivial)
const SEEDS: u64 = 8;
const LOADS: &[usize] = &[5, 10, 20, 40, 80, 160, 320];

// ---- deterministic PRNG (splitmix64) + samplers ---------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        // uniform in [0,1)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gaussian(&mut self) -> f64 {
        // Box-Muller
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

// ---- radix-2 iterative FFT (in-place) -------------------------------------

fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = if inverse { 1.0 } else { -1.0 } * std::f64::consts::TAU / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        for x in re.iter_mut() {
            *x /= n as f64;
        }
        for x in im.iter_mut() {
            *x /= n as f64;
        }
    }
}

/// Circular convolution a (*) b of two real vectors.
fn circ_conv(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let (mut ar, mut ai) = (a.to_vec(), vec![0.0; n]);
    let (mut br, mut bi) = (b.to_vec(), vec![0.0; n]);
    fft(&mut ar, &mut ai, false);
    fft(&mut br, &mut bi, false);
    let (mut cr, mut ci) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        cr[i] = ar[i] * br[i] - ai[i] * bi[i];
        ci[i] = ar[i] * bi[i] + ai[i] * br[i];
    }
    fft(&mut cr, &mut ci, true);
    cr
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

// ---- task + metric --------------------------------------------------------

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

fn dprime(correct: &[f64], misbound: &[f64]) -> f64 {
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var = |v: &[f64], m: f64| v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    let mc = mean(correct);
    let mm = mean(misbound);
    let sd = (0.5 * (var(correct, mc) + var(misbound, mm)))
        .sqrt()
        .max(1e-3);
    ((mc - mm) / sd).clamp(0.0, 50.0)
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, sd)
}

// ---- three systems --------------------------------------------------------

fn run_hrr(n: usize, seed: u64) -> f64 {
    let cb: Vec<Vec<f64>> = (0..N_SYMBOLS)
        .map(|i| {
            let mut rng = Rng::new(mix(i as u64, seed));
            (0..D).map(|_| rng.gaussian()).collect()
        })
        .collect();
    let mut r1 = Rng::new(mix(0x401E1, seed));
    let mut r2 = Rng::new(mix(0x401E2, seed));
    let role1: Vec<f64> = (0..D).map(|_| r1.gaussian()).collect();
    let role2: Vec<f64> = (0..D).map(|_| r2.gaussian()).collect();
    let facts = build_facts(n, seed);

    let mut bundle = vec![0.0f64; D];
    for f in &facts {
        for (acc, v) in bundle.iter_mut().zip(circ_conv(&role1, &cb[f.a])) {
            *acc += v;
        }
        for (acc, v) in bundle.iter_mut().zip(circ_conv(&role2, &cb[f.b])) {
            *acc += v;
        }
    }
    // Membership readout (matched to P): compare the composed query pair against
    // the bundle. correct = role1(*)a is a summed member; mis = role1(*)b is not.
    let mut correct = Vec::with_capacity(n);
    let mut misbound = Vec::with_capacity(n);
    for f in &facts {
        correct.push(cosine(&circ_conv(&role1, &cb[f.a]), &bundle));
        misbound.push(cosine(&circ_conv(&role1, &cb[f.b]), &bundle));
    }
    dprime(&correct, &misbound)
}

fn run_map(n: usize, seed: u64) -> f64 {
    let cb: Vec<Vec<f64>> = (0..N_SYMBOLS)
        .map(|i| {
            let mut rng = Rng::new(mix(i as u64, seed));
            (0..D).map(|_| rng.sign()).collect()
        })
        .collect();
    let mut r1 = Rng::new(mix(0x3A01, seed));
    let mut r2 = Rng::new(mix(0x3A02, seed));
    let role1: Vec<f64> = (0..D).map(|_| r1.sign()).collect();
    let role2: Vec<f64> = (0..D).map(|_| r2.sign()).collect();
    let facts = build_facts(n, seed);

    let mut bundle = vec![0.0f64; D];
    for f in &facts {
        for i in 0..D {
            bundle[i] += role1[i] * cb[f.a][i];
            bundle[i] += role2[i] * cb[f.b][i];
        }
    }
    // Membership readout (matched to P and HRR): compose the query pair and
    // compare it against the bundle.
    let mut correct = Vec::with_capacity(n);
    let mut misbound = Vec::with_capacity(n);
    for f in &facts {
        let qa: Vec<f64> = (0..D).map(|i| role1[i] * cb[f.a][i]).collect();
        let qb: Vec<f64> = (0..D).map(|i| role1[i] * cb[f.b][i]).collect();
        correct.push(cosine(&qa, &bundle));
        misbound.push(cosine(&qb, &bundle));
    }
    dprime(&correct, &misbound)
}

/// Sparse permutation stack (the surviving candidate): bind by idx^mask, bloom
/// bundle, corrected-containment readout.
fn run_p(n: usize, seed: u64) -> f64 {
    let cb: Vec<EntangledHVec> = (0..N_SYMBOLS)
        .map(|i| EntangledHVec::new_with_density(D, SPARSE_DENOM, mix(i as u64, seed)))
        .collect();
    let mask1 = ((mix(seed, 0xC0DE_1111) % D as u64) as u32).max(1);
    let mask2 = ((mix(seed, 0xC0DE_2222) % D as u64) as u32).max(1);
    let facts = build_facts(n, seed);
    let inv = |hv: &EntangledHVec, mask: u32| -> EntangledHVec {
        let mut idx: Vec<u32> = hv.indices().iter().map(|&i| i ^ mask).collect();
        idx.sort_unstable();
        EntangledHVec::from_indices(idx, D)
    };

    let mut pairs = Vec::with_capacity(2 * n);
    for f in &facts {
        pairs.push(inv(&cb[f.a], mask1));
        pairs.push(inv(&cb[f.b], mask2));
    }
    let bundle = EntangledHVec::bundle_bloom(&pairs);
    let mut correct = Vec::with_capacity(n);
    let mut misbound = Vec::with_capacity(n);
    for f in &facts {
        correct.push(inv(&cb[f.a], mask1).corrected_containment(&bundle));
        misbound.push(inv(&cb[f.b], mask1).corrected_containment(&bundle));
    }
    dprime(&correct, &misbound)
}

fn fft_self_check() {
    // Convolving a delta with x must return x (circular identity).
    let mut delta = vec![0.0; D];
    delta[0] = 1.0;
    let x: Vec<f64> = (0..D).map(|i| (i as f64 * 0.1).sin()).collect();
    let c = circ_conv(&delta, &x);
    let err: f64 = c.iter().zip(&x).map(|(a, b)| (a - b).abs()).sum();
    assert!(err < 1e-6, "FFT self-check failed: err={err}");
}

fn main() {
    fft_self_check();
    println!("binding-baselines | D={D} (matched) | sparse active=D/{SPARSE_DENOM} | symbols={N_SYMBOLS} | seeds={SEEDS}");
    println!("d' = correct-binding vs mis-binding (chance floor 0). Same discriminator as §10-12.");
    println!(
        "Storage/vector (matched D flatters dense): HRR={D} f64, MAP={D} signs, P=~{} indices.\n",
        D / SPARSE_DENOM
    );
    println!(
        "{:>5}  {:>15}  {:>15}  {:>15}",
        "N", "HRR d'(sd)", "MAP d'(sd)", "P sparse d'(sd)"
    );

    for &n in LOADS {
        let mut hrr = Vec::with_capacity(SEEDS as usize);
        let mut map = Vec::with_capacity(SEEDS as usize);
        let mut p = Vec::with_capacity(SEEDS as usize);
        for seed in 0..SEEDS {
            hrr.push(run_hrr(n, seed));
            map.push(run_map(n, seed));
            p.push(run_p(n, seed));
        }
        let (hm, hs) = mean_sd(&hrr);
        let (mm, ms) = mean_sd(&map);
        let (pm, ps) = mean_sd(&p);
        println!(
            "{n:>5}  {:>8.2} ({:>4.2})  {:>8.2} ({:>4.2})  {:>8.2} ({:>4.2})",
            hm, hs, mm, ms, pm, ps
        );
    }

    println!("\nRead: if P (sparse) matches HRR/MAP at matched D, it is competitive at a");
    println!("fraction of the storage (potentially novel). If P loses, dense baselines win as");
    println!("expected-given-more-bits; only a small gap would be interesting.");
}
