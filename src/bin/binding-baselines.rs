// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Pre-registration step 2, hardened (strong baselines) from
// docs/PREREGISTRATION-binding-readout.md.
//
// Compares the sparse permutation stack (P) against the field's strong incumbent
// bindings, HRR (Plate, circular convolution) and MAP (Gayler, elementwise
// product), on the role-filler mis-binding task, at matched dimensionality.
//
//   P   = sparse permutation bind (idx^mask), bloom bundle. Verification via
//         corrected containment; retrieval via inverse-permute + containment rank.
//   HRR = real dense N(0,1/D), circular-convolution bind. Verification via
//         cosine-to-bundle; retrieval via circular-correlation unbind + cosine rank.
//   MAP = bipolar dense {-1,+1}, elementwise-product bind. Verification via
//         cosine-to-bundle; retrieval via product-unbind + cosine rank.
//
// Hardening over the first run (which reported verification d' only, matched-D,
// 8 seeds):
//   - ROC/AUC alongside d' (threshold-free, no Gaussian assumption).
//   - RETRIEVAL rank-1 accuracy (unbind -> reconstruct -> nearest codebook entry)
//     -- the axis originally assumed to favor dense; measured here for all three,
//     since a permutation bind is invertible and P can in fact unbind.
//   - D-sweep (512, 2048, 8192) at fixed density 1/256 (sparse active = D/256).
//   - 20 seeds for verification.
//
// MATCHED-D flatters the dense systems on storage (HRR: D f64; MAP: D signs; P:
// ~D/256 indices); reported alongside.
//
// Run: cargo run --release --bin binding-baselines

use holographic_memory::EntangledHVec;

const SPARSE_DENOM: usize = 256;
const N_SYMBOLS: usize = 512; // shared codebook (real interference)
const D_SWEEP: &[usize] = &[512, 2048, 8192];
const VERIF_LOADS: &[usize] = &[20, 40, 80, 160, 320];
const VERIF_SEEDS: u64 = 20;
const RET_D: usize = 2048;
const RET_LOADS: &[usize] = &[20, 40, 80, 160];
const RET_SEEDS: u64 = 10;

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
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gaussian(&mut self) -> f64 {
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

/// Circular correlation = unbind: recovers what was convolved with `a`.
fn circ_corr(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let (mut ar, mut ai) = (a.to_vec(), vec![0.0; n]);
    let (mut br, mut bi) = (b.to_vec(), vec![0.0; n]);
    fft(&mut ar, &mut ai, false);
    fft(&mut br, &mut bi, false);
    let (mut cr, mut ci) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        cr[i] = ar[i] * br[i] + ai[i] * bi[i];
        ci[i] = ar[i] * bi[i] - ai[i] * br[i];
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

// ---- task + metrics -------------------------------------------------------

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

/// AUC = P(correct score > mis-bound score), the Mann-Whitney estimator.
/// Threshold-free and distribution-free (no Gaussian assumption like d').
fn auc(correct: &[f64], misbound: &[f64]) -> f64 {
    let mut wins = 0.0;
    let total = (correct.len() * misbound.len()) as f64;
    for &c in correct {
        for &m in misbound {
            if c > m {
                wins += 1.0;
            } else if (c - m).abs() < 1e-12 {
                wins += 0.5;
            }
        }
    }
    wins / total
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, sd)
}

// ---- system builders (return codebook, roles, bundle) ---------------------

fn dense_codebook(d: usize, seed: u64, gaussian: bool) -> Vec<Vec<f64>> {
    (0..N_SYMBOLS)
        .map(|i| {
            let mut rng = Rng::new(mix(i as u64, seed));
            (0..d)
                .map(|_| if gaussian { rng.gaussian() } else { rng.sign() })
                .collect()
        })
        .collect()
}

fn dense_role(d: usize, seed: u64, gaussian: bool) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..d)
        .map(|_| if gaussian { rng.gaussian() } else { rng.sign() })
        .collect()
}

fn hrr_bundle(d: usize, facts: &[Fact], cb: &[Vec<f64>], r1: &[f64], r2: &[f64]) -> Vec<f64> {
    let mut bundle = vec![0.0f64; d];
    for f in facts {
        for (acc, v) in bundle.iter_mut().zip(circ_conv(r1, &cb[f.a])) {
            *acc += v;
        }
        for (acc, v) in bundle.iter_mut().zip(circ_conv(r2, &cb[f.b])) {
            *acc += v;
        }
    }
    bundle
}

fn map_bundle(d: usize, facts: &[Fact], cb: &[Vec<f64>], r1: &[f64], r2: &[f64]) -> Vec<f64> {
    let mut bundle = vec![0.0f64; d];
    for f in facts {
        for i in 0..d {
            bundle[i] += r1[i] * cb[f.a][i];
            bundle[i] += r2[i] * cb[f.b][i];
        }
    }
    bundle
}

fn sparse_masks(seed: u64, d: usize) -> (u32, u32) {
    let m1 = ((mix(seed, 0xC0DE_1111) % d as u64) as u32).max(1);
    let m2 = ((mix(seed, 0xC0DE_2222) % d as u64) as u32).max(1);
    (m1, m2)
}

fn perm(hv: &EntangledHVec, mask: u32, d: usize) -> EntangledHVec {
    let mut idx: Vec<u32> = hv.indices().iter().map(|&i| i ^ mask).collect();
    idx.sort_unstable();
    EntangledHVec::from_indices(idx, d)
}

// ---- verification: (d', auc) per system -----------------------------------

fn verif_hrr(d: usize, n: usize, seed: u64) -> (f64, f64) {
    let cb = dense_codebook(d, seed, true);
    let (r1, r2) = (
        dense_role(d, mix(0x401E1, seed), true),
        dense_role(d, mix(0x401E2, seed), true),
    );
    let facts = build_facts(n, seed);
    let bundle = hrr_bundle(d, &facts, &cb, &r1, &r2);
    let (mut c, mut m) = (Vec::new(), Vec::new());
    for f in &facts {
        c.push(cosine(&circ_conv(&r1, &cb[f.a]), &bundle));
        m.push(cosine(&circ_conv(&r1, &cb[f.b]), &bundle));
    }
    (dprime(&c, &m), auc(&c, &m))
}

fn verif_map(d: usize, n: usize, seed: u64) -> (f64, f64) {
    let cb = dense_codebook(d, seed, false);
    let (r1, r2) = (
        dense_role(d, mix(0x3A01, seed), false),
        dense_role(d, mix(0x3A02, seed), false),
    );
    let facts = build_facts(n, seed);
    let bundle = map_bundle(d, &facts, &cb, &r1, &r2);
    let (mut c, mut m) = (Vec::new(), Vec::new());
    for f in &facts {
        let qa: Vec<f64> = (0..d).map(|i| r1[i] * cb[f.a][i]).collect();
        let qb: Vec<f64> = (0..d).map(|i| r1[i] * cb[f.b][i]).collect();
        c.push(cosine(&qa, &bundle));
        m.push(cosine(&qb, &bundle));
    }
    (dprime(&c, &m), auc(&c, &m))
}

fn verif_p(d: usize, n: usize, seed: u64) -> (f64, f64) {
    let cb: Vec<EntangledHVec> = (0..N_SYMBOLS)
        .map(|i| EntangledHVec::new_with_density(d, SPARSE_DENOM, mix(i as u64, seed)))
        .collect();
    let (mask1, mask2) = sparse_masks(seed, d);
    let facts = build_facts(n, seed);
    let mut pairs = Vec::with_capacity(2 * n);
    for f in &facts {
        pairs.push(perm(&cb[f.a], mask1, d));
        pairs.push(perm(&cb[f.b], mask2, d));
    }
    let bundle = EntangledHVec::bundle_bloom(&pairs);
    let (mut c, mut m) = (Vec::new(), Vec::new());
    for f in &facts {
        // Probe role1 with the correct filler (a) vs the mis-bound filler (b).
        c.push(perm(&cb[f.a], mask1, d).corrected_containment(&bundle));
        m.push(perm(&cb[f.b], mask1, d).corrected_containment(&bundle));
    }
    (dprime(&c, &m), auc(&c, &m))
}

// ---- retrieval: rank-1 accuracy per system --------------------------------

fn argmax_correct<F: Fn(usize) -> f64>(n_symbols: usize, truth: usize, score: F) -> bool {
    let mut best = usize::MAX;
    let mut best_s = f64::NEG_INFINITY;
    for s in 0..n_symbols {
        let v = score(s);
        if v > best_s {
            best_s = v;
            best = s;
        }
    }
    best == truth
}

fn retr_hrr(d: usize, n: usize, seed: u64) -> f64 {
    let cb = dense_codebook(d, seed, true);
    let (r1, r2) = (
        dense_role(d, mix(0x401E1, seed), true),
        dense_role(d, mix(0x401E2, seed), true),
    );
    let facts = build_facts(n, seed);
    let bundle = hrr_bundle(d, &facts, &cb, &r1, &r2);
    let unbound = circ_corr(&r1, &bundle);
    let hits = facts
        .iter()
        .filter(|f| argmax_correct(N_SYMBOLS, f.a, |s| cosine(&unbound, &cb[s])))
        .count();
    hits as f64 / n as f64
}

fn retr_map(d: usize, n: usize, seed: u64) -> f64 {
    let cb = dense_codebook(d, seed, false);
    let (r1, r2) = (
        dense_role(d, mix(0x3A01, seed), false),
        dense_role(d, mix(0x3A02, seed), false),
    );
    let facts = build_facts(n, seed);
    let bundle = map_bundle(d, &facts, &cb, &r1, &r2);
    let unbound: Vec<f64> = (0..d).map(|i| r1[i] * bundle[i]).collect();
    let hits = facts
        .iter()
        .filter(|f| argmax_correct(N_SYMBOLS, f.a, |s| cosine(&unbound, &cb[s])))
        .count();
    hits as f64 / n as f64
}

fn retr_p(d: usize, n: usize, seed: u64) -> f64 {
    let cb: Vec<EntangledHVec> = (0..N_SYMBOLS)
        .map(|i| EntangledHVec::new_with_density(d, SPARSE_DENOM, mix(i as u64, seed)))
        .collect();
    let (mask1, mask2) = sparse_masks(seed, d);
    let facts = build_facts(n, seed);
    let mut pairs = Vec::with_capacity(2 * n);
    for f in &facts {
        pairs.push(perm(&cb[f.a], mask1, d));
        pairs.push(perm(&cb[f.b], mask2, d));
    }
    let bundle = EntangledHVec::bundle_bloom(&pairs);
    // Unbind role1 by inverse-permuting the bundle (idx^mask1 is an involution),
    // then rank codebook symbols by containment in the unbound bundle.
    let unbound = perm(&bundle, mask1, d);
    let hits = facts
        .iter()
        .filter(|f| argmax_correct(N_SYMBOLS, f.a, |s| cb[s].corrected_containment(&unbound)))
        .count();
    hits as f64 / n as f64
}

// ---- self-check + main ----------------------------------------------------

fn fft_self_check() {
    let d = 2048;
    let mut delta = vec![0.0; d];
    delta[0] = 1.0;
    let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
    let c = circ_conv(&delta, &x);
    let err: f64 = c.iter().zip(&x).map(|(a, b)| (a - b).abs()).sum();
    assert!(err < 1e-6, "FFT self-check failed: err={err}");
}

fn main() {
    fft_self_check();
    println!("binding-baselines (hardened) | symbols={N_SYMBOLS} | verif seeds={VERIF_SEEDS} | ret seeds={RET_SEEDS}");
    println!(
        "Chance: verification d'=0 / AUC=0.5; retrieval acc=1/{N_SYMBOLS}={:.4}.\n",
        1.0 / N_SYMBOLS as f64
    );

    // --- Verification: d' and AUC, D-sweep ---
    println!("== VERIFICATION (membership: is this pair bound?) ==");
    for &d in D_SWEEP {
        let active = d / SPARSE_DENOM;
        println!("\n-- D={d} (HRR {d} f64, MAP {d} signs, P {active} idx) --  d' | AUC");
        println!(
            "{:>5}  {:>18}  {:>18}  {:>18}",
            "N", "HRR", "MAP", "P sparse"
        );
        for &n in VERIF_LOADS {
            let mut hd = Vec::new();
            let mut ha = Vec::new();
            let mut md = Vec::new();
            let mut ma = Vec::new();
            let mut pd = Vec::new();
            let mut pa = Vec::new();
            for seed in 0..VERIF_SEEDS {
                let (a, b) = verif_hrr(d, n, seed);
                hd.push(a);
                ha.push(b);
                let (a, b) = verif_map(d, n, seed);
                md.push(a);
                ma.push(b);
                let (a, b) = verif_p(d, n, seed);
                pd.push(a);
                pa.push(b);
            }
            println!(
                "{n:>5}  {:>7.2}|{:>5.3}({:>4.2})  {:>7.2}|{:>5.3}({:>4.2})  {:>7.2}|{:>5.3}({:>4.2})",
                mean_sd(&hd).0,
                mean_sd(&ha).0,
                mean_sd(&hd).1,
                mean_sd(&md).0,
                mean_sd(&ma).0,
                mean_sd(&md).1,
                mean_sd(&pd).0,
                mean_sd(&pa).0,
                mean_sd(&pd).1,
            );
        }
    }

    // --- Retrieval: rank-1 accuracy (the scope-defining axis) ---
    println!("\n== RETRIEVAL (unbind -> reconstruct -> nearest codebook symbol), D={RET_D} ==");
    println!(
        "{:>5}  {:>14}  {:>14}  {:>14}",
        "N", "HRR acc", "MAP acc", "P sparse acc"
    );
    for &n in RET_LOADS {
        let mut h = Vec::new();
        let mut m = Vec::new();
        let mut p = Vec::new();
        for seed in 0..RET_SEEDS {
            h.push(retr_hrr(RET_D, n, seed));
            m.push(retr_map(RET_D, n, seed));
            p.push(retr_p(RET_D, n, seed));
        }
        println!(
            "{n:>5}  {:>8.3} ({:>4.2})  {:>8.3} ({:>4.2})  {:>8.3} ({:>4.2})",
            mean_sd(&h).0,
            mean_sd(&h).1,
            mean_sd(&m).0,
            mean_sd(&m).1,
            mean_sd(&p).0,
            mean_sd(&p).1,
        );
    }
}
