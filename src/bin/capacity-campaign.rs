// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// §23. The naive superposition + linear-cleanup readout tops out near the Plate SNR
// limit (~0.1-0.2·D); the counting ceiling (V=64, N=256, D=1024) is ~1.33·D. That
// gap is OPEN. This pits a MULTITUDE of readout/code mechanisms against the naive
// baseline on the SAME single-bundle retrieval task (recover object given exact key
// from ONE bundle of M facts), sweeping load past 1.0·D to find each arm's knee.
//
//   base        — dense phasors, one-shot linear score argmax (incumbent)
//   whiten      — per-dim field magnitude normalization before scoring
//   hopfield-lo — iterative modern-Hopfield softmax cleanup, low beta
//   hopfield-hi — iterative modern-Hopfield softmax cleanup, high beta
//   ens2 / ens4 — K independent sub-bundles (D/K each), summed scores (voting)
//
// f64 complex here (capacity, not determinism, is the question; the winning arm
// would be re-checked for integer determinism like §20). Run:
//   cargo run --release --bin capacity-campaign

use std::f64::consts::TAU;

const D: usize = 1024;
const O: usize = 64;
const SEEDS: u64 = 5;
const QUERIES: usize = 48;
const ITERS: usize = 12;

type Cx = (Vec<f64>, Vec<f64>);

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Unit-phasor vector of `dim` dims in Z_N, deterministic in (id, salt, seed).
fn phasor(id: u64, salt: u64, seed: u64, n: u32, dim: usize) -> Cx {
    let mut re = vec![0.0; dim];
    let mut im = vec![0.0; dim];
    for d in 0..dim {
        let p = (mix(id.wrapping_mul(0x9E37) ^ (d as u64) ^ salt, seed) % n as u64) as f64;
        let a = TAU * p / n as f64;
        re[d] = a.cos();
        im[d] = a.sin();
    }
    (re, im)
}

/// Complex product a⊙b (phase-add binding).
fn cmul(a: &Cx, b: &Cx) -> Cx {
    let n = a.0.len();
    let mut r = vec![0.0; n];
    let mut i = vec![0.0; n];
    for d in 0..n {
        r[d] = a.0[d] * b.0[d] - a.1[d] * b.1[d];
        i[d] = a.0[d] * b.1[d] + a.1[d] * b.0[d];
    }
    (r, i)
}

fn conj(a: &Cx) -> Cx {
    (a.0.clone(), a.1.iter().map(|&x| -x).collect())
}

/// Re<a, b> = Σ a̅·b real part = Σ (a.re·b.re + a.im·b.im).
fn dot(a: &Cx, b: &Cx) -> f64 {
    let mut s = 0.0;
    for d in 0..a.0.len() {
        s += a.0[d] * b.0[d] + a.1[d] * b.1[d];
    }
    s
}

struct Facts {
    keys: Vec<Cx>,
    objs: Vec<Cx>,
    fact_obj: Vec<usize>,
}

fn gen(m: usize, seed: u64, n: u32, dim: usize) -> Facts {
    let objs = (0..O)
        .map(|o| phasor(o as u64, 0xB0, seed, n, dim))
        .collect();
    let keys = (0..m)
        .map(|i| phasor(i as u64, 0xA0, seed, n, dim))
        .collect();
    let fact_obj = (0..m)
        .map(|i| (mix(i as u64, seed ^ 0xC0) % O as u64) as usize)
        .collect();
    Facts {
        keys,
        objs,
        fact_obj,
    }
}

fn build_field(f: &Facts, dim: usize) -> Cx {
    let mut re = vec![0.0; dim];
    let mut im = vec![0.0; dim];
    for i in 0..f.keys.len() {
        let b = cmul(&f.keys[i], &f.objs[f.fact_obj[i]]);
        for d in 0..dim {
            re[d] += b.0[d];
            im[d] += b.1[d];
        }
    }
    (re, im)
}

fn argmax_scores(scores: &[f64]) -> usize {
    (0..scores.len())
        .max_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap())
        .unwrap()
}

/// One-shot linear readout: score_o = Re<field, bind(q, obj_o)>.
fn read_base(field: &Cx, q: &Cx, objs: &[Cx]) -> usize {
    let scores: Vec<f64> = objs.iter().map(|o| dot(field, &cmul(q, o))).collect();
    argmax_scores(&scores)
}

/// Per-dim magnitude-normalized field before scoring.
fn read_whiten(field: &Cx, q: &Cx, objs: &[Cx]) -> usize {
    let dim = field.0.len();
    let mut wr = vec![0.0; dim];
    let mut wi = vec![0.0; dim];
    for d in 0..dim {
        let m = (field.0[d] * field.0[d] + field.1[d] * field.1[d]).sqrt() + 1e-9;
        wr[d] = field.0[d] / m;
        wi[d] = field.1[d] / m;
    }
    let w = (wr, wi);
    let scores: Vec<f64> = objs.iter().map(|o| dot(&w, &cmul(q, o))).collect();
    argmax_scores(&scores)
}

/// Iterative modern-Hopfield cleanup on the unbound value.
fn read_hopfield(field: &Cx, q: &Cx, objs: &[Cx], beta: f64) -> usize {
    let dim = field.0.len();
    let sq = (dim as f64).sqrt();
    // v0 = unbind(field, q) = field ⊙ conj(q)
    let mut v = cmul(field, &conj(q));
    let mut scores = vec![0.0; O];
    for _ in 0..ITERS {
        for (o, ob) in objs.iter().enumerate() {
            scores[o] = dot(&v, ob);
        }
        // softmax at a temperature scaled to the score magnitude (~sqrt(dim))
        let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
        let exps: Vec<f64> = scores
            .iter()
            .map(|&s| (beta * (s - mx) / sq).exp())
            .collect();
        let z: f64 = exps.iter().sum::<f64>() + 1e-12;
        // v = Σ_o softmax_o · obj_o
        let mut nr = vec![0.0; dim];
        let mut ni = vec![0.0; dim];
        for (o, ob) in objs.iter().enumerate() {
            let w = exps[o] / z;
            for d in 0..dim {
                nr[d] += w * ob.0[d];
                ni[d] += w * ob.1[d];
            }
        }
        v = (nr, ni);
    }
    argmax_scores(&scores)
}

/// K independent sub-bundles of D/K dims each; summed per-candidate scores.
fn recall_ensemble(m: usize, n: u32, k: usize) -> f64 {
    let sub = D / k;
    let (mut hits, mut tot) = (0i64, 0i64);
    for seed in 0..SEEDS {
        // independent fields per sub-bundle
        let subs: Vec<(Facts, Cx)> = (0..k)
            .map(|c| {
                let f = gen(m, seed ^ ((c as u64 + 1) << 20), n, sub);
                let field = build_field(&f, sub);
                (f, field)
            })
            .collect();
        let step = (m / QUERIES).max(1);
        for i in (0..m).step_by(step) {
            let truth = subs[0].0.fact_obj[i];
            let best = (0..O)
                .max_by(|&a, &b| {
                    let sa: f64 = subs
                        .iter()
                        .map(|(f, fld)| dot(fld, &cmul(&f.keys[i], &f.objs[a])))
                        .sum();
                    let sb: f64 = subs
                        .iter()
                        .map(|(f, fld)| dot(fld, &cmul(&f.keys[i], &f.objs[b])))
                        .sum();
                    sa.partial_cmp(&sb).unwrap()
                })
                .unwrap();
            hits += (best == truth) as i64;
            tot += 1;
        }
    }
    100.0 * hits as f64 / tot as f64
}

/// Sparse-key code: each key is active (unit) on only a fraction `s` of dims, zero
/// elsewhere (objects dense). Reduces per-dimension collision (Frady/Kleyko).
fn recall_sparse(m: usize, n: u32, s: f64) -> f64 {
    let (mut hits, mut tot) = (0i64, 0i64);
    for seed in 0..SEEDS {
        let mut f = gen(m, seed, n, D);
        for (i, key) in f.keys.iter_mut().enumerate() {
            for d in 0..D {
                if (mix(d as u64, seed ^ (i as u64) ^ 0xE0) as f64 / u64::MAX as f64) >= s {
                    key.0[d] = 0.0;
                    key.1[d] = 0.0;
                }
            }
        }
        let field = build_field(&f, D);
        let step = (m / QUERIES).max(1);
        for i in (0..m).step_by(step) {
            let pred = read_base(&field, &f.keys[i], &f.objs);
            hits += (pred == f.fact_obj[i]) as i64;
            tot += 1;
        }
    }
    100.0 * hits as f64 / tot as f64
}

fn recall_single<F: Fn(&Cx, &Cx, &[Cx]) -> usize>(m: usize, n: u32, read: F) -> f64 {
    let (mut hits, mut tot) = (0i64, 0i64);
    for seed in 0..SEEDS {
        let f = gen(m, seed, n, D);
        let field = build_field(&f, D);
        let step = (m / QUERIES).max(1);
        for i in (0..m).step_by(step) {
            let pred = read(&field, &f.keys[i], &f.objs);
            hits += (pred == f.fact_obj[i]) as i64;
            tot += 1;
        }
    }
    100.0 * hits as f64 / tot as f64
}

fn main() {
    let loads = [0.1f64, 0.2, 0.3, 0.5, 0.75, 1.0];
    let n = 256u32;
    println!(
        "capacity-campaign | D={D} O={O} seeds={SEEDS} N={n} | top-1 recall %, chance={:.1}%",
        100.0 / O as f64
    );
    println!("counting ceiling (V=64) ~1.33·D. Task: recover object from ONE bundle, exact key.\n");
    type Arm = (&'static str, Box<dyn Fn(usize) -> f64>);
    let arms: Vec<Arm> = vec![
        ("base", Box::new(move |m| recall_single(m, n, read_base))),
        (
            "whiten",
            Box::new(move |m| recall_single(m, n, read_whiten)),
        ),
        (
            "hopfield-lo",
            Box::new(move |m| recall_single(m, n, |f, q, o| read_hopfield(f, q, o, 2.0))),
        ),
        (
            "hopfield-hi",
            Box::new(move |m| recall_single(m, n, |f, q, o| read_hopfield(f, q, o, 8.0))),
        ),
        ("ens2", Box::new(move |m| recall_ensemble(m, n, 2))),
        ("ens4", Box::new(move |m| recall_ensemble(m, n, 4))),
        ("sparse.5", Box::new(move |m| recall_sparse(m, n, 0.5))),
        ("sparse.25", Box::new(move |m| recall_sparse(m, n, 0.25))),
        ("sparse.1", Box::new(move |m| recall_sparse(m, n, 0.1))),
    ];
    print!("{:>12}", "arm \\ M/D");
    for l in loads {
        print!("  {:>5}", format!("{l:.2}"));
    }
    println!("   knee@90 knee@50");
    for (name, f) in &arms {
        let recs: Vec<f64> = loads
            .iter()
            .map(|&l| f(((l * D as f64) as usize).max(1)))
            .collect();
        print!("{name:>12}");
        for r in &recs {
            print!("  {:>4.0}%", r);
        }
        let knee = |thresh: f64| {
            loads
                .iter()
                .zip(&recs)
                .filter(|(_, &r)| r >= thresh)
                .map(|(&l, _)| l)
                .fold(0.0, f64::max)
        };
        println!("   {:>6.2} {:>6.2}", knee(90.0), knee(50.0));
    }
    println!("\nKill: no arm beats base's 50%-knee by >0.05·D -> readout tricks don't unlock");
    println!("capacity on the dense phase substrate; the lever is a different code (sparse");
    println!("block codes on EntangledHVec, ECC output codes) -> next campaign.");
}
