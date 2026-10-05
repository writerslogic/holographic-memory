// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sweep of subset-union ("Bloom") bundle capacity: each item contributes a
//! deterministic k'-subset of its 64 active indices. Items always have 64
//! active indices (density denominator dim/64), so k'=64 is the full-item
//! baseline at every D. Writes benchmarks/results/bundle_subset_sweep.json.

use holographic_memory::core::bundle_subset::{bundle_subset, contains_subset, subset_indices};
use holographic_memory::core::entangled::EntangledHVec;
use serde_json::{json, Value};

const DIMS: [usize; 2] = [16384, 65536];
const SUBSETS: [usize; 6] = [4, 8, 11, 16, 32, 64];
const ITEM_SIZE: usize = 64;
const SEEDS: usize = 5;
const PROBES_PER_SEED: usize = 4000;
const GRID_START: f64 = 16.0;
const GRID_RATIO: f64 = 1.15;
const STOP_FPR: f64 = 0.01;
const MAX_N: usize = 2_000_000;

struct SeedState {
    bits: Vec<u64>,
    items: Vec<EntangledHVec>,
    probes: Vec<Vec<u32>>,
    seed_base: u64,
    dim: usize,
    k: usize,
    set_bits: usize,
}

impl SeedState {
    fn new(dim: usize, k: usize, seed: usize) -> Self {
        let seed_base = (seed as u64 + 1) << 40;
        let probes = (0..PROBES_PER_SEED)
            .map(|i| {
                let p = EntangledHVec::new_with_density(
                    dim,
                    dim / ITEM_SIZE,
                    seed_base | (1 << 39) | i as u64,
                );
                subset_indices(&p, k)
            })
            .collect();
        Self {
            bits: vec![0; dim.div_ceil(64)],
            items: Vec::new(),
            probes,
            seed_base,
            dim,
            k,
            set_bits: 0,
        }
    }

    fn grow_to(&mut self, n: usize) {
        while self.items.len() < n {
            let it = EntangledHVec::new_with_density(
                self.dim,
                self.dim / ITEM_SIZE,
                self.seed_base | self.items.len() as u64,
            );
            for i in subset_indices(&it, self.k) {
                let (w, m) = (i as usize / 64, 1u64 << (i % 64));
                if self.bits[w] & m == 0 {
                    self.bits[w] |= m;
                    self.set_bits += 1;
                }
            }
            self.items.push(it);
        }
    }

    fn false_positives(&self) -> usize {
        self.probes
            .iter()
            .filter(|p| {
                p.iter()
                    .all(|&i| (self.bits[i as usize / 64] >> (i % 64)) & 1 == 1)
            })
            .count()
    }
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    (m, var.sqrt())
}

fn main() {
    let mut out_dims = Vec::new();
    for &dim in &DIMS {
        let mut out_k = Vec::new();
        for &k in &SUBSETS {
            let mut states: Vec<SeedState> =
                (0..SEEDS).map(|s| SeedState::new(dim, k, s)).collect();
            let mut points: Vec<Value> = Vec::new();
            let mut n_f = GRID_START;
            let mut last_n = 0usize;
            loop {
                let n = n_f.round() as usize;
                n_f *= GRID_RATIO;
                if n == last_n {
                    continue;
                }
                last_n = n;
                if n > MAX_N {
                    break;
                }
                let mut fprs = Vec::new();
                let mut dens = Vec::new();
                let mut fp_total = 0usize;
                for s in states.iter_mut() {
                    s.grow_to(n);
                    let fp = s.false_positives();
                    fp_total += fp;
                    fprs.push(fp as f64 / PROBES_PER_SEED as f64);
                    dens.push(s.set_bits as f64 / dim as f64);
                }
                // Cross-check the library builder and membership test against
                // the bitset on the first grid point of every (D, k').
                if points.is_empty() {
                    let s = &states[0];
                    let b = bundle_subset(&s.items, k);
                    assert_eq!(b.indices().len(), s.set_bits);
                    assert!(s.items.iter().all(|it| contains_subset(&b, it, k)));
                }
                let (fpr, fpr_std) = mean_std(&fprs);
                let (density, _) = mean_std(&dens);
                let analytic_density = 1.0 - (1.0 - k as f64 / dim as f64).powi(n as i32);
                points.push(json!({
                    "n": n,
                    "fpr_mean": fpr,
                    "fpr_std_over_seeds": fpr_std,
                    "false_positives": fp_total,
                    "probes_total": SEEDS * PROBES_PER_SEED,
                    "density_measured": density,
                    "density_analytic": analytic_density,
                    "theory_fpr_density_pow_k_measured": density.powi(k as i32),
                    "theory_fpr_density_pow_k_analytic": analytic_density.powi(k as i32),
                }));
                eprintln!("D={dim} k'={k} N={n} fpr={fpr:.5} density={density:.4}");
                if fpr > STOP_FPR {
                    break;
                }
            }
            let cap = |thr: f64| -> Value {
                points
                    .iter()
                    .filter(|p| p["fpr_mean"].as_f64().unwrap() <= thr)
                    .map(|p| p["n"].as_u64().unwrap())
                    .max()
                    .map_or(Value::Null, |n| json!(n))
            };
            out_k.push(json!({
                "subset_size": k,
                "capacity_fpr_le_1pct": cap(0.01),
                "capacity_fpr_le_0_1pct": cap(0.001),
                "points": points,
            }));
        }
        out_dims.push(json!({
            "dim": dim,
            "k_star_at_n1000": dim as f64 / 1000.0 * std::f64::consts::LN_2,
            "subsets": out_k,
        }));
    }
    let doc = json!({
        "description": "Subset-union Bloom bundle capacity. Items have 64 active indices; each contributes a deterministic k'-subset; membership = all k' present. FPR measured on non-member probes.",
        "item_active_indices": ITEM_SIZE,
        "seeds": SEEDS,
        "probes_per_seed": PROBES_PER_SEED,
        "grid": {"start": GRID_START, "ratio": GRID_RATIO, "stop_when_fpr_gt": STOP_FPR},
        "capacity_note": "Largest grid N with mean FPR <= threshold; resolution is the grid ratio. FPR resolution is 1/probes_total (one FP in 20000 = 0.005%), so a 0.1% capacity is 20 FPs.",
        "dims": out_dims,
    });
    let path = "benchmarks/results/bundle_subset_sweep.json";
    std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap()).expect("write results");
    eprintln!("wrote {path}");
}
