// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sorted-u32 intersection kernels: |a| = 64 against |b| in 64..16384.
//! Group names are `b{|b|}_ov{overlap%}`; function names are the kernels.

use criterion::{criterion_group, criterion_main, Criterion};
use holographic_memory::core::intersection::*;
use std::hint::black_box;

const UNIVERSE: u32 = 262_144;
const NA: usize = 64;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn idx(&mut self) -> u32 {
        (self.next() % UNIVERSE as u64) as u32
    }
}

fn make_pair(nb: usize, overlap_pct: usize, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut rng = Rng(seed);
    let mut a = std::collections::BTreeSet::new();
    while a.len() < NA {
        a.insert(rng.idx());
    }
    let a: Vec<u32> = a.into_iter().collect();
    let shared = NA * overlap_pct / 100;
    let mut b = std::collections::BTreeSet::new();
    // Random subset of `a` of size `shared` via partial Fisher-Yates.
    let mut pool = a.clone();
    for k in 0..shared {
        let r = k + (rng.next() as usize) % (pool.len() - k);
        pool.swap(k, r);
        b.insert(pool[k]);
    }
    while b.len() < nb {
        let v = rng.idx();
        if a.binary_search(&v).is_err() {
            b.insert(v);
        }
    }
    (a, b.into_iter().collect())
}

/// Frozen copy of the pre-change aarch64 dispatch (merge, or gallop when 8x skewed).
fn baseline(a: &[u32], b: &[u32]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if small.len() * 8 < large.len() {
        galloping_count(small, large)
    } else {
        merge_branchy_count(small, large)
    }
}

fn bench(c: &mut Criterion) {
    for &nb in &[64usize, 256, 1024, 4096, 16384] {
        for &ov in &[0usize, 10, 50, 100] {
            let (a, b) = make_pair(nb, ov, (nb * 1000 + ov) as u64);
            let mut g = c.benchmark_group(format!("b{nb}_ov{ov}"));
            let mut run = |name: &str, f: &dyn Fn(&[u32], &[u32]) -> usize| {
                g.bench_function(name, |bn| bn.iter(|| f(black_box(&a), black_box(&b))));
            };
            run("baseline", &baseline);
            run("merge_branchy", &merge_branchy_count);
            run("merge_branchless", &merge_branchless_count);
            run("gallop", &galloping_count);
            #[cfg(target_arch = "aarch64")]
            run("neon", &neon_intersection_count);
            run("final", &sparse_intersection_count);
            g.finish();
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
