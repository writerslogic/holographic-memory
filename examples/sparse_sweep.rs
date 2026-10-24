// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Recall/QPS sweep of the learned-sparse index on big-ann-benchmarks sparse files.
//!
//! `sparse_sweep BASE.csr QUERIES.csr GT [postings block_fraction energy] [cuts] [factors]`
//! where `cuts` and `factors` are comma lists. Recall@10 is reported over all queries and over
//! the even and odd query rows separately. Threads follow `RAYON_NUM_THREADS`.

use std::time::Instant;

use holographic_memory::core::sparse_impact::{
    CsrView, SparseBuildParams, SparseImpactIndex, SparseSearchParams,
};

fn map(path: &str) -> memmap2::Mmap {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    // SAFETY: read-only map of a file nothing truncates while this process runs.
    unsafe { memmap2::Mmap::map(&file) }.unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// big-ann ground truth: int32 nq, k, then nq*k int32 ids and nq*k float32 scores.
fn read_gt(path: &str) -> (usize, usize, Vec<u32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let word = |i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    let (nq, k) = (word(0) as usize, word(1) as usize);
    (nq, k, (0..nq * k).map(|i| word(2 + i) as u32).collect())
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T> {
    s.split(',')
        .map(|x| x.parse().ok().expect("bad list value"))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: sparse_sweep BASE QUERIES GT [postings frac energy] [cuts] [factors]");
        std::process::exit(2);
    }
    let base = map(&args[1]);
    let queries = map(&args[2]);
    let (gt_nq, gt_k, gt) = read_gt(&args[3]);
    let arg = |i: usize, d: &str| args.get(i).cloned().unwrap_or_else(|| d.to_string());
    let build = SparseBuildParams {
        postings_per_term: arg(4, "4000").parse().unwrap(),
        block_fraction: arg(5, "0.1").parse().unwrap(),
        summary_energy: arg(6, "0.4").parse().unwrap(),
        ..SparseBuildParams::default()
    };
    let cuts: Vec<usize> = list(&arg(7, "10"));
    let factors: Vec<f32> = list(&arg(8, "0.8"));

    let docs = CsrView::from_spmat(&base).expect("base");
    let qs = CsrView::from_spmat(&queries).expect("queries");
    let nq = qs.indptr.len() - 1;
    assert_eq!(nq, gt_nq, "query and ground-truth counts differ");
    let started = Instant::now();
    let index = SparseImpactIndex::build(docs, build).expect("build");
    let (blocks, postings) = index.block_stats();
    eprintln!(
        "build {:.1}s docs {} blocks {blocks} postings {postings} index {:.2} GiB params {build:?}",
        started.elapsed().as_secs_f64(),
        index.n_docs(),
        index.memory_bytes() as f64 / f64::from(1 << 30)
    );

    let k = 10;
    let recall = |found: &[u32], rows: &mut dyn Iterator<Item = usize>| {
        let (mut hit, mut n) = (0usize, 0usize);
        for q in rows {
            let truth = &gt[q * gt_k..q * gt_k + k];
            hit += found[q * k..q * k + k]
                .iter()
                .filter(|d| truth.contains(d))
                .count();
            n += k;
        }
        hit as f64 / n as f64
    };
    println!(
        "postings,block_fraction,energy,query_cut,heap_factor,qps,recall,recall_even,recall_odd"
    );
    for &query_cut in &cuts {
        for &heap_factor in &factors {
            let params = SparseSearchParams {
                query_cut,
                heap_factor,
            };
            let mut best = f64::INFINITY;
            let mut found = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                found = index.search_batch(qs, k, params).expect("search");
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "{},{},{},{query_cut},{heap_factor},{:.0},{:.4},{:.4},{:.4}",
                build.postings_per_term,
                build.block_fraction,
                build.summary_energy,
                nq as f64 / best,
                recall(&found, &mut (0..nq)),
                recall(&found, &mut (0..nq).step_by(2)),
                recall(&found, &mut (1..nq).step_by(2)),
            );
        }
    }
}
