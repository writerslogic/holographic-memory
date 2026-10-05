// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fidelity and latency of `EntangledHVec::from_dense` on synthetic data.
//!
//! Usage: `encoder-eval [--bench]`. Default writes
//! `benchmarks/results/encoder_eval.json`; `--bench` prints median encode
//! time per (embedding dim, D) and writes nothing.

use anyhow::{Context, Result};
use holographic_memory::EntangledHVec;
use rayon::prelude::*;
use serde::Serialize;
use std::time::Instant;

const SEED: u64 = 0x5EED_E0C0_DE01;
const EMB_DIM: usize = 384;
const N_DOCS: usize = 10_000;
const N_QUERIES: usize = 500;
const N_CLUSTERS: usize = 99;
const TOP_K: usize = 10;
const DIMS: [usize; 3] = [4096, 16384, 65536];
/// Noise norm relative to the unit-norm cluster centre, one per cluster class.
const NOISE_LEVELS: [(&str, f64); 3] = [("low", 0.25), ("medium", 0.6), ("high", 1.2)];
const LATENCY_SAMPLES: usize = 200;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn gauss_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.gauss()).collect()
    }
}

fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
}

fn sample(rng: &mut Rng, centre: &[f64], noise: f64) -> Vec<f32> {
    let mut noise_vec = rng.gauss_vec(centre.len());
    normalize(&mut noise_vec);
    let mut v: Vec<f64> = centre
        .iter()
        .zip(&noise_vec)
        .map(|(c, n)| c + noise * n)
        .collect();
    normalize(&mut v);
    v.into_iter().map(|x| x as f32).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn jaccard(a: &[u32], b: &[u32]) -> f64 {
    let (mut i, mut j, mut inter) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Top-k ids by descending score; ties broken by lower id.
fn top_k(scores: &[f64], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    let cmp = |a: &usize, b: &usize| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b));
    ids.select_nth_unstable_by(k - 1, cmp);
    ids.truncate(k);
    ids.sort_by(cmp);
    ids
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time_encode_us(v: &[f32], d: usize, reps: usize) -> f64 {
    let times = (0..reps)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(EntangledHVec::from_dense(std::hint::black_box(v), d));
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    median(times)
}

fn bench() {
    let mut rng = Rng(SEED);
    for emb in [64usize, 384, 768, 1536] {
        let mut v = rng.gauss_vec(emb);
        normalize(&mut v);
        let v: Vec<f32> = v.into_iter().map(|x| x as f32).collect();
        for d in DIMS {
            let reps = if d >= 65536 { 40 } else { 200 };
            time_encode_us(&v, d, 3);
            println!(
                "emb={emb} D={d} median_encode_us={:.1}",
                time_encode_us(&v, d, reps)
            );
        }
    }
}

#[derive(Serialize)]
struct NoiseResult {
    noise_level: &'static str,
    noise_norm_ratio: f64,
    n_queries: usize,
    recall_at_10: f64,
}

#[derive(Serialize)]
struct DimResult {
    sparse_dim: usize,
    active_count: usize,
    recall_at_10_overall: f64,
    by_noise: Vec<NoiseResult>,
    encode_us_per_vector_median_single_thread: f64,
}

#[derive(Serialize)]
struct Report {
    dataset: &'static str,
    note: &'static str,
    encoder: &'static str,
    seed: u64,
    embedding_dim: usize,
    n_docs: usize,
    n_queries: usize,
    n_clusters: usize,
    top_k: usize,
    ground_truth: &'static str,
    sparse_ranking: &'static str,
    random_baseline_recall_at_10: f64,
    results: Vec<DimResult>,
}

fn main() -> Result<()> {
    if std::env::args().any(|a| a == "--bench") {
        bench();
        return Ok(());
    }
    let mut rng = Rng(SEED);
    let mut centres: Vec<Vec<f64>> = (0..N_CLUSTERS).map(|_| rng.gauss_vec(EMB_DIM)).collect();
    centres.iter_mut().for_each(|c| normalize(c));
    let level_of = |c: usize| c % NOISE_LEVELS.len();

    let docs: Vec<Vec<f32>> = (0..N_DOCS)
        .map(|i| {
            let c = i % N_CLUSTERS;
            sample(&mut rng, &centres[c], NOISE_LEVELS[level_of(c)].1)
        })
        .collect();
    // Queries are fresh draws from the same clusters, not corpus members.
    let queries: Vec<(usize, Vec<f32>)> = (0..N_QUERIES)
        .map(|q| {
            let c = (q * 7 + 3) % N_CLUSTERS;
            (
                level_of(c),
                sample(&mut rng, &centres[c], NOISE_LEVELS[level_of(c)].1),
            )
        })
        .collect();

    let truth: Vec<Vec<usize>> = queries
        .par_iter()
        .map(|(_, q)| {
            let s: Vec<f64> = docs.iter().map(|d| f64::from(dot(q, d))).collect();
            top_k(&s, TOP_K)
        })
        .collect();

    let mut results = Vec::new();
    for d in DIMS {
        let doc_sp: Vec<EntangledHVec> = docs
            .par_iter()
            .map(|v| EntangledHVec::from_dense(v, d))
            .collect();
        let q_sp: Vec<EntangledHVec> = queries
            .par_iter()
            .map(|(_, v)| EntangledHVec::from_dense(v, d))
            .collect();
        let hits: Vec<usize> = q_sp
            .par_iter()
            .zip(&truth)
            .map(|(q, t)| {
                let s: Vec<f64> = doc_sp
                    .iter()
                    .map(|x| jaccard(q.indices(), x.indices()))
                    .collect();
                let got = top_k(&s, TOP_K);
                got.iter().filter(|g| t.contains(g)).count()
            })
            .collect();

        let by_noise = NOISE_LEVELS
            .iter()
            .enumerate()
            .map(|(l, (name, ratio))| {
                let (h, n) = queries
                    .iter()
                    .zip(&hits)
                    .filter(|((lv, _), _)| *lv == l)
                    .fold((0, 0), |(h, n), (_, x)| (h + x, n + 1));
                NoiseResult {
                    noise_level: name,
                    noise_norm_ratio: *ratio,
                    n_queries: n,
                    recall_at_10: h as f64 / (n * TOP_K) as f64,
                }
            })
            .collect();

        let lat = median(
            docs.iter()
                .take(LATENCY_SAMPLES)
                .map(|v| time_encode_us(v, d, 1))
                .collect(),
        );
        let r = DimResult {
            sparse_dim: d,
            active_count: doc_sp[0].indices().len(),
            recall_at_10_overall: hits.iter().sum::<usize>() as f64 / (N_QUERIES * TOP_K) as f64,
            by_noise,
            encode_us_per_vector_median_single_thread: lat,
        };
        eprintln!(
            "D={d} recall@10={:.4} encode_us={lat:.0}",
            r.recall_at_10_overall
        );
        results.push(r);
    }

    let report = Report {
        dataset: "SYNTHETIC",
        note: "Synthetic Gaussian-cluster unit-norm embeddings; not evidence about real text-embedding neighbour structure.",
        encoder: "signed-projection-v2 (EntangledHVec::from_dense)",
        seed: SEED,
        embedding_dim: EMB_DIM,
        n_docs: N_DOCS,
        n_queries: N_QUERIES,
        n_clusters: N_CLUSTERS,
        top_k: TOP_K,
        ground_truth: "dense cosine (unit-norm dot) top-10 over the corpus, ties by lower id",
        sparse_ranking: "sparse Jaccard top-10 over the corpus, ties by lower id",
        random_baseline_recall_at_10: TOP_K as f64 / N_DOCS as f64,
        results,
    };
    let path = "benchmarks/results/encoder_eval.json";
    std::fs::write(path, serde_json::to_string_pretty(&report)?).context(path)?;
    Ok(())
}
