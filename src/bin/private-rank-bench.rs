// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Private ranking at scale: glove-100 rows quantized to low-bit codes and
//! ranked with the clustered private-scoring protocol
//! (`private_query::clustered`).
//!
//! `sweep` measures recall without encryption. Decoding is exact (the failure
//! bound is checked per configuration and `protocol` verifies every decoded
//! score), so the private result is the top 10 of the probed clusters'
//! plaintext scores. `protocol` runs the encrypted protocol for chosen
//! configurations, compares every decoded score with the plaintext score and
//! times each phase.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use holographic_memory::core::private_query::clustered::{
    DenseClient, DenseMatrix, DenseParams, DenseServer, FAILURE_LOG2_BUDGET,
};
use holographic_memory::core::private_query::{ERROR_SIGMA, ERROR_TAIL, LWE_DIM, SEED_LEN};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde_json::{json, Value};

const DIM: usize = 100;
const TOP: usize = 10;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Recall and communication for every (clusters, probes, widths) at each size.
    Sweep {
        #[command(flatten)]
        common: Common,
        #[arg(long, value_delimiter = ',', default_value = "1,16,32,64,128,256,1024")]
        clusters: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,32,64")]
        probes: Vec<usize>,
    },
    /// The encrypted protocol for configurations `clusters:probes:db_bits:query_bits`.
    Protocol {
        #[command(flatten)]
        common: Common,
        #[arg(long, value_delimiter = ',')]
        configs: Vec<String>,
        /// Queries run through the protocol per configuration (after 2 warm-up).
        #[arg(long, default_value_t = 30)]
        protocol_queries: usize,
    },
}

#[derive(Args)]
struct Common {
    /// Directory with train.f32 and test.f32 (glove-100-angular).
    #[arg(long)]
    data: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "100000,1000000")]
    sizes: Vec<usize>,
    /// `db_bits:query_bits` pairs.
    #[arg(long, value_delimiter = ',', default_value = "1:4,2:4,4:4")]
    widths: Vec<String>,
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    /// Cluster capacity as a multiple of the mean cluster size.
    #[arg(long, default_value_t = 1.25)]
    capacity: f64,
    #[arg(long, default_value_t = 100_000)]
    kmeans_sample: usize,
    #[arg(long, default_value_t = 12)]
    kmeans_iters: usize,
    #[arg(long)]
    out: PathBuf,
}

fn read_f32(path: &Path, rows: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let available = bytes.len() / (4 * DIM);
    ensure!(
        available >= rows,
        "{} has {available} rows, need {rows}",
        path.display()
    );
    let mut v: Vec<f32> = bytes[..rows * DIM * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    for row in v.chunks_exact_mut(DIM) {
        let norm = row
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::MIN_POSITIVE);
        row.iter_mut().for_each(|x| *x /= norm);
    }
    Ok(v)
}

/// Odd-integer levels `±1, ±3, .., ±(2^bits - 1)` with step 2; `bits = 1` is
/// the sign (a bipolar hypervector). The scale puts three standard deviations
/// of a unit vector's coordinate (`1/sqrt(d)`) at the top level's outer edge.
fn quantize(v: &[f32], bits: u32) -> Vec<i8> {
    let level = (1i32 << bits) - 1;
    let scale = (1u32 << bits) as f32 * (DIM as f32).sqrt() / 3.0;
    v.iter()
        .map(|&x| {
            if bits == 1 {
                if x >= 0.0 {
                    1
                } else {
                    -1
                }
            } else {
                (2 * (x * scale / 2.0).floor() as i32 + 1).clamp(-level, level) as i8
            }
        })
        .collect()
}

fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Nearest centroid order: ascending `||x - c||²`, i.e. descending `x·c - ||c||²/2`.
fn centroid_order(cent: &[f32], half_norms: &[f32], x: &[f32], take: usize) -> Vec<u32> {
    let mut s: Vec<(f32, u32)> = cent
        .chunks_exact(DIM)
        .zip(half_norms)
        .enumerate()
        .map(|(k, (c, h))| (h - dot_f32(x, c), k as u32))
        .collect();
    let take = take.min(s.len());
    if take < s.len() {
        s.select_nth_unstable_by(take, |a, b| a.0.total_cmp(&b.0));
        s.truncate(take);
    }
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    s.into_iter().map(|(_, k)| k).collect()
}

fn half_norms(cent: &[f32]) -> Vec<f32> {
    cent.chunks_exact(DIM)
        .map(|c| dot_f32(c, c) / 2.0)
        .collect()
}

/// Lloyd's k-means on a random sample, seeded from distinct sample points.
fn kmeans(data: &[f32], k: usize, sample: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / DIM;
    let mut rng = StdRng::seed_from_u64(seed);
    let idx = rand::seq::index::sample(&mut rng, n, sample.min(n).max(k)).into_vec();
    let pts: Vec<f32> = idx
        .iter()
        .flat_map(|&i| data[i * DIM..(i + 1) * DIM].iter().copied())
        .collect();
    let m = idx.len();
    let mut cent = pts[..k * DIM].to_vec();
    for iter in 0..iters {
        let h = half_norms(&cent);
        let assign: Vec<u32> = pts
            .par_chunks_exact(DIM)
            .map(|x| centroid_order(&cent, &h, x, 1)[0])
            .collect();
        let mut sums = vec![0f64; k * DIM];
        let mut counts = vec![0usize; k];
        for (x, &a) in pts.chunks_exact(DIM).zip(&assign) {
            counts[a as usize] += 1;
            for (s, &v) in sums[a as usize * DIM..][..DIM].iter_mut().zip(x) {
                *s += f64::from(v);
            }
        }
        for c in 0..k {
            let dst = &mut cent[c * DIM..(c + 1) * DIM];
            if counts[c] == 0 {
                // Reseed an empty cluster with a sample point.
                let j = (c * 7919 + iter * 104_729) % m;
                dst.copy_from_slice(&pts[j * DIM..(j + 1) * DIM]);
            } else {
                for (x, s) in dst.iter_mut().zip(&sums[c * DIM..(c + 1) * DIM]) {
                    *x = (s / counts[c] as f64) as f32;
                }
            }
        }
    }
    cent
}

struct Layout {
    centroids: Vec<f32>,
    members: Vec<Vec<u32>>,
    /// Rows not in their nearest cluster because it was full.
    displaced: usize,
    kmeans_ms: f64,
}

impl Layout {
    fn slots(&self) -> usize {
        self.members.iter().map(Vec::len).max().unwrap_or(0)
    }
}

/// Assigns every row to its nearest cluster with room, rows closest to a
/// centroid first, under a capacity of `capacity · n / k`.
fn layout(data: &[f32], k: usize, common: &Common) -> Layout {
    let n = data.len() / DIM;
    let start = Instant::now();
    if k == 1 {
        let mut centroid = vec![0f32; DIM];
        for x in data.chunks_exact(DIM) {
            centroid
                .iter_mut()
                .zip(x)
                .for_each(|(c, v)| *c += v / n as f32);
        }
        return Layout {
            centroids: centroid,
            members: vec![(0..n as u32).collect()],
            displaced: 0,
            kmeans_ms: 0.0,
        };
    }
    let cent = kmeans(data, k, common.kmeans_sample, common.kmeans_iters, k as u64);
    let h = half_norms(&cent);
    let near: Vec<Vec<u32>> = data
        .par_chunks_exact(DIM)
        .map(|x| centroid_order(&cent, &h, x, 8))
        .collect();
    let best: Vec<f32> = data
        .par_chunks_exact(DIM)
        .zip(&near)
        .map(|(x, o)| h[o[0] as usize] - dot_f32(x, &cent[o[0] as usize * DIM..][..DIM]))
        .collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_by(|&a, &b| best[a as usize].total_cmp(&best[b as usize]));
    let cap = ((common.capacity * n as f64 / k as f64).ceil() as usize).max(1);
    let mut members = vec![Vec::new(); k];
    let mut displaced = 0;
    for &row in &order {
        let r = row as usize;
        let target = near[r]
            .iter()
            .copied()
            .find(|&c| members[c as usize].len() < cap)
            .unwrap_or_else(|| {
                centroid_order(&cent, &h, &data[r * DIM..(r + 1) * DIM], k)
                    .into_iter()
                    .find(|&c| members[c as usize].len() < cap)
                    .expect("total capacity covers every row")
            });
        displaced += usize::from(target != near[r][0]);
        members[target as usize].push(row);
    }
    Layout {
        centroids: cent,
        members,
        displaced,
        kmeans_ms: start.elapsed().as_secs_f64() * 1e3,
    }
}

/// Indices of the `TOP` best scores, ties broken by lower index.
fn top_by_score(candidates: impl Iterator<Item = (u32, i32)>) -> Vec<(u32, i32)> {
    let mut v: Vec<(u32, i32)> = candidates.collect();
    let cmp = |a: &(u32, i32), b: &(u32, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
    if v.len() > TOP {
        v.select_nth_unstable_by(TOP - 1, cmp);
        v.truncate(TOP);
    }
    v.sort_by(cmp);
    v
}

/// Tie-aware recall@10: a returned item counts when its score is at least the
/// 10th best plaintext score over the whole database.
fn recall_codes(found: &[(u32, i32)], threshold: i32) -> f64 {
    found.iter().filter(|(_, s)| *s >= threshold).count() as f64 / TOP as f64
}

fn recall_f32(found: &[(u32, i32)], truth: &[u32]) -> f64 {
    found.iter().filter(|(i, _)| truth.contains(i)).count() as f64 / TOP as f64
}

fn parse_pair(s: &str) -> Result<(u32, u32)> {
    let (a, b) = s.split_once(':').context("expected a:b")?;
    Ok((a.parse()?, b.parse()?))
}

fn summary(samples: &mut [f64]) -> Value {
    samples.sort_by(f64::total_cmp);
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    json!({ "median": at(0.5), "p10": at(0.1), "p90": at(0.9), "min": samples[0],
            "max": samples[samples.len() - 1], "samples": samples.len() })
}

struct Corpus<'a> {
    data: &'a [f32],
    queries: &'a [f32],
    truth: Vec<Vec<u32>>,
}

/// Exact f32 top-10 by inner product (rows and queries are unit vectors).
fn f32_truth(data: &[f32], queries: &[f32]) -> Vec<Vec<u32>> {
    queries
        .par_chunks_exact(DIM)
        .map(|q| {
            let scored = data
                .chunks_exact(DIM)
                .enumerate()
                .map(|(i, x)| (i as u32, (dot_f32(q, x) * 1e6) as i32));
            top_by_score(scored).into_iter().map(|(i, _)| i).collect()
        })
        .collect()
}

fn config_row(
    n: usize,
    bits: (u32, u32),
    lay: &Layout,
    probes: usize,
    codes: &[i8],
) -> (Value, Option<DenseParams>) {
    let k = lay.members.len();
    let slots = lay.slots();
    let bx = (1u32 << bits.0) - 1;
    let bu = (1u32 << bits.1) - 1;
    let norms: Vec<u64> = codes
        .chunks_exact(DIM)
        .map(|c| {
            c.iter()
                .map(|&x| (i64::from(x) * i64::from(x)) as u64)
                .sum()
        })
        .collect();
    let row_norm_sq = (0..slots)
        .map(|s| {
            lay.members
                .iter()
                .filter_map(|m| m.get(s))
                .map(|&i| norms[i as usize])
                .sum::<u64>()
        })
        .max()
        .unwrap_or(1)
        .max(1);
    let params = DenseParams::new(DIM, k, slots, bx, bu, row_norm_sq).ok();
    let worst = DenseParams::worst_row_norm_sq(DIM, k, bx);
    let row = json!({
        "rows": n, "db_bits": bits.0, "query_bits": bits.1, "clusters": k, "probes": probes,
        "slots": slots, "padding": (k * slots) as f64 / n as f64,
        "displaced_fraction": lay.displaced as f64 / n as f64,
        "fraction_probed": (probes * slots) as f64 / n as f64,
        "samples_per_secret": k * DIM,
        "row_norm_sq": row_norm_sq, "row_norm_sq_worst": worst,
        "params_valid": params.is_some(),
        "plaintext_modulus": params.map(|p| p.plaintext_modulus()),
        "failure_log2_per_probe": params.map(|p| p.failure_log2()),
        "exact": params.map(|p| p.exact()),
        "upload_bytes": probes * k * DIM * 4,
        "download_bytes": probes * slots * 4,
        "hint_bytes": slots * LWE_DIM * 4,
        "centroid_bytes": if k > 1 { k * DIM * 4 } else { 0 },
        "public_matrix_bytes": k * DIM * LWE_DIM * 4,
        "server_macs": probes * slots * k * DIM,
        "kmeans_ms": lay.kmeans_ms,
    });
    (row, params)
}

fn environment() -> Value {
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;
    json!({
        "arch": std::env::consts::ARCH, "os": std::env::consts::OS, "avx2": avx2,
        "rayon_threads": rayon::current_num_threads(), "crate_version": env!("CARGO_PKG_VERSION"),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let (common, mode) = match &cli.command {
        Command::Sweep { common, .. } => (common, "sweep"),
        Command::Protocol { common, .. } => (common, "protocol"),
    };
    let widths: Vec<(u32, u32)> = common
        .widths
        .iter()
        .map(|w| parse_pair(w))
        .collect::<Result<_>>()?;
    let max_n = *common.sizes.iter().max().context("no sizes")?;
    let train = read_f32(&common.data.join("train.f32"), max_n)?;
    let test = read_f32(&common.data.join("test.f32"), common.queries)?;

    let mut rows = Vec::new();
    for &n in &common.sizes {
        let start = Instant::now();
        let sized = Corpus {
            data: &train[..n * DIM],
            queries: &test,
            truth: f32_truth(&train[..n * DIM], &test),
        };
        eprintln!("n={n}: f32 truth {:.1}s", start.elapsed().as_secs_f64());
        match &cli.command {
            Command::Sweep {
                clusters, probes, ..
            } => sweep(&sized, common, clusters, probes, &widths, &mut rows),
            Command::Protocol {
                configs,
                protocol_queries,
                ..
            } => {
                for config in configs {
                    let parts: Vec<usize> = config
                        .split(':')
                        .map(str::parse)
                        .collect::<Result<_, _>>()?;
                    ensure!(parts.len() == 4, "config {config} is not C:P:db:query");
                    let bits = (parts[2] as u32, parts[3] as u32);
                    rows.push(protocol(
                        &sized,
                        common,
                        parts[0],
                        parts[1],
                        bits,
                        *protocol_queries,
                    )?);
                }
            }
        }
    }

    let report = json!({
        "benchmark": "private_rank_scale",
        "mode": mode,
        "note": "Experimental. glove-100-angular train-row prefix and test-query prefix, unit-normalised, quantized to odd-integer levels. recall_codes is tie-aware recall@10 against the exhaustive plaintext ranking of the same codes; recall_f32 is recall@10 against the exact f32 inner-product top 10.",
        "dataset": common.data.file_name().map(|f| f.to_string_lossy().into_owned()),
        "queries": common.queries,
        "capacity": common.capacity,
        "kmeans": { "sample": common.kmeans_sample, "iters": common.kmeans_iters },
        "lwe": { "n": LWE_DIM, "modulus_bits": 32, "error_sigma": ERROR_SIGMA, "error_tail": ERROR_TAIL,
                 "secret": "uniform", "failure_budget_log2_per_probe": FAILURE_LOG2_BUDGET },
        "environment": environment(),
        "rows": rows,
    });
    std::fs::write(&common.out, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!("wrote {}", common.out.display());
    Ok(())
}

fn sweep(
    sized: &Corpus,
    common: &Common,
    clusters: &[usize],
    probes: &[usize],
    widths: &[(u32, u32)],
    rows: &mut Vec<Value>,
) {
    let n = sized.data.len() / DIM;
    let layouts: Vec<Layout> = clusters
        .iter()
        .map(|&k| {
            let l = layout(sized.data, k, common);
            eprintln!(
                "n={n} C={k}: S={} displaced={} {:.0} ms",
                l.slots(),
                l.displaced,
                l.kmeans_ms
            );
            l
        })
        .collect();
    let max_p = probes.iter().copied().max().unwrap_or(1);
    // orders[layout][query] = nearest clusters, up to max_p.
    let orders: Vec<Vec<Vec<u32>>> = layouts
        .iter()
        .map(|l| {
            let h = half_norms(&l.centroids);
            sized
                .queries
                .par_chunks_exact(DIM)
                .map(|q| centroid_order(&l.centroids, &h, q, max_p))
                .collect()
        })
        .collect();
    for &bits in widths {
        let codes = quantize(sized.data, bits.0);
        let qcodes = quantize(sized.queries, bits.1);
        let configs: Vec<(usize, usize)> = (0..layouts.len())
            .flat_map(|li| {
                probes
                    .iter()
                    .copied()
                    .filter(move |&p| p <= clusters[li])
                    .map(move |p| (li, p))
            })
            .collect();
        // per query: (plaintext ceiling vs f32, recall per config)
        let per_query: Vec<(f64, Vec<(f64, f64)>)> = qcodes
            .par_chunks_exact(DIM)
            .enumerate()
            .map(|(qi, q)| {
                let scores: Vec<i32> = codes.chunks_exact(DIM).map(|x| dot_i8(x, q)).collect();
                let all = top_by_score(scores.iter().enumerate().map(|(i, &s)| (i as u32, s)));
                let threshold = all[TOP - 1].1;
                let ceiling = recall_f32(&all, &sized.truth[qi]);
                let per = configs
                    .iter()
                    .map(|&(li, p)| {
                        let found = top_by_score(
                            orders[li][qi][..p]
                                .iter()
                                .flat_map(|&k| layouts[li].members[k as usize].iter())
                                .map(|&i| (i, scores[i as usize])),
                        );
                        (
                            recall_codes(&found, threshold),
                            recall_f32(&found, &sized.truth[qi]),
                        )
                    })
                    .collect();
                (ceiling, per)
            })
            .collect();
        let nq = per_query.len() as f64;
        let ceiling = per_query.iter().map(|(c, _)| c).sum::<f64>() / nq;
        for (ci, &(li, p)) in configs.iter().enumerate() {
            let rc = per_query.iter().map(|(_, v)| v[ci].0).sum::<f64>() / nq;
            let rf = per_query.iter().map(|(_, v)| v[ci].1).sum::<f64>() / nq;
            let (mut row, _) = config_row(n, bits, &layouts[li], p, &codes);
            row["recall_codes"] = json!(rc);
            row["recall_f32"] = json!(rf);
            row["plaintext_codes_recall_f32"] = json!(ceiling);
            eprintln!(
                "n={n} bits={}:{} C={} P={p}: recall {rc:.3} (f32 {rf:.3}, ceiling {ceiling:.3}) valid={}",
                bits.0, bits.1, clusters[li], row["params_valid"]
            );
            rows.push(row);
        }
    }
}

fn protocol(
    sized: &Corpus,
    common: &Common,
    k: usize,
    probes: usize,
    bits: (u32, u32),
    runs: usize,
) -> Result<Value> {
    let n = sized.data.len() / DIM;
    ensure!(probes >= 1 && probes <= k, "probes must be in 1..=clusters");
    let lay = layout(sized.data, k, common);
    let codes = quantize(sized.data, bits.0);
    let qcodes = quantize(sized.queries, bits.1);
    let (mut row, params) = config_row(n, bits, &lay, probes, &codes);
    let params = params.context("parameters fail the decoding-failure budget")?;

    let start = Instant::now();
    let matrix = DenseMatrix::expand(params, [42; SEED_LEN]);
    let matrix_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    let server = DenseServer::new(&matrix, &codes, &lay.members)?;
    let hint_ms = start.elapsed().as_secs_f64() * 1e3;
    let client = DenseClient::new(matrix.clone());
    let h = half_norms(&lay.centroids);

    let (mut prep, mut enc, mut ans, mut dec) = (vec![], vec![], vec![], vec![]);
    let (mut checked, mut mismatches, mut recall) = (0usize, 0usize, 0f64);
    let total = runs + 2;
    ensure!(total <= sized.queries.len() / DIM, "not enough queries");
    for qi in 0..total {
        let qf = &sized.queries[qi * DIM..(qi + 1) * DIM];
        let qc = &qcodes[qi * DIM..(qi + 1) * DIM];
        let chosen: Vec<usize> = centroid_order(&lay.centroids, &h, qf, probes)
            .into_iter()
            .map(|c| c as usize)
            .collect();
        let t = Instant::now();
        let prepared = client.prepare(probes);
        let t_prep = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let (queries, state) = prepared.encode(qc, &chosen)?;
        let t_enc = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let answers = server.answer(&queries)?;
        let t_ans = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let scores = state.decode(server.hint(), &answers)?;
        let t_dec = t.elapsed().as_secs_f64() * 1e3;
        if qi < 2 {
            continue;
        }
        prep.push(t_prep);
        enc.push(t_enc);
        ans.push(t_ans);
        dec.push(t_dec);
        let mut found = Vec::new();
        for (p, &c) in chosen.iter().enumerate() {
            for (slot, &got) in scores[p].iter().enumerate() {
                let expected = lay.members[c]
                    .get(slot)
                    .map_or(0, |&i| dot_i8(&codes[i as usize * DIM..][..DIM], qc));
                checked += 1;
                mismatches += usize::from(got != expected);
                if let Some(&i) = lay.members[c].get(slot) {
                    found.push((i, got));
                }
            }
        }
        let all: Vec<i32> = codes.chunks_exact(DIM).map(|x| dot_i8(x, qc)).collect();
        let threshold =
            top_by_score(all.iter().enumerate().map(|(i, &s)| (i as u32, s)))[TOP - 1].1;
        recall += recall_codes(&top_by_score(found.into_iter()), threshold);
    }
    eprintln!(
        "n={n} C={k} P={probes} bits={}:{}: answer median {:.2} ms, mismatches {mismatches}/{checked}",
        bits.0,
        bits.1,
        summary(&mut ans.clone())["median"]
    );
    row["protocol_queries"] = json!(runs);
    row["scores_checked"] = json!(checked);
    row["score_mismatches"] = json!(mismatches);
    row["recall_codes"] = json!(recall / runs as f64);
    row["matrix_expand_ms"] = json!(matrix_ms);
    row["hint_build_ms"] = json!(hint_ms);
    row["client_prepare_ms"] = summary(&mut prep);
    row["client_encode_ms"] = summary(&mut enc);
    row["server_answer_ms"] = summary(&mut ans);
    row["client_decode_ms"] = summary(&mut dec);
    Ok(row)
}
