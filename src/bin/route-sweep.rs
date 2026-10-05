// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Measures recall@10 and latency of each retrieval route (exact scan, inverted,
//! IVF, NSG) across store size, dimension and data distribution, so the query
//! planner thresholds can be set from data. The route that runs is forced by
//! which indices are trained, and confirmed with `explain_query`.
//!
//! Usage: route-sweep [--ns 1000,3000] [--dims 4096,16384] [--queries 200]
//!                    [--train-budget-secs 600] [--out path.json]

use anyhow::{bail, Context, Result};
use holographic_memory::{EntangledHVec, HmsCore, RetrievalResult};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

const K: usize = 10;
const CLUSTER_BASES: usize = 300;
const PERTURB: f64 = 0.2;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Replace `frac` of the active indices with fresh random ones, keeping the count.
fn perturb(v: &EntangledHVec, frac: f64, dim: usize, rng: &mut Rng) -> EntangledHVec {
    let target = v.indices().len();
    let drop = ((target as f64) * frac).round() as usize;
    let mut idx = v.indices().to_vec();
    for _ in 0..drop {
        idx.swap_remove(rng.below(idx.len()));
    }
    idx.sort_unstable();
    while idx.len() < target {
        let c = rng.below(dim) as u32;
        if let Err(pos) = idx.binary_search(&c) {
            idx.insert(pos, c);
        }
    }
    EntangledHVec::from_indices(idx, dim)
}

fn gen_data(kind: &str, n: usize, dim: usize) -> Vec<EntangledHVec> {
    match kind {
        "random" => (0..n)
            .map(|i| EntangledHVec::new_deterministic(dim, i as u64))
            .collect(),
        _ => {
            let bases: Vec<EntangledHVec> = (0..CLUSTER_BASES)
                .map(|b| EntangledHVec::new_deterministic(dim, 1_000_000_007 + b as u64))
                .collect();
            let mut rng = Rng(0xC1D5_7E12);
            (0..n)
                .map(|i| perturb(&bases[i % CLUSTER_BASES], PERTURB, dim, &mut rng))
                .collect()
        }
    }
}

/// Exact top-K similarities, descending. Insertion into a small sorted array.
fn exact_topk(vectors: &[EntangledHVec], q: &EntangledHVec) -> [f64; K] {
    let mut best = [f64::NEG_INFINITY; K];
    for v in vectors {
        let s = q.similarity(v);
        if s > best[K - 1] {
            let mut p = K - 1;
            while p > 0 && best[p - 1] < s {
                best[p] = best[p - 1];
                p -= 1;
            }
            best[p] = s;
        }
    }
    best
}

fn pct(sorted_us: &[f64], p: f64) -> f64 {
    let i = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[i]
}

/// Tie-tolerant recall@K: a returned vector counts when its exact similarity
/// reaches the exact K-th best similarity.
fn recall(res: &[RetrievalResult], vectors: &[EntangledHVec], q: &EntangledHVec, kth: f64) -> f64 {
    let mut seen = std::collections::HashSet::new();
    let mut hit = 0;
    for r in res.iter().take(K) {
        let Some(i) = r.id.strip_prefix('v').and_then(|s| s.parse::<usize>().ok()) else {
            continue;
        };
        if i < vectors.len() && seen.insert(i) && q.similarity(&vectors[i]) >= kth - 1e-12 {
            hit += 1;
        }
    }
    hit as f64 / K as f64
}

fn summarize(mut lat_us: Vec<f64>, recalls: &[f64]) -> Value {
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    json!({
        "recall_at_10": recalls.iter().sum::<f64>() / recalls.len() as f64,
        "p50_us": pct(&lat_us, 0.5),
        "p95_us": pct(&lat_us, 0.95),
        "mean_us": lat_us.iter().sum::<f64>() / lat_us.len() as f64,
        "queries": lat_us.len(),
    })
}

fn run_route(
    hms: &HmsCore,
    vectors: &[EntangledHVec],
    queries: &[EntangledHVec],
    kth: &[f64],
) -> Value {
    for q in queries.iter().take(10) {
        let _ = hms.query(q, K as u32);
    }
    let route = hms.explain_query(&queries[0], K as u32).route;
    let mut lat = Vec::with_capacity(queries.len());
    let mut rec = Vec::with_capacity(queries.len());
    for (q, &kt) in queries.iter().zip(kth) {
        let t = Instant::now();
        let res = hms.query(q, K as u32);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
        rec.push(recall(&res, vectors, q, kt));
    }
    let mut v = summarize(lat, &rec);
    v["route_reported"] = json!(route);
    v
}

/// Last training time per (dataset, dim, route), used to skip a route whose
/// extrapolated training time (assuming N^1.3) exceeds the budget.
#[derive(Default)]
struct Prev(HashMap<(String, usize, String), (usize, f64)>);
impl Prev {
    fn key(kind: &str, dim: usize, route: &str) -> (String, usize, String) {
        (kind.to_string(), dim, route.to_string())
    }
    fn estimate(&self, kind: &str, dim: usize, route: &str, n: usize) -> f64 {
        match self.0.get(&Self::key(kind, dim, route)) {
            Some(&(pn, secs)) => secs * (n as f64 / pn as f64).powf(1.3),
            None => 0.0,
        }
    }
    fn record(&mut self, kind: &str, dim: usize, route: &str, n: usize, secs: f64) {
        self.0.insert(Self::key(kind, dim, route), (n, secs));
    }
}

fn sweep_point(
    kind: &str,
    n: usize,
    dim: usize,
    nq: usize,
    budget: f64,
    prev: &mut Prev,
) -> Result<Value> {
    eprintln!("== {kind} D={dim} N={n}");
    let vectors = gen_data(kind, n, dim);
    let mut rng = Rng(0x5EED ^ n as u64 ^ dim as u64);
    let queries: Vec<EntangledHVec> = (0..nq)
        .map(|_| perturb(&vectors[rng.below(n)], 0.25, dim, &mut rng))
        .collect();

    let mut kth = Vec::with_capacity(nq);
    let mut lat = Vec::with_capacity(nq);
    for q in &queries {
        let t = Instant::now();
        let top = exact_topk(&vectors, q);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
        kth.push(top[K - 1]);
    }
    let exact = summarize(lat, &vec![1.0; nq]);

    let dir = tempfile::tempdir()?;
    let hms = HmsCore::new(dim as u32, Some(dir.path().display().to_string()), None)?;
    let t = Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        hms.memorize(format!("v{i}"), v.clone())
            .with_context(|| format!("memorize v{i}"))?;
    }
    let memorize_secs = t.elapsed().as_secs_f64();

    let inverted = run_route(&hms, &vectors, &queries, &kth);
    if inverted["route_reported"] != json!("inverted") {
        bail!(
            "untrained store did not route to inverted: {}",
            inverted["route_reported"]
        );
    }
    let mut routes = json!({ "exact": exact, "inverted": inverted });
    for name in ["ivf", "nsg"] {
        let est = prev.estimate(kind, dim, name, n);
        if est > budget {
            eprintln!("  skip {name}: estimated train {est:.0}s > budget {budget:.0}s");
            routes[name] = json!({
                "skipped": format!("estimated training {est:.0}s exceeds budget {budget:.0}s")
            });
            continue;
        }
        let t = Instant::now();
        if name == "ivf" {
            hms.train_ivf()?;
        } else {
            hms.train_nsg()?;
        }
        let secs = t.elapsed().as_secs_f64();
        prev.record(kind, dim, name, n, secs);
        let mut r = run_route(&hms, &vectors, &queries, &kth);
        r["train_secs"] = json!(secs);
        if r["route_reported"] != json!(name) {
            bail!(
                "forced {name} but explain_query reported {}",
                r["route_reported"]
            );
        }
        routes[name] = r;
    }
    Ok(json!({
        "dataset": kind, "dim": dim, "n": n, "active_per_vector": vectors[0].indices().len(),
        "memorize_secs": memorize_secs, "routes": routes,
    }))
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn list(s: Option<&str>, default: &[usize]) -> Vec<usize> {
    s.map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn machine() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ns = list(
        arg(&args, "--ns"),
        &[1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
    );
    let dims = list(arg(&args, "--dims"), &[4096, 16384]);
    let nq: usize = arg(&args, "--queries")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let budget: f64 = arg(&args, "--train-budget-secs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(600.0);
    let out = arg(&args, "--out").unwrap_or("benchmarks/results/route_sweep.json");

    let mut points = Vec::new();
    let mut prev = Prev::default();
    let write = |points: &Vec<Value>, done: bool| -> Result<()> {
        let doc = json!({
            "schema_version": 1,
            "machine": machine(),
            "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
            "k": K, "queries_per_point": nq, "train_budget_secs": budget,
            "complete": done,
            "query_model": "stored vector with 25% of active indices replaced by random ones",
            "recall_model": "tie-tolerant recall@10 vs exact Jaccard scan",
            "cluster_model": format!(
                "{CLUSTER_BASES} base vectors, each copy has {:.0}% indices replaced",
                PERTURB * 100.0
            ),
            "latency": "single-threaded caller, wall time per HmsCore::query, warm, microseconds",
            "points": points,
        });
        std::fs::write(out, serde_json::to_vec_pretty(&doc)?)?;
        Ok(())
    };
    for &dim in &dims {
        for kind in ["random", "clustered"] {
            for &n in &ns {
                points.push(sweep_point(kind, n, dim, nq, budget, &mut prev)?);
                write(&points, false)?;
            }
        }
    }
    write(&points, true)?;
    eprintln!("wrote {out}");
    Ok(())
}
