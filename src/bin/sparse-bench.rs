// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Learned sparse retrieval benchmark: serves encoded BEIR corpora with the HMS impact index
//! and reports nDCG@10, R@100, single-thread QPS, serving RSS and index bytes; evaluates a
//! TREC run file (an external system's output) with the same metric code; tunes and reports
//! a min-max blend of the sparse scores with exact dense (MiniLM) scores.
//!
//! Inputs are written by `benchmarks/public/learned_sparse_encode.py`.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use holographic_memory::core::sparse::{ImpactIndex, SparseVector};
use serde_json::json;

#[derive(Parser)]
struct Args {
    /// Directory with corpus.sparse, queries_{test,tune}.sparse, *_ids.json, qrels_*.tsv.
    #[arg(long)]
    sparse: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Timing repeats (median reported).
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    /// Evaluate this TREC run file (qid Q0 docid rank score tag) instead of serving.
    #[arg(long)]
    run_file: Option<PathBuf>,
    /// Dense corpus/query rows (f32 little-endian) for the blend.
    #[arg(long)]
    dense_corpus: Option<PathBuf>,
    #[arg(long)]
    dense_queries: Option<PathBuf>,
    #[arg(long)]
    dense_tune_queries: Option<PathBuf>,
    #[arg(long, default_value_t = 384)]
    dim: usize,
    /// Which split the dense tune rows belong to; the sparse tune split is fixed by the encoder.
    #[arg(long)]
    label: Option<String>,
    /// With `--run-file`: query ids in the run are 0-based positions into the test id list.
    #[arg(long)]
    run_qid_index: bool,
}

fn read_sparse(path: &Path) -> Result<(u32, Vec<SparseVector>)> {
    let mut f = BufReader::new(File::open(path).with_context(|| path.display().to_string())?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    ensure!(&magic == b"HMSSPRS1", "bad magic in {}", path.display());
    let mut u = [0u8; 4];
    f.read_exact(&mut u)?;
    let vocab = u32::from_le_bytes(u);
    f.read_exact(&mut u)?;
    let n = u32::from_le_bytes(u) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        f.read_exact(&mut u)?;
        let len = u32::from_le_bytes(u) as usize;
        let mut ids = vec![0u8; 4 * len];
        f.read_exact(&mut ids)?;
        let mut ws = vec![0u8; 4 * len];
        f.read_exact(&mut ws)?;
        let ids: Vec<u32> = ids
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let weights: Vec<f32> = ws
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        out.push(SparseVector { ids, weights });
    }
    Ok((vocab, out))
}

fn read_ids(path: &Path) -> Result<Vec<String>> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}

/// qrels: query id -> (doc id -> relevance).
fn read_qrels(path: &Path) -> Result<HashMap<String, HashMap<String, u32>>> {
    let mut out: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for (i, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        let f: Vec<&str> = line.split('\t').collect();
        if i == 0 && f.first() == Some(&"query-id") {
            continue;
        }
        if f.len() < 3 {
            continue;
        }
        let rel: u32 = f[2].trim().parse().unwrap_or(0);
        out.entry(f[0].to_string())
            .or_default()
            .insert(f[1].to_string(), rel);
    }
    Ok(out)
}

fn read_f32(path: &Path, dim: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len() % (4 * dim) == 0,
        "{} is not {dim}-wide",
        path.display()
    );
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// nDCG@10 and R@100 of ranked document ids per query, BEIR style (graded gains, log2
/// discount from rank 1, ideal from the qrels); queries without qrels are skipped.
fn evaluate(
    ranked: &HashMap<String, Vec<String>>,
    qrels: &HashMap<String, HashMap<String, u32>>,
) -> (f64, f64, usize) {
    let (mut ndcg, mut recall, mut n) = (0f64, 0f64, 0usize);
    for (qid, rels) in qrels {
        let Some(docs) = ranked.get(qid) else {
            n += 1;
            continue;
        };
        let dcg: f64 = docs
            .iter()
            .take(10)
            .enumerate()
            .map(|(i, d)| {
                let g = f64::from(*rels.get(d).unwrap_or(&0));
                g / ((i + 2) as f64).log2()
            })
            .sum();
        let mut gains: Vec<u32> = rels.values().copied().collect();
        gains.sort_unstable_by(|a, b| b.cmp(a));
        let idcg: f64 = gains
            .iter()
            .take(10)
            .enumerate()
            .map(|(i, &g)| f64::from(g) / ((i + 2) as f64).log2())
            .sum();
        if idcg > 0.0 {
            ndcg += dcg / idcg;
        }
        let relevant: Vec<&String> = rels
            .iter()
            .filter(|(_, &r)| r > 0)
            .map(|(d, _)| d)
            .collect();
        if !relevant.is_empty() {
            let hit = docs
                .iter()
                .take(100)
                .filter(|d| rels.get(*d).is_some_and(|&r| r > 0))
                .count();
            recall += hit as f64 / relevant.len() as f64;
        }
        n += 1;
    }
    (ndcg / n.max(1) as f64, recall / n.max(1) as f64, n)
}

fn rss_bytes() -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

fn minmax(v: &mut [f32]) {
    let (lo, hi) = v
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| {
            (l.min(x), h.max(x))
        });
    let range = if hi > lo { hi - lo } else { 1.0 };
    v.iter_mut().for_each(|x| *x = (*x - lo) / range);
}

/// Ranked ids by a blend of min-max normalized sparse and dense scores over all documents.
fn blend_rank(
    sparse: &[f32],
    dense: &[f32],
    alpha: f32,
    corpus_ids: &[String],
    depth: usize,
) -> Vec<String> {
    let mut s = sparse.to_vec();
    let mut d = dense.to_vec();
    minmax(&mut s);
    minmax(&mut d);
    let mut scored: Vec<(f32, usize)> = s
        .iter()
        .zip(&d)
        .enumerate()
        .map(|(i, (a, b))| (alpha * a + (1.0 - alpha) * b, i))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored
        .iter()
        .take(depth)
        .map(|&(_, i)| corpus_ids[i].clone())
        .collect()
}

fn main() -> Result<()> {
    let a = Args::parse();
    let dir = &a.sparse;
    let meta: serde_json::Value = serde_json::from_reader(File::open(dir.join("meta.json"))?)?;
    let corpus_ids = read_ids(&dir.join("corpus_ids.json"))?;
    let qrels_test = read_qrels(&dir.join("qrels_test.tsv"))?;
    let test_ids = read_ids(&dir.join("queries_test_ids.json"))?;
    if let Some(run) = &a.run_file {
        // External system: rank lists from a TREC run file.
        let mut ranked: HashMap<String, Vec<(u32, String)>> = HashMap::new();
        for line in BufReader::new(File::open(run)?).lines() {
            let line = line?;
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 6 {
                continue;
            }
            let qid = if a.run_qid_index {
                let at: usize = f[0].parse().context("integer query position")?;
                test_ids
                    .get(at)
                    .cloned()
                    .context("query position outside the test list")?
            } else {
                f[0].to_string()
            };
            ranked
                .entry(qid)
                .or_default()
                .push((f[3].parse().unwrap_or(u32::MAX), f[2].to_string()));
        }
        let ranked: HashMap<String, Vec<String>> = ranked
            .into_iter()
            .map(|(q, mut v)| {
                v.sort_unstable_by_key(|x| x.0);
                (q, v.into_iter().map(|x| x.1).collect())
            })
            .collect();
        let (ndcg, r100, n) = evaluate(&ranked, &qrels_test);
        let report = json!({"schema_version": 1, "system": "run_file", "run_file": run,
            "model": meta["model"], "dataset": meta["dataset"], "queries": n,
            "ndcg_at_10": ndcg, "recall_at_100": r100});
        std::fs::write(&a.out, serde_json::to_string_pretty(&report)?)?;
        println!("{report}");
        return Ok(());
    }
    let (vocab, docs) = read_sparse(&dir.join("corpus.sparse"))?;
    let (_, queries) = read_sparse(&dir.join("queries_test.sparse"))?;
    ensure!(
        docs.len() == corpus_ids.len() && queries.len() == test_ids.len(),
        "id lists mismatch"
    );
    let rss_static = rss_bytes();
    let t = Instant::now();
    let index = ImpactIndex::build(&docs, vocab)?;
    let build_secs = t.elapsed().as_secs_f64();
    let docs_nnz: u64 = docs.iter().map(|d| d.ids.len() as u64).sum();
    drop(docs);
    let rss_serving = rss_bytes();
    // Quality at depth 100 (R@100 needs it; nDCG@10 is a prefix).
    let mut ranked: HashMap<String, Vec<String>> = HashMap::new();
    for (q, qid) in queries.iter().zip(&test_ids) {
        let top = index.top_k(q, 100);
        ranked.insert(
            qid.clone(),
            top.iter()
                .map(|&(d, _)| corpus_ids[d as usize].clone())
                .collect(),
        );
    }
    let (ndcg, r100, n) = evaluate(&ranked, &qrels_test);
    // Single-thread QPS at k = 10 and k = 100, median of repeats, warmed.
    let mut qps = BTreeMap::new();
    for &k in &[10usize, 100] {
        let mut runs = Vec::new();
        for _ in 0..a.repeats + 1 {
            let t = Instant::now();
            let mut sink = 0usize;
            for q in &queries {
                sink += index.top_k(q, k).len();
            }
            let secs = t.elapsed().as_secs_f64();
            std::hint::black_box(sink);
            runs.push(queries.len() as f64 / secs);
        }
        runs.remove(0);
        runs.sort_by(f64::total_cmp);
        qps.insert(
            format!("k{k}"),
            json!({"median": runs[runs.len() / 2], "runs": runs}),
        );
    }
    let mut report = json!({
        "schema_version": 1, "system": "hms_impact_index", "model": meta["model"], "license": meta["license"],
        "model_url": meta["url"], "dataset": meta["dataset"], "documents": index.documents(), "queries": n,
        "vocab": vocab, "postings": index.postings(), "corpus_nonzeros": docs_nnz,
        "index_bytes": index.bytes(), "bytes_per_posting": index.bytes() as f64 / index.postings().max(1) as f64,
        "impact_scale": index.scale(), "build_seconds": build_secs,
        "ndcg_at_10": ndcg, "recall_at_100": r100, "qps_single_thread": qps,
        "rss_bytes": {"static": rss_static, "serving": rss_serving},
        "method": "MaxScore document-at-a-time over 8-bit impacts with 128-posting blocks and block maxima; exact top-k over the quantized impacts; one thread; quality at depth 100"
    });
    if let (Some(dc), Some(dq)) = (&a.dense_corpus, &a.dense_queries) {
        let dense = read_f32(dc, a.dim)?;
        ensure!(dense.len() == corpus_ids.len() * a.dim, "dense corpus rows");
        let dense_q = read_f32(dq, a.dim)?;
        ensure!(
            dense_q.len() == queries.len() * a.dim,
            "dense test query rows"
        );
        // Tune alpha on the tune split when its dense rows are given, else report alpha 0.5.
        let score_all = |q: &SparseVector, dq: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let s: Vec<f32> = (0..corpus_ids.len() as u32)
                .map(|d| index.score_document(q, d))
                .collect();
            let d: Vec<f32> = dense
                .chunks_exact(a.dim)
                .map(|row| row.iter().zip(dq).map(|(x, y)| x * y).sum())
                .collect();
            (s, d)
        };
        let alphas: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
        let mut chosen = 0.5f32;
        let mut tuning = json!(null);
        if let Some(dt) = &a.dense_tune_queries {
            let (_, tune_q) = read_sparse(&dir.join("queries_tune.sparse"))?;
            let tune_ids = read_ids(&dir.join("queries_tune_ids.json"))?;
            let qrels_tune = read_qrels(&dir.join("qrels_tune.tsv"))?;
            let dense_t = read_f32(dt, a.dim)?;
            ensure!(
                dense_t.len() == tune_q.len() * a.dim,
                "dense tune query rows ({} vs {})",
                dense_t.len() / a.dim,
                tune_q.len()
            );
            let scores: Vec<(Vec<f32>, Vec<f32>)> = tune_q
                .iter()
                .zip(dense_t.chunks_exact(a.dim))
                .map(|(q, dq)| score_all(q, dq))
                .collect();
            let mut table = Vec::new();
            let mut best = (f64::NEG_INFINITY, 0.5f32);
            for &alpha in &alphas {
                let ranked: HashMap<String, Vec<String>> = scores
                    .iter()
                    .zip(&tune_ids)
                    .map(|((s, d), qid)| (qid.clone(), blend_rank(s, d, alpha, &corpus_ids, 100)))
                    .collect();
                let (nd, _, _) = evaluate(&ranked, &qrels_tune);
                table.push(json!({"alpha": alpha, "ndcg_at_10": nd}));
                if nd > best.0 {
                    best = (nd, alpha);
                }
            }
            chosen = best.1;
            tuning = json!({"split": meta["tune_split"], "queries": tune_ids.len(), "grid": table, "chosen_alpha": chosen});
        }
        let ranked: HashMap<String, Vec<String>> = queries
            .iter()
            .zip(dense_q.chunks_exact(a.dim))
            .zip(&test_ids)
            .map(|((q, dq), qid)| {
                let (s, d) = score_all(q, dq);
                (qid.clone(), blend_rank(&s, &d, chosen, &corpus_ids, 100))
            })
            .collect();
        let (bn, br, _) = evaluate(&ranked, &qrels_test);
        let dense_only: HashMap<String, Vec<String>> = queries
            .iter()
            .zip(dense_q.chunks_exact(a.dim))
            .zip(&test_ids)
            .map(|((q, dq), qid)| {
                let (s, d) = score_all(q, dq);
                (qid.clone(), blend_rank(&s, &d, 0.0, &corpus_ids, 100))
            })
            .collect();
        let (dn, dr, _) = evaluate(&dense_only, &qrels_test);
        report["blend"] = json!({"alpha": chosen, "tuning": tuning, "ndcg_at_10": bn, "recall_at_100": br,
            "dense_only_ndcg_at_10": dn, "dense_only_recall_at_100": dr,
            "method": "min-max over all documents per query of the exact sparse score and the exact dense (MiniLM-L6) dot product, alpha * sparse + (1 - alpha) * dense, alpha chosen on the tune split"});
    }
    if let Some(label) = &a.label {
        report["label"] = json!(label);
    }
    if a.out.exists() {
        bail!("refusing to overwrite {}", a.out.display());
    }
    std::fs::write(&a.out, serde_json::to_string_pretty(&report)?)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"ndcg_at_10": ndcg, "recall_at_100": r100, "qps": report["qps_single_thread"], "index_bytes": index.bytes()})
        )?
    );
    Ok(())
}
