// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deletion indistinguishability experiment.
//!
//! For each target document X drawn from a held-out pool, store A ingests the base corpus with X
//! committed as its own transaction at a random batch boundary, then deletes X. The shadow
//! store B runs the same transactions and maintenance without X. At every checkpoint the
//! experiment compares every public observable (lexical, hybrid and vector search results with
//! their score bits, single-term queries over X's terms, which expose document frequency, IDF
//! and the length normalisation, listings, counters, index status) and every file under the
//! store directory, then scores two deletion-inference attackers:
//!
//! - the oracle holds B as a shadow and answers "deleted" when any observable differs, so its
//!   accuracy is `0.5 + 0.5 * P(differs)`; no attacker with the same view can do better;
//! - the Ghost Echoes attacker has no shadow: it thresholds features of probes around X (top
//!   scores, graph-versus-exact overlap near X, single-term score levels), fit on the even
//!   targets and scored on the odd ones.
//!
//! Checkpoints: `present` (X still stored; positive control), `deleted` (after
//! `delete_document`), `purged` (after `compact_scrubbing`, the `forget(purge)` path),
//! `reopened` (both stores closed, reopened and maintained), `readmitted` (more documents
//! committed and maintained after the reopen). A second B built on a one-thread pool is the
//! determinism control.
//!
//! Usage: `forget-experiment --dataset nfcorpus|glove --data DIR --work DIR --out FILE
//!   [--targets N] [--later M] [--base N] [--seed S] [--label TEXT]`

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use holographic_memory::core::documents::{
    EncodeMode, IngestOptions, VectorIndexMode, VectorQueryOptions,
};
use holographic_memory::core::HmsConfig;
use holographic_memory::{
    DocumentInput, DocumentResult, EmbeddingSpace, HmsCore, MemorizeBatchItem, SearchOptions,
};
use serde_json::{json, Value};

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
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[derive(Clone)]
struct Doc {
    id: String,
    text: String,
    embedding: Vec<f32>,
}

struct Dataset {
    name: String,
    dim: usize,
    hv_dim: u32,
    encode: EncodeMode,
    base: Vec<Doc>,
    targets: Vec<Doc>,
    later: Vec<Doc>,
    queries: Vec<(String, Vec<f32>)>,
    batch: usize,
    graph_min_rows: usize,
    raw_vectors: usize,
}

struct Args {
    dataset: String,
    data: PathBuf,
    work: PathBuf,
    out: PathBuf,
    targets: usize,
    later: usize,
    base: usize,
    seed: u64,
    label: String,
}

fn parse_args() -> Result<Args> {
    let mut map = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        let Some(name) = key.strip_prefix("--") else {
            bail!("unexpected argument {key}");
        };
        map.insert(name.to_string(), args.next().context("missing value")?);
    }
    let get = |k: &str| map.get(k).cloned();
    Ok(Args {
        dataset: get("dataset").context("--dataset")?,
        data: get("data").context("--data")?.into(),
        work: get("work").context("--work")?.into(),
        out: get("out").context("--out")?.into(),
        targets: get("targets").map_or(Ok(100), |v| v.parse())?,
        later: get("later").map_or(Ok(20), |v| v.parse())?,
        base: get("base").map_or(Ok(50_000), |v| v.parse())?,
        seed: get("seed").map_or(Ok(20261021), |v| v.parse())?,
        label: get("label").unwrap_or_default(),
    })
}

fn read_f32(path: &Path, dim: usize, rows: usize) -> Result<Vec<Vec<f32>>> {
    let mut file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut bytes = vec![0u8; rows * dim * 4];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(dim * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect())
}

fn read_jsonl(path: &Path) -> Result<Vec<(String, String)>> {
    let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    std::io::BufReader::new(file)
        .lines()
        .map(|line| {
            let value: Value = serde_json::from_str(&line?)?;
            Ok((
                value["id"].as_str().context("id")?.to_string(),
                value["text"].as_str().context("text")?.to_string(),
            ))
        })
        .collect()
}

fn split(
    mut docs: Vec<Doc>,
    targets: usize,
    later: usize,
    rng: &mut Rng,
) -> (Vec<Doc>, Vec<Doc>, Vec<Doc>) {
    let mut order: Vec<usize> = (0..docs.len()).collect();
    rng.shuffle(&mut order);
    let held: BTreeSet<usize> = order[..targets + later].iter().copied().collect();
    let target_set: Vec<usize> = order[..targets].to_vec();
    let later_set: Vec<usize> = order[targets..targets + later].to_vec();
    let pick = |set: &[usize], docs: &[Doc]| set.iter().map(|&i| docs[i].clone()).collect();
    let t = pick(&target_set, &docs);
    let l = pick(&later_set, &docs);
    let mut i = 0;
    docs.retain(|_| {
        i += 1;
        !held.contains(&(i - 1))
    });
    (docs, t, l)
}

fn load(args: &Args, rng: &mut Rng) -> Result<Dataset> {
    match args.dataset.as_str() {
        "nfcorpus" => {
            let corpus = read_jsonl(&args.data.join("corpus.jsonl"))?;
            let vectors = read_f32(&args.data.join("corpus.f32"), 384, corpus.len())?;
            let docs = corpus
                .into_iter()
                .zip(vectors)
                .map(|((id, text), embedding)| Doc {
                    id,
                    text,
                    embedding,
                })
                .collect();
            let (base, targets, later) = split(docs, args.targets, args.later, rng);
            let queries = read_jsonl(&args.data.join("queries.jsonl"))?;
            let qvec = read_f32(&args.data.join("queries.f32"), 384, queries.len())?;
            Ok(Dataset {
                name: "nfcorpus".into(),
                dim: 384,
                hv_dim: 1024,
                encode: EncodeMode::Eager,
                base,
                targets,
                later,
                queries: queries
                    .into_iter()
                    .zip(qvec)
                    .take(20)
                    .map(|((_, text), v)| (text, v))
                    .collect(),
                batch: 256,
                graph_min_rows: 1000,
                raw_vectors: 64,
            })
        }
        "glove" => {
            let total = args.base + args.targets + args.later;
            let rows = read_f32(&args.data.join("train.f32"), 100, total)?;
            let docs: Vec<Doc> = rows
                .into_iter()
                .enumerate()
                .map(|(i, embedding)| Doc {
                    id: format!("g{i:07}"),
                    text: format!("item{i}"),
                    embedding,
                })
                .collect();
            let base = docs[..args.base].to_vec();
            let targets = docs[args.base..args.base + args.targets].to_vec();
            let later = docs[args.base + args.targets..].to_vec();
            let test = read_f32(&args.data.join("test.f32"), 100, 10)?;
            Ok(Dataset {
                name: "glove-100".into(),
                dim: 100,
                hv_dim: 64,
                encode: EncodeMode::Lexical,
                base,
                targets,
                later,
                queries: test.into_iter().map(|v| (String::new(), v)).collect(),
                batch: 4096,
                graph_min_rows: 1000,
                raw_vectors: 0,
            })
        }
        other => bail!("unknown dataset {other}"),
    }
}

fn config(ds: &Dataset) -> HmsConfig {
    HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: ds.name.clone(),
            revision: "1".into(),
            dimensions: ds.dim,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    }
}

fn input(doc: &Doc) -> Result<DocumentInput> {
    let embedding: Vec<f64> = doc.embedding.iter().map(|&v| f64::from(v)).collect();
    Ok(serde_json::from_value(json!({
        "id": doc.id,
        "text": doc.text,
        "chunkWords": 4096,
        "embeddings": [embedding],
    }))?)
}

fn open(dir: &Path, ds: &Dataset) -> Result<HmsCore> {
    HmsCore::new(ds.hv_dim, Some(dir.display().to_string()), Some(config(ds)))
}

fn ingest(core: &HmsCore, ds: &Dataset, docs: &[Doc]) -> Result<()> {
    let inputs = docs.iter().map(input).collect::<Result<Vec<_>>>()?;
    core.memorize_documents_with(inputs, &IngestOptions { encode: ds.encode })?;
    Ok(())
}

fn maintain(core: &HmsCore, ds: &Dataset) -> Result<()> {
    core.rebuild_vector_index(ds.graph_min_rows)?;
    core.train_document_index()?;
    Ok(())
}

/// Ingests the base corpus in batches, with `insert = (X, j)` committed before batch `j`.
fn build(dir: &Path, ds: &Dataset, insert: Option<(&Doc, usize)>) -> Result<HmsCore> {
    let core = open(dir, ds)?;
    if ds.raw_vectors > 0 {
        let items: Vec<MemorizeBatchItem> = (0..ds.raw_vectors)
            .map(|i| MemorizeBatchItem {
                id: format!("raw{i:03}"),
                text: format!(
                    "raw vector {i} {}",
                    ds.base[i]
                        .text
                        .split_whitespace()
                        .take(8)
                        .collect::<Vec<_>>()
                        .join(" ")
                ),
            })
            .collect();
        core.memorize_batch(&items)?;
    }
    let batches: Vec<&[Doc]> = ds.base.chunks(ds.batch).collect();
    for (j, batch) in batches.iter().enumerate() {
        if let Some((x, at)) = insert {
            if at == j {
                ingest(&core, ds, std::slice::from_ref(x))?;
            }
        }
        ingest(&core, ds, batch)?;
    }
    if let Some((x, at)) = insert {
        if at >= batches.len() {
            ingest(&core, ds, std::slice::from_ref(x))?;
        }
    }
    maintain(&core, ds)?;
    Ok(core)
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn hash_value(value: &Value) -> u64 {
    hash_bytes(&serde_json::to_vec(value).expect("serialize observation"))
}

fn results(r: Result<Vec<DocumentResult>>) -> Value {
    match r {
        Ok(hits) => serde_json::to_value(hits).expect("serialize results"),
        Err(e) => json!({ "error": e.to_string() }),
    }
}

fn top_score(v: &Value) -> f64 {
    v.get(0)
        .and_then(|h| h.get("score"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

fn ids(v: &Value) -> BTreeSet<String> {
    v.as_array()
        .map(|hits| {
            hits.iter()
                .filter_map(|h| h["documentId"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn overlap(a: &Value, b: &Value) -> f64 {
    let (a, b) = (ids(a), ids(b));
    if b.is_empty() {
        return 1.0;
    }
    a.intersection(&b).count() as f64 / b.len() as f64
}

/// The probes around one target: its text, its distinct terms, its embedding and its nearest
/// neighbours in the base corpus, plus a fixed query sample.
struct Probe {
    text: String,
    terms: Vec<String>,
    embedding: Vec<f64>,
    neighbours: Vec<Vec<f64>>,
}

fn probe_for(x: &Doc, ds: &Dataset) -> Probe {
    let text: String = x
        .text
        .split_whitespace()
        .take(48)
        .collect::<Vec<_>>()
        .join(" ");
    let mut seen = BTreeSet::new();
    let terms = x
        .text
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() >= 3 && seen.insert(t.clone()))
        .take(16)
        .collect();
    let norm = |v: &[f32]| -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / n).collect()
    };
    let q = norm(&x.embedding);
    let mut scored: Vec<(f32, usize)> = ds
        .base
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let n = norm(&d.embedding);
            (q.iter().zip(&n).map(|(a, b)| a * b).sum(), i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let neighbours = scored
        .iter()
        .take(5)
        .map(|&(_, i)| ds.base[i].embedding.iter().map(|&v| f64::from(v)).collect())
        .collect();
    Probe {
        text,
        terms,
        embedding: x.embedding.iter().map(|&v| f64::from(v)).collect(),
        neighbours,
    }
}

#[derive(Default)]
struct Obs {
    api: BTreeMap<String, u64>,
    features: Vec<f64>,
}

const FEATURES: [&str; 7] = [
    "graph_top1_x",
    "graph_exact_overlap_x",
    "graph_exact_overlap_nn",
    "lexical_top1_x",
    "term_top1_mean",
    "hybrid_top1_x",
    "graph_hits_k100_x",
];

fn observe(core: &HmsCore, ds: &Dataset, p: &Probe) -> Result<Obs> {
    let mut obs = Obs::default();
    let mut put = |name: String, value: &Value| {
        obs.api.insert(name, hash_value(value));
    };
    let lexical = |k: u32| SearchOptions {
        k: Some(k),
        lexical_weight: Some(1.0),
        semantic_weight: Some(0.0),
        ..Default::default()
    };
    let x_lex10 = results(core.search_documents(&p.text, &lexical(10)));
    put("lex/x/k10".into(), &x_lex10);
    put(
        "lex/x/k100".into(),
        &results(core.search_documents(&p.text, &lexical(100))),
    );
    put(
        "default/x/k10".into(),
        &results(core.search_documents(
            &p.text,
            &SearchOptions {
                k: Some(10),
                ..Default::default()
            },
        )),
    );
    let mut term_top = Vec::new();
    for term in &p.terms {
        let r = results(core.search_documents(term, &lexical(10)));
        term_top.push(top_score(&r));
        put(format!("term/{term}"), &r);
    }
    for (i, (text, _)) in ds
        .queries
        .iter()
        .enumerate()
        .filter(|(_, (t, _))| !t.is_empty())
    {
        put(
            format!("lex/q{i}"),
            &results(core.search_documents(text, &lexical(10))),
        );
    }
    let hybrid = |embedding: &[f64], fusion: &str| SearchOptions {
        k: Some(10),
        embedding: Some(embedding.to_vec()),
        fusion: Some(fusion.into()),
        lexical_weight: Some(1.0),
        semantic_weight: Some(1.0),
        min_semantic_score: Some(-1.0),
        ..Default::default()
    };
    let x_hyb = results(core.search_documents(&p.text, &hybrid(&p.embedding, "blend")));
    put("hybrid/x/blend".into(), &x_hyb);
    put(
        "hybrid/x/rrf".into(),
        &results(core.search_documents(&p.text, &hybrid(&p.embedding, "rrf"))),
    );
    for (i, (text, v)) in ds.queries.iter().enumerate().take(10) {
        let v: Vec<f64> = v.iter().map(|&x| f64::from(x)).collect();
        put(
            format!("hybrid/q{i}"),
            &results(core.search_documents(text, &hybrid(&v, "blend"))),
        );
    }
    let vector = |e: &[f64], k: usize, mode: VectorIndexMode| {
        results(core.search_vectors(
            e,
            k,
            &VectorQueryOptions {
                mode,
                ..Default::default()
            },
        ))
    };
    let x_graph = vector(&p.embedding, 10, VectorIndexMode::Graph);
    let x_exact = vector(&p.embedding, 10, VectorIndexMode::Exact);
    let x_graph100 = vector(&p.embedding, 100, VectorIndexMode::Graph);
    put("vgraph/x/k10".into(), &x_graph);
    put("vexact/x/k10".into(), &x_exact);
    put("vgraph/x/k100".into(), &x_graph100);
    put(
        "vexact/x/k100".into(),
        &vector(&p.embedding, 100, VectorIndexMode::Exact),
    );
    put(
        "vauto/x/k10".into(),
        &vector(&p.embedding, 10, VectorIndexMode::Auto),
    );
    let mut nn_overlap = Vec::new();
    for (i, e) in p.neighbours.iter().enumerate() {
        let g = vector(e, 10, VectorIndexMode::Graph);
        let x = vector(e, 10, VectorIndexMode::Exact);
        nn_overlap.push(overlap(&g, &x));
        put(format!("vgraph/nn{i}"), &g);
        put(format!("vexact/nn{i}"), &x);
    }
    for (i, (_, v)) in ds.queries.iter().enumerate().take(10) {
        let v: Vec<f64> = v.iter().map(|&x| f64::from(x)).collect();
        put(
            format!("vgraph/q{i}"),
            &vector(&v, 10, VectorIndexMode::Graph),
        );
        put(
            format!("vexact/q{i}"),
            &vector(&v, 10, VectorIndexMode::Exact),
        );
    }
    let mut status = serde_json::to_value(core.vector_index_status())?;
    let generation = status["generation"].take();
    put("status/vector_index_generation".into(), &generation);
    put("status/vector_index".into(), &status);
    put("status/revision".into(), &json!(core.revision()));
    put(
        "status/document_count".into(),
        &json!(core.document_count()),
    );
    put(
        "status/document_index_trained".into(),
        &json!(core.document_index_trained()),
    );
    put(
        "status/index_status".into(),
        &serde_json::to_value(core.index_status())?,
    );
    put(
        "list/all".into(),
        &serde_json::to_value(core.list_documents(None, 1 << 20))?,
    );
    let mean = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    obs.features = vec![
        top_score(&x_graph),
        overlap(&x_graph, &x_exact),
        mean(&nn_overlap),
        top_score(&x_lex10),
        mean(&term_top),
        top_score(&x_hyb),
        x_graph100.as_array().map_or(0, Vec::len) as f64,
    ];
    Ok(obs)
}

/// Every file under `dir` except the lock, keyed by its path with generation directory names
/// normalised (they are reported separately), valued by a hash of length and content.
fn files(dir: &Path) -> Result<(BTreeMap<String, u64>, Vec<String>)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(&path, out)?;
            } else {
                out.push(path);
            }
        }
        Ok(())
    }
    let mut all = Vec::new();
    walk(dir, &mut all)?;
    let mut generations = BTreeSet::new();
    let mut keyed = Vec::new();
    for path in all {
        let rel = path.strip_prefix(dir)?.display().to_string();
        if rel == ".hms.lock" {
            continue;
        }
        let mut key = String::new();
        for (i, part) in rel.split('/').enumerate() {
            if i > 0 {
                key.push('/');
            }
            if part.starts_with("gen-") {
                generations.insert(part.to_string());
                key.push_str("gen-*");
            } else {
                key.push_str(part);
            }
        }
        keyed.push((key, path));
    }
    let mut map = BTreeMap::new();
    for (key, path) in keyed {
        let mut bytes = std::fs::read(&path)?;
        // Small files (CURRENT, indices.json) name the generation; its name is reported apart.
        if bytes.len() <= 1 << 22 {
            for name in &generations {
                bytes = replace_all(&bytes, name.as_bytes(), b"gen-*");
            }
        }
        map.insert(
            key,
            hash_bytes(&bytes) ^ (bytes.len() as u64).rotate_left(17),
        );
    }
    Ok((map, generations.into_iter().collect()))
}

fn replace_all(haystack: &[u8], needle: &[u8], with: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(haystack.len());
    let mut i = 0;
    while i < haystack.len() {
        if haystack[i..].starts_with(needle) {
            out.extend_from_slice(with);
            i += needle.len();
        } else {
            out.push(haystack[i]);
            i += 1;
        }
    }
    out
}

fn differing(a: &BTreeMap<String, u64>, b: &BTreeMap<String, u64>) -> Vec<String> {
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    keys.into_iter()
        .filter(|k| a.get(*k) != b.get(*k))
        .cloned()
        .collect()
}

fn component(probe: &str) -> String {
    let mut parts = probe.split('/');
    let head = parts.next().unwrap_or_default();
    match head {
        "status" | "list" => format!("{head}/{}", parts.next().unwrap_or_default()),
        _ => head.to_string(),
    }
}

fn ln_choose(n: usize, k: usize) -> f64 {
    (1..=k).map(|i| ((n - k + i) as f64 / i as f64).ln()).sum()
}

fn binom_cdf(k: usize, n: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
    }
    (0..=k)
        .map(|i| (ln_choose(n, i) + i as f64 * p.ln() + (n - i) as f64 * (1.0 - p).ln()).exp())
        .sum::<f64>()
        .min(1.0)
}

/// Two-sided 95% Clopper-Pearson interval for `k` successes in `n` trials.
fn clopper_pearson(k: usize, n: usize) -> [f64; 2] {
    let bisect = |f: &dyn Fn(f64) -> bool| {
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if f(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let lower = if k == 0 {
        0.0
    } else {
        bisect(&|p| 1.0 - binom_cdf(k - 1, n, p) >= 0.025)
    };
    let upper = if k == n {
        1.0
    } else {
        bisect(&|p| binom_cdf(k, n, p) <= 0.025)
    };
    [lower, upper]
}

/// Percentile bootstrap (2000 resamples of targets) of the mean of per-target values.
fn bootstrap(values: &[f64], rng: &mut Rng) -> [f64; 2] {
    if values.is_empty() {
        return [0.0, 0.0];
    }
    let mut means: Vec<f64> = (0..2000)
        .map(|_| {
            (0..values.len())
                .map(|_| values[rng.below(values.len())])
                .sum::<f64>()
                / values.len() as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    [means[50], means[1949]]
}

/// One target's attacker features on the store that deleted it and on the never-stored store.
type Pair = (Vec<f64>, Vec<f64>);

/// The shadow-free attacker: the single feature, direction and threshold with the best
/// accuracy on the even targets, scored on the odd targets (one deleted and one never-stored
/// observation each).
fn ghost_echoes(pairs: &[Pair], rng: &mut Rng) -> Value {
    let (train, test): (Vec<_>, Vec<_>) = pairs.iter().enumerate().partition(|(i, _)| i % 2 == 0);
    let accuracy = |set: &[(usize, &Pair)], f: usize, sign: f64, t: f64| -> f64 {
        let mut correct = 0usize;
        for (_, (deleted, never)) in set {
            correct += usize::from(sign * deleted[f] > t) + usize::from(sign * never[f] <= t);
        }
        correct as f64 / (2 * set.len()).max(1) as f64
    };
    let mut best = (0.5, 0usize, 1.0f64, f64::INFINITY);
    for f in 0..FEATURES.len() {
        for sign in [1.0, -1.0] {
            let mut thresholds: Vec<f64> = train
                .iter()
                .flat_map(|(_, (d, n))| [sign * d[f], sign * n[f]])
                .collect();
            thresholds.push(f64::INFINITY);
            for t in thresholds {
                let acc = accuracy(&train, f, sign, t);
                if acc > best.0 {
                    best = (acc, f, sign, t);
                }
            }
        }
    }
    let (_, f, sign, t) = best;
    let per_target: Vec<f64> = test
        .iter()
        .map(|(_, (d, n))| {
            (f64::from(u8::from(sign * d[f] > t)) + f64::from(u8::from(sign * n[f] <= t))) / 2.0
        })
        .collect();
    let acc = per_target.iter().sum::<f64>() / per_target.len().max(1) as f64;
    json!({
        "feature": FEATURES[f],
        "direction": sign,
        "train_accuracy": best.0,
        "test_accuracy": acc,
        "test_accuracy_ci95": bootstrap(&per_target, rng),
        "test_targets": per_target.len(),
    })
}

type Files = (BTreeMap<String, u64>, Vec<String>);

#[derive(Default)]
struct Tally {
    api: Vec<BTreeSet<String>>,
    files: Vec<BTreeSet<String>>,
    generation_names_differ: usize,
    pairs: Vec<Pair>,
}

/// Trials in which each component (probe family or file) differed, outside `unstable`.
fn components(
    diffs: &[BTreeSet<String>],
    unstable: &BTreeSet<String>,
    name: fn(&str) -> String,
) -> (BTreeMap<String, usize>, BTreeMap<String, usize>) {
    let (mut stable, mut randomized) = (BTreeMap::new(), BTreeMap::new());
    for diff in diffs {
        let (mut s, mut r) = (BTreeSet::new(), BTreeSet::new());
        for probe in diff {
            if unstable.contains(probe) {
                r.insert(name(probe));
            } else {
                s.insert(name(probe));
            }
        }
        for c in s {
            *stable.entry(c).or_insert(0) += 1;
        }
        for c in r {
            *randomized.entry(c).or_insert(0) += 1;
        }
    }
    (stable, randomized)
}

impl Tally {
    fn add(&mut self, a: &Obs, b: &Obs, fa: &Files, fb: &Files) {
        self.api
            .push(differing(&a.api, &b.api).into_iter().collect());
        self.files
            .push(differing(&fa.0, &fb.0).into_iter().collect());
        self.generation_names_differ += usize::from(fa.1 != fb.1);
        self.pairs.push((a.features.clone(), b.features.clone()));
    }

    /// Probes and files that differ between two builds of the never-stored store: values that
    /// are randomized per store instance, which carry no information about the history.
    fn unstable(&self) -> (BTreeSet<String>, BTreeSet<String>) {
        (
            self.api.iter().flatten().cloned().collect(),
            self.files.iter().flatten().cloned().collect(),
        )
    }

    fn report(&self, unstable: &(BTreeSet<String>, BTreeSet<String>), rng: &mut Rng) -> Value {
        let n = self.api.len();
        let flags = |diffs: &[BTreeSet<String>], unstable: &BTreeSet<String>| -> Vec<bool> {
            diffs
                .iter()
                .map(|d| d.iter().any(|p| !unstable.contains(p)))
                .collect()
        };
        let api_flags = flags(&self.api, &unstable.0);
        let file_flags = flags(&self.files, &unstable.1);
        let raw_flags = flags(&self.api, &BTreeSet::new());
        let (api_components, api_randomized) = components(&self.api, &unstable.0, component);
        let (file_components, _) = components(&self.files, &unstable.1, str::to_string);
        let summary = |flags: &[bool], rng: &mut Rng| {
            let k = flags.iter().filter(|&&d| d).count();
            let per_target: Vec<f64> = flags.iter().map(|&d| if d { 1.0 } else { 0.5 }).collect();
            let cp = clopper_pearson(k, n);
            json!({
                "differ": k,
                "p_differ": k as f64 / n.max(1) as f64,
                "p_differ_cp95": cp,
                "oracle_accuracy": 0.5 + 0.5 * k as f64 / n.max(1) as f64,
                "oracle_accuracy_bootstrap95": bootstrap(&per_target, rng),
                "oracle_accuracy_cp95": [0.5 + 0.5 * cp[0], 0.5 + 0.5 * cp[1]],
            })
        };
        let any: Vec<bool> = api_flags
            .iter()
            .zip(&file_flags)
            .map(|(a, f)| *a || *f)
            .collect();
        json!({
            "targets": n,
            "api": summary(&api_flags, rng),
            "api_including_randomized_fields": summary(&raw_flags, rng),
            "files": summary(&file_flags, rng),
            "api_or_files": summary(&any, rng),
            "api_components": api_components,
            "api_randomized_components": api_randomized,
            "file_components": file_components,
            "generation_names_differ": self.generation_names_differ,
            "ghost_echoes": ghost_echoes(&self.pairs, rng),
        })
    }
}

const CHECKPOINTS: [&str; 5] = ["present", "deleted", "purged", "reopened", "readmitted"];

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut rng = Rng(args.seed);
    let ds = load(&args, &mut rng)?;
    std::fs::create_dir_all(&args.work)?;
    let started = Instant::now();
    let fresh = |name: &str| -> Result<PathBuf> {
        let dir = args.work.join(name);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(dir)
    };
    let batches = ds.base.len().div_ceil(ds.batch);
    eprintln!(
        "{}: base {} targets {} later {} batches {batches}",
        ds.name,
        ds.base.len(),
        ds.targets.len(),
        ds.later.len()
    );

    // Shadow stores, one per checkpoint state, built once.
    let t = Instant::now();
    let b_live = build(&fresh("b-live")?, &ds, None)?;
    let build_seconds = t.elapsed().as_secs_f64();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let b_control = pool.install(|| build(&fresh("b-control")?, &ds, None))?;
    let b_purged = build(&fresh("b-purged")?, &ds, None)?;
    b_purged.compact_scrubbing()?;
    let reopened_dir = fresh("b-reopened")?;
    drop({
        let core = build(&reopened_dir, &ds, None)?;
        core.compact_scrubbing()?;
        core
    });
    let b_reopened = open(&reopened_dir, &ds)?;
    maintain(&b_reopened, &ds)?;
    let readmitted_dir = fresh("b-readmitted")?;
    drop({
        let core = build(&readmitted_dir, &ds, None)?;
        core.compact_scrubbing()?;
        core
    });
    let b_readmitted = open(&readmitted_dir, &ds)?;
    maintain(&b_readmitted, &ds)?;
    ingest(&b_readmitted, &ds, &ds.later)?;
    maintain(&b_readmitted, &ds)?;
    let shadow_files = [
        files(&args.work.join("b-live"))?,
        files(&args.work.join("b-purged"))?,
        files(&reopened_dir)?,
        files(&readmitted_dir)?,
    ];
    let control_files = files(&args.work.join("b-control"))?;
    eprintln!(
        "shadow stores built in {:.1}s (one build {build_seconds:.1}s)",
        started.elapsed().as_secs_f64()
    );

    let mut tallies: Vec<Tally> = CHECKPOINTS.iter().map(|_| Tally::default()).collect();
    let mut control = Tally::default();
    let a_dir = args.work.join("a");
    for (trial, x) in ds.targets.iter().enumerate() {
        let t = Instant::now();
        let at = rng.below(batches + 1);
        let probe = probe_for(x, &ds);
        if a_dir.exists() {
            std::fs::remove_dir_all(&a_dir)?;
        }
        let a = build(&a_dir, &ds, Some((x, at)))?;
        let b_obs = observe(&b_live, &ds, &probe)?;
        control.add(
            &observe(&b_control, &ds, &probe)?,
            &b_obs,
            &control_files,
            &shadow_files[0],
        );
        // present: X stored (positive control)
        tallies[0].add(
            &observe(&a, &ds, &probe)?,
            &b_obs,
            &files(&a_dir)?,
            &shadow_files[0],
        );
        ensure!(a.delete_document(&x.id)?, "target {} was not stored", x.id);
        tallies[1].add(
            &observe(&a, &ds, &probe)?,
            &b_obs,
            &files(&a_dir)?,
            &shadow_files[0],
        );
        a.compact_scrubbing()?;
        tallies[2].add(
            &observe(&a, &ds, &probe)?,
            &observe(&b_purged, &ds, &probe)?,
            &files(&a_dir)?,
            &shadow_files[1],
        );
        drop(a);
        let a = open(&a_dir, &ds)?;
        maintain(&a, &ds)?;
        tallies[3].add(
            &observe(&a, &ds, &probe)?,
            &observe(&b_reopened, &ds, &probe)?,
            &files(&a_dir)?,
            &shadow_files[2],
        );
        ingest(&a, &ds, &ds.later)?;
        maintain(&a, &ds)?;
        tallies[4].add(
            &observe(&a, &ds, &probe)?,
            &observe(&b_readmitted, &ds, &probe)?,
            &files(&a_dir)?,
            &shadow_files[3],
        );
        drop(a);
        let differ: Vec<String> = tallies
            .iter()
            .map(|t| {
                format!(
                    "{}{}",
                    t.api.last().map_or(0, BTreeSet::len),
                    t.files.last().map_or(0, BTreeSet::len)
                )
            })
            .collect();
        eprintln!(
            "trial {trial} {} at batch {at}: api/files differ {} ({:.1}s)",
            x.id,
            differ.join(" "),
            t.elapsed().as_secs_f64()
        );
    }
    // The shadow stores must not have changed under the probes.
    ensure!(
        files(&args.work.join("b-live"))?.0 == shadow_files[0].0,
        "shadow store files changed during the run"
    );

    let unstable = control.unstable();
    let mut checkpoints = serde_json::Map::new();
    for (name, tally) in CHECKPOINTS.iter().zip(&tallies) {
        checkpoints.insert((*name).into(), tally.report(&unstable, &mut rng));
    }
    let report = json!({
        "experiment": "forget-indistinguishability",
        "label": args.label,
        "dataset": ds.name,
        "base_documents": ds.base.len(),
        "targets": ds.targets.len(),
        "later_documents": ds.later.len(),
        "raw_vectors": ds.raw_vectors,
        "batch": ds.batch,
        "graph_min_rows": ds.graph_min_rows,
        "hypervector_dimensions": ds.hv_dim,
        "encode": format!("{:?}", ds.encode),
        "seed": args.seed,
        "checkpoints": checkpoints,
        "randomized_probes": unstable.0,
        "randomized_files": unstable.1,
        "determinism_control_one_thread": control.report(&(BTreeSet::new(), BTreeSet::new()), &mut rng),
        "seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}
