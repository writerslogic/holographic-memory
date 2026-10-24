use anyhow::{ensure, Context, Result};
use holographic_memory::core::qgraph::{
    diagnostics::TraversalDiagnostic, BuildParams, Graph, Metric, Residency, SearchKnobs, VGraph,
    VSearchParams,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const MAX_ROWS: usize = 65536;
const MAX_MATRIX_BYTES: usize = 64 * 1024 * 1024;
const FIXTURE_DECLARATION: &str =
    "773451815a234dfb7036db97b6e09a1214762f4d1264c0749f46d24ac5fb8c0d";
const FIXTURE_INDEX: &str = "84881d9f5041f63b5042cb5d9161792e49d999a02e307599bc4fd114bf916899";
const FIXTURE_GRAPH: u64 = 0x64840985d7e672d7;

struct Input {
    base: Vec<f32>,
    queries: Vec<f32>,
    dim: usize,
    degree: usize,
    build_ef: usize,
    efs: Vec<usize>,
    index: Option<PathBuf>,
    bindings: Value,
}

fn matrix_bytes(rows: usize, dim: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "row count outside 1..=65536"
    );
    ensure!((1..=8192).contains(&dim), "dimension outside 1..=8192");
    rows.checked_mul(dim)
        .and_then(|n| n.checked_mul(4))
        .filter(|&bytes| bytes <= MAX_MATRIX_BYTES)
        .context("matrix exceeds 64 MiB")
}

fn file_sha(path: &Path, maximum: u64) -> Result<String> {
    ensure!(fs::metadata(path)?.len() <= maximum, "file exceeds bound");
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut total = 0u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok(digest
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect());
        }
        total += count as u64;
        ensure!(total <= maximum, "file grew beyond bound");
        digest.update(&buffer[..count]);
    }
}

fn read_bounded(path: &Path, maximum: usize, expected: &str) -> Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    let size = file.metadata()?.len();
    ensure!(size <= maximum as u64, "file exceeds allocation bound");
    let mut bytes = Vec::with_capacity(size as usize + 1);
    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= maximum && bytes.len() as u64 == size,
        "file changed while reading"
    );
    ensure!(
        Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            == expected,
        "input hash mismatch"
    );
    Ok(bytes)
}

fn read_matrix(path: &Path, rows: usize, dim: usize, expected: &str) -> Result<Vec<f32>> {
    let size = matrix_bytes(rows, dim)?;
    let bytes = read_bounded(path, size, expected)?;
    ensure!(bytes.len() == size, "wrong matrix size");
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(values.iter().all(|x| x.is_finite()), "nonfinite matrix");
    ensure!(
        values
            .chunks_exact(dim)
            .all(|row| row.iter().any(|&x| x != 0.0)),
        "zero angular row"
    );
    Ok(values)
}

fn read_ids(path: &Path, count: usize, expected: &str) -> Result<HashSet<u32>> {
    let bytes = read_bounded(path, count * 4, expected)?;
    ensure!(bytes.len() == count * 4, "wrong ID matrix size");
    let ids: HashSet<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    ensure!(ids.len() == count, "duplicate partition ID");
    Ok(ids)
}

fn prepared(input: &Path, index: &Path) -> Result<Input> {
    let input = input.canonicalize()?;
    let declaration_path = input.join("declaration.json");
    let declaration: Value = serde_json::from_slice(&read_bounded(
        &declaration_path,
        16384,
        FIXTURE_DECLARATION,
    )?)?;
    ensure!(
        declaration["official_test_arrays_read"] == false
            && declaration["construction_count"] == MAX_ROWS
            && declaration["development_count"] == 256
            && declaration["construction_development_overlap"] == 0,
        "invalid train-only declaration"
    );
    let required = [
        "construction.f32",
        "development.f32",
        "construction.ids.u32",
        "development.ids.u32",
    ];
    let files = declaration["files"]
        .as_object()
        .context("missing input bindings")?;
    ensure!(files.len() == required.len(), "unexpected input bindings");
    for name in required {
        ensure!(
            files.get(name).and_then(Value::as_str).is_some(),
            "missing input hash: {name}"
        );
    }
    let expected = |name: &str| files[name].as_str().expect("validated input hash");
    let construction = read_ids(
        &input.join("construction.ids.u32"),
        MAX_ROWS,
        expected("construction.ids.u32"),
    )?;
    let development = read_ids(
        &input.join("development.ids.u32"),
        256,
        expected("development.ids.u32"),
    )?;
    ensure!(construction.is_disjoint(&development), "partition overlap");
    ensure!(
        file_sha(index, 1024 * 1024 * 1024)? == FIXTURE_INDEX,
        "saved index hash mismatch"
    );
    Ok(Input {
        base: read_matrix(
            &input.join("construction.f32"),
            MAX_ROWS,
            100,
            expected("construction.f32"),
        )?,
        queries: read_matrix(
            &input.join("development.f32"),
            256,
            100,
            expected("development.f32"),
        )?,
        dim: 100,
        degree: 64,
        build_ef: 500,
        efs: vec![80, 240],
        index: Some(index.canonicalize()?),
        bindings: json!({
            "kind": "frozen_train_only_fixture", "input": input,
            "declaration_sha256": FIXTURE_DECLARATION, "files": files,
            "index_sha256": FIXTURE_INDEX, "graph_hash": format!("{FIXTURE_GRAPH:016x}"),
            "official_test_arrays_read": false, "partition_ids_checked_disjoint": true,
        }),
    })
}

fn synthetic() -> Input {
    let dim = 32;
    let row = |id: usize| {
        (0..dim).map(move |j| {
            let z = ((id as u64 + 1) * 0x9e3779b9) ^ ((j as u64 + 3) * 0x85ebca6b);
            ((z ^ (z >> 17)) % 10007) as f32 / 5003.0 - 1.0
        })
    };
    Input {
        base: (0..257).flat_map(row).collect(),
        queries: (257..265).flat_map(row).collect(),
        dim,
        degree: 16,
        build_ef: 80,
        efs: vec![10, 32, 64],
        index: None,
        bindings: json!({"kind": "deterministic_synthetic", "construction_rows": 257, "development_rows": 8, "official_test_arrays_read": false}),
    }
}

fn exact_truth(base: &[f32], query: &[f32], dim: usize, k: usize) -> Result<Vec<u32>> {
    ensure!(
        dim > 0 && base.len().is_multiple_of(dim),
        "invalid base geometry"
    );
    let rows = base.len() / dim;
    matrix_bytes(rows, dim)?;
    ensure!(
        query.len() == dim && (1..=rows).contains(&k),
        "invalid query geometry or k"
    );
    ensure!(
        base.iter().chain(query).all(|x| x.is_finite()),
        "nonfinite truth input"
    );
    let qnorm = query
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(qnorm > 0.0, "zero truth query");
    let mut scores = Vec::with_capacity(rows);
    for (id, row) in base.chunks_exact(dim).enumerate() {
        let norm = row
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        ensure!(norm > 0.0, "zero truth row");
        let score = row
            .iter()
            .zip(query)
            .map(|(&x, &q)| (f64::from(x) / norm) * (f64::from(q) / qnorm))
            .sum::<f64>();
        scores.push((score, id as u32));
    }
    let compare = |a: &(f64, u32), b: &(f64, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    if k < rows {
        scores.select_nth_unstable_by(k, compare);
    }
    scores[..k].sort_unstable_by(compare);
    Ok(scores[..k].iter().map(|&(_, id)| id).collect())
}

fn checked_set(ids: &[u32], n: usize) -> Result<HashSet<u32>> {
    ensure!(ids.iter().all(|&id| (id as usize) < n), "ID outside index");
    let set: HashSet<_> = ids.iter().copied().collect();
    ensure!(set.len() == ids.len(), "duplicate diagnostic ID");
    Ok(set)
}

fn classify(
    truth: &[u32],
    report: &TraversalDiagnostic,
    n: usize,
) -> Result<BTreeMap<&'static str, Vec<u32>>> {
    ensure!(
        (1..=MAX_ROWS).contains(&n) && !truth.is_empty(),
        "invalid truth/index size"
    );
    checked_set(truth, n)?;
    let evaluated = checked_set(&report.evaluated, n)?;
    let beam = checked_set(&report.beam, n)?;
    let retained = checked_set(&report.retained, n)?;
    let reranked = checked_set(&report.rerank_candidates, n)?;
    let returned = checked_set(&report.results, n)?;
    ensure!(
        beam.is_subset(&evaluated) && retained.is_subset(&evaluated),
        "unscored beam/retention ID"
    );
    ensure!(
        reranked.is_subset(&retained) && returned.is_subset(&retained),
        "invalid final candidate boundary"
    );
    ensure!(
        reranked.is_empty() || returned.is_subset(&reranked),
        "returned ID was not reranked"
    );
    let mut classes: BTreeMap<_, Vec<_>> = [
        "never_scored",
        "scored_not_retained",
        "retained_not_reranked",
        "reranked_not_returned",
        "returned",
    ]
    .into_iter()
    .map(|name| (name, Vec::new()))
    .collect();
    for &id in truth {
        let name = if returned.contains(&id) {
            "returned"
        } else if !evaluated.contains(&id) {
            "never_scored"
        } else if !retained.contains(&id) {
            "scored_not_retained"
        } else if !reranked.contains(&id) {
            "retained_not_reranked"
        } else {
            "reranked_not_returned"
        };
        classes
            .get_mut(name)
            .expect("fixed classification names")
            .push(id);
    }
    Ok(classes)
}

fn run(input: Input, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "output already exists");
    let n = input.base.len() / input.dim;
    let q = input.queries.len() / input.dim;
    ensure!(q <= 256, "at most 256 development queries");
    let k = 10;
    let truth = input
        .queries
        .chunks_exact(input.dim)
        .map(|query| exact_truth(&input.base, query, input.dim, k))
        .collect::<Result<Vec<_>>>()?;
    let graph = Graph::build_owned(
        input.base,
        input.dim,
        &BuildParams {
            metric: Metric::Angular,
            degree: input.degree,
            build_ef: input.build_ef,
            quantized_search: true,
            ..Default::default()
        },
    )?;
    let mut index = if let Some(path) = &input.index {
        ensure!(
            graph.structure_hash() == FIXTURE_GRAPH,
            "rebuilt source graph differs from frozen fixture"
        );
        let index = VGraph::open(path, Residency::Heap)?;
        ensure!(
            file_sha(path, 1024 * 1024 * 1024)? == FIXTURE_INDEX,
            "index changed while opening"
        );
        index
    } else {
        VGraph::from_graph_with(&graph, 8, 8, true, true, 3, false)?
    };
    let before_bytes = index.index_bytes();
    let mut settings = Vec::new();
    for ef in input.efs {
        let params = VSearchParams {
            ef,
            rerank: 64,
            ..Default::default()
        };
        let mut queries = Vec::new();
        let mut totals = BTreeMap::<&str, usize>::new();
        for (qi, query) in input.queries.chunks_exact(input.dim).enumerate() {
            let mut production = Vec::new();
            let evaluations =
                index
                    .searcher_with(SearchKnobs::OFF)
                    .search(query, k, params, &mut production)?;
            let report = index.diagnose(&graph, query, k, params, None, false)?;
            ensure!(
                report.results == production && production.len() == k,
                "diagnostic results differ from ordinary search"
            );
            let classes = classify(&truth[qi], &report, n)?;
            for (&name, ids) in &classes {
                *totals.entry(name).or_default() += ids.len();
            }
            queries.push(json!({
                "query_id": qi, "truth": truth[qi], "classification": classes,
                "production_ordered_ids": production, "production_distance_evaluations": evaluations,
                "trace": {"evaluated": report.evaluated, "beam": report.beam, "retained": report.retained, "rerank_candidates": report.rerank_candidates, "results": report.results},
            }));
        }
        ensure!(
            totals.values().sum::<usize>() == q * k,
            "classification is not a partition of truth"
        );
        settings.push(json!({"ef": ef, "rerank": params.rerank, "truth_class_counts": totals, "queries": queries}));
    }
    ensure!(
        index.index_bytes() == before_bytes,
        "diagnosis changed index geometry"
    );
    let result = json!({
        "schema": "hms-vector-discovery-diagnosis-v1", "diagnostic_only": true,
        "input_bindings": input.bindings, "construction_count": n, "query_count": q,
        "dim": input.dim, "k": k, "graph_hash": format!("{:016x}", graph.structure_hash()),
        "logical_index_bytes": before_bytes, "settings": settings,
        "truth_method": "Exhaustive cosine on original f32 values promoted to f64; f64 norms and sequential normalized-product sum; descending score then original local ID for ties",
        "scope": "Candidate-stage attribution only; no timing, speedup, competitor, held-out leaderboard, or memory-win claim. Evaluated IDs are unique; production work also counts repeated/descent/rerank evaluations and is not the same metric.",
    });
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &result)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    println!(
        "{}",
        json!({"output": output, "sha256": file_sha(output, 1024 * 1024 * 1024)?})
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    match args.get(1).and_then(|s| s.to_str()) {
        Some("synthetic") if args.len() == 3 => run(synthetic(), Path::new(&args[2])),
        Some("prepared") if args.len() == 5 => run(
            prepared(Path::new(&args[2]), Path::new(&args[3]))?,
            Path::new(&args[4]),
        ),
        _ => anyhow::bail!(
            "usage: vector_discovery_diagnosis synthetic OUTPUT | prepared INPUT INDEX OUTPUT"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_exclusive_and_validates_boundaries() {
        let mut trace = TraversalDiagnostic {
            evaluated: vec![1, 2, 3, 4],
            beam: vec![2, 3, 4],
            retained: vec![2, 3, 4],
            rerank_candidates: vec![3, 4],
            results: vec![4],
        };
        let result = classify(&[0, 1, 2, 3, 4], &trace, 5).unwrap();
        for (name, id) in [
            ("never_scored", 0),
            ("scored_not_retained", 1),
            ("retained_not_reranked", 2),
            ("reranked_not_returned", 3),
            ("returned", 4),
        ] {
            assert_eq!(result[name], vec![id]);
        }
        assert!(classify(&[4], &trace, 4).is_err());
        assert!(classify(&[], &trace, 5).is_err());
        assert!(classify(&[4, 4], &trace, 5).is_err());
        trace.retained.push(0);
        assert!(classify(&[0], &trace, 5).is_err());
        trace.retained.pop();
        trace.rerank_candidates = vec![1];
        assert!(classify(&[1], &trace, 5).is_err());
        trace.rerank_candidates.clear();
        assert_eq!(
            classify(&[3, 4], &trace, 5).unwrap()["retained_not_reranked"],
            vec![3]
        );
    }

    #[test]
    fn exact_cosine_handles_ties_scaling_and_topk_edges() {
        let base = [1.0, 0.0, 2.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        assert_eq!(exact_truth(&base, &[7.0, 0.0], 2, 1).unwrap(), vec![0]);
        assert_eq!(
            exact_truth(&base, &[7.0, 0.0], 2, 3).unwrap(),
            vec![0, 1, 2]
        );
        assert_eq!(
            exact_truth(&base, &[7.0, 0.0], 2, 4).unwrap(),
            vec![0, 1, 2, 3]
        );
        assert!(exact_truth(&base, &[7.0, 0.0], 2, 5).is_err());
        assert!(exact_truth(&base, &[7.0, 0.0], 2, 0).is_err());
        assert!(exact_truth(&base, &[0.0, 0.0], 2, 1).is_err());
        assert!(exact_truth(&base, &[f32::NAN, 0.0], 2, 1).is_err());
    }

    #[test]
    fn matrix_bounds_precede_allocation() {
        assert!(matrix_bytes(0, 1).is_err());
        assert_eq!(matrix_bytes(MAX_ROWS - 1, 1).unwrap(), (MAX_ROWS - 1) * 4);
        assert_eq!(matrix_bytes(MAX_ROWS, 1).unwrap(), MAX_ROWS * 4);
        assert!(matrix_bytes(MAX_ROWS + 1, 1).is_err());
        assert!(matrix_bytes(1, 0).is_err());
        assert!(matrix_bytes(1, 8191).is_ok());
        assert!(matrix_bytes(1, 8192).is_ok());
        assert!(matrix_bytes(1, 8193).is_err());
        assert!(matrix_bytes(16383, 1024).is_ok());
        assert_eq!(matrix_bytes(16384, 1024).unwrap(), MAX_MATRIX_BYTES);
        assert!(matrix_bytes(16385, 1024).is_err());
    }
}
