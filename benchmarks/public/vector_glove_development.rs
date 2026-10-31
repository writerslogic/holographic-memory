use anyhow::{ensure, Context, Result};
use serde_json::json;
use std::{
    fs,
    io::{BufWriter, Read, Write},
    path::Path,
    time::Instant,
};
use writerslogic_warm::core::qgraph::{Residency, SearchKnobs, VGraph, VMetric, VSearchParams};

const N: usize = 65536;
const Q: usize = 256;
const D: usize = 100;
const K: usize = 10;
const SEED: u64 = 539365412;

fn rows(path: &Path, count: usize) -> Result<(Vec<f32>, u32)> {
    ensure!(
        fs::metadata(path)?.len() == (count * D * 4) as u64,
        "wrong matrix size"
    );
    let bytes = fs::read(path)?;
    let crc = crc32fast::hash(&bytes);
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(values.iter().all(|x| x.is_finite()), "nonfinite matrix");
    ensure!(
        values
            .as_chunks::<D>()
            .0
            .iter()
            .all(|r| r.iter().any(|&x| x != 0.0)),
        "zero angular row"
    );
    Ok((values, crc))
}

fn file_crc(path: &Path) -> Result<u32> {
    let mut file = fs::File::open(path)?;
    let mut hash = crc32fast::Hasher::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            return Ok(hash.finalize());
        }
        hash.update(&buffer[..n]);
    }
}

fn order(len: usize, seed: u64) -> Vec<usize> {
    let mut keys: Vec<_> = (0..len)
        .map(|id| {
            let mut z = seed.wrapping_add((id as u64).wrapping_mul(0x9e3779b97f4a7c15));
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            (z ^ (z >> 31), id)
        })
        .collect();
    keys.sort_unstable();
    keys.into_iter().map(|(_, id)| id).collect()
}

fn exact_truth(base: &[f32], queries: &[f32]) -> Vec<Vec<u32>> {
    let normalized: Vec<f64> = base
        .as_chunks::<D>()
        .0
        .iter()
        .flat_map(|row| {
            let norm = row
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            row.iter().map(move |&x| f64::from(x) / norm)
        })
        .collect();
    queries
        .as_chunks::<D>()
        .0
        .iter()
        .map(|query| {
            let norm = query
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            let q: Vec<_> = query.iter().map(|&x| f64::from(x) / norm).collect();
            let mut scores: Vec<_> = normalized
                .as_chunks::<D>()
                .0
                .iter()
                .enumerate()
                .map(|(id, row)| {
                    (
                        row.iter().zip(&q).map(|(x, y)| x * y).sum::<f64>(),
                        id as u32,
                    )
                })
                .collect();
            let compare = |a: &(f64, u32), b: &(f64, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
            scores.select_nth_unstable_by(K, compare);
            scores[..K].sort_unstable_by(compare);
            scores[..K].iter().map(|&(_, id)| id).collect()
        })
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(args.len() >= 4, "usage: vector-glove-development INPUT INDEX OUTPUT [--repeats 7] [--efs 10,20,40,80,120,160,240,400,640] [--knobs off|env]");
    let input = Path::new(&args[1]);
    let index_path = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    ensure!(!output.exists(), "output already exists");
    let mut repeats = 7usize;
    let mut efs = vec![10usize, 20, 40, 80, 120, 160, 240, 400, 640];
    let mut knob_mode = "off";
    ensure!((args.len() - 4) % 2 == 0, "options require values");
    for pair in args[4..].as_chunks::<2>().0 {
        let value = pair[1].to_str().context("option is not UTF-8")?;
        match pair[0].to_str() {
            Some("--repeats") => repeats = value.parse()?,
            Some("--efs") => {
                efs = value
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<_>, _>>()?
            }
            Some("--knobs") => knob_mode = value,
            _ => anyhow::bail!("unknown option"),
        }
    }
    ensure!((1..=1000).contains(&repeats), "repeats must be 1..=1000");
    ensure!(
        !efs.is_empty() && efs.len() <= 16 && efs.iter().all(|ef| (K..=4096).contains(ef)),
        "invalid ef grid"
    );
    let mut unique = efs.clone();
    unique.sort_unstable();
    unique.dedup();
    ensure!(unique.len() == efs.len(), "duplicate ef");
    ensure!(repeats * efs.len() <= 1000, "at most 1000 repeat/ef cells");
    let knobs = match knob_mode {
        "off" => SearchKnobs::OFF,
        "env" => SearchKnobs::from_env(),
        _ => anyhow::bail!("knobs must be off or env"),
    };
    let declaration_path = input.join("declaration.json");
    let declaration_bytes = fs::read(&declaration_path)?;
    let declaration: serde_json::Value = serde_json::from_slice(&declaration_bytes)?;
    ensure!(
        declaration["official_test_arrays_read"] == false
            && declaration["construction_count"] == N
            && declaration["development_count"] == Q
            && declaration["construction_development_overlap"] == 0,
        "input declaration mismatch"
    );
    let input = input.canonicalize()?;
    let (base, base_crc) = rows(&input.join("construction.f32"), N)?;
    let (queries, queries_crc) = rows(&input.join("development.f32"), Q)?;
    let truth_started = Instant::now();
    let truth = exact_truth(&base, &queries);
    let truth_s = truth_started.elapsed().as_secs_f64();
    drop(base);
    let index_path = index_path.canonicalize()?;
    let serialized_bytes = fs::metadata(&index_path)?.len();
    ensure!(
        serialized_bytes <= 1024 * 1024 * 1024,
        "index exceeds 1 GiB harness bound"
    );
    let index_crc = file_crc(&index_path)?;
    let index = VGraph::open(&index_path, Residency::Heap).context("open saved index")?;
    ensure!(
        index.len() == N && index.dim() == D && index.metric() == VMetric::Angular,
        "index geometry mismatch"
    );
    let mut searcher = index.searcher_with(knobs);
    let mut out = Vec::with_capacity(K);
    for &ef in &efs {
        for query in queries.as_chunks::<D>().0 {
            searcher.search(
                query,
                K,
                VSearchParams {
                    ef,
                    rerank: 64,
                    ..Default::default()
                },
                &mut out,
            )?;
        }
    }
    let mut trials = Vec::with_capacity(repeats * efs.len());
    eprintln!("QUERY_STARTED");
    for repeat in 0..repeats {
        let query_order = order(Q, SEED.wrapping_add(repeat as u64));
        for ei in order(
            efs.len(),
            SEED ^ (repeat as u64).wrapping_mul(0x9e3779b97f4a7c15),
        ) {
            let ef = efs[ei];
            let params = VSearchParams {
                ef,
                rerank: 64,
                ..Default::default()
            };
            let mut latencies_ns = vec![0u64; Q];
            let mut evaluations = vec![0usize; Q];
            let mut ids: Vec<_> = (0..Q).map(|_| Vec::<u32>::with_capacity(K)).collect();
            let batch_started = Instant::now();
            for &qi in &query_order {
                let started = Instant::now();
                let work = searcher.search(&queries[qi * D..(qi + 1) * D], K, params, &mut out)?;
                latencies_ns[qi] = started.elapsed().as_nanos().try_into()?;
                evaluations[qi] = work;
                ids[qi].extend_from_slice(&out);
            }
            let batch_ns: u64 = batch_started.elapsed().as_nanos().try_into()?;
            ensure!(
                latencies_ns.iter().all(|&n| n > 0),
                "timer resolution insufficient"
            );
            for result in &ids {
                ensure!(
                    result.len() == K && result.iter().all(|&id| (id as usize) < N),
                    "invalid result IDs"
                );
                let mut unique = result.clone();
                unique.sort_unstable();
                unique.dedup();
                ensure!(unique.len() == K, "duplicate result IDs");
            }
            trials.push(json!({"repeat": repeat, "ef": ef, "query_order": query_order, "latencies_ns": latencies_ns, "batch_ns": batch_ns, "ids": ids, "distance_evaluations": evaluations}));
        }
    }
    eprintln!("QUERY_FINISHED");
    let result = json!({
        "schema": "warm-glove-development-v1", "development_only": true,
        "input": input, "index": index_path, "input_declaration": declaration,
        "binding_crc32": {"construction.f32": base_crc, "development.f32": queries_crc, "declaration.json": crc32fast::hash(&declaration_bytes), "index": index_crc},
        "construction_count": N, "query_count": Q, "dim": D, "k": K, "rerank": 64,
        "efs": efs, "repeats": repeats, "order_seed": SEED, "warmup_passes_per_ef": 1,
        "knob_mode": knob_mode,
        "knobs": {"prefetch": knobs.prefetch, "batch_est": knobs.batch_est, "frontier": knobs.frontier, "block_screen": knobs.block_screen, "rerank_z": knobs.rerank_z},
        "truth": truth, "truth_generation_s": truth_s,
        "truth_method": "Exhaustive cosine on source f32 values promoted to f64; f64 normalization and sequential dot sum; descending cosine then ascending local ID for ties. Independent of native f32 normalization and quantization.",
        "residency": "Heap", "logical_index_bytes": index.index_bytes(), "serialized_index_bytes": serialized_bytes,
        "memory_scope": "Logical native index bytes and serialized file bytes only; neither RSS, peak, USS nor allocator retention.",
        "timing_scope": "Individual latency includes one Instant elapsed call and native search; batch also includes timers, work/result copies, loop overhead. Ground truth, index open, warmup, validation and JSON serialization excluded. Repeats in one process are conditional diagnostics, not independent hardware replications.",
        "trials": trials
    });
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &result)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_cosine_ignores_scale_and_breaks_ties_by_local_id() {
        let mut base = vec![0.0f32; 12 * D];
        for (id, row) in base.as_chunks_mut::<D>().0.iter_mut().enumerate() {
            row[0] = if id == 11 { -1.0 } else { (id + 1) as f32 };
            row[1] = if id < 2 || id == 11 {
                0.0
            } else {
                (id * id) as f32
            };
        }
        let mut query = vec![0.0f32; D];
        query[0] = 9.0;
        assert_eq!(
            exact_truth(&base, &query),
            vec![(0..10).collect::<Vec<u32>>()]
        );
        query[0] = -3.0;
        assert_eq!(
            exact_truth(&base, &query)[0],
            vec![11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        );
    }

    #[test]
    fn fixture_reader_rejects_bad_length_zero_and_nonfinite_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.f32");
        for size in [D * 4 - 1, D * 4 + 1] {
            fs::write(&path, vec![0u8; size]).unwrap();
            assert!(rows(&path, 1).is_err());
        }
        for value in [0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bytes = vec![0u8; D * 4];
            bytes[..4].copy_from_slice(&value.to_le_bytes());
            fs::write(&path, bytes).unwrap();
            assert!(rows(&path, 1).is_err());
        }
        let mut bytes = vec![0u8; D * 4];
        bytes[..4].copy_from_slice(&1.0f32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        let (matrix, crc) = rows(&path, 1).unwrap();
        assert_eq!(matrix[0], 1.0);
        assert_eq!(crc, crc32fast::hash(&bytes));
    }
}
