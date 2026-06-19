// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use holographic_memory::core::algebra::HolographicAlgebra;
use holographic_memory::core::clifford::CliffordVec;
use holographic_memory::core::entangled::EntangledHVec;
use holographic_memory::HmsCore;
use std::time::Instant;
use tempfile::TempDir;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|a| a == "--json");

    let dim = 16384;
    let scales = [100, 500, 1000, 5000];

    if !json_output {
        println!("HMS Stress Test & Benchmark Report");
        println!("===================================");
        println!("Dimension: {}", dim);
        println!("Scales: {:?}", scales);
        println!();
    }

    let mut results = Vec::new();

    // --- Micro-benchmarks: EntangledHVec vs CliffordVec ---
    let micro = run_micro_benchmarks(dim);
    if !json_output {
        print_micro(&micro);
    }
    results.push(serde_json::to_value(&micro).unwrap());

    // --- Similarity distribution analysis ---
    let sim_dist = run_similarity_distribution(dim, 1000);
    if !json_output {
        print_sim_dist(&sim_dist);
    }
    results.push(serde_json::to_value(&sim_dist).unwrap());

    // --- Query scaling ---
    let mut scaling_results = Vec::new();
    for &n in &scales {
        let sr = run_query_scaling(dim, n);
        if !json_output {
            print_scaling(&sr);
        }
        scaling_results.push(sr);
    }
    results.push(serde_json::to_value(&scaling_results).unwrap());

    // --- Retrieval quality comparison ---
    let quality = run_retrieval_quality(dim, 500, 50);
    if !json_output {
        print_quality(&quality);
    }
    results.push(serde_json::to_value(&quality).unwrap());

    // --- Codebook capacity estimation ---
    let capacity = run_capacity_estimation(dim);
    if !json_output {
        print_capacity(&capacity);
    }
    results.push(serde_json::to_value(&capacity).unwrap());

    if json_output {
        let report = serde_json::json!({
            "micro_benchmarks": results[0],
            "similarity_distribution": results[1],
            "query_scaling": results[2],
            "retrieval_quality": results[3],
            "capacity_estimation": results[4],
            "config": {
                "dim": dim,
                "scales": scales,
            }
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}

#[derive(serde::Serialize)]
struct MicroBenchmark {
    operation: String,
    entangled_ns: f64,
    clifford_ns: f64,
    ratio: f64,
}

fn run_micro_benchmarks(dim: usize) -> Vec<MicroBenchmark> {
    let iterations = 10_000;
    let mut results = Vec::new();

    let e1 = EntangledHVec::new_deterministic(dim, 1);
    let e2 = EntangledHVec::new_deterministic(dim, 2);
    let c1 = CliffordVec::from_seed(dim, 1);
    let c2 = CliffordVec::from_seed(dim, 2);

    // from_seed / new_deterministic
    let t = time_ns(iterations, || {
        std::hint::black_box(EntangledHVec::new_deterministic(dim, 42));
    });
    let tc = time_ns(iterations, || {
        std::hint::black_box(CliffordVec::from_seed(dim, 42));
    });
    results.push(MicroBenchmark {
        operation: "from_seed".into(),
        entangled_ns: t,
        clifford_ns: tc,
        ratio: tc / t,
    });

    // similarity
    let t = time_ns(iterations, || {
        std::hint::black_box(e1.similarity(&e2));
    });
    let tc = time_ns(iterations, || {
        std::hint::black_box(c1.similarity(&c2));
    });
    results.push(MicroBenchmark {
        operation: "similarity".into(),
        entangled_ns: t,
        clifford_ns: tc,
        ratio: tc / t,
    });

    // bind
    let t = time_ns(iterations, || {
        std::hint::black_box(e1.bind(&e2));
    });
    let tc = time_ns(iterations, || {
        std::hint::black_box(c1.bind(&c2));
    });
    results.push(MicroBenchmark {
        operation: "bind".into(),
        entangled_ns: t,
        clifford_ns: tc,
        ratio: tc / t,
    });

    // bundle 10
    let evecs: Vec<EntangledHVec> = (0..10)
        .map(|i| EntangledHVec::new_deterministic(dim, i))
        .collect();
    let cvecs: Vec<CliffordVec> = (0..10)
        .map(|i| CliffordVec::from_seed(dim, i))
        .collect();
    let t = time_ns(iterations / 10, || {
        std::hint::black_box(EntangledHVec::bundle(&evecs));
    });
    let tc = time_ns(iterations / 10, || {
        std::hint::black_box(CliffordVec::bundle(&cvecs));
    });
    results.push(MicroBenchmark {
        operation: "bundle_10".into(),
        entangled_ns: t,
        clifford_ns: tc,
        ratio: tc / t,
    });

    // permute
    let t = time_ns(iterations, || {
        std::hint::black_box(e1.permute(7));
    });
    let tc = time_ns(iterations, || {
        std::hint::black_box(c1.permute(7));
    });
    results.push(MicroBenchmark {
        operation: "permute".into(),
        entangled_ns: t,
        clifford_ns: tc,
        ratio: tc / t,
    });

    results
}

fn time_ns(iterations: usize, mut f: impl FnMut()) -> f64 {
    // Warmup
    for _ in 0..iterations / 10 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

#[derive(serde::Serialize)]
struct SimDistribution {
    representation: String,
    self_sim_mean: f64,
    random_pair_mean: f64,
    random_pair_std: f64,
    random_pair_min: f64,
    random_pair_max: f64,
    bind_self_recovery: f64,
}

fn run_similarity_distribution(dim: usize, n_pairs: usize) -> Vec<SimDistribution> {
    let mut results = Vec::new();

    // EntangledHVec
    let mut self_sims = Vec::with_capacity(100);
    let mut pair_sims = Vec::with_capacity(n_pairs);
    for i in 0..100 {
        let v = EntangledHVec::new_deterministic(dim, i);
        self_sims.push(v.similarity(&v));
    }
    for i in 0..n_pairs {
        let a = EntangledHVec::new_deterministic(dim, i as u64 * 2);
        let b = EntangledHVec::new_deterministic(dim, i as u64 * 2 + 1);
        pair_sims.push(a.similarity(&b));
    }
    let bind_a = EntangledHVec::new_deterministic(dim, 1000);
    let bind_b = EntangledHVec::new_deterministic(dim, 2000);
    let ab = bind_a.bind(&bind_b);
    let recovered = ab.bind(&bind_b);
    results.push(SimDistribution {
        representation: "EntangledHVec".into(),
        self_sim_mean: mean(&self_sims),
        random_pair_mean: mean(&pair_sims),
        random_pair_std: std_dev(&pair_sims),
        random_pair_min: pair_sims.iter().cloned().fold(f64::INFINITY, f64::min),
        random_pair_max: pair_sims
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        bind_self_recovery: recovered.similarity(&bind_a),
    });

    // CliffordVec
    let mut self_sims = Vec::with_capacity(100);
    let mut pair_sims = Vec::with_capacity(n_pairs);
    for i in 0..100 {
        let v = CliffordVec::from_seed(dim, i);
        self_sims.push(v.similarity(&v));
    }
    for i in 0..n_pairs {
        let a = CliffordVec::from_seed(dim, i as u64 * 2);
        let b = CliffordVec::from_seed(dim, i as u64 * 2 + 1);
        pair_sims.push(a.similarity(&b));
    }
    let bind_a = CliffordVec::from_seed(dim, 1000);
    let bind_b = CliffordVec::from_seed(dim, 2000);
    let ab = bind_a.bind(&bind_b);
    let b_rev = bind_b.reverse();
    let recovered = ab.bind(&b_rev);
    results.push(SimDistribution {
        representation: "CliffordVec".into(),
        self_sim_mean: mean(&self_sims),
        random_pair_mean: mean(&pair_sims),
        random_pair_std: std_dev(&pair_sims),
        random_pair_min: pair_sims.iter().cloned().fold(f64::INFINITY, f64::min),
        random_pair_max: pair_sims
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        bind_self_recovery: recovered.similarity(&bind_a),
    });

    results
}

#[derive(serde::Serialize)]
struct ScalingResult {
    n_patterns: usize,
    insert_total_ms: f64,
    brute_force_query_us: f64,
    hopfield_query_us: f64,
    hopfield_slowdown: f64,
    brute_top1_id: String,
    hopfield_top1_id: String,
    top1_agreement: bool,
}

fn run_query_scaling(dim: usize, n: usize) -> ScalingResult {
    let dir = TempDir::new().unwrap();
    let hms = HmsCore::new(dim as u32, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    let insert_start = Instant::now();
    for i in 0..n {
        let text = format!(
            "Document about {} discussing {} with reference to {}",
            TOPICS[i % TOPICS.len()],
            VERBS[i % VERBS.len()],
            OBJECTS[i % OBJECTS.len()]
        );
        let vec = hms.encode_text(&text);
        hms.memorize(format!("doc_{}", i), vec).unwrap();
    }
    let insert_ms = insert_start.elapsed().as_secs_f64() * 1000.0;

    let query_text = "Document about science discussing analysis";
    let q_vec = hms.encode_text(query_text);

    let n_queries = 100;
    let bf_start = Instant::now();
    let mut bf_result = Vec::new();
    for _ in 0..n_queries {
        bf_result = hms.query(&q_vec, 5);
    }
    let bf_us = bf_start.elapsed().as_secs_f64() * 1_000_000.0 / n_queries as f64;

    let hf_start = Instant::now();
    let mut hf_result = Vec::new();
    for _ in 0..n_queries {
        hf_result = hms.query_hopfield(&q_vec, 5);
    }
    let hf_us = hf_start.elapsed().as_secs_f64() * 1_000_000.0 / n_queries as f64;

    let bf_top1 = bf_result.first().map(|r| r.id.clone()).unwrap_or_default();
    let hf_top1 = hf_result.first().map(|r| r.id.clone()).unwrap_or_default();

    ScalingResult {
        n_patterns: n,
        insert_total_ms: insert_ms,
        brute_force_query_us: bf_us,
        hopfield_query_us: hf_us,
        hopfield_slowdown: hf_us / bf_us,
        top1_agreement: bf_top1 == hf_top1,
        brute_top1_id: bf_top1,
        hopfield_top1_id: hf_top1,
    }
}

#[derive(serde::Serialize)]
struct QualityResult {
    n_patterns: usize,
    n_queries: usize,
    top1_agreement_rate: f64,
    top5_overlap_rate: f64,
    hopfield_avg_results: f64,
    hopfield_sparsity_rate: f64,
}

fn run_retrieval_quality(dim: usize, n_patterns: usize, n_queries: usize) -> QualityResult {
    let dir = TempDir::new().unwrap();
    let hms = HmsCore::new(dim as u32, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

    for i in 0..n_patterns {
        let text = format!(
            "The {} {} the {} near the {}",
            TOPICS[i % TOPICS.len()],
            VERBS[i % VERBS.len()],
            OBJECTS[i % OBJECTS.len()],
            TOPICS[(i + 3) % TOPICS.len()]
        );
        let vec = hms.encode_text(&text);
        hms.memorize(format!("doc_{}", i), vec).unwrap();
    }

    let mut top1_agree = 0;
    let mut top5_overlap_sum = 0.0;
    let mut total_hf_results = 0usize;
    let mut sparse_count = 0;

    for q in 0..n_queries {
        let text = format!(
            "{} {} the {}",
            TOPICS[q % TOPICS.len()],
            VERBS[(q + 1) % VERBS.len()],
            OBJECTS[(q + 2) % OBJECTS.len()]
        );
        let q_vec = hms.encode_text(&text);

        let bf = hms.query(&q_vec, 5);
        let hf = hms.query_hopfield(&q_vec, 5);

        if let (Some(bf1), Some(hf1)) = (bf.first(), hf.first()) {
            if bf1.id == hf1.id {
                top1_agree += 1;
            }
        }

        let bf_ids: std::collections::HashSet<&str> = bf.iter().map(|r| r.id.as_str()).collect();
        let hf_ids: std::collections::HashSet<&str> = hf.iter().map(|r| r.id.as_str()).collect();
        let overlap = bf_ids.intersection(&hf_ids).count();
        let union = bf_ids.union(&hf_ids).count();
        if union > 0 {
            top5_overlap_sum += overlap as f64 / union as f64;
        }

        total_hf_results += hf.len();
        if hf.len() < 5 {
            sparse_count += 1;
        }
    }

    QualityResult {
        n_patterns,
        n_queries,
        top1_agreement_rate: top1_agree as f64 / n_queries as f64,
        top5_overlap_rate: top5_overlap_sum / n_queries as f64,
        hopfield_avg_results: total_hf_results as f64 / n_queries as f64,
        hopfield_sparsity_rate: sparse_count as f64 / n_queries as f64,
    }
}

#[derive(serde::Serialize)]
struct CapacityEstimate {
    dim: usize,
    active_bits: usize,
    entangled_theoretical_capacity: f64,
    clifford_n: usize,
    clifford_algebra_dim: usize,
    clifford_terms: usize,
    measured_entangled_crosstalk: Vec<(usize, f64)>,
}

fn run_capacity_estimation(dim: usize) -> CapacityEstimate {
    let active = dim / 256;
    let n = ((dim as f64).log2().ceil()) as usize;

    // Theoretical: for sparse binary with rho = 1/256,
    // capacity ~ C(D, D*rho) / noise_floor
    // Simplified: patterns recoverable ~ D / (rho * ln(N)) for N patterns
    let theoretical = (dim as f64) / (active as f64).ln();

    // Empirical crosstalk measurement
    let mut crosstalk = Vec::new();
    for &n_pat in &[10, 50, 100, 500, 1000] {
        let patterns: Vec<EntangledHVec> = (0..n_pat)
            .map(|i| EntangledHVec::new_deterministic(dim, i as u64 * 7 + 13))
            .collect();

        let mut max_cross = 0.0f64;
        let sample_size = n_pat.min(100);
        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                let sim = patterns[i].similarity(&patterns[j]);
                max_cross = max_cross.max(sim);
            }
        }
        crosstalk.push((n_pat, max_cross));
    }

    CapacityEstimate {
        dim,
        active_bits: active,
        entangled_theoretical_capacity: theoretical,
        clifford_n: n,
        clifford_algebra_dim: 1 << n,
        clifford_terms: 64,
        measured_entangled_crosstalk: crosstalk,
    }
}

// --- Printing helpers ---

fn print_micro(benchmarks: &[MicroBenchmark]) {
    println!("Micro-Benchmarks: EntangledHVec vs CliffordVec");
    println!("{:-<70}", "");
    println!(
        "{:<15} {:>12} {:>12} {:>10}",
        "Operation", "Entangled", "Clifford", "Ratio"
    );
    println!(
        "{:<15} {:>12} {:>12} {:>10}",
        "", "(ns)", "(ns)", "(C/E)"
    );
    println!("{:-<70}", "");
    for b in benchmarks {
        println!(
            "{:<15} {:>12.1} {:>12.1} {:>10.2}x",
            b.operation, b.entangled_ns, b.clifford_ns, b.ratio
        );
    }
    println!();
}

fn print_sim_dist(dists: &[SimDistribution]) {
    println!("Similarity Distribution Analysis");
    println!("{:-<70}", "");
    for d in dists {
        println!("  {}", d.representation);
        println!("    Self-similarity (mean):   {:.6}", d.self_sim_mean);
        println!(
            "    Random pair (mean/std):   {:.6} +/- {:.6}",
            d.random_pair_mean, d.random_pair_std
        );
        println!(
            "    Random pair (min/max):    {:.6} / {:.6}",
            d.random_pair_min, d.random_pair_max
        );
        println!("    Bind-unbind recovery:     {:.6}", d.bind_self_recovery);
    }
    println!();
}

fn print_scaling(s: &ScalingResult) {
    println!("Query Scaling: {} patterns", s.n_patterns);
    println!("{:-<70}", "");
    println!("  Insert total:        {:.1} ms", s.insert_total_ms);
    println!("  Brute-force query:   {:.1} us", s.brute_force_query_us);
    println!("  Hopfield query:      {:.1} us", s.hopfield_query_us);
    println!("  Hopfield slowdown:   {:.2}x", s.hopfield_slowdown);
    println!("  Top-1 agreement:     {}", s.top1_agreement);
    println!(
        "  BF top-1: {}  HF top-1: {}",
        s.brute_top1_id, s.hopfield_top1_id
    );
    println!();
}

fn print_quality(q: &QualityResult) {
    println!(
        "Retrieval Quality: {} patterns, {} queries",
        q.n_patterns, q.n_queries
    );
    println!("{:-<70}", "");
    println!("  Top-1 agreement:     {:.1}%", q.top1_agreement_rate * 100.0);
    println!(
        "  Top-5 Jaccard:       {:.1}%",
        q.top5_overlap_rate * 100.0
    );
    println!(
        "  Hopfield avg results: {:.1}",
        q.hopfield_avg_results
    );
    println!(
        "  Hopfield sparse rate: {:.1}% (returned < 5)",
        q.hopfield_sparsity_rate * 100.0
    );
    println!();
}

fn print_capacity(c: &CapacityEstimate) {
    println!("Capacity Estimation");
    println!("{:-<70}", "");
    println!("  Dimension:             {}", c.dim);
    println!("  Active bits (rho):     {} (1/256)", c.active_bits);
    println!(
        "  Theoretical capacity:  {:.0} patterns",
        c.entangled_theoretical_capacity
    );
    println!("  Clifford n:            {} (Cl({},0))", c.clifford_n, c.clifford_n);
    println!("  Clifford algebra dim:  {}", c.clifford_algebra_dim);
    println!("  Clifford sparse terms: {}", c.clifford_terms);
    println!("  Crosstalk by pattern count:");
    for (n, cross) in &c.measured_entangled_crosstalk {
        println!("    N={:<5}  max similarity = {:.6}", n, cross);
    }
    println!();
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

const TOPICS: &[&str] = &[
    "science", "history", "art", "music", "math", "physics", "biology",
    "chemistry", "literature", "philosophy", "engineering", "medicine",
    "ecology", "astronomy", "geology", "economics", "psychology",
    "linguistics", "anthropology", "sociology",
];

const VERBS: &[&str] = &[
    "explores", "analyzes", "discusses", "examines", "reveals",
    "compares", "demonstrates", "investigates", "questions", "describes",
];

const OBJECTS: &[&str] = &[
    "patterns", "structures", "processes", "relationships", "theories",
    "models", "systems", "networks", "hierarchies", "transformations",
];
