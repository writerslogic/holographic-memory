// HMS scaling benchmark: capacity walls, noise tolerance, throughput, and memory
// at multiple dimensions and densities.
//
// Usage:
//   hms-scaling --dim 65536 --density 256 --json
//   hms-scaling --dim 65536 --density 1024 --json
//   hms-scaling --all --json

use holographic_memory::core::bloom_memory::HolographicBloomMemory;
use holographic_memory::core::entangled::{hash_u64, EntangledHVec, DEFAULT_RHO_DENOM};
use holographic_memory::core::hopfield::{hopfield_query, HopfieldConfig};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|a| a == "--json");
    let run_all = args.iter().any(|a| a == "--all");

    let dims_and_densities: Vec<(usize, usize)> = if run_all {
        vec![
            (16384, 256),
            (65536, 256),
            (65536, 1024),
            (131072, 256),
            (131072, 1024),
            (262144, 1024),
            (262144, 4096),
            (524288, 1024),
            (524288, 4096),
        ]
    } else {
        let dim = args
            .iter()
            .position(|a| a == "--dim")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(16384);
        let density = args
            .iter()
            .position(|a| a == "--density")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_RHO_DENOM);
        vec![(dim, density)]
    };

    let mut all_results = Vec::new();

    for &(dim, density_denom) in &dims_and_densities {
        let active = dim / density_denom;
        if !json_output {
            eprintln!("[D={} density=1/{} active={}]", dim, density_denom, active);
        }
        let result = run_scaling_suite(dim, density_denom, json_output);
        all_results.push(result);
    }

    if json_output {
        let report = serde_json::json!({ "scaling_benchmark": all_results });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}

fn run_scaling_suite(dim: usize, density_denom: usize, json_output: bool) -> serde_json::Value {
    let active = dim / density_denom;
    let t_start = Instant::now();

    if !json_output {
        eprintln!("  capacity wall...");
    }
    let capacity_wall = find_capacity_wall(dim, density_denom);

    if !json_output {
        eprintln!("  structured retrieval stress...");
    }
    let struct_stress = structured_stress(dim, density_denom);

    if !json_output {
        eprintln!("  noise tolerance...");
    }
    let noise_curve = noise_tolerance(dim, density_denom);

    if !json_output {
        eprintln!("  throughput...");
    }
    let throughput = measure_throughput(dim, density_denom);

    let bytes_per_item = active * 4;
    let bytes_float32 = dim * 4;
    let elapsed_s = t_start.elapsed().as_secs_f64();

    serde_json::json!({
        "dim": dim,
        "density_denom": density_denom,
        "active_indices": active,
        "elapsed_seconds": elapsed_s,
        "capacity_wall": capacity_wall,
        "structured_retrieval_stress": struct_stress,
        "noise_tolerance": noise_curve,
        "throughput": throughput,
        "memory": {
            "bytes_per_sparse_item": bytes_per_item,
            "bytes_per_dense_float32": bytes_float32,
            "compression_ratio": bytes_float32 as f64 / bytes_per_item as f64,
        },
    })
}

fn measure_at_n(
    dim: usize,
    density_denom: usize,
    n_items: usize,
    probe_size: usize,
) -> serde_json::Value {
    let items: Vec<EntangledHVec> = (0..n_items)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, i as u64 * 137 + 7))
        .collect();

    let mut mem = HolographicBloomMemory::new(dim);
    mem.insert_batch(&items);
    let density = mem.density();

    let probe_n = probe_size.min(n_items);
    let mut member_scores = Vec::new();
    for item in items.iter().take(probe_n) {
        member_scores.push(mem.contains(item));
    }

    let mut non_member_scores = Vec::new();
    for i in 0..probe_size {
        let nm = EntangledHVec::new_with_density(dim, density_denom, 999000 + i as u64 * 31);
        non_member_scores.push(mem.contains(&nm));
    }

    let recall = member_scores.iter().filter(|&&s| s >= 0.5).count() as f64 / probe_n as f64;
    let fpr = non_member_scores.iter().filter(|&&s| s >= 0.5).count() as f64 / probe_size as f64;
    let member_min = member_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let non_member_max = non_member_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let d_prime = d_prime_val(
        mean(&member_scores),
        std_dev(&member_scores),
        mean(&non_member_scores),
        std_dev(&non_member_scores),
    );
    let member_mean = mean(&member_scores);
    let non_member_mean = mean(&non_member_scores);

    serde_json::json!({
        "n_items": n_items,
        "density": density,
        "recall": recall,
        "fpr": fpr,
        "member_min": member_min,
        "member_mean": member_mean,
        "non_member_max": non_member_max,
        "non_member_mean": non_member_mean,
        "gap": member_min - non_member_max,
        "d_prime": d_prime,
    })
}

fn find_capacity_wall(dim: usize, density_denom: usize) -> serde_json::Value {
    let probe_size = 50;
    let mut results = Vec::new();
    let base = density_denom.max(256) / 256;
    let mut n_items = 50 * base;
    let mut last_pass = 0usize;

    loop {
        let measurement = measure_at_n(dim, density_denom, n_items, probe_size);
        let recall = measurement["recall"].as_f64().unwrap_or(0.0);
        results.push(measurement);

        if recall < 0.95 {
            break;
        }
        last_pass = n_items;

        if n_items < 200 * base {
            n_items += 50 * base;
        } else if n_items < 1000 * base {
            n_items += 100 * base;
        } else if n_items < 5000 * base {
            n_items += 500 * base;
        } else if n_items < 20000 * base {
            n_items += 2000 * base;
        } else {
            n_items += 5000 * base;
        }

        if n_items > 200000 {
            break;
        }
    }

    let first_fail = results
        .last()
        .and_then(|r| r["n_items"].as_u64())
        .unwrap_or(0) as usize;
    if last_pass > 0 && first_fail > last_pass && (first_fail - last_pass) > base {
        let mut lo = last_pass;
        let mut hi = first_fail;
        let min_step = base.max(1);
        while hi - lo > min_step {
            let mid = (lo + hi) / 2;
            let mid_aligned = (mid / min_step) * min_step;
            if mid_aligned <= lo || mid_aligned >= hi {
                break;
            }
            let measurement = measure_at_n(dim, density_denom, mid_aligned, probe_size);
            let recall = measurement["recall"].as_f64().unwrap_or(0.0);
            results.push(measurement);
            if recall >= 0.95 {
                lo = mid_aligned;
            } else {
                hi = mid_aligned;
            }
        }
        results.sort_by_key(|r| r["n_items"].as_u64().unwrap_or(0));
    }

    let wall = results
        .iter()
        .rev()
        .find(|r| r["recall"].as_f64().unwrap_or(0.0) >= 0.95)
        .map(|r| r["n_items"].as_u64().unwrap_or(0))
        .unwrap_or(0);

    let soft_wall = results
        .iter()
        .rev()
        .find(|r| r["d_prime"].as_f64().unwrap_or(0.0) >= 2.0)
        .map(|r| r["n_items"].as_u64().unwrap_or(0))
        .unwrap_or(0);

    let gap_wall = results
        .iter()
        .rev()
        .find(|r| r["gap"].as_f64().unwrap_or(0.0) > 0.0)
        .map(|r| r["n_items"].as_u64().unwrap_or(0))
        .unwrap_or(0);

    serde_json::json!({
        "wall_at_95_recall": wall,
        "soft_wall_dprime_2": soft_wall,
        "gap_wall_positive_gap": gap_wall,
        "sweep": results,
    })
}

fn structured_stress(dim: usize, density_denom: usize) -> serde_json::Value {
    let vocab_size = 50;
    let n_facts = 40;
    let role_counts = [2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100];

    let entities: Vec<EntangledHVec> = (0..vocab_size)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, 0xE000 + i as u64))
        .collect();

    let mut results = Vec::new();
    for &n_roles in &role_counts {
        let active = dim / density_denom;
        let comp_density_est = 1.0 - (1.0 - active as f64 / dim as f64).powi(n_roles);
        if comp_density_est > 0.98 {
            break;
        }

        let mut facts: Vec<Vec<usize>> = Vec::new();
        let mut comps = Vec::new();
        for t in 0..n_facts {
            let mut fillers = Vec::new();
            for r in 0..n_roles {
                let idx = hash_u64(0x7000 + r as u64, t as u64) as usize % vocab_size;
                fillers.push(idx);
            }
            let bindings: Vec<EntangledHVec> = fillers
                .iter()
                .enumerate()
                .map(|(r, &idx)| entities[idx].permute(r + 1))
                .collect();
            comps.push(EntangledHVec::bundle_bloom(&bindings));
            facts.push(fillers);
        }

        let mut correct = 0;
        let total = n_facts * n_roles;
        for (t, fillers) in facts.iter().enumerate() {
            let comp = &comps[t];
            for (r, &true_idx) in fillers.iter().enumerate() {
                let mut best = 0;
                let mut best_s = f64::NEG_INFINITY;
                for (e, ent) in entities.iter().enumerate() {
                    let s = ent.permute(r + 1).containment_similarity(comp);
                    if s > best_s {
                        best_s = s;
                        best = e;
                    }
                }
                if best == true_idx {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f64 / total as f64;
        results.push(serde_json::json!({
            "n_roles": n_roles,
            "accuracy": accuracy,
            "composition_density": comp_density_est,
            "total_queries": total,
        }));

        if accuracy < 0.5 {
            break;
        }
    }

    serde_json::json!(results)
}

fn noise_tolerance(dim: usize, density_denom: usize) -> serde_json::Value {
    let n_items = 100;
    let n_probes = 30;
    let corruptions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    let items: Vec<EntangledHVec> = (0..n_items)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, 0xBF00 + i as u64))
        .collect();

    let patterns: Vec<(String, EntangledHVec)> = items
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("{}", i), v.clone()))
        .collect();
    let hop_config = HopfieldConfig {
        beta: 100.0,
        alpha: 2.0,
        max_iter: 3,
    };

    let mut results = Vec::new();
    for &corruption in &corruptions {
        let mut jaccard_correct = 0;
        let mut hopfield_correct = 0;

        for i in 0..n_probes {
            let target_idx = hash_u64(0xCC00, i as u64) as usize % n_items;
            let noisy = corrupt_vector(
                &items[target_idx],
                dim,
                density_denom,
                corruption,
                0xDD00 + i as u64,
            );

            let jaccard_best = items
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.similarity(&noisy)
                        .partial_cmp(&b.similarity(&noisy))
                        .unwrap()
                })
                .unwrap()
                .0;
            if jaccard_best == target_idx {
                jaccard_correct += 1;
            }

            let hop_results = hopfield_query(&noisy, &patterns, &hop_config, 1);
            if !hop_results.is_empty()
                && hop_results[0].id.parse::<usize>().ok() == Some(target_idx)
            {
                hopfield_correct += 1;
            }
        }

        results.push(serde_json::json!({
            "corruption": corruption,
            "jaccard_accuracy": jaccard_correct as f64 / n_probes as f64,
            "hopfield_accuracy": hopfield_correct as f64 / n_probes as f64,
        }));
    }

    serde_json::json!({
        "n_items": n_items,
        "n_probes": n_probes,
        "results": results,
    })
}

fn measure_throughput(dim: usize, density_denom: usize) -> serde_json::Value {
    let n_ops = 1000;

    let t0 = Instant::now();
    let vecs: Vec<EntangledHVec> = (0..n_ops)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, i as u64))
        .collect();
    let encode_ns = t0.elapsed().as_nanos() as f64 / n_ops as f64;

    let t0 = Instant::now();
    for i in 0..n_ops - 1 {
        let _ = vecs[i].bind(&vecs[i + 1]);
    }
    let bind_ns = t0.elapsed().as_nanos() as f64 / (n_ops - 1) as f64;

    let t0 = Instant::now();
    let bundle_batch = 10;
    let bundle_iters = n_ops / bundle_batch;
    for i in 0..bundle_iters {
        let batch: Vec<&EntangledHVec> = (0..bundle_batch)
            .map(|j| &vecs[i * bundle_batch + j])
            .collect();
        let _ = EntangledHVec::bundle_bloom(&batch);
    }
    let bundle_ns = t0.elapsed().as_nanos() as f64 / bundle_iters as f64;

    let bundle_100 = EntangledHVec::bundle_bloom(&vecs[..100].iter().collect::<Vec<_>>());
    let t0 = Instant::now();
    for i in 0..n_ops - 1 {
        let _ = vecs[i].similarity(&vecs[i + 1]);
    }
    let similarity_ns = t0.elapsed().as_nanos() as f64 / (n_ops - 1) as f64;

    let t0 = Instant::now();
    for v in vecs.iter().take(n_ops) {
        let _ = v.containment_similarity(&bundle_100);
    }
    let containment_ns = t0.elapsed().as_nanos() as f64 / n_ops as f64;

    let t0 = Instant::now();
    for v in vecs.iter().take(n_ops) {
        let _ = v.corrected_containment(&bundle_100);
    }
    let corrected_ns = t0.elapsed().as_nanos() as f64 / n_ops as f64;

    let t0 = Instant::now();
    for v in vecs.iter().take(n_ops) {
        let _ = v.permute(1);
    }
    let permute_ns = t0.elapsed().as_nanos() as f64 / n_ops as f64;

    serde_json::json!({
        "n_ops": n_ops,
        "encode_ns": encode_ns,
        "bind_ns": bind_ns,
        "bundle_bloom_10_ns": bundle_ns,
        "similarity_ns": similarity_ns,
        "containment_ns": containment_ns,
        "corrected_containment_ns": corrected_ns,
        "permute_ns": permute_ns,
        "encode_ops_per_sec": 1e9 / encode_ns,
        "bind_ops_per_sec": 1e9 / bind_ns,
        "bundle_bloom_10_ops_per_sec": 1e9 / bundle_ns,
        "similarity_ops_per_sec": 1e9 / similarity_ns,
        "containment_ops_per_sec": 1e9 / containment_ns,
        "permute_ops_per_sec": 1e9 / permute_ns,
    })
}

fn corrupt_vector(
    v: &EntangledHVec,
    dim: usize,
    density_denom: usize,
    pct: f64,
    seed: u64,
) -> EntangledHVec {
    let indices = v.indices();
    let n_flip = (indices.len() as f64 * pct) as usize;
    let mut new_indices: Vec<u32> = indices.to_vec();
    let n_remove = n_flip.min(new_indices.len());
    for i in 0..n_remove {
        let idx = hash_u64(seed, i as u64) as usize % new_indices.len();
        new_indices.swap_remove(idx);
    }
    for i in 0..n_flip {
        let idx = (hash_u64(seed.wrapping_add(0x1000), i as u64) % dim as u64) as u32;
        if new_indices.binary_search(&idx).is_err() {
            new_indices.push(idx);
        }
    }
    new_indices.sort_unstable();
    new_indices.dedup();
    let target = dim / density_denom;
    new_indices.truncate(target + n_flip);
    EntangledHVec::from_indices(new_indices, dim)
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_dev(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let m = mean(vals);
    let var = vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (vals.len() - 1) as f64;
    var.sqrt()
}

fn d_prime_val(mean_a: f64, std_a: f64, mean_b: f64, std_b: f64) -> f64 {
    let pooled = ((std_a * std_a + std_b * std_b) / 2.0).sqrt();
    if pooled < 1e-15 {
        return if (mean_a - mean_b).abs() < 1e-15 {
            0.0
        } else {
            f64::INFINITY
        };
    }
    (mean_a - mean_b) / pooled
}
