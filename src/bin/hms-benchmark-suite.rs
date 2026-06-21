// HMS VSA/HDC benchmark suite with pre-registered experiments.

use holographic_memory::core::bloom_memory::HolographicBloomMemory;
use holographic_memory::core::encoding::encode_text_internal;
use holographic_memory::core::entangled::{hash_u64, EntangledHVec, DEFAULT_RHO_DENOM};
use holographic_memory::core::hopfield::{hopfield_query, HopfieldConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|a| a == "--json");
    let quick = args.iter().any(|a| a == "--quick");

    let primary_dim = if quick { 4096 } else { 16384 };

    if !json_output {
        eprintln!("HMS Benchmark Suite");
        eprintln!("===================");
    }

    eprint_stage("1/8 Capacity load curves", json_output);
    let capacity = run_capacity_sweep(primary_dim, quick);

    eprint_stage("2/8 Binding fidelity", json_output);
    let fidelity = run_binding_fidelity(primary_dim);

    eprint_stage("3/8 Composition depth", json_output);
    let composition = run_composition_depth(primary_dim, quick);

    eprint_stage("4/8 Hopfield associative memory", json_output);
    let hopfield = run_hopfield_benchmarks(primary_dim, quick);

    eprint_stage("5/8 Encoding quality", json_output);
    let encoding = run_encoding_quality(primary_dim);

    eprint_stage("6/8 [PRE-REG] Sparse-binary vs dense Hopfield", json_output);
    let sparse_vs_dense = run_sparse_vs_dense_hopfield(primary_dim, quick);

    eprint_stage("7/8 Holographic Bloom Memory", json_output);
    let hbm = run_holographic_bloom_memory(primary_dim, quick);

    eprint_stage("8/8 Compositional text encoding", json_output);
    let comp_text = run_compositional_text(primary_dim);

    let report = serde_json::json!({
        "meta": {
            "suite_version": "4.0.0",
            "primary_dim": primary_dim,
            "quick_mode": quick,
            "active_density": format!("1/{}", DEFAULT_RHO_DENOM),
        },
        "capacity": capacity,
        "binding_fidelity": fidelity,
        "composition_depth": composition,
        "hopfield": hopfield,
        "encoding": encoding,
        "preregistered": {
            "sparse_vs_dense_hopfield": sparse_vs_dense,
        },
        "holographic_bloom_memory": hbm,
        "compositional_text": comp_text,
    });

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        print_human_report(&report);
    }
}

fn eprint_stage(label: &str, json: bool) {
    if !json {
        eprintln!("[{}]", label);
    }
}

// ===========================================================================
// 1. CAPACITY LOAD CURVES
// ===========================================================================

fn run_capacity_sweep(dim: usize, quick: bool) -> serde_json::Value {
    let trials = if quick { 20 } else { 50 };
    let k_values: Vec<usize> = if quick {
        vec![1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
    } else {
        let mut ks = Vec::new();
        let mut k = 1;
        while k <= 500.min(dim / 16) {
            ks.push(k);
            k = if k < 10 {
                k + 1
            } else if k < 50 {
                k + 5
            } else if k < 100 {
                k + 10
            } else {
                k + 25
            };
        }
        ks
    };
    let active = dim / DEFAULT_RHO_DENOM;
    let mut majority_results = Vec::new();
    let mut bloom_results = Vec::new();

    for &k in &k_values {
        let mut maj_signal = Vec::new();
        let mut maj_noise = Vec::new();
        let mut maj_density = 0.0;
        let mut maj_recovered = 0;

        let mut blm_signal = Vec::new();
        let mut blm_noise = Vec::new();
        let mut blm_density = 0.0;
        let mut blm_recovered = 0;

        for trial in 0..trials {
            let seed_base = trial as u64 * 10000 + dim as u64 * 100000;
            let components: Vec<EntangledHVec> = (0..k)
                .map(|i| EntangledHVec::new_deterministic(dim, seed_base + i as u64))
                .collect();

            // Majority-vote bundle
            let maj_bundle = EntangledHVec::bundle(&components);
            maj_density += maj_bundle.indices().len() as f64 / dim as f64;
            let test_idx = trial % k;
            let sig = maj_bundle.similarity(&components[test_idx]);
            maj_signal.push(sig);
            let noise_vec = EntangledHVec::new_deterministic(dim, seed_base + k as u64 + 999);
            let noi = maj_bundle.similarity(&noise_vec);
            maj_noise.push(noi);
            let threshold = active as f64 / dim as f64 * 2.0;
            if sig > threshold.max(noi * 2.0) {
                maj_recovered += 1;
            }

            // Bloom-filter bundle (set union + corrected containment)
            let blm_bundle = EntangledHVec::bundle_bloom(&components);
            blm_density += blm_bundle.indices().len() as f64 / dim as f64;
            let sig_b = components[test_idx].corrected_containment(&blm_bundle);
            blm_signal.push(sig_b);
            let noi_b = noise_vec.corrected_containment(&blm_bundle);
            blm_noise.push(noi_b);
            if sig_b > 0.5 && sig_b > noi_b + 0.1 {
                blm_recovered += 1;
            }
        }

        maj_signal.sort_by(|a, b| a.partial_cmp(b).unwrap());
        maj_noise.sort_by(|a, b| a.partial_cmp(b).unwrap());
        blm_signal.sort_by(|a, b| a.partial_cmp(b).unwrap());
        blm_noise.sort_by(|a, b| a.partial_cmp(b).unwrap());

        majority_results.push(serde_json::json!({
            "dim": dim, "k": k,
            "recovery_rate": maj_recovered as f64 / trials as f64,
            "signal_mean": mean(&maj_signal), "signal_std": std_dev(&maj_signal),
            "noise_mean": mean(&maj_noise), "noise_std": std_dev(&maj_noise),
            "d_prime": d_prime_val(mean(&maj_signal), std_dev(&maj_signal), mean(&maj_noise), std_dev(&maj_noise)),
            "density_mean": maj_density / trials as f64,
        }));

        bloom_results.push(serde_json::json!({
            "dim": dim, "k": k,
            "recovery_rate": blm_recovered as f64 / trials as f64,
            "signal_mean": mean(&blm_signal), "signal_std": std_dev(&blm_signal),
            "noise_mean": mean(&blm_noise), "noise_std": std_dev(&blm_noise),
            "d_prime": d_prime_val(mean(&blm_signal), std_dev(&blm_signal), mean(&blm_noise), std_dev(&blm_noise)),
            "density_mean": blm_density / trials as f64,
        }));
    }
    serde_json::json!({
        "majority_vote": majority_results,
        "bloom_filter": bloom_results,
        "note": "Majority vote uses Jaccard; Bloom filter uses containment similarity"
    })
}

// ===========================================================================
// 2. BINDING FIDELITY
// ===========================================================================

fn run_binding_fidelity(dim: usize) -> serde_json::Value {
    let n_pairs = 500;
    let mut signal = Vec::new();
    let mut noise = Vec::new();

    for i in 0..n_pairs {
        let a = EntangledHVec::new_deterministic(dim, i as u64 * 3);
        let b = EntangledHVec::new_deterministic(dim, i as u64 * 3 + 1);
        let unrelated = EntangledHVec::new_deterministic(dim, i as u64 * 3 + 2);
        let recovered = a.bind(&b).bind(&a);
        signal.push(recovered.similarity(&b));
        noise.push(a.bind(&b).similarity(&unrelated));
    }
    signal.sort_by(|a, b| a.partial_cmp(b).unwrap());
    noise.sort_by(|a, b| a.partial_cmp(b).unwrap());

    serde_json::json!({
        "signal_mean": mean(&signal), "signal_std": std_dev(&signal),
        "noise_mean": mean(&noise), "noise_std": std_dev(&noise),
        "d_prime": d_prime_val(mean(&signal), std_dev(&signal), mean(&noise), std_dev(&noise)),
        "n_pairs": n_pairs,
    })
}

// ===========================================================================
// 3. COMPOSITION DEPTH
// ===========================================================================

fn run_composition_depth(dim: usize, quick: bool) -> serde_json::Value {
    let max_depth = if quick { 10 } else { 20 };
    let trials = if quick { 30 } else { 100 };
    let mut results = Vec::new();

    for depth in 1..=max_depth {
        let mut fidelities = Vec::new();
        for trial in 0..trials {
            let seed_base = trial as u64 * 1000 + depth as u64 * 100000;
            let target = EntangledHVec::new_deterministic(dim, seed_base);
            let mut composed = target.clone();
            let mut keys = Vec::new();
            for d in 0..depth {
                let key = EntangledHVec::new_deterministic(dim, seed_base + d as u64 + 1);
                composed = composed.bind(&key);
                keys.push(key);
            }
            for key in keys.iter().rev() {
                composed = composed.bind(key);
            }
            fidelities.push(composed.similarity(&target));
        }
        results.push(serde_json::json!({
            "depth": depth,
            "fidelity_mean": mean(&fidelities),
            "fidelity_std": std_dev(&fidelities),
        }));
    }
    serde_json::json!(results)
}

// ===========================================================================
// 4. HOPFIELD BENCHMARKS
// ===========================================================================

fn run_hopfield_benchmarks(dim: usize, quick: bool) -> serde_json::Value {
    let pattern_counts: Vec<usize> = if quick {
        vec![5, 10, 20, 50, 100, 200]
    } else {
        vec![5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    };
    let queries_per = if quick { 10 } else { 30 };
    let mut capacity = Vec::new();

    for &n in &pattern_counts {
        let patterns: Vec<(String, EntangledHVec)> = (0..n)
            .map(|i| {
                (
                    format!("p{}", i),
                    EntangledHVec::new_deterministic(dim, i as u64 * 7 + 1),
                )
            })
            .collect();

        for &(name, alpha) in &[("sparsemax", 2.0), ("1.5-entmax", 1.5)] {
            let config = HopfieldConfig {
                beta: 100.0,
                alpha,
                max_iter: 1,
            };
            let nq = queries_per.min(n);
            let mut exact = 0;
            for q in 0..nq {
                if hopfield_query(&patterns[q].1, &patterns, &config, 10)
                    .first()
                    .is_some_and(|r| r.id == patterns[q].0)
                {
                    exact += 1;
                }
            }
            capacity.push(serde_json::json!({
                "n_patterns": n, "entmax_type": name,
                "exact_retrieval_rate": exact as f64 / nq as f64,
            }));
        }
    }

    // Noise recovery
    let corruption_pcts = if quick {
        vec![0.0, 0.1, 0.2, 0.3, 0.5]
    } else {
        vec![0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    };
    let n_pat = 100;
    let trials = if quick { 20 } else { 50 };
    let patterns: Vec<(String, EntangledHVec)> = (0..n_pat)
        .map(|i| {
            (
                format!("p{}", i),
                EntangledHVec::new_deterministic(dim, i as u64 * 13 + 3),
            )
        })
        .collect();
    let mut noise_recovery = Vec::new();

    for &pct in &corruption_pcts {
        let config = HopfieldConfig {
            beta: 100.0,
            alpha: 2.0,
            max_iter: 1,
        };
        let mut rec = 0;
        for trial in 0..trials {
            let idx = trial % n_pat;
            let corrupted =
                corrupt_vector(&patterns[idx].1, dim, pct, trial as u64 + idx as u64 * 1000);
            if hopfield_query(&corrupted, &patterns, &config, 5)
                .first()
                .is_some_and(|r| r.id == patterns[idx].0)
            {
                rec += 1;
            }
        }
        noise_recovery.push(serde_json::json!({
            "corruption_pct": pct,
            "recovery_rate": rec as f64 / trials as f64,
        }));
    }

    serde_json::json!({ "capacity": capacity, "noise_recovery": noise_recovery })
}

// ===========================================================================
// 5. ENCODING QUALITY
// ===========================================================================

fn run_encoding_quality(dim: usize) -> serde_json::Value {
    let base_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning algorithms process large datasets efficiently",
        "quantum computing leverages superposition and entanglement",
        "distributed systems require careful coordination of state",
    ];
    let levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7];
    let mut perturbation = Vec::new();

    for &pct in &levels {
        let mut sims = Vec::new();
        for (idx, &text) in base_texts.iter().enumerate() {
            let original = encode_text_internal(text, dim);
            for v in 0..10u64 {
                let corrupted =
                    encode_text_internal(&corrupt_text(text, pct, idx as u64 * 100 + v), dim);
                sims.push(original.similarity(&corrupted));
            }
        }
        perturbation.push(serde_json::json!({
            "corruption_pct": pct,
            "similarity_mean": mean(&sims),
            "similarity_std": std_dev(&sims),
        }));
    }
    serde_json::json!({ "perturbation": perturbation })
}

// ===========================================================================
// 6. PRE-REGISTERED: Sparse-binary vs dense-Gaussian Hopfield
// ===========================================================================
fn run_sparse_vs_dense_hopfield(dim: usize, quick: bool) -> serde_json::Value {
    let n_patterns = if quick { 50 } else { 100 };
    let trials = if quick { 20 } else { 50 };
    let config = HopfieldConfig {
        beta: 100.0,
        alpha: 2.0,
        max_iter: 1,
    };

    let sparse_patterns: Vec<(String, EntangledHVec)> = (0..n_patterns)
        .map(|i| {
            (
                format!("s{}", i),
                EntangledHVec::new_deterministic(dim, i as u64 * 17 + 5),
            )
        })
        .collect();
    let dense_patterns: Vec<(String, EntangledHVec)> = (0..n_patterns)
        .map(|i| {
            let dense = generate_pseudo_gaussian(dim, i as u64 * 23 + 7);
            (format!("d{}", i), EntangledHVec::from_dense(&dense, dim))
        })
        .collect();

    let sparse_active = dim / DEFAULT_RHO_DENOM;
    let sparse_info_bits = log2_binomial(dim, sparse_active);
    let dense_active_mean = dense_patterns
        .iter()
        .map(|(_, v)| v.indices().len() as f64)
        .sum::<f64>()
        / n_patterns as f64;

    let sparse_ct = pairwise_crosstalk(&sparse_patterns, 200);
    let dense_ct = pairwise_crosstalk(&dense_patterns, 200);

    let path_info = serde_json::json!({
        "similarity_kernel": "Jaccard",
        "entmax_alpha": config.alpha,
        "beta": config.beta,
        "note": "Both pattern types use identical hopfield_query code path"
    });

    // Basin sweep
    let corruption_levels = if quick {
        vec![0.0, 0.1, 0.2, 0.3, 0.5]
    } else {
        vec![0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    };
    let mut basin = Vec::new();
    for &pct in &corruption_levels {
        let mut s_rec = 0;
        let mut d_rec = 0;
        for trial in 0..trials {
            let idx = trial % n_patterns;
            let sc = corrupt_vector(&sparse_patterns[idx].1, dim, pct, trial as u64 * 31);
            if hopfield_query(&sc, &sparse_patterns, &config, 5)
                .first()
                .is_some_and(|r| r.id == sparse_patterns[idx].0)
            {
                s_rec += 1;
            }
            let dc = corrupt_vector(&dense_patterns[idx].1, dim, pct, trial as u64 * 37);
            if hopfield_query(&dc, &dense_patterns, &config, 5)
                .first()
                .is_some_and(|r| r.id == dense_patterns[idx].0)
            {
                d_rec += 1;
            }
        }
        basin.push(serde_json::json!({
            "corruption_pct": pct,
            "sparse_recovery": s_rec as f64 / trials as f64,
            "dense_recovery": d_rec as f64 / trials as f64,
        }));
    }

    // Capacity sweep
    let cap_counts: Vec<usize> = if quick {
        vec![10, 25, 50, 100, 200]
    } else {
        vec![10, 25, 50, 100, 200, 500, 1000]
    };
    let cap_q = if quick { 10 } else { 20 };
    let mut cap_results = Vec::new();
    for &n in &cap_counts {
        let sp: Vec<(String, EntangledHVec)> = (0..n)
            .map(|i| {
                (
                    format!("s{}", i),
                    EntangledHVec::new_deterministic(dim, i as u64 * 41 + 11),
                )
            })
            .collect();
        let dp: Vec<(String, EntangledHVec)> = (0..n)
            .map(|i| {
                let d = generate_pseudo_gaussian(dim, i as u64 * 43 + 13);
                (format!("d{}", i), EntangledHVec::from_dense(&d, dim))
            })
            .collect();
        let nq = cap_q.min(n);
        let mut se = 0;
        let mut de = 0;
        for q in 0..nq {
            if hopfield_query(&sp[q].1, &sp, &config, 10)
                .first()
                .is_some_and(|r| r.id == sp[q].0)
            {
                se += 1;
            }
            if hopfield_query(&dp[q].1, &dp, &config, 10)
                .first()
                .is_some_and(|r| r.id == dp[q].0)
            {
                de += 1;
            }
        }
        cap_results.push(serde_json::json!({
            "n_patterns": n,
            "sparse_exact": se as f64 / nq as f64,
            "dense_exact": de as f64 / nq as f64,
        }));
    }

    let s_wins_basin = basin
        .iter()
        .filter(|r| {
            r["sparse_recovery"].as_f64().unwrap_or(0.0)
                > r["dense_recovery"].as_f64().unwrap_or(0.0)
        })
        .count();
    let s_wins_cap = cap_results
        .iter()
        .filter(|r| {
            r["sparse_exact"].as_f64().unwrap_or(0.0) > r["dense_exact"].as_f64().unwrap_or(0.0)
        })
        .count();
    let dense_wins_both = s_wins_basin == 0 && s_wins_cap == 0;

    serde_json::json!({
        "description": "Sparse binary vs dense-projected in Hopfield retrieval",
        "kill_condition": "Dense matches or beats sparse on BOTH basin width AND capacity cliff",
        "information_content": {
            "sparse_active_bits": sparse_active,
            "sparse_info_bits_approx": sparse_info_bits,
            "dense_active_mean": dense_active_mean,
            "note": "Equal-D comparison; information content differs. Confound stated.",
        },
        "hopfield_path": path_info,
        "crosstalk": {
            "sparse_mean": sparse_ct.0, "sparse_std": sparse_ct.1,
            "dense_mean": dense_ct.0, "dense_std": dense_ct.1,
        },
        "basin_of_attraction": basin,
        "capacity_sweep": cap_results,
        "sparse_wins_basin_points": s_wins_basin,
        "sparse_wins_capacity_points": s_wins_cap,
        "kill_fires": dense_wins_both,
        "strong_claim_holds": !dense_wins_both,
    })
}

// ===========================================================================
// 8. CLIFFORDVEC ABLATION
// ===========================================================================

// ===========================================================================
// 7. HOLOGRAPHIC BLOOM MEMORY
// ===========================================================================
// Bloom-filter bundling with density-corrected containment similarity.
// Members score 1.0 (exact containment). Non-members score ~0.0 after
// subtracting the density floor. Capacity: hundreds of items at D=16384.

fn run_holographic_bloom_memory(dim: usize, quick: bool) -> serde_json::Value {
    let item_counts: Vec<usize> = if quick {
        vec![10, 25, 50, 100, 200]
    } else {
        vec![10, 25, 50, 100, 200, 300, 500]
    };
    let n_probe = if quick { 20 } else { 50 };

    let mut results = Vec::new();

    for &n_items in &item_counts {
        let items: Vec<EntangledHVec> = (0..n_items)
            .map(|i| EntangledHVec::new_deterministic(dim, i as u64 * 137 + 7))
            .collect();

        let mut mem = HolographicBloomMemory::new(dim);
        mem.insert_batch(&items);
        let density = mem.density();

        let mut member_scores = Vec::new();
        let probe_members = n_probe.min(n_items);
        for item in items.iter().take(probe_members) {
            member_scores.push(mem.contains(item));
        }

        let mut non_member_scores = Vec::new();
        for i in 0..n_probe {
            let nm = EntangledHVec::new_deterministic(dim, 555000 + i as u64 * 31);
            non_member_scores.push(mem.contains(&nm));
        }

        let member_min = member_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let non_member_max = non_member_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let gap = member_min - non_member_max;

        let threshold = 0.5;
        let recall =
            member_scores.iter().filter(|&&s| s >= threshold).count() as f64 / probe_members as f64;
        let false_pos = non_member_scores
            .iter()
            .filter(|&&s| s >= threshold)
            .count() as f64
            / n_probe as f64;

        results.push(serde_json::json!({
            "n_items": n_items,
            "density": density,
            "member_mean": mean(&member_scores),
            "member_min": member_min,
            "non_member_mean": mean(&non_member_scores),
            "non_member_max": non_member_max,
            "non_member_std": std_dev(&non_member_scores),
            "gap": gap,
            "d_prime": d_prime_val(
                mean(&member_scores), std_dev(&member_scores),
                mean(&non_member_scores), std_dev(&non_member_scores)
            ),
            "recall_at_0.5": recall,
            "false_positive_rate_at_0.5": false_pos,
        }));
    }

    // Structured retrieval via role-as-permutation with Bloom bundling.
    // Each role is a unique permutation offset. Composition = union of
    // permuted entities. Query = containment of role-permuted candidate.
    // Breaks swap symmetry: permute(ent, 1) ≠ permute(ent, 2).
    let n_facts = if quick { 20 } else { 50 };
    let vocab_size = if quick { 15 } else { 30 };
    let entities: Vec<EntangledHVec> = (0..vocab_size)
        .map(|i| EntangledHVec::new_deterministic(dim, 0xF000 + i as u64))
        .collect();
    let agent_offset = 1usize;
    let patient_offset = 2usize;

    let mut facts = Vec::new();
    let mut compositions = Vec::new();
    for t in 0..n_facts {
        let a_idx = hash_u64(0x8181, t as u64) as usize % vocab_size;
        let mut b_idx = hash_u64(0x8182, t as u64) as usize % vocab_size;
        if b_idx == a_idx {
            b_idx = (a_idx + 1) % vocab_size;
        }
        facts.push((a_idx, b_idx));

        let comp = EntangledHVec::bundle_bloom(&[
            entities[a_idx].permute(agent_offset),
            entities[b_idx].permute(patient_offset),
        ]);
        compositions.push(comp);
    }

    let mut fact_mem = HolographicBloomMemory::new(dim);
    fact_mem.insert_batch(&compositions);

    let mut struct_correct = 0;
    for (t, &(a_idx, _b_idx)) in facts.iter().enumerate() {
        let comp = &compositions[t];
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (e, entity) in entities.iter().enumerate() {
            let probe = entity.permute(agent_offset);
            let score = probe.containment_similarity(comp);
            if score > best_score {
                best_score = score;
                best_idx = e;
            }
        }
        if best_idx == a_idx {
            struct_correct += 1;
        }
    }

    let structured = serde_json::json!({
        "n_facts": n_facts,
        "vocab_size": vocab_size,
        "agent_recovery_accuracy": struct_correct as f64 / n_facts as f64,
        "method": "role_as_permutation + bloom_containment",
    });

    // Scaling sweep: vary n_roles, vocab_size, n_facts
    let scale_configs: Vec<(usize, usize, usize)> = if quick {
        vec![(2, 15, 20), (3, 20, 30), (5, 30, 40)]
    } else {
        vec![
            (2, 30, 50),
            (3, 50, 80),
            (5, 50, 100),
            (8, 80, 120),
            (10, 100, 150),
        ]
    };

    let mut scaling_results = Vec::new();
    for &(n_roles, vs, nf) in &scale_configs {
        let ents: Vec<EntangledHVec> = (0..vs)
            .map(|i| EntangledHVec::new_deterministic(dim, 0xE000 + i as u64))
            .collect();

        let mut facts: Vec<Vec<usize>> = Vec::new();
        let mut comps = Vec::new();
        for t in 0..nf {
            let mut fillers = Vec::new();
            for r in 0..n_roles {
                let idx = hash_u64(0x7000 + r as u64, t as u64) as usize % vs;
                fillers.push(idx);
            }
            let bindings: Vec<EntangledHVec> = fillers
                .iter()
                .enumerate()
                .map(|(r, &idx)| ents[idx].permute(r + 1))
                .collect();
            comps.push(EntangledHVec::bundle_bloom(&bindings));
            facts.push(fillers);
        }

        let mut correct = 0;
        let total = nf * n_roles;
        for (t, fillers) in facts.iter().enumerate() {
            let comp = &comps[t];
            for (r, &true_idx) in fillers.iter().enumerate() {
                let mut best = 0;
                let mut best_s = f64::NEG_INFINITY;
                for (e, ent) in ents.iter().enumerate() {
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

        scaling_results.push(serde_json::json!({
            "n_roles": n_roles,
            "vocab_size": vs,
            "n_facts": nf,
            "accuracy": correct as f64 / total as f64,
            "total_queries": total,
        }));
    }

    // Bloom + Hopfield hybrid: corrupted query → Bloom narrows candidates → Hopfield selects
    let hybrid_n = if quick { 100 } else { 500 };
    let hybrid_corruptions = [0.3, 0.5, 0.7];
    let hybrid_items: Vec<EntangledHVec> = (0..hybrid_n)
        .map(|i| EntangledHVec::new_deterministic(dim, 0xBF00 + i as u64))
        .collect();

    let patterns: Vec<(String, EntangledHVec)> = hybrid_items
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("{}", i), v.clone()))
        .collect();
    let hop_config = HopfieldConfig {
        beta: 100.0,
        alpha: 2.0,
        max_iter: 3,
    };
    let n_hybrid_probes = if quick { 30 } else { 100 };
    let top_k = 20;

    let mut hybrid_results = Vec::new();
    for &corruption in &hybrid_corruptions {
        let mut jaccard_correct = 0;
        let mut hopfield_full_correct = 0;
        let mut hopfield_bloom_correct = 0;

        for i in 0..n_hybrid_probes {
            let target_idx = hash_u64(0xCC00, i as u64) as usize % hybrid_n;
            let noisy = corrupt_vector(
                &hybrid_items[target_idx],
                dim,
                corruption,
                0xDD00 + i as u64,
            );

            let jaccard_best = hybrid_items
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

            let hop_full = hopfield_query(&noisy, &patterns, &hop_config, 1);
            if !hop_full.is_empty() && hop_full[0].id.parse::<usize>().ok() == Some(target_idx) {
                hopfield_full_correct += 1;
            }

            let mut scored: Vec<(usize, f64)> = hybrid_items
                .iter()
                .enumerate()
                .map(|(j, item)| (j, noisy.similarity(item)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let candidates: Vec<(String, EntangledHVec)> = scored
                .iter()
                .take(top_k)
                .map(|&(j, _)| (format!("{}", j), hybrid_items[j].clone()))
                .collect();
            let hop_narrow = hopfield_query(&noisy, &candidates, &hop_config, 1);
            if !hop_narrow.is_empty() && hop_narrow[0].id.parse::<usize>().ok() == Some(target_idx)
            {
                hopfield_bloom_correct += 1;
            }
        }

        hybrid_results.push(serde_json::json!({
            "corruption": corruption,
            "jaccard_nn_accuracy": jaccard_correct as f64 / n_hybrid_probes as f64,
            "hopfield_full_accuracy": hopfield_full_correct as f64 / n_hybrid_probes as f64,
            "hopfield_top_k_accuracy": hopfield_bloom_correct as f64 / n_hybrid_probes as f64,
            "top_k": top_k,
        }));
    }

    let hybrid = serde_json::json!({
        "n_items": hybrid_n,
        "n_probes": n_hybrid_probes,
        "results": hybrid_results,
        "method": "jaccard_prefilter + hopfield_cleanup",
    });

    serde_json::json!({
        "description": "Holographic Bloom memory with role-as-permutation structured retrieval",
        "capacity_sweep": results,
        "structured_retrieval": structured,
        "scaling_sweep": scaling_results,
        "bloom_hopfield_hybrid": hybrid,
    })
}

// ===========================================================================
// 10. COMPOSITIONAL TEXT ENCODING
// ===========================================================================
// Encode natural-language sentences as structured hypervectors using
// role-as-permutation binding + Bloom bundling. Demonstrate retrieval of
// role fillers (subject, verb, object) from sentence representations.

fn run_compositional_text(dim: usize) -> serde_json::Value {
    let subject_offset = 1usize;
    let verb_offset = 2usize;
    let object_offset = 3usize;

    let sentences: Vec<(&str, &str, &str)> = vec![
        ("dog", "chases", "cat"),
        ("cat", "watches", "bird"),
        ("bird", "eats", "worm"),
        ("fish", "swims", "river"),
        ("wolf", "hunts", "deer"),
        ("hawk", "catches", "mouse"),
        ("bear", "climbs", "tree"),
        ("fox", "jumps", "fence"),
        ("owl", "hunts", "rabbit"),
        ("snake", "eats", "frog"),
        ("eagle", "flies", "mountain"),
        ("shark", "swims", "ocean"),
    ];

    let mut vocab: std::collections::HashMap<&str, EntangledHVec> =
        std::collections::HashMap::new();
    for &(s, v, o) in &sentences {
        for word in [s, v, o] {
            vocab
                .entry(word)
                .or_insert_with(|| encode_text_internal(word, dim));
        }
    }

    let compositions: Vec<EntangledHVec> = sentences
        .iter()
        .map(|&(s, v, o)| {
            EntangledHVec::bundle_bloom(&[
                vocab[s].permute(subject_offset),
                vocab[v].permute(verb_offset),
                vocab[o].permute(object_offset),
            ])
        })
        .collect();

    let all_words: Vec<&str> = vocab.keys().copied().collect();

    let mut subject_correct = 0;
    let mut verb_correct = 0;
    let mut object_correct = 0;
    for (i, &(true_s, true_v, true_o)) in sentences.iter().enumerate() {
        let comp = &compositions[i];

        let best_s = all_words
            .iter()
            .max_by(|a, b| {
                vocab[**a]
                    .permute(subject_offset)
                    .containment_similarity(comp)
                    .partial_cmp(
                        &vocab[**b]
                            .permute(subject_offset)
                            .containment_similarity(comp),
                    )
                    .unwrap()
            })
            .unwrap();
        if *best_s == true_s {
            subject_correct += 1;
        }

        let best_v = all_words
            .iter()
            .max_by(|a, b| {
                vocab[**a]
                    .permute(verb_offset)
                    .containment_similarity(comp)
                    .partial_cmp(&vocab[**b].permute(verb_offset).containment_similarity(comp))
                    .unwrap()
            })
            .unwrap();
        if *best_v == true_v {
            verb_correct += 1;
        }

        let best_o = all_words
            .iter()
            .max_by(|a, b| {
                vocab[**a]
                    .permute(object_offset)
                    .containment_similarity(comp)
                    .partial_cmp(
                        &vocab[**b]
                            .permute(object_offset)
                            .containment_similarity(comp),
                    )
                    .unwrap()
            })
            .unwrap();
        if *best_o == true_o {
            object_correct += 1;
        }
    }

    let n = sentences.len() as f64;

    let mut mem = HolographicBloomMemory::new(dim);
    mem.insert_batch(&compositions);

    let query_word = "chases";
    let query_binding = vocab[query_word].permute(verb_offset);
    let mut query_results = Vec::new();
    for (i, comp) in compositions.iter().enumerate() {
        let score = query_binding.containment_similarity(comp);
        if (score - 1.0).abs() < 1e-10 {
            let &(s, v, o) = &sentences[i];
            query_results.push(serde_json::json!({
                "sentence": format!("{} {} {}", s, v, o),
                "score": score,
            }));
        }
    }

    serde_json::json!({
        "description": "Natural language sentences encoded as role-permuted Bloom bundles",
        "n_sentences": sentences.len(),
        "vocab_size": vocab.len(),
        "subject_accuracy": subject_correct as f64 / n,
        "verb_accuracy": verb_correct as f64 / n,
        "object_accuracy": object_correct as f64 / n,
        "overall_accuracy": (subject_correct + verb_correct + object_correct) as f64 / (3.0 * n),
        "verb_query": {
            "query": query_word,
            "role": "verb",
            "matches": query_results,
        },
    })
}

// ===========================================================================
// Utilities
// ===========================================================================

fn corrupt_vector(v: &EntangledHVec, dim: usize, pct: f64, seed: u64) -> EntangledHVec {
    let indices = v.indices();
    let n_flip = (indices.len() as f64 * pct) as usize;
    let mut new_indices: Vec<u32> = indices.to_vec();
    let n_remove = n_flip.min(new_indices.len());
    for i in 0..n_remove {
        let idx = hash_u64(seed, i as u64) as usize % new_indices.len();
        new_indices.swap_remove(idx);
    }
    for i in 0..n_remove {
        new_indices.push((hash_u64(seed + 0xBEEF, i as u64) % dim as u64) as u32);
    }
    new_indices.sort_unstable();
    new_indices.dedup();
    EntangledHVec::from_indices(new_indices, dim)
}

fn corrupt_text(text: &str, pct: f64, seed: u64) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = (chars.len() as f64 * pct) as usize;
    let mut result = chars;
    for i in 0..n {
        let pos = hash_u64(seed, i as u64) as usize % result.len();
        let ch = (hash_u64(seed + 0xDEAD, i as u64) % 26) as u8 + b'a';
        result[pos] = ch as char;
    }
    result.into_iter().collect()
}

fn generate_pseudo_gaussian(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for i in (0..dim).step_by(2) {
        let u1 = (hash_u64(seed, i as u64) as f64 / u64::MAX as f64).max(1e-10);
        let u2 = hash_u64(seed, i as u64 + 1) as f64 / u64::MAX as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        v[i] = (r * theta.cos()) as f32;
        if i + 1 < dim {
            v[i + 1] = (r * theta.sin()) as f32;
        }
    }
    v
}

fn pairwise_crosstalk(patterns: &[(String, EntangledHVec)], max_pairs: usize) -> (f64, f64) {
    let mut sims = Vec::new();
    let n = patterns.len();
    let mut count = 0;
    'outer: for i in 0..n {
        for j in (i + 1)..n {
            sims.push(patterns[i].1.similarity(&patterns[j].1));
            count += 1;
            if count >= max_pairs {
                break 'outer;
            }
        }
    }
    (mean(&sims), std_dev(&sims))
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
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn d_prime_val(sm: f64, ss: f64, nm: f64, ns: f64) -> f64 {
    let d = ((ss * ss + ns * ns) / 2.0).sqrt();
    if d < 1e-15 {
        f64::INFINITY
    } else {
        (sm - nm) / d
    }
}

fn log2_binomial(n: usize, k: usize) -> f64 {
    if k == 0 || k >= n {
        return 0.0;
    }
    let p = k as f64 / n as f64;
    let h = -p * p.ln() - (1.0 - p) * (1.0 - p).ln();
    n as f64 * h / std::f64::consts::LN_2
}

fn print_human_report(report: &serde_json::Value) {
    println!("HMS Benchmark Suite Results");
    println!("==========================\n");

    if let Some(cap) = report.get("capacity") {
        println!(
            "1. CAPACITY (D={})",
            report["meta"]["primary_dim"].as_u64().unwrap_or(0)
        );
        if let Some(maj) = cap["majority_vote"].as_array() {
            println!("   MAJORITY VOTE:");
            println!("   K   | Recovery | d'");
            for e in maj {
                println!(
                    "   {:>4} | {:>6.1}%  | {:>5.1}",
                    e["k"].as_u64().unwrap_or(0),
                    e["recovery_rate"].as_f64().unwrap_or(0.0) * 100.0,
                    e["d_prime"].as_f64().unwrap_or(0.0)
                );
            }
        }
        if let Some(blm) = cap["bloom_filter"].as_array() {
            println!("   BLOOM FILTER (containment similarity):");
            println!("   K   | Recovery | d'");
            for e in blm {
                println!(
                    "   {:>4} | {:>6.1}%  | {:>5.1}",
                    e["k"].as_u64().unwrap_or(0),
                    e["recovery_rate"].as_f64().unwrap_or(0.0) * 100.0,
                    e["d_prime"].as_f64().unwrap_or(0.0)
                );
            }
        }
        println!();
    }

    if let Some(bf) = report.get("binding_fidelity") {
        println!(
            "2. BINDING FIDELITY  d'={:.1}",
            bf["d_prime"].as_f64().unwrap_or(0.0)
        );
        println!();
    }

    if let Some(pr) = report.get("preregistered") {
        if let Some(svd) = pr.get("sparse_vs_dense_hopfield") {
            println!("\nPRE-REGISTERED: Sparse vs Dense Hopfield");
            println!(
                "  Kill fires: {}",
                svd["kill_fires"].as_bool().unwrap_or(true)
            );
        }
    }

    if let Some(hbm) = report.get("holographic_bloom_memory") {
        println!("\n7. HOLOGRAPHIC BLOOM MEMORY");
        println!("   Bloom bundling + density-corrected containment\n");
        if let Some(sweep) = hbm["capacity_sweep"].as_array() {
            println!("   Items | Density | Recall  | FPR     | d'      | Gap");
            println!("   ------+---------+---------+---------+---------+------");
            for e in sweep {
                println!(
                    "   {:>5} | {:>6.3}  | {:>6.1}% | {:>6.3}% | {:>7.1} | {:.4}",
                    e["n_items"].as_u64().unwrap_or(0),
                    e["density"].as_f64().unwrap_or(0.0),
                    e["recall_at_0.5"].as_f64().unwrap_or(0.0) * 100.0,
                    e["false_positive_rate_at_0.5"].as_f64().unwrap_or(0.0) * 100.0,
                    e["d_prime"].as_f64().unwrap_or(0.0),
                    e["gap"].as_f64().unwrap_or(0.0)
                );
            }
        }
        if let Some(sr) = hbm.get("structured_retrieval") {
            println!("\n   STRUCTURED RETRIEVAL (role-as-permutation + bloom):");
            println!(
                "   Agent recovery: {:.1}% ({} facts, vocab={})",
                sr["agent_recovery_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                sr["n_facts"].as_u64().unwrap_or(0),
                sr["vocab_size"].as_u64().unwrap_or(0)
            );
        }
        if let Some(scale) = hbm["scaling_sweep"].as_array() {
            println!("\n   SCALING SWEEP:");
            println!("   Roles | Vocab | Facts | Accuracy");
            println!("   ------+-------+-------+---------");
            for e in scale {
                println!(
                    "   {:>5} | {:>5} | {:>5} | {:>6.1}%",
                    e["n_roles"].as_u64().unwrap_or(0),
                    e["vocab_size"].as_u64().unwrap_or(0),
                    e["n_facts"].as_u64().unwrap_or(0),
                    e["accuracy"].as_f64().unwrap_or(0.0) * 100.0
                );
            }
        }
        if let Some(bh) = hbm.get("bloom_hopfield_hybrid") {
            println!(
                "\n   BLOOM+HOPFIELD HYBRID (N={}):",
                bh["n_items"].as_u64().unwrap_or(0)
            );
            println!("   Corrupt | Jaccard | Hopfield | Hop-Top-K");
            println!("   --------+---------+----------+----------");
            if let Some(rs) = bh["results"].as_array() {
                for r in rs {
                    println!(
                        "   {:>6.0}% | {:>6.1}% | {:>7.1}% | {:>8.1}%",
                        r["corruption"].as_f64().unwrap_or(0.0) * 100.0,
                        r["jaccard_nn_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                        r["hopfield_full_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                        r["hopfield_top_k_accuracy"].as_f64().unwrap_or(0.0) * 100.0
                    );
                }
            }
        }
    }

    if let Some(ct) = report.get("compositional_text") {
        println!("\n8. COMPOSITIONAL TEXT ENCODING");
        println!("    Role-as-permutation sentence encoding\n");
        println!(
            "    Sentences: {}  Vocab: {}",
            ct["n_sentences"].as_u64().unwrap_or(0),
            ct["vocab_size"].as_u64().unwrap_or(0)
        );
        println!(
            "    Subject accuracy: {:.1}%",
            ct["subject_accuracy"].as_f64().unwrap_or(0.0) * 100.0
        );
        println!(
            "    Verb accuracy:    {:.1}%",
            ct["verb_accuracy"].as_f64().unwrap_or(0.0) * 100.0
        );
        println!(
            "    Object accuracy:  {:.1}%",
            ct["object_accuracy"].as_f64().unwrap_or(0.0) * 100.0
        );
        println!(
            "    Overall:          {:.1}%",
            ct["overall_accuracy"].as_f64().unwrap_or(0.0) * 100.0
        );
    }
}
