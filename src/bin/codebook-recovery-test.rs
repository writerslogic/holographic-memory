// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Codebook recovery experiment: three-arm test measuring whether Bloom-bundled
//! codebook-composed facts (XOR triples) can be recovered via argmax query,
//! or whether the Category 4 density plateau is a mere density artifact.

use holographic_memory::core::entangled::{hash_u64, EntangledHVec};
use std::collections::{HashMap, HashSet};

const DIM: usize = 16384;
const DENOM: usize = 256;
const N_CONCEPTS: usize = 100;
const MAX_FACTS: usize = 5000;

fn load_points() -> Vec<usize> {
    vec![10, 50, 100, 200, 500, 1000, 2000, 5000]
}

/// Density of a bundle.
fn density(bundle: &EntangledHVec) -> f64 {
    bundle.indices().len() as f64 / DIM as f64
}

struct Fact {
    country_idx: usize,
    capital_idx: usize,
    composed: EntangledHVec,
}

fn main() {
    // ── Generate codebook ──────────────────────────────────────────────
    let concepts: Vec<EntangledHVec> = (0..N_CONCEPTS)
        .map(|i| EntangledHVec::new_with_density(DIM, DENOM, i as u64 * 7 + 42))
        .collect();

    // Fixed relation concept (separate seed space)
    let relation = EntangledHVec::new_with_density(DIM, DENOM, 0xCAFE_BABE);

    // ── Generate facts: country_i XOR relation XOR capital_j ─────────
    // Use country from [0..100), capital from [0..100), take first MAX_FACTS pairs
    let mut facts: Vec<Fact> = Vec::with_capacity(MAX_FACTS);
    'outer: for ci in 0..N_CONCEPTS {
        for ki in 0..N_CONCEPTS {
            if facts.len() >= MAX_FACTS {
                break 'outer;
            }
            let composed = concepts[ci].bind(&relation).bind(&concepts[ki]);
            facts.push(Fact {
                country_idx: ci,
                capital_idx: ki,
                composed,
            });
        }
    }

    // ── Arm A: codebook composition ─────────────────────────────────
    let all_composed: Vec<&EntangledHVec> = facts.iter().map(|f| &f.composed).collect();

    // ── Arm B setup: bounded pool, independent ──────────────────────
    // Measure the distinct index pool across ALL arm-A facts
    let mut arm_a_pool: HashSet<u32> = HashSet::new();
    for f in &facts {
        for &idx in f.composed.indices() {
            arm_a_pool.insert(idx);
        }
    }
    let pool_indices: Vec<u32> = {
        let mut v: Vec<u32> = arm_a_pool.into_iter().collect();
        v.sort_unstable();
        v
    };
    let pool_size = pool_indices.len();

    // For each fact, measure its active count to match in arm B
    let arm_a_active_counts: Vec<usize> =
        facts.iter().map(|f| f.composed.indices().len()).collect();
    let avg_active =
        arm_a_active_counts.iter().sum::<usize>() as f64 / arm_a_active_counts.len() as f64;

    // Generate arm-B items: random samples from the bounded pool
    let arm_b_items: Vec<EntangledHVec> = (0..MAX_FACTS)
        .map(|i| {
            let target_k = arm_a_active_counts[i];
            let mut selected: Vec<u32> = Vec::with_capacity(target_k);
            let seed = i as u64 * 31 + 0xB00B;
            for p in 0..target_k {
                let pool_idx = hash_u64(seed, p as u64) as usize % pool_size;
                selected.push(pool_indices[pool_idx]);
            }
            selected.sort_unstable();
            selected.dedup();
            EntangledHVec::from_indices(selected, DIM)
        })
        .collect();

    // ── Arm C: unbounded IID baseline ───────────────────────────────
    let arm_c_items: Vec<EntangledHVec> = (0..MAX_FACTS)
        .map(|i| EntangledHVec::new_with_density(DIM, DENOM, i as u64 * 1_000_003 + 0xDEAD))
        .collect();

    // ── Non-member probes (for gap metrics on B and C) ──────────────
    let n_probes = 200usize;
    let probes_b: Vec<EntangledHVec> = (0..n_probes)
        .map(|i| {
            let target_k = avg_active.round() as usize;
            let seed = i as u64 * 97 + 0xF00D;
            let mut selected: Vec<u32> = Vec::with_capacity(target_k);
            for p in 0..target_k {
                let pool_idx = hash_u64(seed, p as u64) as usize % pool_size;
                selected.push(pool_indices[pool_idx]);
            }
            selected.sort_unstable();
            selected.dedup();
            EntangledHVec::from_indices(selected, DIM)
        })
        .collect();

    let probes_c: Vec<EntangledHVec> = (0..n_probes)
        .map(|i| {
            EntangledHVec::new_with_density(
                DIM,
                DENOM,
                (MAX_FACTS + n_probes + i) as u64 * 1_000_003 + 0xBEEF,
            )
        })
        .collect();

    // ── Kill criterion ──────────────────────────────────────────────
    println!("KILL CRITERION: Codebook recovery at n=5000 must exceed 80% top-1 AND arm A must beat arm B gap by clear margin.");
    println!(
        "If recovery < 80%: codebook stores concepts not facts, Category 4 is density artifact."
    );
    println!("If A and B gap tie: structure claim dead, only bounded-pool effect is real.");
    println!();

    // ── Print setup info ────────────────────────────────────────────
    println!(
        "# Setup: D={}, denom={} (k={}), {} concepts, {} max facts",
        DIM,
        DENOM,
        DIM / DENOM,
        N_CONCEPTS,
        MAX_FACTS
    );
    println!(
        "# Arm-A index pool size: {} / {} = {:.4} of dimension",
        pool_size,
        DIM,
        pool_size as f64 / DIM as f64
    );
    println!("# Arm-A avg active count per fact: {:.1}", avg_active);
    println!();

    // ══════════════════════════════════════════════════════════════════
    // ARM A: Codebook composition (100 concepts, structured)
    // ══════════════════════════════════════════════════════════════════
    println!("# ARM A: Codebook composition (100 concepts, structured)");
    println!(
        "# arm\tn_items\trecovery_accuracy\tn_ambiguous\tn_correct_but_ambig\tbundle_density\tgap"
    );

    let load_pts = load_points();
    let mut arm_a_final_recovery = 0.0f64;

    for &n in &load_pts {
        let n = n.min(facts.len());
        let bundle = EntangledHVec::bundle_bloom(&all_composed[..n]);
        let d = density(&bundle);

        // Recovery test: for each stored fact, query all 100 candidate capitals
        let mut correct = 0usize;
        let mut ambiguous = 0usize;
        let mut correct_but_ambig = 0usize;

        for fact in &facts[..n] {
            let country_idx = fact.country_idx;
            let capital_idx = fact.capital_idx;

            let mut pass_count = 0usize;
            let mut correct_passes = false;

            for cand_j in 0..N_CONCEPTS {
                let candidate_fact = concepts[country_idx]
                    .bind(&relation)
                    .bind(&concepts[cand_j]);
                let cc = candidate_fact.corrected_containment(&bundle);
                if cc >= 0.99 {
                    pass_count += 1;
                    if cand_j == capital_idx {
                        correct_passes = true;
                    }
                }
            }

            if correct_passes && pass_count == 1 {
                correct += 1;
            } else if correct_passes && pass_count > 1 {
                ambiguous += 1;
                correct_but_ambig += 1;
            }
        }

        let recovery = correct as f64 / n as f64;
        if n == facts.len().min(MAX_FACTS) || n == 5000 {
            arm_a_final_recovery = recovery;
        }

        // Gap metric for arm A
        let member_mins: Vec<f64> = (0..n.min(200))
            .map(|fi| facts[fi].composed.corrected_containment(&bundle))
            .collect();
        let member_min = member_mins.iter().cloned().fold(f64::INFINITY, f64::min);

        // Non-member probes: random compositions NOT in the bundle
        let mut nonmember_maxes: Vec<f64> = Vec::new();
        for pi in 0..200usize {
            let ci = (n + pi) % N_CONCEPTS;
            let ki = ((n + pi) / N_CONCEPTS + 50) % N_CONCEPTS;
            let probe = concepts[ci].bind(&relation).bind(&concepts[ki]);
            let is_stored = facts[..n]
                .iter()
                .any(|f| f.country_idx == ci && f.capital_idx == ki);
            if !is_stored {
                nonmember_maxes.push(probe.corrected_containment(&bundle));
            }
        }
        let nonmember_max = if nonmember_maxes.is_empty() {
            f64::NEG_INFINITY
        } else {
            nonmember_maxes
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let gap = member_min - nonmember_max;

        println!(
            "A\t{}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}",
            n, recovery, ambiguous, correct_but_ambig, d, gap
        );
    }

    println!();

    // ══════════════════════════════════════════════════════════════════
    // ARM B: Bounded pool, independent (same density growth, no structure)
    // ══════════════════════════════════════════════════════════════════
    println!("# ARM B: Bounded pool, independent (same density growth, no structure)");
    println!("# arm\tn_items\tbundle_density\tmember_min\tnonmember_max\tgap");

    for &n in &load_pts {
        let n = n.min(arm_b_items.len());
        let slice: Vec<&EntangledHVec> = arm_b_items[..n].iter().collect();
        let bundle = EntangledHVec::bundle_bloom(&slice);
        let d = density(&bundle);

        let member_ccs: Vec<f64> = arm_b_items[..n]
            .iter()
            .map(|v| v.corrected_containment(&bundle))
            .collect();
        let member_min = member_ccs.iter().cloned().fold(f64::INFINITY, f64::min);

        let probe_ccs: Vec<f64> = probes_b
            .iter()
            .map(|v| v.corrected_containment(&bundle))
            .collect();
        let nonmember_max = probe_ccs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gap = member_min - nonmember_max;

        println!(
            "B\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            n, d, member_min, nonmember_max, gap
        );
    }

    println!();

    // ══════════════════════════════════════════════════════════════════
    // ARM C: Unbounded IID baseline
    // ══════════════════════════════════════════════════════════════════
    println!("# ARM C: Unbounded IID baseline");
    println!("# arm\tn_items\tbundle_density\tmember_min\tnonmember_max\tgap");

    for &n in &load_pts {
        let n = n.min(arm_c_items.len());
        let slice: Vec<&EntangledHVec> = arm_c_items[..n].iter().collect();
        let bundle = EntangledHVec::bundle_bloom(&slice);
        let d = density(&bundle);

        let member_ccs: Vec<f64> = arm_c_items[..n]
            .iter()
            .map(|v| v.corrected_containment(&bundle))
            .collect();
        let member_min = member_ccs.iter().cloned().fold(f64::INFINITY, f64::min);

        let probe_ccs: Vec<f64> = probes_c
            .iter()
            .map(|v| v.corrected_containment(&bundle))
            .collect();
        let nonmember_max = probe_ccs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gap = member_min - nonmember_max;

        println!(
            "C\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            n, d, member_min, nonmember_max, gap
        );
    }

    println!();

    // ══════════════════════════════════════════════════════════════════
    // DIAGNOSTIC: Recovery accuracy by concept sharing count (n=5000)
    // ══════════════════════════════════════════════════════════════════
    println!("# DIAGNOSTIC: Recovery accuracy by concept sharing count (n=5000)");
    println!("# sharing_bucket\tn_facts\trecovery_accuracy");

    let n_diag = MAX_FACTS.min(facts.len());
    let bundle_diag = EntangledHVec::bundle_bloom(&all_composed[..n_diag]);

    // Count concept usage
    let mut concept_usage: HashMap<usize, usize> = HashMap::new();
    for f in &facts[..n_diag] {
        *concept_usage.entry(f.country_idx).or_insert(0) += 1;
        *concept_usage.entry(f.capital_idx).or_insert(0) += 1;
    }

    // For each fact, compute max sharing of its two concepts and recovery
    struct DiagFact {
        max_sharing: usize,
        recovered: bool,
    }
    let mut diag_facts: Vec<DiagFact> = Vec::with_capacity(n_diag);

    // Progress indicator for the slow diagnostic pass
    for fact in &facts[..n_diag] {
        let country_idx = fact.country_idx;
        let capital_idx = fact.capital_idx;
        let max_sharing = std::cmp::max(
            *concept_usage.get(&country_idx).unwrap_or(&0),
            *concept_usage.get(&capital_idx).unwrap_or(&0),
        );

        let mut pass_count = 0usize;
        let mut correct_passes = false;
        for cand_j in 0..N_CONCEPTS {
            let candidate_fact = concepts[country_idx]
                .bind(&relation)
                .bind(&concepts[cand_j]);
            let cc = candidate_fact.corrected_containment(&bundle_diag);
            if cc >= 0.99 {
                pass_count += 1;
                if cand_j == capital_idx {
                    correct_passes = true;
                }
            }
        }
        let recovered = correct_passes && pass_count == 1;
        diag_facts.push(DiagFact {
            max_sharing,
            recovered,
        });
    }

    // Bucket by sharing count
    let mut buckets: HashMap<usize, (usize, usize)> = HashMap::new();
    for df in &diag_facts {
        let entry = buckets.entry(df.max_sharing).or_insert((0, 0));
        entry.0 += 1;
        if df.recovered {
            entry.1 += 1;
        }
    }

    let mut bucket_keys: Vec<usize> = buckets.keys().cloned().collect();
    bucket_keys.sort();
    for &k in &bucket_keys {
        let (total, recovered) = buckets[&k];
        let acc = recovered as f64 / total as f64;
        println!("{}\t{}\t{:.4}", k, total, acc);
    }

    println!();

    // ── Verdict ─────────────────────────────────────────────────────
    println!("# ═══════════════════════════════════════════════════════════");
    println!("# VERDICT");
    if arm_a_final_recovery >= 0.80 {
        println!(
            "# Recovery at n=5000: {:.1}% >= 80% => codebook stores FACTS, not just concepts.",
            arm_a_final_recovery * 100.0
        );
    } else {
        println!(
            "# Recovery at n=5000: {:.1}% < 80% => KILL: codebook stores concepts not facts, Category 4 is density artifact.",
            arm_a_final_recovery * 100.0
        );
    }
    println!("# ═══════════════════════════════════════════════════════════");
}
