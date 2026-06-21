// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Multi-bundle architectures for extending Bloom-bundle capacity.
//!
//! Flat Bloom bundling (binary OR of active indices) at D=16384, denom=256,
//! 200 probes hits gap=0 at n~700 due to density saturation.
//! This binary tests three multi-bundle schemes that keep per-bundle density low.

use holographic_memory::core::entangled::{hash_u64, EntangledHVec};

const DIM: usize = 16384;
const DENOM: usize = 256;
const N_PROBES: usize = 200;
const MAX_ITEMS: usize = 10000;

fn load_points() -> Vec<usize> {
    vec![10, 50, 100, 200, 500, 700, 1000, 2000, 5000, 10000]
}

fn mean(vals: &[f64]) -> f64 {
    vals.iter().sum::<f64>() / vals.len().max(1) as f64
}

fn fmin(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn fmax(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Density of a bundle: fraction of dimension with active indices.
fn density(bundle: &EntangledHVec) -> f64 {
    bundle.indices().len() as f64 / DIM as f64
}

/// Generate all items and non-member probes upfront.
fn generate_items_and_probes() -> (Vec<EntangledHVec>, Vec<EntangledHVec>) {
    let items: Vec<EntangledHVec> = (0..MAX_ITEMS)
        .map(|i| EntangledHVec::new_with_density(DIM, DENOM, i as u64 * 37 + 1))
        .collect();
    let probes: Vec<EntangledHVec> = (0..N_PROBES)
        .map(|i| EntangledHVec::new_with_density(DIM, DENOM, (MAX_ITEMS + i) as u64 * 37 + 9999))
        .collect();
    (items, probes)
}

/// Measure gap statistics for a set of bundles.
/// Query: max corrected_containment across all bundles.
fn measure_gap(
    members: &[EntangledHVec],
    probes: &[EntangledHVec],
    bundles: &[EntangledHVec],
) -> (f64, f64, f64, f64, f64) {
    let member_sims: Vec<f64> = members
        .iter()
        .map(|item| {
            bundles
                .iter()
                .map(|b| item.corrected_containment(b))
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();

    let nonmember_sims: Vec<f64> = probes
        .iter()
        .map(|item| {
            bundles
                .iter()
                .map(|b| item.corrected_containment(b))
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();

    let m_mean = mean(&member_sims);
    let m_min = fmin(&member_sims);
    let nm_mean = mean(&nonmember_sims);
    let nm_max = fmax(&nonmember_sims);
    let gap = m_min - nm_max;

    (m_mean, m_min, nm_mean, nm_max, gap)
}

// --- Flat Bloom baseline -------------------------------------------------

fn run_flat_bloom(items: &[EntangledHVec], probes: &[EntangledHVec]) {
    println!(
        "# FLAT BLOOM BASELINE  D={} denom={} probes={}",
        DIM, DENOM, N_PROBES
    );
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    for &n in &load_points() {
        if n > items.len() {
            break;
        }
        let bundle = EntangledHVec::bundle_bloom(&items[..n]);
        let d = density(&bundle);
        let bundles = vec![bundle];
        let (m_mean, m_min, nm_mean, nm_max, gap) = measure_gap(&items[..n], probes, &bundles);

        println!(
            "flat_bloom\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            n, m_mean, m_min, nm_mean, nm_max, gap
        );

        if d > 0.995 {
            break;
        }
    }
    println!();
}

// --- Approach 1: K-way Round-Robin Split ---------------------------------

fn run_round_robin(items: &[EntangledHVec], probes: &[EntangledHVec], k: usize) {
    println!(
        "# ROUND-ROBIN k={}  D={} denom={} probes={}",
        k, DIM, DENOM, N_PROBES
    );
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    for &n in &load_points() {
        if n > items.len() {
            break;
        }

        // Partition items into k buckets by round-robin.
        let mut buckets: Vec<Vec<&EntangledHVec>> = vec![Vec::new(); k];
        for (i, item) in items[..n].iter().enumerate() {
            buckets[i % k].push(item);
        }

        // Build one Bloom bundle per bucket.
        let bundles: Vec<EntangledHVec> = buckets
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| EntangledHVec::bundle_bloom(b))
            .collect();

        let (m_mean, m_min, nm_mean, nm_max, gap) = measure_gap(&items[..n], probes, &bundles);

        println!(
            "rr_k{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            k, n, m_mean, m_min, nm_mean, nm_max, gap
        );
    }
    println!();
}

// --- Approach 2: Power-of-Two-Choices ------------------------------------

fn run_power_of_two(items: &[EntangledHVec], probes: &[EntangledHVec], k: usize) {
    println!(
        "# POWER-OF-TWO-CHOICES k={}  D={} denom={} probes={}",
        k, DIM, DENOM, N_PROBES
    );
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    for &n in &load_points() {
        if n > items.len() {
            break;
        }

        // For each item, hash to 2 candidate bundles, insert into the less dense one.
        let mut buckets: Vec<Vec<&EntangledHVec>> = vec![Vec::new(); k];

        for (i, item) in items[..n].iter().enumerate() {
            let h1 = hash_u64(i as u64, 0xA1B2C3D4) as usize % k;
            let mut h2 = hash_u64(i as u64, 0xD4C3B2A1) as usize % k;

            // Ensure two different choices when possible.
            if h2 == h1 && k > 1 {
                h2 = (h2 + 1) % k;
            }

            // Pick the bucket with fewer items (proxy for lower density).
            let choice = if buckets[h1].len() <= buckets[h2].len() {
                h1
            } else {
                h2
            };
            buckets[choice].push(item);
        }

        let bundles: Vec<EntangledHVec> = buckets
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| EntangledHVec::bundle_bloom(b))
            .collect();

        let (m_mean, m_min, nm_mean, nm_max, gap) = measure_gap(&items[..n], probes, &bundles);

        println!(
            "po2_k{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            k, n, m_mean, m_min, nm_mean, nm_max, gap
        );
    }
    println!();
}

// --- Approach 3: Density-Gated Cascade -----------------------------------

fn run_density_cascade_fast(items: &[EntangledHVec], probes: &[EntangledHVec], threshold: f64) {
    println!(
        "# DENSITY-GATED CASCADE threshold={:.2}  D={} denom={} probes={}",
        threshold, DIM, DENOM, N_PROBES
    );
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    // Precompute: how many items fit per bundle at this threshold?
    // n items give density ~ 1 - (1 - 1/DENOM)^n.
    // Solve: n = ln(1 - threshold) / ln(1 - 1/DENOM)
    let items_per_bundle = ((1.0 - threshold).ln() / (1.0 - 1.0 / DENOM as f64).ln()) as usize;
    let items_per_bundle = items_per_bundle.max(1);

    for &n in &load_points() {
        if n > items.len() {
            break;
        }

        // Split items into sequential chunks of items_per_bundle.
        let n_bundles = n.div_ceil(items_per_bundle);
        let bundles: Vec<EntangledHVec> = (0..n_bundles)
            .map(|bi| {
                let start = bi * items_per_bundle;
                let end = (start + items_per_bundle).min(n);
                EntangledHVec::bundle_bloom(&items[start..end])
            })
            .collect();

        let actual_n_bundles = bundles.len();
        let (m_mean, m_min, nm_mean, nm_max, gap) = measure_gap(&items[..n], probes, &bundles);

        println!(
            "cascade_{:.0}pct\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t# {} bundles, ~{} items/bundle",
            threshold * 100.0,
            n,
            m_mean,
            m_min,
            nm_mean,
            nm_max,
            gap,
            actual_n_bundles,
            items_per_bundle
        );
    }
    println!();
}

fn main() {
    println!("# Multi-Bundle Architecture Experiments");
    println!(
        "# D={} denom={} probes={} max_items={}",
        DIM, DENOM, N_PROBES, MAX_ITEMS
    );
    println!("# gap = member_min - nonmember_max (>0 => perfect member/non-member separation)");
    println!();

    let (items, probes) = generate_items_and_probes();

    // -- Baseline ---------------------------------------------------------
    run_flat_bloom(&items, &probes);

    // -- Approach 1: K-way Round-Robin ------------------------------------
    for k in [2, 4, 8, 16] {
        run_round_robin(&items, &probes, k);
    }

    // -- Approach 2: Power-of-Two-Choices ---------------------------------
    for k in [2, 4, 8, 16] {
        run_power_of_two(&items, &probes, k);
    }

    // -- Approach 3: Density-Gated Cascade --------------------------------
    for threshold in [0.50, 0.40, 0.30] {
        run_density_cascade_fast(&items, &probes, threshold);
    }

    // -- Summary ----------------------------------------------------------
    println!("# CAPACITY SUMMARY");
    println!("# For each scheme, find the largest n_items where gap > 0.");
    println!("# Flat Bloom baseline collapses at ~700 items.");
    println!("# Round-Robin k=N should extend capacity ~Nx.");
    println!("# Power-of-Two-Choices should match or beat Round-Robin.");
    println!("# Density-Gated Cascade with threshold 0.50 should extend capacity proportionally.");
    println!("# The multiplier over baseline is capacity(scheme) / 700.");
}
