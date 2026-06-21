// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Window-experiments: compare four approaches for bounded-capacity Bloom bundling.
//!
//! 1. Sliding Window (FIFO) -- keep most recent W items, drop oldest
//! 2. Majority-Vote Bundle -- threshold-based majority vote (no Bloom)
//! 3. Reservoir Sampling Bundle -- uniform-random replacement, periodic re-bundle
//! 4. Tiered Window -- two tiers with temporal priority decay

use holographic_memory::core::entangled::EntangledHVec;

const DIM: usize = 16384;
const DENOM: usize = 256;
const MAX_ITEMS: usize = 5000;
const N_PROBES: usize = 200;

fn load_points() -> Vec<usize> {
    vec![10, 50, 100, 200, 500, 700, 1000, 2000, 5000]
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}
fn fmin(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn fmax(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Generate all items and non-member probes up front.
fn generate_items() -> (Vec<EntangledHVec>, Vec<EntangledHVec>) {
    let items: Vec<EntangledHVec> = (0..MAX_ITEMS)
        .map(|i| EntangledHVec::new_with_density(DIM, DENOM, i as u64 * 37 + 1))
        .collect();
    let non_members: Vec<EntangledHVec> = (0..N_PROBES)
        .map(|i| {
            EntangledHVec::new_with_density(
                DIM,
                DENOM,
                (MAX_ITEMS + i) as u64 * 37 + 9999,
            )
        })
        .collect();
    (items, non_members)
}

/// Print TSV header.
fn print_header() {
    println!(
        "scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap\tfirst100_recall"
    );
}

/// Measure corrected containment stats for a bundle against members and non-members.
fn measure_stats(
    members: &[EntangledHVec],
    non_members: &[EntangledHVec],
    bundle: &EntangledHVec,
) -> (f64, f64, f64, f64, f64) {
    let member_sims: Vec<f64> = members
        .iter()
        .map(|item| item.corrected_containment(bundle))
        .collect();
    let member_mean = mean(&member_sims);
    let member_min = fmin(&member_sims);

    let nonmember_sims: Vec<f64> = non_members
        .iter()
        .map(|item| item.corrected_containment(bundle))
        .collect();
    let nonmember_mean = mean(&nonmember_sims);
    let nonmember_max = fmax(&nonmember_sims);
    let gap = member_min - nonmember_max;

    (member_mean, member_min, nonmember_mean, nonmember_max, gap)
}

/// Recall rate: fraction of first-100 items whose corrected containment > 0.5.
fn recall_first100(items: &[EntangledHVec], bundle: &EntangledHVec) -> f64 {
    let count = items.len().min(100);
    if count == 0 {
        return 1.0;
    }
    let recalled = items[..count]
        .iter()
        .filter(|item| item.corrected_containment(bundle) > 0.5)
        .count();
    recalled as f64 / count as f64
}

// ---------------------------------------------------------------------------
// Approach 1: Sliding Window (FIFO)
// ---------------------------------------------------------------------------

fn run_sliding_window(items: &[EntangledHVec], non_members: &[EntangledHVec], window_size: usize) {
    let scheme = format!("sliding_w{}", window_size);

    for &n in &load_points() {
        if n > MAX_ITEMS {
            break;
        }

        // The window contains the most recent `window_size` items (or all if n < window_size).
        let window_start = n.saturating_sub(window_size);
        let window = &items[window_start..n];

        let bundle = EntangledHVec::bundle_bloom(window);

        // Members = items currently in the window
        let (mm, mmin, nmm, nmmax, gap) = measure_stats(window, non_members, &bundle);
        let r100 = recall_first100(items, &bundle);

        println!(
            "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
            scheme, n, mm, mmin, nmm, nmmax, gap, r100
        );
    }
}

// ---------------------------------------------------------------------------
// Approach 2: Majority-Vote Bundle
// ---------------------------------------------------------------------------

fn run_majority_vote(items: &[EntangledHVec], non_members: &[EntangledHVec]) {
    for &n in &load_points() {
        if n > MAX_ITEMS {
            break;
        }

        let bundle = EntangledHVec::bundle(&items[..n]);

        // For majority vote, use Jaccard similarity instead of containment,
        // since the bundle is not a Bloom filter (it is sparse like individual items).
        let member_sims: Vec<f64> = items[..n]
            .iter()
            .map(|item| item.similarity(&bundle))
            .collect();
        let nonmember_sims: Vec<f64> = non_members
            .iter()
            .map(|item| item.similarity(&bundle))
            .collect();

        let mm = mean(&member_sims);
        let mmin = fmin(&member_sims);
        let nmm = mean(&nonmember_sims);
        let nmmax = fmax(&nonmember_sims);
        let gap = mmin - nmmax;

        // Recall for majority vote: check if first-100 items have sim > baseline
        let r100_count = items[..100.min(n)]
            .iter()
            .filter(|item| item.similarity(&bundle) > nmmax)
            .count();
        let r100 = r100_count as f64 / 100.0_f64.min(n as f64);

        println!(
            "majority\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
            n, mm, mmin, nmm, nmmax, gap, r100
        );
    }
}

// ---------------------------------------------------------------------------
// Approach 3: Reservoir Sampling Bundle
// ---------------------------------------------------------------------------

fn run_reservoir(items: &[EntangledHVec], non_members: &[EntangledHVec], reservoir_size: usize) {
    use holographic_memory::core::entangled::hash_u64;

    let scheme = format!("reservoir_r{}", reservoir_size);

    for &n in &load_points() {
        if n > MAX_ITEMS {
            break;
        }

        // Reservoir sampling: deterministic via hash_u64 for reproducibility.
        let mut reservoir: Vec<usize> = Vec::with_capacity(reservoir_size);

        for i in 0..n {
            if i < reservoir_size {
                reservoir.push(i);
            } else {
                // Replace element j with probability reservoir_size / (i+1)
                let r = (hash_u64(42, i as u64) % (i as u64 + 1)) as usize;
                if r < reservoir_size {
                    reservoir[r] = i;
                }
            }
        }

        let reservoir_items: Vec<&EntangledHVec> =
            reservoir.iter().map(|&idx| &items[idx]).collect();
        let bundle = EntangledHVec::bundle_bloom(&reservoir_items);

        // Members = items currently in the reservoir
        let member_sims: Vec<f64> = reservoir
            .iter()
            .map(|&idx| items[idx].corrected_containment(&bundle))
            .collect();
        let mm = mean(&member_sims);
        let mmin = fmin(&member_sims);

        let nonmember_sims: Vec<f64> = non_members
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .collect();
        let nmm = mean(&nonmember_sims);
        let nmmax = fmax(&nonmember_sims);
        let gap = mmin - nmmax;

        // Recall: fraction of first 100 items still in reservoir
        let first100_in_reservoir = reservoir.iter().filter(|&&idx| idx < 100).count();
        let r100 = first100_in_reservoir as f64 / 100.0_f64.min(n as f64);

        println!(
            "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
            scheme, n, mm, mmin, nmm, nmmax, gap, r100
        );
    }
}

// ---------------------------------------------------------------------------
// Approach 4: Tiered Window
// ---------------------------------------------------------------------------

fn run_tiered_window(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    w1: usize,
    w2: usize,
) {
    let scheme = format!("tiered_{}_{}", w1, w2);

    for &n in &load_points() {
        if n > MAX_ITEMS {
            break;
        }

        // Tier 1: most recent w1 items
        let t1_start = n.saturating_sub(w1);
        let tier1_items = &items[t1_start..n];
        let bundle_t1 = EntangledHVec::bundle_bloom(tier1_items);

        // Tier 2: items from position (n - w1 - w2) to (n - w1)
        let t2_end = t1_start;
        let t2_start = t2_end.saturating_sub(w2);
        let has_tier2 = t2_end > t2_start;
        let bundle_t2 = if has_tier2 {
            Some(EntangledHVec::bundle_bloom(&items[t2_start..t2_end]))
        } else {
            None
        };

        // Combined score: max(tier1_score, tier2_score * 0.5)
        let combined_score = |item: &EntangledHVec| -> f64 {
            let s1 = item.corrected_containment(&bundle_t1);
            let s2 = bundle_t2
                .as_ref()
                .map(|b| item.corrected_containment(b) * 0.5)
                .unwrap_or(0.0);
            s1.max(s2)
        };

        // Members = all items in either tier
        let all_tier_start = t2_start.min(t1_start);
        let member_sims: Vec<f64> = items[all_tier_start..n]
            .iter()
            .map(&combined_score)
            .collect();
        let mm = mean(&member_sims);
        let mmin = fmin(&member_sims);

        let nonmember_sims: Vec<f64> = non_members
            .iter()
            .map(&combined_score)
            .collect();
        let nmm = mean(&nonmember_sims);
        let nmmax = fmax(&nonmember_sims);
        let gap = mmin - nmmax;

        // Recall of first 100 items
        let r100_count = items[..100.min(n)]
            .iter()
            .filter(|item| combined_score(item) > 0.5)
            .count();
        let r100 = r100_count as f64 / 100.0_f64.min(n as f64);

        println!(
            "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
            scheme, n, mm, mmin, nmm, nmmax, gap, r100
        );
    }
}

// ---------------------------------------------------------------------------
// Flat Bloom baseline (for comparison)
// ---------------------------------------------------------------------------

fn run_flat_bloom(items: &[EntangledHVec], non_members: &[EntangledHVec]) {
    for &n in &load_points() {
        if n > MAX_ITEMS {
            break;
        }

        let bundle = EntangledHVec::bundle_bloom(&items[..n]);
        let density = bundle.indices().len() as f64 / DIM as f64;

        let (mm, mmin, nmm, nmmax, gap) = measure_stats(&items[..n], non_members, &bundle);
        let r100 = recall_first100(items, &bundle);

        println!(
            "flat_bloom\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
            n, mm, mmin, nmm, nmmax, gap, r100
        );

        if density > 0.995 {
            break;
        }
    }
}

// ---------------------------------------------------------------------------

fn main() {
    println!(
        "# Window experiments: D={} denom={} max_items={} probes={}",
        DIM, DENOM, MAX_ITEMS, N_PROBES
    );
    println!("# gap = member_min - nonmember_max (>0 means perfect separation)");
    println!(
        "# first100_recall = fraction of first 100 items retrievable at each load point"
    );
    println!();

    let (items, non_members) = generate_items();

    // Baseline: flat Bloom
    print_header();
    run_flat_bloom(&items, &non_members);
    println!();

    // Approach 1: Sliding Window with various sizes
    print_header();
    run_sliding_window(&items, &non_members, 100);
    println!();
    print_header();
    run_sliding_window(&items, &non_members, 200);
    println!();
    print_header();
    run_sliding_window(&items, &non_members, 500);
    println!();

    // Approach 2: Majority-Vote
    print_header();
    run_majority_vote(&items, &non_members);
    println!();

    // Approach 3: Reservoir Sampling with various sizes
    print_header();
    run_reservoir(&items, &non_members, 100);
    println!();
    print_header();
    run_reservoir(&items, &non_members, 200);
    println!();
    print_header();
    run_reservoir(&items, &non_members, 500);
    println!();

    // Approach 4: Tiered Window (w1=100, w2=400)
    print_header();
    run_tiered_window(&items, &non_members, 100, 400);
    println!();

    // Summary
    println!("# SUMMARY");
    println!("# flat_bloom: unbounded growth, gap hits 0 around n=700");
    println!(
        "# sliding_wN: bounded at N items, gap stays positive, old items forgotten (FIFO)"
    );
    println!("# majority: threshold vote, sparse output, limited capacity");
    println!(
        "# reservoir_rN: bounded at N items, gap stays positive, uniform forgetting"
    );
    println!("# tiered: two-tier temporal priority, wider effective window");
}
