// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Nonlinear bundling experiments for breaking the Bloom capacity wall.
//!
//! Binary OR bundling (Bloom) hits gap=0 at n~700 for D=16384, denom=256.
//! Any LINEAR readout on an ADDITIVE bundle reads component statistics, not
//! structure. These four approaches introduce nonlinearity to break the wall.
//!
//! Approaches:
//! 1. Top-K Counting: z-score readout on per-index frequency counts
//! 2. Saturation Counting: counting with capped max (4, 8, 16, 255)
//! 3. Competitive Inhibition: log-compressed excess over expected baseline
//! 4. Frequency-Inverse Weighting: TF-IDF style 1/log(1+count) weighting

use holographic_memory::core::entangled::EntangledHVec;

const DIM: usize = 16384;
const DENOM: usize = 256;
const MAX_ITEMS: usize = 50000;
const N_PROBES: usize = 200;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn load_points() -> Vec<usize> {
    vec![10, 50, 100, 200, 500, 700, 1000, 2000, 5000, 10000, 20000, 50000]
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

/// Pre-generate all items and non-member probes once.
fn generate_vectors() -> (Vec<EntangledHVec>, Vec<EntangledHVec>) {
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

// ---------------------------------------------------------------------------
// Approach 1: Top-K Counting (z-score readout)
// ---------------------------------------------------------------------------

struct CountingBundle {
    counts: Vec<u16>,
    n_items: usize,
}

impl CountingBundle {
    fn new() -> Self {
        Self {
            counts: vec![0u16; DIM],
            n_items: 0,
        }
    }

    fn add(&mut self, item: &EntangledHVec) {
        for &idx in item.indices() {
            self.counts[idx as usize] = self.counts[idx as usize].saturating_add(1);
        }
        self.n_items += 1;
    }

    /// Z-score: (observed - expected) / sqrt(expected)
    /// where expected = n * k / D, and k = item's active count.
    /// Under Poisson model, each position hit ~ Poisson(n*k/D).
    /// Sum of k Poisson(lambda) = Poisson(k*lambda).
    /// For a member: sum includes their own +1 at each of k positions.
    fn z_score(&self, query: &EntangledHVec) -> f64 {
        let k = query.indices().len() as f64;
        let n = self.n_items as f64;
        let expected = n * k / DIM as f64;
        // Poisson variance of sum = expected (since sum of k independent
        // Poisson(n/D) is Poisson(k*n/D))
        let variance = expected;

        let observed: f64 = query
            .indices()
            .iter()
            .map(|&idx| self.counts[idx as usize] as f64)
            .sum();

        if variance <= 0.0 {
            return 0.0;
        }
        (observed - expected) / variance.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Approach 2: Saturation Counting
// ---------------------------------------------------------------------------

struct SaturationBundle {
    counts: Vec<u8>,
    n_items: usize,
    sat_max: u8,
}

impl SaturationBundle {
    fn new(sat_max: u8) -> Self {
        Self {
            counts: vec![0u8; DIM],
            n_items: 0,
            sat_max,
        }
    }

    fn add(&mut self, item: &EntangledHVec) {
        for &idx in item.indices() {
            let c = &mut self.counts[idx as usize];
            if *c < self.sat_max {
                *c += 1;
            }
        }
        self.n_items += 1;
    }

    /// Z-score using saturated counts. Saturation clips the Poisson tail,
    /// so we compute empirical mean and variance across all DIM positions
    /// as the null model.
    fn z_score(&self, query: &EntangledHVec) -> f64 {
        let k = query.indices().len() as f64;
        if k == 0.0 || self.n_items == 0 {
            return 0.0;
        }

        // Empirical mean and variance of saturated counts across all positions
        let global_mean: f64 =
            self.counts.iter().map(|&c| c as f64).sum::<f64>() / DIM as f64;
        let global_var: f64 = self
            .counts
            .iter()
            .map(|&c| {
                let d = c as f64 - global_mean;
                d * d
            })
            .sum::<f64>()
            / DIM as f64;

        let expected = k * global_mean;
        // Variance of sum of k iid draws with variance global_var
        let variance = k * global_var;

        let observed: f64 = query
            .indices()
            .iter()
            .map(|&idx| self.counts[idx as usize] as f64)
            .sum();

        if variance <= 0.0 {
            return 0.0;
        }
        (observed - expected) / variance.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Approach 3: Competitive Inhibition
// ---------------------------------------------------------------------------

struct InhibitionBundle {
    counts: Vec<u16>,
    n_items: usize,
}

impl InhibitionBundle {
    fn new() -> Self {
        Self {
            counts: vec![0u16; DIM],
            n_items: 0,
        }
    }

    fn add(&mut self, item: &EntangledHVec) {
        for &idx in item.indices() {
            self.counts[idx as usize] = self.counts[idx as usize].saturating_add(1);
        }
        self.n_items += 1;
    }

    /// Nonlinear readout: for each query index, compute excess = count - expected,
    /// then apply sign-preserving log compression: sign(x)*ln(1+|x|).
    ///
    /// This is the competitive inhibition: popular positions (high count) have
    /// diminishing returns. A member's signal comes from positions where their
    /// +1 contribution sits above baseline. The log compression means that
    /// outlier positions (hit by many items) don't dominate the score.
    ///
    /// Normalize by sqrt(k) for scale independence.
    fn z_score(&self, query: &EntangledHVec) -> f64 {
        let k = query.indices().len() as f64;
        let n = self.n_items as f64;
        if k == 0.0 || n == 0.0 {
            return 0.0;
        }

        // Expected count at any position: n * (D/DENOM) / D = n / DENOM
        let expected_per_pos = n / DENOM as f64;

        let observed: f64 = query
            .indices()
            .iter()
            .map(|&idx| {
                let c = self.counts[idx as usize] as f64;
                let excess = c - expected_per_pos;
                // Sign-preserving log compression
                if excess >= 0.0 {
                    excess.ln_1p()
                } else {
                    -(-excess).ln_1p()
                }
            })
            .sum();

        // Normalize by sqrt(k) so the score doesn't scale with item density
        observed / k.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Approach 4: Frequency-Inverse Weighting (TF-IDF for bundles)
// ---------------------------------------------------------------------------

struct IdfBundle {
    counts: Vec<u16>,
    n_items: usize,
}

impl IdfBundle {
    fn new() -> Self {
        Self {
            counts: vec![0u16; DIM],
            n_items: 0,
        }
    }

    fn add(&mut self, item: &EntangledHVec) {
        for &idx in item.indices() {
            self.counts[idx as usize] = self.counts[idx as usize].saturating_add(1);
        }
        self.n_items += 1;
    }

    /// IDF-weighted membership score.
    ///
    /// For each query index, contribute log(1 + N / max(1, count)).
    /// Rare positions (low count) contribute log(1+N) ~ large.
    /// Popular positions (high count) contribute ~ log(2).
    /// Positions with count=0 contribute 0 (not in bundle at all).
    ///
    /// A member has count >= 1 at ALL its positions (own contribution).
    /// A non-member only has count > 0 where the bundle happened to land.
    /// The IDF weighting amplifies rare positions where only the member
    /// (and few others) contributed, suppressing popular positions that
    /// provide no discrimination.
    fn idf_score(&self, query: &EntangledHVec) -> f64 {
        let k = query.indices().len() as f64;
        let n = self.n_items as f64;
        if k == 0.0 || n == 0.0 {
            return 0.0;
        }

        let observed: f64 = query
            .indices()
            .iter()
            .map(|&idx| {
                let c = self.counts[idx as usize] as f64;
                if c > 0.0 {
                    (1.0 + n / c).ln()
                } else {
                    0.0
                }
            })
            .sum();

        // Empirical mean and variance of IDF weights across all positions
        let idf_values: Vec<f64> = self
            .counts
            .iter()
            .map(|&c| {
                let c = c as f64;
                if c > 0.0 {
                    (1.0 + n / c).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let idf_mean: f64 = idf_values.iter().sum::<f64>() / DIM as f64;
        let idf_var: f64 = idf_values
            .iter()
            .map(|&v| {
                let d = v - idf_mean;
                d * d
            })
            .sum::<f64>()
            / DIM as f64;

        let expected = k * idf_mean;
        let variance = k * idf_var;

        if variance <= 0.0 {
            return 0.0;
        }
        (observed - expected) / variance.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Measurement runners (each returns gap data for summary)
// ---------------------------------------------------------------------------

fn run_bloom(items: &[EntangledHVec], non_members: &[EntangledHVec]) -> Vec<(usize, f64)> {
    println!("# Bloom baseline (corrected containment)");
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    let mut gaps = Vec::new();

    for &n in &load_points() {
        if n > items.len() {
            break;
        }
        let bundle = EntangledHVec::bundle_bloom(&items[..n]);

        let member_sims: Vec<f64> = items[..n]
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .collect();
        let nonmember_sims: Vec<f64> = non_members
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .collect();

        let mm = mean(&member_sims);
        let mmin = fmin(&member_sims);
        let nm = mean(&nonmember_sims);
        let nmax = fmax(&nonmember_sims);
        let gap = mmin - nmax;
        gaps.push((n, gap));

        println!(
            "bloom\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            n, mm, mmin, nm, nmax, gap
        );

        if bundle.indices().len() as f64 / DIM as f64 > 0.995 {
            break;
        }
    }
    println!();
    gaps
}

fn run_counting(items: &[EntangledHVec], non_members: &[EntangledHVec]) -> Vec<(usize, f64)> {
    println!("# Approach 1: Top-K Counting (z-score)");
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    let mut bundle = CountingBundle::new();
    let mut item_idx = 0;
    let mut gaps = Vec::new();

    for &n in &load_points() {
        if n > items.len() {
            break;
        }
        while item_idx < n {
            bundle.add(&items[item_idx]);
            item_idx += 1;
        }

        let member_scores: Vec<f64> = items[..n]
            .iter()
            .map(|item| bundle.z_score(item))
            .collect();
        let nonmember_scores: Vec<f64> = non_members
            .iter()
            .map(|item| bundle.z_score(item))
            .collect();

        let mm = mean(&member_scores);
        let mmin = fmin(&member_scores);
        let nm = mean(&nonmember_scores);
        let nmax = fmax(&nonmember_scores);
        let gap = mmin - nmax;
        gaps.push((n, gap));

        println!(
            "counting\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            n, mm, mmin, nm, nmax, gap
        );
    }
    println!();
    gaps
}

fn run_saturation(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    sat_max: u8,
) -> Vec<(usize, f64)> {
    println!("# Approach 2: Saturation Counting (max={})", sat_max);
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    let mut bundle = SaturationBundle::new(sat_max);
    let mut item_idx = 0;
    let mut gaps = Vec::new();

    for &n in &load_points() {
        if n > items.len() {
            break;
        }
        while item_idx < n {
            bundle.add(&items[item_idx]);
            item_idx += 1;
        }

        let member_scores: Vec<f64> = items[..n]
            .iter()
            .map(|item| bundle.z_score(item))
            .collect();
        let nonmember_scores: Vec<f64> = non_members
            .iter()
            .map(|item| bundle.z_score(item))
            .collect();

        let mm = mean(&member_scores);
        let mmin = fmin(&member_scores);
        let nm = mean(&nonmember_scores);
        let nmax = fmax(&nonmember_scores);
        let gap = mmin - nmax;
        gaps.push((n, gap));

        println!(
            "sat{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            sat_max, n, mm, mmin, nm, nmax, gap
        );
    }
    println!();
    gaps
}

fn run_inhibition(items: &[EntangledHVec], non_members: &[EntangledHVec]) -> Vec<(usize, f64)> {
    println!("# Approach 3: Competitive Inhibition (log-compressed excess)");
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    let mut bundle = InhibitionBundle::new();
    let mut item_idx = 0;
    let mut gaps = Vec::new();

    for &n in &load_points() {
        if n > items.len() {
            break;
        }
        while item_idx < n {
            bundle.add(&items[item_idx]);
            item_idx += 1;
        }

        let member_scores: Vec<f64> = items[..n]
            .iter()
            .map(|item| bundle.z_score(item))
            .collect();
        let nonmember_scores: Vec<f64> = non_members
            .iter()
            .map(|item| bundle.z_score(item))
            .collect();

        let mm = mean(&member_scores);
        let mmin = fmin(&member_scores);
        let nm = mean(&nonmember_scores);
        let nmax = fmax(&nonmember_scores);
        let gap = mmin - nmax;
        gaps.push((n, gap));

        println!(
            "inhibit\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            n, mm, mmin, nm, nmax, gap
        );
    }
    println!();
    gaps
}

fn run_idf(items: &[EntangledHVec], non_members: &[EntangledHVec]) -> Vec<(usize, f64)> {
    println!("# Approach 4: Frequency-Inverse Weighting (IDF)");
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    let mut bundle = IdfBundle::new();
    let mut item_idx = 0;
    let mut gaps = Vec::new();

    for &n in &load_points() {
        if n > items.len() {
            break;
        }
        while item_idx < n {
            bundle.add(&items[item_idx]);
            item_idx += 1;
        }

        let member_scores: Vec<f64> = items[..n]
            .iter()
            .map(|item| bundle.idf_score(item))
            .collect();
        let nonmember_scores: Vec<f64> = non_members
            .iter()
            .map(|item| bundle.idf_score(item))
            .collect();

        let mm = mean(&member_scores);
        let mmin = fmin(&member_scores);
        let nm = mean(&nonmember_scores);
        let nmax = fmax(&nonmember_scores);
        let gap = mmin - nmax;
        gaps.push((n, gap));

        println!(
            "idf\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            n, mm, mmin, nm, nmax, gap
        );
    }
    println!();
    gaps
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

fn find_capacity_wall(scheme: &str, gaps: &[(usize, f64)]) {
    // Find first n where gap <= 0
    let mut first_negative_n: Option<usize> = None;
    for &(n, gap) in gaps {
        if gap <= 0.0 {
            first_negative_n = Some(n);
            break;
        }
    }

    match first_negative_n {
        Some(wall) => {
            let multiplier = wall as f64 / 700.0;
            eprintln!(
                "  {:<12} wall at n={:<6} gap<=0  ({:.1}x over Bloom ~700)",
                scheme, wall, multiplier
            );
        }
        None => {
            let last_n = gaps.last().map(|(n, _)| *n).unwrap_or(0);
            let last_gap = gaps.last().map(|(_, g)| *g).unwrap_or(0.0);
            eprintln!(
                "  {:<12} gap STILL POSITIVE at n={} (gap={:.4})  (>{:.0}x over Bloom ~700)",
                scheme, last_n, last_gap, last_n as f64 / 700.0
            );
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    eprintln!(
        "Generating {} items + {} probes at D={}, denom={}...",
        MAX_ITEMS, N_PROBES, DIM, DENOM
    );
    let (items, non_members) = generate_vectors();
    eprintln!("Done. Running experiments.\n");

    println!("# Nonlinear Bundling Experiments");
    println!("# D={}, denom={}, probes={}", DIM, DENOM, N_PROBES);
    println!(
        "# gap = member_min - nonmember_max (>0 = perfect separation)"
    );
    println!();

    // --- Bloom baseline ---
    let bloom_gaps = run_bloom(&items, &non_members);

    // --- Approach 1: Top-K Counting ---
    let counting_gaps = run_counting(&items, &non_members);

    // --- Approach 2: Saturation Counting (4 variants) ---
    let sat4_gaps = run_saturation(&items, &non_members, 4);
    let sat8_gaps = run_saturation(&items, &non_members, 8);
    let sat16_gaps = run_saturation(&items, &non_members, 16);
    let sat255_gaps = run_saturation(&items, &non_members, 255);

    // --- Approach 3: Competitive Inhibition ---
    let inhibit_gaps = run_inhibition(&items, &non_members);

    // --- Approach 4: IDF ---
    let idf_gaps = run_idf(&items, &non_members);

    // --- Summary ---
    eprintln!();
    eprintln!("=== CAPACITY SUMMARY ===");
    eprintln!("  Bloom baseline wall: ~700 (from prior experiments)");
    find_capacity_wall("bloom", &bloom_gaps);
    find_capacity_wall("counting", &counting_gaps);
    find_capacity_wall("sat4", &sat4_gaps);
    find_capacity_wall("sat8", &sat8_gaps);
    find_capacity_wall("sat16", &sat16_gaps);
    find_capacity_wall("sat255", &sat255_gaps);
    find_capacity_wall("inhibit", &inhibit_gaps);
    find_capacity_wall("idf", &idf_gaps);
}
