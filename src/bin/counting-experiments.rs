// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Counting-Bloom experiments: measure capacity of alternative bundling
// approaches against the flat binary Bloom baseline.
//
// Approaches:
//   1. Counting Bloom — per-index u32 counts instead of binary OR
//   2. Exponential Decay Counting — periodic count decay for temporal weighting
//   3. Normalized Counting — cosine similarity between query and count vector
//
// For each approach we measure the "gap" metric:
//   gap = member_min - nonmember_max  (of the relevant score)
// Positive gap means perfect separation; crossing zero marks capacity wall.

use holographic_memory::core::entangled::EntangledHVec;

// ---------------------------------------------------------------------------
// Counting Bloom bundle
// ---------------------------------------------------------------------------

struct CountingBloom {
    counts: Vec<u32>,
    n_items: usize,
    n_blocks: usize, // number of active indices per item
}

impl CountingBloom {
    fn new(dim: usize, n_blocks: usize) -> Self {
        Self {
            counts: vec![0u32; dim],
            n_items: 0,
            n_blocks,
        }
    }

    fn add(&mut self, item: &EntangledHVec) {
        for &idx in item.indices() {
            self.counts[idx as usize] += 1;
        }
        self.n_items += 1;
    }

    /// Score = (sum of counts at query indices) / n_blocks - expected_noise.
    /// Expected noise for a random query = n_items * n_blocks / dim
    /// (each of the n_blocks query indices hits a count that averages
    /// n_items * n_blocks / dim).
    fn score(&self, query: &EntangledHVec) -> f64 {
        let sum: u64 = query
            .indices()
            .iter()
            .map(|&idx| self.counts[idx as usize] as u64)
            .sum();
        let raw = sum as f64 / self.n_blocks as f64;
        let expected = self.n_items as f64 * self.n_blocks as f64 / self.counts.len() as f64;
        raw - expected
    }
}

// ---------------------------------------------------------------------------
// Exponential Decay Counting Bloom
// ---------------------------------------------------------------------------

struct DecayCountingBloom {
    counts: Vec<f64>,
    n_items: usize,
    n_blocks: usize,
    decay_factor: f64,
    consolidation_interval: usize,
    items_since_decay: usize,
}

impl DecayCountingBloom {
    fn new(dim: usize, n_blocks: usize, decay_factor: f64, consolidation_interval: usize) -> Self {
        Self {
            counts: vec![0.0; dim],
            n_items: 0,
            n_blocks,
            decay_factor,
            consolidation_interval,
            items_since_decay: 0,
        }
    }

    fn add(&mut self, item: &EntangledHVec) {
        for &idx in item.indices() {
            self.counts[idx as usize] += 1.0;
        }
        self.n_items += 1;
        self.items_since_decay += 1;

        if self.items_since_decay >= self.consolidation_interval {
            for c in self.counts.iter_mut() {
                *c *= self.decay_factor;
            }
            self.items_since_decay = 0;
        }
    }

    /// Score: same structure as counting Bloom but on decayed counts.
    /// Expected noise adjusts for decay: effective_items * n_blocks / dim.
    /// Since decay is multiplicative, effective_items is hard to compute exactly.
    /// We use the sum of all counts / n_blocks as a proxy for effective_n.
    fn score(&self, query: &EntangledHVec) -> f64 {
        let sum: f64 = query
            .indices()
            .iter()
            .map(|&idx| self.counts[idx as usize])
            .sum();
        let raw = sum / self.n_blocks as f64;

        let total_counts: f64 = self.counts.iter().sum();
        let effective_n = total_counts / self.n_blocks as f64;
        let expected = effective_n * self.n_blocks as f64 / self.counts.len() as f64;
        raw - expected
    }
}

// ---------------------------------------------------------------------------
// Normalized Counting Bloom (cosine similarity)
// ---------------------------------------------------------------------------

struct NormalizedCountingBloom {
    counts: Vec<u32>,
    n_items: usize,
    n_blocks: usize,
}

impl NormalizedCountingBloom {
    fn new(dim: usize, n_blocks: usize) -> Self {
        Self {
            counts: vec![0u32; dim],
            n_items: 0,
            n_blocks,
        }
    }

    fn add(&mut self, item: &EntangledHVec) {
        for &idx in item.indices() {
            self.counts[idx as usize] += 1;
        }
        self.n_items += 1;
    }

    /// Cosine similarity between query (binary) and count vector.
    /// cos(q, c) = (sum of counts at query indices) / (|q| * ||c||)
    /// where |q| = sqrt(n_blocks) and ||c|| = sqrt(sum of squared counts).
    ///
    /// Then subtract expected cosine for a random query.
    fn score(&self, query: &EntangledHVec) -> f64 {
        let dot: u64 = query
            .indices()
            .iter()
            .map(|&idx| self.counts[idx as usize] as u64)
            .sum();

        let norm_sq: u64 = self.counts.iter().map(|&c| (c as u64) * (c as u64)).sum();
        if norm_sq == 0 {
            return 0.0;
        }

        let q_norm = (self.n_blocks as f64).sqrt();
        let c_norm = (norm_sq as f64).sqrt();

        let cosine = dot as f64 / (q_norm * c_norm);

        // Expected cosine for random query: E[dot] = n_blocks * (sum_counts / dim)
        // so E[cos] = n_blocks * (sum_counts / dim) / (sqrt(n_blocks) * c_norm)
        //           = sqrt(n_blocks) * sum_counts / (dim * c_norm)
        let sum_counts: u64 = self.counts.iter().map(|&c| c as u64).sum();
        let expected = q_norm * sum_counts as f64 / (self.counts.len() as f64 * c_norm);

        cosine - expected
    }
}

// ---------------------------------------------------------------------------
// Measurement harness
// ---------------------------------------------------------------------------

const LOAD_POINTS: &[usize] = &[
    10, 20, 50, 100, 200, 500, 700, 1000, 2000, 5000, 10000, 20000, 50000,
];

fn mean(vals: &[f64]) -> f64 {
    vals.iter().sum::<f64>() / vals.len() as f64
}
fn fmin(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn fmax(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn measure_bloom(items: &[EntangledHVec], non_members: &[EntangledHVec], dim: usize) {
    println!("# BLOOM BASELINE dim={}", dim);

    for &n_items in LOAD_POINTS {
        if n_items > items.len() {
            break;
        }
        let bundle = EntangledHVec::bundle_bloom(&items[..n_items]);
        let density = bundle.indices().len() as f64 / dim as f64;

        let member_sims: Vec<f64> = items[..n_items]
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .collect();
        let member_mean = mean(&member_sims);
        let member_min = fmin(&member_sims);

        let nonmember_sims: Vec<f64> = non_members
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .collect();
        let nonmember_mean = mean(&nonmember_sims);
        let nonmember_max = fmax(&nonmember_sims);

        println!(
            "bloom\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            n_items,
            member_mean,
            member_min,
            nonmember_mean,
            nonmember_max,
            member_min - nonmember_max
        );

        // Stop if bundle is saturated
        if density > 0.995 {
            break;
        }
    }
}

fn measure_counting(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    dim: usize,
    n_blocks: usize,
) {
    println!("# COUNTING BLOOM dim={}", dim);

    let mut bundle = CountingBloom::new(dim, n_blocks);
    let mut next_point = 0;

    for i in 0..items.len() {
        bundle.add(&items[i]);

        if next_point < LOAD_POINTS.len() && (i + 1) == LOAD_POINTS[next_point] {
            let n_items = i + 1;

            let member_scores: Vec<f64> = items[..n_items]
                .iter()
                .map(|item| bundle.score(item))
                .collect();
            let nonmember_scores: Vec<f64> =
                non_members.iter().map(|item| bundle.score(item)).collect();

            let member_mean = mean(&member_scores);
            let member_min = fmin(&member_scores);
            let nonmember_mean = mean(&nonmember_scores);
            let nonmember_max = fmax(&nonmember_scores);

            println!(
                "counting\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                n_items,
                member_mean,
                member_min,
                nonmember_mean,
                nonmember_max,
                member_min - nonmember_max
            );

            next_point += 1;
        }
    }
}

fn measure_decay_counting(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    dim: usize,
    n_blocks: usize,
    decay_factor: f64,
    consolidation_interval: usize,
) {
    println!(
        "# DECAY COUNTING dim={} decay={} interval={}",
        dim, decay_factor, consolidation_interval
    );

    let mut bundle = DecayCountingBloom::new(dim, n_blocks, decay_factor, consolidation_interval);
    let mut next_point = 0;

    for i in 0..items.len() {
        bundle.add(&items[i]);

        if next_point < LOAD_POINTS.len() && (i + 1) == LOAD_POINTS[next_point] {
            let n_items = i + 1;

            let member_scores: Vec<f64> = items[..n_items]
                .iter()
                .map(|item| bundle.score(item))
                .collect();
            let nonmember_scores: Vec<f64> =
                non_members.iter().map(|item| bundle.score(item)).collect();

            let member_mean = mean(&member_scores);
            let member_min = fmin(&member_scores);
            let nonmember_mean = mean(&nonmember_scores);
            let nonmember_max = fmax(&nonmember_scores);

            println!(
                "decay\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                n_items,
                member_mean,
                member_min,
                nonmember_mean,
                nonmember_max,
                member_min - nonmember_max
            );

            next_point += 1;
        }
    }
}

fn measure_normalized(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    dim: usize,
    n_blocks: usize,
) {
    println!("# NORMALIZED COUNTING dim={}", dim);

    let mut bundle = NormalizedCountingBloom::new(dim, n_blocks);
    let mut next_point = 0;

    for i in 0..items.len() {
        bundle.add(&items[i]);

        if next_point < LOAD_POINTS.len() && (i + 1) == LOAD_POINTS[next_point] {
            let n_items = i + 1;

            let member_scores: Vec<f64> = items[..n_items]
                .iter()
                .map(|item| bundle.score(item))
                .collect();
            let nonmember_scores: Vec<f64> =
                non_members.iter().map(|item| bundle.score(item)).collect();

            let member_mean = mean(&member_scores);
            let member_min = fmin(&member_scores);
            let nonmember_mean = mean(&nonmember_scores);
            let nonmember_max = fmax(&nonmember_scores);

            println!(
                "normalized\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                n_items,
                member_mean,
                member_min,
                nonmember_mean,
                nonmember_max,
                member_min - nonmember_max
            );

            next_point += 1;
        }
    }
}

fn main() {
    let dim = 16384;
    let density_denom = 256;
    let n_blocks = dim / density_denom; // 64 active indices per item
    let max_items = 50000;
    let n_probes = 200;

    println!(
        "# Counting experiments: D={} denom={} n_blocks={}",
        dim, density_denom, n_blocks
    );
    println!(
        "# Generating {} items + {} non-member probes...",
        max_items, n_probes
    );
    println!("# gap = member_min - nonmember_max (>0 = perfect separation)");
    println!();

    // Pre-generate all items and non-members (shared across experiments)
    let items: Vec<EntangledHVec> = (0..max_items)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, i as u64 * 37 + 1))
        .collect();
    let non_members: Vec<EntangledHVec> = (0..n_probes)
        .map(|i| {
            EntangledHVec::new_with_density(dim, density_denom, (max_items + i) as u64 * 37 + 9999)
        })
        .collect();

    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    // 1. Bloom baseline
    measure_bloom(&items, &non_members, dim);
    println!();

    // 2. Counting Bloom
    measure_counting(&items, &non_members, dim, n_blocks);
    println!();

    // 3. Exponential Decay Counting
    measure_decay_counting(&items, &non_members, dim, n_blocks, 0.9, 100);
    println!();

    // 4. Normalized Counting
    measure_normalized(&items, &non_members, dim, n_blocks);
    println!();

    // Summary: find zero-crossing points
    println!("# SUMMARY: Zero-crossing analysis");
    println!("# (Re-running to find exact crossing points)");
    println!();

    // Re-run each to find crossing points
    let bloom_cross = find_crossing_bloom(&items, &non_members, dim);
    let counting_cross = find_crossing_counting(&items, &non_members, dim, n_blocks);
    let decay_cross = find_crossing_decay(&items, &non_members, dim, n_blocks, 0.9, 100);
    let norm_cross = find_crossing_normalized(&items, &non_members, dim, n_blocks);

    println!("Scheme\t\t\tCrossing_n\tMultiplier_vs_Bloom");
    println!("bloom\t\t\t{}\t\t{:.1}x", bloom_cross, 1.0);
    println!(
        "counting\t\t{}\t\t{:.1}x",
        counting_cross,
        if bloom_cross > 0 {
            counting_cross as f64 / bloom_cross as f64
        } else {
            f64::INFINITY
        }
    );
    println!(
        "decay(0.9/100)\t\t{}\t\t{:.1}x",
        decay_cross,
        if bloom_cross > 0 {
            decay_cross as f64 / bloom_cross as f64
        } else {
            f64::INFINITY
        }
    );
    println!(
        "normalized\t\t{}\t\t{:.1}x",
        norm_cross,
        if bloom_cross > 0 {
            norm_cross as f64 / bloom_cross as f64
        } else {
            f64::INFINITY
        }
    );
}

// ---------------------------------------------------------------------------
// Zero-crossing finders (scan load points for sign change)
// ---------------------------------------------------------------------------

fn find_crossing_bloom(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    dim: usize,
) -> usize {
    for &n_items in LOAD_POINTS {
        if n_items > items.len() {
            break;
        }
        let bundle = EntangledHVec::bundle_bloom(&items[..n_items]);
        let density = bundle.indices().len() as f64 / dim as f64;

        let member_min = items[..n_items]
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .fold(f64::INFINITY, f64::min);

        let nonmember_max = non_members
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .fold(f64::NEG_INFINITY, f64::max);

        if member_min - nonmember_max <= 0.0 || density > 0.995 {
            return n_items;
        }
    }
    // Never crossed within tested range
    *LOAD_POINTS.last().unwrap_or(&0)
}

fn find_crossing_counting(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    dim: usize,
    n_blocks: usize,
) -> usize {
    let mut bundle = CountingBloom::new(dim, n_blocks);
    let mut next_point = 0;

    for i in 0..items.len() {
        bundle.add(&items[i]);

        if next_point < LOAD_POINTS.len() && (i + 1) == LOAD_POINTS[next_point] {
            let n_items = i + 1;

            let member_min = items[..n_items]
                .iter()
                .map(|item| bundle.score(item))
                .fold(f64::INFINITY, f64::min);
            let nonmember_max = non_members
                .iter()
                .map(|item| bundle.score(item))
                .fold(f64::NEG_INFINITY, f64::max);

            if member_min - nonmember_max <= 0.0 {
                return n_items;
            }
            next_point += 1;
        }
    }
    // Never crossed: report beyond max as indicator
    *LOAD_POINTS.last().unwrap_or(&0)
}

fn find_crossing_decay(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    dim: usize,
    n_blocks: usize,
    decay_factor: f64,
    consolidation_interval: usize,
) -> usize {
    let mut bundle = DecayCountingBloom::new(dim, n_blocks, decay_factor, consolidation_interval);
    let mut next_point = 0;

    for i in 0..items.len() {
        bundle.add(&items[i]);

        if next_point < LOAD_POINTS.len() && (i + 1) == LOAD_POINTS[next_point] {
            let n_items = i + 1;

            let member_min = items[..n_items]
                .iter()
                .map(|item| bundle.score(item))
                .fold(f64::INFINITY, f64::min);
            let nonmember_max = non_members
                .iter()
                .map(|item| bundle.score(item))
                .fold(f64::NEG_INFINITY, f64::max);

            if member_min - nonmember_max <= 0.0 {
                return n_items;
            }
            next_point += 1;
        }
    }
    *LOAD_POINTS.last().unwrap_or(&0)
}

fn find_crossing_normalized(
    items: &[EntangledHVec],
    non_members: &[EntangledHVec],
    dim: usize,
    n_blocks: usize,
) -> usize {
    let mut bundle = NormalizedCountingBloom::new(dim, n_blocks);
    let mut next_point = 0;

    for i in 0..items.len() {
        bundle.add(&items[i]);

        if next_point < LOAD_POINTS.len() && (i + 1) == LOAD_POINTS[next_point] {
            let n_items = i + 1;

            let member_min = items[..n_items]
                .iter()
                .map(|item| bundle.score(item))
                .fold(f64::INFINITY, f64::min);
            let nonmember_max = non_members
                .iter()
                .map(|item| bundle.score(item))
                .fold(f64::NEG_INFINITY, f64::max);

            if member_min - nonmember_max <= 0.0 {
                return n_items;
            }
            next_point += 1;
        }
    }
    *LOAD_POINTS.last().unwrap_or(&0)
}
