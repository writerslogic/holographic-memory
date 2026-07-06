// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// §19. The Bloom membership store (core::bloom_memory) bundles by OR set-union and
// reads out with density-corrected containment. Members score EXACTLY 1.0 (all k
// indices are in the union by construction), so the discrimination collapse is
// purely non-members' scores rising toward 1.0 as the union saturates (false
// positives ~ d^k, exploding as density d->1). OR-union discards COUNT information.
//
// This pits the incumbent binary readout against a counting bundle + Poisson
// z-score readout on the SAME inserted set, measured by distribution-free AUC
// (Mann-Whitney), and locates each readout's membership wall.
//
// Run: cargo run --release --bin bloom-wall

const D: usize = 16384;
const K: usize = 64; // active indices per item (denom 256)
const SEEDS: u64 = 20;
const PROBE: usize = 200; // members / non-members sampled for AUC

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Deterministic k-subset of [0, D) for item `id` (rare collisions accepted; they
/// only slightly lower the effective k, identically for both readouts).
fn indices(id: u64, seed: u64) -> Vec<usize> {
    (0..K)
        .map(|j| {
            (mix(id.wrapping_mul(0x100_0001).wrapping_add(j as u64), seed) % D as u64) as usize
        })
        .collect()
}

/// AUC via Mann-Whitney: P(member score > non-member score), ties = 0.5.
fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    let mut wins = 0.0f64;
    for &p in pos {
        for &n in neg {
            wins += if p > n {
                1.0
            } else if p == n {
                0.5
            } else {
                0.0
            };
        }
    }
    wins / (pos.len() as f64 * neg.len() as f64)
}

/// Returns (binary_auc, counting_auc) averaged over seeds at load n.
fn eval(n: usize) -> (f64, f64) {
    let (mut bin_sum, mut cnt_sum) = (0.0f64, 0.0f64);
    for seed in 0..SEEDS {
        let mut count = vec![0u32; D];
        for id in 0..n as u64 {
            for &idx in &indices(id, seed) {
                count[idx] += 1;
            }
        }
        let union_size = count.iter().filter(|&&c| c > 0).count();
        let d = union_size as f64 / D as f64;
        let lambda = n as f64 * K as f64 / D as f64;

        // corrected containment over the OR union (binary readout)
        let binary = |ix: &[usize]| -> f64 {
            let present = ix.iter().filter(|&&i| count[i] > 0).count() as f64 / K as f64;
            if d >= 1.0 {
                0.0
            } else {
                (present - d) / (1.0 - d)
            }
        };
        // Poisson z-sum over active indices (counting readout)
        let counting = |ix: &[usize]| -> f64 {
            let s: f64 = ix.iter().map(|&i| count[i] as f64 - lambda).sum();
            s / lambda.max(1e-9).sqrt()
        };

        let members: Vec<Vec<usize>> = (0..PROBE.min(n) as u64)
            .map(|id| indices(id, seed))
            .collect();
        let nonmembers: Vec<Vec<usize>> = (0..PROBE as u64)
            .map(|j| indices(9_000_000 + j, seed))
            .collect();

        let bp: Vec<f64> = members.iter().map(|ix| binary(ix)).collect();
        let bn: Vec<f64> = nonmembers.iter().map(|ix| binary(ix)).collect();
        let cp: Vec<f64> = members.iter().map(|ix| counting(ix)).collect();
        let cn: Vec<f64> = nonmembers.iter().map(|ix| counting(ix)).collect();
        bin_sum += auc(&bp, &bn);
        cnt_sum += auc(&cp, &cn);
    }
    (bin_sum / SEEDS as f64, cnt_sum / SEEDS as f64)
}

fn main() {
    let loads = [
        100usize, 200, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000, 8000,
    ];
    println!("bloom-wall | D={D} k={K} seeds={SEEDS} | AUC member vs non-member (chance 0.50)");
    println!("{:>6}  {:>12}  {:>12}", "n", "binary AUC", "counting AUC");
    let (mut bwall99, mut bwall95, mut cwall99, mut cwall95) = (0usize, 0usize, 0usize, 0usize);
    for n in loads {
        let (b, c) = eval(n);
        if bwall99 == 0 && b < 0.99 {
            bwall99 = n;
        }
        if bwall95 == 0 && b < 0.95 {
            bwall95 = n;
        }
        if cwall99 == 0 && c < 0.99 {
            cwall99 = n;
        }
        if cwall95 == 0 && c < 0.95 {
            cwall95 = n;
        }
        println!("{n:>6}  {b:>11.4}  {c:>11.4}");
    }
    let wall = |n: usize| {
        if n == 0 {
            ">8000".to_string()
        } else {
            n.to_string()
        }
    };
    println!(
        "\nwall (first n below AUC): binary <0.99 @ {}, <0.95 @ {}",
        wall(bwall99),
        wall(bwall95)
    );
    println!(
        "                        counting <0.99 @ {}, <0.95 @ {}",
        wall(cwall99),
        wall(cwall95)
    );
    println!("Kill: counting <0.95 wall <= 1.2x binary -> counts don't help; ceiling is");
    println!("field saturation, fix is more D / sharding, not a smarter readout.");
}
