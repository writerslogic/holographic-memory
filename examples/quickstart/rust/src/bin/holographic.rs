// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The "holographic" part in 40 lines: bind roles to fillers, superpose them in one vector,
//! recall by unbinding, and damage the vector to watch recall degrade instead of fail.
//! Run: `cargo run --release --manifest-path examples/quickstart/rust/Cargo.toml --bin holographic`

use holographic_memory::EntangledHVec;

const DIM: usize = 16_384;

fn atom(name: &str) -> EntangledHVec {
    let seed = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    });
    EntangledHVec::new_deterministic(DIM, seed)
}

fn main() {
    let roles = ["capital", "language", "currency", "continent", "river"];
    let fillers = ["paris", "french", "euro", "europe", "seine"];
    let codebook: Vec<(&str, EntangledHVec)> = fillers.iter().map(|f| (*f, atom(f))).collect();

    // Superpose five role-filler bindings into ONE vector about "france".
    let bound: Vec<EntangledHVec> = roles
        .iter()
        .zip(&fillers)
        .map(|(r, f)| atom(r).bind(&atom(f)))
        .collect();
    let france = EntangledHVec::bundle(&bound);

    // Recall by unbinding a role and finding the nearest known filler.
    let recall = |memory: &EntangledHVec, role: &str| {
        let noisy = memory.bind(&atom(role));
        codebook
            .iter()
            .max_by(|a, b| a.1.similarity(&noisy).total_cmp(&b.1.similarity(&noisy)))
            .map(|(name, _)| *name)
            .unwrap_or("-")
    };
    for (role, filler) in roles.iter().zip(&fillers) {
        let got = recall(&france, role);
        println!(
            "france.{role:<9} -> {got:<7} {}",
            if got == *filler { "ok" } else { "WRONG" }
        );
        assert_eq!(got, *filler);
    }

    // Damage: drop a growing share of the superposed vector's active bits. Recall degrades
    // gradually; no single bit held any one fact.
    let bits: Vec<u32> = france.indices().to_vec();
    println!("\nbits kept  recalled of {}", roles.len());
    for keep_percent in [100usize, 75, 50, 35, 25] {
        let kept: Vec<u32> = bits
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| i * 100 / bits.len() < keep_percent)
            .map(|(_, b)| b)
            .collect();
        let damaged = EntangledHVec::from_indices(kept, DIM);
        let right = roles
            .iter()
            .zip(&fillers)
            .filter(|(r, f)| recall(&damaged, r) == **f)
            .count();
        println!("{keep_percent:>8}%  {right}");
    }
}
