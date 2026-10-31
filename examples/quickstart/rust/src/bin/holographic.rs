// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The "holographic" part in a few lines: bind a role to a filler, superpose many items in ONE
//! vector, test membership, and damage the vector to watch the signal degrade gradually.
//! Run: `cargo run --release --manifest-path examples/quickstart/rust/Cargo.toml --bin holographic`

use writerslogic_warm::EntangledHVec;

const DIM: usize = 16_384;
const ITEMS: u64 = 100;

fn atom(seed: u64) -> EntangledHVec {
    EntangledHVec::new_deterministic(DIM, seed)
}

/// Share of `item`'s active bits that are set in `memory` (1.0 = fully present).
fn presence(memory: &EntangledHVec, item: &EntangledHVec) -> f64 {
    let hits = item
        .indices()
        .iter()
        .filter(|bit| memory.indices().binary_search(bit).is_ok())
        .count();
    hits as f64 / item.indices().len() as f64
}

fn main() {
    // Bind is its own inverse: unbinding a role from role (x) filler returns the filler.
    let (role, filler) = (atom(1), atom(2));
    let bound = role.bind(&filler);
    assert_eq!(role.bind(&bound).indices(), filler.indices());
    println!("bind(role, filler) unbound with role recovers the filler exactly");

    // Superpose {ITEMS} atoms into ONE fixed-size vector (a union bundle).
    let members: Vec<EntangledHVec> = (100..100 + ITEMS).map(atom).collect();
    let strangers: Vec<EntangledHVec> = (10_000..10_000 + ITEMS).map(atom).collect();
    let memory = EntangledHVec::bundle_bloom(&members);
    println!(
        "{ITEMS} items in one vector of {} active bits out of {DIM}\n",
        memory.indices().len()
    );

    // Damage: keep only a share of the bundle's bits. No single bit held any one item, so every
    // member degrades together and gradually; strangers stay near the noise floor.
    println!("bits kept  members (mean presence)  strangers (mean presence)");
    let bits = memory.indices().to_vec();
    for keep_percent in [100usize, 75, 50, 25, 10] {
        let kept: Vec<u32> = bits
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| (i * 7919) % 100 < keep_percent)
            .map(|(_, bit)| bit)
            .collect();
        let damaged = EntangledHVec::from_indices(kept, DIM);
        let mean = |set: &[EntangledHVec]| {
            set.iter().map(|v| presence(&damaged, v)).sum::<f64>() / set.len() as f64
        };
        println!(
            "{keep_percent:>8}%  {:>22.2}  {:>25.2}",
            mean(&members),
            mean(&strangers)
        );
    }
}
