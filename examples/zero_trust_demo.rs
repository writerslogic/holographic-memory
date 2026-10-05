// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Client-side vector masking: a store ranks masked vectors without seeing the
//! original coordinates. This is keyed obfuscation, not encryption; the store
//! still learns every pairwise similarity. See `core::mask` for the limits.

use holographic_memory::core::entangled::EntangledHVec;
use holographic_memory::core::mask::VectorMask;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const DIM: usize = 16384;

/// Toy lexical encoder: the union of one deterministic vector per lowercase
/// word, so texts sharing words share active coordinates.
fn encode(text: &str) -> EntangledHVec {
    let mut indices: Vec<u32> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .flat_map(|w| {
            let mut hasher = DefaultHasher::new();
            w.to_lowercase().hash(&mut hasher);
            EntangledHVec::new_deterministic(DIM, hasher.finish())
                .indices()
                .to_vec()
        })
        .collect();
    indices.sort_unstable();
    indices.dedup();
    EntangledHVec::from_indices(indices, DIM)
}

fn main() -> anyhow::Result<()> {
    // The passphrase stays on the client. The salt is public and shared by
    // every client of the collection.
    let mask = VectorMask::derive(b"client passphrase", b"collection-salt-01", DIM)?;

    let docs = [
        "The Q3 financial earnings were surprisingly high due to the merger.",
        "Operation Midnight will commence on Tuesday at 0400 hours.",
        "Patient 402 has a history of severe allergic reactions to penicillin.",
    ];
    let store: Vec<EntangledHVec> = docs
        .iter()
        .map(|text| mask.apply(&encode(text)))
        .collect::<anyhow::Result<_>>()?;

    let query = encode("What were the financial earnings?");
    let masked_query = mask.apply(&query)?;

    // The store ranks using masked vectors only.
    let (best, score) = store
        .iter()
        .map(|v| masked_query.similarity(v))
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("store is not empty");

    let plain_score = query.similarity(&encode(docs[best]));
    println!("best match: {:?}", docs[best]);
    println!("masked similarity {score:.4}, plaintext similarity {plain_score:.4}");
    println!(
        "masked vector overlap with its plaintext: {:.4}",
        store[best].similarity(&encode(docs[best]))
    );
    anyhow::ensure!(best == 0, "expected the earnings document to rank first");
    anyhow::ensure!(score == plain_score, "masking must preserve similarity");
    Ok(())
}
