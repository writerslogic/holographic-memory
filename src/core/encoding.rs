// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::entangled::EntangledHVec;

/// Character trigrams (n=3) for text encoding.
///
/// Captures character-level morphological similarity (e.g., "cat" and "cats"
/// share 2 of 3 trigrams). Does NOT capture word-level semantics — synonyms
/// like "cat" and "feline" produce unrelated vectors.
const NGRAM_SIZE: usize = 3;

pub fn encode_text_internal(text: &str, dim: usize) -> EntangledHVec {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return EntangledHVec::from_indices(vec![], dim);
    }
    if chars.len() < NGRAM_SIZE {
        let vecs: Vec<EntangledHVec> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| EntangledHVec::new_deterministic(dim, *c as u64).permute(i))
            .collect();
        return EntangledHVec::bundle(&vecs);
    }
    let ngrams: Vec<EntangledHVec> = chars
        .windows(NGRAM_SIZE)
        .map(|window| {
            let mut chunk = EntangledHVec::new_deterministic(dim, window[0] as u64).permute(0);
            for (i, c) in window.iter().enumerate().skip(1) {
                let next = EntangledHVec::new_deterministic(dim, *c as u64).permute(i);
                chunk = chunk.bind(&next);
            }
            chunk
        })
        .collect();
    EntangledHVec::bundle(&ngrams)
}
