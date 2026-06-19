// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::entangled::EntangledHVec;
use fxhash::FxHasher;
use std::hash::Hasher;

/// Multi-scale text encoding combining character n-grams, word tokens, and
/// word bigrams into a single sparse hypervector.
///
/// Architecture (based on Rahimi et al. 2016, Joshi et al. 2020):
///
/// 1. **Character n-grams** at scales 2-5 capture morphological similarity.
///    "cat" and "cats" share most bigrams/trigrams; different n-gram sizes
///    provide redundancy and cover different morphological features.
///
/// 2. **Word tokens** with positional binding capture lexical identity.
///    Each word is hashed as a unit and bound with its sentence position,
///    so word order is partially preserved.
///
/// 3. **Word bigrams** capture local phrase structure.
///    Adjacent word pairs bound together represent collocations.
///
/// Each level uses a unique seed space (LEVEL_CHAR, LEVEL_WORD, LEVEL_PHRASE)
/// to prevent cross-level hash collisions. Levels are bundled with
/// empirically-tuned weights favoring word identity (highest signal-to-noise
/// ratio for retrieval tasks).
///
/// Breaking change: vectors produced by this encoder are incompatible with
/// the previous character-trigram-only encoder. Re-encode stored data.

const CHAR_NGRAM_SIZES: [usize; 4] = [2, 3, 4, 5];

/// Level seed offsets prevent cross-level hash collisions.
const LEVEL_CHAR: u64 = 0x01;
const LEVEL_WORD: u64 = 0x02;
const LEVEL_PHRASE: u64 = 0x03;

pub fn encode_text_internal(text: &str, dim: usize) -> EntangledHVec {
    let text = text.trim();
    if text.is_empty() {
        return EntangledHVec::from_indices(vec![], dim);
    }

    let chars: Vec<char> = text.chars().collect();
    let words: Vec<&str> = text.split_whitespace().collect();

    // Bundle within each level first, then equal-weight bundle across levels.
    // This prevents any single level from dominating majority-vote thresholds.
    let mut level_bundles: Vec<EntangledHVec> = Vec::new();

    // --- Character n-grams at multiple scales ---
    for &n in &CHAR_NGRAM_SIZES {
        if chars.len() < n {
            continue;
        }
        let ngrams: Vec<EntangledHVec> = chars
            .windows(n)
            .map(|window| {
                let mut chunk = EntangledHVec::new_deterministic(
                    dim,
                    seeded(LEVEL_CHAR, n as u64, window[0] as u64),
                )
                .permute(0);
                for (i, &c) in window.iter().enumerate().skip(1) {
                    let next = EntangledHVec::new_deterministic(
                        dim,
                        seeded(LEVEL_CHAR, n as u64, c as u64),
                    )
                    .permute(i);
                    chunk = chunk.bind(&next);
                }
                chunk
            })
            .collect();
        level_bundles.push(EntangledHVec::bundle(&ngrams));
    }

    // --- Word-level encoding with positional binding ---
    if !words.is_empty() {
        let word_vecs: Vec<EntangledHVec> = words
            .iter()
            .enumerate()
            .map(|(pos, word)| {
                let word_hash = hash_str(word, LEVEL_WORD);
                EntangledHVec::new_deterministic(dim, word_hash).permute(pos)
            })
            .collect();
        level_bundles.push(EntangledHVec::bundle(&word_vecs));
    }

    // --- Word bigrams for phrase structure ---
    if words.len() >= 2 {
        let phrase_vecs: Vec<EntangledHVec> = words
            .windows(2)
            .enumerate()
            .map(|(pos, pair)| {
                let h0 = hash_str(pair[0], LEVEL_PHRASE);
                let h1 = hash_str(pair[1], LEVEL_PHRASE);
                let v0 = EntangledHVec::new_deterministic(dim, h0).permute(0);
                let v1 = EntangledHVec::new_deterministic(dim, h1).permute(1);
                v0.bind(&v1).permute(pos)
            })
            .collect();
        level_bundles.push(EntangledHVec::bundle(&phrase_vecs));
    }

    if level_bundles.is_empty() {
        return EntangledHVec::new_deterministic(dim, seeded(LEVEL_CHAR, 1, chars[0] as u64));
    }

    // Combine levels by union. Majority-vote bundling fails across levels because
    // different seed spaces produce non-overlapping index sets (threshold is never met).
    // Union preserves all information; the denser result (~N*64 indices) improves
    // discrimination and Jaccard similarity handles arbitrary set sizes correctly.
    let mut all_indices: Vec<u32> = level_bundles
        .iter()
        .flat_map(|v| v.indices().iter().copied())
        .collect();
    all_indices.sort_unstable();
    all_indices.dedup();
    EntangledHVec::from_indices(all_indices, dim)
}

/// Mix level, scale, and item seeds to prevent cross-level collisions.
fn seeded(level: u64, scale: u64, item: u64) -> u64 {
    let mut h = FxHasher::default();
    h.write_u64(level);
    h.write_u64(scale);
    h.write_u64(item);
    h.finish()
}

/// Hash a string with a level prefix for deterministic word encoding.
fn hash_str(s: &str, level: u64) -> u64 {
    let mut h = FxHasher::default();
    h.write_u64(level);
    h.write(s.as_bytes());
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let dim = 16384;
        let v1 = encode_text_internal("hello world", dim);
        let v2 = encode_text_internal("hello world", dim);
        assert!(
            (v1.similarity(&v2) - 1.0).abs() < f64::EPSILON,
            "Same input must produce identical vectors"
        );
    }

    #[test]
    fn test_empty_input() {
        let v = encode_text_internal("", 16384);
        assert!(v.indices().is_empty());
    }

    #[test]
    fn test_single_char() {
        let v = encode_text_internal("x", 16384);
        assert!(!v.indices().is_empty(), "Single char should produce a vector");
    }

    #[test]
    fn test_morphological_similarity() {
        let dim = 16384;
        let cat = encode_text_internal("cat", dim);
        let cats = encode_text_internal("cats", dim);
        let dog = encode_text_internal("dog", dim);
        let sim_cat_cats = cat.similarity(&cats);
        let sim_cat_dog = cat.similarity(&dog);
        assert!(
            sim_cat_cats > sim_cat_dog,
            "cat/cats ({:.4}) should be more similar than cat/dog ({:.4})",
            sim_cat_cats,
            sim_cat_dog
        );
    }

    #[test]
    fn test_word_identity_matters() {
        let dim = 16384;
        let v1 = encode_text_internal("the quick brown fox", dim);
        let v2 = encode_text_internal("the quick brown fox", dim);
        let v3 = encode_text_internal("completely different words here", dim);
        assert!(
            (v1.similarity(&v2) - 1.0).abs() < f64::EPSILON,
            "Identical text must match"
        );
        assert!(
            v1.similarity(&v3) < 0.3,
            "Unrelated text should have low similarity"
        );
    }

    #[test]
    fn test_word_overlap_similarity() {
        let dim = 16384;
        let v1 = encode_text_internal("the cat sat on the mat", dim);
        let v2 = encode_text_internal("the dog sat on the mat", dim);
        let v3 = encode_text_internal("quantum physics is fascinating", dim);
        let sim_close = v1.similarity(&v2);
        let sim_far = v1.similarity(&v3);
        assert!(
            sim_close > sim_far,
            "Sentences sharing most words ({:.4}) should be more similar than unrelated ({:.4})",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_multi_scale_captures_short_text() {
        let dim = 16384;
        // Two-character text should still produce a meaningful vector
        let v = encode_text_internal("hi", dim);
        assert!(!v.indices().is_empty());
        // Should be similar to "his" (shared bigram "hi")
        let v2 = encode_text_internal("his", dim);
        let v3 = encode_text_internal("xyz", dim);
        assert!(
            v.similarity(&v2) > v.similarity(&v3),
            "hi/his should be more similar than hi/xyz"
        );
    }
}
