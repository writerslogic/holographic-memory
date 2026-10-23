// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Text encoders: text in, sparse hypervector out.
//!
//! `multiscale-words-v1` is specified by `encode_reference`: character n-grams at scales
//! 2-5 (one hypervector per character, permuted by its offset in the window and bound by
//! symmetric difference), word tokens permuted by position, and word bigrams, each level
//! bundled by majority vote, the levels combined by cross-level support. That function is the
//! contract (`docs/file-format/ENCODERS.md`: same text and dimension, same indices forever) and
//! stays reachable for the equivalence test in `tests/ingest_encode.rs`.
//!
//! `TextEncoder::encode` computes the same function through per-thread memo tables and
//! reused scratch buffers, which is where ingest spends its time (`new_deterministic` per
//! character of every n-gram, a sort of every level's concatenated indices, and one heap
//! allocation per intermediate vector). It produces identical indices because:
//!
//! 1. `EntangledHVec::new_deterministic(dim, seed)` is a pure function of `(dim, seed)`, so a
//!    base vector is memoized by seed (`EncodeState::terms`); two words with equal seeds share
//!    a vector in the reference as well, so keying by seed adds no collision.
//! 2. `permute(shift)` maps a sorted index set through `(idx + shift) % dim`, a bijection; the
//!    sorted result is the rotation `rotate_sorted` computes without sorting, and a multiset
//!    that is only counted needs no order at all.
//! 3. `bind` is the symmetric difference of two sorted sets: set-valued, so any correct
//!    implementation yields the same indices. A character n-gram's bound vector depends only on
//!    `(scale, characters)` and is memoized by that exact key (`EncodeState::ngrams`).
//! 4. `EntangledHVec::bundle` depends on its inputs only through the ascending run-length list
//!    `(index, count)` of the concatenated indices. `EncodeState::bundle_flat` builds the same
//!    list from a histogram over `[0, dim)` (or by sorting when `dim` dwarfs the index total)
//!    and then calls the identical `select_nth_unstable_by` on the identical slice; that call's
//!    comparator has ties, so the retained set depends on input order and algorithm, both kept.
//! 5. The cross-level selection orders by `(count desc, index asc)`, a total order, so it is a
//!    function of the level bundles alone.
//!
//! The memo tables are a per-thread cache with a fixed budget (cleared when exceeded), never
//! an input to the result: output is independent of thread, cache state and call history.

use super::entangled::{hash_u64, EntangledHVec, DEFAULT_RHO_DENOM};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextEncoder {
    #[default]
    #[serde(rename = "multiscale-words-v1")]
    MultiscaleWordsV1,
    /// Experimental union within each level, capped by cross-level frequency.
    #[serde(rename = "multiscale-capped-union-v1")]
    MultiscaleCappedUnionV1,
    /// Sparse sketch of a local neural sentence embedding (feature `local-embed`); the store's
    /// documents also get dense chunk embeddings from the same model.
    #[serde(rename = "local-embed-v1")]
    LocalEmbedV1,
}

impl TextEncoder {
    pub fn name(self) -> &'static str {
        match self {
            Self::MultiscaleWordsV1 => "multiscale-words-v1",
            Self::MultiscaleCappedUnionV1 => "multiscale-capped-union-v1",
            Self::LocalEmbedV1 => "local-embed-v1",
        }
    }

    /// Whether vectors come from a neural embedding instead of the text itself.
    pub fn is_embedding(self) -> bool {
        self == Self::LocalEmbedV1
    }

    pub(crate) fn encode(self, text: &str, dim: usize) -> EntangledHVec {
        assert!(
            !self.is_embedding(),
            "embedding encoders are applied to embeddings by the engine, not to text"
        );
        ENCODE_STATE.with(|state| encode_cached(text, dim, self, &mut state.borrow_mut()))
    }

    /// The production path ([`Self::encode`]) for callers outside the crate, such as the
    /// equivalence test; not part of the stable API.
    #[doc(hidden)]
    pub fn encode_text(self, text: &str, dim: usize) -> EntangledHVec {
        self.encode(text, dim)
    }

    /// The reference formulation of the encoder: one `new_deterministic` per character of every
    /// n-gram and sorting bundles, exactly as before the memoized path existed. Kept so the
    /// equivalence test can assert bit-identical output; not part of the stable API.
    #[doc(hidden)]
    pub fn encode_reference(self, text: &str, dim: usize) -> EntangledHVec {
        assert!(
            !self.is_embedding(),
            "embedding encoders are applied to embeddings by the engine, not to text"
        );
        encode_reference(text, dim, self)
    }

    fn bundle(self, vectors: &[EntangledHVec]) -> EntangledHVec {
        match self {
            Self::MultiscaleWordsV1 | Self::LocalEmbedV1 => EntangledHVec::bundle(vectors),
            Self::MultiscaleCappedUnionV1 => EntangledHVec::bundle_bloom(vectors),
        }
    }

    /// Whether a level is bundled by set union (`bundle_bloom`) rather than majority vote.
    fn bundles_by_union(self) -> bool {
        self == Self::MultiscaleCappedUnionV1
    }
}

impl std::str::FromStr for TextEncoder {
    type Err = anyhow::Error;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "multiscale-words-v1" => Ok(Self::MultiscaleWordsV1),
            "multiscale-capped-union-v1" => Ok(Self::MultiscaleCappedUnionV1),
            "local-embed-v1" => Ok(Self::LocalEmbedV1),
            _ => anyhow::bail!("unknown text encoder {name:?}"),
        }
    }
}

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

/// The largest character n-gram scale; the memo key holds this many characters.
const MAX_CHAR_NGRAM: usize = 5;
const _: () = assert!(CHAR_NGRAM_SIZES[CHAR_NGRAM_SIZES.len() - 1] == MAX_CHAR_NGRAM);

/// Level seed offsets prevent cross-level hash collisions.
const LEVEL_CHAR: u64 = 0x01;
const LEVEL_WORD: u64 = 0x02;
const LEVEL_PHRASE: u64 = 0x03;

/// Per-thread budget, in `u32` slots, of memoized base vectors (words, bigram words and
/// characters, keyed by seed): 4 MiB. The table is cleared when a new entry would exceed it.
const TERM_ARENA_BUDGET: usize = 1 << 20;

/// Per-thread budget, in `u32` slots, of memoized character n-gram vectors: 8 MiB.
const NGRAM_ARENA_BUDGET: usize = 1 << 21;

/// A level is bundled through a histogram over `[0, dim)` when `dim` is at most this many
/// times the number of indices to count; sorting them is cheaper beyond that.
const COUNTING_RATIO: usize = 8;

pub fn encode_text_internal(text: &str, dim: usize) -> EntangledHVec {
    TextEncoder::MultiscaleWordsV1.encode(text, dim)
}

thread_local! {
    static ENCODE_STATE: RefCell<EncodeState> = RefCell::new(EncodeState::default());
}

/// A memoized vector's location in an arena.
#[derive(Clone, Copy)]
struct Span {
    start: usize,
    len: usize,
}

impl Span {
    fn range(self) -> std::ops::Range<usize> {
        self.start..self.start + self.len
    }
}

/// Exact identity of a character n-gram: its scale and characters (unused slots are NUL,
/// which no window at that scale can contain in those positions).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct NgramKey {
    n: u8,
    chars: [char; MAX_CHAR_NGRAM],
}

/// Memo tables and scratch buffers of one thread. Invariants: every span in `terms` and
/// `ngrams` lies inside its arena; both tables hold vectors for `dim` only; `counts` is
/// all-zero outside [`EncodeState::bundle_flat`]; scratch vectors carry no state between calls.
#[derive(Default)]
struct EncodeState {
    dim: usize,
    /// seed -> sorted indices of `EntangledHVec::new_deterministic(dim, seed)`.
    terms: fxhash::FxHashMap<u64, Span>,
    term_arena: Vec<u32>,
    /// n-gram -> sorted indices of its bound vector (reference lines: permute each character
    /// by its offset, bind left to right).
    ngrams: fxhash::FxHashMap<NgramKey, Span>,
    ngram_arena: Vec<u32>,
    chars: Vec<char>,
    /// The multiset of (already permuted) indices of every vector in the level being bundled.
    flat: Vec<u32>,
    acc: Vec<u32>,
    acc_next: Vec<u32>,
    rotated: Vec<u32>,
    counts: Vec<u32>,
    selected: Vec<(u32, u32)>,
}

impl EncodeState {
    /// Points the memo tables at `dim`, dropping vectors of another dimension.
    fn prepare(&mut self, dim: usize) {
        if self.dim != dim {
            self.dim = dim;
            self.terms.clear();
            self.term_arena.clear();
            self.ngrams.clear();
            self.ngram_arena.clear();
        }
    }

    /// The bound vector of the character n-gram `window` (length `n`), memoized.
    fn ngram_span(&mut self, n: usize, window: &[char]) -> Span {
        let mut key = NgramKey {
            n: n as u8,
            chars: ['\0'; MAX_CHAR_NGRAM],
        };
        key.chars[..n].copy_from_slice(window);
        if let Some(&span) = self.ngrams.get(&key) {
            return span;
        }
        let dim = self.dim;
        let first = term_span(
            &mut self.terms,
            &mut self.term_arena,
            dim,
            seeded(LEVEL_CHAR, n as u64, window[0] as u64),
        );
        // Copied before the next lookup, which may clear the term arena.
        self.acc.clear();
        self.acc.extend_from_slice(&self.term_arena[first.range()]);
        for (i, &c) in window.iter().enumerate().skip(1) {
            let span = term_span(
                &mut self.terms,
                &mut self.term_arena,
                dim,
                seeded(LEVEL_CHAR, n as u64, c as u64),
            );
            rotate_sorted(
                &self.term_arena[span.range()],
                (i % dim) as u32,
                dim as u32,
                &mut self.rotated,
            );
            symmetric_difference(&self.acc, &self.rotated, &mut self.acc_next);
            std::mem::swap(&mut self.acc, &mut self.acc_next);
        }
        if self.ngram_arena.len() + self.acc.len() > NGRAM_ARENA_BUDGET {
            self.ngrams.clear();
            self.ngram_arena.clear();
        }
        let span = Span {
            start: self.ngram_arena.len(),
            len: self.acc.len(),
        };
        self.ngram_arena.extend_from_slice(&self.acc);
        self.ngrams.insert(key, span);
        span
    }

    /// The bundle (`EntangledHVec::bundle`, or `bundle_bloom` for union encoders) of `count`
    /// vectors whose indices are concatenated in `self.flat`, in any order.
    fn bundle_flat(&mut self, count: usize, encoder: TextEncoder) -> EntangledHVec {
        let dim = self.dim;
        if self.flat.is_empty() {
            return EntangledHVec::from_indices(Vec::new(), dim);
        }
        let union = encoder.bundles_by_union();
        let threshold = if union { 1 } else { (count as u32).div_ceil(2) };
        self.selected.clear();
        if dim <= self.flat.len().saturating_mul(COUNTING_RATIO) {
            if self.counts.len() != dim {
                self.counts.clear();
                self.counts.resize(dim, 0);
            }
            debug_assert!(self.counts.iter().all(|&c| c == 0));
            for &idx in &self.flat {
                self.counts[idx as usize] += 1;
            }
            for (idx, slot) in self.counts.iter_mut().enumerate() {
                let n = std::mem::take(slot);
                if n >= threshold {
                    self.selected.push((idx as u32, n));
                }
            }
        } else {
            self.flat.sort_unstable();
            let mut current = self.flat[0];
            let mut n: u32 = 1;
            for &idx in &self.flat[1..] {
                if idx == current {
                    n += 1;
                } else {
                    if n >= threshold {
                        self.selected.push((current, n));
                    }
                    current = idx;
                    n = 1;
                }
            }
            if n >= threshold {
                self.selected.push((current, n));
            }
        }
        if union {
            let indices: Vec<u32> = self.selected.iter().map(|&(idx, _)| idx).collect();
            return EntangledHVec::from_indices(indices, dim);
        }
        let target_count = (dim / DEFAULT_RHO_DENOM).max(1);
        // Same call on the same slice as the reference: the comparator does not break ties,
        // so a different selection algorithm or input order could retain different indices.
        if self.selected.len() > target_count {
            self.selected
                .select_nth_unstable_by(target_count - 1, |a, b| b.1.cmp(&a.1));
            self.selected.truncate(target_count);
        }
        let mut indices: Vec<u32> = self.selected.iter().map(|&(idx, _)| idx).collect();
        indices.sort_unstable();
        EntangledHVec::from_indices(indices, dim)
    }
}

/// Sorted indices of `EntangledHVec::new_deterministic(dim, seed)`, memoized in `arena`.
/// The returned span is valid until the next call, which may clear the arena.
fn term_span(
    terms: &mut fxhash::FxHashMap<u64, Span>,
    arena: &mut Vec<u32>,
    dim: usize,
    seed: u64,
) -> Span {
    if let Some(&span) = terms.get(&seed) {
        return span;
    }
    let vector = EntangledHVec::new_deterministic(dim, seed);
    let indices = vector.indices();
    if arena.len() + indices.len() > TERM_ARENA_BUDGET {
        terms.clear();
        arena.clear();
    }
    let span = Span {
        start: arena.len(),
        len: indices.len(),
    };
    arena.extend_from_slice(indices);
    terms.insert(seed, span);
    span
}

/// `permute(shift)` of the sorted index set `base`: `(idx + shift) % dim` for each index, in
/// ascending order. Indices at or above `dim - shift` wrap to the front, so the result is a
/// rotation of the input and needs no sort. `shift` must be below `dim`.
fn rotate_sorted(base: &[u32], shift: u32, dim: u32, out: &mut Vec<u32>) {
    out.clear();
    if shift == 0 {
        out.extend_from_slice(base);
        return;
    }
    let wrap = dim - shift;
    let split = base.partition_point(|&idx| idx < wrap);
    out.extend(base[split..].iter().map(|&idx| idx - wrap));
    out.extend(base[..split].iter().map(|&idx| idx + shift));
}

/// Sorted symmetric difference of two sorted index sets (`EntangledHVec::bind`).
fn symmetric_difference(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// [`encode_reference`] computed through the thread's memo tables; see the module head for
/// why every step yields the same indices.
fn encode_cached(
    text: &str,
    dim: usize,
    encoder: TextEncoder,
    state: &mut EncodeState,
) -> EntangledHVec {
    let text = text.trim();
    if text.is_empty() {
        return EntangledHVec::from_indices(Vec::new(), dim);
    }
    state.prepare(dim);
    state.chars.clear();
    state.chars.extend(text.chars());
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut level_bundles: Vec<EntangledHVec> = Vec::with_capacity(CHAR_NGRAM_SIZES.len() + 2);

    // --- Character n-grams at multiple scales ---
    for &n in &CHAR_NGRAM_SIZES {
        let windows = match state.chars.len().checked_sub(n) {
            Some(rest) => rest + 1,
            None => continue,
        };
        state.flat.clear();
        for start in 0..windows {
            let mut window = ['\0'; MAX_CHAR_NGRAM];
            window[..n].copy_from_slice(&state.chars[start..start + n]);
            let span = state.ngram_span(n, &window[..n]);
            state
                .flat
                .extend_from_slice(&state.ngram_arena[span.range()]);
        }
        level_bundles.push(state.bundle_flat(windows, encoder));
    }

    let d = dim as u32;

    // --- Word-level encoding with positional binding ---
    if !words.is_empty() {
        state.flat.clear();
        for (pos, word) in words.iter().enumerate() {
            let span = term_span(
                &mut state.terms,
                &mut state.term_arena,
                dim,
                hash_str_seed(word, LEVEL_WORD),
            );
            let shift = (pos % dim) as u32;
            state.flat.extend(
                state.term_arena[span.range()]
                    .iter()
                    .map(|&idx| (idx + shift) % d),
            );
        }
        level_bundles.push(state.bundle_flat(words.len(), encoder));
    }

    // --- Word bigrams for phrase structure ---
    if words.len() >= 2 {
        state.flat.clear();
        for (pos, pair) in words.windows(2).enumerate() {
            let left = term_span(
                &mut state.terms,
                &mut state.term_arena,
                dim,
                hash_str_seed(pair[0], LEVEL_PHRASE),
            );
            // Copied before the next lookup, which may clear the term arena.
            state.acc_next.clear();
            state
                .acc_next
                .extend_from_slice(&state.term_arena[left.range()]);
            let right = term_span(
                &mut state.terms,
                &mut state.term_arena,
                dim,
                hash_str_seed(pair[1], LEVEL_PHRASE),
            );
            rotate_sorted(
                &state.term_arena[right.range()],
                (1 % dim) as u32,
                d,
                &mut state.rotated,
            );
            symmetric_difference(&state.acc_next, &state.rotated, &mut state.acc);
            let shift = (pos % dim) as u32;
            state
                .flat
                .extend(state.acc.iter().map(|&idx| (idx + shift) % d));
        }
        level_bundles.push(state.bundle_flat(words.len() - 1, encoder));
    }

    if level_bundles.is_empty() {
        return EntangledHVec::new_deterministic(dim, seeded(LEVEL_CHAR, 1, state.chars[0] as u64));
    }
    combine_levels(&level_bundles, dim)
}

/// The reference formulation; see [`TextEncoder::encode_reference`].
fn encode_reference(text: &str, dim: usize, encoder: TextEncoder) -> EntangledHVec {
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
        level_bundles.push(encoder.bundle(&ngrams));
    }

    // --- Word-level encoding with positional binding ---
    if !words.is_empty() {
        let word_vecs: Vec<EntangledHVec> = words
            .iter()
            .enumerate()
            .map(|(pos, word)| {
                let word_hash = hash_str_seed(word, LEVEL_WORD);
                EntangledHVec::new_deterministic(dim, word_hash).permute(pos)
            })
            .collect();
        level_bundles.push(encoder.bundle(&word_vecs));
    }

    // --- Word bigrams for phrase structure ---
    if words.len() >= 2 {
        let phrase_vecs: Vec<EntangledHVec> = words
            .windows(2)
            .enumerate()
            .map(|(pos, pair)| {
                let h0 = hash_str_seed(pair[0], LEVEL_PHRASE);
                let h1 = hash_str_seed(pair[1], LEVEL_PHRASE);
                let v0 = EntangledHVec::new_deterministic(dim, h0).permute(0);
                let v1 = EntangledHVec::new_deterministic(dim, h1).permute(1);
                v0.bind(&v1).permute(pos)
            })
            .collect();
        level_bundles.push(encoder.bundle(&phrase_vecs));
    }

    if level_bundles.is_empty() {
        return EntangledHVec::new_deterministic(dim, seeded(LEVEL_CHAR, 1, chars[0] as u64));
    }
    combine_levels(&level_bundles, dim)
}

/// Combine levels by frequency-weighted selection. Each index gets a count of how many level
/// bundles contain it; indices with cross-level support are prioritized. Output is
/// fixed-density (dim/256 active indices) to ensure consistent Jaccard comparisons across
/// different-length texts.
fn combine_levels(level_bundles: &[EntangledHVec], dim: usize) -> EntangledHVec {
    let active_count = (dim / DEFAULT_RHO_DENOM).max(1);
    let mut freq: fxhash::FxHashMap<u32, u32> =
        fxhash::FxHashMap::with_capacity_and_hasher(active_count * 4, Default::default());
    for bundle in level_bundles {
        for &idx in bundle.indices() {
            *freq.entry(idx).or_insert(0) += 1;
        }
    }

    let mut scored: Vec<(u32, u32)> = freq.into_iter().collect();
    if scored.len() > active_count {
        scored.select_nth_unstable_by(active_count - 1, |a, b| {
            b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(active_count);
    }

    let mut indices: Vec<u32> = scored.into_iter().map(|(idx, _)| idx).collect();
    indices.sort_unstable();
    EntangledHVec::from_indices(indices, dim)
}

fn seeded(level: u64, scale: u64, item: u64) -> u64 {
    hash_u64(hash_u64(level, scale), item)
}

pub(crate) fn hash_str_seed(s: &str, level: u64) -> u64 {
    let mut h = level;
    for &byte in s.as_bytes() {
        h = hash_u64(h, byte as u64);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let v = encode_text_internal("", 16384);
        assert!(v.indices().is_empty());
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
        assert!(!encode_text_internal("x", dim).indices().is_empty());
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

    #[test]
    fn rotation_matches_permute() {
        for dim in [1usize, 2, 255, 256, 1024] {
            for seed in 0..32u64 {
                let base = EntangledHVec::new_deterministic(dim, seed);
                for shift in 0..dim.min(7) {
                    let mut out = Vec::new();
                    rotate_sorted(base.indices(), shift as u32, dim as u32, &mut out);
                    assert_eq!(
                        out,
                        base.permute(shift).indices(),
                        "dim {dim} shift {shift}"
                    );
                }
            }
        }
    }

    #[test]
    fn memoized_path_matches_reference_at_density_boundaries() {
        let texts = [
            "x",
            "hi",
            "aa",
            "aaaa aaaa aaaa",
            "the cat sat on the mat",
            "Café records explain backups. Restore deleted documents from verified snapshots.",
            "ünïcödé text with punctuation!!! and NUMBERS 12345, repeated repeated repeated words",
            "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k l m n o p",
        ];
        for encoder in [
            TextEncoder::MultiscaleWordsV1,
            TextEncoder::MultiscaleCappedUnionV1,
        ] {
            for dim in [1usize, 2, 3, 255, 256, 257, 1024, 4096, 16384] {
                for text in texts {
                    let reference = encoder.encode_reference(text, dim);
                    let cached = encoder.encode(text, dim);
                    assert_eq!(cached.dim, reference.dim);
                    assert_eq!(
                        cached.indices(),
                        reference.indices(),
                        "{} dim {dim} text {text:?}",
                        encoder.name()
                    );
                    // A second call answers from the memo tables.
                    assert_eq!(encoder.encode(text, dim).indices(), reference.indices());
                }
            }
        }
    }
}
