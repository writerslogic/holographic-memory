// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Document-local structural sketches and their counterfactual foils.
//!
//! # Mechanism
//! For an ordered stem sequence `w_0 .. w_{n-1}` the directional pair term at
//! distance `delta in {1, 2, 3}` is `bind(I[w_i], rho^delta(I[w_{i+delta}]))`.
//! The document sketch is `S_d = sign(sum of all pair terms)` stored as 2,048
//! sign bits (`[u64; 32]`, 256 bytes), plus the un-signed sum clamped to int8
//! for the ablation (2,048 bytes).
//!
//! The query sketch `Q+` is built the same way. Its foils are order-breaking
//! counterfactuals with the same token multiset:
//! * `F_rev`: every pair reversed (`bind(I[w_{i+delta}], rho^delta(I[w_i]))`),
//!   i.e. the sketch of the reversed sequence;
//! * `F_delta` for each `delta`: only the pairs at that distance reversed.
//! * callers may push extra foils from alternative surface forms (Part B pushes
//!   the agent/patient-swapped rendering of a structured query).
//!
//! The per-distance forward and reverse partial sums are shared between `Q+`
//! and its foils, so building all four generic foils costs twice `Q+`, not five
//! times.
//!
//! # Scores
//! * positive-only: `<Q+, S_d>`;
//! * mean-foil contrast: `<Q+ - lambda * mean_m(F_m), S_d>`, computed as ONE
//!   dot product of a precomputed integer contrast vector with `S_d`. With
//!   `lambda = L / 4` and `M` foils the vector `4M * (Q+ - lambda * mean F)` is
//!   exact in `i32`, so the score is exact and identical to the linear
//!   combination of the individual dots (asserted in tests);
//! * max-foil contrast: `<Q+, S_d> - max_m <F_m, S_d>`;
//! * cleanup energy (ablation baseline): cosine of `Q+` against the un-signed
//!   int8 trace of the document.
//!
//! # Hashed-pair baseline
//! `PairTable` keeps exact `(w_i, w_j, delta)` counts keyed by a 64-bit hash,
//! sorted so that overlap is a merge join. It answers whether the 2,048-bit
//! compression adds anything beyond explicit ordered-pair features.

use crate::hv::{
    accumulate, add_accum, atom, bind, clamp_i8, dot_bits, dot_bits_i8, dot_i32_bits, norm_i8,
    pair_key, rotate, sign_bits, tie_breaker, Accum, BitVec, DIM,
};

/// Pair distances contributing to a sketch.
pub const DELTAS: [usize; 3] = [1, 2, 3];
/// Number of generic foils (`F_rev` plus one per distance).
pub const GENERIC_FOILS: usize = 1 + DELTAS.len();

/// Per-token atoms: `I[w]` and `rho^delta(I[w])` for each distance.
struct TokenAtoms {
    plain: BitVec,
    rotated: [BitVec; DELTAS.len()],
}

fn token_atoms(seeds: &[u64]) -> Vec<TokenAtoms> {
    seeds
        .iter()
        .map(|&seed| {
            let plain = atom(seed);
            let rotated = [
                rotate(&plain, DELTAS[0]),
                rotate(&plain, DELTAS[1]),
                rotate(&plain, DELTAS[2]),
            ];
            TokenAtoms { plain, rotated }
        })
        .collect()
}

fn zero_accum() -> Box<Accum> {
    Box::new([0i32; DIM])
}

/// Forward and reversed per-distance pair sums of one stem sequence.
pub struct PairSums {
    /// `forward[k]` sums `bind(I[w_i], rho^d(I[w_{i+d}]))` for `d = DELTAS[k]`.
    pub forward: [Box<Accum>; DELTAS.len()],
    /// `reverse[k]` sums `bind(I[w_{i+d}], rho^d(I[w_i]))` for `d = DELTAS[k]`.
    pub reverse: Option<[Box<Accum>; DELTAS.len()]>,
    /// Number of forward pair terms accumulated (over all distances).
    pub pairs: usize,
}

/// Accumulates the pair sums of `seeds`; `with_reverse` also builds the reversed sums.
pub fn pair_sums(seeds: &[u64], with_reverse: bool) -> PairSums {
    let atoms = token_atoms(seeds);
    let mut forward = [zero_accum(), zero_accum(), zero_accum()];
    let mut reverse = if with_reverse {
        Some([zero_accum(), zero_accum(), zero_accum()])
    } else {
        None
    };
    let mut pairs = 0usize;
    for (k, &delta) in DELTAS.iter().enumerate() {
        if atoms.len() <= delta {
            continue;
        }
        for i in 0..atoms.len() - delta {
            let a = &atoms[i];
            let b = &atoms[i + delta];
            accumulate(&mut forward[k], &bind(&a.plain, &b.rotated[k]));
            if let Some(rev) = reverse.as_mut() {
                accumulate(&mut rev[k], &bind(&b.plain, &a.rotated[k]));
            }
            pairs += 1;
        }
    }
    PairSums {
        forward,
        reverse,
        pairs,
    }
}

/// A document's structural sketch.
pub struct DocSketch {
    /// `sign(sum of pair terms)` as 2,048 sign bits (256 bytes).
    pub bits: BitVec,
    /// The un-signed sum clamped to int8 (ablation).
    pub int8: Box<[i8; DIM]>,
    /// Euclidean norm of `int8` (for cleanup-energy cosine).
    pub int8_norm: f64,
    /// Number of pair terms superposed.
    pub pairs: usize,
}

/// Builds the sketch of an ordered stem sequence.
pub fn doc_sketch(seeds: &[u64]) -> DocSketch {
    let sums = pair_sums(seeds, false);
    let mut total = zero_accum();
    for acc in sums.forward.iter() {
        add_accum(&mut total, acc);
    }
    let bits = sign_bits(&total, &tie_breaker());
    let int8 = clamp_i8(&total);
    let int8_norm = norm_i8(&int8);
    DocSketch {
        bits,
        int8,
        int8_norm,
        pairs: sums.pairs,
    }
}

/// A query's positive sketch and its foil set.
pub struct QuerySketch {
    /// `Q+`.
    pub positive: BitVec,
    /// `F_rev`, `F_delta1`, `F_delta2`, `F_delta3`, then any pushed foils.
    pub foils: Vec<BitVec>,
    /// Number of forward pair terms in `Q+`.
    pub pairs: usize,
}

/// Builds `Q+` and the four generic foils from an ordered stem sequence.
pub fn query_sketch(seeds: &[u64]) -> QuerySketch {
    let tie = tie_breaker();
    let sums = pair_sums(seeds, true);
    let reverse = sums
        .reverse
        .as_ref()
        .expect("pair_sums(with_reverse = true) always fills the reversed sums");
    let mut pos = zero_accum();
    let mut rev = zero_accum();
    for k in 0..DELTAS.len() {
        add_accum(&mut pos, &sums.forward[k]);
        add_accum(&mut rev, &reverse[k]);
    }
    let positive = sign_bits(&pos, &tie);
    let mut foils = Vec::with_capacity(GENERIC_FOILS + 1);
    foils.push(sign_bits(&rev, &tie));
    for k in 0..DELTAS.len() {
        let mut partial = zero_accum();
        for j in 0..DELTAS.len() {
            if j == k {
                add_accum(&mut partial, &reverse[j]);
            } else {
                add_accum(&mut partial, &sums.forward[j]);
            }
        }
        foils.push(sign_bits(&partial, &tie));
    }
    QuerySketch {
        positive,
        foils,
        pairs: sums.pairs,
    }
}

impl QuerySketch {
    /// Appends the sketch of an alternative surface form (e.g. the role-swapped query) as a foil.
    pub fn push_foil(&mut self, seeds: &[u64]) {
        let sums = pair_sums(seeds, false);
        let mut total = zero_accum();
        for acc in sums.forward.iter() {
            add_accum(&mut total, acc);
        }
        self.foils.push(sign_bits(&total, &tie_breaker()));
    }

    /// Positive-only score `<Q+, S_d>`.
    #[inline]
    pub fn positive_score(&self, doc: &DocSketch) -> i32 {
        dot_bits(&self.positive, &doc.bits)
    }

    /// `<F_m, S_d>` for every foil, in order.
    #[inline]
    pub fn foil_scores(&self, doc: &DocSketch) -> Vec<i32> {
        self.foils.iter().map(|f| dot_bits(f, &doc.bits)).collect()
    }

    /// Max-foil contrast `<Q+, S_d> - max_m <F_m, S_d>` (no `lambda`).
    #[inline]
    pub fn max_foil_score(&self, doc: &DocSketch) -> i32 {
        let pos = self.positive_score(doc);
        let worst = self
            .foils
            .iter()
            .map(|f| dot_bits(f, &doc.bits))
            .max()
            .unwrap_or(0);
        pos - worst
    }

    /// Cleanup-energy ablation: cosine of `Q+` against the un-signed int8 trace.
    #[inline]
    pub fn cleanup_energy(&self, doc: &DocSketch) -> f64 {
        if doc.int8_norm == 0.0 {
            return 0.0;
        }
        f64::from(dot_bits_i8(&self.positive, &doc.int8)) / (doc.int8_norm * (DIM as f64).sqrt())
    }

    /// The precomputed integer contrast vector for `lambda = lambda_quarters / 4`.
    pub fn contrast_vector(&self, lambda_quarters: u32) -> ContrastVector {
        let m = self.foils.len() as i32;
        let scale = 4 * m.max(1);
        let mut q = Box::new([0i32; DIM]);
        let lam = lambda_quarters as i32;
        for i in 0..DIM {
            let w = i / 64;
            let b = i % 64;
            let p = (((self.positive[w] >> b) & 1) as i32) * 2 - 1;
            let mut foil_sum = 0i32;
            for f in &self.foils {
                foil_sum += (((f[w] >> b) & 1) as i32) * 2 - 1;
            }
            // 4M * (p - lambda * foil_sum / M) = 4M*p - 4*lambda*foil_sum = 4M*p - L*foil_sum
            q[i] = scale * p - lam * foil_sum;
        }
        let total: i64 = q.iter().map(|&x| i64::from(x)).sum();
        ContrastVector { q, total, scale }
    }
}

/// `4M * (Q+ - lambda * mean_m F_m)` as an exact integer vector.
pub struct ContrastVector {
    /// The scaled contrast vector.
    pub q: Box<[i32; DIM]>,
    /// `sum_i q_i`, precomputed for the single-dot-product scorer.
    pub total: i64,
    /// `4M`; divide a raw score by it to recover `<Q+ - lambda * mean F, S_d>`.
    pub scale: i32,
}

impl ContrastVector {
    /// Raw contrast score (scaled by `scale`): one dot product with the document bits.
    #[inline]
    pub fn score_raw(&self, doc: &DocSketch) -> i64 {
        dot_i32_bits(&self.q, self.total, &doc.bits)
    }

    /// `<Q+ - lambda * mean F, S_d>` in sketch units.
    #[inline]
    pub fn score(&self, doc: &DocSketch) -> f64 {
        self.score_raw(doc) as f64 / f64::from(self.scale)
    }
}

/// Mean-foil contrast derived from individual dots (used for lambda sweeps).
#[inline]
pub fn mean_foil_from_dots(pos: i32, foils: &[i32], lambda: f64) -> f64 {
    if foils.is_empty() {
        return f64::from(pos);
    }
    let mean = foils.iter().map(|&x| f64::from(x)).sum::<f64>() / foils.len() as f64;
    f64::from(pos) - lambda * mean
}

/// Exact ordered-pair feature counts `(w_i, w_{i+delta}, delta)`, sorted by key.
pub struct PairTable {
    entries: Vec<(u64, u32)>,
    total: u32,
}

impl PairTable {
    /// Builds the table of an ordered stem sequence.
    pub fn build(seeds: &[u64]) -> Self {
        let mut keys: Vec<u64> = Vec::new();
        for &delta in DELTAS.iter() {
            if seeds.len() <= delta {
                continue;
            }
            for i in 0..seeds.len() - delta {
                keys.push(pair_key(seeds[i], seeds[i + delta], delta));
            }
        }
        keys.sort_unstable();
        let mut entries: Vec<(u64, u32)> = Vec::new();
        for key in keys {
            match entries.last_mut() {
                Some((k, c)) if *k == key => *c += 1,
                _ => entries.push((key, 1)),
            }
        }
        let total = entries.iter().map(|&(_, c)| c).sum();
        Self { entries, total }
    }

    /// Number of pair occurrences.
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Multiset overlap `sum_f min(count_self, count_other)`.
    pub fn overlap(&self, other: &PairTable) -> u32 {
        let (mut i, mut j, mut acc) = (0usize, 0usize, 0u32);
        while i < self.entries.len() && j < other.entries.len() {
            let (ka, ca) = self.entries[i];
            let (kb, cb) = other.entries[j];
            if ka == kb {
                acc += ca.min(cb);
                i += 1;
                j += 1;
            } else if ka < kb {
                i += 1;
            } else {
                j += 1;
            }
        }
        acc
    }

    /// Fraction of this (query) table's pair occurrences found in `doc`, in `[0, 1]`.
    pub fn coverage(&self, doc: &PairTable) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        f64::from(self.overlap(doc)) / f64::from(self.total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hv::stem_seed;

    fn seeds(words: &[&str]) -> Vec<u64> {
        words.iter().map(|w| stem_seed(w)).collect()
    }

    #[test]
    fn order_changes_the_sketch_but_not_the_pair_count() {
        let a = doc_sketch(&seeds(&["alice", "warned", "bob", "after", "the", "audit"]));
        let b = doc_sketch(&seeds(&["bob", "warned", "alice", "after", "the", "audit"]));
        assert_eq!(a.pairs, b.pairs);
        assert_ne!(a.bits, b.bits);
    }

    #[test]
    fn reversed_foil_matches_reversed_sequence_sketch() {
        let fwd = seeds(&["alice", "warned", "bob", "after", "the", "audit"]);
        let mut rev = fwd.clone();
        rev.reverse();
        let q = query_sketch(&fwd);
        let r = doc_sketch(&rev);
        assert_eq!(q.foils[0], r.bits);
    }

    #[test]
    fn single_dot_contrast_equals_linear_combination() {
        let q = query_sketch(&seeds(&["the", "cat", "sat", "on", "the", "mat"]));
        let d = doc_sketch(&seeds(&[
            "yesterday", "the", "cat", "sat", "on", "the", "mat", "and", "slept",
        ]));
        for lq in [0u32, 2, 3, 4, 6] {
            let cv = q.contrast_vector(lq);
            let one = cv.score(&d);
            let lin = mean_foil_from_dots(q.positive_score(&d), &q.foil_scores(&d), lq as f64 / 4.0);
            assert!((one - lin).abs() < 1e-9, "lambda/4={lq}: {one} vs {lin}");
        }
    }

    #[test]
    fn pair_table_overlap_is_multiset_min() {
        let a = PairTable::build(&seeds(&["a", "b", "a", "b"]));
        let b = PairTable::build(&seeds(&["a", "b"]));
        assert_eq!(b.overlap(&a), 1);
        assert_eq!(a.overlap(&b), 1);
        assert!((b.coverage(&a) - 1.0).abs() < 1e-12);
    }
}
