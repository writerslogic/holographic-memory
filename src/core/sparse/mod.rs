// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Learned sparse retrieval: an impact index over learned sparse vectors (SPLADE-style
//! vocabulary-sized vectors, or any sparse term weights) with exact dot-product top-k.
//!
//! Documents are stored as postings per term, ordered by document id in blocks of at most
//! [`BLOCK`] entries: a 12-byte block header (first id, count, block maximum impact, payload
//! offset) and a payload of varint id gaps followed by one byte per impact. Learned weights are
//! quantized to 8-bit impacts with one global scale, so a score is `scale * sum q_t * impact_t`.
//! Queries run document-at-a-time MaxScore (Turtle and Flood, 1995): terms sorted by their
//! maximal contribution, the lowest ones become non-essential once the k-th best score exceeds
//! their summed bound and are only consulted for documents that can still qualify, with the
//! block maxima skipping whole blocks that cannot. The result is the exact top-k over the
//! quantized impacts.

use std::collections::BinaryHeap;

use crate::core::error::WarmError;

/// Postings per block.
pub const BLOCK: usize = 128;

/// A sparse vector: term ids ascending with their weights.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseVector {
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
}

impl SparseVector {
    /// Builds a vector from `(id, weight)` pairs, sorting them by id and dropping zeros and
    /// negative weights; duplicate ids are summed.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, f32)>) -> Self {
        let mut pairs: Vec<(u32, f32)> = pairs.into_iter().filter(|p| p.1 > 0.0).collect();
        pairs.sort_unstable_by_key(|p| p.0);
        let mut out = Self::default();
        for (id, w) in pairs {
            if out.ids.last() == Some(&id) {
                *out.weights.last_mut().expect("parallel") += w;
            } else {
                out.ids.push(id);
                out.weights.push(w);
            }
        }
        out
    }

    /// Exact dot product with another vector (both ascending).
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let (mut i, mut j, mut s) = (0, 0, 0f32);
        while i < self.ids.len() && j < other.ids.len() {
            match self.ids[i].cmp(&other.ids[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    s += self.weights[i] * other.weights[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        s
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct BlockHeader {
    first: u32,
    offset: u32,
    count: u16,
    max: u8,
    _pad: u8,
}

/// The impact index.
pub struct ImpactIndex {
    vocab: u32,
    docs: u32,
    scale: f32,
    /// Block range of term `t`: `term_blocks[t]..term_blocks[t + 1]`.
    term_blocks: Vec<u32>,
    term_max: Vec<u8>,
    blocks: Vec<BlockHeader>,
    payload: Vec<u8>,
    postings: u64,
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
fn get_varint(bytes: &[u8], at: &mut usize) -> u32 {
    let mut v = 0u32;
    let mut shift = 0;
    loop {
        let b = bytes[*at];
        *at += 1;
        v |= u32::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return v;
        }
        shift += 7;
    }
}

impl ImpactIndex {
    /// Builds the index over `docs` (document id = position) with a vocabulary of `vocab`
    /// terms. Weights are quantized to impacts `1..=255` with `scale = max weight / 255`;
    /// weights that round to zero are dropped.
    pub fn build(docs: &[SparseVector], vocab: u32) -> Result<Self, WarmError> {
        if docs.len() >= u32::MAX as usize {
            return Err(WarmError::InvalidParam {
                details: "too many documents for u32 ids".into(),
            });
        }
        let mut max_w = 0f32;
        for d in docs {
            if d.ids.len() != d.weights.len() || d.ids.windows(2).any(|p| p[0] >= p[1]) {
                return Err(WarmError::InvalidParam {
                    details: "sparse vector ids must be strictly ascending with one weight each"
                        .into(),
                });
            }
            if let Some(&last) = d.ids.last() {
                if last >= vocab {
                    return Err(WarmError::InvalidParam {
                        details: format!("term id {last} outside the vocabulary of {vocab}"),
                    });
                }
            }
            for &w in &d.weights {
                if !w.is_finite() || w < 0.0 {
                    return Err(WarmError::InvalidParam {
                        details: "sparse weights must be finite and non-negative".into(),
                    });
                }
                max_w = max_w.max(w);
            }
        }
        let scale = if max_w > 0.0 { max_w / 255.0 } else { 1.0 };
        // Postings per term, in document order (documents are visited in id order).
        let mut lists: Vec<Vec<(u32, u8)>> = vec![Vec::new(); vocab as usize];
        let mut postings = 0u64;
        for (doc, d) in docs.iter().enumerate() {
            for (&t, &w) in d.ids.iter().zip(&d.weights) {
                let q = (w / scale).round().clamp(0.0, 255.0) as u8;
                if q > 0 {
                    lists[t as usize].push((doc as u32, q));
                    postings += 1;
                }
            }
        }
        let mut term_blocks = Vec::with_capacity(vocab as usize + 1);
        let mut term_max = Vec::with_capacity(vocab as usize);
        let mut blocks = Vec::new();
        let mut payload = Vec::new();
        for list in &lists {
            term_blocks.push(blocks.len() as u32);
            term_max.push(list.iter().map(|p| p.1).max().unwrap_or(0));
            for chunk in list.chunks(BLOCK) {
                let offset = payload.len() as u32;
                let mut prev = chunk[0].0;
                for &(doc, _) in &chunk[1..] {
                    put_varint(&mut payload, doc - prev);
                    prev = doc;
                }
                payload.extend(chunk.iter().map(|p| p.1));
                blocks.push(BlockHeader {
                    first: chunk[0].0,
                    offset,
                    count: chunk.len() as u16,
                    max: chunk.iter().map(|p| p.1).max().unwrap_or(0),
                    _pad: 0,
                });
            }
        }
        term_blocks.push(blocks.len() as u32);
        Ok(Self {
            vocab,
            docs: docs.len() as u32,
            scale,
            term_blocks,
            term_max,
            blocks,
            payload,
            postings,
        })
    }

    pub fn documents(&self) -> u32 {
        self.docs
    }

    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    /// Stored postings.
    pub fn postings(&self) -> u64 {
        self.postings
    }

    /// Bytes of the index: block directory, payload, and the per-term tables.
    pub fn bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockHeader>()
            + self.payload.len()
            + self.term_blocks.len() * 4
            + self.term_max.len()
    }

    /// Quantization step of the impacts.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Decodes block `b` into `docs` and `impacts`.
    fn decode(&self, b: usize, docs: &mut Vec<u32>, impacts: &mut Vec<u8>) {
        let h = self.blocks[b];
        let n = usize::from(h.count);
        docs.clear();
        impacts.clear();
        let mut at = h.offset as usize;
        let mut doc = h.first;
        docs.push(doc);
        for _ in 1..n {
            doc += get_varint(&self.payload, &mut at);
            docs.push(doc);
        }
        impacts.extend_from_slice(&self.payload[at..at + n]);
    }

    /// Exact score of document `doc` for `query` over the quantized impacts (reference).
    pub fn score_document(&self, query: &SparseVector, doc: u32) -> f32 {
        let (mut docs, mut impacts) = (Vec::new(), Vec::new());
        let mut s = 0f32;
        for (&t, &qw) in query.ids.iter().zip(&query.weights) {
            if t >= self.vocab {
                continue;
            }
            let (lo, hi) = (
                self.term_blocks[t as usize] as usize,
                self.term_blocks[t as usize + 1] as usize,
            );
            if lo == hi {
                continue;
            }
            // Last block whose first id is at most `doc`.
            let b = lo + self.blocks[lo..hi].partition_point(|h| h.first <= doc);
            if b == lo {
                continue;
            }
            self.decode(b - 1, &mut docs, &mut impacts);
            if let Ok(i) = docs.binary_search(&doc) {
                s += qw * f32::from(impacts[i]);
            }
        }
        s * self.scale
    }

    /// The `k` best documents by dot product with `query` (score descending, ties by id),
    /// exact over the quantized impacts.
    pub fn top_k(&self, query: &SparseVector, k: usize) -> Vec<(u32, f32)> {
        if k == 0 {
            return Vec::new();
        }
        // Query terms with postings, ascending by their maximal contribution.
        let mut terms: Vec<(u32, f32, f32)> = query
            .ids
            .iter()
            .zip(&query.weights)
            .filter(|(&t, &w)| {
                w > 0.0
                    && t < self.vocab
                    && self.term_blocks[t as usize] < self.term_blocks[t as usize + 1]
            })
            .map(|(&t, &w)| (t, w, w * f32::from(self.term_max[t as usize])))
            .collect();
        terms.sort_by(|a, b| a.2.total_cmp(&b.2));
        let n = terms.len();
        if n == 0 {
            return Vec::new();
        }
        let mut prefix = vec![0f32; n];
        let mut acc = 0f32;
        for (i, t) in terms.iter().enumerate() {
            acc += t.2;
            prefix[i] = acc;
        }
        let mut cursors: Vec<Cursor> = terms
            .iter()
            .map(|&(t, qw, _)| Cursor::new(self, t, qw))
            .collect();
        let mut heap: BinaryHeap<std::cmp::Reverse<(OrdF32, std::cmp::Reverse<u32>)>> =
            BinaryHeap::with_capacity(k + 1);
        let mut theta = f32::NEG_INFINITY;
        // Terms `p..n` are essential: without any of them a document cannot beat `theta`.
        let mut p = 0usize;
        loop {
            let mut doc = u32::MAX;
            for c in &cursors[p..] {
                doc = doc.min(c.doc);
            }
            if doc == u32::MAX {
                break;
            }
            let mut score = 0f32;
            for c in &mut cursors[p..] {
                if c.doc == doc {
                    score += c.qw * f32::from(c.impact);
                    c.advance(self);
                }
            }
            // Non-essential terms, largest bound first; each one's absence is a bound.
            let mut i = p;
            while i > 0 {
                i -= 1;
                if score + prefix[i] <= theta {
                    break;
                }
                let c = &mut cursors[i];
                let remaining = if i > 0 { prefix[i - 1] } else { 0.0 };
                if let Some(imp) = c.seek(self, doc, theta - score - remaining) {
                    score += c.qw * f32::from(imp);
                }
            }
            if score > theta {
                heap.push(std::cmp::Reverse((OrdF32(score), std::cmp::Reverse(doc))));
                if heap.len() > k {
                    heap.pop();
                }
                if heap.len() == k {
                    theta = heap.peek().expect("k >= 1").0 .0 .0;
                    while p < n && prefix[p] <= theta {
                        p += 1;
                    }
                }
            }
            if p >= n {
                // Every term is non-essential: no unseen document can beat theta.
                break;
            }
        }
        let mut out: Vec<(u32, f32)> = heap
            .into_iter()
            .map(|r| ((r.0).1 .0, (r.0).0 .0 * self.scale))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }
}

/// f32 with a total order, for the result heap.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Traversal state of one query term's postings.
struct Cursor {
    qw: f32,
    lo: usize,
    hi: usize,
    block: usize,
    pos: usize,
    docs: Vec<u32>,
    impacts: Vec<u8>,
    /// Current document (`u32::MAX` at the end) and its impact.
    doc: u32,
    impact: u8,
}

impl Cursor {
    fn new(index: &ImpactIndex, term: u32, qw: f32) -> Self {
        let (lo, hi) = (
            index.term_blocks[term as usize] as usize,
            index.term_blocks[term as usize + 1] as usize,
        );
        let mut c = Self {
            qw,
            lo,
            hi,
            block: lo,
            pos: 0,
            docs: Vec::with_capacity(BLOCK),
            impacts: Vec::with_capacity(BLOCK),
            doc: u32::MAX,
            impact: 0,
        };
        c.load(index, lo);
        c
    }

    fn load(&mut self, index: &ImpactIndex, block: usize) {
        self.block = block;
        self.pos = 0;
        if block < self.hi {
            index.decode(block, &mut self.docs, &mut self.impacts);
            self.doc = self.docs[0];
            self.impact = self.impacts[0];
        } else {
            self.doc = u32::MAX;
            self.impact = 0;
        }
    }

    #[inline]
    fn advance(&mut self, index: &ImpactIndex) {
        self.pos += 1;
        if self.pos < self.docs.len() {
            self.doc = self.docs[self.pos];
            self.impact = self.impacts[self.pos];
        } else {
            let next = self.block + 1;
            self.load(index, next);
        }
    }

    /// Moves to the first document at or beyond `target` and returns its impact if it is
    /// `target` itself. Blocks whose maximal contribution cannot lift the score above the
    /// threshold (`need`, in impact units times `qw`) are skipped without decoding.
    fn seek(&mut self, index: &ImpactIndex, target: u32, need: f32) -> Option<u8> {
        if self.doc > target {
            return None;
        }
        if self.doc == target {
            return Some(self.impact);
        }
        // Past the current block? Find the last block starting at or before the target.
        let in_block = self.block < self.hi
            && self.pos < self.docs.len()
            && target <= *self.docs.last().expect("non-empty block");
        if !in_block {
            let b = self.lo + index.blocks[self.lo..self.hi].partition_point(|h| h.first <= target);
            if b == self.lo {
                return None;
            }
            let b = b - 1;
            if b != self.block || self.pos >= self.docs.len() {
                if self.qw * f32::from(index.blocks[b].max) <= need {
                    // The block cannot matter for this document; stay positioned before it
                    // logically by jumping to it without decoding when later seeks pass it.
                    self.block = b;
                    self.pos = usize::MAX;
                    self.doc = index.blocks[b].first.max(target);
                    self.impact = 0;
                    self.docs.clear();
                    self.impacts.clear();
                    return None;
                }
                self.load(index, b);
            }
        }
        // Within the loaded block.
        let start = if self.pos == usize::MAX { 0 } else { self.pos };
        let at = start + self.docs[start..].partition_point(|&d| d < target);
        if at < self.docs.len() {
            self.pos = at;
            self.doc = self.docs[at];
            self.impact = self.impacts[at];
            (self.doc == target).then_some(self.impact)
        } else {
            let next = self.block + 1;
            self.load(index, next);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64, for reproducible random corpora.
    struct SplitMix(u64);
    impl SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn random_corpus(n: usize, vocab: u32, nnz: usize, seed: u64) -> Vec<SparseVector> {
        let mut rng = SplitMix(seed);
        (0..n)
            .map(|_| {
                SparseVector::from_pairs((0..nnz).map(|_| {
                    (
                        (rng.next_u64() % u64::from(vocab)) as u32,
                        (rng.next_u64() % 1000) as f32 / 100.0 + 0.01,
                    )
                }))
            })
            .collect()
    }

    #[test]
    fn postings_round_trip_and_block_maxima_bound_every_impact() {
        let docs = random_corpus(2000, 300, 40, 1);
        let index = ImpactIndex::build(&docs, 300).expect("build");
        let (mut ids, mut impacts) = (Vec::new(), Vec::new());
        let mut seen = 0u64;
        for t in 0..300u32 {
            let (lo, hi) = (
                index.term_blocks[t as usize] as usize,
                index.term_blocks[t as usize + 1] as usize,
            );
            let mut last = None;
            for b in lo..hi {
                index.decode(b, &mut ids, &mut impacts);
                assert_eq!(ids.len(), impacts.len());
                assert!(ids.len() <= BLOCK && !ids.is_empty());
                assert!(ids.windows(2).all(|p| p[0] < p[1]));
                assert!(impacts.iter().all(|&i| i >= 1 && i <= index.blocks[b].max));
                assert!(impacts.contains(&index.blocks[b].max));
                assert!(index.blocks[b].max <= index.term_max[t as usize]);
                if let Some(l) = last {
                    assert!(ids[0] > l);
                }
                last = ids.last().copied();
                for (&d, &i) in ids.iter().zip(&impacts) {
                    let doc = &docs[d as usize];
                    let at = doc
                        .ids
                        .binary_search(&t)
                        .expect("posting belongs to document");
                    let want = (doc.weights[at] / index.scale()).round() as u8;
                    assert_eq!(i, want);
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, index.postings());
        assert!(
            index.bytes() < seen as usize * 5,
            "{} bytes for {seen} postings",
            index.bytes()
        );
    }

    #[test]
    fn top_k_matches_brute_force_over_quantized_impacts() {
        let docs = random_corpus(3000, 500, 30, 2);
        let index = ImpactIndex::build(&docs, 500).expect("build");
        let queries = random_corpus(60, 500, 12, 3);
        for q in &queries {
            for &k in &[1usize, 10, 100] {
                let got = index.top_k(q, k);
                let mut want: Vec<(u32, f32)> = (0..docs.len() as u32)
                    .map(|d| (d, index.score_document(q, d)))
                    .collect();
                want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                want.truncate(k);
                assert_eq!(got.len(), want.len().min(k));
                // Summation order differs between the two paths, so scores agree to f32
                // rounding; ids must agree wherever the scores are not tied within it.
                let tol = |w: f32| 1e-5 * w.abs().max(1.0);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!((g.1 - w.1).abs() <= tol(w.1), "{g:?} vs {w:?}");
                    let tied = |j: usize| j < want.len() && (want[j].1 - w.1).abs() <= tol(w.1);
                    if g.0 != w.0 {
                        let lo = i.checked_sub(1).is_some_and(tied);
                        assert!(lo || tied(i + 1), "rank {i}: {g:?} vs {w:?} is not a tie");
                    }
                }
            }
        }
        assert!(index.top_k(&SparseVector::default(), 10).is_empty());
        assert!(index.top_k(&queries[0], 0).is_empty());
    }

    #[test]
    fn invalid_vectors_are_rejected() {
        let bad = SparseVector {
            ids: vec![3, 1],
            weights: vec![1.0, 1.0],
        };
        assert!(ImpactIndex::build(&[bad], 10).is_err());
        let outside = SparseVector {
            ids: vec![12],
            weights: vec![1.0],
        };
        assert!(ImpactIndex::build(&[outside], 10).is_err());
        let nan = SparseVector {
            ids: vec![1],
            weights: vec![f32::NAN],
        };
        assert!(ImpactIndex::build(&[nan], 10).is_err());
        let v = SparseVector::from_pairs([(5, 1.0), (2, 0.5), (5, 2.0), (7, 0.0)]);
        assert_eq!(v.ids, vec![2, 5]);
        assert_eq!(v.weights, vec![0.5, 3.0]);
        assert_eq!(v.dot(&SparseVector::from_pairs([(5, 2.0), (9, 1.0)])), 6.0);
    }
}
