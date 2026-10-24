// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Label-filtered exact-metric nearest-neighbour search over 8-bit vectors (Euclidean).
//!
//! Every point carries a set of labels; a query carries one or more labels and asks for the
//! `k` nearest points that hold *all* of them (AND semantics). Each label keeps its sorted
//! member ids; labels with many members also keep a dense membership bitset and an inverted
//! file of their own members (k-means lists trained on that label alone). A query is routed by
//! the posting size of its rarest label:
//!
//! - rarest posting at most `brute_max`: its members are intersected with the other labels and
//!   every match is scored exactly;
//! - otherwise the rarest label's lists are probed nearest-centroid first, members failing the
//!   other labels are skipped, and probing stops once `target` matches have been scored (and at
//!   least `min_lists` lists probed). With `target >= n` every list is probed and the result is
//!   the exact filtered top-k.
//!
//! Distances are exact integers: rows are stored as `x - 128` in i8, the query likewise, and
//! `|q - x|^2 = |q'|^2 + |x'|^2 - 2 q'.x'`, ranked by `|x'|^2 - 2 q'.x'` from the integer dot
//! kernels (VNNI where the CPU has it). Ties break on the smaller id.
//!
//! Deletes set a tombstone that every path checks; inserting a point needs a rebuild.

use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use super::kernels::{dot_i8_query, dot_i8_query_x4, prefetch_bytes};

/// Errors of [`FilteredIndex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterError {
    /// An argument has the wrong shape or an out-of-range value.
    Invalid(String),
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(m) => write!(f, "invalid filtered-index argument: {m}"),
        }
    }
}

impl std::error::Error for FilterError {}

/// Build parameters of a [`FilteredIndex`].
#[derive(Debug, Clone, Copy)]
pub struct FilterBuildParams {
    /// Labels with more members than this get an inverted file of their own.
    pub ivf_min: usize,
    /// Target members per list of a label's inverted file.
    pub cluster_size: usize,
    /// Upper bound on the lists of one label.
    pub max_lists: usize,
    /// Labels with at least this many members get a dense membership bitset.
    pub bitset_min: usize,
    /// Lloyd iterations of the per-label k-means.
    pub kmeans_iters: usize,
    /// Training points per list for the per-label k-means (a sample of the members).
    pub train_per_list: usize,
    /// Labels with an inverted file and at least this many members also keep a copy of their
    /// rows grouped by list, so a probe reads memory sequentially (`usize::MAX` = none).
    pub copy_min: usize,
    /// Seed of the k-means sampling.
    pub seed: u64,
}

impl Default for FilterBuildParams {
    fn default() -> Self {
        Self {
            ivf_min: 20_000,
            cluster_size: 1_000,
            max_lists: 1_024,
            bitset_min: 20_000,
            kmeans_iters: 6,
            train_per_list: 48,
            copy_min: usize::MAX,
            seed: 0x5eed,
        }
    }
}

/// Search parameters of [`FilteredIndex::search`].
#[derive(Debug, Clone, Copy)]
pub struct FilterSearchParams {
    /// Queries whose rarest label has at most this many members are answered exactly by
    /// scoring every match.
    pub brute_max: usize,
    /// Matches to score before an inverted-file search stops.
    pub target: usize,
    /// Lists an inverted-file search probes at least.
    pub min_lists: usize,
    /// Multi-label queries whose rarest label has at most this many members are intersected
    /// first; when the matches number at most `brute_max` they are all scored exactly.
    pub intersect_max: usize,
}

impl Default for FilterSearchParams {
    fn default() -> Self {
        Self {
            brute_max: 20_000,
            target: 5_000,
            min_lists: 1,
            intersect_max: 200_000,
        }
    }
}

impl FilterSearchParams {
    /// Parameters under which every query returns the exact filtered top-k.
    #[must_use]
    pub fn exact() -> Self {
        Self {
            brute_max: usize::MAX,
            target: usize::MAX,
            min_lists: usize::MAX,
            intersect_max: usize::MAX,
        }
    }
}

/// The inverted file of one label: list centroids and the members of each list.
struct LabelIvf {
    /// `lists * dim` centroids in the shifted i8 domain.
    centroids: Vec<i8>,
    /// `|c'|^2` per list.
    centroid_norms: Vec<i32>,
    /// `lists + 1` offsets into `members`.
    offsets: Vec<u32>,
    /// Member ids grouped by list, ascending within a list.
    members: Vec<u32>,
    /// The members' rows in `members` order (`stride` each), or empty.
    rows: Vec<i8>,
    /// `|x'|^2` in `members` order, or empty.
    norms: Vec<i32>,
}

/// Which path a query took, counted for diagnostics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FilterStats {
    /// Queries scored exactly over their matches.
    pub brute: u64,
    /// Queries answered by a label inverted file.
    pub ivf: u64,
    /// Rows scored.
    pub scored: u64,
}

/// A label-filtered index over 8-bit Euclidean vectors. See the module documentation.
pub struct FilteredIndex {
    dim: usize,
    /// `dim` rounded up to a multiple of 16: the row and centroid width.
    stride: usize,
    n: usize,
    /// Rows as `x - 128`, `n * dim`.
    rows: Vec<i8>,
    /// `|x'|^2` per row.
    norms: Vec<i32>,
    /// `labels + 1` offsets into `postings`.
    label_offsets: Vec<u64>,
    /// Member ids of every label, ascending.
    postings: Vec<u32>,
    /// Bitset slot per label (`u32::MAX` = none).
    bitset_slot: Vec<u32>,
    /// `words` words per slot.
    bitsets: Vec<u64>,
    /// Inverted file per label (`u32::MAX` = none), index into `ivfs`.
    ivf_slot: Vec<u32>,
    ivfs: Vec<LabelIvf>,
    deleted: Vec<AtomicU64>,
    live: usize,
    brute_queries: AtomicU64,
    ivf_queries: AtomicU64,
    scored: AtomicU64,
}

#[inline]
fn shift(v: u8) -> i8 {
    (v ^ 0x80) as i8
}

#[inline]
fn prefetch_norm(v: &i32) {
    // SAFETY: a prefetch hint never faults or writes; the pointer comes from a live reference.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
            std::ptr::from_ref(v).cast::<i8>(),
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = v;
}

#[inline]
fn bit(words: &[u64], id: u32) -> bool {
    words[(id >> 6) as usize] >> (id & 63) & 1 == 1
}

/// The `k` best `(key, id)` pairs seen, kept sorted ascending.
struct TopK {
    k: usize,
    items: Vec<(i64, u32)>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    #[inline]
    fn worst(&self) -> (i64, u32) {
        if self.items.len() < self.k {
            (i64::MAX, u32::MAX)
        } else {
            self.items[self.k - 1]
        }
    }

    #[inline]
    fn push(&mut self, key: i64, id: u32) {
        if self.k == 0 || (key, id) >= self.worst() {
            return;
        }
        let at = self.items.partition_point(|&e| e < (key, id));
        self.items.insert(at, (key, id));
        self.items.truncate(self.k);
    }
}

/// Splitmix64, for deterministic k-means sampling.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl FilteredIndex {
    /// Builds the index over `n = data.len() / dim` rows of `data` (row-major u8) and their
    /// labels as CSR (`label_indptr` with `n + 1` entries into `label_ids`). `dim` must be a
    /// positive multiple of 16, label ids below `labels` (0 infers it from the data).
    pub fn build(
        data: &[u8],
        dim: usize,
        label_indptr: &[u64],
        label_ids: &[u32],
        labels: usize,
        params: &FilterBuildParams,
    ) -> Result<Self, FilterError> {
        if dim == 0 {
            return Err(FilterError::Invalid("dim must be positive".into()));
        }
        if !data.len().is_multiple_of(dim) {
            return Err(FilterError::Invalid(
                "data is not a whole number of rows".into(),
            ));
        }
        let n = data.len() / dim;
        if n > u32::MAX as usize - 1 {
            return Err(FilterError::Invalid(format!("{n} rows exceed u32 ids")));
        }
        if label_indptr.len() != n + 1
            || label_indptr[0] != 0
            || label_indptr.windows(2).any(|w| w[0] > w[1])
            || label_indptr[n] as usize != label_ids.len()
        {
            return Err(FilterError::Invalid(
                "label_indptr is not a monotone CSR index over label_ids with n + 1 entries".into(),
            ));
        }
        let max_label = label_ids
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        let labels = if labels == 0 { max_label } else { labels };
        if max_label > labels {
            return Err(FilterError::Invalid(format!(
                "label id {} is out of range for {labels} labels",
                max_label - 1
            )));
        }
        if params.cluster_size == 0 || params.max_lists == 0 {
            return Err(FilterError::Invalid(
                "cluster_size and max_lists must be positive".into(),
            ));
        }

        // Rows are zero-padded (in the shifted domain) to a multiple of 16 for the kernels.
        let stride = dim.next_multiple_of(16);
        let mut rows = vec![0i8; n * stride];
        let mut norms = vec![0i32; n];
        rows.par_chunks_mut(stride)
            .zip(data.par_chunks(dim))
            .zip(norms.par_iter_mut())
            .for_each(|((out, x), norm)| {
                let mut s = 0i32;
                for (o, &v) in out.iter_mut().zip(x) {
                    *o = shift(v);
                    s += i32::from(*o) * i32::from(*o);
                }
                *norm = s;
            });

        // Transpose the point -> labels CSR with a counting sort: ids come out ascending.
        let mut counts = vec![0u64; labels + 1];
        for &l in label_ids {
            counts[l as usize + 1] += 1;
        }
        for i in 0..labels {
            counts[i + 1] += counts[i];
        }
        let label_offsets = counts.clone();
        let mut fill = counts;
        let mut postings = vec![0u32; label_ids.len()];
        for p in 0..n {
            let (a, b) = (label_indptr[p] as usize, label_indptr[p + 1] as usize);
            for &l in &label_ids[a..b] {
                let at = &mut fill[l as usize];
                postings[*at as usize] = p as u32;
                *at += 1;
            }
        }
        // A point listing a label twice would appear twice; collapse adjacent repeats.
        let mut dedup_offsets = vec![0u64; labels + 1];
        let mut write = 0usize;
        for l in 0..labels {
            let (a, b) = (label_offsets[l] as usize, label_offsets[l + 1] as usize);
            let mut prev = u32::MAX;
            for r in a..b {
                let id = postings[r];
                if id != prev {
                    postings[write] = id;
                    write += 1;
                    prev = id;
                }
            }
            dedup_offsets[l + 1] = write as u64;
        }
        postings.truncate(write);
        let label_offsets = dedup_offsets;

        let words = n.div_ceil(64);
        let mut bitset_slot = vec![u32::MAX; labels];
        let dense: Vec<usize> = (0..labels)
            .filter(|&l| {
                let len = (label_offsets[l + 1] - label_offsets[l]) as usize;
                len > 0 && len >= params.bitset_min
            })
            .collect();
        for (slot, &l) in dense.iter().enumerate() {
            bitset_slot[l] = slot as u32;
        }
        let mut bitsets = vec![0u64; dense.len() * words];
        bitsets
            .par_chunks_mut(words.max(1))
            .zip(dense.par_iter())
            .for_each(|(set, &l)| {
                let (a, b) = (label_offsets[l] as usize, label_offsets[l + 1] as usize);
                for &id in &postings[a..b] {
                    set[(id >> 6) as usize] |= 1 << (id & 63);
                }
            });

        let mut ivf_slot = vec![u32::MAX; labels];
        let mut big: Vec<usize> = (0..labels)
            .filter(|&l| (label_offsets[l + 1] - label_offsets[l]) as usize > params.ivf_min)
            .collect();
        // Largest first so the long trainings start early.
        big.sort_by_key(|&l| std::cmp::Reverse(label_offsets[l + 1] - label_offsets[l]));
        let ivfs: Vec<LabelIvf> = big
            .par_iter()
            .map(|&l| {
                let (a, b) = (label_offsets[l] as usize, label_offsets[l + 1] as usize);
                train_label(&rows, &norms, stride, &postings[a..b], params, l as u64)
            })
            .collect();
        for (slot, &l) in big.iter().enumerate() {
            ivf_slot[l] = slot as u32;
        }

        Ok(Self {
            dim,
            stride,
            n,
            rows,
            norms,
            label_offsets,
            postings,
            bitset_slot,
            bitsets,
            ivf_slot,
            ivfs,
            deleted: (0..words).map(|_| AtomicU64::new(0)).collect(),
            live: n,
            brute_queries: AtomicU64::new(0),
            ivf_queries: AtomicU64::new(0),
            scored: AtomicU64::new(0),
        })
    }

    /// Row width.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Rows indexed, deleted ones included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether no rows are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Rows not deleted.
    #[must_use]
    pub fn live(&self) -> usize {
        self.live
    }

    /// Number of labels.
    #[must_use]
    pub fn labels(&self) -> usize {
        self.label_offsets.len() - 1
    }

    /// Members of `label` (deleted ones included).
    #[must_use]
    pub fn label_len(&self, label: u32) -> usize {
        let l = label as usize;
        if l >= self.labels() {
            return 0;
        }
        (self.label_offsets[l + 1] - self.label_offsets[l]) as usize
    }

    /// Labels that have an inverted file.
    #[must_use]
    pub fn ivf_labels(&self) -> usize {
        self.ivfs.len()
    }

    /// Bytes held by the index structures (rows, postings, bitsets, inverted files).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.rows.len()
            + self.norms.len() * 4
            + self.label_offsets.len() * 8
            + self.postings.len() * 4
            + self.bitsets.len() * 8
            + self.deleted.len() * 8
            + self
                .ivfs
                .iter()
                .map(|v| {
                    v.centroids.len()
                        + v.centroid_norms.len() * 4
                        + v.offsets.len() * 4
                        + v.members.len() * 4
                        + v.rows.len()
                        + v.norms.len() * 4
                })
                .sum::<usize>()
    }

    /// Marks `id` deleted; it is skipped by every search from now on. Returns whether it was
    /// live.
    pub fn delete(&mut self, id: u32) -> bool {
        if id as usize >= self.n {
            return false;
        }
        let w = &self.deleted[(id >> 6) as usize];
        let mask = 1u64 << (id & 63);
        let was = w.fetch_or(mask, Ordering::Relaxed) & mask == 0;
        if was {
            self.live -= 1;
        }
        was
    }

    #[inline]
    fn is_deleted(&self, id: u32) -> bool {
        self.deleted[(id >> 6) as usize].load(Ordering::Relaxed) >> (id & 63) & 1 == 1
    }

    fn posting(&self, label: u32) -> &[u32] {
        let l = label as usize;
        &self.postings[self.label_offsets[l] as usize..self.label_offsets[l + 1] as usize]
    }

    fn bitset(&self, label: u32) -> Option<&[u64]> {
        let slot = self.bitset_slot[label as usize];
        (slot != u32::MAX).then(|| {
            let words = self.n.div_ceil(64);
            &self.bitsets[slot as usize * words..(slot as usize + 1) * words]
        })
    }

    #[inline]
    fn row(&self, id: u32) -> &[i8] {
        let at = id as usize * self.stride;
        &self.rows[at..at + self.stride]
    }

    /// Path counters since the last [`Self::reset_stats`].
    #[must_use]
    pub fn stats(&self) -> FilterStats {
        FilterStats {
            brute: self.brute_queries.load(Ordering::Relaxed),
            ivf: self.ivf_queries.load(Ordering::Relaxed),
            scored: self.scored.load(Ordering::Relaxed),
        }
    }

    /// Clears the path counters.
    pub fn reset_stats(&self) {
        self.brute_queries.store(0, Ordering::Relaxed);
        self.ivf_queries.store(0, Ordering::Relaxed);
        self.scored.store(0, Ordering::Relaxed);
    }

    /// The `k` nearest live points holding every label in `labels`, nearest first, written to
    /// `out` (cleared first) as `(squared distance, id)`. An empty `labels` matches nothing;
    /// a label id out of range matches nothing.
    pub fn search(
        &self,
        query: &[u8],
        labels: &[u32],
        k: usize,
        params: &FilterSearchParams,
        out: &mut Vec<(u32, u32)>,
    ) -> Result<(), FilterError> {
        out.clear();
        if query.len() != self.dim {
            return Err(FilterError::Invalid(format!(
                "query has {} coordinates, index {}",
                query.len(),
                self.dim
            )));
        }
        if k == 0 || labels.is_empty() || labels.iter().any(|&l| l as usize >= self.labels()) {
            return Ok(());
        }
        let mut order: Vec<u32> = labels.to_vec();
        order.sort_by_key(|&l| (self.label_len(l), l));
        order.dedup();
        let rarest = order[0];
        let others = &order[1..];

        let mut q = vec![0i8; self.stride];
        for (o, &v) in q.iter_mut().zip(query) {
            *o = shift(v);
        }
        let qsum: i32 = q.iter().map(|&v| i32::from(v)).sum();
        let qnorm: i64 = q.iter().map(|&v| i64::from(v) * i64::from(v)).sum();
        let mut top = TopK::new(k);

        let n_rare = self.label_len(rarest);
        let ivf = self.ivf_slot[rarest as usize];
        let matches = if n_rare <= params.brute_max
            || ivf == u32::MAX
            || (!others.is_empty() && n_rare <= params.intersect_max)
        {
            Some(self.intersect(rarest, others))
        } else {
            None
        };
        let scored = match matches {
            Some(m) if m.len() <= params.brute_max || ivf == u32::MAX => {
                self.brute_queries.fetch_add(1, Ordering::Relaxed);
                self.score_ids(&q, qsum, &m, &mut top);
                m.len()
            }
            _ => {
                self.ivf_queries.fetch_add(1, Ordering::Relaxed);
                self.search_ivf(&self.ivfs[ivf as usize], &q, qsum, others, params, &mut top)
            }
        };
        self.scored.fetch_add(scored as u64, Ordering::Relaxed);
        out.extend(
            top.items
                .iter()
                .map(|&(key, id)| (u32::try_from(key + qnorm).unwrap_or(u32::MAX), id)),
        );
        Ok(())
    }

    /// Whether `id` holds every label in `others` (bitsets first, then sorted postings).
    #[inline]
    fn holds(&self, id: u32, others: &[u32], cursors: &mut [usize]) -> bool {
        for (j, &l) in others.iter().enumerate() {
            if let Some(set) = self.bitset(l) {
                if !bit(set, id) {
                    return false;
                }
            } else {
                // Ids arrive ascending, so a forward galloping cursor per label suffices.
                let p = self.posting(l);
                let c = &mut cursors[j];
                if *c < p.len() && p[*c] < id {
                    // Invariant: p[lo] < id.
                    let (mut lo, mut step) = (*c, 1);
                    while lo + step < p.len() && p[lo + step] < id {
                        lo += step;
                        step *= 2;
                    }
                    let hi = (lo + step + 1).min(p.len());
                    *c = lo + 1 + p[lo + 1..hi].partition_point(|&v| v < id);
                }
                if *c >= p.len() || p[*c] != id {
                    return false;
                }
            }
        }
        true
    }

    /// The live members of `rarest` that hold every label in `others`, ascending.
    fn intersect(&self, rarest: u32, others: &[u32]) -> Vec<u32> {
        let mut cursors = vec![0usize; others.len()];
        self.posting(rarest)
            .iter()
            .copied()
            .filter(|&id| !self.is_deleted(id) && self.holds(id, others, &mut cursors))
            .collect()
    }

    /// Scores every id of `ids` (all live and matching) into `top`.
    fn score_ids(&self, q: &[i8], qsum: i32, ids: &[u32], top: &mut TopK) {
        const AHEAD: usize = 8;
        let (chunks, tail) = ids.as_chunks::<4>();
        for (c, ch) in chunks.iter().enumerate() {
            for &p in ids.iter().skip(c * 4 + AHEAD).take(4) {
                prefetch_bytes(self.row(p));
                prefetch_norm(&self.norms[p as usize]);
            }
            let rows = [
                self.row(ch[0]),
                self.row(ch[1]),
                self.row(ch[2]),
                self.row(ch[3]),
            ];
            let d = dot_i8_query_x4(q, rows, qsum);
            for j in 0..4 {
                let key = i64::from(self.norms[ch[j] as usize]) - 2 * i64::from(d[j]);
                top.push(key, ch[j]);
            }
        }
        for &id in tail {
            let key = i64::from(self.norms[id as usize])
                - 2 * i64::from(dot_i8_query(q, self.row(id), qsum));
            top.push(key, id);
        }
    }

    /// Scores the copied rows at positions `pos` of `ivf` into `top`.
    fn score_copied(&self, ivf: &LabelIvf, q: &[i8], qsum: i32, pos: &[u32], top: &mut TopK) {
        let s = self.stride;
        let row = |p: u32| &ivf.rows[p as usize * s..(p as usize + 1) * s];
        let (chunks, tail) = pos.as_chunks::<4>();
        for ch in chunks {
            let d = dot_i8_query_x4(q, [row(ch[0]), row(ch[1]), row(ch[2]), row(ch[3])], qsum);
            for (&p, &dot) in ch.iter().zip(&d) {
                let key = i64::from(ivf.norms[p as usize]) - 2 * i64::from(dot);
                top.push(key, ivf.members[p as usize]);
            }
        }
        for &p in tail {
            let key =
                i64::from(ivf.norms[p as usize]) - 2 * i64::from(dot_i8_query(q, row(p), qsum));
            top.push(key, ivf.members[p as usize]);
        }
    }

    fn search_ivf(
        &self,
        ivf: &LabelIvf,
        q: &[i8],
        qsum: i32,
        others: &[u32],
        params: &FilterSearchParams,
        top: &mut TopK,
    ) -> usize {
        let lists = ivf.centroid_norms.len();
        let mut order: Vec<(i64, u32)> = (0..lists)
            .map(|c| {
                let cent = &ivf.centroids[c * self.stride..(c + 1) * self.stride];
                let key =
                    i64::from(ivf.centroid_norms[c]) - 2 * i64::from(dot_i8_query(q, cent, qsum));
                (key, c as u32)
            })
            .collect();
        order.sort_unstable();
        let mut scored = 0usize;
        let mut batch: Vec<u32> = Vec::with_capacity(4096);
        let mut cursors = vec![0usize; others.len()];
        for (probed, &(_, c)) in order.iter().enumerate() {
            if scored >= params.target && probed >= params.min_lists {
                break;
            }
            let (a, b) = (
                ivf.offsets[c as usize] as usize,
                ivf.offsets[c as usize + 1] as usize,
            );
            let members = &ivf.members[a..b];
            cursors.fill(0);
            if ivf.rows.is_empty() {
                batch.clear();
                batch.extend(
                    members
                        .iter()
                        .copied()
                        .filter(|&id| !self.is_deleted(id) && self.holds(id, others, &mut cursors)),
                );
                self.score_ids(q, qsum, &batch, top);
                scored += batch.len();
            } else {
                // Positions (into the copied rows) of the passing members.
                batch.clear();
                batch.extend((a as u32..b as u32).filter(|&pos| {
                    let id = ivf.members[pos as usize];
                    !self.is_deleted(id) && self.holds(id, others, &mut cursors)
                }));
                self.score_copied(ivf, q, qsum, &batch, top);
                scored += batch.len();
            }
        }
        scored
    }

    /// [`Self::search`] for `nq` queries in parallel: `queries` row-major `nq * dim`, query
    /// labels as CSR (`label_indptr` with `nq + 1` entries). Writes ids to `out` (`nq * k`,
    /// -1 where fewer than `k` points match).
    pub fn search_batch(
        &self,
        queries: &[u8],
        label_indptr: &[u64],
        label_ids: &[u32],
        k: usize,
        params: &FilterSearchParams,
        out: &mut [i64],
    ) -> Result<(), FilterError> {
        if !queries.len().is_multiple_of(self.dim) {
            return Err(FilterError::Invalid("queries are not whole rows".into()));
        }
        let nq = queries.len() / self.dim;
        if label_indptr.len() != nq + 1
            || label_indptr.windows(2).any(|w| w[0] > w[1])
            || label_indptr[nq] as usize > label_ids.len()
        {
            return Err(FilterError::Invalid(
                "query label CSR does not match the queries".into(),
            ));
        }
        if out.len() != nq * k {
            return Err(FilterError::Invalid(format!(
                "out has {} slots, need {}",
                out.len(),
                nq * k
            )));
        }
        if k == 0 {
            return Ok(());
        }
        out.par_chunks_mut(k)
            .zip(queries.par_chunks(self.dim))
            .enumerate()
            .try_for_each_init(Vec::new, |found, (i, (row, q))| {
                let labels = &label_ids[label_indptr[i] as usize..label_indptr[i + 1] as usize];
                self.search(q, labels, k, params, found)?;
                row.fill(-1);
                for (slot, &(_, id)) in row.iter_mut().zip(found.iter()) {
                    *slot = i64::from(id);
                }
                Ok(())
            })
    }
}

/// k-means over the members of one label in the shifted i8 domain, then one assignment pass
/// over every member.
fn train_label(
    rows: &[i8],
    norms: &[i32],
    dim: usize,
    members: &[u32],
    params: &FilterBuildParams,
    salt: u64,
) -> LabelIvf {
    let lists = members
        .len()
        .div_ceil(params.cluster_size)
        .clamp(1, params.max_lists);
    let row = |id: u32| &rows[id as usize * dim..(id as usize + 1) * dim];
    let train_n = (lists * params.train_per_list).min(members.len());
    // Deterministic sample without replacement: a stride walk from a seeded start.
    let mut sample: Vec<u32> = Vec::with_capacity(train_n);
    let start = (mix(params.seed ^ salt) % members.len() as u64) as usize;
    let stride = members.len() as f64 / train_n as f64;
    for i in 0..train_n {
        sample.push(members[(start + (i as f64 * stride) as usize) % members.len()]);
    }
    let mut centroids: Vec<i8> = Vec::with_capacity(lists * dim);
    for c in 0..lists {
        let pick = sample[(c as f64 * train_n as f64 / lists as f64) as usize];
        centroids.extend_from_slice(row(pick));
    }
    let mut cnorms = centroid_norms(&centroids, dim);
    let assign = |x: &[i8], centroids: &[i8], cnorms: &[i32]| -> u32 {
        let qsum: i32 = x.iter().map(|&v| i32::from(v)).sum();
        let mut best = (i64::MAX, 0u32);
        for (c, cent) in centroids.chunks_exact(dim).enumerate() {
            let key = i64::from(cnorms[c]) - 2 * i64::from(dot_i8_query(x, cent, qsum));
            if key < best.0 {
                best = (key, c as u32);
            }
        }
        best.1
    };
    for _ in 0..params.kmeans_iters {
        let labels: Vec<u32> = sample
            .iter()
            .map(|&id| assign(row(id), &centroids, &cnorms))
            .collect();
        let mut sums = vec![0i64; lists * dim];
        let mut counts = vec![0u32; lists];
        for (&id, &c) in sample.iter().zip(&labels) {
            counts[c as usize] += 1;
            let s = &mut sums[c as usize * dim..(c as usize + 1) * dim];
            for (acc, &v) in s.iter_mut().zip(row(id)) {
                *acc += i64::from(v);
            }
        }
        for c in 0..lists {
            if counts[c] == 0 {
                continue;
            }
            let cnt = i64::from(counts[c]);
            for j in 0..dim {
                let s = sums[c * dim + j];
                // Round half away from zero.
                let m = if s >= 0 {
                    (2 * s + cnt) / (2 * cnt)
                } else {
                    -((-2 * s + cnt) / (2 * cnt))
                };
                centroids[c * dim + j] = m.clamp(-128, 127) as i8;
            }
        }
        cnorms = centroid_norms(&centroids, dim);
    }
    let assignment: Vec<u32> = if members.len() > 50_000 {
        members
            .par_iter()
            .map(|&id| assign(row(id), &centroids, &cnorms))
            .collect()
    } else {
        members
            .iter()
            .map(|&id| assign(row(id), &centroids, &cnorms))
            .collect()
    };
    let mut offsets = vec![0u32; lists + 1];
    for &c in &assignment {
        offsets[c as usize + 1] += 1;
    }
    for c in 0..lists {
        offsets[c + 1] += offsets[c];
    }
    let mut fill = offsets.clone();
    let mut grouped = vec![0u32; members.len()];
    for (&id, &c) in members.iter().zip(&assignment) {
        grouped[fill[c as usize] as usize] = id;
        fill[c as usize] += 1;
    }
    let (copy_rows, copy_norms) = if members.len() >= params.copy_min {
        let mut r = Vec::with_capacity(grouped.len() * dim);
        for &id in &grouped {
            r.extend_from_slice(row(id));
        }
        (r, grouped.iter().map(|&id| norms[id as usize]).collect())
    } else {
        (Vec::new(), Vec::new())
    };
    LabelIvf {
        centroids,
        centroid_norms: cnorms,
        offsets,
        members: grouped,
        rows: copy_rows,
        norms: copy_norms,
    }
}

fn centroid_norms(centroids: &[i8], dim: usize) -> Vec<i32> {
    centroids
        .chunks_exact(dim)
        .map(|c| c.iter().map(|&v| i32::from(v) * i32::from(v)).sum())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        data: Vec<u8>,
        indptr: Vec<u64>,
        ids: Vec<u32>,
        dim: usize,
    }

    /// Clustered u8 rows with a heavy-tailed label distribution; coordinates are drawn from a
    /// few values so exact distance ties occur.
    fn fixture(n: usize, dim: usize, labels: u32, seed: u64) -> Fixture {
        let mut s = seed;
        let mut next = || {
            s = mix(s);
            s
        };
        let centers: Vec<Vec<u8>> = (0..8)
            .map(|_| (0..dim).map(|_| (next() % 256) as u8).collect())
            .collect();
        let mut data = Vec::with_capacity(n * dim);
        let mut indptr = vec![0u64];
        let mut ids = Vec::new();
        for _ in 0..n {
            let c = &centers[(next() % 8) as usize];
            for &v in c {
                let noise = (next() % 5) as i32 * 8 - 16;
                data.push((i32::from(v) + noise).clamp(0, 255) as u8);
            }
            let tags = 1 + next() % 5;
            for _ in 0..tags {
                // Zipf-like: small label ids are common.
                let r = (next() % 10_000) as f64 / 10_000.0;
                ids.push(((r * r * r) * f64::from(labels)) as u32);
            }
            indptr.push(ids.len() as u64);
        }
        Fixture {
            data,
            indptr,
            ids,
            dim,
        }
    }

    fn brute(f: &Fixture, deleted: &[bool], q: &[u8], labels: &[u32], k: usize) -> Vec<(u32, u32)> {
        let n = f.data.len() / f.dim;
        let mut all: Vec<(u32, u32)> = (0..n)
            .filter(|&p| !deleted[p])
            .filter(|&p| {
                let tags = &f.ids[f.indptr[p] as usize..f.indptr[p + 1] as usize];
                labels.iter().all(|l| tags.contains(l))
            })
            .map(|p| {
                let row = &f.data[p * f.dim..(p + 1) * f.dim];
                let d: u32 = row
                    .iter()
                    .zip(q)
                    .map(|(&a, &b)| (i32::from(a) - i32::from(b)).pow(2) as u32)
                    .sum();
                (d, p as u32)
            })
            .collect();
        all.sort_unstable();
        all.truncate(k);
        all
    }

    fn queries(f: &Fixture, count: usize, labels: u32, seed: u64) -> Vec<(Vec<u8>, Vec<u32>)> {
        let mut s = seed;
        let mut next = || {
            s = mix(s);
            s
        };
        let n = f.data.len() / f.dim;
        (0..count)
            .map(|i| {
                let p = (next() % n as u64) as usize;
                let q: Vec<u8> = f.data[p * f.dim..(p + 1) * f.dim]
                    .iter()
                    .map(|&v| v.wrapping_add((next() % 3) as u8))
                    .collect();
                let tags = &f.ids[f.indptr[p] as usize..f.indptr[p + 1] as usize];
                let mut ls = vec![tags[(next() % tags.len() as u64) as usize]];
                if i % 2 == 1 {
                    ls.push((next() % u64::from(labels)) as u32);
                }
                (q, ls)
            })
            .collect()
    }

    fn small_params() -> FilterBuildParams {
        FilterBuildParams {
            ivf_min: 200,
            cluster_size: 50,
            max_lists: 64,
            bitset_min: 300,
            kmeans_iters: 4,
            train_per_list: 16,
            copy_min: 400,
            seed: 7,
        }
    }

    #[test]
    fn exact_setting_equals_filtered_brute_force() {
        let labels = 40;
        let f = fixture(4_000, 32, labels, 1);
        let idx =
            FilteredIndex::build(&f.data, f.dim, &f.indptr, &f.ids, 0, &small_params()).unwrap();
        assert!(
            idx.ivf_labels() > 0,
            "fixture must exercise the inverted-file path"
        );
        let deleted = vec![false; 4_000];
        let mut out = Vec::new();
        for (q, ls) in queries(&f, 300, labels, 2) {
            // brute_max 0 forces the inverted file for every label that has one.
            let ivf_exact = FilterSearchParams {
                brute_max: 0,
                ..FilterSearchParams::exact()
            };
            for p in [FilterSearchParams::exact(), ivf_exact] {
                idx.search(&q, &ls, 10, &p, &mut out).unwrap();
                assert_eq!(out, brute(&f, &deleted, &q, &ls, 10), "labels {ls:?}");
            }
        }
        let s = idx.stats();
        assert!(s.brute > 0 && s.ivf > 0, "{s:?}");
    }

    #[test]
    fn deletes_are_never_returned_and_rest_stays_exact() {
        let labels = 30;
        let f = fixture(3_000, 20, labels, 3);
        let mut idx =
            FilteredIndex::build(&f.data, f.dim, &f.indptr, &f.ids, 0, &small_params()).unwrap();
        let mut deleted = vec![false; 3_000];
        for id in (0..3_000).step_by(3) {
            assert!(idx.delete(id as u32));
            deleted[id] = true;
        }
        assert!(!idx.delete(0));
        assert_eq!(idx.live(), 2_000);
        let mut out = Vec::new();
        for (q, ls) in queries(&f, 200, labels, 4) {
            idx.search(
                &q,
                &ls,
                7,
                &FilterSearchParams {
                    brute_max: 0,
                    ..FilterSearchParams::exact()
                },
                &mut out,
            )
            .unwrap();
            assert_eq!(out, brute(&f, &deleted, &q, &ls, 7));
        }
    }

    #[test]
    fn batch_matches_single_and_pads_with_minus_one() {
        let labels = 25;
        let f = fixture(2_000, 16, labels, 5);
        let idx =
            FilteredIndex::build(&f.data, f.dim, &f.indptr, &f.ids, 0, &small_params()).unwrap();
        let qs = queries(&f, 64, labels, 6);
        let mut flat = Vec::new();
        let mut indptr = vec![0u64];
        let mut lids = Vec::new();
        for (q, ls) in &qs {
            flat.extend_from_slice(q);
            lids.extend_from_slice(ls);
            indptr.push(lids.len() as u64);
        }
        let k = 10;
        let p = FilterSearchParams::default();
        let mut out = vec![0i64; qs.len() * k];
        idx.search_batch(&flat, &indptr, &lids, k, &p, &mut out)
            .unwrap();
        let mut one = Vec::new();
        for (i, (q, ls)) in qs.iter().enumerate() {
            idx.search(q, ls, k, &p, &mut one).unwrap();
            let row = &out[i * k..(i + 1) * k];
            for (j, &got) in row.iter().enumerate() {
                assert_eq!(got, one.get(j).map_or(-1, |&(_, id)| i64::from(id)));
            }
        }
        // A label nobody holds and an out-of-range label both match nothing.
        idx.search(&qs[0].0, &[labels + 100], k, &p, &mut one)
            .unwrap();
        assert!(one.is_empty());
    }

    #[test]
    fn rejects_malformed_input() {
        let f = fixture(100, 16, 5, 9);
        let p = small_params();
        assert!(FilteredIndex::build(&f.data, 15, &f.indptr, &f.ids, 0, &p).is_err());
        assert!(FilteredIndex::build(&f.data, 16, &f.indptr[..50], &f.ids, 0, &p).is_err());
        assert!(FilteredIndex::build(&f.data, 16, &f.indptr, &f.ids, 2, &p).is_err());
        let idx = FilteredIndex::build(&f.data, 16, &f.indptr, &f.ids, 0, &p).unwrap();
        let mut out = Vec::new();
        assert!(idx
            .search(&[0u8; 8], &[0], 3, &FilterSearchParams::default(), &mut out)
            .is_err());
    }
}
