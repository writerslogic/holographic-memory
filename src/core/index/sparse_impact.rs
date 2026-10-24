// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Inner-product top-k over learned sparse vectors (SPLADE-style), with a blocked inverted
//! index and a forward index.
//!
//! The layout follows the blocked inverted index of SEISMIC (Bruch et al., arXiv 2404.18812):
//!
//! - every coordinate keeps its `postings_per_term` largest postings (static pruning);
//! - each kept list is split into blocks by assigning every document to the most similar of a
//!   random sample of the list's documents (shallow clustering);
//! - every block stores a summary: the coordinate-wise maximum of its documents, pruned to the
//!   coordinates that carry `summary_energy` of its mass and quantized to one byte;
//! - a forward index holds every document in full, so a candidate's score is its exact inner
//!   product with the query.
//!
//! A search walks the lists of the `query_cut` heaviest query coordinates and scores a block's
//! documents only when the summary's inner product with the query, times `heap_factor`, reaches
//! the current k-th best score.
//!
//! With [`SparseBuildParams::exact`] and [`SparseSearchParams::exact`] (no static pruning, full
//! summaries rounded up, every query coordinate, `heap_factor` 1) a summary bounds the score of
//! every document in its block, so a skipped block cannot hold a result and the search returns
//! the exact top-k, ties broken by the lower document id. That needs non-negative document
//! values, which the build enforces; negative query weights only lower a score, so the bound
//! uses the positive part of the query.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

/// Largest supported coordinate count: coordinate ids are stored in two bytes.
pub const MAX_DIMS: usize = 1 << 16;

/// Errors from building a [`SparseImpactIndex`] or validating a query.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseIndexError {
    /// The CSR arrays are inconsistent (lengths, offsets or an out-of-range coordinate).
    Malformed(String),
    /// A document value is negative or not finite.
    BadValue { doc: usize, value: f32 },
    /// More coordinates or documents than the format holds.
    TooLarge(String),
}

impl std::fmt::Display for SparseIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(m) => write!(f, "malformed sparse matrix: {m}"),
            Self::BadValue { doc, value } => {
                write!(
                    f,
                    "document {doc} has value {value}; values must be finite and >= 0"
                )
            }
            Self::TooLarge(m) => write!(f, "sparse matrix too large: {m}"),
        }
    }
}

impl std::error::Error for SparseIndexError {}

/// A borrowed CSR matrix: row `r` holds `indices[indptr[r]..indptr[r + 1]]` with `data` values.
#[derive(Clone, Copy)]
pub struct CsrView<'a> {
    pub indptr: &'a [i64],
    pub indices: &'a [i32],
    pub data: &'a [f32],
    pub n_dims: usize,
}

impl<'a> CsrView<'a> {
    /// Reads the big-ann-benchmarks `spmat` layout: int64 rows, cols and nnz, then int64
    /// indptr, int32 indices and float32 data. `bytes` must be 8-byte aligned (a memory map is).
    pub fn from_spmat(bytes: &'a [u8]) -> Result<Self, SparseIndexError> {
        let bad = |m: String| SparseIndexError::Malformed(m);
        let header: &[i64] = bytes
            .get(..24)
            .and_then(|h| bytemuck::try_cast_slice(h).ok())
            .ok_or_else(|| bad("spmat header is truncated or misaligned".into()))?;
        let size = |v: i64| usize::try_from(v).map_err(|_| bad(format!("negative size {v}")));
        let (rows, cols, nnz) = (size(header[0])?, size(header[1])?, size(header[2])?);
        let span = |n: usize, w: usize| {
            n.checked_mul(w)
                .ok_or_else(|| bad("spmat size overflows".into()))
        };
        let indptr_end = 24 + span(rows + 1, 8)?;
        let indices_end = indptr_end + span(nnz, 4)?;
        let data_end = indices_end + span(nnz, 4)?;
        if bytes.len() != data_end {
            return Err(bad(format!(
                "file is {} bytes, header implies {data_end}",
                bytes.len()
            )));
        }
        let cast_err = |e: bytemuck::PodCastError| bad(format!("misaligned section: {e:?}"));
        Ok(Self {
            indptr: bytemuck::try_cast_slice(&bytes[24..indptr_end]).map_err(cast_err)?,
            indices: bytemuck::try_cast_slice(&bytes[indptr_end..indices_end]).map_err(cast_err)?,
            data: bytemuck::try_cast_slice(&bytes[indices_end..data_end]).map_err(cast_err)?,
            n_dims: cols,
        })
    }

    /// Checks lengths, monotone offsets and coordinate range.
    pub fn validate(&self) -> Result<usize, SparseIndexError> {
        let rows = self
            .indptr
            .len()
            .checked_sub(1)
            .ok_or_else(|| SparseIndexError::Malformed("indptr is empty".into()))?;
        if self.indices.len() != self.data.len() {
            return Err(SparseIndexError::Malformed(format!(
                "{} indices but {} values",
                self.indices.len(),
                self.data.len()
            )));
        }
        if self.indptr[0] != 0 || self.indptr[rows] != self.indices.len() as i64 {
            return Err(SparseIndexError::Malformed(
                "indptr must start at 0 and end at nnz".into(),
            ));
        }
        if self.indptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(SparseIndexError::Malformed("indptr decreases".into()));
        }
        if let Some(&bad) = self
            .indices
            .iter()
            .find(|&&c| c < 0 || c as usize >= self.n_dims)
        {
            return Err(SparseIndexError::Malformed(format!(
                "coordinate {bad} outside [0, {})",
                self.n_dims
            )));
        }
        Ok(rows)
    }

    /// Coordinates and values of row `r`.
    pub fn row(&self, r: usize) -> (&'a [i32], &'a [f32]) {
        let (a, b) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
        (&self.indices[a..b], &self.data[a..b])
    }
}

/// Build-time parameters.
#[derive(Clone, Copy, Debug)]
pub struct SparseBuildParams {
    /// Postings kept per coordinate (the largest values); `usize::MAX` keeps every posting.
    pub postings_per_term: usize,
    /// Blocks per kept list as a fraction of its length (at least one block).
    pub block_fraction: f32,
    /// Fraction of a summary's mass its kept coordinates must carry; 1 keeps all of them.
    pub summary_energy: f32,
    /// Postings transposed at once during the build (bounds the build's working memory).
    pub transpose_budget: usize,
    /// Seed of the centroid sample.
    pub seed: u64,
    /// Store forward values as half floats (2 bytes, relative error below 2^-11); scores are
    /// then exact inner products with the rounded values.
    pub half_values: bool,
}

impl SparseBuildParams {
    /// Parameters under which a search with [`SparseSearchParams::exact`] is exact.
    pub fn exact() -> Self {
        Self {
            postings_per_term: usize::MAX,
            summary_energy: 1.0,
            half_values: false,
            ..Self::default()
        }
    }
}

impl Default for SparseBuildParams {
    fn default() -> Self {
        Self {
            postings_per_term: 4000,
            block_fraction: 0.1,
            summary_energy: 0.4,
            transpose_budget: 1 << 28,
            seed: 0x5eed,
            half_values: true,
        }
    }
}

/// Query-time parameters.
#[derive(Clone, Copy, Debug)]
pub struct SparseSearchParams {
    /// Query coordinates (heaviest first) whose lists are walked.
    pub query_cut: usize,
    /// A block is scored when `summary score * heap_factor` reaches the k-th best score.
    pub heap_factor: f32,
}

impl SparseSearchParams {
    pub fn exact() -> Self {
        Self {
            query_cut: usize::MAX,
            heap_factor: 1.0,
        }
    }
}

/// Blocked inverted index with summaries plus a forward index.
pub struct SparseImpactIndex {
    n_dims: usize,
    n_docs: usize,
    /// Forward index: document `d` is `fwd_ids/fwd_vals[fwd_offsets[d]..fwd_offsets[d + 1]]`.
    fwd_offsets: Vec<u64>,
    fwd_ids: Vec<u16>,
    fwd_vals: Vec<f32>,
    /// The forward values as IEEE half floats when [`SparseBuildParams::half_values`] is set
    /// (`fwd_vals` is then empty).
    fwd_half: Vec<u16>,
    /// Blocks of coordinate `t`: `term_blocks[t]..term_blocks[t + 1]`.
    term_blocks: Vec<u32>,
    /// Documents of block `b`: `block_docs[block_doc_offsets[b]..block_doc_offsets[b + 1]]`.
    block_doc_offsets: Vec<u64>,
    block_docs: Vec<u32>,
    /// Summary of block `b`: `sum_ids/sum_codes[block_sum_offsets[b]..block_sum_offsets[b + 1]]`,
    /// a coordinate's maximum being at most `sum_codes * block_scales[b]`.
    block_sum_offsets: Vec<u64>,
    sum_ids: Vec<u16>,
    sum_codes: Vec<u8>,
    block_scales: Vec<f32>,
}

/// One kept list after clustering: its blocks' documents and summaries.
struct TermBlocks {
    doc_counts: Vec<u32>,
    docs: Vec<u32>,
    sum_counts: Vec<u32>,
    sum_ids: Vec<u16>,
    sum_codes: Vec<u8>,
    scales: Vec<f32>,
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Per-thread scratch for the build: dense maxima and a coordinate-to-centroid map.
struct BuildScratch {
    dense: Vec<f32>,
    touched: Vec<u16>,
    starts: Vec<u32>,
}

impl BuildScratch {
    fn new(n_dims: usize) -> Self {
        Self {
            dense: vec![0.0; n_dims],
            touched: Vec::new(),
            starts: vec![u32::MAX; n_dims + 1],
        }
    }
}

impl SparseImpactIndex {
    /// Builds the index over the rows of `csr`. Zero values are dropped.
    pub fn build(csr: CsrView<'_>, params: SparseBuildParams) -> Result<Self, SparseIndexError> {
        let n_docs = csr.validate()?;
        if csr.n_dims > MAX_DIMS {
            return Err(SparseIndexError::TooLarge(format!(
                "{} coordinates, at most {MAX_DIMS}",
                csr.n_dims
            )));
        }
        if n_docs > u32::MAX as usize {
            return Err(SparseIndexError::TooLarge(format!("{n_docs} documents")));
        }
        if let Some(i) = csr.data.iter().position(|v| !(v.is_finite() && *v >= 0.0)) {
            let doc = csr.indptr.partition_point(|&p| p <= i as i64) - 1;
            return Err(SparseIndexError::BadValue {
                doc,
                value: csr.data[i],
            });
        }
        let n_dims = csr.n_dims;

        // Forward index (rows keep their order; zeros dropped).
        let mut fwd_offsets = Vec::with_capacity(n_docs + 1);
        fwd_offsets.push(0u64);
        let nnz = csr.data.iter().filter(|&&v| v > 0.0).count();
        let mut fwd_ids = Vec::with_capacity(nnz);
        let mut fwd_vals = Vec::with_capacity(nnz);
        let mut df = vec![0u64; n_dims];
        for r in 0..n_docs {
            let (ids, vals) = csr.row(r);
            for (&c, &v) in ids.iter().zip(vals) {
                if v > 0.0 {
                    fwd_ids.push(c as u16);
                    fwd_vals.push(v);
                    df[c as usize] += 1;
                }
            }
            fwd_offsets.push(fwd_ids.len() as u64);
        }

        let mut index = Self {
            n_dims,
            n_docs,
            fwd_offsets,
            fwd_ids,
            fwd_vals,
            fwd_half: Vec::new(),
            term_blocks: Vec::with_capacity(n_dims + 1),
            block_doc_offsets: vec![0],
            block_docs: Vec::new(),
            block_sum_offsets: vec![0],
            sum_ids: Vec::new(),
            sum_codes: Vec::new(),
            block_scales: Vec::new(),
        };
        index.term_blocks.push(0);

        // Transpose a range of coordinates at a time, then cluster each list in parallel.
        let budget = params.transpose_budget.max(1) as u64;
        let mut lo = 0usize;
        while lo < n_dims {
            let mut hi = lo;
            let mut total = 0u64;
            while hi < n_dims && (hi == lo || total + df[hi] <= budget) {
                total += df[hi];
                hi += 1;
            }
            let mut starts = Vec::with_capacity(hi - lo + 1);
            starts.push(0u64);
            for &d in &df[lo..hi] {
                starts.push(starts.last().copied().unwrap_or(0) + d);
            }
            let mut cursor = starts.clone();
            let mut postings = vec![(0u32, 0f32); total as usize];
            for d in 0..n_docs {
                let (ids, vals) = index.row(d as u32);
                for (&c, &v) in ids.iter().zip(vals) {
                    let c = c as usize;
                    if (lo..hi).contains(&c) {
                        let slot = &mut cursor[c - lo];
                        postings[*slot as usize] = (d as u32, v);
                        *slot += 1;
                    }
                }
            }
            let lists: Vec<&mut [(u32, f32)]> = {
                let mut rest = postings.as_mut_slice();
                let mut out = Vec::with_capacity(hi - lo);
                for t in 0..hi - lo {
                    let (head, tail) = rest.split_at_mut((starts[t + 1] - starts[t]) as usize);
                    out.push(head);
                    rest = tail;
                }
                out
            };
            let this = &index;
            let built: Vec<TermBlocks> = lists
                .into_par_iter()
                .enumerate()
                .map_init(
                    || BuildScratch::new(n_dims),
                    |scratch, (i, list)| this.cluster_list(lo + i, list, params, scratch),
                )
                .collect();
            for tb in built {
                index.append(tb);
            }
            lo = hi;
        }
        if params.half_values {
            index.fwd_half = index.fwd_vals.par_iter().map(|&v| f16_bits(v)).collect();
            index.fwd_vals = Vec::new();
        }
        Ok(index)
    }

    fn row(&self, d: u32) -> (&[u16], &[f32]) {
        let (a, b) = (
            self.fwd_offsets[d as usize] as usize,
            self.fwd_offsets[d as usize + 1] as usize,
        );
        (&self.fwd_ids[a..b], &self.fwd_vals[a..b])
    }

    /// Inner product of document `d` (after `build`) with a dense query.
    #[inline]
    fn score(&self, d: u32, dense: &[f32]) -> f32 {
        let (a, b) = (
            self.fwd_offsets[d as usize] as usize,
            self.fwd_offsets[d as usize + 1] as usize,
        );
        if self.fwd_half.is_empty() {
            dot(&self.fwd_ids[a..b], &self.fwd_vals[a..b], dense)
        } else {
            dot_half(&self.fwd_ids[a..b], &self.fwd_half[a..b], dense)
        }
    }

    /// Hint the cache to load the forward row of document `d`.
    #[inline]
    fn prefetch_row(&self, d: u32) {
        let (a, b) = (
            self.fwd_offsets[d as usize] as usize,
            self.fwd_offsets[d as usize + 1] as usize,
        );
        prefetch(self.fwd_ids[a..].as_ptr().cast(), (b - a) * 2);
        if self.fwd_half.is_empty() {
            prefetch(self.fwd_vals[a..].as_ptr().cast(), (b - a) * 4);
        } else {
            prefetch(self.fwd_half[a..].as_ptr().cast(), (b - a) * 2);
        }
    }

    fn append(&mut self, tb: TermBlocks) {
        let mut doc_at = 0usize;
        let mut sum_at = 0usize;
        for ((&dc, &sc), &scale) in tb.doc_counts.iter().zip(&tb.sum_counts).zip(&tb.scales) {
            let (dc, sc) = (dc as usize, sc as usize);
            self.block_docs
                .extend_from_slice(&tb.docs[doc_at..doc_at + dc]);
            self.block_doc_offsets.push(self.block_docs.len() as u64);
            self.sum_ids
                .extend_from_slice(&tb.sum_ids[sum_at..sum_at + sc]);
            self.sum_codes
                .extend_from_slice(&tb.sum_codes[sum_at..sum_at + sc]);
            self.block_sum_offsets.push(self.sum_ids.len() as u64);
            self.block_scales.push(scale);
            doc_at += dc;
            sum_at += sc;
        }
        let blocks = self.block_scales.len() as u32;
        self.term_blocks.push(blocks);
    }

    /// Prunes, clusters and summarizes the list of coordinate `term`.
    fn cluster_list(
        &self,
        term: usize,
        list: &mut [(u32, f32)],
        params: SparseBuildParams,
        scratch: &mut BuildScratch,
    ) -> TermBlocks {
        let mut out = TermBlocks {
            doc_counts: Vec::new(),
            docs: Vec::new(),
            sum_counts: Vec::new(),
            sum_ids: Vec::new(),
            sum_codes: Vec::new(),
            scales: Vec::new(),
        };
        if list.is_empty() {
            return out;
        }
        let by_value_desc =
            |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        let keep = params.postings_per_term.min(list.len());
        if keep == 0 {
            return out;
        }
        if keep < list.len() {
            list.select_nth_unstable_by(keep - 1, by_value_desc);
        }
        let list = &mut list[..keep];

        let n_blocks = ((keep as f64 * f64::from(params.block_fraction)).ceil() as usize)
            .max(1)
            .min(keep);
        let assign: Vec<u32> = if n_blocks == 1 {
            vec![0; keep]
        } else {
            self.assign_to_centroids(term, list, n_blocks, params.seed, scratch)
        };

        // Group by block, documents ascending within a block.
        let mut order: Vec<u32> = (0..keep as u32).collect();
        order.sort_unstable_by_key(|&i| (assign[i as usize], list[i as usize].0));
        let mut groups: Vec<(f32, &[u32])> = Vec::with_capacity(n_blocks);
        let mut start = 0;
        while start < keep {
            let b = assign[order[start] as usize];
            let mut end = start;
            let mut top = 0f32;
            while end < keep && assign[order[end] as usize] == b {
                top = top.max(list[order[end] as usize].1);
                end += 1;
            }
            groups.push((top, &order[start..end]));
            start = end;
        }
        // Blocks with the largest value of this coordinate first: the threshold rises early.
        groups.sort_by(|a, b| b.0.total_cmp(&a.0));

        for (_, members) in groups {
            for &i in members {
                out.docs.push(list[i as usize].0);
            }
            out.doc_counts.push(members.len() as u32);
            self.summarize(
                members.iter().map(|&i| list[i as usize].0),
                params.summary_energy,
                scratch,
                &mut out,
            );
        }
        out
    }

    /// Picks `n_blocks` documents of `list` as centroids and assigns every document to the
    /// centroid with the largest inner product (ties to the lower centroid).
    fn assign_to_centroids(
        &self,
        term: usize,
        list: &[(u32, f32)],
        n_blocks: usize,
        seed: u64,
        scratch: &mut BuildScratch,
    ) -> Vec<u32> {
        let mut state = seed ^ (term as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
        // Partial Fisher-Yates over positions.
        let mut pos: Vec<u32> = (0..list.len() as u32).collect();
        for i in 0..n_blocks {
            let j = i + (splitmix(&mut state) % (list.len() - i) as u64) as usize;
            pos.swap(i, j);
        }
        // Inverted map of the centroids: coordinate -> (centroid, value).
        let mut entries: Vec<(u16, u32, f32)> = Vec::new();
        for (c, &p) in pos[..n_blocks].iter().enumerate() {
            let (ids, vals) = self.row(list[p as usize].0);
            entries.extend(ids.iter().zip(vals).map(|(&t, &v)| (t, c as u32, v)));
        }
        entries.sort_unstable_by_key(|e| (e.0, e.1));
        let starts = &mut scratch.starts;
        let mut touched = Vec::new();
        for (i, e) in entries.iter().enumerate() {
            let t = e.0 as usize;
            if starts[t] == u32::MAX {
                starts[t] = i as u32;
                touched.push(t);
            }
        }
        let mut scores = vec![0f32; n_blocks];
        let mut assign = Vec::with_capacity(list.len());
        for &(doc, _) in list {
            scores.iter_mut().for_each(|s| *s = 0.0);
            let (ids, vals) = self.row(doc);
            for (&t, &v) in ids.iter().zip(vals) {
                let s = starts[t as usize];
                if s == u32::MAX {
                    continue;
                }
                for e in entries[s as usize..].iter().take_while(|e| e.0 == t) {
                    scores[e.1 as usize] += v * e.2;
                }
            }
            let mut best = 0usize;
            for (c, &s) in scores.iter().enumerate() {
                if s > scores[best] {
                    best = c;
                }
            }
            assign.push(best as u32);
        }
        for t in touched {
            starts[t] = u32::MAX;
        }
        assign
    }

    /// Appends the summary of `docs`: coordinate maxima pruned to `energy` of their mass,
    /// quantized upward to one byte.
    fn summarize(
        &self,
        docs: impl Iterator<Item = u32>,
        energy: f32,
        scratch: &mut BuildScratch,
        out: &mut TermBlocks,
    ) {
        let BuildScratch { dense, touched, .. } = scratch;
        for d in docs {
            let (ids, vals) = self.row(d);
            for (&t, &v) in ids.iter().zip(vals) {
                let slot = &mut dense[t as usize];
                if *slot == 0.0 {
                    touched.push(t);
                }
                *slot = slot.max(v);
            }
        }
        let mut coords: Vec<(u16, f32)> = touched.iter().map(|&t| (t, dense[t as usize])).collect();
        for &t in touched.iter() {
            dense[t as usize] = 0.0;
        }
        touched.clear();
        if energy < 1.0 {
            coords.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let total: f64 = coords.iter().map(|c| f64::from(c.1)).sum();
            let target = total * f64::from(energy.max(0.0));
            let mut acc = 0f64;
            let mut keep = 0;
            while keep < coords.len() && (keep == 0 || acc < target) {
                acc += f64::from(coords[keep].1);
                keep += 1;
            }
            coords.truncate(keep);
        }
        coords.sort_unstable_by_key(|c| c.0);
        let max = coords.iter().fold(0f32, |m, c| m.max(c.1));
        // Above max / 255 so that the bound survives rounding in the summed inner products.
        let scale = if max > 0.0 {
            max / 255.0 * (1.0 + 1e-4)
        } else {
            1.0
        };
        for &(t, v) in &coords {
            let mut code = (v / scale).ceil().min(255.0) as u8;
            while code < 255 && f32::from(code) * scale < v {
                code += 1;
            }
            out.sum_ids.push(t);
            out.sum_codes.push(code);
        }
        out.sum_counts.push(coords.len() as u32);
        out.scales.push(scale);
    }

    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    pub fn n_dims(&self) -> usize {
        self.n_dims
    }

    /// Bytes held by the index arrays.
    pub fn memory_bytes(&self) -> usize {
        self.fwd_offsets.len() * 8
            + self.fwd_ids.len() * 2
            + self.fwd_vals.len() * 4
            + self.fwd_half.len() * 2
            + self.term_blocks.len() * 4
            + self.block_doc_offsets.len() * 8
            + self.block_docs.len() * 4
            + self.block_sum_offsets.len() * 8
            + self.sum_ids.len() * 2
            + self.sum_codes.len()
            + self.block_scales.len() * 4
    }

    /// Number of blocks and kept postings.
    pub fn block_stats(&self) -> (usize, usize) {
        (self.block_scales.len(), self.block_docs.len())
    }

    /// Scratch for [`SparseSearcher::search`]; one per thread.
    pub fn searcher(&self) -> SparseSearcher<'_> {
        SparseSearcher {
            index: self,
            dense: vec![0.0; self.n_dims],
            dense_pos: vec![0.0; self.n_dims],
            visited: vec![0; self.n_docs.div_ceil(64)],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// The exact top-k by scanning the forward index (reference for tests and ground truth).
    pub fn brute_force(&self, ids: &[u32], vals: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut dense = vec![0f32; self.n_dims];
        for (&c, &v) in ids.iter().zip(vals) {
            if (c as usize) < self.n_dims {
                dense[c as usize] += v;
            }
        }
        let mut heap = BinaryHeap::new();
        for d in 0..self.n_docs as u32 {
            let s = self.score(d, &dense);
            if s > 0.0 {
                offer(&mut heap, k, Hit { score: s, doc: d });
            }
        }
        into_sorted(heap)
    }
}

#[inline]
fn dot(ids: &[u16], vals: &[f32], dense: &[f32]) -> f32 {
    let mut acc = [0f32; 4];
    let (id4, id_rest) = ids.as_chunks::<4>();
    let (v4, v_rest) = vals.as_chunks::<4>();
    for (i, v) in id4.iter().zip(v4) {
        for j in 0..4 {
            acc[j] += dense[i[j] as usize] * v[j];
        }
    }
    let mut tail = 0f32;
    for (&i, &v) in id_rest.iter().zip(v_rest) {
        tail += dense[i as usize] * v;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// Hint the cache to load `bytes` bytes at `ptr`. A no-op where no prefetch is wired up.
#[inline]
fn prefetch(ptr: *const u8, bytes: usize) {
    let end = ptr as usize + bytes;
    let mut line = ptr as usize & !63;
    while line < end {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: PRFM is a hint: it never faults and writes no register or memory, whatever
        // the address.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) line,
                options(nostack, preserves_flags, readonly)
            );
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: PREFETCHT0 (SSE, part of the x86_64 baseline) is a hint: it never faults
        // and writes no register or memory, whatever the address.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                line as *const i8,
            );
        }
        line += 64;
    }
}

/// IEEE binary16 bits of a non-negative finite `v`, rounded to nearest; values below the
/// smallest normal half become that normal (`v > 0`) or zero, values above the largest
/// become the largest.
fn f16_bits(v: f32) -> u16 {
    const MIN_NORMAL: f32 = 6.103_515_6e-5;
    if v <= 0.0 {
        return 0;
    }
    let v = v.clamp(MIN_NORMAL, 65504.0);
    let bits = v.to_bits() - ((127 - 15) << 23);
    let rounded = bits + 0x0fff + ((bits >> 13) & 1);
    ((rounded >> 13).min(0x7bff)) as u16
}

/// Value of half bits produced by [`f16_bits`] (zero or normal).
#[inline]
fn f16_value(h: u16) -> f32 {
    let bits = (u32::from(h) << 13) + ((127 - 15) << 23);
    if h == 0 {
        0.0
    } else {
        f32::from_bits(bits)
    }
}

#[inline]
fn dot_half(ids: &[u16], vals: &[u16], dense: &[f32]) -> f32 {
    let mut acc = [0f32; 4];
    let (id4, id_rest) = ids.as_chunks::<4>();
    let (v4, v_rest) = vals.as_chunks::<4>();
    for (i, v) in id4.iter().zip(v4) {
        for j in 0..4 {
            acc[j] += dense[i[j] as usize] * f16_value(v[j]);
        }
    }
    let mut tail = 0f32;
    for (&i, &v) in id_rest.iter().zip(v_rest) {
        tail += dense[i as usize] * f16_value(v);
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// A scored document; `Ord` puts the worst result (lowest score, then highest id) on top of
/// the max-heap so it is the one evicted.
#[derive(Clone, Copy, PartialEq)]
struct Hit {
    score: f32,
    doc: u32,
}

impl Eq for Hit {}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.doc.cmp(&other.doc))
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn offer(heap: &mut BinaryHeap<Hit>, k: usize, hit: Hit) {
    if heap.len() < k {
        heap.push(hit);
    } else if let Some(mut worst) = heap.peek_mut() {
        if hit < *worst {
            *worst = hit;
        }
    }
}

fn into_sorted(heap: BinaryHeap<Hit>) -> Vec<(u32, f32)> {
    heap.into_sorted_vec()
        .into_iter()
        .map(|h| (h.doc, h.score))
        .collect()
}

/// Per-thread search state.
pub struct SparseSearcher<'a> {
    index: &'a SparseImpactIndex,
    dense: Vec<f32>,
    dense_pos: Vec<f32>,
    visited: Vec<u64>,
    touched: Vec<u32>,
    heap: BinaryHeap<Hit>,
}

impl SparseSearcher<'_> {
    /// The top `k` documents by inner product with the query `(ids, vals)`, best first, as
    /// `(doc, score)`; documents scoring zero or less are never returned. Duplicate query
    /// coordinates add up; coordinates outside the index are ignored.
    pub fn search(
        &mut self,
        ids: &[u32],
        vals: &[f32],
        k: usize,
        params: SparseSearchParams,
    ) -> Vec<(u32, f32)> {
        let index = self.index;
        if k == 0 {
            return Vec::new();
        }
        let mut terms: Vec<(u32, f32)> = Vec::with_capacity(ids.len());
        for (&c, &v) in ids.iter().zip(vals) {
            if (c as usize) < index.n_dims && v.is_finite() && v != 0.0 {
                if self.dense[c as usize] == 0.0 && self.dense_pos[c as usize] == 0.0 {
                    terms.push((c, 0.0));
                }
                self.dense[c as usize] += v;
            }
        }
        for t in terms.iter_mut() {
            let w = self.dense[t.0 as usize];
            t.1 = w;
            self.dense_pos[t.0 as usize] = w.max(0.0);
        }
        terms.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let walked = terms
            .iter()
            .take(params.query_cut)
            .take_while(|t| t.1 > 0.0);

        let heap = &mut self.heap;
        heap.clear();
        for &(t, _) in walked {
            let blocks =
                index.term_blocks[t as usize] as usize..index.term_blocks[t as usize + 1] as usize;
            for b in blocks {
                if heap.len() == k {
                    let (s0, s1) = (
                        index.block_sum_offsets[b] as usize,
                        index.block_sum_offsets[b + 1] as usize,
                    );
                    let mut bound = 0f32;
                    for (&c, &code) in index.sum_ids[s0..s1].iter().zip(&index.sum_codes[s0..s1]) {
                        bound += self.dense_pos[c as usize] * f32::from(code);
                    }
                    bound *= index.block_scales[b];
                    let threshold = heap.peek().map_or(f32::MIN, |h| h.score);
                    if bound * params.heap_factor < threshold {
                        continue;
                    }
                }
                let (d0, d1) = (
                    index.block_doc_offsets[b] as usize,
                    index.block_doc_offsets[b + 1] as usize,
                );
                let docs = &index.block_docs[d0..d1];
                if let Some(&first) = docs.first() {
                    index.prefetch_row(first);
                }
                for (i, &d) in docs.iter().enumerate() {
                    if let Some(&next) = docs.get(i + 1) {
                        index.prefetch_row(next);
                    }
                    let (w, bit) = ((d >> 6) as usize, 1u64 << (d & 63));
                    if self.visited[w] & bit != 0 {
                        continue;
                    }
                    if self.visited[w] == 0 {
                        self.touched.push(w as u32);
                    }
                    self.visited[w] |= bit;
                    let s = index.score(d, &self.dense);
                    if s > 0.0 {
                        offer(heap, k, Hit { score: s, doc: d });
                    }
                }
            }
        }
        for &w in &self.touched {
            self.visited[w as usize] = 0;
        }
        self.touched.clear();
        for t in &terms {
            self.dense[t.0 as usize] = 0.0;
            self.dense_pos[t.0 as usize] = 0.0;
        }
        into_sorted(std::mem::take(heap))
    }
}

impl SparseImpactIndex {
    /// Searches the CSR query rows in parallel; row `i` of the result holds query `i`'s ids,
    /// best first, padded with `u32::MAX`.
    pub fn search_batch(
        &self,
        queries: CsrView<'_>,
        k: usize,
        params: SparseSearchParams,
    ) -> Result<Vec<u32>, SparseIndexError> {
        let nq = queries.validate()?;
        let mut out = vec![u32::MAX; nq * k];
        if k == 0 {
            return Ok(out);
        }
        out.par_chunks_mut(k).enumerate().for_each_init(
            || (self.searcher(), Vec::new()),
            |(searcher, ids), (q, row)| {
                let (qi, qv) = queries.row(q);
                ids.clear();
                ids.extend(qi.iter().map(|&c| c as u32));
                for (slot, (doc, _)) in row.iter_mut().zip(searcher.search(ids, qv, k, params)) {
                    *slot = doc;
                }
            },
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Csr {
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
        n_dims: usize,
    }

    impl Csr {
        fn view(&self) -> CsrView<'_> {
            CsrView {
                indptr: &self.indptr,
                indices: &self.indices,
                data: &self.data,
                n_dims: self.n_dims,
            }
        }
    }

    /// Random rows with a skewed coordinate distribution; values on a coarse grid so that ties
    /// occur, some rows empty and some duplicated.
    fn random_csr(rows: usize, n_dims: usize, max_nnz: usize, state: &mut u64) -> Csr {
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in 0..rows {
            if r % 17 == 5 && r > 0 {
                let (a, b) = (indptr[r - 1] as usize, indptr[r] as usize);
                indices.extend_from_within(a..b);
                data.extend_from_within(a..b);
            } else {
                let nnz = (splitmix(state) % (max_nnz as u64 + 1)) as usize;
                let mut cols: Vec<i32> = (0..nnz)
                    .map(|_| {
                        let u = splitmix(state) % (n_dims as u64 * n_dims as u64);
                        (u as f64).sqrt() as i32
                    })
                    .collect();
                cols.sort_unstable();
                cols.dedup();
                for c in cols {
                    indices.push(c);
                    data.push((1 + splitmix(state) % 8) as f32 * 0.25);
                }
            }
            indptr.push(indices.len() as i64);
        }
        Csr {
            indptr,
            indices,
            data,
            n_dims,
        }
    }

    #[test]
    fn exact_mode_matches_brute_force() {
        let mut state = 7u64;
        for (rows, n_dims, frac) in [(600, 64, 0.1f32), (2000, 300, 0.05), (50, 16, 0.5)] {
            let docs = random_csr(rows, n_dims, 24, &mut state);
            let params = SparseBuildParams {
                block_fraction: frac,
                transpose_budget: 997,
                ..SparseBuildParams::exact()
            };
            let index = SparseImpactIndex::build(docs.view(), params).unwrap();
            let queries = random_csr(80, n_dims + 4, 12, &mut state);
            let mut searcher = index.searcher();
            for q in 0..80 {
                let (a, b) = (queries.indptr[q] as usize, queries.indptr[q + 1] as usize);
                let ids: Vec<u32> = queries.indices[a..b].iter().map(|&c| c as u32).collect();
                let mut vals = queries.data[a..b].to_vec();
                if q % 9 == 0 && !vals.is_empty() {
                    vals[0] = -vals[0];
                }
                for k in [1, 10, 37] {
                    let want = index.brute_force(&ids, &vals, k);
                    let got = searcher.search(&ids, &vals, k, SparseSearchParams::exact());
                    assert_eq!(got, want, "rows {rows} query {q} k {k}");
                }
            }
        }
    }

    #[test]
    fn approximate_mode_returns_true_scores() {
        let mut state = 11u64;
        let docs = random_csr(3000, 200, 30, &mut state);
        let index = SparseImpactIndex::build(
            docs.view(),
            SparseBuildParams {
                postings_per_term: 100,
                ..SparseBuildParams::default()
            },
        )
        .unwrap();
        let params = SparseSearchParams {
            query_cut: 5,
            heap_factor: 0.8,
        };
        let mut searcher = index.searcher();
        let (ids, vals) = ([3u32, 40, 41, 150], [1.0f32, 0.5, 0.75, 2.0]);
        let got = searcher.search(&ids, &vals, 10, params);
        assert!(!got.is_empty());
        let mut dense = vec![0f32; 200];
        for (&c, &v) in ids.iter().zip(&vals) {
            dense[c as usize] = v;
        }
        for w in got.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
        for &(d, s) in &got {
            assert_eq!(s, index.score(d, &dense));
        }
        let batch = index
            .search_batch(
                CsrView {
                    indptr: &[0, 4],
                    indices: &[3, 40, 41, 150],
                    data: &vals,
                    n_dims: 200,
                },
                10,
                params,
            )
            .unwrap();
        let ids_only: Vec<u32> = got.iter().map(|h| h.0).collect();
        assert_eq!(&batch[..ids_only.len()], ids_only.as_slice());
    }

    #[test]
    fn half_values_round_to_nearest() {
        for v in [
            6.2e-5f32, 0.001, 0.1, 0.333, 1.0, 1.5, 2.75, 7.9, 1000.0, 65504.0,
        ] {
            let h = f16_value(f16_bits(v));
            assert!((h - v).abs() <= v * (1.0 / 2048.0), "{v} -> {h}");
        }
        assert_eq!(f16_value(f16_bits(0.0)), 0.0);
        assert_eq!(f16_value(f16_bits(1e9)), 65504.0);
        assert!(f16_value(f16_bits(1e-9)) > 0.0);
    }

    #[test]
    fn rejects_bad_input() {
        let bad = |indptr: Vec<i64>, indices: Vec<i32>, data: Vec<f32>| {
            SparseImpactIndex::build(
                CsrView {
                    indptr: &indptr,
                    indices: &indices,
                    data: &data,
                    n_dims: 4,
                },
                SparseBuildParams::exact(),
            )
            .is_err()
        };
        assert!(bad(vec![0, 1], vec![4], vec![1.0]));
        assert!(bad(vec![0, 1], vec![0], vec![-1.0]));
        assert!(bad(vec![0, 1], vec![0], vec![f32::NAN]));
        assert!(bad(vec![0, 2], vec![0], vec![1.0]));
        assert!(bad(vec![], vec![], vec![]));
    }
}
