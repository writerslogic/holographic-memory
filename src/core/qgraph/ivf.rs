// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Experimental angular IVF with contiguous, centroid-relative scalar codes.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::invalid;
use super::kernels::{
    dot, dot_f32_i4, dot_f32_i8, dot_i4_inline, dot_i8_inline, interleave_for_i4,
};
use super::vertex::{code_at, quantize, quantize4};
use crate::core::error::WarmError;

const MAGIC: &[u8; 8] = b"HMSIVF01";
const HEADER: usize = 24;
const MAX_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_DIM: usize = 4096;
const BOUND_ROWS: usize = 64;

#[derive(Default)]
struct CodeBounds {
    offsets: Vec<u32>,
    minimum: Vec<i8>,
    maximum: Vec<i8>,
    scales: Vec<[f32; 2]>,
}

fn bounds_vector<T: Clone>(len: usize, value: T) -> Result<Vec<T>, WarmError> {
    let mut vector = Vec::new();
    vector
        .try_reserve_exact(len)
        .map_err(|_| WarmError::CapacityExceeded {
            limit: "IVF runtime bounds allocation".into(),
        })?;
    vector.resize(len, value);
    Ok(vector)
}

impl CodeBounds {
    fn fits_budget(index_bytes: usize, cache_bytes: usize) -> bool {
        index_bytes
            .checked_add(cache_bytes)
            .is_some_and(|total| total <= MAX_BYTES)
    }

    fn build(index: &QuantizedIvf) -> Result<Self, WarmError> {
        let blocks = index
            .offsets
            .windows(2)
            .map(|range| (range[1] - range[0]) as usize / BOUND_ROWS)
            .sum::<usize>();
        let cells = blocks
            .checked_mul(index.padded)
            .ok_or_else(|| invalid("IVF runtime bounds shape overflow".into()))?;
        let bytes = index
            .offsets
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|bytes| bytes.checked_add(cells.checked_mul(2)?))
            .and_then(|bytes| {
                bytes.checked_add(blocks.checked_mul(std::mem::size_of::<[f32; 2]>())?)
            })
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()));
        let Some(bytes) = bytes.filter(|&bytes| Self::fits_budget(index.serialized_bytes(), bytes))
        else {
            return Ok(Self::default());
        };
        let mut bounds = Self {
            offsets: bounds_vector(index.offsets.len(), 0)?,
            minimum: bounds_vector(cells, i8::MAX)?,
            maximum: bounds_vector(cells, -i8::MAX)?,
            scales: bounds_vector(blocks, [f32::INFINITY, 0.0])?,
        };
        let stride = index.padded * usize::from(index.bits) / 8;
        let mut block = 0;
        for (cluster, range) in index.offsets.windows(2).enumerate() {
            let start = range[0] as usize;
            let full_end = start + (range[1] as usize - start) / BOUND_ROWS * BOUND_ROWS;
            for first in (start..full_end).step_by(BOUND_ROWS) {
                let at = block * index.padded;
                let minimum = &mut bounds.minimum[at..at + index.padded];
                let maximum = &mut bounds.maximum[at..at + index.padded];
                for row in first..first + BOUND_ROWS {
                    let code = &index.codes[row * stride..(row + 1) * stride];
                    for (coordinate, (low, high)) in
                        minimum.iter_mut().zip(maximum.iter_mut()).enumerate()
                    {
                        let value = code_at(code, index.bits, coordinate);
                        *low = (*low).min(value);
                        *high = (*high).max(value);
                    }
                    bounds.scales[block][0] = bounds.scales[block][0].min(index.scales[row]);
                    bounds.scales[block][1] = bounds.scales[block][1].max(index.scales[row]);
                }
                block += 1;
            }
            bounds.offsets[cluster + 1] = block as u32;
        }
        debug_assert_eq!(block, blocks);
        debug_assert!(bounds.allocated_bytes() >= bytes);
        if !Self::fits_budget(index.serialized_bytes(), bounds.allocated_bytes()) {
            return Ok(Self::default());
        }
        Ok(bounds)
    }

    fn allocated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.offsets.capacity() * std::mem::size_of::<u32>())
            .saturating_add(self.minimum.capacity())
            .saturating_add(self.maximum.capacity())
            .saturating_add(self.scales.capacity() * std::mem::size_of::<[f32; 2]>())
    }

    fn distance_lower_bound(
        &self,
        block: usize,
        positive: &[i8],
        negative: &[i8],
        query_scale: f32,
        center_distance: f32,
    ) -> Option<f32> {
        let start = block * positive.len();
        let minimum = &self.minimum[start..start + positive.len()];
        let maximum = &self.maximum[start..start + positive.len()];
        // At most 4096 * 127 * 127; the integer envelope and native dot cannot overflow.
        let ip = dot_i8_inline(positive, maximum) + dot_i8_inline(negative, minimum);
        let row_scale = self.scales[block][usize::from(ip >= 0)];
        let lower = center_distance - query_scale * row_scale * ip as f32;
        lower.is_finite().then_some(lower)
    }
}

/// Codes are grouped by assignment, with original row ids retained for results.
pub struct QuantizedIvf {
    n: usize,
    dim: usize,
    padded: usize,
    bits: u8,
    residual_bits: u8,
    centers: Vec<f32>,
    offsets: Vec<u32>,
    ids: Vec<u32>,
    scales: Vec<f32>,
    residual_scales: Vec<f32>,
    codes: Vec<i8>,
    residual: Vec<i8>,
    bounds: CodeBounds,
}

/// Limits the lists scanned and the coarse candidates scored with the float query.
#[derive(Clone, Copy, Debug)]
pub struct IvfSearchParams {
    pub nprobe: usize,
    /// Zero returns coarse estimates. Otherwise at least `k` candidates are re-scored.
    pub rerank: usize,
}

#[derive(Clone, Copy, Debug)]
struct Hit {
    distance: f32,
    id: u32,
    row: usize,
    cluster: usize,
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

fn layout(
    n: usize,
    dim: usize,
    clusters: usize,
    bits: u8,
    residual: u8,
) -> Result<usize, WarmError> {
    if n == 0
        || n >= u32::MAX as usize
        || !(1..=MAX_DIM).contains(&dim)
        || clusters == 0
        || clusters > n
        || !matches!(bits, 4 | 8)
        || !matches!(residual, 0 | 4 | 5 | 8)
    {
        return Err(invalid("invalid IVF shape or code width".into()));
    }
    let padded = dim.next_multiple_of(32);
    let stride = padded * usize::from(bits) / 8;
    let residual_stride = padded * usize::from(residual) / 8;
    let size = (|| {
        HEADER
            .checked_add(clusters.checked_mul(dim)?.checked_mul(4)?)?
            .checked_add(clusters.checked_add(1)?.checked_mul(4)?)?
            .checked_add(n.checked_mul(8 + if residual > 0 { 4 } else { 0 })?)?
            .checked_add(n.checked_mul(stride.checked_add(residual_stride)?)?)?
            .checked_add(4)
    })()
    .filter(|&size| size <= MAX_BYTES)
    .ok_or_else(|| invalid("IVF index exceeds the 2 GiB allocation limit".into()))?;
    Ok(size)
}

fn validate_units(
    rows: &[f32],
    dim: usize,
    label: &str,
    allow_zero: bool,
) -> Result<(), WarmError> {
    for row in rows.chunks_exact(dim) {
        if row.iter().any(|x| !x.is_finite()) {
            return Err(invalid(format!("{label} contains a non-finite coordinate")));
        }
        let norm: f64 = row.iter().map(|&x| f64::from(x).powi(2)).sum();
        if (norm - 1.0).abs() > 0.001 && !(allow_zero && norm == 0.0) {
            return Err(invalid(format!("{label} must contain unit vectors")));
        }
    }
    Ok(())
}

fn code5(code: &[i8], i: usize) -> i8 {
    let bit = i * 5;
    let word = u16::from(code[bit / 8] as u8)
        | (u16::from(code.get(bit / 8 + 1).copied().unwrap_or(0) as u8) << 8);
    (((word >> (bit % 8)) as i8) << 3) >> 3
}

fn quantize5(x: &[f32], code: &mut [i8]) -> f32 {
    code.fill(0);
    let max = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if max == 0.0 {
        return 0.0;
    }
    let scale = max / 15.0;
    for (i, &x) in x.iter().enumerate() {
        let value = (x / scale).round().clamp(-15.0, 15.0) as i8 as u8 & 31;
        let bit = i * 5;
        code[bit / 8] |= (value << (bit % 8)) as i8;
        if bit % 8 > 3 {
            code[bit / 8 + 1] |= (value >> (8 - bit % 8)) as i8;
        }
    }
    scale
}

fn dot_f32_i5(query: &[f32], code: &[i8]) -> f32 {
    let mut accumulators = [0.0f32; 8];
    for (i, &q) in query.iter().enumerate() {
        accumulators[i % 8] += q * f32::from(code5(code, i));
    }
    accumulators.iter().sum()
}

impl QuantizedIvf {
    /// Assignments and centroids come from the same externally trained coarse quantizer.
    /// Input rows must be finite unit or zero vectors; centroids must be unit vectors.
    /// Zero rows use the zero inner-product convention. No original vectors are retained.
    pub fn build(
        rows: &[f32],
        dim: usize,
        centers: &[f32],
        assignments: &[u32],
        bits: u8,
        residual_bits: u8,
    ) -> Result<Self, WarmError> {
        if dim == 0 || !rows.len().is_multiple_of(dim) || !centers.len().is_multiple_of(dim) {
            return Err(invalid("IVF input does not contain complete rows".into()));
        }
        let n = rows.len() / dim;
        let clusters = centers.len() / dim;
        layout(n, dim, clusters, bits, residual_bits)?;
        if assignments.len() != n || assignments.iter().any(|&x| x as usize >= clusters) {
            return Err(invalid(
                "IVF assignments do not match the rows and centroids".into(),
            ));
        }
        validate_units(rows, dim, "IVF input", true)?;
        validate_units(centers, dim, "IVF centroids", false)?;
        let mut offsets = vec![0u32; clusters + 1];
        for &cluster in assignments {
            offsets[cluster as usize + 1] += 1;
        }
        for cluster in 1..=clusters {
            offsets[cluster] += offsets[cluster - 1];
        }
        let mut cursor = offsets[..clusters].to_vec();
        let mut ids = vec![0; n];
        for (id, &cluster) in assignments.iter().enumerate() {
            let next = &mut cursor[cluster as usize];
            ids[*next as usize] = id as u32;
            *next += 1;
        }
        let padded = dim.next_multiple_of(32);
        let stride = padded * usize::from(bits) / 8;
        let residual_stride = padded * usize::from(residual_bits) / 8;
        let mut index = Self {
            n,
            dim,
            padded,
            bits,
            residual_bits,
            centers: centers.to_vec(),
            offsets,
            ids,
            scales: vec![0.0; n],
            residual_scales: vec![0.0; if residual_bits > 0 { n } else { 0 }],
            codes: vec![0; n * stride],
            residual: vec![0; n * residual_stride],
            bounds: CodeBounds::default(),
        };
        let mut relative = vec![0.0; dim];
        for cluster in 0..clusters {
            let center = &index.centers[cluster * dim..(cluster + 1) * dim];
            for slot in index.offsets[cluster] as usize..index.offsets[cluster + 1] as usize {
                let id = index.ids[slot] as usize;
                let row = &rows[id * dim..(id + 1) * dim];
                for ((r, &x), &c) in relative.iter_mut().zip(row).zip(center) {
                    *r = x - c;
                }
                let code = &mut index.codes[slot * stride..(slot + 1) * stride];
                let scale = if bits == 4 {
                    quantize4(&relative, code)
                } else {
                    quantize(&relative, code)
                };
                index.scales[slot] = scale;
                if residual_bits > 0 {
                    for (i, r) in relative.iter_mut().enumerate() {
                        *r -= scale * f32::from(code_at(code, bits, i));
                    }
                    let residual =
                        &mut index.residual[slot * residual_stride..(slot + 1) * residual_stride];
                    index.residual_scales[slot] = match residual_bits {
                        4 => quantize4(&relative, residual),
                        5 => quantize5(&relative, residual),
                        _ => quantize(&relative, residual),
                    };
                }
            }
        }
        index.bounds = CodeBounds::build(&index)?;
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn clusters(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn serialized_bytes(&self) -> usize {
        // Construction and loading both validate this layout before allocating.
        layout(
            self.n,
            self.dim,
            self.clusters(),
            self.bits,
            self.residual_bits,
        )
        .expect("validated IVF layout")
    }

    /// Runtime code-envelope allocation including cache fields; allocator metadata excluded.
    pub fn auxiliary_bytes(&self) -> usize {
        self.bounds.allocated_bytes()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.serialized_bytes());
        out.extend_from_slice(MAGIC);
        for x in [self.n as u32, self.dim as u32, self.clusters() as u32] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&[self.bits, self.residual_bits, 0, 0]);
        for &x in &self.centers {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for &x in self.offsets.iter().chain(&self.ids) {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for &x in self.scales.iter().chain(&self.residual_scales) {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend(self.codes.iter().chain(&self.residual).map(|&x| x as u8));
        out.extend_from_slice(&crc32fast::hash(&out).to_le_bytes());
        out
    }

    /// Checks shape, exact length, CRC, ids, offsets, scales and codes before use.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WarmError> {
        if bytes.len() < HEADER + 4 || &bytes[..8] != MAGIC || bytes[22..24] != [0, 0] {
            return Err(invalid("invalid IVF header".into()));
        }
        let read_u32 = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let n = read_u32(8) as usize;
        let dim = read_u32(12) as usize;
        let clusters = read_u32(16) as usize;
        let (bits, residual_bits) = (bytes[20], bytes[21]);
        if bytes.len() != layout(n, dim, clusters, bits, residual_bits)? {
            return Err(invalid("IVF length does not match its shape".into()));
        }
        if crc32fast::hash(&bytes[..bytes.len() - 4]) != read_u32(bytes.len() - 4) {
            return Err(invalid("IVF checksum mismatch".into()));
        }
        let mut at = HEADER;
        let mut floats = |len: usize| {
            let end = at + len * 4;
            let v = bytes[at..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            at = end;
            v
        };
        let centers = floats(clusters * dim);
        validate_units(&centers, dim, "IVF centroids", false)?;
        let mut integers = |len: usize| {
            let end = at + len * 4;
            let v = bytes[at..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            at = end;
            v
        };
        let offsets = integers(clusters + 1);
        if offsets[0] != 0
            || offsets[clusters] as usize != n
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(invalid("invalid IVF list offsets".into()));
        }
        let ids = integers(n);
        let mut seen = vec![false; n];
        for &id in &ids {
            if id as usize >= n || std::mem::replace(&mut seen[id as usize], true) {
                return Err(invalid("IVF ids must be a permutation of row ids".into()));
            }
        }
        let mut floats = |len: usize| {
            let end = at + len * 4;
            let v = bytes[at..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            at = end;
            v
        };
        let scales = floats(n);
        let residual_scales = floats(if residual_bits > 0 { n } else { 0 });
        if scales
            .iter()
            .chain(&residual_scales)
            .any(|x| !x.is_finite() || *x < 0.0 || *x > 2.01)
        {
            return Err(invalid("invalid IVF code scale".into()));
        }
        let padded = dim.next_multiple_of(32);
        let stride = padded * usize::from(bits) / 8;
        let residual_stride = padded * usize::from(residual_bits) / 8;
        let codes = bytes[at..at + n * stride]
            .iter()
            .map(|&x| x as i8)
            .collect();
        at += n * stride;
        let residual = bytes[at..at + n * residual_stride]
            .iter()
            .map(|&x| x as i8)
            .collect();
        let mut index = Self {
            n,
            dim,
            padded,
            bits,
            residual_bits,
            centers,
            offsets,
            ids,
            scales,
            residual_scales,
            codes,
            residual,
            bounds: CodeBounds::default(),
        };
        for (codes, width) in [(&index.codes, bits), (&index.residual, residual_bits)] {
            if width == 5 {
                if codes
                    .chunks_exact(index.padded * 5 / 8)
                    .any(|code| (0..index.padded).any(|i| code5(code, i) == -16))
                {
                    return Err(invalid("invalid IVF quantization code".into()));
                }
                continue;
            }
            if codes.iter().any(|&b| {
                if width == 4 {
                    (b as u8 & 0x0f) == 8 || (b as u8 >> 4) == 8
                } else {
                    b == i8::MIN
                }
            }) {
                return Err(invalid("invalid IVF quantization code".into()));
            }
        }
        index.bounds = CodeBounds::build(&index)?;
        Ok(index)
    }
}

/// Reusable query buffers; one searcher belongs to one query worker.
pub struct IvfSearcher<'a> {
    index: &'a QuantizedIvf,
    query: Vec<f32>,
    query_code: Vec<i8>,
    query_positive: Vec<i8>,
    query_negative: Vec<i8>,
    interleaved: Vec<i8>,
    centers: Vec<Hit>,
    center_scores: Vec<f32>,
    heap: BinaryHeap<Hit>,
    results: Vec<Hit>,
    scanned: usize,
    evaluated: usize,
}

impl<'a> IvfSearcher<'a> {
    pub fn new(index: &'a QuantizedIvf) -> Self {
        Self {
            index,
            query: vec![0.0; index.padded],
            query_code: vec![0; index.padded],
            query_positive: vec![0; index.padded],
            query_negative: vec![0; index.padded],
            interleaved: vec![0; index.padded],
            centers: Vec::with_capacity(index.clusters()),
            center_scores: vec![0.0; index.clusters()],
            heap: BinaryHeap::new(),
            results: Vec::new(),
            scanned: 0,
            evaluated: 0,
        }
    }

    pub fn scanned(&self) -> usize {
        self.scanned
    }

    /// Complete coarse-code dot evaluations in the most recent successful query.
    pub fn evaluated(&self) -> usize {
        self.evaluated
    }

    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        params: IvfSearchParams,
    ) -> Result<Vec<(u32, f32)>, WarmError> {
        self.search_with_pruning(query, k, params, true)
    }

    /// Reference traversal of every selected-list code with identical scoring and reranking.
    pub fn search_unpruned(
        &mut self,
        query: &[f32],
        k: usize,
        params: IvfSearchParams,
    ) -> Result<Vec<(u32, f32)>, WarmError> {
        self.search_with_pruning(query, k, params, false)
    }

    fn search_with_pruning(
        &mut self,
        query: &[f32],
        k: usize,
        params: IvfSearchParams,
        prune: bool,
    ) -> Result<Vec<(u32, f32)>, WarmError> {
        let index = self.index;
        let prune = prune && !index.bounds.offsets.is_empty();
        if query.len() != index.dim
            || k == 0
            || k > index.n
            || params.nprobe == 0
            || params.nprobe > index.clusters()
            || params.rerank > index.n
            || query.iter().any(|x| !x.is_finite())
        {
            return Err(invalid(
                "invalid IVF query shape, value or search limit".into(),
            ));
        }
        let max = query.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if max == 0.0 {
            return Err(invalid("IVF query must be nonzero".into()));
        }
        let norm = query.iter().map(|&x| (x / max).powi(2)).sum::<f32>().sqrt();
        self.query.fill(0.0);
        for (dst, &x) in self.query.iter_mut().zip(query) {
            *dst = (x / max) / norm;
        }
        let scale = quantize(&self.query, &mut self.query_code);
        if prune {
            for ((positive, negative), &value) in self
                .query_positive
                .iter_mut()
                .zip(&mut self.query_negative)
                .zip(&self.query_code)
            {
                *positive = value.max(0);
                *negative = value.min(0);
            }
        }
        if index.bits == 4 {
            interleave_for_i4(&self.query_code, &mut self.interleaved);
        }
        self.centers.clear();
        for (cluster, center) in index.centers.chunks_exact(index.dim).enumerate() {
            let distance = -dot(&self.query[..index.dim], center);
            self.center_scores[cluster] = distance;
            self.centers.push(Hit {
                distance,
                id: cluster as u32,
                row: 0,
                cluster,
            });
        }
        if params.nprobe < self.centers.len() {
            self.centers.select_nth_unstable(params.nprobe);
            self.centers.truncate(params.nprobe);
        }
        self.centers.sort_unstable();
        self.heap.clear();
        self.scanned = 0;
        self.evaluated = 0;
        let limit = k.max(params.rerank);
        let stride = index.padded * usize::from(index.bits) / 8;
        for center in &self.centers {
            let start = index.offsets[center.cluster] as usize;
            let end = index.offsets[center.cluster + 1] as usize;
            self.scanned += end - start;
            for first in (start..end).step_by(BOUND_ROWS) {
                let last = (first + BOUND_ROWS).min(end);
                if prune && last - first == BOUND_ROWS && self.heap.len() == limit {
                    let block = index.bounds.offsets[center.cluster] as usize
                        + (first - start) / BOUND_ROWS;
                    if let Some(lower) = index.bounds.distance_lower_bound(
                        block,
                        &self.query_positive,
                        &self.query_negative,
                        scale,
                        center.distance,
                    ) {
                        // Strict numeric separation leaves signed-zero and id ties untouched.
                        if lower > self.heap.peek().expect("nonempty candidate heap").distance {
                            continue;
                        }
                    }
                }
                for row in first..last {
                    self.evaluated += 1;
                    let code = &index.codes[row * stride..(row + 1) * stride];
                    let ip = if index.bits == 4 {
                        dot_i4_inline(&self.interleaved, code)
                    } else {
                        dot_i8_inline(&self.query_code, code)
                    };
                    let hit = Hit {
                        distance: center.distance - scale * index.scales[row] * ip as f32,
                        id: index.ids[row],
                        row,
                        cluster: center.cluster,
                    };
                    if self.heap.len() < limit {
                        self.heap.push(hit);
                    } else if hit < *self.heap.peek().expect("nonempty candidate heap") {
                        *self.heap.peek_mut().expect("nonempty candidate heap") = hit;
                    }
                }
            }
        }
        self.results.clear();
        self.results.extend(self.heap.drain());
        if params.rerank > 0 {
            let residual_stride = index.padded * usize::from(index.residual_bits) / 8;
            for hit in &mut self.results {
                let code = &index.codes[hit.row * stride..(hit.row + 1) * stride];
                let ip = if index.bits == 4 {
                    dot_f32_i4(&self.query, code)
                } else {
                    dot_f32_i8(&self.query, code)
                };
                let mut score = -self.center_scores[hit.cluster] + index.scales[hit.row] * ip;
                if index.residual_bits > 0 {
                    let residual =
                        &index.residual[hit.row * residual_stride..(hit.row + 1) * residual_stride];
                    let ip = match index.residual_bits {
                        4 => dot_f32_i4(&self.query, residual),
                        5 => dot_f32_i5(&self.query, residual),
                        _ => dot_f32_i8(&self.query, residual),
                    };
                    score += index.residual_scales[hit.row] * ip;
                }
                hit.distance = -score;
            }
        }
        self.results.sort_unstable();
        Ok(self
            .results
            .iter()
            .take(k)
            .map(|h| (h.id, h.distance))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_budget_is_combined_and_disabled_cache_uses_the_reference_traversal() {
        let cache = 4096;
        assert!(CodeBounds::fits_budget(MAX_BYTES - cache - 1, cache));
        assert!(CodeBounds::fits_budget(MAX_BYTES - cache, cache));
        assert!(!CodeBounds::fits_budget(MAX_BYTES - cache + 1, cache));
        assert!(!CodeBounds::fits_budget(usize::MAX, cache));

        let mut index =
            QuantizedIvf::build(&[1.0, 0.0].repeat(128), 2, &[1.0, 0.0], &[0; 128], 4, 5).unwrap();
        let bytes = index.to_bytes();
        index.bounds = CodeBounds::default();
        assert_eq!(index.auxiliary_bytes(), std::mem::size_of::<CodeBounds>());
        let mut search = IvfSearcher::new(&index);
        let mut reference = IvfSearcher::new(&index);
        for rerank in [0, 64, 65, 128] {
            let params = IvfSearchParams { nprobe: 1, rerank };
            let actual = search.search(&[1.0, 0.0], 10, params).unwrap();
            let expected = reference.search_unpruned(&[1.0, 0.0], 10, params).unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|&(id, distance)| (id, distance.to_bits()))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|&(id, distance)| (id, distance.to_bits()))
                    .collect::<Vec<_>>()
            );
            assert_eq!(search.scanned(), 128);
            assert_eq!(search.evaluated(), 128);
        }
        assert_eq!(index.to_bytes(), bytes);
    }

    #[test]
    fn integer_envelopes_bound_native_dots_at_maximum_dimension() {
        for bits in [4u8, 8] {
            let maximum = if bits == 4 { 7 } else { 127 };
            for value in [-maximum, maximum] {
                let query = vec![127; MAX_DIM];
                let negative = vec![0; MAX_DIM];
                let code = vec![
                    if bits == 4 {
                        ((value as u8 & 15) | ((value as u8 & 15) << 4)) as i8
                    } else {
                        value
                    };
                    MAX_DIM * usize::from(bits) / 8
                ];
                let mut interleaved = vec![0; MAX_DIM];
                interleave_for_i4(&query, &mut interleaved);
                let ip = if bits == 4 {
                    dot_i4_inline(&interleaved, &code)
                } else {
                    dot_i8_inline(&query, &code)
                };
                assert_eq!(ip, MAX_DIM as i32 * 127 * i32::from(value));
                let bounds = CodeBounds {
                    minimum: vec![value; MAX_DIM],
                    maximum: vec![value; MAX_DIM],
                    scales: vec![[f32::from_bits(1), 2.01]],
                    ..CodeBounds::default()
                };
                for query_scale in [f32::from_bits(1), f32::MIN_POSITIVE, 1.0 / 127.0] {
                    for center in [-1.0, -0.0, 0.0, 1.0] {
                        let lower = bounds
                            .distance_lower_bound(0, &query, &negative, query_scale, center)
                            .unwrap();
                        for row_scale in [f32::from_bits(1), f32::MIN_POSITIVE, 1.0 / 127.0, 2.01] {
                            let exact = center - query_scale * row_scale * ip as f32;
                            assert!(lower <= exact);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nonfinite_envelopes_fall_back_even_when_individual_scores_are_finite() {
        let bounds = CodeBounds {
            minimum: vec![0; 32],
            maximum: vec![1; 32],
            scales: vec![[f32::MAX / 2.0; 2]],
            ..CodeBounds::default()
        };
        let query = [1; 32];
        let negative = [0; 32];
        for axis in 0..32 {
            let mut code = [0; 32];
            code[axis] = 1;
            let ip = query
                .iter()
                .zip(code)
                .map(|(&q, c)| i32::from(q) * c)
                .sum::<i32>();
            assert!((0.0 - 1.0 * (f32::MAX / 2.0) * ip as f32).is_finite());
        }
        assert_eq!(
            bounds.distance_lower_bound(0, &query, &negative, 1.0, 0.0),
            None
        );
        assert_eq!(
            bounds.distance_lower_bound(0, &query, &negative, f32::INFINITY, 0.0),
            None
        );
        assert_eq!(
            bounds.distance_lower_bound(0, &query, &negative, 0.0, f32::NAN),
            None
        );
    }

    #[test]
    fn packed_five_bit_residuals_preserve_signed_values_and_padding() {
        let values = (-15..=15).map(|x| x as f32).collect::<Vec<_>>();
        let mut code = vec![0; 32 * 5 / 8];
        assert_eq!(quantize5(&values, &mut code), 1.0);
        for (i, &value) in values.iter().enumerate() {
            assert_eq!(f32::from(code5(&code, i)), value);
        }
        assert_eq!(code5(&code, 31), 0);
        let query = (0..32).map(|x| (x - 16) as f32).collect::<Vec<_>>();
        let exact = query.iter().zip(&values).map(|(q, x)| q * x).sum::<f32>();
        assert_eq!(dot_f32_i5(&query, &code), exact);
    }
}
