// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Frozen public basis, document codes, query transforms, and the three index
//! representations compared by the kill test:
//!
//! * [`SplitIndex`]: int8 x16 head + `B`-bit sign residual per document,
//!   contiguous `[u64; 3 + W]` records (layout in the crate docs).
//! * [`Int8Index`]: full-dimension int8 rows with a per-row f32 scale.
//! * [`Primary`]: the f32 rows (exact scan baseline and the rerank source)
//!   plus the per-document liveness flag shared by every method.
//!
//! Encoding is single-threaded and allocation-free per document through
//! [`Scratch`].

use anyhow::{ensure, Result};
use serde::Serialize;

use crate::data::Rng;
use crate::{dot_f32, f32_to_f16_bits, HEAD_DIM};

/// Word counts (`B / 64`) the scan kernels are monomorphized for.
pub const SUPPORTED_WORDS: [usize; 8] = [1, 2, 3, 4, 6, 8, 12, 16];

/// Record words before the residual bits: two words of int8 head, one of scalars.
pub const RECORD_PREFIX_WORDS: usize = 3;

/// How the 16-dim head subspace is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum BasisKind {
    /// Top-16 principal subspace of the calibration sample.
    Pca,
    /// Fixed random orthonormal basis (data-independent).
    Random,
}

/// Query-side evaluation of the residual sign code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ResidualMode {
    /// 1-bit query, XOR + popcount Hamming estimator.
    Sym1,
    /// 4-bit query as four bit planes, AND + popcount per plane.
    Asym4,
}

/// Parameters of [`fit_basis`].
pub struct BasisParams {
    /// Head subspace kind.
    pub kind: BasisKind,
    /// Embedding dimension `D`.
    pub dim: usize,
    /// Residual bits `B` (multiple of 64, at most the padded dimension).
    pub residual_bits: usize,
    /// Seed for the random signs, coordinate subset and random basis.
    pub seed: u64,
    /// Subspace-iteration sweeps for PCA.
    pub pca_iters: usize,
}

/// The frozen public basis `(mu, U, R)`.
pub struct Basis {
    /// Head subspace kind.
    pub kind: BasisKind,
    /// `D`.
    pub dim: usize,
    /// `P = next_power_of_two(D)`.
    pub padded: usize,
    /// `B`.
    pub bits: usize,
    /// `W = B / 64`.
    pub words: usize,
    /// Centre, length `D`.
    pub mu: Vec<f32>,
    /// `HEAD_DIM` orthonormal rows of length `D`, row-major.
    pub u: Vec<f32>,
    /// Random signs (`+1`/`-1`) for the first `D` padded coordinates.
    pub signs: Vec<f32>,
    /// The `B` kept coordinates of the rotated vector (sorted).
    pub select: Vec<u32>,
    /// `1 / sqrt(P)`: makes the Walsh-Hadamard transform orthonormal.
    pub scale: f32,
    /// `P / B`: compensates the dropped coordinates in the residual estimator.
    pub keep_ratio: f32,
    /// Fraction of calibration variance captured by the head (PCA diagnostics;
    /// computed for both kinds on the calibration sample).
    pub head_variance_fraction: f32,
}

/// Modified Gram-Schmidt on `HEAD_DIM` rows of length `d`.
fn orthonormalize(v: &mut [f32], d: usize, rng: &mut Rng) {
    for k in 0..HEAD_DIM {
        for j in 0..k {
            let (lo, hi) = v.split_at_mut(k * d);
            let vj = &lo[j * d..(j + 1) * d];
            let vk = &mut hi[..d];
            let p = dot_f32(vk, vj);
            for t in 0..d {
                vk[t] -= p * vj[t];
            }
        }
        let vk = &mut v[k * d..(k + 1) * d];
        let mut n = crate::norm_f32(vk);
        if n < 1e-12 {
            for x in vk.iter_mut() {
                *x = rng.gaussian();
            }
            n = crate::norm_f32(vk);
        }
        let inv = 1.0 / n;
        for x in vk.iter_mut() {
            *x *= inv;
        }
    }
}

/// Fit the basis on a calibration sample (`calib` is row-major, `n x dim`).
pub fn fit_basis(p: &BasisParams, calib: &[f32]) -> Result<Basis> {
    let d = p.dim;
    ensure!(d > 0, "dim must be positive");
    ensure!(
        calib.len() >= d && calib.len() % d == 0,
        "calibration sample must hold at least one row of {} floats",
        d
    );
    let n = calib.len() / d;
    let padded = d.next_power_of_two().max(64);
    let bits = p.residual_bits;
    ensure!(
        bits % 64 == 0 && bits > 0 && bits <= padded,
        "residual bits {} must be a positive multiple of 64 and at most the padded dim {}",
        bits,
        padded
    );
    let words = bits / 64;
    ensure!(
        SUPPORTED_WORDS.contains(&words),
        "residual words {} not in the monomorphized set {:?}; pick --residual-bits from {:?}",
        words,
        SUPPORTED_WORDS,
        SUPPORTED_WORDS.map(|w| w * 64)
    );
    let mut rng = Rng::new(p.seed ^ 0xBA51_5BA5_15BA_515B);

    let mut mu = vec![0.0f32; d];
    for row in calib.chunks_exact(d) {
        for j in 0..d {
            mu[j] += row[j];
        }
    }
    let inv_n = 1.0 / n as f32;
    for m in mu.iter_mut() {
        *m *= inv_n;
    }
    let mut centred = calib.to_vec();
    for row in centred.chunks_exact_mut(d) {
        for j in 0..d {
            row[j] -= mu[j];
        }
    }
    let total_var: f32 = centred.iter().map(|x| x * x).sum::<f32>() * inv_n;

    let mut u = vec![0.0f32; HEAD_DIM * d];
    for x in u.iter_mut() {
        *x = rng.gaussian();
    }
    orthonormalize(&mut u, d, &mut rng);
    if p.kind == BasisKind::Pca {
        let mut acc = vec![0.0f32; HEAD_DIM * d];
        for _ in 0..p.pca_iters.max(1) {
            acc.iter_mut().for_each(|x| *x = 0.0);
            for row in centred.chunks_exact(d) {
                let mut proj = [0.0f32; HEAD_DIM];
                for k in 0..HEAD_DIM {
                    proj[k] = dot_f32(row, &u[k * d..(k + 1) * d]);
                }
                for k in 0..HEAD_DIM {
                    let s = proj[k];
                    let a = &mut acc[k * d..(k + 1) * d];
                    for j in 0..d {
                        a[j] += s * row[j];
                    }
                }
            }
            u.copy_from_slice(&acc);
            orthonormalize(&mut u, d, &mut rng);
        }
    }
    let mut head_var = 0.0f32;
    for row in centred.chunks_exact(d) {
        for k in 0..HEAD_DIM {
            let h = dot_f32(row, &u[k * d..(k + 1) * d]);
            head_var += h * h;
        }
    }
    head_var *= inv_n;
    let head_variance_fraction = if total_var > 0.0 {
        head_var / total_var
    } else {
        0.0
    };

    let signs: Vec<f32> = (0..d)
        .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut select = rng.sample_indices(padded, bits);
    select.sort_unstable();

    Ok(Basis {
        kind: p.kind,
        dim: d,
        padded,
        bits,
        words,
        mu,
        u,
        signs,
        select,
        scale: 1.0 / (padded as f32).sqrt(),
        keep_ratio: padded as f32 / bits as f32,
        head_variance_fraction,
    })
}

/// In-place Walsh-Hadamard transform (unnormalized); `v.len()` is a power of two.
pub fn fwht(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            let (lo, hi) = v[i..i + 2 * h].split_at_mut(h);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let x = *a;
                let y = *b;
                *a = x + y;
                *b = x - y;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// Per-thread working buffers for encoding and query preparation.
pub struct Scratch {
    /// Head coefficients.
    pub head: [f32; HEAD_DIM],
    /// Centred vector, then the residual (length `D`).
    pub resid: Vec<f32>,
    /// Padded rotation buffer (length `P`).
    pub pad: Vec<f32>,
    /// Kept rotated coordinates (length `B`).
    pub sel: Vec<f32>,
}

impl Scratch {
    /// Buffers sized for `basis`.
    pub fn new(basis: &Basis) -> Self {
        Self {
            head: [0.0; HEAD_DIM],
            resid: vec![0.0; basis.dim],
            pad: vec![0.0; basis.padded],
            sel: vec![0.0; basis.bits],
        }
    }
}

impl Basis {
    /// Split `x` (optionally centred by `mu`) into `s.head` and `s.resid`.
    pub fn project(&self, x: &[f32], centre: bool, s: &mut Scratch) {
        let d = self.dim;
        if centre {
            for j in 0..d {
                s.resid[j] = x[j] - self.mu[j];
            }
        } else {
            s.resid[..d].copy_from_slice(&x[..d]);
        }
        for k in 0..HEAD_DIM {
            s.head[k] = dot_f32(&s.resid, &self.u[k * d..(k + 1) * d]);
        }
        for k in 0..HEAD_DIM {
            let h = s.head[k];
            let uk = &self.u[k * d..(k + 1) * d];
            for j in 0..d {
                s.resid[j] -= h * uk[j];
            }
        }
    }

    /// Rotate `s.resid` into `s.sel`: sign flip, zero pad, orthonormal
    /// Walsh-Hadamard, coordinate subset.
    pub fn rotate(&self, s: &mut Scratch) {
        let d = self.dim;
        let p = self.padded;
        for j in 0..d {
            s.pad[j] = s.resid[j] * self.signs[j];
        }
        for j in d..p {
            s.pad[j] = 0.0;
        }
        fwht(&mut s.pad);
        let sc = self.scale;
        for v in s.pad.iter_mut() {
            *v *= sc;
        }
        for (dst, &idx) in s.sel.iter_mut().zip(&self.select) {
            *dst = s.pad[idx as usize];
        }
    }

    /// Record words per document.
    pub fn stride(&self) -> usize {
        RECORD_PREFIX_WORDS + self.words
    }

    /// Write the record for the vector currently held in `s` (after
    /// [`Basis::project`] with centring and [`Basis::rotate`]).
    pub fn encode_record(&self, s: &Scratch, out: &mut [u64]) {
        debug_assert_eq!(out.len(), self.stride());
        let mut maxabs = 0.0f32;
        for h in s.head.iter() {
            maxabs = maxabs.max(h.abs());
        }
        let head_scale = maxabs / 127.0;
        let inv = if head_scale > 0.0 { 1.0 / head_scale } else { 0.0 };
        let mut hb = [0u8; HEAD_DIM];
        for k in 0..HEAD_DIM {
            let q = (s.head[k] * inv).round().clamp(-127.0, 127.0) as i8;
            hb[k] = q as u8;
        }
        out[0] = u64::from_le_bytes(hb[0..8].try_into().expect("8 head bytes"));
        out[1] = u64::from_le_bytes(hb[8..16].try_into().expect("8 head bytes"));

        let mut sum_sq = 0.0f32;
        let mut sum_abs = 0.0f32;
        let mut ones: u32 = 0;
        for (wi, chunk) in s.sel.chunks_exact(64).enumerate() {
            let mut word = 0u64;
            for (t, &v) in chunk.iter().enumerate() {
                sum_sq += v * v;
                sum_abs += v.abs();
                if v > 0.0 {
                    word |= 1u64 << t;
                }
            }
            ones += word.count_ones();
            out[RECORD_PREFIX_WORDS + wi] = word;
        }
        let rho = if sum_abs > 0.0 {
            self.keep_ratio * sum_sq / sum_abs
        } else {
            0.0
        };
        out[2] = (f32_to_f16_bits(head_scale) as u64)
            | ((f32_to_f16_bits(rho) as u64) << 16)
            | (((ones & 0xffff) as u64) << 32);
    }
}

/// Query transformed against a [`Basis`] for one [`ResidualMode`].
pub struct SplitQuery {
    /// Residual mode the planes/bits were built for.
    pub mode: ResidualMode,
    /// Int16-quantized query head widened to i32.
    pub head_q: [i32; HEAD_DIM],
    /// Query head scale (`max|U q| / 32767`).
    pub a_q: f32,
    /// `sym1`: query sign bits (`W` words).
    pub bits: Vec<u64>,
    /// `asym4`: bit planes of the 4-bit query (`4 x W` words).
    pub planes: [Vec<u64>; 4],
    /// `sym1`: `mean|z| * B`.
    pub b_q: f32,
    /// `sym1`: `2 * mean|z|`.
    pub c_q: f32,
    /// `asym4`: `2 * delta`.
    pub delta2: f32,
    /// `asym4`: `2 * lo`.
    pub lo2: f32,
    /// `asym4`: `sum_i z_i`.
    pub zsum: f32,
}

impl SplitQuery {
    /// Zeroed query buffers for `basis`.
    pub fn new(basis: &Basis, mode: ResidualMode) -> Self {
        let w = basis.words;
        Self {
            mode,
            head_q: [0; HEAD_DIM],
            a_q: 0.0,
            bits: vec![0; w],
            planes: [vec![0; w], vec![0; w], vec![0; w], vec![0; w]],
            b_q: 0.0,
            c_q: 0.0,
            delta2: 0.0,
            lo2: 0.0,
            zsum: 0.0,
        }
    }
}

impl Basis {
    /// Transform query `q` (unit norm) into `out`.
    pub fn prepare_query(&self, q: &[f32], s: &mut Scratch, out: &mut SplitQuery) {
        self.project(q, false, s);
        let mut maxabs = 0.0f32;
        for h in s.head.iter() {
            maxabs = maxabs.max(h.abs());
        }
        let a_q = maxabs / 32767.0;
        let inv = if a_q > 0.0 { 1.0 / a_q } else { 0.0 };
        for k in 0..HEAD_DIM {
            out.head_q[k] = (s.head[k] * inv).round().clamp(-32767.0, 32767.0) as i32;
        }
        out.a_q = a_q;
        self.rotate(s);
        let z = &s.sel;
        let b = self.bits as f32;
        match out.mode {
            ResidualMode::Sym1 => {
                let mut sum_abs = 0.0f32;
                for (wi, chunk) in z.chunks_exact(64).enumerate() {
                    let mut word = 0u64;
                    for (t, &v) in chunk.iter().enumerate() {
                        sum_abs += v.abs();
                        if v > 0.0 {
                            word |= 1u64 << t;
                        }
                    }
                    out.bits[wi] = word;
                }
                let m = sum_abs / b;
                out.b_q = m * b;
                out.c_q = 2.0 * m;
            }
            ResidualMode::Asym4 => {
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                let mut zsum = 0.0f32;
                for &v in z.iter() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                    zsum += v;
                }
                let mut delta = (hi - lo) / 15.0;
                if delta.is_nan() || delta <= 0.0 {
                    delta = 1.0;
                }
                let inv = 1.0 / delta;
                for (wi, chunk) in z.chunks_exact(64).enumerate() {
                    let mut p = [0u64; 4];
                    for (t, &v) in chunk.iter().enumerate() {
                        let u = ((v - lo) * inv).round().clamp(0.0, 15.0) as u64;
                        for (j, plane) in p.iter_mut().enumerate() {
                            *plane |= ((u >> j) & 1) << t;
                        }
                    }
                    for j in 0..4 {
                        out.planes[j][wi] = p[j];
                    }
                }
                out.delta2 = 2.0 * delta;
                out.lo2 = 2.0 * lo;
                out.zsum = zsum;
            }
        }
    }
}

/// The split-manifold index: contiguous records plus the basis.
pub struct SplitIndex {
    /// Document slots.
    pub n: usize,
    /// Words per record.
    pub stride: usize,
    /// `n * stride` words.
    pub codes: Vec<u64>,
    /// The frozen basis.
    pub basis: Basis,
    /// Residual mode the index is scanned with.
    pub mode: ResidualMode,
}

impl SplitIndex {
    /// Allocate `n` zeroed slots.
    pub fn with_capacity(basis: Basis, n: usize, mode: ResidualMode) -> Self {
        let stride = basis.stride();
        Self {
            n,
            stride,
            codes: vec![0u64; n * stride],
            basis,
            mode,
        }
    }

    /// Encode `x` into slot `id` (also used for replace).
    pub fn encode(&mut self, id: usize, x: &[f32], s: &mut Scratch) {
        self.basis.project(x, true, s);
        self.basis.rotate(s);
        let st = self.stride;
        self.basis.encode_record(s, &mut self.codes[id * st..(id + 1) * st]);
    }

    /// Zero the scalar word of slot `id` so it scores 0; returns bytes touched.
    pub fn delete(&mut self, id: usize) -> usize {
        self.codes[id * self.stride + 2] = 0;
        8
    }

    /// Bytes per document in the scanned array.
    pub fn bytes_per_doc(&self) -> usize {
        self.stride * 8
    }
}

/// Full-dimension int8 rows with a per-row scale (the baseline quantizer).
pub struct Int8Index {
    /// Document slots.
    pub n: usize,
    /// `D`.
    pub dim: usize,
    /// `n * dim` codes.
    pub codes: Vec<i8>,
    /// Per-row `max|x| / 127`.
    pub scales: Vec<f32>,
}

impl Int8Index {
    /// Allocate `n` zeroed slots.
    pub fn with_capacity(n: usize, dim: usize) -> Self {
        Self {
            n,
            dim,
            codes: vec![0i8; n * dim],
            scales: vec![0.0f32; n],
        }
    }

    /// Quantize `x` into slot `id`.
    pub fn encode(&mut self, id: usize, x: &[f32]) {
        let d = self.dim;
        let mut maxabs = 0.0f32;
        for v in x.iter() {
            maxabs = maxabs.max(v.abs());
        }
        let scale = maxabs / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let row = &mut self.codes[id * d..(id + 1) * d];
        for j in 0..d {
            row[j] = (x[j] * inv).round().clamp(-127.0, 127.0) as i8;
        }
        self.scales[id] = scale;
    }

    /// Zero the row scale so it scores 0; returns bytes touched.
    pub fn delete(&mut self, id: usize) -> usize {
        self.scales[id] = 0.0;
        4
    }

    /// Bytes per document in the scanned arrays.
    pub fn bytes_per_doc(&self) -> usize {
        self.dim + 4
    }
}

/// Int8-quantized query for [`Int8Index`].
pub struct Int8Query {
    /// Quantized coordinates.
    pub q: Vec<i8>,
    /// `max|q| / 127`.
    pub scale: f32,
}

impl Int8Query {
    /// Zeroed query of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            q: vec![0i8; dim],
            scale: 0.0,
        }
    }

    /// Quantize `q` in place.
    pub fn prepare(&mut self, q: &[f32]) {
        let mut maxabs = 0.0f32;
        for v in q.iter() {
            maxabs = maxabs.max(v.abs());
        }
        self.scale = maxabs / 127.0;
        let inv = if self.scale > 0.0 { 1.0 / self.scale } else { 0.0 };
        for (dst, &v) in self.q.iter_mut().zip(q) {
            *dst = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// The f32 rows plus the shared liveness flags.
pub struct Primary {
    /// Document slots.
    pub n: usize,
    /// `D`.
    pub dim: usize,
    /// `n * dim` floats.
    pub rows: Vec<f32>,
    /// 1 = live, 0 = deleted.
    pub live: Vec<u8>,
}

impl Primary {
    /// Take ownership of the corpus rows.
    pub fn new(rows: Vec<f32>, dim: usize) -> Self {
        let n = rows.len() / dim;
        Self {
            n,
            dim,
            rows,
            live: vec![1u8; n],
        }
    }

    /// Row `id`.
    #[inline(always)]
    pub fn row(&self, id: usize) -> &[f32] {
        &self.rows[id * self.dim..(id + 1) * self.dim]
    }

    /// Clear the liveness flag; returns bytes touched.
    pub fn delete(&mut self, id: usize) -> usize {
        self.live[id] = 0;
        1
    }

    /// Overwrite row `id` and mark it live; returns bytes touched.
    pub fn replace(&mut self, id: usize, x: &[f32]) -> usize {
        let d = self.dim;
        self.rows[id * d..(id + 1) * d].copy_from_slice(&x[..d]);
        self.live[id] = 1;
        d * 4 + 1
    }

    /// Bytes per document in the scanned array (flag included).
    pub fn bytes_per_doc(&self) -> usize {
        self.dim * 4 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{synthetic, SyntheticParams};

    #[test]
    fn fwht_scaled_preserves_norm_and_is_self_inverse() {
        let mut v: Vec<f32> = (0..128).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let orig = v.clone();
        let n0 = crate::norm_f32(&v);
        fwht(&mut v);
        let sc = 1.0 / (128f32).sqrt();
        for x in v.iter_mut() {
            *x *= sc;
        }
        assert!((crate::norm_f32(&v) - n0).abs() < 1e-3);
        fwht(&mut v);
        for x in v.iter_mut() {
            *x *= sc;
        }
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn basis_is_orthonormal_and_estimator_tracks_cosine() {
        let ds = synthetic(&SyntheticParams {
            n: 600,
            dim: 96,
            clusters: 6,
            n_queries: 4,
            seed: 3,
            spread: 0.75,
            spectrum_alpha: 0.5,
        });
        let basis = fit_basis(
            &BasisParams {
                kind: BasisKind::Pca,
                dim: 96,
                residual_bits: 128,
                seed: 3,
                pca_iters: 16,
            },
            &ds.docs[..500 * 96],
        )
        .expect("basis");
        for a in 0..HEAD_DIM {
            for b in 0..HEAD_DIM {
                let d = dot_f32(&basis.u[a * 96..(a + 1) * 96], &basis.u[b * 96..(b + 1) * 96]);
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((d - want).abs() < 1e-4, "u[{a}].u[{b}] = {d}");
            }
        }
        assert!(basis.head_variance_fraction > 0.0 && basis.head_variance_fraction <= 1.0);
        let mut ix = SplitIndex::with_capacity(basis, ds.n_docs, ResidualMode::Sym1);
        let mut s = Scratch::new(&ix.basis);
        for i in 0..ds.n_docs {
            ix.encode(i, ds.doc(i), &mut s);
        }
        let mut sq = SplitQuery::new(&ix.basis, ResidualMode::Sym1);
        ix.basis.prepare_query(ds.query(0), &mut s, &mut sq);
        let mut heap = crate::TopK::new(32);
        crate::scan::scan_split(&ix, &sq, &mut heap);
        let q_mu = dot_f32(ds.query(0), &ix.basis.mu);
        let mut err = 0.0f32;
        for &(est, id) in heap.entries() {
            let exact = dot_f32(ds.query(0), ds.doc(id as usize));
            err += (est + q_mu - exact).abs();
        }
        let mean_err = err / heap.entries().len() as f32;
        assert!(mean_err < 0.25, "mean |estimate - exact| = {mean_err}");
    }
}
