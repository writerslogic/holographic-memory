// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inner loops of the quantized graph: rotation, float dot product, and the FastScan table
//! lookups that turn edge codes into inner-product estimates.

/// Bits per query coordinate; FastScan table entries (four of them summed) must fit a u8.
pub(crate) const QUERY_BITS: usize = 4;
const ROUNDS: usize = 3;

/// SplitMix64: small, seedable and stable across platforms and crate versions.
pub(crate) struct SplitMix(pub(crate) u64);

impl SplitMix {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    #[cfg(test)]
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Random orthogonal transform: zero-pad to a power of two (at least 64), then three rounds
/// of random sign flips followed by a fast Walsh-Hadamard transform. Not Haar-distributed,
/// but close enough for the RaBitQ estimator in practice (checked by the bias test).
pub(crate) struct Rotation {
    dim: usize,
    padded: usize,
    signs: Vec<f32>,
    scale: f32,
}

impl Rotation {
    /// `bits` is the requested output length (0 = automatic); see `BuildParams::code_bits`.
    pub(crate) fn new(dim: usize, bits: usize, seed: u64) -> Self {
        let padded = dim.max(bits).next_power_of_two().max(64);
        let mut rng = SplitMix(seed);
        let signs = (0..ROUNDS * padded)
            .map(|_| if rng.next_u64() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        Self {
            dim,
            padded,
            signs,
            scale: (padded as f32).powf(-0.5 * ROUNDS as f32),
        }
    }

    pub(crate) fn padded(&self) -> usize {
        self.padded
    }

    pub(crate) fn apply(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(out.len(), self.padded);
        out[..self.dim].copy_from_slice(x);
        out[self.dim..].fill(0.0);
        for signs in self.signs.chunks_exact(self.padded) {
            for (o, s) in out.iter_mut().zip(signs) {
                *o *= s;
            }
            fwht(out);
        }
        for o in out.iter_mut() {
            *o *= self.scale;
        }
    }
}

/// Unnormalized in-place Walsh-Hadamard transform; `x.len()` must be a power of two.
fn fwht(x: &mut [f32]) {
    let mut h = 1;
    while h < x.len() {
        for chunk in x.chunks_exact_mut(2 * h) {
            let (a, b) = chunk.split_at_mut(h);
            for (u, v) in a.iter_mut().zip(b.iter_mut()) {
                let t = *u;
                *u = t + *v;
                *v = t - *v;
            }
        }
        h *= 2;
    }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 16];
    let ((ca, ta), (cb, tb)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    let tail: f32 = ta.iter().zip(tb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for ((s, p), q) in acc.iter_mut().zip(x).zip(y) {
            *s += p * q;
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Edges per FastScan batch: one 16-byte register of nibbles holds one position of 32 codes.
pub(crate) const BATCH: usize = 32;

/// A query coordinate vector quantized to [`QUERY_BITS`] unsigned bits per coordinate, stored
/// as FastScan lookup tables: `lut[16 g + c] = sum_t bit_t(c) * q[4 g + t]` for each group `g`
/// of four coordinates. For any bit vector `b`,
/// `sum_i b_i x_i ~= lo * popcount(b) + delta * raw(b)` with `raw(b) = sum_i b_i q_i`.
pub(crate) struct QueryCode {
    pub(crate) lut: Vec<u8>,
    pub(crate) lo: f32,
    pub(crate) delta: f32,
    /// Sum of the dequantized coordinates.
    pub(crate) sum: f32,
}

impl QueryCode {
    /// `dims` is the rotated (padded) dimension, a multiple of 64.
    pub(crate) fn new(dims: usize) -> Self {
        Self {
            lut: vec![0; 4 * dims],
            lo: 0.0,
            delta: 1.0,
            sum: 0.0,
        }
    }

    pub(crate) fn encode(&mut self, x: &[f32]) {
        debug_assert_eq!(4 * x.len(), self.lut.len());
        let (lo, hi) = x
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        let levels = ((1 << QUERY_BITS) - 1) as f32;
        let delta = if hi > lo { (hi - lo) / levels } else { 1.0 };
        let mut total = 0u32;
        for (group, table) in x
            .as_chunks::<4>()
            .0
            .iter()
            .zip(self.lut.as_chunks_mut::<16>().0)
        {
            let mut q = [0u8; 4];
            for (o, &v) in q.iter_mut().zip(group) {
                *o = ((v - lo) / delta).round().clamp(0.0, levels) as u8;
                total += u32::from(*o);
            }
            table[0] = 0;
            for c in 1..16 {
                table[c] = table[c & (c - 1)] + q[c.trailing_zeros() as usize];
            }
        }
        self.lo = lo;
        self.delta = delta;
        self.sum = lo * x.len() as f32 + delta * total as f32;
    }
}

/// Packs one bit plane of up to [`BATCH`] codes (`dims` bits each, as u64 words) into the
/// FastScan layout: byte `16 g + k` holds bits `4g..4g+3` of code `k` in its low nibble and
/// of code `k + 16` in its high nibble. Missing codes are zero.
pub(crate) fn pack_batch(codes: &[&[u64]], dims: usize, out: &mut [u8]) {
    assert!(codes.len() <= BATCH);
    assert_eq!(out.len(), 4 * dims);
    out.fill(0);
    for (j, code) in codes.iter().enumerate() {
        let (lane, shift) = (j % 16, 4 * (j / 16));
        for g in 0..dims / 4 {
            let nib = (code[g / 16] >> (4 * (g % 16))) & 0xF;
            out[16 * g + lane] |= (nib as u8) << shift;
        }
    }
}

/// `out[j] = sum_p 2^p sum_i bit_i(plane_p(code_j)) * q_i` for the [`BATCH`] codes whose
/// packed bit planes (one or two, each `lut.len()` bytes) are concatenated in `planes`.
pub(crate) fn fastscan(planes: &[u8], lut: &[u8], out: &mut [u32; BATCH]) {
    let n = lut.len();
    assert!(n > 0 && (planes.len() == n || planes.len() == 2 * n));
    #[cfg(target_arch = "aarch64")]
    if n.is_multiple_of(64) && n <= 16 * MAX_NEON_GROUPS {
        return if planes.len() == n {
            neon::fastscan::<1>(planes, lut, out)
        } else {
            neon::fastscan::<2>(planes, lut, out)
        };
    }
    fastscan_scalar(&planes[..n], lut, out);
    if planes.len() == 2 * n {
        let mut high = [0u32; BATCH];
        fastscan_scalar(&planes[n..], lut, &mut high);
        out.iter_mut().zip(high).for_each(|(o, h)| *o += 2 * h);
    }
}

/// Groups the NEON kernel can sum in u16 lanes: each table entry is at most 4 * 15 = 60.
#[cfg(target_arch = "aarch64")]
const MAX_NEON_GROUPS: usize = u16::MAX as usize / 60;

pub(crate) fn fastscan_scalar(packed: &[u8], lut: &[u8], out: &mut [u32; BATCH]) {
    out.fill(0);
    for (codes, table) in packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(lut.as_chunks::<16>().0)
    {
        for (k, &c) in codes.iter().enumerate() {
            out[k] += u32::from(table[usize::from(c & 0xF)]);
            out[k + 16] += u32::from(table[usize::from(c >> 4)]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::BATCH;

    /// Sums of the table entries over one packed plane, as four u16x8 lanes (codes 0-7, 8-15,
    /// 16-23, 24-31).
    ///
    /// # Safety
    /// `pc` and `pl` must be valid for `64 * chunks` bytes, and `240 * chunks` must fit a u16.
    #[inline(always)]
    unsafe fn plane(pc: *const u8, pl: *const u8, chunks: usize) -> [uint16x8_t; 4] {
        // SAFETY: the caller guarantees the reads; NEON is part of the aarch64 baseline. Four
        // table entries of at most 60 sum to at most 240, so the u8 additions cannot overflow.
        unsafe {
            let mask = vdupq_n_u8(0x0F);
            let mut a = [vdupq_n_u16(0); 4];
            for chunk in 0..chunks {
                let (mut lo, mut hi) = (vdupq_n_u8(0), vdupq_n_u8(0));
                for t in 0..4 {
                    let off = 64 * chunk + 16 * t;
                    let c = vld1q_u8(pc.add(off));
                    let l = vld1q_u8(pl.add(off));
                    lo = vaddq_u8(lo, vqtbl1q_u8(l, vandq_u8(c, mask)));
                    hi = vaddq_u8(hi, vqtbl1q_u8(l, vshrq_n_u8::<4>(c)));
                }
                a[0] = vaddw_u8(a[0], vget_low_u8(lo));
                a[1] = vaddw_high_u8(a[1], lo);
                a[2] = vaddw_u8(a[2], vget_low_u8(hi));
                a[3] = vaddw_high_u8(a[3], hi);
            }
            a
        }
    }

    pub(super) fn fastscan<const P: usize>(planes: &[u8], lut: &[u8], out: &mut [u32; BATCH]) {
        let n = lut.len();
        assert!((P == 1 || P == 2) && planes.len() == P * n && n.is_multiple_of(64));
        // SAFETY: plane `p < P` is `planes[p n..(p + 1) n]` and the table is `lut`, `n` bytes
        // each (asserted), that is `n / 64` chunks; the caller bounds `n` so that the u16 lanes
        // cannot overflow. The stores write exactly the 32 u32 of `out`.
        unsafe {
            let a = plane(planes.as_ptr(), lut.as_ptr(), n / 64);
            let b = if P == 2 {
                plane(planes.as_ptr().add(n), lut.as_ptr(), n / 64)
            } else {
                [vdupq_n_u16(0); 4]
            };
            let o = out.as_mut_ptr();
            for (q, (&x, &y)) in a.iter().zip(&b).enumerate() {
                let lo = vmlal_n_u16(vmovl_u16(vget_low_u16(x)), vget_low_u16(y), 2);
                let hi = vmlal_high_n_u16(vmovl_high_u16(x), y, 2);
                vst1q_u32(o.add(8 * q), lo);
                vst1q_u32(o.add(8 * q + 4), hi);
            }
        }
    }
}

/// Hint the cache to load `words`. A no-op where no prefetch instruction is wired up.
#[inline]
pub(crate) fn prefetch(words: &[u64]) {
    #[cfg(target_arch = "aarch64")]
    for line in words.chunks(16) {
        // SAFETY: PRFM is a hint: it never faults, writes no register or memory, and the
        // address points into a live slice.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) line.as_ptr(),
                options(nostack, preserves_flags, readonly)
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = words;
}

pub(crate) fn as_f32(w: &[u64]) -> &[f32] {
    // SAFETY: u64 alignment satisfies f32 alignment, the f32 view covers exactly the same
    // bytes, every bit pattern is a valid f32, and the lifetime is tied to `w`.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast::<f32>(), w.len() * 2) }
}

pub(crate) fn as_f32_mut(w: &mut [u64]) -> &mut [f32] {
    // SAFETY: as in `as_f32`; the exclusive borrow of `w` is carried over to the view.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<f32>(), w.len() * 2) }
}

pub(crate) fn as_u8(w: &[u64]) -> &[u8] {
    // SAFETY: u8 has alignment 1, the view covers exactly the same bytes, every bit pattern is
    // a valid u8, and the lifetime is tied to `w`.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast::<u8>(), w.len() * 8) }
}

pub(crate) fn as_u8_mut(w: &mut [u64]) -> &mut [u8] {
    // SAFETY: as in `as_u8`; the exclusive borrow of `w` is carried over to the view.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<u8>(), w.len() * 8) }
}

pub(crate) fn as_u32(w: &[u64]) -> &[u32] {
    // SAFETY: as in `as_f32`; every bit pattern is a valid u32.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast::<u32>(), w.len() * 2) }
}

pub(crate) fn as_u32_mut(w: &mut [u64]) -> &mut [u32] {
    // SAFETY: as in `as_f32_mut`; every bit pattern is a valid u32.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<u32>(), w.len() * 2) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the quantized query value of every coordinate, read back from the tables.
    fn levels(qc: &QueryCode) -> Vec<u32> {
        qc.lut
            .as_chunks::<16>()
            .0
            .iter()
            .flat_map(|t| [1, 2, 4, 8].map(|c| u32::from(t[c])))
            .collect()
    }

    #[test]
    fn fastscan_matches_bitwise_sum() {
        let mut rng = SplitMix(7);
        // 64..=4096 dimensions covers both the NEON kernel and its u16 bound; 8192 takes the
        // scalar path.
        for dims in [64, 128, 192, 256, 512, 1024, 4096, 8192] {
            let x: Vec<f32> = (0..dims).map(|_| rng.next_f32() - 0.5).collect();
            let mut qc = QueryCode::new(dims);
            qc.encode(&x);
            let q = levels(&qc);
            for n_codes in [1, 17, BATCH] {
                let codes: Vec<Vec<u64>> = (0..n_codes)
                    .map(|_| (0..dims / 64).map(|_| rng.next_u64()).collect())
                    .collect();
                // All-ones codes reach the u16 bound of the NEON path.
                let codes: Vec<Vec<u64>> = std::iter::once(vec![u64::MAX; dims / 64])
                    .chain(codes.into_iter().skip(1))
                    .collect();
                let refs: Vec<&[u64]> = codes.iter().map(Vec::as_slice).collect();
                let mut packed = vec![0u8; 4 * dims];
                pack_batch(&refs, dims, &mut packed);
                let (mut fast, mut slow) = ([0u32; BATCH], [0u32; BATCH]);
                fastscan(&packed, &qc.lut, &mut fast);
                fastscan_scalar(&packed, &qc.lut, &mut slow);
                assert_eq!(fast, slow, "dims = {dims}");
                for (j, &got) in fast.iter().enumerate() {
                    let want: u32 = codes.get(j).map_or(0, |c| {
                        (0..dims)
                            .filter(|&i| c[i / 64] >> (i % 64) & 1 == 1)
                            .map(|i| q[i])
                            .sum()
                    });
                    assert_eq!(got, want, "dims = {dims}, code {j}");
                }
                // Two planes: the second (weight 2) holds the codes in reverse order.
                let rev: Vec<&[u64]> = refs.iter().rev().copied().collect();
                let mut two = packed.clone();
                two.resize(8 * dims, 0);
                pack_batch(&rev, dims, &mut two[4 * dims..]);
                let mut high = [0u32; BATCH];
                fastscan_scalar(&two[4 * dims..], &qc.lut, &mut high);
                let mut fused = [0u32; BATCH];
                fastscan(&two, &qc.lut, &mut fused);
                for j in 0..BATCH {
                    assert_eq!(fused[j], slow[j] + 2 * high[j], "dims = {dims}, code {j}");
                }
            }
        }
    }

    #[test]
    fn dot_matches_naive() {
        let mut rng = SplitMix(3);
        for len in 0..70 {
            let a: Vec<f32> = (0..len).map(|_| rng.next_f32() - 0.5).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.next_f32() - 0.5).collect();
            let naive: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y)).sum();
            assert!((f64::from(dot(&a, &b)) - naive).abs() < 1e-5);
        }
    }

    #[test]
    fn rotation_preserves_inner_products() {
        let rot = Rotation::new(100, 0, 11);
        let mut rng = SplitMix(5);
        let a: Vec<f32> = (0..100).map(|_| rng.next_f32() - 0.5).collect();
        let b: Vec<f32> = (0..100).map(|_| rng.next_f32() - 0.5).collect();
        let (mut ra, mut rb) = (vec![0.0; 128], vec![0.0; 128]);
        rot.apply(&a, &mut ra);
        rot.apply(&b, &mut rb);
        assert!((dot(&ra, &rb) - dot(&a, &b)).abs() < 1e-4);
        assert!((dot(&ra, &ra) - dot(&a, &a)).abs() < 1e-4);
    }

    #[test]
    fn query_code_reconstructs_bit_sums() {
        let mut rng = SplitMix(9);
        let x: Vec<f32> = (0..128).map(|_| rng.next_f32() - 0.5).collect();
        let mut qc = QueryCode::new(128);
        qc.encode(&x);
        let code = [rng.next_u64(), rng.next_u64()];
        let mut packed = vec![0u8; 4 * 128];
        pack_batch(&[&code], 128, &mut packed);
        let mut raw = [0u32; BATCH];
        fastscan(&packed, &qc.lut, &mut raw);
        let pop = code.iter().map(|c| c.count_ones()).sum::<u32>() as f32;
        let approx = qc.lo * pop + qc.delta * raw[0] as f32;
        let exact: f32 = (0..128)
            .filter(|i| code[i / 64] >> (i % 64) & 1 == 1)
            .map(|i| x[i])
            .sum();
        // Rounding error is at most delta / 2 per selected coordinate.
        assert!((approx - exact).abs() <= qc.delta / 2.0 * pop + 1e-4);
        let q = levels(&qc);
        let total: f32 = q.iter().map(|&v| v as f32).sum();
        assert!((qc.sum - (qc.lo * 128.0 + qc.delta * total)).abs() < 1e-4);
    }
}
