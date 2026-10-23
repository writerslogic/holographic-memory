// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Batched 4-bit lookup-table scan over a block of 32 codes (the FastScan form of André,
//! Kermarrec and Le Scouarnec, 2015, as used by FAISS, RaBitQ and SymphonyQG).
//!
//! A block holds `m` groups of 16 bytes. Byte `i` of group `j` carries nibble `j` of code
//! `i` in its low half and of code `i + 16` in its high half, so one 16-byte load and two
//! table lookups score nibble `j` of all 32 codes. A query is a table of `m * 16` bytes: entry
//! `16 * j + v` is the contribution of nibble value `v` at position `j`. Each code's score is
//! the sum of its `m` entries.

/// Codes per block.
pub(crate) const BLOCK: usize = 32;

/// Writes `out[c] = sum_j lut[16 * j + nibble_j(c)]` for the 32 codes of `codes`
/// (`16 * m` bytes) and the table `lut` (`16 * m` bytes). Any `m` is allowed; sums are exact
/// in `u32`.
#[inline]
pub(crate) fn scan32(codes: &[u8], lut: &[u8], m: usize, out: &mut [u32; BLOCK]) {
    assert!(codes.len() >= 16 * m && lut.len() >= 16 * m);
    out.fill(0);
    // At most 256 groups of at most 255 fit a u16 lane without overflow.
    for (c, l) in codes[..16 * m]
        .chunks(16 * 256)
        .zip(lut[..16 * m].chunks(16 * 256))
    {
        let mut part = [0u16; BLOCK];
        scan32_u16(c, l, c.len() / 16, &mut part);
        for (o, p) in out.iter_mut().zip(part) {
            *o += u32::from(p);
        }
    }
}

/// [`scan32`] for `m <= 256` groups, accumulating in `u16`.
#[inline]
fn scan32_u16(codes: &[u8], lut: &[u8], m: usize, out: &mut [u16; BLOCK]) {
    debug_assert!(m <= 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline; `codes` and `lut` hold at least
        // `16 * m` bytes (checked by the public caller).
        unsafe { neon::scan32(codes, lut, m, out) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected; lengths as above.
            unsafe { avx2::scan32(codes, lut, m, out) };
            return;
        }
        scan32_scalar(codes, lut, m, out);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    scan32_scalar(codes, lut, m, out);
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn scan32_scalar(codes: &[u8], lut: &[u8], m: usize, out: &mut [u16; BLOCK]) {
    out.fill(0);
    for j in 0..m {
        add_group(&codes[16 * j..16 * j + 16], &lut[16 * j..16 * j + 16], out);
    }
}

/// Adds the entries of one group (16 code bytes, 16 table bytes) to the 32 sums.
#[inline]
fn add_group(codes: &[u8], lut: &[u8], out: &mut [u16; BLOCK]) {
    for i in 0..16 {
        out[i] += u16::from(lut[(codes[i] & 15) as usize]);
        out[i + 16] += u16::from(lut[(codes[i] >> 4) as usize]);
    }
}

/// Packs 32 codes of `m` nibbles each (`nibbles[c * m + j]`, values below 16) into the block
/// layout `scan32` reads; `block` holds `16 * m` bytes.
pub(crate) fn pack32(nibbles: &[u8], m: usize, block: &mut [u8]) {
    assert!(nibbles.len() == BLOCK * m && block.len() == 16 * m);
    for j in 0..m {
        for i in 0..16 {
            let lo = nibbles[i * m + j];
            let hi = nibbles[(i + 16) * m + j];
            debug_assert!(lo < 16 && hi < 16);
            block[16 * j + i] = lo | (hi << 4);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::BLOCK;

    #[inline]
    pub(super) unsafe fn scan32(codes: &[u8], lut: &[u8], m: usize, out: &mut [u16; BLOCK]) {
        let mask = vdupq_n_u8(0x0f);
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_u16(0),
            vdupq_n_u16(0),
            vdupq_n_u16(0),
            vdupq_n_u16(0),
        );
        let (mut b0, mut b1, mut b2, mut b3) = (
            vdupq_n_u16(0),
            vdupq_n_u16(0),
            vdupq_n_u16(0),
            vdupq_n_u16(0),
        );
        let (cp, lp) = (codes.as_ptr(), lut.as_ptr());
        let mut j = 0;
        // Two independent accumulator sets break the add latency chain.
        while j + 2 <= m {
            let c = vld1q_u8(cp.add(16 * j));
            let t = vld1q_u8(lp.add(16 * j));
            let rlo = vqtbl1q_u8(t, vandq_u8(c, mask));
            let rhi = vqtbl1q_u8(t, vshrq_n_u8::<4>(c));
            a0 = vaddw_u8(a0, vget_low_u8(rlo));
            a1 = vaddw_high_u8(a1, rlo);
            a2 = vaddw_u8(a2, vget_low_u8(rhi));
            a3 = vaddw_high_u8(a3, rhi);
            let c = vld1q_u8(cp.add(16 * j + 16));
            let t = vld1q_u8(lp.add(16 * j + 16));
            let rlo = vqtbl1q_u8(t, vandq_u8(c, mask));
            let rhi = vqtbl1q_u8(t, vshrq_n_u8::<4>(c));
            b0 = vaddw_u8(b0, vget_low_u8(rlo));
            b1 = vaddw_high_u8(b1, rlo);
            b2 = vaddw_u8(b2, vget_low_u8(rhi));
            b3 = vaddw_high_u8(b3, rhi);
            j += 2;
        }
        if j < m {
            let c = vld1q_u8(cp.add(16 * j));
            let t = vld1q_u8(lp.add(16 * j));
            let rlo = vqtbl1q_u8(t, vandq_u8(c, mask));
            let rhi = vqtbl1q_u8(t, vshrq_n_u8::<4>(c));
            a0 = vaddw_u8(a0, vget_low_u8(rlo));
            a1 = vaddw_high_u8(a1, rlo);
            a2 = vaddw_u8(a2, vget_low_u8(rhi));
            a3 = vaddw_high_u8(a3, rhi);
        }
        let o = out.as_mut_ptr();
        vst1q_u16(o, vaddq_u16(a0, b0));
        vst1q_u16(o.add(8), vaddq_u16(a1, b1));
        vst1q_u16(o.add(16), vaddq_u16(a2, b2));
        vst1q_u16(o.add(24), vaddq_u16(a3, b3));
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{add_group, BLOCK};

    /// Two 16-byte groups per 256-bit register: lanes 0..16 score group `j`, lanes 16..32
    /// group `j + 1`, both with their own table half.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn scan32(codes: &[u8], lut: &[u8], m: usize, out: &mut [u16; BLOCK]) {
        let mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        // 16-bit accumulators: codes 0..8 | 8..16 (low halves), 16..24 | 24..32 (high halves),
        // each for the even group (lane 0..16) and the odd group (lane 16..32).
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        let (cp, lp) = (codes.as_ptr(), lut.as_ptr());
        let mut j = 0;
        while j + 2 <= m {
            let c = _mm256_loadu_si256(cp.add(16 * j).cast());
            let t = _mm256_loadu_si256(lp.add(16 * j).cast());
            let rlo = _mm256_shuffle_epi8(t, _mm256_and_si256(c, mask));
            let rhi = _mm256_shuffle_epi8(t, _mm256_and_si256(_mm256_srli_epi16::<4>(c), mask));
            // Widen each 128-bit half's 16 bytes to 16-bit lanes and accumulate.
            let lo0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rlo));
            let lo1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(rlo));
            let hi0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rhi));
            let hi1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(rhi));
            a0 = _mm256_add_epi16(a0, lo0);
            a1 = _mm256_add_epi16(a1, lo1);
            a2 = _mm256_add_epi16(a2, hi0);
            a3 = _mm256_add_epi16(a3, hi1);
            j += 2;
        }
        // a0: codes 0..16 of even groups; a1: codes 0..16 of odd groups; a2/a3 codes 16..32.
        let lo = _mm256_add_epi16(a0, a1);
        let hi = _mm256_add_epi16(a2, a3);
        let o = out.as_mut_ptr();
        _mm256_storeu_si256(o.cast(), lo);
        _mm256_storeu_si256(o.add(16).cast(), hi);
        if j < m {
            add_group(&codes[16 * j..16 * j + 16], &lut[16 * j..16 * j + 16], out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::kernels::SplitMix;
    use super::*;

    fn random_block(m: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut rng = SplitMix(seed);
        let nibbles: Vec<u8> = (0..BLOCK * m)
            .map(|_| (rng.next_u64() & 15) as u8)
            .collect();
        let lut: Vec<u8> = (0..16 * m).map(|_| (rng.next_u64() & 255) as u8).collect();
        let mut block = vec![0u8; 16 * m];
        pack32(&nibbles, m, &mut block);
        (nibbles, lut, block)
    }

    #[test]
    fn scan_matches_direct_sum_at_every_width() {
        for &m in &[1usize, 2, 3, 16, 25, 32, 64, 100, 255, 256, 257, 300, 520] {
            let (nibbles, lut, block) = random_block(m, m as u64);
            let mut out = [0u32; BLOCK];
            scan32(&block, &lut, m, &mut out);
            for c in 0..BLOCK {
                let want: u32 = (0..m)
                    .map(|j| u32::from(lut[16 * j + nibbles[c * m + j] as usize]))
                    .sum();
                assert_eq!(out[c], want, "code {c} m {m}");
            }
            let mut scalar = [0u16; BLOCK];
            if m <= 256 {
                scan32_scalar(&block, &lut, m, &mut scalar);
                for c in 0..BLOCK {
                    assert_eq!(u32::from(scalar[c]), out[c]);
                }
            }
        }
    }

    /// Kill test of the neighbour-block estimator (lever 1A.1): ns per neighbour of the
    /// block scan at the nibble counts of 1-, 2- and 4-bit codes against the 8-bit SDOT
    /// gather estimate, same method as `raw_screen_microbenchmark` (300k rows, dim 256,
    /// 20k expansions of 32 neighbours, best of 5). Kill rule: block scan < 0.25 of est.
    /// `cargo test --release block_scan_microbenchmark -- --ignored --nocapture`; writes
    /// JSON to `$HMS_BLOCK_BENCH_OUT` when set.
    #[test]
    #[ignore = "microbenchmark: run in release with --ignored --nocapture"]
    fn block_scan_microbenchmark() {
        use super::super::kernels::{dot_i8_inline, prefetch_bytes};
        use std::time::Instant;
        const N: usize = 300_000;
        const DIM: usize = 256;
        const DEG: usize = 32;
        const EXPANSIONS: usize = 20_000;
        let mut rng = SplitMix(7);
        let codes: Vec<i8> = (0..N * DIM)
            .map(|_| (rng.next_u64() & 0xff) as i8)
            .collect();
        let q: Vec<i8> = (0..DIM).map(|_| (rng.next_u64() & 0xff) as i8).collect();
        let ids: Vec<u32> = (0..EXPANSIONS * DEG)
            .map(|_| (rng.next_u64() % N as u64) as u32)
            .collect();
        let code = |v: u32| &codes[v as usize * DIM..(v as usize + 1) * DIM];
        let est = |batch: &[u32]| -> i64 {
            batch.iter().for_each(|&v| prefetch_bytes(code(v)));
            batch
                .iter()
                .map(|&v| i64::from(dot_i8_inline(&q, code(v))))
                .sum()
        };
        let mut best = f64::INFINITY;
        let mut acc = 0i64;
        for _ in 0..5 {
            let t = Instant::now();
            for batch in ids.as_chunks::<DEG>().0 {
                acc = acc.wrapping_add(est(batch));
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e9 / ids.len() as f64);
        }
        let t_est = best;
        let mut report = format!(
            "{{\"n_rows\": {N}, \"dim\": {DIM}, \"degree\": {DEG}, \"expansions\": {EXPANSIONS}, \
             \"est_8bit_sdot_ns_per_neighbour\": {t_est:.2}, \"kill_rule\": \"block scan < 0.25 of est\", \
             \"est_checksum\": {acc}, \"block_scan\": ["
        );
        let mut first = true;
        for (bits, m) in [(1usize, DIM / 4), (2, DIM / 2), (4, DIM)] {
            // One block of 32 neighbour codes per vertex, read at a random vertex per
            // expansion; the lookup table is per query.
            let blocks: Vec<u8> = (0..N * 16 * m)
                .map(|_| (rng.next_u64() & 255) as u8)
                .collect();
            let lut: Vec<u8> = (0..16 * m).map(|_| (rng.next_u64() & 255) as u8).collect();
            let block = |v: u32| &blocks[v as usize * 16 * m..(v as usize + 1) * 16 * m];
            let mut out = [0u32; BLOCK];
            let mut best = f64::INFINITY;
            let mut sum = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                let mut s = 0u64;
                for (e, batch) in ids.as_chunks::<DEG>().0.iter().enumerate() {
                    let v = batch[0];
                    if let Some(next) = ids.as_chunks::<DEG>().0.get(e + 1) {
                        for line in block(next[0]).chunks(128) {
                            // SAFETY-free hint: the prefetch helper takes i8 slices.
                            prefetch_bytes(unsafe {
                                std::slice::from_raw_parts(line.as_ptr().cast::<i8>(), line.len())
                            });
                        }
                    }
                    scan32(block(v), &lut, m, &mut out);
                    s += out.iter().map(|&x| u64::from(x)).sum::<u64>();
                }
                sum = s;
                best = best.min(t.elapsed().as_secs_f64() * 1e9 / ids.len() as f64);
            }
            report.push_str(&format!(
                "{}{{\"bits\": {bits}, \"m\": {m}, \"bytes_per_block\": {}, \"ns_per_neighbour\": {best:.2}, \
                 \"ratio_to_est\": {:.3}, \"pass\": {}, \"checksum\": {sum}}}",
                if first { "" } else { ", " },
                16 * m,
                best / t_est,
                best < 0.25 * t_est
            ));
            first = false;
        }
        report.push_str("]}");
        println!("{report}");
        if let Ok(path) = std::env::var("HMS_BLOCK_BENCH_OUT") {
            std::fs::write(path, format!("{report}\n")).unwrap();
        }
    }
}
