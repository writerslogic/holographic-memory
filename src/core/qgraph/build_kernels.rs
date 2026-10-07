// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inner loops used only while constructing the graph.

/// Hint the cache to load `row`. A no-op where no prefetch instruction is wired up.
#[inline]
pub(crate) fn prefetch_row(row: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    for line in row.chunks(32) {
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
    let _ = row;
}

/// Hint the cache to load an adjacency list.
#[inline]
pub(crate) fn prefetch_ids(ids: &[u32]) {
    #[cfg(target_arch = "aarch64")]
    for line in ids.chunks(32) {
        // SAFETY: as in `prefetch_row`.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) line.as_ptr(),
                options(nostack, preserves_flags, readonly)
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = ids;
}

/// `[dot(a, b[0]), .., dot(a, b[3])]`, bit-identical to [`super::kernels::dot`]: lane `j` of
/// every accumulator sums the products of coordinates `j mod 16` in order, with a separate
/// multiply and add, and the lanes and tail are reduced as `dot` reduces them.
#[inline]
pub(crate) fn dot4(a: &[f32], b: [&[f32]; 4]) -> [f32; 4] {
    for r in b {
        assert_eq!(r.len(), a.len());
    }
    let body = a.len() / 16 * 16;
    let acc = lanes4(&a[..body], b.map(|r| &r[..body]));
    let mut out = [0f32; 4];
    for ((o, acc), r) in out.iter_mut().zip(&acc).zip(b) {
        let tail: f32 = a[body..].iter().zip(&r[body..]).map(|(x, y)| x * y).sum();
        *o = acc.iter().sum::<f32>() + tail;
    }
    out
}

/// Per-lane sums of `a * b[i]` over 16-float chunks; lengths are equal multiples of 16.
#[cfg(target_arch = "aarch64")]
#[inline]
fn lanes4(a: &[f32], b: [&[f32]; 4]) -> [[f32; 16]; 4] {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};
    // SAFETY: NEON is mandatory on aarch64. Every load reads 4 floats at offset `k + 4 i`
    // with `k + 16 <= a.len() == b[r].len()`, and every store writes 4 floats into a
    // 16-float array at offset 4 i < 16.
    unsafe {
        let z = vdupq_n_f32(0.0);
        // Sixteen named accumulators (row, lane group) so that they stay in registers.
        let (mut s00, mut s01, mut s02, mut s03) = (z, z, z, z);
        let (mut s10, mut s11, mut s12, mut s13) = (z, z, z, z);
        let (mut s20, mut s21, mut s22, mut s23) = (z, z, z, z);
        let (mut s30, mut s31, mut s32, mut s33) = (z, z, z, z);
        let [p0, p1, p2, p3] = b.map(<[f32]>::as_ptr);
        let pa = a.as_ptr();
        let mut k = 0;
        while k < a.len() {
            let x0 = vld1q_f32(pa.add(k));
            let x1 = vld1q_f32(pa.add(k + 4));
            let x2 = vld1q_f32(pa.add(k + 8));
            let x3 = vld1q_f32(pa.add(k + 12));
            macro_rules! row {
                ($p:ident, $a0:ident, $a1:ident, $a2:ident, $a3:ident) => {
                    $a0 = vaddq_f32($a0, vmulq_f32(x0, vld1q_f32($p.add(k))));
                    $a1 = vaddq_f32($a1, vmulq_f32(x1, vld1q_f32($p.add(k + 4))));
                    $a2 = vaddq_f32($a2, vmulq_f32(x2, vld1q_f32($p.add(k + 8))));
                    $a3 = vaddq_f32($a3, vmulq_f32(x3, vld1q_f32($p.add(k + 12))));
                };
            }
            row!(p0, s00, s01, s02, s03);
            row!(p1, s10, s11, s12, s13);
            row!(p2, s20, s21, s22, s23);
            row!(p3, s30, s31, s32, s33);
            k += 16;
        }
        let mut out = [[0f32; 16]; 4];
        let acc = [
            [s00, s01, s02, s03],
            [s10, s11, s12, s13],
            [s20, s21, s22, s23],
            [s30, s31, s32, s33],
        ];
        for (o, acc) in out.iter_mut().zip(acc) {
            for (i, s) in acc.into_iter().enumerate() {
                vst1q_f32(o.as_mut_ptr().add(4 * i), s);
            }
        }
        out
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn lanes4(a: &[f32], b: [&[f32]; 4]) -> [[f32; 16]; 4] {
    let mut out = [[0f32; 16]; 4];
    for (o, r) in out.iter_mut().zip(b) {
        for (x, y) in a.as_chunks::<16>().0.iter().zip(r.as_chunks::<16>().0) {
            for ((s, p), q) in o.iter_mut().zip(x).zip(y) {
                *s += p * q;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::kernels::{dot, SplitMix};
    use super::dot4;

    #[test]
    fn dot4_is_bit_identical_to_dot() {
        let mut rng = SplitMix(7);
        for d in [1, 15, 16, 17, 100, 256] {
            let mut row = || -> Vec<f32> {
                (0..d)
                    .map(|_| (rng.next_u64() >> 40) as f32 / 8388608.0 - 1.0)
                    .collect()
            };
            let a = row();
            let b: Vec<Vec<f32>> = (0..4).map(|_| row()).collect();
            let got = dot4(&a, [&b[0], &b[1], &b[2], &b[3]]);
            for (g, r) in got.iter().zip(&b) {
                assert_eq!(g.to_bits(), dot(&a, r).to_bits(), "d = {d}");
            }
        }
    }
}
