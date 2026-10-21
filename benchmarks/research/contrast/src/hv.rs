// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bipolar hypervector primitives over a fixed 2,048-bit space.
//!
//! # Data layout
//! A hypervector is `[u64; WORDS]` (256 bytes) of sign bits; bit `i` lives at
//! `words[i / 64] >> (i % 64)`. A set bit encodes `+1`, a clear bit `-1`.
//!
//! # Operations
//! * `bind` is elementwise multiplication of bipolar vectors, which is XNOR on
//!   sign bits (`+1 * +1 = +1`, `-1 * -1 = +1`, mixed = `-1`).
//! * `rotate` is the permutation `rho^delta`: a cyclic left shift of the whole
//!   2,048-bit ring by `delta` bits, so `rho^a(rho^b(x)) == rho^(a+b)(x)`.
//! * `accumulate` adds a bipolar vector into an `i32` sum; `sign_bits` turns a
//!   sum back into sign bits (ties broken by a fixed random vector so the result
//!   is deterministic and unbiased); `clamp_i8` keeps the un-signed sum for the
//!   int8 ablation.
//! * `dot_bits` is the bipolar dot product of two sign vectors, i.e.
//!   `DIM - 2 * popcount(a XOR b)`, in `[-DIM, DIM]`.
//! * `dot_i32_bits` and `dot_bits_i8` are mixed dot products used by the
//!   single-dot-product contrast scorer and by the int8 ablation.
//!
//! # Atoms
//! `atom(seed)` draws the random bipolar hypervector `I[w]` of a stem from a
//! splitmix64 stream seeded by the stem's hash. No vocabulary table exists: the
//! same stem always yields the same atom, on any machine, in any process.
//!
//! # Invariants
//! Every function is pure and deterministic. No allocation happens on the
//! per-pair hot path (`bind`, `rotate`, `accumulate`).

/// Dimensionality of the sign-bit sketch.
pub const DIM: usize = 2048;
/// Number of `u64` words holding `DIM` bits.
pub const WORDS: usize = DIM / 64;

/// Sign bits of a bipolar hypervector (bit set = `+1`, clear = `-1`).
pub type BitVec = [u64; WORDS];
/// Un-signed accumulator for a superposition of bipolar vectors.
pub type Accum = [i32; DIM];

/// Splitmix64 step: returns the next 64-bit output and advances `state`.
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// FNV-1a 64-bit hash of `bytes`.
#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The atom seed of a stem: FNV-1a of the UTF-8 bytes, finalized through one
/// splitmix64 step so adjacent hashes do not yield correlated streams.
#[inline]
pub fn stem_seed(stem: &str) -> u64 {
    let mut s = fnv1a64(stem.as_bytes()) ^ 0x5851_F42D_4C95_7F2D;
    splitmix64(&mut s)
}

/// Mixes an ordered `(a, b, delta)` triple into one 64-bit key (hashed-pair baseline).
#[inline]
pub fn pair_key(a: u64, b: u64, delta: usize) -> u64 {
    let mut s = a ^ b.rotate_left(23) ^ (delta as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    splitmix64(&mut s)
}

/// The random bipolar hypervector drawn from `seed`.
#[inline]
pub fn atom(seed: u64) -> BitVec {
    let mut s = seed;
    let mut v = [0u64; WORDS];
    for w in v.iter_mut() {
        *w = splitmix64(&mut s);
    }
    v
}

/// Fixed vector that breaks `sign_bits` ties.
pub fn tie_breaker() -> BitVec {
    atom(0x7E1E_B0A7_1E5E_ED00)
}

/// Elementwise bipolar product (XNOR of sign bits).
#[inline]
pub fn bind(a: &BitVec, b: &BitVec) -> BitVec {
    let mut out = [0u64; WORDS];
    for i in 0..WORDS {
        out[i] = !(a[i] ^ b[i]);
    }
    out
}

/// Cyclic left rotation of the 2,048-bit ring by `delta` bits: output bit `j`
/// is input bit `(j - delta) mod DIM`.
#[inline]
pub fn rotate(v: &BitVec, delta: usize) -> BitVec {
    let delta = delta % DIM;
    let word_shift = delta / 64;
    let bit_shift = delta % 64;
    let mut out = [0u64; WORDS];
    for k in 0..WORDS {
        let src = (k + WORDS - word_shift) % WORDS;
        if bit_shift == 0 {
            out[k] = v[src];
        } else {
            let prev = (src + WORDS - 1) % WORDS;
            out[k] = (v[src] << bit_shift) | (v[prev] >> (64 - bit_shift));
        }
    }
    out
}

/// Adds the bipolar vector `v` into `acc` (`+1` for a set bit, `-1` for a clear bit).
#[inline]
pub fn accumulate(acc: &mut Accum, v: &BitVec) {
    for (w, &word) in v.iter().enumerate() {
        let base = w * 64;
        for b in 0..64 {
            acc[base + b] += (((word >> b) & 1) as i32) * 2 - 1;
        }
    }
}

/// Adds `src` into `dst` elementwise.
#[inline]
pub fn add_accum(dst: &mut Accum, src: &Accum) {
    for i in 0..DIM {
        dst[i] += src[i];
    }
}

/// Sign bits of an accumulator; zero entries take the tie-breaker's bit.
pub fn sign_bits(acc: &Accum, tie: &BitVec) -> BitVec {
    let mut out = [0u64; WORDS];
    for (w, slot) in out.iter_mut().enumerate() {
        let base = w * 64;
        let mut word = 0u64;
        for b in 0..64 {
            let x = acc[base + b];
            let bit = if x > 0 {
                1
            } else if x < 0 {
                0
            } else {
                (tie[w] >> b) & 1
            };
            word |= bit << b;
        }
        *slot = word;
    }
    out
}

/// The accumulator clamped into `i8` (the int8 ablation of the un-signed sum).
pub fn clamp_i8(acc: &Accum) -> Box<[i8; DIM]> {
    let mut out = Box::new([0i8; DIM]);
    for i in 0..DIM {
        out[i] = acc[i].clamp(-127, 127) as i8;
    }
    out
}

/// Bipolar dot product of two sign vectors, in `[-DIM, DIM]`.
#[inline]
pub fn dot_bits(a: &BitVec, b: &BitVec) -> i32 {
    let mut disagree = 0u32;
    for i in 0..WORDS {
        disagree += (a[i] ^ b[i]).count_ones();
    }
    DIM as i32 - 2 * disagree as i32
}

/// Dot product of a real (integer) query vector with a bipolar sign vector:
/// `sum_i q_i * s_i` where `s_i = +1` for a set bit and `-1` otherwise.
/// `total` must equal `sum_i q_i` (precomputed once per query).
#[inline]
pub fn dot_i32_bits(q: &[i32; DIM], total: i64, bits: &BitVec) -> i64 {
    let mut set_sum: i64 = 0;
    for (w, &word) in bits.iter().enumerate() {
        let base = w * 64;
        let mut local: i64 = 0;
        for b in 0..64 {
            let mask = -(((word >> b) & 1) as i32);
            local += i64::from(q[base + b] & mask);
        }
        set_sum += local;
    }
    2 * set_sum - total
}

/// Dot product of a bipolar query with an int8 document trace.
#[inline]
pub fn dot_bits_i8(q: &BitVec, v: &[i8; DIM]) -> i32 {
    let mut set_sum: i32 = 0;
    let mut total: i32 = 0;
    for (w, &word) in q.iter().enumerate() {
        let base = w * 64;
        for b in 0..64 {
            let x = i32::from(v[base + b]);
            total += x;
            let mask = -(((word >> b) & 1) as i32);
            set_sum += x & mask;
        }
    }
    2 * set_sum - total
}

/// Euclidean norm of an int8 trace.
pub fn norm_i8(v: &[i8; DIM]) -> f64 {
    let mut s: i64 = 0;
    for &x in v.iter() {
        let x = i64::from(x);
        s += x * x;
    }
    (s as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_composes_and_wraps() {
        let v = atom(1);
        let a = rotate(&rotate(&v, 3), 5);
        let b = rotate(&v, 8);
        assert_eq!(a, b);
        assert_eq!(rotate(&v, DIM), v);
        let mut unit = [0u64; WORDS];
        unit[0] = 1;
        let r = rotate(&unit, DIM - 1);
        assert_eq!(r[WORDS - 1], 1u64 << 63);
    }

    #[test]
    fn bind_is_involutive_and_dot_is_consistent() {
        let a = atom(7);
        let b = atom(9);
        assert_eq!(bind(&bind(&a, &b), &b), a);
        assert_eq!(dot_bits(&a, &a), DIM as i32);
        let mut acc = [0i32; DIM];
        accumulate(&mut acc, &a);
        let s = sign_bits(&acc, &tie_breaker());
        assert_eq!(s, a);
        let total: i64 = acc.iter().map(|&x| i64::from(x)).sum();
        assert_eq!(dot_i32_bits(&acc, total, &b), i64::from(dot_bits(&a, &b)));
        let v8 = clamp_i8(&acc);
        assert_eq!(dot_bits_i8(&b, &v8), dot_bits(&a, &b));
    }
}
