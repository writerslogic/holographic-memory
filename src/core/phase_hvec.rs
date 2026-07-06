// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantized-phase hypervector (qFHRR, arXiv 2604.25939).
//!
//! Stores D phases as indices in Z_N. Binding is phase-add mod N, unbinding is
//! phase-sub mod N, similarity is the mean cosine of phase differences, and
//! bundling superposes unit phasors (integer cos/sin LUT) and recovers a phase per
//! dimension via `atan2`. The stored state is integer phases and the bundle is a
//! deterministic integer fold of its inputs, so a phase-vector memory is an exact,
//! replay-verifiable function of what was added -- similarity is a float readout at
//! query time, exactly like [`super::entangled::EntangledHVec`].
//!
//! This is the deterministic substrate for the phase resonator
//! ([`super::phase_resonator`]); §20 of `docs/PREREGISTRATION-binding-readout.md`
//! shows it matches float FHRR factorization capacity down to 4-bit phase (N=16),
//! and §15-18 characterise its continuous associative-memory behaviour. Unlike the
//! sparse-binary core, phase binding is NOT self-inverse, so it does not leak
//! pairing the way XOR does.

use std::borrow::Borrow;
use std::f64::consts::TAU;

/// Fixed-point unit for the integer cos/sin LUT used in bundling.
const SCALE: i64 = 1 << 14;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseHVec {
    dim: usize,
    n: u32,
    phases: Vec<u16>,
}

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

/// Integer cos/sin lookup tables of length `n`, scaled by `SCALE`.
fn luts(n: u32) -> (Vec<i64>, Vec<i64>) {
    let cos = (0..n)
        .map(|k| (SCALE as f64 * (TAU * k as f64 / n as f64).cos()).round() as i64)
        .collect();
    let sin = (0..n)
        .map(|k| (SCALE as f64 * (TAU * k as f64 / n as f64).sin()).round() as i64)
        .collect();
    (cos, sin)
}

impl PhaseHVec {
    /// A random phase vector of `dim` dimensions at resolution `n` (phase levels),
    /// deterministic in `seed`. `n` must be in `2..=65536`.
    pub fn new_random(dim: usize, n: u32, seed: u64) -> Self {
        assert!(
            (2..=65536).contains(&n),
            "phase resolution n must be in 2..=65536"
        );
        let phases = (0..dim)
            .map(|d| (mix(d as u64, seed) % n as u64) as u16)
            .collect();
        Self { dim, n, phases }
    }

    /// Construct from explicit phase indices; every phase must be `< n`.
    pub fn from_phases(phases: Vec<u16>, n: u32) -> Self {
        assert!(
            (2..=65536).contains(&n),
            "phase resolution n must be in 2..=65536"
        );
        assert!(
            phases.iter().all(|&p| (p as u32) < n),
            "phase index out of range"
        );
        Self {
            dim: phases.len(),
            n,
            phases,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn phases(&self) -> &[u16] {
        &self.phases
    }

    fn check(&self, other: &Self) {
        assert_eq!(self.dim, other.dim, "phase-vector dimension mismatch");
        assert_eq!(self.n, other.n, "phase-vector resolution mismatch");
    }

    /// Binding: phase-add mod N. Non-self-inverse; inverse is [`Self::unbind`].
    pub fn bind(&self, other: &Self) -> Self {
        self.check(other);
        let n = self.n;
        let phases = self
            .phases
            .iter()
            .zip(&other.phases)
            .map(|(&a, &b)| ((a as u32 + b as u32) % n) as u16)
            .collect();
        Self {
            dim: self.dim,
            n: self.n,
            phases,
        }
    }

    /// Unbinding: phase-sub mod N. `bind(a,b).unbind(b) == a` exactly.
    pub fn unbind(&self, other: &Self) -> Self {
        self.check(other);
        let n = self.n;
        let phases = self
            .phases
            .iter()
            .zip(&other.phases)
            .map(|(&a, &b)| ((a as u32 + n - b as u32) % n) as u16)
            .collect();
        Self {
            dim: self.dim,
            n: self.n,
            phases,
        }
    }

    /// Mean cosine of phase differences in `[-1, 1]`. 1.0 for identical vectors,
    /// ~0 for independent ones. Float readout (query time).
    pub fn similarity(&self, other: &Self) -> f64 {
        self.check(other);
        if self.dim == 0 {
            return 1.0;
        }
        let n = self.n as f64;
        let s: f64 = self
            .phases
            .iter()
            .zip(&other.phases)
            .map(|(&a, &b)| (TAU * (a as f64 - b as f64) / n).cos())
            .sum();
        s / self.dim as f64
    }

    /// Superpose (bundle) phase vectors: sum unit phasors via integer LUT, recover
    /// a phase per dimension with `atan2`. Deterministic integer fold -> the result
    /// is an exact, replay-verifiable function of the inputs (and their order does
    /// not matter, addition being commutative). Empty input yields a 0-dim vector.
    pub fn bundle<V: Borrow<Self>>(vectors: &[V]) -> Self {
        if vectors.is_empty() {
            return Self {
                dim: 0,
                n: 2,
                phases: Vec::new(),
            };
        }
        let first = vectors[0].borrow();
        let (dim, n) = (first.dim, first.n);
        let (cos, sin) = luts(n);
        let mut re = vec![0i64; dim];
        let mut im = vec![0i64; dim];
        for v in vectors {
            let v = v.borrow();
            assert_eq!(v.dim, dim, "phase-vector dimension mismatch in bundle");
            assert_eq!(v.n, n, "phase-vector resolution mismatch in bundle");
            for d in 0..dim {
                re[d] += cos[v.phases[d] as usize];
                im[d] += sin[v.phases[d] as usize];
            }
        }
        let phases = (0..dim)
            .map(|d| {
                let ang = (im[d] as f64).atan2(re[d] as f64);
                let idx = (ang / TAU * n as f64).round() as i64;
                idx.rem_euclid(n as i64) as u16
            })
            .collect();
        Self { dim, n, phases }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_unbind_is_exact_roundtrip() {
        let a = PhaseHVec::new_random(512, 256, 1);
        let b = PhaseHVec::new_random(512, 256, 2);
        let recovered = a.bind(&b).unbind(&b);
        assert_eq!(
            recovered.phases(),
            a.phases(),
            "unbind(bind(a,b),b) must equal a"
        );
    }

    #[test]
    fn similarity_self_is_one_random_near_zero() {
        let a = PhaseHVec::new_random(4096, 256, 7);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-9);
        let b = PhaseHVec::new_random(4096, 256, 8);
        assert!(
            a.similarity(&b).abs() < 0.05,
            "independent vectors ~orthogonal"
        );
    }

    #[test]
    fn binding_is_not_self_inverse() {
        // XOR leaks because bind(a,a) collapses; phase-add does not.
        let a = PhaseHVec::new_random(2048, 256, 3);
        let self_bound = a.bind(&a);
        assert!(
            self_bound.similarity(&a).abs() < 0.2,
            "phase self-bind must not collapse back to a (non-self-inverse)"
        );
    }

    #[test]
    fn bundle_preserves_membership() {
        let members: Vec<PhaseHVec> = (0..5)
            .map(|i| PhaseHVec::new_random(4096, 256, 100 + i))
            .collect();
        let bundle = PhaseHVec::bundle(&members);
        let member_sim = bundle.similarity(&members[0]);
        let nonmember = PhaseHVec::new_random(4096, 256, 999);
        assert!(
            member_sim > bundle.similarity(&nonmember) + 0.05,
            "a bundled member ({member_sim:.3}) must score above a non-member"
        );
    }

    #[test]
    fn bundle_is_deterministic_and_order_free() {
        let v: Vec<PhaseHVec> = (0..4)
            .map(|i| PhaseHVec::new_random(256, 64, 10 + i))
            .collect();
        let ab = PhaseHVec::bundle(&v);
        let mut rev = v.clone();
        rev.reverse();
        let ba = PhaseHVec::bundle(&rev);
        assert_eq!(
            ab.phases(),
            ba.phases(),
            "bundle must be order-independent and deterministic"
        );
    }
}
