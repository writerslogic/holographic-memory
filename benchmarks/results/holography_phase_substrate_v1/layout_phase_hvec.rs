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
use std::sync::OnceLock;

/// Fixed-point unit for the integer cos/sin LUT used in bundling.
const SCALE: i64 = 1 << 14;

// Caching power-of-two resolutions bounds the initialized table payload to less than 1.5 MiB.
const CACHED_RESOLUTIONS: usize = 16;
static COSINES: [OnceLock<Box<[f64]>>; CACHED_RESOLUTIONS] =
    [const { OnceLock::new() }; CACHED_RESOLUTIONS];
static BUNDLE_LUTS: [OnceLock<Box<[[i16; 2]]>>; CACHED_RESOLUTIONS] =
    [const { OnceLock::new() }; CACHED_RESOLUTIONS];

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

/// Integer unit phasors scaled by `SCALE`; both components fit exactly in `i16`.
fn luts(n: u32) -> Box<[[i16; 2]]> {
    (0..n)
        .map(|k| {
            let angle = TAU * k as f64 / n as f64;
            [
                (SCALE as f64 * angle.cos()).round() as i16,
                (SCALE as f64 * angle.sin()).round() as i16,
            ]
        })
        .collect()
}

fn cached_resolution(n: u32) -> Option<usize> {
    n.is_power_of_two().then(|| n.trailing_zeros() as usize - 1)
}

fn cosine_lut(n: u32, dim: usize) -> Option<&'static [f64]> {
    let entry = &COSINES[cached_resolution(n)?];
    if n > 256 && dim < n as usize {
        return entry.get().map(AsRef::as_ref);
    }
    Some(entry.get_or_init(|| {
        (0..n)
            .map(|delta| (TAU * delta as f64 / n as f64).cos())
            .collect()
    }))
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

    /// Number of dimensions (phase components).
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Phase resolution: the number of discrete phase levels `N` (each phase is in
    /// `0..n`).
    pub fn n(&self) -> u32 {
        self.n
    }
    /// The raw phase indices, one per dimension, each in `0..n`.
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
            .map(|(&a, &b)| {
                let sum = a as u32 + b as u32;
                (if sum >= n { sum - n } else { sum }) as u16
            })
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
            .map(|(&a, &b)| {
                let difference = a as u32 + n - b as u32;
                (if difference >= n {
                    difference - n
                } else {
                    difference
                }) as u16
            })
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
        let s: f64 = if let Some(cos) = cosine_lut(self.n, self.dim) {
            self.phases
                .iter()
                .zip(&other.phases)
                .map(|(&a, &b)| cos[a.abs_diff(b) as usize])
                .sum()
        } else {
            let n = self.n as f64;
            self.phases
                .iter()
                .zip(&other.phases)
                .map(|(&a, &b)| (TAU * (a as f64 - b as f64) / n).cos())
                .sum()
        };
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
        for v in vectors {
            let v = v.borrow();
            assert_eq!(v.dim, dim, "phase-vector dimension mismatch in bundle");
            assert_eq!(v.n, n, "phase-vector resolution mismatch in bundle");
        }
        if dim == 0 || vectors.len() == 1 {
            return first.clone();
        }
        if n == 2 {
            let mut ones = vec![0usize; dim];
            for v in vectors {
                for (count, &phase) in ones.iter_mut().zip(&v.borrow().phases) {
                    *count += phase as usize;
                }
            }
            let phases = ones
                .into_iter()
                .map(|count| u16::from(count > vectors.len() / 2))
                .collect();
            return Self { dim, n, phases };
        }
        let owned;
        let lut = if let Some(index) = cached_resolution(n) {
            BUNDLE_LUTS[index].get_or_init(|| luts(n))
        } else {
            owned = luts(n);
            &owned
        };
        let mut re = vec![0i64; dim];
        let mut im = vec![0i64; dim];
        for v in vectors {
            let v = v.borrow();
            for ((re, im), &phase) in re.iter_mut().zip(&mut im).zip(&v.phases) {
                let [cos, sin] = lut[phase as usize];
                *re += cos as i64;
                *im += sin as i64;
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

    #[test]
    fn bundle_empty_is_zero_dim() {
        let empty: Vec<PhaseHVec> = Vec::new();
        let b = PhaseHVec::bundle(&empty);
        assert_eq!(b.dim(), 0);
        assert!(b.phases().is_empty());
    }

    #[test]
    fn bundle_single_element_is_identity() {
        let a = PhaseHVec::new_random(512, 256, 42);
        let b = PhaseHVec::bundle(std::slice::from_ref(&a));
        assert_eq!(
            b.phases(),
            a.phases(),
            "bundling one vector must return its own phases"
        );
    }

    #[test]
    fn zero_dim_similarity_is_one() {
        // Empty vectors are vacuously identical (guarded in `similarity`).
        let a = PhaseHVec::from_phases(Vec::new(), 2);
        let b = PhaseHVec::from_phases(Vec::new(), 2);
        assert_eq!(a.dim(), 0);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn resolution_outside_2_to_65536_is_rejected_and_bounds_roundtrip() {
        for (n, valid) in [
            (1u32, false),
            (2, true),
            (3, true),
            (65535, true),
            (65536, true),
            (65537, false),
        ] {
            let built = std::panic::catch_unwind(|| PhaseHVec::new_random(128, n, 3));
            assert_eq!(built.is_ok(), valid, "n={n}");
            if let Ok(a) = built {
                let b = PhaseHVec::new_random(128, n, 4);
                assert_eq!(a.n(), n, "n={n}");
                assert_eq!(a.bind(&b).unbind(&b).phases(), a.phases(), "n={n}");
                assert!((a.similarity(&a) - 1.0).abs() < 1e-9, "n={n}");
            }
        }
    }

    #[test]
    fn phase_arithmetic_matches_modular_reference_at_wrap_boundaries() {
        for n in [2, 3, 15, 16, 17, 255, 256, 257, 65535, 65536] {
            let phases = [0, 1, n / 2, n - 2, n - 1];
            for a in phases {
                for b in phases {
                    let left = PhaseHVec::from_phases(vec![a as u16], n);
                    let right = PhaseHVec::from_phases(vec![b as u16], n);
                    assert_eq!(left.bind(&right).phases(), &[((a + b) % n) as u16]);
                    assert_eq!(left.unbind(&right).phases(), &[((a + n - b) % n) as u16]);
                }
            }
        }
    }

    #[test]
    fn similarity_matches_direct_cosine_for_all_phase_differences() {
        for n in [2, 3, 15, 16, 17, 255, 256, 257, 65535, 65536] {
            let left = PhaseHVec::from_phases((0..n).map(|p| p as u16).collect(), n);
            let right = PhaseHVec::from_phases(vec![0; n as usize], n);
            let reference = left
                .phases()
                .iter()
                .map(|&p| (TAU * p as f64 / n as f64).cos())
                .sum::<f64>()
                / n as f64;
            assert_eq!(left.similarity(&right).to_bits(), reference.to_bits());
            assert_eq!(right.similarity(&left).to_bits(), reference.to_bits());
            for pivot in [0, 1, n / 2, n - 2, n - 1] {
                let right = PhaseHVec::from_phases(vec![pivot as u16; n as usize], n);
                let reference = left
                    .phases()
                    .iter()
                    .map(|&p| (TAU * (p as f64 - pivot as f64) / n as f64).cos())
                    .sum::<f64>()
                    / n as f64;
                assert_eq!(left.similarity(&right).to_bits(), reference.to_bits());
            }
        }
    }

    #[test]
    fn bundles_match_direct_phasor_sums_at_all_resolutions() {
        for n in [2, 3, 15, 16, 17, 255, 256, 257, 65535, 65536] {
            let vectors: Vec<_> = (0..7)
                .map(|seed| PhaseHVec::new_random(129, n, seed))
                .collect();
            for count in [1, 2, 3, 7] {
                let reference: Vec<u16> = (0..129)
                    .map(|d| {
                        let (re, im) = vectors[..count].iter().fold((0i64, 0i64), |(re, im), v| {
                            let phase = TAU * v.phases()[d] as f64 / n as f64;
                            (
                                re + (SCALE as f64 * phase.cos()).round() as i64,
                                im + (SCALE as f64 * phase.sin()).round() as i64,
                            )
                        });
                        let angle = (im as f64).atan2(re as f64);
                        ((angle / TAU * n as f64).round() as i64).rem_euclid(n as i64) as u16
                    })
                    .collect();
                assert_eq!(PhaseHVec::bundle(&vectors[..count]).phases(), reference);
            }
        }
    }

    #[test]
    fn singleton_bundle_matches_fixed_point_roundtrip_for_every_phase() {
        for n in [2, 3, 16, 256, 257, 65535, 65536] {
            let phases = PhaseHVec::from_phases((0..n).map(|p| p as u16).collect(), n);
            let reference: Vec<u16> = phases
                .phases()
                .iter()
                .map(|&p| {
                    let angle = TAU * p as f64 / n as f64;
                    let re = (SCALE as f64 * angle.cos()).round() as i64;
                    let im = (SCALE as f64 * angle.sin()).round() as i64;
                    (((im as f64).atan2(re as f64) / TAU * n as f64).round() as i64)
                        .rem_euclid(n as i64) as u16
                })
                .collect();
            assert_eq!(reference, phases.phases());
            assert_eq!(PhaseHVec::bundle(&[&phases]).phases(), reference);
        }
    }

    #[test]
    fn binary_bundle_matches_phasor_sum_for_every_vote_split() {
        let zero = PhaseHVec::from_phases(vec![0], 2);
        let one = PhaseHVec::from_phases(vec![1], 2);
        for count in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256, 257] {
            for ones in 0..=count {
                let mut vectors = vec![&zero; count];
                vectors[..ones].fill(&one);
                let (re, im) = vectors.iter().fold((0i64, 0i64), |(re, im), v| {
                    let phase = TAU * v.phases()[0] as f64 / 2.0;
                    (
                        re + (SCALE as f64 * phase.cos()).round() as i64,
                        im + (SCALE as f64 * phase.sin()).round() as i64,
                    )
                });
                let reference = (((im as f64).atan2(re as f64) / TAU * 2.0).round() as i64)
                    .rem_euclid(2) as u16;
                assert_eq!(PhaseHVec::bundle(&vectors).phases(), &[reference]);
            }
        }
    }

    #[test]
    fn zero_dim_bundles_still_check_resolution() {
        let vectors = [
            PhaseHVec::from_phases(Vec::new(), 16),
            PhaseHVec::from_phases(Vec::new(), 256),
        ];
        assert!(std::panic::catch_unwind(|| PhaseHVec::bundle(&vectors)).is_err());
    }
}
