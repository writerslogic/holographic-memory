// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experimental finite-alphabet codes with fully accounted storage and allocation-free cleanup.

use super::{entangled::hash_u64, qgraph::kernels::fwht};
use anyhow::{ensure, Result};
use nalgebra::Complex;
use serde::Serialize;
use std::f64::consts::TAU;
use std::mem::size_of;

pub const MAX_DIM: usize = 8192;
pub const MAX_LABELS: usize = 256;

fn random_phase(seed: u64, row: usize, coordinate: usize) -> u16 {
    let row_seed = hash_u64(seed ^ 0xC0DE, row as u64);
    (hash_u64(row_seed, coordinate as u64) & 15) as u16
}

fn shape(dim: usize, count: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_DIM).contains(&dim),
        "dimension must be in 1..=8192"
    );
    ensure!(
        (1..=MAX_LABELS).contains(&count),
        "count must be in 1..=256"
    );
    dim.checked_mul(count)
        .ok_or_else(|| anyhow::anyhow!("codebook size overflows"))
}

#[derive(Debug)]
pub struct PackedPhaseCodes {
    dim: usize,
    count: usize,
    bytes: Vec<u8>,
}

impl PackedPhaseCodes {
    pub fn random(dim: usize, count: usize, seed: u64) -> Result<Self> {
        let length = shape(dim, count)?;
        let mut bytes = vec![0; length.div_ceil(2)];
        for index in 0..length {
            let phase = random_phase(seed, index / dim, index % dim) as u8;
            bytes[index / 2] |= phase << (4 * (index % 2));
        }
        Ok(Self { dim, count, bytes })
    }

    pub fn from_phases(dim: usize, phases: &[u16]) -> Result<Self> {
        ensure!(
            dim > 0 && phases.len().is_multiple_of(dim),
            "invalid row shape"
        );
        let count = phases.len() / dim;
        let length = shape(dim, count)?;
        ensure!(phases.iter().all(|&p| p < 16), "phase must be in 0..16");
        let mut bytes = vec![0; length.div_ceil(2)];
        for (index, &phase) in phases.iter().enumerate() {
            bytes[index / 2] |= (phase as u8) << (4 * (index % 2));
        }
        Ok(Self { dim, count, bytes })
    }

    pub fn phase(&self, row: usize, coordinate: usize) -> u16 {
        assert!(
            row < self.count && coordinate < self.dim,
            "code coordinate out of range"
        );
        let index = row * self.dim + coordinate;
        ((self.bytes[index / 2] >> (4 * (index % 2))) & 15) as u16
    }

    pub fn storage_bytes(&self) -> usize {
        size_of::<Self>() + self.bytes.capacity()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueCodeKind {
    UnpackedRandom,
    PackedRandom,
    RegeneratedRandom,
    Walsh,
}

#[derive(Debug)]
pub struct ValueCodebook {
    kind: ValueCodeKind,
    dim: usize,
    labels: usize,
    seed: u64,
    unpacked: Vec<u16>,
    packed: Vec<u8>,
    lut: [Complex<f64>; 16],
}

impl ValueCodebook {
    pub fn new(kind: ValueCodeKind, dim: usize, labels: usize, seed: u64) -> Result<Self> {
        let length = shape(dim, labels)?;
        ensure!(
            labels >= 2 && labels.is_power_of_two(),
            "labels must be a power of two in 2..=256"
        );
        ensure!(
            kind != ValueCodeKind::Walsh || dim.is_multiple_of(labels),
            "Walsh dimension must be divisible by label count"
        );
        let mut result = Self {
            kind,
            dim,
            labels,
            seed,
            unpacked: Vec::new(),
            packed: Vec::new(),
            lut: std::array::from_fn(|p| {
                let angle = TAU * p as f64 / 16.0;
                Complex::new(angle.cos(), angle.sin())
            }),
        };
        if kind == ValueCodeKind::UnpackedRandom {
            result.unpacked = (0..length)
                .map(|i| random_phase(seed, i / dim, i % dim))
                .collect();
        } else if kind == ValueCodeKind::PackedRandom {
            result.packed = vec![0; length.div_ceil(2)];
            for i in 0..length {
                result.packed[i / 2] |=
                    (random_phase(seed, i / dim, i % dim) as u8) << (4 * (i % 2));
            }
        }
        Ok(result)
    }

    pub fn phase(&self, label: usize, coordinate: usize) -> u16 {
        assert!(
            label < self.labels && coordinate < self.dim,
            "value coordinate out of range"
        );
        let index = label * self.dim + coordinate;
        match self.kind {
            ValueCodeKind::UnpackedRandom => self.unpacked[index],
            ValueCodeKind::PackedRandom => {
                ((self.packed[index / 2] >> (4 * (index % 2))) & 15) as u16
            }
            ValueCodeKind::RegeneratedRandom => random_phase(self.seed, label, coordinate),
            ValueCodeKind::Walsh => {
                8 * ((label & (coordinate % self.labels)).count_ones() % 2) as u16
            }
        }
    }

    pub fn phasor(&self, phase: u16) -> Complex<f64> {
        self.lut[usize::from(phase)]
    }

    pub fn scores(&self, raw: &[Complex<f64>], scores: &mut [f64]) -> Result<()> {
        ensure!(
            raw.len() == self.dim && scores.len() == self.labels,
            "readout shape mismatch"
        );
        ensure!(
            raw.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "readout contains non-finite values"
        );
        scores.fill(0.0);
        if self.kind == ValueCodeKind::Walsh {
            for (d, z) in raw.iter().enumerate() {
                scores[d % self.labels] += z.re;
            }
            fwht(scores);
        } else {
            for (label, score) in scores.iter_mut().enumerate() {
                *score = raw
                    .iter()
                    .enumerate()
                    .map(|(d, &z)| (z * self.phasor(self.phase(label, d)).conj()).re)
                    .sum();
            }
        }
        ensure!(
            scores.iter().all(|x| x.is_finite()),
            "readout scores overflow"
        );
        Ok(())
    }

    pub fn storage_bytes(&self) -> usize {
        size_of::<Self>() + self.unpacked.capacity() * size_of::<u16>() + self.packed.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_packed_and_regenerated_codes_have_identical_scores() {
        for dim in [1, 3, 64, 129] {
            let books: Vec<_> = [
                ValueCodeKind::UnpackedRandom,
                ValueCodeKind::PackedRandom,
                ValueCodeKind::RegeneratedRandom,
            ]
            .into_iter()
            .map(|kind| ValueCodebook::new(kind, dim, 16, 91).unwrap())
            .collect();
            let raw: Vec<_> = (0..dim)
                .map(|d| Complex::new((d as f64 * 0.3).cos(), (d as f64 * 0.7).sin()))
                .collect();
            let mut scores = vec![0.0; 16];
            books[0].scores(&raw, &mut scores).unwrap();
            for book in &books[1..] {
                let mut other = vec![0.0; 16];
                book.scores(&raw, &mut other).unwrap();
                assert_eq!(scores, other);
                for label in 0..16 {
                    for d in 0..dim {
                        assert_eq!(books[0].phase(label, d), book.phase(label, d));
                    }
                }
            }
        }
    }

    #[test]
    fn transform_scores_match_explicit_complex_inner_products() {
        for labels in [2, 16, 64, 256] {
            let dim = labels * 3;
            let book = ValueCodebook::new(ValueCodeKind::Walsh, dim, labels, 3).unwrap();
            let raw: Vec<_> = (0..dim)
                .map(|d| Complex::new((d as f64 * 0.7).cos(), (d as f64 * 0.3).sin()))
                .collect();
            let mut scores = vec![0.0; labels];
            book.scores(&raw, &mut scores).unwrap();
            for (label, &score) in scores.iter().enumerate() {
                let expected: f64 = raw
                    .iter()
                    .enumerate()
                    .map(|(d, &z)| {
                        let angle = TAU * f64::from(book.phase(label, d)) / 16.0;
                        (z * Complex::new(angle.cos(), -angle.sin())).re
                    })
                    .sum();
                assert!((score - expected).abs() < 1e-10);
            }
            for label in 0..labels {
                let code: Vec<_> = (0..dim)
                    .map(|d| book.phasor(book.phase(label, d)))
                    .collect();
                book.scores(&code, &mut scores).unwrap();
                for (other, &score) in scores.iter().enumerate() {
                    assert!((score - if other == label { dim as f64 } else { 0.0 }).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn packed_odd_rows_preserve_nibble_boundaries() {
        let phases = [0, 14, 15, 1, 15, 2];
        let codes = PackedPhaseCodes::from_phases(3, &phases).unwrap();
        for (i, phase) in phases.into_iter().enumerate() {
            assert_eq!(codes.phase(i / 3, i % 3), phase);
        }
        assert!(PackedPhaseCodes::from_phases(3, &[0, 1]).is_err());
        assert!(PackedPhaseCodes::from_phases(1, &[16]).is_err());
        assert!(PackedPhaseCodes::from_phases(0, &[]).is_err());
        let codes = PackedPhaseCodes::from_phases(3, &[15, 14, 0]).unwrap();
        assert_eq!(codes.bytes, [239, 0]);
    }

    #[test]
    fn dimensions_labels_and_readout_are_bounded_before_allocation() {
        for dim in [MAX_DIM - 1, MAX_DIM] {
            assert!(ValueCodebook::new(ValueCodeKind::RegeneratedRandom, dim, 2, 0).is_ok());
        }
        for dim in [0, MAX_DIM + 1, usize::MAX] {
            assert!(ValueCodebook::new(ValueCodeKind::RegeneratedRandom, dim, 2, 0).is_err());
        }
        assert!(ValueCodebook::new(ValueCodeKind::Walsh, MAX_DIM, MAX_LABELS, 0).is_ok());
        for labels in [0, 1, MAX_LABELS - 1, MAX_LABELS + 1] {
            assert!(ValueCodebook::new(ValueCodeKind::RegeneratedRandom, 64, labels, 0).is_err());
        }
        for dim in [63, 65] {
            assert!(ValueCodebook::new(ValueCodeKind::Walsh, dim, 64, 0).is_err());
        }
        let book = ValueCodebook::new(ValueCodeKind::Walsh, 64, 64, 0).unwrap();
        assert!(book
            .scores(&[Complex::new(0.0, 0.0); 63], &mut [0.0; 64])
            .is_err());
        assert!(book
            .scores(&[Complex::new(0.0, 0.0); 64], &mut [0.0; 63])
            .is_err());
        assert!(book
            .scores(&[Complex::new(f64::NAN, 0.0); 64], &mut [0.0; 64])
            .is_err());
        assert!(book
            .scores(&[Complex::new(f64::MAX, 0.0); 64], &mut [0.0; 64])
            .is_err());
    }
}
