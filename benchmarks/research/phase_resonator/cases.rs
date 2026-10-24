// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::{baseline, phase_hvec::PhaseHVec, phase_resonator};

#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    pub dim: usize,
    pub n: u32,
    pub factors: usize,
    pub entries: usize,
    pub max_iter: usize,
    pub seed: u64,
}

pub struct Dataset {
    pub books: Vec<Vec<PhaseHVec>>,
    pub composites: Vec<PhaseHVec>,
    pub targets: Vec<Vec<usize>>,
}

pub fn dataset(s: Scenario, trials: usize, corrupt: bool) -> Dataset {
    let books: Vec<Vec<PhaseHVec>> = (0..s.factors)
        .map(|axis| {
            (0..s.entries)
                .map(|entry| {
                    PhaseHVec::new_random(s.dim, s.n, s.seed ^ ((axis as u64) << 40) ^ entry as u64)
                })
                .collect()
        })
        .collect();
    let mut composites = Vec::new();
    let mut targets = Vec::new();
    for trial in 0..trials {
        let target: Vec<usize> = (0..s.factors)
            .map(|axis| (trial * 7 + axis * 3 + 1) % s.entries)
            .collect();
        let mut composite = books[0][target[0]].clone();
        for axis in 1..s.factors {
            composite = composite.bind(&books[axis][target[axis]]);
        }
        if corrupt {
            let noise = PhaseHVec::new_random(s.dim, s.n, s.seed ^ 0xBAD0_0000 ^ trial as u64);
            let mut phases = composite.phases().to_vec();
            for d in (trial % 10..s.dim).step_by(10) {
                phases[d] = noise.phases()[d];
            }
            composite = PhaseHVec::from_phases(phases, s.n);
        }
        composites.push(composite);
        targets.push(target);
    }
    Dataset {
        books,
        composites,
        targets,
    }
}

pub fn config(max_iter: usize) -> phase_resonator::ResonatorConfig {
    phase_resonator::ResonatorConfig {
        max_iter,
        convergence_threshold: 0.999,
    }
}

pub fn baseline_config(max_iter: usize) -> baseline::ResonatorConfig {
    baseline::ResonatorConfig {
        max_iter,
        convergence_threshold: 0.999,
    }
}

pub fn assert_equal(expected: &[baseline::FactorResult], actual: &[phase_resonator::FactorResult]) {
    assert_equal_context(expected, actual, "benchmark or boundary case");
}

fn assert_equal_context(
    expected: &[baseline::FactorResult],
    actual: &[phase_resonator::FactorResult],
    context: impl std::fmt::Debug,
) {
    assert_eq!(actual.len(), expected.len());
    for (old, new) in expected.iter().zip(actual) {
        assert_eq!(new.factor_idx, old.factor_idx, "factor index");
        assert_eq!(new.codebook_entry, old.codebook_entry, "factor ID");
        assert_eq!(new.converged, old.converged, "convergence");
        assert_eq!(new.iterations, old.iterations, "iterations");
        assert_eq!(
            new.similarity.to_bits(),
            old.similarity.to_bits(),
            "similarity: {context:?}"
        );
    }
}

pub fn check(s: Scenario, trials: usize, corrupt: bool) {
    let data = dataset(s, trials, corrupt);
    let old = baseline::PhaseResonator::with_config(&data.books, baseline_config(s.max_iter));
    let expanded = phase_resonator::PhaseResonator::with_config(&data.books, config(s.max_iter));
    let packed =
        phase_resonator::PhaseResonator::with_packed_codebooks(&data.books, config(s.max_iter));
    for (query, composite) in data.composites.iter().enumerate() {
        let expected = old.factorize(composite);
        assert_equal_context(
            &expected,
            &expanded.factorize(composite),
            (s, corrupt, query, "expanded"),
        );
        assert_equal_context(
            &expected,
            &packed.factorize(composite),
            (s, corrupt, query, "packed"),
        );
        assert_equal_context(
            &expected,
            &phase_resonator::phase_resonator_factorize(
                composite,
                &data.books,
                &config(s.max_iter),
            ),
            (s, corrupt, query, "one-shot"),
        );
    }
}
