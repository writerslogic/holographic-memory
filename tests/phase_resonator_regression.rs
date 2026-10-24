// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use holographic_memory::core::{phase_hvec, phase_resonator};

#[allow(dead_code)]
#[path = "../benchmarks/results/holography_resonator_v1/baseline.rs"]
mod baseline;
#[allow(dead_code)]
#[path = "../benchmarks/research/phase_resonator/cases.rs"]
mod cases;

use cases::{assert_equal, baseline_config, check, config, Scenario};
use phase_hvec::PhaseHVec;
use phase_resonator::PhaseResonator;

#[test]
fn preserves_factors_through_capacity_knee_and_corrupted_queries() {
    for seed in [0x5EED, 0xC0DE] {
        for n in [16, 256] {
            for entries in [8, 24, 40, 48] {
                let s = Scenario {
                    dim: 1024,
                    n,
                    factors: 3,
                    entries,
                    max_iter: 40,
                    seed,
                };
                check(s, 2, false);
                check(s, 1, true);
            }
        }
    }
}

#[test]
fn packed_phase_width_and_table_cutoff_boundaries_match_baseline() {
    for n in [2, 15, 16, 17, 255, 256, 257, 4095, 4096, 4097, 65535, 65536] {
        for dim in [0, 1, 16, 17] {
            check(
                Scenario {
                    dim,
                    n,
                    factors: 2,
                    entries: 3,
                    max_iter: 6,
                    seed: 0xAB,
                },
                2,
                false,
            );
        }
    }
    for n in [4095, 4096, 4097] {
        check(
            Scenario {
                dim: 1025,
                n,
                factors: 2,
                entries: 3,
                max_iter: 6,
                seed: 0xCAFE,
            },
            1,
            false,
        );
    }
}

#[test]
fn iteration_limits_and_single_axis_match_baseline() {
    for max_iter in [0, 1, 2, 3, 4, 5] {
        for (factors, entries) in [(1, 1), (1, 7), (4, 5)] {
            check(
                Scenario {
                    dim: 127,
                    n: 16,
                    factors,
                    entries,
                    max_iter,
                    seed: 3,
                },
                2,
                false,
            );
        }
    }
}

#[test]
fn ties_keep_the_last_entry_including_odd_nibble_boundaries() {
    for n in [16, 256, 65536] {
        let entry = PhaseHVec::from_phases(vec![0, 1, (n - 1) as u16], n);
        let books = vec![vec![entry.clone(); 3], vec![entry.clone(); 5]];
        let composite = entry.bind(&entry);
        let old = baseline::PhaseResonator::with_config(&books, baseline_config(8));
        let expected = old.factorize(&composite);
        assert_eq!(expected[0].codebook_entry, 2);
        assert_eq!(expected[1].codebook_entry, 4);
        assert_eq!(expected[0].iterations, 4);
        assert!(expected[0].converged);
        assert_equal(
            &expected,
            &PhaseResonator::new(&books).factorize(&composite),
        );
        assert_equal(
            &expected,
            &PhaseResonator::with_packed_codebooks(&books, config(8)).factorize(&composite),
        );
    }
}

#[test]
fn dimension_mismatch_is_rejected_before_factorization() {
    let books = vec![vec![PhaseHVec::new_random(16, 16, 1)]];
    for packed in [false, true] {
        let index = if packed {
            PhaseResonator::with_packed_codebooks(&books, config(8))
        } else {
            PhaseResonator::new(&books)
        };
        for dim in [15, 16, 17] {
            let query = PhaseHVec::new_random(dim, 16, 1);
            let result = std::panic::catch_unwind(|| index.factorize(&query));
            assert_eq!(result.is_ok(), dim == 16);
            let result = std::panic::catch_unwind(|| {
                phase_resonator::phase_resonator_factorize(&query, &books, &config(8))
            });
            assert_eq!(result.is_ok(), dim == 16);
        }
    }
}

#[test]
fn invalid_codebooks_are_rejected_by_both_backends() {
    let good = PhaseHVec::new_random(16, 16, 1);
    for books in [
        vec![vec![]],
        vec![vec![good.clone()], vec![]],
        vec![vec![good.clone(), PhaseHVec::new_random(15, 16, 2)]],
        vec![vec![good.clone()], vec![PhaseHVec::new_random(17, 16, 2)]],
        vec![vec![good.clone(), PhaseHVec::new_random(16, 15, 2)]],
        vec![vec![good.clone()], vec![PhaseHVec::new_random(16, 17, 2)]],
    ] {
        assert!(std::panic::catch_unwind(|| PhaseResonator::new(&books)).is_err());
        assert!(
            std::panic::catch_unwind(|| PhaseResonator::with_packed_codebooks(&books, config(8)))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| phase_resonator::phase_resonator_factorize(
                &good,
                &books,
                &config(8)
            ))
            .is_err()
        );
    }
}

#[test]
fn empty_outer_codebooks_accept_any_composite_in_both_backends() {
    let query = PhaseHVec::new_random(17, 65536, 1);
    assert!(PhaseResonator::new(&[]).factorize(&query).is_empty());
    assert!(PhaseResonator::with_packed_codebooks(&[], config(8))
        .factorize(&query)
        .is_empty());
    assert!(phase_resonator::phase_resonator_factorize(&query, &[], &config(8)).is_empty());
}

#[test]
fn composite_resolution_mismatch_is_rejected_by_all_entry_points() {
    let books = vec![vec![PhaseHVec::new_random(16, 16, 1)]];
    for n in [15, 16, 17] {
        let query = PhaseHVec::new_random(16, n, 1);
        assert_eq!(
            std::panic::catch_unwind(|| PhaseResonator::new(&books).factorize(&query)).is_ok(),
            n == 16
        );
        assert_eq!(
            std::panic::catch_unwind(
                || PhaseResonator::with_packed_codebooks(&books, config(8)).factorize(&query)
            )
            .is_ok(),
            n == 16
        );
        assert_eq!(
            std::panic::catch_unwind(|| phase_resonator::phase_resonator_factorize(
                &query,
                &books,
                &config(8)
            ))
            .is_ok(),
            n == 16
        );
    }
}
