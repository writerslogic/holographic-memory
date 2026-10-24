// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Standalone comparison against the phase-vector implementation at 12c80dec.

#[allow(dead_code)]
#[path = "../../results/holography_phase_substrate_v1/baseline_phase_hvec.rs"]
mod baseline;
#[allow(dead_code)]
#[path = "../../../src/core/phase_hvec.rs"]
mod current;

use std::hint::black_box;
use std::time::Instant;

fn elapsed(mut run: impl FnMut(), iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    start.elapsed().as_secs_f64() * 1e9
}

fn calibrate(run: &mut impl FnMut(), target_ns: f64) -> usize {
    let mut iterations = 1;
    loop {
        let ns = elapsed(&mut *run, iterations);
        if ns >= target_ns / 4.0 || iterations >= 1_000_000 {
            return ((iterations as f64 * target_ns / ns.max(1.0)) as usize).clamp(1, 1_000_000);
        }
        iterations *= 4;
    }
}

fn benchmark(
    operation: &str,
    mut old: impl FnMut(),
    mut new: impl FnMut(),
    target_ns: f64,
    repeats: usize,
    trial: usize,
) {
    let old_iterations = calibrate(&mut old, target_ns);
    let new_iterations = calibrate(&mut new, target_ns);
    let mut old_ns = Vec::with_capacity(repeats);
    let mut new_ns = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let (a, b) = if (repeat + trial).is_multiple_of(2) {
            let a = elapsed(&mut old, old_iterations);
            let b = elapsed(&mut new, new_iterations);
            (a, b)
        } else {
            let b = elapsed(&mut new, new_iterations);
            let a = elapsed(&mut old, old_iterations);
            (a, b)
        };
        old_ns.push(a / old_iterations as f64);
        new_ns.push(b / new_iterations as f64);
    }
    println!(
        "{{\"operation\":\"{operation}\",\"baseline_iterations\":{old_iterations},\"current_iterations\":{new_iterations},\"baseline_ns\":{old_ns:?},\"current_ns\":{new_ns:?}}}"
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 6, "expected: N DIM SAMPLE_MS REPEATS TRIAL");
    let n: u32 = args[1].parse().unwrap();
    let dim: usize = args[2].parse().unwrap();
    let target_ns: f64 = args[3].parse::<f64>().unwrap() * 1e6;
    let repeats: usize = args[4].parse().unwrap();
    let trial: usize = args[5].parse().unwrap();
    assert!(target_ns.is_finite() && target_ns > 0.0 && repeats > 0);
    let old: Vec<_> = (0..32)
        .map(|seed| baseline::PhaseHVec::new_random(dim, n, seed))
        .collect();
    let new: Vec<_> = (0..32)
        .map(|seed| current::PhaseHVec::new_random(dim, n, seed))
        .collect();
    for (a, b) in old.iter().zip(&new) {
        assert_eq!(a.phases(), b.phases());
    }

    // Each invocation is a fresh process, before either global LUT is initialized.
    let mut old_similarity = 0.0;
    let mut new_similarity = 0.0;
    let mut old_bundle = None;
    let mut new_bundle = None;
    let (cold_similarity_old_ns, cold_similarity_new_ns) = if trial.is_multiple_of(2) {
        let a = elapsed(
            || old_similarity = black_box(&old[0]).similarity(black_box(&old[1])),
            1,
        );
        let b = elapsed(
            || new_similarity = black_box(&new[0]).similarity(black_box(&new[1])),
            1,
        );
        (a, b)
    } else {
        let b = elapsed(
            || new_similarity = black_box(&new[0]).similarity(black_box(&new[1])),
            1,
        );
        let a = elapsed(
            || old_similarity = black_box(&old[0]).similarity(black_box(&old[1])),
            1,
        );
        (a, b)
    };
    let (cold_bundle_old_ns, cold_bundle_new_ns) = if trial.is_multiple_of(2) {
        let a = elapsed(
            || old_bundle = Some(baseline::PhaseHVec::bundle(black_box(&old[..8]))),
            1,
        );
        let b = elapsed(
            || new_bundle = Some(current::PhaseHVec::bundle(black_box(&new[..8]))),
            1,
        );
        (a, b)
    } else {
        let b = elapsed(
            || new_bundle = Some(current::PhaseHVec::bundle(black_box(&new[..8]))),
            1,
        );
        let a = elapsed(
            || old_bundle = Some(baseline::PhaseHVec::bundle(black_box(&old[..8]))),
            1,
        );
        (a, b)
    };
    assert_eq!(old_similarity.to_bits(), new_similarity.to_bits());
    assert_eq!(old_bundle.unwrap().phases(), new_bundle.unwrap().phases());

    let mut similarity_checks = 1;
    let mut phase_checks = 0;
    for i in 0..32 {
        for j in 0..32 {
            assert_eq!(
                old[i].similarity(&old[j]).to_bits(),
                new[i].similarity(&new[j]).to_bits()
            );
            similarity_checks += 1;
        }
        assert_eq!(old[i].bind(&old[0]).phases(), new[i].bind(&new[0]).phases());
        assert_eq!(
            old[i].unbind(&old[0]).phases(),
            new[i].unbind(&new[0]).phases()
        );
        phase_checks += 2 * dim;
    }
    for count in [1, 2, 8, 32] {
        assert_eq!(
            baseline::PhaseHVec::bundle(&old[..count]).phases(),
            current::PhaseHVec::bundle(&new[..count]).phases()
        );
        phase_checks += dim;
    }
    println!(
        "{{\"n\":{n},\"dim\":{dim},\"trial\":{trial},\"similarity_checks\":{similarity_checks},\"exact_phase_checks\":{phase_checks},\"max_similarity_delta\":0,\"cold_similarity_baseline_ns\":{cold_similarity_old_ns},\"cold_similarity_current_ns\":{cold_similarity_new_ns},\"cold_bundle8_baseline_ns\":{cold_bundle_old_ns},\"cold_bundle8_current_ns\":{cold_bundle_new_ns}}}"
    );
    benchmark(
        "similarity",
        || {
            black_box(black_box(&old[0]).similarity(black_box(&old[1])));
        },
        || {
            black_box(black_box(&new[0]).similarity(black_box(&new[1])));
        },
        target_ns,
        repeats,
        trial,
    );
    benchmark(
        "bind",
        || {
            black_box(black_box(&old[0]).bind(black_box(&old[1])));
        },
        || {
            black_box(black_box(&new[0]).bind(black_box(&new[1])));
        },
        target_ns,
        repeats,
        trial,
    );
    benchmark(
        "unbind",
        || {
            black_box(black_box(&old[0]).unbind(black_box(&old[1])));
        },
        || {
            black_box(black_box(&new[0]).unbind(black_box(&new[1])));
        },
        target_ns,
        repeats,
        trial,
    );
    for count in [1, 2, 8, 32] {
        benchmark(
            &format!("bundle{count}"),
            || {
                black_box(baseline::PhaseHVec::bundle(black_box(&old[..count])));
            },
            || {
                black_box(current::PhaseHVec::bundle(black_box(&new[..count])));
            },
            target_ns,
            repeats,
            trial,
        );
    }
}
