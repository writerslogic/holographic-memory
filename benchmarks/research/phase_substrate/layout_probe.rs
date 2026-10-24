// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
#[path = "../../results/holography_phase_substrate_v1/baseline_phase_hvec.rs"]
mod baseline;
#[allow(dead_code)]
#[path = "../../results/holography_phase_substrate_v1/layout_phase_hvec.rs"]
mod current;
#[allow(dead_code)]
#[path = "../../../target/holography-phase-substrate/wide_aos.rs"]
mod wide_aos;
#[allow(dead_code)]
#[path = "../../../target/holography-phase-substrate/wide_soa.rs"]
mod wide_soa;

use std::hint::black_box;
use std::time::Instant;

fn elapsed(run: &mut impl FnMut(usize), variant: usize, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        run(variant);
    }
    start.elapsed().as_secs_f64() * 1e9
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "expected: N DIM SAMPLE_MS REPEATS");
    let n: u32 = args[1].parse().unwrap();
    let dim: usize = args[2].parse().unwrap();
    let target_ns: f64 = args[3].parse::<f64>().unwrap() * 1e6;
    let repeats: usize = args[4].parse().unwrap();
    let baseline: Vec<_> = (0..32)
        .map(|seed| baseline::PhaseHVec::new_random(dim, n, seed))
        .collect();
    let current: Vec<_> = (0..32)
        .map(|seed| current::PhaseHVec::new_random(dim, n, seed))
        .collect();
    let wide_aos: Vec<_> = (0..32)
        .map(|seed| wide_aos::PhaseHVec::new_random(dim, n, seed))
        .collect();
    let wide_soa: Vec<_> = (0..32)
        .map(|seed| wide_soa::PhaseHVec::new_random(dim, n, seed))
        .collect();
    for count in [2, 8, 32] {
        let expected = baseline::PhaseHVec::bundle(&baseline[..count]);
        assert_eq!(
            expected.phases(),
            current::PhaseHVec::bundle(&current[..count]).phases()
        );
        assert_eq!(
            expected.phases(),
            wide_aos::PhaseHVec::bundle(&wide_aos[..count]).phases()
        );
        assert_eq!(
            expected.phases(),
            wide_soa::PhaseHVec::bundle(&wide_soa[..count]).phases()
        );
        let mut run = |variant| match variant {
            0 => {
                black_box(baseline::PhaseHVec::bundle(black_box(&baseline[..count])));
            }
            1 => {
                black_box(current::PhaseHVec::bundle(black_box(&current[..count])));
            }
            2 => {
                black_box(wide_aos::PhaseHVec::bundle(black_box(&wide_aos[..count])));
            }
            3 => {
                black_box(wide_soa::PhaseHVec::bundle(black_box(&wide_soa[..count])));
            }
            _ => unreachable!(),
        };
        let mut iterations = 1;
        loop {
            let ns = elapsed(&mut run, 0, iterations);
            if ns >= target_ns / 4.0 {
                iterations = ((iterations as f64 * target_ns / ns) as usize).max(1);
                break;
            }
            iterations *= 4;
        }
        let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(repeats));
        for repeat in 0..repeats {
            for offset in 0..4 {
                let variant = (repeat + offset) % 4;
                samples[variant].push(elapsed(&mut run, variant, iterations) / iterations as f64);
            }
        }
        println!("{{\"n\":{n},\"dim\":{dim},\"bundle_count\":{count},\"iterations\":{iterations},\"baseline_ns\":{:?},\"i16_aos_ns\":{:?},\"i64_aos_ns\":{:?},\"i64_soa_ns\":{:?}}}", samples[0], samples[1], samples[2], samples[3]);
    }
}
