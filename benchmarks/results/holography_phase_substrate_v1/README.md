# Phase-vector substrate evidence

This isolates `PhaseHVec` kernels against the implementation at
`12c80decb094d57ee13ba187a01bf1de8788ca2f`; the complete original module is
[`baseline_phase_hvec.rs`](baseline_phase_hvec.rs). Public phase storage remains
`Vec<u16>` and `phases() -> &[u16]`. This is a kernel optimization, with no claim of
better retrieval accuracy or packed storage for public phase vectors.

The production changes remove division from modular binding/unbinding, reuse
exact cosine values for power-of-two resolutions, reuse fixed-point phasors, and
avoid phasor reconstruction for singleton and binary-phase bundles. Cosine lookup
uses the absolute, unwrapped phase difference and preserves summation order.
The original fixed-point scale, integer accumulation, and phase rounding are
unchanged for general bundles. Binary bundling counts phase-one votes and emits
one only for a strict majority; a tie matches `atan2(0, 0)` and emits zero.

Above resolution 256, phasor table entries are adjacent `i16` components, each
within ±16,384, instead of separate `i64` arrays: four bytes per phase instead of
sixteen. They widen exactly into the original `i64` accumulators. Resolutions
at or below 256 use adjacent `i64` components because the layout diagnostic
measured better throughput for these small tables. Persistent lookup tables exist only
for the sixteen power-of-two resolutions 2..65,536. Their total possible payload
is 1,578,928 bytes (excluding headers and allocator overhead), and they are
initialized lazily. Unlike the original temporary tables, these remain allocated.
The binary bundle path never initializes its phasor table. Compared with using
`i16` at every resolution, widening the remaining small cached tables costs only
6,096 bytes; avoiding the now-unused binary phasor table saves eight bytes.
Non-power-of-two phasor tables remain temporary. Large cosine tables are built
only when dimension is at least resolution; a shorter comparison uses the
original direct calculation until such a table already exists.

## Reproduction

From the repository root:

```sh
uv run --no-project python benchmarks/research/phase_substrate/run.py \
  --label serial-validation \
  --output benchmarks/results/holography_phase_substrate_v1/serial.json

uv run --no-project python benchmarks/research/phase_substrate/layouts.py \
  --label serial-validation
```

Both runners use standalone Rust with the repository's release optimization
settings: optimization level 3, LTO, and one codegen unit. They write generated
executables and layout variants beneath the local `target/` directory. They need
no added dependencies or Cargo manifest changes. Full source hashes, compiler
settings, and raw samples are retained in the JSON artifacts.

The main probe covers resolutions 2, 16, 256, 257, and 65,536 at dimensions 64,
512, 4,096, and 16,384, plus resolution/dimension 65,536. Each shape runs in three
fresh processes. Each process captures the first similarity and eight-vector
bundle call before cache initialization, verifies outputs, then runs five
alternating warm measurements per operation. Calibration targets 10 ms per
measurement. Operations are similarity, bind, unbind, and bundles of 1, 2, 8,
and 32 vectors. Fixtures are deterministic and allocated outside timing. Warm
timings include output allocation and destruction; cold timings include output
allocation but retain the result until after timing. The cold measurements have
only three samples and are particularly noisy.

The layout probe compares the original baseline, the earlier `i16` adjacent
components, widened adjacent `i64` components, and separate `i64` arrays. The
three cached variants share validation, iteration, allocation of accumulators,
and phase recovery. It checks exact phase equality and rotates measurement order
over nine repeats. It covers resolutions 16/256, dimensions 4,096/16,384/65,536,
and bundles of 2/8/32. It uses the archived
[`layout_phase_hvec.rs`](layout_phase_hvec.rs), whose hash matches the measured
diagnostic source, to reproduce the comparison that motivated the final hybrid
layout. The main probe always builds the final production source.

“Serial” means serialized within this task. Unrelated external Rust workloads
were observed on the host; exclusive machine access was not available.

## Verification

The final accumulator correction and its 16-test validation are recorded in
[`accumulator_regression.md`](accumulator_regression.md). It resolves the
process-dependent dimension-512 regression found during the first integrated
sweep and preserves that failed/intermediate sweep with its exact source.

[`validation.txt`](validation.txt): 15 focused phase-vector tests passed. The tests cover modular
wrap boundaries, every cosine phase difference at representative and boundary
resolutions, empty dimensions, invalid resolution bounds, fixed-point bundle
equivalence, every singleton phase through resolution 65,536, and every binary
vote split for odd/even bundle sizes around 2, 8, 16, 32, and 256.
The same artifact records `cargo clippy --locked --lib -- -D warnings` passing.

The worktree-local Cargo invocations set `CARGO_BUILD_JOBS=2` and an absolute
`CARGO_TARGET_DIR` under this worktree. The touched Rust files also pass rustfmt
checks, and `git diff --check` is clean.

## Initial measurements

[`exploratory.json`](exploratory.json) was collected on Apple M4, macOS 27,
Rust 1.96.0 while other agents were active. It predates the binary-majority fast
path, the low-resolution wide tables, and two additional tests. Its exact source
is archived in [`exploratory_phase_hvec.rs`](exploratory_phase_hvec.rs) and matches
the recorded source hash. It checked 64,575 similarity outputs bit-for-bit and 34,846,464
phase outputs exactly, with zero differences. These numbers describe synthetic
kernel comparisons, not end-to-end retrieval benchmarks.

Selected warm medians, in microseconds:

| Resolution | Dimension | Operation | Baseline | Optimized | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 16 | 4,096 | similarity | 10.065 | 2.660 | 3.78× |
| 16 | 4,096 | bind | 2.143 | 0.256 | 8.37× |
| 16 | 4,096 | unbind | 2.148 | 0.318 | 6.76× |
| 256 | 64 | bundle, 8 vectors | 2.143 | 0.711 | 3.01× |
| 65,536 | 4,096 | bundle, 8 vectors | 458.490 | 63.274 | 7.25× |

Cache initialization changes first-call costs. For resolution 256/dimension 64,
the first similarity median was 0.709 → 1.417 µs; repeated calls amortize that
setup. For resolution 65,536/dimension 4,096, the first eight-vector bundle was
504.209 → 426.250 µs. Non-power-of-two similarity intentionally retains its
original calculation. Large low-resolution bundle timings were mixed in the
concurrent run, motivating the separate layout diagnostic rather than assuming
all bundle workloads improve.

## Layout diagnosis and final selection

[`layouts.json`](layouts.json) captures the task-serialized diagnostic. At
resolution 16/dimension 16,384 with eight vectors, the baseline and compact-table
medians were 374.900 and 382.180 µs (0.98×), compared with 0.86× during the
concurrent run. This suggests the original 14% regression was substantially
timing noise. The widened adjacent table measured 352.390 µs, 8% faster than
the compact table, while separate `i64` arrays measured 389.580 µs. Across most
resolution-16 cases, widened adjacent components were 5–10% faster; the separate
arrays were consistently slower. These are measured layout effects. Attributing
the difference specifically to integer-widening instructions is an inference,
not a hardware-counter measurement.

The final implementation therefore uses adjacent `i64` components at resolutions
up to 256 and compact adjacent `i16` components above 256. It retains the
substantial large-table space saving with approximately 6 KiB of additional
small-table payload. The 15 focused tests and clippy result refer to this final
selection; final performance should be assessed with the main probe against the
current production source.

## Final integrated sweep

[`serial.json`](serial.json) tests the final interleaved accumulator across all
21 shapes and three fresh processes: 64,575 bitwise similarity comparisons and
34,846,464 exact phase-coordinate comparisons pass. At resolution16/dimension4096,
warm similarity is 3.78× faster, binding 8.79×, and unbinding 6.47×. Eight-vector
bundling improves 1.37× there, 12.50× at binary resolution, and 9.71× at resolution
65536/dimension4096. The corrected resolution16/dimension512 bundle32 median is
7.99µs versus13.91µs in the baseline.

The non-power-of-two resolution257/dimension16384 eight-vector bundle measured
0.911× baseline speed in this sweep. The bounded seven-process
[threshold control](threshold_diagnosis.json) did not show a reproducible benefit
from widening the small-table cutoff to512 (246.9 versus253.6µs, with broad
overlapping variability). The cutoff remains256; the observed loss is retained.
Cold lookup setup also has a cost: resolution256/dimension64 similarity takes
1.334µs initially versus0.875µs in the baseline, while warm calls are4.65× faster.
These are isolated kernels on a shared host, not end-to-end search speedups.
