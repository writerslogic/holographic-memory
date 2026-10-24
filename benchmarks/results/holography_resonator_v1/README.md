# Resonator codebook packing and query workspace

The implementation preserves the existing resonator update, argmax tie, and
three-stable-round stopping rules. The default index uses contiguous expanded
phasors, cached initial estimates, bounded phase lookups, and a fixed number of
query allocations. `PhaseResonator::with_packed_codebooks` adds opt-in 4-, 8-, or
16-bit codebook storage with streamed lookup during dot products and projection.

This is a software memory/throughput tradeoff. It does not increase factorization
capacity, turn the floating query into integer arithmetic, or reproduce an optical
paper's physical parallelism. The default remains expanded because packed lookup
costs throughput relative to the optimized expanded implementation.

## Reproduction

Run from the repository root. This harness uses the actual library source files
and a frozen baseline, without changing Cargo manifests or adding dependencies:

```sh
mkdir -p target/resonator-validation
rustc --edition=2021 -C opt-level=3 -C lto -C codegen-units=1 \
  benchmarks/research/phase_resonator/bench.rs \
  -o target/resonator-validation/phase-resonator-bench
target/resonator-validation/phase-resonator-bench 7 \
  > benchmarks/results/holography_resonator_v1/reproduced.jsonl
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target/resonator-validation" \
  CARGO_PROFILE_TEST_OPT_LEVEL=1 cargo test --locked --test phase_resonator_regression
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target/resonator-validation" \
  CARGO_PROFILE_TEST_OPT_LEVEL=1 cargo test --locked --lib phase_resonator
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target/resonator-validation" \
  cargo test --locked --doc phase_resonator
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target/resonator-validation" \
  cargo clippy --locked --lib --test phase_resonator_regression -- -D warnings
```

`baseline.rs` is `src/core/phase_resonator.rs` from commit `12c80dec`, before
`#[cfg(test)]`, with rustfmt's trailing-blank-line normalization. Its SHA-256 is
`a427db1179fe32a2ccd85daac5aa0623a022b6304b3e5cf840bd0a5400c2a91d`.
The integration regression tests compile this baseline alongside the current
library and compare factor IDs, convergence, iteration counts, and similarity
bits. The baseline depends only on unchanged phase construction/binding behavior.

The timing harness fixes seed `0x5EED`, alternates backend order across samples,
and checks parity on all queries before timing. Each case includes one query with
every tenth coordinate replaced by a deterministic random phase. Query allocation
counters are measured separately and suspended during timed query runs. Build
times are single observations and are not robust throughput measurements.

Heap figures include allocations owned by the index: codebooks, initial estimates,
lookup tables, and vector metadata. They exclude allocator overhead, caller-owned
input codebooks, and other simultaneously live benchmark indexes. Index stack
sizes are reported separately. Inputs can be dropped after constructing an index.
These figures are allocation accounting, not process RSS measurements.

## Exploratory measurements

The final integrated run is in [serial-final.jsonl](serial-final.jsonl), with
[aggregates](serial-summary.json) and [source/compiler hashes](serial-environment.json).
It serializes this task's benchmarks, while unrelated host work may remain. The
optimized default takes 0.483/0.495 ms per query for the small 4-/8-bit cases,
versus 0.664/0.678 ms in the baseline (about 1.37× faster). Packed versions take
0.547/0.523 ms and reduce owned index heap from 591,624 bytes to 68,152/90,424 bytes
(8.68×/6.54× smaller). All four clean queries in each small case are recovered,
and every benchmark query has bit-identical baseline/new factor results.

The final large stress cases recover one of two clean queries at 4 bits and zero
of two at 8 bits; their 28.45×/15.05× heap reductions describe storage, not added
capacity. At high phase resolution, packing still increases retained heap and
slows decoding. These limits remain part of the result.

`exploratory-final.jsonl` contains seven timed samples per case on the Apple M4
configuration in `environment.txt`. Other work was running on the host, so these
latencies are exploratory and should be reproduced for deployment decisions.
They precede a separate-trigonometric-pass fix for the uncached high-resolution
conversion; the common `N=16/256` paths are unchanged by that fix.

Latency is median batch time divided by the number of queries. Every row uses
three factors. The large cases use six iterations as a bandwidth stress test and
recover zero of the two clean queries; they do not establish useful capacity at
that codebook size.

| Case (`N`, `D`, entries/axis) | Baseline ms/query | Expanded ms/query | Packed ms/query | Baseline index heap | Packed index heap |
|---|---:|---:|---:|---:|---:|
| Small (16, 1024, 12) | 0.654 | 0.482 | 0.536 | 591,624 B | 68,152 B |
| Small (256, 1024, 12) | 0.675 | 0.488 | 0.520 | 591,624 B | 90,424 B |
| Knee (16, 1024, 48) | 2.932 | 2.596 | 3.194 | 2,366,280 B | 123,448 B |
| Knee (256, 1024, 48) | 3.736 | 3.370 | 3.891 | 2,366,280 B | 201,016 B |
| Large (16, 8192, 256) | 90.034 | 78.643 | 93.075 | 100,700,232 B | 3,539,512 B |
| Large (256, 8192, 256) | 90.086 | 78.702 | 88.991 | 100,700,232 B | 6,689,080 B |
| High resolution (65536, 1024, 12) | 0.771 | 0.678 | 0.825 | 591,624 B | 1,171,768 B |

The large codebook payload is 96 MiB before packing, exceeding the reported 4 MiB
L2 cache. Total owned index heap falls by 28.5x at `N=16` and 15.1x at `N=256`.
Codebook payload alone falls by 32x/16x; the total reductions are smaller because
the index retains floating initial estimates and a shared phasor table.

Both new backends allocate 12 times per three-factor query, independent of the
iteration count, versus 63–141 allocations/query averaged over these baseline
cases. At `D=1024` they allocate 82,184 bytes/query, versus 459,136–1,049,104 bytes
in the baseline cases. This reduces allocation volume, not the numerical precision
of the working estimates. The default expanded index retains slightly more memory
than the baseline because it caches the initial estimates and phase table.

Packing is not always a memory win: at `N=65536`, the shared table alone uses 1 MiB,
so the small high-resolution packed index is larger than the expanded one. There
is no automatic switch to packing, and no public `PhaseHVec` layout change.

## Differential coverage and diagnostics

The eight regression tests include 48 within-capacity/knee queries (`D=1024`,
`F=8/24/40/48`, `N=16/256`, two seeds), including 16 corrupted queries; phase-width
and lookup cutoff boundaries; zero, one, even, and odd dimensions; odd entry
counts; one and four factors; `max_iter=0..5`; and exact argmax ties. Each generated
case compares the expanded, packed, and one-shot entry points against the frozen
baseline. Invalid empty axes and dimension/resolution mismatches are rejected at
entry points; an empty outer collection still returns no factors.

An exact-bit test initially caught a one-ULP similarity difference at `N=65535`,
`D=1`. Paired cosine/sine evaluation on the uncached conversion path differed from
the baseline's separate passes. Restoring separate conversion passes resolved the
failure; `regression-tests.log` records all eight tests passing without relaxing
the equality check. `regression-diagnostic.log` preserves the original failing
case. This is same-platform evidence, not a cross-platform floating-point promise.

Focused checks passed on the final implementation: eight integration regression
tests, nine existing resonator unit tests, one public API doctest, and Clippy with
`-D warnings` for the library and regression target. Their output is retained in
`regression-tests.log`, `unit-tests.log`, `doctests.log`, and `clippy.log`.
Touched Rust files pass rustfmt, and the tracked patch passes `git diff --check`.
Full repository and combined-feature gates are handled at integration.

`exploratory-initial.jsonl` records the first packed implementation, which decoded
each entry into a floating scratch vector before scoring. It required extra
workspace and was slower than streamed lookup. Its source is preserved in
`packed_scratch_experiment.rs` and can be selected for reproduction by adding
`--cfg resonator_scratch` to the `rustc` command. It is not a shipped backend.
`exploratory-streamed.jsonl` records the first streaming experiment. Those first
two runs kept allocation counters active during timing; the final harness suspends
them. Cross-run timing differences include host contention and instrumentation,
so use each run's within-run baseline comparison, not a precise cross-run speedup.
