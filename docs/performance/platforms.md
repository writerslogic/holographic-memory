# Distance kernels by platform

Scope: the integer and float distance kernels of `core::qgraph` (`src/core/qgraph/kernels.rs`):
8-bit and 4-bit code dot products (`dot_i8`, `dot_i4`), the float re-rank dots (`dot_f32_i8`,
`dot_f32_i4`) and the 1-bit screen popcount. Every kernel is checked bit for bit against the
scalar reference (`every_integer_kernel_matches_scalar`, `every_float_kernel_matches_scalar`:
lengths 16 to 512, extreme values, all-minimum nibbles). Kill rule: a kernel ships only if it beats the
incumbent in `dot_kernel_microbench`; a loser stays in the tree as test-only code so the kill
test reproduces, and is listed here.

Run: `cargo test --release --lib dot_kernel_microbench -- --ignored --nocapture`.
Raw data: `benchmarks/results/platform_aarch64_microbench_v1.json` (M4, 6 runs, min and median),
`platform_x86_modal_microbench_v1.json` (12 Modal containers, 3 with AVX512-VNNI),
`platform_x86_modal_e2e_v1.json`, `platform_aarch64_glove_v1.json`.

## Result table (ns per call, minimum over runs; ratio < 1 is faster than the incumbent)

| Arch | Kernel (dim) | Incumbent | Candidate | cold ratio | hot ratio | Decision |
|---|---|---|---|---|---|---|
| aarch64 M4 | i8 (128, 256) | SDOT 2 acc | SDOT 4 acc | 0.85, 0.79 | 0.87, 0.92 | killed: cold ratios were 0.99 and 1.08 to 1.0 in two earlier batches, so the gain is run noise; memory bound |
| aarch64 M4 | i4 (128, 256) | SDOT | SDOT, one shift | 1.18, 1.23 | 1.24, 1.16 | killed (earlier batch: 0.98, 1.2) |
| aarch64 M4 | i8 / i4 | SDOT | SMMLA (I8MM) 2 rows | n/a | 1.50, 2.63 | killed: the query fills both tile rows, half of each tile is wasted |
| aarch64 M4 | f32 re-rank i8 / i4 | autovectorized | explicit NEON | 1.03 to 1.35 | 0.90 to 1.31 | killed (LLVM already vectorizes the 16-lane form) |
| aarch64 M4 | i8 | SDOT | widening NEON (no dotprod) | 0.99, 1.09 | 1.37, 2.23 | fallback for cores without dotprod, kept |
| aarch64 M4 | i4 | scalar | widening NEON (no dotprod) | 0.07, 0.12 | 0.04, 0.06 | shipped: new fallback, replaces a scalar loop that was 20 to 40x slower |
| x86 AVX512-VNNI | i8 (128, 256) | AVX2 | `vpdpbusd` ymm, biased | 0.90, 0.76 | 0.87, 0.69 | shipped |
| x86 AVX512-VNNI | i4 (128, 256) | AVX2 | `vpdpbusd` xmm, biased | 0.70, 0.78 | 0.71, 0.83 | shipped |
| x86 AVX512-VNNI | i8 | `vpdpbusd` ymm | 512-bit BW, `vpdpwssd`, `vpdpbusd` | 0.91 to 1.02 | 0.95 to 1.28 | killed: no consistent gain over ymm, licence risk |
| x86 AVX512-VNNI | i4 | `vpdpbusd` | `vpdpwssd` ymm | 1.27, 1.21 | 1.20, 1.13 | killed |
| x86 AVX2-only hosts | all | scalar | AVX2 (already shipped) | 0.23, 0.41 (i8) | 0.18, 0.15 (i8) | unchanged |

Apple M4: the existing SDOT kernels were already at the memory-bound floor. Nothing on aarch64
beat them, so the aarch64 hot path is unchanged. Dispatch there is compile-time SDOT where the
target enables `dotprod` (all Apple cores) and runtime detection otherwise.

Float re-rank on x86 and aarch64: left autovectorized (no explicit kernel won).

## End to end (glove-100-angular, matched recall 0.90 and 0.95)

- aarch64: the shipped code equals the base binary. The one end-to-end test run was the killed
  one-shift i4 variant against base (9 alternating rounds, 1 and 4 concurrent processes):
  ratios 0.95 and 1.04 (1 process), 1.20 and 1.26 (4), under load 43 to 120 from other jobs, with round
  ratios from 0.71 to 2.3. That is noise, not a result. Recall and exact evaluations per query
  were identical (0.9005 / 5810.9 at ef 256, 0.9518 / 14882.1 at ef 768).
- x86 AVX512-VNNI (Modal, one 8 vCPU container, 7 alternating rounds, same binary with
  `HMS_QGRAPH_KERNEL=avx2` against `vnni`): ef 256 / 736 at 1 thread 0.998 and 1.036, at 4 threads
  1.013 and 0.975 (median of round ratios, min 0.65, max 1.50). The microbenchmark gain of
  10 to 30 percent per call does not show in QPS at this noise; the search is bound by row fetch
  latency, not the dot product. Recall and exact evaluations per query were identical for both
  kernels at every ef (0.9005 / 5810.9, 0.95065 / 14341.3), as the bit-exactness implies.
  A 21-round rerun found no VNNI container in 24 attempts (`platform_x86_modal_e2e_v2.raw.json`).
- The AVX512-VNNI kernels are therefore a per-call win (microbenchmark) with no measured QPS
  change. They stay because they cost nothing and cannot change results.

## Recall gate

`cargo test --release --lib qgraph` (26 tests including the recall tests) passes on the M4, and on
the AVX512 container once per tier (`HMS_QGRAPH_KERNEL=scalar`, `avx2`, `vnni`), all 26 passing
(see `qgraph_tests_by_tier` in the e2e file).

## Knob

x86: `HMS_QGRAPH_KERNEL=scalar|avx2|vnni|auto` (read once, never above what the CPU supports).
Default `auto`.

## Not obtained

No non-dotprod ARM core was available, so the widening NEON i4
fallback is measured only as a function on the M4 (not end to end). WASM SIMD128 belongs to
another agent.
