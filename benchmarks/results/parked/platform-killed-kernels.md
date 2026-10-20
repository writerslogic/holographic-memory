# Parked: platform kernels killed by the microbenchmark

`platform-killed-kernels.patch` applies forward to `src/core/qgraph/kernels.rs` as shipped and
re-adds the losing candidates as test-only code, plus the extended `dot_kernel_microbench`:

- aarch64: SDOT with four accumulators, one-shift i4 SDOT, SMMLA (I8MM) two-row, explicit NEON f32 re-rank dots
- x86: 512-bit BW, `vpdpwssd` and biased `vpdpbusd` i8, ymm `vpdpwssd` i4

Evidence: `platform_aarch64_microbench_v1.json` (ratios to SDOT), `platform_x86_modal_microbench_v1.json`
(the 512-bit and `vpdpwssd` rows), `platform_aarch64_glove_v1.json` (end to end for the one-shift i4:
noise). Summary table and kill rule: `docs/performance/platforms.md`.
