<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase-superposition routing: two rejected synthetic screens

Neither frozen screen qualifies for a separate ANN development experiment. This rejects these routing rules and budgets on these synthetic distributions; it does not rule out holographic representations generally. No production behavior, official dataset, heldout ANN query, or durable format was changed or accessed.

The first screen retains all 12 phase configurations, exact-kernel controls, equal-memory reservoirs, and random order. The second asks whether a phase field adds value to geometry-selected witnesses. It does not pass its combined-memory gate: witness-only routing explains every structured-case gain, while the hybrid loses to the equal-budget reservoir on mixed blocks. Geometric page organization, separate addressing, and capacity budgeting remain plausible directions; these runs do not establish an improvement from them.

## Mechanism and limits

For each block, retain `H_b = Σ_x exp(i Ωx/σ)` with independent standard-normal columns of Ω, then read `Re[φ(q) · conjugate(H_b)] / D`. This is the paired sine/cosine random Fourier feature construction: its expectation is `Σ_x exp(-||q-x||²/(2σ²))`, a kernel-density sum, not a nearest-neighbor maximum. The code reuses the mathematical pattern already present in [HLI](../../research/hli/hli.py); this is a routing application, not a new encoding. [Rahimi and Recht, 2007](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf) gives the kernel construction; [Plate, 1995](https://pages.ucsd.edu/~msereno/170/readings/06-Holographic.pdf) discusses superposition and noisy associative retrieval.

Exact RBF scores isolate finite-feature interference from the density objective. They are the infinite-feature target, not an upper bound on every finite-sketch routing outcome. A small correctness counterexample shows that ten farther neighbors can outrank one exact neighbor even without sketch error. Population normalization changes the objective and also loses on some cases below.

Multiplying both a query and a block field by the same unit-modulus reference phase cancels in their conjugate product. Independent per-block reference rotations therefore add no selectivity when both operands are rotated consistently; this identity is tested. Distinct frequency bands change kernels and cost more state/work. No untested multiband mixture, zero-hop retrieval claim, or physical optical capacity claim follows from these results.

## Frozen phase screen

[PLAN.json](../../research/holographic_router/PLAN.json) and [its pre-run seal](pre-run-seal.json) fix 24 datasets: four scenarios, nominal populations 256/2,048, and seeds 17/29/43. Eight blocks alternate full and quarter populations, totaling 1,280/10,240 float64 vectors in 32 dimensions; each case has 32 queries. Phase widths 64/256, bandwidths 0.5/1.5/4, sum/mean readouts, and projection seeds 101/211/307 produce 864 phase readouts. The 1,128 total method records include all controls.

Coherent blocks have one Gaussian lobe; multimodal blocks have opposite lobes that cancel their centroid; mixed blocks share one distribution; rare blocks contain true neighbors competing with denser, farther clouds in other blocks. Routing coverage is the fraction of oracle top-10 IDs retained after selecting blocks; it is not end-to-end ANN recall. The frozen rule requires at least 99.75% coverage at fanout 4/8 in every coherent/multimodal/rare case and projection, plus a 10-point mean gain over centroid on multimodal cases. **Zero of 12 configurations qualify.**

All values below are mean coverage percentages at fanout 4/8. Each phase cell averages six datasets and three projection seeds. Every control cell averages the same six datasets. These are descriptive means, not confidence intervals. [aggregate.json](aggregate.json) retains all fanouts, minima/maxima, and every negative mixed/rare comparison; [raw cases](run-synthetic-v1/cases.jsonl) retain every ordering, top-10 ID, and mutation cost.

| Method | coherent | multimodal | mixed | rare |
|---|---:|---:|---:|---:|
| Phase 64, σ=0.5, sum | 61.46 | 55.21 | 50.00 | 83.85 |
| Phase 64, σ=0.5, mean | 61.98 | 56.60 | 50.36 | 86.46 |
| Phase 64, σ=1.5, sum | 100.00 | 100.00 | 50.42 | 72.74 |
| Phase 64, σ=1.5, mean | 100.00 | 100.00 | 49.50 | 73.44 |
| Phase 64, σ=4, sum | 100.00 | 100.00 | 79.90 | 71.35 |
| Phase 64, σ=4, mean | 100.00 | 100.00 | 54.05 | 72.92 |
| Phase 256, σ=0.5, sum | 67.71 | 61.98 | 50.14 | 93.92 |
| Phase 256, σ=0.5, mean | 68.75 | 60.76 | 50.26 | 94.44 |
| Phase 256, σ=1.5, sum | 100.00 | 100.00 | 49.91 | 81.94 |
| Phase 256, σ=1.5, mean | 100.00 | 100.00 | 49.10 | 81.25 |
| Phase 256, σ=4, sum | 100.00 | 100.00 | 79.90 | 85.24 |
| Phase 256, σ=4, mean | 100.00 | 100.00 | 54.55 | 83.16 |

| Method | coherent | multimodal | mixed | rare |
|---|---:|---:|---:|---:|
| Centroid | 100.00 | 49.48 | 56.72 | 75.00 |
| Random order | 40.62 | 40.62 | 48.85 | 40.62 |
| Reservoir 12 | 100.00 | 100.00 | 53.54 | 65.10 |
| Reservoir 48 | 100.00 | 100.00 | 56.46 | 84.38 |
| Exact RBF σ=0.5, sum | 100.00 | 100.00 | 84.69 | 100.00 |
| Exact RBF σ=0.5, mean | 100.00 | 100.00 | 83.70 | 100.00 |
| Exact RBF σ=1.5, sum | 100.00 | 100.00 | 80.83 | 100.00 |
| Exact RBF σ=1.5, mean | 100.00 | 100.00 | 76.77 | 100.00 |
| Exact RBF σ=4, sum | 100.00 | 100.00 | 79.90 | 100.00 |
| Exact RBF σ=4, mean | 100.00 | 100.00 | 57.45 | 100.00 |
| Oracle hit-count ranking | 100.00 | 100.00 | 87.81 | 100.00 |

The bandwidth that preserves coherent/multimodal blocks can miss rare true neighbors. Narrower bandwidth improves rare coverage but loses coherent/multimodal selectivity at the frozen widths. Broad sum scores favor large mixed blocks; their high mixed coverage is retained without attributing it to rare-neighbor protection. The exact-kernel controls and each failed finite sketch remain in the assessment.

## Frozen witness followup

[hybrid_PLAN.json](../../research/holographic_router/hybrid_PLAN.json) and [its seal](hybrid-pre-run-seal.json) fix 30 fresh datasets with seeds 59/71/83 and projections 401/503/601. The phase component is width 64, bandwidth 1.5, sum readout. Witness selection uses base rows only: start nearest the centroid, then repeatedly choose the row farthest from its nearest selected witness, breaking ties by row ID. The hybrid uses the first 36 of the same 48-witness sequence. At fanout f, reserve ceil(f/2) witness-ranked blocks and fill with phase-ranked blocks.

The additional interior-rare scenario surrounds true neighbors with antithetic shells, so they are not merely farthest from the centroid. Their measured centroid-distance percentiles span 10.94–71.88%. There is no query-trained witness selection or post-run grid expansion.

| Method | coherent | multimodal | mixed | rare | interior_rare |
|---|---:|---:|---:|---:|---:|
| Centroid | 100.00 | 49.48 | 57.03 | 75.00 | 100.00 |
| Geometry 36 | 100.00 | 100.00 | 50.68 | 100.00 | 100.00 |
| Geometry 48 | 100.00 | 100.00 | 49.22 | 100.00 | 100.00 |
| Reservoir 48 | 100.00 | 100.00 | 59.84 | 81.77 | 100.00 |
| Phase 64 | 100.00 | 100.00 | 50.99 | 74.48 | 93.23 |
| Hybrid | 100.00 | 100.00 | 54.24 | 100.00 | 100.00 |
| Oracle hit-count ranking | 100.00 | 100.00 | 87.34 | 100.00 | 100.00 |

All structured cases reach 100% at fanout 4/8 with both the hybrid and geometry-36 alone. Geometry-36 costs 73,728 bytes versus the hybrid's 98,368 bytes; phase gets no credit for that witness-only result. In mixed cases the hybrid averages 54.24%, versus 59.84% for the equal-budget reservoir, with a worst individual loss of 13.75 percentage points. The frozen hybrid rule requires no loss over 0.25 points against the better equal-budget control and at least one scenario gain of 2 points over both controls; it fails both requirements. Geometry-only also loses on mixed blocks. [Raw cases](run-hybrid-v1/cases.jsonl) preserve those failures.

## Memory, work, and interference

These are NumPy array payload bytes, including shared Ω and block counts where used, not allocator overhead or an RSS claim. Witness IDs used during selection are not needed for block scoring. Every budget is for all eight blocks together. The smallest dataset has 327,680 raw-vector bytes, so the 24,640-byte phase component does not trigger the preregistered storage stop.

| Router | Stored bytes | Dot-product multiply-accumulates/query | Sin/cos evaluations/query |
|---|---:|---:|---:|
| Centroid | 2,048 | 256 | 0 |
| Phase 64 | 24,640 | 3,072 | 128 |
| Reservoir 12, phase-screen accounting | 24,640 | 3,072 | 0 |
| Phase 256 | 98,368 | 12,288 | 512 |
| Reservoir 48, phase-screen accounting | 98,368 | 12,288 | 0 |
| Geometry 36 | 73,728 | 9,216 | 0 |
| Geometry/reservoir 48, hybrid accounting | 98,304 | 12,288 | 0 |
| Hybrid | 98,368 | 12,288 | 128 |

The first screen charges reservoirs an additional eight int64 counts; the fixed-size followup omits them. Both controls fit their respective total budgets. Counts above describe dominant dot products, excluding norm calculations, reductions, ranking, and Python overhead. One phase insertion/deletion needs 32D projection multiply-accumulates and 2D trig evaluations; replacement doubles that work. Witness selection takes repeated full-block distance scans; dynamic witness maintenance was not implemented or costed.

With independent uniform background phases, a normalized readout has noise variance n/(2D), and one matching member contributes signal 1. Thus the **theoretical** signal-to-noise ratio is sqrt(2D/n). This assumes independent phases and is not a measured law for correlated ANN vectors. A separate frozen 64-trial uniform-phase simulation reports:

| Background rows | Phase width | Observed noise SD | IID theory SD |
|---:|---:|---:|---:|
| 64 | 64 | 0.710 | 0.707 |
| 64 | 256 | 0.400 | 0.354 |
| 256 | 64 | 1.619 | 1.414 |
| 256 | 256 | 0.734 | 0.707 |
| 1024 | 64 | 2.916 | 2.828 |
| 1024 | 256 | 1.387 | 1.414 |
| 4096 | 64 | 5.575 | 5.657 |
| 4096 | 256 | 2.980 | 2.828 |

All noise samples are in [the phase summary](run-synthetic-v1/summary.json). Only 64–4,096 background rows were simulated; no production-scale capacity was measured. Deletion/replacement/reuse was compared with full re-summation at 10,240 live rows: the largest accumulator discrepancy was 9.10e-13 and query-score discrepancy 2.28e-13, with exact counts. This is floating-point roundoff, not bitwise reversibility or evidence of indefinite update stability. [audit.json](audit.json) records each state.

## Reproduction and verification

[environment.json](environment.json) records CPython 3.14.7, NumPy 2.5.4, uv 0.12.10, macOS 27.0 build 26A428, Apple Accelerate 1.11/vecLib 3.11, and requested thread limits of two. NumPy reports its BLAS version as unknown; framework versions and the OS build are the available exact identifiers. Effective BLAS thread counts and load averages were not independently recorded. Both runs occurred during concurrent local work: their raw timing fields are exploratory and cannot support throughput, latency, or retention claims.

From the repository root, use new empty output directories for runs:

```sh
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
uv run --no-project --python 3.14.7 --with numpy==2.5.4 python -m unittest discover -s benchmarks/research/holographic_router -p 'test_*.py'
uv run --no-project --python 3.14.7 --with numpy==2.5.4 python benchmarks/research/holographic_router/probe.py --output ./routing-phase-reproduction
uv run --no-project --python 3.14.7 --with numpy==2.5.4 python benchmarks/research/holographic_router/hybrid_probe.py --output ./routing-hybrid-reproduction
uv run --no-project --python 3.14.7 --with numpy==2.5.4 python benchmarks/research/holographic_router/audit.py benchmarks/results/holographic_router_synthetic_v1/run-synthetic-v1
uv run --no-project --python 3.14.7 --with numpy==2.5.4 python benchmarks/research/holographic_router/audit_hybrid.py benchmarks/results/holographic_router_synthetic_v1/run-hybrid-v1
uv run --no-project python benchmarks/research/holographic_router/summarize.py benchmarks/results/holographic_router_synthetic_v1
uv run --no-project python benchmarks/research/holographic_router/check_evidence.py benchmarks/results/holographic_router_synthetic_v1
shasum -a 256 -c benchmarks/results/holographic_router_synthetic_v1/SHA256SUMS
```

The first audit regenerates source/query hashes and independently recomputes exact top-10 IDs with coordinatewise distances, all routing coverages, and the frozen decisions. The hybrid audit additionally reselects every witness independently, recomputes scores with complex exponentials and coordinatewise distances, reconstructs every selected page, and checks combined bytes. All 1,458 method records match. [validation.json](validation.json) records correctness tests and the required scoped Rust gate; hashes cover plans, sources, and results. Reproductions on different NumPy/BLAS/platform combinations may differ in floating-point ties or RNG-generated bytes; the committed outputs remain the sealed observations, not a promise of cross-platform byte identity.
