<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Coupled superposition: positive recovery, expensive decoder

This bounded audit reproduces an accuracy advantage for the repository's existing
spatially coupled Gaussian encoder. It does not introduce a production memory
backend or claim a general capacity theorem.

The [frozen plan](../../research/holographic_capacity/PLAN.json) fixes `D=1248`,
`M=352`, alphabet 64, 300 existing block-AMP iterations, and seeds 101/211/307.
All methods share the labels and signal dimension. The coupled design uses 32
column blocks and width 8; the independently addressed control uses width 1.
Noise is the same Gaussian draw within each seed, scaled by each signal's standard
deviation. The three seeds were not used to select these parameters.

| Encoder | Clean symbol accuracy, mean (range) | 10% relative AWGN accuracy, mean (range) |
|---|---:|---:|
| Dense Gaussian | 67.90% (50.57–100%) | 49.62% (44.89–55.11%) |
| Spatially coupled | 100% (100–100%) | 100% (100–100%) |
| Independent shards | 50.38% (45.74–53.69%) | 48.96% (45.17–51.99%) |

These are observations over 1,056 symbols per cell, clustered in only three
generated problems. They are not independent-sample confidence intervals or an
end-to-end retrieval evaluation. Changing the encoder does change this particular
decoder's behavior; the experiment does not prove an impossibility result for
other decoders or parameter choices.

## Full state cost

Every implementation currently allocates a dense `float64` design matrix with
`D * M * 64 = 28,114,944` entries: **224,919,552 bytes**. The signal itself is only
9,984 bytes. Including array metadata used by the decoder gives persistent array
payloads of 224,942,344 bytes (dense), 224,952,320 (coupled), and 224,950,528
(sharded), before Python objects, allocator overhead, or AMP temporaries. The
returned posterior alone adds 180,224 bytes. This is payload accounting, not RSS.

Only 46,137,344 bytes of the coupled matrix and 7,028,736 bytes of the sharded
matrix are nonzero. A block-structured operator could omit the zeros, but that is
an unimplemented optimization and these smaller numbers are not the measured
allocations. Regenerating a matrix from a seed trades retained state for work;
that tradeoff was not measured here.

The task stores 352 labels from a known 64-symbol alphabet: 2,112 bits (264 bytes)
if represented directly with known slot order. That representation does not
provide the tested noisy soft readout, but it illustrates why symbol recovery
alone is insufficient evidence of a smaller useful database. Codebook generation,
query semantics, updates, and full memory costs must be addressed before promotion.

The old dropout helper reduces `d` after slicing rows without rescaling the
retained matrix; its original normalization therefore deserves a separate audit.
This run uses clean and additive-noise cases only. Historical dropout numbers are
not revalidated by this result.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
uv run --no-project --python 3.14.7 --with numpy==2.5.4 python \
  benchmarks/research/holographic_capacity/probe.py --output ./capacity-reproduction
```

[Raw cases](cases.jsonl), [aggregates](summary.json), and
[source/environment hashes](environment.json) retain all observations. The probe
uses the existing encoder and decoder in `scripts/superposition_floor_coded.py`.
Timing fields were collected during concurrent work and support no speed claim.
