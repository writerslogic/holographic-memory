# Deterministic quantized-phase resonator for VSA factorization

**What this is.** A validation, not a discovery. It pairs two published mechanisms
— the quantized-phase FHRR substrate (qFHRR) and the Frady/Kent resonator network —
and measures whether the pairing costs anything. It does not: phase quantization down
to 4 bits/dimension is *free* for resonator factorization across the operating range.
The result is an integer-only, replay-exact factorization capability on a substrate
whose stored state is a deterministic function of what was written.

This document reports numbers produced by a run *in this repository*
(`resonator-factorize`, 24 seeds). It does not restate figures from memory; every
number below carries its seed count and standard deviation.

## Claim

A resonator network run at finite phase resolution `N` (the qFHRR
bundle-recover quantization) matches the float FHRR resonator's factorization
capacity — the size of the search space `F^k` it can decode — across `F` from within
capacity to past the knee. The quantized capacity knee does **not** shift left of the
float knee. Down to `N = 16` (4-bit phase) the two curves overlap within ±1σ at every
`F` tested.

The contribution is the *combination plus this measurement*: qFHRR (arXiv 2604.25939)
explicitly does not do resonators or factorization; Frady/Kent resonator networks are
float. The open edge is whether quantization costs factorization capacity. It does
not — so a deterministic, integer-substrate resonator is a real capability, not a
degraded approximation of the float one.

## Method

`src/bin/resonator-factorize.rs`. `D = 1024`, `k = 3` factors (search space `F^3`;
two factors is trivially within capacity and measures nothing — see "Why three
factors" below).

- Each factor axis has a codebook of `F` random phase vectors. Phases are drawn in
  `Z_N` (quantized, qFHRR) or continuous (`N = 0`, the float FHRR baseline).
- A composite is the bind (phase-add) of one true entry per axis:
  `c = x_a ⊙ y_b ⊙ z_c`.
- The resonator maintains a superposition estimate per factor, initialized to the
  snapped sum of that factor's codebook. Each iteration, for each factor: unbind the
  other factors' current estimates, project (similarity-weighted) onto the factor's
  codebook, and snap each dimension back to a unit phasor quantized to `N` levels
  (the qFHRR recover step). Early-stop when the argmax per factor is stable for 3
  iterations; cap 40 iterations.
- Readout: argmax similarity per factor. Success = all three factors correct.
- Metric: factorization accuracy, mean ± std over 24 seeds × 32 trials per cell.
  **Chance = 1 / F^3.**

**Why three factors.** With two factors the task is 100% everywhere in the tested
range — an answer by construction, not a capacity measurement. Three factors is the
honest Frady/Kent benchmark: accuracy falls as `F^3` crosses the substrate's capacity.

**Cheapest disconfirming test first.** `F = 16` at `N = 256`: if the resonator did
not recover all three factors near-perfectly at this within-capacity size, the
dynamics or the quantized substrate would be broken. It scores 98±2% — the setup is
sound before sweeping the knee.

## Results

3-factor factorization accuracy, **mean ± std over 24 seeds × 32 trials**, `D = 1024`,
40 iterations. `float` = continuous phase (FHRR baseline); `N = 256` = 8-bit phase;
`N = 16` = 4-bit phase. Run in this loop, 2026-07-07.

| F (search space F³) | chance (1/F³) | float | N = 256 (8-bit) | N = 16 (4-bit) |
|---------------------|---------------|-------|-----------------|----------------|
| 16 (4 096)          | 0.024%        | 99±2  | 98±2            | 98±3           |
| 24 (13 824)         | 0.0072%       | 89±5  | 87±7            | 89±6           |
| 32 (32 768)         | 0.0031%       | 77±5  | 77±9            | 77±8           |
| 40 (64 000)         | 0.0016%       | 69±9  | 65±7            | 67±8           |
| 48 (110 592)        | 0.0009%       | 59±8  | 59±9            | 56±9           |

The capacity knee sits at `F ≈ 24–32` (search space ~14k–33k ≈ `D^1.5 = 32k` for
`D = 1024`), matching Frady/Kent theory — evidence the setup is validated, not rigged.
Float, 8-bit, and 4-bit phase overlap within ±1σ at every `F`; the largest gap
(`F = 40`, 69 vs 65) is under 1σ, and the deviations are non-monotonic in `N` — noise,
not a quantization trend. Every cell is orders of magnitude above the `1/F³` chance
floor even past the knee.

## Validation hardening (wider seeds, phase-bit and dimension sweeps)

To tighten confidence in the headline, `src/bin/resonator-sweep.rs` extends the *same*
fixed design (identical dynamics, verified below) along two controlling parameters:
phase resolution across float/8/6/4/3/2 bits (`N ∈ {0,256,64,16,8,4}`) and dimension
`D ∈ {512, 1024, 2048}`, at **30 seeds × 32 trials**. This adds no new mechanism — it
sweeps parameters of the §20 experiment to see how far "quantization is free" holds.

**Integrity check (the run reproduces §20 exactly).** The sweep first re-runs the
frozen configuration — `D = 1024`, 24 seeds, `N ∈ {0, 256, 16}` — and reproduces the
Results table above cell-for-cell (99±2 / 98±2 / 98±3 at `F = 16` … 59±8 / 59±9 / 56±9
at `F = 48`). The reimplemented dynamics are therefore faithful, so the extended
numbers below are trustworthy.

**Phase-bit robustness at `D = 1024` (30 seeds).** float / 8-bit / 6-bit / 4-bit /
3-bit / 2-bit:

| F        | float | 8-bit | 6-bit | 4-bit | 3-bit | 2-bit |
|----------|-------|-------|-------|-------|-------|-------|
| 16       | 98±3  | 98±2  | 98±3  | 98±3  | 98±3  | 95±4  |
| 24       | 89±5  | 87±7  | 89±5  | 90±6  | 87±6  | 80±7  |
| 32       | 76±6  | 78±9  | 78±7  | 78±7  | 75±8  | 72±9  |
| 40       | 69±8  | 65±8  | 66±8  | 67±8  | 63±7  | 61±9  |
| 48       | 58±8  | 58±9  | 58±8  | 56±9  | 56±9  | 51±10 |

Float through **4-bit** overlaps within ±1σ at every `F` — the §20 "4-bit is free"
claim survives the wider seed set. **3-bit** also tracks float within noise. **2-bit**
(`N = 4`) is the first resolution with a consistent deficit, largest near the knee
(`F = 24`: 80±7 vs 89±5, ≈1.5σ). So "free" holds down to ~3–4 bits/dimension and
begins to cost at 2 bits — the headline is confirmed, and its lower boundary is now
measured, not assumed.

**Dimension robustness (30 seeds).** The free-quantization pattern holds at every `D`;
absolute capacity scales with `D` as Frady/Kent predict (~`D^1.5`), so the knee moves
right with larger `D`, but the quantized columns track float within ±1σ at each `D`
(down to 3-bit). Selected float / 4-bit / 2-bit cells:

| D    | F=16 (fl/4b/2b) | F=32 (fl/4b/2b) | F=48 (fl/4b/2b) |
|------|-----------------|-----------------|-----------------|
| 512  | 89±5 / 90±7 / 81±8 | 58±8 / 60±8 / 54±8 | 35±8 / 35±7 / 37±8 |
| 1024 | 98±3 / 98±3 / 95±4 | 76±6 / 78±7 / 72±9 | 58±8 / 56±9 / 51±10 |
| 2048 | 100±1 / 100±0 / 99±2 | 93±4 / 91±6 / 83±6 | 76±8 / 75±5 / 64±12 |

Every cell above sits orders of magnitude over the `1/F³` chance floor (< 0.03%).
Reproduce with `cargo run --release --bin resonator-sweep` (std-only, deterministic;
seeds `0..24` for the repro block, `0..30` for the sweep).

## Interpretation and honest scope

Phase quantization is free for resonator factorization: a deterministic integer
resonator (down to 4 bits/dim) matches the float resonator's capacity across the range.
That makes the pairing — qFHRR substrate + Frady/Kent dynamics — an integer-only,
replay-exact neurosymbolic factorizer.

Scope this claim does **not** exceed:

- **Single clean composite.** This is the standard Frady/Kent benchmark: one bound
  product, no bundle. Factoring a fact out of a *superposition* of many stored facts
  is a separate regime where quantization noise can compound with bundle interference;
  it is not measured here (see §21 in `PREREGISTRATION-binding-readout.md`, which is
  underpowered and not shipped).
- **Float readout arithmetic.** The isolated variable is the phase resolution `N` of
  the stored/snapped state. The resonator's inner products are computed in float here;
  a fully fixed-point resonator (integer LUT arithmetic end to end) is an
  implementation step, its determinism already validated by qFHRR and the substrate
  tests. The stored substrate is integer and replay-verifiable; the query is a float
  computation over it.
- **Capacity is per-composite factorization, not superposition storage.** This does
  not change HMS's bundle-capacity story (sharding); it is about decoding a single
  bound tuple.

## Reproduction

```
cargo run --release --bin resonator-factorize
```

The bin is self-contained (std only; it reimplements the phase dynamics), so no
cargo features are needed. Deterministic: seeds `0..24` are fixed in the binary, so
the table above reproduces exactly. The library-level capability is exercised by:

```
cargo test --lib --features experimental phase_resonator
cargo test --doc --features experimental phase_resonator   # runs the PhaseResonator doctest
```

Public API: `holographic_memory::core::PhaseResonator` (reusable index over fixed
codebooks) and `holographic_memory::core::phase_resonator_factorize` (one-shot),
over the `holographic_memory::core::PhaseHVec` substrate.

## References

- **qFHRR** — quantized-phase Fourier Holographic Reduced Representations. arXiv
  2604.25939 (Apr 2026). The discrete-phase substrate: integer bind/unbind/similarity/
  bundle via modular arithmetic and LUTs, ~lossless vs complex FHRR at K ≥ 16. Provides
  the deterministic substrate; does not address resonators or factorization.
- **Frady, Kent, Olshausen, Sommer (2020)** — "Resonator Networks, 1: An Efficient
  Solution for Factoring High-Dimensional, Distributed Representations of Data
  Structures." *Neural Computation* 32(12):2311–2331 (companion part 2 by Kent et al.,
  same issue). The resonator dynamics (superposition estimates, alternating unbind +
  cleanup) and the `~D^1.5` capacity scaling this run reproduces. These networks are
  float. See `docs/PREREGISTRATION-binding-readout.md` §20 for the full citation chain
  as the project maintains it.

The citable entry is the intersection these two lines leave open: the SOTA
factorization method run on the deterministic quantized substrate, measured to cost
nothing in capacity.
