# The superposition-recovery floor is computational, not information-theoretic

**Question (posed cold).** A VSA memory superposes M key-value pairs into one vector
`s = Σ bind(k_i, v_i)`. Naive unbind + cleanup recovers reliably only up to
`M/D ≈ 0.1–0.3` before crosstalk dominates. Is that floor an information wall, or can
some decoder push blind recovery toward `M/D ≈ 1` **while keeping soft/graceful
readout** (i.e. not by switching to an exact coded store with a hard cliff)?

**Answer.** Across **seven independent decoder families explored cold**, none crossed
`M/D ≈ 0.27` for blind single-bundle recovery. But the floor is **not** an information
wall: the true solution is a stable, verifiable fixed point far past it. The barrier is
a **statistical-to-computational gap** — the solution *exists and is checkable* to at
least `M/D ≈ 0.75`, but no polynomial-time method tested can *find* it blindly past
`~0.27`. This is an honest negative that sharpens HMS's positioning, not a defeat: it
tells you exactly why capacity beyond the floor must come from sharding (side
information) or from an exact code (giving up soft recall).

This corrects a loose earlier reading. §25/§26 of `PREREGISTRATION-binding-readout.md`
reached `M ≈ D` — but only with **exact coded key-value stores** (pinv / Reed-Solomon /
XOR), which have a hard cliff and lose soft similarity. That is not "beating the
superposition floor with graceful readout"; it is choosing the coded-store corner of
the tradeoff. For *soft* superposition recovery the floor stands.

## The two verified facts

Both reproduced independently (not just taken from the exploration agents), on a
genuine capacity test: unitary keys (exact-inverse HRR — the strong Plate/Frady
baseline), a **shared** L=64 value codebook reused across the M facts (real crosstalk,
not one-symbol-per-fact). Script: `scripts/superposition_floor.py` (numpy, seeds 0..4).

**(1) The floor.** Blind recovery — naive HRR unbind, and Approximate Message Passing
(AMP, the Bayes-optimal iterative decoder) — saturates near `M/D ≈ 0.25–0.27`,
dimension-invariant (the transition sharpens with D, confirming it is a real phase
transition, not a small-D artifact):

| M/D | naive (D=512) | AMP (D=512) | AMP soft-readout |
|-----|---------------|-------------|------------------|
| 0.20 | 0.48 | 1.00 | 1.00 |
| 0.25 | 0.37 | 1.00 | 1.00 |
| 0.30 | 0.31 | 0.41 | 0.27 |
| 0.40 | 0.25 | 0.30 | 0.14 |

**(2) The gap is computational.** Initialise a hard Gauss-Seidel solver *at* the true
support (a "genie") and it stays there — and no random restart ever finds a competing
exact fit — all the way to `M/D = 1.5` (the largest load tested):

| M/D | genie-init stays | truth residual | random best-of-8 residual |
|-----|------------------|----------------|---------------------------|
| 0.50 | 1.000 | 0.0000 | 3.46 |
| 0.75 | 1.000 | 0.0000 | 3.17 |
| 1.00 | 1.000 | 0.0000 | 3.11 |
| 1.50 | 1.000 | 0.0000 | 3.04 |

So the true solution is a stable, exactly-fitting, verifiable fixed point to **≥1.5·D**,
while every blind method is trapped in spurious minima (residual ~3 vs truth's 0). In
noiseless exact arithmetic the information is plainly there — 256 real numbers carry the
~2300 bits of `M·log₂L` easily — so the barrier is **entirely computational**:
*verifiable ≠ findable.* This refutes a strict "info-theoretically impossible past
~0.5·D" reading and matches the known SPARC/CDMA hard-phase picture (a low algorithmic
threshold ≈0.27 far below the information threshold). The **practical** capacity limit
is instead set by quantization/noise — which is exactly why deterministic HMS shards.

Soft/graceful readout is preserved throughout: the AMP posterior on the true entry
degrades continuously across the transition (no cliff) — the property an exact code
would destroy.

## What was explored, and why each floored

Seven mechanisms, run cold with an explicit mandate to *test* any bound invoked rather
than assume it. Per-mechanism curves are from the exploration agents; the floor and the
computational-gap crux (rows 1–2's numbers) were independently reproduced.

| Mechanism | Reached (blind) | Verdict — the bound that actually bit (tested) |
|-----------|-----------------|-------------------------------------------------|
| Naive HRR unbind | ~0.1–0.2 | crosstalk SNR — the baseline floor |
| **AMP / compressed sensing** | **~0.25–0.27** | AMP algorithmic threshold; genie-verifiable to 0.75 (the computational gap, measured) |
| Modern-Hopfield readout | ~0.27 (no gain) | reduces to nearest-neighbour on a given codebook; softmax can't beat the unbind-SNR decision boundary |
| SIC / onion-peeling | ~0.19–0.25 | error propagation: a wrong hard subtraction poisons the residual (instrumented). Strictly dominated by AMP |
| Nonlinear / higher-order lift | ~0.12–0.14 | data-processing bound at matched storage; the matrix-memory "win" is a storage illusion (matched-reals accounting) |
| Low-coherence codebooks (ETF/Zadoff-Chu) | ~0.27 (no gain) | coherence is the *wrong* predictor — a design below the Welch bound still failed; what binds is interference Gaussianity, not coherence |
| Spatial coupling + stored hint | ~0.27 (no gain) | noiseless/interference-limited substrate: AMP potential threshold sits at the floor; band coupling makes mid-chain load ω·(M/D)>1 (seed ignites, wave dies). Hint costs ~as many bits as it saves |
| Learned / unrolled decoder | ~0.27 (no gain) | converges to the AMP fixed point; generalises to the same null, does not memorise — confirms the barrier is fundamental, not tuning |

The convergence is the result: eight decoders (including the naive baseline), one floor,
each failing for a *different, tested* reason, all consistent with a single
computational hard phase.

## Attacking the assumptions: does any stronger solver break 0.27?

The "0.27 is fundamental" claim rests on two *assumptions*, not proven theorems: that
AMP is optimal, and that the statistical-to-computational hard phase is real. The shared
codebook actually **violates** AMP's iid state-evolution premise, so AMP need not be
optimal here. Six solvers, each validated at M/D=0.10 (all 1.000), pushed past the
floor (`scripts/superposition_floor_probes.py`, D=256, 5 seeds):

| M/D | naive | AMP | AMP+refine | multistart×32 | AMP-anneal | **VAMP** |
|-----|-------|-----|------------|---------------|------------|----------|
| 0.25 | 0.41 | 0.875 | 0.86 | 0.86 | 0.89 | **1.000** |
| 0.30 | 0.31 | 0.39 | 0.29 | 0.26 | 0.39 | **0.41** |
| 0.40 | 0.23 | 0.28 | 0.17 | 0.16 | 0.28 | **0.29** |
| 0.50 | 0.21 | 0.21 | 0.12 | 0.08 | 0.21 | **0.21** |

Two findings, both honest:

- **Assumption #1 was partially false — and it didn't matter.** VAMP/OAMP (built for
  correlated designs) *is* strictly better than AMP: it holds 100% at M/D=0.25 where AMP
  has already frayed to 0.875, confirming the shared-codebook correlation was hurting
  AMP. But it only nudges the reliable knee ~0.25→0.27; at M/D=0.30 it collapses with
  everything else. The best available solver confirms the floor rather than breaking it.
- **Local search *hurts* past the floor** (AMP+refine, multistart both fall *below* plain
  AMP at M/D≥0.30). Refining a blind estimate drives it *into* a spurious minimum — a
  direct, positive signature of the trap-dominated landscape the hard-phase predicts.

So the assumption-attack strengthens the conclusion: the floor survived VAMP, annealing,
local-search refinement, and 32-restart search, and the one place AMP was genuinely
suboptimal (the correlation) buys only ~0.02 in M/D.

## What it means for HMS

- **Capacity beyond the floor is sharding, confirmed the hard way.** The only thing that
  reaches past `0.27·D` is a genie / side-information init — i.e. already knowing most of
  the answer, which is storage. This is the `capacity_is_sharding` position, now backed
  by a measured statistical-to-computational gap rather than an assumed bound.
- **The two escapes each cost something concrete.** Exact coded store → `M ≈ D` but hard
  cliff, no soft recall (§26). Side information → past the floor but at the bit-cost of
  the side information (sharding). Soft superposition recovery → floored at `~0.27·D`.
  There is no free lunch, and now we know *why* (computational hardness), not just *that*.
- **The holographic layer's value is not superposition capacity.** It is the preserved
  soft/graceful readout (continuous degradation, noise robustness) and — the HMS novelty
  — verifiability/plasticity, not a higher fact-per-dimension count.

## Honest scope

- Single clean bundle, one substrate class (unitary-key circular-convolution HRR / FHRR,
  shared Gaussian codebook), D up to 512–1024. The hard-phase is empirical here and
  consistent with SPARC/CDMA theory; it is not *proven* for this exact ensemble.
- "Computational hardness" means *no polynomial method tested* crossed it; it is not a
  proof that none exists. A genuinely novel superposition code remains open (and would be
  publishable) — but coupling, coherence design, learning, peeling, nonlinear lift, and
  Hopfield are now measured negatives, not untried ideas.
- The exploration prototypes live under `/tmp/floor/` (ephemeral); the durable,
  independently-reproduced crux is `scripts/superposition_floor.py`.

**Reproduce the crux:** `uv run python scripts/superposition_floor.py`
