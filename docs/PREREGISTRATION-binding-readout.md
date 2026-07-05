# 024 — Pre-registration: non-self-inverse binding + nonlinear readout

**Status:** proposed (pre-registration only — no implementation authorized yet).
**Author:** (pending review)
**Relates to:** `core/entangled.rs` (`bind`, `similarity`), `core/resonator.rs`,
`core/block_codes.rs`, `docs/RESEARCH.md`, `docs/COMPARISON.md`.

This document pre-registers an experiment **before any code is written**, per the
project's research discipline. It states an open question, strong baselines, a
metric, a strong-outcome threshold, and a kill-condition that can fire on a
plausible negative. Nothing here asserts a result. The experiment must be run
and the curve reported before any claim enters a summary, changelog, or paper.

## 1. Background and the open question

HMS's live algebra is:

- **Bind** (`entangled.rs:bind`): symmetric difference (XOR) of sparse active-index
  sets. This operator is **self-inverse**: `a ⊕ a = 0`, and `bind` is its own
  unbind.
- **Readout** (`entangled.rs:similarity`): Jaccard overlap of active-index sets —
  a linear set-membership statistic on the (additively bundled) superposition.

Two known limitations from the standing notes and `RESEARCH.md`:

1. Self-inverse XOR is the weak VSA binding. The field has non-self-inverse
   bindings (HRR circular convolution, MAP element-wise product with a distinct
   unbind) that carry strictly more algebraic structure. `RESEARCH.md` records
   "BSC+XOR is insufficient → use HRR, MAP" as settled prior art, not an open
   claim.
2. Linear readout on an additive bundle reads first-order statistics (which
   symbols are present), not binding structure (how they are composed). Recovering
   structure requires a nonlinearity *before* the linear comparison.

**Open question (what this experiment tests):** On *compositional* retrieval —
distinguishing correctly-bound role–filler structures from same-symbol
mis-bindings — does replacing (self-inverse XOR bind, Jaccard readout) with
(a non-self-inverse bind, a nonlinear readout) yield a measurable capacity /
discrimination improvement at matched dimensionality `D` and matched density,
*after* accounting for its cost?

This is the open edge. That HRR/MAP beat XOR on generic binding is **settled and
will be labelled validation, not discovery.** The unproven part is whether, in
*this* sparse-index substrate and at matched `D`/density/cost, the swap pays off
for compositional discrimination — and by how much, against which specific
baseline, at what latency cost.

## 2. Task (the property under test)

Capacity-at-interference for bound structure:

- Build `N` facts, each a bound set of `R` role–filler pairs drawn from a
  **shared** codebook of `S` symbols (shared so that interference is real;
  unique-symbol-per-fact is banned — it has an answer by construction and
  measures nothing).
- Bundle the `N` facts into one superposition.
- Query: given a probe role (and optionally a partial structure), decide whether
  a specific `(role, filler)` pair is bound in a target fact — against
  distractors that contain the **same symbols in a different binding**
  (mis-binding distractors, not random-symbol distractors).

The discriminator is the mis-binding case: XOR+Jaccard is expected to be weakest
exactly where the same symbols appear but the composition differs. If the new
operators cannot separate correct-binding from mis-binding better than the
baseline, the contribution is empty.

## 3. Systems compared (baselines are strong incumbents)

| Arm | Bind | Readout | Role |
|-----|------|---------|------|
| B0  | XOR (self-inverse), sparse | Jaccard | incumbent / control (current HMS) |
| B1  | **HRR** (circular convolution), dense | matched-filter (circular correlation) | strong incumbent baseline |
| B2  | **MAP** (element-wise product), bipolar | cosine | strong incumbent baseline |
| X   | non-self-inverse candidate on the sparse substrate | nonlinear readout (e.g. k-WTA / entmax before overlap) | the proposed change |

B1 and B2 are the incumbents the field would compare against; they are not
strawmen and must be implemented faithfully (correct unbind, correct
normalization). X must be held to **matched `D`, matched density/active-count,
matched query budget** — no free parameters that B0/B1/B2 do not also get.

## 4. Metric

- Primary: **d′** (sensitivity) and **ROC / AUC** for correct-binding vs
  mis-binding, not bare mean similarity.
- Report full **distributions** and effect size, not point means.
- **Sweep the controlling parameter** `N` (load) across the capacity knee; mark
  the **chance floor** explicitly.
- **≥ 20 seeds** near the knee; report variance. Seed everything (codebook draw,
  fact composition, query order).
- Report **cost** alongside benefit on the same axis: wall-clock per query and
  per bundle, and memory, for each arm. A latency multiplier is part of the
  result, not a footnote.

## 5. Strong-outcome condition (what would make X worth adopting)

At matched `D`, density, and query budget, arm X shows a **d′ improvement over
B0 that also holds against the better of B1/B2** (i.e. X is not merely recovering
what HRR/MAP already give), sustained across the swept load `N` near the knee,
with the cost multiplier over B0 stated. Concretely, pre-registered threshold:
**Δd′ ≥ 0.5 over B0 and ≥ 0.2 over max(B1,B2) at the 50%-recall load, across ≥20
seeds with non-overlapping confidence intervals.** (Thresholds are the
commitment; they are falsifiable and set before any run.)

## 6. Kill-condition (must be able to fire on a plausible negative)

Abort and record a negative result if **any** of:

- X's d′ does not exceed B0 by the pre-registered margin at the knee (the swap
  buys nothing on this substrate) — **or** —
- X only matches B1/B2 (then the honest finding is "use HRR/MAP", not "adopt X"),
  which `RESEARCH.md` already knows — **or** —
- the cost multiplier is so large that at iso-latency (give B0 more `D`/seeds for
  the same wall-clock) B0 closes the d′ gap.

A flat or negative result here is **publishable and will be reported as such**
("non-self-inverse binding gives no compositional advantage over XOR on the
sparse-index substrate at matched cost, despite richer algebra") — it is not a
reason to keep tuning X until it wins.

## 7. Run order (cheapest disconfirming test first)

1. Smallest version: `S` small, `R = 2`, single seed, `N` swept coarsely — just
   B0 vs X on the mis-binding discriminator. If the kill-condition fires here,
   **pivot; do not build the full harness.**
2. Only if step 1 survives: add B1/B2, full seed count, fine `N` sweep, cost
   instrumentation.

## 8. Implementation constraints (when/if authorized)

- **Additive, no regression.** The existing XOR/Jaccard path stays the default
  and untouched; X lands as an opt-in behind an `experimental` module + feature,
  never by mutating `bind`/`similarity`. (Consistent with the "no regression to
  holographic" constraint.)
- Experiment/benchmark code is **not** auto-generated to pass. Tests assert the
  property in their title (shared symbols, real interference) and include the
  disconfirming variant.
- No result, table, or number is written before its run. Hypotheses stay
  questions with the kill-condition above.

## 9. What is explicitly NOT claimed here

- That X wins. (Unknown until section 5/6 resolves against the curve.)
- That HRR/MAP beating XOR is a contribution. (Settled prior art; validation
  only.)
- Any capacity number. (None exists until the sweep is run and seeded.)

## 10. Step-1 result (run 2026-07-04)

Harness: `src/bin/binding-discriminator.rs`. D=16384, density 1/256, shared
codebook of 2048 symbols, R=2 roles, 8 seeds. Each binding uses its NATIVE stack
(the fair comparison — on this sparse substrate the binding and readout are
coupled):
- B0 (incumbent): XOR bind, majority-vote bundle, query = XOR-unbind(role) then
  Jaccard vs candidate filler.
- X (candidate): permutation bind (`hash_permute(role)`), bloom-union bundle,
  query = density-corrected containment.

d' (correct-binding vs mis-binding), mean over 8 seeds:

| N   | B0 XOR d' | X perm d' | X/B0 |
|-----|-----------|-----------|------|
| 40  | 0.86      | 15.2      | 18x  |
| 80  | 0.70      | 6.1       | 8.8x |
| 160 | 0.60      | 3.7       | 6.1x |
| 320 | 0.46      | 2.3       | 4.9x |
| 640 | 0.34      | 1.19      | 3.5x |

**The step-1 kill-condition did NOT fire.** X clears B0 by 3.5x-18x across the
load sweep, with both at their native readout.

Caveats carried forward (this is a promising signal, not a validated finding):
- **Sentinel-inflated low N.** X d' at N=5,10,20 (115, 45, 29) sit in the
  near-zero-variance regime where the d' estimator is unstable; only N>=40 is
  trustworthy. The conclusion rests on the N>=40 rows.
- **Weak incumbent, not the strong baseline.** Beating XOR only confirms the
  settled "BSC is weak on sparse" prior art. The contribution question — does X
  beat HRR/MAP? — is untested. Not citable on this result alone.
- **Density confound.** X's bloom stack is density-preserving; B0's XOR doubles
  active-count. X's advantage is not yet separated from its density edge.
- **Stack-vs-stack, not binding-in-isolation.** Binding and readout are coupled
  on this substrate, so this compares whole stacks, not the bind operator alone.
- **First readout was a strawman, then fixed.** An initial run gave both bindings
  a containment readout and reported B0 d'=0.00; that under-served XOR (its native
  query is unbind, not containment). The table above uses XOR's native stack.

Consequence per the pre-registration: step 1 survives, so proceed to step 2 —
add HRR and MAP baselines with their proper unbind readouts, a density-matched
XOR control, more seeds near the knee (N in 40..160), and ROC/AUC alongside d'.
Step 2 requires implementing HRR (dense circular convolution) and MAP (bipolar)
off the sparse index-set substrate — a design choice to be surfaced before
building, per the experiment-code rule.
