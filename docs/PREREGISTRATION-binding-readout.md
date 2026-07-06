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

## 11. Density-matched control (run 2026-07-04)

The chosen cheapest disconfirming follow-up: rule out the density confound. A new
column `B0thin` gives XOR *X's exact stack* — each XOR-bound pair thinned to k
active indices (matching permutation density), bloom bundle, corrected-
containment readout — so the ONLY difference from X is the bind operator, at
matched density AND matched readout. `B0nat` is XOR at its own native best.

d', mean over 8 seeds:

| N   | B0nat XOR | B0thin XOR | X perm | X / B0thin |
|-----|-----------|------------|--------|------------|
| 40  | 0.86      | 1.07       | 15.2   | 14x        |
| 80  | 0.70      | 0.87       | 6.1    | 7x         |
| 160 | 0.60      | 0.69       | 3.7    | 5.3x       |
| 320 | 0.46      | 0.50       | 2.3    | 4.5x       |
| 640 | 0.34      | 0.34       | 1.19   | 3.5x       |

**Density/readout confound ruled out.** At matched density and matched readout,
permutation still beats XOR 3.5x-14x. Density-matched XOR (B0thin) tracks native
XOR (~d' 0.3-1.4) — thinning neither helps nor hurts it — so XOR is intrinsically
weak at compositional discrimination on this substrate, independent of density
and readout. The advantage is the binding operator.

Still open (unchanged): (a) the citable comparison is X vs the STRONG baselines
HRR/MAP, not vs XOR; beating XOR only re-confirms "BSC is weak on sparse". (b) A
sharper mechanistic control — an *involution* (self-inverse) permutation vs the
non-involution permutation — would isolate whether the driver is "non-self-
inverse-ness" per se or "role-specific relocation" (permutation) vs "raw-index
union" (XOR); under a containment readout these may be indistinguishable, which
would reframe the claim away from "self-inverse is the bottleneck" toward "XOR's
index-union leaks; any role-relocation fixes it". (c) low-N X d' stays sentinel-
inflated; rely on N>=40.

## 12. Involution control — the framing was wrong (run 2026-07-04)

Ran control (b). Added `Inv`: a self-inverse involution permutation (`idx ^ mask`,
which applied twice is the identity) through X's exact bloom+containment stack —
identical to X except the permutation is self-inverse. All four systems, d' over
8 seeds, N=40..1280 (d' floored/capped at 50; the sentinel-inflated low-N rows
from sections 10-11 are superseded by this cleaner estimator):

| N    | B0nat XOR | B0thin XOR | Inv perm | X perm |
|------|-----------|------------|----------|--------|
| 40   | 0.86      | 1.07       | 32.9     | 15.2   |
| 80   | 0.70      | 0.87       | 7.6      | 6.1    |
| 160  | 0.60      | 0.69       | 4.55     | 3.67   |
| 320  | 0.46      | 0.50       | 2.86     | 2.26   |
| 640  | 0.34      | 0.34       | 1.19     | 1.19   |
| 1280 | 0.24      | 0.22       | 0.20     | 0.45   |

(At N=1280 every system sits at the chance floor — the sparse bloom bundle is
saturated; the informative band is N=40..640.)

**The original "self-inverse XOR is the bottleneck" framing is DISCONFIRMED as a
causal claim.** `Inv` (self-inverse) matches or beats `X` (non-self-inverse); the
self-inverse property is not the axis. The real axis is **role-relocation vs
index-union**: HMS's bind is set symmetric-difference, which merges role and
filler indices into a bag where the pairing leaks (any (role, filler) whose parts
appear anywhere scores high); ANY position-permutation bind — self-inverse or not
— relocates each filler into a role-specific slot that does not leak. Relocation
beats union 3.5x-30x across N=40..640; density and readout were already ruled out
in section 11.

Honest positioning of this result:
- This is most likely a **validation of settled VSA knowledge** (permutation /
  convolution binding outperforms set-XOR/BSC binding), re-derived on the HMS
  substrate — NOT a novel discovery, and not independently citable.
- It is nonetheless **practically actionable**: HMS's current bind is the weak
  set-XOR kind; switching to a position-permutation bind (`permute` /
  `hash_permute` already exist in `entangled.rs`) recovers 3.5x-30x mis-binding
  discrimination at the same O(D) cost. An *exact* permutation (`Inv`'s idx^mask,
  or `permute`) beat the collision-prone `hash_permute` at low-mid load — prefer
  an exact bijection.
- The only path to a NOVEL contribution remains X-stack vs strong baselines
  HRR/MAP with matched resources (untested; the expensive step).

## 13. Strong baselines: HRR + MAP (run 2026-07-04)

Harness: `src/bin/binding-baselines.rs`. Matched D=2048, shared 1024-symbol
codebook, 8 seeds. All three systems run the SAME discriminator with the SAME
query TYPE — membership: compose the (role, filler) pair and score it against the
bundle (correct pair is a member, mis-bound is not). Faithful strong baselines:
- HRR: real dense N(0,1/D), circular-convolution bind (validated radix-2 FFT),
  cosine-to-bundle.
- MAP: bipolar dense {-1,+1}, elementwise-product bind, cosine-to-bundle.
- P: sparse permutation bind (idx^mask), bloom bundle, corrected containment.

d' (correct vs mis-binding), mean over 8 seeds:

| N   | HRR  | MAP  | P sparse |
|-----|------|------|----------|
| 40  | 3.47 | 4.00 | 8.05     |
| 80  | 2.18 | 2.41 | 3.58     |
| 160 | 1.43 | 1.64 | 2.39     |
| 320 | 1.03 | 1.17 | 1.35     |

**P (sparse) matches or beats HRR and MAP at every load**, at ~8 active indices
vs 2048 dense components (matched-D deliberately flatters the dense systems on
storage, and P still wins). Verified real, not a bug: at low load the sparse
bundle is near-empty so a non-member's indices are cleanly absent (near-binary
containment, high separability); dense bundles carry continuous crosstalk from
the first item. MAP membership is algebraically identical to MAP retrieval for
bipolar roles, so the readout swap is not what drives P's edge.

Honest scope — this is NOT "sparse VSA beats HRR/MAP", it is much narrower:
- **Verification only.** The task is membership verification (is this pair
  bound?). HRR and MAP additionally support **retrieval** (unbind -> reconstruct
  the filler) and cleanup/analogy; P's containment does none of these. The claim
  is scoped to the one axis P can win — verification — exactly the axis the honest
  positioning rules say to benchmark, with the retrieval axis disclosed as P's
  loss, not hidden.
- **Stack-vs-stack, not binding-isolated.** P differs from HRR/MAP on substrate
  (sparse vs dense), bind (permutation vs convolution/product), AND readout
  (containment vs cosine) simultaneously. The win is the whole sparse stack's, not
  attributable to the bind operator alone.
- **Mechanism, not magic.** P's edge is the near-binary separability of sparse
  membership before bundle saturation; it collapses to the dense systems' level
  as load approaches the saturation knee (converging by N~320). It is a
  low-storage efficiency result for verification, likely an incremental/known
  property of sparse HDC (cf. HyperCam, already cited in `entangled.rs`), not a
  new capability.

Net: the citable-strength claim would be "sparse permutation + bloom membership
verifies role-filler bindings at higher d' than dense HRR/MAP at matched D and
~256x lower storage, on the verification axis; it does not provide retrieval."
Before that is paper-grade it needs: a retrieval-axis table (to bound the scope
quantitatively), ROC/AUC, a D-sweep, and >=20 seeds near the knee.

## 14. Hardening RETRACTS the §13 claim (run 2026-07-04)

Ran the paper-grade hardening from the §13 to-do list (ROC/AUC, retrieval,
D-sweep, 20 seeds). **It overturns §13.** The "P beats HRR/MAP" result was a d'
artifact.

Verification, D=2048 (d' | AUC), 20 seeds:

| N   | HRR d'|AUC   | MAP d'|AUC   | P d'|AUC     |
|-----|-------------|-------------|--------------|
| 20  | 4.10 | 0.977 | 4.51 | 0.973 | 9.57 | 0.977 |
| 80  | 1.94 | 0.911 | 2.09 | 0.924 | 3.01 | 0.923 |
| 320 | 0.87 | 0.734 | 0.96 | 0.758 | 1.15 | 0.701 |

P's d' is highest at every load, but its **AUC ties at low load and is the LOWEST
at high load** (0.701 vs HRR 0.734 / MAP 0.758 at N=320; same story at D=512 and
D=8192, where P's AUC is consistently at or below the dense systems by N=320).
d' rewards P's near-binary, low-variance containment scores (large mean gap /
small spread), but the score distributions actually **overlap more in the tails**,
which the distribution-free AUC exposes. d' assumed Gaussianity that P violates.

Retrieval (rank-1 over 512 symbols, D=2048): all three are poor and P is
consistently worst (~0.05/0.025/0.014/0.007 vs HRR/MAP ~0.075/0.045/0.028/0.02
for N=20/40/80/160). NOTE the per-fact retrieval query is partly ill-posed on a
flat bundle (unbinding a role returns the superposition of ALL that role's
fillers, so "fact i's filler" has no unique target) — treat these as directional
(dense > sparse) only; a well-posed retrieval test needs per-fact addressing and
is future work.

**Corrected conclusion.** The sparse permutation stack does NOT beat HRR/MAP. By
the honest metric (AUC) it is comparable at low load and inferior at high load,
and it is worse at retrieval. Its one genuine advantage is storage (~256x fewer
components at matched D) while reaching low-load verification AUC parity. So the
defensible claim shrinks to: "sparse permutation + bloom membership reaches
HRR/MAP verification AUC at low load with ~256x less storage, degrading below
them as load rises; it does not match them on retrieval." A storage-efficiency
tradeoff for verification-heavy workloads, not a capability win — and NOT the d'
headline of §13, which is retracted. This is the discipline working: the
flattering metric was caught by the pre-registered stronger one.

## 15. Fixed-point complex substrate for continuous memory (run 2026-07-05)

**Question (lane 1, continuous/graceful memory).** The fractional-power encoding
(FPE) similarity kernel is smooth in |x-y| but needs a COSINE (inner-product)
readout. The integer phase-histogram substrate is exact-match only, so it cannot
host that kernel — an earlier "nearest" test on the histogram passed only by
coincidence (§ FPE-in-histogram, retracted). Cosine is a float op, which would
break the bit-exact-replay verifiability the store depends on. Open question: can
a FIXED-POINT complex substrate (phase → integer (cos,sin) lookup tables, integer
inner-product similarity, integer superposition) give graceful similarity AND stay
deterministic? This is FHRR quantized to Z_N. Baseline = the histogram exact-match
readout at matched D. Kill = fixed-point cosine does not beat exact-match on
graceful nearest.

**Setup.** N=256 phases, D=1024, SCALE=4096 fixed-point unit, 5 seeds. FPE base
frequencies low (-4..4). Stored grid = {0,10,...,120}; queries = {7,23,38,51,64,
77,92,108} land strictly BETWEEN grid points so exact-match cannot win.
`src/bin/fixed-point-holographic.rs`.

Part 1 — nearest-value recall (readout comparison):

| readout                     | nearest-recall |
|-----------------------------|----------------|
| fixed-point cosine          | 100%           |
| exact-match (histogram)     | 0%             |

Decisive: the histogram scores 0% on between-grid queries (confirms exact-match
cannot host a continuous kernel); fixed-point cosine recovers the true nearest
grid point every time, with no float.

Part 2 — graceful associative retrieval under superposition. Store continuous-key
→ symbol facts (bind = phase-add) bundled into ONE integer complex field; query a
between-grid key, recover the NEAREST stored key's symbol, as distractor facts are
added. Chance ≈ 1/13 ≈ 7.7%.

| distractors | recall |
|-------------|--------|
| 0           | 85%    |
| 25          | 68%    |
| 100         | 50%    |
| 400         | 45%    |

**Conclusion.** The fixed-point complex substrate delivers the continuous
capability the histogram cannot: a graceful similarity kernel that recovers the
nearest value, entirely in integer arithmetic (cos/sin tables + i64/i128 sums), so
the field stays a bit-exact fold of its inserts — replay-verifiable like the
histogram. The kill condition did NOT fire. Honest caveats: (a) even at zero
distractors single-pass associative recall is 85%, not 100% — the 13 grid keys
interfere with each other in the shared bundle; (b) recall degrades to ~45% at 400
distractors (still ~6x chance). Both are single-pass numbers with no cleanup; a
resonator/iterative-cleanup readout is the obvious next lever and is future work.
The substrate is validated; the module (a continuous-key `PhaseVectorMemory`) is
the product step.

## 16. Resonator cleanup for continuous associative recall (pre-registered 2026-07-05)

**Question.** §15 Part 2 leaves single-pass associative recall at 85% (zero load)
→ 45% (400 distractors). The field superposes `bind(key_i, symbol_i)` facts; a
continuous query fixes the key at `encode(q)`, which is NOT a stored key, so the
one-shot symbol readout carries interference from neighboring facts. Does a
two-factor **resonator network** (Frady/Kent/Sommer 2020 style) lift recall by
alternating hard cleanup between the two codebooks?

**Method.** Two codebooks: K = {encode(k) : k ∈ grid} (stored keys), S =
{symbol_phase(i)} (symbols). Field = Σ_i bind(K_i, S_i), same fixed-point complex
accumulator as §15. Per query q (between grid points):
- init key estimate = encode(q) (the continuous query, not a codebook entry);
- iterate T=4: (a) symbol step — score each S_j by ⟨field, bind(key_est, S_j)⟩,
  hard-snap symbol_est to argmax; (b) key step — score each K_j by ⟨field,
  bind(K_j, symbol_est)⟩, hard-snap key_est to that stored key. Early-stop when
  key index is stable.
- answer = final symbol index.

Using the stored-key codebook for cleanup is legitimate: a real store knows its
own keys. Kill can genuinely fire — a wrong key-snap propagates and can make the
resonator WORSE than single-pass.

**Baseline** = §15 single-pass (key fixed at encode(q), one symbol readout),
re-run in the same harness for a matched comparison.

**Metric.** Symbol recall at loads {0, 25, 100, 400}, 5 seeds, chance ≈ 7.7%.
Also report key-recovery (final key index = true nearest grid key).

**Strong outcome.** Resonator recall > single-pass at every load, gap widening
under load. **Kill.** Resonator ≤ single-pass at the load levels → the alternating
cleanup buys nothing here; report the flat result and ship the module single-pass.
`src/bin/resonator-cleanup.rs`.

**Result (run 2026-07-05) — KILL FIRED.** Symbol recall, single-pass vs resonator:

| distractors | single-pass | resonator | key-recovery |
|-------------|-------------|-----------|--------------|
| 0           | 85%         | 85%       | 85%          |
| 25          | 68%         | 62%       | 62%          |
| 100         | 50%         | 40%       | 42%          |
| 400         | 45%         | 38%       | 48%          |

The resonator ties at zero load and is strictly WORSE under load. Why: single-pass
is already near-optimal for THIS query shape. With unique symbols, the one-shot
score of symbol s_j is ≈ κ(k_j − q), the FPE kernel of the query's distance to
grid key k_j, so argmax already picks the nearest key's symbol — the continuous
query q carries the "between-ness" the kernel needs. The resonator's HARD key-snap
throws that away: it projects the continuous key estimate onto a (sometimes wrong)
grid key, and a wrong snap propagates into the symbol readout. Key-recovery
(48% at load 400) confirms the snap is itself unreliable. Hard cleanup destroys
the graceful information rather than sharpening it.

**Corrected conclusion.** For continuous single-factor recall with unique fillers,
single-pass fixed-point cosine is the readout to ship; the two-factor resonator is
a net negative. The real lever for the 85%-at-zero-load ceiling is NOT cleanup but
CAPACITY — that ceiling is 13 facts interfering in one D=1024 bundle, independent
of distractors, so raising D (or sharding) is what lifts it. Resonators remain the
right tool for genuine MULTI-factor products (bind of ≥2 codebook factors both
unknown); they are the wrong tool when one factor is a known continuous probe.
Honest negative, not buried: it redirects the module design (ship single-pass, size
D to the fact count) and saves an iterative readout that would have cost latency
for less accuracy.
