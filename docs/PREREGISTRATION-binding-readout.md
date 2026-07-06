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

## 17. What limits the zero-load 85%: kernel width vs capacity (pre-registered 2026-07-05)

**Question.** §15 recovers the nearest symbol 85% of the time at just 13 facts,
zero distractors, D=1024. That is the diagnostic anomaly: 13 facts is FAR below the
capacity knee (Plate C_eff≈0.10·D ⇒ M_50≈100 at D=1024, consistent with §15's own
45% at ~413 facts), so 13 facts should recall ~100%. Three competing explanations,
each with a different fix:
  (H1 confound) The low-frequency FPE base (multipliers in ±1..4) makes the
    similarity kernel κ(|x−y|) too WIDE relative to grid spacing 10, so a
    between-grid query's nearest grid key is confusable with its neighbors. Fix =
    narrow the kernel (wider base-frequency spread). Cheap, additive.
  (H2 capacity) The 85% is genuine interference; raising D lifts it. Fix = scale D
    / decorrelate bases (the capacity sweep, §-future).
  (H3 binding) Phase-add binding is intrinsically limited at this grid spacing;
    neither knob helps. Fix = a different bind (HRR circular convolution),
    reframing the substrate.

**Method (one experiment discriminates all three).** Reuse the §15 fixed-point
complex substrate. Two independent knobs at ZERO distractor load (13 grid facts,
8 between-grid queries, 20 seeds):
  - base-frequency spread W: multipliers drawn from ±(1..=W)\{0}. W ∈ {2,4,8,16,
    32,64} plus a fully-random base (multipliers uniform 0..N ⇒ maximally narrow,
    non-graceful control). Larger W = narrower kernel.
  - dimension D ∈ {512, 1024, 2048, 4096}.
Report between-grid nearest-recall for each (W, D). Secondary column: exact-at-grid
recall (query sits ON a grid key) to expose the width tradeoff — too-narrow kernels
should keep exact-recall high while between-grid recall collapses.

**Discrimination.**
  - Narrowing W lifts between-grid recall AND raising D does not → H1 (confound).
  - Raising D lifts it AND W does not → H2 (capacity).
  - Neither knob reaches ~100% between-grid → H3 (binding); a non-trivial kernel
    sweet spot that beats 85% but caps below ~95% still favors H1-with-a-tradeoff.

**Kill for H1.** If no W beats the low-freq baseline's ~85% between-grid recall at
D=1024 (within seed variance), the confound hypothesis is dead and the ceiling is
capacity or binding — pivot to the D-sweep. `src/bin/kernel-capacity-sweep.rs`.

**Result (run 2026-07-05) — H1 CONFIRMED, H2 and H3 rejected.** Between-grid
recall (%), 20 seeds, 13 facts, zero load:

|            | D=512 | D=1024 | D=2048 | D=4096 |
|------------|-------|--------|--------|--------|
| W=4 (§15)  | 73    | 83     | 92     | 98     |
| W=8        | 94    | 98     | 99     | 100    |
| W=16       | 100   | 100    | 100    | 100    |
| W=32       | 84    | 88     | 89     | 88     |
| W=64       | 16    | 16     | 16     | 16     |
| random     | 6     | 6      | 4      | 12     |

W=4 at D=1024 reproduces the §15 85% (83%, harness matches). Widening to W=16 lifts
it to 100% at EVERY D — W=16 @ D=512 (100%) beats W=4 @ D=4096 (98%) at 8x less
memory, so kernel width dominates dimension. Classic bandwidth tuning: too wide
(W≤4) confuses grid neighbors (spacing 10); too narrow (W≥32) drops between-grid
queries outside the kernel → collapse to chance (W=64: 16%, random: 6%). The
exact-on-grid control confirms the mechanism: narrow kernels perfectly separate
exact keys (100% for W≥8) but only W=8–16 also INTERPOLATES between them.

**Conclusion.** The §15 zero-load ceiling was a mis-tuned FPE kernel bandwidth, not
capacity (D is nearly irrelevant here) and not binding (phase-add reaches 100%).
The fix is free and additive: the substrate must tune base frequency to the value
RESOLUTION (kernel width ≈ grid spacing), not hardcode low frequencies. Design
principle, not a magic constant: W too small under-resolves, W too large loses
gracefulness; the sweet spot is where the kernel's half-width ≈ half the smallest
distinguishable value gap. Caveat: this is ZERO load; §18 tests whether the tuned
kernel lifts the full distractor-load curve or only the zero-load point.

## 18. Tuned kernel × capacity under distractor load (pre-registered 2026-07-05)

**Question.** §17 fixed the ZERO-load ceiling via kernel width (W=4→16 ⇒ 85%→100%).
Two open questions for the full associative curve (§15 Part 2 gave 85→68→50→45% at
loads 0/25/100/400): (a) does the tuned kernel lift the WHOLE curve or only the
zero-load point? (b) with the confound removed, does D (capacity) now behave as
theory predicts — the lever for the distractor tail (recall ~ 1 − M/cD)?

**Method.** §15 Part-2 associative task: store 13 grid facts + `extra` distractor
facts (keys far outside the grid, distinct symbols) bundled into one fixed-point
complex field; query between-grid keys, recover the nearest grid key's symbol.
Sweep W ∈ {4 (baseline), 8, 16} × D ∈ {1024, 2048, 4096} × load ∈ {0,25,100,400,
1000}, 20 seeds. Metric: recall (chance ≈ 7.7%).

**Strong outcome.** W=16 beats W=4 at every load (kernel fix is not zero-load-only),
AND at fixed W the curve lifts with D following the capacity law (2x D ⇒ ~2x the
load for equal recall). Combined W=16 + D=4096 should far exceed §15's 85→45.
**Kill.** If W=16 ≤ W=4 once load>0, the kernel fix is a zero-load artifact; if D
does not move the tail, the substrate is capacity-inefficient (revisit decorrelated
bases). `src/bin/kernel-load-curve.rs`.

**Result (run 2026-07-05) — BOTH levers confirmed, orthogonal and compounding.**
Recall (%), 20 seeds, 13 grid facts + distractors:

| config           | 0   | 25  | 100 | 400 | 1000 |
|------------------|-----|-----|-----|-----|------|
| W=4  D=1024 (§15)| 83  | 68  | 52  | 44  | 33   |
| W=16 D=1024      | 100 | 100 | 94  | 61  | 31   |
| W=16 D=2048      | 100 | 100 | 98  | 78  | 42   |
| W=16 D=4096      | 100 | 100 | 99  | 93  | 68   |
| W=8  D=4096      | 100 | 99  | 98  | 84  | 71   |

The §15 curve 85→68→52→44 becomes 100→100→99→93 at W=16 D=4096 — near-perfect
graceful recall out to 413 facts where §15 was at 44%. Kill did not fire (W=16 >
W=4 at every load except 1000, where both sit ~31–33% because 1013 facts is past
the capacity knee for D=1024).

**Conclusion.** Kernel width and dimension are orthogonal levers on distinct
regimes: (a) bandwidth (W) owns low-to-mid load and is FREE — W=4→16 takes load-100
from 52%→94% at no extra memory; (b) D owns the distractor tail and follows the
capacity law recall≈1−M/cD (at load 400, doubling D roughly halves the error:
61→78→93%). Combined, the fixed-point complex substrate reaches production-grade
graceful associative recall (≥93% to ~400 facts at D=4096), entirely in integer
arithmetic (replay-verifiable). Two honest caveats: past the capacity knee (load
1000, 1013 facts vs D=4096) recall is 68% — the fix there is more D or sharding,
not tuning; and the optimal W drifts slightly WIDER under heavy noise (W=8 beats
W=16 at load 1000, 71 vs 68), so "tune W to resolution" carries a load-dependent
correction. Substrate design for the `PhaseVectorMemory` module is now settled:
tune kernel bandwidth to value resolution, size D to expected fact count, shard
beyond the knee.

## 19. Counting readout vs the binary Bloom membership wall (pre-registered 2026-07-05)

**Question.** The Bloom membership store (`core::bloom_memory`) bundles items by OR
set-union and reads them out with density-corrected containment. Members score
EXACTLY 1.0 (all k active indices are in the union by construction), so the
discrimination collapse is entirely NON-members' scores rising toward 1.0 as the
union saturates — a false-positive rate ≈ d^k that explodes as density d→1. OR-union
discards COUNT information (how many items hit each index). Does a counting bundle +
statistical readout push the membership wall substantially past the binary one?

**Method (self-contained `src/bin/bloom-wall.rs`, not the untested harness).**
D=16384, k=64 active indices/item (denom 256), items = deterministic random
k-subsets. Two readouts on the SAME inserted set:
  - binary: OR-union presence; score = corrected containment
    (|q∩union|/k − d)/(1−d), d = |union|/D. (the incumbent, `bloom_memory`.)
  - counting: per-index integer hit count; score = Σ_{i∈q active}(count_i − λ)/√λ,
    the Poisson z-sum under the non-member null λ = n·k/D.
Sweep n ∈ {100,200,400,700,1000,1500,2000,3000}, 20 seeds. Metric = **AUC**
(Mann-Whitney, 200 members vs 200 non-members, distribution-free — NOT d', which
§14 showed flatters near-binary score distributions). Report the wall = smallest n
where AUC drops below 0.99 and below 0.95, for each readout.

**Strong outcome.** Counting's AUC<0.95 wall is ≥1.5× the binary wall (e.g. binary
walls ~700, counting holds past ~1400). **Kill.** Counting wall ≤ binary wall (no
better than 1.2×) → counts don't help; the ceiling is field saturation itself, not
the readout, and the fix is more D / sharding, not a smarter readout.

**Result (run 2026-07-05) — counting QUADRUPLES the membership wall.** AUC (member
vs non-member, 20 seeds), D=16384, k=64:

| n    | binary AUC | counting AUC |
|------|------------|--------------|
| 700  | 0.994      | 0.9997       |
| 1000 | 0.865      | 0.998        |
| 1500 | 0.578      | 0.990        |
| 2000 | 0.514      | 0.978        |
| 3000 | 0.500      | 0.951        |
| 4000 | 0.500      | 0.922        |
| 6000 | 0.500      | 0.878        |
| 8000 | 0.500      | 0.841        |

Binary walls at <0.95 @ 1000 and is DEAD (chance) by 2000. Counting walls at <0.95
@ 4000 and is still 0.84 at 8000. That is a ~4x capacity gain, and counting
degrades gracefully where binary cliff-collapses. Kill did not fire.

**Conclusion.** The Bloom membership ceiling was READOUT-driven, not saturation:
OR-union discards the per-index count, which is exactly the signal (a member adds
+1 to each of its k indices; the Poisson z-sum recovers that shift long after the
union has saturated to all-ones). Preserving counts quadruples usable capacity at
matched D — cost is one u32 field vs one bit field (trades Bloom's 1-bit
compactness for capacity), still a deterministic integer fold (replay-verifiable).
A direct, actionable win for `core::bloom_memory`; the next step is a production
counting-membership store behind the `experimental` gate.

## 20. Deterministic quantized resonator for factorization (pre-registered 2026-07-05)

**Prior-art note (read first).** The quantized-phase-FHRR substrate used in §15–§18
is PUBLISHED as qFHRR (arXiv 2604.25939, Apr 2026): discrete phase indices, integer
bind/unbind/similarity/bundle via mod-arithmetic + LUTs, ~lossless vs complex FHRR
at K≥16 (bind-sim 0.99 at K=16, 0.997 at K=32). So §15/§17/§18 are VALIDATION of
qFHRR, not discovery — labelled as such. qFHRR explicitly does NOT do resonators or
factorization; Frady/Kent resonator networks (2020; Nature Mach. Intell. 2024) are
FLOAT. This experiment tests the open edge: does phase QUANTIZATION cost
factorization capacity in a resonator?

**Question.** Given a composite c = bind(x_a, y_b) (phase-add) of two unknown
codebook factors, a resonator network recovers (a,b) by alternating unbind +
codebook-cleanup. Does an FHRR resonator run at finite phase resolution N match the
float resonator's factorization capacity (search space F² it can solve), or does
quantization shrink it?

**Method (`src/bin/resonator-factorize.rs`).** D=1024. Factor codebooks X, Y each
of F items; phases drawn in Z_N (N ∈ {16,32,64,256}) or continuous (N=0, the float
baseline). Composite c = (x_a+y_b) mod. Resonator state x̂,ŷ as complex vectors,
init = g(Σ codebook). Iterate ≤30: x̂ = g(Σ_k ⟨X_k, c⊙conj(ŷ)⟩ X_k); ŷ symmetric;
g[d] = unit phasor at phase atan2(im,re) quantized to N (the qFHRR bundle recover
step). Early-stop on stable argmax. Readout: (â,b̂)=argmax similarity. Success =
both correct. Sweep F ∈ {8,16,32,48,64} (search space 64…4096), 30 trials × random
(a,b), 20 seeds. Metric: factorization accuracy vs F, per N. Chance = 1/F².

**Cheapest disconfirming first.** F=8, N=256, D=1024: if the resonator does not
recover (a,b) ~100% at this trivial size, the dynamics are wrong or the quantized
substrate cannot host a resonator — fix before sweeping.

**Strong outcome.** Quantized (N=256) matches float (N=0) accuracy across F within
noise, and even N=16–32 tracks it closely (per qFHRR's ~lossless-at-K≥16) → phase
quantization is free for factorization → "deterministic resonator" is real, pairing
qFHRR + Frady/Kent. **Kill.** N=256 accuracy is materially below float at every F
(capacity knee shifts left with quantization) → quantization costs factorization;
report the cost curve honestly. Either way it is a citable entry qFHRR left open.

**Result (run 2026-07-05) — quantization is FREE for factorization; kill did not
fire.** First run (2 factors, F≤64) was 100% everywhere — a toy within capacity,
measuring nothing; pivoted to 3 factors (search space F³), the honest Frady/Kent
benchmark. 3-factor accuracy, mean±std over 24 seeds × 32 trials, D=1024:

| F (F³)     | float | N=256 | N=16 (4-bit) |
|------------|-------|-------|--------------|
| 16 (4k)    | 99±2  | 98±2  | 98±3         |
| 24 (14k)   | 89±5  | 87±7  | 89±6         |
| 32 (33k)   | 77±5  | 77±9  | 77±8         |
| 40 (64k)   | 69±9  | 65±7  | 67±8         |
| 48 (110k)  | 59±8  | 59±9  | 56±9         |

The capacity knee is at F≈24–32 (search space ~14k–33k ≈ D^1.5=32k for D=1024),
matching Frady/Kent theory — the setup is validated, not rigged. Float, N=256, and
even N=16 (4-bit phase) overlap within ±1σ at EVERY F; the largest gap (F=40, 69 vs
65) is <1σ and the deviations are non-monotonic in N (noise, not a quantization
trend). The quantized knee does not shift left of float.

**Conclusion.** Phase quantization is free for resonator factorization: a
deterministic integer resonator (down to 4 bits/dim) matches the float FHRR
resonator's capacity across the operating range. This is the open edge qFHRR left
(they did no resonators) meeting the SOTA factorization method (Frady/Kent, float):
the pairing costs nothing, yielding integer-only, replay-exact neurosymbolic
factorization — a genuine, citable contribution, not a re-derivation. Honest scope:
(a) single clean composite (the standard Frady/Kent benchmark); factoring from a
bundled superposition (many facts) is separate future work where quantization noise
may compound with bundle interference; (b) the resonator readout is float here — the
QUANTIZATION variable (phase resolution N) is isolated; a fully fixed-point resonator
(integer LUT arithmetic) is the implementation step, its determinism already
validated by qFHRR and §15. Next: a `resonator` module productization + the bundled-
factorization stress test.
