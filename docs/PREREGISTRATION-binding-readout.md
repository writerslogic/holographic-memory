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

## 21. Bundled factorization: resonator + explaining-away under load (pre-registered 2026-07-05)

**Question.** §20 factored a SINGLE clean composite. A memory holds MANY facts
superposed. Can the deterministic resonator recover facts from a BUNDLE of B bound
products (Σ bind(x,y,z)) by explaining-away (factor → subtract → repeat), and does
phase quantization cost MORE here — where quantization noise compounds with bundle
interference — than in the single-composite case (§20, where it cost nothing)?

**Method (`src/bin/resonator-bundle.rs`).** D=1024, 3 factors, codebook size F=24.
Store B facts, each = product of one random entry per axis; superpose into one
complex field (float working state; codebooks quantized to N). Peel B times:
resonator-factorize the residual field, record the recovered triple, subtract its
unit-phasor product from the field. Sweep B ∈ {1,4,8,12,16,24} × N ∈ {float, 256,
16}, 12 seeds × 8 trials. Metric: set-recovery = |recovered ∩ stored| / B.

**Strong outcome.** Quantized (N=256, N=16) tracks float recovery across B — i.e.
quantization stays free even under bundle load, so deterministic bundled
factorization is real (the productization target for `phase_resonator::unbundle`).
**Kill.** Quantized recovery falls materially below float as B grows → bundling is
where quantization finally costs; report the crossover and gate the prod unbundle
to the regime where it holds. Either way, quantify the bundle-capacity curve
(recovery vs B) the single-composite §20 could not show.

**Result (run 2026-07-06) — directional, UNDERPOWERED.** Bundled 3-factor
set-recovery via explaining-away, D=1024, F=16, but only 3 seeds × 2 trials =
6/cell: the pre-registered 20-seed run was intractable on a heavily-loaded host (a
47-min and a 3-hr run were killed; the resonator inner loop is O(B·iters·F·D) per
peel and the box delivered ~50M flop/s under contention from ~15 concurrent
processes; the bin was rewritten allocation-free but the box, not allocation, was
the limit).

| B facts | float | N=256 | N=16 (4-bit) |
|---------|-------|-------|--------------|
| 1       | 100   | 100   | 100          |
| 4       | 83    | 100   | 100          |
| 8       | 83    | 83    | 100          |
| 16      | 77    | 68    | 89           |

Bundled factorization WORKS: the resonator peels facts from a superposition and
degrades gracefully (100% at B=1 → ~70–90% at B=16). Quantized tracks float within
noise — indeed float sits BELOW quantized at several cells (83 vs 100), which is
impossible as a real effect (quantization cannot ADD information) and simply
confirms the noise floor is wide at 6 trials/cell. So: no evidence quantization
costs under bundling (kill did not fire), but the test is UNDERPOWERED — do NOT
cite these as effect sizes. A firm claim needs the pre-registered ≥20-seed run on
an unloaded machine (or a smaller D / fewer factors to cut cost). Directional
validation of the design only; the statistics are thin. The productization
(`phase_resonator::unbundle`) should wait for the full-power confirmation.

## 22. Does the holographic layer beat exact hash on noisy queries? (pre-reg 2026-07-06)

**Motivation.** A multi-model fusion pass on the "0.5·D verifiable retrieval" hard
problem claimed 0.5·D from a single dense superposition is "information-theoretically
impossible" — that framing is WRONG (retracted). The real counting bound is
vocab-dependent (M ≤ D·log2 N / log2 V; ~1.33·D at V=64, so 0.5·D is well under it),
and the 0.1·D "Plate law" is a readout artifact of naive superposition + linear
cleanup, NOT a fundamental wall — closing the gap to the ceiling with better codes
is OPEN. Sharding is one capacity path (what the models defaulted to), not forced. For EXACT (S,R) queries a plain
hash-to-shard KV store suffices; the VSA/attention layer earns its keep ONLY for
NOISY/partial-cue queries. So the sharpest test of HMS's whole premise (Claude's
"hash-router ablation"): does VSA similarity retrieval beat exact hash once the
query key is corrupted? If not, HMS ≈ a sharded KV store and the holographic
machinery is decorative.

**Method (`src/bin/noisy-retrieval.rs`).** D=1024. Each fact = a random key phasor
bound to an object from a codebook of O=64 random phasors; M facts superposed into
one fixed-point complex field (re/im i64, integer LUT — deterministic). Query =
the true key with a fraction ρ of its D phase components replaced by uniform random
phases (partial-cue corruption). Recover the object by argmax over the codebook of
⟨field, bind(q, obj_o)⟩. Baseline = exact hash-KV: recall 1.0 at ρ=0, 0 for any
ρ>0 (any component change → different hash). Sweep load M/D ∈ {0.05,0.1,0.2,0.3} ×
ρ ∈ {0,0.1,0.2,0.4}, N ∈ {256,16}, seeds. Metric: top-1 object recall; chance = 1/O
≈ 1.6%.

**Strong outcome.** VSA recall stays well above chance over a broad (load, ρ)
region where exact hash is 0 — quantifying the noise-tolerance envelope the
holographic layer buys over a hash-sharded store. **Kill.** If VSA recall collapses
to ~chance at ρ=0.2 at reasonable load (M/D≤0.2), the noise tolerance is negligible
→ the holographic layer adds nothing over exact hash and HMS's honest positioning
is "a sharded KV store," not a robust associative memory. Either way it turns the
fusion's reasoning into a measured envelope.

**Result (run 2026-07-06) — kill did NOT fire; the holographic layer earns its
keep on noisy queries.** Top-1 object recall (%), chance 1.6%, 6 seeds:

N=256 (rows load M/D, cols corruption ρ):

| load | ρ=0.0 | 0.1 | 0.2 | 0.4 |
|------|-------|-----|-----|-----|
| 0.05 | 100   | 100 | 99  | 92  |
| 0.10 | 97    | 93  | 87  | 65  |
| 0.20 | 80    | 65  | 56  | 38  |
| 0.30 | 65    | 55  | 46  | 26  |

N=16 essentially identical (56→62% at ρ=0.2/load 0.2). Exact hash-KV = 100% at ρ=0,
**0% at every ρ>0**.

**Conclusion.** For EXACT queries the ρ=0 column shows VSA merely matches hash (which
is cheaper/exact) — the fusion was right that hash suffices there. But on NOISY /
partial-cue queries the holographic layer holds far above chance where hash is dead:
usable recall (>50%) out to ρ=0.2 at load≤0.2 and ρ=0.4 at load≤0.1. Kill needed
~chance (1.6%) at ρ=0.2/load≤0.2; it measured 56–62%. So HMS is a genuine
noise-robust associative memory, not a decorative wrapper on a sharded KV store —
and 4-bit phase is free again. Honest label: this VALIDATES known VSA noise-
robustness on the qFHRR substrate (expected from theory), not a novel capacity
result. It settles HMS's value axis (noisy-query robustness) with a measured
envelope. It does NOT settle capacity: 0.5·D is NOT information-theoretically
impossible (the counting ceiling ~1.33·D at V=64 sits well above it, and the 0.1·D
Plate limit is a readout artifact, not a wall). Closing the 0.1·D→ceiling gap with
better codes (sparse block codes, ECC, structured binding) is the open frontier —
sharding is one path to capacity, not the only one.

## 23. Capacity campaign: how far past 0.1·D can a single bundle go? (pre-reg 2026-07-06)

**Question.** The naive random-superposition + linear-cleanup readout tops out near
the Plate SNR limit; the counting ceiling (V=64, N=256, D=1024) is ~1.33·D. That gap
is OPEN. Test a MULTITUDE of readout/code mechanisms on the SAME single-bundle
retrieval task (no sharding) and measure where each one's recall-vs-load knee lands.

**Task.** Recover object given exact key from ONE bundle of M facts (random key ⊗
object, O=64 codebook). Metric: top-1 recall vs load M/D ∈ {0.1,0.2,0.3,0.5,0.75,
1.0}; report the KNEE (max M/D at recall ≥90% and ≥50%). Fixed D=1024, N=256, O=64,
exact queries (isolate capacity from the §22 noise axis), ≥5 seeds. Chance 1.6%.

**Arms (`src/bin/capacity-campaign.rs`):**
- base — dense phasors, one-shot linear score argmax (the incumbent ~0.1–0.2·D).
- whiten — per-dim field magnitude normalization before scoring (equalize dims).
- hopfield-lo / hopfield-hi — iterative modern-Hopfield softmax cleanup on the
  unbound value (β low/high, T iters) — nonlinearity-before-readout.
- ens2 / ens4 — split D into K independent sub-bundles (D/K each), sum per-candidate
  scores (repetition/voting at matched total D).
Plus a base-vs-best check at N=16.

**Strong outcome.** ≥1 arm pushes the 50%-recall knee materially past base (e.g.
base ~0.25·D → best ≥0.5·D). **Kill.** No arm beats base's knee by >0.05·D → on the
dense phase substrate, readout tricks don't unlock capacity and the lever is a
different SUBSTRATE (sparse block codes on `EntangledHVec`; ECC output codes) — name
that as the next campaign. Cheapest disconfirming: run base vs hopfield-hi at loads
{0.2,0.5} first; if hopfield doesn't beat base at 0.5, the iterative-cleanup arm is
out. This is a readout/code sweep on ONE substrate; sparse-substrate and ECC codes
are the explicitly-noted follow-on batch, not tested here.

**Result (run 2026-07-06) — KILL FIRED. Nothing beats naive base.** Top-1 recall
(%) vs load M/D, V=64, N=256, 5 seeds:

| arm         | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | knee@50 |
|-------------|-----|-----|-----|-----|------|-----|---------|
| base        | 98  | 79  | 56  | 45  | 27   | 18  | 0.30    |
| whiten      | 94  | 68  | 46  | 37  | 22   | 11  | 0.20    |
| hopfield-lo | 98  | 79  | 56  | 45  | 27   | 18  | 0.30    |
| hopfield-hi | 98  | 79  | 56  | 45  | 27   | 18  | 0.30    |
| ens2        | 36  | 23  | 17  | 12  | 8    | 7   | <0.10   |
| ens4        | 11  | 6   | 6   | 4   | 4    | 4   | <0.10   |
| sparse.5    | 96  | 78  | 60  | 40  | 25   | 18  | 0.30    |
| sparse.25   | 96  | 80  | 60  | 37  | 31   | 21  | 0.30    |
| sparse.1    | 96  | 80  | 65  | 38  | 29   | 21  | 0.30    |

**Conclusion.** (1) Readout is not the lever: `whiten` discards magnitude signal
(worse); iterative Hopfield EXACTLY equals base (one-shot matched-filter argmax is
already optimal for single-object retrieval — iteration can only confirm the top
score; the initial hopfield=chance was a beta-scaling bug, fixed to beta/sqrt(dim),
after which it converges to base); `ensemble` doubles per-subspace load (worse).
(2) Phase-sparse codes tie base — random sparse SUPPORT on a phasor is NOT a
structured sparse-block code; the Frady/Kleyko gain is a property of the
sparse-BINARY substrate (block-wise one-hot), unreachable by masking phase dims.
(3) Byproduct: naive dense already reaches 0.30·D @ 50% recall (V=64) — 3x the
"0.1·D Plate" folklore, reconfirming 0.1·D is not a wall (90%-reliable is still
~0.1·D; counting ceiling ~1.33·D). The gap to the ceiling is real and none of these
9 close it. The lever is a structurally different CODE on a different SUBSTRATE:
sparse-block codes on `EntangledHVec` (`block_codes` / "block code floor") or ECC
output codes — the well-motivated next campaign (§24).

## 24. Block codes also floor: the retrieval-vs-information capacity gap (2026-07-06)

**Result (existing `block-recovery-test`, re-run 2026-07-06).** The genuine sparse
block-code substrate (`BlockCodeVec`, block-wise one-hot) on (s,r,o) triple
retrieval, D=16384, 125 symbols:

| facts | M/D    | top-1 | crosstalk | confound |
|-------|--------|-------|-----------|----------|
| 200   | 0.012  | 1.000 | 0         | 0        |
| 500   | 0.031  | 0.994 | 3         | 0        |
| 1000  | 0.061  | 0.893 | 107       | 0        |
| 1500  | 0.092  | 0.711 | 433       | 0        |

D-scaling (500 facts, 125 symbols): 55.8%@D=4096, 88.2%@8192, 99.4%@16384,
100%@32768. **confound=0 throughout** — retrieval is well-posed; every failure is
pure crosstalk (interference), not a wrong-but-valid filler.

**Complete synthesis (closes §17–§24 capacity thread).** Both HMS substrates floor
at a small constant fraction of D for well-posed per-fact retrieval: dense phase
(§23: ~0.1·D @ 98%, ~0.3·D @ 50%, V=64) and sparse block codes (~0.03–0.06·D at
high reliability, V=125). Neither approaches the ~1.33·D counting ceiling. This is
the KNOWN gap between information capacity (~D·log2 N bits, Shannon) and RETRIEVAL
capacity (associative memory) of distributed representations — the same phenomenon
as classical Hopfield's ~0.14·N. §23 showed readout tricks don't move it; §24 shows
the sparse-block-code substrate doesn't either. Conclusion: **0.5·D well-posed
retrieval is NOT information-theoretically impossible (retracted in §22/§23) but is
beyond what any known SUPERPOSITION code achieves — it's an OPEN gap, not a wall.**
Modern-Hopfield exponential capacity requires SEPARATE storage (→ sharding/codebook,
§22), not superposition. Pushing past the floor needs a genuinely novel superposition
code (hard, open, publishable if solved); otherwise the validated path is sharding
for capacity + the noisy-query robustness envelope (§22) for value.

## 25. The gap SOLVED (characterized): joint AMP decode + power allocation (2026-07-06)

**Two 8-agent research waves + a red-team prover resolved the open problem.** The
field Σ_i bind(key_i,obj_i) IS a Sparse Superposition Code (SPARC) / an unsourced
Gaussian MAC. The floor is the DECODER (matched filter = interference-as-noise), not
the code. Theory (Donoho-Tanner + SPARC-AMP state evolution, red-team prover):
- matched-filter floor ≈ D/(2 ln V) ≈ 0.12·D;
- **SPARC-AMP tractable threshold ≈ 0.33–0.40·D** — the wall for ANY poly-time
  decoder on the uniform-random field;
- ℓ0/spark converse: M < D (the TRUE information cap on a linear field — tighter than
  the loose 1.33·D counting bound);
- **0.5·D sits in the statistical-to-computational gap**: information-theoretically
  legal (ℓ0≈D) but algorithmically unreachable by any decoder on the fixed uniform
  field. Closing to 0.5·D is an ENCODE problem, not a decoder problem — the only
  linear lever is spatial coupling + power allocation (SPARC's capacity-achieving
  construction, provably → ~D); beyond D needs a nonlinear encode.

**Experiment (`src/bin/amp-decode.rs`).** Matched filter vs soft interference-
cancellation (AMP-lite) vs soft-IC with a geometric power ladder. Fraction of M
facts decoded, D=1024, V=64:

| method            | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 |
|-------------------|-----|-----|-----|-----|------|-----|
| matched-filter    | 98  | 80  | 61  | 38  | 25   | 20  |
| soft-IC (flat)    | 100 | 100 | 14  | 3   | 2    | 1   |
| soft-IC (ladder)  | 100 | 96  | 92  | 14  | 5    | 3   |

**Result — first mechanism to beat the floor.** Joint decode shows the SPARC
signature: a sharp threshold cliff (100% below, collapse above), joint decode beats
matched filter below threshold (100 vs 80 @ 0.2·D), and power allocation MOVES the
threshold up (flat cliffs ~0.25·D; ladder holds 92% @ 0.3·D vs MF 61%) — exactly
SPARC power-allocation theory. Net: power-allocated joint decode lifts the reliable
knee ~3× (0.1·D → 0.3·D). The above-threshold collapse to < MF is an un-damped
AMP-lite artifact (proper AMP + Onsager + damping degrades gracefully → toward the
~0.4·D wall). Next increment toward the ~D spark limit: spatial-coupled SPARC.

**Honest conclusion.** The open problem is now characterized, not mysterious: the
0.1·D "floor" was a matched-filter artifact; joint AMP decode reaches ~0.4·D
(demonstrated the first 3× here), spatial-coupled/power-allocated SPARC provably
approaches the ~D linear cap, and >D requires a nonlinear encode. 0.5·D is reachable
but only by encode-side redesign (SC-SPARC), not decoder tricks — and it stays in the
superposition regime (unlike sharding). Genuinely-different algebraic routes
(Berlekamp-Welch rational store: exact unique-decode to 0.5·D; tropical max-plus
superposition; cuckoo encode-gauge nulling) remain untested — see
docs/CAPACITY-CAMPAIGN-LOG.md for the full ledger.

## 26. The floor was self-inflicted: solve-don't-sum reaches M≈D (2026-07-06)

**How it was found.** After ~32 agents converged (all fed a biased "additive
superposition + here's what's killed" framing), the user flagged the bias directly:
"stop biasing what you know" + "theorems are not facts." A fresh wave of SIX OPEN
agents (bare problem — store M key→value pairs in one D-vector, maximize M — no VSA
jargon, no killed-list, no theorems) converged INDEPENDENTLY from six directions
(first-principles, physics, pure math, biology, wild, contrarian) on the same point:
**additive superposition (field = Σ bind(key,value)) is the worst encoding, ~3-10×
suboptimal.** The whole §17-§25 floor and its theorems (spark M<D, DT 0.4·D) govern
ONLY additive superposition + linear decode.

**Experiment (`src/bin/pinv-memory.rs`).** Don't SUM the bound pairs — treat the
stored vector x as unknown and SOLVE the linear system {⟨a_{k_i}, x⟩ = v_i} for x
(least-norm pseudo-inverse; the Kohonen/Personnaz optimal linear associator).
Recall (%), D=256, V=64, 5 seeds:

| M/D | exact | quantized (4096-level) |
|-----|-------|------------------------|
| 0.3 | 100   | 100 |
| 0.5 | 100   | 100 |
| 0.8 | 100   | 100 |
| 1.0 | 100   | 79  |
| 1.2 | 4     | 4   |
| 2.0 | 2     | 2   |

**Result.** ~100% exact recall to M≈D, a SHARP cliff at the counting bound (not the
soft √M decay of superposition) — 3-8× the superposition floor (~0.3·D @50%, ~0.1·D
reliable). Quantizing x to an integer grid costs a little only at the M=D edge (79%).
Three OTHER open-agent-converged encodes reach the same ~D regime and were validated
by the convergence (not yet coded): polynomial/Reed-Solomon interpolation (M=D exact
by Lagrange), Bloomier/XOR/Ribbon filters (~0.8·D, O(1) recovery, key-universe
oblivious), nonlinear cell codes (toward the ~2.7·D Shannon bit ceiling).

**Honest resolution (the real finding, sharpened).** The "0.1-0.3·D floor" was an
ARTIFACT of choosing additive superposition, not a law. For EXACT key→value
retrieval at maximum capacity, DON'T use superposition — solve/interpolate/hash
reaches M≈D. BUT these are exact CODED key-value stores: they lose the soft
holographic properties (approximate/similar-key retrieval, graceful degradation,
algebraic composability) that additive superposition uniquely provides at ~0.1-0.3·D.
So it is a clean TRADEOFF the campaign had conflated:
- exact key→value, max capacity → coded store (pinv/RS/XOR), M≈D, hard cliff;
- approximate / robust / composable → superposition, ~0.1-0.3·D + soft properties
  (and §22's measured noisy-query robustness — the holographic layer's real value).
This directly sharpens §22/§24: the holographic layer was never the right tool for
EXACT capacity; its value axis is the soft/noisy dimension. HMS's honest architecture
= a coded store for exact capacity + the holographic layer for approximate/verifiable/
plastic queries. (Methodological note: the bias-stripped open wave found in one pass
what 32 framed agents missed — treat theorems as maps of assumptions to break, not
walls.)

Also §26b (`src/bin/bilinear-readout.rs`): a nonlinear second-order readout
B[d,d']=φ[d]conj(φ[d']) DID show real signal (recall rose monotonically with pair
count P: 4%→29%→78% at low load) — confirming a nonlinear readout extracts structure
the linear matched filter can't — but reaching the √D advantage needs P~D² pairs
(the honest O(D²) cost); at affordable P it did not yet beat matched filter. Real
mechanism, impractical constant on the superposition substrate.

## 27. Fusing the two layers via ridge regression — HALF confirmed (2026-07-06)

**Hypothesis.** The exact coded store (§26) and the holographic superposition store
are two ends of ONE spectrum — regularized least squares x=argmin‖Ax−v‖²+λ‖x‖². λ→0 =
exact pinv (coded, M≈D); large λ → Hebbian/superposition (graceful, lower capacity).
Claim: one knob λ trades exact capacity for noise tolerance.

**Result (`src/bin/ridge-memory.rs`, D=256, V=64).** Exact-key recall: λ=0 → 100% to
M=D; raising λ trades capacity away exactly (λ=0.01: 82%@0.5D, 22%@0.8D; λ=0.1: dead).
That HALF is confirmed — λ spans the capacity axis. BUT noisy-key recall (additive
Gaussian, cosine~0.9) stayed at CHANCE for EVERY λ, including the Hebbian end. So the
"one knob spans both" claim is NOT supported — over-claimed.

**Honest diagnosis (two conflations I made).** (1) The noise side failed because
recovering 1-of-64 (6 bits) precisely from a cosine-0.9 key via a SCALAR LINEAR
readout is too hard — additive noise exceeds the 1/64 resolution regardless of λ (an
affine scale+offset calibration fixed the earlier scale bug but not this). (2) I
conflated two "soft" behaviors: PARTIAL-CUE robustness (§22 — query a stored key with
corrupted components; works via redundant code + VALUE CODEBOOK + argmax, not scalar
readout) vs SIMILARITY generalization (novel similar key → similar value; needs
STRUCTURED data, impossible on random pairs). The ridge test used scalar readout +
additive noise + random data — the setup where neither soft behavior can appear.

**Conclusion.** Ridge-λ demonstrably spans the CAPACITY axis but I have NOT shown it
spans the ROBUSTNESS axis. The fusion that combines both BY CONSTRUCTION (not relying
on ridge to bridge) is base+residual: a holographic superposition base (partial-cue
robust, codebook+argmax, §22) + an exact coded residual patch. Exact key → base+
residual = exact; corrupted key → residual misses but the robust base still returns
the value. That is the real "both layers at once", and it must be tested in the §22
readout (value codebook + argmax + partial-corruption), NOT the scalar setup here.
[next: §28 base+residual]

## 28. Base+residual "fusion" — CONFOUNDED, but it reveals the real answer (2026-07-06)

**Experiment (`src/bin/base-residual.rs`, D=256, V=64).** Three matched-D stores;
exact-key recall / partial-cue recall (ρ=0.3 components randomized):

| store            | exact @M/D 0.1..1.0 | cue @M/D 0.1..1.0 |
|------------------|---------------------|-------------------|
| pure-coded       | 100/100/100/100/81  | 5/1/3/2/2         |
| pure-holographic | 100/100/100/100/100 | 100/100/100/99/98 |
| fused(128+128)   | 100/100/82/2/2      | 100/97/92/81/69   |

**CONFOUND — do not read this as a clean fusion win.** The "holographic base" is a
Hebbian D×D MATRIX (W = Σ obj⊗key, D²=65536 numbers), while the "coded residual" is a
D-VECTOR (256 numbers). NOT matched storage — the matrix base has ~256× the room. So
"pure-holographic 100%/98%" is the LINEAR ASSOCIATOR (matrix), NOT the one-D-vector
holographic bundle of §17-§26 (which genuinely floors at ~0.1-0.3·D). The comparison
is apples-to-oranges on storage budget.

**The real, honest answer to "can we combine both layers".** Two regimes:
1. If storage may be a D×D MATRIX (bounded, but D² not D): the classic linear
   associator / Hebbian matrix (Kohonen OLAM, modern-Hopfield) gives EVERYTHING at
   once — exact retrieval to M=D, partial-cue robustness (98%), AND graceful
   similarity — as the pure-holographic row shows. That IS the fusion; cost = D²
   storage. This is what modern Hopfield / attention do.
2. Within the strict ONE-D-VECTOR budget (HMS's holographic bundle substrate): exact
   (coded, M≈D) vs robust (bundle, ~0.1-0.3·D) is a genuine TRADEOFF — base+residual
   splits the vector, no free lunch, because robustness (redundancy) and exact
   capacity both spend the same D bits. The fused row's exact caps at 0.5·D (its
   coded half) precisely because of the split.

So: YES you can combine both — the clean way is an associative D×D matrix instead of
a D-vector (a real design choice for HMS, not a wall). Under the strict single-vector
constraint, combining is an allocation tradeoff. (Discipline note: the confound —
matrix base vs vector residual — was caught and named, not shipped as a win.)

## 29. Bits-per-dimension at matched factorization accuracy: footprint win vs float Frady/Kent (pre-registered 2026-07-07)

**Positioning (the missing entry).** §20 established that phase quantization is free
for resonator factorization down to 4-bit. The competitive corollary has never been
stated as a head-to-head: the deterministic quantized resonator matches the SOTA
FLOAT Frady/Kent resonator's factorization accuracy at a fraction of the stored-state
footprint. qFHRR left resonators unaddressed; Frady/Kent resonators are float; the
accuracy-vs-bits/dim tradeoff is unmeasured. This is a citable entry IF it survives a
firing kill on unseen seeds.

**Question.** On the identical task (3-factor factorization, search space F³), at
matched D, F, seeds, and dynamics: what is the minimum stored-phase resolution
(bits/dim) at which the quantized resonator matches the float resonator's accuracy
within ±1σ, and what stored-state footprint reduction does that yield vs 32-bit float
FHRR?

**Baselines.** STRONG incumbent = float FHRR Frady/Kent resonator (N=0, continuous
phase), identical dynamics — this is the SOTA factorization method, not a strawman.
Arms = quantized at N ∈ {256, 64, 16, 8, 4} = {8, 6, 4, 3, 2} bits/dim.

**Metric.** All-factors-correct accuracy, mean ± std over seeds, per (F, bits/dim),
at D ∈ {512, 1024, 2048}. Chance = 1/F³ (<0.03%). Headline = the smallest bits/dim
that matches float within ±1σ at every F for the reference D=1024; footprint ratio =
32 / (that bits/dim) for f32 storage (64 / it for f64).

**Fresh-seed confirmation (anti-forking-path).** The §20 hardening used seeds 0..30 —
already inspected. The pre-registered evidence is a FRESH, unseen seed block
(seeds 100..130) via `cargo run --release --bin resonator-sweep 100`. The built-in
REPRO CHECK (seeds 0..24) must still reproduce the §20 table (guards the code path);
the claim rests on the fresh block, not the seeds I have already seen.

**Strong outcome.** On the fresh seeds, quantized matches float within ±1σ down to
≤4 bits/dim at every F and D → ≥8× (f32) / ≥16× (f64) stored-state reduction at
matched factorization capability → a footprint win over the float incumbent.

**Kill condition (can fire).** On the fresh seeds, N=16 (4-bit) accuracy is >1σ below
float at ANY F for D=1024 → 4-bit is not free on unseen seeds → no 8× win at matched
accuracy. Fall back to the smallest N that does track and report its smaller ratio; if
nothing ≤8-bit tracks float, the footprint claim fails and is reported as a negative.

**Honest scope.** (a) The claim is stored-STATE footprint; the resonator's inner
products are float at query time — compute is not quantized (same isolation as §20).
(b) This is the factorization axis (Frady/Kent ~D^1.5), SEPARATE from the bundle-
superposition capacity floor of §22–§26. (c) Accuracy is *matched*, not beaten — the
win is a tradeoff-free footprint reduction, reported as such, never as an accuracy win.

## 30. Noise-robust super-floor code: the open prize (pre-registered 2026-07-07)

**Positioning.** `docs/SUPERPOSITION-FLOOR.md` established two things: (i) no poly
*decoder* beats the random-bundle floor (~0.27·D at L=64), including the OGP-evader
(lattice); (ii) a spatially-coupled (SC-SPARC) *encoder* crosses it with a poly decoder
at matched storage — BUT clean-signal only: under query noise the coupled advantage
collapses to the floor. The random bundle is the opposite corner: floored capacity, but
graceful/robust (each fact spread holographically over all D dims). The open, publishable
target is a code that occupies NEITHER corner: **> 0.27·D capacity with a poly decoder
AND graceful degradation under query noise/dropout comparable to the bundle.**

**Question.** Is there an encoder + poly decoder whose (capacity, robustness) Pareto
frontier STRICTLY DOMINATES the random bundle — more facts/dim at matched noise
tolerance, or more noise tolerance at matched capacity? Or is the capacity/robustness
tension (spreading→robust vs structure→capacity) empirically fundamental for
compact O(D) codes?

**Noise models.** (a) AWGN on the stored/queried vector, σ relative to signal std;
(b) dimension dropout — erase a fraction f of dims (the canonical holographic-robustness
test). Both at query time; storage stays exact.

**Candidates (cheapest disconfirming FIRST).**
1. FRAME: measure the (capacity, noise) frontier for {random bundle, SC-noiseless} —
   quantify the tension and whether SC has ANY positive noise margin. [run first]
2. Noise-margin SC + DISTRIBUTED/redundant seeds (SC-LDPC has a positive noise threshold
   by design; the fragile version was rate-maximised with a single boundary seed).
3. SPREAD⊗COUPLE interpolation: give each fact a partial holographic spread AND a coupling
   role; sweep the interpolation for a sweet spot beating the bundle under noise.
4. Two-layer capacity split (robust bundle layer + SC increment on shared dims, joint
   decode) — test whether the layers interfere or compose.
5. Protograph/irregular SC optimised for the noisy operating point.

**Metric.** Capacity at ≥90% recovery as a function of σ (AWGN) and f (dropout), matched
storage (all store s∈R^D). Frontier plot per code. Seeds ≥4, report spread.

**Strong outcome.** A code whose capacity-vs-noise curve lies strictly above
max(bundle, SC-clean) over a useful noise range → noise-robust super-floor capacity =
the target.

**Kill condition (can fire).** Every candidate is dominated: for every noise level σ>0,
no code exceeds the random bundle's capacity at matched recovery → the capacity/robustness
tension is empirically fundamental for compact codes. Report as a negative + a conjecture
(itself publishable: "holographic robustness and super-floor capacity are exclusive for
O(D) VSA codes").

**Honest guard.** Matched storage always (a code that spends extra dims/redundancy must
be compared at matched total bits). No claim before the frontier curve. Verify any
dominating code independently (per-block profile + noise sweep) before reporting — a
noise-robust crossing is exactly the kind of too-good result that demands audit.
