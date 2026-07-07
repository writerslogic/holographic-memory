# Capacity Campaign Log — closing the retrieval-vs-information gap

**Problem.** Well-posed per-fact retrieval (recover specific O given exact (S,R)) from
ONE superposition, pushing reliable load past ~0.3·D toward the counting ceiling
(~1.33·D at D=1024, N=256, V=64). All standard VSA codes floor at a small constant
fraction of D (§17–§24). Running log so avenues are not repeated. Append results.

**Core finding (§23/§24 + 4-agent fanout 2026-07-06).** The floor is the DECODER, not
the keys: matched-filter (MF) capacity is M ≈ D/(2 ln V) ≈ 0.12·D (CDMA derivation,
matches §23). The information is present but entangled; a single-correlation linear
decode cannot extract it. Lever = JOINT interference-cancelling decode.

## KILLED / DOA — do not retry
- Readout tricks on dense phase (§23): whitening (worse), iterative Hopfield cleanup
  (= base; per-query argmax already optimal), ensembles (split doubles per-subspace load).
- Sparse phase-support codes (§23): tie base.
- Sparse block codes / EntangledHVec (§24): also floor ~0.03–0.06·D, confound=0.
- Keys-alone (Gold/Kasami/Zadoff-Chu/Hadamard) WITHOUT a structured codebook: object
  phases re-randomize crosstalk → no gain (CDMA derivation). [cheap confirm pending]
- Cover-free / d-disjunct group testing: sublinear load, worse than random (worst-case
  zero-error costs capacity).
- Decorrelating multiuser detector: needs M·V ≤ D → M ≤ 16 at V=64; worse than MF.
- Sidon/B_h key sets (~√D, caps M low); FlyHash (fixed map, no new info); chaos/
  reservoir (linear readout on bounded state); low-rank tensor (violates one-vector
  constraint = cheating).
- [wave 2] Spectral / super-resolution (Prony/MUSIC/ESPRIT): DOA for KNOWN keys —
  collapses to the (killed) decorrelator; reachable only ~0.25·D; needs off-grid +
  exp-small noise, quantization breaks it. Prior art (Poore 2026 wave-geometric
  duality arXiv 2604.22863).
- [wave 2] Temporal / sequence / permutation-trajectory: DOA — Frady-Sommer proved
  linear-in-D, NO order bonus; bounded-ISI+BCJR collapses to AMP; non-overlap =
  sharding in disguise.
- [wave 2] Learned/optimized codes: DOA at CODE level (random ≈ optimal for the
  AWGN-like channel); the only win is a learned DECODER = AMP; memorization-on-
  deploy-facts is cheating.
- [wave 2] Lattice/compute-and-forward AS A SEPARATOR: DOA — CF decodes integer
  SUMS, we need to UN-sum. (Lattices fine as an integer substrate, not a separator.)
- [wave 2] Moment-tensor / CP decomposition (Jennrich unique): D^3 storage =
  cheating; sketched → collapses to AMP.
- [wave 2] Hierarchical coarse/fine phase & q-ary block×phase: FLOOR-INVARIANT
  (re-partitioning D splits SNR identically) / prior art (Frady block codes). NOT a
  capacity contribution.
- KEY PRINCIPLE (hybrid agent): any substrate that merely RE-PARTITIONS D is
  floor-invariant. Wins come ONLY from (a) coding-gain redundancy or (b) breaking
  symmetric power, or (c) a genuinely different algebra/measurement geometry.

## PROMISING — test queue (priority order)
1. **AMP / SPARC decode + Onsager [STRONGEST].** Field IS a sparse-superposition code
   (M sections, one-hot object per section). AMP is capacity-achieving; HMS resonator
   lacks the Onsager term + section structure. Needs the LINEAR count field, not
   phase-normalized. Prior art: FKS did CS/OMP (arXiv 2305.16873, ~2×); AMP+Onsager+
   query-conditioning is the un-imported piece. → testing now (soft-IC / AMP-lite first).
2. Confidence-gated SIC (CDMA): decode high-margin facts, resynthesize+subtract, iterate
   (~0.3–0.5·D reachable). Special case of #1.
3. Quantitative/adder-channel group testing: RS/Vandermonde structured measurements +
   integer least-squares/syndrome decode; recover key-set then solve for fillers.
4. Query-conditioned CS: exploit that the key is KNOWN (stronger prior than FKS).
5. **Spatial coupling (SC-AMP/SC-LDPC) [theory-backed]:** band-diagonal fact→block
   allocation triggers threshold saturation → pushes the AMP threshold toward the
   ceiling (ECC agent). The way to get the last stretch after plain AMP.
6. Power ladder (geometric amplitudes on the count field): the SPARC knob that makes
   AMP capacity-achieving; a knob inside the AMP test, not a separate substrate.

## GENUINELY-NOVEL, NON-COLLAPSING (survive the AMP-collapse — test after AMP)
These do NOT reduce to AMP/matched-filter; they change the algebra or encode geometry:
- **Berlekamp-Welch / rational-function store [likely reaches 0.5·D]:** object =
  coeffs of N(x)/Q(x) over GF(p); facts are (key,value) interpolation constraints;
  ALGEBRAICALLY exact unique-decode to M ≤ (D−1)/2 = 0.5·D, with WB error-correction.
  Integer-exact/deterministic. Caveat: it's exact interpolation (a coded KV store),
  not additive superposition — reaches 0.5·D by LEAVING the superposition regime
  (the algebraic analog of the sharding escape). Adjacent to residue-HDC (prior art);
  the WB error-correction layer is the differentiator.
- **Tropical (max-plus) superposition:** superpose with coordinate-wise MAX
  (idempotent) + bind with add; masking is rank-order not summation → NOT the √M
  dilution law. Genuinely unused in VSA, integer-exact. Wildcard — masking may
  collapse recall; cheap to disconfirm.
- **Cuckoo encoder-gauge nulling:** spend a per-fact free phase/gauge (readout-
  invariant) via coordinate descent to make crosstalk DESIGNED-DESTRUCTIVE before
  storage — the untouched ENCODE-side axis. Uses PhaseHVec. Novel; test small.
- **RRNS redundant-residue content code:** object as residues under redundant
  coprime moduli + CRT/Reed-Solomon error correction → genuine coding gain, integer-
  native. Only wins if per-band error rate is low enough that correction beats
  spending the dims (the experiment).

## SUBSTRATE IDEAS (mix / new — per user)
- LINEAR COUNT field (integer re/im accumulator, NOT atan2 phase-normalized): decisive —
  enables CS/AMP; phase-only kills it. HMS's fixed-point complex field already is this.
- q-ary block code: block-wise one-hot × a phase per slot (block low-collision × phase
  density). [untested]
- Spectral / super-resolution substrate: facts as complex exponentials at distinct
  frequencies; recover via MUSIC/ESPRIT/matrix-pencil (parametric, beats √M). [NEW]
- Polynomial-root substrate: field = coefficients of Π(x − z_i); recover roots. [NEW]
- Residue/CRT HDC (Kymn/Frady 2311.04872): prior art — cite, pair with a joint decoder.

## THEORY (red-team prover, decisive)
- matched-filter floor 0.12·D; **SPARC-AMP tractable threshold ~0.33–0.40·D** = the
  wall for ANY poly-time decoder on the uniform field; ℓ0/spark converse **M < D**
  (true linear cap, tighter than the loose 1.33·D counting bound).
- **0.5·D is in the statistical-to-computational gap** (legal but algorithmically
  unreachable on the fixed uniform field). Closing to 0.5·D is an ENCODE problem:
  spatial coupling + power allocation (linear, → ~D) or nonlinear/algebraic encode.
- Per-avenue caps: AMP/SIC/OMP ≤ ~0.4·D; spectral, learned, group-testing-value all
  capped by DT; spatial-coupled/power-allocated SPARC = OPEN (→ ~D); Berlekamp-Welch
  = exact to 0.5·D but leaves superposition.

## RESULTS
- §25 (amp-decode.rs): joint AMP-lite decode + power ladder = FIRST mechanism to beat
  the floor. Sharp SPARC threshold cliff; 100% below vs matched-filter 80% @0.2·D;
  power ladder moves threshold 0.25→0.3·D (92% @0.3·D vs MF 61%). Reliable knee lifted
  ~3× (0.1·D→0.3·D). Above-threshold collapse = un-damped AMP-lite artifact; proper
  AMP+Onsager+damping → ~0.4·D wall. See PREREGISTRATION §25.
- NEXT (untested): spatial-coupled/power-allocated SPARC (→ ~D, theory-backed);
  Berlekamp-Welch (exact 0.5·D, algebraic); tropical semiring; cuckoo gauge-nulling.
(see docs/PREREGISTRATION-binding-readout.md §§17–25 for full measurements)
