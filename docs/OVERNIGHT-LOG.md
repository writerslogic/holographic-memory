# Overnight consolidation + validation log

Loop started 2026-07-07 ~02:40 PDT. Goal: turn the validated §20 deterministic
phase-resonator result into a shipped, citable, reproducible capability, then
harden its validation and prod-code quality. No new research claims unattended.

Gate per unit: `cargo test --lib --features experimental` + `cargo clippy
--workspace --features experimental -- -D warnings` must pass; commit; log one line.

## Progress

- **Item 1 done** (02:5x): `PhaseResonator` reusable-index entry point + clean
  `core::{PhaseResonator, phase_resonator_factorize}` re-export + compiling 3-factor
  doctest. Free fn kept for python.rs. Gate: 15 lib tests + 1 doctest + clippy clean
  (all `--features experimental`). Baseline §20 tests unchanged and green.
- **Item 2 done**: `docs/DETERMINISTIC-RESONATOR.md`. Re-ran `resonator-factorize`
  (24 seeds × 32 trials, D=1024) in this loop; output REPRODUCES §20 exactly
  (F=16: 99±2/98±2/98±3 float/N256/N16 ... F=48: 59±8/59±9/56±9). No surprise, no
  quarantine. Doc labels it validation of qFHRR + Frady/Kent, cites both, gives repro
  command + chance floor (1/F³). All numbers carry seed count + std.
- **Item 3 done**: `resonator-factorize.rs` reproducibility header — documents the
  fixed seed set (0..24), pure-function determinism (no RNG/clock/threads), integer-
  rounded stable output, and the CORRECT repro command (was missing the required
  `--features experimental`). Comment-only; experiment logic and output unchanged.
- **Correction**: item 3 wrongly claimed the bin "requires the experimental feature".
  It does not — `resonator-factorize` is std-only (auto-discovered, no required-
  features), verified it builds+runs with plain `cargo run --release --bin
  resonator-factorize`. Fixed the header and the doc repro command to drop the
  misleading `--features experimental`. Lib/doctest commands (which DO need the
  feature) are unchanged.
- **Item 5 done**: prod-code quality on phase_hvec.rs + phase_resonator.rs. Accessor
  doc comments; hardened `argmax` comparator against a NaN panic (finite-value
  behavior unchanged); added boundary tests — empty/single-element bundle, zero-dim
  similarity, N=2 min & N=65536 max resolution, empty codebooks, single factor/entry,
  N=2 factorization, and a mismatched-resolution panic test. 25 phase tests pass;
  clippy clean. §20 tests untouched.
- **Item 4 done**: `resonator-sweep.rs` hardening artifact + folded results into
  DETERMINISTIC-RESONATOR.md. 30 seeds × 32 trials; phase bits float/8/6/4/3/2
  (N∈{0,256,64,16,8,4}) × D∈{512,1024,2048}. Built-in REPRO CHECK (D=1024, 24 seeds,
  N∈{0,256,16}) reproduces the frozen §20 table CELL-FOR-CELL → reimplementation
  faithful. Result: 4-bit confirmed free under wider seeds; 3-bit also ~free; 2-bit
  (N=4) is the first resolution with a consistent small deficit near the knee (F=24:
  80±7 vs 89±5). CONFIRMS + bounds §20; does not contradict/beat it → no quarantine.
  Dimension axis: pattern holds at all D, capacity scales ~D^1.5 (Frady/Kent).

## Closeout (queue exhausted ~03:2x)

All five queue items committed (+1 self-caught correction). Verification:
- Full lib suite `cargo test --workspace --lib --features experimental`: **337 passed,
  0 failed** — no crate-wide regression from the mod.rs re-exports or argmax change.
- Every file I authored passed scoped clippy (`--lib --features experimental`, and each
  new bin individually). Item 1 doctest green.
- Python-binding work (Cargo.toml/lib.rs/python.rs/pyproject.toml/publish-pypi.yml)
  left untouched and uncommitted throughout. Nothing pushed.

### NEEDS REVIEW (pre-existing, NOT introduced this session — not fixed)
- `cargo clippy --workspace -- -D warnings` (default features, no `--features
  experimental`) fails with ONE error: `needless_range_loop` in
  `src/bin/resonator-bundle.rs:123` (`for f in 0..FACTORS`). This bin is the §21
  bundled-factorization artifact; last touched in commit 61eb870 (2026-07-06, before
  this loop) and NOT in this session's diff — the lint predates my work and surfaces
  only because the bin is auto-discovered under default features. Left unfixed by
  design: §21 is the "do NOT ship / underpowered" path, it is experiment code (research
  carve-out: no unprompted edits), and it is outside the queue scope. Trivial mechanical
  fix (`for (f, _) in ...enumerate()`) if the user wants the default-features workspace
  clippy green; flagged for a decision rather than touched unattended.
  **RESOLVED (on user request to review):** applied the outcome-neutral enumerate
  rewrite (`for (f, cb) in books.iter().enumerate()` + `cb.iter()`; `cb === &books[f]`,
  books has exactly FACTORS entries). Left the argmax-closure `books[f]` site alone.
  Full default-features `cargo clippy --workspace -- -D warnings` now GREEN across all
  38 bins (0 warnings). No logic/outcome change to the §21 experiment.

### §29 bits/dim head-to-head (pre-registered, on user request)
- Pre-registered §29 (footprint vs float Frady/Kent) with a firing kill; added optional
  seed-base to resonator-sweep for a fresh-seed confirmation. Ran seeds 100..130 (unseen):
  REPRO CHECK still matches §20; kill did NOT fire — 4-bit equals/exceeds float within
  ±1σ at every F and D (D=1024: 97/90/78/70/60 vs 97/88/77/67/58). Claim confirmed:
  4 bits/dim matches float factorization accuracy → 8x (f32)/16x (f64) smaller stored
  state. Folded into DETERMINISTIC-RESONATOR.md. Accuracy matched, not beaten.

### Superposition-floor cold exploration (on user request — attended)
- Launched parallel cold explorations (zero-bias, test-the-bound). Early findings:
  - CS/AMP: blind AMP reaches M/D≈0.25–0.27 (2–3x naive) WITH soft readout; BUT
    genie-init hard decode holds 100% to M/D≈0.75 and self-corrects from a corrupted
    start → floor is COMPUTATIONAL (finding the solution), not information-theoretic
    (it exists + is verifiable to ≥0.75D). Refutes strict "0.5D impossible"; lever =
    initialization (spatial coupling / stored hint). NEEDS INDEPENDENT VERIFICATION.
  - Modern-Hopfield readout: clean negative on capacity (reduces to nearest-neighbor,
    SNR-bound), real positive on output fidelity. Not a capacity lever.
  - Nonlinear/higher-order lift: clean negative at matched storage (data-processing
    bound, tested); matrix-memory "win" is a storage illusion.
- **SYNTHESIS (7 mechanisms, converged).** Also ran SIC/peeling, low-coherence
  codebooks, spatial-coupling+hint, learned/unrolled decoder — all floor at M/D≈0.27
  blind, each for a distinct TESTED reason (error propagation; coherence is the wrong
  predictor; noiseless substrate kills threshold-saturation; learning converges to the
  AMP fixed point). Independently REPRODUCED the crux (scripts/superposition_floor.py):
  AMP 100%→M/D=0.25 then floors; genie-init stays 1.0 to 0.75 while random-init
  collapses. Verdict: the floor is a COMPUTATIONAL hard phase, not an information wall —
  solution verifiable/stable to ≥0.75D but unfindable blindly. CORRECTION to my earlier
  gloss: 0.75 is the genie STABILITY radius (start at truth), NOT an algorithmically
  reachable threshold; no tested poly method crosses 0.27. We did NOT beat the floor for
  soft superposition; §25/§26's M≈D was exact CODED stores (hard cliff, no soft recall).
  Written up as docs/SUPERPOSITION-FLOOR.md (honest negative; reinforces capacity=sharding).
- **ASSUMPTION ATTACK (on user request — test the theorems, run the suggestion).**
  Inventoried theorem-based assumptions; ran the suggested VAMP/OAMP + power-ladder/
  local-search solvers (scripts/superposition_floor_probes.py). Findings: (1) AMP-
  optimality assumption partially FALSE — VAMP holds 1.0 at M/D=0.25 where AMP frayed to
  0.875 (shared-codebook correlation hurt AMP), but it only nudges the knee ~0.25→0.27;
  nothing crosses 0.30. (2) Local-search refine + 32-restart are WORSE past the floor
  (fall into spurious minima) — positive hard-phase signature. (3) Identifiability
  ceiling probe: genie stable + no competing exact fit to M/D=1.5 (was only tested to
  0.75) → in noiseless arithmetic the barrier is ENTIRELY computational; the practical
  wall is quantization/noise → sharding. Floor at ~0.27 survives all six solvers.

### Encoder co-design CROSSES the floor (on user push: "question the encoder, quadruple effort")
- Wave 2 questioned the fixed-random-encoder assumption. Exact solvers (ILP/Grobner/DFS)
  confirmed identifiability past 0.27 at EXP cost; survey propagation, spectral/tensor,
  evolutionary, hybrid-syndrome all floored (each with a tested mechanism). BUT a
  co-designed SPATIALLY-COUPLED (SC-SPARC) encoder CROSSES: independently verified
  (scripts/superposition_floor_coded.py) — at M/D=0.29 random=0.43 vs coupled=0.92, matched
  D/M/storage, poly block-AMP decoder, soft readout preserved. Per-block profile shows the
  decoding wave propagates through fully-interfering MIDDLE blocks → real threshold
  saturation, NOT sharding. Guard: block-AMP reproduces the floor on the random code.
  HONEST CAVEATS: (1) known coding theory (SC-SPARC threshold saturation), not a new algo;
  (2) CLEAN-SIGNAL only — noisy query collapses the advantage back to the floor, so it
  raises clean capacity (~+33-50% facts/real), not HMS's robust/noisy capacity. Folded into
  docs/SUPERPOSITION-FLOOR.md. Lattice OGP-evader agent still pending (to collect).
- **Lattice OGP-evader RESOLVED (negative).** Collected + ran the head-to-head myself:
  at matched (D,L), LLL is STRICTLY WORSE than AMP (D=48,L=8,M/D=0.5: amp 0.94 vs lat
  0.13; L=16,0.3: amp 1.0 vs lat 0.09). Embedding valid (100% at low load), so real
  negative — LLL approx factor is exp in lattice dim (D+M+N), collapses below the AMP
  threshold; the earlier "100% at 0.5" was an L=4 confound (easy for AMP too). Last
  poly-ALGORITHM candidate closed. Only verified crossing = encoder co-design (SC).
  EXHAUSTIVE VERDICT: ~13 decoder methods / 8 families floor on the random code; exact
  solvers = identifiability at exp cost; lattice(OGP-evader) worse than AMP; the floor
  falls ONLY to encoder co-design (spatially-coupled), clean-signal only.

### Post-queue (on "continue")
- Verified the shipped DETERMINISTIC-RESONATOR.md tables transcribe the raw sweep
  output cell-for-cell (no hand-copy error). Both D=1024 and the D-sweep cells match.
- Discoverability/citability: added a "Deterministic Quantized Substrate (validated)"
  section to docs/RESEARCH.md pointing to DETERMINISTIC-RESONATOR.md, naming the
  `PhaseResonator`/`phase_resonator_factorize` API, and correcting the resonator
  attribution by addition (Frady/Kent/Sommer 2020 is the primary source; the existing
  "Kanerva 2022" line left intact). Docs-only; API names + repro commands verified.
