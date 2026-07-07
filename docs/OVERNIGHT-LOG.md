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
