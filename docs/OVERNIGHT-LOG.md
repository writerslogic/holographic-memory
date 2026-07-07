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
