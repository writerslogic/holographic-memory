<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Format work handoff

Done: spec and `docs/STABILITY.md` (PR #119, merged); `core::format` (constants, offline
verifier, upgrade registry), `hms-format` binary, golden stores and `tests/format_{golden,layout,crash}.rs`,
`scripts/format-gate.sh` + `.github/workflows/format.yml`, fuzz target `format_readers`
(needs the `[[bin]]` stanza in `fuzz/Cargo.toml`, added here), fixes for three crash windows
(`indices.json` sidecar, torn `audit.bin` entry, non-atomic `encryption.salt`).

Not done:
- Real-binary tests: run `hms-admin reencode`, `hms-format verify/upgrade`, and `hms-server`
  (tenant dirs) against each other (`tests/format_binaries.rs`); `upgrade` is exercised only by
  its unit test, not on v2/v3 goldens yet.
- Run `cargo +nightly fuzz run format_readers` and add a corpus.
- Re-read `origin/main` for the embeddings agent's new encoders: add each to
  `docs/file-format/ENCODERS.md` and commit a golden (`HMS_GOLDEN_WRITE=1`, see
  `tests/golden/README` notes in `tests/format_golden.rs`).
- Purge agent changes to `documents.text` truncation: update `docs/file-format/CACHES.md`.
- `feat/c-abi` exists: update the "no C ABI" row of `docs/STABILITY.md` when it lands.
- Slim `torn_append_at_every_byte` (5 min in debug).

Next command: `HMS_GOLDEN_WRITE=1 cargo test --features security,experimental --test format_golden -- --ignored generate_goldens`
after adding a spec to `specs()`.
