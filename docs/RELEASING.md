<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Releasing HMS

One tag produces every artefact, a provenance attestation and an SBOM for each of them, and (after
approval) the publishes. Nothing publishes from a branch, a pull request or a manual run.

| Tag | Workflow | Produces |
| --- | --- | --- |
| `vX.Y.Z` | `.github/workflows/release.yml` | the crate, 8 npm native packages and the root npm package, Python wheels and sdist (as release assets), SBOMs, SLSA provenance, a draft GitHub release; then publishes the crate and the npm packages |
| `py-vX.Y.Z` | `.github/workflows/publish-pypi.yml` | the same wheels and sdist plus `holographic-sdk`, published to PyPI through Trusted Publishing |

Push both tags for a full release (the wheels build twice; PyPI trusted publishing is bound to
`publish-pypi.yml`, so it cannot move into `release.yml`).

## What is built

- **Python** (`python-wheels.yml`): abi3 (`cp39-abi3`) wheels for manylinux2014 x86_64 and aarch64,
  musllinux_1_2 x86_64 and aarch64, macOS arm64 and x86_64, Windows x64, and the sdist. Each wheel is
  installed into an empty virtualenv with no source tree and no network, then a real associative
  search and a resonator factorization run through it (`scripts/ci/smoke_python.py`). Linux wheels
  are additionally installed in a Debian 11 (glibc 2.31, Python 3.9) or Alpine container.
- **npm** (`npm-packages.yml`): `holographic-memory-{darwin-arm64, darwin-x64, linux-arm64-gnu,
  linux-arm64-musl, linux-x64-gnu, linux-x64-musl, win32-arm64-msvc, win32-x64-msvc}` plus the
  JavaScript-only root `holographic-memory`, whose `optionalDependencies` select the binding by
  os, cpu and libc. glibc bindings are linked against glibc 2.17, musl bindings with zig. The
  tarballs are installed into an empty project on a machine of every target (Alpine containers for
  musl, Debian 11 for glibc) with the root's optional dependencies pointed at the local tarballs,
  so npm itself must pick exactly one package, and a document search must succeed
  (`scripts/ci/smoke-node.cjs`). The root `package.json` (`napi.targets`) is the single list of
  platforms and `scripts/ci/npm-platform-packages.cjs` refuses to pack if it disagrees with
  `optionalDependencies` or any binding is missing.
- **Crate**: `cargo package --locked`, which also builds the packaged tarball.
- **SBOM**: a CycloneDX 1.5 JSON per artefact (`cargo-cyclonedx`, `npm sbom`), named
  `<artefact>.cdx.json` and paired by `scripts/ci/pair-sboms.py`, which refuses a release where any
  artefact lacks one.
- **Provenance**: `actions/attest-build-provenance` signs a SLSA provenance statement for every
  artefact and SBOM, and `actions/attest-sbom` binds each SBOM to its artefact. The bundle is also a
  release asset (`provenance.intoto.jsonl`).

## One-time setup (repository settings, by the maintainer)

These cannot be committed; the publish jobs wait on them.

1. Environments (Settings, Environments), each with required reviewers: `release`, `crates-io`,
   `npm`, `pypi`.
2. crates.io: add this repository's `release.yml` and environment `crates-io` as a Trusted Publisher
   for `holographic-memory`. The job uses `rust-lang/crates-io-auth-action` and holds no token.
3. npm: put an automation token for the `holographic-memory` and `holographic-memory-*` names in the
   `npm` environment as `NPM_TOKEN`. The first publish of the seven new platform package names
   needs a token; afterwards configure Trusted Publishing for every package on npmjs.com and delete
   the secret. Publishing uses `--provenance`.
4. PyPI: the trusted publisher is bound to `publish-pypi.yml` and environment `pypi`. Keep both.

## Cut a release

1. **Version.** `uv run python scripts/ci/bump-version.py X.Y.Z` sets Cargo.toml, Cargo.lock,
   package.json with its `optionalDependencies`, and package-lock.json; `scripts/ci/check-versions.py`
   runs in every artefact job and again against the tag, so a mismatch fails the build.
2. **Changelog.** `git cliff -c cliff.toml -o CHANGELOG.md`, committed as `chore: update changelog`.
   Tag the commit before any `[skip ci]` commit (those run no workflows).
3. **Dry run.** Merge to `main`, then run Actions, Release, Run workflow on `main`, and run
   `publish-pypi` the same way. Both build, smoke-test, SBOM and attest everything and skip the
   publish jobs. Do not tag until both are green. The `Reproducible build` workflow must also be
   green on that commit.
4. **Tag and push** (signed): `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`, and the same
   for `py-vX.Y.Z`. Approve the `release`, `crates-io`, `npm` and `pypi` environments when asked.
5. **GitHub release.** `release.yml` leaves a *draft* release holding every artefact, SBOM,
   `SHA256SUMS` and the provenance bundle. Write the notes, then publish it. The data releases
   (`evidence-v*`) use other tags and never trigger these workflows.
6. **npm un-deprecation** (required while the package carries the old deprecation; the previous
   `0.3.0` was deprecated):

   ```sh
   npm view holographic-memory deprecated            # the message, if any, that npm shows today
   npm deprecate "holographic-memory@*" ""            # an empty message removes the deprecation
   npm view holographic-memory@X.Y.Z deprecated       # must print nothing
   npm dist-tag ls holographic-memory                 # latest must be X.Y.Z
   ```

   Run it after the npm publish job is green, as an owner of the package (`npm login`, 2FA on).
   Then confirm the registry page shows no deprecation banner and `npm install holographic-memory`
   resolves to X.Y.Z. The `verify-published` job already installs the published packages and runs a
   search; this step is only about the deprecation flag.
7. **Verify** what was shipped (see below).

## Verify an artefact

```sh
gh attestation verify holographic_vsa-X.Y.Z-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
  --repo writerslogic/holographic-memory
gh attestation verify holographic-memory-linux-x64-gnu-X.Y.Z.tgz --repo writerslogic/holographic-memory
npm audit signatures            # in a project that depends on holographic-memory
sha256sum -c SHA256SUMS         # next to the downloaded release assets
```

## Reproducible builds

`.github/workflows/reproducible.yml` builds `hms-eval`, `hms-server`, the napi `cdylib`, the Python
wheel and the crate on two independent runners from different checkout paths and fails unless the
SHA-256 of every artefact is identical (`scripts/ci/repro-build.sh`, `scripts/ci/compare-hashes.py`).
Inputs that are pinned: the committed `Cargo.lock` (`--locked`), `SOURCE_DATE_EPOCH` (the commit
time), `maturin`, no incremental compilation, and `--remap-path-prefix` for the checkout, cargo home
and rustup home. The Rust toolchain is the runner's current `stable`, so both legs must run in the
same workflow run.

Documented exceptions: see the table below, filled from the workflow's output.

| Artefact | Reproducible | Reason |
| --- | --- | --- |
| (set by the first green runs) | | |

Not covered: the per-platform npm bindings and the cross-compiled wheels, which depend on the zig
and napi cross toolchains downloaded at build time (their SHA-256 are in `SHA256SUMS` and the
attestations, not compared).

## If something goes wrong

- A publish job failed before publishing: fix and re-run the failed job; every publish is
  idempotent (`cargo publish` and `npm publish` refuse a version that exists, PyPI uses
  `skip-existing`).
- Published a bad version: crates.io `cargo yank --version X.Y.Z`; npm `npm deprecate
  holographic-memory@X.Y.Z "reason"` (unpublish only within 72 hours); PyPI yank in the web UI (no
  API). Then release X.Y.(Z+1); never re-tag.
- A tag pushed by mistake: delete the draft release first, then the tag, before approving any
  environment.
