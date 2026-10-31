<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Releasing WARM

The engine and binding tags produce artefacts, provenance attestations and SBOMs. MCP launchers
have their own version and release tag. Nothing publishes from a branch, a pull request or a manual
run.

| Tag | Workflow | Produces |
| --- | --- | --- |
| `vX.Y.Z` | `.github/workflows/release.yml` | the crate, 8 npm native packages and the root npm package, Python wheels and sdist (as release assets), SBOMs, SLSA provenance, a draft GitHub release; then publishes the crate and the npm packages |
| `py-models-vX.Y.Z` | `.github/workflows/warm-models-release.yml` | the pinned default offline `warm-models` pack, published to PyPI through Trusted Publishing |
| `py-vX.Y.Z` | `.github/workflows/warm-py-release.yml` | `warm-py` wheels and sdist, published to PyPI through Trusted Publishing |
| `sdk-vX.Y.Z` | `.github/workflows/warm-sdk-release.yml` | `warm-sdk` wheel and sdist, published to PyPI through Trusted Publishing |
| `mcp-vX.Y.Z` | `.github/workflows/mcp-release.yml` | five `warm-mcp` binaries, checksum tables embedded in npm and PyPI launchers, a GitHub release, then the launcher packages and official MCP Registry entry |

Publish the model pack before the native wheel that offers it through `[embed]`. The Python
distributions have separate release tags and Trusted Publisher workflows. For a full engine release,
push both `vX.Y.Z` and `py-vX.Y.Z`; the native wheels build twice.

## MCP launcher release

`writerslogic-warm-mcp` on npm and PyPI is versioned separately from the engine. Its launchers
download `warm-mcp` from a `mcp-vX.Y.Z` GitHub release and verify its hash against a table inside
the package. The release workflow builds macOS arm64 and x64, Linux arm64 and x64 (glibc 2.17
minimum), and Windows x64 binaries. It rejects missing or extra assets, tests both launchers with a
local asset server, packs and installs both distributions, creates the GitHub release, then publishes
to npm, PyPI and the official MCP Registry. The Registry entry uses both packages, and the package
metadata proves ownership of `io.github.writerslogic/warm-mcp`.

Before the first release, add a **pending Trusted Publisher** for the new project at
`https://pypi.org/manage/account/publishing/`: project `writerslogic-warm-mcp`, owner
`writerslogic`, repository `writerslogic-warm`, workflow `warm-mcp-release.yml`, environment `pypi`.
PyPI creates the project on the first successful upload. The repository's `NPM_TOKEN` secret must
have publish access to `writerslogic-warm-mcp`.

To check the pipeline without publishing, run **MCP launcher release** with `workflow_dispatch` or
open a pull request touching its files. To publish, tag the commit containing the tested workflow
with `mcp-vX.Y.Z` where X.Y.Z matches the three launcher version strings, and push the tag. The
workflow never replaces an existing release or registry version. If one publish job fails after the
GitHub release exists, correct its credentials or publisher configuration and rerun that job only.

After the official Registry entry is active, submit the repository through Glama's **Add MCP
Server** form. `glama.json` names the maintainer, as in `scrivener-mcp`. Verify third-party
directories separately; an official Registry entry does not confirm their listings. At the 0.1.2
release, [mcp.so](https://mcp.so/Submit?type=server) has a free submission path and an optional
$39 paid publication option. Use the free path; do not pay. PulseMCP is not accepting submissions.
The [modelcontextprotocol/servers repository](https://github.com/modelcontextprotocol/servers)
now contains reference servers and directs community server discovery to the official Registry.

After Glama approves the listing, a signed-in maintainer must [configure its Docker release](https://glama.ai/mcp/servers/writerslogic/warm/admin/dockerfile).
That form generates a Dockerfile from its fields; the repository-root `Dockerfile` is built and
tested separately by `mcp-container.yml`. Clear the **Pinned Commit SHA** (`5250df9` points to a
commit before the repository Dockerfile and current tool definitions); leave it empty so Glama can
sync newer commits. Pinning a SHA requires updating it for every subsequent release.

Set **Base Image** to `debian:trixie-slim` and **Node.js version** to `24`. Glama installs Node and
`mcp-proxy` in its generated image. Enter this **Build Step** after publishing the MCP launcher to
npm:

```sh
npm install -g --no-audit --no-fund writerslogic-warm-mcp@latest && WARM_MCP_CACHE_DIR=/tmp/warm-mcp-cache writerslogic-warm-mcp --version && install -m 755 /tmp/warm-mcp-cache/*/*/warm-mcp /usr/local/bin/warm-mcp && install -d -m 1777 /data
```

The npm launcher verifies its downloaded binary against the checksum embedded in that release;
the build copies the verified binary into the image so it needs no download at runtime. Set **CMD
Arguments** to `["mcp-proxy","--","/usr/local/bin/warm-mcp","--data-dir","/data"]`, **Environment
Variables JSON Schema** to `{"type":"object","properties":{}}`, and **Placeholder Parameters**
to `{}`. Persistent memory on a hosted deployment requires Glama's optional `/data` volume.

Click Deploy; after its build test succeeds, choose Make Release and publish the MCP launcher
version. The build step picks up npm's latest published launcher on each build. Glama requires a
new build test and release publication for each version. A release enables hosted deployment and
security scans; a GitHub tag alone does not.

Then check the [server score](https://glama.ai/mcp/servers/writerslogic/warm)
and confirm it can deploy and introspect all eight tools. Review every tool's six Tool Definition
Quality Score dimensions, the server coherence score, and the listed security and maintenance
findings. Use Glama's specific findings to improve any dimension below 5/5, any overall score below
the highest tier, or any reported risk; then confirm the next scan reflects the fix. The
`mcp-container.yml` CI check covers the Docker build and stdio introspection before submission.

Afterward, confirm `npm view writerslogic-warm-mcp version` and the PyPI project page show X.Y.Z.
Run `uvx writerslogic-warm-mcp --version` and `npx -y writerslogic-warm-mcp --version` on a
supported platform with an empty launcher cache; these commands print the underlying engine's
version, which is independent of the launcher version.

## What is built

- **Python** (`python-wheels.yml`): abi3 (`cp39-abi3`) wheels for manylinux2014 x86_64 and aarch64,
  musllinux_1_2 x86_64 and aarch64, macOS arm64 and x86_64, Windows x64, and the sdist. Each wheel is
  installed into an empty virtualenv with no source tree and no network, then a real associative
  search and a resonator factorization run through it (`scripts/ci/smoke_python.py`). Linux wheels
  are additionally installed in a Debian 11 (glibc 2.31, Python 3.9) or Alpine container.
- **npm** (`npm-packages.yml`): `writerslogic-warm-{darwin-arm64, darwin-x64, linux-arm64-gnu,
  linux-arm64-musl, linux-x64-gnu, linux-x64-musl, win32-arm64-msvc, win32-x64-msvc}` plus the
  JavaScript-only root `writerslogic-warm`, whose `optionalDependencies` select the binding by
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
   for `writerslogic-warm`. The job uses `rust-lang/crates-io-auth-action` and holds no token.
3. npm: provide an automation token for the `writerslogic-warm` and
   `writerslogic-warm-*` names as `NPM_TOKEN` (a repository or `npm` environment secret).
   The first publish of the eight platform package names needs a token. Publishing uses
   `--provenance`. The token must also permit `npm deprecate` until the 0.3.0 listing is corrected;
   npm Trusted Publishing alone does not authorize that command.
4. PyPI: add pending Trusted Publishers under owner `writerslogic`, repository
   `warm`, environment `pypi`: `warm-models` uses
   `warm-models-release.yml`, `warm-py` uses `warm-py-release.yml`, and
   `warm-sdk` uses `warm-sdk-release.yml`. The MCP launcher uses
   `warm-mcp-release.yml` with the same owner, repository and environment. PyPI permits three pending
   projects at a time; publish one before registering the fourth.

## Cut a release

1. **Version.** `uv run python scripts/ci/bump-version.py X.Y.Z` sets Cargo.toml, Cargo.lock,
   package.json with its `optionalDependencies`, and package-lock.json; `scripts/ci/check-versions.py`
   runs in every artefact job and again against the tag, so a mismatch fails the build.
2. **Changelog.** `git cliff -c cliff.toml -o CHANGELOG.md`, committed as `chore: update changelog`.
   Tag the commit before any `[skip ci]` commit (those run no workflows).
3. **Dry run.** Merge to `main`, then run Actions, Release and Python native release on `main`.
   Run Python models release and Python SDK release when those packages change. The dry runs build
   and test without publishing. The `Reproducible build` workflow must also be green on that commit.
4. **Tag and push** (signed): `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`, and the same
   for `py-vX.Y.Z`. Publish `py-models-vX.Y.Z` first if the native `[embed]` extra needs a new model
   pack. Publish `sdk-vX.Y.Z` when the SDK changes. Approve the `release`, `crates-io`, `npm` and
   `pypi` environments when asked.
5. **GitHub release.** `release.yml` leaves a *draft* release holding every artefact, SBOM,
   `SHA256SUMS` and the provenance bundle. Write the notes, then publish it. The data releases
   (`evidence-v*`) use other tags and never trigger these workflows.
6. **npm listing.** The protected `correct-npm-deprecation` job runs after `publish-npm` and
   replaces the incorrect AGPL message on `0.3.0` while keeping that version deprecated. Verify:

   ```sh
   npm view writerslogic-warm@0.3.0 deprecated       # accurate upgrade message
   npm view writerslogic-warm@X.Y.Z deprecated       # must print nothing
   npm dist-tag ls writerslogic-warm                 # latest must be X.Y.Z
   ```

   If the correction job fails after publication, fix its token and rerun that job. Then confirm
   the latest registry page shows no deprecation banner, the README logo renders, and
   `npm install writerslogic-warm` resolves to X.Y.Z. npm can acknowledge a new package before
   its version is available; a [`0.0.0-stage` placeholder](https://docs.npmjs.com/staged-publishing/)
   alone is not a published binding.
   The `verify-published` job waits for the root and every declared optional platform package
   with a fresh npm cache, then installs the published package and runs a search.
7. **Verify** what was shipped (see below).

## Verify an artefact

```sh
gh attestation verify writerslogic_warm_py-X.Y.Z-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
  --repo writerslogic/warm
gh attestation verify writerslogic-warm-linux-x64-gnu-X.Y.Z.tgz --repo writerslogic/warm
npm audit signatures            # in a project that depends on writerslogic-warm
sha256sum -c SHA256SUMS         # next to the downloaded release assets
```

## Reproducible builds

`.github/workflows/reproducible.yml` builds `warm-eval`, `warm-server`, the napi `cdylib`, the Python
wheel and the crate on two independent runners from different checkout paths and fails unless the
SHA-256 of every artefact is identical (`scripts/ci/repro-build.sh`, `scripts/ci/compare-hashes.py`).
Inputs that are pinned: the committed `Cargo.lock` (`--locked`), `SOURCE_DATE_EPOCH` (the commit
time), `maturin`, no incremental compilation, and `--remap-path-prefix` for the checkout, cargo home
and rustup home. The Rust toolchain is the runner's current `stable`, so both legs must run in the
same workflow run.

Result of the first runs (two runners, different checkout directories): `warm-eval`, `warm-server`,
the napi `cdylib` and the crate are bit-for-bit identical. The Python wheel is identical member for
member except `*.dist-info/sboms/*` (and `RECORD`, which lists its hash), the SBOM maturin embeds, whose `bom-ref`s contain the absolute
checkout directory; the workflow therefore allows the wheel hash to differ and fails if any other
member differs.

| Artefact | Reproducible | Reason |
| --- | --- | --- |
| `warm-eval`, `warm-server`, napi `cdylib`, `.crate` | yes | |
| Python wheel (manylinux x86_64, built natively) | all members except the embedded SBOM | absolute checkout path in the SBOM bom-refs |

Not covered: the per-platform npm bindings and the cross-compiled wheels, which depend on the zig
and napi cross toolchains downloaded at build time (their SHA-256 are in `SHA256SUMS` and the
attestations, not compared).

## If something goes wrong

- A publish job failed before publishing: fix and re-run the failed job; every publish is
  idempotent (`cargo publish` and `npm publish` refuse a version that exists, PyPI uses
  `skip-existing`).
- A tag's crate upload failed because the archive is too large: fix the `Cargo.toml` package
  contents on `main` and verify `cargo package --locked --no-default-features` reports less than
  9.5 MB compressed. The `crates-io` environment normally permits only `v*` tags; temporarily
  add `main` as an allowed deployment branch while retaining the `v*` policy and required reviewer.
  Run `Release` on `main` with `publish_crate_only=true` and approve the `crates-io` environment.
  Remove the temporary `main` policy after the job finishes. This protected run packages,
  attests, and publishes only the crate.
  Verify that the crate index checksum matches the `crate-recovery` artifact and the registry
  download. If the GitHub release is still a draft, replace its crate, crate SBOM, and matching
  `SHA256SUMS` entries before publishing. If it is already immutable, leave those assets intact;
  record the recovery commit, registry checksum, and attestation in the editable release notes so
  readers can distinguish the published crate from the original oversized release asset. Never
  move the signed release tag.
- Published a bad version: crates.io `cargo yank --version X.Y.Z`; npm `npm deprecate
  writerslogic-warm@X.Y.Z "reason"` (unpublish only within 72 hours); PyPI yank in the web UI (no
  API). Then release X.Y.(Z+1); never re-tag.
- A tag pushed by mistake: delete the draft release first, then the tag, before approving any
  environment.
