# WARM migration

WARM means **WritersLogic Adaptive Retention & Memory**. Repository: `writerslogic/warm`; product: `https://writerslogic.com/warm/`; documentation: `https://writerslogic.com/warm/docs/`.

## Scope and names

The user authorized the rebrand, publication and deployment of existing edits, and retirement of the legacy registry packages. Backward compatibility is not a release requirement. Customer data, cryptographic identities, billing prices and object IDs remain protected. No paid inference or benchmark runs were used.

| Surface | Distribution / command | Import or implementation identity |
|---|---|---|
| Rust | `warm-rust` | `writerslogic_warm` library, explicitly named to preserve binding linkage |
| npm | `writerslogic-warm` and eight platform packages | `WarmMemorySystem` |
| Native Python | `warm-python` | `writerslogic_warm` |
| Python HTTP SDK | `warm-sdk` | `writerslogic_warm_sdk` |
| Python model packs | `warm-models` | `writerslogic_warm_models` |
| Python MCP launcher | `warm-mcp` | `writerslogic_warm_mcp` |
| npm MCP launcher | `writerslogic-warm-mcp` | downloads `warm-mcp` |
| CLI | `warm-server`, `warm-mcp`, `warm-admin`, `warm-embed`, `warm-format`, `warm-eval` | Rust executables |
| Benchmark tooling | `warmbench` | historical result keys remain unchanged |

`scripts/rename-warm.zsh` invokes the bounded Python renamer. `rename-plan.json` and `rename-applied.json` record the initial transformation and hashes; subsequent fixes and naming decisions are in Git history. Original bytes remain locally in `.git/warm-migration-backup/`. Do not reapply the initial transform to the final source.

## Preserved identities

Stored file magic, lock/key filenames, signed credential/context URIs, embedding schemas, model formats and reserved `hms.`/`hms:` metadata retain their original bytes. Historical measurements remain attributed to their original snapshots. Physical Cloudflare Worker/D1/R2 names and Stripe object IDs remain stable. Public service names and copy are WARM; canonical cloud is `warm-cloud.writerslogic.com`, billing is `/api/warm/`. Existing stores do not silently move to empty directories.

The immutable JSON-LD context SHA256 is `ec2dbc721f02dc1f06a7d03f589c5a7e43b5fa0ff8184d6a384516b774790b45`.

## Verified validation

- Rust library: 799 passed, 12 ignored; MCP: 40 passed; server: 54 passed. Subsequent server-abuse/store-lifecycle integration checks: 5 passed; reliability: 10 passed. All-target/all-feature Clippy passed with warnings denied.
- Native Python: 119 passed; native JavaScript adapters: 139 passed; Node package: 10 passed, 3 intentional skips. Python SDK/adapters, C, Go, Java, .NET, Swift and JVM checks passed.
- Docs generators: 19 passed; renamer: 3 passed; benchmark tooling: 43 passed, 20 expected skips. Raw benchmark results were not rewritten.
- Docs CI `41165031767`: accessibility 36 views without serious/critical violations; search 46/46; mobile Lighthouse home 99 and concepts 86, desktop 100/99, other categories 100. Original thresholds retained. Navigation pruning reduces the concepts DOM by 42%.
- Public registry installs verified npm search/signatures/provenance, Rust import/vector similarity, native Python 16/16 searches and resonator factorization, SDK imports, and model-pack checksum `6a1cc24734a57e4524ecdab1466b2c14535a059b393c76c418ca458ec29602dd`.

Local detailed logs are under `target/warm-migration/`; the committed workflow runs and registry provenance provide durable release evidence.

## Publication and deployments

Published and verified: `warm-rust` 0.6.4, npm core/platform packages 0.6.4, `warm-python` 0.6.4, `warm-sdk` 0.1.3, `warm-models` 0.1.0. Source 0.6.5 corrects the MCP tool description to derive its reserved prefix from the actual implementation. Rust 0.6.5 and MCP launcher 0.1.5 are the final release work in progress.

Python Trusted Publisher filenames: `warm-py-release.yml`, `warm-sdk-release.yml`, `warm-models-release.yml`, `warm-mcp-release.yml`; owner `writerslogic`, repository `warm`, environment `pypi`. Rust Trusted Publishing uses `release.yml` and environment `crates-io`; configuration was created through the registry API and recorded in `rust-trusted-publisher.json`.

Main website `21a8a874` deployed as `4d45ad0f`; legacy site `39bfc6b4`; hosted cloud `41b5ea26`. Hosted health/transparency and billing portal checks passed. Checkout remains intentionally disabled as it was before migration. Stripe product branding changed without modifying prices, subscriptions or object IDs. Hosted evidence lives in the cloud repository's operations and acceptance records.

The old product site uses HTTP 301 redirects. The old documentation hostname uses GitHub Pages HTML/JavaScript redirects because DNS-write authorization was unavailable. Docs deployment `41165031767` verified old latest/versioned paths, query strings and anchors; signed context responses remain exact bytes. Canonical docs deployment guards require the complete reference/search payload. Final active-doc changes are being restaged with the website.

## Legacy registry retirement

The user deleted both `holographic-memory` and the superseded `writerslogic-warm` crates; public endpoints returned 404. The supported Rust name is `warm-rust`.

npm automation-token deletion was rejected with 403; interactive authenticated/2FA deletion of `holographic-memory` was also rejected with 422. The legacy core had 579 downloads and MCP 770 downloads in the registry's measured week, exceeding npm's less-than-300 rule for packages older than 72 hours. Core/platform versions have migration deprecations; MCP deprecation follows its replacement release. Permanent deletion remains subject to registry policy. Exact evidence is in `npm-retirement-policy.json`; no new WARM package is targeted by the retirement workflow.
