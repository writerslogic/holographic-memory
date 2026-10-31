# WARM migration

WARM means **WritersLogic Adaptive Retention & Memory**. Canonical repository: `writerslogic/warm`; product: `https://writerslogic.com/warm`; documentation: `https://writerslogic.com/warm/docs/`.

## Authorized scope

Rename core, SDKs, tooling, documentation, product sites, and hosted service. Include existing working-tree edits in website/service deployments. Preserve customer data, cryptographic identities, billing objects/prices, and backwards-compatible endpoints. No paid inference or benchmark runs. Source publication and deployment, including existing edits, are authorized. Backward compatibility is not a release requirement; stored data and service identities remain protected.

## Execution

`scripts/rename-warm.zsh` invokes the bounded Python renamer. `rename-plan.json` and `rename-applied.json` record every transformed file/path and before/after hashes. Original bytes are retained locally in `.git/warm-migration-backup/`. Subsequent compatibility edits intentionally retain legacy names; do not reapply the bulk transformation to those aliases.

New source package family: `writerslogic-warm` (Rust/npm), `writerslogic-warm-py` (native Python), `writerslogic-warm-sdk` (HTTP Python SDK); native Python/Rust import `writerslogic_warm`. JavaScript framework adapters use `@writerslogic/warm-*`. New executables use `warm-*`. New package manifests do not imply that renamed distributions have already been published.

## Compatibility boundary

Stored file magic, `.hms.lock`, key filenames, credential types/URIs, signed contexts, embedding schema IDs, model pack formats, MCP authorization claims/scopes, and existing customer tokens retain their original bytes. Historical measurements/reports remain attributed to their original commits. Existing Cloudflare Worker/D1/R2 names and Stripe object IDs are stable implementation identifiers; public service names and product copy change to WARM. Existing stores must not silently move to a new empty data directory.

## Validation and deployment

Implementation validation completed: Rust library 799 passed (12 ignored), MCP 40 passed, server 54 passed; native Python 119 passed; native JavaScript adapters 139 passed; Node package 10 passed (3 intentional skips). Python SDK and integration, C ABI, Go, Java, .NET, Swift and JVM checks passed. Docs generator 19 and rename utility 3 tests passed. See local validation logs and hosted operations records for detailed scope.

WARM website, legacy product redirect, cloud services, billing branding and shared webhook have been deployed. Final source publication and GitHub Pages legacy documentation redirects are being completed. Canonical docs are staged with the product site and deployment guards require their presence.

Renamed registry distributions are not published; installation is currently from source. Legacy wire identifiers, signed JSON-LD contexts, physical infrastructure names and historical evidence remain unchanged intentionally. Existing aliases do not block release, but ongoing compatibility is not promised.
