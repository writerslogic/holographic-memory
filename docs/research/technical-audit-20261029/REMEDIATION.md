<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Technical audit remediation

The user authorized implementation, verification, merging and pushing all audit defect and architectural fixes to `main` on 29 October 2026. The historical assessment remains in REPORT.md; this file records implementation and acceptance evidence.

Constraints: no paid API/cloud credits, model downloads or live benchmarks; use local/free checks. Preserve unrelated work. Use signed, DCO-signed-off commits and the repository PR/check workflow. Do not claim speculative research or certification as delivered implementation.

Baseline: `de4a2fd6` preserves the 14 pre-existing audited storage/service fixes. Its exact prior working tree passed 894 focused Rust tests, strict TypeScript checks and version checks. Root integration branch is `fix/technical-audit-20261029` in `/Volumes/A/holographic-memory`.

| Findings | Owner and isolated branch | State | Acceptance |
|---|---|---|---|
| D01, D03, D14; importer resume, SDK fidelity | boundaries / `fix/audit-boundaries` | Integrated; focused acceptance checks pass | Checked native allocations; collision-free versioned identities; content-bound resume; canonical snapshot containment; retryable failed saves. |
| D05, D06, D09, D15, D16, D17; embed limits/downloads | identity / `fix/audit-identity` | Integrated | Monotonic issuance; actual embedding-function fingerprint; strict COSE; typed valid benchmark evidence/clustered uncertainty; correct release SBOM graph. |
| D07, D08, D10, D12; analytics budgets | retrieval / `fix/audit-retrieval` | Integrated | Canonical exact/scoring oracles; checked allocation/numeric bounds; composed filtered execution; explicit bounded analytics. |
| D02, D04, D11, D13; typed contracts/effect recovery | coordinator + retrieval/identity isolated branches | Integrated; complete CI gate required | Recoverable restore; revision-conditional policy writes; coherent governance and sidecar outcomes; bounded MCP resources; useful generated API contracts. |
| R18 native offline workflow validation | coordinator | Rust/HTTP/Node pass, including restart and quote checks | Native ingest/recall/supersede/forget/restart/source checks using frozen local fixtures; no paid evaluation. |
| Final integration and main publication | coordinator | Required PR checks and merge record are authoritative | Required default/all-feature lint/tests, MSRV, generated contracts, affected bindings/interop/package checks; resolve PR checks and merge; verify origin/main contains all fixes. |

Shared decisions: model identity keeps the public EmbeddingSpace shape and uses a versioned effective-function fingerprint in revision; changed nominal managed-model stores require explicit reembedding, never relabeling. Memory policy work uses optimistic revision-conditional core commits rather than holding a mutex across policy callbacks or I/O. Resource exhaustion must be explicit; no silently partial results. Agent builds use private worktree target directories.

## Verified integration milestones

- `ac184197`, `47b1cd2d`: core revision-conditional policy commits, bounded coherent result assembly, policy callbacks outside the memory-index lock. Multiple-wrapper/raw-writer, stale forgetting/consolidation and same-wrapper callback reentry regressions pass.
- `df830388`: parent tenant ownership lease and durable restore intent; process-exit tests at intent/rename/commit boundaries recover a complete old or committed replacement store. Imported record writes/deletes require matching typed ownership under the service mutation lease.
- `d6075a16`: committed-outcome errors propagate through service cleanup; staged vector rows are retained if the core mutation is already durable. Forced post-publication index-persistence failure verifies typed outcome and reopen.
- Focused integrated Rust run: **763 library + 19 MCP + 51 HTTP unit tests pass**, 12 existing development ignores; after the callback/result-assembly follow-up, all 8 memory-policy regressions pass.
- Frozen synthetic lifecycle `tests/fixtures/native_memory_lifecycle_v1.json`: **19 cases over three fresh opens**. Rust and real HTTP verify source quote/URI fidelity, idempotence/conflict, temporal history, unsupported-query emptiness, forgetting and restart. This verifies native plumbing and fixture outcomes; it is not a learned-model answer-quality or held-out benchmark result.
- Independent Draft 2020-12 checker: **12 actual HTTP memory request/reply pairs + 8 invalid-shape checks pass**. Strict MkDocs build and 18 documentation-generator unit tests pass.

## Verification limits retained

No paid API/cloud execution, model-weight downloads, held-out live evaluation or claimed performance improvement. Process-interruption checks do not establish hardware power-loss behavior. Reasoning byte budgets are accounted allocations, not RSS measurements. SCITT receipt bytes remain explicitly unverified without deployment-specific inclusion/trust policy. General VC/cryptographic certification and cross-toolchain legacy encoder-tie stability remain unverified. The historical 6.2/10 score is not recalculated from implementation effort or test counts.

## Durable ownership and format compatibility

Governance uses bounded snapshot/shadow planning and a revision-conditional journal transaction.
Distinct live triple owners survive duplicate-vector maintenance; atom/composite deletes are
scoped so raw-vector aliases are preserved. Refinement rebinds owned composites; live state is
checked against reopen/compaction. Schema 5 introduces these records with an appended golden
store and an explicit `hms-format` schema upgrade. Existing stores keep writing their original
format until upgraded. HTTP MCP now has connection/header/body/stream, queue, session, lifetime,
request-count and retained-message budgets; session retirement preserves SDK request association.

Component validation records: [boundaries-validation.json](boundaries-validation.json),
[identity-validation.json](identity-validation.json), [retrieval-validation.json](retrieval-validation.json).
These preserve completed scoped checks and explicitly identify checks still pending integration.
Final CI results apply to the integrated PR head; they must pass before publication to main.

The native addon build now targets the Rust library rather than also compiling every research
binary. Committed-effect HTTP failures remain replayable within the idempotency cache's TTL
and capacity, so a retained key does not repeat an already durable mutation.

## Final integrated local checks and PR

[PR #334](https://github.com/writerslogic/holographic-memory/pull/334) is the authoritative final
platform/feature/fuzz gate. Fresh integrated local checks pass: governance6, MCP28, native Node9
(including async/sync governance and shared19-case lifecycle), packed CommonJS/ESM distribution,
TypeScript, committed-failure replay1 and provenance22 after the atomic compatibility change.
The complete required PR checks must pass before merge; component artifacts retain the earlier
scoped source state rather than silently treating partial runs as completed final gates.

First PR validation identified Rust1.99's deprecation of atomic fetch_update and a missing restore
module in the server fuzz driver. The issuance path uses a checked compare-exchange loop to
preserve the MSRV; the fuzz harness now compiles the shipped restore module. Six Python static
review findings were corrected without suppressions;51 focused tests pass.

The complete all-feature gate also retained the existing tenant-directory isolation assertions:
server ownership files moved to the dedicated `tenant-locks/` control directory rather than
relaxing those assertions or mixing non-tenant entries into `tenants/`.

The subsequent full platform run cleared the earlier namespace and fuzz-driver failures.
Latest Rust Clippy identified the large native transport-error variant; boxing that failure-only
payload reduces the enum while preserving its error context (no lint suppression). Scoped MCP
Clippy passes locally. Stdio now honors the configured request-byte cap through the SDK reader,
whose prior default was16MiB; two boundary/recovery/cancellation regressions pass.

Final local `cargo +1.99.0 clippy --all-targets --all-features -- -D warnings` passes. The last
all-target warning was an obsolete local dimension binding after the model-space identity
change; it was removed after confirming its use is now supplied by `Embedder::space()`.
A fault-injection regression verifies both document deletion and Memory forgetting advance
HTTP cache generations when data is durable but purge/index persistence fails. Conditional
reads return the changed state instead of304. The same test passes for both route families.
