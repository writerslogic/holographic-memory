<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Technical audit remediation

The user authorized implementation, verification, merging and pushing all audit defect and architectural fixes to `main` on 29 October 2026. The historical assessment remains in REPORT.md; this file records implementation and acceptance evidence.

PR [#334](https://github.com/writerslogic/holographic-memory/pull/334) merged to `main` as
`0c075265c27c21373bf5a43aad5c951abcf366d3`. Its final head had 152 successful checks,
16 neutral/skipped entries, and no pending or failed checks; GitHub verified the merge signature.

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

## Additional bounded concerns

Follow-up to REPORT.md §IV: [PR #335](https://github.com/writerslogic/holographic-memory/pull/335),
based on the merged PR above. Its final-head checks and merge record are the authoritative
integrated gate; this section records implementation and local evidence.

| Concern | Resolution and acceptance evidence |
|---|---|
| Sidecar effects after commit | Retain the public typed committed outcome and retained-key HTTP replay. Add an actual failed audit-sync create/delete/reopen regression. Explicitly document that reopening does not reconstruct missing sidecar effects, that the replay cache is process-local, and that external repair requires caller-owned durable effect IDs. No automatic outbox or exactly-once delivery claim. |
| Import resume/synchronization | Existing content-bound prefix checks and owned split replacement remain. Reports now declare observed-stream/source-managed snapshot semantics and retention of source-deleted records. A completed rerun regression proves updates do not accidentally prune absent source records. Identity/core importer suite: 42 pass against the merged baseline's local HTTP binary. |
| Local pack downloads | Retain unique temporary ownership and per-cache-file single flight. Add local HTTP corruption/retry/concurrency coverage of the actual downloader and cache publication; no model registry or model-weight download. |
| Embedding resource/fidelity contract | Add positive configurable hard text/byte/token/output/concurrent-call limits, checked allocation sizes and fail-fast admission. Node rejects numeric coercion/wrap inputs. Rust/Node/Python report both UTF-8 byte and token truncation in input order. Checked sparse encoding propagates failures before durable writes; core batches share one admission instead of exhausting their own call limit. The soft per-forward budget and limits' scope are explicit; numerical agreement is not downstream task equivalence. |
| Encoder stability | Add opt-in `multiscale-words-v2`, resolving majority ties by frequency descending then coordinate ascending, with independent reference selection, boundary/cache parity, explicit source reencoding and a new golden store. Keep default V1 and all existing goldens unchanged. No claim that legacy equal-key selection is portable across future toolchains. |
| Privacy/erasure claims | Preserve the existing narrow masking and bundling-only DP threat models. Explicitly distinguish signed engine-operation claims from universal physical erasure; independent cryptographic/finite-precision DP certification remains unverified. |

The follow-up's exact-head CI gate must pass before publication. Shared binding checks use
deterministic two-layer random-weight
fixtures, not downloaded pretrained weights or a claim of model quality. Caller-owned inputs,
foreign-runtime copies, separately loaded models, hardware power loss and retained results are
outside a Rust embedder call's allocation limits.

Local checks pass: embedding25, encoder8, importer42, committed-sidecar1, encoder/migration7,
golden-format6, ingest2, and checked embedding-store7. All-feature/all-target Clippy passes with
Rust1.99; strict MkDocs and workflow lint pass. The final binding/runtime and platform checks
are recorded by the PR's exact-head gate. See [bounded-validation.json](bounded-validation.json)
and its retained passing logs. The source commits are `6cd497c` and `107f0e86`; later evidence-only
commits do not claim additional implementation or performance gains.

## Audit log integrity follow-up

The October 31 follow-up reproduced two audit-sidecar defects on the merged WARM
baseline: unknown operation codes silently disappeared from query results, and a
partial append misaligned subsequent records until reopening. Both new regressions
failed against the old implementation. Reads now reject invalid complete records
with their byte offset even outside the timestamp filter. Opening and appending
share a same-handle torn-tail repair; appends serialize repair and writing, sync
the truncation, and stop if repair fails. Complete records are never discarded.

Eight focused audit tests pass, including operation corruption at read-buffer
boundaries, one-byte and 104/105/106-byte interrupted tails, read-only repair
failure, concurrent append preservation and reopen parity. The security-enabled
library suite passes 672 tests with 12 existing ignores. Reproduce with Rust 1.99:

```sh
cargo test --locked --lib core::audit::tests
cargo test --locked --features security --lib
cargo test --locked --features security --test allocation_bounds --test format_crash --test committed_sidecars --test format_golden
cargo clippy --locked --features security --lib --tests -- -D warnings
```

The follow-up PR's checks record the final platform gate. This does not add a
durable outbox, reconstruct missing sidecar events, verify signatures on read,
or establish physical power-loss behavior. The audit format and existing typed
committed-effect contract remain unchanged; `docs/SECURITY.md` defines the reader
and recovery guarantees.
