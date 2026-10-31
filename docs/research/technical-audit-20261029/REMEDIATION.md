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
The handle is opened for read/write and seeks to EOF within that same writer lock:
Windows [requires `FILE_WRITE_DATA` for truncation](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntsetinformationfile),
which Rust's append-only opening mode omits. Reopen-and-append checks preserve
the existing prefix without first reading the reopened file.

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

## Production-readiness improvement program

The active October 31 objective is production readiness and demonstrated leading
quality across WARM's features. The historical scorecards remain evidence about
their reviewed revisions; passing more tests alone does not earn a new score or
establish competitive superiority. Implementation and verification proceed in
bounded milestones, with no paid development API calls or model downloads.

| Priority | Capability / shared constraint | Required evidence | State |
|---|---|---|---|
| 1 | HTTP retry safety and retained-memory bounds | Post-commit panic never blindly repeats work; byte/capacity pressure preserves protected keys; expiry reclaims allocations; real HTTP and concurrency regressions | Merged in PR #352 (`f28095e0`); all CI checks pass |
| 2 | Restart-safe committed effects | Transaction-backed effect identities and intents, idempotent delivery, explicit ambiguous outcomes, replay/compaction/restore invariants, process-interruption tests | Merged in PR #353 (`79609861`); provenance and HTTP restart receipts remain separate work |
| 3 | Retrieval quality and workload performance | Reproducible local baselines, correctness-equivalent competitor workloads, update/churn/filter coverage, latency and allocation evidence | Merged in PR #354 (`6a2ec2b7`); verified allocation reduction; historical tail regression and competitive evidence remain open |
| 4 | Security lifecycle and assurance | Signing-key rotation and historical verification policy, adversarial validation, actionable dependency dispositions | Existing limitations in `docs/SECURITY.md`; no certification claim |
| 5 | Interface and standards fidelity | Shared lifecycle/error/serialization checks across HTTP, native bindings, MCP and framework adapters; authoritative applicable requirements | Existing lifecycle fixtures are the baseline; expand where coverage is missing |
| 6 | Operational recovery and observability | Restore generation and retry-state interactions, per-tenant isolation/resource admission, bounded diagnostics and operator acceptance checks | Review before introducing persistent request/effect state |

### HTTP retry safety milestone

The shared write pipeline previously discarded its running key when a worker
panicked, even if that worker had already committed. Count-based cache eviction
could also discard completed keys before TTL, and cached JSON trees had no retained
byte budget. The implementation now reserves uncertain keys for reconciliation,
rejects new keys when protected records fill capacity, and bounds retained response
buffer allocations independently. Replays share serialized bytes; completion-order
expiry reclaims both keys and bytes without expiring live workers. Metrics expose
record count, retained buffer bytes and unreplayable keys.

Acceptance covers actual committed writes followed by panic, response-budget
overflow, concurrent completions, capacity pressure, TTL boundaries, tenant and
fingerprint separation, and real HTTP responses. Final local and CI results will
be recorded on the milestone PR. This is process-local retry protection, not a
durable outbox or an exactly-once guarantee. The remaining restart boundary must
be closed with durable transaction identities, not by persisting a cache and
assuming its write is atomic with the core mutation.

Local validation: 62 server tests pass; the final admission/retention changes pass
all 11 focused limit tests; three real-HTTP idempotency tests and the generated
OpenAPI/security check pass. Strict server/API-test Clippy passes. The initial
unit run preceded the documentation updates and failed the two reference-drift
checks; both pass after the configuration and error references were updated.
The response-buffer accounting includes serialization capacity, not merely JSON
length, but excludes active sends and process-wide allocations. Capacity planning
and the changed full-cache admission behavior are explicit in `docs/SERVER.md`.

### Restart-safe audit delivery: implementation under validation

Scope: make enabled engine audit effects recoverable without reapplying the data
mutation. This does **not** by itself make HTTP requests restart-idempotent, deliver
arbitrary external effects exactly once, or repair past missing audit records.

Verified constraints:

- `engine/mutation.rs::commit_payload_at_revision` appends one durable transaction
  before publishing live state. A pending effect belongs in that same transaction;
  a separately written request cache or sidecar intent has an unsafe crash interval.
- Core revision counters are process-local opaque values, not durable event IDs.
  Use a persisted unique effect ID rather than treating revision as a stable identity.
- Compaction rewrites state, so pending effects must be included in its snapshot
  and tail replay. Its own audit effect must be published with the new generation,
  not appended to the old generation after compaction has already committed.
- Document checkpoints exclude non-document state today. Effect metadata must be
  represented in the journal alongside the checkpoint and accepted by overlay
  replay without hydrating all documents or disabling checkpoints permanently.
- The existing 105-byte audit format has no delivery identity. It cannot deduplicate
  identical timestamp/operation/ID-hash records reliably. Preserve legacy history
  and introduce a versioned effect-aware stream instead of guessing identity from
  payload equality. Old stores must explicitly upgrade before writing new mutations.

Implemented mechanism and mandatory invariants:

1. Introduce schema-gated pending-audit and acknowledgement mutation records. An
   intent carries a bounded stable identity and immutable signed event payload.
   Validate pending count/bytes and identity conflicts under the writer lease
   **before** appending the business mutation. Exhaustion must fail before commit.
2. Keep a bounded FIFO of pending effects in committed journal order. A serialized
   drainer delivers the front effect, syncs the audit stream, then durably acknowledges
   it in the core journal. Stop on any delivery or acknowledgement error. New business
   writes never skip an unresolved delivery and never reapply an old data mutation.
3. Give the new audit stream fixed-size checksummed records and a version marker;
   bind the effect identity into its signed message. Delivery can recognize the
   last record after a crash between delivery and acknowledgement. This is valid
   only with one ordered drainer and no bypass writer: at most one delivered effect
   may lack an acknowledgement. A deduplicated retry must sync again before ack.
4. Preserve the pending FIFO during compaction and restore; retain legacy audit
   history in reads. Capture compaction's event in the generation being published.
   Expose pending/recovery status explicitly. A missing or corrupt sidecar must not
   silently acknowledge effects or reconstruct unsupported historical claims.
5. Replay the original signed bytes after restart, including across signing-key
   changes; do not relabel an old event by re-signing it as a new delivery. A new
   signature-message helper must distinguish legacy records from effect-aware ones.

Acceptance before merge: schema validation and explicit upgrade; new frozen golden
store; current and legacy format verification; default/security/all-feature builds;
crash injection before/after core append, audit write/sync and acknowledgement;
concurrent writes and bounded backlog; compaction with pending/partially delivered
effects; document-checkpoint overlay parity; restart with a repaired sidecar; snapshot
and restore coverage; signature/identity tampering checks. Hardware power-loss and
external-history anchoring remain unverified without separate evidence.

Follow-on design remains necessary for provenance delivery (its chained sequence
must be assigned under the provenance writer's own ordering contract), durable HTTP
receipts for multi-transaction operations, restore incarnation boundaries, and
per-tenant fair admission. Do not claim these are solved by the audit stream.

Implementation notes from the source trace:

- Reuse a separate `mutation_gate::MutationGate::maintenance()` lease for delivery:
  its ticket queue releases its mutex before I/O. Do not hold the core writer lease
  while delivering and then recursively acquire it to acknowledge.
- Add pending/ack variants to `checkpoint_accepts` and exclude them from
  `image.other_state`. Compaction writes pending records alongside the document
  checkpoint; nonempty metadata frames naturally select overlay replay at open.
- Prefer a separate versioned `audit.effects.bin` stream so legacy `audit.bin`
  history requires no destructive conversion. Reads must cover both. Format
  verification, known-file inventory, snapshots, upgrade and goldens must cover
  the new file. A new public signable-message helper must expose the correct
  domain-separated message including effect ID while preserving legacy verification.
- The first durable-delivery scope is the existing audited engine operations:
  vector memorize, delete and compaction. Document/meaning audit coverage needs
  an explicit extension; do not quietly describe that scope as all engine writes.
- Upgrade registry steps currently hardcode schema 5 while their implementation
  writes the current schema. Adding schema 6 requires updating planner descriptions
  and targets as well as the constant, plus a 5-to-6 step and new frozen golden.
- Delivery/ack fault tests must prove that a retried already-written audit record
  is re-synced before acknowledging. A failed core acknowledgement must stop the
  drainer so checking only the audit stream's last effect ID remains sound.

Validation progress (not yet a merge/readiness claim): the initial 36 audit-filtered unit
checks passed, including delivered/unacknowledged replay, failed delivery plus compaction,
concurrent writes, queue saturation, and process exits after core commit, effect write,
sync, and acknowledgement. The expanded run passed 39 checks and exposed a test setup
error: `has_document()` intentionally hydrates a checkpoint. The corrected test asserts
lexical results and mapped serving directly and passes; audit delivery itself preserves the
checkpoint. Additional crash points before commit and after compaction publication, pending
snapshot copy/restore, scanner/upgrade compatibility and final gates remain in progress.

The implementation adds one durable acknowledgement transaction per delivered audit event;
this is extra write/sync cost, not a measured performance improvement. Production validation
must include audit-enabled write latency and recovery throughput. Individual record signatures
do not detect removal of whole records without an external history anchor. No hardware
power-loss, generalized provenance recovery or restart-safe HTTP receipt claim is made.

Pre-merge review found that an empty current journal does not prove a store has no audit
history: compaction with logging disabled may leave only a document checkpoint. Initialization
now durably creates the audit header before publishing `store.json`, and every subsequent
schema-6 open requires the existing stream, including empty and checkpoint-only stores. New
regressions cover both cases and interruption between header creation and schema publication.


### Integrated audit delivery acceptance

PR [#353](https://github.com/writerslogic/warm/pull/353) merged as `79609861` after
all 71 check entries completed without failures or pending work (neutral/skipped
jobs remain distinct from successful tests). The merged tree matches tested head
`5431df4b`. Local final checks include 43 audit-filtered tests, the original-signature
recovery and legacy-upgrade regressions, and 13 initialization/delivery checks with
seven process-interruption phases. Strict security/experimental/server Clippy passed.
The full local golden sweep was stopped under filesystem saturation; the complete
platform/format matrix is established by the final CI gate, not that partial local run.
Durable delivery covers vector memorize/delete and compaction, with the limitations
in the design above; no score increase or generalized exactly-once claim follows.

### Retrieval capture allocations and audit-cost validation

The measured candidate replaces changed-ID String copies with bounded identities of
base-index owners and avoids building slot membership when there is no dense delta.
The index pins those allocations; the first replaced owner remains excluded through
repeated replacement, deletion/recreation and subsequent index publication. Existing
128-document, 1024-chunk and 64 KiB changed-ID limits are unchanged.

Evidence in [retrieval-tail-20261031/validation.json](retrieval-tail-20261031/validation.json)
compares the fixed `f28095e0` baseline and preserved candidate patch over five alternating
rounds, 128 queries per group, a 10,000-document/16-dimensional corpus and 32 updates.
All 5,120 full ordered result comparisons match the exhaustive oracle. Every one of
640 paired updated-hot queries allocates exactly 22 fewer times (178 to 156); peak
requested bytes remain unchanged. Medians of per-round updated-hot total p50/p99
change by -1.27%/-7.29%, but paired timing variability and concurrent host activity
preclude a production latency claim. These data do not resolve the older 32-sample
historical tail regression or establish superiority against another implementation.

Dense tests pass 12 checks; search tests pass 14 with one existing ignore; strict
Clippy and formatting pass. The owner-lifetime regression retains captured queries
through repeated replacement, delete/recreate, retraining and destruction of the
live store. Raw samples, source/binary hashes, logs, host observations, the planned
acceptance criteria and summary script are retained. Reproduce the measured source
hashes without rebuilding with the `source_reproduction_command` in `validation.json`;
`summarize.py` regenerates the numerical summary offline.
The benchmark source stays pinned to its measured revision after integration.

The new `audit_delivery` benchmark separately measures disabled, unsigned and signed
single-writer commits, with reopen/result/signature checks outside the timing region.
All three correctness modes and strict benchmark Clippy pass. The first local timing
attempt was aborted after filesystem contention appeared and Criterion projected a
run beyond the 240-second budget; no timing comparison is claimed. Its exact command,
exit status and logs remain in [audit-cost-20261031/validation.json](audit-cost-20261031/validation.json).
Use the command and limits in `docs/production-readiness.md` on the deployment filesystem;
audit write cost and backlog recovery throughput remain unverified production criteria.
The retrieval/benchmark follow-up requires its own complete PR gate before merge.


### Merged retrieval and security follow-ups

PR [#354](https://github.com/writerslogic/warm/pull/354) merged as `6a2ec2b7` after all
109 check entries completed without failures or pending jobs. The merged tree matches
its tested head `ed953da8`; neutral/skipped jobs are not counted as passing tests.
This completes the integration gate above, without strengthening its timing claims.

PR [#355](https://github.com/writerslogic/warm/pull/355) merged as `5db4341a` after the
109-entry final gate. Candle 0.11.1 became available during the investigation and
removed the vulnerable ZIP 8.6 dependency; the lockfile and GitHub dependency graph
both resolve ZIP 9.2.0. Local checks pass 30 model, 22 embedding and seven store tests
(one existing embedding ignore), strict Clippy and formatting. Cargo Audit reports
zero known vulnerabilities; bincode/paste maintenance warnings remain. Evidence and
the advisory database revision are in `candle-security-20261031/`.

The remaining esbuild and ZIP GitHub alerts were stale against the verified patched
lockfile and dependency graph. Their dismissals record the exact version/range evidence,
not risk acceptance. The unpatched deepmerge development-fixture alert remains open;
it must not be presented as fixed or removed by dropping integration coverage.

PR [#356](https://github.com/writerslogic/warm/pull/356) merged as `3f787a5e` after all
69 check entries completed without failure or pending work. Node now exposes audit
signatures, their exact signed bytes and optional delivery IDs; Rust exports named
`AuditEntry`/`AuditOp` types. Three native tests verify legacy/signed/unsigned behavior,
independently generated public keys, process restart, tampering and safe-integer timestamp
bounds. Build, TypeScript, Clippy and formatting pass; evidence is in
`audit-interface-20261031/`. Historical public keys remain necessary: this does not add
key rotation, whole-history authentication, or independent cryptographic assurance.

### Conditional-request semantics

Source inspection found literal comparison of weak entity tags, inspection of only one
header line, and comma splitting inside quoted opaque values. Three new regression groups
fail on the prior implementation. The replacement parses admitted header bytes without
allocation, combines header lines, preserves quoted commas/opaque bytes and compares
validators independently of their strength. A lone wildcard remains valid; malformed
conditions are ignored in favor of a full response, including mixed wildcard/tag lists.

The applicable requirements are [RFC 9110 §13.1.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-13.1.2)
and [§8.8.3.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-8.8.3.2).
The existing real-HTTP conditional-read test now covers strong/weak equivalence, repeated
fields, quoted commas, empty 304 bodies, returned validators, invalid-condition fallback
and invalidation after a write. This closes the inspected `If-None-Match` matching gap;
it is not a claim of complete HTTP conformance. The follow-up requires its final PR gate.


A deterministic concurrent-write regression subsequently reproduced a second issue:
a document update became visible before the separate tenant generation bump, and two
reads returned changed metadata under the old validator, including a stale 304. Validators
now pair a digest of the exact serialized representation with the process epoch; serialization occurs
on the blocking worker and the same bytes are sent for 200. This also removes reliance on
restore/write timing for cache correctness. Response validators are no longer a shared tenant
revision token. The tradeoff is explicit: 304 responses render and hash their representation;
no latency improvement is claimed, and large metadata-page cost needs workload measurement.


PR [#357](https://github.com/writerslogic/warm/pull/357) merged as `171c80d3` after
67 successful and four skipped check entries, with no pending/failed checks or unresolved
review threads. Its merged tree matches tested head `5d71997a`. This completes the
conditional-request integration gate above; rendering cost remains unmeasured.

### Authentication and request-header ambiguity

Four real-HTTP regression groups fail on `3f787a5e`: lowercase Bearer schemes were
refused, while duplicate Authorization, X-Tenant-ID and Idempotency-Key fields were
silently reduced to the first value. Conflicting headers could therefore execute a
write using one of the supplied identities or retry keys. This establishes ambiguous
request handling, not a demonstrated authorization bypass or intermediary exploit.

The server now rejects repeated authentication fields with 401/Bearer challenge,
repeated tenant fields with 400 `invalid_tenant`, and repeated retry keys with 422
`invalid_request`, before mutation or key reservation. Identical duplicates are also
rejected. HTTP and MCP accept case-insensitive Bearer schemes and one or more ASCII
spaces before the unchanged case-sensitive secret. MCP static-token mode reuses the
OAuth header parser; OAuth retains its separate token-format and length checks.

[RFC 9110 §11.1](https://www.rfc-editor.org/rfc/rfc9110.html#section-11.1) defines
case-insensitive authentication schemes; [RFC 6750 §2.1](https://www.rfc-editor.org/rfc/rfc6750.html#section-2.1)
permits one or more spaces. Rejecting duplicates is the explicit WARM policy for these
singleton fields, consistent with the sender restriction in
[RFC 9110 §5.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-5.3).
No-auth mode remains explicitly unauthenticated. This is scoped protocol hardening,
not full HTTP/OAuth conformance or a new restart-safe idempotency guarantee.

Validation and the final PR gate are recorded in `header-semantics-20261031/validation.json`.

Local acceptance passes seven authentication tests, all 19 HTTP integration tests,
and all 41 MCP tests; strict scoped Clippy, formatting and diff checks pass. The first
parallel HTTP run had four empty responses and a burst-timing failure. Serial execution
cleared the empty responses; the rate test now initializes its store and replenishes
the warm-up token before measuring the burst, retaining every existing rate/quota
assertion. All failed attempts remain in the evidence archive. No production timeout,
rate limit, test assertion or validation gate was relaxed.


### Request-body error classification and cleanup

Four regressions fail on `171c80d3`: buffered reads label transport failure as 413;
restore labels it 422; streaming ingest returns 400 with `invalid_request`, whose
registered contract is 422. One `invalid_body` 400 problem now describes unreadable
bodies consistently. Buffered reads retain 413 only for Axum's documented
`LengthLimitError`; 408 deadlines remain unchanged. The underlying transport error
text never enters the problem response. This follows the distinction between
[RFC 9110 §15.5.1](https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.1)
and [§15.5.14](https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.14).

Seven body-handling and four error-contract checks pass, including multi-frame size
boundaries, real-router error injection, work-slot/gauge reclamation, no buffered
mutation or retry-key reservation, restore-upload cleanup, and preservation of a
previously committed NDJSON batch. Strict binary Clippy, formatting and diff checks
pass. Evidence: `body-errors-20261031/`. Response delivery after a network disconnect
is not guaranteed, and streaming retries still require reconciliation. The complete
PR gate remains required before merge.

The same follow-up registers and documents the existing `index_unavailable` 409
code, which lacked an entry in `error::CODES` and the error reference. A real-router
check verifies the specific problem title and successful explicit exact-search fallback
on the same empty store. Its baseline failed on the generic Conflict title; the 409
routing behavior itself was already implemented. The targeted check and refreshed
error-contract, Clippy and formatting gates pass.


The existing Windows job ran default features only (`default = []`), excluding both
HTTP and MCP runtime contracts. Its existing default-feature run is retained, followed
by an explicit `server,mcp` nextest step scoped to both service binaries and the five
server integration targets. Actionlint passes locally. Windows runtime acceptance is
pending the new PR's exact-head CI, not inferred from local macOS results.


PR [#358](https://github.com/writerslogic/warm/pull/358) merged as `5192ab7b` after
68 successful and four skipped check entries, with no pending/failed checks or unresolved
review threads. Its merged tree matches tested head `60088616`. This completes the
header-semantics integration gate, including the pre-existing Windows default-feature job.
It does not establish the expanded Windows service coverage added in the next follow-up.

### Enforced CI protection

Main's GitHub protection had `strict: true` but no required status contexts. After
verifying the existing CI `check` aggregate succeeded, required status protection now
names `check` from GitHub Actions app 15368. The aggregate fails unless all twelve
listed prerequisite jobs succeed, including Windows tests and workflow lint. The
strict up-to-date requirement and captured owner-review/admin settings remain intact.
Before/after API evidence and the successful originating check are retained in
`branch-protection-20261031/`. Conditional workflows outside this aggregate are still
reviewed separately; this rule does not claim to enforce every GitHub check.


After integrating main `5192ab7b`, the complete local server unit suite passes all
72 tests with `server,mcp`; refreshed scoped Clippy and formatting pass. MCP source
matches the 41-test validated header follow-up. The expanded Windows runtime gate
remains pending, so local results are not presented as cross-platform verification.

The expanded Windows step exposed an unconditional `tokio::net::UnixListener`
import (CI run 41194768596), so the required aggregate correctly failed. Unix-only
imports, listener variants, match arms and cleanup are now platform-guarded. A
non-Unix CLI regression checks the existing explicit unsupported-platform error.
The real Unix socket integration test and strict service Clippy pass locally;
Windows runtime results remain pending. The failed CI log is preserved.

Source inspection also found that the newly enabled service tests used Unix
`kill` for shutdown and SIGHUP. Windows children now get their own process groups;
the harness sends targeted Ctrl+Break and the server routes it through its existing
drain/flush path. Windows reloads exercise the authenticated admin endpoints, while
Unix retains SIGHUP coverage. No graceful-shutdown assertions were removed.
All 13 local operations tests and the key-rotation test pass; strict scoped Clippy
and formatting pass. Windows execution remains pending. The only added direct
dependency is test-only `windows-sys`, already present at 0.61.2 in the lockfile.
