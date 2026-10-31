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

Windows CI run 41197071913 passed 159/164 service tests, including graceful
shutdown/drain, readiness, TLS/key reloads and non-Unix socket rejection. Five
snapshot-related tests failed. The unit-level log reports access denied: snapshot
copying reopened each destination read-only before flushing, while Windows
`FlushFileBuffers` requires a writable handle. Windows copying now retains that
handle through permission restoration and flush; other platforms keep native copy.
A portable regression covers nested read-only data, byte counts, permissions and
process-state exclusions. Five local snapshot checks and refreshed strict Clippy
and formatting pass; Windows acceptance remains pending.

This fix does not establish full Windows power-loss durability: directory-sync
helpers are still no-ops there. The separate core admin migration `copy_tree` also
still skips Windows file flushes; that requires its own correction and validation.

Windows run 41199300291 passed the read-only snapshot regression and cleared the
access-denied failures, but finished 162/165: an idle test assumed writes took less
than one second, a readiness test exhausted its two-second drain window, and a
backup round-trip received an empty response without identifying the request.
Eviction testing now injects explicit time at the idle boundary (-1/at/+1 ms),
while production retains its monotonic clock. The readiness fixture pre-opens its
tenant and tests a five-second delay with the matching minimum-time assertion.

A stopped-server regression also exposed `try_send` panicking on connection
refusal; it fails before the helper fix and passes afterward. Ordinary request
failures now identify the method/path and recent server events. The Windows service
step schedules independent fixtures serially, retaining every test and the
multi-client concurrency inside them; socket read timeouts remain 20 seconds.
All 13 local operations checks, the idle-boundary test, stopped-server/readiness
checks, strict Clippy, formatting and actionlint pass. The backup timeout’s precise
cause remains unconfirmed; the next Windows run must pass and is not overridden.

A review of the unmerged Windows copy fix found that byte-stream copying would
omit NTFS alternate streams preserved by `std::fs::copy`. Native copying is now
retained everywhere. For a read-only Windows destination, only the unpublished
copy’s read-only bit is temporarily cleared to acquire a writable flush handle;
attributes are restored before propagating open errors or flushing. The Windows
regression now verifies an alternate stream as well as ordinary bytes and
permissions. The portable test, Clippy and formatting pass locally; Windows
acceptance of this final metadata-preserving variant remains pending.

### Python HTTP transport failure and lifetime contracts

The plain-HTTP optimized SDK transport combined timeout budgets and classified
post-send failures on fresh sockets as connection failures. It now applies separate
connect/write/read socket deadlines and reports the matching HTTPX error class,
without retrying requests. Direct HTTPS misuse is rejected before I/O (normal HTTPS
clients already used HTTPX TLS). Body production happens before connection state
changes, and a weak close registry stops retaining exited workers’ sockets.

Five initial regressions and one thread-lifetime regression failed before their
respective fixes. Real loopback checks plus client, memory and importer retry tests
pass on CPython 3.10.19 and 3.14.7: 47 passed, one optional-orjson skip on each.
Evidence, failed attempts and reproducible regression patches are retained in
`python-transport-20261031/`. No latency, global pool-limit, automatic retry or
restart-safe idempotency improvement is claimed. Full integration CI remains pending.

Six additional SDK end-to-end tests pass against the actual local `warm-server`
process, covering synchronous/asynchronous workflows, errors and request payloads.
This complements the transport fixtures; it does not validate Windows execution.

### Migration and upgrade file-flush ordering

Source inspection found `admin::copy_tree` explicitly skipped Windows file flushes,
and format upgrades copied retained/root-level files without flushing their contents
before publication. A shared crate-private copy helper now flushes copied files and
preserves permissions, retaining writable handles on Windows and native copies on
Unix. Upgrade publication syncs staged directory entries where supported.

Four admin tests and eight upgrade regressions pass locally, including read-only
nested files, process-lock exclusion, mixed-version tenant arenas and read-only
deployment metadata. Strict library/test Clippy and formatting pass. Evidence is
in `migration-flush-20261031/`. These tests verify data and permissions, not hardware
power-loss behavior; the missing flushes were established by source inspection, not
by claiming a new test fails on the baseline. Windows execution and PR integration
remain pending. Windows directory-sync limitations and throughput are unverified.

The migration helper likewise retains native file copies rather than stream
copying, preserving NTFS alternate streams. It restores the unpublished Windows
copy’s read-only attribute after acquiring a writable flush handle, including the
open-error path. Windows-only stream-preservation assertions cover both nested
migration files and top-level upgrade metadata. The refined helper passes the
four admin/eight upgrade checks and strict Clippy/format gates locally; its
Windows execution is still pending. This does not prove power-loss durability.

PR [#359](https://github.com/writerslogic/warm/pull/359) merged as `f78b2175`
after 102 successful and 11 skipped check entries, no unresolved reviews and zero
open code-scanning alerts. The merged tree exactly matches tested head `f0845612`.
Windows run 41203375960 passed 806 default tests (14 skipped) and all 165 service
tests (none skipped), including native-copy alternate streams, backup/restore,
shutdown and idle boundaries. The final Windows log and check state are archived
in `body-errors-20261031/`. This closes that PR’s integration gate; SDK/migration
follow-up changes still require their own up-to-date CI gate.

### Dependency graph refresh and constrained remediation (2026-10-31)

The refreshed GitHub dependency graph has **210 open alert records / 201 distinct
advisory IDs**: 208 SDK optional-graph records, one docs-generator minimatch
record, and the previously documented JavaScript deepmerge record. Severity by
record is 10 critical, 72 high, 99 medium, 29 low. This supersedes earlier alert
counts, not earlier verified code-scanning results. The snapshot is
`dependency-refresh-20261031/open-alerts.jsonl`; package presence is not proof
that a vulnerable operation is reachable through WARM. Base SDK dependencies
remain httpx/sniffio; consumer environments resolve their own dependencies.

Dependabot PR #360 updates pyasn1 0.6.4 to 0.6.5, covering four decoder DoS
advisories. Its uv lock rewrite also removes onnxruntime 1.24.3 from the Python
3.10 fork, retaining 1.22.0 there and 1.31.1 on newer Python. PyPI metadata declares
1.24.3 supports Python 3.10 but publishes no cp310 wheel; 1.22.0 publishes four.
This incidental solver change is disclosed, not claimed as a measured importer
runtime improvement. No other package/version pair changes in the accepted lock.
`uv lock --check --offline` passes; focused base SDK tests pass on Python 3.10.19
and 3.14.7 (39 passed, one optional-orjson skip, 46 subtests each). This does not
validate every optional framework or demonstrate absence of dependency exploits.

Rejected changes and remaining work:

- Broad uv updates left vulnerable legacy forks and removed a newer cohere fork.
  Forcing virtualenv >=21.14.9 introduced older instructor/LangChain variants;
  forcing exactly 21.14.9 removed virtualenv through an instructor downgrade.
  Neither trial is committed. Patched virtualenv requires packaging >=26.3;
  the legacy graph retains packaging 25. Resolve parent constraints before
  claiming that an alert disappearing from a lock represents a safe upgrade.
- Embedchain 0.1.128 constrains mem0ai<0.2 and chromadb<0.6. Semantic Kernel's
  openapi-core 0.19.5 constrains Werkzeug<3.1.2, below patched 3.1.9. Pillow 13
  and filelock 4 require Python >=3.11. A support-policy question is pending;
  no extra was retired and Python 3.10 support remains. Acceptance: supported
  extra combinations resolve without vulnerable mandatory dependencies and
  pass their real adapter contracts; otherwise document a deliberate policy
  decision rather than silently downgrade or suppress alerts.
- TypeDoc 0.28.20 requires minimatch ^10.2.5. Patched minimatch 11.0.1 removes
  `matchOne`, called by TypeDoc glob traversal. A scoped override installs and
  reports zero npm advisories but breaks that API: it was reverted. Inspected
  TypeDoc calls do not enable the advisory's optimizationLevel>=2 condition.
  Keep the alert open pending compatible upstream support; acceptance includes
  both generated references and glob traversal, not an install-only audit.

SDK installation guidance now recommends the needed extra rather than `all`,
without removing compatibility. `docs/SECURITY.md` records the unresolved
constraints. Resolver trials, metadata, API probe, and test logs are archived in
`dependency-refresh-20261031/investigation.tar.gz` with a SHA-256 index. No paid
services, model downloads, advisory suppressions, or unsupported safety claims.

After integration with main `78f722cd`, the same focused dependency-validation
commands pass 46 tests, one optional-orjson skip, and 46 subtests on each Python
version, including the newly merged transport regressions. PR #361 itself merged
only after 68 successful and four skipped checks, with no pending/failing checks
or unresolved review threads. Its Windows gate passed 807 baseline tests (14
existing skips) and 165 service tests (zero skips). Tested and merged trees match;
`python-transport-20261031/validation.json` and its final-state/log evidence record
the gate. Code-scanning open alerts were zero at that merge; dependency alerts
remain tracked separately above.

### Uploaded archive resource bounds and read-only durability (2026-10-31)

`restore_upload` extracted tar metadata before checking entry limits. In tar
0.4.46, the normal iterator buffers GNU long names/links and local PAX records
with `read_all()` before yielding a regular entry. The default 8 GiB upload
limit therefore did not provide a suitable metadata-allocation bound. A valid
65,537-byte PAX-comment fixture reproduced acceptance beyond the proposed bound;
the new rejection regression fails before the change. No OOM or RSS result is
claimed, and this route requires the existing admin authorization.

A seek-based raw-header preflight now limits each extension to 64 KiB before
normal extraction. It reuses tar's header/checksum and PAX parsers, preserves
PAX size overrides across GNU extension headers, and rejects unsupported sparse
entries before their parser can allocate. Checked offsets, effective file sizes,
existing aggregate byte/entry limits, duplicate/dangling metadata and truncation
are validated before extraction. File payloads are skipped by seeking, not loaded
into memory. This is an application resource limit, not a tar-standard limit.

The Windows upload flush also opened read-only extracted files directly for
writing. It now shares the snapshot-copy helper: clear only the unpublished
file's read-only attribute when needed, acquire the write handle, restore
attributes even if opening failed, then sync. Native copies and their alternate
streams remain unchanged. The Windows test checks that the old write-open path
returns PermissionDenied before exercising the corrected flush.

Local acceptance: five archive tests, including limit-1/limit/limit+1 metadata,
PAX/GNU interoperability and malformed/overflow cases; all 13 real-server
operations tests; strict server/operations Clippy and formatting. The earlier
read-only-only gate also passed 74 server unit tests and the snapshot-copy test.
Two test runs used an incorrect HTTP-400 expectation; it was corrected to WARM's
existing HTTP-422 invalid-request contract, with no production error downgrade.
`archive-restore-20261031/` retains logs, hashes and a reproducible baseline patch.
Windows CI remains required; filesystem power-loss durability is not established.

The first full archive CI gate caught Clippy's `byte_char_slices` lint in a new
test, which the earlier binary-only local Clippy target did not compile. The test
now uses the equivalent byte string, and local strict Clippy includes `--tests`.
CI failure logs and the expanded local check are retained; no lint was disabled.


Windows CI at `cdf80c39` passed all five archive regressions, including the
read-only permission check, but the full service gate was 169 passed / one failed.
The existing off-box round trip received no restore response within its 20-second
read deadline. Resource pressure from the harness's full-archive copies is a
hypothesis, not a measured cause. The follow-up removes the body clone and encoded
request copy, drops the owned upload before reading the response, and retains the
same payload, timeout and restore assertions. Transport failures now report read
elapsed time, byte count and I/O errors; restore-stage events distinguish receive,
upload sync, unpack, store sync, verification and installation without logging
archive contents. The failed Windows log is retained. No memory or latency gain
is claimed; a green final Windows gate remains necessary.

The final local source passes 78 server unit tests, 19 HTTP API tests and 13
operations tests, plus strict Clippy including test code and scoped formatting.
The first default-parallel API run hit four 20-second response timeouts; the
serial rerun matches CI and passes without changing deadlines. Both logs remain
available; this is not evidence of unconstrained parallel-workload performance.


The diagnostics gate at `3e036396` passed the full off-box round trip with its
original payload and 20-second read deadline. Windows baseline: 807 pass / 14
skip; service: 169 pass / one failure. The remaining failure was different: the
size-limit test sent a 1,073,753,088-byte archive to a 4,096-byte limit. The server
logged 413 after 70 ms, but the write-then-read harness received Windows 10053
immediately on reading. [RFC 9110 §15.5.14](https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.14)
permits terminating oversized requests; it does not guarantee that a client
continuing a massive upload receives that final response.

Size rejection is now tested at 4,095 / 4,096 / 4,097 actual bytes: the in-limit
truncated archives must receive 422, the over-limit body 413, all with problem
JSON and empty staging. Full-size valid restore and corrupted/truncated archives
remain covered. The hostile-archive fixture moves the full corruption payload
and copies only the truncated half, eliminating redundant full-file allocations.
No deadline or production limit changes. Large rejected uploads can still end
in transport errors; no universally graceful-response or RSS/latency claim is
made. The second failed gate is retained, and new exact-head CI remains required.

The revised size-boundary cases and all 13 operations tests pass locally, as do
strict operations Clippy, scoped formatting and diff validation. Windows CI
remains required for this revision.

PR [#360](https://github.com/writerslogic/warm/pull/360) merged as `30cd5d46`
after 69 successful and three skipped exact-head checks, with no pending/failed
checks or unresolved reviews and no open code-scanning alerts. The tested head
`abc82f4a` and merge share tree `813b1c50d74d7e7855bfe8767fc5792a96718709`.
Windows passed 807 baseline tests (14 existing skips) and all 170 service tests
(no skips), including full restore, hostile archives and size boundaries. Admin
merge bypassed owner review only. The exact gate and Windows log are retained.
An immediate post-merge dependency-alert read still reported 210 open records,
including four pyasn1 entries; no alert-closure count is inferred from the merge.

### Python LangChain identity and filter contracts (2026-10-31)

Fourteen real-server regressions fail on `78f722cd`: supplied IDs are ignored,
`from_texts(ids=...)` passes IDs to the connection constructor and fails, retriever
metadata filters are discarded, and `zip` silently truncates uploads when text,
metadata, or embedding counts differ. Empty input also invokes the embedder.
These are adapter contract failures, not retrieval-quality measurements.

The adapter now preserves caller IDs, supplies secure UUIDs for absent IDs,
returns stored IDs on Documents, forwards similarity filters, and checks input
cardinality before embedding and output cardinality before upload. Explicit IDs
from `add_documents` override Document IDs through the framework's base contract.
`from_texts` separates IDs from connection options and closes its owned client
on failure; callers can explicitly `close()` successful stores. Existing
LangChain [VectorStore contracts](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore)
and the installed implementations were inspected before changing the adapter.

LangChain 0.2.11 has a Document ID field but its inherited `add_documents` still
loses those IDs (13 pass / one fail in the compatibility trial). The declared
minimum is now 0.3.86, matching the already locked legacy branch, which passes all
17 final adapter tests on Python 3.10.19. Current 1.7.3 passes the full SDK suite
on Python 3.14.7: 77 tests and 46 subtests, with 13 upstream asyncio deprecation
warnings. The locked modern version, 1.7.0, also passes all 17 adapter contracts
on Python 3.14. CI uses the two locked versions (0.3.86/3.10 and 1.7.0/3.14)
against the real server with framework tracing disabled. All embeddings and
server calls are local; no paid APIs, model downloads or relevance/latency claims.

Offline lock validation, scoped actionlint/zizmor and distribution builds pass.
uv 0.13.6 preserves lock revision 5 and every package/version pair; the final
delta from PR360 is only the two LangChain requirement lower bounds. An older
local resolver rewrote markers; that normalization was discarded.
The pending dependency remediation must be integrated before this follow-up merges.
No extra or Python version was retired. Legacy dependency advisories remain open;
passing API contracts does not establish framework security. Scored retrieval,
MMR and adapter deletion remain explicitly unsupported. Evidence and the 14-case
baseline reproduction patch are in `langchain-contracts-20261031/`; final PR checks
remain required before merge.


### Python LlamaIndex query and deletion contracts (2026-10-31)

The adapter ignored metadata filters, query modes and constraints, and created an
unowned HTTP client for each property access. The initial real-server suite
reported 34 failures on `363b837b`; one empty-ID-list expectation was later
corrected from actual framework behavior, leaving 33 retained defect assertions. It now applies flat AND/equality filters before
server-side top-k, rejects unsupported modes/operators/ID restrictions and
projection options, rejects duplicate filter keys, and owns one explicitly
closable client. Original node text is stored separately from metadata; a
regression caught the previous metadata-prefixed text returned as node content.

The framework's [reference-delete contract](https://developers.llamaindex.ai/python/framework-api-reference/indices/vector/#llama_index.core.indices.vector_store.base.VectorStoreIndex.delete_ref_doc)
means source-document deletion, but WARM
passed that value to node deletion. A source ID colliding with an unrelated node
therefore deleted that node (HTTP 200 in the retained baseline). Reference and
filtered deletion now raise before writes. `delete_nodes` implements explicit-ID
deletion, validates the ID list before starting, tolerates only documented
missing-node errors, and propagates authorization failures. Multiple deletes are
sequential and can partially commit; they are not an atomic source-document purge.
This prevents the false deletion contract without claiming reference ownership
or complete node/relationship reconstruction is implemented.

The actual LlamaIndex [vector-store types](https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/)
and installed Pydantic bridge were inspected.
Its minimum 0.10.0 passes 42 final contracts on Python 3.10; 0.14.25 passes the
full SDK suite on Python 3.14 (125 tests, including 48 LlamaIndex tests, plus
46 subtests; 13 upstream asyncio deprecation warnings). The older framework
rejects nested filters during Pydantic construction; the newer one delegates
rejection to WARM. An initial compatibility test expected only the latter and
was corrected without changing production validation. Existing dependency bounds
remain. CI exercises both versions against the real server; all vectors are local.
Scoped workflow lint and package builds pass. Logs, hashes and the initial
34-case baseline patch are in `llamaindex-contracts-20261031/`; no performance,
relevance or overall security improvement is inferred from these contract tests.
The integrated exact-head PR gate remains required before publication.


The added `VectorStoreIndex` insertion/retrieval workflow initially failed on
both versions: text-storing indexes pass `node_ids=[]` for unrestricted search.
Empty ID lists now preserve that framework convention; nonempty unsupported ID
restrictions still raise. The real workflow and explicit empty-list cases pass
on both versions using the framework's deterministic local MockEmbedding, with
no LLM or model-weight calls. The earlier failed expectation and workflow logs
remain archived; direct adapter tests alone had missed this caller contract.


### Remaining adapter prerequisites: source observations, not runtime validation

- `writerslogic_warm_sdk/haystack/document_store.py`: `count_documents` returns
  zero and `filter_documents` an empty list unconditionally; `write_documents`
  ignores its duplicate policy. Reproduce with the real framework before choosing
  support scope. Counting can use existing tenant stats; correct FAIL/SKIP writes
  need atomic server semantics, not a race-prone client preflight.
- `writerslogic_warm_sdk/semantic_kernel/memory_store.py`: collection methods are
  stubs; upsert/remove use raw record IDs and nearest matching omits collection
  filtering. Installed-framework abstract-method/signature compatibility has not
  been checked here, so runtime cross-collection consequences remain unverified.
- HTTP `documents::list_documents` returns IDs/metadata only. Native
  `Core::get_document` returns text, version and chunk count, but only first-chunk
  text (with the documented whitespace normalization). Adapter document recovery
  requires an explicit bounded retrieval contract, not fabricated empty results
  or similarity search masquerading as enumeration. No new endpoint is implemented.

These are follow-up prerequisites; the current framework fixes establish neither
full Haystack/Semantic Kernel support nor all-adapter production readiness.


### Metadata listing projection and cold checkpoint reads (2026-10-31)

HTTP and Node metadata listings discarded text after copying full document records.
A 15-row fixture allocated 998,897 bytes. The new projection uses live metadata
and selected checkpoint records without copying stored text: 13,057 bytes live,
13,777 reopened. Counts no longer hydrate the checkpoint. Sorted overlay merging
preserves replacements, tombstones, cursor order and replay behavior. Deep offline
verification now explicitly validates the full checkpoint and propagates decoder
errors; a CRC-valid slot corruption regression guards against weakening this path.

All nine checkpoint tests, 19 HTTP API tests, two Node tests and the allocation-bound
regression pass locally, as do strict Clippy including tests, scoped formatting
and generated-artifact verification. Exact-head CI remains required. The evidence directory
`document-listing-20261031/` retains baseline, intermediate failed cold-read and
candidate logs plus a baseline reproduction patch. Allocation results do not
establish RSS, latency, throughput, or competitive leadership; chunk decoding and
metadata sizes still affect work.


### Haystack silent-result and duplicate-policy corrections (2026-10-31)

Nine real-server baseline failures confirm false zero counts, fabricated empty
filter results, and ignored duplicate policies. Counts now use authenticated,
tenant-scoped stats, exposed in both SDK clients. FAIL/SKIP and unknown policy
types reject before mutation; NONE preserves WARM's existing overwrite default.
Filtering explicitly raises until faithful full-document retrieval exists.

Twelve contract tests pass on Haystack 2.0.0/Python 3.10 and 2.27.0/Python 3.14,
including actual DocumentWriter use and sync/async authentication isolation.
The whole SDK run passes 71 tests and 46 subtests with three documented optional
skips; package builds and workflow lints pass. Evidence and reproducible baseline
are in `haystack-contracts-20261031/`. Exact-head CI remains required. This is
partial adapter support, not full DocumentStore conformance: atomic FAIL/SKIP,
full-document enumeration/filtering and serialization remain prerequisites.


### Semantic Kernel minimum-version instantiation blocker (2026-10-31)

Runtime inspection with semantic-kernel 1.39.4 / Python 3.10 confirms that all 12
MemoryStoreBase methods remain abstract: the adapter uses obsolete `_async`
names. Normal instantiation raises TypeError before any backend request. The
prior unverified compatibility concern is now reproduced; source-level collection
isolation concerns are still not runtime disclosure findings. SDK docs flag the
adapter as unusable. `semantic-kernel-compatibility-20261031/` records the probe,
log and restoration acceptance criteria. No extra or Python support was retired;
renaming methods without replacing collection stubs would enable unsafe behavior.


### Metadata listing / Haystack merge gate (2026-10-31)

PR #363 merged at `68e016dba33eee777e3630ce27ec4b9a953995b3` after 68 checks
succeeded and four skipped; required aggregate passed. The signed merge and
exact tested head `79ce353407fee9df816bf60b5fda2a16dc3589cc` share tree
`15ed4efc790d38704f41dee04f0fd589defbd051`. Windows passed 810 baseline tests
(14 skips) and 170 service tests; Python passed both real Haystack boundaries
(12 each) alongside LangChain/LlamaIndex and SDK suites. Evidence:
`haystack-listing-gate-20261031/`. This supersedes the pending-gate notes above;
partial framework support and dependency risks remain.


### Retained chunks and text-cache integrity (2026-10-31)

A truncated cache reproduced a panic while closing a store with a due checkpoint.
A same-length UTF-8 byte change reproduced altered text after close/reopen: unchecked
cache bytes entered an authoritative checkpoint. Fallible JSON/checkpoint writers
and process-local plaintext CRC32 values now reject damaged reads while preserving
journal recovery. CRC32 is not malicious-tamper authentication; encrypted records
retain AEAD, and durable format bytes are unchanged.

A new Rust/HTTP/sync+async Python read returns one retained chunk, metadata, source
span and opaque revision. It checks tenant/read authorization, reports absence and
unretained text distinctly, and retries boundedly when document/segment identities
change during unlocked I/O. Checkpoint record decoding enforces ingest limits before
allocation. This supersedes the earlier absence of an HTTP text-retrieval endpoint;
it does not recover omitted original whitespace, unretained text or original
embeddings, and does not establish full Haystack/Semantic Kernel conformance.

Five text-integrity units, five deterministic read-concurrency units, six
recovery/chunk integration cases and the allocation regression pass locally. Reading
one 64 KiB chunk allocated 66,233 bytes live/compacted and 66,281 reopened; this is
requested allocation, not RSS, latency or an HTTP response bound. Ten checkpoint tests, including purge and encryption, six golden-store checks
(one intentional generator skip), 20 HTTP tests, strict Clippy and 16 generated
exports pass. SDK tests pass 62 tests / 46 subtests on Python 3.10 and 3.14
(three locally uninstalled framework modules skipped). An initial CI assertion
incorrectly expected HTTP-generated document versions to stay unchanged; it was
corrected and reused caller versions are tested natively. Exact-head CI remains
required after that correction. Evidence: `document-chunks-20261031/`.

The existing infallible get/search text-hydration paths still use `expect` for cache
I/O failures. A follow-up must propagate errors through already-fallible engine/query
APIs while preserving intentional deletion/scrubbing behavior; inject truncation and
same-length corruption, require an explicit error and successful journal recovery,
and retain concurrent-delete/search regressions. Do not hide corruption as no hits
or missing text. Atomic duplicate policies and faithful framework record/collection
semantics remain separate prerequisites.

Follow-up call map (source inspection, not additional runtime reproductions):
`search.rs::RankedHit::read` feeds `read_ranked` / `execute_search` and
`engine/document_api.rs::run_searches`; `vector_index.rs::to_result` feeds
`read_vector_hits` / `search_vector_snapshot`. Preserve the distinction between
`None` (a scrubbed/stale capture requiring retry) and `Err` (storage failure).
`DocumentStore::chunk_text` also reaches lexical phrase fallback, deferred-encoding
fingerprints and memory document views; those consumers must not silently turn
errors into absent text. Existing `document_snapshot_tests` cover retained captures
and concurrent deletion. Memory/get-view API changes require their own caller audit.


### Fallible query text hydration (2026-10-31)

Four runtime probes reproduce eight panics: text, vector-snapshot, phrase-fallback
and captured queries each panic on truncated or same-length damaged cache text,
despite already exposing Result contracts. Ranked hydration now propagates errors;
a scrubbed capture remains a distinct retry outcome. Phrase fallback uses fallible
text access, and in-memory search forwards the result. Four paired candidate
probes pass; expanded mixed healthy/damaged-hit fixtures and seven existing
snapshot/concurrency/recovery cases also pass (11 tests total). Evidence:
`query-errors-20261031/`. Broader validation and CI remain pending.

This closes the query-result slice of the preceding call map; deferred encoding
is addressed below, while infallible views remain. Existing read guards across query text I/O remain;
a separate change must capture and revalidate store/segment identity without
weakening erasure or coherent retries. No new lock or critical-section extension
is part of this error-propagation change.


### Deferred-encoding cache failures and committed outcomes (2026-10-31)

Two more probes reproduce four panics during encoding preparation or fingerprint
validation at installation, for both truncation and same-length damage. The latter
can happen after the encoding transaction is durable. Text/fingerprint reads now
propagate errors; the existing post-commit wrapper reports CommittedMutationError
instead of unwinding before reporting durability. Both tests pass: preparation
leaves journal/revision unchanged, while a failed committed installation advances
them and replay recovers encoded state from original text. Encoding is idempotent
after repair/reopen. Evidence and independent reproduction patch are included in
`query-errors-20261031/`; broader final validation and exact-head CI are pending.

Remaining scope: legacy raw `WarmCore::query` logs deferred-catch-up failures and
can omit pending vectors; existing get/list and memory-view helpers remain
infallible. Do not claim full query completeness or universal I/O recovery.


The combined query/encoding follow-up passes its final local gate: 11 snapshot,
119 document (one deliberate benchmark skip), two deferred fault/replay, four
ingest/postings, and 20 HTTP API tests; strict Clippy and scoped formatting pass.
It is folded into PR #364 before merge, so its final exact-head CI covers chunk
access, cache integrity, and error propagation together. The previous 3b5b61dd
head had not completed its Windows gate and is not recorded as fully validated.


### Maintained SDK integrations and dependency retirement (2026-10-31)

User policy: prioritize maintained integrations and retire extras that cannot be
secured. This supersedes the unresolved support-policy gate in the earlier
October 31 dependency investigation. This change retires the SDK's `all`,
`embedchain`, and legacy `semantic-kernel` extras and deletes their two obsolete
adapter modules. No callers outside the retired modules were found; the separate
current Semantic Kernel implementation under `integrations/python` is preserved,
with its security gate explicit. Historical audit and rename evidence is retained.

The SDK floor moves from Python 3.10 to 3.11, matching Python's published October
1, 2026 end of life for 3.10. This is an intentional compatibility change, not a
resolver marker that silently omits a requested integration. The Pydantic AI extra
uses the official provider-free `pydantic-ai-slim>=2.64.0`; applications explicitly
choose provider extras. The resolver constraint `filelock>=4.2.6` is not an
incompatible override: all declared upstream requirements remain satisfied. It
applies to this repository, not to independently resolved consumer installations.

Before/after scan: every registry package/version pair in both complete locks was
queried against the public OSV API, including inactive interpreter/extra forks.
The SDK lock shrinks from 458 to 250 total packages (457 to 249 registry pairs).
Affected pairs fall from 24 to 2; distinct GHSA IDs fall from 195 to 5. Raw OSV IDs
include PYSEC aliases: 319 before, 10 after; these are not independent exploit
counts. Separate GitHub alert snapshot: 206 open records / 197 distinct GHSA IDs,
204 records for the SDK lock and two JavaScript locks. The two databases and their
record granularities differ; neither count measures reproduced WARM exploits.

Remaining SDK matches: Chroma 1.5.9 (four distinct GHSA IDs; direct importer/test
and transitive CrewAI dependency) and DiskCache 5.6.3 (one; DSPy dependency).
Current upstream package versions are still affected. Chroma importer code calls
`PersistentClient` or `HttpClient` and read operations, not an HTTP server; that
observation alone does not prove safe loading of stored collection configurations.
DiskCache requires attacker write access to a cache directory. These integrations
remain outside a security-qualified release until reachability and cache/source
isolation are verified or dependencies replaced/retired. This pass does not claim
zero advisories, secure optional extras, or safety of the separate integrations
packages. No alert suppression, broken dependency overrides, or paid API/model
calls were used.

Validation: uv 0.13.6 resolves/checks the lock and builds wheel/sdist. An isolated
installed wheel verifies Python metadata, removal of retired modules/extras, and
real Pydantic AI tool registration. Offline FunctionModel + real local server
checks pass on Python 3.11 and 3.14 (4 each): tenant isolation, normal retrieval,
authorization errors, encoder failures, and connection closure. The latter exposed
and fixes a leaked per-call client; both success and exceptions now close it.
Base SDK on 3.11: 61 passed, 5 skips (four absent optional framework modules and
orjson), 46 subtests. Existing minimum and current framework contracts pass:
LangChain 0.3.86/1.7.3 (17/17), LlamaIndex 0.10.0/0.14.25 (42/48), Haystack
2.0.0/3.4.1 (12/12), on Python 3.11/3.14 respectively. CI pins those versions and
adds the Pydantic AI cases. Local checks reuse the existing server binary whose
SHA-256 is recorded; final exact-head CI must build and test the server itself.
The locked Python 3.12 importer group passes 173 cases with one opt-in embedded
Weaviate skip (54.86s); it uses local stores and fixture HTTP servers. The optional
fast transport passes 31 cases on Python 3.14. Workflow lint, 214 pinned-action
checks, and diff whitespace checks pass. Final PR gate remains pending.

Evidence: [maintained-integrations-20261031/](maintained-integrations-20261031/)
contains the repeatable `scan_lock.py`, hash-bound complete OSV responses,
GitHub alert snapshot, primary advisory records, lock/build logs, contract logs,
and wheel check. `docs/SECURITY.md` corrects the older mem0ai 0.1.116 attribution:
GHSA-7587-9684-52r6 currently starts at 0.1.117, so that one attribution was wrong;
other legacy mem0 advisories were independent. Primary references:
[Python support](https://devguide.python.org/versions/),
[Pydantic AI slim installation](https://pydantic.dev/docs/ai/overview/install/),
[OSV query API](https://google.github.io/osv.dev/post-v1-querybatch/).


### Retained-text/query merge gate (2026-10-31)

PR #364 final head `9368fcb25a6ee87ca00491ff020d47ceb57e8ddc` passed
all 72 exact-head checks (success/skipped), including the required `check` from
app 15368. No unresolved review threads or open code-scanning alerts remained.
Merge `68aa40792698f4288c61f20c2e11bee215651c76` has a verified signature
and exactly the tested tree `ff2e97b3d19d3067bd032a084ef6e405ebe6412d`.
Only the owner-review requirement was bypassed after completion of all gates.
Evidence: [query-errors-20261031/final-gate.json](query-errors-20261031/final-gate.json).

### DSPy cache isolation and compatible dependency profile (2026-10-31)

Follow-up to the remaining DiskCache advisory: official DSPy 3.4.1 wheel source
shows lazy default-cache creation and separate disk/memory branches in
`dspy.clients.cache.Cache`. Its public `configure_cache(enable_disk_cache=False,
enable_memory_cache=True)` avoids importing/constructing DiskCache if applied
before the first cache/LM use. Keep this process-wide choice in application setup;
the WARM adapter does not silently mutate another application's global cache.
The runnable offline example and quickstart now select this profile.

A fresh-process regression forbids every DiskCache import, runs the real WARM
retrieval/DummyLM example, then exercises a cache miss and an in-memory write/read.
The cache round trip matters: DummyLM alone did not access the default cache, so
an initial example-only check was insufficient. The strengthened baseline fails
with `AssertionError: ['diskcache']`; the configured example passes. All 11 DSPy
retriever/isolation tests pass against the real local WARM server using Python
3.12, DSPy 3.4.1, LiteLLM 1.103.10, and Filelock 4.2.6, without paid calls.

Resolver tradeoff made explicit: current LiteLLM 1.106.6 requires Filelock <4;
the SDK constraint chooses compatible LiteLLM 1.103.10 rather than violating that
upper bound. This is a deliberate version rollback, not a latest-everywhere
upgrade. The lock also selects deprecated 1.3.1 (previous maximum 3.0.0), NetworkX
3.6.1 (3.7), and NumPy 2.4.6 (2.5.4). The refreshed importer group passes its 173
tests with this graph, and the added DSPy CI step pins the tested LiteLLM/Filelock
pair. Unconstrained consumer installations can still select vulnerable Filelock;
documentation provides explicit install constraints. No package override or
advisory suppression is introduced. This preserves a useful supported workflow
without claiming every optional framework path has been security-qualified.

DiskCache remains installed and the full-lock OSV advisory count is unchanged.
The mitigation covers the configured cache path; arbitrary direct DiskCache use,
untrusted program deserialization, and later global cache reconfiguration are
outside it. Chroma's four advisory matches and CrewAI's dependency remain open.
The separate integrations package's whole dependency graph is not yet audited.
Evidence: [dspy-cache-isolation-20261031/](dspy-cache-isolation-20261031/).


### CrewAI and SmolAgents native tool contracts (2026-10-31)

Inspection and actual framework calls reproduced two defects: CrewAI's WARM tool
created a client for every call without closing it; SmolAgents 1.26.0 rejected the
WARM tool constructor because its optional `top_k` parameter lacked the required
nullable input declaration. The latter failed before any search could run.
Eleven baseline cases failed (four unclosed CrewAI clients and seven SmolAgents
construction failures). The tool descriptions also incorrectly promised
"FHE-lite"/secure holographic storage; those claims are replaced with WARM search.

CrewAI now closes its owned client on normal return, service rejection, and
encoder exceptions. SmolAgents declares optional input semantics explicitly:
omitted/None `top_k` means three, but zero still reaches the service's 422 rejection
rather than being silently replaced by the default. Its reusable client has an
explicit close/context-manager lifetime. Repeated close is harmless. Tests call
real framework entry points (`BaseTool.run`, `Tool.__call__`) against a real local
service, including tenant isolation, authorization, and both error paths. The
client spy only tracks actual client lifetimes; service/encoder results are not
mocked to restate the implementation.

The SDK extras now require the actual verified framework versions, CrewAI 1.15.33
and SmolAgents 1.26.0. Current CrewAI limits Python to <3.14, so SDK documentation
states its 3.11–3.13 range. The registry package/version set is unchanged from the
previous complete OSV scan: 249 pairs, with only root dependency metadata changed
in the lock. This does not resolve Chroma advisories or qualify CrewAI's built-in
storage, provider integrations, or arbitrary agents as secure.

Validation: 11 cases passed on Python 3.11, 3.12, and 3.13; after adding the zero/one/
two boundary checks, the final 13-case suite passed on 3.12. Wheel/sdist builds,
uv 0.13.6 lock check, workflow lint, and diff checks pass. CI pins CrewAI 1.15.33,
SmolAgents 1.26.0, LiteLLM 1.103.10, and Filelock 4.2.6, with telemetry disabled and
only local encoders/service calls. Final PR/merge gate remains pending.
Evidence: [framework-tools-20261031/](framework-tools-20261031/).


### Archived Phidata adapter retirement and final SDK scan (2026-10-31)

The remaining Phidata extra resolves 2.7.10, whose upstream repository
[agno-agi/phidata](https://github.com/agno-agi/phidata) is archived. Inspecting and
instantiating WARM's actual adapter fails: `create`, `delete`, `doc_exists`,
`exists`, `name_exists`, and `upsert` remain abstract. Its `drop` also silently
succeeds without deleting anything, and inserted IDs derive from process-random
Python hashes when names are absent. The constructor failure is measured; those
additional defects are source observations, not runtime exploit claims.

Under the user's maintained-integration policy, remove the `phidata` extra and
adapter rather than implement a new lifecycle/storage contract for an archived
framework. No other production callers were found. Documentation removes the
advertised working connector and explicitly does not claim an Agno replacement.
The separate current integrations and historical audit/rename records remain.

The final SDK lock has 246 total packages / 245 registry package-version pairs,
compared with 458 / 457 before this work. This step removes only phidata, gitdb,
gitpython, and smmap; no registry package/version additions or upgrades occur.
A fresh complete OSV scan of this exact final lock still matches the same four
Chroma GHSA IDs and one DiskCache GHSA ID (10 IDs including PYSEC aliases).
Retirement is justified by archived support and broken contracts, not an invented
Phidata vulnerability. The final installed wheel confirms all four retired extras
and three modules are absent, with the Python 3.11 floor intact. Wheel/sdist and
lock checks pass. Final exact-head CI/merge gate remains pending.
Evidence: [phidata-retirement-20261031/](phidata-retirement-20261031/), including
hash-bound scan, exact lock delta, upstream archive metadata and constructor output.


### Publish the DSPy security floor to consumers (2026-10-31)

The initial remediation constrained the repository's Filelock only. A normal
consumer install of DSPy still chose LiteLLM 1.106.6 and vulnerable Filelock
3.32.7 unless the documented profile was explicitly supplied. Close that packaging
gap: both `warm-sdk[dspy]` and `writerslogic-warm-integrations[dspy]` now declare
`filelock>=4.2.6`; the SDK depends directly on maintained `dspy>=3.4.1` rather than
the old `dspy-ai` metapackage. These are ordinary published requirements, not
resolver overrides. Incompatible application pins now fail resolution.

Independent installs of the newly built SDK wheel's DSPy extra and the separate
integrations package's extra, without a repository lock or external constraints,
both select DSPy 3.4.1, Filelock 4.2.6, and compatible LiteLLM 1.103.10. Both real
retrievers construct. These are package-install/constructor checks; the earlier
11-case offline retrieval/cache suite verifies the same DSPy/LiteLLM/Filelock
versions. Disk caching must still be disabled before use as documented; this
floor does not remove the installed DiskCache advisory.

The final candidate lock now contains 245 total packages / 244 registry pairs:
only the redundant `dspy-ai` metapackage is additionally removed. The fresh
complete hash-bound OSV scan again reports four Chroma GHSA IDs and one DiskCache
GHSA ID. This supersedes the preceding 246-package snapshot for final release
assessment. Build and lock checks pass; final exact-head CI/merge remains pending.
Evidence: [dspy-consumer-constraints-20261031/](dspy-consumer-constraints-20261031/).

## Maintained Semantic Kernel dependency security (2026-10-31)

The separate `writerslogic-warm-integrations[semantic-kernel]` adapter remains
implemented and usable; it is distinct from the retired legacy SDK extra.
Fresh resolution selected Semantic Kernel 1.46.0 → OpenAPI Core 0.19.5 →
Werkzeug 3.1.1. A complete OSV scan of 100 registry package/version pairs matched
four GHSA IDs (eight IDs including aliases), all in Werkzeug's Windows
`safe_join` handling. This is dependency advisory evidence, not a demonstrated
exploit in WARM's Rust server or the vector-store adapter.

The published extra now requires `werkzeug>=3.1.9`. The resolver selects OpenAPI
Core 0.19.4, which satisfies Semantic Kernel's `>=0.18,<0.20` requirement, instead
of overriding 0.19.5's `<3.1.2` cap. The cap's rationale is documented in upstream
[issue 938](https://github.com/python-openapi/openapi-core/issues/938); the
[0.19.4…0.19.5 changes](https://github.com/python-openapi/openapi-core/compare/0.19.4...0.19.5)
are archived for review. No package metadata or advisory is suppressed. This
tradeoff validates WARM's vector-store workflows, not every OpenAPI or Django
integration available through Semantic Kernel.

Twelve real-server tests pass with Semantic Kernel 1.45.0/Python 3.11 and
1.46.0/Python 3.14, including custom embedding generators and `Kernel.invoke`.
The first run exposed an existing stale `HmsConfig` reference (11 pass, one
NameError); it now uses the imported `WarmConfig`, with assertions unchanged.
Both boundaries are added to the existing live-server CI job. No paid model
providers or model downloads are used.

Wheel and sdist build; wheel metadata includes the extra-scoped security floor.
A fresh wheel consumer selecting the extra resolves Werkzeug 3.1.9/OpenAPI Core
0.19.4 without external constraints. Installing the bare package plus Semantic
Kernel separately does not activate an extra's requirements and still resolves
the vulnerable version; use the documented extra. A new hash-bound universal
lock scan reports zero matches across 100 registry pairs (103 total packages).
That snapshot does not establish vulnerability absence or cover other extras.
Workflow lint and all 214 action pins pass. Exact-head remote CI and merge are
still pending.

Evidence: [semantic-kernel-security-20261031/](semantic-kernel-security-20261031/)
contains resolution locks, OSV responses, upstream metadata, compatibility and
consumer logs, and the tested server binary hash. The server was built from the
separately validated PR #366 source tree; this change adds no Rust code.
