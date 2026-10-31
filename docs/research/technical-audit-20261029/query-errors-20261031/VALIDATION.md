# Query and deferred-encoding text-cache error propagation

Baseline source: `67f3b5701a5d20abc93657aa46e99901314c4897` from PR 364,
including its cache checksum protection. The branch then fast-forwarded to
`3b5b61ddb2002523b790cba12ea7504e019c914d` (test/evidence correction only).
Local macOS arm64, Rust 1.99.0, debug tests; no paid services or model downloads.

Four baseline probes each triggered both a same-length UTF-8 cache change and a
truncated record. Text queries, vector snapshot queries, phrase fallback, and
captured queries all panicked at `text_of(...).expect(...)` instead of returning
their existing `Result` error. Four failing tests / eight caught query panics are
retained in `query-baseline.log.gz`; `baseline-repro.patch.gz` adds just the probes
to baseline. Reproduce with `cargo +1.99.0 test --locked --features security --lib
queries_report_cache_errors_without_panicking -- --nocapture`.

Ranked text/vector hydration now returns `Result`; a scrubbed capture remains
`Ok(None)` and triggers the existing coherent retry, while I/O or checksum failures
propagate as errors. Phrase fallback uses fallible chunk text. In-memory document
search forwards the same result. Error paths do not return partial hits or turn
corruption into missing text. No public successful-result shape, durable format,
or locking protocol changes. Existing get/list and memory-view
helpers still have infallible text reads; this is not universal I/O recovery.
The existing query hydration paths still hold read guards while reading text;
removing those requires separate snapshot-identity/retry validation, especially
for canonical segment replacement and phrase fallback. No additional locks or
longer critical sections were introduced here.

The first implementation failed to compile because three additional callers/tests
still expected a vector; their enclosing implementations were inspected and
updated. This was a development error, not a product regression. The paired
single-document candidate then passed all four probes. Final probes include one
healthy and one damaged hit, verifying errors instead of partial query success;
restoring cache bytes makes the same live engine answer correctly again.

Eleven snapshot tests pass, including the expanded probes, retained snapshots,
concurrent replacement/deletion, canonical rebuild failure, filtered hybrid
queries, compaction and reopen. Broader document, Clippy, HTTP and exact-head CI
checks are pending. Commands:

- `cargo +1.99.0 test --locked --features security --lib document_snapshot_tests`
- `cargo +1.99.0 test --locked --features security --lib core::documents`
- `cargo +1.99.0 clippy --locked --features server,node-api,security --lib --bin warm-server --tests -- -D warnings`
- `cargo +1.99.0 test --locked --features server,security --test server_api -- --test-threads=1`

No latency, throughput, RSS, model-quality or competitive-advantage claim follows
from these checks. Existing historical lock behavior remains an architectural
limitation, not a newly measured bottleneck.


## Deferred encoding and durable error outcomes

Two additional baseline tests each exercise truncation and same-length corruption.
They ran after the query-only fix, before changing deferred encoding or fingerprint
propagation, and reproduced four panics. The independent deferred reproduction
patch applies to baseline 3b5b61dd; that extracted patch has not separately been
rerun on a clean checkout. Unlike query corruption, installation can fail after
its encoding transaction is durable: the baseline replay assertions confirmed the
journal still installs the correct encoding after reopening.

`document_fingerprint` now distinguishes absent text from read failure with
Result<Option<u64>>. Preparation propagates text/fingerprint errors before writing;
installation propagates its error through the existing post-commit wrapper.
Both final tests pass, with both fault kinds: pre-commit journal checkpoint and
revision stay unchanged, and text repair permits one successful encoding followed
by an idempotent zero-work retry. Post-commit failure returns CommittedMutationError,
advances the revision and journal checkpoint, leaves the live document pending,
and reopens into the encoded state using the original journal text. Recovery does
not repair the damaged cache manually; opening rebuilds it. A repeated encode
then performs zero work. No transaction format or fingerprint algorithm changed.

The legacy infallible raw-vector `WarmCore::query` remains best-effort: it logs
catch-up errors and can omit pending document vectors. Its contract/binding
migration is separate; this change does not establish strict completeness for
that API. Memory views and legacy get/list reads also remain outside this fix.


## Final local gate

The expanded implementation passes 11 snapshot tests, 119 document tests
(one deliberately ignored latency benchmark), two deferred fault/replay tests,
four ingest/encoding/postings integration tests, strict Clippy including tests
(server/node-api/security), and 20 serial HTTP API tests. Scoped rustfmt and
`git diff --check` pass. Final logs are distinct from earlier runs. These changes
are being integrated into PR 364 before merge; exact-head CI is still required.
No SDK method or wire schema changes in this follow-up; PR 364's SDK and schema
checks will rerun in CI against the combined source.
