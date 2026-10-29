# Local evidence-program development check

This is a synthetic correctness check of the unpublished
[evidence research crate](../../research/evidence/RUN.md), using generated text,
supplied assertions and typed queries. It is an oracle-extraction control, not a
LongMemEval/Bemula run or an end-to-end QA score. No model, paid API or cloud
service is used. The production engine and its retrieval policy are unchanged.

Reproduce from the HMS checkout root with cached Rust dependencies:

```sh
export CARGO_TARGET_DIR="$PWD/.hms-local/evidence/target"
cargo test --offline --locked --manifest-path benchmarks/research/evidence/Cargo.toml
cargo run --offline --locked --manifest-path benchmarks/research/evidence/Cargo.toml -- demo
```

The [demo report](demo.json) retains the query, retrieved sources/claims, gaps,
retrieval steps, source/serialized packet budgets and logical proof witnesses.
It demonstrates payment versus attendance scope, plans versus actual events,
repeated event mentions, an amount recovered from another source, ambiguous
event identities, and abstention when the world budget is incomplete. Its
separate forgetting case cuts two alternative declared proofs while protecting
an independent preference, then checks deletion/reopen, exact-version replay
rejection and fresh-version relearning on actual temporary HMS stores.

The [verification record](verification.json) records 81 passing tests with the
optional `narrative-lens` bridge enabled, Clippy with warnings denied, a Rust 1.89
compatibility check, formatting and a strict documentation build. Runtime tests
used Rust 1.96 on macOS arm64. That original development run used an uncommitted local
sibling checkout for the optional bridge; its source snapshot was not archived.
The public crate now pins the bridge to the public Git revision recorded in
its manifest and lockfile. A [fresh public-pin verification](verification-public-pin.json)
records the same 81 tests, strict Clippy, formatting and the Rust 1.89 check
against that pinned dependency. Rust API generation
was skipped by the documentation build's default configuration.

Resolved values apply only to the supplied observed assertions. Neither exact
quotation matching nor supplied event identities establish extraction quality
or factual truth. Scope repair enumerates the supplied annotation catalogue;
the packet budget does not bound total catalogue scans or store reads.
The proof check covers its declared finite monotone rules, without a guarantee
against unrestricted inference, future data collection or backup recovery.

No benchmark improvement, speed improvement or production readiness is claimed.
