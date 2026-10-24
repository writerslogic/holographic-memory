# Exact episode context and a bounded phase-memory control

The production change preserves the original immediate-context candidate pass,
then uses remaining capacity to recover transitive premises by exact stored
session/version identity. Every added closure is atomic under source, original
round and serialized UTF-8 byte limits. It does not change candidate ranking,
the aggregation gate or the context's `unverified` status.

The native regression supplies the correct partial-cue anchor. Before the fix,
four returned turns omit a rare departure fact; after the fix, all nine original
turns are recovered, including the fact, with no unresolved links. The baseline
test was run against `12c80dec`'s `context.rs` and failed at `4 != 9`; see
`native-before.txt`. This is a concrete context-recovery failure, not a measured
end-to-end QA accuracy change.

Preserving candidate breadth matters: transitive closure can span a long
session's prefix. The regression with a nine-turn priority episode and a
69-turn episode retains both candidates' eight immediate sources under a
nine-source budget. Under the default 64-source budget it returns 13 sources:
nine from the first episode plus the second episode's four immediate sources.
The long episode's unresolved links remain explicit. Enriching before admitting
other candidates would have displaced useful evidence.

The [frozen protocol](../../research/holographic_context/PROTOCOL.md) separately
tests partial-cue episode lookup. It uses only generated data and the existing
compact context encoder. The oracle controls isolate context recovery from
locating the episode; they are not claims about HMS ranking quality.

All methods have the same 4096-byte, nine-source, five-round output limits.
Exact posting payload fits within the phase memory vector's payload budget in
every setting, even before counting the phase codebook. Held-out synthetic audit
results follow; each cell contains 48 queries over three seeds.

| Episodes | Complex dimensions | Phase, unique cue | Phase, split cues with temporal distractors | Exact intersection, each condition |
| --- | --- | --- | --- | --- |
| 64 | 512 | 28/48 | 14/48 | 48/48 |
| 64 | 2048 | 46/48 | 30/48 | 48/48 |
| 256 | 512 | 1/48 | 4/48 | 48/48 |
| 256 | 2048 | 11/48 | 12/48 | 48/48 |

Entries measure exact rare-source recovery after verifying the located episode
against every cue. Incorrect phase locations abstain; no method returns sources
from an incorrect episode/version. Across 384 configuration-query observations,
phase lookup recovers 146, exact intersection recovers 384, the transitive oracle
recovers 384, and the pre-fix immediate-context oracle recovers zero. The same 192
held-out queries run at both dimensions; these are not 384 independent examples.

Exact postings use 1340 bytes for 64 episodes and 5228 bytes for 256 episodes.
The phase vector alone uses 8192 or 32768 bytes; cached phase codebooks add about
1.16–17.63 MB of logical numeric payload. Shared token/episode identities and
Python object overhead are excluded from both representations. Regenerating
phase codes would reduce resident payload at additional computational cost.

Median locator time across held-out observations was approximately 1.46 µs for
exact intersection and 6.11 ms for phase lookup. These are Python mechanism
measurements on macOS/arm64 collected during concurrent work, not Rust
serving-performance estimates or controlled latency measurements. Output
contexts averaged about 780 bytes for the immediate oracle and 914 bytes for
the transitive oracle: more original evidence consumes more bytes while still
fitting the same budget.

The tested global FHRR sidecar fails its promotion gate. It is retained only as
research code. This bounded negative result does not rule out other phase
representations, compressed storage requirements, or noisy semantic cues. The
measured production improvement is exact premise recovery when capacity remains;
there is no claimed broad QA or serving-speed improvement.

Reproduce from the repository root:

```sh
uv run --no-project python benchmarks/research/holographic_context/probe.py --output benchmarks/results/holographic_context_v1/probe.json
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target" cargo test --locked --features conversation-memory --test conversation_context --test conversation_memory
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target" cargo test --locked --features conversation-memory --lib core::memory_api::conversation::context::tests
rustfmt --edition 2021 --check src/core/memory_api/conversation/context.rs tests/conversation_context.rs
git diff --check
```

`probe.json` records the frozen protocol, script and encoder hashes, corpus hashes,
per-query source digests, all aggregate counts and raw timings. `native-after.txt`
and `native-unit.txt` record focused native validation. Native tests cover exact
byte/source/round boundaries, missing deleted premises, duplicate anchors, replay,
cycle forgery, session/version isolation and competing long-session candidates.
