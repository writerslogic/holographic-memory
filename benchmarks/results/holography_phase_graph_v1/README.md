<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Directed phase-graph retrieval

The old encoding added the same entity phase in the subject and object slots.
Consequently `(alice, parent_of, bob)` and `(bob, parent_of, alice)` had exactly
the same stored trace. This was a structural ambiguity, independent of dimension
or cleanup strength. The regression also showed an incoming-edge query returning
the outgoing neighbor instead.

Subject and object now use distinct deterministic hash domains. Binding remains
integer phase addition. Stored events and their commitments retain their format;
the experimental histogram is in-memory state and is recomputed under the new
encoding. Event replay and SHA-256 commitment checks pass.

Scoring hashes each identifier once, query ranking prepares the two known-role
phases once, and normalized scores derive the coincidence floor from the event
count rather than scanning the entire histogram. Candidate ranking still scans
every known entity. It now needs a temporary `4 * dimension` byte cue vector;
there is no added persistent candidate index or dense per-entity cache. Ties
retain the first entity, as before. An unknown query can still return the best
candidate: this API does not implement an abstention threshold.

## Regression evidence

[`before.txt`](before.txt) records the two expected failures at baseline
`12c80dec`; [`after.txt`](after.txt) records ten graph tests passing, including
the security feature. Cases cover reversed edges, repeated-relation paths,
background interference, integer replay, overflowing histogram dimensions, and
optimized ranking against exhaustive scores at phase-resolution boundaries.
The engine facade's existing test also covers a directed two-hop parent chain.

## Benchmark protocol

The final five-trial release-settings run is in
[serial/summary.json](serial/summary.json), with
[every observation](serial/cases.jsonl) and
[source/compiler metadata](serial/environment.json). Median retrieval-query
latencies (each query ranks all known entities) are:

| Facts | Baseline | Improved | Speedup | Baseline forward / backward / three-hop correct | Improved, each direction |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.761 ms | 0.084 ms | 32.77× | 13 / 13 / 0 of 32 | 32 of 32 |
| 512 | 22.974 ms | 0.692 ms | 33.20× | 13 / 14 / 2 of 32 | 32 of 32 |
| 2,048 | 91.705 ms | 2.746 ms | 33.40× | 13 / 17 / 1 of 32 | 32 of 32 |

All trials agree on the integer retrieval outcomes. These are deliberately
directed-chain fixtures that expose the old role ambiguity; they are not a
representative knowledge-graph accuracy estimate. Normalized triple scoring is
19.6–27.2× faster and insertion is 13.8–16.9× faster on these fixtures.

The [driver](../../research/phase_graph/bench.rs) and
[runner](../../research/phase_graph/run.py) compile the actual before/after graph
sources and their hash mixer with optimization level 3, LTO, and one codegen unit.
Both builds substitute `std::HashMap` for the membership-only `FastMap`; timed
query loops never access that map. This is an isolated graph benchmark, not
end-to-end HMS document retrieval.

Three chain fixtures use `(facts, dimension, phases)` of `(128, 512, 16)`,
`(512, 1024, 256)`, and `(2048, 1024, 256)`. Identifiers are canonical-record-style
strings, for example `episode/entity/00000000/canonical-record`; shorter strings
reduce the hashing work saved by this implementation. Each fixture samples 32
interior facts and tests forward, backward, and three-hop recovery. String
fixtures are constructed outside timing. Timings include query scratch allocation.
Each trial alternates before/after process order; summaries use trial medians.

The [initial run](exploratory/summary.json) found all 96 forward, backward, and
three-hop queries correct per direction in the new implementation across the
three fixtures; the original had frequent directional failures. That run occurred
during concurrent work and used optimization level 3 without LTO. It is retained
as exploratory evidence, separate from the final release-settings run.

```sh
uv run --no-project python benchmarks/research/phase_graph/run.py \
  --trials 5 --output ./phase-graph-reproduction
cargo test --locked --lib --features experimental,security core::phase_graph::tests
cargo test --locked --lib --features experimental phase_graph_relate_and_retrieve
```

Task-local serialization does not imply an exclusive host: unrelated Rust jobs
were observed. Raw samples and exact source/compiler metadata accompany each run.
