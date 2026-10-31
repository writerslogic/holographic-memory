# Role-sensitive episode retrieval admission experiment

Local research harness for the frozen [plan](../../results/vsa_episodes_v1/PLAN.json). No production integration or paid service. Results and the promotion decision belong in `benchmarks/results/vsa_episodes_v1/REPORT.md`.

## Reproduce

From the repository root, with Rust 1.99.0 and cached dependencies:

```sh
CARGO_TARGET_DIR=target/vsa-episodes-eval cargo +1.99.0 test --offline --locked --release --manifest-path benchmarks/research/vsa_episodes/Cargo.toml
CARGO_TARGET_DIR=target/vsa-episodes-eval cargo +1.99.0 build --offline --locked --release --manifest-path benchmarks/research/vsa_episodes/Cargo.toml
uv run --no-project --offline python benchmarks/research/vsa_episodes/run.py --dev
uv run --no-project --offline python benchmarks/research/vsa_episodes/run.py
uv run --no-project --offline python benchmarks/research/vsa_episodes/summarize.py
```

The runner refuses to overwrite existing results. Archive the complete result directory before a deliberate replication. Development seed 11 is separate from held-out seeds 101, 211, 307. Evaluation is capped at 600 seconds; compilation is outside the run. Do not tune parameters from evaluation results.

## Contract and implementation

Each document has canonical triples, tenant, half-open validity interval, active flag, version, UTF-8 source, and exact byte spans. Queries contain one or two partially or fully specified triples. Every clause must match within a single fact; two clauses must match within the same document. Verification returns source quotes and current versions. Canonical annotations are supplied equally, with no NLP extraction or entailment claim.

Each corpus contains role-reversed document pairs with identical roleless token multisets. There are 128 entities and 16 predicates; two linked facts plus random distractors per episode. The six query classes are full triple, subject/predicate, predicate/object, two clauses, role reversal, and absent entity. Sixteen queries per class are reused across initial and mutated states, so changes test stale-answer rejection. Queries are deliberately bounded and synthetic, not a representative application distribution.

* Exact: sorted role/filler posting lists, shortest-list candidate intersection, then source verification. Role association is checked within a fact, avoiding false joins across facts.
* BM25: existing research implementation over roleless entity/predicate IDs, 32 eligible candidates, then the same verifier. This does not benchmark native HMS hybrid search or dense embeddings.
* VSA: actual `PhaseHVec` at 2,048 binary dimensions. Role-qualified atoms bind into each of seven nonempty role projections. Facts are bundled using production binary majority semantics, including zero-valued ties. Each projection is packed to 256 bytes. Query projections use the same codebook and native binding; popcount similarity is checked against native phase similarity. Eligible documents are ranked by summed clause similarity; 32 or 128 candidates are verified. All codebook allocations count.
* VSA fallback: try 32 candidates; if fewer than five results survive verification, run exact retrieval. Its retained memory and lifecycle cost include both indexes. The fallback never consults the oracle.

Sixteen replacements change a fact and increment source version; sixteen deletions deactivate documents. Exact postings update incrementally, VSA recomputes changed signatures, and the reference BM25 rebuilds because it exposes no update API. A canonical JSON snapshot is written/read and each index rebuilt. Every result is compared with its corresponding pre-reopen result. This tests harness lifecycle semantics, not the native storage journal, crash recovery, or fsync durability.

## Measurements and limits

Recall is verified relevant results divided by `min(5, number of eligible relevant documents)`, averaged only over nonempty queries. Empty-query output is checked separately through the safety-error counter. `by_class: null` means no positive query in that class, not perfect recall. An independently coded exhaustive relational oracle checks exact retrieval and output eligibility. Source span/text/version checks supplement relevance checks.

Query timings include query encoding, eligibility, ranking, verification, returned quote allocation/deallocation, and any fallback. One warmup and three measured sweeps alternate method order. Report p50/p95, not a tail SLA; 96 queries per phase and uncontrolled host load limit timing precision. Allocation accounting itself adds instrumentation overhead. Build/update/reopen are single observations per case, not robust microbenchmarks; large effects can reject admission, but small differences require controlled replication.

The allocator counts retained requested heap bytes, including map/vector capacity, signatures, and codebook. Common canonical records are added to every arm; fallback includes both exact and VSA indexes. Initial and mutated retained states are reported. Stack, allocator bookkeeping, fragmentation, executable pages, and transient peaks are excluded. `/usr/bin/time -l` reports whole-process peak RSS, with multiple indexes and snapshots co-resident; it is not per-arm RSS.

Lifecycle gate ratios add shared canonical generation, mutation preparation, or snapshot read time to per-index build/update/rebuild cost. Synthetic canonical generation is common preparation, not a model of external ingestion. Snapshot writing is shared and reported separately; it is not an fsync durability benchmark. No claim of measured concurrency, million-document scaling, arbitrary variable binding, neural inference, or hardware acceleration.
