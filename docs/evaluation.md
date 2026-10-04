# Retrieval and workload evaluation

## Checked-in inputs and reports

The [2026-09-14 reports](../benchmarks/review-2026-09-14) were produced locally on an Apple M4,
macOS arm64, with the format-2 working tree. The Node evaluation records Node version, model
identity, preprocessing fingerprint, artifact revisions, dtype, and dataset SHA-256. Rust
reports record engine version, dimensions, corpus size, architecture, and operating system.

- [Operational relevance fixture](../tests/fixtures/relevance.json): 12 documents and 12 distinct labeled queries.
- [Paraphrase fixture](../tests/fixtures/semantic-paraphrases.json): 10 documents and 10 queries with less direct lexical overlap.
- [Paraphrase results](../benchmarks/review-2026-09-14/paraphrase-relevance.json): lexical, hybrid, optional reranked relevance and latency, plus projection fidelity.
- Concurrent sparse query/update reports at [1,200](../benchmarks/review-2026-09-14/workload-1200.json)
  and [10,000](../benchmarks/review-2026-09-14/workload-10000.json) stored vectors.

These are small curated regression datasets, not externally validated benchmark collections.
The queries are distinct from the stored passages, but this is not a blind evaluation. Do not
extrapolate semantic accuracy or capacity from them. Workload timings are short local runs
with normal OS caching; they are not throughput guarantees or comparisons with the previous
implementation. Training happens before the concurrent interval. Updates invalidate the ANN
cache, so the timed workload includes fallback retrieval.

## Run document relevance without a model

```sh
cargo run --release --bin hms-eval -- \
  --dataset tests/fixtures/relevance.json --dimensions 4096 --k 3 --assert-min-recall 0.90
```

The JSON output contains document-deduplicated recall@k, MRR@k, nDCG@k, and query latency. CI
runs this fixture alongside the existing synthetic self-recall floor. Add domain documents
and independent relevance labels to test application behavior; do not merely add exact
copies of stored text as queries.

## Run real local model evaluation

Build the native module with `npm run build` and install `@huggingface/transformers` separately.
Obtain model artifacts in explicit local directories. The recorded run used these ONNX q8
artifacts:

| Role | Model | Artifact revision |
|---|---|---|
| Embedding | Xenova/all-MiniLM-L6-v2 | 751bff37182d3f1213fa05d7196b954e230abad9 |
| Reranking | Xenova/ms-marco-MiniLM-L-6-v2 | a09144355adeed5f58c8ed011d209bf8ee5a1fec |

```sh
node scripts/evaluate-local-model.mjs \
  --model-dir /path/to/minilm \
  --revision 751bff37182d3f1213fa05d7196b954e230abad9 \
  --dataset tests/fixtures/semantic-paraphrases.json \
  --reranker-dir /path/to/reranker \
  --reranker-revision a09144355adeed5f58c8ed011d209bf8ee5a1fec \
  --output /tmp/paraphrase-results.json
```

Omit both reranker options to evaluate embeddings alone. Model inference uses mean-pooled,
L2-normalized embeddings; query/document prefixes and dtype are included in the adapter's
schema revision fingerprint. The application must configure the store from `embedder.space`.
Reusing an artifact label after changing its files defeats revision checking, so keep local
artifacts immutable. See [Transformers.js local-model configuration](https://huggingface.co/docs/transformers.js/custom_usage).

The evaluation reports hit@1, hit@5, MRR@5, and p95 query latency for lexical and hybrid
retrieval, with a separate reranked result when enabled. Model loading and ingestion are
excluded from query latency; the first query is not separately warmed up. It also embeds the
same corpus into the sparse signed projection and measures its top-5 neighbor overlap against
exact dense cosine using distinct queries. That measures approximation fidelity, not semantic
relevance. Exact dense document retrieval avoids this sparse projection loss.

## Concurrent workload

```sh
cargo run --release --bin hms-workload -- --vectors 1200 --operations 300 --dimensions 4096
cargo run --release --bin hms-workload -- --vectors 10000 --operations 1000 --dimensions 4096
```

One writer replaces vectors and also deletes/reinserts every fifth ID while one reader queries
top-10 neighbors. Reports include p95/p99 latency, elapsed time, and final live count. For
capacity planning, increase the corpus and workload duration, exercise real embeddings and
metadata filters, monitor resident memory/disk use, and include compaction and maintenance.

## Distribution checks

`npm test` exercises the actual native public APIs across child-process restarts and checks
queue backpressure. `npm run test:package` packs the built application, installs the tarball
in a temporary project, and loads CJS, ESM, the native module, and the document adapter. CI
runs native/package checks on Linux, macOS, and Windows; local results establish only the
current machine's behavior. Publishing remains a separate release action.

`npm run test:types` compiles a consumer of the generated native declarations and document adapter under strict TypeScript settings, including invalid-input type checks.
