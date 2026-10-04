# Production readiness

## Supported surfaces

Rust `HmsCore` and Node `HolographicMemorySystem` expose persistent vector, document, and
structured knowledge operations. `holographic-memory/semantic` adds bounded ingestion, local
ONNX embeddings, hybrid retrieval, and optional local cross-encoder reranking. Python currently
exposes the quantized-phase substrate, not the document engine. The Rust MSRV is 1.89.

## Compatibility and migration

This implementation introduces **storage format 2 and embedding schema 2**. `store.json` binds
a store to its dimensions, text encoder (`multiscale-words-v1`), signed dense projection
(`signed-projection-v2`), similarity metric, meaning-memory mode, encryption mode, and optional
embedding model/revision/dimensions/normalization/metric. A mismatch fails at construction.
Configure the same embedding space and security settings on every reopen.

**Existing stores without an embedding schema cannot be opened by this engine.** Old unsigned
projections discarded information; a bit-level conversion cannot recover it. Keep the old store
and its original application version. Re-encode the original source documents into a new path:

```sh
cargo run --release --bin hms-admin -- reencode sources.jsonl ./store-v2 --dimensions 16384
cargo run --release --bin hms-admin -- verify ./store-v2
```

Each UTF-8 JSONL line is a `DocumentInput`, for example:

```json
{"id":"guide","text":"Restore a verified backup.","sourceUri":"guide.md","version":"2","metadata":{"project":"alpha"}}
```

The CLI streams bounded lines, builds a temporary store, verifies its frames, and publishes the
new destination by rename. An invalid line leaves the destination absent and the source intact.
It refuses an existing destination. This command migrates source documents; reingest graph
relations, triples, and rules through their public APIs from your original source of truth.
For externally generated embeddings, use Rust `reencode_documents` with an `EmbeddingSpace`
configuration or Node `DocumentMemory.ingest` into a new store. Changing models requires new
embeddings, not relabeling the old vectors.

`hms-admin migrate SOURCE DESTINATION` makes a verified copy of an arena and its sidecars;
it does **not** change its text/vector encoding or upgrade an old embedding schema. Inspection
and verification understand legacy frames and the active `CURRENT` generation:

```sh
cargo run --release --bin hms-admin -- inspect ./store-v2
cargo run --release --bin hms-admin -- verify ./store-v2
cargo run --release --bin hms-admin -- migrate ./store-v2 ./store-copy
```

## Durability and concurrency

Every engine holds an exclusive `.hms.lock` for its lifetime. Use one instance per store and
share it across tasks; a second process or instance cannot open it. Persistent mutations acquire
one transaction gate, validate all operations, append one CRC-framed transaction, and update
live state while queries are excluded. Batches and a document replacement commit as a unit.
Relations, relation removals, relation types, triples, composition rules, and derived-fact
ownership are replayed alongside vectors and documents.

Compaction writes and verifies a new arena generation, flushes its files, then atomically
publishes a `CURRENT` pointer. Old segments are removed only after publication. Interruption
tests terminate child processes before publication, after publication, and after mapping;
reopening must see a complete generation. Compaction excludes persistent mutations and queries
for the snapshot/publication interval. It can therefore cause latency spikes on large stores.

Files are flushed before publication; directory entries are synced on Unix. Windows uses file
flushes and atomic replacement without directory fsync. Power-loss guarantees still depend on
the filesystem and hardware. The tests exercise process interruption, not a simulated disk
controller or power failure. Call `flush()` before controlled shutdown and measure recovery
with representative stores. Audit/provenance sidecars are separate from the arena transaction;
an error in those sidecars can be returned after the main data mutation has committed.

## Index maintenance

Vector updates and deletes modify affected postings and reuse slots. They invalidate trained
ANN caches when necessary. Inserts update a trained index when supported. Persisted ANN files
have checksums, shard topology, and an arena generation/offset checkpoint. Stale or damaged
caches are discarded; retrieval falls back to indices rebuilt from durable records.

Ingestion no longer trains indices inside the write call. `indexStatus()` reports recommendations.
Call `maintainIndices()` between ingestion batches or from a scheduled job. Native Node methods
run off the JavaScript event loop. Rust callers can use
`Arc<HmsCore>::maintain_indices_background()`. Training uses a snapshot outside the transaction
gate and publishes only if the data revision still matches. If concurrent writes changed the
store, the call returns a retryable maintenance error; ingestion remains committed.

## Documents and resource limits

`memorizeDocument` chunks UTF-8 text and atomically replaces any document with the same ID.
`searchDocuments` returns document IDs, versions, source URIs, metadata, passages, and byte
offsets. Metadata equality, source URI, and document-ID filters apply before ranking.
`deleteDocument` removes every owned chunk, including on restart. `memorizeFile` now reads a
bounded UTF-8 file as a document; use these document search/delete APIs for its output.

- Document text: 8 MiB; ID: 4,096 bytes; up to 4,096 chunks.
- Default chunking: 256 whitespace-delimited words, with 32 words of overlap.
- Each chunk: at most 64 KiB; chunkWords: 1–4,096; overlap must be smaller.
- Metadata: JSON object up to 64 KiB; source URI: 8,192 bytes; version: 1–1,024 bytes.
- Dense embeddings: finite, nonzero, schema-matching dimensions; at most 1,048,576 values per document.
- `memorizeBatch`: 1–4,096 items and 8 MiB of combined IDs/text, committed atomically.
- Serialized arena transaction: at most 50 MiB; overlap and JSON encoding count toward this limit.
- Native document results: k 1–1,000, candidate limit k–4,096. The JS adapter caps candidates at 1,000.
- `DocumentMemory` accepts at most 32 pending operations by default; a full queue rejects explicitly.
  Its async-iterable `ingest` waits for each document and applies backpressure.

Raw passages are stored by default. `storeText: false` omits passages but retains lexical terms,
metadata, source identifiers, and embeddings; it is not an anonymization control. Reranking
requires stored passages. `memorizeFile` uses the default text-storage policy; use explicit
`DocumentInput` settings when a different policy is needed.

Hybrid document retrieval currently scores eligible chunks with BM25 and exact dense cosine,
combines ranked candidates with weighted reciprocal-rank fusion, then optionally reranks them.
It scans eligible chunks and keeps dense embeddings in memory. NSG/IVF acceleration applies to
the sparse vector API; it does not make document semantic retrieval sublinear. Measure corpus
size, memory, and latency before selecting a deployment capacity.

## Security

The standard npm build includes the Rust `security` feature. Custom builds reject requested
security/provenance capabilities they cannot provide. `securityStatus()` reports active
capabilities and the configured embedding space. An encrypted store cannot silently reopen
as plaintext, and wrong-key errors are surfaced. See [PRIVACY.md](PRIVACY.md) for the precise
scope of encryption and differential privacy. Lossy encoding alone does not protect secrets.

## Reproducible validation

```sh
cargo test --locked --no-default-features
cargo test --locked --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo +1.89.0 check --locked --no-default-features
cargo bench --no-run --no-default-features
npm run build
npm test
npm run test:types
npm run test:package
cargo run --release --bin hms-eval -- --dataset tests/fixtures/relevance.json --dimensions 4096 --k 3 --assert-min-recall 0.90
cargo run --release --bin hms-workload -- --vectors 1200 --operations 300 --dimensions 4096
```

`hms-eval` emits relevance, rank, and latency metrics; `hms-workload` measures concurrent query
and update p95/p99. `scripts/evaluate-local-model.mjs` compares lexical and hybrid search with
actual local models, records model revisions and a dataset digest, and measures sparse
projection fidelity against dense-cosine neighbors. See [evaluation.md](evaluation.md).
Small curated fixtures and synthetic self-recall are regression signals, not general quality
or capacity guarantees. CI runs default/all-feature Rust tests, a Rust MSRV check, deterministic
retrieval floors, and native/packed Node tests on Linux, macOS, and Windows.
