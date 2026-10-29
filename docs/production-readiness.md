# Production readiness

## Supported surfaces

Rust `HmsCore` and Node `HolographicMemorySystem` expose persistent vector, document, and
structured knowledge operations. `holographic-memory/semantic` adds bounded ingestion, local
ONNX embeddings, hybrid retrieval, and optional local cross-encoder reranking. Python currently
exposes the quantized-phase substrate, not the document engine. The Rust MSRV is 1.89.

## Compatibility and migration

New stores use **storage format 4**. On Unix, 1 GiB logical segments start with 64 MiB
of mapped file capacity and grow in 64 MiB increments before appending a frame. Frame packing, logical
offsets, transaction encoding and the 50 MiB raw-frame limit match the 1 GiB layout.
Growth flushes the old mapping, extends and fully syncs the file, then remaps under the
existing writer guard. Other platforms retain full-size allocation before mapping.
Windows requires closing mapped views before extending a file; its allocation path stays
fixed. Writes crossing a capacity boundary on Unix pay that allocation
cost; this does not establish an unconditional latency improvement.

Existing supported formats 1–3 retain their recorded geometry through writes and compaction
(formats 1/2 use fixed 1 GiB segments; format 3 uses fixed 64 MiB segments). Opening them
does not upgrade their layout. Older binaries reject format 4 before changing persistent
data (opening still updates the ownership-lock PID). Storage health reports the current
layout version and mapped file capacity (not RSS). `hms-admin migrate` copies the source format;
re-encoding original documents into a new destination creates format 4.

**Embedding schema 2** is unchanged. `store.json` binds
a store to its dimensions, text encoder (`multiscale-words-v1` by default), signed dense projection
(`signed-projection-v2`), similarity metric, meaning-memory mode, encryption mode, and optional
embedding model/revision/dimensions/normalization/metric. A mismatch fails at construction.
Configure the same embedding space and security settings on every reopen.

**Existing stores without an embedding schema cannot be opened by this engine.** Old unsigned
projections discarded information; a bit-level conversion cannot recover it. Keep the old store
and its original application version. Re-encode the original source documents into a new path:

```sh
cargo run --release --features cli --bin hms-admin -- reencode sources.jsonl ./store-v2 --dimensions 16384
cargo run --release --features cli --bin hms-admin -- verify ./store-v2
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

The experimental `multiscale-capped-union-v1` encoder unions primitive vectors within each
text feature level before applying the existing cross-level frequency cap. It requires the
`experimental` feature and explicit selection; enabling that feature preserves the default
encoder. To evaluate it with original source documents in a separate store:

```sh
cargo run --release --features experimental,cli --bin hms-admin -- reencode sources.jsonl ./union-store --dimensions 16384 --text-encoder multiscale-capped-union-v1
```

In Rust, select `HmsConfig.text_encoder = TextEncoder::MultiscaleCappedUnionV1`, importing
`TextEncoder` from `holographic_memory::core::encoding`. Reopen with the same encoder;
the schema rejects a different encoder before loading arena data. Both text encoding and
document chunk encoding use the configured version. The lexical document search API and
supplied dense embeddings retain their existing scoring. Capped union remains a research
option; synthetic gains do not establish document relevance or a speed improvement.

`hms-admin migrate SOURCE DESTINATION` makes a verified copy of an arena and its sidecars;
it does **not** change its text/vector encoding or upgrade an old embedding schema. Inspection
and verification understand legacy frames and the active `CURRENT` generation:

```sh
cargo run --release --features cli --bin hms-admin -- inspect ./store-v2
cargo run --release --features cli --bin hms-admin -- verify ./store-v2
cargo run --release --features cli --bin hms-admin -- migrate ./store-v2 ./store-copy
```

## Durability and concurrency

Every engine holds an exclusive `.hms.lock` for its lifetime. Use one instance per store and
share it across tasks; a second process or instance cannot open it. Persistent mutations validate
and serialize their operations, reserve an exclusive writer lease, then synchronously append
one CRC-framed transaction. The writer queue holds its state mutex only for reservation and
release; queries can capture the prior live revision during the append. Reader exclusion covers
publication of the whole live batch and its revision, before acknowledgment. Compaction
checkpoints and other writers wait for the same lease, so they cannot pair an unpublished
journal tail with an older live snapshot. Batches and a document replacement commit as a unit.
An unacknowledged durable frame may replay after process interruption; successful acknowledgment
still follows both durable append and live publication.
Relations, relation removals, relation types, triples, composition rules, and derived-fact
ownership are replayed alongside vectors and documents.

Compaction writes and verifies a new arena generation, flushes its files, then atomically
publishes a `CURRENT` pointer. Old segments are removed only after publication. Interruption
tests terminate child processes before publication, after publication, and after mapping;
reopening must see a complete generation. Compaction captures a coherent snapshot while
excluding mutations, then rebuilds and verifies staging while mutations and readers proceed.
It shares immutable document owners and copies sparse vectors and other snapshot records.
Concurrent compactors serialize. Two catch-up passes copy the acknowledged journal outside
the mutation gate, retaining each original transaction and encrypted payload. Final tail replay,
flush, publication and index persistence exclude mutations; readers can proceed. This final
pause depends on outstanding writes and filesystem costs and has no fixed latency bound.
Snapshot memory and rebuild costs can still affect serving latency; no throughput improvement
has been measured for this implementation.

A mutation returns only after its journal frame is synced: `msync(MS_SYNC)` on Linux,
`msync` followed by `fcntl(F_FULLFSYNC)` on macOS (plain `msync`/`fsync` there leaves data in
the drive cache), `FlushViewOfFile` and `FlushFileBuffers` on Windows. The per-platform
guarantee is in [file-format/CRASH-CONSISTENCY.md](file-format/CRASH-CONSISTENCY.md).
On macOS the drive-cache flush dominates a single-document commit: a replace took 4.6 ms and a
delete 4.0 ms (median of 7 runs, Apple M4 internal SSD) against 0.51 and 0.23 ms without it
(`benchmarks/results/durable_commit_macos_v1.json`). Batch APIs commit many documents in one
transaction and pay the flush once.

Files are flushed before publication; directory entries are synced on Unix. Windows uses file
flushes and atomic replacement without directory fsync. Power-loss guarantees still depend on
the filesystem and hardware. The tests exercise process interruption, not a simulated disk
controller or power failure. Call `flush()` before controlled shutdown and measure recovery
with representative stores. Audit/provenance sidecars are separate from the arena transaction;
an error in those sidecars can be returned after the main data mutation has committed.
Those failures, publication failures after a durable append, and failed post-publication
compaction cleanup carry `CommittedMutationError`. HTTP reports `mutation_committed`;
Node errors carry the same code prefix. Inspect the resulting state and reopen or reconcile
it before retrying. This contract distinguishes durable data from completed auxiliary effects;
it does not supply an automatic sidecar repair service or exactly-once external delivery.

Memory policy writes compare the planned core revision inside the journal writer lease.
Independent wrappers and raw document writers therefore cannot commit stale conflict or
supersession decisions. Snapshot reads and policy operations have bounded retries and may
return `MemoryError::Concurrent` (`concurrent_mutation` over HTTP/Node). Retrying the same
fact ID/content is idempotent. Multi-batch forgetting can commit a prefix before contention;
retry the same scope to finish it.

Tenant restore publishes a durable intent before moving directories. A lifetime parent lock
survives the move; recovery processes the intent before a missing tenant directory can be
created. An interrupted uncommitted restore rolls back to the complete old store. These are
process-interruption guarantees, subject to the filesystem/hardware limits above.

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
combines ranked candidates by a weighted blend of min-max normalized BM25 and cosine scores (reciprocal-rank fusion on request), then optionally reranks them.
Unfiltered lexical queries use exact matching postings and cached collection statistics.
Filtered queries retain adaptive scan/postings scoring with exact filtered statistics.
Untrained dense retrieval scans eligible chunks and keeps embeddings in memory. Explicit
`train_document_index()` builds a separate exact f64 box tree for document semantic
and hybrid queries, including broad filtered searches; conservative bounds can skip vector dot products while preserving scores,
ordering and source fields. Rust `maintain_indices()` also builds an eligible document index;
Node exposes `trainDocumentIndex()` and `documentIndexTrained`. The server/SDK maintenance
route is documented in [SERVER.md](SERVER.md). NSG/IVF remain separate sparse-vector strategies.
Measure corpus size, memory, and latency before selecting a deployment capacity.

See [retrieval contracts](guides/retrieval-contracts.md) for exact versus approximate routing,
filter/phrase composition and explicit reasoning budgets, and [upgrade notes](guides/audit-fix-migrations.md)
for model and importer compatibility.

Document search prepares queries before acquiring the read gate, captures immutable document
owners and relevant postings/statistics under the gate, then scores and builds results after
releasing it. Model query variants share one capture, preventing revision mixing during fusion.
An overlapping query can return its captured revision after replacement or deletion; a query
captured after the acknowledged mutation sees the new state. Captures share source and embedding
buffers and retain their owners until scoring completes. They still allocate candidate and
posting references. A trained unfiltered semantic query pins the index without capturing every
dense row; hybrid capture also follows requested lexical postings. Filtered and fallback dense
capture remains linear in eligible chunks. This has correctness and mutation-progress
regressions, but no new end-to-end performance measurement.

The dense cache supports 128–250,000 embedded chunks and 1–4,096 dimensions, bounded by
64 Mi embedding coordinates, 64 MiB of box bounds and 80 MiB of estimated auxiliary memory.
It shares source/embedding buffers. Smaller, unsupported, stale and filtered cases use exact
scanning; equal-vector data can require an exhaustive leaf. Updates invalidate the cache;
restart rebuilds it explicitly. Training runs outside locks and publishes only a current
document generation, with idempotent concurrent publication. Worst-case search remains a
full scan. JSON float decoding uses the lossless roundtrip option so persisted f64 embeddings
and numeric metadata retain their values across restart.

`memorize_documents()` / Node `memorizeDocuments()` validates a bounded batch before one journal
transaction. Native batches require unique IDs and at most 4,096 documents/chunks, 8 MiB of
aggregate source fields, 8 MiB of metadata, 1,048,576 supplied or inferred embedding values,
and 32 MiB of prepared transaction JSON. HTTP batches preserve ordered last-write-wins IDs.
Validation failures leave existing documents untouched; durable replay uses the existing
transaction recovery protocol.

Model-stage installation and direct model search reject invalid weights and native result
limits. Fusion rejects duplicate chunk identities, conflicting source versions and nonfinite
scores; query-dependent ranking scores do not define source identity.

The `experimental` angular `qgraph::ivf::QuantizedIvf` is a separate index for supplied
unit/zero vectors and trained unit centroids, with 4/8-bit coarse codes and optional
4/5/8-bit residual codes. It validates serialized shape, checksum, offsets, ids and
codes before use. It is not connected to document semantic retrieval or the default
sparse index. Competitive development evidence lives in
`benchmarks/results/superiority_v3/ann_competition/`; local timings are ineligible
for superiority claims.

IVF now has conservative runtime code envelopes that can skip complete coarse-code evaluations
without changing selected lists, rankings or serialization. `scanned()` remains selected-list
population; `evaluated()` counts actual coarse-code row evaluations. The CLI reports both plus
`auxiliary_index_bytes`, which is separate from serialized `index_bytes`; per-worker query
mask buffers add two padded byte arrays. The combined serialized/cache budget remains bounded,
with unpruned fallback when cache allocation would exceed it. Synthetic parity and work-count
evidence is in `superiority_v3/ivf_routing_v1/`; public workload latency is unmeasured.

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
cargo run --release --features cli --bin hms-workload -- --vectors 1200 --operations 300 --dimensions 4096
```

`hms-eval` emits relevance, rank, and latency metrics; `hms-workload` measures concurrent query
and update p95/p99. `scripts/evaluate-local-model.mjs` compares lexical and hybrid search with
actual local models, records model revisions and a dataset digest, and measures sparse
projection fidelity against dense-cosine neighbors. See [evaluation.md](evaluation.md).
Small curated fixtures and synthetic self-recall are regression signals, not general quality
or capacity guarantees. CI runs default/all-feature Rust tests, a Rust MSRV check, deterministic
retrieval floors, and native/packed Node tests on Linux, macOS, and Windows.
