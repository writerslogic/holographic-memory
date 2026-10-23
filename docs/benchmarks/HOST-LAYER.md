# Host-layer and write-path costs (development measurements)

These are the measurements used to decide which host-layer changes to keep: the memory API
write side, the hms-server request path and the Python `VectorIndex` batch path. They are
development measurements, not a standings run: HMS only, no comparators, on an Apple M4
(10 cores, 32 GB, macOS) shared with other work (1-minute load average 9 to 34, recorded per
row). All runs are at `origin/main` 8fddc13f.

## Memory API writes: `remember` and `forget`

`cargo run --release --example memory_write_bench -- 10000 1000 100 <vocabulary>` runs on a
durable store of 4,096 dimensions:

- 10,000 single-fact `remember` calls;
- 10,000 more facts through `remember_many` in batches of 100;
- 1,000 single-id `forget` calls over the 20,000 facts.

Each fact is 24 words drawn with a skewed rank distribution from `vocabulary` pseudo-words.
Allocation counts come from a counting global allocator local to the example. The run was
single-threaded on the caller side, at load 28 to 34.

| Operation | Vocabulary | p50 µs | p95 µs | Allocations / call | Bytes allocated / call |
|---|---|---|---|---|---|
| `remember` (1 fact) | 4,096 | 3,990 | 10,440 | 208 | 36 KB |
| `remember_many` (100 facts) | 4,096 | 8,791 | — | 15,289 | 5.0 MB |
| `forget` (1 id) | 4,096 | 7,307 | 9,070 | 212 | 359 KB |
| `remember` (1 fact) | 24 | 3,974 | 10,112 | 175 | 28 KB |
| `remember_many` (100 facts) | 24 | 5,242 | — | 9,041 | 1.4 MB |
| `forget` (1 id) | 24 | 9,001 | 10,007 | 213 | 7.8 MB |

A wall-clock profile (samply, 4 kHz) of the same workload shows where the time goes:

- **`remember`**: the journal append and its `F_FULLFSYNC` take 72% of the time, 51% in the
  flush alone. Planning, stamping and the index update in `memory_api/write.rs` take under 1%.
  Write-side code in the memory API has no measurable cost to remove.
- **`forget`**: 62% is `DocumentStore::remove` → `PostingLists::remove`, which decodes and
  re-encodes the whole posting list of every term in the deleted document. The bytes allocated
  per call grow with the store: 1.6 MB at 2,000 facts and 12.6 MB at 10,000 facts with the
  24-word vocabulary. The other 32% is two drive-cache flushes per call: the delete commit's
  `F_FULLFSYNC` and `sync_text_segment`, which makes the scrubbed text durable before `forget`
  returns.
- **Forget certificates**: the per-item cost was not measured. The certificate exists only in
  the research crate `benchmarks/research/forget` (`DeletionCertificate` over a counter
  bundle); `Memory::forget` issues none.

Both remaining `forget` costs sit in the engine and the document store
(`src/core/documents/postings.rs`, the commit order in `src/core/engine`), not in the memory API:

- A posting-list delete that does not rewrite the list (tombstone, or remove within a block)
  would remove the O(list) work.
- Zeroing the text before the journal flush would let one drive-cache flush cover both, but
  that changes the crash-ordering argument. The second flush is what makes the forget
  guarantee hold, so it stays until such a change is made and argued.

## hms-server request path

A profile of 20,000 sequential `POST /api/v1/query` requests (10,000 rows, dimension 100,
`index: graph`, `ef: 64`, one keep-alive connection, load 9 to 12) splits server CPU samples
like this:

| Part | Share of samples |
|---|---|
| `vector_search` | 7.5% |
| of which the graph search (`search_inner`) | 6.2% |
| response as a `serde_json::Value` tree, then text | about 1% |
| request JSON parse | 0.6% |
| `get_document` per hit | 0.7% |
| tokio, hyper and the blocking-pool hop | the rest of the active samples; idle workers dominate the total |

No single host-layer item is a large share. Two options were measured and not kept:

- **Serializing `Matches` directly instead of through `Value`.** It would save about 1% of
  samples, and it would also change the key order of the response body (`Value` maps are
  sorted), so the response format would change.
- **Reusing searchers across requests.** Cost: see below.

Per-request latency in an hms-only A/B (p50 µs, 3 rounds): query 111 to 115, replace 3,940
to 3,980, delete 3,080 to 4,000, text search 69 to 122, memory `remember` 3,990 to 4,010,
memory `forget` 7,020 to 7,990. Replace, delete, `remember` and `forget` are one or two
`F_FULLFSYNC` each.

## Searcher reuse (Python `batch_query`, hms-server `vindex::search`)

`VGraph::searcher()` allocates visited marks of one word per vector. On the M4, minimum of
2,000 repetitions:

| n | create only | search, fresh searcher | search, reused searcher |
|---|---|---|---|
| 10,000 | 0.46 µs | 29.3 µs | 28.5 µs |
| 100,000 | 1.38 µs | 37.0 µs | 34.8 µs |

At 100,000 rows a fresh searcher adds at most 2.2 µs, about 6% of one query. Python
`batch_query` makes four searchers per thread per call, which is under 0.1% of a
10,000-query batch. Pooling them only pays for batches smaller than about four rows per
thread, and the pooled searchers would hold n words each after the call.

An interleaved A/B of a pooled-searcher `batch_query` (100,000 × 128, 4 threads) could not be
read at load 18 to 36: each build won one round. It was not kept. The per-request searcher in
`vindex::search` is about 2% of a 110 µs request and was left as is.

Zero-copy numpy views in `batch_query` were not tried. The query copy is about 5 MB for
10,000 × 128, about 1 ms against roughly 190 ms of search. `query` and `batch_query` copy
under the GIL on purpose, so that another thread cannot change the array during the search.

## Unmeasured until benchmark runs resume

- Linux delete latency with the vector-log sync skipped after a committed delete (PR #244).
- The re-measured server standing against Qdrant, Meilisearch and Typesense at N = 1k / 10k /
  100k.

Both use the server harness, for example
`./benchmarks/competitive/hms-bench run --suite smoke --mode server --only ann --rounds 3`,
with `HMS_BENCH_HMS_SERVER` naming the binary of each arm.
