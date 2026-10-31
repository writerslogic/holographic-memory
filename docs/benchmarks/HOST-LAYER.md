# Host-layer and write-path costs

Load-independent counts used to decide which host-layer changes to keep. Latencies and profile
shares from the same work were taken on a shared, loaded machine and are development
measurements: they are in PR #245, not here. All counts are at `origin/main` 8fddc13f.

## Memory API writes: allocations per call

`cargo run --release --example memory_write_bench -- <remembers> <forgets> 100 <vocabulary>`
runs on a durable store of 4,096 dimensions:

- `<remembers>` single-fact `remember` calls;
- as many facts again through `remember_many` in batches of 100;
- `<forgets>` single-id `forget` calls.

Each fact is 24 words drawn with a skewed rank distribution from `<vocabulary>` pseudo-words.
Counts come from a counting global allocator local to the example.

| Facts stored | Vocabulary | `remember`: allocs / bytes per call | `remember_many` (100): allocs / bytes per call | `forget`: allocs / bytes per call |
|---|---|---|---|---|
| 20,000 | 4,096 | 208 / 36 KB | 15,289 / 5.0 MB | 212 / 359 KB |
| 20,000 | 24 | 175 / 28 KB | 9,041 / 1.4 MB | 213 / 7.8 MB |

How the bytes per `forget` grow with the store (300 forgets):

| Facts stored | Vocabulary 4,096 | Vocabulary 24 |
|---|---|---|
| 2,000 | 46 KB | 678 KB |
| 10,000 | 187 KB | 3.9 MB |
| 20,000 | 359 KB | 7.8 MB |

The bytes allocated per `forget` grow linearly with the number of stored facts that share the
deleted fact's terms. The cause is `DocumentStore::remove` → `PostingLists::remove`, which
decodes and re-encodes the whole posting list of every term in the deleted document.

The other fixed cost of `forget` is two drive-cache flushes per call:

1. the delete commit's journal flush;
2. `sync_text_segment`, which makes the scrubbed text durable before `forget` returns.

`remember` costs one journal flush. The memory API's own write code (planning, stamps, the
supersession index) is a negligible share of either call.

Both remaining `forget` costs are in the document store and the engine, not the memory API:

- **Posting-list rewrite.** A posting-list delete that does not rewrite the list (a tombstone,
  or removal inside one block) would remove the O(list) work.
- **Second flush.** Zeroing the text before the journal flush would let one drive-cache flush
  cover both. That changes the crash-ordering argument behind the forget guarantee, so it needs
  that argument made first.

The deletion certificates in `benchmarks/research/forget` (`DeletionCertificate` over a
counter bundle) are research code that `Memory::forget` does not issue, so the certificate
cost per item does not apply to this path.

## Host-layer changes considered and not kept

- **Reusing searchers.** `VGraph::searcher()` allocates visited marks of one word per vector.
  Reusing them in Python `batch_query` (four searchers per thread per call) or in warm-server
  `vindex::search` (one per request) saves at most the first-touch page faults of those marks.
  That is a few percent of one query at 100,000 rows, and well under 1% of a 10,000-query batch.
  Pooled searchers would also keep n words each alive after the call.
- **Serializing the `POST /api/v1/query` response directly instead of through
  `serde_json::Value`.** `serde_json`'s `preserve_order` feature is not enabled in the server
  build, so `Value` objects serialize with sorted keys. Direct serialization would emit
  declaration order and change the response bytes.
- **Zero-copy numpy views in `batch_query`.** `query` and `batch_query` copy the queries while
  holding the GIL, so that another thread cannot change the array during the GIL-released
  search. The copy is linear in the input and small next to the search.
