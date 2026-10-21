# server-ann: the vertex graph index on the hms-server vector query path

Round-2 package. Nothing here was built, tested or run by the author (round rules); every
command below reproduces it. Expected numbers are inferences from the committed
measurements, labelled as such.

## 1. Why the index path was not used, and what `exact+index` did

- `deploy/bench/server_bench.py:166,187-192`: the `exact+index` arm is the `exact` arm plus one
  `POST /api/v1/index/maintain` after ingest.
- `src/bin/hms-server/documents.rs` (`maintain_index`, formerly lines 563-584) calls
  `HmsCore::train_document_index` (`src/core/engine/document_api.rs`, formerly 140-163), which
  builds `documents::dense::DenseIndex` (`src/core/documents/dense.rs:152-233`): a bounding-box
  tree over the `f64` embeddings (split on the widest coordinate, 64-row leaves). Its search
  (`dense.rs:311-390`) prunes a node only when the coordinate-wise upper bound of the dot
  product is below the current k-th score. At 384 dimensions each split tightens one of 384
  coordinates, so every node's bound stays close to 1 and nothing is pruned: the "index" visits
  every leaf and computes the same `f64` iterator dot products as the scan, through
  `Arc<StoredDocument>` pointer chasing, plus heap overhead. Hence recall 1.0 and 10-20% lower
  QPS than `exact` at every size in `benchmarks/results/server_v1.json` (686 vs 828 at 1k,
  795 vs 895 at 10k, 158 vs 178 at 50k, 16 clients).
- The vertex graph (`src/core/qgraph/vertex.rs`) had no caller in the engine's document path
  or in the server (`rg qgraph src/core/engine src/core/documents src/bin/hms-server` was empty;
  its callers were the research binaries in `src/bin/*bench*.rs` and `src/core/format/scan.rs`).
- Independently of the index, the generic vector path is slow per query:
  `documents::search::capture_query` (`src/core/documents/search.rs:208-237`) clones one
  `Arc<StoredDocument>` per stored chunk (50k contended atomic increments and decrements per
  query, shared cache lines across all worker threads) and allocates a slot table per query;
  `execute_chunks` (`search.rs:389-454`) computes BM25 norms and `f64` dot products per chunk.
  Measured: 37 ms per query at 50k x 384 with one client (0.74 us per row).
- `DocumentStore::invalidate_dense_index` (`src/core/documents/index.rs:159-165`) drops the
  bounding-box index on every insert, replace and delete, so it never survives a write.

## 2. What changed (additive)

Engine, `src/core/documents/vector_index.rs` (new; module head documents mechanism, layout,
consistency), re-exported from `src/core/documents.rs`:

- `pub trait VectorIndex { kind, rows, search(query: &[f32], k, ef) -> Vec<(row, cosine)> }`
  with two implementations: `ExactScan` (contiguous l2-normalised `f32` matrix, 8-lane dot,
  bounded top-k heap, deterministic ties by row) and `GraphIndex` (`qgraph::VGraph` built by
  `Graph::build_owned` + `VGraph::from_graph(.., residual = true)`, fetched ids re-scored
  exactly against the matrix). `VectorIndexKind::Scan` is the reserved third slot for
  `benchmarks/research/scan`; nothing implements it.
- `VectorSnapshot`: rows (`Arc<StoredDocument>` + chunk), matrix, member set, the exact scan
  and the optional graph, captured at one store generation. Held by `HmsCore.vector_index`
  (`VectorIndexSlot`: `RwLock<Option<Arc<VectorSnapshot>>>` + build mutex).
- Route per query: `VectorQueryOptions { mode: auto|exact|graph, ef, min_score }`; `auto` uses
  the graph when the snapshot has one (built when rows >= `graph_min_rows`, default 4096),
  the exact scan otherwise. `graph` without a graph index is `GraphUnavailable`.
- Deletes and replaces: the store bumps `generation` per mutation (`index.rs:159-165`); a
  stale snapshot validates each hit by `Arc` identity against the live store (dropped or
  replaced documents fall out: tombstone by identity, with over-fetch of up to `4k` rows), and
  documents not covered by the snapshot (inserted or replaced since the capture) are scored
  exactly in a delta pass over the live store (one walk of `store.documents` with a hash
  probe per document, a dot product per uncovered chunk). Cost bound: never more than the
  generic exact path; when fewer than `k` hits survive, the query falls back to the generic
  path. `rebuild_recommended` after `REBUILD_AFTER_MUTATIONS = 64` mutations (or when no
  snapshot exists over >= 128 embedded chunks).
- Maintenance: `HmsCore::rebuild_vector_index(graph_min_rows)`, `maintain_vector_index`
  (rebuild only when recommended). Hooked into `train_document_index`
  (`src/core/engine/document_api.rs`) and `maintain_indices` (`src/core/engine/query.rs`).
  `HmsCore::search_vectors(embedding, k, &options)` is the new engine entry;
  `search_documents` is untouched.

Server (`src/bin/hms-server/**`):

- `POST /api/v1/query`: new optional fields `index` (`auto|exact|graph`) and `ef` (1..=4096).
  Response shape unchanged (`score` is still the exact cosine; the graph route re-scores its
  candidates exactly). Filtered queries keep the generic path; `index: "graph"` with a filter
  is 400. `index: "graph"` with no graph built is 409 `index_unavailable`.
- Flags: `--query-index` (default `auto`), `--query-ef` (default 128), `--graph-min-rows`
  (default 4096); env `HMS_QUERY_INDEX`, `HMS_QUERY_EF`, `HMS_GRAPH_MIN_ROWS`.
- `POST /api/v1/index/maintain` builds the vector snapshot (graph above `--graph-min-rows`)
  before the bounding-box index; its reply and `GET /api/v1/stats` carry a new `vectorIndex`
  object (`rows`, `graph`, `current`, `staleMutations`, `rebuildRecommended`).
- Scheduled compaction (`tenants.rs::compact`) runs `maintain_vector_index` after a
  successful compaction, so the snapshot is rebuilt without an explicit maintain call once
  64 mutations accumulated (or on the first tick after ingest, when none exists).

Concurrency (deliverable 3): there is no exclusive lock on the query path. The handler takes a
work-slot permit (`--max-concurrent-work`, semaphore in `limits.rs`) and runs on
`spawn_blocking` (`request.rs:99-108`); `AppState::core` hands out an `ArcRwLockReadGuard`
(`tenants.rs:128-131`); the engine took `mutation_gate.read()` and `documents.read()`. The
measured 50k scaling (21.5 QPS at 1 client to 178 at 16, with load average 16-30 from other
processes) confirms queries ran in parallel; the cross-core cost was the per-query `Arc`
refcount traffic over 50k shared documents and the per-row `f64` work, not a lock. The new
path holds the store read lock only to read the generation and the text segment (and, when
stale, to validate hits), and searches an immutable shared snapshot with no per-row atomics.
`--max-concurrent-work` bounds in-flight queries; the committed protocol's 10 matches the 10
cores of the bench machine.

## 3. Commands (run from the repo root; private CARGO_TARGET_DIR per the project memory)

```sh
cargo fmt --all -- --check || cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features server,security -- -D warnings
cargo test --lib core::documents::vector_index
# The OpenAPI document is generated from the request/response types; the new query fields and
# the vectorIndex objects change it. Regenerate it once (additive diff expected in
# docs/server/openapi.json), then the full server suite:
HMS_UPDATE_OPENAPI=1 cargo test --features server --test server_api
cargo test --features server --test server_api --test server_ann --test server --test server_ops
```

Pass criteria of `tests/server_ann.rs`: graph recall@10 >= 0.95 against the exact route at the
default ef on 3000 x 16-d clustered vectors (`--graph-min-rows 512`); the exact route equals
a local f64 reference scan; a deleted id is absent from both routes before and after the next
maintenance; a replaced vector is top-1 with score > 0.9999 on both routes before and after;
`index: "graph"` is 409 before maintenance; `ef: 0` and `index: "bogus"` are 400.

Server benchmark (the protocol of `benchmarks/results/server_v1.json`, same flags; the
`exact+index` arm now exercises the graph at 10k and 50k and the snapshot exact scan at 1k):

```sh
cargo build --release --features server
QDRANT=$(command -v qdrant || find ~/.cache/hms-bench -maxdepth 4 -type f -name qdrant | head -1)
uv run deploy/bench/server_bench.py compare \
  --hms-bin target/release/hms-server --qdrant-bin "$QDRANT" \
  --sizes 1000,10000,50000 --concurrency 1,4,16 --duration 6 --hms-work 10 \
  --out benchmarks/results/server_v2.json
jq -c '.results[] | {system, variant, n, recall_at_10, index: .index.index_train_seconds, qps: [.load[].qps]}' benchmarks/results/server_v2.json
```

`compare` has no system filter: it always runs hms `exact`, hms `exact+index`, Qdrant HNSW,
Qdrant exact and SQLite+FastAPI (the Qdrant arm is built in; the harness needs the binary that
produced `qdrant 1.19.2` in server_v1). To bring the `exact` hms arm onto the snapshot path
without the explicit maintain call, add `--compaction-interval-seconds 1` to `Hms.command` in
`deploy/bench/server_bench.py:168-173` (the scheduled tick builds the snapshot after the first
ingest); that is a protocol change and should be recorded in the `method.hms_flags` string.

Expected (inference from server_v1, not a measurement): the 1k rows were HTTP-bound at
~830 QPS with one client and ~7,800 with 16; a graph query at 50k x 384 costs on the order of
0.1-0.3 ms against 37 ms today, so the `exact+index` arm at 10k and 50k should approach the 1k
numbers. Pass thresholds: recall@10 >= 0.95 at 10k and 50k; QPS at 16 clients >= 2x Qdrant
HNSW at the same size (server_v1: 2,458 at 10k, 1,899 at 50k), and >= 3x at one client (480
and 693). If recall is below 0.95, raise `--query-ef` (256, then 512; the accepted qgraph row
needed 736 on 1.18M glove-100 vectors) and re-run; each doubling of ef roughly halves graph
throughput before the HTTP floor. Record `index.index_train_seconds` (graph build at 50k x 384
with `BuildParams::default()` on the rayon pool; if it exceeds the harness's client timeout the
maintain call fails and the arm reports no index: lower `build_ef` in
`src/core/documents/vector_index.rs::build_vector_snapshot` in that case).

## 4. Root-crate APIs used (for fast compile-error triage)

- `qgraph::Graph::build_owned(Vec<f32>, dim, &BuildParams)` `src/core/qgraph/mod.rs:866`;
  `BuildParams::default()` `mod.rs:84`.
- `qgraph::VGraph::from_graph(&Graph, residual: bool)` `src/core/qgraph/vertex.rs:520`;
  `VGraph::len` `vertex.rs:1032`; `VGraph::searcher` `vertex.rs:1201`;
  `VSearcher::search(&mut self, &[f32], k, VSearchParams, &mut Vec<u32>) -> Result<usize, HmsError>`
  `vertex.rs:1417`; `VSearchParams { ef, rerank, .. }` `vertex.rs:31-58`.
- `DocumentStore` fields `documents`, `embedded_documents`, `generation` (`pub(super)`,
  `src/core/documents/index.rs:14-21`), `embedded_count` (`:145`), `get` (`:185`),
  `text_segment` (`:267`); `index::text_of` (`:103`).
- `StoredDocument`/`StoredChunk` (`src/core/documents.rs:177-209`), `DocumentResult` (`:162`),
  `normalize_embedding` (`:366`), `SearchOptions` (`:84`).
- `HmsCore` private fields `config`, `mutation_gate`, `documents` (`src/core/engine/mod.rs`).
- Server: `ApiError::{new, bad, internal}` (`src/bin/hms-server/error.rs:65-94`),
  `AppState.cfg`, `CoreGuard` deref to `HmsCore` (`tenants.rs`), `log_event!`.
- Crates already in the root manifest: `fxhash`, `parking_lot`, `serde`, `anyhow`, `schemars`.
  No dependency added.

## 5. Open risks

- Not compiled. Likely trouble spots: the `&dyn VectorIndex` coercions in
  `VectorSnapshot::index`; `PeekMut` assignment in `TopK::offer`; `VGraph: Send + Sync`
  (plain `Vec` fields as far as read; if it is not, the snapshot cannot be shared and the
  graph must move behind a `Mutex`, which would serialise graph queries).
- `train_document_index` now also maintains the vector snapshot; engine tests that call it on
  stores with >= 4096 embedded chunks pay a graph build.
- The stale-path delta walk is O(live documents) per query (about 1-2 ms at 50k); it exists
  only between a mutation and the next maintenance/compaction tick.
- The graph is rebuilt from scratch by maintenance (no incremental insert); build time at 50k
  x 384 is unmeasured.
