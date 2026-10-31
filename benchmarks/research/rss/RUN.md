# rss: f32 chunk embeddings, a snapshot without its own matrix, and the hypervector switch

Round-2 package. Nothing here was built, tested or run by the author (round rules); every
command below is for the mediator. Expected numbers are arithmetic and inference from the
committed measurements, labelled as such.

## 1. Measured problem and where the bytes were

`benchmarks/research/server-ann/server_v2.json`, 50k x 384 (`rss_mb_after_ingest`): warm
639 (`exact`) and 672 MB (`exact+index`); Qdrant 202-214 MB under load; SQLite+FastAPI
236-372 MB. Live-heap arithmetic of the pre-change layout at 50k x 384:

| component (before) | bytes | MB |
|---|---|---|
| chunk embeddings, `Option<Vec<f64>>` (`src/core/documents.rs`, `StoredChunk::embedding`) | 50k x 384 x 8 | 153.6 |
| vector snapshot matrix, a second `f32` copy (`vector_index.rs::capture_vector_rows`, built by the first vector query or maintenance) | 50k x 384 x 4 | 76.8 |
| graph build input, `matrix.to_vec()` (transient, freed with the builder) | 50k x 384 x 4 | 76.8 transient |
| `VGraph` (8-bit codes + 8-bit residual + 32 x u32 adjacency + scales; `qgraph/vertex.rs::from_graph` = `from_graph_with(g, 8, 8, ..)`) | 50k x ~904 | ~45 (estimate) |
| bounding-box index nodes, `f64` low/high (`dense.rs`, `exact+index` arm only) | 1563 nodes x 384 x 16 | 9.6 |
| hypervectors (`EntangledHVec`, 4096/256 = 16 active indices x 4 B + 32 B header) | 50k x 96 | 4.8 |
| BTree tables (documents, slots, owners, sources, metadata index, postings over 2 terms) | | ~30-40 (estimate) |
| journal pages: the journal is a segmented mmap arena (`src/core/storage.rs:67-77`, `Arc<Mmap>` read segments + `MmapMut` active segment); a committed document is JSON (`src/core/engine/mutation.rs:170`) of ~384 x 19 chars = ~7.3 KB of `f64` text per document, LZ4 framed; the pages written through the mapping are resident | 50k x ~4-7 KB | ~200-350 (inference) |

The sum of the live heap (~320-330 MB without the bbox index) leaves ~300 MB of the measured
639 MB, which the mmapped journal pages (~6 KB per document, linear in N, matching the gap at
10k as well: 183 measured vs ~95 live) explain. The breakdown script separates these.

## 2. What changed (additive; stores from either version open with either version)

- `src/core/documents.rs`: `Embedding(Arc<[f32]>)` replaces `Option<Vec<f64>>` in
  `StoredChunk::embedding` (`Option<Embedding>`). Values are normalised in `f64` at ingest
  (`normalize_embedding`, now generic over `f32`/`f64` input so the replay check in
  `mutation.rs:229-235` compiles unchanged) and rounded to `f32`. One method,
  `Embedding::cosine(&[f64]) -> f64`, widens each stored value and accumulates in `f64`
  (sequential, clamped): the generic path (`search.rs::execute_chunks`), the bounding-box
  index (`dense.rs::search`) and the snapshot re-score (`vector_index.rs`) all call it, so a
  chunk has one semantic score however it is reached.
  Wire form: the same `embedding` key, a JSON array of the `f32` values in shortest
  round-trip decimals (serde_json/ryu); legacy records hold `f64` arrays under the same key
  and both parse as "numbers rounded to the nearest `f32`" (serde_json `f32`
  deserialization parses as `f64` and casts, round-to-nearest). A chunk without an embedding
  serializes byte for byte as before (`"embedding":null`). No format or schema version
  changes, no new field: the previous release reads the new arrays as `f64`.
- `src/core/documents/vector_index.rs`: `ExactScan { rows: Arc<[Embedding]> }` scans the
  store's own allocations (one pointer per row); `VectorCapture` clones handles (reference
  counts), no matrix. The graph build assembles its contiguous `Vec<f32>` for the build only
  and drops it with the builder. The stale delta pass dots the stored `f32` rows directly.
- `src/core/documents/dense.rs`: node bounds are exact `f32` min/max (half the bytes);
  `upper` widens them and keeps the outward rounding over the `f64` dot.
- `src/bin/warm-server/config.rs`, `documents.rs`, `docs/server/configuration.md`:
  `--ingest-encode eager|deferred|lexical` (`WARM_INGEST_ENCODE`, default `eager`), validated
  at startup, passed per batch as `IngestOptions { encode }` to
  `WarmCore::memorize_documents_with`. The store's encoder in the server is the default text
  encoder (`tenants.rs:339-349` sets only `embedding_space`), so `deferred` is accepted
  (text is stored); with a `LocalEmbedV1` encoder the engine refuses `deferred` at ingest
  (`documents.rs::prepare_with_options`), reported as 500 by the batch endpoint.
- Compile fallout outside the package, kept minimal: `src/core/engine/memory_access.rs:87`
  (`to_f64()` for `DocumentParts.embedding: Option<Vec<f64>>`), `src/core/engine/mutation.rs`
  test (dropped the `Vec::capacity` assertion on embeddings; `Arc<[f32]>` is exact by
  construction), `src/core/documents/{search.rs,compact_term_tests.rs,dense_tests.rs}`.
- New: `tests/rss_embeddings.rs`, `benchmarks/research/rss/rss_breakdown.py`, this file.

Expected RSS arithmetic at 50k x 384 after the change:

| component (after) | bytes | MB | delta |
|---|---|---|---|
| chunk embeddings, `Arc<[f32]>` (+16 B Arc header each) | 50k x (1536 + 16) | 74.0 | -79.6 |
| vector snapshot: `RowRef` + `Embedding` handle per row | 50k x 32 | 1.6 | -75.2 |
| graph build input (transient only, unchanged) | | 0 resident | 0 |
| `VGraph` | | ~45 | 0 |
| bbox nodes, `f32` bounds (`exact+index` only) | 1563 x 384 x 8 | 4.8 | -4.8 |
| hypervectors with `--ingest-encode lexical` | 0 | 0 | -4.8 (and the encode time per chunk) |
| journal pages: `f32` shortest decimals average ~10 chars against ~19 | ~55% of before | -90 to -150 (inference) |
| tables, allocator retention | | unchanged | 0 |

Inference: 639-672 MB -> about 300-350 MB after ingest at 50k x 384 with the default flags,
about 290-340 with `--ingest-encode lexical`. That is below Qdrant's 1485 MB `rss_mb_after_ingest`
at 50k and above its 202-214 MB under load; the 180 MB target is not reachable by this change
alone by arithmetic. The remaining lever the breakdown isolates is journal residency
(`residual_mb`): if it is the ~200 MB the inference says, the next package is a binary
embedding frame in the journal (1536 B per document instead of ~4 KB of decimals) or
`madvise(MADV_DONTNEED)`/unmapping of sealed read segments after replay, neither of which
touches result scores.

## 2b. Round 2b: journal page release, graph transient, text tables

Measured on the integrated branch (`breakdown_f32_50k.json`, 50k x 384, RSS after ingest):
eager 342, lexical 246, lexical+snapshot 248, lexical+graph 338 MB; components: hypervectors
96, text tables 64, graph 84, embeddings 64 (3.5 B/value), snapshot 6.5, residual 177 MB.

(1) Journal pages. The arena maps every segment (`storage.rs`: `Vec<Arc<Mmap>>` sealed
segments, `MmapMut` active segment, 64 MiB capacity steps). Pages become resident when
`discover_offset` scans the active segment at open (every written page and the zero tail up
to capacity), when replay copies each frame out, and when `write_frame` writes a committed
frame through the mapping. Serving never reads them again (documents live in the store, text
in the text segment via `pread`), only the next replay or a compaction does, and both read
from the file. New `src/core/os_memory.rs` (declared in `src/core/mod.rs`) returns them:
- Linux: `madvise(MADV_DONTNEED)` on the shared file mapping (page-table entries dropped;
  page cache and file keep the bytes, dirty or clean).
- macOS: `madvise(MADV_DONTNEED)`/`MADV_FREE` only deactivate file-backed pages (resident
  until memory pressure, invisible to `ps`), so the pages are discarded with
  `msync(MS_SYNC | MS_INVALIDATE)` (dirty pages written first, resident copies freed, the
  next access refaults). Symbols are declared by hand (`libc` is an optional dependency);
  constants from `<sys/mman.h>` are given at each use.
- Hooks: end of `PersistentArena::new` (the whole active mapping after `discover_offset`);
  `write_frame` after `flush_range` for durable frames (the frame plus its end marker;
  non-durable compaction staging writes are left alone, so staging gets no extra syncs);
  `load_from_log` (`src/core/engine/mod.rs`) after each replayed frame via the new
  `PersistentArena::release_pages(offset, len)`; `replace_compacted` after the swap
  (`release_all_pages` over the published generation).
- Knob: `WARM_RELEASE_JOURNAL_PAGES` (`0`, `false`, `off`, `no` disable; default on), read
  once per process (`os_memory::release_journal_pages_enabled`). Not a server flag, so the
  flag/doc test is untouched.
- Correctness: every release happens after the bytes are on disk (`flush_range` MS_SYNC) or
  in the page cache; readers refault. A unit test (`os_memory::tests`) writes through a
  mapping, releases, reads the same bytes back and checks the file.

(2) Text tables (64 MB for 200-byte texts): no duplicate table. `StoredTerms`
(`packed_terms.rs`) is three exact boxes per chunk (~400 B at 30 terms, 20 MB at 50k);
posting lists (`postings.rs`) are varint `Vec<u8>` per term with `shrink_to_fit` on rebuild
(~5 MB); the text itself lives in the text segment (`engine/mod.rs:296` attaches it; chunk
text is not inline); the slot/owner tables are per chunk, not per byte. The rest of the 64 MB
is the journal share of the longer records (text plus the serialized term map, ~560 B more
JSON per document, ~20-25 MB of resident frame pages) and allocator retention, both covered
by (1) and (3). No code change; the `lexical_long_text` arm after this round shows what is
left (expected ~25-30 MB: terms 20, postings 5).

(3) Graph transient. `Graph::build_owned` keeps the `f32` matrix (`unit`, 76.8 MB) and
50k small neighbour `Vec`s until `drop(built)` after `VGraph::from_graph`; the matrix is a
single large allocation (returned to the OS by both allocators), the neighbour lists are
small chunks the allocator keeps. `build_vector_snapshot` now calls
`os_memory::release_free_heap()` after the drop: glibc `malloc_trim(0)`, macOS
`malloc_zone_pressure_relief(NULL, 0)`. Measured graph 84 MB against the `VGraph` arithmetic
of 45 MB (50k x (384 codes + 384 residual + 128 adjacency + 8)); the ~39 MB gap is the
retention this returns (inference; the breakdown's `graph_mb` is the check).

Expected RSS per arm at 50k x 384 after this round (inference from the measured components;
idle ~25 MB):

| arm | before | expected after | arithmetic |
|---|---|---|---|
| eager | 342 | ~165 | 342 - 177 journal |
| lexical | 246 | ~70 | 246 - 177 |
| lexical_keep_pages (new; `WARM_RELEASE_JOURNAL_PAGES=0`) | | ~246 | the before value, as the control |
| lexical_snapshot (after maintain) | 248 | ~72 | +6.5 snapshot and bbox |
| lexical_graph (after maintain) | 338 | ~115-125 | 70 + 45 VGraph + 0-10 retention; target < 200 |
| lexical_long_text | 310 | ~100 | 70 + 25-30 terms/postings |

If macOS `ps` RSS does not drop for the `lexical` arm (journal_pages_released_mb near 0),
`msync(MS_INVALIDATE)` did not discard the pages on this kernel; the fallback is to unmap
sealed segments after replay and re-map on demand in `read_slice` (`read_segments` becomes
`Vec<Option<Arc<Mmap>>>` with the path kept), which removes their RSS unconditionally, and
to shrink the active segment's capacity step (`SEGMENT_SIZE`) so the zero tail is small.

Commands for this round (after section 3's fmt/clippy/tests, plus):

```sh
cargo test --lib core::os_memory core::storage
cargo test --test store_lifecycle --test format_crash --test reliability --test documents
cargo build --release --features server
uv run --no-project python -I benchmarks/research/rss/rss_breakdown.py --warm-bin target/release/warm-server \
  --n 50000 --dim 384 --dims 384,128 --label f32-release --out benchmarks/research/rss/breakdown_f32_release_50k.json
uv run --no-project python -I benchmarks/research/rss/rss_breakdown.py --warm-bin target/release/warm-server \
  --n 10000 --dim 384 --dims 384,128 --label f32-release --out benchmarks/research/rss/breakdown_f32_release_10k.json
jq '.components, (.arms | map_values(.rss_after_ingest_mb))' benchmarks/research/rss/breakdown_f32_release_50k.json
```

Pass: `components.journal_pages_released_mb` >= 120 at 50k; `arms.lexical_graph.rss_after_maintain_mb`
< 200; `arms.eager.rss_after_ingest_mb` < 200; `graph_mb` <= 55; the store tests above pass
(replay after release reads the same frames). Then the server benchmark of section 3 with the
flag set through `WARM_INGEST_ENCODE=lexical`, recording `rss_mb_after_ingest` per size.

## 3. Commands (repo root; private `CARGO_TARGET_DIR` per the project memory)

```sh
cargo fmt --all -- --check || cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features server,security -- -D warnings
cargo test --lib core::documents
cargo test --test rss_embeddings --test documents --test document_dense --test embed_store --test ingest_encode --test format_golden --test format_layout --test memory_api
cargo test --features server --test rss_embeddings --test server --test server_api --test server_ann --test server_ops
cargo test --features server --bin warm-server    # every_flag_has_an_environment_variable_and_a_documented_row
```

Pass criteria of `tests/rss_embeddings.rs`: on 300 x 64-d documents and 20 queries, the
generic path and the snapshot path return identical ids and scores; every score equals the
`f64` cosine of the stored `f32` vector within 1e-12 and the cosine of the client's `f64`
vector within 1e-6; a reopen reproduces the scores; `EncodeMode::Lexical` and `Eager` give
identical vector results and identical text results. With `server`: `--ingest-encode lexical`
ingests and answers `query_vector` with the same exactness, `stats.vectors == 3`.

Known failing assertion, by design: `tests/server.rs:284`
(`scores_are_exact_cosine_and_documents_must_fit_one_chunk`) asserts `|score - cosine_f64| < 1e-9`.
The score is now the exact cosine of the stored `f32` vector; its distance from the `f64`
cosine is the rounding of the stored coordinates, about 2e-9 typical and 6e-8 worst case at
384 dimensions (`Var = sum(q_i^2 v_i^2) 2^-48 / 3`), so about half the comparisons exceed
1e-9. Apply before the server suite (the package owner did not edit `tests/server.rs`):

```sh
sed -i '' 's/assert!((gs - es).abs() < 1e-9, "{gid}: score {gs} vs cosine {es}");/assert!((gs - es).abs() < 1e-6, "{gid}: score {gs} vs cosine {es}");/' tests/server.rs
```

`tests/server.rs:748` (`< 1e-9` against 1.0 for a vector queried with itself) passes: the
cosine of a vector with its own `f32` rounding is `1 - O(2^-48)`.

Breakdown (both binaries; `--label` records which):

```sh
git worktree add /tmp/warm-main origin/main && (cd /tmp/warm-main && cargo build --release --features server)
cargo build --release --features server
# Baseline binary (no --ingest-encode flag): every arm ingests eagerly, so hypervectors_mb is
# absent and the embedding slope, snapshot, graph and residual components are measured.
uv run --no-project python -I benchmarks/research/rss/rss_breakdown.py --warm-bin /tmp/warm-main/target/release/warm-server \
  --no-encode-flag --n 50000 --dim 384 --dims 384,128 --label main --out benchmarks/research/rss/breakdown_main_50k.json
uv run --no-project python -I benchmarks/research/rss/rss_breakdown.py --warm-bin target/release/warm-server \
  --n 50000 --dim 384 --dims 384,128 --label f32 --out benchmarks/research/rss/breakdown_f32_50k.json
uv run --no-project python -I benchmarks/research/rss/rss_breakdown.py --warm-bin target/release/warm-server \
  --n 10000 --dim 384 --dims 384,128 --label f32 --out benchmarks/research/rss/breakdown_f32_10k.json
jq '.components' benchmarks/research/rss/breakdown_*_50k.json
```

Each 50k run ingests five to six times (one fresh server per arm; the corpus is generated in
pure Python, about 20 s per arm at 50k x 384 plus the server's ingest), so budget about
15-25 minutes per binary. `arms.eager.rss_after_ingest_mb` of the baseline run should
reproduce `server_v2.json` (639 MB at 50k, 183 MB at 10k) within the run-to-run noise of
that measurement; if it does not, the harness difference (6-decimal client values, `ps`
instead of psutil) is the first thing to reconcile before comparing the new binary.

Output schema (`warm-rss-breakdown/1`): `arms.<name>` = `{encode, dim, text_bytes, maintain,
graph_min_rows, rss_idle_mb, rss_after_ingest_mb, ingest_seconds, vectors,
data_dir_bytes_after_ingest, [maintain_seconds, vector_index, rss_after_maintain_mb]}`;
`components` = `{hypervectors_mb, text_tables_mb, snapshot_and_bbox_mb, graph_mb,
embedding_bytes_per_value_measured, embeddings_mb, residual_mb, arithmetic{...}}`.
Pass thresholds for the new binary at 50k x 384: `embedding_bytes_per_value_measured` in
4.0..=5.0 (was ~8 + 4 with the snapshot built; the slope arms do not build one); `hypervectors_mb`
>= 0 and <= 10; `snapshot_and_bbox_mb` <= 10 (was ~86); `arms.eager.rss_after_ingest_mb` <= 400.
If `residual_mb` is above 150, journal residency is confirmed as the next lever
(`data_dir_bytes_after_ingest` gives its upper bound).

Server benchmark (the protocol of `benchmarks/research/server-ann/RUN.md`; the comparison arm
sets the flag through the environment, which the harness passes through, so
`deploy/bench/server_bench.py` is unchanged; record the flag in `method.warm_flags` by hand):

```sh
QDRANT=$(command -v qdrant || find ~/.cache/warm-bench -maxdepth 4 -type f -name qdrant | head -1)
uv run deploy/bench/server_bench.py compare \
  --warm-bin target/release/warm-server --qdrant-bin "$QDRANT" \
  --sizes 1000,10000,50000 --concurrency 1,4,16 --duration 6 --warm-work 10 \
  --out benchmarks/results/server_v3_f32.json
WARM_INGEST_ENCODE=lexical uv run deploy/bench/server_bench.py compare \
  --warm-bin target/release/warm-server --qdrant-bin "$QDRANT" \
  --sizes 1000,10000,50000 --concurrency 1,4,16 --duration 6 --warm-work 10 \
  --out benchmarks/results/server_v3_f32_lexical.json
jq -c '.results[] | {system, variant, n, recall_at_10, rss: .rss_mb_after_ingest, qps: [.load[].qps]}' benchmarks/results/server_v3_f32*.json
```

Pass: recall@10 unchanged (1.0 exact, >= 0.95 graph) against `server_v2.json`; QPS of the
`exact` arm within 15% of `server_v2.json` at 16 clients (the exact scan now follows one
pointer per row instead of streaming one matrix; at 1.5 KB per row the prefetcher covers most
of it; if the 50k `exact` QPS drops more than 15%, the fallback is to keep a contiguous
`Arc<[f32]>` matrix in the snapshot and drop it from the store's `Embedding` only for
chunks the snapshot covers, which is a larger change). `rss_mb_after_ingest` at 50k <= 400
(both arms), lexical <= eager.

## 4. Root-crate APIs used (for fast compile-error triage)

- `serde` with `rc` (Cargo.toml:24); `Embedding` implements `Serialize`/`Deserialize` by
  hand (`collect_seq`, `Vec<f32>`), so no reliance on serde's `Arc<[T]>` impls.
- `EntangledHVec::from_dense(&[f32], usize)` `src/core/entangled.rs:131`; `DEFAULT_RHO_DENOM = 256` (`:12`).
- `qgraph::Graph::build_owned(Vec<f32>, dim, &BuildParams)` `src/core/qgraph/mod.rs:866`,
  `BuildParams::default()` (`:81`, degree 32); `VGraph::from_graph(&Graph, bool)`
  `src/core/qgraph/vertex.rs:520`; `VSearchParams`, `VGraph::searcher`, `VGraph::len` as before.
- `DocumentStore` fields `documents`, `embedded_documents`, `generation` (`src/core/documents/index.rs:14-22`),
  `embedded_count` (`:154`), `get` (`:235`), `text_segment`; `index::text_of` (`:112`);
  `ChunkWithText::serialize` (`:90-109`) is unchanged and emits the new form through
  `Embedding: Serialize`.
- `WarmCore::memorize_documents_with(Vec<DocumentInput>, &IngestOptions)` `src/core/engine/document_api.rs:77-81`;
  `search_vectors` (`:178`), `rebuild_vector_index` (`:265`); `WarmCore::new(u32, Option<String>, Option<WarmConfig>)` `src/core/engine/mod.rs:116`.
- `EncodeMode: FromStr` `src/core/documents.rs:109`; `IngestOptions` (`:125`).
- Server: `Config` (`src/bin/warm-server/config.rs:80`), `AppState.cfg`, `ApiError::internal`
  (`error.rs`), `tests/server_support::{Server::open, Request::{req, tenant, json, send}, embedding, DIM}`.
- Crates: none added.

## 5. Open risks

- Not compiled. Likely trouble spots: deref coercions `&Embedding -> &[f32]` at generic call
  sites (`normalize_embedding::<T>` in `mutation.rs:235`, `from_dense` in `documents.rs`);
  `Arc<[f32]>: FromIterator` in `Embedding::from_f64`; `Vec<f32>::deserialize` turbofish
  form; the `axis` closure capture in the `vector_index.rs` unit test; the graph build on
  80 x 16-d rows in that test (if `Graph::build_owned` needs more rows or dims, drop the
  `1` from `[usize::MAX, 1]`; `tests/server_ann.rs` covers the graph end to end).
- `tests/server.rs:284` fails until its tolerance is 1e-6 (section 3): the cost of `f32`
  storage is exactly that rounding, and it is now documented on `Embedding`.
- `tests/golden/*` fixtures carry `f64` embedding arrays; they parse through the same
  `Embedding` deserializer. If a golden expectation compares semantic scores exactly to the
  `f64` cosine, it needs the same 1e-6 tolerance (`rg score tests/format_golden.rs
  tests/format_common/mod.rs` found no score comparison).
- The exact-scan QPS risk and its fallback are in section 3.
- `MemoryDocumentStore::export` (`memory.rs:196`) and the journal now write `f32` decimals;
  `export_import_round_trips_search_results_exactly` compares a round trip of the new form
  with itself and should pass; a fixture of the old form is not byte-identical after a
  rewrite (compaction), which is the expected effect of the feature, not a format change.
