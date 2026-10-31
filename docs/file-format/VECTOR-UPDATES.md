<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Persisted vector updates, version 1

## Structure and visibility

`UpdatableVGraph` owns a directory containing an immutable `WARMVGR01` vertex
index, an immutable full-precision row file, a checkpoint manifest, and an
append-only update log. Existing standalone vertex files remain readable and
unchanged. Creating an updatable index requires the original vectors: quantized
vertex codes cannot reconstruct them without accumulating error on rebuilds.
External ids are `u32`, excluding `u32::MAX`; dense graph slots are private.

A mutable delta uses the construction implementation's greedy search and Vamana
robust prune, including reciprocal links. A nearest-parent routing tree retains
one incoming path per allocated delta slot, using two u32 links per slot.
Reciprocal pruning can otherwise remove a vertex's last incoming edge, as the
repeated-replacement regression demonstrates. Routing-tree links survive pruning
and logical deletion until consolidation. When the beam covers the whole delta,
live rows are scored directly rather than maintaining a full sorted search pool.
Slots are never reused before consolidation. Base deletions use a bitmap; delta deletions retain routing slots
but remove their external-id lookup. Replace invalidates the previous slot and
inserts a new one in one operation. One id may occur only once between commits.
Updates become visible when their in-memory operation completes; durability
requires `commit`. Dropping without committing discards the pending batch.

With no updates, search calls the existing vertex search unchanged, with reused
per-thread scratch. `search_into` also reuses the caller's result buffer; dense
external ids need no translation. With updates it widens the base beam and candidate count by
the inverse live fraction, preserving routing through deleted slots,
filters deleted answers, doubles the beam and retries when too few live candidates
remain, searches the delta with the same compensation, and merges candidates using original
full-precision distances with external-id tie breaking. Original rows stay mapped
and are touched only for candidate reranking and consolidation. `k` is clamped
to the live count before allocation. Empty indexes are represented without a
vertex file because the standalone vertex format requires a nonempty graph.

[MERIT](https://arxiv.org/html/2607.29173v1), sections 3–4, motivates bounded
reverse-neighbor discovery, local MST repair, and versioned edge invalidation.
This design retains deleted slots as routing vertices instead of breaking those
paths. It therefore needs neither reverse edges nor versioned base edges. Local
base repair would add an overlay lookup to the serving loop and duplicate topology
ownership; it is deferred pending measurements of sustained high deletion rates.
This is an engineering tradeoff, not a claim of MERIT-equivalent update costs.
Consolidation removes routing tombstones. The default advisory policy recommends
it at 10% allocated delta slots or 20% base tombstones, with a minimum 1024 delta
slots; thresholds are configurable. Commit never hides a rebuild in its latency.
A consolidation either rebuilds the base graph over every live row or, with
`ConsolidationMode::Merge`, keeps the previous base graph: lists that pointed at
a deleted row take that row's live out-neighbours as candidates and are pruned
again (FreshDiskANN's delete consolidation), the delta rows are linked in by the
build's first insertion pass, and the build's second pass then searches again for
the delta rows, the base rows they link to and the rows that lost a deleted
neighbour. Upper layers and the entry are recomputed. Merging writes the same
generation files and publishes them the same way, so the on-disk format and
recovery are unchanged. It falls back to a rebuild without a base graph, for a
renumbered base or one with a screen or blocks, when the delta outnumbers the
surviving base rows, or once the rows inserted and deleted by merges since the last
rebuild (counted in memory from open) would reach the caller's `rebuild_after`
multiple of the live rows.

## Ownership and synchronization

A directory lock excludes other writable opens, including other processes.
Drop explicitly unlocks the file so a descriptor inherited during a concurrent
process launch cannot retain ownership after the index closes.
An atomic writer permit serializes insert/delete/replace/commit/consolidate;
concurrent writers receive `WriterBusy`. A reader/writer lock protects only
in-memory state. No lock guard is held during file I/O. Searches borrow one
consistent state for their CPU work; consolidation snapshots the state, releases
the guard, builds and writes files, and publishes the new mapped base with a
short write guard. Readers continue on the old state while rebuilding.
An ambiguous append, sync, or checkpoint publication failure requires reopening
before further writes. Searches can continue to observe the in-memory state.

## Publication and recovery

Each committed batch is one length-prefixed CRC32 frame. A frame is validated in
full, including unique ids and operation preconditions, before any operation is
applied. Replay uses the same insertion order and pruning as live updates.
Incomplete final headers or payloads are truncated to the previous frame boundary
on open; a complete frame with a bad checksum is a typed corruption error, even
at EOF. A batch gets one durable file sync using `durable_file` after append.
The directory entry for the log is durable before updates can be accepted.

Consolidation first commits any pending batch, then writes fresh generation
files named by the committed sequence. Files are synced before publication.
The manifest is atomically replaced and its parent directory synced. Its
checkpoint sequence makes old frames at or below that sequence redundant.
Only afterwards is the log atomically rotated to an empty log whose origin is
the checkpoint. Thus interruption before publication replays against the old
base; interruption after publication opens the new base and skips the old prefix.
A torn rotation exposes either complete log. Old generation files are removed
only after publication and rotation; mappings held by readers remain valid.

## Binary contract

All integers and floats are little-endian. Every header contains the byte-order
marker `0x0102030405060708`; big-endian hosts refuse these files. CRC32 is IEEE.
Unknown versions are refused as newer than supported, never interpreted as v1.
Limits are validated before allocation: dimensions 1..=2^20, ids below 2^32-1,
frame payload at most 64 MiB and at most 65536 operations. Non-finite vectors and
squared norms that overflow f32 are rejected. Checked arithmetic bounds lengths.

* `CURRENT`: `WARMVUP01`, order (u64), identity (16 bytes), sixteen u64 words,
  CRC32 of all preceding bytes. Words: dimension, metric (0 angular, 1 Euclidean),
  checkpoint sequence, degree, build ef, alpha f32 bits, seed, quantized-search
  flag, vertex bits, residual bits, aligned flag, reordered flag, id bytes,
  screen flag, block bits, reserved zero.
* `rows-<sequence>.bin`: `WARMVRW01`, order, dimension, count, metric, sequence,
  row offset (all u64), header CRC32 (u32), id-table CRC32 (u32): 64 bytes.
  Sorted external ids (`count` u32) follow. Rows start at the next 64-byte boundary;
  each is `dimension` f32 values followed by its CRC32. The header and id ordering
  and id-table checksum are validated at open; row checksums and finite values
  are validated on access.
* `updates.log`: `WARMVLG01`, order, identity (16 bytes), dimension, metric,
  origin sequence (u64), header CRC32: 60 bytes. The identity binds it to CURRENT.
* Batch frame: payload length u32, its bitwise complement u32, sequence u64,
  operation count u32, CRC32 u32, payload. The checksum covers the first 20 header
  bytes and payload. Sequences start at origin + 1 and increase exactly by one.
  Records: tag u8 (1 insert, 2 delete, 3 replace), external id u32; insert/replace
  additionally carry exactly `dimension` f32 values. No trailing bytes allowed.

The vertex format stays version 1. Golden vector directories and byte drift tests
cover CURRENT, row files and update logs independently of text-store formats.
Python and Node bindings are follow-up work.

## Verification and measurements

Acceptance includes randomized update sequences against exact kNN and a rebuilt
base, bytewise torn-tail recovery, checksum rejection, reopen equivalence,
consolidation interruption points, 50k-vector recall comparison, and concurrent
readers. Measurements and exact reproduction commands are recorded here as they
are completed; SIFT-1M timing runs only on Linux under the shared timing lock.

### Recorded evidence

`benchmarks/results/ann_updatable_x86_v1.json` contains the source revisions,
executable hashes, commands, dataset checksum and raw measurements from one
SIFT-1M comparison on the Intel Core Ultra 9 185H. Each measured child held
`~/warm-timing.lock`; compilation and graph construction used cores 12–21.
No SIFT-1M measurements were run on macOS.

Recall was identical at all 20 ef/rerank configurations. At target recalls
0.90/0.95/0.99, immutable-base QPS was 27,752/20,769/10,485 and updatable-index
QPS was 26,465/20,175/10,068. The base benchmark uses an aggregate timer; the
update and hnswlib workers time individual queries. These single-run differences
of 2.9–4.6% do not isolate synchronization cost from the timing boundary or noise.
An identically timed, 50-sample Criterion comparison on 10k 32-dimensional vectors
measured 24.97 microseconds before and 24.85 for the final clean updatable index
with a reused output buffer (-0.5%); the allocating API measured 24.78 microseconds.
The same immutable binary measured 23.66–24.97 microseconds across runs. The artifact
retains the earlier pairs, including 23.66/24.86 and 24.24/24.58 microseconds.
This variability prevents a speedup claim; the final pair shows no clean-path
regression. The one permitted full SIFT comparison was used before the final
fixes, so its QPS numbers are not a measurement of the final binary.

Mapped serving RSS from `/proc/<pid>/smaps_rollup` was 468.0 MiB before and
479.2 MiB after; hnswlib was 898.7 MiB. The page-cache-evicted open plus first
query took 198.9 ms. Replacing 10,000 ids including commit took 1.291 seconds;
deleting 10,000 ids including commit took 8.405 ms. The existing hnswlib adapter
measured 1.311 ms per replacement and 0.02969 ms per deletion, including its save
boundary. Its `save_index` has different durability semantics from WARM's synced log.

The update tests cover randomized insert/delete/replace sequences against exact
kNN and a freshly built base; committed reopen equivalence; every byte offset of
a torn final frame; corrupt checksums and headers; three consolidation publication
interruptions; torn unpublished vertex files; concurrent readers during I/O;
allocation limits; bounded manifest reads; inherited-lock release; id-table
checksums; reusable output buffers; delta routing reachability after replacement;
and dimension-safe scratch reuse. The 50k-vector test uses
16 dimensions, 100 queries, 500 deletes and 500 replacements, and bounds the recall
change after consolidation by 0.002. Golden-byte regeneration passes on ARM macOS
and x86 Linux.

Reproduce the full comparison on Linux, after preparing SIFT-1M with
`benchmarks/competitive/warmbench/h5_convert.py` and building the before and after
`public-bench` binaries in separate target directories:

```sh
python benchmarks/competitive/ann_updatable_measure.py \
  --data "$WARM_BENCH_DIR/ann-update/data" \
  --work "$WARM_BENCH_DIR/ann-update/measurement" \
  --before "$WARM_BENCH_DIR/ann-update/target-before/dist/public-bench" \
  --after "$WARM_BENCH_DIR/ann-update/target-after/dist/public-bench" \
  --python-hnsw "$WARM_BENCH_DIR/ann-update/hnsw-env/bin/python" \
  --before-commit 6685568 --commit "$(git rev-parse HEAD)" \
  --out "$WARM_BENCH_DIR/ann-update/ann_updatable_x86_v1.json"
```

The hnswlib environment uses Python 3.13, numpy 2.5.4 and hnswlib 0.8.0.
The artifact records the exact original invocation and the measured executables'
source provenance. Python and Node API bindings remain outside this change.

Reproduce the smaller Criterion comparison in each source checkout, using a
private target directory per checkout (the before checkout needs the static-only
benchmark source embedded in the result artifact):

```sh
CARGO_TARGET_DIR="$WARM_BENCH_DIR/ann-update/target-after" \
  taskset -c 12-21 cargo +1.96.0 bench --profile dist --bench vector_updates --no-run
CARGO_TARGET_DIR="$WARM_BENCH_DIR/ann-update/target-after" RAYON_NUM_THREADS=10 \
  flock -F "$HOME/warm-timing.lock" taskset -c 2,3 \
  "$WARM_BENCH_DIR/ann-update/target-after/dist/deps/vector_updates-c4f1efd2bcfee7d8" \
  --bench --save-baseline release-final
```

Final verification on ARM macOS with Rust 1.99.0 passed formatting, both
all-target Clippy modes (default and all features), 566 default tests, 880
all-features tests, and cargo-deny. Both test runs include doctests and leave five
explicitly ignored tests. All 18 vector-update tests pass. The competitive adapter
wiring checks pass (two tests). Linux CI with Rust 1.99.1 passed 889 all-features
tests and two doctests; the MSRV 1.89 job also passed. The result artifact records
the verified source revisions and command exit codes.

### Direct streaming cost fixes

Construction scratch clears only the bitset words visited by its previous search,
before resizing for a different graph or routing layer. Each nonzero word has one
entry in the touched-word list. Its logical additional payload is at most four
bytes per bitset word; vector spare capacity is additional.

Mixed queries request the complete widened base candidate set with residual
reranking disabled, then retain the existing original-row scoring and external-id
tie breaking. The requested candidate count, traversal width, and live-count
retries are unchanged. Clean-base queries retain their configured reranking.
Both delta scoring paths use the Euclidean norms computed during insertion;
angular scoring is unchanged.

The regression fingerprint `166293721`, captured from `1cfb43f` before these
changes, covers ordered query results and encoded generation bytes across eleven
encoding configurations, including reordered/compressed ids, screens, and neighbor
blocks. It includes clean, deleted/replaced/inserted, replayed, and consolidated
states, narrow and exhaustive beams, and exact ties. Separate regressions check
visited-word boundaries and graph-size reuse, and bitwise delta score equivalence
around the dot-product block boundaries and for zero Euclidean rows and queries.

`benches/vector_updates.rs` adds synthetic 100-dimensional Euclidean mixed-base
and delta-only query cases at `ef` 64, 1024, and 2048. The ignored
`build_visited_reset_workload` diagnostic compares sparse and dense visitation at
8,421,866 vertices. These measurements do not establish recall or runtime for the
official 1,280-step streaming workload. Graph reuse, local repair, and recall
changes remain separate work requiring measurements on that workload.
