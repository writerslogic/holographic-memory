# ANN graph search: runtime knobs, kill tests and the same-machine comparison

Package: search-path knobs in `src/core/qgraph/` (no new index format), an identity test, and
the commands to measure each knob on glove-100-angular and to measure hnswlib and FAISS HNSW on
the same machine with the competitive harness. Nothing in this package was built, tested or
run by its author (research-round rule); every command below is for the mediator.

## What exists already (levers that needed no code)

| Lever | State on `origin/main` |
|---|---|
| (3) visited set | `Marks` in `vertex.rs`: a per-query epoch-stamped `u8` array (cleared every 255 searches), not a hash set. Nothing to do. |
| (4) no per-query allocation | `VSearcher` owns every buffer (`fresh`, `survivors`, `keys`, `results`, the `Pool` sorted array, the query codes); `search` only clears them. Nothing to do. |
| (5) early termination | The pool rule (stop when none of the `ef` best seen is unexpanded) plus `--patience`. New: the frontier rule below. |
| (2) block scoring, FastScan | `fastscan.rs` (`scan32`, AVX2 + NEON) and `VGraph::with_blocks` (SymphonyQG-style neighbour blocks, 1/2/4-bit codes of each neighbour's residual to the vertex, per-edge scale and offset, calibrated error model `block_kappa`) exist and are exposed as `--block-bits 1|2|4` (`--fuse` optional). In that path the block estimate *orders* the search; the results come from the expanded vertices' own rows. New: the row-batched kernel (`batch_est`) and the block *screen* (knob 6) below. |
| (1) prefetch | Baseline prefetches the next candidate's neighbour list (`peek_next`) and the code row of every admitted neighbour. It did not prefetch the per-vertex scale (its own 4.7 MB array, one random line per neighbour), the rows the re-rank reads, or anything one expansion ahead. New levels below. |

## Knobs (all default off; results bit-identical to `origin/main` when off)

Read once per process from the environment by `VGraph::searcher()` (`SearchKnobs::from_env`,
`src/core/qgraph/vertex.rs`), or passed explicitly with `VGraph::searcher_with(SearchKnobs)`.
`public-bench ann-qgraph` calls `searcher()`, so the env vars are the interface; no binary was
edited.

| Env var | Values | Mechanism | Results |
|---|---|---|---|
| `WARM_QGRAPH_PREFETCH` | `0` (default) .. `3` | 1: also prefetch each admitted neighbour's scale and, before the re-rank, the `rerank` rows (code + residual + scales). 2: also the neighbour list of the second-nearest unexpanded candidate (`Pool::peek_unexpanded(1)`). 3: also a lookahead: after scoring an expansion, prefetch the code rows (or 1-bit screen rows) and scales of the next candidate's not-yet-visited neighbours (`Marks::seen`), so they travel during the next pop. | identical at every level |
| `WARM_QGRAPH_BATCH_EST` | `0` / `1` | Admitted neighbours are estimated four rows at a time (`VGraph::est4` -> `kernels::dot_i4_inline_x4`, AVX2 `dot_i4_avx2_x4`: the query's widened halves are loaded once per 16 packed bytes, four row loads in flight, four accumulators). Elsewhere (NEON, scalar, 8-bit rows) it falls back to four single calls. | identical (same integer sums) |
| `WARM_QGRAPH_FRONTIER` | `1.0` (default, off) .. `0.0` | With a full pool, stop when the nearest unexpanded estimate is past the point `f` of the way from the `k`-th best to the `ef`-th best estimate (`est > kth + f * (worst - kth)`). `1.0` never fires. | changes results: fewer expansions per `ef`; compare the recall/QPS curve over `ef` |
| `WARM_QGRAPH_BLOCK_SCREEN` | `0` / `1`; needs `--block-bits 1` (or 2, 4) | Knob 6. The neighbour block is scanned once per pop (`scan32`); a fresh neighbour whose block estimate, improved by `--screen-sigmas` deviations of the calibrated per-edge error (`sqrt(block_kappa * s^2 * |q - x_v|^2)`), cannot enter the full pool is skipped (counted in `screened_per_query`); survivors' 4-bit rows are prefetched together and estimated; the pool holds row estimates as in the plain search, and results come from the pool. `--screen-sigmas` sweeps the margin. | changes results by the screened candidates only; with an infinite margin it reproduces the plain search exactly (tested) |
| `WARM_QGRAPH_HUGEPAGES` | `0` / `1` | At index encode time, `madvise(MADV_HUGEPAGE)` on the 2 MiB-aligned interior of the code rows, residual rows, scales, 3-byte neighbour lists and neighbour blocks before they are first touched (Linux only; no-op elsewhere). Needs `/sys/kernel/mm/transparent_hugepage/enabled` = `always` or `madvise`. Check it took: `grep AnonHugePages /proc/<pid>/smaps_rollup` while the sweep runs (expect >= the index size minus a few MB). | identical |

Combination semantics: independent; `WARM_QGRAPH_BLOCK_SCREEN=1` without `--block-bits` is a
no-op, `--fuse` is ignored under the block screen, `WARM_QGRAPH_PREFETCH>=3` lookahead applies to
the plain (row) path only (the block paths prefetch the next record already).

## Code map (names are `rg`-able; line numbers from the edited tree)

* `src/core/qgraph/vertex.rs`: `SearchKnobs`, `env_knob`, `huge_pages`, `advise_huge_pages`,
  `huge_capacity` (top of file, after `VSearchParams`); `Lines::zeroed` (advised allocation);
  `from_graph_with` (advised `scales`, `residual`, `adj3`); `VGraph::prefetch_scale`,
  `prefetch_rerank_rows`, `est4`; `VGraph::searcher` / `searcher_with`; `Marks::seen`;
  `Pool::peek_unexpanded`; `VSearcher::knobs`, `VSearcher::score_fresh` (shared row scoring,
  batched or not); `VSearcher::search` (frontier rule at the loop head, prefetch levels, the
  `block_screen` branch before the existing `if blocks` branch, lookahead after `score_fresh`,
  re-rank prefetch, `if blocks && !block_screen` for the result source).
* `src/core/qgraph/kernels.rs`: `dot_i4_inline_x4` (after `dot_i4_inline`) and
  `x86::dot_i4_avx2_x4` (after `x86::dot_i4_avx2`).
* `src/core/qgraph/mod.rs`: `pub use vertex::{SearchKnobs, ...}`.
* `tests/qgraph_prefetch.rs`: identity and pruning tests (below).

Root-crate APIs used by the test: `core::qgraph::{BuildParams, Graph::build, VGraph::from_graph_with,
VGraph::with_blocks, VGraph::searcher_with, VGraph::{len, has_blocks, block_bytes, index_bytes},
VSearcher::{search, last_screened}, VSearchParams, SearchKnobs}`.

## Gates (run first)

```sh
cargo test --release --test qgraph_prefetch          # 50k x 100; three tests, one shared build
cargo test --test qgraph_prefetch                    # debug: 12k x 100 (the build is slow unoptimised)
cargo clippy --all-targets -- -D warnings
cargo test --all-features                            # the rest of the suite; the block path is used by mod.rs tests
```

The test asserts, per query, identical result ids and identical estimate counts for prefetch
1/2/3, `batch_est`, both together, and `frontier` 1.0 against all-off, on three encodings
(the measured one; one with the 1-bit vertex screen; 8-bit rows with delta-coded lists); the
block screen with `screen_sigmas = +inf` must reproduce the plain search over the same graph
exactly, and at one sigma must screen something, make fewer estimates in total, and keep
recall >= 0.8 x baseline. `frontier` 0.5 must never make more estimates than the baseline on
any query. The baseline recall sanity bound is 0.5; if that fails, the fixture (not a knob) is
at fault.

## Kill tests on glove-100-angular (x86, same machine, one process per arm)

Data: `~/.cache/warm-bench/glove-100-angular/{train.f32,test.f32,neighbors.i32,meta.json}` as
written by `benchmarks/public/prepare.py ann glove-100-angular` (present on this machine).
Configuration string: the accepted one from `benchmarks/public/ann_modal_confirm.py` `CONFIG`
(4-bit rows, 8-bit residual, degree 32, build ef 200, aligned rows, BFS reorder, 3-byte ids,
rerank 64, holdout 2000 train queries), with its ef sweep.

```sh
BIN=target/release/public-bench
cargo build --release --features cli,mimalloc --bin public-bench
DATA=~/.cache/warm-bench/glove-100-angular
OUT=benchmarks/results/research/ann; mkdir -p $OUT
CACHE=$OUT/glove-100.graph      # one graph for every arm: identical structure (graph_hash in each JSON)
CONFIG="--index vertex --vertex-bits 4 --residual --residual-bits 8 --degree 32 --build-ef 200 \
  --align-rows --reorder --id-bytes 3 --rerank 64 --ef 128,192,256,320,384,512,736 --repeats 5 \
  --holdout 2000 --max-load 1000 --max-wait-secs 0"

# Arm 0: baseline (all knobs off). Builds and saves the graph.
$BIN ann-qgraph --data $DATA --out $OUT/base.json --graph-cache $CACHE $CONFIG
# Arm P1..P3: prefetch levels (expect identical recall_at_10 per ef, higher qps).
WARM_QGRAPH_PREFETCH=1 $BIN ann-qgraph --data $DATA --out $OUT/prefetch1.json --graph-cache $CACHE $CONFIG
WARM_QGRAPH_PREFETCH=2 $BIN ann-qgraph --data $DATA --out $OUT/prefetch2.json --graph-cache $CACHE $CONFIG
WARM_QGRAPH_PREFETCH=3 $BIN ann-qgraph --data $DATA --out $OUT/prefetch3.json --graph-cache $CACHE $CONFIG
# Arm B: batched estimation (identical recall).
WARM_QGRAPH_BATCH_EST=1 $BIN ann-qgraph --data $DATA --out $OUT/batch.json --graph-cache $CACHE $CONFIG
# Arm H: huge pages (identical recall; Linux with THP enabled=always|madvise).
cat /sys/kernel/mm/transparent_hugepage/enabled
WARM_QGRAPH_HUGEPAGES=1 $BIN ann-qgraph --data $DATA --out $OUT/hugepages.json --graph-cache $CACHE $CONFIG
# Arm F: frontier rule (recall changes; compare curves). Sweep three values.
for F in 0.75 0.5 0.25; do
  WARM_QGRAPH_FRONTIER=$F $BIN ann-qgraph --data $DATA --out $OUT/frontier-$F.json --graph-cache $CACHE $CONFIG
done
# Arm S: block screen (knob 6) on 1-bit neighbour blocks; sweep the margin. Memory: index_bytes.
BLOCKS="--block-bits 1 --screen-sigmas 0.5,1,2 --ef 128,192,256,320,384,512,736"
WARM_QGRAPH_BLOCK_SCREEN=1 $BIN ann-qgraph --data $DATA --out $OUT/blockscreen-1bit.json --graph-cache $CACHE \
  ${CONFIG/--ef 128,192,256,320,384,512,736/} $BLOCKS
# Control for S: the existing block path on the same blocks (block estimate orders the search).
$BIN ann-qgraph --data $DATA --out $OUT/blocks-1bit-order.json --graph-cache $CACHE \
  ${CONFIG/--ef 128,192,256,320,384,512,736/} --block-bits 1 --ef 128,192,256,320,384,512,736
# Arm ALL: the identity knobs together, then plus the block screen.
WARM_QGRAPH_PREFETCH=3 WARM_QGRAPH_BATCH_EST=1 WARM_QGRAPH_HUGEPAGES=1 \
  $BIN ann-qgraph --data $DATA --out $OUT/all-identity.json --graph-cache $CACHE $CONFIG
WARM_QGRAPH_PREFETCH=3 WARM_QGRAPH_BATCH_EST=1 WARM_QGRAPH_HUGEPAGES=1 WARM_QGRAPH_BLOCK_SCREEN=1 \
  $BIN ann-qgraph --data $DATA --out $OUT/all-blockscreen.json --graph-cache $CACHE \
  ${CONFIG/--ef 128,192,256,320,384,512,736/} $BLOCKS
```

Interleave arms (base, P1, P2, P3, B, H, base, ...) over two rounds if the machine is shared;
the `--repeats 5` median is per arm, so the order of arms is the only cross-arm control.

Output JSON (`public-bench ann-qgraph`): `sweep[]` rows with `params.ef`, `params.screen_sigmas`,
`recall_at_10`, `qps_single_thread` (median of `qps_runs`), `mean_exact_evals_per_query`,
`screened_per_query`, `latency_us` (with `--latency`); top level `index_bytes`,
`residual_bytes`, `build.graph_hash`, `rss_bytes`.

Readout per arm:

1. `graph_hash` equal across arms (same cached graph), else stop.
2. Identity arms (P1-P3, B, H, ALL-identity): `recall_at_10` and `mean_exact_evals_per_query`
   equal to `base.json` at every ef, bit for bit. Any difference is a bug; kill the arm.
3. QPS at recall 0.95 and 0.90, two ways: (a) the harness rule, the highest
   `qps_single_thread` among rows with `recall_at_10 >= target`; (b) linear interpolation of
   `log(qps)` against recall between the two ef rows bracketing the target (reported next to
   (a); (a) is the number comparable to the competitive table).
4. Verdict per knob: keep if QPS@0.95 by (a) is >= 1.05 x base and QPS@0.90 is not below base;
   kill if < 1.03 x or below base at either target. For F and S (result-changing), apply the
   same rule on the interpolated curve and report the recall ceiling (the best recall reached
   in the sweep must stay >= base's best, otherwise the knob caps recall).
5. Memory for S: `index_bytes` minus base's, divided by 1,181,514 (n after holdout); expected
   768 B/node (below). Report it next to the QPS.

## Expected gains (arithmetic; measurement decides)

Measured: 1,165 QPS at 0.95 = 858 us/query; 3,019 at 0.90 = 331 us/query. A faster dot
kernel changed nothing, so the traversal is bound by dependent random fetches.

Per expansion on the baseline (degree 32, ~16 fresh neighbours once the search is under way):
one neighbour-list line (prefetched one pop earlier), 16 code lines (64 B rows, one line each,
prefetched at admission and consumed within ~100 ns, so partly exposed), 16 scale lines
(not prefetched, exposed, overlapped only by out-of-order execution), ~16 visited-mark lines
(L2/L3), then 16 estimates (~8 ns each) and pool inserts. With ~90 ns DRAM latency and 10-12
line-fill buffers: codes ~2 rounds (~180 ns) partly hidden, scales ~1.5 rounds (~140 ns)
exposed, list ~90 ns when the frontier moved, compute ~150 ns: roughly 450-600 ns per
expansion. At ef 384 (near 0.95 on this set) that is ~400-500 expansions, 180-300 us of the
858 us; the rest is the re-rank (64 x three unprefetched rows: ~20 us), the descent, and the
pool/mark traffic that scales with ef. Read `mean_exact_evals_per_query` from base.json to
replace the expansion estimate with the measured count (evals ~= expansions x fresh + 64).

* P1 (scales + re-rank rows): removes the exposed scale round and most of the re-rank misses:
  -100 to -160 ns per expansion, ~+15-25% QPS if the traversal is 60% of the query.
* P3 (lookahead): code and scale lines of the next pop travel during this pop's compute: the
  exposed memory time per expansion falls to about one round (~90 ns): 1.5-2x on the
  traversal, +30-60% end to end. P2 alone is worth little (the second list is rarely the next
  pop) and is included to isolate it.
* B (batched rows): arithmetic is not the bottleneck (the VNNI null result); expect 0-5%.
  Included as a kill test of that statement in the prefetched regime, where it may matter.
* H (huge pages): each DRAM miss also pays a page walk (~20-30 ns with cached PTEs) on a
  190 MB working set over 46k 4 KiB pages against ~1.5k L2-TLB entries; with 2 MiB pages the
  whole index fits the TLB: 5-15% on the exposed misses. On a gVisor/Modal container the
  advice may be ignored (check `AnonHugePages`).
* F (frontier): fewer expansions at a given ef with a recall drop; at matched recall the
  literature on relaxed termination gives 1.1-1.4x. It is the only knob that can also lose.
* S (block screen, 1-bit): replaces 16 random row fetches per expansion by one contiguous
  768 B record (12 lines, hardware-prefetched) plus the survivors' rows. If the 1-bit estimate
  at one sigma screens 60-80% of fresh neighbours (the vertex screen's regime), random misses
  per expansion fall from ~32 (code + scale) to ~6-13: 2-3x on the traversal, 1.5-2.2x end to
  end, at +768 B/node = +907 MB on 1.18M vectors (base index ~345 MB: 64 B code + 96 B list +
  4 B scale + 128 B residual + 4 B residual scale per node plus the upper layers), i.e. 3.6x
  the memory. Record format: `blocks_per_vertex * (16 * m + 8 * 32)` bytes with
  `m = block_dim * bits / 4`, `block_dim` = the rotation width (128 for d = 100), so
  16 * 32 + 256 = 768 at 1 bit (512 B of codes, 128 B of per-edge scales, 128 B of offsets).
  The compact layout the literature uses (sign bits + one f16 factor per edge, ~576 B, or
  ~400 B unpadded) is a format change and is not in this package.

Stacked (P3 + H + S at matched recall): 858 us -> roughly 300-450 us, 2.2-3.3k QPS at 0.95.
Against hnswlib/FAISS HNSW estimated at 2.3-4.7k on the same machine, that is parity to a
modest lead, not a wide margin; the same-machine measurement below is what settles it. The
numbers above are a model, not a measurement; the kill tests replace them.

## hnswlib and FAISS HNSW on the same machine (competitive harness)

The harness (`benchmarks/competitive/warm-bench`, `warmbench/runner.py`, `ann_worker.py`) runs
WARM and every comparator in interleaved rounds on one machine, one query at a time in a single
client thread, per-query `perf_counter_ns` inside the worker process; builds use
`spec["threads"]` cores, search one (`OMP_NUM_THREADS=1`, `hnswlib set_num_threads(1)`,
`faiss.omp_set_num_threads(1)`). WARM is always included and runs
`public-bench ann-qgraph` with the accepted configuration (`runner.py` `_warm_ann_round`:
`--index vertex --degree 32 --build-ef 200 --residual --residual-bits 8 --vertex-bits 4
--id-bytes 3 --align-rows --reorder --rerank 64 --ef 64,96,128,192,256,384,512,736,1024
--repeats 2 --latency`) against the hdf5 test queries (10,000 in the heavy suite), so its rows
are comparable to the comparators' in the same file, not to the holdout rows above. The env
is inherited (`run_measured` merges `os.environ`), so the knobs apply to the WARM row.

```sh
./benchmarks/competitive/warm-bench install hnswlib faiss-hnsw        # pinned envs under ~/.cache/warm-bench/competitive/envs
cargo build --release --features cli,mimalloc --bin public-bench                              # the harness does not build
# Full glove-100-angular (1,183,514 train, 10,000 test), 3 rounds, WARM + hnswlib + FAISS HNSW:
./benchmarks/competitive/warm-bench run --suite heavy --only ann --systems hnswlib faiss-hnsw --datasets glove-100-angular
# The same with the knobs on the WARM row (one file per setting; name them in the readout):
WARM_QGRAPH_PREFETCH=3 WARM_QGRAPH_HUGEPAGES=1 \
  ./benchmarks/competitive/warm-bench run --suite heavy --only ann --systems hnswlib faiss-hnsw --datasets glove-100-angular
./benchmarks/competitive/warm-bench table                              # regenerates docs/benchmarks/COMPARISON.md
```

Output: `benchmarks/results/competitive/heavy-ann-glove-100-angular-n1183514-<hardware class>.json`
(`systems.<key>.builds.<build>.serve_rounds[].configs[]` with `params`, `recall_at_10`,
`repeats[].qps`). The smaller `--suite standard` (250k) is the quick check; the `heavy` run is
the one that answers the question at 1.18M.

Recall matching (docs/benchmarks/README.md, "Matched recall"): each comparator sweeps its
documented settings, HNSW with M 16 and 32, efConstruction 200, ef in
`10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024` (`ann_worker.py` `EF_SWEEP`,
`efSearch = max(ef, k)`); for each target recall@10 (0.90, 0.95, 0.99) the row with recall at
or above the target and the highest pooled median QPS over all rounds is chosen; a target
not reached is reported with its best recall, never dropped; comparators are never tuned
below their documented defaults. Apply the same selection to the WARM rows in that file and
report, per target, WARM QPS / best-comparator QPS with the 95% interval the table generator
computes (`warm-bench table`). The block screen needs `--block-bits 1` on the WARM command,
which the harness does not pass: for that arm run `public-bench ann-qgraph` by hand on the
harness's data directory (`~/.cache/warm-bench/competitive/data/glove-100-angular-1183514/`,
written by the run above) with the harness's WARM flags plus `--block-bits 1 --screen-sigmas 1`
and `WARM_QGRAPH_BLOCK_SCREEN=1`, and select its rows with the same rule.

## Open risks

* Not compiled here. Likeliest compile faults: borrow of `self.fresh` across
  `self.score_fresh` in `search` (fields are accessed directly, which the borrow checker
  accepts; if not, inline the call), `extern "C" madvise` under `#[cfg(target_os = "linux")]`
  (edition 2021, no `unsafe extern` needed), the `tier::get() >= tier::AVX2` comparison in
  `dot_i4_inline_x4`.
* The frontier rule reads `self.pool.keys[k - 1]`; `k` <= `ef` <= pool length once full, and
  the rule is skipped for `k == 0`.
* `measure_block_kappa` (unchanged) calls `self.searcher()`, which now reads the env once;
  harmless.
* Huge pages: `glibc` places large allocations in their own mappings, so the advised interior
  covers all but the first and last partial 2 MiB; on gVisor the advice may be a silent no-op.
* Lookahead reads the next candidate's list a pop early; when an insert lands at the pool
  cursor the lookahead target changes and one list fetch is exposed. The arm P3 vs P2
  difference measures the net effect.
