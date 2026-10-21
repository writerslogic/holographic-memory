# hms-research-witness: occupancy-balanced rare-witness decoding kill test

Package: `benchmarks/research/witness/` (standalone crate `hms-research-witness`,
not a workspace member, own `[workspace]` table). Layout mirrors
`benchmarks/research/scan/` (patterns copied, no dependency on it).

Dependencies, all at the root crate's versions: `anyhow 1.0`, `clap 4.4
(derive)`, `rayon 1.8`, `serde 1.0 (derive)`, `serde_json 1.0
(float_roundtrip)`. No new third-party crates. The `holographic-memory` path
dependency is intentionally omitted (no root-crate API is called; see the
header of `Cargo.toml`). `Cargo.lock`, `target/` and `out/*.json` are
git-ignored inside the package.

Nothing in this package has been compiled or run by its author (research-round
rule). Expect the first `cargo build` to surface small type errors; the code is
plain Rust 2021 (MSRV 1.89 declared).

## Root-crate APIs used

None. Every kernel is local: `src/lib.rs` (`dot_f32`, `exact_topk`, `rerank`,
`Index::query`), `src/bank.rs` (`fwht`, SRHT bank, quantile fit, selector),
`src/postings.rs` (CSR build), `src/join.rs` (counting threshold-join),
`src/data.rs` (loaders, PRNG copied from `scan/src/data.rs`).

## Mechanism (what is implemented)

Module heads carry the full description and invariants (`src/lib.rs`,
`src/bank.rs`). In short:

* `M` signed channels = `M/2` SRHT directions x 2 tails. Directions come in
  blocks of `Dp = next_power_of_two(D)` (128 for glove, 256 for nytimes); block
  `b` is `H_Dp (d_b * x)` with a seeded sign vector `d_b`. `M` must be a
  multiple of `2 Dp` (4096 and 32768 are, for both datasets).
* Tables fitted on a seeded 50k-row sample: per direction mean / 1/std (query
  pre-filter only); per signed channel 12 tail quantiles at `2^-1 .. 2^-12` and
  one membership threshold per `B`, the empirical quantile at tail probability
  `B/M`. Equal expected occupancy `N B / M` per channel; mean memberships per
  vector = `B` on the sample by construction, variable per vector.
* Postings: one strictly increasing `u32` list per channel, CSR (`offsets[M+1]`,
  `ids[]`), plain u32 (no delta-varint in this round).
* Query: `M/2` projections -> z-score pre-filter over directions (Gaussian cut
  sized to pass ~`4Q`, halved until >= `2Q` pass) -> exact tail-grid score for
  each survivor's channel on the query's sign side -> top `Q` by quantile rank
  -> counting threshold-join (u8 counter per id, emit on reaching `t`) -> exact
  cosine rerank of the emitted candidates in one bulk pass -> top 10.
* Ground truth: exact top-10 by cosine recomputed by the binary (batched 8
  queries per corpus pass); the shipped `neighbors.i32` is only used for a
  loader cross-check (`gt_file_agreement`, expected ~1.0 on the full corpus).
* Query sets: a seeded shuffle of the 10,000 test rows; the first 1,000 are the
  reporting set, the next 500 the disjoint selection set. Stage-1 PASS/FAIL is
  printed for the reporting set; stage 2 picks the configurations that passed
  on the selection set.
* Rayon is used only in the build (sample projection + per-direction sorts,
  corpus projection + membership). The merge order is fixed, so the postings
  are identical for any `--build-threads`. Everything on the query side is
  single-threaded.

## Data found on this machine

* `~/.cache/hms-bench/glove-100-angular/`: `meta.json` (n_train 1,183,514,
  n_test 10,000, dim 100, n_neighbors 100), `train.f32` 473,405,600 B
  (= 1,183,514 x 100 x 4), `test.f32` 4,000,000 B, `neighbors.i32`
  4,000,000 B (10,000 x 100 x 4).
* `~/.cache/hms-bench/nytimes-256-angular/`: same layout: `meta.json`
  (n_train 290,000, n_test 10,000, dim 256, n_neighbors 100), `train.f32`
  296,960,000 B (= 290,000 x 256 x 4), `test.f32` 10,240,000 B,
  `neighbors.i32` 4,000,000 B, plus an `extra_work/` directory of other
  systems' result files (ignored). Row norms were not verified by hand; the
  loader l2-normalizes every row, so angular = cosine = dot either way.
* SIFT-1M is not on this machine (not run).

## Commands (run from the repo root)

```sh
export RUSTFLAGS="-C target-cpu=native"
M=benchmarks/research/witness/Cargo.toml
OUT=benchmarks/research/witness/out

# 0. unit tests (FWHT, normal quantile, occupancy balance, grid monotonicity,
#    selector, join tally/emit/reset, postings sorted + thread-count independence)
cargo test --release --manifest-path $M

# 1. smoke (5 minutes max): 200k-row glove prefix, reduced sweep
cargo run --release --manifest-path $M -- --dataset glove --n 200000 --ms 4096 --bs 16 --qs 16,32 --ts 2,3 --out $OUT/smoke_glove_200k.json

# 2. stage 1, glove-100-angular, full sweep M{4096,32768} x B{16,24,32} x Q{16,32,64} x t{2,3}
cargo run --release --manifest-path $M -- --dataset glove --out $OUT/stage1_glove100_1183514.json

# 3. stage 1, nytimes-256-angular, same sweep
cargo run --release --manifest-path $M -- --dataset nytimes --out $OUT/stage1_nytimes256_290000.json

# 4. stage 2 for the configurations that passed stage 1 on the selection set
#    (stage 1 is re-run in the same process; add --timed-all to time every
#    configuration, or pin one):
cargo run --release --manifest-path $M -- --dataset glove --timed
cargo run --release --manifest-path $M -- --dataset glove --config 32768,16,32,2 --config 32768,24,16,2

# 5. single-thread build-time number for the stage-2 row (optional, slower build)
cargo run --release --manifest-path $M -- --dataset glove --build-threads 1 --config 32768,16,32,2
```

Dataset directories default to `~/.cache/hms-bench/glove-100-angular` and
`~/.cache/hms-bench/nytimes-256-angular` (`--data-dir` overrides). `--n`
truncates the corpus to a prefix (smoke runs only; the shipped ground truth
cross-check is then skipped). Other knobs: `--sample 50000` (raise to 200000
for M=32768 if the `[warn]` line about thin quantile ranks appears; at the
default the `B=16, M=32768` threshold rests on 24 sample rows per channel),
`--report-queries 1000 --select-queries 500 --seed 42 --top 10 --runs 5
--warmup 100 --window-rows 65536`, and every gate value (see `--help`).

Output: stage 1 writes `$OUT/stage1_<dataset>_<n>.json` (or `--out`); stage 2
writes one `$OUT/stage2_<dataset>_<n>_m<M>_b<B>_q<Q>_t<t>.json` per
configuration (`--out-dir` overrides the directory). Stdout carries the
compact stage-1 table (one row per configuration plus a rank-10
survival-by-similarity line), the Pareto frontier, and the stage-2 rows;
stderr carries progress (`[data]`, `[gt]`, `[bank]`, `[warn]`).

Expected wall clock, glove full corpus, ~10 cores: load 2-5 s; ground truth
15-40 s (177 GFLOP, single thread); per M: fit 5-30 s, build 5-20 s; stage-1
statistics 1-2 min (dominated by the M=4096 joins, up to 2.4 MB of ids per
query); stage 2 about 30 s per configuration. Total 5-10 minutes per dataset.
nytimes is roughly a quarter of that.

## Pass / fail thresholds

Stage 1 (every one of these, printed per configuration; `--min-*` / `--max-*`
flags override):

1. candidate recall@10 >= 0.98;
2. rank-10 survival >= 0.95 (the 10th exact neighbour has >= t witnesses);
3. median / p95 emitted candidates <= 1,000 / 5,000;
4. median / p95 posting bytes streamed <= 100 KB / 250 KB;
5. mean projection + selection time <= 20 us.

Stage 2 (printed per timed configuration): recall@10 >= 0.95 and mean
complete-query latency <= 70 us (median of 5 passes over the 1,000 reporting
queries after 100 warmup queries, i.e. >= ~14,300 QPS). The baseline row
(`public-bench ann-qgraph`, 1,165 QPS at recall 0.95) is supplied by the
mediator; the binary prints only the witness numbers.

## Memory arithmetic (glove, N = 1,183,514, D = 100)

Posting ids, 4 B each, `4 B N`:

| B | ids | bytes |
|---|---|---|
| 16 | 18,936,224 | 75,744,896 (75.7 MB) |
| 24 | 28,404,336 | 113,617,344 (113.6 MB) |
| 32 | 37,872,448 | 151,489,792 (151.5 MB) |

Plus offsets `4 (M+1)` (16.4 KB at M=4096, 131 KB at M=32768), bank tables
(signs `4 M/2`, mean + 1/std `8 M/2`, grid `48 M`, tau `4 M`: 0.24 MB at
M=4096, 1.9 MB at M=32768) and rerank rows `4 N D` = 473,405,600 B (473.4 MB,
unpadded f32). The JSON reports each term.

Expected posting bytes streamed per query (`4 Q N B / M`, perfectly balanced
channels; the measured median/p95 are what the gate uses):

| M | B | ids/list | Q=16 | Q=32 | Q=64 |
|---|---|---|---|---|---|
| 4096 | 16 | 4,623 | 296 KB | 592 KB | 1,184 KB |
| 4096 | 24 | 6,935 | 444 KB | 888 KB | 1,775 KB |
| 4096 | 32 | 9,246 | 592 KB | 1,184 KB | 2,367 KB |
| 32768 | 16 | 578 | 37 KB | 74 KB | 148 KB |
| 32768 | 24 | 867 | 56 KB | 111 KB | 222 KB |
| 32768 | 32 | 1,156 | 74 KB | 148 KB | 296 KB |

So at full glove scale the 100 KB median gate can only be met by M=32768 with
(B=16, Q<=32), (B=24, Q=16) or (B=32, Q=16); the whole M=4096 family is
outside the byte gate by arithmetic and is swept only for the recall-vs-bytes
curve. The `expected_posting_bytes` / `exp_KB` column prints this number next
to the measured one. For nytimes (N = 290,000) divide by 4.08.

Per-query projection cost: `M/2 log2(Dp)` add/subs = 115k (glove, M=32768) or
131k (nytimes, M=32768); 14k / 16k at M=4096.

## Output JSON

Schemas (`hms-research-witness/stage1/v1`, `hms-research-witness/stage2/v1`)
are spelled out in `out/SCHEMA.md`. Top level, stage 1:

```text
{ schema, generated_unix_s,
  dataset: { kind, name, dir, n_docs, n_train_total, dim, dim_pad, n_report_queries,
             n_select_queries, sample_rows, seed, top, load_ms, ground_truth_ms,
             gt_file_agreement | null, target },
  sweep: { ms[], bs[], qs[], ts[] }, gate: { ...stage-1 limits },
  banks: [ { m, directions, blocks, grid_levels, table_bytes, fit_ms, build_all_budgets_ms,
             build_threads, budgets: [ { b, tail_p, sample_rank, total_memberships,
             mean_memberships_per_vector, posting_bytes, occupancy{...} } ] } ],
  configs: [ { m, b, q, t, expected_posting_bytes, report: SetStats, select: SetStats,
               fails_report[], fails_select[], pass_report, pass_select } ],
  curve: [ CurvePoint sorted by median_posting_bytes ], pareto: [ CurvePoint ],
  n_pass_report, n_pass_select }
SetStats = { n_queries, candidate_recall_at_top, rank10_survival,
             candidates{mean,median,p95,max}, posting_bytes{...}, projection_selection_us{...},
             mean_true_sim_rank10, survival_by_sim: [ { lo, hi, n, rank10_survival, all10_n, all10_survival } ] }
```

Stage 2: `{ schema, generated_unix_s, dataset, config{m,b,q,t}, top,
build{fit_ms, postings_ms, total_ms, threads, total_memberships,
mean_memberships_per_vector, occupancy}, index_bytes{postings, bank_tables,
thresholds, rerank_rows, total}, query{n_queries, runs, warmup_queries,
run_ms[], median_run_ms, qps, mean_us, latency_us{...}, candidates{...},
posting_bytes{...}}, recall_at_top, thresholds[{name, value, limit, pass}],
pass }`.

## Known limits / risks

- Untested code. Likely first-compile trouble spots: the rayon `map_init`
  closure signature in `src/postings.rs::build`, the `pool.install` closures in
  `src/bank.rs::fit_tables` (they borrow `rows` / `cols` mutably), and clap's
  `value_parser = parse_config` on `Vec<ConfigKey>`.
- Quantile-rank selection uses a z-score pre-filter before the exact grid
  score; a channel with a heavy upper tail but modest variance could be cut
  before scoring. The fallback (cut halves until >= 2Q directions pass) bounds
  the damage; `--qs` with a larger Q plus the grid score is the diagnostic.
- Membership thresholds at `B/M = 2^-11` are 24th-of-50k order statistics;
  channel occupancy will scatter by ~20% around `N B / M` (the `[bank]` line
  prints min/p05/p50/p95/max). `--sample 200000` tightens it at 4x fit cost.
- The join uses a u8 counter per id (1.18 MB for glove, L2-resident) and resets
  by re-walking the lists, so reset costs as much as the join. A 2-bit or
  epoch-stamped counter would halve that if the join is what misses 70 us.
- The rerank reads `candidates x 400 B` of unpadded f32 rows with no explicit
  prefetch (stable Rust has no portable prefetch); independent loads overlap
  on their own, but this is the one random-access term left in the query.
- Ties at the 10th exact neighbour are resolved by id; recall is set overlap.
