# hms-research-scan: graph-free split-manifold scan kill test

Package: `benchmarks/research/scan/` (standalone crate `hms-research-scan`, not a
workspace member, own `[workspace]` table). No root-crate API is called and the
`holographic-memory` path dependency is intentionally omitted (see `Cargo.toml`);
dependencies are `anyhow 1.0`, `clap 4.4 (derive)`, `serde 1.0 (derive)`,
`serde_json 1.0 (float_roundtrip)`, all already used by the root crate at the
same versions. No new third-party crates. `Cargo.lock` and `target/` are
git-ignored inside the package.

Round 2 additions (not compiled by the author): `--head-dim 16|32|64`
(alias `--head-dims`), `--residual-mult M` (M independent SRHT rotations,
`M x padded` bits available), and `--sweep` (head {16,32,64} x mult {1,2} on one
dataset against shared int8/exact baselines, compact table + JSON).

## What it measures

Mechanism, record layout and invariants: module docs at the top of
`src/lib.rs`; basis fitting and codes in `src/codes.rs`; kernels in
`src/scan.rs`; datasets in `src/data.rs`; orchestration/JSON in `src/main.rs`.

Three representations, same binary, same primary f32 array for rerank:

| method | scanned bytes/doc | query path |
|---|---|---|
| `split` | `8 x (H/8 + 1 + B/64)`: 120 at H=16, B=768 (D=768); 184 at H=64, B=1024 (D=384, mult 2) | transform once (H dots + M SRHTs), linear scan (i8 head dot + XOR/popcount or 4-plane AND/popcount), bounded top-K heap, exact f32 rerank of K, top-10 |
| `int8` | `D + 4` (int8 row + f32 scale) | int8 query, linear i8 dot scan, same heap, same rerank |
| `exact` | `4 D + 1` (f32 row + liveness byte) | f32 dot scan straight to top-10 |

Residual estimator (unbiased for any `M`): `rho = (P / B) |y|^2 / sum|y_i|`
over the `B` kept coordinates of the `M x P` concatenated rotations; see
`src/lib.rs` docs. Default `B` = `D` rounded up to 64 (capped at `P`) for
`M = 1`, and all `M x P` bits for `M >= 2` (384-d: 384 / 1024; 768-d: 768 / 2048;
100-d: 128 / 256). Supported `B/64` word counts: 1,2,3,4,6,8,12,16,24,32.

Delete = zero the record's scalar word (split: 8 B; int8: the 4 B scale) plus
the 1-byte liveness flag in the primary array; scans never read the flag, the
rerank / exact admission does. Replace = re-encode into the slot + rewrite the
primary row + flag. Bytes touched are reported per method (single mode only).

Timing is single-thread: a recall pass (untimed), `--warmup` queries, then
`--runs` (default 5) full passes over the query set; QPS = queries / median
pass time. Encode time covers the per-document encode loop only (basis fit is
reported separately as `basis_fit_ms`).

## Commands (run from the repo root)

Build flags: `-C target-cpu=native` lets LLVM use NEON dotprod / AVX2 popcnt;
set it on every invocation so cargo does not rebuild between runs.

```sh
export RUSTFLAGS="-C target-cpu=native"
M=benchmarks/research/scan/Cargo.toml
OUT=benchmarks/research/scan/out

# 0. unit tests (f16, FWHT, heap, basis orthonormality for head 16/32/64 and mult 1/2, K=N == exact, delete leak)
cargo test --release --manifest-path $M

# 1. sweeps (the round-2 question): head {16,32,64} x mult {1,2}, shared int8/exact baselines
cargo run --release --manifest-path $M -- --dataset nfcorpus --n 50000 --sweep --out $OUT/nfcorpus_50k_pca_sym1_sweep.json
cargo run --release --manifest-path $M -- --dataset nfcorpus --n 50000 --sweep --residual-mode asym4 --out $OUT/nfcorpus_50k_pca_asym4_sweep.json
cargo run --release --manifest-path $M -- --dataset glove --n 50000 --sweep --out $OUT/glove_50k_pca_sym1_sweep.json
cargo run --release --manifest-path $M -- --dataset synthetic --n 50000 --dim 768 --sweep --out $OUT/synthetic_50k_pca_sym1_sweep.json
# narrower/faster sweep: --sweep-heads 32,64 --sweep-mults 2 --runs 3

# 2. single configurations (full report incl. delete/replace): pick the sweep winner, e.g.
cargo run --release --manifest-path $M -- --dataset nfcorpus --n 50000 --head-dim 64 --residual-mult 2 --out $OUT/nfcorpus_50k_h64_m2_sym1.json
cargo run --release --manifest-path $M -- --dataset nfcorpus --n 10000 --head-dim 64 --residual-mult 2 --out $OUT/nfcorpus_10k_h64_m2_sym1.json
cargo run --release --manifest-path $M -- --dataset glove --n 50000 --head-dim 32 --residual-mult 2 --out $OUT/glove_50k_h32_m2_sym1.json

# 3. round-1 baselines (unchanged defaults: head 16, mult 1)
cargo run --release --manifest-path $M -- --dataset synthetic --n 50000 --dim 768 --out $OUT/synthetic_50k_pca_sym1.json
cargo run --release --manifest-path $M -- --dataset synthetic --n 10000 --dim 768 --out $OUT/synthetic_10k_pca_sym1.json
cargo run --release --manifest-path $M -- --dataset synthetic --n 50000 --dim 768 --basis random --out $OUT/synthetic_50k_random_sym1.json
cargo run --release --manifest-path $M -- --dataset nfcorpus --n 50000 --out $OUT/nfcorpus_50k_pca_sym1.json
```

Explicit bit counts: `--residual-bits` must be a multiple of 64 and at most
`M x padded dim` (384-d: 512 for M=1, 1024 for M=2; `--residual-bits 768`
needs `--residual-mult 2`).

Dataset paths default to `~/.cache/hms-bench/nfcorpus` and
`~/.cache/hms-bench/glove-100-angular` (`--nfcorpus-dir`, `--glove-dir` to
override). glove is loaded from the already-converted `train.f32` / `test.f32`
in that directory (no hdf5 parsing, no new dependency); ground truth is
recomputed on the subset, so `neighbors.i32` is not used. Real datasets take
`dim` from their `meta.json`; `--dim` applies to synthetic only.

Each run prints a table, one `[PASS]`/`[FAIL]` line per threshold, an overall
line, and `wrote <path>`. Default output file:
`benchmarks/research/scan/out/<dataset>_<n>_<basis>_<mode>.json`
(`..._sweep.json` with `--sweep`).

Expected wall clock: a single 50k run 1-3 minutes; a 6-configuration sweep
roughly 6x the split part (basis fit + encode + 3 K x 6 passes per
configuration) plus one int8/exact baseline.

Other knobs: `--ks 64,256,1024` (default), `--runs 5`, `--warmup 50`,
`--calib 1000`, `--pca-iters 32`, `--clusters 200`, `--spread 0.75`,
`--spectrum-alpha 0.5`, `--distractor-noise 0.6`, `--deletes 1000`, `--seed 42`,
thresholds `--target-recall 0.95 --min-qps 2500 --min-ratio 1.5
--max-encode-ms-50k 250 --threshold-max-k 256`.

## Pass/fail thresholds

Evaluated on every run and on every sweep row; the encode limit scales
linearly with `--n` so a 10k run is judged at 50 ms.

1. split recall@10 >= 0.95 at some K <= 256.
2. split single-thread QPS >= 2,500 at K* (smallest K <= 256 reaching 0.95).
3. split QPS / int8+rerank QPS >= 1.5 at matched recall (each method at its own K*).
4. split encode time for 50k docs < 250 ms.
5. (single mode) no deleted id returned by any method after 1,000 deletes.
6. (single mode) replaced slots are found rank-1 by their own new vector (50 checks).

A sweep prints `[PASS]`/`[FAIL]` per row (thresholds 1-4) and an overall line
that passes if any row passes.

## Output JSON schema

Single mode (`schema = "hms-research-scan/v2"`):

```text
{
  schema, generated_unix_s,
  dataset: { kind, name, source, n_docs, n_real_docs, dim, n_queries, seed, load_ms, ground_truth_ms },
  config:  { basis, residual_mode, ks[], runs, warmup_queries, rerank_top, calib_vectors, pca_iters, threads, target },
  split_config: { head_dim, head_words, residual_mult, residual_bits, residual_words, padded_dim, padded_total,
                  head_variance_fraction, basis_fit_ms },
  methods: {
    split: { bytes_per_doc, encode_ms, encode_us_per_doc,
             per_k: [ { k, recall_at_top, qps, us_per_query, median_run_ms, run_ms[] } ] },
    int8:  { same },
    exact: { same, per_k has the single entry k = rerank_top }
  },
  ablation: { target_recall, threshold_max_k, split_k_star, int8_k_star,
              split_qps_at_k_star, int8_qps_at_k_star, exact_qps,
              ratio_split_vs_int8_matched, ratio_split_vs_exact_matched,
              ratio_at_equal_k: [ { k, split_recall, int8_recall, split_qps, int8_qps, ratio_split_vs_int8 } ] },
  mutation: { ops, leak_check_k, deleted_leaks, replace_self_hits, replace_self_checks,
              split: { delete_bytes_touched, delete_ns_per_op, replace_bytes_touched, replace_us_per_op },
              int8:  { same }, exact: { same } },
  thresholds: [ { name, value, limit, pass } ],
  pass
}
```

Sweep mode (`schema = "hms-research-scan/sweep-v1"`):

```text
{
  schema, generated_unix_s, dataset, config,            (as above)
  int8: MethodReport, exact: MethodReport, int8_k_star,
  rows: [ { split_config, bytes_per_doc, encode_ms, per_k[], k_star, qps_at_k_star,
            ratio_vs_int8_matched, thresholds[], pass } ],
  any_pass
}
```

`recall_at_top` is recall@`rerank_top` (10) against the exact cosine top-10
computed by the same binary before any mutation. `k_star` fields are `null`
when no K <= `threshold_max_k` reaches the target; the matched ratio is then
`null` and threshold 3 fails.

## Root-crate APIs used

None. The crate does not depend on `holographic-memory`; all kernels are local
(`src/lib.rs` `dot_f32` / `dot_i8` lane-accumulator dots, `u64::count_ones`
popcounts, `src/codes.rs` `fwht`).

## Known limits / risks

- Round-2 code is uncompiled: likely spots are the const-generic dispatch
  macros in `src/scan.rs` (`dispatch_kernel!` / `dispatch_words!`, 60
  monomorphized kernels) and the partial moves out of `SplitEval` in
  `src/main.rs`.
- The `sym1` residual estimator approximates `sum_i sign(y_i) z_i` by
  `mean|z| (B - 2 hamming)`; `asym4` is the exact 4-bit-query form.
- The int8 baseline is plain auto-vectorized Rust (no `sdot` intrinsics), the
  same treatment the split kernel gets. A hand-tuned int8 kernel would narrow
  the ratio; the JSON records the compiler target features for context.
- Head scale and rho are stored as IEEE half (`f32_to_f16_bits`, round to
  nearest even); the scan decodes them with a branch-free multiply.
- Larger heads and `mult >= 2` raise bytes/doc and encode time (each extra
  rotation is one more FWHT per document); the thresholds catch both.
