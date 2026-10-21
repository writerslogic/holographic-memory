# hms-research-scan: graph-free split-manifold scan kill test

Package: `benchmarks/research/scan/` (standalone crate `hms-research-scan`, not a
workspace member, own `[workspace]` table). No root-crate API is called and the
`holographic-memory` path dependency is intentionally omitted (see `Cargo.toml`);
dependencies are `anyhow 1.0`, `clap 4.4 (derive)`, `serde 1.0 (derive)`,
`serde_json 1.0 (float_roundtrip)`, all already used by the root crate at the
same versions. No new third-party crates. `Cargo.lock` and `target/` are
git-ignored inside the package.

Nothing in this package has been compiled or run by its author (research-round
rule). Expect the first `cargo build` to surface small type errors; the code is
plain Rust 2021 with no unstable features (MSRV 1.89 declared, local toolchain
1.96).

## What it measures

Mechanism, record layout and invariants: module docs at the top of
`src/lib.rs`; basis fitting and codes in `src/codes.rs`; kernels in
`src/scan.rs`; datasets in `src/data.rs`; orchestration/JSON in `src/main.rs`.

Three representations, same binary, same primary f32 array for rerank:

| method | scanned bytes/doc (D=768, B=768) | query path |
|---|---|---|
| `split` | 120 (= 8 x (3 + 12) words: int8x16 head, half-float head scale + rho + popcount word, 768 sign bits) | transform once (16 dots + SRHT), linear scan (i8 head dot + XOR/popcount or 4-plane AND/popcount), bounded top-K heap, exact f32 rerank of K, top-10 |
| `int8` | 772 (768 int8 + f32 row scale) | int8 query, linear i8 dot scan, same heap, same rerank |
| `exact` | 3073 (768 f32 + liveness byte) | f32 dot scan straight to top-10 |

Delete = zero the record's scalar word (split: 8 B; int8: the 4 B scale) plus
the 1-byte liveness flag in the primary array; scans never read the flag, the
rerank / exact admission does. Replace = re-encode into the slot + rewrite the
primary row + flag. Bytes touched are reported per method.

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

# 0. unit tests (f16 round trip, FWHT, heap, basis orthonormality, K=N == exact, delete leak)
cargo test --release --manifest-path $M

# 1. main kill test: synthetic 50k x 768, 200 clusters, 500 queries, PCA basis, 1-bit query
cargo run --release --manifest-path $M -- --dataset synthetic --n 50000 --dim 768 --out $OUT/synthetic_50k_pca_sym1.json

# 2. same at 10k
cargo run --release --manifest-path $M -- --dataset synthetic --n 10000 --dim 768 --out $OUT/synthetic_10k_pca_sym1.json

# 3. ablations on the 50k synthetic: 4-bit query planes; random (data-independent) basis
cargo run --release --manifest-path $M -- --dataset synthetic --n 50000 --dim 768 --residual-mode asym4 --out $OUT/synthetic_50k_pca_asym4.json
cargo run --release --manifest-path $M -- --dataset synthetic --n 50000 --dim 768 --basis random --out $OUT/synthetic_50k_random_sym1.json

# 4. real distribution: NFCorpus MiniLM 384-d (3,633 real rows + on-manifold distractors to 50k, 323 real queries)
cargo run --release --manifest-path $M -- --dataset nfcorpus --n 50000 --out $OUT/nfcorpus_50k_pca_sym1.json
cargo run --release --manifest-path $M -- --dataset nfcorpus --n 50000 --residual-mode asym4 --out $OUT/nfcorpus_50k_pca_asym4.json
cargo run --release --manifest-path $M -- --dataset nfcorpus --n 10000 --out $OUT/nfcorpus_10k_pca_sym1.json

# 5. glove-100-angular (pre-converted raw f32 in ~/.cache/hms-bench/glove-100-angular, seeded 50k subset, 500 test queries)
cargo run --release --manifest-path $M -- --dataset glove --n 50000 --out $OUT/glove_50k_pca_sym1.json
cargo run --release --manifest-path $M -- --dataset glove --n 50000 --residual-mode asym4 --out $OUT/glove_50k_pca_asym4.json
```

Dataset paths default to `~/.cache/hms-bench/nfcorpus` and
`~/.cache/hms-bench/glove-100-angular` (`--nfcorpus-dir`, `--glove-dir` to
override). glove is loaded from the already-converted `train.f32` / `test.f32`
in that directory (no hdf5 parsing, no new dependency); ground truth is
recomputed on the subset, so `neighbors.i32` is not used. Real datasets take
`dim` from their `meta.json`; `--dim` applies to synthetic only.

Each run prints a table, one `[PASS]`/`[FAIL]` line per threshold, an overall
line, and `wrote <path>`. If `--out` is omitted the file is
`benchmarks/research/scan/out/<dataset>_<n>_<basis>_<mode>.json`.

Expected wall clock per 50k run: roughly 1-3 minutes (ground truth ~ 5 s, then
3 K values x 2 methods x 6 passes + exact 6 passes over 500 queries).

Other knobs: `--ks 64,256,1024` (default), `--runs 5`, `--warmup 50`,
`--residual-bits <multiple of 64 <= padded dim>` (default: dim rounded up to
64; supported word counts 1,2,3,4,6,8,12,16 i.e. 64..1024 bits), `--calib 1000`,
`--pca-iters 32`, `--clusters 200`, `--spread 0.75`, `--spectrum-alpha 0.5`,
`--distractor-noise 0.6`, `--deletes 1000`, `--seed 42`, thresholds
`--target-recall 0.95 --min-qps 2500 --min-ratio 1.5 --max-encode-ms-50k 250
--threshold-max-k 256`.

## Pass/fail thresholds (panel)

Evaluated on every run; the encode limit scales linearly with `--n` so the 10k
run is judged at 50 ms.

1. split recall@10 >= 0.95 at some K <= 256.
2. split single-thread QPS >= 2,500 at K* (smallest K <= 256 reaching 0.95).
3. split QPS / int8+rerank QPS >= 1.5 at matched recall (each method at its own K*).
4. split encode time for 50k docs < 250 ms.
5. no deleted id returned by any method after 1,000 deletes (sanity, must be 0).
6. replaced slots are found rank-1 by their own new vector (sanity, 50 checks).

The verdict for the panel is thresholds 1-4 on `synthetic_50k_pca_sym1.json`,
confirmed (or killed) by the two real-distribution runs.

## Output JSON schema (`schema = "hms-research-scan/v1"`)

```text
{
  schema, generated_unix_s,
  dataset: { kind, name, source, n_docs, n_real_docs, dim, n_queries, seed, load_ms, ground_truth_ms },
  config:  { basis, head_dim, residual_bits, residual_words, padded_dim, residual_mode,
             ks[], runs, warmup_queries, rerank_top, calib_vectors, pca_iters,
             head_variance_fraction, threads, target },
  basis_fit_ms,
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

`recall_at_top` is recall@`rerank_top` (10) against the exact cosine top-10
computed by the same binary before any mutation. `k_star` fields are `null`
when no K <= `threshold_max_k` reaches the target; the matched ratio is then
`null` and threshold 3 fails.

## Root-crate APIs used

None. The crate does not depend on `holographic-memory`; all kernels are local
(`src/lib.rs` `dot_f32` / `dot_i8` lane-accumulator dots, `u64::count_ones`
popcounts, `src/codes.rs` `fwht`).

## Known limits / risks

- Untested code: compile errors are possible; the most likely spots are the
  `&[u64; W]` `try_into` conversions in `src/scan.rs` and the closure
  coercions to `&mut dyn FnMut(usize, &mut Vec<u32>)` in `src/main.rs`.
- The `sym1` residual estimator approximates `sum_i sign(y_i) z_i` by
  `mean|z| (B - 2 hamming)`; `asym4` is the exact 4-bit-query form. If `sym1`
  misses recall but `asym4` passes it at lower QPS, the honest verdict is the
  `asym4` row.
- The int8 baseline is plain auto-vectorized Rust (no `sdot` intrinsics), the
  same treatment the split kernel gets. A hand-tuned int8 kernel would narrow
  the ratio; the JSON records the compiler target features for context.
- Head scale and rho are stored as IEEE half (`f32_to_f16_bits`, round to
  nearest even); the scan decodes them with a branch-free multiply.
- Synthetic clusters are an easy structure for a 16-dim PCA head; the NFCorpus
  and glove runs are the real-distribution check and should be weighed more.
