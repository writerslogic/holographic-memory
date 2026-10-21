# hqb: holographic query batching kill test

Can K concurrent queries be superposed into one d-vector so that a single
pass over the corpus yields candidates for all K, each then reranked exactly?
The scan would cost d*N/K multiply-accumulates (MACs) per query instead of
d*N. This package measures recall@10 after rerank, candidates per query, scan
and rerank MACs per query, single-core wall time per query, and
recall-per-FLOP against the single-query scan, for every variant, batch width
K in {1,2,4,8,16} and pool size C in {200,1000,5000}, on two 50k corpora.

Pure Python + numpy; no root-crate API is used (nothing to quote). No build.

## Command

```sh
# full run (default output: benchmarks/research/hqb/out/hqb_v1.json)
uv run --no-project --with numpy python benchmarks/research/hqb/hqb.py

# 10-20 s smoke run first if desired (n=5000, 50 queries, out/hqb_quick.json)
uv run --no-project --with numpy python benchmarks/research/hqb/hqb.py --quick

# synthetic only, if ~/.cache/hms-bench/nfcorpus is unavailable
uv run --no-project --with numpy python benchmarks/research/hqb/hqb.py --datasets synthetic
```

Expected full-run wall time on one core: 1.5 to 3 minutes (dominated by the
timed passes; the corpus matvec for 50k x 768 f32 is a few ms). Peak memory
about 700 MB. Thread caps (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, ...) are forced to 1 inside the
script before numpy is imported; the values are echoed in `config.thread_env`.

Options (defaults in brackets): `--datasets synthetic,nfcorpus`, `--n 50000`,
`--queries 500`, `--dim 768`, `--clusters 200`, `--spread 0.75`,
`--spectrum-alpha 0.5`, `--distractor-noise 0.6`, `--seed 42`,
`--ks 1,2,4,8,16`, `--cs 200,1000,5000`, `--runs 5`, `--warmup 1`,
`--timing-queries 16`, `--nfcorpus-dir ~/.cache/hms-bench/nfcorpus`, `--out`.

Exit code: 0 after a completed run (PASS or FAIL is the verdict line, not
the exit code); 2 for bad arguments or an unloadable dataset; 1 if the JSON
cannot be written.

## Datasets

* `synthetic`: 50,000 x 768 anisotropic Gaussian clusters, the recipe of
  `benchmarks/research/scan/src/data.rs` (200 centres `normalize(w*g)` with
  `w_j=(j+1)^-0.5`, doc i in cluster i mod 200, `normalize(c + 0.75*(w*g)/|w|)`),
  500 held-out queries drawn the same way. numpy PCG64 replaces xoshiro, so
  rows are not bit-identical to the Rust generator.
* `nfcorpus`: the 3,633 real MiniLM-L6 384-d rows from
  `~/.cache/hms-bench/nfcorpus/corpus.f32` padded to 50,000 with on-manifold
  distractors `normalize(x_j + 0.6*g/sqrt(384))` around random real rows;
  queries are the real queries after a seeded shuffle. The file holds 323
  queries, so `n_queries = 323` (the brief's 500 is `n_queries_requested`);
  no synthetic queries are added to the real set.

Ground truth is the exact top-10 by dot product (rows are l2-normalised).

## Variants

| variant | mode | select | one pass costs | candidates per query |
|---|---|---|---|---|
| `single` | | exact | d*N per query | none (exact) |
| `v2` ordinary batching (baseline) | | exact | d*N per query (corpus streamed once per K) | none (exact) |
| `v1` superposition, shared pool | `key` (signed-permutation keys P_i) or `sum` (P_i = I) | `abs` (top-C by |s|) or `pos` (top-C by s) | d*N per group = d*N/K per query, rerank C*d per query | C shared by the K queries |
| `v3` sign-split (disjoint blocks of d/K coords) | `axis` (coordinate permutation) or `rot` (Haar rotation first) | `own` (top-C by own partial score) | d*N per group = d*N/K per query, rerank C*d per query | C per query |

`v1/key` is the construction as stated in the brief; a scalar mixed score
cannot be unbound, so it is expected to behave like random selection.
`v1/sum` is the control where mixing preserves the sum of per-query scores.
`v3` is, per query, a random (d/K)-dimensional coordinate-subspace sketch;
the superposition only packs K sketches into one vector and the recall of a
query does not depend on which other queries share its group.

Timing: for each (variant, mode, select, K, C) the real per-group code path
(superpose or transform, matvec(s), argpartition, gather + exact rerank
matmul, top-10) runs over `max(2, timing_queries // K)` full groups: one
warmup pass (whose top-10s are compared with ground truth and reported as
`path_recall_at_10_timing_groups`, the cross-check that the timed path and
the batched recall arithmetic agree), then `--runs` timed passes; the median
of microseconds per query is reported with the min and all runs. Recall over
all queries is computed with the same selection rules in batched form, where
the exact rerank is a lookup into the exact score matrix (identical values,
different BLAS call).

## Pass rule

`PASS` is printed as the last stdout line when, on every evaluated dataset,
some `v1` or `v3` row has `K >= 4`, `C <= 1000`, `recall_at_10 >= 0.95` and
`scan_flop_reduction_vs_single >= 2` (which equals K for v1/v3). Otherwise
`FAIL`. Per-dataset verdicts and the best eligible row per variant are
printed just above and stored in `verdict.per_dataset` and
`datasets.<name>.best_eligible`. Note the rule counts scan FLOPs only;
`total_flops_per_query` (scan + rerank) and `recall_per_total_flop_vs_single`
are reported alongside so the rerank cost is visible.

## Output: `out/hqb_v1.json`

```
{
  schema: "hqb_v1", generated_at, seed,
  config: { ks[], cs[], top, runs, warmup, timing_queries, n, queries,
            synthetic{dim,clusters,spread,spectrum_alpha},
            nfcorpus{dir,distractor_noise}, quick, flop_convention, thread_env{} },
  environment: { python, numpy, platform, machine, blas },
  datasets: {
    <name>: {
      name, source, n_docs, n_real_docs, dim, n_queries, n_queries_requested,
      load_ms, ground_truth_ms, extra{}, single_us_per_query,
      rows: [ {
        variant: "single"|"v2"|"v1"|"v3", mode: ""|"key"|"sum"|"axis"|"rot",
        select: "exact"|"abs"|"pos"|"own", K, C (0 for exact), groups, exact_scan,
        recall_at_10,                       # after exact rerank, mean over all queries
        rerank_candidates_per_query,        # C (0 for exact scans)
        scan_flops_per_query,               # nominal MACs: d*N (exact) or d*N/K
        scan_flops_per_query_actual,        # v1 pays a full pass for a short last group
        rerank_flops_per_query,             # C*d
        total_flops_per_query,              # scan + rerank
        scan_flop_reduction_vs_single,      # d*N / scan_flops_per_query (= K)
        recall_per_scan_flop_vs_single,     # recall * reduction
        recall_per_total_flop_vs_single,    # recall * d*N / total
        us_per_query_median, us_per_query_min, us_per_query_runs[],
        timing_groups, timing_queries, path_recall_at_10_timing_groups,
        speedup_vs_single_wall              # single us / this us
      } ... ],
      best_eligible: { v1: {...}, v3: {...} },   # highest recall at K>=4, C<=1000
      pass_rows: [...], pass: bool, notes: []
    }
  },
  verdict: { pass: bool, rule, per_dataset: {name: bool} },
  total_ms
}
```

FLOP convention: one multiply-accumulate = 1, as in the brief (d*N per
single-query scan). The query transform of `v3/rot` (K*d*d per group) and
the superposition itself are not counted as scan FLOPs but are inside the
wall time.

## What to read in the result

* `v1/sum` vs `v1/key`: whether any mixed-score signal survives (key should
  sit at recall ~ C/N; sum should degrade with K as the other queries' scores
  add noise, faster on nfcorpus whose query-doc cosines are small).
* `v3/rot` vs `v3/axis`: whether spreading the spectrum rescues the subspace
  sketch; `v3` at K=4 means 192 (synthetic) or 96 (nfcorpus) coordinates per
  query.
* `v2` rows: recall 1.0 at d*N MACs per query but a wall-time speedup from
  streaming the corpus once per K queries; a superposition that beats it on
  recall-per-FLOP must still beat it on wall time to matter for the
  cache-resident scan.
