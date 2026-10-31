# Output JSON schemas

Files in this directory are written by `warm-research-witness` (see `../RUN.md`
for the commands). JSON results are git-ignored; this file documents their
shape. All byte counts are bytes, times are milliseconds unless the field name
ends in `_us` (microseconds), fractions are in `[0, 1]`.

## `stage1_<dataset>_<n>.json` (`schema = "warm-research-witness/stage1/v1"`)

```text
{
  schema: string, generated_unix_s: u64,
  dataset: {
    kind: "glove" | "nytimes", name, dir,
    n_docs, n_train_total, dim, dim_pad,
    n_report_queries, n_select_queries, sample_rows, seed, top,
    load_ms, ground_truth_ms,
    gt_file_agreement: f64 | null,   // mean overlap of recomputed top-`top` with neighbors.i32[:top]
    target: string                   // arch + detected SIMD features
  },
  sweep: { ms: [usize], bs: [usize], qs: [usize], ts: [u8] },
  gate: { min_candidate_recall, min_rank10_survival, max_median_candidates, max_p95_candidates,
          max_median_posting_kb, max_p95_posting_kb, max_projsel_us },
  banks: [ {
    m, directions (= m/2), blocks (= directions / dim_pad), grid_levels (12), table_bytes,
    fit_ms, build_all_budgets_ms, build_threads,
    budgets: [ {
      b, tail_p (= b/m), sample_rank (sample rows at or above the threshold),
      total_memberships, mean_memberships_per_vector, posting_bytes,
      occupancy: { channels, total, mean, min, p05, p50, p95, max, empty }
    } ]
  } ],
  configs: [ {
    m, b, q, t,
    expected_posting_bytes,          // 4 * q * n_docs * b / m
    report: SetStats, select: SetStats,
    fails_report: [string], fails_select: [string],
    pass_report: bool, pass_select: bool
  } ],
  curve:  [ CurvePoint ],            // every config, reporting set, sorted by median_posting_bytes asc
  pareto: [ CurvePoint ],            // frontier of curve (fewer bytes, higher candidate recall)
  n_pass_report, n_pass_select
}

SetStats = {
  n_queries,
  candidate_recall_at_top,           // mean over queries of |emitted ∩ exact top-`top`| / top
  rank10_survival,                   // fraction of queries whose `top`-th exact neighbour has >= t witnesses
  candidates:              Summary,  // ids with >= t witnesses per query
  posting_bytes:           Summary,  // 4 * sum of selected list lengths per query
  projection_selection_us: Summary,  // per-query wall time of project + select
  mean_true_sim_rank10,              // mean cosine of the `top`-th exact neighbour
  survival_by_sim: [ {               // 11 bins: [-1,0), [0,0.1), ..., [0.9,1.0]
    lo, hi, n, rank10_survival,      // rank-10 neighbour binned by its cosine
    all10_n, all10_survival          // all top-`top` neighbours binned by their cosine
  } ]
}
Summary    = { mean, median, p95, max }   // nearest-rank percentiles
CurvePoint = { m, b, q, t, median_posting_bytes, p95_posting_bytes, median_candidates,
               candidate_recall_at_top, rank10_survival, pass }
```

## `stage2_<dataset>_<n>_m<M>_b<B>_q<Q>_t<t>.json` (`schema = "warm-research-witness/stage2/v1"`)

```text
{
  schema, generated_unix_s,
  dataset: (as above),
  config: { m, b, q, t }, top,
  build: { fit_ms, postings_ms, total_ms, threads, total_memberships,
           mean_memberships_per_vector, occupancy: {...} },
  index_bytes: { postings (ids + offsets), bank_tables (signs + mean + inv_std + grid),
                 thresholds (4 * m), rerank_rows (4 * n_docs * dim), total },
  query: {
    n_queries, runs, warmup_queries,
    run_ms: [f64],                   // one entry per timed pass over the reporting queries
    median_run_ms, qps,              // qps = n_queries / median_run_ms
    mean_us,                         // median_run_ms * 1e3 / n_queries  (gate value)
    latency_us: Summary,             // per-query wall time from one instrumented pass
    candidates: Summary, posting_bytes: Summary
  },
  recall_at_top,                     // complete query vs exact top-`top`, reporting set
  thresholds: [ { name, value, limit, pass } ],
  pass
}
```
