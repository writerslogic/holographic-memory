# roc: dense-channel relevance gate on NFCorpus

Prerequisite measurement: may a MiniLM-L6 cosine channel vote next to BM25 on
NFCorpus, and by how much. Standalone Python (numpy only), no Rust, no network, no root-crate API.

## Command

```sh
uv run --no-project --with numpy python benchmarks/research/roc/roc.py
```

Options: `--cache <dir>` (default `~/.cache/hms-bench/nfcorpus`), `--out <json>` (default
`benchmarks/research/roc/out/roc_v1.json`; the directory is created). Works from any cwd.

Expected wall clock: 10-40 s on one CPU core (BM25 indexing a few seconds, 2 x 1440 sigmoid-gate
configurations on compact candidate arrays a few seconds, ROC under 2 s). Deterministic: fixed
tie-break, fixed bootstrap seed, no hash-order dependence (query terms are iterated sorted).

## Inputs (read-only)

`<cache>/{meta.json, corpus.jsonl, queries.jsonl, qrels.tsv, corpus.f32, queries.f32}`. The script
checks the jsonl row counts against meta.json and the f32 byte sizes against n x dim x 4, and
records row norms and sha256 of every input in the output. It exits with a message naming the
file on any mismatch.

Split facts recorded in the output: the cache holds 323 queries / 12334 judgments (grades 1 and
2 only), which is the BEIR nfcorpus *test* split by count; the reference numbers (BM25 0.2916,
MiniLM 0.3024, gte-small 0.3398) were measured on dev (324 queries) and are printed for
orientation only. BEIR train qrels are in `~/.cache/hms-bench/downloads/nfcorpus/qrels/train.tsv`
but there are no embeddings for train queries, so the disjoint tuning split is by 0-based query
index parity in queries.jsonl order: tune = odd, eval = even (primary fold); the swapped fold and
a count-weighted cross-fit mean are also reported.

## Output

`benchmarks/research/roc/out/roc_v1.json` (schema `hms.research.roc/1`), floats rounded to 6
decimals, `null` for undefined AUCs. Top-level keys:

| key | content |
|---|---|
| `protocol` | exact text of every procedure (BM25, nDCG, candidates, each fusion, gates, split, ROC, recall, oracle, bootstrap, verdict rule) |
| `constants` | k1, b, candidate K, RRF k, gate top-N, bands, weight grid, sigmoid grid, tie epsilon |
| `data` | counts, qrels grade counts, dropped judgments, split identification, embedding row norms, BM25 index size, input sha256s |
| `baselines_all_queries` | BM25 and cosine nDCG@10 over all 323 queries and per parity half; reference points |
| `static_curves_all_queries` | nDCG@10 vs dense weight (0..1 step 0.05) for min-max, z-score, weighted RRF; standard RRF k=60; information only (scored on evaluation queries) |
| `gate_statistic_diagnostics` | per statistic (`ig_kl`, `ig_cv`, `z_margin`): Spearman vs (cos - bm25) per-query nDCG, vs each channel, quantiles, mean channel gap in the low/high half |
| `fusion_selection.primary` | per family: selected params, tune and eval nDCG@10, configs tried; `best`, `best_static`, `best_gated` with deltas vs BM25 eval |
| `fusion_selection.swapped`, `.cross_fit` | the mirrored fold and the count-weighted mean of the two eval scores |
| `bootstrap_primary_best_minus_bm25` | paired percentile bootstrap (2000 resamples, seed 0) of eval-split best fusion minus BM25 |
| `roc` | `band1..band4` (BM25 top-200 positive-score rank quartiles, band1 highest), `all_bands`, `outside_top50`, `all_docs`: cosine pooled/macro AUC over (relevant, non-relevant) pairs, BM25 control AUC in the same cells, pair and cell counts |
| `recall` | `@100`, `@200`: BM25, cosine, union (top-K of each, up to 2K docs), union gain |
| `oracle` | best channel per query (mean, win counts), best min-max weight per query, best of any static sweep |
| `verdict` | `GATE_OPEN` / `GATE_CLOSED` with the two tested numbers and thresholds |
| `timing_s` | per phase |

Fusion families (all ranked inside the union of BM25 top-200 and cosine top-200, parameters
chosen on the tune split): `minmax` (weight sweep), `z` (weight sweep), `rrf` (weighted, k=60),
`rrf_k60_equal` (standard RRF, no tuning), `gate_rankcal_kl_minmax` (principled, label-free:
dense weight = 1 - midrank ECDF of the KL-from-uniform statistic of the top-50 BM25 softmax),
`gate_rankcal_selected` (same calibration, statistic and base normalization picked on tune),
`gate_kl_raw_minmax` (dense weight = 1 - ig_kl, no calibration), `gate_sigmoid` (grid over
statistic x {minmax, z} x centre quantile x temperature x w_lo x w_hi, 1440 configs).

## Pass / fail

The script prints a one-screen summary ending in

```
VERDICT: GATE_OPEN|GATE_CLOSED  (AUC a >= 0.55: yes|no; fusion gain g >= 0.02: yes|no)
```

* `a` = `roc.all_bands.cos_pooled_auc`: pair-pooled AUC of cosine for ordering relevant above
  non-relevant documents that sit in the same BM25 score band.
* `g` = `fusion_selection.primary.best.ndcg10_eval - fusion_selection.primary.bm25.ndcg10_eval`
  (family and parameters chosen on the odd-index tune split, scored on the even-index eval split).

GATE_OPEN requires both. Read with the verdict: the bootstrap CI of `g` (162 eval queries; a
0.02 point estimate can sit inside noise), the swapped fold, `oracle.best_channel` (ceiling for
any one-channel-per-query gate) and `best_gated.delta_vs_best_static_eval` (whether the
dispersion gate buys anything over a fixed weight). The BM25 control AUCs in `roc` should sit
near 0.5 inside bands (the bands remove most of BM25's own order) and well above 0.5 in
`all_docs`; if they do not, the band construction is wrong.

Exit code 0 with the JSON written is the only success signal; any input mismatch exits non-zero
with the offending path in the message.

## Root-crate APIs, dependencies

None from the root crate. Dependency: numpy (1.26 or 2.x), supplied by `--with numpy`.
Python >= 3.11.

---

# relevance: closing the oracle gap above static fusion (round 2)

`relevance.py` imports `roc.py` (same directory) for input parsing, BM25, metrics, pools and the
bootstrap, and tests seven query-time mechanisms against the z-score-sum baseline that round 1
selected (eval 0.3459, oracles 0.3774 best channel / 0.3970 best weight).

## Command

```sh
uv run --no-project --with numpy python benchmarks/research/roc/relevance.py
```

Options: `--cache <dir>` (default `~/.cache/hms-bench/nfcorpus`), `--out <json>` (default
`benchmarks/research/roc/out/relevance_v1.json`). Same input checks as `roc.py` (row counts,
f32 sizes, sha256 recorded). Deterministic: fixed tie-break, bootstrap seed 0, no hash-order
dependence. Expected wall clock 20-60 s on one core (BM25 a few seconds; 2 folds x up to 9
PRF corpus re-scores of 323 x 3633 x 384; everything else is on compact pools). Budget: 3 min.

## Protocol (exact)

* Queries with IDCG@10 = 0 are dropped. BM25 (k1 1.2, b 0.75, HMS tokenizer) and cosine
  (`Q D^T`, l2-normalised rows) as in `roc.py`.
* Pool per query: BM25 top-200 (positive scores only) | cosine top-200; PRF adds | cos' top-200.
  Both channels z-scored over the pool; fused = (1-w) z(BM25) + w z(dense); fused top-100 with
  the lower-document-index tie-break. nDCG@10 (linear gain) and recall@100 on that list.
* Folds: `primary` tunes on odd 0-based query indices (queries.jsonl order) and evaluates on even;
  `swapped` mirrors it. Every parameter is chosen by tune-fold mean nDCG@10; each family's
  held-out number is reported once. The chosen family is the non-baseline family with the highest
  tune-fold score. Nothing that reads eval-fold labels enters a choice.
* Families (dense weight grid 0..1 step 0.05 everywhere; `n_configs` = configurations compared
  on the tune fold):
  * `zsum` baseline (weight w0 by tune); `rrf_k60` competitor baseline (untuned).
  * (a) `prf_dense`: q' = normalize(q + alpha mean(D[zsum fused top-m])), m in {3,5,10}, alpha in
    {0.5,1,2}; cos' over the whole corpus; re-fuse with a second-stage weight (189 configs).
  * (c) `gate_jaccard20`: Jaccard of BM25 top-20 (positive) and cosine top-20, bucketed by
    tune-fold quantiles (2/3/4 buckets); each bucket takes the weight maximising the tune mean of
    its members; a bucket with no tune member takes the global tune-best weight.
  * (d) `gate_cos_margin` (cos top1-top2), `gate_bm25_margin` ((s1-s2)/s1, 0 when s1 <= 0),
    `gate_margin_diff` (difference of the two tune-fold midrank ECDFs), `gate_margin_2d`
    (2x2 or 3x3 quantile grid over both margins); same bucket rule.
  * (e) `centroid_rerank`: + beta z(cos(doc, normalize(mean D[positive BM25 top-m]))) on the
    baseline pool, m in {3,5}, beta in {0.25,0.5,1,2} (168 configs); centroid = q when there is no
    BM25 hit.
  * (f) `two_stage_prf`: zsum fused top-100 re-ranked by (1-lambda) z(fused) + lambda z(cos'),
    lambda in {0.5,0.75,1} over the nine (m, alpha) PRF queries (27 configs); recall@100 is the
    baseline's by construction.
  * (g) `prf_gate_jaccard20`, `prf_gate_margin`: the nine PRF tables with the second-stage weight
    chosen per bucket by the (c) or (d) gates computed on the first-pass channels (27 / 99 configs).
* Per family and fold: `ndcg10_tune`, `ndcg10_eval`, `recall100_tune/eval`, `delta_vs_zsum_eval`,
  `gap_closed_best_weight_eval` = (family - zsum)/(best-z-weight-per-query oracle - zsum) and
  `gap_closed_best_channel_eval` on the eval fold (null when the oracle does not exceed zsum),
  `bootstrap_vs_zsum_eval` (paired percentile bootstrap, 2000 resamples, seed 0, over eval
  queries), `cost.extra_dots` (dim-384 dot products per query beyond the baseline's single corpus
  pass: PRF = corpus size 3633, centroid rerank = mean pool width, two-stage = 100, gates = 0).
* `folds.<fold>.chosen` adds `bootstrap_vs_rrf_eval` and `passes_win_rule`;
  `best_family_by_eval_info_only` is printed for diagnosis only and never drives selection.

## Output

`benchmarks/research/roc/out/relevance_v1.json` (schema `hms.research.relevance/1`), floats to 6
decimals. Keys: `protocol` (procedure text), `constants`, `data` (counts, pool width, input
sha256s), `all_queries` (channel and oracle numbers over all scored queries, the zsum weight curve,
statistic quantiles), `folds.primary` / `folds.swapped` (`bm25`, `cosine`, `oracle_eval`,
`families`, `chosen`), `verdict`, `timing_s`.

## Pass / fail

The summary prints one table per fold (family, tune, eval, R@100, delta vs zsum, CI95, gap closed
vs both oracles, extra dots, configs) and ends with

```
VERDICT: WIN|NO_WIN  (chosen <family>: delta +x.xxxx >= 0.01: yes|no; ci95 [...] excludes 0: yes|no; swapped fold <family> delta +x.xxxx passes: yes|no)
```

WIN iff the family chosen on the primary tune fold beats `zsum` by >= 0.01 nDCG@10 on the primary
eval fold (162 queries) with the bootstrap CI95 lower bound > 0. Read with it: whether the swapped
fold selects the same family and also passes (`verdict.swapped_fold`), `cross_fit_delta_vs_zsum`,
and the `n_configs` of the winner (a 189-configuration grid on 161 tune queries can select noise;
the held-out CI is the guard). A `prf_dense` win costs one extra corpus pass per query; a gate win
costs nothing. Exit 0 with the JSON written is the only success signal; input mismatches exit
non-zero naming the file.

Dependencies: numpy only (`--with numpy`), Python >= 3.11, `roc.py` beside the script.
