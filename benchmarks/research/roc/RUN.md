# roc: dense-channel relevance gate on NFCorpus

Prerequisite measurement for the panel: may a MiniLM-L6 cosine channel vote next to BM25 on
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
