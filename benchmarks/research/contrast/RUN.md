# contrast: counterfactual structural contrast kill test

Crate `hms-research-contrast` (standalone, `benchmarks/research/contrast/`, binary `hms-contrast`).
Nothing outside this directory is touched. No `Cargo.lock` is committed.

## Mechanism under test

Ingest: for a document's ordered stems `w_0..w_{n-1}`, `S_d = sign(sum over delta in {1,2,3}, i of
bind(I[w_i], rho^delta(I[w_{i+delta}])))` with `I[w] = splitmix64 stream seeded by FNV-1a(stem)`
(no vocabulary table), `bind = XNOR` on sign bits (bipolar multiply), `rho = cyclic rotate` of the
2,048-bit ring. Stored as `[u64; 32]` (256 bytes) plus the un-signed sum clamped to int8 (ablation).

Query: `Q+` built the same way; foils with the same token multiset but broken order:
`F_rev` (every pair reversed), `F_delta` for each distance (only pairs at that distance reversed), and,
in Part B where roles are known, the agent/patient-swapped rendering of the query.

Scores: positive-only `<Q+,S_d>`; mean-foil contrast `<Q+ - lambda*mean(F), S_d>` as ONE dot
product of a precomputed exact-integer contrast vector (`lambda` in quarters); max-foil contrast
`<Q+,S_d> - max_m <F_m,S_d>`; abstention gate = threshold on the accept statistic calibrated at
TAR 0.80.

Baselines in the same binary: own BM25 over the same stems (root constants `k1 = 1.2`, `b = 0.75`),
raw unigram overlap, cleanup energy (cosine of `Q+` against the int8 un-signed trace), exact hashed
ordered-pair `(w_i, w_j, delta)` counts with merge-join overlap, exact tuple lookup (Part B only).

## Root-crate APIs

None are called: every needed root function is `pub(crate)`. Copied instead:

* tokenizer + Snowball English stemmer: `src/core/documents.rs:381-384` (stemmer) and
  `src/core/documents.rs:388-403` (`terms`), reproduced order-preserving in `src/bm25.rs::stems`
  using the root dependency `rust-stemmers = "1.2.0"` (root `Cargo.toml:72`);
* BM25 constants and weight formula: `src/core/documents.rs:116-121` (defaults) and
  `src/core/documents.rs:147-156` (`norm`, `weight`). The root idf was not read; this crate uses the
  Lucene/BEIR form `ln(1 + (N - df + 0.5) / (df + 0.5))`. The root's document search is only
  reachable through `HmsCore` and was not used, so Part A's BM25 is an own implementation over the
  same stems (expect a number near but not equal to the measured root BM25 0.2916).

Dependencies (all already in the root manifest, same versions): `serde 1.0`, `serde_json 1.0`,
`anyhow 1.0`, `clap 4.4` (derive), `rust-stemmers 1.2.0`. No new third-party crates.

## Commands

Run from anywhere; output paths are absolute (resolved from the crate directory at compile time).

```sh
cd /Volumes/A/holographic-memory/benchmarks/research/contrast

# unit tests (hv rotate/bind/dot identities, foil == reversed-sequence sketch,
# single-dot contrast == linear combination, synth twin multiset invariant, metrics)
cargo test --release

# Part A: NFCorpus rerank, alpha/lambda tuned on SciFact, reported on all 323 NFCorpus queries
cargo run --release --bin hms-contrast -- nfcorpus
#   -> out/part_a.json ; stdout line "PART A: ... -> PASS|FAIL"

# Part A robustness variant: tune on the first 100 NFCorpus queries, report on the remaining 223
cargo run --release --bin hms-contrast -- nfcorpus --tune nfcorpus-head --head 100 --out out/part_a_head100.json

# Part B: synthetic agent memory (2,000 docs, 500 + 500 queries, 20 candidates, 7 timed repeats)
cargo run --release --bin hms-contrast -- synth
#   -> out/part_b.json, out/synth_corpus.jsonl, out/synth_queries.jsonl
#   stdout "PART B: ... -> PASS|FAIL" and zero or more "DEMOTE: ..." lines

# both with defaults
cargo run --release --bin hms-contrast -- all
```

Options: `nfcorpus --data DIR --tune-data DIR --candidates 200 --head 100 --out PATH`;
`synth --seed U64 --docs 2000 --queries 500 --candidates 20 --repeats 7 --warmup 1 --out PATH --no-dump`.
Data defaults: `$HOME/.cache/hms-bench/nfcorpus` and `$HOME/.cache/hms-bench/scifact`.

Expected wall clock (single thread): Part A ~1-2 min (sketching 3,633 + 5,183 abstracts dominates),
Part B ~1 min (7 timed passes over 8 methods).

## Tuning / calibration protocol

* Part A: fusion `minmax(BM25) + alpha * minmax(feature)` within each query's BM25 top-200.
  Grid `alpha in {0, .05, .1, .2, .3, .5, .75, 1, 1.5, 2}`, `lambda in {.25, .5, .75, 1, 1.5, 2}` (mean-foil only).
  Default tuning set: the SciFact split at `~/.cache/hms-bench/scifact` (300 queries, 339 qrels; this is
  the BEIR `scifact` query file present on the machine, which is its test split, not train; it is a
  disjoint corpus, so no NFCorpus query is touched by tuning). The `--tune nfcorpus-head` mode tunes on
  the first 100 NFCorpus queries and reports on the untouched 223.
  nDCG@10 is graded (gain = qrel score, pytrec_eval form); queries without qrels are skipped; qrel docs
  absent from the corpus keep their gain in the ideal DCG.
* Part B: queries are ordered present then absent; the first half of each class is the calibration
  half (chooses `lambda` for mean-foil by AUC and the accept threshold at TAR 0.80), the second half
  gives FAR / TAR / AUC. `far_at_tar80_oracle` uses a threshold set on the evaluation half itself.
  Accept statistic = top candidate's score; for a present query whose top candidate is not the source
  document the statistic is `-inf` (counted as a reject). Latency = median over `--repeats` passes
  (after `--warmup`) of one full pass over all 1,000 queries, divided by 1,000; each pass rebuilds the
  query-side representation (sketch, foils, contrast vector, pair table, term counts) and scores the
  20 candidates. `added_latency_us = method - bm25`.
  Voice: 30% of document facts are passive; a query uses the document's voice with p = 0.7, so the
  `voice_match` / `voice_mismatch` breakdown shows what order-based matching loses across voices.

## Pass / fail

Part A (stdout `PART A:`; `verdict.pass` in JSON): best of {mean-foil, max-foil} eval nDCG@10
`>= 0.312` AND `>= positive-only + 0.008`.

Part B (stdout `PART B:`; `verdict` in JSON): the contrast method with the lowest calibrated FAR must
satisfy all of: `FAR <= 0.5 * min(FAR of bm25, raw_overlap, cleanup_energy)`;
`AUC >= max(baseline AUC) + 0.10`; `added_latency_us < 100`.
DEMOTE rule: a `DEMOTE:` line is printed (and listed in `verdict.demote`) for each of
`hashed_pairs`, `exact_tuple` whose calibrated FAR is `<=` the contrast FAR at lower per-query latency.
Exact tuple lookup is the oracle wherever tuples exist; it is expected to print DEMOTE for Part B, and
the interesting question is whether hashed pairs also do.

## Output files

All under `benchmarks/research/contrast/out/`:

* `part_a.json`, optional `part_a_head100.json`
* `part_b.json`, `synth_corpus.jsonl` (one `SynthDoc` per line: id, twin, text, facts),
  `synth_queries.jsonl` (one `SynthQuery` per line: id, doc, present, text, swapped_text, triple,
  voice_match, passive, candidates)

### part_a.json

```
{
  "part": "A",
  "tune_mode": "Scifact" | "NfcorpusHead",
  "eval_set":  { "name", "dir", "n_queries", "candidates_k", "recall_at_k",
                 "n_docs", "sketch_build_us_per_doc", "mean_pairs_per_doc" },
  "tune_set":  { same fields },
  "variants": [
    { "name": "bm25" | "positive" | "positive_int8" | "mean_foil" | "max_foil" | "hashed_pairs",
      "alpha": f64, "lambda": f64,
      "ndcg10_tune": f64, "ndcg10_eval": f64,
      "grid": [ { "alpha", "lambda", "ndcg10" } ... ] }
  ],
  "verdict": { "pass": bool, "best_contrast": "mean_foil" | "max_foil",
               "ndcg10_best_contrast", "ndcg10_positive_only", "ndcg10_bm25",
               "threshold_abs": 0.312, "threshold_gain_over_positive": 0.008 }
}
```
`positive_int8` is the ablation that scores `Q+` against the un-signed int8 trace (cleanup energy).

### part_b.json

```
{
  "part": "B",
  "config": { "seed", "n_docs", "n_queries_per_class", "candidates",
              "p_passive", "p_voice_match", "p_query_cause" },
  "n_docs", "n_queries", "calibration": str, "repeats", "warmup", "sketch_build_us_per_doc",
  "methods": [
    { "name": "bm25" | "raw_overlap" | "cleanup_energy" | "positive" | "mean_foil" | "max_foil"
              | "hashed_pairs" | "exact_tuple",
      "lambda": f64 | null,
      "threshold": f64,
      "far_at_tar80_calibrated": f64, "tar_realized": f64, "far_at_tar80_oracle": f64,
      "auc_eval": f64, "auc_all": f64,
      "latency_us_per_query": f64, "added_latency_us": f64,
      "voice_match":    { "n_present", "n_absent", "far", "tar", "auc" },
      "voice_mismatch": { same } }
  ],
  "verdict": { "pass", "pass_far", "pass_auc", "pass_latency",
               "contrast_method", "contrast_far", "contrast_auc", "contrast_added_latency_us",
               "baseline_best_far", "baseline_best_auc",
               "demote": [ { "method", "far", "latency_us_per_query" } ... ] }
}
```

## Known limits

* Sketch capacity: an abstract of ~250 stems superposes ~750 pair terms into 2,048 bits, so each
  query pair's expected signal is ~`2048 * sqrt(2 / (pi * 750))` ~ 60 against ~45 noise. Part A
  measures whether that regime carries any rerank signal; it is not tuned around.
* Passive voice is not canonicalized: an order-based sketch of `b was warned by a` does not match
  `a warned b`. The `voice_mismatch` breakdown quantifies the loss instead of hiding it.
* `added_latency_us` for the contrast methods includes building `Q+`, four generic foils and the
  role-swap foil (about 2x the cost of `Q+` thanks to shared per-distance partial sums) plus the
  contrast vector; a 20-candidate dot pass is < 2 us.
