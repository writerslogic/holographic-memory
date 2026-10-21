# hli: holographic late interaction kill test

Can ColBERT-style token-level retrieval (MaxSim over contextual token embeddings) be compressed
into ONE superposed VSA hypervector per document and keep a useful share of the MaxSim-minus-dense
gap at single-vector cost? Standalone Python (numpy + torch + sentence-transformers); no Rust, no
root-crate API. BM25, readers, metrics and the bootstrap are imported from
`benchmarks/research/roc/roc.py` (`read_jsonl` :100, `read_qrels` :126, `build_bm25` :195,
`bm25_scores` :229, `topk` :258, `ideal_dcg` :274, `ndcg10` :280, `topk_mask` :293,
`recall_at` :302, `rounded` :665).

## Command

```sh
uv run --no-project --with numpy --with torch --with transformers --with sentence-transformers \
    python benchmarks/research/hli/hli.py
```

Works from any cwd. Model: `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (the revision recorded in `~/.cache/hms-bench/*/meta.json`,
already in the Hugging Face hub cache), loaded on CPU through `SentenceTransformer(id, revision=..., device="cpu")`
as `benchmarks/embedding/bench_torch_cpu.py` does; `max_seq_length = 256`.

Options (defaults in brackets): `--datasets nfcorpus scifact`, `--cache-root ~/.cache/hms-bench`,
`--out benchmarks/research/hli/out/hli_v1.json`, `--dims 2048 4096 8192 16384`,
`--buckets 1 4 16`, `--kmeans 2 4 8`, `--maps real real_c sign rff`, `--rff-gamma 4`,
`--expand-weight 1`, `--no-additions`, `--threads [min(10, cpu_count)]`, `--batch-size 64`,
`--token-chunk 8192`, `--acc-budget-gb 6`, `--no-timing`, `--time-queries 4`, `--time-repeats 5`,
`--max-docs 0`, `--seed 20241020`.

## Runtime and memory

Both datasets, full grid: about 10-25 min on a 10-core CPU (fp32 GEMM through torch dominates;
roughly 400 TFLOP of scoring + 100 TFLOP of projection), plus 1-3 min of token embedding on the
first run. Peak RSS about 12-14 GB at D = 16384 on SciFact. Progress and per-row results stream to
stderr with elapsed seconds; the JSON is rewritten after every dataset.

Token embeddings are cached in `benchmarks/research/hli/out/<dataset>.<docs|queries>.{tokens,offsets,ids,pooled}.npy`
(float16, ~1.5 GB for both datasets) with a `.cache.json` guard (model, revision, max length,
item count) and the model's word-embedding matrix in `out/word_embeddings.npy`; all gitignored by
`benchmarks/research/hli/.gitignore`. Reruns skip the model entirely.

Resume: every scored `hli` / `hli_puresum` row is appended to `out/<dataset>.rows.partial.json`
as soon as it is measured (with a fingerprint: seed, gamma, expansion weight, subsample, corpus
and query counts). A rerun reloads matching rows, skips their configurations (and the bundling of
accumulators nobody still needs), and recomputes only the cheap dense / BM25 / MaxSim rows, which
carry the per-query reference metrics for the bootstrap. Delete the partial file to rescore.

Run order: D descending (16384 first), and within a D the IDF-weighted and K-bundle rows
(A, A+B, B), then the other additions, then the plain map x bucket grid; `--dims` is sorted
descending whatever order is given.

If a run must be shorter: `--dims 2048 4096 8192` roughly halves the scoring time (D = 16384 is
half the FLOPs), `--no-additions` drops additions A/B/C/E (about 30% of the scoring), `--no-timing`
drops the single-thread micro-benchmark (~2 min), `--maps real sign` drops the centred and RFF maps.
SciFact has 5183 documents, so no subsampling is applied by default; `--max-docs N` keeps every
judged document plus random others up to N (seeded) and records it under `stats.subsample`.

## Inputs (read-only)

`~/.cache/hms-bench/{nfcorpus,scifact}/{meta.json, corpus.jsonl, queries.jsonl, qrels.tsv, corpus.f32}`.
NFCorpus: 3633 docs, 323 queries, 12334 judgments (grades 1-2). SciFact: 5183 docs, 300 queries,
339 judgments. Queries without a positive judgment are dropped (count recorded). The pooled vectors
recomputed here are compared with the cached `corpus.f32` (`stats.pooled_vs_cached_corpus_f32_cosine_mean`,
expected ~1.0; a lower value means the stack drifted from the one that built the cache).

## Protocol

Tokens: last hidden state per word piece (384-d), documents and queries truncated at 256 word
pieces, [CLS]/[SEP] dropped, every token l2-normalised. Pooled vector: the model's own mean pooling
+ l2 (dense baseline). `stats.doc_tokens.frac_at_cap` is the share of documents whose word-piece
count hit 256 (truncated or exactly full).

Mechanisms (rows), every one from the same token bank:

| mechanism | score |
|---|---|
| `dense` | cosine of pooled vectors |
| `bm25` | HMS BM25 (k1 1.2, b 0.75) from roc.py; hybrid partner |
| `maxsim`, `maxsim_idf`, `maxsim_centered` | sum over query tokens of the max cosine over document tokens, / n_q (idf-weighted query tokens; corpus-mean-centred tokens) |
| `hli` | token hypervector h(x) = f(R^T x), grouped, bundled, quantised; see below |
| `hli_puresum` | cosine of the summed query hypervector against the bundle (control; collapses to dense) |

HLI token maps (`token_map`): `real` R^T x; `real_c` R^T (x - mu) / |x - mu| (mu = corpus-mean
token); `sign` sign(R^T x) / sqrt(D); `rff` sqrt(2/D) cos(sqrt(2 gamma D) R^T x + b), random Fourier
features of exp(-gamma |x - y|^2) so that the bundle dot product is a sum of kernel values (a soft
max), gamma = 4. R is one fixed 384 x 16384 Gaussian draw (entries N(0, 1/D) after scaling), the
first D columns for each D.

Grouping (`grouping`, `n_groups`, `store`): `pos<B>` relative position bucket floor(p B / n), bound
by cyclic permutation `np.roll(., g D / B)` and superposed into one D-vector (`sup`); `kmeans<K>`
seeded spherical k-means over the document's tokens, K separate D-vectors (`sep`, memory K x D).
Score: sum over query tokens of the max over groups of the cosine-normalised dot (unbinding on the
query side: G rolled query copies), divided by the query weight mass.

Bundle (`bundle`): `int8` per-row symmetric int8 + fp32 scale (dequantised and l2-normalised for
scoring; memory G x (D + 4) bytes); `bin` sign of the sum, binarised query tokens, score = (+-1
dot) / D, an affine function of the Hamming distance (memory G x D / 8 bytes).

Additions (flags in the row label):
- A `idf`: wordpiece idf (BM25 form, document frequency over the corpus) weights each token
  hypervector before summing and each query token in the score.
- B `kmeans<K>|sep`: K semantic bundles per document, K in {2, 4, 8}.
- C `exp`: expansion by binding. Each document token x becomes l2(x + lambda sum_i s_i p_i) before
  projection, where p_i are the corpus-mean contextual embeddings (prototypes) of the 3 vocabulary
  items nearest to x's word piece in the model's tied word-embedding matrix (cosine s_i, clipped at
  0, among word pieces that occur in the corpus), lambda = `--expand-weight`. Documents only.
- D rerank: every row also reports `metrics.rerank` with candidates = dense top-100 union
  positive-BM25 top-100 (non-candidates unretrieved); `stats.rerank_candidates.recall_ceiling` is
  the candidate set's recall.
- E `msub`: crosstalk cancellation, M_{d,g} -= W_{d,g} h_bar with h_bar the (idf-)weighted mean
  token hypervector over the corpus and W_{d,g} the group's weight mass, applied before quantisation.

Rows per D, in run order: additions on the `real` map: A, A+B (K = 8, 4, 2), B (K = 8, 4, 2),
A+E, A+C, A+C+E, A+C+E+kmeans4, A+C+E+pos4, C, E, each as int8 and bin; then the grid
{real, real_c, sign, rff} x {pos1, pos4, pos16} x {int8; bin for real and sign}; plus the pure-sum
control for each map at pos1/int8. Superposed (`sup`) rows store one D-vector whatever the bucket
count; only `sep` rows store K.

Evaluation modes per row: `first_stage` (exhaustive), `rerank` (D), `hybrid_bm25` (per-query
z-score of the row + z-score of BM25; the hybrid partner for every row, compared with dense +
BM25). Metrics: nDCG@10 (linear gain) and recall@100, on all queries and on the odd-index (tune) /
even-index (eval) query split. Paired percentile bootstrap (2000 resamples, seed 0, one shared
index matrix per dataset and one per eval split) of the nDCG@10 difference against dense (or
against dense + BM25 in hybrid mode). `gap_closed_all` = (row - dense) / (maxsim - dense).

Costs: `memory_bytes_per_doc` (dense 1536; maxsim mean tokens x 384 x 2 for fp16, fp32 also
recorded; int8 bundle G x (D + 4); binary G x D / 8), `madds_per_doc` per query (mean query tokens
x G x D; dense 384; maxsim q x d x 384; pure-sum D), `bitops_per_doc` for binary rows,
`scoring_wall_s` for one full pass at `--threads`, and `ms_per_query_1thread` = median of 5 repeats
after 1 warm-up on 4 queries with torch threads = 1.

Capacity diagnostic (`capacity`): for 200 sampled documents per D, the fraction of a document's own
tokens whose similarity to its bundle (real map, pos1, real and binary bundle) exceeds the 99th
percentile of 2000 foreign tokens, plus the own-vs-foreign AUC, by token-count bin
(1-32, 33-64, 65-128, 129-256); and the same on synthetic isotropic unit tokens for n in
{16, 32, 64, 128, 256} (20 trials). Foreign real tokens can be semantically identical to stored
ones, so the real-token numbers are a lower bound on crosstalk-free recoverability.

Determinism: projection and RFF phases from `--seed`; k-means seed 7; capacity sampling seed 11;
bootstrap seed 0; ties broken by lower document index; torch threads fixed.

## Pass / fail

Per dataset and D (`datasets.<name>.verdict.per_dim.<D>.pass`): PASS iff some `hli` row (not the
pure-sum control) with `memory_bytes_per_doc <= 4 x 1536` beats dense on first-stage nDCG@10 over
all queries by >= 0.02 with the bootstrap 95% CI excluding zero. `verdict.per_dim.<D>.pass_all_datasets`
and `verdict.overall_pass` (some D passes on both datasets) are printed at the end as
`D=<D>: PASS|FAIL` and `OVERALL: PASS|FAIL`. Memory-eligible rows: every superposed int8 row at
D = 2048 and 4096, kmeans2 int8 at D = 2048, and every binary row with (stored vectors) x D <= 49152
bits (all superposed binary rows; kmeans2 up to D = 16384, kmeans4 up to 8192, kmeans8 up to 4096).

`verdict.per_dim.<D>.best` is the best memory-eligible row on all queries with its delta, CI and
`gap_closed`; `rerank_any_pass` repeats the criterion in rerank mode. `verdict.best_by_tune_split`
selects, per mode and for both memory classes, the row with the highest tune-split nDCG@10 and
reports its eval-split nDCG@10, recall@100, delta vs dense and CI (eval bootstrap), memory and gap
closed; that is the "best combination per dataset" answer and is free of test-set selection.

## Output schema (`hms.research.hli/1`, floats rounded to 6 decimals)

```
schema, generated_at, command, model{id, revision, max_seq_length, hidden},
machine{platform, machine, cpu_count, threads, load_average_1m, python, numpy, torch, sentence_transformers},
config{dims, buckets, kmeans, maps, rff_gamma, expand_weight, additions, seed, ...},
protocol{...}                       prose of every rule above
datasets.<name>:
  stats{n_docs, n_queries, n_queries_dropped_no_qrels, qrels, subsample, doc_tokens{mean, median, max, min, total,
        frac_at_cap}, query_tokens{...}, pooled_vs_cached_corpus_f32_cosine_mean, embedding_cache_hit,
        rerank_candidates{mean_per_query, recall_ceiling}, split{n_tune, n_eval}, expansion{vocab_size,
        occurring_wordpieces, top, mean_neighbour_cosine, lambda}}
  rows[]: {label, mechanism, dim, token_map, grouping, n_groups, store, bundle, idf, expand, meansub,
           memory_bytes_per_doc, memory_x_dense, pass_memory, madds_per_doc, bitops_per_doc,
           metrics{first_stage|rerank|hybrid_bm25: {ndcg10{all,tune,eval}, recall100{all,tune,eval},
                   vs_ref_ndcg10_all{mean, ci95, p_mean_le_0, n_resamples, n_queries, seed}, vs_ref_ndcg10_eval{...},
                   vs_ref_recall100_all, vs_maxsim_ndcg10_all, gap_closed_all}},
           scoring_wall_s, threads, ms_per_query_1thread{median_ms_per_query, min_ms_per_query, repeats, warmup,
           n_queries, threads}, pass_quality, pass}
  capacity{real_tokens.<D>{by_token_count.<bin>{n_docs, frac_recoverable_real, frac_recoverable_binary, auc_real,
           auc_binary}}, synthetic.<D>{by_token_count.<n>{real_tokens_real_bundle, real_tokens_binary_bundle,
           sign_tokens_binary_bundle}}}
  verdict{pass_any_dim, per_dim.<D>{pass, n_rows, n_passing_rows, best{...}, rerank_any_pass},
          best_by_tune_split.<mode>.{memory_le_4x_dense|any_memory}{selected_on, tune_ndcg10, eval{...}}}
  wall_s_so_far
verdict{per_dim.<D>{pass_all_datasets, by_dataset}, overall_pass, wall_s}
```

`vs_ref` is dense for every row (dense + BM25 in hybrid mode); rows carry the row label
`<map>|<grouping><G>|<store>[|idf][|exp]|<bundle>[|msub]|D<D>`.

## Caveats

- MiniLM-L6 is a mean-pooling model, not trained for late interaction; `maxsim` is the upper bound
  this token bank affords. If `maxsim - dense` is small, no compression can pass, and
  `gap_closed_all` is the figure to read.
- Expansion (C) is an untrained stand-in for learned expansion: static word-embedding neighbours
  mapped through corpus prototypes. It is reported as a flag, not as SPLADE.
- Memory for int8 rows counts the stored int8 + scale; scoring dequantises to fp32, so the
  multiply-add counts are the fp32 ones.
- The single-thread micro-benchmark is on 4 queries; `scoring_wall_s` is the full pass at
  `--threads` and is the number to use for throughput.
- First run needs the Hugging Face hub cache (or network) for the pinned revision; the word-embedding
  matrix is read from the loaded model once and cached.

## Result (2026-10-21, `out/hli_v1.json`)

FAIL by the pre-registered rule (>= +0.02 nDCG@10 over dense at <= 4x dense memory on both
datasets). NFCorpus: dense 0.3166, MaxSim 0.3490; best memory-eligible row
`real|kmeans8|sep|idf|bin|D4096` 0.3332, +0.0166 [95% CI +0.0040, +0.0294], 2.67x dense memory,
51% of the MaxSim gap. SciFact: dense 0.6451, MaxSim 0.7120; no memory-eligible row reaches +0.02
(best int8 row at 85x memory +0.0297). D = 16384 and 8192 do not beat D = 4096 on NFCorpus
(+0.0034 and +0.0094). The D = 8192 SciFact rows were not scored (run cut at 90 min); they cannot
change the verdict because the D = 16384 rows already fail. Parked: a binary 4096-bit
late-interaction bundle is a significant but sub-threshold gain over dense and is not productized.
