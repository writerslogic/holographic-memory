# Public benchmarks

Measured 2026-10-05 on an Apple M4 (10 cores, macOS). Results: `benchmarks/results/public_*.json`.
Harness: `benchmarks/public/prepare.py` (downloads and embeds data outside the repository),
`src/bin/public-bench.rs` (drives HMS) and `benchmarks/public/evaluate.py` (scores runs and runs
the comparison libraries on the same data).

The machine was not idle: load average ranged from 4 to 37 during the runs, and every result
file records it. Recall and nDCG are unaffected by load. Latency and queries/second are
indicative; the gaps reported below are far larger than the load effect.

## Text retrieval (BEIR)

BEIR test splits, scored with `pytrec_eval`. Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, L2-normalized, computed by `prepare.py`.

| Method | SciFact nDCG@10 | SciFact R@100 | NFCorpus nDCG@10 | NFCorpus R@100 |
|---|---|---|---|---|
| HMS hybrid (document API, stemmed BM25 + exact cosine, min-max score blend) | **0.729** | 0.953 | **0.355** | 0.322 |
| HMS lexical (document API BM25, Porter stemming) | 0.688 | 0.925 | 0.323 | 0.249 |
| HMS dense (document API, exact cosine) | 0.645 | 0.925 | 0.317 | 0.311 |
| HMS sparse vector path (`from_dense` + inverted index, D=16384) | 0.470 | 0.779 | 0.209 | 0.207 |
| Reference: exact cosine in NumPy, same embeddings | 0.645 | 0.925 | 0.316 | 0.312 |
| Reference: BM25 as published in the BEIR paper (Thakur et al. 2021, main nDCG@10 results) | 0.665 | | 0.325 | |
| Previous HMS hybrid (unstemmed BM25, reciprocal rank fusion; measured 2026-10-05) | 0.683 | 0.954 | 0.344 | 0.325 |
| Previous HMS lexical (unstemmed) | 0.662 | 0.886 | 0.307 | 0.237 |

Re-measured 2026-10-07 (load 70 to 119, which does not affect nDCG) after two document-API
changes chosen on NFCorpus dev and SciFact train qrels only (`benchmarks/results/proxies_2026-10_beir.json`;
the test qrels were read once, for this table): Porter (Snowball English) stemming of BM25 terms
(+0.024 / +0.032 on dev/train; +0.026 / +0.016 on test) and a min-max score blend of the lexical
and semantic lists in place of reciprocal rank fusion (+0.017 / +0.016 on dev/train; RRF remains
available as `fusion: "rrf"`). Chunks stored before stemming keep matching unstemmed query
terms.

- HMS's dense search matches the independent NumPy reference. This is a correctness check of the
  document API; it says nothing new about the embedding model.
- HMS's BM25 is now above the published BM25 on SciFact (+0.023) and 0.002 under it on NFCorpus.
- Hybrid search is the best configuration on both sets: +0.084 / +0.038 nDCG@10 over dense alone.
- The sparse vector path loses about 27% (SciFact) and 34% (NFCorpus) of nDCG@10 relative to exact
  cosine on the same embeddings. It is not suitable as the primary semantic retrieval path; the
  document API is.

## Vector search (ann-benchmarks)

Datasets and ground truth from [ann-benchmarks](https://ann-benchmarks.com): `nytimes-256-angular`
(290,000 train, 10,000 queries) and `glove-100-angular` (1,183,514 train, 10,000 queries). Every
system searches single-threaded, one query at a time; builds use all cores. FAISS 1.15.1,
hnswlib 0.8.0.

HMS's only vector-search path is the sparse one: each dense vector is encoded with `from_dense`
(D=16384, 64 active indices) and searched exactly in the inverted index. Its query time includes
encoding, because the other systems take dense input directly.

| System | nytimes recall@10 | nytimes QPS | glove recall@10 | glove QPS | glove memory |
|---|---|---|---|---|---|
| HMS sparse (D=16384) | 0.34 | 204 | 0.25 | 34 | 2.5 GB RSS |
| FAISS flat (exact) | 0.99 | 206 | 1.00 | 124 | 0.47 GB vectors |
| FAISS HNSW M=16, efSearch=16 | 0.69 | 13,782 | 0.56 | 19,861 | 0.64 GB |
| FAISS HNSW M=16, efSearch=128 | 0.88 | 2,875 | 0.83 | 4,111 | 0.64 GB |
| FAISS HNSW M=32, efSearch=512 | 0.96 | 509 | 0.96 | 682 | 0.80 GB |
| hnswlib M=16, ef=128 | 0.88 | 1,202 | 0.83 | 3,368 | 0.63 GB (estimate) |

Full sweeps are in the result files.

- HMS is behind at every operating point. On glove, HNSW reaches higher recall than HMS at roughly
  600 times the throughput, and exact FAISS search is faster than HMS while returning the right
  answer.
- The cause is the encoder, not the index: the inverted index is exact over the sparse codes, but
  the codes keep only part of the neighbour structure. The true top-10 is inside HMS's top-100 for
  54% of results on both sets (`experimental_prefilter` in the result files), so even an exact
  re-rank of 100 candidates would not close the gap.
- Encoding dominates build time (155 s for nytimes, 265 s for glove, all cores) and is about half
  of query time on nytimes.

## Quantized graph index (`core::qgraph`) vs the field

`VGraph` (`core::qgraph::vertex`) is a Vamana-style graph (out-degree 64 on nytimes, 32 on glove,
build_ef 200, alpha 1.0, HNSW-like upper layers for the entry point) with one 8-bit LVQ code per
vertex for the traversal and an 8-bit residual code for the re-rank of the best `rerank` pool
candidates (`public-bench ann-qgraph --index vertex --residual`). An opt-in 4-bit traversal
encoding (`--vertex-bits 4 --residual-bits 8 --align-rows --reorder --id-bytes 3`) costs no recall
and saves 24% of the bytes on nytimes and 16.5% on glove. Parameters were chosen on held-out train
vectors (`--holdout 2000`; the test queries are never read for a decision).
`benchmarks/results/qgraph_merge_verification_heldout.json` (2026-10-07, one clean paired run per
dataset, 11 rounds under one lock) keeps the current search path (x1.30 / x1.38 over the previous
code at recall 0.90 / 0.95 on nytimes, x1.48 / x1.58 on glove, 11 of 11 rounds) and keeps the
8-bit encoding as the default: the 4-bit encoding wins x1.20 / x1.18 on nytimes but only x1.01 /
x0.97 on glove (6 and 3 of 11 rounds), so it is a memory-only gain.

### nytimes-256-angular, test set, Apple M4 (NEON)

`benchmarks/results/public_qgraph_nytimes-256-angular.json` (commit d985308, 2026-10-07):
10,000 test queries, every system single-threaded one query at a time, 3 repeats (median QPS, the
runs are in the file), QPS at a target interpolated in log(QPS) between the Pareto-frontier points
that bracket it; builds use all cores; competitors at the parameters their authors recommend
(`benchmarks/public/ann_extra.py`). Every row ran with the 1-minute load above the gate of 3
(other builds and timings shared the machine; load 5 to 103, recorded per row), so the
absolute QPS are indicative only, and the two HMS rows ran at different loads (9-28 against 7-9).
Table: `uv run --script benchmarks/public/qgraph_table.py nytimes-256-angular`.

| System | QPS @ 0.90 | QPS @ 0.95 | HMS/sys @ 0.90 | Index bytes | sys/HMS bytes | Build s | Load 1m |
|---|---|---|---|---|---|---|---|
| HMS VGraph (8-bit codes, rerank) | 4,409 | not reached | 1.00 | 226,310,960 | 1.00 | 111 (graph from the checked cache: 1) | 9-28 |
| HMS VGraph (4-bit codes, opt-in) | 13,133 | not reached | 0.34 | 171,790,960 | 0.76 | (cache: 4) | 7-9 |
| SymphonyQG (rabitqlib, raw refinement) | 5,913 | not reached | 0.75 | 705,280,166 | 3.12 | 17 | 6-8 |
| SymphonyQG (rabitqlib, 4-bit refinement) | not reached | not reached | | 447,761,210 | 1.98 | 18 | 9-13 |
| FAISS HNSW M=16 | 2,038 | 379 | 2.16 | 338,802,554 | 1.50 | 51 | 11-75 |
| FAISS HNSW M=32 | 2,011 | 509 | 2.19 | 375,882,658 | 1.66 | 93 | 5-64 |
| hnswlib M=16 | 1,352 | 348 | 3.26 | 336,400,000 | 1.49 | 51 | 7-37 |
| RaBitQ HNSW 1+4 | 2,066 | not reached | 2.13 | 95,277,781 | 0.42 | 29 | 11-28 |
| RaBitQ IVF 1+8 | 969 | 803 | 4.55 | 92,637,282 | 0.41 | 15 | 15-29 |
| RaBitQ IVF 1+4 | 940 | not reached | 4.69 | 55,517,282 | 0.25 | 12 | 33-63 |
| Lucene HNSW M=16 | 1,108 | 278 | 3.98 | 406,592,056 | 1.80 | 243 | 6-46 |
| Lucene HNSW M=32 | 358 | 129 | 12.32 | 466,298,840 | 2.06 | 851 | 42-103 |

Read as the file reads it:

- At recall 0.90 the 8-bit VGraph is 2.2x FAISS HNSW (M 16 and 32), 3.3x hnswlib, 2.1x RaBitQ
  HNSW, 4.6x RaBitQ IVF and 4.0x Lucene HNSW on QPS at 0.60-0.67x of their bytes. It LOSES to
  SymphonyQG (the authors' library, raw-vector refinement) by 0.75x on QPS while using 0.32x of its
  bytes. The 4-bit VGraph row is 2.2x SymphonyQG, but it ran at load 7-9 while the 8-bit row ran at
  9-28; the paired held-out ratio of 4-bit over 8-bit is x1.20, not x3. Treat the 4-bit row as
  unconfirmed until a rerun at equal load.
- At recall 0.95 no HMS row is bracketed: both sweeps stopped at ef 512 with recall 0.948 (the test
  set is harder than the held-out train vectors, where ef 384 gave 0.950). FAISS, hnswlib, Lucene
  and RaBitQ IVF 1+8 reach it (379, 509, 348, 129-278 and 803 QPS).
- Memory: RaBitQ IVF (1+4: 55.5 MB, 1+8: 92.6 MB) and RaBitQ HNSW (95.3 MB) are 2.4-4.1x SMALLER
  than the 8-bit VGraph (226 MB) and 1.8-3.1x smaller than the 4-bit one (172 MB). That is a loss
  on index bytes at equal recall.
- Build time: FAISS 51-93 s, SymphonyQG 17 s, RaBitQ 12-29 s, Lucene 243-851 s, VGraph 111 s for
  the full nytimes graph (measured once at load 47 while building the checked cache; the paired
  build numbers are in `qgraph_vertex_build_heldout.json`). Build time is the one metric where
  losing is accepted; it is still a loss against SymphonyQG and RaBitQ here.
- Not run on this machine: NGT (its quantized graph needs the `qbg` command, which the macOS
  Homebrew build lacks; the ONNG stand-in did not build within its 30-minute cap), Glass, DiskANN
  and ScaNN (x86-only builds). They need an x86 machine.
- glove-100-angular was not run in this pass.

The earlier edge-code index (1,024-bit RaBitQ codes per edge, 2.97 GB on nytimes, 7,297 / 1,427
QPS at load below 3) is superseded by `VGraph`; its sweeps stay in
`benchmarks/results/public_qgraph_edge_<set>.json`. The competitors are timed through a Python
per-query loop (Lucene through a Java loop) and HMS through a native loop; that per-query
overhead was not measured.

### Stop rule and 1-bit screen (held-out, 2026-10-07)

`benchmarks/results/qgraph_stop_screen_heldout.json` (held-out train vectors, test set unread). `VSearchParams::patience` ends a
search after that many expansions without a change to the 10 best estimates: 17-29% fewer
estimates at equal recall on both sets, and in paired rounds (load 11-185, gate unmet,
ratios only) x1.33 (4 of 5 rounds) on nytimes and x1.42 (4 of 4) on glove at recall 0.95, neutral
at 0.90; off by default, `--patience` in the bench. The opt-in 1-bit screen before the 8-bit
estimate skips 76% of the estimates on nytimes at equal recall but is slower in nearly every round
(x0.65-0.87): its scalar popcount costs as much as the SDOT estimate; recorded as a negative
result, off by default.

## Agent long-term memory (LongMemEval_S)

Retrieval only, no LLM. Data: LongMemEval_S from the `xiaowu0162/longmemeval-cleaned` release
(revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`, `longmemeval_s_cleaned.json`, sha256
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`), the release the
[LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) now points to (September 2025
clean-up of the history sessions); the paper's numbers predate it. Embeddings: the same
`all-MiniLM-L6-v2` revision as above, L2-normalized. Metrics come from the repository's own
`src/retrieval/eval_utils.py` (commit `9e0b455f`, hash recorded in the result file), applied the way
`run_retrieval.py` and `print_retrieval_metrics.py` do: items are user turns only, a session is
the concatenation of its user turns, and the 30 abstention questions plus questions with no
answer-bearing user turn are excluded, leaving 419 of 500 questions (only 5 of the 56
single-session-assistant questions remain, so that row is not informative). Each question is
searched in a fresh store holding only its own haystack (about 48 sessions, 245 user turns).
Recall is `recall_all` (every evidence item in the top k) and nDCG is `ndcg_any`, the two figures
the official script prints; `recall_any` is in the result file. "Round" is the official
user-turn granularity. Turn-level rows score the round run against the labelled turns;
session-level rows of a round run use the official turn-to-session conversion.
Results: `benchmarks/results/public_longmemeval_s.json`.

| Granularity | Level | Method | Recall@5 | Recall@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|
| Round | Turn | HMS dense (exact cosine) | 0.511 | 0.711 | 0.570 | 0.623 |
| Round | Turn | HMS lexical (BM25) | 0.599 | 0.718 | 0.642 | 0.675 |
| Round | Turn | HMS hybrid | **0.668** | **0.785** | **0.681** | **0.712** |
| Round | Session | HMS dense (exact cosine) | 0.690 | 0.840 | 0.611 | 0.647 |
| Round | Session | HMS lexical (BM25) | 0.668 | 0.768 | 0.658 | 0.684 |
| Round | Session | HMS hybrid | **0.764** | **0.854** | **0.702** | **0.723** |
| Session | Session | HMS dense (exact cosine) | 0.852 | 0.938 | 0.861 | 0.878 |
| Session | Session | HMS lexical (BM25) | 0.823 | 0.893 | 0.862 | 0.877 |
| Session | Session | HMS hybrid | **0.876** | **0.952** | **0.904** | **0.919** |

HMS dense reproduces an independent NumPy exact-cosine ranking on the same embeddings to every
digit (the result file carries both).

On this data and model, hybrid is above dense on every overall metric at every granularity and
level (nDCG@10: +0.089 round/turn, +0.076 round/session, +0.041 session). The runs are single
deterministic passes with no confidence intervals or significance test. Caveats that bound the
comparison: MiniLM truncates input at 256 tokens, so a session embedding sees only the start of
the concatenated user turns (the lexical index sees all of it); HMS BM25 is the engine's own, not
the `rank_bm25` used by the official `flat-bm25`; zero-score items follow scored items in
haystack order in the lexical run.

Recall@10 and nDCG@10 per question type (`recall_all` / `ndcg_any`):

| Type | n | Session run, session | Round run, session (dense) | Round run, session (hybrid) | Round run, turn (dense) | Round run, turn (hybrid) |
|---|---|---|---|---|---|---|
| multi-session | 121 | 0.909 / 0.895 | 0.736 / 0.585 | 0.727 / 0.654 | 0.545 / 0.553 | 0.595 / 0.633 |
| temporal-reasoning | 127 | 0.953 / 0.893 | 0.756 / 0.570 | 0.850 / 0.690 | 0.614 / 0.538 | 0.787 / 0.678 |
| knowledge-update | 72 | 1.000 / 0.982 | 0.986 / 0.686 | 0.986 / 0.772 | 0.903 / 0.673 | 0.986 / 0.772 |
| single-session-preference | 30 | 0.900 / 0.788 | 0.967 / 0.703 | 0.800 / 0.543 | 0.767 / 0.678 | 0.633 / 0.522 |
| single-session-user | 64 | 1.000 / 1.000 | 0.969 / 0.827 | 0.969 / 0.936 | 0.953 / 0.822 | 0.969 / 0.936 |
| single-session-assistant | 5 | 1.000 / 1.000 | 1.000 / 0.900 | 1.000 / 0.877 | 1.000 / 0.900 | 1.000 / 0.877 |

The session-run column is hybrid. The weakest types are multi-session (evidence spread over
several sessions, all of which must be retrieved), single-session-preference, and
temporal-reasoning. Preference is the one type where hybrid is below dense at round granularity
(30 questions; lexical matching works against implicit preference questions). Temporal questions
are where round-level hybrid helps most (turn Recall@10 0.614 to 0.787), but they are still
weaker than the other types, and HMS has no date-aware filtering that would use the question
date.

Published numbers (Wu et al., ICLR 2025, Table 3), LongMemEval_M with Stella V5 1.5B:

| Value / key | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| Round, K = V | 0.582 | 0.481 | 0.692 | 0.512 |
| Round, K = V + LLM-extracted facts | 0.644 | 0.498 | 0.784 | 0.536 |
| Session, K = V | 0.706 | 0.617 | 0.783 | 0.638 |
| Session, K = V + LLM-extracted facts | 0.732 | 0.620 | 0.862 | 0.652 |

These are not comparable with the table above: M has about 500 sessions per history against
about 48 in S, the embedding model is a 1.5B-parameter model against a 22M-parameter one, the
dataset release differs, and the paper's recall variant and level mapping were not re-derived
here. No ranking against the paper is implied.

## What this means

- For semantic search over text, use the document API: exact cosine reproduces the embedding
  model's quality, and hybrid search with the built-in BM25 improves on it.
- Do not use the sparse vector path (`memorize_vector` / `query_vector`) where nearest-neighbour
  recall matters. Improving it needs a different encoder; making the current one faster does not.
- As a vector index, `VGraph` is ahead of FAISS HNSW, hnswlib and Lucene HNSW on single-thread QPS
  at recall 0.90 and below them on bytes on nytimes (Apple M4, load above the gate: indicative), and
  behind SymphonyQG on QPS and behind RaBitQ on bytes. No x86 number exists yet, and the
  distinctive parts of HMS (the compositional VSA algebra, structured memory and provenance) are
  not what these benchmarks measure.

## Reproduce

```sh
uv run --script benchmarks/public/prepare.py beir scifact
uv run --script benchmarks/public/prepare.py ann glove-100-angular
cargo build --release --features cli --bin public-bench
./target/release/public-bench beir --data ~/.cache/hms-bench/scifact --dim 16384 --out scifact.json
uv run --script benchmarks/public/evaluate.py beir scifact scifact.json
./target/release/public-bench ann --data ~/.cache/hms-bench/glove-100-angular --dim 16384 --out glove.json
uv run --script benchmarks/public/evaluate.py ann glove-100-angular glove.json
uv run --script benchmarks/public/prepare.py longmemeval s
./target/release/public-bench longmemeval --data ~/.cache/hms-bench/longmemeval_s --out lme.json
uv run --script benchmarks/public/evaluate.py longmemeval s lme.json
```

## On-device model stages (`local-models`), Apple M4, 32 GB

Measured 2026-10-05 with `public-bench lme-model --stage device` (Metal; encoders f32, LLM bf16;
Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, Qwen3-4B-Instruct-2507 at the pinned revisions), every
stage on, through the document API. The first 10 LongMemEval_S sessions (mean 127 user words) were
ingested one session per document; 10 questions were searched with `k = 10`. Small sample: these
are data points, not a distribution.

| Measure | Value |
|---|---|
| Model load (all three) | 25.6 s |
| Ingest per session (fact extraction + embedding) | mean 13.8 s, median 1.5 s, max 48.7 s |
| Query (LLM rewrite + 2 query embeddings + hybrid search + re-rank of 20) | mean 10.3 s, median 10.6 s, max 12.8 s |
| Peak memory footprint (`/usr/bin/time -l`) | 17.5 GB (max RSS 9.9 GB) |

Ingest time is dominated by the LLM's output length (sessions with many facts take tens of
seconds); query latency by the rewrite generation. The CPU path was checked for correctness, not
timed end to end. Embedder and re-ranker parity with the Python stages on 48 turns / 16 questions /
48 pairs: cosine >= 0.9999997, p(yes) max |diff| 8.4e-8 (CPU) and 7.4e-8 (Metal).

### Batched model stages on Metal

`benchmarks/results/local_models_batching.json`. HMS now vendors the Qwen3 forward pass so that
Metal and CUDA can batch: the decoder runs continuous batching with left-padded rows, the encoders
run right-padded batches. Inputs: the first 64 distinct LongMemEval_S haystack sessions
(`prompts::fact_prompt`, greedy, up to 1536 new tokens), 96 user turns, 96 questions and 96
(question, turn) pairs. Timing is paired (arms alternating, 2 rounds, 1-minute load average below 8 before every run). Throughput
excludes model load.

| Stage | Batch 1 | Best batch | Parity vs batch 1 |
|---|---|---|---|
| Fact extraction, 16 sessions | 0.074 sessions/s, 9.6 tokens/s, peak 8.9 GB | batch 8: 0.146 sessions/s, 18.1 tokens/s, peak 13.7 GB | 56 / 64 outputs identical; the other 8 parse to different fact lists (205 vs 202 facts; one session went from 3 facts to none) |
| Embedding, 96 turns | 22.1 turns/s | batch 4: 27.1 turns/s (30.2 in the default-arm run) | min cosine 1 - 5.4e-12, max abs diff 3.6e-7 |
| Re-ranking, 96 pairs | 9.1 pairs/s | batch 1 (batch 2: 8.8, 4: 8.7, 8: 8.0) | batch 16: max p(yes) diff 3.6e-7 |

On the full 64 sessions, untimed: batch 1 took 829 s (7,851 tokens) and batch 8 434 s (7,812
tokens), peak memory footprint 9.6 GB and 19.2 GB.

Defaults on Metal and CUDA: the decoder stays at batch 1 because batch 8 changed 12.5% of fact
outputs, which is above the 5% bar for making it the default; set `--batch` (or
`Generator::set_batch`) to trade that for about 2x throughput. The embedder defaults to batch 4 and
the re-ranker to batch 1. CUDA was not built or run for this measurement, so the CUDA defaults remain
unverified with batching.
