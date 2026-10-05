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
| HMS hybrid (document API, BM25 + exact cosine, rank fusion) | **0.683** | **0.954** | **0.344** | **0.325** |
| HMS lexical (document API BM25) | 0.662 | 0.886 | 0.307 | 0.237 |
| HMS dense (document API, exact cosine) | 0.645 | 0.925 | 0.317 | 0.311 |
| HMS sparse vector path (`from_dense` + inverted index, D=16384) | 0.470 | 0.779 | 0.209 | 0.207 |
| Reference: exact cosine in NumPy, same embeddings | 0.645 | 0.925 | 0.316 | 0.312 |
| Reference: BM25 as published in the BEIR paper (Thakur et al. 2021, main nDCG@10 results) | 0.665 | | 0.325 | |

- HMS's dense search matches the independent NumPy reference. This is a correctness check of the
  document API; it says nothing new about the embedding model.
- HMS's built-in BM25 is within 0.003 of the published BM25 on SciFact and 0.018 below it on
  NFCorpus (HMS lowercases and splits on non-alphanumeric characters, with no stemming or stopword removal).
- Hybrid search is the best configuration on both sets: +0.038 / +0.027 nDCG@10 over dense alone.
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

## LongMemEval retrieval pipeline (S dev split)

Protocol: `benchmarks/public/longmemeval_split.json` freezes 100 dev / 400 held-out question ids
(seed 20261005, stratified by question type and abstention; S and M share the same 500 ids). Only
dev is used for tuning; 84 dev questions survive the official scoring filter, so one question moves
a recall by 0.012. Held-out is scored once, in the final M run.

Pipeline: `longmemeval_pipeline.py` (keys, fusion, official scoring), `longmemeval_modal.py` (GPU
stages on Modal with content-hash caches in the `hms-lme` volume and a hard cost cap; HMS BM25 and
exact cosine from `public-bench lme-scores`), `longmemeval_sweep.py` (dev ablation and tuning,
writes `longmemeval_config.json` and `benchmarks/results/longmemeval_dev_ablation.json`).

Dev baseline (MiniLM hybrid, the harness above restricted to dev,
`benchmarks/results/longmemeval_dev_minilm_baseline.json`): session run R@5 0.893, R@10 0.952,
nDCG@5 0.899, nDCG@10 0.911; round run, turn level R@5 0.607, R@10 0.786, nDCG@5 0.649,
nDCG@10 0.687.

A date-range filter in the document API is not needed for this benchmark: every question's store is
ranked exhaustively, so a date filter or boost applied to the full ranking gives the same result as
an engine-side filter.

Dev models (small, pinned in `longmemeval_modal.py`): Qwen3-Embedding-0.6B (model-card query
instruction, last-token pooling, L2), Qwen3-Reranker-0.6B (model-card yes/no prompt),
Qwen3-4B-Instruct-2507 for user-fact extraction per distinct session text and for query rewrite /
sub-queries / time range. The LLM stage ran on an L4; vLLM on the cheaper T4 was not tried.
Embedding and re-ranking ran on T4. Levers were measured on cached scores (`longmemeval_sweep.py`):
first each lever alone against the baseline, then greedy ascent, then pruning. The baseline uses
user-turn keys for turns, session keys for sessions, the original question, document-API hybrid
RRF and max aggregation. The objective is the mean of recall_all@5/10 and ndcg_any@5/10 at that
level. Greedy ascent kept a change only if it raised the objective by at least 0.005. Pruning then
removed any kept lever whose removal cost less than 0.005.

| Lever (value) | Session, alone | Turn, alone | Kept (drop-one loss) |
|---|---|---|---|
| Lexical weight (0.5; session kept 0.0, dense only) | +0.013 | +0.003 | session 0.0 (0.015), turn 0.5 (0.016) |
| Min-max score blend instead of RRF | +0.017 | +0.005 | no |
| User-turn keys merged into the session ranking | +0.016 | n/a | session (0.021) |
| Round keys (user turn + assistant reply) | +0.010 | -0.015 | no |
| Facts as their own ranked list | -0.152 | -0.148 | no |
| Facts in the value keys' index (paper's "separate" join) | +0.027 | +0.025 | both (0.019 / 0.014) |
| Fact-expanded turn keys (K = V + facts) | +0.018 | +0.014 | turn, weight 2 (0.008) |
| Fact-expanded session keys | -0.006 | n/a | no |
| Aggregation sum2 / sum3 / RRF instead of max | <= -0.003 | <= -0.002 | no |
| LLM rewrite as an extra query | +0.011 | +0.014 | turn (0.015) |
| LLM sub-queries (weight 0.25) | +0.008 | +0.004 | no |
| Session prior for turns | n/a | +0.018 | no |
| Time-range boost from `question_date` (0 to 31 days padding) | <= -0.002 | <= +0.003 | no |
| Cross-encoder re-rank of the top 20 (weight 2; turn top 50: +0.074) | +0.040 | +0.052 | session w 1 (0.015), turn w 8 (0.042) |

Notes on the levers:
- Facts as a separate list lose because extraction yields facts for only some sessions, and rank
  fusion then favours any session that has a fact.
- The time lever did run. All 100 query outputs parsed, and 17 dev questions received a range.
  The gold sessions fall inside it for 9 of those 17; the 4B model's date arithmetic is loose
  (for example, "two weeks ago" from 2023/02/01 became 2022/12/15 to 12/31).
- Noise floor: re-embedding the same texts in different batches moved the session baseline by
  about 0.004 (fp16). Kept levers with drop-one losses near 0.008 are therefore close to noise.

Final dev metrics come from the end-to-end Modal dry run of the exact job on an empty cache
(`benchmarks/results/longmemeval_dev_final.json`; 84 scored dev questions; official
`eval_utils.py`; recall_all / ndcg_any). The full-S MiniLM numbers quoted earlier (session R@10
0.952, turn R@10 0.785) are not like-for-like with these; the dev baseline rows are:

| Level | Recall@5 | Recall@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| Session, MiniLM hybrid baseline (dev) | 0.893 | 0.952 | 0.899 | 0.911 |
| Session, pipeline | **0.964** | **0.988** | **0.951** | **0.958** |
| Turn, MiniLM hybrid baseline (dev) | 0.607 | 0.786 | 0.649 | 0.687 |
| Turn, pipeline | **0.821** | **0.929** | **0.791** | **0.823** |

R@10 by question type (turn level / session level):

| Type | n | Turn R@10 | Session R@10 |
|---|---|---|---|
| multi-session | 24 | 0.833 | 0.958 |
| temporal-reasoning | 25 | 0.920 | 1.000 |
| knowledge-update | 15 | 1.000 | 1.000 |
| single-session-user | 13 | 1.000 | 1.000 |
| single-session-preference | 6 | 1.000 | 1.000 |

Multi-session is the weakest type at turn level (R@5 0.625). These numbers were tuned on dev, so
they say nothing about held-out performance.

Cost of the dry run:
- In-job estimate: $0.85.
- `modal billing report`: $1.02 for that app (GPU, CPU and image build), read before the billing
  hour closed.
- All Modal use for this work, including development runs and failed attempts: $4.35.

Stage times, from the dry run's `cost.gpu_calls`:

| Stage | Hardware | Work | Time |
|---|---|---|---|
| Fact extraction | L4 | 4,506 sessions | 1,250 s |
| Query LLM | L4 | 100 questions | 59 s |
| Embedding | T4 | 50,585 texts | 458 s |
| Re-ranking | T4 | 3,997 pairs | 273 s |
| HMS scoring | driver CPU | 50 questions | about 15 s |

### The single M run (not executed; needs the maintainer's go-ahead)

```sh
uvx modal run benchmarks/public/longmemeval_modal.py --dataset m --part all --models large \
  --cap 40 --tag m-all-large --out benchmarks/results/longmemeval_m_large.json
```

Large models, all on H100: Qwen3-Embedding-8B @ `1d8ad4ca9b3d`, Qwen3-Reranker-8B @
`77d193c791ed`, and Qwen3-30B-A3B-Instruct-2507 @ `0d7cf23991f4`. Metrics are written for dev,
held-out and all separately.

LongMemEval_M has 237,046 session instances but 48,783 distinct session texts (measured; 1,213,940
user-turn instances, 251,622 distinct). Fact extraction is keyed by content alone, so it makes about
49k LLM calls. With dates in the prompt it would make about 237k.

The estimate below is an inference, not a measurement. Dev stage times were scaled by M's distinct
counts (about 10.8x dev), by parameter count (8B against 0.6B) and by an assumed H100-to-T4/L4
throughput ratio of 4x to 12x:

| Stage | Estimate |
|---|---|
| Fact extraction | about 1 to 1.5 H100-hours |
| Embedding (about 540k keys) | about 1.5 H100-hours |
| Re-ranking (about 20k pairs) | about 0.5 H100-hours |
| HMS scoring | about 30 minutes of driver CPU |

Expected spend is $15 to $25, with 3 to 4 hours of wall time at 4 GPU containers per stage. The
`--cap 40` stop is checked before every wave. The in-job estimate ran about 20% below the bill on
S, which is why the cap leaves twice the margin.

Not yet verified, because the large-model path has never run:
- the H100 functions;
- vLLM loading the 30B MoE model on one H100;
- the M download and sha256 check;
- driver memory for the 2.7 GB JSON (64 GiB allotted);
- HMS ingestion at about 4,000 keys per question;
- the `--part all` metrics path.

## Holographic fact memory prototype (S dev split, numpy)

`benchmarks/public/holographic_memory_proto.py`, results in `benchmarks/results/holographic_dev.json`.
Dev split only; held-out untouched. Representation: binary spatter codes (XOR binding, majority
superposition), D = 16,384; component and atom codes are sign codes of Qwen3-Embedding-0.6B
embeddings under one fixed Gaussian projection, after centering (see capacity). Time is a
month-bucket code with graded overlap between adjacent months; session and turn codes are random.
LLM stages (Qwen3-4B-Instruct-2507 on an L4, cached by content under tag `holo-struct-dev`): one
structuring pass over the cached free-text facts into (entity, attribute, value, when), one query
parse into (entity, attribute, latest/earliest/in-range), and the answer judge. Modal spend for all
of it: $0.76 on the ledger plus one preempted L4 shard (about $0.12) that was lost before its ledger
write. The answer judge is the same model and prompt for every system.

**Capacity (facts per trace before cleanup accuracy drops below 0.95; codebook of 4,000 real
values, attributes distinct within a trace).** Uncentered sign codes fail at 8 facts whatever D:
the raw embeddings share a common direction (mean pairwise cosine 0.43), so their sign codes agree
on 64% of bits and every superposition collapses onto that component. Centered sign codes:

| Facts per trace | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| BSC, random codes, D = 8,192 / 16,384 / 32,768 / 65,536 | 1.0 / 1.0 / 1.0 / 1.0 | 0.98 / 1.0 / 1.0 / 1.0 | 0.81 / 0.99 / 1.0 / 1.0 | 0.37 / 0.77 / 1.0 / 1.0 |
| BSC, centered sign codes, same D | 1.0 / 1.0 / 1.0 / 1.0 | 0.95 / 0.98 / 0.98 / 1.0 | 0.55 / 0.57 / 0.21 / 0.90 | 0.10 / 0.12 / 0.16 / 0.11 |
| HRR (circular convolution), sign codes, D = 8,192 / 16,384 | 1.0 / 1.0 | 1.0 / 1.0 | 0.77 / 0.96 | 0.25 / 0.52 |
| BSC two-level (sessions of 8, then superposed), sign codes, D = 16,384 / 65,536 | 0.95 / 1.0 | 0.19 / 0.68 | 0.0 / 0.05 | 0.0 / 0.0 |

Random codes scale with D as Plate's analysis predicts; semantic codes saturate at about 128 to 256
facts per trace and more D does not buy more, because the crosstalk is structured (near-synonymous
attributes and values share bits by design). A second level of superposition halves the usable
load again. HRR is slightly better than BSC at equal D but needs real-valued traces; BSC was kept
because the sign codes are already binary and XOR is its own inverse. D = 16,384 was chosen as the
smallest D at which the mean per-entity load on dev (156 records for `user`, max 239) is at the
0.95 point for single-level traces.

**EAV fact memory (the §5.7 design): a negative result.** Structuring produced 17,867 records from
13,351 cached facts (83% about `user`); each question's store has 189 records over 48 sessions.
Answer accuracy (judge; 94 non-abstention dev questions):

| Question type (n) | Holographic flat | Hierarchical (user trace, then session traces) | Compositional (top-5 sessions by binding) | Retrieval top-1 turn | Retrieval top-5 turns | Either (flat or top-1) |
|---|---|---|---|---|---|---|
| knowledge-update (15) | 0.267 | 0.267 | 0.333 | 0.733 | 1.0 | 0.733 |
| multi-session (24) | 0.042 | 0.042 | 0.292 | 0.208 | 0.417 | 0.25 |
| single-session-assistant (11) | 0.0 | 0.0 | 0.0 | 0.455 | 0.364 | 0.455 |
| single-session-preference (6) | 0.0 | 0.0 | 0.0 | 0.5 | 0.833 | 0.5 |
| single-session-user (13) | 0.231 | 0.308 | 0.308 | 0.923 | 1.0 | 0.923 |
| temporal-reasoning (25) | 0.12 | 0.2 | 0.16 | 0.36 | 0.76 | 0.4 |
| overall (94) | 0.117 | 0.149 | 0.213 | 0.479 | 0.702 | 0.5 |

Controls over the same fact sentences (dense sign-code index 0.277, sparse 64-of-16,384 index
0.255) show most of the gap is the extraction ceiling, not the hologram: the gold answer string
appears in some structured value for 27 of 81 short-answer questions and in some fact sentence for
33 (none for single-session-assistant, whose answers are in assistant turns the extractor never
sees). Time binding changed no answer (with and without time: 0.117; knowledge-update 0.267 both
ways) because the month code only separates values when two records of the same attribute exist in
different months and the cleanup already preferred one of them. Where it loses: aggregation
questions (26 parsed as needing several records; a single-value lookup cannot answer them, the
compositional list helps multi-session, 0.292 against top-1's 0.208, but stays below top-5's
0.417); extraction and structuring (the breed in "a collar suitable for a Golden Retriever" became
the record `plan = get new collar with name tag`); cleanup crosstalk (correct answers score 0.20 to
0.26 agreement, incorrect ones median 0.23, so no abstention threshold separates them; the six
abstention questions score 0.20 to 0.32). Session recall from the user trace alone: R_any@5 0.548,
R_all@5 0.155 (fact index 0.857 / 0.679; tuned pipeline 0.976 / 0.964). Query cost: flat trace
292 D-bit ops (one cleanup per month bucket), hierarchical 4,380, fact-index scan 138.

Robustness of the EAV variant: with 5/10/20/30/40/50% of trace bits flipped the flat answer stays
at 0.117 / 0.117 / 0.138 / 0.128 / 0.106 / 0.011 (same answer as the clean trace 97 / 94 / 85 / 77
/ 56 / 1%), the sparse-code index drops 0.255 to 0.234 / 0.191 / 0.117 / 0.053 / 0.043 / 0.0 and
the dense-code index holds 0.277 to 0.245 until 40% then 0.011 at 50%. Eight shards of each entity
trace (every record in a random half of them) answer 0.117 after deleting 10 / 25 / 50% of shards;
the fact index with the same fraction of items deleted goes 0.277 to 0.223 / 0.266 / 0.202.

**Turn-atom memory (no extraction), official metrics.** Atoms are the sign codes of raw user turns
and of the cached fact sentences (385 per haystack, 8 per session); each atom is bound with a random
turn code, its session code and the session's month code; session traces, window traces (W
consecutive sessions) and the user trace are majorities. A content query is not a factor of any
trace, so the maintainer's "unbind with the query" step is replaced by its defined form: a trace is
unbound with an atom's key and compared with the query code, one Hamming per atom and no codebook
(the cleanup codebook of all turns is therefore unused). Official `eval_utils.py` (pinned commit),
84 scored dev questions, D = 16,384, same codes for the flat Hamming scan:

| Coarse trace (atoms per trace) | Session R_all@5 | Session nDCG@10 | Turn R_all@5 (from session traces) | Turn nDCG@10 |
|---|---|---|---|---|
| user trace (385) | 0.143 | 0.310 | 0.738 | 0.800 |
| window of 16 sessions (130) | 0.560 | 0.683 | | |
| window of 4 sessions (32) | 0.833 | 0.849 | | |
| one trace per session (8) | 0.940 | 0.924 | | |
| flat Hamming scan over the atoms (no traces) | 0.976 | 0.977 | 0.786 | 0.838 |
| tuned pipeline (dense + BM25 + facts + re-ranker, `longmemeval_dev_final.json`) | 0.964 | 0.958 | 0.821 | 0.823 |

The session traces hold their 8 atoms almost losslessly: turns resolved from them reach R_all@5
0.738 / nDCG@10 0.800 against 0.786 / 0.838 for the flat scan over the same codes, with the store
reduced from 385 atom codes to 48 session codes, and the LLM rewrite as the query did not help
(0.155 session R_all@5 from the user trace). The user trace is over capacity, in line with the
curve above: session recall from one trace per haystack is 0.143, and recovers only when a trace
holds at most about 32 atoms (window of 4 sessions, 0.833). Per type, turns from session traces
match the flat scan on single-session-user (1.0), single-session-assistant (1.0) and
knowledge-update (0.80), beat it on multi-session (0.67 against 0.62) and preference (0.67 against 0.50), and lose on temporal-reasoning (0.64 against 0.88). Query cost: 864 D-bit
ops hierarchical (coarse over the user trace plus fine over every session trace) against 385 for
the flat scan; with window traces the coarse stage is n_sessions times the window's atoms.

Robustness of the turn-atom variant (window-4 coarse, session-trace fine; turns ordered globally by
score): flipping 5 / 10 / 20 / 30 / 40 / 50% of every trace's bits gives session R_all@5 0.845 /
0.714 / 0.607 / 0.440 / 0.143 / 0.036 and turn R_all@5 0.726 / 0.762 / 0.714 / 0.679 / 0.440 /
0.012 (clean: 0.833 / 0.738). Storing each session trace as 8 shards (every atom in a random half)
and deleting 10 / 25 / 50% of the shards: session R_all@5 0.905 / 0.905 / 0.929, turn 0.583 /
0.571 / 0.631; the flat index with the same fraction of atoms deleted outright: session 0.964 /
0.952 / 0.917, turn 0.750 / 0.690 / 0.488. At 50% loss the sharded traces keep turn recall 0.631
against the index's 0.488, but the shards cost 0.74 of the fine-stage recall at zero loss (0.583
against 0.738, averaging agreements over half-full shards adds noise), so the "cut the hologram in
half" property is real and paid for up front; a replicated index of the same byte budget was not
measured.

**Plain statement.** On dev, no holographic variant answers questions better than the tuned
chunk-retrieval pipeline: the EAV design loses on every question type (0.117 against 0.479 top-1)
and its ceiling is extraction, not the algebra; the extraction-free turn-atom design gets within
0.05 of a flat scan over the same codes at the turn level while storing one vector per session, and
loses at the session level unless traces are kept at about 32 atoms. The properties the name
implies hold in the measured form: answers and rankings degrade gradually up to 30 to 40% bit
flips where the sparse index fails by 20%, and shard loss costs little; the cost is capacity, which
semantic codes cap at a few hundred items per trace regardless of D. Not verified: a replicated
index at equal bytes as the shard control, time codes bound into the query, re-ranking on top of
the holographic candidates, and anything on the held-out split or on M. No Rust was written.

## What this means

- For semantic search over text, use the document API: exact cosine reproduces the embedding
  model's quality, and hybrid search with the built-in BM25 improves on it.
- Do not use the sparse vector path (`memorize_vector` / `query_vector`) where nearest-neighbour
  recall matters. Improving it needs a different encoder; making the current one faster does not.
- HMS does not compete with HNSW libraries as a vector index. Its distinctive parts are the
  compositional VSA algebra, structured memory and provenance features, which these benchmarks do
  not measure.

## Reproduce

```sh
uv run --script benchmarks/public/prepare.py beir scifact
uv run --script benchmarks/public/prepare.py ann glove-100-angular
cargo build --release --bin public-bench
./target/release/public-bench beir --data ~/.cache/hms-bench/scifact --dim 16384 --out scifact.json
uv run --script benchmarks/public/evaluate.py beir scifact scifact.json
./target/release/public-bench ann --data ~/.cache/hms-bench/glove-100-angular --dim 16384 --out glove.json
uv run --script benchmarks/public/evaluate.py ann glove-100-angular glove.json
uv run --script benchmarks/public/prepare.py longmemeval s
./target/release/public-bench longmemeval --data ~/.cache/hms-bench/longmemeval_s --out lme.json
uv run --script benchmarks/public/evaluate.py longmemeval s lme.json
```
