# Embedding model selection

Source of every number: `benchmarks/results/embedding_selection_v1.json` (scripts in `benchmarks/embedding/`).
Splits: SciFact **train** queries (809), NFCorpus **dev** queries (324), LongMemEval_S frozen **dev** ids
(94 non-abstention questions, turn-level, session score = best turn). No test split and none of the 400
held-out ids was read. Licence rule: MIT or Apache-2.0 from the model card.

## Stage 1: fp32 PyTorch reference (nDCG@10; LongMemEval: nDCG@10 / recall_any@5)

| Model | Licence | Params | SciFact-train | NFCorpus-dev | LME-dev | Engine supports |
|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | apache-2.0 | 23M | 0.6602 | 0.3024 | 0.8942 / 0.968 | yes |
| sentence-transformers/all-MiniLM-L12-v2 | apache-2.0 | 33M | 0.6231 | 0.3097 | 0.8881 / 0.947 | yes |
| BAAI/bge-small-en-v1.5 | mit | 33M | 0.7469 | 0.3276 | 0.9000 / 0.968 | yes |
| thenlper/gte-small | mit | 33M | 0.7377 | 0.3398 | 0.9082 / 0.957 | yes |
| intfloat/e5-small-v2 | mit | 33M | 0.7078 | 0.3222 | 0.9170 / 0.957 | yes |
| intfloat/multilingual-e5-small | mit | 118M | 0.6766 | 0.2974 | 0.9080 / 0.957 | yes |
| Snowflake/snowflake-arctic-embed-s | apache-2.0 | 33M | 0.7094 | 0.3101 | 0.8991 / 0.957 | yes |
| Snowflake/snowflake-arctic-embed-xs | None | 23M | not evaluated: licence None is not MIT/Apache-2.0 | | | |
| mixedbread-ai/mxbai-embed-xsmall-v1 | apache-2.0 | 24M | 0.6284 | 0.2639 | 0.8396 / 0.947 | yes |
| avsolatorio/GIST-small-Embedding-v0 | mit | 33M | 0.7301 | 0.3332 | 0.9010 / 0.957 | yes |
| ibm-granite/granite-embedding-30m-english | apache-2.0 | 30M | 0.7486 | 0.3149 | 0.9089 / 0.968 | yes |
| ibm-granite/granite-embedding-small-english-r2 | apache-2.0 | 48M | 0.7623 | 0.3338 | 0.8950 / 0.957 | no (ModernBERT) |

Differences below about 0.01 are inside the noise of these query counts. gte-small, bge-small and
granite-30m are statistically close overall; the default is **ibm-granite/granite-embedding-30m-english**: best SciFact among
engine-supported models, equal best LongMemEval recall@5, and 6 layers against 12 for gte/bge/e5, so
about half the compute and a 34 MB int8 pack. gte-small is the better choice on NFCorpus-dev
(0.3398 against 0.3149) and is in the registry. `granite-embedding-small-english-r2` (ModernBERT) scores
higher on SciFact but needs an architecture the engine does not implement. Engine-side runs of gte-small
and bge-small were not completed, so the choice rests on
stage-1 quality plus the layer count, not on a measured engine latency comparison.

## Rust engine, int8 (granite-embedding-30m-english)

* nDCG@10 through the engine: SciFact-train 0.7491, NFCorpus-dev 0.3152, LME-dev 0.9092 (recall_any@5 0.968); the fp32 reference is in the table above, so int8 costs no measurable quality.
* Agreement with the fp32 reference on the fixed 71-text sample: min cosine 0.99963 (passages), 0.99975 (queries); bar 0.999. int8 uses one scale per 64 weights: on all-MiniLM-L6-v2 per-row scales gave min cosine 0.9973, groups of 128/64/32/16 gave 0.9988/0.9994/0.9994/0.9995.
* Pack: 34.1 MB. Peak RSS embedding 2000 passages: 182 MB. Cold start (process to first embedding): 396 ms wall, 280 ms of it loading and hashing the pack.
* Latency per passage (SciFact docs, about 330 tokens, best of 2): 279 ms at 1 thread, 77 ms at 8; per query 13.8 / 3.8 ms. Measured on a shared machine at load average 20; indicative only. The engine does not use Apple's AMX, so PyTorch/Accelerate is faster per core.

## Retrieval gain and cost through the document store (`embedding_store_v1.json`)

`TextEncoder::LocalEmbedV1` (granite-embedding-30m-english int8, 8 threads) against the default lexical
store, same documents and queries, nDCG@10 (SciFact-train / NFCorpus-dev; LongMemEval was not run here):

| Variant | SciFact-train | NFCorpus-dev |
|---|---|---|
| lexical default | 0.7027 | 0.2916 |
| semantic only | 0.7491 | 0.3151 |
| hybrid (default blend) | **0.7681** | **0.3435** |

Hybrid gains +0.065 and +0.052 nDCG@10 over the lexical default. Cost (shared machine, indicative):
ingest 9.6 docs/s against 735 docs/s lexical on SciFact (7.1 against 282 on NFCorpus), peak process RSS
368 MB / 345 MB against 97 MB / 80 MB, query latency about 12 ms / 8 ms (query embedding included)
against 0.1 ms, process start to a working one-document store 0.31 s with the embedder (the
default-encoder figure in the file, 0.75 s, is dominated by machine load). The embedder is opt-in; the
default build and its memory are unaffected.
