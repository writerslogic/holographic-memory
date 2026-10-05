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
| Reference: BM25 as published in the BEIR paper (Thakur et al. 2021, Table 2) | 0.665 | | 0.325 | |

- HMS's dense search matches the independent NumPy reference. This is a correctness check of the
  document API; it says nothing new about the embedding model.
- HMS's built-in BM25 is within 0.003 of the published BM25 on SciFact and 0.018 below it on
  NFCorpus (different tokenization; no stemming).
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
```
