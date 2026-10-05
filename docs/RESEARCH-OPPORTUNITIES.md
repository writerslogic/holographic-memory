# Algorithm opportunities and private-search scoping (2026-10-04)

Status: findings and plans only. Nothing here is implemented. Each item is labelled
**measured** (reproduced or read from a committed results file), **inferred** (follows from the
code or from theory, not yet timed), or **conjecture**.

## 1. README performance claims and their sources

All result files now live in `benchmarks/results/`.

| README claim | Source | Note |
|---|---|---|
| Capacity table: 2,478 / 9,800 / 58,432 items, encode ops/s | `benchmark_scaling_results.json` (Modal run, 2026-06-19), rows with 64 active indices | Exact match. See 2.2: the column measures saturation, not usable capacity. |
| KG 100%, analogy 100%, multi-hop 100%, d' 353.7 | `research_bench_16384_256.json` | 2,120 facts, 350 analogies, 20 chains. Single dimension, single density. |
| Noise tolerance 100% at 30/50/70% corruption | `benchmark_scaling_results.json` `.noise_tolerance` | 100 stored items, 30 probes. One N; says nothing about larger stores. |
| NSG recall and latency in `docs/evaluation.md` | `benchmarks/review-2026-09-14/*.json` | 1,200 and 10,000 vectors, D=4096, Apple M4. |

`benchmark_scaling_results_coarse.json` and `bench_local_*.json` are earlier or local runs of
the same sweep and are not cited by the README. `benchmark-results*.json`,
`benchmark_results_full.json` and `bench_suite_full_16384.json` are suite outputs with no
README claim attached.

## 2. Findings

### 2.1 Exact-scan results come back worst-first (measured, defect)

`Shard::brute_force_scan` and `ShardManager::query` call `into_sorted_vec()` and then
`reverse()`. `RetrievalResult`'s `Ord` is already inverted, so `into_sorted_vec()` is
descending by similarity and the `reverse()` makes it ascending. The inverted-index route
does not reverse and is correct. A probe (40 `from_scalar` vectors at D=16384, query equal to
item 20, k=5) returned `s-18=0.753 s-22=0.778 s-19=0.882 s-21=0.882 s-20=1.000`. Every store
under 1,000 vectors takes this route, as does every multi-shard merge. No existing test
asserts result order. Present before the agy commits.

Fix: drop the two `reverse()` calls; add one test asserting descending order on each route
(brute force, inverted, NSG, IVF, multi-shard).

### 2.2 The "hard wall" column is the bundle saturation point (measured)

Sweep for D=16384, 64 active indices, 50 probes per point:

| N | bundle density | recall | FPR | d' |
|---|---|---|---|---|
| 300 | 0.695 | 1.00 | 0.00 | 14.96 |
| 500 | 0.865 | 1.00 | 0.04 | 6.88 |
| 1,000 | 0.981 | 1.00 | 0.32 | 2.02 |
| 2,000 | 0.9998 | 1.00 | 1.00 | 0.00 |
| 2,478 | 0.9999 | 1.00 | 1.00 | 0.00 |
| 2,479 | 1.000 | 0.00 | 0.00 | 0.00 |

Recall stays at 1.0 because a union bundle always contains its members; it drops to 0 only
when every bit is set and `corrected_containment` returns 0. 2,478 is within 0.3% of the
coupon-collector estimate D·ln(D)/k = 2,484. At that point the false-positive rate is 100%.
The same file already records the meaningful limits: positive member/non-member gap up to
N=500, d' ≥ 2 up to N=1,000. The README column should be one of those, with the FPR stated.

### 2.3 Bundle capacity is limited by using all 64 indices per item (inferred, partly measured)

A union bundle is a Bloom filter with k = active indices. FPR for full containment is
density^k; at N=1,000 that predicts 0.9814^64 = 0.30 against a measured 0.32. The optimum is
k* = (D/N)·ln 2, about 11 at N=1,000, for a predicted FPR near 4e-4. Committed local runs
agree in direction: at D=16384 with 16 active indices the gap wall is 1,200 and the d' wall
2,000, versus 500 and 1,000 with 64.

Hypothesis: inserting a deterministic k*-subset of each item's indices into Bloom bundles
raises FPR-bounded capacity several-fold at fixed D without changing the item vectors.
Measurement: sweep subset size {4, 8, 11, 16, 32, 64} × N from 100 until FPR > 1%, at
D ∈ {16384, 65536}, at least 1,000 non-member probes so that 0.1% is resolvable, 5 seeds;
report the largest N with FPR ≤ 1% and ≤ 0.1%.

### 2.4 Exact inverted-index scoring is probably better than NSG well past 1,000 vectors (inferred)

Scoring through the inverted index is exact and costs about N·k²/D counter increments per
query for random codes: roughly 25,000 at N=100,000, k=64, D=16384. The committed NSG report
is recall@1 0.94 at 1,200 vectors with a 116 µs mean, where an exact answer is available. The
planner sends a query to the inverted index only when it has at most 4 active indices or no
trained index exists.

Measurement: N ∈ {1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6}, D ∈ {4096, 16384}, random codes and
`from_dense` codes from real embeddings (posting-list skew matters), recall@10 against brute
force plus p50/p95 for brute force, inverted, NSG and IVF. The deliverable is the crossover N
per route, or a statement that none was reached in the tested range.

Related code observations (read from source; cost not yet timed): NSG search allocates
`vec![false; n]` per query; the Hopfield route clones every stored vector per query; the
inverted route sorts query dimensions by document frequency but never prunes.

### 2.5 Intersection kernel dispatch (inferred)

- On x86-64 the AVX2 block kernel is tried before the skew check, so a 64-element query
  against a large union bundle walks the large side linearly instead of galloping.
- On aarch64 (the machine all September reports came from) there is no SIMD path; the
  fallback merge is branchy.
- Fixed-size vectors (64 indices) could use a branchless merge or a block layout.

Measurement: criterion microbench over |a| = 64 and |b| ∈ {64, 256, 1k, 4k, 16k}, overlap
∈ {0, 10, 50, 100}%, on both architectures; then the end-to-end exact scan at N=1,000.

### 2.6 Dense-to-sparse encoder (weak measurement)

`from_dense` evaluates D/2 hashed projections of the full embedding per call (about 3.1M hash
evaluations for a 384-d embedding at D=16384). The only fidelity figure is dense-neighbour
recall@5 = 0.60 on a 10-document fixture, which supports no conclusion. Measurement: recall@10
of sparse top-k against dense cosine top-k on a public retrieval set of at least 10k
documents, sweeping D and active count, plus encode latency.

## 3. Private similarity search: threat-model comparison

HMS today runs locally, so there is no remote party to hide a query from. The table matters
only if a hosted or shared deployment is planned. "Server" means whoever operates storage and
compute.

| Approach | Hides from server | Still leaks / trust required | Cost reported in the primary source |
|---|---|---|---|
| Local only (current) | Everything; no server | Device compromise | None |
| Keyed permutation (`src/core/mask.rs`) | Index labels | All pairwise overlaps, equality, frequencies, access patterns; known pairs reveal the mapping. Obfuscation, not encryption | Negligible |
| TEE with remote attestation (RFC 9334) | Data and queries from the operator | Hardware vendor, enclave code, side channels; storage access patterns unless combined with ORAM (Opal) | Near native; Opal reports 29× the throughput of its secure baseline |
| Single-server linearly homomorphic scoring (Tiptoe) | Query and which results were fetched | Corpus is readable by the server; server work is linear in corpus size | 360M documents: 145 core-s, 56.9 MiB, 2.7 s per query |
| Two non-colluding servers, LSH + DPF (Servan-Schreiber et al.) | Query | Both operators must not collude; corpus readable by servers | Sublinear communication |
| Client-driven encrypted index over ORAM (Compass) | Data, queries, results, access patterns | Client state and several round trips | About 1 s user-perceived latency |
| Client graph search over PIR (Pacmann) | Query | Corpus readable by server; client storage | 100M vectors, 90% of non-private ANN quality |
| Single-server HE + MPC (Panther) | Query | Corpus readable by server | 10M points: 18 s, 284 MB per query |
| Differential privacy + anonymity + SHE (Wally) | Query, up to (ε=0.1, δ=2⁻²⁶) | Needs many concurrent clients and an anonymous network | 7–29× Tiptoe throughput at 3.2M entries |

Fit to HMS (conjecture until prototyped): HMS scores are integer intersection counts bounded
by the active count, so a Tiptoe-style scheme needs a plaintext modulus only slightly above
64 and the server-side work is a sparse matrix times a ciphertext vector. That suits a shared
or public corpus with private queries. It does not protect a user's own stored memories from
the host; for that the realistic choices are a TEE with verified attestation, or an
ORAM-backed encrypted index with a seconds-scale latency budget.

Embeddings are not a privacy boundary by themselves: text can be reconstructed from them
(Morris et al. 2023).

## 4. Outcomes (2026-10-05)

All runs on an Apple M4. Several ran while other builds were compiling, so latencies are
indicative only; recall, capacity and correctness figures are unaffected by load.

1. **Result ordering (2.1), fixed.** Exact-scan, multi-shard and federated queries returned
   top-k worst-first. Regression test `every_query_route_returns_best_match_first`.
2. **Capacity column (2.2), relabelled** in 0.6.1 from the committed sweep.
3. **Subset bundles (2.3), measured, partly confirmed.** `benchmarks/results/bundle_subset_sweep.json`
   (64 active indices, 5 seeds x 4,000 probes). Inserting an 8-index subset per item stores about
   2.7x more items at FPR <= 1% (D=16384: 606 -> 1,611) and about 2x at FPR <= 0.1%. Less than
   the unconstrained Bloom optimum suggests; the grid stops at the first point above 1%.
   Experimental module `bundle_subset`; not wired into the engine.
4. **Routing (2.4), measured and applied.** `benchmarks/results/route_sweep.json` (N = 1e3..1e6,
   D = 4096 and 16384, random and clustered codes). The exact inverted index had recall@10 = 1.0
   and the lowest latency at every point; NSG recall fell to about 0 on clustered codes at
   N >= 1e5. The planner now sends every sparse query to the inverted index.
5. **Intersection kernels (2.5), negative result.** A branchless merge was 2-10x slower and a NEON
   block kernel won only on identical inputs, so the aarch64 kernels are unchanged. On x86 the
   skew check now runs before AVX2 (unmeasured: no x86 machine).
6. **Encoder (2.6), measured on synthetic data.** `from_dense` is 2-3.5x faster with bit-identical
   output. Recall@10 of sparse top-10 against dense cosine top-10 is 0.13 / 0.23 / 0.36 at
   D = 4096 / 16384 / 65536, but the true top-10 is inside the sparse top-100 at 0.76 / 0.96 /
   1.00 (clusters of 100). That did not hold on real data: on the public benchmarks
   (`docs/PUBLIC-BENCHMARKS.md`) the true top-10 is inside the sparse top-100 for only 54% of
   results on nytimes and glove, and the sparse path loses 27-34% of nDCG@10 on BEIR. The
   encoder, not the index, limits the sparse path; a new versioned encoder is the open problem.
7. **Private search (3), prototype.** `private-search` feature and `docs/PRIVATE-SEARCH.md`:
   SimplePIR-style scoring that hides the query from an honest-but-curious server. 0.88 ms server
   time per query at N = 1e5, with a 410 MB one-time client hint. The database stays visible to
   the server.
8. **SDK service.** `hms-server` (feature `server`) serves the Python SDK with exact cosine through
   the document API; the SDK's live tests run against it.

## Sources

- Henzinger, Dauterman, Corrigan-Gibbs, Zeldovich. Private Web Search with Tiptoe. SOSP 2023. https://dl.acm.org/doi/10.1145/3600006.3613134
- Zhu, Patel, Zaharia, Popa. Compass: Encrypted Semantic Search with High Accuracy. OSDI 2025. https://eprint.iacr.org/2024/1255
- Zhou, Shi, Fanti. Pacmann: Efficient Private Approximate Nearest Neighbor Search. ICLR 2025. https://eprint.iacr.org/2024/1600
- Li et al. Panther: Private Approximate Nearest Neighbor Search in the Single Server Setting. CCS 2025. https://eprint.iacr.org/2024/1774
- Asi et al. Scalable Private Search with Wally. https://arxiv.org/abs/2406.06761
- Servan-Schreiber, Langowski, Devadas. Private Approximate Nearest Neighbor Search with Sublinear Communication. IEEE S&P 2022. https://eprint.iacr.org/2021/1157
- Kaviani, Ozdarendeli, Zhu, Ding, Popa. Opal: Private Memory for Personal AI. https://arxiv.org/abs/2604.02522
- Birkholz et al. Remote ATtestation procedureS (RATS) Architecture. RFC 9334. https://www.rfc-editor.org/rfc/rfc9334
- Morris, Kuleshov, Shmatikov, Rush. Text Embeddings Reveal (Almost) As Much As Text. EMNLP 2023. https://arxiv.org/abs/2310.06816

## 5. Strategy after the public benchmarks (2026-10-05)

**Measured strengths.** Local hybrid retrieval (BM25 + exact cosine, rank fusion) beats either
component and matches published baselines on BEIR; a transactional, crash-safe, optionally signed
store in one embedded engine. **Unmeasured:** VSA reasoning (binding, analogy, multi-hop) on any
real benchmark. **Measured weakness:** the dense-to-sparse encoder (section 4, item 6).

### 5.1 Encoder: dense binary sign codes instead of sparse top-k codes (measured, numpy)

Random-rotation sign codes (SimHash; dense binary hypervectors, i.e. the BSC representation that
supports XOR binding and majority bundling) were tested against the exact top-10 on the
public data (`all-MiniLM-L6-v2` for BEIR; 1,000 nytimes queries):

| Code | SciFact top-100 + re-rank | NFCorpus | nytimes | Hamming top-10 alone (SciFact / nytimes) |
|---|---|---|---|---|
| current sparse (64 of 16,384) | | | 0.54 | |
| sign codes, 384 bits (48 B) | 0.897 | 0.852 | 0.714 | 0.503 / 0.449 |
| sign codes, 1,024 bits (128 B) | 0.987 | 0.955 | 0.878 | 0.654 / 0.606 |
| sign codes, 4,096 bits (512 B) | 0.999 | 0.997 | 0.972 | 0.812 / 0.772 |

Sign codes are standard binary quantization, not a novel technique; the point is that they are
also valid VSA hypervectors, so one representation can serve retrieval and the algebra. A linear
Hamming scan will not match HNSW throughput at 1M vectors; the claim to pursue is recall and
memory, not speed. Next: a versioned `sign-projection-v1` encoder in Rust, measured with
`public-bench` on all four sets.

### 5.2 Where "superior" is testable: agent long-term memory (LongMemEval)

LongMemEval (Wu et al., ICLR 2025, arXiv 2410.10813, MIT) scores retrieval of the evidence for
500 questions over long chat histories, including temporal reasoning, knowledge updates and
abstention. Published retrieval on LongMemEval_M with Stella V5 1.5B (Table 3, read from the PDF):

| Value / key | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| Round, K = V | 0.582 | 0.481 | 0.692 | 0.512 |
| Round, K = V + LLM-extracted facts | 0.644 | 0.498 | 0.784 | 0.536 |
| Session, K = V | 0.706 | 0.617 | 0.783 | 0.638 |
| Session, K = V + LLM-extracted facts | 0.732 | 0.620 | 0.862 | 0.652 |

Falsifiable plan: (1) with one embedding model held fixed, compare exact-cosine dense retrieval
against HMS (hybrid, round/session keys, time-aware filtering) using the repository's official
retrieval evaluation, one store per question; (2) report the paper's numbers separately with the
model-size difference stated; (3) add engine features (date-range filters, update handling) only
where the baseline shows failures. LLM fact extraction needs an API and is a separate decision.

### 5.3 LongMemEval_M with larger models (plan, 2026-10-05)

Target: beat the paper on LongMemEval_M retrieval with open-weights models run locally.

1. **Reproduce the paper's baseline first** on the cleaned M release: Stella V5 1.5B
   (`NovaSearch/stella_en_1.5B_v5` @ `7817065`, MIT), encoded exactly as the official
   `src/retrieval/run_retrieval.py` does (no query instruction, mean pooling, 1024-d `2_Dense_1024`
   head, max 512 tokens), K = V at round and session level, official `eval_utils.py`. Every later
   number is compared with this reproduction, not only with the PDF, so a cleaned-data effect is
   not mistaken for a model gain.
2. **Larger embedder:** `Qwen/Qwen3-Embedding-8B` @ `1d8ad4ca9b3d` (Apache-2.0, 7.6B parameters,
   about 5x Stella), with the model card's query instruction and last-token pooling.
3. Then, one lever at a time on the same harness: hybrid (BM25 + dense), session aggregation of
   round scores, time-aware filtering from `question_date` (needs a date-range filter in the
   document API), cross-encoder rerank with `Qwen/Qwen3-Reranker-4B` (Apache-2.0; cost measured on
   a sample first).
4. Operational: data and model caches on /Volumes/A, embeddings in resumable shards keyed by
   content hash, load average recorded, nothing else timed during runs.

Open decision: an LLM for fact-key extraction (the paper's largest gain) and for end-to-end QA,
local (Qwen3 via MLX) or API.

### 5.4 Targets after review (2026-10-05)

**Speed track (beat HNSW).** Target: higher single-thread QPS than FAISS HNSW and hnswlib at
recall@10 = 0.90 and 0.95 on glove-100-angular and nytimes-256-angular, all systems timed in the
same session on an idle machine with repeated runs. Approach: rotated 1-bit codes (RaBitQ; also
HMS's binary hypervectors) searched inside a graph built on exact distances, codes of each
vertex's neighbours stored contiguously, popcount/estimator SIMD kernels, exact re-rank of a small
candidate set (the quantized-graph design of SymphonyQG, SIGMOD 2025). `src/core/nsg` is not
reused: it runs on the sparse codes and its recall collapsed on clustered data.

Estimator check (numpy, 1,000 queries, exact top-10 recall):

| Set | Bits | RaBitQ est. top-10 | Hamming top-10 | RaBitQ top-100 + re-rank | Hamming top-100 + re-rank |
|---|---|---|---|---|---|
| SciFact | 384 | 0.625 | 0.582 | 0.984 | 0.949 |
| nytimes | 256 | 0.580 | 0.449 | 0.881 | 0.731 |
| nytimes | 1,024 (padded) | 0.761 | 0.661 | 0.973 | 0.930 |
| glove (300k subset) | 128 | 0.243 | 0.192 | 0.628 | 0.476 |

The estimator beats plain Hamming at native length; zero-padding does not add information
(multi-bit extended codes are the route to more precision). glove needs larger candidate sets,
which the graph search provides.

**Accuracy track (LongMemEval).** Goal: as close to 1.0 as achievable, reported per question type.
Rules: a stratified split frozen before any tuning (100 dev / 400 held-out questions, by type);
tune only on dev, report held-out, and label full-set numbers as tuned. Iterate on S; one final
run on M.

**Compute.** Qwen3-Embedding-8B in fp16 on this M4 via PyTorch MPS measured 72 tokens/s on user
turns and 123 tokens/s on sessions. LongMemEval_M deduplicates to 49,018 sessions (about 120M
tokens) and 254,089 user turns (about 15M tokens), so a full M pass with the 8B model is about
1.5 days (turns) to 2 weeks (sessions) locally. Local LLM fact extraction over M is slower still.
M-scale runs with 8B-class models need a rented GPU or a smaller model.

### 5.5 LongMemEval single-run protocol (2026-10-05)

The large-model run on LongMemEval_M happens once, on Modal, after everything else is fixed:

1. **Frozen split.** 100 dev / 400 held-out questions, stratified by question type, chosen with a
   fixed seed and committed before any tuning. All tuning on dev only.
2. **Pipeline built and ablated on S dev** with small models locally: multi-granularity indexing
   (user turns, rounds, sessions, extracted facts), hybrid BM25 + dense, session aggregation of
   turn scores, time-aware handling from `question_date` and session dates, query rewriting /
   decomposition for multi-session questions, cross-encoder re-ranking. Each lever is kept only if
   it improves dev, and its gain is recorded.
3. **Dry run** of the exact Modal job on S end to end (cheap) so the M run cannot fail midway;
   resumable shards, a hard spending cap, cost reported.
4. **Single M run** with the strongest open models (Qwen3-Embedding-8B, Qwen3-Reranker-8B, a Qwen3
   LLM for fact extraction and query rewriting), official `eval_utils.py`, every retrieval metric
   at both granularities, per question type, held-out and full set reported separately.
5. **Reporting.** Every number as measured, beside the paper's published baselines and the local
   Stella reproduction. A metric where HMS does not lead is reported as such.

### 5.6 Local model stages inside HMS (2026-10-05)

Decision: fact extraction, query rewriting, embedding and re-ranking move from the benchmark
scripts into HMS, so the LongMemEval headline exercises HMS's own stages and runs on-device.

Order: (1) build the stages in HMS; (2) re-run S dev through them on Modal and compare with the
tuned dev numbers (embedder and re-ranker within the ~0.004 noise floor; the generative LLM stages
re-measured, not assumed equal); (3) only then the single M run, with a cost estimate from the
HMS-stage timings, after maintainer approval.

Constraints: candle with the exact tuned safetensors revisions (Qwen3-Embedding-0.6B/8B,
Qwen3-Reranker-0.6B/8B, Qwen3-4B-Instruct-2507 / Qwen3-30B-A3B-Instruct-2507); a `local-models`
feature that builds CPU-only on every OS; Metal only via a macOS target dependency; CUDA only in
the Modal image, never under `--all-features`; MSRV impact checked; `cargo deny` on the full tree;
models load from local directories with pinned revisions and are never downloaded by the library.
Report on-device ingest time per session and query latency (query rewriting adds an LLM call per
query). llama.cpp/GGUF is a later, separately measured speed path.
