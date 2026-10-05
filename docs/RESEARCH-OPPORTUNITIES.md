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

## 4. Ranked options

1. Fix result ordering (2.1). Correctness, small, testable.
2. README capacity column (2.2): relabelled in 0.6.1 from the committed sweep. Still open: regenerate
   with at least 1,000 non-member probes so FPR ≤ 1% and ≤ 0.1% limits can be reported.
3. Bundle-capacity sweep over subset size (2.3). Largest expected gain; theory and partial data agree.
4. Routing sweep, then planner thresholds from the measured crossover (2.4).
5. Intersection kernel microbench and dispatch fix (2.5).
6. Encoder evaluation on a real corpus (2.6).
7. Private search: decide the deployment model first. If hosted personal memory, scope TEE
   attestation verification; if shared corpus, prototype homomorphic scoring as an experiment.

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
