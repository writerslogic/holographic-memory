# Practical VSA admission experiment — 30 October 2026

**Decision: NO_GO.** No VSA arm meets the frozen promotion gates. Keep this design in research; do not promote it into the production retrieval path.

This answers a bounded question: can role-bound episode signatures beat an exact structural index when both receive the same canonical facts? It does **not** establish that every VSA architecture or application is unhelpful. It also does not measure natural-language understanding, learned extraction, dense search, hardware acceleration, or production HMS APIs.

## Evidence and decision

Eighteen held-out cases: three seeds × 256/1,024 documents × 2/8/32 facts per document. Each case has 96 queries in six classes, measured before and after 16 replacements and 16 deletions. All indexes are reconstructed from a canonical snapshot and checked for result parity. There were 17,280 diagnostic method/query executions across initial/mutated states, in addition to timed sweeps and reopen comparisons. These are repeated synthetic queries, not independent real-world examples.

| Method | Worst case recall | Median p50 (µs) | Median latency / exact | Median total heap / exact | Promote |
|---|---:|---:|---:|---:|---|
| exact_role_postings | 100.0% | 0.42 | 1.00× | 1.00× | Reference |
| bm25_verified | 81.0% | 4.25 | 9.54× | 1.70× | Control |
| vsa_verified_32 | 100.0% | 10.40 | 26.83× | 6.34× | No |
| vsa_verified_128 | 100.0% | 11.40 | 28.47× | 6.34× | No |
| vsa_verified_32_fallback | 100.0% | 11.04 | 26.89× | 6.68× | No |

“Worst case” is the lowest phase-level average over the 36 initial/mutated case-phases, not minimum individual-query recall. Other table entries are medians across those case-phases. Heap includes canonical source records, signatures/postings, codebook, and both indexes for fallback. A ratio above 1 favors exact indexing.

All five methods produced **zero observed unsupported, wrong-version, wrong-namespace, invalid-time, deleted, or incorrectly quoted outputs**. Exact retrieval achieved 100% recall in every case; all reopen comparisons passed. Safety comes from exact canonical verification shared by every arm, not from hypervector similarity. The same verifier rejects role confusion and cross-fact joins.

The frozen gate requires recall ≥99% and within one percentage point of exact in every case; a ≥20% latency or retained-memory advantage at matched quality; and no >10% query/ingest/update/reopen regression anywhere. [Machine-readable decisions](decision.json) show each gate separately. No evaluation parameter tuning was performed.

## Interference and lifecycle

| Facts / episode | Exact recall | BM25 recall | VSA 32 recall | VSA 128 recall | VSA fallback recall |
|---|---:|---:|---:|---:|---:|
| 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 32 | 100.0% | 89.8% | 100.0% | 100.0% | 100.0% |

Recall here averages the nonempty-query phase averages at each interference level. Empty queries are excluded from recall and checked for false output. Raw per-class results remain in [results.json](evaluation/results.json).

| Method | Ingest ratio | Update ratio | Reopen ratio |
|---|---:|---:|---:|
| exact_role_postings | 1.00× | 1.00× | 1.00× |
| bm25_verified | 0.69× | 2.90× | 0.86× |
| vsa_verified_32 | 47.33× | 14.69× | 30.05× |
| vsa_verified_128 | 47.33× | 14.69× | 30.05× |
| vsa_verified_32_fallback | 47.93× | 15.54× | 30.35× |

Lifecycle entries are median ratios to exact, including shared canonical preparation or snapshot reading. These are single measurements per case; small differences would require controlled replication. BM25 rebuilds on mutation because the reused reference implementation exposes no incremental update API. This limitation does not affect the decisive comparison with the incrementally maintained exact index.

The fallback invoked exact retrieval for a median 99.0% of queries. It retains and maintains both indexes. It therefore tests whether approximate candidates add value to the exact path rather than assuming that a fallback is free.

## Why the design fails to earn promotion

1. **Canonical structure already solves the target ambiguity.** Role/filler postings narrow candidates directly, and fact-level verification preserves subject/predicate/object association. Role-sensitive retrieval is useful; that alone does not establish a need for VSA.
2. **Partial matching has a material representation cost.** Seven projections cost 1,792 signature bytes per episode before the codebook or canonical sources. Even an unrealistically free codebook leaves an initial total-heap ratio of 1.40–6.79× exact for this layout. This is an arithmetic lower bound, not a benchmark of an optimized implementation.
3. **There is no quality dividend over exact retrieval here.** VSA matched exact recall and outperformed the roleless BM25 control at 32 facts per episode: the representation worked. It nevertheless added cost without improving the decisive baseline. The fallback triggered for almost every query because most had fewer than five valid answers; it provided no observed recall gain in this workload.
4. **The tested query path scans eligible signatures.** It also creates native phase vectors for query binding. Packed codebooks, direct bitwise binding, or another candidate index could reduce overhead, but their gains were not measured. They would be a new hypothesis, not a reason to reinterpret this failure as success.

The experiment uses the actual production phase primitives and existing research popcount/BM25 implementations. It does not pretend that this small harness is a fully optimized VSA system or that 1,024 documents establish scaling behavior.

## Practical engineering decision

* **Proceed with exact, evidence-backed structural retrieval as the reference design.** Before product work, validate the actual application's relation/query contract, annotation provenance, and end-to-end source fidelity. This experiment supports that baseline, not an unqualified production-readiness claim.
* **Keep VSA opt-in and outside authoritative storage.** There is no measured justification here for making it a default retrieval dependency or replacing canonical facts.
* **Do not spend paid model/API credits or broaden the benchmark to rescue this arm.** No arm passed the agreed local admission gate. The appropriate action is to stop promotion of this design.
* **Reopen VSA productization only for a materially different workload and explicit advantage.** Require a concrete user need, a strong exact/non-VSA baseline, a fixed resource budget, and falsifiable held-out success criteria. Potential improvements remain research until independently measured.

This work changed only research harness/artifact files. It did not implement production features or repairs. Other workspace work may proceed independently.

## Reproduction and limitations

See the [harness README](../../research/vsa_episodes/README.md), [frozen plan](PLAN.json), [final test log](tests-final.log), [source/binary hashes and environment](evaluation/manifest.json), and [run/RSS log](evaluation/execution.log). Fixtures and post-mutation snapshots are preserved per case. All 24 tests passed. Evaluation completed in 28.89 seconds on Apple M4, arm64; whole-process peak RSS was 27.83 MiB. All recorded source/plan inputs still match the pre-run hashes.

The plan references audit commit `2e4c91d`; execution HEAD was `b10329c629c80550649625b178d9cdcc045cd5ee` because unrelated work advanced the repository. The three reused phase, popcount, and BM25 files had no diff from the audit commit when checked before evaluation; the manifest pins the actual consumed files and binary.

Annotations are oracle inputs shared equally; no extraction accuracy/cost is included. Synthetic entities, exact literals, small corpora, a single thread, co-resident indexes, allocator instrumentation, and uncontrolled host activity limit external validity. Retained requested heap bytes exclude fragmentation, allocator metadata, stack, and transient peaks. Whole-process RSS is reported separately and cannot be assigned to an individual arm. Snapshot reopen is a harness check, not evidence of production crash recovery or durability. The quality/safety findings are stronger than microsecond timing precision; this is an admission decision for the tested design, not a universal VSA verdict.
