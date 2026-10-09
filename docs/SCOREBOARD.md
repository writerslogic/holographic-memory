# Research scoreboard, October 2026

Blank matched cells have no measured result. Vendor-native values are separate;
100 reused dev questions cannot establish a margin against a vendor's full 500.

| Competitor | Frontier | Metric | Matched condition | HMS | Them | Margin | Status | File | Vendor-native |
|---|---|---|---|---|---|---|---|---|---|
| FAISS HNSW, IVF-PQ | ANN | single-thread QPS at recall@10 0.90 and 0.95; index bytes | nytimes-256, glove-100; ann-benchmarks recall; 11 paired rounds; load gate; same harness | | | | | | |
| hnswlib | ANN | same | same | | | | | | |
| SymphonyQG | ANN speed | same | same, x86 AVX-512; raw, 4-bit, 8-bit refinement | | | | no x86 run | | |
| RaBitQ IVF | ANN bytes | index bytes at matched recall; QPS | same | | | | no x86 run | | |
| SPLADE v3 | text | BEIR nDCG@10 | SciFact, NFCorpus, then full average; encoder pinned | | | | | | |
| Zep | memory | LongMemEval S answer accuracy; citation support | same reader, judge, token cap; dev 100 labeled dev; held-out 400 requires separate authorization | dev 87/100; citation spans 201/204 |  |  | dev measured | [answers](../benchmarks/results/longmemeval_dev_answers_v1.json), [check](../benchmarks/results/longmemeval_dev_answers_v1_summary.json) | 451/500 |
| Mem0 | memory | same | same | dev 87/100; citation spans 201/204 |  |  | dev measured | [answers](../benchmarks/results/longmemeval_dev_answers_v1.json), [check](../benchmarks/results/longmemeval_dev_answers_v1_summary.json) | 474/500 managed v3 top-50 |

Dev uses gpt-5.4-2026-03-05 for the reader and official judge. First-five, knapsack and operand score 83/100, 82/100 and 87/100; their verbatim citation-span scores are 187/192, 180/186 and 201/204. The local Qwen3-4B arms score 47/100, 46/100 and 48/100, including four malformed outputs as incorrect.

The full answer checker rejects six corruptions; the [local evidence checker](../benchmarks/results/weak_reader_evidence_v1_check.json) rejects nine. Repeated judging disagrees on 12/400 calls. Citation-span validation checks retained-source text membership separately from answer correctness.
