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
| Zep | memory | LongMemEval S answer accuracy; citation support | same reader, judge, token cap; dev 100 labeled dev; held-out 400 requires separate authorization | | | | | | 451/500 |
| Mem0 | memory | same | same | | | | | | 474/500 managed v3 top-50 |
