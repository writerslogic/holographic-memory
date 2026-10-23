# ann-benchmarks harness, single core

These results come from running HMS and its competitors through the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) runner (commit `2e081ad`) on an Intel Core Ultra 9 185H under WSL2 with AVX2 and no AVX-512. Each run is pinned to one CPU, for the index build and for every query. HMS is commit `19f4371`, using the wrapper in [`benchmarks/ann-benchmarks/hms`](../../benchmarks/ann-benchmarks/hms). The results file [`ann_benchmarks_harness_wl02_v1.json`](../../benchmarks/results/ann_benchmarks_harness_wl02_v1.json) has every point, each competitor's version, and each deviation from upstream's docker runner. The main deviations are:

- `--local` with `taskset` replaces docker;
- `--runs 2`;
- up to 10 runs in parallel;
- subsets of the competitors' upstream run groups.

The table gives the best QPS at recall@10 of at least 0.90, 0.95 and 0.99, for HMS and for the fastest competitor that was run. The last column is the ratio of the two. Noise is large on this host: two HMS settings that did identical work in the same run differed by up to 1.37x. Treat ratios within 1.37x as ties. Variation between runs made at different times was not measured, so a gap a little beyond that is a probable loss rather than a firm one.

| Dataset | Recall | HMS QPS | Fastest competitor | QPS | HMS / competitor |
|---|---|---|---|---|---|
| glove-100-angular | 0.90 | 1,457 | scann | 2,985 | 0.49 |
| glove-100-angular | 0.95 | 1,019 | scann | 1,535 | 0.66 |
| glove-100-angular | 0.99 | 511 | scann | 592 | 0.86 |
| sift-128-euclidean | 0.90 | 4,649 | faiss-ivfpqfs | 5,276 | 0.88 |
| sift-128-euclidean | 0.95 | 2,819 | faiss-ivfpqfs | 3,787 | 0.74 |
| sift-128-euclidean | 0.99 | 1,267 | faiss-ivfpqfs | 1,895 | 0.67 |
| fashion-mnist-784-euclidean | 0.90 | 9,454 | glass | 14,180 | 0.67 |
| fashion-mnist-784-euclidean | 0.95 | 9,454 | glass | 14,180 | 0.67 |
| fashion-mnist-784-euclidean | 0.99 | 7,316 | glass | 7,809 | 0.94 |

HMS is not the fastest at any of these points. Four are ties within the noise: GloVe 0.99, SIFT 0.90, SIFT 0.95 and Fashion-MNIST 0.99. GloVe 0.95, SIFT 0.99 and Fashion-MNIST 0.90/0.95 trail by about 1.5x, which is probable rather than firm. GloVe 0.90 is a clear loss at 2.0x.

On GloVe, recall grows slowly with the pool size. Degree 64 needs ef 120 for 0.93 and ef 640 for 0.99, so traversal cost dominates there. Re-ranking the whole pool barely changes recall at a given ef.

Competitors that were not run include NGT/ONNG/QSG-NGT and DiskANN/Vamana. The host has no AVX-512, so glass, faiss and HMS all run their AVX2 paths; published AVX-512 rankings do not carry over. hnswlib's GloVe points are far below its published position relative to scann and are not validated.

![glove-100-angular](ann-benchmarks/glove-100-angular.png)
![sift-128-euclidean](ann-benchmarks/sift-128-euclidean.png)
![fashion-mnist-784-euclidean](ann-benchmarks/fashion-mnist-784-euclidean.png)

## Single-thread search after `perf/single-thread-search` (v2, unlocked rounds)

[`ann_benchmarks_single_thread_wl02_v2.json`](../../benchmarks/results/ann_benchmarks_single_thread_wl02_v2.json) pairs the code of the v1 standing (`791749b`, 4-bit traversal codes) against this branch on the same host and the same indexes: degree 32, build_ef 500, 8-bit residual, one pinned CPU, 2,000 test queries per configuration, two rounds interleaved before/after, median QPS. Three arms: the code before; the code after on the same 4-bit index, which returns the same ids for every configuration (the file carries the result hashes); and the code after with 8-bit traversal codes, now the wrapper's `code_bits` default. These rounds ran outside the host's timing lock, which another job held for the whole window, at a 1-minute load of 9 to 25: absolute QPS are below the v1 harness numbers and the before/after ratios are inflated where the old code waited on memory. A run of the same arms with 10,000 queries and three rounds under the lock is queued and replaces this table when it lands. The competitor column repeats the v1 standing; the v1 noise note (ratios within 1.37x are ties) applies. GloVe is degree 32 here, so its 0.95 and 0.99 points (degree 64 in v1) are not reached by the swept pool sizes.

| Dataset | Recall | HMS before (4-bit) | HMS after (4-bit, same results) | HMS after (8-bit codes) | Fastest competitor (v1) | QPS |
|---|---|---|---|---|---|---|
| fashion-mnist-784-euclidean | 0.90 | 6772 | 9402 | 11302 | glass | 14180 |
| fashion-mnist-784-euclidean | 0.95 | 5198 | 9362 | 11302 | glass | 14180 |
| fashion-mnist-784-euclidean | 0.99 | 5040 | 7448 | 6774 | glass | 7809 |
| sift-128-euclidean | 0.90 | 4314 | 6792 | 12676 | faiss-ivfpqfs | 5276 |
| sift-128-euclidean | 0.95 | 4314 | 6792 | 7406 | faiss-ivfpqfs | 3787 |
| sift-128-euclidean | 0.99 | 1174 | 2030 | 3679 | faiss-ivfpqfs | 1895 |
| glove-100-angular | 0.90 | 538 | 1184 | 1874 | scann | 2985 |
| glove-100-angular | 0.95 | - | - | - | scann | 1535 |
| glove-100-angular | 0.99 | - | - | - | scann | 592 |

GloVe 0.90, 0.95 and 0.99 remain losses to scann on these rounds (degree 32; v1 used degree 64 for 0.95 and 0.99, not rebuilt here). Fashion-MNIST 0.90 and 0.95 remain behind glass. Fashion-MNIST 0.99 and the three SIFT points are ahead of the v1 competitor numbers, but the two measurements were not made at the same time or under the same load, so a harness rerun decides them.

What changed, from the profile in the results file (`profile_before`): on this host each row estimate waited on the scale (and, for Euclidean, the norm term) of the row, two random 4-byte reads that were not prefetched with the code row; the re-rank kernels were scalar; the entry descent read upper-layer rows one at a time. Prefetching those, and AVX2/NEON re-rank kernels that keep the scalar arithmetic, leave every result unchanged. On SIFT the 4-bit max-abs code is what forces the large pool: with 8-bit codes the same graph reaches 0.99 at ef 160 instead of 320 and 0.95 at ef 45 instead of 80. On GloVe the two code widths give the same recall at every ef (0.9170 vs 0.9165 at ef 320), so its pool size is a property of the graph, and GloVe keeps its losses.
