# ann-benchmarks harness, single core

These results come from running HMS and its competitors through the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) runner (commit `2e081ad`) on writerslogic-02, an Intel Core Ultra 9 185H under WSL2 with AVX2 and no AVX-512. Each run is pinned to one CPU, for the index build and for every query. HMS is commit `19f4371`, using the wrapper in [`benchmarks/ann-benchmarks/hms`](../../benchmarks/ann-benchmarks/hms). The results file [`ann_benchmarks_harness_wl02_v1.json`](../../benchmarks/results/ann_benchmarks_harness_wl02_v1.json) has every point, each competitor's version, and each deviation from upstream's docker runner. The main deviations are:

- `--local` with `taskset` replaces docker;
- `--runs 2`;
- up to 9 runs in parallel;
- subsets of the competitors' upstream run groups.

The table gives the best QPS at recall@10 of at least 0.90, 0.95 and 0.99, for HMS and for the fastest competitor that was run. The last column is the ratio of the two. Run-to-run noise is large on this host. Two settings that did identical work differed by up to 1.37x, and the same configuration measured an hour apart differed by up to 1.47x. Treat ratios within about 1.3x as ties.

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

HMS is not the fastest at any of these points. GloVe at 0.90 is the clearest loss (2.0x). The other gaps are 1.1x to 1.5x. The SIFT gap at 0.90 and the 0.99 gaps on GloVe and Fashion-MNIST are inside the noise.

On GloVe, recall grows slowly with the pool size. Degree 64 needs ef 120 for 0.93 and ef 640 for 0.99, so traversal cost dominates there. Re-ranking the whole pool barely changes recall at a given ef.

Competitors that were not run include NGT/ONNG/QSG-NGT and DiskANN/Vamana. Glass and faiss use their AVX2 paths on this host, so their standing in the published AVX-512 results does not carry over.

![glove-100-angular](ann-benchmarks/glove-100-angular.png)
![sift-128-euclidean](ann-benchmarks/sift-128-euclidean.png)
![fashion-mnist-784-euclidean](ann-benchmarks/fashion-mnist-784-euclidean.png)
