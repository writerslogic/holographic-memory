# big-ann-benchmarks filtered track (YFCC-10M): development results

Status: development measurements only. No standings run has been made; the leaderboard is run
by the track maintainers on Azure D8lds_v5 (8 vCPU Xeon Platinum 8370C).

## Task

yfcc-10M: 10M CLIP vectors, 192-d uint8, L2; every item has a bag of tags (200,386-tag
vocabulary, 10.8 tags per item on average). 100k queries, each with 1 or 2 tags (61.6% / 38.4%)
that must all be in the item's bag. Ranked by QPS at recall@10 >= 0.90 on the public queries.
Live board (upstream 89a3aba, fetched 2026-10-23): pinecone 85,491 (closed), zilliz 84,596
(closed), parlayivf 37,902, puck 19,193, hwtl_sdu_anns_filter 15,059 (closed), wm_filter 14,468,
dhq 13,671, fdufilterdiskann 5,680, pyanns 5,185, faissplus 3,777, faiss 3,033, cufe 2,917.

Exact match counts of the public queries (from the base metadata):

| matches | queries | 1-tag | 2-tag |
|---|---|---|---|
| < 1k | 24,055 | 8,051 | 16,004 |
| 1k - 10k | 22,800 | 4,058 | 18,742 |
| 10k - 100k | 23,443 | 19,815 | 3,628 |
| 100k - 1M | 24,044 | 24,044 | 0 |
| >= 1M | 5,658 | 5,658 | 0 |

## Index (`core::qgraph::filter::FilteredIndex`)

Per-tag sorted posting lists, membership bitsets for tags with at least `bitset_min` members, and
for every tag with more than `ivf_min` members a k-means inverted file over that tag's members
alone (about `cluster_size` per list, at most `max_lists`); tags with at least `copy_min` members
also keep their rows copied in list order. A query with at most `brute_max` matches (one tag: its
posting; two tags: the intersection, computed when the rarer tag has at most `intersect_max`
members) is scored exactly over every match; otherwise the rarer tag's lists are probed
nearest-centroid first, skipping members without the other tag, until `target` matches are
scored. Distances are exact integer L2 from the i8 dot kernels (AVX-VNNI / AVX512-VNNI / AVX2 /
NEON). With `brute_max = target = usize::MAX` results equal exact filtered brute force (tests,
checked on the scalar, AVX2 and AVX-VNNI tiers; the AVX512-VNNI tier is not verified on hardware).

## Harness runs (public queries)

Intel Core Ultra 9 185H, `taskset -c 0-7` (4 P-cores x 2 hyperthreads), 8 search threads,
`run.py --nodocker --neurips23track filter --dataset yfcc-10M --runs 3`, under the timing lock but
on a host shared with other jobs (load average 18-23 on 22 threads). Build ivf_min 5000,
cluster_size 1000, max_lists 2048, copy_min 50000. Results:
`benchmarks/results/bigann_filter_yfcc10m_dev_2026-10-23.json`.

| run | HMS commit | bitset_min | brute_max | target | recall@10 | QPS |
|---|---|---|---|---|---|---|
| 1 | 372cddb | 5000 | 5000 | 2000 | 0.8857 | 17,530 |
| 1 | 372cddb | 5000 | 5000 | 2500 | 0.9045 | 14,130 |
| 1 | 372cddb | 5000 | 10000 | 2500 | 0.9067 | 16,260 |
| 1 | 372cddb | 5000 | 5000 | 3000 | 0.9194 | 14,232 |
| 1 | 372cddb | 5000 | 5000 | 4000 | 0.9397 | 12,416 |
| 2 | fb28403 | 20000 | 10000 | 2000 | 0.8892 | 17,413 |
| 2 | fb28403 | 20000 | 5000 | 2500 | 0.9045 | 16,205 |
| 2 | fb28403 | 20000 | 10000 | 2500 | 0.9067 | 15,505 |
| 2 | fb28403 | 20000 | 20000 | 2500 | 0.9106 | 13,801 |
| 2 | fb28403 | 20000 | 10000 | 3000 | 0.9208 | 15,940 |

Index size reported by the harness: run 1 10.6 GB, run 2 9.8 GB (bitsets 3.4 GB -> 0.7 GB).
Build 167-179 s. Run-to-run noise on this shared host is large (e.g. target 3000 measured faster
than target 2500 in run 2); treat differences under ~15% as noise.

Development QPS at recall@10 >= 0.90: about 16k on 4 hyperthreaded P-cores. That sits between
puck (19,193) and hwtl_sdu_anns_filter / wm_filter (~15k), below parlayivf (37,902) and the
closed entries. The estimate is uncalibrated: no other entry was run on this host.

## Parameter sweeps (private queries, tuning)

`query.private.2727415019` with its ground truth, the same 8 CPUs, without the lock, load
average in brackets. QPS per band times that band's queries as one batch.

| sweep | build | brute_max:target | recall | QPS | recall by band (<1k, 1k-10k, 10k-100k, 100k-1M, >=1M) |
|---|---|---|---|---|---|
| r1 [~15-22] | ivf_min 20000, bitset_min 20000, max_lists 1024, no copies, routing by rarer posting | 20000:2000 | 0.8903 | 12,279 | 1.0, 0.998, 0.879, 0.751, 0.642 |
| r1 | | 20000:5000 | 0.9521 | 8,221 | 1.0, 1.0, 0.964, 0.885, 0.798 |
| r1 | | 20000:10000 | 0.9799 | 5,762 | 1.0, 1.0, 0.991, 0.950, 0.895 |
| r1 | | 50000:5000 | 0.9557 | 7,151 | 1.0, 1.0, 0.980, 0.885, 0.798 |
| r1 | | 50000:10000 | 0.9805 | 4,978 | 1.0, 1.0, 0.994, 0.950, 0.895 |
| r1 | | 10000:5000 | 0.9521 | 7,283 | as 20000:5000 |
| r2 [10-12] | ivf_min 5000, bitset_min 5000, max_lists 2048, copy_min 50000, routing by match count | 5000:1500 | 0.8587 | 23,788 | 1.0, 0.976, 0.819, 0.702, 0.627 |
| r2 | | 5000:2500 | 0.9043 | 18,336 | 1.0, 0.990, 0.892, 0.785, 0.715 |
| r2 | | 5000:4000 | 0.9396 | 14,179 | 1.0, 0.998, 0.944, 0.856, 0.795 |
| r2 | | 10000:2500 | 0.9065 | 18,768 | 1.0, 1.0, 0.892, 0.785, 0.715 |
| r3 [14-29] | as r2 with bitset_min 50000, copy_min 20000 | 1000:2500 | 0.9042 | 5,686 | 1.0, 0.990, 0.892, 0.785, 0.715 |
| r3 | | 2500:2500 | 0.9042 | 6,459 | |
| r3 | | 5000:2500 | 0.9043 | 13,659 | |
| r3 | | 2500:3500 | 0.9308 | 9,289 | 1.0, 0.996, 0.932, 0.838, 0.772 |

r3 ran at load average up to 29 and is not comparable for QPS; its recall column stands.

## Not done

- Graph search with a tag predicate for the common tags (filtered-Vamana / ACORN style) and
  per-list tag signatures (Bloom) to skip lists: not built; the remaining recall loss is in the
  1-tag queries with more than 100k matches (0.785 and 0.715 at target 2500).
- A 4-bit screen with 8-bit re-rank over the copied lists.
- Inserts need a rebuild; deletes are tombstones.

## Post-freeze command (Azure D8lds_v5, maintainers' procedure)

```
python install.py --neurips23track filter --algorithm hms
python run.py --neurips23track filter --algorithm hms --dataset yfcc-10M
python data_export.py --out res.csv
```

with `neurips23/filter/hms` from the fork branch `filter/hms`, whose Dockerfile pins the HMS
commit.
