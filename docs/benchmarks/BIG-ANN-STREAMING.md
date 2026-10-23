# big-ann-benchmarks streaming track (NeurIPS'23)

The streaming track replays a runbook of inserts, deletes and searches from an empty index
(`neurips23/runbooks/delete_runbook.yaml` on msturing-10M-clustered: 100-d float32, L2,
at most 6M live rows, 33 search checkpoints of 10,000 public queries) and scores the mean
recall@10 over the checkpoints, under 8 GiB of DRAM and one hour. HMS enters it with
`UpdatableIndex` (`src/python/vector_index.rs` over `core::qgraph::UpdatableVGraph`): a
memory-mapped base graph, an in-memory delta graph of the rows inserted since the last
consolidation, and a rebuild of the base over the live rows when the delta or the deleted
share passes the adapter's thresholds (delta 1.0x the base, 30% deleted, 100k rows minimum).

Every number below traces to `benchmarks/results/bigann_streaming_delete_runbook_v1.json`,
which records the HMS commit, the harness commit (`89a3aba`), the machine and the parameters.

## Where the recall went

The recall deficit at the ef the leaderboard entries use (64-128) was diagnosed on the first
1M rows of the dataset (float64 brute-force ground truth over the live rows, 1,000 public
queries) and on the first 12 steps of the official runbook against the harness ground truth,
with per-search-step recall recorded. The hypotheses that deletes or the delta were the loss
are falsified by that record: recall is low at the first checkpoint, before any delete, and
stays flat across deletes and consolidations (at ef 1024 the 33 official checkpoints sit in
0.91-0.98 with no trend). Two settings of the build explain the gap to the static
`VectorIndex` on the same rows and the gap to the track baseline:

| 1M prefix, R32 L64, recall@10 | ef 64 | ef 128 | ef 256 |
|---|---|---|---|
| static `VectorIndex` (alpha 1.0) | 0.716 | 0.810 | 0.888 |
| `UpdatableIndex` base, alpha 1.2 (the old default) | 0.631 | 0.743 | 0.826 |
| `UpdatableIndex` base, alpha 1.0 | 0.716 | 0.811 | 0.888 |
| `UpdatableIndex` delta, alpha 1.2 | 0.619 | 0.714 | 0.798 |
| `UpdatableIndex` delta, alpha 1.0 | 0.667 | 0.763 | |
| static `VectorIndex` R64 L128 | 0.836 | 0.912 | 0.957 |
| `UpdatableIndex` base R64 L128, alpha 1.2 | 0.796 | 0.887 | 0.941 |

* `alpha`, the pruning parameter of the second construction pass, defaulted to 1.2 in
  `UpdatableIndex.create` (and in the track adapter) while `VectorIndex` and
  `BuildParams::default()` use 1.0. At degree 32 the alpha-1.2 prune keeps nearer, shorter
  edges and loses 7-9 points of recall@10 at ef 64-128 on both the base and the delta. With
  alpha 1.0 the consolidated base reproduces the static index exactly (the regression test
  `consolidated_base_matches_static_build_at_equal_params` pins that equality). The default
  is now 1.0.
* The remaining distance to the DiskANN baseline (R64, Ls 100: 0.924) is the degree: R64 L128
  gains another 12 points at ef 64 and 10 at ef 128 on the prefix, at 2x the build time per
  consolidation and about 128 more bytes per live row; R32 peaked at 7.47 GiB
  (sampled summed RSS, Modal, ef 1024) and at 4.85 GiB (GNU time, writerslogic-02, ef 64 and 128), so
  whether R64 fits the 8 GiB cap is unmeasured; it needs one official R64 run.
* Candidate retention (`reservoir`) changes nothing at any ef (0.7161 at 0, 256 and 512), so
  neither quantized estimates nor pool eviction is a loss mechanism here; the queries are
  simply far from the clustered rows (in-distribution queries reach 0.973 at ef 64 on the
  same graph).

Per search step of the official runbook (harness ground truth, same wheel, R32 L64 T8):

| step (live rows) | ef 64 before | ef 64 after | ef 128 before | ef 128 after |
|---|---|---|---|---|
| 2 (256k) | 0.706 | 0.777 | 0.825 | 0.875 |
| 4 (492k) | 0.712 | 0.771 | 0.814 | 0.862 |
| 6 (825k) | 0.586 | 0.692 | 0.710 | 0.796 |
| 8 (725k) | 0.605 | 0.705 | 0.725 | 0.805 |
| 10 (981k) | 0.654 | 0.739 | 0.761 | 0.830 |
| 12 (1.47M) | 0.644 | 0.721 | 0.738 | 0.807 |
| mean of the six | 0.651 | 0.734 | 0.762 | 0.829 |

## Official runs (delete runbook, `run.py --nodocker`, 8 CPUs, 8 GiB)

| HMS | build | ef | recall@10 | wall | peak RSS | host |
|---|---|---|---|---|---|---|
| ff70f3e | R32 L64 alpha 1.2 | 64 | 0.6814 | about 18 min per ef | 7.78 GB (both) | Modal 8 vCPU |
| ff70f3e | R32 L64 alpha 1.2 | 128 | 0.7755 | | | Modal 8 vCPU |
| cc5dede | R32 L64 alpha 1.2 | 1024 | 0.9301 | 36.0 min | 7.47 GiB | Modal 8 vCPU |
| 791749b, alpha 1.0 passed explicitly | R32 L64 alpha 1.0 | 64 | 0.7623 | 59:40 for both replays, nice 19 | 4.85 GiB (GNU time, both) | writerslogic-02, cores 0-7 |
| 791749b, alpha 1.0 passed explicitly | R32 L64 alpha 1.0 | 128 | 0.8401 | | | writerslogic-02, cores 0-7 |

The baseline the track README quotes is DiskANN at R64 L50 Ls100: 0.924 in about 23 minutes
on the competition VM (Azure D8lds v5). The writerslogic-02 runs are recall runs at `nice 19`
on a shared host (Core Ultra 9 185H, 8 of 22 threads, 8 GiB cgroup cap); their wall time is
an upper bound, not a timing claim.
