# big-ann-benchmarks streaming track (NeurIPS'23)

The streaming track replays a runbook of inserts, deletes and searches from an empty index
(`neurips23/runbooks/delete_runbook.yaml` on msturing-10M-clustered: 100-d float32, L2,
at most 6M live rows, 33 search checkpoints of 10,000 public queries) and scores the mean
recall@10 over the checkpoints, under 8 GiB of DRAM and one hour. HMS enters it with
`UpdatableIndex` (`src/python/vector_index.rs` over `core::qgraph::UpdatableVGraph`): a
memory-mapped base graph, an in-memory delta graph of the rows inserted since the last
consolidation, and a rebuild of the base over the live rows when the delta or the deleted
share passes the adapter's thresholds (delta 1.0x the base, 30% deleted, 100k rows minimum).

Every number below traces to `benchmarks/results/bigann_streaming_delete_runbook_v1.json`
or, from "Operating point" on, `benchmarks/results/bigann_streaming_delete_runbook_v2.json`;
both record the HMS commit, the harness commit (`89a3aba`), the machine and the parameters.

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
  (sampled summed RSS, Modal, ef 1024) and at 4.85 GiB (GNU time, writerslogic-02, ef 64 and 128), and
  the official R64 L64 run at ef 256 peaked at 5.57 GiB on Modal (see the official runs table).
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
| c80aa52 (main) | R32 L64 alpha 1.0 | 64 | 0.7623 | 14:24 (runbook steps 864 s) | 4.98 GiB | Modal cpu 8.0 |
| 082766e | R32 L64 alpha 1.0 | 64 | 0.7623 | 13:32 (runbook steps 811 s) | 4.85 GiB | Modal cpu 8.0 |
| 082766e | R32 L64 alpha 1.0 | 128 | 0.8401 | 18:53 (1,132 s) | 4.84 GiB | Modal cpu 8.0 |
| 082766e | R32 L64 alpha 1.0 | 384 | 0.9226 | 20:51 (1,250 s) | 4.87 GiB | Modal cpu 8.0 |
| 082766e | R32 L64 alpha 1.0 | **512** | **0.9371** | **15:41 (941 s)** | 4.84 GiB | Modal cpu 8.0 |
| 082766e | R64 L64 alpha 1.0 | 256 | 0.9353 | 15:09 (908 s) | 5.57 GiB | Modal cpu 8.0 |

The baseline the track README quotes is DiskANN at R64 L50 Ls100: 0.924 in about 23 minutes
on the competition VM (Azure D8lds v5). The writerslogic-02 runs are recall runs at `nice 19`
on a shared host (Core Ultra 9 185H, 8 of 22 threads, 8 GiB cgroup cap); their wall time is
an upper bound, not a timing claim.

The Modal runs at 082766e each replay the runbook once (one round per row), the R32 ones in
four containers started together, under `/usr/bin/time -v` around `run.py`, with an 8 GiB hard
memory limit and non-preemptible scheduling. `cpu=8.0` gives eight schedulable CPUs; Modal
bills `cpu` in physical cores and hides the topology inside the container, so whether these are
eight cores or eight hyperthreads is not known, and the compute is not matched to the
competition VM's 8 vCPU (D8lds v5). Wall time spread between the containers is host noise as
much as ef: the ef 512 replay finished before the ef 384 one, and the runbook-step sums differ
by 33% in the direction opposite to the search work. The runbook-step sum is the harness's own
per-step time; the wall includes start-up and the dataset range reads.

## Operating point

At R32 L64 alpha 1.0, **ef 512 scores 0.9371 against the track baseline's 0.924, at a 4.84 GiB
peak, in a 15:41 replay (941 s of runbook steps)**; ef 384 falls just short (0.9226). Every one
of the five Modal replays finished in 13:32-20:51 against the baseline's 23 minutes, one round
each and on different hardware from the baseline's, so that is an observation, not a matched
timing comparison. The ef 64 and ef 128 rows reproduce the Core Ultra 9 numbers of the 791749b runs (alpha 1.0 passed explicitly) to
the fourth digit, and main without the 3-byte ids scores the same 0.7623 on Modal, so the
change leaves the results unchanged.

The choice came from full-runbook replays on a shared Intel Core Ultra 9 185H (8 threads per
replay, nice 19, load average 11-25; `replay.py`: the official runbook through `UpdatableIndex`
with the adapter's policy, recall@10 against the harness ground truth at every checkpoint,
several ef per pass). Mean recall@10 over the checkpoints each replay reached (rows marked
partial were stopped early; their means cover the early checkpoints, which score higher than
the late ones, so they are not comparable with full means):

| build | checkpoints | ef 64 | ef 128 | ef 192 | ef 256 | ef 384 | ef 512 | ef 768 |
|---|---|---|---|---|---|---|---|---|
| R32 L64 | 33 of 33 | | | | 0.8971 | 0.9226 | 0.9371 | 0.9537 |
| R64 L64 | 33 of 33 | 0.8281 | 0.8915 | 0.9195 | 0.9353 | | | |
| R64 L48 | 11 of 33 (partial) | | 0.8563 | 0.8940 | 0.9160 | 0.9411 | | |
| R64 L128 | 6 of 33 (partial) | 0.8533 | 0.9238 | 0.9524 | 0.9669 | | | |

Each complete replay reproduced the official score to the fourth digit (R32 ef 384 0.9226, ef
512 0.9371; R64 ef 256 0.9353), so a full replay is the official score without the harness.

Degree 64 reaches the same recall at half the ef: R64 L64 at ef 256 scores 0.9353 (0.0018 below
R32 L64 at ef 512) at 0.73 GiB more memory, and its 908 s of runbook steps are
indistinguishable from R32's 941 s given the spread between containers. R32 ef 512 is the
configured point. In both, the inserts and deletes, which run the four rebuilds of 1.9M to 3.8M
live rows, are 80-87% of the step time. The R64 L128 and R64 L48 replays were stopped early.

## Memory

Peak RSS at R32 is 4.84-4.87 GiB for the benchmark process (4.97 GiB sampled with the Modal
parent), reached near the end of the runbook, when the live count (5.2M) and the mapped base
pages searches touch are largest; at R64 it is 5.57 GiB. The anonymous peak comes earlier, in
the step-49 rebuild of 3.8M rows (old delta, new adjacency, Python): 2.1 GiB at R32 and 2.7 GiB
at R64 in the replays. The 8 GiB limit is not binding at either point. The earlier 7.47-7.78 GB
peaks were measured at ff70f3e and cc5dede, before #205 moved consolidation onto a mapped
scratch matrix and a bounded encoder instead of an owned row copy plus a full owned graph and
encoding; the drop is consistent with that change, which this one does not touch.

This change narrows the consolidated base's neighbour ids from 4 to 3 bytes whenever the
generation has fewer than 2^24 - 1 rows (4-byte ids above that; variable-width encodings are
untouched), as the static `VectorIndex` already does: the base vertex file shrinks from 380 to
348 bytes per row at R32 (508 to 444 at R64), about 166 MB at 5.2M rows. Measured on Modal at
ef 64, one round each, run one after the other: main (c80aa52) peaks at 5,350,993,920 bytes
(4.98 GiB) for the benchmark process and 5,572,100,096 sampled with the parent; this change at
5,212,020,736 (4.85 GiB) and 5,432,266,752, 139 MB lower (2.6%), with the same score (0.7623).

On the Core Ultra 9 host (no reclaim pressure, 12 GiB scope), the replays' RSS peaks split into
anonymous memory (delta graph, build adjacency, Python) and mapped files (base, scratch rows,
and the replay's own `numpy.memmap` of the source, which the official adapter does not hold):
R32 L64 peaks at 4.55 GiB VmRSS (2.0 GiB anonymous, 2.5 GiB mapped) at step 73 with 5.2M live
rows; R64 L64 at 6.04 GiB (2.0 anonymous, 4.1 mapped) at step 68 with 4.8M live rows.

## Final runbook (msturing-30M-clustered): loss

One run at the same setting (R32 L64 alpha 1.0, ef 512, 082766e, Modal cpu 8.0, 8 GiB) did not
finish: the runner stopped `run.py` at its 3,600 s limit after 541 of 1,280 steps (4.3M live
rows at that point), so no recall was recorded. Measured over those steps: 270 searches at 3.88 s
each (1,049 s), 143 insert steps 1,089 s, 128 delete steps 1,461 s (deletes include the
tombstone-triggered rebuilds); peak RSS 6.02 GiB for the benchmark process, 6.22 GiB summed with
the parent, so memory was not the limit. Projection, not measured: the 640 searches alone need
about 2,480 s at ef 512, so this runbook needs a lower ef and fewer or cheaper rebuilds to fit
the hour.

## Merged consolidation (development measurements)

`UpdatableIndex.consolidate(merge=True)` keeps the previous base graph instead of rebuilding
it: it splices deleted rows out of the lists that pointed at them, links the delta in with the
build's first insertion pass, and runs the build's second pass over the delta, the base rows it
links to and the rows that lost a deleted neighbour (`docs/file-format/VECTOR-UPDATES.md`). The
fork adapter turns it on with `"merge": true`. Each Modal row below is one round (`run.py
--nodocker`, harness 89a3aba, `cpu=8.0`, 8 GiB, non-preemptible, R32 L64 alpha 1.0 T8, ef 512,
HMS 9b26418, which is c1ec45d before its rebase onto main); the control is the same image
without merging, run in the same hour, since step sums vary by about a third between containers.

| delete_runbook | delta / deleted ratio | recall@10 | wall | runbook steps: insert / search / delete s | peak RSS |
|---|---|---|---|---|---|
| rebuild (control) | 1.0 / 0.3 | 0.9371 | 17:42 | 1,061: 505 / 149 / 408 | 4.70 GiB |
| merge | 1.0 / 0.3 | 0.9376 | 12:07 | 726: 382 / 103 / 241 | 4.69 GiB |
| merge | 0.25 / 0.15 | 0.9444 | 17:32 | 1,052: 696 / 50 / 306 | 4.93 GiB |

A merge searches again for about four to six rows per changed row, so it beats a rebuild only
while the churn per consolidation is a fraction of the live set: merging at a quarter of the
base spends more on inserts than rebuilding at doubling. The configured point merges at the
rebuild thresholds (1.0 / 0.3): by construction a delta-triggered consolidation there usually falls back to a rebuild (the delta is then at least as large as the surviving base), so the merges replace the tombstone-triggered rebuilds.

Final runbook (msturing-30M-clustered), ef 256: no recall was recorded. With the image of 9b26418 at ef 256, merging at the rebuild thresholds (1.0 / 0.3) reached step 672 of 1,280 in 1,926 s of runbook steps (searches averaged about 1.0 s; the #238 run at ef 512 needed 3,599 s for its first 541 steps), then the launcher's 8 GiB guard on the summed benchmark and parent RSS stopped it at 8.60 GB during the step-673 consolidation (a merge after deletes, 8.42M live rows; the same step at which earlier rebuild-based candidates ran out of time or memory). Merging at 0.25 / 0.15 reached step 638 in 2,838 s before the same guard (8.61 GB). The ef 512 run at 0.25 / 0.15 was stopped at step 214 (1,461 s of steps against the #238 run's 976 s for the same steps) once it could not finish. The merge read the previous base's adjacency back into memory and kept those pages through the insertion passes; 50017d4 drops them as soon as the last list is read.
