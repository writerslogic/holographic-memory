# Separate coarse-routing coverage from score approximation

This is a bounded hypothesis for a future development experiment, not authorization
to run it and not a positive result. The rejected prototype is not re-enabled.

Measured evidence: on yi, C256/r64 reaches validation recall 0.88063 at 32 probes,
0.94356 at 64 and0.98247 at 128; C1024/r64 reaches 0.77733 at 32,0.86071 at 64 and
0.92502 at 128. These are the maximum recalls over the four registered rerank
values, derived from yi-development.json. The chosen model needs 256 probes out
of 1024. These observations suggest an expensive coarse-selection stage, but they
do not prove that routing causes the recall loss: truncation/quantization can
also discard neighbors after a correct list is visited. No per-stage profile
was recorded.

A useful next kill test is metric-consistent coarse routing with no low-rank
scoring. The recovered prototype uses Euclidean Lloyd assignments but inner-product
centroid routing. The primary LoRANN description uses spherical clustering for
inner product/cosine (paper Section 2.2). Compare the existing partition with a
spherical partition, keeping every other parameter fixed, and measure the exact
fraction of true neighbors present in the probed list union before any score
approximation. This establishes a ceiling before implementing another search path.

Proposed small preregistration, to be sealed before execution:

1. Use the same 32,768 learn rows for training; reserve learn rows 33868..34379 in the existing seeded permutation for this
   diagnostic after checking cross-study ledgers for prior evaluation. This run
   did not train or score those rows.
   Never read official test or select from its observed result. Keep current
   validation outcomes as historical evidence only.
2. Fix clusters 1024 and seed 24301, compare Euclidean versus spherical assignments,
   and probe 16/32/64. Compute exact top 100 from original base rows, list-union
   recall, visited-row count, centroid cost and degree of list imbalance.
3. Stop if the spherical alternative cannot retain >=0.98 neighbors at<=32
   probes while visiting<=5% of rows. These are proposed engineering thresholds,
   not data-derived promises. Report both routes and all failed cells.
4. Only if that ceiling passes, preregister a second experiment comparing full
   dimension i8 contiguous scans plus exact rerank against the graph. This avoids
   each selected list's rank-by-dimension query projection. Validate on a further
   untouched learn slice; do not reuse official test as a tuning signal.

This differs from the earlier rejected k-means graph-entry-table experiment:
it first checks candidate-set coverage and cost, without changing graph entry
points or claiming a throughput gain. If coverage fails, prioritize query-aware
graph quality/connectivity rather than another IVF implementation. The current
10M training-completeness investigation is independent and takes precedence.

Primary source: [LoRANN, NeurIPS 2024, Sections 2.2–3](https://proceedings.neurips.cc/paper_files/paper/2024/file/b939da3932e88ded5e9b08026e35069d-Paper-Conference.pdf).
