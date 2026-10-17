# Parked: 1A.4 locality insertion order (2026-10-16)

Mechanism: insert vertices sorted by a seeded 256-centroid k-means cluster, then by a seeded
projection, so that a batch's construction searches share their rows and codes in cache.
Measured on nytimes-256-angular, held-out train split (5 threads, load 112-139): passes 78.2 s
and 89.4 s (greedy CPU 282 s and 324 s) against the shuffled 198.7 s and 53.2 s (764 s and
189 s); recall 0.8926 / 0.9341 at ef 96 / 384 against 0.9049 / 0.9498. The first pass gets
cheaper, the second pass dearer, and the result is worse. The order's prefix also feeds the
upper layers, which then sample one cluster (upper phase 0.2 s, 22k fresh against 38M), so the
entry points degrade; that confound must be removed (shuffled prefix for the layers) before any
retry. Patch: 1A4-nn-descent-init.patch carries this lever's code (`locality_order`).
