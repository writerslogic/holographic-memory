# Parked: 1A.4 lever 1, NN-descent initialization (2026-10-16)

Mechanism: 8 rounds of NN-descent (k = degree, forward and sampled reverse joins through new
entries) build the k-NN lists, pruned at alpha 1.0, replacing the first insertion pass.
Measured on nytimes-256-angular, held-out train split (288k rows, degree 64, build_ef 200, 5
threads, 1-minute load 92-135): NN-descent 295.8 s wall against the first pass's 198.7 s; the
remaining pass 48.8 s against 53.2 s; peak RSS 4.29 GB against 0.98 GB (reverse-edge table and
per-round update lists); recall 0.8976 / 0.9396 at ef 96 / 384 against 0.9049 / 0.9498.
Why it loses: the construction search examines about 7.5k fresh neighbours per insertion, while
NN-descent at k = 64 scores 47M exact candidates over 8 rounds (160 per vertex per round after
the new-entry filter) with 1 KB f32 rows, and the local joins touch more distinct rows.
Patch: 1A4-nn-descent-init.patch (also carries the quantized-search and locality-order levers).
