# Neighbour-major fast-scan blocks: kill test (development measurement)

Question: would a SymphonyQG-style neighbour-major vertex block (each vertex's neighbour ids, the
neighbours' fast-scan codes and their factors, plus the vertex's own code, in one contiguous
fixed-degree record) close the single-thread gap to the RaBitQ/fast-scan graph leaders on VIBE?

The index already has the code half of that layout: `VGraph::with_blocks` stores, per vertex,
32-code fast-scan records of the neighbours' residual codes (1, 2 or 4 bits per rotated
coordinate, per-edge scale and offset), scanned with `fastscan::scan32` (AVX2 `vpshufb`, NEON
`tbl`, scalar). The new format would only merge the two remaining reads per expansion (the
neighbour ids and the vertex's own 8-bit row) into that record. Decision rule, written before
the run: with R = QPS(per-row path) / QPS(1-bit blocks) at recall@100 0.95 on a 768-d set,
R >= 1.3 means contiguity cannot recover the loss and the format is not built.

## Result

Paired, interleaved, locked, 1 thread pinned, Intel Core Ultra 9 185H (no AVX-512), commit
8e66e387, 3 rounds, host load 5.1-10.3 (other work on the host; the arms are interleaved per
round, so the load hits every arm). Raw points, protocol and the lock log:
[`benchmarks/results/qgraph_neighbour_blocks_ab_x86_v1.json`](../../benchmarks/results/qgraph_neighbour_blocks_ab_x86_v1.json).

| Dataset (recall@100) | Target | Per-row path | 1-bit blocks | 2-bit blocks |
|---|---|---|---|---|
| arxiv-nomic-768, 100k slice, 256 queries | 0.90 | 4,013 | 3,618 (0.90x) | 3,716 (0.93x) |
| | 0.95 | 4,013 | 2,974 (0.74x) | 3,082 (0.77x) |
| | 0.99 | 2,979 | 2,002 (0.67x) | 2,374 (0.80x) |
| glove-100-angular, 10k queries | 0.90 | 1,353 | does not reach 0.90 by ef 1200 | 1,021 (0.75x) |

QPS is the median over rounds; recall targets are interpolated log-linearly between the
bracketing ef points (the per-row path already reaches 0.977 at ef 100, the floor for k = 100).
Degree 32 does not reach 0.95 on GloVe on either path.

R = 4,013 / 2,974 = 1.35: the new format is not built. Noise: with 256 queries and this host
load, one configuration's three rounds spread up to 1.5x (the medians are what the table
uses), so R is known to perhaps +-0.2; the recall side of the loss (ef ~170 against <= 100 for
0.95) is exact and load-independent.

## Why

* At matched ef the 1-bit blocks are 1.12-1.26x faster per query on the 768-d slice (medians
  of this run; 1.09-1.37x in an earlier back-to-back run), so the block read itself is cheap
  enough. The loss is recall per
  ef: neighbour order from 1-bit residual estimates needs about 1.7x the ef of the 8-bit
  per-row estimates for 0.95 (ef ~170 against <= 100).
* A 1-bit record is already 4,352 bytes per vertex at 768-d (32 codes of the 1024-wide
  rotation plus 256 bytes of factors). Folding in the 768-byte own row and 96 bytes of ids
  merges about 0.9 KB into a stream that is already prefetched one expansion ahead; it does not
  touch the estimate quality that costs 1.35x.
* Earlier 4-bit result on nytimes (`qgraph_block4_untimed_x86_v1.json`): no ef penalty, 0.76-0.79x
  of the per-row path, index 2.7 GB.

Not covered: Euclidean sets (sift-128, fashion-mnist; the blocks are angular only), AVX-512
`vpshufb` on 64-byte registers (the timing host has no AVX-512), and full-size VIBE sets
(the slice is 100k rows; its 8-bit rows, 77 MB, already exceed the host's last-level cache).
