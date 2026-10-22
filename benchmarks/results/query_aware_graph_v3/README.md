# Exact byte-VNNI validation

The kernel is rejected by the predeclared six-point criterion; final Rust validation selects the unchanged v2 kernel. The frozen patch and all results are retained.

Reserved learn validation only; frozen v1 official results are unchanged.

Fixed adjacency, same 8-bit codes and 8-bit residuals, graph/search settings and construction rows; v2 versus v3 source. writerslogic-02, Rust1.96.0/native, one VIBE-selected CPU, one thread, three alternating paired blocks, saved fifth pass after four warm passes.

| Dataset / ef/rerank | Actual recall | Reference QPS median [min–max] | Byte-VNNI QPS median [min–max] | Median delta | Paired deltas | Observed result |
|---|---:|---:|---:|---:|---|---|
| llama / 100/100 | 0.91523438 | 5424.77 [5394.31–5692.84] | 5574.03 [5545.71–5575.08] | +2.75% | +3.35%, -2.58%, +2.75% | inconclusive |
| llama / 256/400 | 0.99070313 | 2587.51 [1882.96–2589.60] | 2583.41 [2296.39–2631.39] | -0.16% | -0.16%, +21.96%, +1.61% | inconclusive |
| llama / 512/400 | 0.99710937 | 1526.21 [1491.42–1530.87] | 1438.71 [1311.96–1459.15] | -5.73% | -6.02%, -12.03%, -4.39% | slower in all observed blocks |
| yi / 100/100 | 0.90550781 | 7153.21 [7097.31–7156.38] | 7002.17 [6741.13–7108.54] | -2.11% | -5.76%, -1.34%, -0.67% | slower in all observed blocks |
| yi / 256/400 | 0.97546875 | 3319.19 [3316.67–3354.59] | 3217.87 [3029.71–3304.83] | -3.05% | -8.65%, -4.08%, -0.43% | slower in all observed blocks |
| yi / 512/400 | 0.99089844 | 1837.99 [1823.61–1842.33] | 1834.39 [1698.31–1861.64] | -0.20% | -6.87%, +1.05%, -0.20% | inconclusive |

Ordered returned IDs match in every paired group. Observed ranges are not statistical confidence intervals. No combined compact-layout/VNNI timing or new official-test timing was performed.

| Dataset | Reference build seconds | Byte-VNNI build seconds | Encoded bytes per index | Co-resident process peak RSS, KiB |
|---|---:|---:|---:|---:|
| llama | 333.905 | 336.200 | 206,536,668 | 1,194,452 |
| yi | 169.411 | 172.089 | 150,906,736 | 954,668 |

Build times are single observations. Process RSS includes both indexes and input/runtime buffers; isolated index peak RSS is unmeasured.

Reproduce from signed2d4967e plus [kernel.patch](kernel.patch), [kernel-v3.sh](kernel-v3.sh), and the source/binary/compiler receipts. Selection and the sole final Cargo gate use [final-gate.sh](final-gate.sh). Decisions and limits: [ledger.md](ledger.md). Machine-readable points: [summary.json](summary.json).
