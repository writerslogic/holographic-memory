# Production search cost and adjacency layout

No tested point shows a repeatable speedup with separated observed ranges. Compact adjacency is retained as an explicit memory option, default off.

Reserved learn validation only; no new official evaluation. The frozen v1 official results remain separate.

Unpinned instrumented aggregate stages; no production QPS, cache, allocation, bandwidth or instruction inference.

| Dataset / layout | Actual recall | Encoded bytes | Persisted bytes | Encode query % | Entry % | Adjacency / visited / prefetch % | Scoring / pool % | Rerank / sort % | Other % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama / fixed8 | 0.99070313 | 206,536,668 | 206,622,512 | 0.07 | 1.06 | 47.35 | 37.69 | 11.08 | 2.76 |
| llama / compact8 | 0.99070313 | 158,488,097 | 206,622,512 | 0.07 | 1.01 | 48.31 | 37.09 | 10.79 | 2.72 |
| yi / fixed8 | 0.99089844 | 150,906,736 | 150,982,476 | 0.05 | 0.90 | 45.48 | 36.37 | 13.45 | 3.75 |
| yi / compact8 | 0.99089844 | 115,835,997 | 150,982,476 | 0.05 | 0.89 | 46.76 | 35.62 | 13.01 | 3.68 |

Exact ordered returned IDs, persisted bytes and operation counts match between layouts on all profiled queries.

| Dataset / ef/rerank | Actual recall | Fixed QPS median [min–max] | Compact QPS median [min–max] | Median delta | Paired deltas | Observed timing result |
|---|---:|---:|---:|---:|---|---|
| llama / 100/100 | 0.91523438 | 5381.38 [4620.49–5395.96] | 5180.40 [5147.43–5317.49] | -3.73% | -3.73%, +15.09%, -4.61% | inconclusive |
| llama / 256/400 | 0.99070313 | 2514.06 [2498.62–2523.08] | 2375.39 [2372.91–2515.33] | -5.52% | -0.31%, -4.93%, -5.61% | slower in all observed blocks |
| llama / 512/400 | 0.99710937 | 1455.12 [1376.72–1465.55] | 1377.09 [1253.20–1395.39] | -5.36% | -14.49%, -5.36%, +1.36% | inconclusive |
| yi / 100/100 | 0.90550781 | 6970.13 [6913.69–6970.37] | 6741.31 [6728.03–6872.21] | -3.28% | -3.29%, -2.69%, -1.40% | slower in all observed blocks |
| yi / 256/400 | 0.97546875 | 3078.23 [3036.80–3306.59] | 3160.77 [3118.90–3220.58] | +2.68% | +4.08%, +1.32%, -2.60% | inconclusive |
| yi / 512/400 | 0.99089844 | 1826.05 [1810.29–1906.86] | 1802.30 [1594.04–1844.95] | -1.30% | -11.95%, -5.48%, +1.03% | inconclusive |

One harness-selected CPU, five passes, saved fifth pass, three alternating paired blocks; observed ranges are not confidence intervals.

| Dataset | Fixed build seconds | Compact build seconds | Co-resident process peak RSS, KiB |
|---|---:|---:|---:|
| llama | 350.092 | 345.635 | 1,216,140 |
| yi | 176.030 | 176.085 | 956,880 |

Compaction saves live encoded adjacency capacity. Save expands the original padding, so persisted-file size does not shrink. Save/open/compaction times in [summary.json](summary.json) are single diagnostic observations. Isolated index RSS/peak build RSS remain unmeasured.

Sources, settings, stage totals and work counts: [summary.json](summary.json). Drivers: [profile-v2.sh](profile-v2.sh), [layout-v2.sh](layout-v2.sh). Decisions: [ledger.md](ledger.md).
