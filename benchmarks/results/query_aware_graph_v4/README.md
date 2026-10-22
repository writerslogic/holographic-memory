# Direct IP serving and independent candidate retention

Reserved learn validation only: 256 queries per full dataset, 32768 disjoint construction queries. No untouched final set is certified; these results do not establish a new official result.

Direct IP improves median QPS by 27.12–47.95% across the six tested dataset/beam points and saves about 18.4% of encoded bytes. The reservoir recovers low-beam recall at a measurable cost; at high effort, direct serving without the reservoir is the faster tested variant. Both features remain independently selectable.

The measured source is reconstructed by applying [measured-source.patch](measured-source.patch) to commit 4d5168f7ce8253b15aefe2f7ce24f0e248de4deb; final-source.sha256 and final-callers.sha256 identify its files. The PR integrates later main changes separately; its functional checks are recorded in [pr-integration.json](pr-integration.json). Timings were not rerun after integration. Large diagnostic JSONs and raw HDF5 files use Git LFS; fetch them with `git lfs pull --include='benchmarks/results/query_aware_graph_v4/**'` before regenerating this report.

Both changes are opt-in. Construction remains the lifted Euclidean graph; direct serving encodes the original 128 coordinates with explicit negative-inner-product semantics. The independent reservoir receives every row score before beam admission, including deduplicated descent scores. The existing ef >= k rule remains in force; screens and neighbor blocks reject independent retention.

```python
index = holographic_memory.VectorIndex("ip", code_bits=8, direct_ip=True)
index.fit(base, training_queries=construction_queries)
index.set_query_arguments(ef=100, rerank=400, reservoir=400)
```

Omitting direct_ip and reservoir preserves legacy serving and retention. Rust callers use VGraph::from_lifted_ip_graph and VSearchParams::reservoir. VGraph::metric now returns VMetric; construction Metric converts with Into. Serving metric word 2 identifies IP in HMSVGR01, which older readers reject. Lifted HMSGRF04 construction caches are unchanged.

## Storage and quantization

| Dataset | Lifted encoded bytes | Direct encoded bytes | Saving | Lift dominates quantization scale | Mean lifted/direct scale ratio |
|---|---:|---:|---:|---:|---:|
| llama | 206,536,668 | 168,512,296 | 18.4105% | 20.4016% | 1.080378 |
| yi | 150,906,736 | 123,105,908 | 18.4225% | 6.8387% | 1.014902 |

The measured 8-bit primary stride is 256→128 bytes; residual stride 144→128; norm storage 4→0 bytes per vector. Total savings are 148 bytes per vector plus 64 mean-array bytes. Encoded bytes exclude query scratch and runtime overhead and are not isolated RSS. Coordinate reconstruction errors are recorded in the diagnostic JSON; the lifted and direct coordinate RMSEs use different widths and are not a query-score error comparison.

## Loss breakdown

Coverage and recall are percentages. These rows use lifted serving; direct rows and all original-float oracle reranks are retained in the JSON. Oracle work is diagnostic only.

| Dataset | ef | Reservoir | Evaluated coverage | Beam coverage | Retained coverage | Rerank-set coverage | Returned recall | Float oracle over evaluated IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama | 100 | 0 | 95.4922% | 91.5234% | 91.5234% | 91.5234% | 91.5234% | 95.4922% |
| llama | 100 | 400 | 95.4922% | 91.5234% | 95.4922% | 95.4922% | 95.4922% | 95.4922% |
| llama | 256 | 0 | 99.0820% | 99.0820% | 99.0820% | 99.0820% | 99.0703% | 99.0820% |
| llama | 256 | 400 | 99.0820% | 99.0820% | 99.0820% | 99.0820% | 99.0703% | 99.0820% |
| llama | 512 | 0 | 99.7266% | 99.7266% | 99.7266% | 99.7266% | 99.7109% | 99.7266% |
| llama | 512 | 400 | 99.7266% | 99.7266% | 99.7266% | 99.7266% | 99.7109% | 99.7266% |
| yi | 100 | 0 | 91.7969% | 90.5508% | 90.5508% | 90.5508% | 90.5508% | 91.7969% |
| yi | 100 | 400 | 91.7969% | 90.5508% | 91.7969% | 91.7969% | 91.7969% | 91.7969% |
| yi | 256 | 0 | 97.5508% | 97.5469% | 97.5469% | 97.5469% | 97.5469% | 97.5508% |
| yi | 256 | 400 | 97.5508% | 97.5469% | 97.5508% | 97.5508% | 97.5508% | 97.5508% |
| yi | 512 | 0 | 99.0937% | 99.0937% | 99.0937% | 99.0937% | 99.0898% | 99.0937% |
| yi | 512 | 400 | 99.0937% | 99.0937% | 99.0937% | 99.0937% | 99.0898% | 99.0937% |

Changing only retention preserves every evaluated-ID set, ordered traversal beam and expansion count in all full-data diagnostic queries. At ef100, the 400-candidate reservoir recovers encountered true neighbors that the beam discards. At ef256/512, most misses are unencountered neighbors; original-float reranking adds at most 0.02 percentage points. Yi's low-beam discovery ceiling remains about 91.8%. Direct serving makes small, mixed recall changes; a precision improvement is not established.

## Paired query timings

writerslogic-02, Core Ultra 9 185H, one VIBE-selected CPU (0) and one worker, 8-bit traversal and 8-bit residual codes; each representation is built once and reused across all settings. Four arms, three alternating blocks, five passes per block; QPS uses only the saved fifth pass. All arms share two resident indexes. Ranges are observed minima/maxima, not confidence intervals. Rerank=400, patience=0, k=100; the legacy beam therefore refines only 100/256/400 candidates at ef100/256/512.

Default ordered IDs match all frozen v2 blocks (2304 lists per dataset); every measured four-arm result and encoded size matches the native diagnostic. The raw files, source/binary hashes, frozen input manifests, CPU/thread attributes and exact schedules are retained.

### llama

| Serving / retention | ef | Actual recall | QPS median [min–max] |
|---|---:|---:|---:|
| lifted | 100 | 91.5234% | 5325.70 [5252.44–5388.57] |
| lifted | 256 | 99.0703% | 2358.01 [2319.28–2472.26] |
| lifted | 512 | 99.7109% | 1425.14 [1405.70–1464.13] |
| lifted+reservoir | 100 | 95.4922% | 3616.63 [3594.71–3736.13] |
| lifted+reservoir | 256 | 99.0703% | 2002.88 [1740.83–2168.09] |
| lifted+reservoir | 512 | 99.7109% | 1374.28 [1362.10–1387.47] |
| direct | 100 | 91.7852% | 7529.48 [7398.94–7611.35] |
| direct | 256 | 99.0586% | 3488.74 [3235.35–3503.90] |
| direct | 512 | 99.6680% | 1972.99 [1797.27–2084.69] |
| direct+reservoir | 100 | 95.5312% | 4820.97 [4807.85–4900.29] |
| direct+reservoir | 256 | 99.0586% | 2897.86 [2854.63–3003.87] |
| direct+reservoir | 512 | 99.6680% | 1755.20 [1723.24–1866.36] |

| Required recall | Legacy: fastest sampled qualifying QPS | Direct | Direct + reservoir |
|---|---:|---:|---:|
| 90% | 5325.70 (ef=100, recall=91.5234%) | 7529.48 (ef=100, recall=91.7852%) | 4820.97 (ef=100, recall=95.5312%) |
| 95% | 2358.01 (ef=256, recall=99.0703%) | 3488.74 (ef=256, recall=99.0586%) | 4820.97 (ef=100, recall=95.5312%) |
| 99% | 2358.01 (ef=256, recall=99.0703%) | 3488.74 (ef=256, recall=99.0586%) | 2897.86 (ef=256, recall=99.0586%) |

### yi

| Serving / retention | ef | Actual recall | QPS median [min–max] |
|---|---:|---:|---:|
| lifted | 100 | 90.5508% | 6666.85 [6619.12–6877.98] |
| lifted | 256 | 97.5469% | 3200.44 [3063.21–3215.80] |
| lifted | 512 | 99.0898% | 1766.87 [1756.46–1843.32] |
| lifted+reservoir | 100 | 91.7969% | 4231.66 [4226.91–4280.89] |
| lifted+reservoir | 256 | 97.5508% | 2531.41 [2526.58–2558.20] |
| lifted+reservoir | 512 | 99.0898% | 1681.25 [1670.63–1718.91] |
| direct | 100 | 90.5781% | 8475.13 [8106.18–8492.37] |
| direct | 256 | 97.5430% | 4159.61 [4032.78–4163.56] |
| direct | 512 | 99.0859% | 2295.34 [2290.66–2377.05] |
| direct+reservoir | 100 | 91.7930% | 4976.64 [4894.67–5263.81] |
| direct+reservoir | 256 | 97.5469% | 3098.51 [3083.61–3319.78] |
| direct+reservoir | 512 | 99.0859% | 2119.34 [2051.25–2151.60] |

| Required recall | Legacy: fastest sampled qualifying QPS | Direct | Direct + reservoir |
|---|---:|---:|---:|
| 90% | 6666.85 (ef=100, recall=90.5508%) | 8475.13 (ef=100, recall=90.5781%) | 4976.64 (ef=100, recall=91.7930%) |
| 95% | 3200.44 (ef=256, recall=97.5469%) | 4159.61 (ef=256, recall=97.5430%) | 3098.51 (ef=256, recall=97.5469%) |
| 99% | 1766.87 (ef=512, recall=99.0898%) | 2295.34 (ef=512, recall=99.0859%) | 2119.34 (ef=512, recall=99.0859%) |

## Verification and limits

Measured-source Linux all-features gate: 991 Rust tests passed, 6 ignored; 52 native Python tests passed. Rust 1.89 MSRV library gate: 603 passed, 4 ignored. Formatting passes. The measured-source Clippy receipt records two baseline lints in documents/postings.rs and triple_store.rs; no lint was suppressed.

After integration with main at 1cfb43f, the fresh Linux gate passed 994 Rust tests (6 ignored) and 52 native Python tests. The Rust 1.89 library gate passed 606 tests (4 ignored). Source hashes, commands, logs and the separate passing Rust 1.96 all-features Clippy result are recorded in pr-integration.json. These checks do not remeasure the frozen timings.

Tests cover recovery after beam rejection, upper-layer deduplication/epoch wrap, unchanged traversal with patience/frontier/batched scoring, 127/128/129 storage boundaries, original magnitudes/IDs/ties/zero queries, and mapped/prefaulted/heap persistence. Compile-failure logs from updating complete parameter literals are preserved separately.

No topology or kernel change is included. Discovery remains the main high-recall limit; these measurements do not select a new degree policy. The three sampled ef values are not an exhaustive effort sweep. Validation margins remain narrow at 99%, and historical official results cannot certify the new variants. No new official or competitor-superiority claim is made; the previous inconclusive/stopped in-distribution regression evidence remains unchanged.

Reproduction: build-diagnose.sh, final-gate.sh, timing.sh and generate_report.py in this directory; the VIBE adapter/driver are benchmarks/public/qag_serving_{module,execute}.py. Decisions and execution history: [ledger.md](ledger.md).
