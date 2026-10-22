# Query-aware graph evidence

Official full-dataset performance remains unmeasured.
Regression gate: **inconclusive**. Formatting: **unrun**. Required Rust gate: **unrun**.

Baseline SHA: `08d37778a66b58709461456e5b9a7ab6a05b399f`. Source decisions, configurations, seeds and commands are retained in [ledger.md](ledger.md); dataset/runtime fingerprints are in [provenance.json](provenance.json).

B is ordinary entry with coded routing; O starts at the true nearest base vector; V is true-top100 coverage among all unique distance-evaluated vertices during B; E uses exact routing; OE combines oracle entry and exact routing. All diagnostic rows use 256 held-out learn queries, degree=64, build_ef=500, seed=24301, ef=2000, rerank=1000 and exact original-IP ground truth.

| Held-out learn split | B | O | V during B | E | OE | B retained coverage | B rerank-candidate coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| [llama 200k prefix](llama-prefix-diagnosis.json) | 0.9655078125 | 0.96578125 | 0.99203125 | 0.991484375 | 0.99171875 | 0.9862890625 | 0.9659375 |
| [llama full base](llama-full-b4.json) | 0.95484375 | 0.955 | 0.9877734375 | 0.9896875 | 0.9896875 | 0.9791015625 | 0.9551171875 |
| [yi full base](yi-full-b4.json) | 0.9802734375 | 0.9804296875 | 0.98359375 | 0.9833984375 | 0.983515625 | 0.9831640625 | 0.9808984375 |

Observed prefix oracle-entry gain is 0.0002734375; exact-routing gain is 0.0259765625. The prefix supports routing/retention precision as the larger loss at that operating point. Full-base diagnostic rows retain the narrower evidence for each dataset.

**Conditional projected-graph ablation, untimed.** Both rows use the same llama 200k graph, 8-bit codes plus 8-bit residuals, 32,768 construction queries, quota 32, seed 24301 and the same 256 held-out learn queries. Only ef/rerank changes between rows; E/OE replace coded routing within each row. These are prefix diagnostics, not official-test results.

| Projected prefix config | B | V during B | B retained coverage | B rerank-candidate coverage | E | OE | E − B |
|---|---:|---:|---:|---:|---:|---:|---:|
| [ef=100,rr=100](llama-prefix-proj-q32768-d32-ef100.json) | 0.9159375 | 0.9566796875 | 0.9159375 | 0.9159375 | 0.9577734375 | 0.95734375 | +0.0418359375 |
| [ef=160,rr=200](llama-prefix-proj-q32768-d32-ef160.json) | 0.9789453125 | 0.979453125 | 0.9789453125 | 0.9789453125 | 0.9801953125 | 0.979765625 | +0.0012500000 |

At ef=100, evaluated top-100 coverage exceeds retained coverage and exact routing raises recall. This supports testing whether distance refinement before admission helps at small beams. No E/OE QPS or refinement speedup was measured.

**Official full-dataset Pareto points.** Every measured QPS entry is the saved fifth pass from each of three alternating blocks: median [min–max], one thread, writerslogic-02, common VIBE harness/affinity/cache policy. Selection uses actual minimum strict ID recall across repetitions ≥target, with no interpolation. Config p is the stale-expansion patience; RoarGraph L is its query width.

| Dataset | Recall target | Before: strict recall; QPS [min–max]; config | After: strict recall; QPS [min–max]; config | RoarGraph: strict recall; QPS [min–max]; config | MLANN: strict recall; QPS [min–max]; config | After vs best measured CPU |
|---|---:|---|---|---|---|---|
| llama-128-ip | 0.90 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| llama-128-ip | 0.95 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| llama-128-ip | 0.99 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| yi-128-ip | 0.90 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| yi-128-ip | 0.95 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| yi-128-ip | 0.99 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |

Comparator deltas cover only the frozen measured RoarGraph M48/M96 and MLANN RF10 depth14 configurations, plus matching separate RF80 depth13 reports where available. RF80 depth13 remains unrun in the saved evidence. These measurements establish neither the global best CPU competitor nor cross-machine parity. The frozen construction policy uses the same 32,768 query rows for all trained arms. MLANN has no exposed seed control; each run reuses one forest across repetitions.

**Full-base validation only.** These 256 queries come from reserved learn rows and are excluded from construction; this table does not substitute for official-test results.

| Dataset | Recall target | Before: strict recall; QPS [min–max]; config | After: strict recall; QPS [min–max]; config | RoarGraph: strict recall; QPS [min–max]; config | MLANN: strict recall; QPS [min–max]; config | After vs best measured CPU |
|---|---:|---|---|---|---|---|
| [llama-128-ip](runs/llama-full-validation-v1/report.json) | 0.90 | 0.9345703125; 894.87 [890.00–923.85]; ef=1000,rr=1000,p=0 | 0.9076953125; 6,411.97 [6,260.44–6,769.64]; ef=160,rr=200,p=8 | 0.9562890625; 6,865.81 [6,078.39–6,992.28]; M=96,L=120 | unreached (max 0.8848828125) | roargraph96: -6.61% |
| [llama-128-ip](runs/llama-full-validation-v1/report.json) | 0.95 | 0.95484375; 501.34 [497.65–516.77]; ef=2000,rr=1000,p=0 | 0.9558984375; 4,948.22 [4,703.39–5,002.71]; ef=160,rr=200,p=20 | 0.9562890625; 6,865.81 [6,078.39–6,992.28]; M=96,L=120 | unreached (max 0.8848828125) | roargraph96: -27.93% |
| [llama-128-ip](runs/llama-full-validation-v1/report.json) | 0.99 | unreached (max 0.95484375) | 0.990703125; 2,607.41 [2,393.87–2,623.30]; ef=256,rr=400,p=0 | unreached (max 0.9784375) | unreached (max 0.8848828125) | unmeasured |
| [yi-128-ip](runs/yi-full-validation-v1/report.json) | 0.90 | 0.9567578125; 1,139.31 [1,121.03–1,143.54]; ef=1000,rr=1000,p=0 | 0.9055078125; 6,941.56 [6,443.72–6,963.55]; ef=100,rr=100,p=0 | 0.919921875; 4,202.83 [3,954.59–4,215.02]; M=96,L=120 | unreached (max 0.76546875) | roargraph96: +65.16% |
| [yi-128-ip](runs/yi-full-validation-v1/report.json) | 0.95 | 0.9567578125; 1,139.31 [1,121.03–1,143.54]; ef=1000,rr=1000,p=0 | 0.97546875; 3,244.87 [3,132.42–3,282.69]; ef=256,rr=400,p=0 | 0.965; 964.14 [938.38–988.31]; M=96,L=800 | unreached (max 0.76546875) | roargraph96: +236.55% |
| [yi-128-ip](runs/yi-full-validation-v1/report.json) | 0.99 | unreached (max 0.9802734375) | 0.9908984375; 1,867.72 [1,842.02–1,868.89]; ef=512,rr=400,p=0 | unreached (max 0.965) | unreached (max 0.76546875) | unmeasured |

**In-distribution regression gate.** Both arms use the legacy 4-bit path with no training queries at frozen ef/rerank settings. The gate requires identical returned IDs and uses the three observed paired QPS ratios: all ≥0.97 pass, all <0.97 fail, mixed evidence is inconclusive. The observed paired range is not a statistical confidence interval. Glove's VIBE cosine alias preserves original angular coordinates and saved neighbor IDs.

| Dataset / original metric | Gate | Matched strict recall / config | Before QPS [min–max] | After QPS [min–max] | Median delta | Paired deltas |
|---|---|---|---|---|---:|---|
| sift-128-euclidean | inconclusive | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| glove-100-angular | inconclusive | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| yahoo-minilm-384-normalized | inconclusive | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |

**Resource observations.**

| Dataset / phase | Arm | Build seconds (one build) | Encoded index bytes | Size delta vs baseline |
|---|---|---:|---:|---:|
| [llama-128-ip / validation](runs/llama-full-validation-v1/report.json) | baseline4 | 277.851 | 128,432,748 | +0.00% |
| [llama-128-ip / validation](runs/llama-full-validation-v1/report.json) | candidate8 | 337.015 | 206,536,668 | +60.81% |
| [llama-128-ip / validation](runs/llama-full-validation-v1/report.json) | roargraph48 | 97.338 | unmeasured | unmeasured |
| [llama-128-ip / validation](runs/llama-full-validation-v1/report.json) | roargraph96 | 149.317 | unmeasured | unmeasured |
| [llama-128-ip / validation](runs/llama-full-validation-v1/report.json) | mlann_rf10_depth14 | 7.988 | unmeasured | unmeasured |
| [yi-128-ip / validation](runs/yi-full-validation-v1/report.json) | baseline4 | 117.947 | 93,802,528 | +0.00% |
| [yi-128-ip / validation](runs/yi-full-validation-v1/report.json) | candidate8 | 171.510 | 150,906,736 | +60.88% |
| [yi-128-ip / validation](runs/yi-full-validation-v1/report.json) | roargraph48 | 73.934 | unmeasured | unmeasured |
| [yi-128-ip / validation](runs/yi-full-validation-v1/report.json) | roargraph96 | 108.263 | unmeasured | unmeasured |
| [yi-128-ip / validation](runs/yi-full-validation-v1/report.json) | mlann_rf10_depth14 | 7.946 | unmeasured | unmeasured |

Build times are single observations. Main-run process RSS and USS include all co-resident arms. Separate RF80 process measurements include its input buffers and runtime; isolated index-only peak RSS and competitor serialized sizes are unmeasured.

| Dataset / phase | Process peak RSS KiB | VIBE USS delta KiB |
|---|---:|---:|
| [llama-128-ip / validation](runs/llama-full-validation-v1/report.json) | 2,357,024 | 2,021,088 |
| [yi-128-ip / validation](runs/yi-full-validation-v1/report.json) | 1,883,088 | 1,581,436 |

**Persisted formats and compatibility.**

| Format / contract | Change | Verification |
|---|---|---|
| Graph cache `HMSGRF04` | no layout change | required Rust gate covers compatibility tests; current status above |
| Query-trained Graph cache writes | rejected before touching the target, because the cache key omits training queries | rejection contract belongs to the required Rust gate |
| Encoded VGraph cache `HMSVGR01` | existing persisted layout, including trained adjacency | round-trip contracts belong to the required Rust gate; no layout version bump or new persisted fields |
| Legacy no-training path/default 4-bit codes | preserved by optional API | None/default byte-parity and binding contracts belong to the recorded validation gates |

**Verification.**

| Check | Exact command / evidence | Status |
|---|---|---|
| Formatting | `cargo fmt --check` | unrun |
| Required Rust gate | `cargo +1.89.0 test --all-features` | unrun |
| Native Python contracts | No execution receipt | unrun |
| Benchmark Python contracts | No execution receipt | unrun |
| Native benchmark wrapper | No execution receipt | unrun |
| Additional Cargo suites | Clippy, default-feature suite, feature-matrix sweeps | unrun under the requested validation budget |

**Historical target, separate protocol.** The supplied llama standing file reports VIBE epsilon-based recall and best-of-five QPS under concurrent pinned workers. These sourced values are not new paired medians and are excluded from the comparisons above.

| llama target | Historical HMS QPS | Historical RoarGraph QPS |
|---:|---:|---:|
|0.90|700.8253394103064|5174.602867638178|
|0.95|290.3855707947149|2947.0522964206903|
|0.99|unreached (max recall 0.9585)|519.6949572300871|

Source: [supplied standing JSON](../vibe_v1/llama-128-ip-standing-wl02.json); current audit and strict-ID reconstructions are in [provenance.json](provenance.json).

Regenerate this file with `uv run --no-project python benchmarks/results/query_aware_graph_v1/generate_report.py` after pulling completed run reports; add `--checks benchmarks/results/query_aware_graph_v1/checks.json` when the sole final-gate receipt exists.
