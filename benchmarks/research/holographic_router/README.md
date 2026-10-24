<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Synthetic phase-summary routing probes

The two frozen probes test block routing from additive phase summaries, then a fixed-budget phase/witness combination. Both are rejected under their preregistered rules. See the [evidence and reproduction commands](../../results/holographic_router_synthetic_v1/README.md).

`probe.py`, both plans, and original correctness tests are preserved byte-for-byte from their pre-run seals. The active `hybrid_probe.py` removes one unused import after review; [frozen-hybrid-probe.py.gz](frozen-hybrid-probe.py.gz) preserves its exact original bytes. `recorded_source.py` accepts that archive only when its hash matches the recorded source and the active script differs solely by the documented import removal. The independent auditors regenerate truth and metrics; `summarize.py` retains every registered configuration and negative mixed/rare comparison. These are bounded synthetic research tools, not a production router or durable format.
