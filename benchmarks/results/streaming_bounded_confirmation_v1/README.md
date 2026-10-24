# Frozen bounded-consolidation confirmation

The selected cap250k/ef128 configuration improves consolidation and recall but
regresses query time in every heldout phase at1M and3M. Retain this experiment as
tradeoff evidence only. No production coordinator or default change is retained.

All figures below compare the selected layout against global ef256. Query ratios
are medians of the three paired ratios, each using five128-query batches.

| Prefix | Phase | Control hits /1280 | Selected hits /1280 | Query time ratio |
| --- | --- | ---: | ---: | ---: |
| 1M | Before | 1073 | 1092 | 1.823799 |
| 1M | Churn | 1092 | 1104 | 1.500101 |
| 1M | After consolidation | 1066 | 1097 | 1.844706 |
| 3M | Before | 1099 | 1148 | 4.378259 |
| 3M | Churn | 1097 | 1153 | 3.246266 |
| 3M | After consolidation | 1080 | 1155 | 4.626302 |

| Prefix | Consolidation time | Consolidation VmHWM | Reset cgroup peak | Bytes written | Mutation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1M | 0.547827 | 0.305441 | 0.548053 | 0.447621 | 0.750823 |
| 3M | 0.215057 | 0.119521 | 0.452915 | 0.171387 | 0.826801 |

These are paired median ratios; every individual ratio remains in
`confirmation-summary.json`. In particular,3M round1 mutation regressed from
0.673295101s to3.893946219s (5.783416830x). Its cause is unmeasured; the pair was
retained without retry. Later3M mutation ratios were0.305991 and0.826801.

All12 arms completed with successful Rust test and zero `/usr/bin/time` exit
receipts. Raw IDs passed liveness, duplicate, independent float64 recall, reopen
and repeated-fingerprint checks. Query IDs640..767, cap250k/ef128 and global
ef256 were frozen before these runs. All six paired rounds and all costs remain
in the record. No additional setting was selected after confirmation.

`query-overhead.md` records verified source mechanisms: every segment is searched,
stored-row scoring repeats, and unequal graph sizes invalidate the single
thread-local search scratch. Their timing shares remain unmeasured. A separate
query-major/segment-major diagnostic is prepared outside this evidence commit;
it was neither built nor run and did not alter the frozen experiment.

## Evidence and reproduction

- `protocol.md`, `selection.json`, and `selection-verification.json` preserve the
  registration and independently verified selection.
- `tuning-evidence.tar.gz` and `summary-tune.json` retain every tuning arm.
- `confirmation-evidence.tar.gz` contains all confirmation logs, raw answer
  arrays, preflight records, per-arm receipts, exact executed scripts and hashes.
- `heldout-truth.tar.gz` contains the independent live-row oracle outputs and
  their preparation receipts. Input hashes and hardware are in the accompanying
  input metadata and `machine.json`.
- `manifest.json` hashes this package. `decision.json` records completion and
  lock release. `final-gate.log` and `focused-linux.log` are historical frozen
  harness checks; `evidence-final-gate.log` checks the evidence PR's current base.

From this directory, unpack both archives into a new empty directory and run:

```sh
mkdir replay
tar -xzf heldout-truth.tar.gz -C replay
tar -xzf confirmation-evidence.tar.gz -C replay
uv run --no-project --with numpy python confirm.py \
  --evidence replay/evidence --truth replay/truth --output replay/summary.json
cmp confirmation-summary.json replay/summary.json
uv run --no-project python -m unittest test_confirm.py
```

This rerun requires no index build, official vector download, or ANN query.
The independent confirmation rerun produced the same summarySHA256
`2a9e9b5af0b3ba83c479e1f7f18bac3bcc0575c42a7452019c76d7740a433dca`.

The frozen native source is commit`06101dfd`; its production parent is`b228cae0`.
For source reconstruction, run `git apply --unidiff-zero frozen-harness.patch`
on that production parent.
`frozen-partitioned_probe.rs` independently preserves the exact test harness;
itsSHA256 is`66005c7c2c58c6900a647b97a7bb4d5b19e7781d7c445a95f36cea0120235c0b`.
The measured binarySHA256 is
`6c3401a81769323dd32796ea75ae45e9d6edccb22a7bc50af9aea3d60a10e8d2`.

These are prefix development measurements on the recorded host, using eight
threads, fresh8GiB/no-swap scopes and the shared timing lock. They establish no
full30M replay,8GiB/one-hour resource compliance, standings result, or production
readiness. Standings, full live benchmarks, Modal and paid execution remain frozen.
