# Consolidation scratch allocation experiment

Decision: reject centered-row scratch reuse (`279fe809`) under the registered
240k-row gate. Median full consolidation time was 0.9980616x, above the required
strict 0.98x. Two of three pairs were faster; isolated RSS passed, and every
published file matched its paired control. The native change is preserved only
as a patch. The default implementation stays unchanged; the conditional 1M scale
check was not run.

| Round / order | Consolidation ratio | Isolated RSS ratio | Reset cgroup peak ratio | Written bytes ratio | Files identical |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 / parent,candidate | 1.0433504 | 0.9992402 | 0.9974080 | 1.0 | yes |
| 2 / candidate,parent | 0.9970978 | 0.9987594 | 1.0004972 | 1.0 | yes |
| 3 / parent,candidate | 0.9980616 | 1.0001519 | 1.0000160 | 1.0 | yes |
| Median | 0.9980616 | 0.9992402 | 1.0000160 | 1.0 | yes |

Ratios are candidate/control; all losses remain. These are development
consolidation measurements, not mixed-workload throughput, official-data recall,
30M resource compliance or standings evidence.

`summary.json` records source/binary hashes, raw per-arm statistics and the frozen
retention checks. `raw/` contains all six receipts, immutable-fixture/output
hashes, build logs, platform details and the parent allocation profile. The
counting allocator attributed 423176 allocations and 214.2 MiB allocated bytes to
encoding; its timing is instrumentation evidence only. Lower allocation counts
from earlier diagnosis did not establish a full-consolidation speedup.

The fixture has 240000 deterministic synthetic 100D Euclidean rows: 200000 initial
base rows, deletion of every seventh base ID, then 40000 inserted tail rows,
leaving 211428 live rows. Both arms use degree 32/build-ef 64, eight Rayon threads on
CPUs 0–7, a fresh copied fixture and 4 GiB/no-swap scopes. Each pair holds the shared
timing lock below 14 minutes; builds use private targets, CPUs 18–19, jobs 2/nice 19.
The frozen rule requires median consolidation ratio<0.98, at least 2/3 faster
pairs, median isolated RSS ratio<=1.05 and identical published files.

To reconstruct source, start from production parent`ac235724`, apply
`measurement-harness.patch` with `git apply --unidiff-zero` to both arms, then `scratch-candidate.patch` only to
the candidate. These correspond to comparison parent`6fb34c40` and candidate
`279fe809`; `source-verification.json` confirms the sealed 576-file source archives
differ only in`src/core/qgraph/vertex/file.rs`. `build-pair.sh`, `locked-step.sh`
and `paired.sh` preserve the executed setup/timing commands. The build script's
existing cached dependency target is optional setup; a clean private target can
be built from the same locked sources. Timing uses the ordinary allocator in
`qgraph_consolidation`; the profiler creates the fixture and supplies separate
phase attribution.

Recompute the decision and exercise its boundary/failure checks with:

```sh
uv run --no-project python summarize.py raw
uv run --no-project python -m unittest -v test_summarize.py
```
