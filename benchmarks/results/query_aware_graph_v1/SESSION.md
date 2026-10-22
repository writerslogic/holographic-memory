# Resume state — 2026-10-22

## Objective and workspace

Finish the original query-aware Rust search experiment and deliver one signed, unmerged PR to main; draft unless targets/checks pass. Preserve the user's original detailed task constraints. Work only in `/Volumes/A/.sol-wt-qag`, branch `feat/query-aware-graph`; original `/Volumes/A/holographic-memory` has unrelated changes. Read original `CLAUDE.md` there first. Baseline is `08d37778a66b58709461456e5b9a7ab6a05b399f`; do not recreate/reset the worktree.

Signed source commits: projection/API85b0af0, integration3f285ce, patience API f73305b624305836211c8157975c4f1c37f10df9, frozen controllers e0c8096f7cad6c297eafdacc63b13057daf7852c, RF80/smoke836615c39911df5bb94fe0b039a3a96f318659d8. No push/PR yet. Signing works with `git -c user.email=david@writerslogic.com commit -S -m '<type>: <Imperative description>'`.

Canonical evidence: this directory's ledger.md, provenance.json, README.md, raw JSON/HDF5, sources/, logs/, configs/, runs/. `generate_report.py` regenerates README from completed reports and optional checks.json. Detailed original task instructions remain in conversation; ledger records settled experimental decisions.

## Running work: verify before doing anything

Remote task root `writerslogic-02:~/sol-qag`. Do not start overlapping jobs/builds or mutate frozen remote runtime/helper sources.

- Main batch outer flock PID400113, driver400114; local tool session7442 (may not survive thread switch). Last verified running ~10:14 Pacific: official llama worker404750 building. Log `~/sol-qag/logs/full-evaluation-v1.log`.
- Batch `~/sol-qag/full-evaluation-v1.sh` already completed full llama/yi validation; next official llama, official yi, all three default4 regressions, corrected prefix v2 report. It holds `~/hms-timing.lock` continuously.
- RF80 comparator queued behind it: outer flock PID404709, local session67691. Script `~/sol-qag/mlann-rf80-v1.sh`, log `~/sol-qag/logs/mlann-rf80-v1.log`. Only builds RF80/depth13 MLANN, two official datasets, nine fixed query groups each. Local immutable pre-test declaration `benchmarks/vibe/mlann_pair/frozen.json`.
- Monitor bounded log tails/PIDs. If processes ended, inspect completion/failure receipts. Do NOT rerun whole existing batches: controllers refuse existing directories. Resume only missing phases with their explicit existing freeze/script commands; preserve previous results. Never kill unrelated lock holders.

## Verified results and closed budget

All3 HMS interventions ×6 small configurations are exhausted. No further HMS configurations/interventions without renewed authorization. Precision, projection, stale-expansion stopping are implemented; defaults/None path preserved. Projection uses optional original-coordinate training queries through Python, first32768 rows, forward/reverse quotas32, all base edges retained; default codes4, opt-in8, residual8. Patience defaults0.

Full validation reports have been pulled under `runs/{llama,yi}-full-validation-v1/` with frozen.json/raw/receipts. All256 validation queries are reserved learn rows, disjoint from32k construction; never official test. Config set frozen before official evaluation.

| Dataset / target | Actual recall | Candidate QPS median | Config |
|---|---:|---:|---|
|llama .90|.9076953125|6412.0|ef160/rr200/p8|
|llama .95|.9558984375|4948.2|ef160/rr200/p20|
|llama .99|.990703125|2607.4|ef256/rr400/p0|
|yi .90|.9055078125|6941.6|ef100/rr100/p0|
|yi .95|.97546875|3244.9|ef256/rr400/p0|
|yi .99|.9908984375|1867.7|ef512/rr400/p0|

Llama .95 misses: qualifying Roar96/L120 .9562890625 at6865.8 QPS (candidate72.1%); gap exceeds observed ranges. Llama .90 passes93.4%. These are validation, not official claims. Yi early stopping generalizes poorly. Required full official accuracy/QPS and regressions remain pending. RF10/depth14 MLANN saturates below.90; RF80 was declared before official results to strengthen comparator coverage.

B/O/V/E/OE full diagnostic tables are already in README. Original architecture is scalar mean-centered codes, no production FWHT or exact-f32 rerank. Oracle entry adds little. Existing projected-prefix ef100 ablation B=.9159375,V=.9566796875,E=.9577734375 motivates selective admission refinement as a future hypothesis; exact-routing QPS is unmeasured. User asked whether this is the best possible result: answered no, budget establishes no architectural ceiling; compact adjacency is another unproven next experiment (capacity128, mean64.4). Do not promise either design's result.

## Artifacts and remaining actions

Remote SOs: `artifacts/baseline-holographic_memory.so` SHA91beaaf58f5bced0060e6337052edad5aa1dd8c6733fece137b927c9d0788b7e; `artifacts/patience-holographic_memory.so` SHA31988dc57c8998e12f2daabb8e6b77086775d8532d60f475ac12095d7d2d0a26. Both Rust1.96/native/featurepython. `trained-holographic_memory.so` is older projection-only source. Untimed patience diagnostics accidentally built with stable1.99, recorded; no1.99 candidate timings accepted. Never use mutable stable for comparison builds.

1. Let frozen main and RF80 jobs finish. Pull completed `runs/{llama,yi}-official-v1/` excluding evaluation.hdf5 and validation-export.hdf5; include raw.hdf5/logs/JSON, plus corresponding `freezes/<id>/frozen.json` into each canonical run. Pull regression-runs/default4-v1 into canonical `regression/default4-v1/`; pull competitors/mlann-rf80 into canonical same path. Pull corrected prefix summary-v2 and keep original receipt/raw. README generator validates scope/source/harness/pool/affinity before RF80 merge. Do not edit frozen helpers to resolve report-only issues until all jobs finish.
2. After timings, run new real-native `benchmarks/vibe/qag_hms_smoke.py` on Linux via uv, with current hms/module.py copied to a separate task-local wrapper path, new SO and optional baseline SO. It is committed but native execution is unrun. Already verified: native Python38 tests, local benchmark52 tests, preparation43 tests, combined integration/measurement patches apply to VIBE7ac76c1. Logs canonical.
3. Finish all code/experiments before final gate. **No cargo check/test/clippy has run.** Run cargo fmt --check, then **exactly once** on Linux `cargo +1.89.0 test --all-features`; no retry, no extra suites. Rust1.89 is installed but rustfmt component is absent; install it under lock if choosing1.89 formatting. Stable is1.99.1. Pin explicit toolchain, use task target dir, no cargo clean. CARGO_BUILD_JOBS=2 is sensible for27GiB RAM. Ensure remote repo source matches committed Rust/Cargo files before gate. On failure record exact error and leave draft; later fixes remain unverified.
4. Record actual check receipts in checks.json (generator supports fmt/cargo/native_python/benchmark_python/wrapper_native with command/status/exit_code/log, cargo attempts1). Regenerate README, update ledger/provenance, signed commit/push, create ONE PR to main, never merge. GitHub connector create_pull_request supports repository_full_name/base/head/title/body/draft; gh auth/signing verified. Do not push add-hms or alter VIBE#15.

Timing contract: SSH Linux only, flock, bench.sh affinity, all5 thread env vars1, saved fifth pass ×3 alternating blocks, median[min–max], actual strict recall≥target, no interpolation/rounded claims. Single-build resources and aggregate co-resident RSS explicitly scoped. Format unchanged HMSGRF04/HMSVGR01; trained Graph cache writes reject before touching file, encode/save VGraph instead. Rust compatibility tests remain unrun until final gate. Raw nested HDF5 uses recursive LFS attribute; force-add intended ignored *.log receipts. No performance impossibility claims.
