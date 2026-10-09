# Continuation state (written 2026-10-05, for the next session)

Durable copy of the hand-off. Delete this file once the items below are merged or resolved.

## State (updated 2026-10-06)

| Workstream | State |
|---|---|
| Quantized-graph index | `public-bench ann-qgraph --holdout N` tunes on held-out train vectors (test set unread). nytimes held-out grid in `benchmarks/results/public_qgraph_tuning_nytimes-256-angular.json`: degree 64 / 512-bit codes wins at both targets (7,715 / 2,087 QPS at 0.90 / 0.95 on held-out queries) with 1.77 GB against 2.95 GB for the old 1,024-bit setting. Alpha and build_ef were not swept. Running unattended: `/Volumes/A/.hms-target/logs/qgraph_pipeline.sh` (log `qgraph_pipeline.log`; tuning JSONs in `logs/tune/glove`, finals in `logs/final`) = glove held-out grid (d64 b128/256/512, d32 b256), auto-pick, then final HMS runs (repeats 3, load < 3) and `evaluate.py ann ... --repeats 3` for both sets, writing `benchmarks/results/public_qgraph_<set>.json`. The machine is shared with other sessions' builds (load 20-40 observed), so runs wait up to 6 h for the gate; check `load_gate_met` on every row before using a number. A first glove tuning attempt ran entirely at load ~38 and was discarded (`logs/tune/glove_unmet`). Next: when the pipeline finishes, merge glove tuning with `logs/merge_tuning.py`, rewrite the qgraph section of `docs/PUBLIC-BENCHMARKS.md` (drop the contamination caveat, state alpha/build_ef untuned, new memory ratio, glove same-session numbers), commit. |
| Local model stages | Every HMS stage runs on L4 (candle 0.11 CUDA kernels do not build for T4). The `s-dev-hms` run was stopped at 18:00 PDT 2026-10-06 after Modal billed $14.40 against the $8 cap (driver preemptions cancelled billed-but-unledgered L4 shards); 1,500 of 4,506 fact sessions done, no retrieval metrics. Budget now reserves wave cost before launch (simulated preemption verified). LLM-stage quality on those 1,500 sessions: 74% identical fact output, 96% of facts matched (`benchmarks/results/longmemeval_dev_hms_stages.json`). At ~17x vLLM's per-session time, a full HMS S-dev run costs roughly $8 GPU plus driver time; do not relaunch without the maintainer's approval and a new cap. Batched GPU decoding in HMS is the fix that makes it affordable. The `s-dev-hms` ledger on the `hms-lme` volume is set to the billed $14.40. |
| Holographic memory | Done for dev: 3 seeds and the equal-bytes control are in `holographic_dev.json` and the doc. Shards beat an equal-bytes index only beyond ~30% loss; no variant beats the index on accuracy; no Rust port. |

## State (2026-10-07): everything is on `main`, pushed (58d1ff0; origin/main is in sync)

There are no other branches, worktrees or stashes. Merged into `main` with signed merge commits:
the build-time speedups, the per-vertex 8-bit LVQ index `VGraph`, batched Metal model stages,
build-bounds screening (ccb4d8a) and the opt-in 4-bit search encoding (fcf0056). The 16 qgraph
unit tests pass and the release binary builds. GitHub CI is GREEN on 0f169e3 (format, clippy 1.99, tests, MSRV 1.89, deny, retrieval-quality, napi on three platforms, CodeQL, coverage). CI's clippy runs stable, currently 1.99.0, which has lints the local `cargo +1.98.0 clippy` gate does not: run `cargo +1.99.0 clippy --all-targets --all-features -- -D warnings` locally before pushing (1.99.0 is installed).

Recorded as merged WITHOUT their code (`-s ours`; reachable only as second parents of these
merge commits, so `git show <tip>` and `git diff main <tip>` still work):
- 18e541d, memory-first qgraph: integer vector stores and per-vertex codes, layout screening,
  a timed glove run of an i8 vertex-code index. Superseded by VGraph; its results files
  (`benchmarks/results/qgraph_memory_first_*.json`) exist only there.
- 1993058, edge-index speed: FastScan edge kernel, batched pool merge, 2-bit edge codes scanned
  in one NEON pass; results in `qgraph_speed_first_heldout.json`, only there. Targets the
  per-edge QGraph; the FastScan kernel and pool merge may be worth porting to VGraph.
Four raw pareto logs that were never committed are in
`/Volumes/A/.hms-target/logs/worktree_ignored/wf-4_pareto_raw/`. The scratch target dirs
`/Volumes/A/.hms-target-*` were left in place (`.hms-target-vsearch/bin/pb-base` is the baseline
binary the next session uses).

Status of the claims (verify before quoting; results files are in `benchmarks/results/`):
- Build bounds: nytimes build ~1.9x faster with a bit-identical graph; slower on glove, so on
  only at dim >= 128 (untested between 100 and 256 dims).
- 4-bit encoding (`--residual --vertex-bits 4 --residual-bits 8 --align-rows --reorder
  --id-bytes 3`, rerank 32 nytimes / 64 glove): memory -24.1% nytimes, -16.5% glove (certain).
  Speed UNCONFIRMED: paired ratios 0.79-1.98 across runs at load 39-270; the default path now
  runs the new pool/kernel code at x0.94 / x1.06 (inconclusive; possible regression).
- `--graph-cache` is keyed by filename only; must be fixed before any published run.

## Prompt 1 progress (2026-10-07, Fable session)

Step 1 DONE (`benchmarks/results/qgraph_merge_verification_heldout.json`, raw logs in
`/Volumes/A/.hms-target/logs/merge_verify/`):
- The graph cache is now format `HMSGRF02`: its header records n, dim, degree, build_ef, alpha,
  seed and a fingerprint of the vectors, and `Graph::load` rejects any mismatch (test
  `graph_cache_round_trips_and_rejects_mismatches`). Old `HMSGRF01` files are rejected (bad
  magic); delete them to rebuild. `ann-qgraph` output now records `build.graph_hash` (stable FNV
  over the structure). Checked caches for the held-out graphs:
  `/Volumes/A/.hms-target/logs/merge_verify/cache/{nytimes-256-angular_d64_b200,glove-100-angular_d32_b200}_v2.graph`.
- (a) The new default search path beats the old one on both sets at both targets in 11 of 11
  paired rounds: nytimes x1.30 (0.90) / x1.38 (0.95), glove x1.48 / x1.58. KEEP the new default
  path; nothing is reverted.
- (b) The recommended encoding (`--vertex-bits 4 --residual-bits 8 --align-rows --reorder
  --id-bytes 3`) wins on nytimes (x1.20 / x1.18, 11 of 11) but not on glove (x1.01 at 0.90, 6 of 11;
  x0.97 at 0.95, 3 of 11), so step 4 uses the DEFAULT encoding (`--residual`, 8-bit codes) on both
  sets and the recommended encoding is reported as a memory-only gain (-24.1% / -16.5%).
- Load was 12-29 throughout (other sessions); paired ratios are the result, absolute QPS are not.

Step 2 DONE: `benchmarks/public/evaluate.py ann ... --extra <system>...` runs each extra
competitor in its own process through `benchmarks/public/ann_extra.py` (faiss-cpu and rabitqlib
each bundle an OpenMP runtime; a rabitqlib build segfaults once FAISS is imported in the same
process). Extras: `symphonyqg` (rabitqlib.SymqgIndex, the SymphonyQG authors' library, NEON),
`rabitq` (rabitqlib IVF 1+4 / 1+8 and HNSW 1+4), `ngt` (NGT-onng via ngtpy and the Homebrew `ngt`
command; NGT-qg needs `qbg`, absent from the macOS build), `lucene_hnsw` (lucene-core 9.12.3,
`benchmarks/public/lucene_hnsw/LuceneHnsw.java`, Java 17); `glass`, `diskann`, `scann` are
recorded as "not run on M4" with the reason (x86-only builds). Extra packages come from the
command line: `uv run --with ngt --with 'rabitqlib>=0.5.2' --script benchmarks/public/evaluate.py
ann <set> <hms.json>... --extra ...`. `--no-builtin`, `--queries N` and `--train-rows N` are
smoke-test switches (not publishable); `--extra-timeout-secs` (default 5400) turns a system that
has not built and run in time into "not run on M4". `evaluate.py ann-merge <set> <part.json>...
--out public_qgraph_<set>.json` joins per-system parts so each holds the timing lock briefly.
`benchmarks/public/qgraph_table.py <set>` prints the doc table from the merged file.

Step 3 DONE: the full gate ran once after the step-1 Rust changes (log
`/Volumes/A/.hms-target/logs/merge_verify/gate.log`): fmt, clippy 1.99 (both feature sets), tests
(both), 1.89 check, deny: all rc 0; `uvx ruff check benchmarks/public` clean (it also fixed four
pre-existing lints in `batching_*.py`). The harness is Python only, so no Rust rerun was needed.

Step 4 PARTIAL, stopped on the maintainer's instruction so that the whole comparison is rerun
once after the parallel session's search changes (patience stop, 1-bit screen) land: nytimes is
done and merged into `benchmarks/results/public_qgraph_nytimes-256-angular.json` (binary of
commit d985308; every row above the load gate, load 5-103; HMS at 0.95 not bracketed because the
ef sweep stopped at 512 with recall 0.948; NGT, Glass, DiskANN, ScaNN recorded as not run on M4).
glove-100-angular was NOT run. Parts and logs: `/Volumes/A/.hms-target/logs/merge_verify/test/`
(`<set>_part_*.json`, `<set>_hms_{default,optin}.json`, `step4.log`). Full-train graphs are in the
checked caches `/Volumes/A/.hms-target/logs/merge_verify/cache/*_full_*_v2.graph` (valid for
degree 64 / 32, build_ef 200, alpha 1.0, seed 0x5EED; any other build parameters rebuild).
The old edge-index results are `benchmarks/results/public_qgraph_edge_<set>.json`.

Rerun recipe (both sets; PREPARED 2026-10-07 22:00 and waiting for the maintainer's go: the
search changes are committed at e517080, the pinned binary
`/Volumes/A/.hms-target-step4/release/public-bench` is built from e517080, and `run_step4b.sh`
sweeps `--patience 0,256` on nytimes and `0,384` on glove beside fixed ef, no `--screen`, which
measured negative in `qgraph_stop_screen_heldout.json`; `evaluate.py` splits a sweep file into one
series per patience value). For any later commit: build the binary of the commit
under test outside the shared target dir (`git worktree add /Volumes/A/.hms-wt-step4 <commit>`,
`CARGO_TARGET_DIR=/Volumes/A/.hms-target-step4 cargo build --release --bin public-bench`; the
shared `/Volumes/A/.hms-target/release/public-bench` is rebuilt by whichever session builds last),
set `B=` in `/Volumes/A/.hms-target/logs/merge_verify/test/run_step4b.sh` to it, then
`run_step4b.sh nytimes-256-angular all` and `run_step4b.sh glove-100-angular all` (ef up to 1,024
/ 1,536, 60-s load wait, NGT capped at 30 min, HMS sweeps embedded at merge time via
`--hms-files`), then `uv run --script benchmarks/public/qgraph_table.py <set>` and rewrite the
tables in `docs/PUBLIC-BENCHMARKS.md` from the files. Run it when the machine is as quiet as it
gets; every row's load is recorded and the doc must say when the gate was unmet.

Step 5 PARTIAL: the "Quantized graph index" section of `docs/PUBLIC-BENCHMARKS.md` carries the
step-1 verification and the preliminary nytimes table with its caveats; README unchanged (no
gate-met number to cite). Step 6 DONE (pushed). Step 7 DONE: `docs/NEXT-SESSIONS.md` holds the
specs; its (A) "systems that beat HMS" list is provisional (nytimes only) and is re-derived after
the rerun.

Next: (1) the parallel session commits and times its changes; (2) rerun step 4 as above; (3)
rewrite both tables and the README vector-search row from the files; (4) then prompts 2 and 3.

## Ideas-tree session (2026-10-07, Fable; prompt "a step, not a percent")

Artefact: `docs/research/IDEAS-2026-10.md` (axioms, relaxations, mappings, counterfactuals, scored
tree, checkpoints 1 and 2). Proxy code `benchmarks/public/proxies/`, proxy results
`benchmarks/results/proxies_2026-10*.json`. Done and past their kill tests (held-out / dev only):
- E1 `VSearchParams::patience` (`--patience`, swept): 17-29% fewer evaluations at equal recall
  on both sets.
- E2 1-bit screen (`VGraph::from_graph_with(.., screen)`, `--screen --screen-sigmas`): nytimes
  76% of 8-bit estimates skipped at +-0.001 recall, +5% index bytes; glove 55% (marginal).
- E3 document API: Porter-stemmed BM25 (per-chunk `stemmed` flag keeps old stores matching) and
  min-max score blend (`fusion: "rrf"` keeps the old behaviour). BEIR test re-measured once:
  hybrid nDCG@10 0.683 -> 0.729 SciFact, 0.344 -> 0.355 NFCorpus (`public_scifact.json`,
  `public_nfcorpus.json` regenerated; doc table rewritten).
- E4 qgraph builds and searches return `Result<_, HmsError>` naming the cause; two property
  tests added; 19 qgraph tests pass.
Pruned with reasons in the doc: entry point, alpha/build_ef, 8-bit code variants, exact
re-rank, relative-slack stop, stop list, valid-time extraction at ingest. LongMemEval
per-session quota passed its dev proxy by one question but FAILS on the tuned pipeline's cached
dev scores (`benchmarks/results/longmemeval_dev_quota.json`, run dirs downloaded to
`~/.cache/hms-bench/longmemeval_s/runs/`): not adopted.

Paired timing DONE (`benchmarks/results/qgraph_stop_screen_heldout.json`): E1 patience x1.33 /
x1.42 at 0.95 (nytimes 4/5, glove 4/4 rounds), neutral at 0.90; E2 screen x0.65-0.87, a negative
result (scalar screen costs as much as the SDOT estimate), code stays opt-in and off. Next
test-set comparison (step 4 rerun) should add HMS rows with `--patience 256` (nytimes) /
`--patience 384` (glove) and, on nytimes, the 4-bit encoding; see IDEAS doc checkpoint 3.
Earlier note on the in-flight run: paired timing of E1/E2 (8-bit and 4-bit encodings, coordinator
`/Volumes/A/.hms-target/logs/merge_verify/paired.py`, tags `e12ny` 5 rounds and `e12gl` 4
rounds, outputs `e12{ny,gl}_{E,D}.json`; an 11-round attempt hit the 14-minute deadline before
the processes wrote their files, so rounds were cut). Report:
`uv run python benchmarks/public/proxies/stop_screen_report.py /Volumes/A/.hms-target/proxies/e1
/Volumes/A/.hms-target/logs/merge_verify benchmarks/results/qgraph_stop_screen_heldout.json`.
The gate ran once (`/Volumes/A/.hms-target/proxies/gate.log`); clippy failed on one complex
type (fixed with an alias) and is being rerun with the qgraph tests
(`/Volumes/A/.hms-target/proxies/gate2.log`). Then: results file, doc section "Quantized graph
index" gains a paired-ratio paragraph for E1/E2, changelog, signed commits (stage only this
session's hunks; the other session shares the tree), push.

Library defaults are unchanged (patience 0, no screen, from_graph as before); the recommended
bench configuration is decided by the paired ratios and written in the doc.

## Acceptance criteria (binding)
- Vector index: single-thread QPS at recall@10 = 0.90 and 0.95 on glove-100-angular and nytimes-256-angular versus
  FAISS HNSW and hnswlib re-timed in the same session on an idle machine (1-min load < 3), medians of 3 runs. Report
  the result either way.
- Model stages: dev parity with the tuned pipeline (`benchmarks/results/longmemeval_dev_final.json`) within ~0.004 for
  embedder/re-ranker; LLM stages re-measured; on-device ingest/query/memory numbers recorded.
- Holographic memory: dev-only; answer accuracy per type vs retrieval top-1 with the same judge; capacity curve;
  bit-corruption curve vs the chunk index; shard-loss curve vs the chunk index; hierarchical (user -> session -> fact)
  recall; compositional temporal/multi-session accuracy. A negative result is reported as such.

## Then, in order
1. Merge what passes; regenerate `CHANGELOG.md` with `git cliff -c cliff.toml -o CHANGELOG.md`, commit as
   `chore: update changelog`, push.
2. With HMS stages in place: smoke-test the large models on S dev on Modal (cap $8), then ask the maintainer before the
   single LongMemEval_M run:
   `uvx modal run benchmarks/public/longmemeval_modal.py --dataset m --part all --models large --cap 40 --tag m-all-large --out benchmarks/results/longmemeval_m_large.json`
   (also run `--models small` as the on-device headline). Score held-out and full set separately; report per type,
   with bootstrap CIs; put the paper's Table 3 numbers beside ours, labelled.
3. Add the end-to-end QA score (local LLM reader + official judge) so the headline matches the paper's.
4. Release 0.6.2 / SDK 0.1.3 (ordering fix, README logo) only with explicit approval.

## Standing rules
- Held-out LongMemEval ids (`benchmarks/public/longmemeval_split.json`) are never used for tuning.
- No claim without a measurement in `benchmarks/results/`; label synthetic data; state load averages for timings.
- No push, tag, publish, yank, or Modal spend beyond stated caps without the maintainer's explicit instruction. Check `modal billing report` after any run; the ledger is an estimate.
- Credentials: the PyPI and Gemini credentials read by a prior agent still need rotation by the maintainer.

## Superiority session 1 startup (2026-10-09)

Next: finish Item 1's frozen 100-dev reader/judge report and independent checker.
Then: commit Item 1 before starting Item 2's matched native harness.
Then: Item 3 learned termination; Item 4 is next if Item 3's kill test fails.

Branch `research/2026-10-superiority`, base `a46108c`; existing work preserved.
Rust 1.96.0, uv 0.12.10, Python 3.13.9, NumPy 2.2.6; all requested inputs present.
Startup load 2.74, timing lock free; `/Volumes/A` has 6.0 TiB free.
Startup/preregistration/spend: `benchmarks/results/{startup,preregistration,spend}_2026-10.json`.
Spend $0; scoreboard created with empty matched cells and separate vendor-native values.
Reader producer, evidence preparation and independent checker are being built; no calls yet.

Item 1 protocol frozen: `reader_protocol_v1.json`, SHA256 `66a5d33412117c4ef5c6d0a89b0a26d0194103dc515d5a891a2593b2afcc566d`. Complete-source reconstruction reproduces all 300 packet hashes; six actual-source tests pass. GPT snapshot available; weak Qwen3-4B weights/revision present. Preparation counts complete inputs for both tokenizers and checks API framing. Calls begin only after preparation. Logs: `target/superiority-validation/{prepare,gates}.log`; spend $0. Vendor conditions differ, so matched margins remain empty.

Checkpoint (2026-10-09 17:13 UTC): all five Rust gates pass (`target/superiority-validation/gates.json`). Prepared all 100 dev questions and 600 reader-arm records; independent partial checker validates all source occurrences, full-input counts and shared cap. Local weak reader is running under the frozen batch-1 protocol (`weak-reader-run.log`, outputs `weak_reader_outputs.jsonl`). GPT generation rejected HTTP429; probe confirms `credit_balance_exhausted` / `insufficient_quota` (`benchmarks/results/reader_api_blocker_2026-10.json`). User received official API billing link. No GPT answers or accepted scoreboard rows; reserve $21.79889 for rejected calls is conservative accounting, not observed paid usage. D1 remains open; Item2 has not started.

Checkpoint checker: `benchmarks/public/check_superiority_checkpoint.py`; verified artifact `benchmarks/results/checkpoint_2026-10_check.json`. Answer report is explicitly incomplete and unaccepted. Source-operand prerequisite independently rechecked: five gates pass, report hash `4e8f5346d10bb187ac11378bfe677e31cc9a96763079b80a9d4da3953ea8178e`. Startup record has its own later load snapshot; 2.74 above was the initial shell probe, not a timing gate. The weak manifest references the prepared-file hash before an artifact-only fingerprint refresh; all frozen reader strings and protocol hashes are unchanged.

17:24 UTC: checkpoint `9fa7893` pushed. User replenished $100 and supplied a replacement API key; stored privately in `~/.env` (mode600) via no-echo input. GPT calls now succeed with unchanged prompts. Retry runner `resume_longmemeval_readers.py --attempt 4` is running (`readers-new-credential.log`); all prior failed requests remain recorded. Local reader completed its first ten distinct prompts, serving 26 arm records; continuation is running. No D1 score accepted yet; no Item2 work started.

### Session 1 checkpoint 2026-10-09T17:54:23.980731+00:00

1. Finish the frozen dev reader, judge and 400 repeat calls; pass the independent checker and commit the dev memory rows.
2. Run and commit the native FAISS/hnswlib harness on both datasets and recall targets.
3. Run learned early termination's preregistered kill test.

Matched primary answers are complete: first-five 83/100, anchored knapsack 82/100, operand 87/100; mechanical supported claims are 187/192, 180/186, 201/204. These are raw dev results pending the final checker. 197/400 repeats and 72/300 local arm records are recorded. Frozen prompts and the base producer are unchanged. Incremental weak judging will consume completed local chunks after the repeat writer exits.

Results: `benchmarks/results/longmemeval_dev_answers_v1.json`, `benchmarks/results/reader_checkpoint_2026-10-09T1754.json`, `benchmarks/results/spend_checkpoint_2026-10-09T1754.json`. Usage-supported cost $5.415003; conservative unverified reservations $22.021348; ledger exposure $27.436351 under the $120 item cap. Five Rust commit gates and seven checker tests pass; three incremental stream-boundary tests pass. Running root writer session 21658; local runner session 61378 and telemetry-label watcher 7920. Scoreboard diff: no accepted rows yet. D1, D2 and D3 remain open.

### Session 1 checkpoint 2026-10-09T18:24:42.365948+00:00

1. Finish the local dev reader and incremental judges, pass both independent checkers with tamper tests, and commit the dev memory rows.
2. Run and commit the native FAISS/hnswlib harness on both datasets and recall targets.
3. Run learned early termination's preregistered kill test.

Matched dev correctness is 83/100 first-five, 82/100 knapsack and 87/100 operand; the final answer checker remains pending. The fixed study completed 400 repeat judge calls: 12 disagreements (3%), two unstable questions (10%), zero failures. The local reader has 100 arm records; two fail the frozen JSON contract and count as incorrect. All 98 valid local records have judge outcomes. Its durable model/weight/chunk/token-count bundle passes the separate independent partial checker and eight tamper cases; this establishes neither D1 nor an accepted scoreboard row.

Results: `benchmarks/results/longmemeval_dev_answers_v1.json`, `benchmarks/results/reader_checkpoint_2026-10-09T1824.json`, `benchmarks/results/spend_checkpoint_2026-10-09T1824.json`, `benchmarks/results/weak_reader_evidence_v1.json` and its `weak_reader_evidence_v1/` sidecars. Usage-supported cost $5.492053; conservative reservations $22.021348; ledger exposure $27.513401. Five Rust commit gates pass. Previous checkpoint e785732 is pushed. Running: incremental judge writer 31363, local runner 61378, telemetry watcher 7920, bundle updater 3004. Frozen prompts and generation sources are unchanged. Scoreboard diff: none; D1, D2 and D3 remain open.

### Session 1 checkpoint 2026-10-09T19:00:27.584849+00:00

1. Finish the frozen local reader and incremental judges; pass both independent checkers with tamper tests and commit the dev memory rows.
2. Run and commit the native FAISS/hnswlib harness on both datasets and recall targets.
3. Run learned early termination's preregistered kill test.

Matched dev accuracy remains 83/100 first-five, 82/100 knapsack and 87/100 operand; 400 judge repeats give 3% disagreement. The staged snapshot contains 167/300 local records, including two JSON-contract failures counted incorrect; 147 valid local records have judge outcomes. The GPU profile is preliminary: it is saturated, KV caching works, and an isolated Metal SDPA candidate is being prepared without changing the frozen run. ANN work has not started; relaxing the item order is awaiting the user's answer.

Results: `benchmarks/results/reader_checkpoint_2026-10-09T1900.json`, `spend_checkpoint_2026-10-09T1900.json`, the staged answer report and weak evidence bundle. Usage-supported cost $5.543031; conservative reservations $22.021348; exposure $27.564378. Five Rust commit gates pass. Previous checkpoint 573ee71 is pushed. Running: weak reader 61378, incremental judges 31363, evidence updater 3004. Scoreboard diff: none; D1, D2 and D3 remain open.

### Session 1 checkpoint 2026-10-09T20:04:41.238076+00:00

1. Finish the frozen local reader and incremental judges; pass both independent checkers with tamper tests and commit the dev memory rows.
2. Run and commit the native FAISS/hnswlib harness on both datasets and recall targets.
3. Run learned early termination's preregistered kill test.

The staged 19:56 UTC snapshot has 252/300 local records: 250 valid, two JSON failures counted incorrect, and 250 valid judge outcomes. Matched dev accuracy remains 83/100, 82/100 and 87/100; 400 repeat judges give 3% disagreement. The independent partial answer checker validates 552 completed reader/judge records; the 11-chunk local bundle passes eight tamper cases. Final acceptance remains pending. Results: `benchmarks/results/reader_checkpoint_2026-10-09T1956.json`, `answer_partial_check_checkpoint5_2026-10-09.json`, `weak_reader_evidence_v1_partial_check_checkpoint5_2026-10-09.json`.

Two operational speed pilots produced no completed native predictions: the load gate blocked one, and the unchanged control exceeded the other's 120-second cap. No SDPA quality or speed result exists. Exact draft sources and attempts are archived in `benchmarks/results/metal_sdpa_draft_2026-10-09/`; the independent suspension/archive checker rejects seven corruptions. All elapsed values are preliminary. The frozen reader continues unchanged; ANN work has not started.

Usage-supported cost $5.616808; conservative reservations $22.021348; exposure $27.638156 (`benchmarks/results/spend_checkpoint_2026-10-09T1956.json`). All five Rust commit gates pass (checkpoint5); source fingerprints remain unchanged. Previous checkpoint 72b3ea2 is pushed. Running: local reader61378, incremental judges31363, evidence updater3004. Scoreboard diff: none; D1, D2 and D3 remain open.

### Session 1 checkpoint 2026-10-09T20:36:05.201056+00:00

1. Finish the final six frozen local prompts and incremental judges; pass both full independent checkers with tamper tests and commit the dev memory rows.
2. Run and commit the native FAISS/hnswlib harness on both datasets and recall targets.
3. Run learned early termination's preregistered kill test.

The staged snapshot has 294/300 local records: 290 valid, four malformed answers counted incorrect, and 290 valid judge outcomes. The final six distinct prompts are running. Matched dev accuracy remains 83/100, 82/100 and 87/100; 400 repeat judges give 3% disagreement. The partial answer checker validates 594 completed records; the 14-chunk local bundle passes eight tamper cases. Results: `benchmarks/results/reader_checkpoint_2026-10-09T2034.json`, `answer_partial_check_checkpoint6_2026-10-09.json`, `weak_reader_evidence_v1_partial_check_checkpoint6_2026-10-09.json`.

All five Rust gates pass (`gates_checkpoint6_2026-10.json`). Both test commands use opt-level1 with debug assertions and overflow checks enabled. Preliminary library-suite execution durations are 7.98s default and 11.98s all-features; the initial optimized dependency rebuild is recorded separately. Reader prompts, binary and generation parameters are unchanged. Previous checkpoint13415cb is pushed.

Usage-supported cost $5.651206; conservative reservations $22.021348; exposure $27.672553 (`benchmarks/results/spend_checkpoint_2026-10-09T2034.json`). Running: local reader61378, incremental judges31363, evidence updater3004. Scoreboard diff: none; D1, D2 and D3 remain open.

### Session 1 Item1 acceptance 2026-10-09T20:46:03.178652+00:00

1. Finish the approved parallel native harness implementation; freeze train-only operating points, then run and commit the matched native comparisons.
2. Run learned early termination's preregistered kill test after Item2 acceptance.
3. Begin fetch pipelining next if learned termination fails its kill test; otherwise start it in session2.

All100 dev questions are complete for the three matched-reader arms and three local-reader arms. Matched correctness is 83/100, 82/100 and 87/100; verbatim citation-span scores are 187/192, 180/186 and 201/204. Local correctness is 47/100, 46/100 and 48/100; four malformed local answers remain incorrect. Repeats: 12/400 disagreements, two unstable questions among20, zero judge failures. The full answer checker rejects six corruptions; the full local evidence checker rejects nine and verifies all300 records/146 distinct requests/15 native chunks.

Accepted artifacts: `benchmarks/results/longmemeval_dev_answers_v1.json`, its `_summary.json` and `_spend.json`, `weak_reader_evidence_v1.json`, and `weak_reader_evidence_v1_check.json`. The report SHA is `79aef5ef72dd8d1677783a9411e9e8feb78c15c1fe699f36a59efc7c74433887`. Five Rust commit gates pass (`gates_checkpoint7_2026-10.json`), with optimized tests retaining assertions and overflow checks. All reader/judge jobs and evidence updater have exited.

Scoreboard diff: Zep and Mem0 now carry HMS dev87/100 and citation201/204; their matched vendor cells/margins remain empty. Vendor-native500-question values stay separate. D1 is established by the full checks and this acceptance commit; D2 and D3 remain open. Usage-supported cost $5.6556705, conservative reservations $22.0213475, exposure $27.677018. Checkpoint865f399 is pushed.

The user approved parallel Item2 implementation at20:39 UTC. Draft producer/native bridge work is isolated in `/Volumes/A/hms-native-harness` at base865f399 with target `/Volumes/A/.hms-target-native-harness`; root owns the independent checker and scoreboard. Native performance timing waits for this Item1 commit and load below3; no ANN test inputs have been used for tuning.

### Session 1 checkpoint 2026-10-09T21:13:55.826119+00:00

1. Freeze the reviewed native protocol, complete train-only calibration and full native builds, then collect the 11 paired ANN rounds and commit checked scoreboard rows.
2. Run learned early termination against the strongest documented patience control after Item2 acceptance.
3. Start fetch pipelining next if learned termination fails; otherwise carry it to session2.

D1 is committed and pushed in `bbc42d9`: matched dev83/82/87, local dev47/46/48, citation201/204 for matched operand, judge disagreements12/400. No memory requests remain running. Scoreboard diff since that acceptance: none; D2 and D3 remain open. Spend exposure remains $27.677018, comprising $5.6556705 usage-supported charges and $22.0213475 conservative reservations.

Native producer/bridge drafts are isolated in `/Volumes/A/hms-native-harness` with target `/Volumes/A/.hms-target-native-harness`. A regression using16,384 real glove train vectors passed complete persistence/16-query output identity and rejects malformed headers/layers; Python adapter validation is pending a macOS27 link/import workaround. Documented controls are build_ef200, nytimes rerank16/64 and patience256, glove rerank16 and patience384. No ANN test vectors or performance timings have run. Root's independent `benchmarks/public/check_native_harness.py` is drafted and parses; full calibration/measurement checks and tamper tests await real producer artifacts.

Local hardware pins and official distance-threshold recall references are in `docs/COMPETITORS.md`. Blocking resource for x86: dedicated bare-metal provider credentials; provisioning spend $0 and x86 cells remain no x86 run. All five root commit gates pass; default/all-feature suites pass337/521 tests, with numerical proof in `benchmarks/results/gates_checkpoint8_2026-10.json`. All operational elapsed values remain preliminary, with no ANN timing claim.
