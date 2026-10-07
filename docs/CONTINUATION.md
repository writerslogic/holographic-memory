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
unit tests pass and the release binary builds; the FULL GATE HAS NOT BEEN RUN locally on this main (the GitHub CI run on 58d1ff0 is the only full check so far; read it with `gh run list --branch main`).

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

Next: run prompt 1 (Fable) from the prompts file. Step 1 verifies the merge and decides the
defaults with one clean paired held-out run, then the competitor harness, the gate, the M4
test-set comparison, the doc rewrite, and the specs for the Opus and Sonnet sessions.

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
