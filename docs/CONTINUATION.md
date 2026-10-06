# Continuation state (written 2026-10-05, for the next session)

Durable copy of the hand-off. Delete this file once the items below are merged or resolved.

## State (updated 2026-10-06)

| Workstream | State |
|---|---|
| Quantized-graph index | `public-bench ann-qgraph --holdout N` tunes on held-out train vectors (test set unread). nytimes held-out grid in `benchmarks/results/public_qgraph_tuning_nytimes-256-angular.json`: degree 64 / 512-bit codes wins at both targets (7,715 / 2,087 QPS at 0.90 / 0.95 on held-out queries) with 1.77 GB against 2.95 GB for the old 1,024-bit setting. Alpha and build_ef were not swept. Running unattended: `/Volumes/A/.hms-target/logs/qgraph_pipeline.sh` (log `qgraph_pipeline.log`; tuning JSONs in `logs/tune/glove`, finals in `logs/final`) = glove held-out grid (d64 b128/256/512, d32 b256), auto-pick, then final HMS runs (repeats 3, load < 3) and `evaluate.py ann ... --repeats 3` for both sets, writing `benchmarks/results/public_qgraph_<set>.json`. The machine is shared with other sessions' builds (load 20-40 observed), so runs wait up to 6 h for the gate; check `load_gate_met` on every row before using a number. A first glove tuning attempt ran entirely at load ~38 and was discarded (`logs/tune/glove_unmet`). Next: when the pipeline finishes, merge glove tuning with `logs/merge_tuning.py`, rewrite the qgraph section of `docs/PUBLIC-BENCHMARKS.md` (drop the contamination caveat, state alpha/build_ef untuned, new memory ratio, glove same-session numbers), commit. |
| Local model stages | Every HMS stage runs on L4 (candle 0.11 CUDA kernels do not build for T4). `--engine hms --models small --cap 8 --tag s-dev-hms` is running (ledger $3.55 at 16:04; ~17x slower than vLLM per session; facts alone project to ~$5.3, so it is expected to stop at the cap before retrieval metrics). LLM-stage quality measured on 1,500 sessions: 74% identical fact output, 96% of facts matched (`benchmarks/results/longmemeval_dev_hms_stages.json`). Next: record the run's final ledger and outcome in that file; do not raise the cap without the maintainer. Batched GPU decoding in HMS is the fix that would make the run affordable. |
| Holographic memory | Done for dev: 3 seeds and the equal-bytes control are in `holographic_dev.json` and the doc. Shards beat an equal-bytes index only beyond ~30% loss; no variant beats the index on accuracy; no Rust port. |

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
- No push, tag, publish, yank, or Modal spend beyond stated caps without the maintainer's explicit instruction.
- Credentials: the PyPI and Gemini credentials read by a prior agent still need rotation by the maintainer.
