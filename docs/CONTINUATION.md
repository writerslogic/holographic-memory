# Continuation state (written 2026-10-05, for the next session)

Durable copy of the hand-off. Delete this file once the items below are merged or resolved.

## Three agents were running in worktrees under `.claude/worktrees/` when the session ended

| Worktree branch | Task | State at hand-off |
|---|---|---|
| (merged) | Quantized-graph ANN index (`src/core/qgraph/`, `public-bench ann-qgraph`) | On main. nytimes: 2.4x the best HNSW QPS at recall 0.90, 1.9x at 0.95, but 8-9x the memory and parameters tuned on 2,000 of the test queries (contaminated; re-tune on held-out train vectors). glove: HMS run only; FAISS/hnswlib re-run NOT done, so no glove claim. Next: clean re-tune, glove competitor re-run on an idle machine, memory reduction (degree 64 -> smaller or compressed neighbour codes). |
| (merged) | Local model stages inside HMS (`src/core/models/`, `local-models` feature) | On main. Embedder/re-ranker match Python within 1e-6 on a 48-item sample; LLM stages unmeasured for quality; `local-models` needs Rust >= 1.94 (candle 0.11), all other features still 1.89; GPUs run batch 1 (candle mask bug). On-device M4/Metal: ingest median 1.5 s per session (max 49 s), query ~10 s (dominated by LLM rewrite), peak RSS 9.9 GB. Next: `modal run benchmarks/public/longmemeval_modal.py --engine hms --models small --cap 8 --tag s-dev-hms` and compare with `benchmarks/results/longmemeval_dev_final.json`; check projected cost after the first wave (HMS decodes one sequence at a time). |
| (merged) | Holographic fact memory prototype | On main as `486ec35`: `benchmarks/results/holographic_dev.json`, doc section in `docs/PUBLIC-BENCHMARKS.md`. EAV variant negative (extraction ceiling); turn-atom traces close to flat scan; shard-loss robustness shown (50% shards deleted: session 0.929 vs index 0.917, turn 0.631 vs 0.488) at a cost in zero-loss recall; capacity ~128-256 atoms per 16k-bit trace with centered codes. Next for a Rust port: per-session/window traces + hierarchical unbinding; replicated-index control at equal bytes; multiple seeds. |

If the agents are gone, collect their work by hand: inspect each worktree's `git status` and `git log main..HEAD`,
run the gate in the worktree, then fast-forward or cherry-pick onto `main` (sign with
`git -c user.email=david@writerslogic.com commit -S`), remove the worktree and branch. Nothing is accepted without
its gate passing and its numbers in `benchmarks/results/`.

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
