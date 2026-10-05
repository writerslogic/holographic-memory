# Continuation state (written 2026-10-05, for the next session)

Durable copy of the hand-off. Delete this file once the items below are merged or resolved.

## Three agents were running in worktrees under `.claude/worktrees/` when the session ended

| Worktree branch | Task | State at hand-off |
|---|---|---|
| `worktree-agent-a74af5826c592ef18` | Quantized-graph ANN index (`src/core/qgraph/`, `public-bench ann-qgraph`) | Commit `bb1277c` + uncommitted work; timed comparison vs FAISS/hnswlib in progress |
| `worktree-agent-afbc860a90d53dcac` | Local model stages inside HMS (`src/core/models/`, `local-models` feature) | Uncommitted work in progress |
| `worktree-agent-a864a5c529281f471` | Holographic fact memory prototype (`benchmarks/public/holographic_memory_proto.py`) | Structuring pass on Modal (tag `holo-struct-dev`, cap $5); uncommitted |

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
