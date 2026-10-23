# LoCoMo: HMS against the published Mem0 and Zep results

HMS answers LoCoMo questions through the hms-server memory API. Scoring follows Mem0's published
evaluation protocol, and HMS is compared with the numbers in the Mem0 paper:

> P. Chhikara, D. Khant, S. Aryan, T. Singh, D. Yadav. *Mem0: Building Production-Ready AI Agents
> with Scalable Long-Term Memory.* arXiv:2504.19413, 2025. Tables 1 and 2.

Results: [`benchmarks/results/locomo_v1.json`](../../benchmarks/results/locomo_v1.json) (per arm,
split and category, with per-pass J). Harness: [`benchmarks/public/locomo/`](../../benchmarks/public/locomo/).

## Setup

| | |
|---|---|
| Dataset | `locomo10.json` from snap-research/locomo 3eb6f2c (sha256 `79fa87e9…98ff4`), 10 conversations, categories 1-4 (1,540 questions). Category 5 (adversarial) is excluded, as in Mem0. |
| Protocol | mem0ai/mem0 `evaluation/` at 393a4fd5 (the paper-era code). The non-graph `ANSWER_PROMPT` reader runs as one system message. Recall is top_k=30 per speaker, and each memory is rendered `"{timestamp}: {memory}"`. The `ACCURACY_PROMPT` judge uses `json_object` output. F1 is set-token F1, BLEU-1 is nltk BLEU-1, both on the raw response. Prompts are copied verbatim and never edited. |
| Reader and judge | `gpt-4o-mini-2024-07-18` at temperature 0. Mem0 reports `gpt-4o-mini`, and this is the only gpt-4o-mini chat snapshot. |
| Embeddings | `text-embedding-3-small` (Mem0's default). |
| HMS | Commit c80aa52a, `hms-server` (`--features server`, release). The configuration was frozen on the tuning conversations at ec2da79, before any held-out call. |
| HMS ingest | One tenant per (conversation, speaker). Each turn the speaker said becomes one fact, `"{speaker}: {text}"`, with the session date as event time. No LLM extraction, summaries or captions are used, and no QA field other than the question reaches recall or the reader. |
| Machine | Apple M-series, 10 cores. hms-server ran with 4 tokio workers and 4 concurrent work slots, on a shared host (load average 3 to 20). The run measured accuracy only; no timing was taken. |
| Passes | Each pass is an independent reader and judge run over all 10 conversations: 5 for the primary arm, 3 for each secondary arm, 1 for full-context. Mem0 reports 10. |
| Split | Rule fixed before any run: the first three conversations (conv-26, 30, 41) are the tuning set, and the other seven are held out. |
| Date, spend | 2026-10-23. OpenAI spend was 10.37 USD, measured from API usage. |

### Arms
- **HMS primary.** Hybrid recall (lexical plus text-embedding-3-small, blend fusion), k=30 per
  speaker, with memories capped at 1,764 o200k tokens to match Mem0's reported context size.
- **HMS uncapped.** The same recall with k=30 per speaker only, which matches Mem0's count. This
  gives about 2,900 tokens of context.
- **HMS lexical.** No embedding model at all, capped at 1,764 tokens.
- **Full-context (ours).** Mem0's full-context baseline (`rag.py`, chunk_size -1). The whole
  conversation goes to the reader, with no memory system. This arm checks that our reader, judge
  and scoring reproduce Mem0's 72.90.

### Category ids
Columns below use LoCoMo's own category ids: 1 multi-hop (282 questions), 2 temporal (321),
3 open-domain (96) and 4 single-hop (841).

The Mem0 paper's Table 1 column labels do not follow those ids. Its numbers are re-keyed here by
weighting its J columns by the id counts. That weighting reproduces its Table 2 overall J for
Mem0 (66.88), Mem0^g (68.44) and Zep (65.99) only under one mapping: paper "single-hop" = id 1,
"temporal" = id 2, "multi-hop" = id 3 and "open-domain" = id 4. The published values are
unchanged; only their column placement differs from the paper's labels.

## All 10 conversations (the paper's setting; includes the 3 tuning conversations)

Each cell is F1 / BLEU-1 / J in percent. Overall J is question-weighted; ± is the standard
deviation across passes. Context is the mean number of memory tokens given to the reader.

| System | 1 multi-hop | 2 temporal | 3 open-domain | 4 single-hop | Overall J | Context |
|---|---|---|---|---|---|---|
| **HMS primary** (1,764-token cap, 5 passes) | 35.86 / 23.90 / 63.90 | **59.90** / **48.77** / **71.59** | 27.40 / 20.48 / 48.96 | 57.34 / 46.63 / 80.52 | 73.65 ± 0.39 | 1,749 |
| HMS uncapped (k=30/speaker, 3 passes) | 37.33 / 25.56 / 67.38 | 58.73 / 48.14 / 71.55 | 26.87 / 19.96 / 47.57 | **57.82** / **46.77** / 82.16 | **75.09** ± 0.19 | 2,884 |
| HMS lexical (no embeddings, 3 passes) | 28.59 / 18.15 / 53.43 | 54.31 / 44.57 / 63.97 | 23.79 / 17.39 / 41.32 | 50.70 / 41.43 / 71.82 | 64.91 ± 0.46 | 1,749 |
| Mem0 (paper) | **38.72** / **27.13** / 67.13 | 48.93 / 40.51 / 55.51 | **28.64** / **21.58** / 51.15 | 47.65 / 38.72 / 72.93 | 66.88 ± 0.15 | 1,764 |
| Mem0^g (paper) | 38.09 / 26.03 / 65.71 | 51.55 / 40.28 / 58.13 | 24.32 / 18.82 / 47.19 | 49.27 / 40.30 / 75.71 | 68.44 ± 0.17 | 3,616 |
| Zep, Mem0 harness (paper) | 35.74 / 23.30 / 61.70 | 42.00 / 34.53 / 49.31 | 19.37 / 14.82 / 41.35 | 49.56 / 38.92 / 76.60 | 65.99 ± 0.16 | 3,911 |
| Best RAG, 256-token chunks, k=2 (paper) | | | | | 60.97 ± 0.20 | |
| Full-context (paper) | | | | | 72.90 ± 0.19 | 26,031 |
| Full-context (ours, 1 pass) | 32.45 / 24.50 / **70.92** | 33.02 / 24.73 / 49.22 | 23.46 / 17.35 / **59.38** | 48.23 / 33.78 / **85.14** | 73.44 | 25,322 |

Our full-context run reaches 73.44 against the paper's 72.90, so the reader, judge and scoring
reproduce Mem0's baseline to within about half a point.

## Held-out 7 conversations (tuning conversations excluded)

These rows come from the same passes as the table above. No published system reports this split,
so the comparison here is against our own full-context run.

| System | 1 multi-hop | 2 temporal | 3 open-domain | 4 single-hop | Overall J | Context |
|---|---|---|---|---|---|---|
| **HMS primary** | 33.79 / 22.33 / 64.13 | 56.51 / 46.55 / 66.49 | 29.35 / 21.46 / 44.00 | 57.70 / 47.31 / 79.75 | 71.97 ± 0.47 | 1,749 |
| HMS uncapped | 35.28 / 23.82 / 66.83 | 55.07 / 45.55 / 66.96 | 29.25 / 21.13 / 43.11 | 58.59 / 47.81 / 81.54 | 73.48 ± 0.35 | 2,877 |
| HMS lexical | 25.50 / 16.35 / 50.80 | 50.83 / 42.34 / 58.73 | 24.71 / 17.25 / 37.33 | 51.81 / 42.59 / 71.81 | 63.17 ± 0.71 | 1,749 |
| Full-context (ours) | 31.69 / 24.73 / 69.71 | 31.13 / 23.94 / 48.05 | 25.02 / 17.88 / 57.33 | 49.33 / 34.33 / 84.40 | 72.73 | 26,377 |

On the held-out conversations, HMS primary scores 71.97, which is 0.8 points below full-context
(72.73). It uses 1/15 of the context to get there.

## Where HMS loses

- **Multi-hop (id 1).** Mem0 is ahead on every metric: J 67.13 vs 63.90 (primary), F1 38.72 vs
  35.86 and BLEU-1 27.13 vs 23.90. Mem0^g also leads on J (65.71). Evidence recall is the weakest
  here, at 0.60 for primary: list questions need many turns, and the 1,764-token cap cuts them off.
  Uncapped recall raises multi-hop J to 67.38, level with Mem0.
- **Open-domain (id 3).** Mem0 is ahead on J (51.15 vs 48.96), F1 (28.64 vs 27.40) and BLEU-1
  (21.58 vs 20.48). This category has only 96 questions, so one question moves J by about 1 point.
- **Against full-context.** Full-context beats every HMS arm on J for multi-hop, open-domain and
  single-hop. HMS's overall lead over full-context on all 10 conversations comes from temporal
  questions (71.59 vs 49.22). On the held-out conversations, full-context leads HMS primary
  overall.
- **Lexical-only HMS.** Without embeddings, HMS scores 64.91, below Mem0 (66.88), Mem0^g and Zep
  overall, and below Mem0 on J in every category except temporal.

## Not protocol-matched: Zep's own number

Zep reports 75.14 ± 0.17 J using its own harness (getzep/zep-papers `kg_architecture_agent_memory/locomo_eval`;
Zep blog 2025-05-06), also with gpt-4o-mini as reader and judge. Its per-id J from one run is
74.11, 66.04, 67.71 and 79.79.

That harness uses Zep's own reader prompt and retrieval template, so it is a different protocol
and is not placed in the tables above. Mem0 reran Zep's code with Mem0's reader prompt and got
58.44 ± 0.20 (getzep/zep-papers issue 5). Zep in turn disputes how Mem0 configured Zep for the
65.99 row. The only protocol-matched Zep number is the paper's 65.99.

## Differences from the published runs

- **Passes.** HMS ran 5 reader+judge passes (primary), not 10. In HMS, retrieval is
  deterministic, so the ± covers reader and judge variance only. Mem0's ± also includes its
  stochastic LLM fact extraction at ingest.
- **Snapshot.** The reader and judge are pinned to `gpt-4o-mini-2024-07-18`. The paper names only
  `gpt-4o-mini`. When the paper was written (April 2025), that alias resolved to this snapshot.
- **Labels.** Category columns are re-keyed from the paper's labels to LoCoMo ids, as described
  in "Category ids" above.
- **Ingest.** HMS stores raw turns and makes no LLM calls at ingest. Mem0 stores LLM-extracted
  facts. Both are read with the same reader prompt and memory rendering.
- **Tuning overlap.** The all-10 rows include the three conversations used to choose the
  configuration. The held-out rows are the clean estimate.

## Reproduce

```sh
cargo build --release --features server --bin hms-server
B="uv run --no-project --with openai==1.109.1 --with nltk==3.9.1 --with tiktoken==0.12.0 --with jinja2==3.1.6 \
  python benchmarks/public/locomo/locomo_bench.py --server-bin target/release/hms-server \
  --data locomo10.json --convs all --max-spend 12"
$B --out final-primary  --embed --token-budget 1764 --runs 5
$B --out final-uncapped --embed --runs 3
$B --out final-lexical  --token-budget 1764 --runs 3
$B --out final-fullctx  --full-context --runs 1
```

Reader and judge calls are cached by prompt hash, so a rerun pays only for prompts that change.
`--dry-run` counts cache misses and their input tokens without making any chat call.
`assemble_results.py` writes the results JSON from the four `predictions.jsonl` files. A full run
from an empty cache costs about 11 USD at list price, about 6 of it for full-context.
