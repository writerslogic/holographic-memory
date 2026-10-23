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
| Date, spend | 2026-10-23. This run's OpenAI spend was 10.37 USD, measured from API usage. It reused 6,892 of the 7,700 primary reader answers, cached by prompt hash from an interrupted run on 2026-10-22; their prompts were byte-identical, so the cache keys matched. That earlier run cost about 3.5 USD (0.96 measured, about 2.5 estimated), so the campaign total is about 13.9 USD. |

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
  35.86 and BLEU-1 27.13 vs 23.90. Mem0^g is also ahead on every metric (65.71 J, 38.09 F1,
  26.03 BLEU-1). Evidence recall is the weakest here, at 0.60 for primary: list questions need
  many turns, and the 1,764-token cap cuts them off. Uncapped recall raises multi-hop J to 67.38,
  level with Mem0. Uncapped still trails Mem0 and Mem0^g on multi-hop F1 (37.33) and BLEU-1 (25.56).
- **Open-domain (id 3).** Mem0 is ahead of both primary and uncapped on J (51.15 vs 48.96 and
  47.57), F1 (28.64 vs 27.40 and 26.87) and BLEU-1 (21.58 vs 20.48 and 19.96). This category has
  only 96 questions, so one question moves J by about 1 point.
- **Against full-context.** Full-context beats every HMS arm on J for multi-hop, open-domain and
  single-hop. HMS's overall lead over full-context on all 10 conversations comes from temporal
  questions (71.59 vs 49.22). On the held-out conversations, full-context leads HMS primary
  overall.
- **Lexical-only HMS.** Without embeddings, HMS scores 64.91, below Mem0 (66.88), Mem0^g and Zep
  overall, and below Mem0 on J in every category except temporal.

## Retrieval round 2: why multi-hop and open-domain still lose

Results: [`benchmarks/results/locomo_v2.json`](../../benchmarks/results/locomo_v2.json). This round
tried to close the multi-hop and open-domain gaps through retrieval alone. The reader, judge,
prompts, Mem0 protocol and raw-turn ingest were unchanged. **No candidate passed, so the J
numbers above stand, and every loss listed above remains.** The round spent no OpenAI money.

### Protocol
- **Diagnosis.** Misses were diagnosed offline from the #236 primary outputs on all 10
  conversations. LoCoMo's evidence annotations were used only to score evidence recall and to
  classify misses. No question text, gold answer, annotation or per-question outcome entered any
  retrieval code, parameter or test fixture.
- **Selection.** Each lever and its grid were written down before the lever was first measured.
  The configuration was chosen on the 3 tuning conversations by a fixed rule: highest mean evidence
  recall over ids 1 and 3, without lowering overall evidence recall.
- **Held-out.** The 7 held-out conversations were scored once per final candidate. They are the
  clean estimate; all-10 rows overlap the tuning set.
- **Second gate.** Every candidate also had to not lower answer-session recall on the 100
  LongMemEval dev questions. That run used the memory API with the cached all-MiniLM-L6-v2 turn
  embeddings, retrieval only (`benchmarks/public/longmemeval_memory_recall.py`).
- **Count.** Configurations tried: 8 on the LoCoMo tuning set, 3 held-out looks (the baseline and
  two candidates), and 8 LongMemEval dev runs. Two of those dev runs were invalid: they mixed
  unembedded assistant turns into hybrid recall, and were discarded.

### Diagnosis (evidence recall at the 1,764-token cap)
In the primary arm, 100 of 282 multi-hop and 49 of 96 open-domain answers were judged wrong in most
passes. Those wrong answers were missing 498 evidence turns between them:

| Where the missing evidence turn was | Turns |
|---|---|
| Within 2 turns of a retrieved turn, same session | 258 |
| In a retrieved session, farther away | 152 |
| Shares no content word with the question | 365 |
| An image turn whose content is only in the caption (Mem0's protocol does not ingest captions) | 202 |
| Inside k=30 per speaker but cut by the token cap | 104 |

The categories overlap, so the counts do not add up to 498.

Evidence sits in one speaker's memory for 232 of 282 multi-hop and 82 of 92 open-domain questions,
yet round-robin packing gives each speaker about 18 of the ~36 slots. Many open-domain wrong answers
(15 of 49) had every evidence turn in context. Uncapped recall scores lower on open-domain than
capped (47.57 vs 48.96), so that category is limited by the reader, not by the cap.

### Levers and results
Evidence recall, overall / id 1 / id 3:

| Configuration | Tuning 3 | Held-out 7 (one look) |
|---|---|---|
| Baseline: #236 primary, round-robin packing | .843 / .642 / .476 | .810 / .590 / .589 |
| Episode context β=0.25 | .855 / .648 / .524 | |
| Episode context β=0.5 (candidate B) | .851 / .639 / .548 | .841 / .620 / .611 |
| Episode context β=1.0 | .827 / .586 / .500 | |
| Cosine packing (candidate A) | .852 / .710 / .595 | .834 / .650 / .593 |
| Zipper packing | .856 / .676 / .595 | not scored |
| Cosine packing + episode 0.5 | .857 / .714 / .571 | |
| Zipper packing + episode 0.5 | .860 / .675 / .524 | |

The three levers:
- **Episode context**, a memory-API recall option. Each candidate's score gains β times the best
  score among the other candidates that share its event time.
- **Cosine packing**, in the adapter. Both speakers' k=30 lists are merged by raw query cosine.
  The cosine is comparable across the two tenants; the blend score is not, because it is min-max
  normalized per tenant.
- **Zipper packing**, in the adapter. Each speaker's recall order is kept, and the next slot goes to
  the speaker whose next memory has the higher cosine.

Both held-out candidates raised evidence recall on LoCoMo. Both failed the LongMemEval dev gate:

| LongMemEval dev (answer-session recall) | any@5 | all@5 | any@10 | all@10 |
|---|---|---|---|---|
| Single tenant, recall order | .82 | .69 | .83 | .73 |
| Single tenant, episode β=0.5 | .72 | .49 | .78 | .63 |
| Single tenant, reordered by cosine (the within-list part of A) | .75 | .55 | .78 | .65 |
| Two tenants by session parity, round-robin packing | .82 | .66 | .85 | .73 |
| Two tenants by session parity, zipper packing | .81 | .63 | .83 | .73 |

- **Episode context: rejected.** On LongMemEval, boosting one session's siblings pushes the other
  answer sessions out of the top k.
- **Cosine packing: rejected.** Its within-list reordering discards the lexical signal. The gate
  used MiniLM, which is weaker than text-embedding-3-small. That is a limitation of the gate, but it
  does not change the decision.
- **Zipper packing: rejected, with no held-out look.** On a single list zipper is the identity, so
  it was gated on a two-tenant split instead. There it drops at cuts 5 and 10, by 1 to 3 of 100
  questions; at a cut of 20 it scores .86 / .75 against .85 / .74.

On the order of events: the rule that promoted zipper was written after the A and B held-out looks.
The rule takes the best tuning configuration whose every part passes the second gate. Zipper was
already in the pre-registered grid.

### What still loses, and why
- **Multi-hop vs Mem0 (63.90 vs 67.13 J).** The cap is the binding limit: uncapped recall reaches
  67.38. At the cap, the evidence for list questions is spread across many sessions and usually
  belongs to one speaker. Packing by relevance across speakers recovers part of it on LoCoMo
  (held-out evidence recall .590 to .650). Every packing rule tried so far lowered LongMemEval
  recall, so none ships.
- **Open-domain vs Mem0 (48.96 vs 51.15).** Not bound by the cap: 15 of the 49 wrong answers had all
  of their evidence in context, and 21 had none of it. Two questions out of 96 separate the two
  systems, which is within pass-to-pass noise.
- **Single-hop and overall vs full-context.** Full-context reads every turn. Evidence recall at the
  cap is .905 on single-hop (tuning) and .893 (held-out).

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
- **Context tokens.** HMS context is the o200k token count of the two rendered memory lists. The
  paper does not say which tokenizer produced its 1,764.
- **Full-context passes.** Our full-context row is a single pass.
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
Round 2 runs retrieval only, with no spend: `$B --convs conv-26,conv-30,conv-41 --embed --token-budget 1764
--retrieval-only --pack {round-robin,cosine,zipper}`. The LongMemEval gate runs as `longmemeval_memory_recall.py
--cache-dir ~/.cache/hms-bench [--order cosine | --split --pack zipper]`, with `HMS_MAX_TENANTS=256` and numpy
added to the `uv` line. Episode context was measured with an unshipped recall option, described above.
`assemble_results.py` writes the results JSON from the four `predictions.jsonl` files. A full run
from an empty cache costs about 13 USD at list price: about 6 for full-context, 3 for the primary
arm, 2.1 for uncapped and 1.6 for lexical.
