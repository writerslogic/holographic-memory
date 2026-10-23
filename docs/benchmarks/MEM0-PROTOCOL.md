# HMS vs Mem0 under Mem0's own harness (memory-benchmarks @ 4b61c5d)

Status: **prepared, not run.** The adapter, commands and anti-gaming rules are in
[`benchmarks/public/mem0_protocol/README.md`](../../benchmarks/public/mem0_protocol/README.md).
Everything below "HMS" is filled from a run's `manifest.json` and the harness result JSON.

## Mem0's reference numbers

Mem0's reference numbers come from the committed result files in `results/platform/` at `4b61c5d`:
gpt-5 answerer and judge via Azure, top-200, single pass.

The repository README states **LoCoMo 92.5% (1425/1540)** and **LongMemEval 94.4% (472/500)**. The
committed top-200 result files say **91.56% (1410/1540)** and **93.4% (467/500)**. The files carry
merged re-run subsets (`merged_from_questions`) and metadata keys that the committed `run.py` does
not write, so Mem0's internal harness version differs from the public one. Both figures are shown.
The 64.1 / 48.6 BEAM figures are the **average rubric score** x100. The pass rates are 70.1 / 50.5.

### LoCoMo (categories 1-4, 1,540 q)

| Category (id, n) | Mem0 top-200 (result file) | HMS |
|---|---|---|
| multi-hop (1, 282) | 93.3 (263/282) | |
| temporal (2, 321) | 92.8 (298/321) | |
| open-domain (3, 96) | 76.0 (73/96) | |
| single-hop (4, 841) | 92.3 (776/841) | |
| **overall** | **91.6 (1410/1540)**; README 92.5 | |
| mean answer-prompt tokens | ~7k (Mem0 docs: 6,956 memory tokens) | |

### LongMemEval-S (500 q)

| Type (n) | Mem0 top-200 (result file) | HMS |
|---|---|---|
| knowledge-update (78) | 96.2 (75/78) | |
| multi-session (133) | 86.5 (115/133) | |
| single-session-assistant (56) | 100.0 (56/56) | |
| single-session-preference (30) | 96.7 (29/30) | |
| single-session-user (70) | 97.1 (68/70) | |
| temporal-reasoning (133) | 93.2 (124/133) | |
| **overall** | **93.4 (467/500)**; README 94.4 | |
| mean answer-prompt tokens | ~6.8k (docs: 6,787) | |

### BEAM 1M (700 q) and 10M (200 q): pass rate (score >= 0.5) and average score x100

| Ability | Mem0 1M pass | Mem0 1M avg | HMS 1M pass | HMS 1M avg | Mem0 10M pass | Mem0 10M avg | HMS 10M pass | HMS 10M avg |
|---|---|---|---|---|---|---|---|---|
| abstention | 55.7 (39/70) | 52.5 | | | 40.0 (8/20) | 40.0 | | |
| contradiction resolution | 48.6 (34/70) | 35.7 | | | 25.0 (5/20) | 32.5 | | |
| event ordering | 60.0 (42/70) | 53.6 | | | 15.0 (3/20) | 20.2 | | |
| information extraction | 75.7 (53/70) | 70.0 | | | 55.0 (11/20) | 56.2 | | |
| instruction following | 88.6 (62/70) | 85.2 | | | 90.0 (18/20) | 82.5 | | |
| knowledge update | 65.7 (46/70) | 65.0 | | | 80.0 (16/20) | 75.0 | | |
| multi-session reasoning | 74.3 (52/70) | 65.2 | | | 30.0 (6/20) | 26.1 | | |
| preference following | 97.1 (68/70) | 88.3 | | | 95.0 (19/20) | 90.4 | | |
| summarization | 68.6 (48/70) | 63.5 | | | 55.0 (11/20) | 46.9 | | |
| temporal reasoning | 67.1 (47/70) | 61.8 | | | 20.0 (4/20) | 16.2 | | |
| **overall** | **70.1 (491/700)** | **64.1** | | | **50.5 (101/200)** | **48.6** | | |

Each HMS row also records the following from its manifest:
- HMS commit and binary sha256
- `retrieval_config` (count-matched or token-matched) and its sha256
- mean answer-prompt tokens and the fraction of the haystack returned
- resolved model snapshot, endpoint (OpenAI, not Azure) and spend
- the audit counts, which must all be zero
- CPU and load average

## Cost (gpt-5 list price $1.25 / $10 per 1M tokens, to confirm against OpenAI pricing)

There is no cache credit. The Batch API's 50% discount is **not available** because the harness makes
synchronous `chat.completions` calls. Using it would mean our own batch implementation of the
answer/judge step, which is a patch, so these figures are full price.

Spent so far: **$0.34** ($0.09 smoke, $0.24 LongMemEval calibration, $0.02 evaluate-only smoke).

| Run | Count-matched (top-200 raw) | Token-matched (`max_context_tokens: 7000`) | Basis |
|---|---|---|---|
| LoCoMo | **$46.6** | **$42.5** (161 memories, 10.2k-token prompt; measured on conv 0) | Prompts counted with `price.py` on conv 0 (152 q, scaled to 1,540). Output from 3 calibrated questions: answer 1,159 completion tokens incl. reasoning, judge 192. Output is ~$21 of the total either way |
| LongMemEval-S | **$59.5** | ~$10 | 30 predicted questions (all single-session-user; the harness orders by type), 87.6k-token prompts, scaled to 500. Output from 2 calibrated questions |
| BEAM 1M | ~$208 | ~$25 | Estimate: 200 x 1,081-token mean chunk = ~216k-token prompts, near the 272k input limit (to confirm; whether any prompt exceeds it is unmeasured). 3.45 nuggets/q, event-ordering extract/align calls. Output assumed like LoCoMo |
| BEAM 10M | ~$59 | ~$7 | Same per-question estimate, 200 q, 3.1 nuggets/q |

Apart from LoCoMo token-matched, the token-matched and BEAM figures are estimates from the formulas above, not `price.py` runs on
predict output. Before any paid BEAM run, rerun `price.py` on its `--predict-only` output, which
costs $0. `price.py` has not yet run on BEAM output, and it is unchecked whether BEAM predict files carry `rubric`. A BEAM-1M conversation-0 predict-only run ($0) settles both.

Uncertainty: on LoCoMo, output (reasoning included) is about 45% of the $47 estimate, and it rests on 3 calls of 597, 1,126 and 1,754 completion tokens. Set each run's `--max-spend` to its price plus a margin. Re-price LongMemEval and BEAM from LoCoMo's real proxy log before spending on them, since the $64 plan has only ~$11 of slack.

### Proposed order within the ~$75 credit

1. **LoCoMo, count-matched, $47.** This is the literal protocol. Its memory text is ~9.1k tokens,
   1.3x Mem0's 6,956; the prompt template adds ~3.2k.
2. **LongMemEval-S, token-matched, ~$10.** The count-matched version is near full context.
3. **BEAM-10M, token-matched, ~$7.**
4. Total about $64. **BEAM-1M (~$25 token-matched)** waits for more credit.

If the founder wants count-matched LongMemEval ($60), it replaces steps 2 and 3, and LoCoMo plus
LongMemEval then come to about $107, which exceeds the credit.

## Wall clock

Ingest throughput depends on the disk, because every `remember` is synced. The machine was an Apple
M-series running macOS, HMS 69d15687, with harness `--max-workers 10`. Development measurements, one
run each and not in the timing lock:
- External USB HDD: 15 facts/s, one LoCoMo conversation alone (load avg about 3).
- Internal SSD: 182 facts/s, which is 2.3 s for LoCoMo conversation 0's 419 turns.
- With three runs sharing the HDD, it fell to 0.7-1.5 facts/s.

Put `--data-dir` on an SSD.

LLM latency through the proxy (gpt-5, OpenAI) was 8-23 s per LoCoMo answer and 3-5 s per judge
call. LongMemEval answers with 87k-token prompts took 9-12 s.

| Run | Estimate |
|---|---|
| LoCoMo | ~1 h. Questions run serially within a conversation, with 10 conversations in parallel at about 20 s per question; the largest conversation bounds it |
| LongMemEval-S | ~15 min ingest (124k pairs) plus ~25 min answer/judge (500 q, 10 workers) |
| BEAM 1M | Ingest 37k chunks of ~1k tokens (rate on SSD not measured). Then ~1 h answer/judge: 35 conversations, 10 in parallel, 20 questions each, ~4.5 calls per question |
| BEAM 10M | Ingest is not measured (~10M tokens per conversation, 10 conversations in parallel), then ~15 min answer/judge |

gpt-5 rate limits (TPM) of the account tier bound the parallel long-prompt runs. That has not been
checked.
