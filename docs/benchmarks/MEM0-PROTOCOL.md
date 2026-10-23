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

## Cost and run plan

Every HMS row is **token-matched**. `retrieval_config.json` caps the returned memory text at 7,000
o200k tokens per question, matching Mem0's documented ~7k memory tokens. The cap is filled in rank
order, it is one constant for the whole run, and it is recorded in the manifest. Recall is
lexical (BM25) only. The literal top-200 (count-matched) variant is documented in the adapter
README, but it is not scheduled.

Prices are gpt-5 list price ($1.25 / $10 per 1M tokens, to confirm). The Batch API is not usable
unpatched. Output is calibrated on 3 LoCoMo calls. Spent so far: **$0.34** ($0.09 smoke, $0.24
LongMemEval calibration, $0.02 evaluate-only smoke). All prompt counts below are from retrieval-only
runs at $0.

| Run (cap) | Answer prompt | Price | Basis |
|---|---|---|---|
| 1. LoCoMo token-matched (cap $48) | 10.2k tokens, 162 memories | $42.6 | conv 0 measured, 152 q scaled to 1,540 |
| 2. LongMemEval-S token-matched (cap $13) | ~7k memory plus template | ~$10 | estimate; the $0 retrieval-only step re-prices it before the paid step |
| 3. BEAM-10M token-matched (cap $10) | ~7.2k | ~$7.2 | BEAM-1M per-question cost x 200 |
| later: BEAM-1M token-matched | 7.2k tokens, 6.3 memories (max 7.3k) | $25.4 | conv 0 measured, 20 q scaled to 700 |
| not run: count-matched | LoCoMo 12.3k, LongMemEval 87.6k, BEAM-1M 227k mean / 242k max | $46.6 / $59.5 / $218 | |

BEAM's `--evaluate-only` only recomputes metrics, so BEAM's paid run is the combined run. Its
retrieval hash is checked against the $0 retrieval-only run. Hybrid keyword+embedding recall is left
out because the shipped local embedder does not serve the memory API (adapter README).

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
