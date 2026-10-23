# HMS under Mem0's benchmark harness (memory-benchmarks)

This directory runs [mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks), Mem0's
current harness for LoCoMo, LongMemEval-S and BEAM, against HMS. The harness runs **unmodified** at
commit `4b61c5d31b9c668a12b4f5e78064248a02c82d2b` with **no patches**. HMS replaces only the memory
backend:

```
memory-benchmarks run.py --backend oss --mem0-host http://127.0.0.1:<shim>
        |  POST /memories, POST /search (Mem0 OSS REST, benchmarks/common/mem0_client.py)
   shim.py  ->  hms-server POST /api/v1/memory/{remember,recall}   (one tenant per harness user_id)
        |  OpenAI calls: OPENAI_BASE_URL -> openai_proxy.py -> api.openai.com (logs params, usage, spend cap)
```

| File | Role |
|---|---|
| `shim.py` | Mem0 OSS REST surface over the hms-server memory API |
| `retrieval_config.json` | Frozen retrieval configuration. It is copied into the manifest with its sha256 |
| `openai_proxy.py` | Pass-through proxy. It records model alias, parameters, resolved snapshot, usage and spend, and enforces a hard USD cap |
| `run_mem0_protocol.py` | Starts server, shim and proxy, runs the harness with the protocol flags, writes `manifest.json` |
| `price.py` | Prices a run from `--predict-only` output using the harness's own prompt builders |
| `manifest_template.json` | Fields every run manifest carries |
| `test_shim.py` | Unit tests for tenant mapping, fact splitting and `created_at` parsing |

Results table: [`docs/benchmarks/MEM0-PROTOCOL.md`](../../../docs/benchmarks/MEM0-PROTOCOL.md).

## License

memory-benchmarks is Apache-2.0. We could vendor it with its LICENSE and a NOTICE of changes, but we
don't. This directory holds no harness code. Each run clones the pinned commit and checks that
no tracked files are modified. Prompts are imported from that clone at run time and never copied
here. The published result JSONs under its `results/platform/` are the source of the Mem0 numbers we
quote.

## What the backend does (fixed before any run)

- **Ingest unit.** Each harness `add()` call becomes one fact: LoCoMo 1 turn (with the
  `[Sharing image - query: ... The image shows: <blip_caption>]` tag the harness itself appends),
  LongMemEval 1 user/assistant pair, BEAM 2 messages. The fact text is one `"{role}: {content}"` line
  per message, holding the strings the harness sent. The same rule applies on every benchmark, with
  no per-benchmark branch.
- **Long facts.** A fact over the engine's single-fact limit (60 KiB / 4000 words here) is split on
  word boundaries into several facts with the same time. That's 6 of 124,345 LongMemEval-S pairs.
- **No model calls at ingest.** The shim and hms-server call no LLM and no embedding model.
- **Time.** The harness `timestamp` (session date, unix seconds) becomes the fact's `eventTime`.
  `created_at` is returned as naive UTC ISO-8601 (`2023-05-08T13:56:00`), which the harness's
  `_to_human_date` parses and sorts. The Mem0 Platform uses the same timestamp semantics.
  Mem0's OSS docker server drops `timestamp`.
- **Search.** `memory/recall` returns `k = limit` (200), lexical (BM25) only, default options, best
  first. Results are `{"id", "memory", "score", "created_at"}`, the normalised shape `mem0_client`
  produces.
- **Isolation.** There is one hms-server tenant per harness `user_id`: one per LoCoMo conversation,
  per LongMemEval question and per BEAM conversation.
- **Optional, off by default:** `max_context_tokens` in the config caps the returned memory text at a
  fixed o200k token budget, as a rank prefix. It is a run-level constant, never per question. See
  "Decision needed".

### Anti-gaming

- The shim receives only what `mem0_client` sends: messages with role, content and timestamp, plus
  the question text and `user_id`. Gold answers, evidence ids, question categories, rubrics and
  dataset fields never reach it.
- The shim branches on no question text pattern and has no per-question or per-benchmark parameter.
- `load_config` rejects unknown keys.
- The configuration is frozen in `retrieval_config.json` before the run. The manifest records its
  sha256 alongside the HMS commit, the dirty flag and the hms-server binary sha256.
- With `--predict-only` first, the predict directory hash goes into the manifest. A later
  `--evaluate-only` scores exactly that retrieval.
- Retrieval is deterministic. Two fresh servers returned identical top-200 lists for all 152
  LoCoMo conversation-0 questions.

## Protocol flags (from Mem0's published result metadata)

`run_mem0_protocol.py` always passes the following flags and refuses any attempt to override them:

| Benchmark | Flags |
|---|---|
| all | `--answerer-model gpt-5 --judge-model gpt-5 --provider openai --top-k 200 --top-k-cutoffs 200 --backend oss` |
| LoCoMo | `--categories 1,2,3,4` (1,540 q), `--dataset-path` locomo10.json (sha256 `79fa87e9...`) |
| LongMemEval-S | `--mode answerer --all-questions` (500 q; the default would sample 30), `--dataset-path longmemeval_s_cleaned.json` |
| BEAM | `--chat-sizes 1M` (35 conversations, 700 q) or `10M` (10 conversations, 200 q) |

The harness defaults that differ from Mem0's headline runs are `--top-k-cutoffs 10,20,50,200` for
LoCoMo and LongMemEval, which costs 4x, and `100` for BEAM.

### Model calls (measured through the proxy in the smoke run)

- **Alias and snapshot:** alias `gpt-5`. OpenAI resolved it to **`gpt-5-2025-08-07`** on every call.
- **Parameters:**
  - Answerer: Chat Completions, one user message, `max_completion_tokens=4096`.
  - Judge (LoCoMo/BEAM): the same, plus `response_format={"type":"json_object"}` and a system prompt.
  - Judge (LongMemEval): plain text with yes/no parsing.
- **Omitted parameters:** no `temperature` (the client omits it for gpt-5, so the API default of 1
  applies), no `reasoning_effort` (the API default applies; documented as `medium`, to confirm),
  no `seed`, no `verbosity`. The proxy confirmed that none of these was sent.
- **Endpoint:** we call `api.openai.com`. **Mem0's published runs used Azure OpenAI**
  (`"provider": "azure"` in every result file), with the deployment and API version not recorded.
  Model weights behind the same snapshot should match, but Azure content filtering and default
  settings are not verified to be identical.
- **Passes and limits:** single pass, `--max-workers 10`, LLM `rpm=200`. LLM and memory calls retry
  5 times.

### Silent failures (check after every run)

When retries run out, `Mem0Client` returns `[]` or `None` and `LLMClient` returns `""`. The harness
scores both as WRONG and keeps going. The manifest's `audit` block counts each case:

- empty `generated_answer`s
- questions with zero retrieved memories
- `failed after` lines and retry lines in the harness log

`llm.finish_reasons` shows `length` truncations, and `ingest.facts` should equal the harness chunk
count. A run with any of these is not reportable until those questions are re-scored.

After a 402 from the proxy's spend cap (`llm.blocked > 0`) or a 429 when credit runs out, the
harness has already written WRONG results with empty answers into the per-question files. A plain
rerun does not fix this. LoCoMo and LongMemEval `--evaluate-only` skip any file that already holds
`cutoff_results` (unless `--rejudge` is given, which re-scores everything), and they refuse to
start while any file is missing. To salvage a run:
1. Delete only the `cutoff_results` key from the files listed in `audit.empty_generated_answers`.
2. Raise `--max-spend`.
3. Rerun `--evaluate-only`.

The BEAM runner has no `--rejudge`, and its skip logic is not checked. Each run writes
`manifest-<predict|evaluate|full>-<UTC stamp>.json`. `retrieval_before` and `retrieval_after` hash
only the question ids and returned memories. On `--evaluate-only`, `audit.retrieval_unchanged` must
be true; this was checked on a 1-question LoCoMo evaluate-only smoke run ($0.02).

## Commands

```sh
# once
git clone https://github.com/mem0ai/memory-benchmarks /path/memory-benchmarks
git -C /path/memory-benchmarks checkout 4b61c5d31b9c668a12b4f5e78064248a02c82d2b
uv venv /path/venv && VIRTUAL_ENV=/path/venv uv pip install -r /path/memory-benchmarks/requirements.txt \
    datasets python-dateutil tiktoken          # BEAM imports datasets and dateutil; price.py uses tiktoken
CARGO_TARGET_DIR=/path/target cargo build --release --features server --bin hms-server
export OPENAI_API_KEY=...                      # never written to any log or manifest
PY=/path/venv/bin/python; H=benchmarks/public/mem0_protocol
COMMON="--clone /path/memory-benchmarks --python $PY --hms-bin /path/target/release/hms-server"

# 1. retrieval only, $0: freeze it (data dir on a local SSD; see wall clock)
$PY $H/run_mem0_protocol.py locomo $COMMON --work runs/locomo --data-dir /ssd/hms-locomo \
    --dataset ~/.cache/hms-bench/locomo/locomo10.json --max-spend 0 -- --predict-only
# 2. price it from the frozen retrieval
$PY $H/price.py --clone /path/memory-benchmarks --benchmark locomo \
    --predict runs/locomo/harness-out/predicted_hms-locomo --calls runs/locomo-smoke/openai_calls.jsonl
# 3. answer + judge exactly that retrieval (no memory calls)
$PY $H/run_mem0_protocol.py locomo $COMMON --work runs/locomo --data-dir /ssd/hms-locomo \
    --dataset ~/.cache/hms-bench/locomo/locomo10.json --max-spend 50 -- --evaluate-only

# LongMemEval-S: same with `longmemeval --dataset .../longmemeval_s_cleaned.json`
# BEAM: `beam-1m` or `beam-10m --dataset /path/beam-cache` (the harness downloads Mohammadta/BEAM
#       1M split, 66 MB parquet / 173 MB json; BEAM-10M, 344 MB parquet)
```

`--evaluate-only` exists in all three runners. Mem0 used it for its top-50 LongMemEval file
(`"evaluate_only": true`). Smoke test (done, $0.09):
`... locomo ... --max-spend 1 -- --conversations 0 --max-questions 3`.

## Decision needed before the real run

Top-200 is a **count** of memories. A Mem0 memory is one extracted fact of about 35 tokens. Its
docs give mean context per query of 6,956 tokens on LoCoMo and 6,787 on LongMemEval. An HMS memory
is a raw turn, pair or chunk, so the same count carries far more text. Measured answer-prompt tokens
per question:

| Benchmark | HMS top-200 (count-matched) | Mem0 top-200 | Note |
|---|---|---|---|
| LoCoMo | 12,285 mean (152 q of conv 0), of which ~9.1k is memory text | 6,956 memory tokens | 1.3x Mem0 |
| LongMemEval-S | 87,577 mean (30 q) | ~6.8k | about 76% of a ~115k-token haystack, so near full context |
| BEAM 1M | ~216k estimated (200 x 1,081-token mean chunk), not measured | n/a | Close to gpt-5's input limit (272k per OpenAI's model page, to confirm); whether any prompt exceeds it is unmeasured |

The count-matched run is the literal protocol. On LongMemEval and BEAM it measures the reader's
long-context ability more than memory retrieval, and BEAM-1M prompts sit near the context limit.
`max_context_tokens: 7000` gives a token-matched variant. It is just as fixed, and stated next to
the score. Pick one of these before any paid run:

- (a) count-matched everywhere;
- (b) token-matched everywhere;
- (c) count-matched LoCoMo plus token-matched LongMemEval and BEAM, each row labelled.

The hybrid recall arm (text-embedding-3-small) is excluded by the "no model calls at ingest" rule.
Allowing it is a separate decision.
