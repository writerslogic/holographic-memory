# WARM under Mem0's benchmark harness (memory-benchmarks)

This directory runs [mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks), Mem0's
current harness for LoCoMo, LongMemEval-S and BEAM, against WARM. The harness runs **unmodified** at
commit `4b61c5d31b9c668a12b4f5e78064248a02c82d2b` with **no patches**. WARM replaces only the memory
backend:

```
memory-benchmarks run.py --backend oss --mem0-host http://127.0.0.1:<shim>
        |  POST /memories, POST /search (Mem0 OSS REST, benchmarks/common/mem0_client.py)
   shim.py  ->  warm-server POST /api/v1/memory/{remember,recall}   (one tenant per harness user_id)
        |  OpenAI calls: OPENAI_BASE_URL -> openai_proxy.py -> api.openai.com (logs params, usage, spend cap)
```

| File | Role |
|---|---|
| `shim.py` | Mem0 OSS REST surface over the warm-server memory API |
| `retrieval_config.json` | Frozen primary retrieval configuration (token-matched: `max_context_tokens: 7000`). It is copied into the manifest with its sha256 |
| `retrieval_config_count_matched.json` | Documented alternative: literal top-200 with no token cap. Not scheduled |
| `openai_proxy.py` | Authenticated local proxy with durable reservations, pinned GPT-5 routing and validated usage |
| `budget_ledger.py` | Private SQLite provider ceiling and immutable final-run allocations; initialization makes no API calls |
| `run_mem0_protocol.py` | Starts server, shim and proxy, runs the harness with the protocol flags, writes `manifest.json` |
| `price.py` | Prices a run from `--predict-only` output using the harness's own prompt builders |
| `manifest_template.json` | Fields every run manifest carries |
| `clear_failed.py` | Clears empty cutoffs only for an entirely failed model-call attempt |
| `test_shim.py` | Unit tests for tenant mapping, fact splitting and `created_at` parsing |

Results table: [`docs/benchmarks/MEM0-PROTOCOL.md`](../../../docs/benchmarks/MEM0-PROTOCOL.md).

## License

memory-benchmarks is Apache-2.0. We could vendor it with its LICENSE and a NOTICE of changes, but we
don't. This directory holds no harness code. Each run clones the pinned commit and checks that
no tracked files are modified. Prompts are imported from that clone at run time and never copied
here. The published result JSONs under its `results/platform/` are the source of the Mem0 numbers we
quote.

## Primary lexical baseline behavior (fixed before its run)

- **Ingest unit.** Each harness `add()` call becomes one fact: LoCoMo 1 turn (with the
  `[Sharing image - query: ... The image shows: <blip_caption>]` tag the harness itself appends),
  LongMemEval 1 user/assistant pair, BEAM 2 messages. The fact text is one `"{role}: {content}"` line
  per message, holding the strings the harness sent. The same rule applies on every benchmark, with
  no per-benchmark branch.
- **Long facts.** A fact over the engine's single-fact limit (60 KiB / 4000 words here) is split on
  word boundaries into several facts with the same time. That's 6 of 124,345 LongMemEval-S pairs.
- **No model calls at ingest.** The shim and warm-server call no LLM and no embedding model.
- **Time.** The harness `timestamp` (session date, unix seconds) becomes the fact's `eventTime`.
  `created_at` is returned as naive UTC ISO-8601 (`2023-05-08T13:56:00`), which the harness's
  `_to_human_date` parses and sorts. The Mem0 Platform uses the same timestamp semantics.
  Mem0's OSS docker server drops `timestamp`.
- **Search.** `memory/recall` with `k = limit` (200), lexical (BM25) only, default options, best
  first, then the token budget. Results are `{"id", "memory", "score", "created_at"}`, the normalised shape `mem0_client`
  produces.
- **Isolation.** There is one warm-server tenant per harness `user_id`: one per LoCoMo conversation,
  per LongMemEval question and per BEAM conversation.
- **Token budget (primary).** `max_context_tokens: 7000` caps the returned memory text at 7,000
  o200k tokens, matching Mem0's documented ~7k memory tokens per query. The budget is filled in rank
  order, and a memory that would overflow it is skipped. It is one run-level constant, never set
  per question. Every reported row is labelled **token-matched**.

### Anti-gaming

- The shim receives only what `mem0_client` sends: messages with role, content and timestamp, plus
  the question text and `user_id`. Gold answers, evidence ids, question categories, rubrics and
  dataset fields never reach it.
- The shim branches on no question text pattern and has no per-question or per-benchmark parameter.
- `load_config` rejects unknown keys.
- The configuration is frozen in `retrieval_config.json` before the run. The manifest records its
  sha256 alongside the WARM commit, the dirty flag and the warm-server binary sha256.
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

New guarded runs preserve the original harness parameters in the audit, then explicitly send
`model=gpt-5-2025-08-07` and `service_tier=default` upstream. Prompts and generation settings are
unchanged. Responses must confirm that snapshot and tier. This routing disclosure replaces the
older byte-for-byte forwarding behavior; it is recorded separately in every new proxy call.

### Silent failures (check after every run)

When retries run out, `Mem0Client` returns `[]` or `None` and `LLMClient` returns `""`. The harness
scores both as WRONG and keeps going. The manifest's `audit` block counts each case:

- empty `generated_answer`s
- questions with zero retrieved memories
- `failed after` lines and retry lines in the harness log

`llm.finish_reasons` distinguishes successful `length` truncations from transport failures.
An empty answer from a successful call remains a scored WRONG outcome under the fixed completion
budget. Do not selectively rescore it. Empty retrievals and empty answers remain visible in the
result audit. Failed ingestion, exhausted transport retries, spend-cap blocks, and missing or
invalid prediction/cutoff records make the wrapper exit nonzero.

`ingest.facts` counts newly stored facts for the optional hybrid adapter; idempotent repeats need
not add a new fact. Check native tenant document counts against the unique source facts when
reusing ingestion state.

The BEAM runner has no `--rejudge`, and its skip logic is not checked. Each run writes
`manifest-<predict|evaluate|full>-<UTC stamp>.json`. `retrieval_before` and `retrieval_after` hash
only the question ids and returned memories. On `--evaluate-only`, `audit.retrieval_unchanged` must
be true; this was checked on a 1-question LoCoMo evaluate-only smoke run ($0.02).

## Commands

These are the exact commands. Paid runs start only with the final WARM SHA from the coordinator. Put
`--data-dir` on a local SSD.

Before a paid final run, create one owner-only directory outside Git and a mode-600 JSON policy
there. Its fields are `provider` (exactly `openai`), `ceiling_usd` (a decimal string), and
`allocations` (final-run identifiers mapped to decimal strings). The coordinator supplies the
reconciled, unspent final-only allocation; historical charges before this baseline must already
be excluded. Allocations must sum to no more than the fixed provider ceiling. Use at most nine
decimal places. Never publish this policy, ledger, financial balances, or credentials.

```sh
# Both paths are outside Git, in a mode-700 directory; PRIVATE_POLICY is mode 600.
# Initialization refuses an existing ledger and makes no paid calls.
uv run --no-project python benchmarks/public/mem0_protocol/budget_ledger.py init \
  --policy "$PRIVATE_POLICY" --ledger "$BUDGET_LEDGER"
```

Every positive `--max-spend` must match its persisted allocation and supply `--budget-ledger`
and `--budget-allocation`. Clean restarts preserve charges; changing CLI caps or work directories
cannot reset them. One proxy owns the provider ledger at a time. Concurrent requests reserve
atomically before network I/O. Crashes, uncertain transport/responses, or audit failures halt
forwarding until the coordinator reconciles durable evidence. There is no automatic reservation
release, reset, top-up, or uncertain-call retry. Identical prompts remain independent, separately
charged calls after known outcomes; the proxy neither caches answers nor changes official retries.

Reservations use the full 400,000-token uncached input bound plus the explicit output limit,
including reasoning. At `max_completion_tokens=4096`, the reservation is $0.54096; validated actual
usage releases the unused portion. Calls may be refused before the allocation reaches zero.
[Official model bounds and standard prices](https://developers.openai.com/api/docs/models/gpt-5)
are the assumption. Only text Chat Completions, one choice, nonstreaming, the documented GPT-5
alias/snapshot and standard tier are admitted. OpenRouter, embeddings, tools, multimodal input,
other endpoints and unpriced models remain unsupported. Provider violations of the pinned price
or token-limit contract are outside this local guarantee.

Only the proxy receives the upstream key. Server, shim and harness child environments have
provider credentials removed; the harness uses a fresh local bearer token as its dummy OpenAI
key. Manifests contain allocation usage, holds and halt state, never provider balances or ceilings.
Existing audit logs remain append-only; historical per-call charges are summed across restarts.
`price.py` without calibration is offline. **`price.py --calibrate` is excluded from this goal and
may not consume either provider's credits, even through the proxy.**

```sh
# once
git clone https://github.com/mem0ai/memory-benchmarks /path/memory-benchmarks
git -C /path/memory-benchmarks checkout 4b61c5d31b9c668a12b4f5e78064248a02c82d2b
uv venv /path/venv && VIRTUAL_ENV=/path/venv uv pip install -r /path/memory-benchmarks/requirements.txt \
    datasets python-dateutil tiktoken          # BEAM imports datasets and dateutil; shim and price.py use tiktoken
git -C /path/warm checkout <final SHA>
CARGO_TARGET_DIR=/path/target cargo build --release --features server --bin warm-server
export OPENAI_API_KEY=...                      # never written to any log or manifest
PY=/path/venv/bin/python; H=benchmarks/public/mem0_protocol
C="--clone /path/memory-benchmarks --python $PY --warm-bin /path/target/release/warm-server"
LOCOMO=~/.cache/warm-bench/locomo/locomo10.json                      # sha256 79fa87e9...
LME=~/.cache/warm-bench/downloads/longmemeval_s_cleaned.json
BEAM=~/.cache/warm-bench/beam                                         # harness caches beam_<size>.json here

# 1. LoCoMo: retrieval only ($0), offline price, then the private final allocation
$PY $H/run_mem0_protocol.py locomo $C --work runs/locomo --data-dir /ssd/locomo --dataset $LOCOMO \
    --max-spend 0 -- --predict-only
$PY $H/price.py --clone /path/memory-benchmarks --benchmark locomo \
    --predict runs/locomo/harness-out/predicted_warm-locomo --calls runs/locomo-smoke/openai_calls.jsonl
$PY $H/run_mem0_protocol.py locomo $C --work runs/locomo --data-dir /ssd/locomo --dataset $LOCOMO \
    --max-spend "$LOCOMO_ALLOCATION_USD" --budget-ledger "$BUDGET_LEDGER" \
    --budget-allocation final-locomo -- --evaluate-only

# 2. LongMemEval-S (same three steps)
$PY $H/run_mem0_protocol.py longmemeval $C --work runs/lme --data-dir /ssd/lme --dataset $LME \
    --max-spend 0 -- --predict-only
$PY $H/run_mem0_protocol.py longmemeval $C --work runs/lme --data-dir /ssd/lme --dataset $LME \
    --max-spend "$LME_ALLOCATION_USD" --budget-ledger "$BUDGET_LEDGER" \
    --budget-allocation final-lme -- --evaluate-only

# 3. BEAM-10M. The BEAM runner's --evaluate-only only recomputes metrics,
#    so the paid run is the combined one. Do the $0 retrieval-only run first, in its own work dir,
#    for pricing and for the retrieval hash.
$PY $H/run_mem0_protocol.py beam-10m $C --work runs/beam10m-predict --data-dir /ssd/beam10m-predict \
    --dataset $BEAM --max-spend 0 -- --predict-only
$PY $H/run_mem0_protocol.py beam-10m $C --work runs/beam10m --data-dir /ssd/beam10m --dataset $BEAM \
    --max-spend "$BEAM_ALLOCATION_USD" --budget-ledger "$BUDGET_LEDGER" \
    --budget-allocation final-beam10m -- --run-id beam10m
# The combined run's retrieval_after.sha256 must equal the predict run's.
# BEAM-1M requires its own predeclared final allocation; use the beam-1m selector.
```

Each run writes `WORK/manifest-<predict|evaluate|full>-<UTC stamp>.json`. Before reporting, check:
- `audit` counts are all zero (and `retrieval_unchanged: true` on evaluate);
- `llm.blocked == 0`;
- `llm.resolved_models` shows a single snapshot.

### Recovering an incomplete paid attempt

Preserve the original failed run before recovery. LoCoMo and LongMemEval `--evaluate-only`
skip existing `cutoff_results`, so rerunning alone does not replace incomplete scored records.

`clear_failed.py WORK` supports only attempts where the manifest proves every logged model call
failed or was blocked, with no successful completion. It clears only empty cutoffs and updates
individual prediction files atomically. It refuses mixed runs: aggregate empty-answer counts do
not establish which questions failed in transport. A successful empty completion, including a
`length` termination, must remain scored WRONG. Do not delete such records or use `--rejudge`
selectively to improve an observed score.

For a mixed run, establish per-question transport provenance before any recovery; the current
proxy logs do not provide that mapping automatically. Keep the attempt marked incomplete until
that evidence exists. BEAM's `--evaluate-only` only aggregates saved results and does not generate
missing answers, so the LoCoMo/LongMemEval recovery helper does not apply to BEAM.

## Cost and prompt sizes (measured with price.py on retrieval-only output)

| Benchmark | Token-matched (primary) | Count-matched (documented, not run) |
|---|---|---|
| LoCoMo | 10.2k-token answer prompt (162 memories), **$42.6** (conv 0, 152 q, scaled to 1,540) | 12.3k (~9.1k memory), $46.6 |
| LongMemEval-S | ~7k memory plus template, **~$10** (estimate) | 87.6k, $59.5 |
| BEAM 1M | 7.2k prompt (6.3 memories, max 7.3k), **$25.4** (conv 0, 20 q, scaled to 700) | 227k mean, 242k max, $218 |
| BEAM 10M | **~$7.2** (BEAM-1M per-question cost x 200) | ~$62 |

Prices are gpt-5 list price, with output calibrated on LoCoMo calls. A BEAM chunk averages about
1.1k tokens, so 7,000 tokens hold about 6 memories.

## Optional hybrid development configuration

The protocol runner rejects embedding configurations before starting processes because embeddings bypass its chat-completion proxy. The presets remain development artifacts; they are unavailable through finite-budget protocol runs. Direct adapter smoke runs do not meter embedding spending.

`retrieval_config_hybrid_dev.json` adds OpenAI `text-embedding-3-large` embeddings at 1024
dimensions to the stored raw facts and query, with semantic2:lexical1 reciprocal rank
fusion, selected on the established development partition. The primary lexical configuration remains unchanged. Both configurations retain the
7,000-token memory cap, original timestamps, and unmodified official answerer and judge.

The embedding response is checked for count, indices, dimension, finite values and token usage.
The run manifest records the embedding configuration and usage separately from answer/judge
spend. Embeddings require `OPENAI_API_KEY`; the adapter never logs the key. This configuration
uses a hosted embedding service, so it does not represent fully local operation.
Small numerical differences between embedding API responses can affect ties; do not assume
byte-identical hybrid retrieval across fresh calls.

Hybrid fact IDs depend on exact text, event time and embedding configuration within each tenant.
A retry reads the stored fact before embedding again; concurrent retries verify the atomically
stored winner. This prevents small embedding-response differences from creating duplicate facts.

The runner resolves expected question IDs with the pinned harness's selectors and checks every
prediction and required scored cutoff. Missing, invalid, or unexpected outputs, failed ingestion,
and exhausted transport retries produce a nonzero wrapper exit even if the upstream runner exits
zero. A successful length-limited call with an empty answer and a recorded wrong score remains a
scored result; the existing empty-answer audit reports it.

Run `live_smoke.py --binary <warm-server> --config retrieval_config_hybrid_dev.json --output <new-dir>`
with the harness Python environment for synthetic ingest, timestamp, semantic retrieval and tenant
isolation checks. Development measurements must use a conversation partition declared before
examining question failures; a smoke result alone is not benchmark accuracy evidence.


### Development presentation order

`retrieval_config_ordered_dev.json` selects the same hybrid results under the same token budget,
then orders selected facts sharing an event timestamp by native ingestion time. Equal ingestion
times retain their native relevance order; missing timestamps stay in their selected slots.
Returned `score` values represent presentation rank (`1 / rank`), so the upstream Mem0 client's
score sort preserves this order before the official prompt sorts by event date. Relevance scores
still determine selection before this step. Facts remain separate with their original IDs, text
and event dates; no adjacency or relationship is inferred from a shared timestamp.

This option is appropriate only when ingestion order reflects source order. The established
LoCoMo development audit verified all1,451 facts: zero missing ingestion timestamps, inversions
or ties across1,381 adjacent same-session pairs. It has not been validated as a general ordering
rule for concurrent ingestion. The answer model, judge, prompt and completion limit are unchanged.

### Reusing frozen development grades in final validation

`reuse_locomo_dev.py` freezes the selected candidate's completed development files and protocol
manifest before final validation. It creates a separate scoring directory from fresh full-run
predictions, retaining a development grade only when the ranked memories, official question/judge
inputs and rendered answer prompt match exactly. Correct, wrong and successful empty answers use
the same rule. A mismatch is scored anew. Both original directories remain untouched, and the
prepared directory appears only after every check succeeds.

The helper checks the same shim, native binary, retrieval configuration, pinned harness and fixed
model/completion budgets. Its receipt records the required resolved model snapshot; the final
evaluation manifest must match it. It refuses incomplete/failed source runs, modified frozen
grades, already scored fresh predictions, unsafe paths and any development source outside
conversation indices0/1/2. Grade validity uses the shared `locomo_comparison.complete_grade` check.

```sh
uv run --no-project --python "$PY" python "$H/reuse_locomo_dev.py" \
    --source /selected-dev/harness-out/predicted_project \
    --source-manifest /selected-dev/manifest-evaluate-TIMESTAMP.json --freeze /results/frozen-dev.json
uv run --no-project --python "$PY" python "$H/reuse_locomo_dev.py" \
    --freeze /results/frozen-dev.json --fresh /final-predict/harness-out/predicted_project \
    --fresh-manifest /final-predict/manifest-predict-TIMESTAMP.json \
    --destination /final-evaluate/harness-out/predicted_project --clone /path/memory-benchmarks
```

### Provider-credit recovery

`quota_resume.py` prepares an independent recovery copy from a reviewed, immutable transport
audit. It verifies saved-file hashes, exact rendered prompt hashes, model settings and every
successful call association. Completed grades are retained regardless of correctness or empty
answers. A successful saved answer whose judge failed is replayed through the pinned official
scoring function with a prompt-checked cached answerer, so only the judge calls the model again.
Only proven quota failures and never-attempted questions receive fresh answer/judge calls.
Ambiguous provenance stops preparation before any destination appears. Original attempts and
all failed transport records remain unchanged. Recovery requires a reconciled, existing final
allocation; no top-up or new paid run is implied. `prepare()` requires `judge`, the pinned
`official_apply` callback and a `replay_spend_usd()` getter when judge-only cases exist. The getter
reports cumulative additional spend beyond the original manifest, including replayed judges;
the receipt subtracts both amounts from the remaining allocation.

New paid launches require the durable guard above. Historical logs used a post-spend threshold
and must not be relabeled as hard-capped evidence. Recovery callbacks must use the guarded proxy
and their existing fixed allocation; these helpers authorize no new paid run.
