# Retrieval and workload evaluation

## LongMemEval memory quality

The objective is the strongest defensible end-to-end memory quality: retrieving
necessary evidence, reasoning over time and entity identity, resolving facts
that change, and abstaining when the history cannot answer. A perfect score on
repeatedly tuned development questions does not establish general memory
capability. Retrieval recall, quotation validity, latency, and answer accuracy
are separate measurements.

A candidate promoted as a memory pipeline must run from source history and a
new query through ordinary indexing and retrieval. Benchmark identifiers,
question-type labels, expected answers, and labeled support cannot direct its
retrieval or generation. Saved answers or citations from development runs
cannot substitute for query-time retrieval; any online multi-stage reader must
charge and report every upstream stage's context and cost. Freeze the complete
policy before its final evaluation, retain every question and failure in the
denominator, and preserve failed experiments. Historical protocols below retain
their original 100/100 dev target fields; those fields are not a generalization
claim or the acceptance rule for a promoted engine.

Tune only on the frozen 100 dev questions in
[the shared S/M split](../benchmarks/public/longmemeval_split.json). Keep the
400 held-out questions sealed until the pipeline, models, prompts, budgets,
abstention rule, and scoring procedure are frozen for the single final M run.
That run cannot be used to select another variant or tune the pipeline.

The [checked v3 dev result](../benchmarks/results/longmemeval_dev_answers_v3_summary.json)
is 88/100 for the matched reader's operand arm (Wilson 95% interval
80.19–93.00%), against 83/100 for first-five and 82/100 for knapsack.
These are repeatedly used exploratory dev questions. The v3 operand arm also
uses larger evidence budgets; this is not an equal-budget comparison or a
held-out result.

The [offline quality diagnostic](../benchmarks/results/longmemeval_dev_quality_v3.json)
recomputes outcomes from saved reader and judge responses without opening the
corpus or making model calls. It retains failures in the 100-question denominator.

| Question type | Correct / total |
|---|---|
| Multi-session | 19/27 |
| Temporal reasoning | 24/26 |
| Knowledge update | 15/16 |
| Single-session assistant | 10/11 |
| Single-session user | 14/14 |
| Single-session preference | 6/6 |

The twelve operand failures comprise six judge-rejected abstentions, five
generated incorrect answers, and one reader failure. Eight failures are
multi-session questions. All six labeled unanswerable questions receive
abstentions; seven answerable-labeled questions also receive abstentions,
including one the judge accepts. Label-based abstention and judge acceptance
are therefore reported separately. Exact quotation validity does not establish
that a claim follows from its citation.

The next experiment should distinguish missing evidence, evidence removed by
the reader token cap, and incorrect use of retained evidence in the multi-session
failures. The existing completeness labels apply before that cap and only to
scored user turns. They cannot establish that the reader had sufficient support.
Temporal and fact-update changes need source-backed cases that distinguish
observation time, event time, validity, entity identity, and plans from completed
events. Changes to abstention need both answerable and unanswerable cases.

The [user-first experiment](../benchmarks/results/longmemeval_user_first_v1/protocol.json)
admits complete user turns before filling the remaining reader budget with
complete assistant continuations. Its prepared packets cover all labeled user
evidence for 82/84 scored questions, compared with 77/84 in the actual v3 reader
inputs. It uses the existing dev candidate pool and a 7,680-token local limit
inside the 8,192-token reader cap. The saved dev archives supply 24,391 verbatim
user turns; assistant continuations are limited to existing v1/v3 snapshots.
The [end-to-end run](../benchmarks/results/longmemeval_user_first_v1/summary.json)
scores **80/100**, regressing from v3's 88/100.
All 100 readers and judges completed successfully. Multi-session accuracy rises
from 19/27 to 20/27, while assistant questions fall from 10/11 to 5/11 and
preferences from 6/6 to 3/6. Preserving more labeled user turns alone is
insufficient: the cap also removes assistant-origin facts and useful context.
This variant is retained as a failed experiment rather than replacing v3.

The [session-closure selector](../benchmarks/public/longmemeval_session_closure.py)
adds the first two user turns from sessions touched by the top 20 retrieved
turns before filling from the remaining pool. Its
[offline reproduction](../benchmarks/results/longmemeval_session_closure_v1.json)
raises actual retained user support from 82/84 to 83/84 at the same local cap,
with no coverage regressions. It has not been scored end to end.

The [full-history run](../benchmarks/results/longmemeval_full_history_v1/summary.json)
evaluates all 24,391 dev user turns and 781 available saved assistant turns
with the same pinned reader and official judge. Its input budget is 131,072
tokens, with a 512-token framing reserve; prepared inputs have a median of
44,014.5 tokens and a maximum of 51,275. This is a larger-context experiment,
with additional generic entity, temporal, and personalization instructions,
and is not an equal-budget comparison. All 84 scored user-evidence cases are
complete before generation. The
[independent validation](../benchmarks/results/longmemeval_full_history_v1/validation.json)
verifies its 199 paid reader/judge calls and failure-inclusive scores. It scores
**88/100**, with five gains and five
regressions against v3 and no aggregate improvement. Knowledge updates
reach 16/16, but multi-session answers remain 19/27; preferences fall to 5/6.
One otherwise supported abstention fails the strict JSON contract because it
includes nonempty claims. The
[paired failure analysis](../benchmarks/results/longmemeval_role_failure_diagnostic_v1.json)
separates missing assistant facts, relation and event identity errors, temporal
scope errors, personalization drift, and schema failure.

The [role-aware retrieval run](../benchmarks/results/longmemeval_role_retrieval_v1/summary.json)
searches complete original assistant replies alongside the cached user ranking,
using equal-weight reciprocal rank fusion and preserving the first five user
results. It retrieves at most 64 complete exchanges under the same larger
context budget. The dev source loader verifies the pinned source-file hash and
all 500 question identifiers, but decodes only the 100 dev records; held-out
record bodies are streamed for framing and hashing without decoding or scoring.
Those dev records contain 24,391 user turns and 24,784 assistant turns, including
18 assistant-only sessions omitted from the earlier user archive. Prepared
packets contain 6,388 user turns and 6,400 assistant turns, with labeled user
support complete for 82/84 scored questions. Median input is 44,400.5 tokens,
with a maximum of 50,698. The
[independently validated result](../benchmarks/results/longmemeval_role_retrieval_v1/validation.json)
is **93/100**, improving on v3's 88/100. This is the best checked exploratory
dev result, with no held-out or production-engine claim.
All 100 reader and judge calls succeed. Knowledge updates reach 16/16, temporal
reasoning 26/26, assistant questions 11/11, user questions 14/14, and preference
questions 6/6. All seven remaining failures are multi-session questions
(20/27): ownership counts, project identity and missing premises, order identity,
wedding identity, habitual device use, payment versus event dates, and one
judge-rejected abstention. Increased assistant availability is useful but does
not establish complete answer quality. This remains exploratory repeated-use
dev evidence under a larger context budget.

The [offline history-budget measurement](../benchmarks/results/longmemeval_history_budget_dev_v1.json)
finds that complete original conversations require 150,437–168,359 input tokens
under this prompt, with a median of 158,135; none fit the 130,560-token local
limit. A generic alternative preserves every user turn and admits a ranked
prefix of complete assistant continuations: it retains 18,175 assistant turns
and all 84/84 labeled user-support cases within that limit. These are source
retention measurements without reader or judge calls.

The [complete-original-history run](../benchmarks/results/longmemeval_complete_history_v1/summary.json)
supplies all 49,175 original dev turns, including every assistant occurrence,
under a 196,608-token input cap with a 512-token framing reserve. Its prepared
inputs have a median of 158,618 tokens and a maximum of 168,842, with 84/84
labeled user-support cases complete. It uses the same pinned GPT-5.4 snapshot
with high reader reasoning and an 8,192-token output limit; the official judge
retains medium reasoning and its 2,048-token output limit. The
[model documentation](https://developers.openai.com/api/docs/models/gpt-5.4)
confirms the snapshot, reasoning setting, and context support. The protocol
also adds generic source-role, entity, relation, temporal, and personalization
bookkeeping instructions. Source availability, context budget, and reasoning
change together, so this run cannot attribute any result to one change alone.
The [independently validated result](../benchmarks/results/longmemeval_complete_history_v1/validation.json)
is **91/100**, including one reader transport failure in the denominator. There
were 99 successful readers and 99 successful judges, with no retries or answer
resampling. Its settled own-call cost is $42.118065, including the conservative
reservation for the failed response, inside its $50 cap and the unchanged $120
dev-reader/judge and $250 global caps. Complete source availability did not
outperform the 93/100 retrieval run on this repeatedly used development set.

The four-prior [independent-reader prototype](../benchmarks/public/longmemeval_verifier.py)
is **unpromoted and unscored**. Its source packets use citations from four saved
development readers, including readers given broad original history. Omitting
their answer text from the final prompt does not remove that upstream access
or make its final 32,768-token context the total query budget. Its generic
source-preservation, premise-closure, token-accounting, and transport mechanisms
are reusable; its fixed development-answer dependencies must be replaced with
ordinary query-time memory retrieval before a general-memory claim. A harmless
[API capability probe](../benchmarks/results/longmemeval_verifier_probe_v1/probe.json)
used no benchmark question. No benchmark generation or scoring has run through
this prototype. These experiments change the benchmark harness, not the
production memory engine's fact-update implementation; their measured results
do not establish production retrieval quality.

The separate opt-in [conversation memory API](memory-api/README.md)
now stores immutable original user and assistant turns through `Memory::remember_many`
and retrieves through `Memory::recall`. It preserves roles and scoped source identity,
hydrates complete originals after chunk retrieval, and admits direct conversation
premises with their anchor under explicit source and byte limits. Query decomposition
is bounded and opt-in, and a separate embedding instruction includes both source roles.
There are no saved-answer, question-ID, question-category, or benchmark-rank dependencies.

The [historical production validation](../benchmarks/results/conversation_quality_validation_2026-10-21.json)
records 788 passing tests; the [completed production proof](../benchmarks/results/conversation_production_validation_v1.json)
records **823**: the same 527 native tests, 3 bridge unit tests, and 293 current Python tests.
The native checks cover original-role evidence, confirmation premises, whole-source
budgets, immutable writes and restart, concurrency, and long-turn hydration.
[All three pinned local models passed the Metal smoke test](../benchmarks/results/conversation_bridge_model_smoke_v1/validation.json),
with [raw JSONL receipts](../benchmarks/results/conversation_bridge_model_smoke_v1/responses.jsonl).
The run uses an [isolated frozen source snapshot](../benchmarks/results/conversation_production_source_snapshot_v1.json)
and its [release build receipt](../benchmarks/results/conversation_bridge_release_build_receipt_v1.json).
Its frozen pipeline ingests complete original conversations into the ordinary memory
index, retrieves 64 distinct sources per role from a pool of 256 returned chunks per
role, and admits premises under limits of 128 sources and 65,536 serialized bytes.
The original query has weight 1, the rewrite weight 1, and all subqueries combined
weight 1; no earlier answers or
cached rankings enter retrieval. The reader is GPT-6.1 Sol with high reasoning and a 4,096-token
output limit and an exact 32,256-token input gate plus a 512-token reserve. Judging uses
the official LongMemEval judge prompt executed by pinned GPT-5.4-2026-03-05; the experiment
has a $14 cap and no held-out run. **[Production dev evaluation stopped before QA](../benchmarks/results/longmemeval_production_closeout_v1.json);
no production QA score is claimed.** These tests
and smoke checks do not establish QA accuracy or generalization. The earlier harness's
93/100 remains separate evidence, and all 400 held-out questions remain unevaluated.

External headline scores also require matching metrics. OpenExp's reported
98.6% measures session Recall@10 without generated answers, while OMEGA's
466/500 correct answers equal 93.2% per-question accuracy; its 95.4% headline
uses task averaging. These do not form a comparable QA leaderboard. The
[primary-source integrity record](../benchmarks/results/longmemeval_external_integrity_research_2026-10-21.json)
links the reports, evaluation code, development accounts, and unresolved
methodological questions. No independent untouched-split reproduction was
established by that review.

Reproduce the diagnostic and its synthetic regression checks with:

```sh
uv run --no-project python benchmarks/public/longmemeval_quality.py
uv run --no-project python -m unittest discover -s benchmarks/public -p 'test_longmemeval_quality.py'
```

## Checked-in inputs and reports

The [2026-09-14 reports](../benchmarks/results/review-2026-09-14) were produced locally on an Apple M4,
macOS arm64. The Node evaluation records Node version, model
identity, preprocessing fingerprint, artifact revisions, dtype, and dataset SHA-256. Rust
reports record engine version, dimensions, corpus size, architecture, and operating system.

- [Operational relevance fixture](../tests/fixtures/relevance.json): 12 documents and 12 distinct labeled queries.
- [Paraphrase fixture](../tests/fixtures/semantic-paraphrases.json): 10 documents and 10 queries with less direct lexical overlap.
- [Paraphrase results](../benchmarks/results/review-2026-09-14/paraphrase-relevance.json): lexical, hybrid, optional reranked relevance and latency, plus projection fidelity.
- Concurrent sparse query/update reports at [1,200](../benchmarks/results/review-2026-09-14/workload-1200.json)
  and [10,000](../benchmarks/results/review-2026-09-14/workload-10000.json) stored vectors.

These are small curated regression datasets, not externally validated benchmark collections.
The queries are distinct from the stored passages, but this is not a blind evaluation. Do not
extrapolate semantic accuracy or capacity from them. Workload timings are short local runs
with normal OS caching; they are not throughput guarantees or comparisons with the previous
implementation. Training happens before the concurrent interval. Updates invalidate the ANN
cache, so the timed workload includes fallback retrieval.

## Run document relevance without a model

```sh
cargo run --release --bin hms-eval -- \
  --dataset tests/fixtures/relevance.json --dimensions 4096 --k 3 --assert-min-recall 0.90
```

The JSON output contains document-deduplicated recall@k, MRR@k, nDCG@k, and query latency. CI
runs this fixture alongside the existing synthetic self-recall floor. Add domain documents
and independent relevance labels to test application behavior; do not merely add exact
copies of stored text as queries.

## Run real local model evaluation

Build the native module with `npm run build` and install `@huggingface/transformers` separately.
Obtain model artifacts in explicit local directories. The recorded run used these ONNX q8
artifacts:

| Role | Model | Artifact revision |
|---|---|---|
| Embedding | Xenova/all-MiniLM-L6-v2 | 751bff37182d3f1213fa05d7196b954e230abad9 |
| Reranking | Xenova/ms-marco-MiniLM-L-6-v2 | a09144355adeed5f58c8ed011d209bf8ee5a1fec |

```sh
node scripts/evaluate-local-model.mjs \
  --model-dir /path/to/minilm \
  --revision 751bff37182d3f1213fa05d7196b954e230abad9 \
  --dataset tests/fixtures/semantic-paraphrases.json \
  --reranker-dir /path/to/reranker \
  --reranker-revision a09144355adeed5f58c8ed011d209bf8ee5a1fec \
  --output /tmp/paraphrase-results.json
```

Omit both reranker options to evaluate embeddings alone. Model inference uses mean-pooled,
L2-normalized embeddings; query/document prefixes and dtype are included in the adapter's
schema revision fingerprint. The application must configure the store from `embedder.space`.
Reusing an artifact label after changing its files defeats revision checking, so keep local
artifacts immutable. See [Transformers.js local-model configuration](https://huggingface.co/docs/transformers.js/custom_usage).

The evaluation reports hit@1, hit@5, MRR@5, and p95 query latency for lexical and hybrid
retrieval, with a separate reranked result when enabled. Model loading and ingestion are
excluded from query latency; the first query is not separately warmed up. It also embeds the
same corpus into the sparse signed projection and measures its top-5 neighbor overlap against
exact dense cosine using distinct queries. That measures approximation fidelity, not semantic
relevance. Exact dense document retrieval avoids this sparse projection loss.

## Concurrent workload

```sh
cargo run --release --features cli --bin hms-workload -- --vectors 1200 --operations 300 --dimensions 4096
cargo run --release --features cli --bin hms-workload -- --vectors 10000 --operations 1000 --dimensions 4096
```

One writer replaces vectors and also deletes/reinserts every fifth ID while one reader queries
top-10 neighbors. Reports include p95/p99 latency, elapsed time, and final live count. For
capacity planning, increase the corpus and workload duration, exercise real embeddings and
metadata filters, monitor resident memory/disk use, and include compaction and maintenance.

## Distribution checks

`npm test` exercises the actual native public APIs across child-process restarts and checks
queue backpressure. `npm run test:package` packs the built application, installs the tarball
in a temporary project, and loads CJS, ESM, the native module, and the document adapter. CI
runs native/package checks on Linux, macOS, and Windows; local results establish only the
current machine's behavior. Publishing remains a separate release action.

`npm run test:types` compiles a consumer of the generated native declarations and document adapter under strict TypeScript settings, including invalid-input type checks.
