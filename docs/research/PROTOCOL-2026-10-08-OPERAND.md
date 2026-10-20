# Operand retrieval: frozen before measurement, 2026-10-08

Evaluate only the existing 100 S dev IDs, of which 84 have scored user-turn
evidence. Keep all 400 held-out IDs untouched. No service calls, paid compute,
new embeddings, reader, judge, answer labels in selection, or parameter sweep.
The 13 diagnosed misses may inform the mechanism; this is repeatedly used dev,
not a fresh confirmation set. Diagnosis is retained in
`target/conversation-geometry-validation/missing-operands.txt` and `/tmp/missing.txt`.

Inputs: the independently validated cached tuned ranking in
`evidence_completeness_dev.json`, its summary, the pinned cleaned S release
SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`,
and the frozen split. Selection sees question text, the first 20 unique ranked
source turns, their actual source ordinals, and opaque SHA-256 identity handles.
It never sees question ID/type, gold labels, gold-derived handles, answers,
scoring flags, oracle choices, or the missing-question diagnostic.

Mechanism 1: parse explicit topic, quantity, quotation and chronology obligations;
ground them in personal declarative source sentences; construct witnessed
topic/session/date or topic/numeric-value operands. Choose full turns by newly
witnessed operand coverage, breaking ties by source ordinal then cached rank.
Fill unused bytes in cached order. This is source operand coverage, not a
minimum over embedding facets, learned geometry, or a rank-value knapsack.

At most one distinct second mechanism is allowed if the first fails the quality
gate: source-premise closure. Include a relevant session's initial source turn
as an explicit context obligation, and retain each quantitative/dated/identity
disambiguation witness; a root and its complementary witness form an atomic
bundle. Generic linguistic rules may differ from mechanism 1, but no per-ID
rules, gold-aware branch, feature fitting, or retrospective success threshold.
Record both attempts, including losses. Do not expand into further variants.

Controls: unchanged tuned first five, original anchored relevance knapsack
(the strong cached control, 68/84), rank-skip and shortest-first retained from
the validated budget report. Recompute controls rather than replacing their
scores. Gold top-20, same-byte anchored and unanchored oracles are diagnostic
only. No new ordinary knapsack proposal is permitted.

Budget: for each question, exact UTF-8 canonical v1 JSON of original first five
unique IDs, with every observed occurrence/date witness, as previously defined.
The new v2 packet contains complete source groups and `inferred: []`; all
wrapper bytes and that explicit inference boundary count against the original
budget. Evidence text is verbatim quoted user text, never a summary. No answer
is asserted: date ordering, arithmetic, absence and semantic entailment remain
inferences until checked by a reader. Operand keys are audit metadata, not a
claim of factual truth. Complete labeled support requires every full labeled
turn, including all aliases, identity, role and date. A partial gold-turn
excerpt, a matching handle without its full text, or a paraphrase counts as
failure. This conservative semantic-support check does not establish answers
or entailment (including anomalous labels noted by the diagnosis).

Success: first >=75/84, then target >=79/84, at the original per-question bytes,
and selector p95 no worse than the Rust strong knapsack control. Report complete
support against both controls, all per-question packets and hashes, paired
differences, bytes, and timing. Measure 11 warmed repetitions of both selectors
in the same persistent release process, alternating their order; retain samples
and compare p95 of per-question medians. Shared cached retrieval/input decoding
is excluded for both; extraction, selection and serialization are included.
Two independently invoked runs must reproduce selections, packet hashes and
quality; timing samples need not be identical. No end-to-end latency claim.

Independent validator uses raw pinned source, not the producer's archive or
selection code, to reconstruct all observed instances, budgets, packet hashes,
oracle outcomes and full-turn support. It must check source/selector/protocol/
split fingerprints, untouched held-out membership, recorded controls, all
aggregates and latency gates. Four separate mutations (packet hash, budget,
oracle outcome, unobserved membership) must each be rejected with nonzero exit.
Report five validation gates: input/split isolation, observed full-source
membership, exact packet hash/bytes, controls/oracles/support, reproduction and
recorded success decisions. Keep logs in
`target/conversation-geometry-validation/`.

Rust final gates: fmt check; clippy all-targets default and all-features with
warnings denied (the default configuration also exercises no-default-features);
locked tests default and all-features. Also explicitly run clippy
no-default-features with warnings denied. Format only touched Rust files.
No commit, push or delegation. Existing unrelated changes remain preserved.
Only after a verified >=75/84 result may the public benchmark and section 17
of IDEAS be edited; publish exact gates and limitations, not a superiority claim.

## Second-mechanism implementation freeze (before its measurement)

The first mechanism is independently reproduced at 59/84 (baseline 68/84),
selector p95 631,295.5 ns versus control 41,270.45 ns. Its complete report,
source code, protocol and validation are retained under
`target/conversation-geometry-validation/operand-predicate*`.

The permitted second mechanism uses personal declarative source statements to
identify relevant episode premises (actual source ordinal 1 or 2), quantity
units, date witnesses and explicitly mentioned companions. Each matched episode
premise is a coverage obligation of weight four; quantity/companion/date
witnesses have weight one. A later witness is bundled with its premise if the
premise is available. Preserve the original top turn, prioritize uncovered
obligations with ordinal/rank ties, then fill remaining bytes in cached order.
Questions without quantity, chronology or transport obligations retain cached
order. Binary earliest-event comparisons additionally retain the last observed
candidate turn of relevant episodes as a boundary; this is an observed boundary,
not a claim of complete conversation reconstruction. No gold-aware decisions.

The compact v2 packet uses `q` for full verbatim quoted source groups and `i`
for inferred answers, always empty. `{"v":2,"q":[],"i":[]}` has exactly the same
21-byte wrapper as the original v1 packet. This removes first-mechanism wrapper
penalties without reducing quotation fidelity or the inference boundary.

Source statement tokenization and exact item byte costs are built in a
question-independent prepared index before warmed queries; retain and report
this indexing time. Both selectors share that prepared source/cost index.
Question parsing, obligation construction, selection and complete packet
serialization stay inside each query timing. The control also reuses prepared
costs. This is warm selector p95 with index construction accounted separately;
it does not claim equal ingest time, total index bytes or end-to-end latency.
The original thresholds, dev isolation, source budgets and full-turn support
criteria remain fixed. No third mechanism or post-result parameter search.

## Correctness repair within mechanism 2 (before repaired measurement)

Its initial implementation scored 71/84 with warm p95 14,743.7 ns versus
35,747.25 ns; retain `operand-premise-initial.json` and its log. That result is
not promoted. Inspection found requested operand types incompletely wired:
number words and contraction suffixes were treated as topic identities, numeric
statements without repeating the topic lost their episode context, and late
date/location/route witnesses were collapsed into a single session key.

Repair those typed obligations without changing weights, thresholds, candidate
limit, budget, anchoring, tie order or controls. A premise is the earliest
observed relevant factual statement, including a later ordinal when initial
messages are generic requests. Parse number words as quantities, named question
entities as identity constraints, and source names/explicit time markers as
separate witnessed values. A self-contained companion/date/location statement
may itself fulfill its episode premise; a context-dependent quantity still
requires its atomic premise bundle. Separate verbatim named route witnesses
when transport is requested. Preserve the initial 71 result. These repairs use
already inspected dev outcomes; all final results remain exploratory repeated
use of dev, with no untouched confirmation or held-out claim.

The first witness repair also scored 71/84: four newly recovered cases were
offset by regressions. Preserve `operand-witness-initial.json` and its source.
Complete the same typed-field wiring before final measurement: explicit
quantities/time expressions themselves establish a quoted factual slot;
source purchase/sale/travel predicates must include booked/earned/sold;
a standalone dated witness fulfills both its premise and observed-date keys;
`recent publications` is not an event chronology request, whereas `most recent`
is. No weights or thresholds change. These are post-inspection implementation
repairs on repeatedly used dev, not an untouched evaluation.

The next repaired implementation scored 74/84; retain `operand-74.json`.
Final operand wiring distinguishes acquisition from later use, requested friend
relationships from other celebrations, ordinal issue numbers from cardinal
counts, and completed collections from uncompleted plans. For completed
collections, retain their last observed quantified confirmation as a boundary
witness. Completion slots must mention a requested object in a declarative
completed/finished statement. No gain weights, budgets, candidate limits or
success thresholds change. Retain all intermediate measurements; do not claim
that these dev-informed correctness repairs were selected on untouched data.

The boundary repair reaches 78/84; retain `operand-78.json`. Its remaining
regression fails to bind question `cooking` to source `baked`. Wire that ordinary
predicate synonym in the existing question parser before the final two runs.
This is a dev-informed lexical correction in mechanism 2, with the same
prospective controls, coverage weights and budgets; it is not independent
confirmation. Stop after final validation; preserve every measured iteration.

## Continuation: complete literal-field wiring before remeasurement

The user requested continuation after the independently reproduced 79/84 result.
Preserve that report, source, protocol, reproduction and validation logs under
`target/conversation-geometry-validation/operand-literal-before/`. No new
mechanism, candidate expansion, budget change, threshold change or held-out
evaluation is authorized by this continuation.

Regression fixtures reveal two incomplete fields in the existing source-premise
mechanism: punctuation prevents extraction of explicit calendar dates, and
requested literal quotations/dates/names do not themselves establish coverage.
Wire those typed fields to existing coverage with weight one and the same
anchor, premise bundles, fill order and tie rules. Literal quotation coverage
requires the exact contiguous text; identity coverage requires a complete name,
not a substring of another name. Normalize slash/dash calendar literals and
reject invalid calendar days. Distinguish dates literally asserted in source
text from observed session-date headings, including aliases. Neither witness
asserts that the observation date is the event date; inference remains explicit.
Do not change the existing source-premise weights or infer an answer.

Verify the newly wired fields with synthetic source-backed regression fixtures,
then reproduce all 100 existing dev rows twice at the original budgets. Retain
the 84-scored >=79/84 gate, strong baseline 68/84, no-worse warm p95 gate, five
independent validation gates, four rejected tamper cases and all Rust gates.
Any measured outcome is repeated-use exploratory dev evidence. Preserve the
previous 79 result if this correction fails; no further quality-driven variants.
