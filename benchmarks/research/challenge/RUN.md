<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# challenge: answer-and-challenge retrieval with scoped erase-units

Standalone crate `hms-research-challenge` at `benchmarks/research/challenge/`. Nothing
outside this directory is edited. The code was written without being compiled or run (round
rules); the first command below is therefore the compile check, and the root-crate API list
at the end is there to fix anything that moved.

## Commands (run from this directory)

```sh
cd benchmarks/research/challenge
export CARGO_TARGET_DIR=/tmp/hms-research-challenge-target   # optional: avoid the shared target dir lock

# 1. compile check (the root crate builds with default features; several minutes cold)
cargo build --release

# 2. unit tests of the contract, generator, scorer and audit rules (no engine, < 1 s)
cargo test --release

# 3. smoke run: 50 families, 2 timed passes (~10-20 s; also exercises the engine ingest)
cargo run --release -- run --families 50 --repeats 2 --out out/smoke.json

# 4. the kill test, test split (held-out templates), 1,000 families x 10 records, 4,000 queries,
#    1 warmup + 5 timed passes per method, 10 families wiped and verified
cargo run --release -- run --split test --out out/results_test.json

# 5. the same on the tune split (for the template hold-out comparison; not the verdict)
cargo run --release -- run --split tune --out out/results_tune.json

# 6. dump the dataset with gold labels for inspection
cargo run --release -- generate --split test --out out/dataset_test.json

# 7. trace audit on the 100 synthetic example queries (writes the JSONL, then audits it)
cargo run --release -- audit --synthetic --examples out/trace_examples.jsonl --out out/audit_synthetic.json

# 8. trace audit on a real agent trace (REQUIRED before the audit number means anything)
cargo run --release -- audit --trace /path/to/real_agent_queries.jsonl --out out/audit_real.json
```

Options of `run`: `--split tune|test` (test), `--families N` (1000), `--conflict-fraction F`
(0.2), `--seed S` (0x5EEDC0DE20260001, decimal or 0x-hex), `--repeats R` (5), `--warmup W`
(1), `--dims D` (1024), `--store-dir DIR` (temporary directory by default), `--wipe-sample N`
(10). `--out` is required everywhere.

Exit codes: 0 PASS (or a non-judging command finished), 1 FAIL, 2 error. The last stdout line
of `run` starts with `PASS:` or `FAIL:` and repeats the numbers behind the verdict.

Expected wall clock for step 4 on a laptop: ingest of 10,000 facts in 10 batches (tens of
seconds at most; the engine encodes every fact into a 1024-d hypervector), arena rebuild
(10,000 `Memory::get` calls, about a second), then 5 methods x (1 + 5) passes x 4,000 queries.
The global BM25 baseline is the slow one (whole-store search); the rest are microseconds.

## Output files and schema

| file | schema | content |
|------|--------|---------|
| `out/results_<split>.json` | `out/results.schema.json` | dataset summary, ingest/arena timings, one report per method (correctness, answer-only accuracy, stale-answer rate, missed-exception rate, conflict handling, p50/p95/p99 latency with per-pass p95s, breakdowns by query kind and family type), challenge-pass statistics, erase check, SQLite note, verdict |
| `out/dataset_<split>.json` | serde of `synth::Dataset` | every family with its 10 records (id, seq, kind, scope, value, text, validity, supersedes, target) and every query with its gold |
| `out/trace_examples.jsonl` | one `audit::TraceQuery` per line | the 100 synthetic example queries |
| `out/audit_*.json` | `out/audit.schema.json` | keyed-by-metadata / keyed-by-rule / unkeyed counts, usable fraction, pass at 50%, rule histogram, samples |

## Pass / fail

`verdict.pass` is true when both hold:

1. `challenge.correctness - max(correctness of the four non-challenge baselines) >= 0.10`
   (absolute, i.e. 10 points). `strongest_baseline` names which baseline set the bar.
2. `challenge.latency.p95_us <= 2.0 x min(p95 of scoped_bm25_latest_first_engine,
   scoped_bm25_latest_first_local)`. The faster scope-filtered BM25 variant is the reference,
   which is the conservative reading.

Correctness is answer-and-evidence correctness at a 4-item evidence budget: a resolved gold
needs the gold primary as the answer and every id of `gold.required_evidence` among the
returned items; an unresolved gold needs status `UnresolvedConflict` with exactly the gold
branches. Every baseline returns a ranking (status `Resolved`, first claim as the answer), so a
baseline is always wrong on an unresolved gold. `answer_only_accuracy` is reported alongside
so the evidence condition's share of the margin is visible. The erase check
(`erase.verified`) is reported, not part of the verdict.

The trace audit passes at `usable_fraction >= 0.5`. On the synthetic examples the number is
meaningless by construction (`synthetic: true`, and the note says so); a real agent trace must
replace them.

## What the mechanism is

`src/contract.rs` (module docs have the full statement): one `(entity, attribute)` family is a
contiguous block. A query carries the key, a scope (qualifier map), an `as_of` time and text.

- Support pass: BM25-lite over the family's *current* claims whose scope covers the query
  scope; the top is the lexical winner.
- Challenge pass, four exact lookups on the same block, similarity-independent:
  (a) a newer or more specific current claim covering the query replaces the lexical winner
  (precedence: specificity, then effective time, then sequence); (b) a retraction voids its
  target entirely, including the supersession links the target carried, so what it replaced
  stands again; (c) a current narrower-scope claim compatible with but not covered by the
  query scope is a branch; (d) current claims at the winner's specificity and effective time
  with no supersession between them are concurrent branches.
- More than one branch means `UnresolvedConflict` with every branch returned. Otherwise
  `Resolved` with the primary, the strongest defeater (the latest valid compatible claim a
  recency-first retriever would have returned, with why it lost: retracted / superseded /
  less specific), what defeated it, and what the primary explicitly replaced, within 4 items.

Two contract decisions a reader may dispute, both documented in the code and applied
identically by the generator's gold: a retraction voids its target's supersession edges
(a withdrawn observation never replaced anything), and a wider claim never overrides a
narrower one by recency alone (only an explicit `supersedes` marker does).

## Storage: what the memory API expresses and what it does not

Every record is stored through the memory API (`src/store.rs`): `subject` = `entity/attribute`
(so `ForgetScope::Subject` erases a family in one call), `valid_from`/`valid_to`,
`supersedes` for explicit markers (a retraction supersedes its target), `entities = [entity]`
and the structured caller metadata every method reads. The challenge index is an in-crate
arena of one block per family rebuilt from `Memory::get` records, because the memory API
cannot express the challenge lookups or the family-contiguous placement:

- its metadata filter is exact top-level equality, `src/core/memory_api/types.rs:98-99`
  (`/// Exact top-level metadata matches, as in document search.`), passed straight to
  document search at `src/core/memory_api/recall.rs:271`; no scope cover/compatibility, no
  latest-per-scope, no retraction-target lookup;
- its subject index is a flat member list `by_subject: HashMap<String, Vec<Member>>`
  without scope, `src/core/memory_api/index.rs:41` and `:57-61`;
- supersession edges are permanent and a superseded fact stays superseded whatever happens to
  the superseder, `src/core/memory_api/index.rs:83-97` and `recall.rs:34-39`
  (`evaluate`: any edge with `effective <= now` makes the fact `Superseded`), so "retraction
  restores the predecessor" is not expressible there;
- facts are ordinary documents keyed by id with no family-contiguous layout,
  `docs/memory-api/PERSISTED-FORMAT.md` ("A fact is an ordinary document ... Everything else
  the API knows is derived from the documents").

The arena is derived state in the sense of `index.rs` itself: rebuilt from the records, never
persisted. Erasing a family = drop the block + `Memory::forget(ForgetScope::Subject)`;
`erase.verified` checks the family is gone from both and that the next family's answers did
not change.

## Baselines (all read the same metadata)

| name | definition |
|------|------------|
| `global_bm25_top4` | `HmsCore::search_documents(query_text, k = 4)` over the whole store |
| `scoped_bm25_latest_first_engine` | `Memory::recall(query_text, subject = family, as_of, history = All, k = 64)`, keep hits with BM25 score > 0 that are valid as of the query and scope-compatible, order latest-first, top 4 |
| `scoped_bm25_latest_first_local` | the same over the arena block with the in-crate BM25-lite (controls for engine overhead in the latency ratio) |
| `scoped_lookup_no_challenge` | claims valid as of the query whose scope covers the query scope, most specific then latest, top 4; no retraction / supersession / tie handling |
| (d) SQLite | **skipped**: `rusqlite` is not a dependency of the root crate (`Cargo.toml` `[dependencies]`, lines 19-58) and the round forbids new third-party crates. The dataset dump (`generate`) has everything a later SQLite implementation needs. |

## Dataset (`src/synth.rs`)

1,000 families = 250 entities (`acct000`..`acct249`) x 4 of 10 attributes; 10 observations per
family at daily slots; 20% of families (by seeded draw; the actual count is in
`dataset.conflict_families`) use the conflict layout. Per family: 8 defaults (D1..D8, global
scope, chained by explicit `supersedes`), 1 exception at `{dim: val}` (dim from region / env /
tier / channel / plan), 1 retraction (of D8 in the normal layout, of the already superseded
D6 in the conflict layout). In the conflict layout D7 and D8 share `valid_from` and both
supersede D6, neither the other. Values are drawn so D7, D8 and the exception differ. Four
queries per family (`in_scope`, `out_of_scope`, `no_scope`, `as_of_mid`), 4,000 in all; half of
`in_scope` queries carry an extra unrelated qualifier (the exact tuple does not exist; a
covering lookup is needed), half of `no_scope` queries carry only an unrelated dimension, half
of all queries mention their scope in the text. Gold is computed from the construction (which
record is the exception, which is retracted, which pair is concurrent), not by running the
contract. Templates: `tune` and `test` pools share no template for claims, exceptions,
retractions, queries or scope hints; the structural draw is identical across splits.

Expected shape of the result (reasoning, not a measurement): `scoped_lookup_no_challenge` is
the strongest baseline and should land near 25-30% correctness (right on `as_of_mid` of normal
families and on `in_scope` of conflict families, wrong wherever the retraction, the ambiguity
or the concurrency matters); the BM25 baselines lower; the challenge method near 100% if the
implementation matches the gold. The latency condition is the one with teeth: the arena
lookups are O(10) per query and should sit well under the engine recall's p95; the local BM25
variant is the honest reference for that comparison.

## Root-crate APIs used (verbatim locations in `src/`)

| API | location |
|-----|----------|
| `HmsCore::new(dimensions: u32, storage_path: Option<String>, config: Option<HmsConfig>) -> Result<Self>` | `src/core/engine/mod.rs:114-118` |
| `HmsCore::search_documents(&self, text: &str, options: &SearchOptions) -> Result<Vec<DocumentResult>>` | `src/core/engine/document_api.rs:101-105` |
| `SearchOptions { k, candidate_limit, filter, source_uri, document_ids, embedding, lexical_weight, semantic_weight, min_semantic_score, fusion, bm25_k1, bm25_b }` (Default) | `src/core/documents.rs:84-103` |
| `DocumentResult { id, document_id, source_uri, version, metadata, text, start_byte, end_byte, score, lexical_score, semantic_score }` | `src/core/documents.rs:162-174` |
| re-exports `holographic_memory::{HmsCore, SearchOptions, DocumentResult}` | `src/lib.rs:26-28, 39` |
| `holographic_memory::core::memory_api` (`pub use policy::*; pub use types::*;`) | `src/core/mod.rs:76`, `src/core/memory_api/mod.rs:65-67` |
| `Memory::new(core: Arc<HmsCore>) -> Self` | `src/core/memory_api/mod.rs:182` |
| `Memory::with_clock(self, clock: Arc<dyn Clock>) -> Self` | `src/core/memory_api/mod.rs:264` |
| `Memory::core(&self) -> &Arc<HmsCore>` | `src/core/memory_api/mod.rs:275` |
| `trait Clock { fn now_millis(&self) -> Millis }`, `ManualClock::new(Millis)`, `ManualClock::set(&self, Millis)` | `src/core/memory_api/policy.rs:15-17, 38-44` |
| `Memory::remember_many(&self, inputs: Vec<RememberInput>) -> Result<Vec<Remembered>>` (batch cap `MAX_BATCH = 4096`) | `src/core/memory_api/write.rs:88`, `:16` |
| `RememberInput { id, text, event_time, valid_from, valid_to, subject, supersedes, entities, metadata, embedding, source_uri }` (Default, `deny_unknown_fields`) | `src/core/memory_api/types.rs:14-40` |
| `Remembered { id, created, supersedes, superseded_by, ingest_time }` | `src/core/memory_api/types.rs:45-54` |
| `Memory::get(&self, id: &str) -> Result<Option<FactRecord>>` | `src/core/memory_api/recall.rs:187` |
| `FactRecord { id, text, status, ingest_time, event_time, valid_from, valid_to, effective_from, subject, entities, supersedes, superseded_by, metadata }` | `src/core/memory_api/types.rs:190-205` |
| `Memory::recall(&self, query: &str, options: &RecallOptions) -> Result<Vec<Recalled>>` | `src/core/memory_api/recall.rs:99` |
| `RecallOptions { k, as_of, known_at, history, subject, entities, filter, embedding, lexical_weight, semantic_weight, min_semantic_score, fusion, candidate_limit, explain }` (Default) | `src/core/memory_api/types.rs:84-111` |
| `HistoryMode::All` | `src/core/memory_api/types.rs:77` |
| `Recalled { id, text, score, status, ingest_time, event_time, valid_from, valid_to, subject, entities, supersedes, superseded_by, metadata, source_uri, explanation }` | `src/core/memory_api/types.rs:116-134` |
| `Memory::forget(&self, scope: &ForgetScope, options: &ForgetOptions) -> Result<ForgetReport>` | `src/core/memory_api/write.rs:424` |
| `ForgetScope::Subject { subject }`, `ForgetOptions { include_chain, dry_run, purge }` (Default), `ForgetReport { forgotten, dry_run, purged }` | `src/core/memory_api/types.rs:214, 227-237, 241-246` |
| `tempfile::tempdir()` (root dependency `tempfile = "3"`) | `Cargo.toml:52` |

Subject restriction in `recall` goes through `Index::subject_ids` to `SearchOptions::document_ids`
(`src/core/memory_api/recall.rs:349-366, 267-278`); facts are stored with `store_text:
Some(true)` and one chunk (`write.rs:388-399`), so `FactRecord::text` and
`DocumentResult::document_id` are populated as this crate assumes.

## Dependencies

`holographic-memory` (path, default features), `serde`, `serde_json`, `anyhow`, `tempfile`:
every crates.io dependency is already a dependency of the root crate at the same version
(`Cargo.toml:24, 27, 35, 52`). No new third-party crate. No `Cargo.lock` is committed.

## Trace format for the audit (`src/audit.rs`)

JSONL, one object per line, unknown fields ignored:

```json
{"id": "q-17", "text": "What is acct017's payment terms?", "entity_id": "acct017",
 "attribute_key": "payment_terms", "scope": {"region": "eu"}, "as_of": 1700950400000}
```

`text` is required. A query is keyed by metadata when `entity_id` and `attribute_key` are
both present; otherwise six template rules run on the lowercased text (`kv`, `possessive`,
`attribute-of-entity`, `attribute-for-entity`, `entity-colon-attribute`,
`identifier-then-attribute`; the module docs give the patterns). The 100 synthetic examples
cover metadata-keyed queries, every rule and free-form text; they only prove the pipeline
runs. **Replace them with a real agent trace before reading the audit's pass/fail.**

## Open risks

- Not compiled. Likely trouble spots if any: trait-object coercion in
  `Memory::new(core).with_clock(clock.clone())` (the root test `tests/memory_api.rs:23` does the
  same), `serde(flatten)` on the tagged `Defeat`/`KeySource` enums (serialize-only paths in the
  reports), and closure type inference in `eval::evaluate`.
- The scope-filtered engine baseline depends on `recall` returning every lexically matching
  family record with `k = 64`; `recall` with `history = All` asks document search for exactly
  `k` (`recall.rs:253-265`), which is more than the 10 records of a family.
- Latency of the engine baselines includes the engine's own search cost; the local BM25 variant
  exists so the verdict's ratio cannot be won on engine overhead alone.
- Gold and contract agree by design on the two semantic decisions above; if the panel rejects
  either, `synth::realize`'s `gold_for` and `contract::Family::answer` both need the change.
