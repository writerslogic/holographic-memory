<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Memory API

`remember`, `recall`, `forget` and `explain` on top of the document engine, with the
semantics agents need: facts that hold over time, newer facts that replace older ones without
destroying them, queries "as of" a moment, and an answer to "why did this come back".

The design rule is minimal opinion. Every policy has a documented default that changes
nothing, every policy has a knob, and the low-level primitives stay reachable: a fact is an
ordinary document, so `search_documents`, `delete_document`, vectors and the raw index keep
working on a store the memory API wrote. Nothing is persisted that an older build cannot
open (see [PERSISTED-FORMAT.md](PERSISTED-FORMAT.md)).

## Quick start

Rust:

```rust
use std::sync::Arc;
use holographic_memory::core::memory_api::*;
use holographic_memory::HmsCore;

let core = Arc::new(HmsCore::new(4096, Some("./store".into()), None)?);
let memory = Memory::new(core);

memory.remember(RememberInput {
    id: Some("home-1".into()),
    text: "Alice lives in Paris".into(),
    subject: Some("alice/city".into()),
    valid_from: Some(1_577_836_800_000),
    ..Default::default()
})?;
memory.remember(RememberInput {
    id: Some("home-2".into()),
    text: "Alice lives in Berlin".into(),
    valid_from: Some(1_735_689_600_000),
    supersedes: Some(vec!["home-1".into()]),   // explicit; or use a policy, below
    ..Default::default()
})?;

let now = memory.recall("where does Alice live", &RecallOptions::default())?;   // Berlin only
let then = memory.recall("where does Alice live", &RecallOptions {
    as_of: Some(1_600_000_000_000), ..Default::default()
})?;                                                                           // Paris
let why = memory.explain("where does Alice live", "home-1", &RecallOptions::default())?;
memory.forget(&ForgetScope::Subject { subject: "alice/city".into() }, &ForgetOptions::default())?;
```

Node (`holographic-memory/memory`):

```js
const { HolographicMemorySystem } = require('holographic-memory')
const { Memory } = require('holographic-memory/memory')

const memory = new Memory(new HolographicMemorySystem(4096, './store'), { supersession: 'latest-by-subject' })
await memory.remember('Alice lives in Paris', { subject: 'alice/city', validFrom: new Date('2020-01-01') })
await memory.remember('Alice lives in Berlin', { subject: 'alice/city', validFrom: new Date('2025-01-01') })
await memory.recall('where does Alice live')                       // Berlin
await memory.recall('where does Alice live', { asOf: new Date('2021-06-01') })   // Paris
```

Python (`holographic-memory-sdk`, against `hms-server`):

```python
from holographic_sdk import HolographicClient

memory = HolographicClient("http://localhost:8080").memory
memory.remember("Alice lives in Paris", subject="alice/city", valid_from=dt.datetime(2020, 1, 1))
memory.recall("where does Alice live", as_of=dt.datetime(2021, 6, 1), explain=True)
```

## Concepts

**Fact.** A document with one chunk of text (at most 64 KiB and 4096 words) and, optionally, a
subject, entities, caller metadata, an embedding, an event time and a validity interval.
Without an id the id is derived from the content, so retrying a call stores one fact.

**Two time axes.** Valid time is when a fact holds: `valid_from` (inclusive) to `valid_to`
(exclusive), plus the `event_time` it describes. Knowledge time is when the store ingested it,
which the store records itself (`ingest_time`) from an injectable clock. `recall` evaluates
validity `as_of` a time (default: now) and can restrict to what was `known_at` an ingest time.
Event time orders supersession and drives recency; it does not by itself open or close a
validity interval. Set `valid_from` for that.

**Supersession.** A fact supersedes the facts listed in its `supersedes`, or the ones a
supersession policy picks. The link is written once, on the new fact; the old fact is not
rewritten, stays stored, and is returned by `recall` with `history` set to `all` or
`historical`, by `history(id)` and by `get(id)`. A supersession takes effect at the new fact's
valid-from (else event time, else ingest time), so `as_of` a time before it still sees the old
fact. Links to facts that were forgotten are ignored.

**Status.** Every returned fact has a status at the evaluation time: `current`, `superseded`,
`expired` (past `valid_to`) or `not-yet-valid` (before `valid_from`).

**Recall.** Document search (BM25, and cosine when an embedding is given) followed by a filter
on status. By default only current facts are returned. With nothing in the store that has a
validity window or a supersession link, and no recency policy, recall asks document search for
exactly `k` results with the options you gave and returns its scores unmodified; this is
asserted bit for bit in the tests.

**Explain.** `recall(..., explain: true)` attaches an `Explanation` to each result, and
`explain(query, id)` explains one fact, including one that did not come back. It reports the
lexical and semantic scores, the fusion and weights, the matched terms, the recency and
historical multipliers, validity at the evaluation time, the rank, and the supersession chain.
Nothing is computed unless it is asked for.

## Configuration and policies

Every field has a default that does nothing surprising. `MemoryConfig` is plain data
(camelCase JSON), so bindings and the server select built-ins by name; Rust callers can also
pass any implementation of the traits through the `with_*` builders.

| knob (`MemoryConfig`) | default | what it changes |
|---|---|---|
| `defaultK` | `10` | results per recall when `k` is absent |
| `defaultHistory` | `current` | `current`, `all` or `historical` when `history` is absent |
| `supersession` | `explicit` | `explicit`: only `supersedes` ids. `latest-by-subject`: a subject names one slot; a fact supersedes the latest earlier-or-equal one and is superseded by the earliest later one, whatever the arrival order |
| `recency` | `none` | `{kind: "exponential", halfLifeMs, basis: "event"\|"ingest", floor}` multiplies scores by `0.5^(age/halfLife)` |
| `historicalWeight` | `1` | multiplier on non-current facts when recall includes them |
| `forgetting` | `keep-all` | what `applyForgetting` removes: `expired-for {graceMs}`, `superseded-for {ageMs}`, `max-facts {limit}` |
| `consolidation` | `none` | what `consolidate` merges: `collapse-duplicates` |
| `onIdConflict` | `reject` | an id that exists with different content: `reject` or `replace` (which discards the old text) |

Rust traits, each with a documented default:

| trait | default | extension point for |
|---|---|---|
| `SupersessionPolicy` | `ExplicitOnly` | deciding links when a fact with a subject arrives |
| `RecencyPolicy` | `NoRecency` | rescoring by age, validity or anything on `RecencyContext` |
| `ForgettingPolicy` | `KeepAll` | choosing facts to remove (run by `apply_forgetting`, dry run first) |
| `ConsolidationPolicy` | `NoConsolidation` | merging facts; a plan only adds facts that supersede their sources |
| `EntityExtractor` | none | filling `entities` from text; a closure works |
| `MemoryObserver` | none | reacting to durable `remember` / `forget`, for a graph or an index |
| `Clock` | `SystemClock` | ingest time and "now"; `ManualClock` makes runs reproducible |

Entities and relationships are hooks, not a graph: the store keeps an exact-match entity
posting and nothing more. Build a graph in an observer if you want one.

Maintenance never runs by itself. `apply_forgetting` and `consolidate` take `dry_run`; the
defaults of the bindings are dry runs. A consolidation only adds a fact that supersedes its
sources (the sources stay as history) unless the policy asks to remove them *and* the call sets
`allow_destructive`.

## Forgetting

`forget` takes exactly one scope (`ids`, `subject`, `entity`, `ingested_before`,
`event_before`, `matching` metadata) and is idempotent: repeating it removes nothing more. It
removes the matched facts, and:

- without `include_chain`, only those. A removed *newest* version makes its predecessor current
  again. A removed *middle* version does not resurrect older ones: the surviving successor is
  re-linked to what the removed fact superseded;
- with `include_chain`, every fact connected to a match by supersession, in both directions, so
  no version of the chain survives;
- with `dry_run`, nothing; the report lists what would go;
- with `purge`, the store is compacted afterwards and what the compaction replaced is
  overwritten (see below).

What "gone" means: removed from search, `has_document`, `recall`, `get`, `history`, `explain`
and the in-memory index at once, and after a restart. Removal is one journal transaction per
4096 facts; a crash between transactions leaves a prefix removed and the same call finishes it.

Physical erasure, checked by `tests/purge.rs` (a marker string scanned for in every file under
the store directory, before reopen, after an abort in a child process, and after restart):

- **Always, before `forget` returns:** the deleted chunks' records in `documents.text` are
  overwritten with zeros and synced. Postings and the memory-API index live in memory and are
  rebuilt from the journal; no file stores terms. A bare `HmsCore::delete_document` zeroes the
  records too but leaves their fsync to the next `forget`, purge or compaction (a design that
  synced the text segment in every delete measured about 4 ms instead of 60 us on macOS;
  `benchmarks/results/purge_v1.json`, recorded before journal commits used `F_FULLFSYNC`). The
  journal frame that records the delete is durable when the call returns.
  `DELETE /api/v1/documents/{id}?purge=true` purges.
- **Updates:** replacing a document zeroes the old text records in `documents.text` the same
  way (synced by the next `forget`, purge or compaction); the journal keeps the old frame
  until a purge.
- **With `purge`, before it returns:** the journal is rewritten without the text, then the
  replaced generation and any unreferenced generation left by an interrupted compaction are
  overwritten with zeros, synced and unlinked.
- **Without `purge`:** the deleted frames stay in the append-only journal until a compaction.
  This is the one bounded exception; use `purge` when the text must be gone.
- **Not controlled by any file API:** copies kept by SSD wear levelling or over-provisioning,
  copy-on-write or snapshotting filesystems, filesystem journals, OS swap or hibernation
  images, backups, and bytes the caller already returned from `recall`. Use full-disk
  encryption (or `security` encryption, which makes the files ciphertext) when those matter.
  Derived data (hashed n-gram vectors in the ANN index files, fact ids) is rewritten without
  the deleted entries by the purge but is not text.

After a purge the store is indistinguishable from one that never held the forgotten facts and
was purged at the same point: the same results, ranks and score bits in lexical, hybrid and
embedding search (exact and graph), the same listings and index status, and byte-identical files
except the generation directory's name, which carries the compaction time. Lexical statistics
(document frequencies, chunk and word counts) are integers and match after any delete; the
purge rebuilds what a delete leaves shaped by history: the text segment and the slots (in id
order, as a reopen lays them out), a published vector snapshot and its graph (with the same
graph choice, so a purge costs one graph build when the snapshot has one), and the dense
document index. Compaction
writes vectors in id order. `HmsCore::revision` and the vector index status `generation`
advance by one per mutation from a random value per open, so a reading does not reveal how
many inserts and deletes came before it. Checked by `tests/forget_equivalence.rs` and by the
paired-store experiment `forget-experiment` (feature `experimental`; results in
`benchmarks/results/forget_equivalence_*.json`). Not covered:

- a delete without `purge`: until the purge the journal holds the frames, the text segment
  holds zeroed records, and a vector snapshot keeps the deleted vertex in its graph, so
  graph-mode results near the deleted vector can differ;
- decisions taken while the fact was stored: a `max-facts` eviction it caused, a
  consolidation or supersession that used it, or a chain forget that re-linked survivors;
- file timestamps, inode numbers and the generation name; encrypted bytes (same lengths,
  fresh nonces); `audit.bin` and the provenance log under `security`, which record the delete;
- retrieval over raw hypervectors (`query`, `query_vector`) and the shard IVF/NSG indexes: a
  delete drops trained shard indexes and leaves a registry slot that later inserts reuse;
- relations, atoms, composites, triples and rules, which compaction writes in their
  containers' order (not checked).

## Guarantees

Each is covered by `tests/memory_api.rs`, `tests/memory_api_server.rs` and the binding tests.

- **History is never lost by supersession.** The old fact keeps its id, text and metadata;
  `history` and `recall(history: all)` return it.
- **As-of queries are consistent.** For a subject slot, the current fact at time t is the
  latest one that had begun by t, for every arrival order (all 24 orders of four facts are
  checked), and `known_at` ignores facts and supersessions ingested later.
- **Idempotent retries.** The same id and payload reports `created: false`, writes nothing and
  keeps the first ingest time; a different payload under an existing id is a `conflict`.
- **Atomic batches.** `remember_many` stores every fact or none; one journal transaction.
- **Deterministic.** Same inputs and clock give byte-identical results across stores; ties
  keep document search's order; ids derived from content are stable.
- **Thread safe.** Writers are serialized, recall runs concurrently with them, and several
  `Memory` values over one core stay correct (a handle notices another writer by the store
  revision and rebuilds its index once).
- **Crash safe.** Mutations are journal transactions; the tests kill a writer with `abort()`
  and recover every acknowledged fact, link and removal.
- **Encrypted stores.** Facts go through the same journal and encryption as documents; there is
  no plaintext side file (`encrypted_stores_hold_facts_and_hide_text_on_disk`, feature
  `security`).

## Cost

See [Measured cost](#measured-cost) for numbers. The default path adds one metadata parse and
one map lookup per result to document search. When the store has validity windows or
supersession links, recall ranks `candidate_limit` candidates (100 by default, at most 1000
examined) instead of `k` and filters them; raise `candidate_limit` when most top candidates
are history. A subject or entity filter limits search to the matching ids and so takes the
filtered-search path.

## Measured cost

`benchmarks/results/memory_api_overhead.json` (`memory-api-bench`, release build, Apple
silicon, synthetic 24-word texts, 400 queries x 5 interleaved passes, lexical search, one
process per store kind so RSS is comparable). `raw` stores plain documents, `memory` stores
the same texts as facts, `history` stores facts of which a fifth are superseded and a fifth
expired. Latencies are mean microseconds per query; `search_documents` and `recall` run on
the same store in the same process.

| facts | RSS raw -> memory | `search_documents` | `recall` default | `recall` with explain | `recall` on `history` store |
|---|---|---|---|---|---|
| 1,000 | 12 -> 14 MB | 17.9 | 19.5 | 111.7 | 16.4 (`history` store's raw: 11.1) |
| 10,000 | 30 -> 34 MB | 64.8 | 89.8 | 214.4 | 175.0 (raw 51.1) |
| 100,000 | 182 -> 196 MB | 71.9 | 76.3 | 139.3 | 194.4 (raw 69.1) |

- Resident memory: +7.7 % at 100,000 facts (about 140 bytes per fact for the stamp, with the
  index empty). A first version stored the stamp as a metadata object and cost +84 %; the
  compact `version` placement removed that.
- Default recall returns the same ranking as `search_documents` for all 2,000 compared
  queries per size. Its latency is within measurement noise of search at 1,000 and 100,000
  facts (+9 % and +6 % mean); the 10,000 row is noisier (+25 us) and the runs are not
  isolated benchmarks, so treat the overhead as "a few percent, tens of microseconds at
  most", not as zero.
- When recall must filter (the `history` store), it ranks 100 candidates instead of `k`: about
  2.5x to 3x document search at 10,000 and 100,000 facts. That cost is the price of correct
  filtering; the raw `history` row shows document search is not slower there.
- Explanations cost about 2x to 6x a recall because they re-tokenize and read chain
  documents; they are off by default.

Retrieval sanity check, `benchmarks/results/memory_api_longmemeval_dev.json`
(`memory-api-lme`): 40 LongMemEval_S dev questions (15 knowledge-update, 25
temporal-reasoning, abstention excluded), lexical retrieval over session facts, no LLM, no
held-out id read. Default recall ranked identically to document search on 40/40 questions;
evaluating as of the question date returned no future-dated fact on 40/40, and as of just
before the newest evidence session removed that session on 40/40. Recall of an evidence
session (recall_any@5/@10): knowledge-update 1.00/1.00 for search and for as-of; exponential
recency did not beat plain search on either type (best: 1.00/1.00 at a 365-day half-life
for knowledge-update; temporal-reasoning 0.88/0.96 for search against at most 0.80/0.88
with recency). As-of the question date lowers temporal-reasoning recall to 0.76/0.84 because
this dataset dates some evidence sessions after the question date. The sample is small and
this is a sanity check, not a result: HMS does not win on temporal reasoning here with
recency or as-of, and no tuned value is claimed.

## HTTP, Node and Python

All three carry one JSON contract implemented once in `Memory::call` (`json.rs`): the camelCase
serde form of the Rust types, unknown fields rejected.

| op | request | response |
|---|---|---|
| `remember` | a fact, or `{"facts": [...]}` | `{"remembered": [...]}` |
| `recall` | `{"query", ...RecallOptions}` | `{"results": [...]}` |
| `explain` | `{"query", "id", ...RecallOptions}` | `{"explanation": {...}}` |
| `forget` | `{"scope": {"kind": ...}, "includeChain", "dryRun", "purge"}` | `{"forgotten", "dryRun", "purged"}` |
| `get` | `{"id"}` | `{"fact": {...} or null}` |
| `history` | `{"id"}` | `{"history": [...]}` |
| `applyForgetting` | `{"includeChain", "dryRun" (default true), "purge"}` | as `forget` |
| `consolidate` | `{"dryRun" (default true), "allowDestructive"}` | `{"created", "forgotten", "dryRun"}` |

`hms-server` serves them as `POST /api/v1/memory/{op}` with the same tenant header, API key and
body limits as the document routes. Errors: 422 `invalid_request`, 404 `not_found`, 409
`conflict`. Embeddings must have the server's `--input-dim` values; without one a fact is
lexical-only. Policies for every tenant come from `--memory-config <file.json>` (or
`HMS_MEMORY_CONFIG`), a `MemoryConfig` in JSON, validated at startup.

Node: `new Memory(system, config)` from `holographic-memory/memory`; times are `Date` or integer
milliseconds, results carry milliseconds, errors are `MemoryError` with `code`. Python:
`HolographicClient(...).memory` / `AsyncHolographicClient(...).memory` (or `MemoryClient`),
fully typed with a `memory.pyi` stub; times are `datetime` or integer milliseconds;
`MemoryError` carries `code` and `status`. The client-side vector mask applies to embeddings
sent through `remember` and `recall`.

## Limits

- Facts are single-chunk: text up to 64 KiB and 4096 words, metadata up to 64 KiB. Split long
  text into several facts.
- A supersession policy relates facts of one subject, so a subject should name one
  single-valued slot (`alice/city`), not a topic. Facts with no subject are linked only
  explicitly.
- A store with many superseded or expired facts relative to current ones needs a larger
  `candidate_limit` to fill `k` current results; at most 1000 candidates are examined.
- The index is derived: the first memory call after open, and the first after any write made
  behind the `Memory`'s back (raw document writes, compaction), scans the documents once.

## Original conversation evidence (Rust, opt-in)

Build with `--features conversation-memory` for `ConversationMemory`. It owns an `HmsCore`
by value; share the wrapper through `Arc` and configure the core before constructing it.
This keeps one writer responsible for its immutable session/version namespace.

```rust
use holographic_memory::core::memory_api::{
    ConversationInput, ConversationMemory, ConversationRecallOptions,
    ConversationRole, ConversationTurn,
};
use holographic_memory::HmsCore;

let core = HmsCore::new(4096, Some("./conversation-store".into()), None)?;
let memory = ConversationMemory::new(core)?;
memory.remember(ConversationInput {
    session_id: "calibration-session".into(),
    version: "v1".into(),
    turns: vec![ConversationTurn {
        source_id: "assistant-observation".into(),
        role: ConversationRole::Assistant,
        text: "The resonator calibration frequency is 37 hertz.".into(),
        observation_time: 1_704_067_200_000,
    }],
})?;
let evidence = memory.recall(
    "What calibration frequency did the assistant report?",
    &ConversationRecallOptions::default(),
)?;
```

`remember` atomically stores a complete ordered session, its manifest, one exact original
per turn, and bounded searchable chunks. Retrying identical input is idempotent; changing
an existing session/version fails. A new version preserves the earlier observations.
Roles, source IDs, turn positions, whitespace and observation timestamps survive hydration.
Assistant statements remain assistant evidence and do not become user assertions. Observation
time does not establish a fact's effective date or remove later-recorded observations.

Recall searches the actual `Memory::recall` path separately for the requested roles. It
oversamples chunks up to `candidate_limit`, collapses them to at most `k` distinct sources
per role, and interleaves the roles with equal-weight reciprocal rank fusion. Each selected
turn includes its opening user context and, for assistant turns, the nearest preceding user
turn within the same session/version. The assistant reply immediately before that user is
also retained, so confirmations can keep the recommendation they refer to. Required context
and its anchor are admitted together;
an oversized or incomplete group is reported in `dropped`. Returned originals are never
clipped. These context links preserve source premises; they do not resolve every discourse
reference or infer missing facts.

The opt-in `expand_aggregation_context` first admits candidates with their immediate
premises in score order. It then uses remaining capacity to complete admitted anchors'
transitive premise links from the exact stored session and version, including anchors
already present in the seed. This recovers earlier original turns even when their text
did not match the query. Each added closure must fit
`ConversationContextLimits.max_sources`, `max_rounds` and the compact context's exact
serialized UTF-8 `max_bytes`; otherwise the existing local context remains intact.
Candidate admission precedes enrichment so a long episode cannot displace other
candidates merely by requiring more premises. A missing stored premise fails expansion
instead of restoring deleted evidence.
The default aggregation gate still applies; set `aggregation_only: false` for other
queries. `unresolved_premise_ids()` reports remaining links when a closure does not fit,
and the context remains `unverified` rather than asserting an exhaustive answer.

`ConversationRecallOptions.max_bytes` counts the exact serialized JSON `sources` array, including provenance; it is
not a tokenizer or complete reader-request budget. Callers must count their final prompt
with the reader's tokenizer. `max_sources` also applies to context turns. Sessions accept
up to 2 MiB of raw text and 4096 stored records; each serialized source and session manifest
must fit 60 KiB. Escaping and provenance count toward that metadata limit. Invalid or
oversized input fails before any session record is written.
Searchable chunks are bounded to 1024 UTF8 bytes and 256 words, reducing truncation by the
local encoder's 2048-token window; hydration restores the complete original.

With `local-models`, configure `extract_facts: false` and set the embedder's query task to
`EmbeddingQueryTask::Conversation` before installing the stages. The constructor rejects
user-only fact extraction and embedding instructions. `StageParams::subquery_weight` enables
up to four distinct query-plan subqueries; its default is zero, preserving existing query
behavior. The configured subquery weight is shared across the admitted queries, so emitting
more queries does not multiply their total influence. Missing usable expansions fall back
to the original query when no positive retrieval weight remains. Generated time ranges do
not silently filter observation dates.

This optional API is experimental and currently Rust-only. Its retrieval regressions use
independently written conversations; no LongMemEval QA score has been established for this
production path. Existing dev harness scores are documented separately in
[evaluation.md](../evaluation.md).
