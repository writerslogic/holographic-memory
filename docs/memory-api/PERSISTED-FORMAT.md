<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# What the memory API persists

The memory API adds no file, no journal record type and no store-level field. A fact is an
ordinary document (`Mutation::Document`, one chunk, text stored) that carries one stamp, the
`_mem` object below. The stamp is stored in the document's `version` string as
`mem1:<compact JSON of the object>` (which keeps the per-fact memory cost to one short string;
measured in `benchmarks/results/memory_api_overhead.json`). If the compact form would exceed
the engine's 1024-byte version limit (many `supersedes` ids or entities), the same object is
stored under the reserved metadata key `_mem` instead and `version` keeps the engine default.
Readers check `version` first, then metadata. Everything else the API knows is derived from the
documents when a `Memory` first needs it.

Consequences, each covered by `tests/memory_api.rs`:

- A store written before the memory API opens unchanged; its documents are plain facts
  (current, no times, no links).
- A store with facts opens in a build without the memory API: the facts are documents whose
  metadata has an extra key.
- Journaling, replay, compaction and encryption apply to facts exactly as to documents.
  There is no plaintext side file.
- Old defaults are bit-identical. Nothing in `DocumentInput`, `SearchOptions` or
  `DocumentResult` changed, and with the default configuration recall asks document search
  for the same results with the same options and returns its scores unmodified
  (`default_recall_is_document_search_when_nothing_is_constrained` asserts bit equality).

## The `_mem` object, version 1

```json
{"v":1,"ingest":1760000000000,"event":1759000000000,"from":1759000000000,"to":1770000000000,
 "subject":"alice/city","supersedes":["fact-17"],"supersededBy":["fact-31"],
 "entities":["Alice"],"hash":"9f2c...32 hex"}
```

| field | type | meaning |
|---|---|---|
| `v` | integer | layout version; this document describes 1 |
| `ingest` | integer ms | when the store ingested the fact (knowledge time); required |
| `event` | integer ms | when the described thing happened; optional |
| `from`, `to` | integer ms | validity interval, `from` inclusive and `to` exclusive; optional |
| `subject` | string | slot key policies match on; optional |
| `supersedes` | string array | ids of facts this fact replaces; written once, on the superseding fact |
| `supersededBy` | string array | ids known at ingest to supersede this fact (a late-arriving older fact); optional |
| `entities` | string array | entities mentioned, exact match; optional |
| `hash` | string | 128-bit FNV-1a hex of the caller's payload, used to recognise an idempotent retry; not a security boundary |

Rules a reader must follow:

- Ignore unknown fields. Later versions add fields without bumping `v`; a reader built for
  version 1 treats a higher `v` as version 1 plus unknown fields. A change that alters the
  meaning of an existing field must use a new key, never reinterpret one.
- A document without `_mem`, or with a `_mem` that lacks an integer `ingest`, is a plain
  document, not an error.
- Times are signed 64-bit Unix milliseconds.
- Absent arrays are empty. Lists are written sorted.
- `supersedes` and `supersededBy` may name ids that no longer exist (the fact was forgotten).
  Readers ignore dangling ids. `supersededBy` is only honoured when the named fact still
  exists.
- Ids derived from content have the form `mem:<32 hex>` (consolidation uses `mem:merged:<32
  hex>`). The prefix `hms:` stays reserved for the engine.
- User metadata lives beside `_mem` in the same object. `_mem` is reserved; the API refuses
  user metadata that uses it.

The metadata size limit (64 KiB) applies to the whole object; the stamp costs under 300 bytes
for a fact with no lists.

## Link semantics

A supersession link `new -> old` is an edge written on `new`. It takes effect on the valid-time
axis at `new`'s effective time (`from`, else `event`, else `ingest`) and on the knowledge-time
axis at `new`'s `ingest`. Old facts are never rewritten to record it, with one exception:
forgetting a fact that sits in the middle of a chain rewrites the surviving fact that
superseded it so it now lists the removed fact's predecessors (same text, embedding, ingest
time and hash; only `supersedes`/`supersededBy` change). That keeps the removal from making an
older version current again. The rewrite and the delete are separate transactions, rewrite
first, so a crash between them leaves extra but valid links and repeating the call finishes
the removal.

## Derived state

`Memory` keeps an in-memory index of reverse links, subject groups and entity postings. It is
rebuilt from the documents (one pass) when the store's revision changed by anything other than
the `Memory`'s own tracked commit, and is never persisted. Opening a store therefore costs
nothing extra; the first memory call after open scans the documents once.

## Physical removal

`forget` deletes documents through the ordinary delete mutation. The journal is append-only, so
the deleted text remains in it until the store compacts; `ForgetOptions::purge` compacts
afterwards. The text segment file (`documents.text`) is a cache rebuilt from the journal when
the store opens, so bytes of deleted text can remain in it until the next open.
