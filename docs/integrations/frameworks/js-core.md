# @holographic-memory/core (JS/TS)

Shared memory API used by the JS adapters. It runs the native engine in process (the `holographic-memory` npm package) and adds a small, stable surface: `remember`, `recall`, `get`, `list`, `forget`. It mirrors the Python core (`holographic_integrations.core`). When the native package ships its own remember/recall/forget/explain, only `HmsStore` changes.

```ts
import { HmsStore } from '@holographic-memory/core'

const store = new HmsStore({ path: './memory' })          // omit path for in-memory
await store.remember([{ id: 'a', text: 'Ada wrote the first program', metadata: { topic: 'history' } }])
const hits = await store.recall('first program', { k: 3, filter: { topic: 'history' } })
const [rec] = await store.get(['a'])                      // undefined when absent
const { records, truncated } = await store.list({ topic: 'history' })
await store.forget(['a'])
await store.close()
```

ESM and CJS are both published. `holographic-memory` is a peer dependency (`>=0.6.0`).

## API (stable)

`new HmsStore(opts)`

| Option | Default | Meaning |
|---|---|---|
| `memory` | created | An existing native `HolographicMemorySystem`; its embedding space must match `embedder` |
| `path` | in-memory | Storage directory for a store this class creates |
| `dimensions` | 4096 | Hypervector dimensions of a created store |
| `embedder` | `HashingEmbedder(384)` | The encoder. Same shape as `holographic-memory/semantic`'s `Embedder`, so `createLocalEmbedder(...)` works directly |
| `rerank` | none | `(query, passages) => Promise<number[]>`, applied to candidates |
| `collection` | none | Namespaces ids and restricts every query inside one store |
| `titleWeight`, `chunkWords`, `overlapWords` | engine defaults | Passed to document ingestion |
| `bm25K1`, `bm25B`, `lexicalWeight`, `semanticWeight`, `fusion` | engine defaults | Passed to every search |
| `overfetch` | 5 | Candidates per requested result when the filter has client-side terms |
| `maxFetch` | 1000 | Ceiling on passages per query (engine limit 1000) |
| `hmsConfig` | none | Native config of a created store, e.g. `{ encryptionEnabled: true, encryptionPassphraseEnv: 'HMS_PASSPHRASE' }` |

Methods: `remember(records): Promise<string[]>` (same id replaces), `recall(query, {k=4, filter, vector, minSemanticScore}): Promise<Hit[]>`, `get(ids): Promise<(StoredRecord|undefined)[]>`, `list(filter?, limit?): Promise<{records, truncated}>`, `forget(ids): Promise<number>`, `forgetWhere(filter?): Promise<number>`, `maintain()`, `flush()`, `close()`. Errors are `HmsError` with `code` in `invalid_argument | limit_exceeded | backend | closed`.

A `Hit` is `{id, text, metadata, score, lexicalScore, semanticScore?, span}`: the whole stored record plus the byte `span` of the best passage. `score` is the engine's fused rank score (or the reranker's), not a cosine; `semanticScore` is the cosine.

## Filters

`filter` is a MongoDB-style object (`{a: 1}`, `{a: {$gt: 1}}`, `$and`, `$or`, `$not`, `$in`, `$nin`, `$ne`, `$exists`, `$contains`, dotted keys) or the exported `Filter` AST. Top-level scalar equalities are pushed to the engine and evaluated before ranking. Everything else (comparisons, `in`, `or`, `not`, arrays/objects, dotted keys) is evaluated in JS on `overfetch * k` candidates, capped at `maxFetch`. Such filters can return fewer than `k` results even when more matches exist deeper in the ranking. `$ne` and `$nin` match records lacking the key; booleans never equal numbers.

## Behavior and limits

- Records are chunked by the engine; a record may be several passages. `recall` returns one hit per record (best passage), so it asks for twice as many passages as results. `get` and `list` reassemble the stored text from passages by byte offset. Trailing whitespace of a document is not preserved by the engine's chunker.
- `get` and `list` enumerate through a probe embedding with `documentIds`/filters, which requires an embedder (default hashing, so always satisfied). `list` returns at most `maxFetch` passages' records and reports `truncated`; page with `forgetWhere` for deletes.
- A store created without an embedder uses the hashing embedder: lexical, not semantic. Pass a real embedder for semantic recall. Its `space` is recorded in the store; reopening with another embedder fails.
- Ids: 1..256 bytes, no `hms:` prefix; metadata keys may not start with `_hms_`. A collection adds `collection/` to the stored id and `_hms_collection` to metadata.
- Writes are serialized inside one `HmsStore`; two processes must not open one path.
- Encryption: `hmsConfig.encryptionEnabled` uses the native engine's at-rest encryption (needs the `security` build, present in the published package).

## Tested against

`holographic-memory` 0.6.0 native build from this repository, Node 26.8, TypeScript 5.9, tsup 8.5. Run `npm test --workspace packages/core` in `integrations/js` (builds, then `node --test`; real native addon, one temp directory per test).
