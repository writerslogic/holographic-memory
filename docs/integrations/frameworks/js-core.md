# @writerslogic/warm-core (JS/TS)

Shared record API used by the JS adapters. It runs the native engine in process (the `writerslogic-warm` npm package) and adds `remember`, `recall`, `get`, `list` and `forget` with explicit collection metadata. It mirrors the Python framework core (`writerslogic_warm_integrations.core`); the native temporal memory API carries its own fact and policy contracts.

```ts
import { WarmStore } from '@writerslogic/warm-core'

const store = new WarmStore({ path: './memory' })          // omit path for in-memory
await store.remember([{ id: 'a', text: 'Ada wrote the first program', metadata: { topic: 'history' } }])
const hits = await store.recall('first program', { k: 3, filter: { topic: 'history' } })
const [rec] = await store.get(['a'])                      // undefined when absent
const { records, truncated } = await store.list({ topic: 'history' })
await store.forget(['a'])
await store.close()
```

ESM and CJS are both published. `writerslogic-warm` is a peer dependency (`>=0.6.0`).

## API (stable)

`new WarmStore(opts)`

| Option | Default | Meaning |
|---|---|---|
| `memory` | created | An existing native `WarmMemorySystem`; its embedding space must match `embedder` |
| `path` | in-memory | Storage directory for a store this class creates |
| `dimensions` | 4096 | Hypervector dimensions of a created store |
| `embedder` | `HashingEmbedder(384)` | The encoder. Same shape as `writerslogic-warm/semantic`'s `Embedder`, so `createLocalEmbedder(...)` works directly |
| `rerank` | none | `(query, passages) => Promise<number[]>`, applied to candidates |
| `collection` | none | Namespaces ids and restricts every query inside one store |
| `titleWeight`, `chunkWords`, `overlapWords` | engine defaults | Passed to document ingestion |
| `bm25K1`, `bm25B`, `lexicalWeight`, `semanticWeight`, `fusion` | engine defaults | Passed to every search |
| `overfetch` | 5 | Retained for configuration compatibility; residual filtering now selects metadata first |
| `maxFetch` | 1000 | Ceiling on passages per query (engine limit 1000) |
| `warmConfig` | none | Native config of a created store, e.g. `{ encryptionEnabled: true, encryptionPassphraseEnv: 'WARM_PASSPHRASE' }` |

Methods: `remember(records): Promise<string[]>` (same id replaces), `recall(query, {k=4, filter, vector, minSemanticScore}): Promise<Hit[]>`, `get(ids): Promise<(StoredRecord|undefined)[]>`, `list(filter?, limit?): Promise<{records, truncated}>`, `forget(ids): Promise<number>`, `forgetWhere(filter?): Promise<number>`, `maintain()`, `flush()`, `close()`. Errors are `WarmError` with `code` in `invalid_argument | limit_exceeded | filter_limit | concurrent_mutation | backend | closed`.

A `Hit` is `{id, text, metadata, score, lexicalScore, semanticScore?, span}`: the whole stored record plus the byte `span` of the best passage. `score` is the engine's fused rank score (or the reranker's), not a cosine; `semanticScore` is the cosine.

## Filters

`filter` is a MongoDB-style object (`{a: 1}`, `{a: {$gt: 1}}`, `$and`, `$or`, `$not`, `$in`, `$nin`, `$ne`, `$exists`, `$contains`, dotted keys) or the exported `Filter` AST. Top-level scalar equalities are pushed to the engine. Residual filters scan paginated metadata before ranking and pass matching document ids into the native query. Because fused scores depend on one candidate population, more than 1000 eligible records raises `filter_limit`; narrow the filter. `$ne` and `$nin` match records lacking the key; booleans never equal numbers.

## Behavior and limits

- Records are chunked by the engine; `recall` returns the best passage per record and widens its passage request when needed. If its configured passage budget prevents proving a complete result, it raises `limit_exceeded`. Reads above 1000 passages fail explicitly. The adapter retains exact whitespace omitted by native passages, the full UTF-8 length and a SHA-256 digest in reserved metadata, committed with the passages. `get`, `list` and `recall` validate source coverage and return byte-exact text, including leading/trailing whitespace. Blank input is rejected. User metadata, retained whitespace, source information and optional vectors together must fit the native 64 KiB metadata limit.
- Records written before this source contract lack evidence of the original boundary/gap bytes. Reading them raises a `backend` error asking for authoritative source reingestion; `remember` with the original text and same id replaces them. Missing historical bytes are never guessed.
- `list` and `forgetWhere` scan native metadata pages in id order, preserving collection scope and evaluating the complete filter. `list` returns at most `maxFetch` records and reports whether more matched; `forgetWhere` continues through every page. These operations require the current native `listDocuments` and `getDocumentInfo` methods.
- A store created without an embedder uses the hashing embedder: lexical, not semantic. Pass a real embedder for semantic recall. Its `space` is recorded in the store; reopening with another embedder fails. Conflicting `warmConfig` identity fields and unsupported normalization/metrics are rejected before opening persistent storage. Mislabeled existing stores require verification and reembedding from source, never blind relabeling.
- Ids: 1..256 bytes, no `hms:` prefix; metadata keys may not start with `_warm_`. A collection adds `collection/` to the stored id and `_warm_collection` to metadata.
- Writes are serialized inside one `WarmStore`; two processes must not open one path.
- `OrderedLog`, shared by the LangChain and OpenAI Agents histories, serializes operations across wrappers using the same native handle, collection and log name. Clear deletes from the tail, including within each batch, so interruption preserves a readable prefix; retry clear before appending. Retrying clear also removes orphan entries left by older versions, using metadata even when historical source bytes are unavailable. `has(id)` checks existence and collection ownership without reconstructing text.
- Encryption: `warmConfig.encryptionEnabled` uses the native engine's at-rest encryption (needs the `security` build, present in the published package).

## Tested against

`writerslogic-warm` 0.6.0 native build from this repository, Node 26.8, TypeScript 5.9, tsup 8.5. Run `npm test --workspace packages/core` in `integrations/js` (builds, then `node --test`; real native addon, one temp directory per test).
