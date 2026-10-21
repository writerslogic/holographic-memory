# LangChain.js (`@holographic-memory/langchain`)

A `VectorStore`, retriever support, and a chat message history for LangChain.js, running the native engine in process through [`@holographic-memory/core`](js-core.md). ESM and CJS, typed.

```sh
npm install @holographic-memory/langchain @langchain/core holographic-memory
```

## Quickstart

```ts
import { Document } from '@langchain/core/documents'
import { HmsVectorStore } from '@holographic-memory/langchain'

const store = await HmsVectorStore.create(embeddings, {   // any LangChain Embeddings
  path: './memory',
  embeddingSpace: { model: 'text-embedding-3-small', revision: '2024-01' },
})
await store.addDocuments([new Document({ pageContent: 'Rust\nOwnership and borrowing.', metadata: { topic: 'code' } })], { ids: ['rust'] })
const docs = await store.similaritySearch('borrowing', 4, { topic: 'code' })
const retriever = store.asRetriever({ k: 4, filter: { topic: 'code' } })
```

A runnable version with chat history is `integrations/js/packages/langchain/examples/quickstart.mjs` (no API key or model: it uses the dependency-free hashing embedder and a fake chat model).

`HmsVectorStore.fromTexts`, `fromDocuments`, `addVectors`, `similaritySearchVectorWithScore`, `similaritySearchWithScore`, `maxMarginalRelevanceSearch`, `delete`, and `asRetriever` (similarity and `mmr`) are all implemented. Extra: `getByIds(ids)`.

Chat history:

```ts
import { HmsChatMessageHistory } from '@holographic-memory/langchain'
const history = new HmsChatMessageHistory({ store: hmsStore, sessionId: 'user-1' })   // an HmsStore
```

Works with `RunnableWithMessageHistory`. `history.search(query, k)` returns the session's most relevant messages.

## Configuration (HMS knobs)

`HmsVectorStore.create(embeddings, options)` takes every `HmsStoreOptions` field of the core plus `embeddingSpace`. The encoder is the `embeddings` object you pass.

| Knob | Option |
|---|---|
| Encoder | the `embeddings` argument; `embeddingSpace` records its identity in the store (the engine refuses a reopen with a different model or revision). Default `langchain-embeddings` / `unversioned`: set both when persisting |
| BM25 | `bm25K1`, `bm25B` |
| Title weight | `titleWeight` (counts the first line of each document that many times) |
| Hybrid fusion | `lexicalWeight`, `semanticWeight`, `fusion: 'blend' \| 'rrf'` |
| Chunking | `chunkWords`, `overlapWords` |
| Encryption at rest | `hmsConfig: { encryptionEnabled: true, encryptionPassphraseEnv: 'HMS_PASSPHRASE' }` |
| Scope | `collection` (namespaces ids and queries inside one store) |
| Limits | `overfetch` (default 5), `maxFetch` (default and ceiling 1000 passages per query) |
| Existing native memory | `new HmsVectorStore(embeddings, { store })` with your own `HmsStore` |

## Search behavior

`similaritySearch` and `similaritySearchWithScore` are hybrid: they send the query text to the engine, which fuses BM25 with cosine over the vector from `embeddings`. The score is the engine's fused rank score (higher is better), not a cosine. `similaritySearchVectorWithScore(vector)` has no text, so it ranks by cosine alone. `maxMarginalRelevanceSearch` re-embeds the top `fetchK` hit texts with the store's embedder (the engine returns text, not vectors) and applies LangChain's `maximalMarginalRelevance`.

## Filters

`FilterType` is a MongoDB-style object or a core `Filter`:

| Filter | Evaluated |
|---|---|
| `{ key: scalar }` (top level, ANDed) | by the engine, before ranking |
| `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$contains`, `$or`, `$not`, nested `$and`, dotted keys, array or object equality | in JS on `overfetch * k` candidates (at most `maxFetch`) |

A filter with client-side terms can return fewer than `k` documents even when more matches exist deeper in the ranking. `k` above `maxFetch` throws `HmsError` with code `limit_exceeded`. LangChain's function-style filters are not supported. Metadata keys may not start with `_hms_`; metadata must be JSON.

`delete` takes exactly one of `{ ids }`, `{ filter }`, `{ deleteAll: true }`.

## Chat history limits

Messages are an ordered log inside the store (record `chat:<session>#<n>`), found by exact id, so a session is unbounded and does not depend on the per-query cap (tested at 99, 100, 101 and 250 messages). The serialized message is stored in metadata, or in the record text when it exceeds 32 KiB (the engine caps metadata at 64 KiB); search over such large messages then sees JSON. One writer per session: two processes appending to the same session can overwrite each other. Use a store (or `collection`) for history separate from documents.

## Not provided

- `@langchain/standard-tests` is not published on npm (404 on 2026-10-20), so the `VectorStoreIntegrationTests` suite cannot be run. `test/vectorstore.test.mjs` covers the same behaviors by hand: add with and without ids, id overwrite, delete by id, search with k and scores, filters, retriever, MMR, persistence.
- A LangChain `BaseStore` (key-value) is not offered: its `yieldKeys` needs unbounded enumeration, which the engine does not support. See [LlamaIndex.TS](llamaindex-ts.md) for a key-value store that fails loudly at the limit.
- A LangGraph.js `BaseStore` or checkpoint saver is not offered; the verified LangGraph adapter is the Python one ([langgraph.md](langgraph.md)).
- The native engine locks a path for the life of the process, so two processes cannot open one path, and reopening a path in the same process after `close()` fails until the handle is garbage collected. Tests that restart do so in a child process.

## Tested against

`@langchain/core` 1.2.19, `holographic-memory` 0.6.0 (native build from this repository), Node 26.8, TypeScript 5.9. `npm test --workspace packages/langchain` in `integrations/js` builds and runs 23 tests against the real native addon, one temp directory per test.
