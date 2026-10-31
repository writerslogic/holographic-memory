# LlamaIndex.TS (`@writerslogic/warm-llamaindex`)

A `BaseVectorStore` and a `BaseKVStore` (for `KVDocumentStore` and `KVIndexStore`) for LlamaIndex.TS, running the native engine in process through [`@writerslogic/warm-core`](js-core.md). ESM and CJS, typed.

```sh
npm install @writerslogic/warm-llamaindex llamaindex @llamaindex/core writerslogic-warm
```

## Quickstart

```ts
import { Document, Settings, VectorStoreIndex, storageContextFromDefaults } from 'llamaindex'
import { WarmVectorStore } from '@writerslogic/warm-llamaindex'

const vectorStore = await WarmVectorStore.create({
  path: './memory',
  embedModel: Settings.embedModel,                 // any LlamaIndex BaseEmbedding
  embeddingSpace: { model: 'text-embedding-3-small', revision: '2024-01' },
})
const storageContext = await storageContextFromDefaults({ vectorStore })
const index = await VectorStoreIndex.fromDocuments([new Document({ text: '...' })], { storageContext })
const response = await index.asQueryEngine({ similarityTopK: 3 }).query({ query: '...' })

// Reopen later without a docstore: the store keeps node text.
const again = await VectorStoreIndex.fromVectorStore(vectorStore)
```

Runnable (no API key: a mock LLM and the hashing embedder): `integrations/js/packages/llamaindex/examples/quickstart.mjs`.

Beyond the `BaseVectorStore` contract (`add`, `delete(refDocId)`, `query`, `client`) the store has `deleteNodes(ids)`, `getNodes(ids)` and `clear()`. Nodes keep their full `_node_content`, relationships and `ref_doc_id`, so `index.deleteRefDoc(id)` works through the store.

Key-value store, for a docstore or index store held in WARM too (use a separate `WarmStore` or `collection`):

```ts
import { KVDocumentStore, KVIndexStore } from 'llamaindex'
import { WarmKVStore, WarmStore } from '@writerslogic/warm-llamaindex'
const kv = new WarmKVStore(new WarmStore({ path: './kv' }))
const storageContext = await storageContextFromDefaults({ vectorStore, docStore: new KVDocumentStore(kv), indexStore: new KVIndexStore(kv) })
```

## Configuration (WARM knobs)

`WarmVectorStore.create(options)` takes every `WarmStoreOptions` field of the core, plus:

| Knob | Option |
|---|---|
| Encoder | `embedModel` (the index's embedding model) and `embeddingSpace` (identity recorded in the store; default `llamaindex-embeddings` / `unversioned`, set both when persisting). Without `embedModel`, pass `embeddingDimensions` to match the vectors you add (default 384) |
| BM25 | `bm25K1`, `bm25B` |
| Title weight | `titleWeight` |
| Hybrid weights | `lexicalWeight`, `semanticWeight`, `fusion: 'blend' \| 'rrf'` (used by `hybrid` and `semantic_hybrid` modes) |
| Chunking | `chunkWords`, `overlapWords` |
| Encryption at rest | `warmConfig: { encryptionEnabled: true, encryptionPassphraseEnv: 'WARM_PASSPHRASE' }` |
| Scope, limits | `collection`, `overfetch` (5), `maxFetch` (1000), `mmrPrefetchFactor` (4) |
| Existing store | `new WarmVectorStore({ store, embedModel })` |

## Query modes

| `VectorStoreQueryMode` | Behavior | `similarities` |
|---|---|---|
| `default` | cosine of `queryEmbedding` | cosine scores (1.0 for an identical text) |
| `hybrid`, `semantic_hybrid` | BM25 over `queryStr` fused with cosine of `queryEmbedding`; `alpha` is ignored, set the weights on the store | the engine's fused rank score |
| `mmr` | top `similarityTopK * mmrPrefetchFactor` re-ranked for diversity; `mmrThreshold` is lambda; hit texts are re-embedded with the store's embedder, so build with `create({ embedModel })` | cosine of the query to each hit |
| `sparse`, `svm`, `linear_regression`, `logistic_regression` | throw `WarmError` | n/a |

`docIds` restricts results to those node ids (client-side, on the over-fetched candidates).

## Metadata filters

All 13 `FilterOperator`s map; `MetadataFilters.condition` is `and` (default) or `or`.

| Operator | Meaning here | Evaluated |
|---|---|---|
| `==` | equal | by the engine when top-level scalars are ANDed, otherwise in JS |
| `!=`, `>`, `<`, `>=`, `<=`, `in`, `nin` | as named; `!=` and `nin` match nodes lacking the key; ordering applies to numbers and strings only | JS |
| `contains` | array metadata contains the value (or substring for strings) | JS |
| `any`, `all` | array metadata contains any, or all, of the values | JS |
| `text_match` | substring of the metadata value (not a full-text query) | JS |
| `is_empty` | key missing, `null`, `''` or `[]` | JS |

JS-evaluated filters run on `overfetch * similarityTopK` candidates (at most `maxFetch`), so they can return fewer than `similarityTopK` nodes. `similarityTopK` above `maxFetch` throws `limit_exceeded`. LlamaIndex nested filter groups do not exist in the 0.6 `MetadataFilters` type used here. Node metadata keys may not start with `_warm_`.

## Key-value store limits

`get`, `put` and `delete` are exact and unbounded; values are JSON text (large values chunk and round-trip, tested at 30,000 words). `getAll(collection)` must enumerate, and the engine returns at most `maxFetch` passages per query, so it throws `limit_exceeded` rather than return a partial map when a collection has `maxFetch` or more entries (9 entries pass and 10 and 11 throw with `maxFetch: 10`; at exactly the cap a truncated answer cannot be told from a complete one). `KVDocumentStore.docs()` and `getAllDocumentHashes()` use `getAll`, so a docstore of more than about 1000 nodes cannot list everything; the vector store itself has no such limit and needs no docstore (`storesText` is true). The KV store was compared operation by operation against `SimpleKVStore`, and `KVDocumentStore` and `KVIndexStore` against their `SimpleKVStore` behavior.

## Not provided

- No LlamaIndex.TS conformance suite is published for vector stores; `test/vectorstore.test.mjs` and `test/kv.test.mjs` cover the documented contract and use LlamaIndex's own `VectorStoreIndex.fromDocuments`, `fromVectorStore`, retriever, query engine (with `MockLLM`), `deleteRefDoc`, `KVDocumentStore` and `KVIndexStore`.
- The native engine locks a path for the life of the process; two processes cannot open one path.

## Tested against

`llamaindex` 0.12.1, `@llamaindex/core` 0.6.22 (the exact version `llamaindex` 0.12.1 depends on; two copies break its constructor checks, so keep one), `writerslogic-warm` 0.6.0 (native build from this repository), Node 26.8, TypeScript 5.9. `npm test --workspace packages/llamaindex` in `integrations/js` builds and runs 18 tests against the real native addon, one temp directory per test.
