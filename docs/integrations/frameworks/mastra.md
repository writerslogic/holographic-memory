# Mastra

Package: `@writerslogic/warm-mastra` (peer: `@mastra/core` >=1.78 <2). Built on `@writerslogic/warm-core` (see [js-core.md](js-core.md)).

`WarmVector` implements `MastraVector` (`createIndex`, `listIndexes`, `describeIndex`, `deleteIndex`, `upsert`, `query`, `updateVector`, `deleteVector`, `deleteVectors`) over the in-process native engine, so it plugs into Mastra Memory semantic recall, `MastraVector` RAG tools and anything else that takes a vector store.

```ts
import { Agent } from '@mastra/core/agent'
import { LibSQLStore } from '@mastra/libsql'
import { Memory } from '@mastra/memory'
import { WarmVector } from '@writerslogic/warm-mastra'

const vector = new WarmVector({ id: 'warm', dimension: 1536, path: './hms-data' })
const agent = new Agent({
  id: 'support', name: 'support', instructions: '...', model: 'openai/...',
  memory: new Memory({
    storage: new LibSQLStore({ id: 'storage', url: 'file:./mastra.db' }),
    vector,
    embedder: yourEmbeddingModel,
    options: { semanticRecall: true },
  }),
})
```

## Options

`id`, `dimension` (vector length of every index; default 1536; fixed for the life of the store), `path` (storage directory; omit for in-memory), `hypervectorDimensions` (engine hypervector size, default 4096), `maxFetch`, `overfetch`, `warmConfig` (native config, e.g. encryption: `{ encryptionEnabled: true, encryptionPassphraseEnv: 'WARM_PASSPHRASE' }`), `memory` (an existing native store created with embedding model `mastra-supplied`).

## Mapping

- An index is a collection inside one engine store. All indexes share `dimension`; `createIndex` with another dimension fails with a `MastraError`. Index metadata (dimension, metric, count) is persisted in a registry collection, so indexes survive restarts. Index names may not contain `/`.
- `query` returns cosine similarity as `score`, best first, with the stored metadata. Filters are applied before ranking (`topK` is not reduced by non-matching neighbours; verified past 100 candidates). With only a `filter` it is a metadata query: it returns up to `topK` matches with `score` 0.
- `includeVector` works: the embedding is kept in the record's metadata (about 20 bytes per dimension) because the engine cannot return it otherwise, and `updateVector` needs it to change only metadata.
- Text: the engine requires non-empty text per record. The record text is `metadata.text` when it is a non-empty string (Mastra Memory and the RAG chunkers set it), otherwise the id. Vectors are always the ones Mastra supplies; the store never embeds.
- `upsert` replaces by id; `deleteFilter` is applied first, in the same serialized step (not one engine transaction). Mastra Memory's message ids are stable, so re-saving is idempotent.
- `updateVector` merges `update.metadata` into existing metadata and replaces the vector when given; by `id` or by `filter`.

## Filters

Mastra's MongoDB-style syntax: `$eq $ne $gt $gte $lt $lte $in $nin $all $elemMatch $exists $regex/$options $not $and $or $nor`, dotted keys for nested fields, `Date` values (compared as ISO strings). Scalar top-level equalities are evaluated by the engine; everything else is evaluated in JS on `overfetch * topK` candidates capped at `maxFetch` (default 1000 passages), so a filter with those terms can return fewer than `topK` results when matches lie deeper than the cap. `$not` at field level and `$nor` are translated to negation of the equivalent positive filter. Date values in metadata are stored as ISO strings.

## Limits

- Metric: cosine only; other metrics fail at `createIndex`. Sparse vectors are rejected.
- `describeIndex().count` is maintained by this adapter (updated inside every write); writes made outside it (through the native handle) are not counted.
- Metadata keys starting with `_warm_` are rejected.
- One writer process per store directory. Writes inside one `WarmVector` are serialized.
- Hybrid (BM25 + vector) search is not reachable through `MastraVector.query`, which has no text parameter; use `@writerslogic/warm-core` directly for it.

## Tested against

`@mastra/core` 1.78.0, `@mastra/memory` 1.38.0, `@mastra/libsql` 1.26.2, Node 26.8. `npm test --workspace packages/mastra` runs 15 contract tests (index lifecycle, validation, cosine scores, every filter operator, pre-ranking filters past 100 candidates, replace, `deleteFilter`, update by id and filter, delete by ids and filter, index isolation) and one end-to-end test: a real `Agent` with `Memory` (LibSQL storage, `semanticRecall`, `lastMessages: 1`) driven by a scripted model, asserting that an old message is retrieved through `WarmVector` into the prompt. Mastra does not publish its vector-store test suite as a package, so the contract tests follow the operations and filter semantics in its `MastraVector` types and docs. `examples/vector.mjs` is a runnable example.
