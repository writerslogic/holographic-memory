# Vercel AI SDK

Package: `@writerslogic/warm-vercel-ai` (peer: `ai` >=7 <8, `zod`). Built on `@writerslogic/warm-core` (see [js-core.md](js-core.md)).

The AI SDK has no vector-store or memory abstraction of its own: its memory guide lists provider-defined tools, hosted memory providers and custom tools. This package is the custom-tool path plus the SDK's documented RAG middleware pattern, so it works with any model provider.

```ts
import { ToolLoopAgent, wrapLanguageModel } from 'ai'
import { WarmStore } from '@writerslogic/warm-core'
import { memoryMiddleware, memoryTools } from '@writerslogic/warm-vercel-ai'

const store = new WarmStore({ path: './memory' })
const agent = new ToolLoopAgent({
  model: wrapLanguageModel({ model: yourModel, middleware: memoryMiddleware(store, { k: 3, scope: { userId } }) }),
  tools: memoryTools(store, { scope: { userId } }),
})
```

## Exports

- `memoryTools(store, {scope, k, names, readOnly})`: `remember`, `recall`, `forget` as AI SDK `tool()`s with `inputSchema` (zod). `scope` metadata is merged into every remembered record and ANDed into every recall; `forget` only deletes records inside the scope. Tool inputs are validated by the SDK; a bad call surfaces as a `tool-error` part and never reaches the store. `metadata` and `filter` inputs are flat string/number/boolean maps (equality filters, evaluated by the engine before ranking).
- `memoryMiddleware(store, {k, filter, scope, minSemanticScore, maxChars, format, queryFrom})`: a `LanguageModelV4Middleware` whose `transformParams` recalls for the last user message and appends the memories as an extra text part of that message. Works for `generateText`, `streamText` and agents. Stored message history is not modified.
- `aiSdkEmbedder(embeddingModel, {model, revision, dimensions})`: any AI SDK embedding model as the store's encoder (uses `embed`/`embedMany`).
- `aiSdkReranker(rerankingModel)`: any AI SDK reranking model as the store's `rerank` hook (uses `rerank`).

WARM knobs (BM25 `k1`/`b`, `titleWeight`, lexical/semantic weights, fusion, encryption through `warmConfig`, collection) are options of `WarmStore`; the adapter adds none of its own.

## Limits

- Filters follow the core rules: tool filters are equality only; `memoryMiddleware({filter})` accepts the full core filter syntax, with client-side terms evaluated on `overfetch * k` candidates.
- `minSemanticScore` is a cosine threshold; the default hashing embedder is lexical, so thresholds are only meaningful with a semantic embedder.
- The middleware adds one `recall` (one embedding call if the encoder is remote) per model call, including each step of a tool loop.
- Verified against the v4 language model specification only (`specificationVersion: 'v4'`).

## Tested against

`ai` 7.0.146, `zod` 4.1.x, Node 26.8. Tests (`npm test --workspace packages/vercel-ai`) use the SDK's own `MockLanguageModelV4`, `MockEmbeddingModelV4` and `MockRerankingModelV4` from `ai/test`, a real tool loop (`generateText` with `isStepCount`), `streamText`, and the real native engine. `examples/agent.mjs` runs a `ToolLoopAgent` with both pieces offline.
