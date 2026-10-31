# OpenAI Agents SDK, JS (`@writerslogic/warm-openai-agents`)

A `Session` and a set of memory tools for the OpenAI Agents SDK for JavaScript/TypeScript (`@openai/agents`), running the native engine in process through [`@writerslogic/warm-core`](js-core.md). ESM and CJS, typed.

```sh
npm install @writerslogic/warm-openai-agents @openai/agents zod writerslogic-warm
```

## Quickstart

```ts
import { Agent, run } from '@openai/agents'
import { WarmSession, WarmStore, createMemoryTools } from '@writerslogic/warm-openai-agents'

const store = new WarmStore({ path: './memory' })
const session = new WarmSession({ store, sessionId: 'ada-chat' })        // conversation memory
const tools = createMemoryTools(store, { scope: { user: 'ada' } })      // long-term memory the model can call

const agent = new Agent({ name: 'assistant', instructions: 'Use memory tools to remember and recall facts.', tools })
await run(agent, 'Remember that I like dark roast coffee.', { session })
```

Runnable without an API key (a scripted model stands in for the LLM): `integrations/js/packages/openai-agents/examples/quickstart.mjs`.

## Session

`WarmSession implements Session`: `getSessionId`, `getItems(limit?)` (latest `limit` items, oldest first, like the SDK's `MemorySession`), `addItems`, `popItem`, `clearSession`. Verified by running the same operation sequence against `MemorySession` and `WarmSession` and comparing results, then with a real `Runner` over a scripted `Model`, where the history reaches the model on the second run. Extra: `search(query, k)` returns the session's items most relevant to a query (hybrid BM25 and cosine).

History is an ordered log inside the store (record `session:<id>#<n>`) read by exact id, so a session is unbounded and independent of the per-query result cap (tested at 99, 100, 101 and 205 items). Each item is stored as JSON, in metadata or, above 32 KiB (the engine caps metadata at 64 KiB), in the record text (tested with a 140 KB item); search over those large items then sees JSON. Items round-trip through JSON, so `undefined` fields (for example `providerData: undefined`) come back absent. One writer per session id: two processes appending to the same session can overwrite each other. The optional Session extensions (`applyHistoryMutations`, `applyHistoryTransaction`, compaction) are not implemented; the SDK works without them.

## Memory tools

`createMemoryTools(store, options)` returns three `tool()`s: `memory_remember({ text, metadata? })`, `memory_recall({ query, limit?, filter? })`, `memory_forget({ id })`.

| Option | Meaning |
|---|---|
| `prefix` | Tool name prefix (default `memory_`) |
| `k`, `maxK` | Default and maximum recall results (4 and 20) |
| `scope` | Tags added to every note and required by recall and forget, so one store serves many users. A tool cannot read or delete another scope's notes |

`filter` accepts exact-match tags (string, number, boolean), evaluated by the engine before ranking.

## Configuration (WARM knobs)

Pass any `WarmStoreOptions` of the core to `new WarmStore(...)`:

| Knob | Option |
|---|---|
| Encoder | `embedder` (default: dependency-free hashing, lexical; `createLocalEmbedder(...)` from `writerslogic-warm/semantic` works directly) |
| BM25 | `bm25K1`, `bm25B` |
| Title weight | `titleWeight` |
| Hybrid | `lexicalWeight`, `semanticWeight`, `fusion` |
| Encryption at rest | `warmConfig: { encryptionEnabled: true, encryptionPassphraseEnv: 'WARM_PASSPHRASE' }` |
| Scope, limits | `collection`, `overfetch`, `maxFetch` |

## Limits

- Use a store (or `collection`) for sessions separate from long-term notes if you want clean `list` results.
- The native engine locks a path for the life of the process; two processes cannot open one path.
- Recall score is the engine's fused rank score, not a cosine; the tool returns ids and text only.

## Tested against

`@openai/agents` 0.20.0 (`@openai/agents-core` 0.20.0), `zod` 4.6.7, `writerslogic-warm` 0.6.0 (native build from this repository), Node 26.8, TypeScript 5.9. `npm test --workspace packages/openai-agents` in `integrations/js` builds and runs 11 tests against the real native addon, one temp directory per test.
