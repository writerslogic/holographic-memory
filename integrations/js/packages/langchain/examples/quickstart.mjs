// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Run: node examples/quickstart.mjs   (after `npm run build` in this package; no API key or model needed)
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Document } from '@langchain/core/documents'
import { FakeListChatModel } from '@langchain/core/utils/testing'
import { RunnableWithMessageHistory } from '@langchain/core/runnables'
import { HashingEmbedder, WarmChatMessageHistory, WarmStore, WarmVectorStore } from '@writerslogic/warm-langchain'

// Any LangChain Embeddings works. This one is the dependency-free hashing embedder (lexical, not semantic);
// swap in OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceTransformersEmbeddings, ...
const hashing = new HashingEmbedder(384)
const embeddings = { embedDocuments: (t) => hashing.encodeDocuments(t), embedQuery: (t) => hashing.encodeQuery(t) }

const path = mkdtempSync(join(tmpdir(), 'warm-langchain-example-'))
const vectorStore = await WarmVectorStore.create(embeddings, {
  path,
  dimensions: 4096,
  embeddingSpace: { model: 'hashing', revision: 'v1' }, // identity of the embedding space, recorded in the store
  titleWeight: 2, // count a document's first line twice in BM25
  fusion: 'blend',
})

await vectorStore.addDocuments(
  [
    new Document({ pageContent: 'Rust\nRust is a systems language with ownership and borrowing.', metadata: { topic: 'code', year: 2023 } }),
    new Document({ pageContent: 'Sourdough\nA live starter and slow fermentation make sourdough bread.', metadata: { topic: 'food', year: 2021 } }),
    new Document({ pageContent: 'Borrow checker\nThe borrow checker enforces aliasing rules at compile time.', metadata: { topic: 'code', year: 2024 } }),
  ],
  { ids: ['rust', 'bread', 'borrow'] },
)

console.log('similarity:', (await vectorStore.similaritySearch('ownership and borrowing', 2)).map((d) => d.id))
console.log('filtered  :', (await vectorStore.similaritySearch('language', 3, { topic: 'code', year: { $gte: 2024 } })).map((d) => d.id))

const retriever = vectorStore.asRetriever({ k: 2, filter: { topic: 'code' } })
console.log('retriever :', (await retriever.invoke('compile time rules')).map((d) => d.id))

// Chat memory: RunnableWithMessageHistory with a store-backed history per session.
const historyStore = new WarmStore({ path: mkdtempSync(join(tmpdir(), 'warm-langchain-history-')), dimensions: 1024 })
const chain = new RunnableWithMessageHistory({
  runnable: new FakeListChatModel({ responses: ['Nice to meet you, Ada.', 'You told me your name is Ada.'] }),
  getMessageHistory: (sessionId) => new WarmChatMessageHistory({ store: historyStore, sessionId }),
})
const config = { configurable: { sessionId: 'demo' } }
await chain.invoke([{ role: 'user', content: 'Hi, I am Ada.' }], config)
await chain.invoke([{ role: 'user', content: 'What is my name?' }], config)
const history = new WarmChatMessageHistory({ store: historyStore, sessionId: 'demo' })
console.log('history   :', (await history.getMessages()).map((m) => `${m.getType()}: ${m.content}`))
