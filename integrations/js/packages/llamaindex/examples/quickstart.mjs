// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Run: node examples/quickstart.mjs   (after `npm run build` in this package; no API key or model needed)
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Document, Settings, VectorStoreIndex, storageContextFromDefaults } from 'llamaindex'
import { BaseEmbedding } from '@llamaindex/core/embeddings'
import { MockLLM } from '@llamaindex/core/llms/mock'
import { HashingEmbedder, WarmKVStore, WarmStore, WarmVectorStore } from '@writerslogic/warm-llamaindex'

// Any LlamaIndex embedding works (OpenAIEmbedding, HuggingFaceEmbedding, OllamaEmbedding...).
// This one is the dependency-free hashing embedder (lexical, not semantic).
const hashing = new HashingEmbedder(384)
class HashEmbedding extends BaseEmbedding {
  async getTextEmbedding(text) {
    return hashing.encodeQuery(text)
  }
}
const embedModel = new HashEmbedding()
Settings.embedModel = embedModel
Settings.llm = new MockLLM({ responseMessage: 'Rust uses ownership and borrowing.' }) // replace with a real LLM

const dir = mkdtempSync(join(tmpdir(), 'warm-llamaindex-example-'))
const vectorStore = await WarmVectorStore.create({
  path: join(dir, 'vectors'),
  dimensions: 4096,
  embedModel,
  embeddingSpace: { model: 'hashing', revision: 'v1' },
  lexicalWeight: 1, // HYBRID mode: BM25 weight
  semanticWeight: 1,
})

// Optional: keep the docstore and index store in WARM too (a separate store).
const kv = new WarmKVStore(new WarmStore({ path: join(dir, 'kv'), dimensions: 1024 }))
const storageContext = await storageContextFromDefaults({ vectorStore })

const index = await VectorStoreIndex.fromDocuments(
  [
    new Document({ id_: 'rust', text: 'Rust is a systems language with ownership and borrowing.', metadata: { topic: 'code', year: 2023 } }),
    new Document({ id_: 'bread', text: 'A live starter and slow fermentation make sourdough bread.', metadata: { topic: 'food', year: 2021 } }),
  ],
  { storageContext },
)

const retriever = index.asRetriever({
  similarityTopK: 2,
  filters: { filters: [{ key: 'year', operator: '>=', value: 2022 }] },
})
for (const { node, score } of await retriever.retrieve('ownership borrowing')) {
  console.log(score.toFixed(3), node.metadata.topic, '|', node.text)
}

const response = await index.asQueryEngine({ similarityTopK: 1 }).query({ query: 'What does Rust use for memory safety?' })
console.log('answer:', response.toString(), '| source:', response.sourceNodes[0].node.metadata.topic)

await kv.put('hello', { world: true })
console.log('kv:', await kv.get('hello'))
