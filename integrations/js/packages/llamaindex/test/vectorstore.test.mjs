// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { Settings, VectorStoreIndex, storageContextFromDefaults } from 'llamaindex'
import { BaseEmbedding } from '@llamaindex/core/embeddings'
import { MockLLM } from '@llamaindex/core/llms/mock'
import { TextNode, Document, MetadataMode } from '@llamaindex/core/schema'
import { VectorStoreQueryMode } from '@llamaindex/core/vector-store'
import { HashingEmbedder, WarmError, WarmVectorStore } from '../dist/index.js'

const hashing = new HashingEmbedder(256)
class HashEmbed extends BaseEmbedding {
  constructor() {
    super()
  }
  async getTextEmbedding(text) {
    return hashing.encodeQuery(text)
  }
}
const embedModel = new HashEmbed()
Settings.embedModel = embedModel
Settings.llm = new MockLLM({ responseMessage: 'mock answer' })

const tmp = () => mkdtempSync(join(tmpdir(), 'warm-li-'))
const make = (opts = {}) => WarmVectorStore.create({ path: tmp(), dimensions: 1024, embedModel, ...opts })

async function embeddedNodes(specs) {
  const nodes = specs.map(([id, text, metadata]) => new TextNode({ id_: id, text, metadata }))
  const vectors = await embedModel.getTextEmbeddings(nodes.map((n) => n.getContent(MetadataMode.NONE)))
  nodes.forEach((n, i) => (n.embedding = vectors[i]))
  return nodes
}
const q = async (text, extra = {}) => ({ queryStr: text, queryEmbedding: await embedModel.getQueryEmbedding({ type: 'text', text }), similarityTopK: 5, mode: VectorStoreQueryMode.DEFAULT, ...extra })

test('VectorStoreIndex.fromDocuments through a storage context, then retrieve and query', async () => {
  const vs = await make()
  const storageContext = await storageContextFromDefaults({ vectorStore: vs })
  const docs = [
    new Document({ id_: 'cats', text: 'Cats are small carnivorous mammals kept as pets.', metadata: { topic: 'animals', year: 2020 } }),
    new Document({ id_: 'rust', text: 'Rust is a systems programming language focused on memory safety.', metadata: { topic: 'code', year: 2023 } }),
    new Document({ id_: 'bread', text: 'Sourdough bread needs a live starter and a long fermentation.', metadata: { topic: 'food', year: 2021 } }),
  ]
  const index = await VectorStoreIndex.fromDocuments(docs, { storageContext })
  const retriever = index.asRetriever({ similarityTopK: 2 })
  const found = await retriever.retrieve('programming language memory safety')
  assert.equal(found.length, 2)
  assert.equal(found[0].node.metadata.topic, 'code')
  assert.equal(found[0].node.getContent(MetadataMode.NONE), 'Rust is a systems programming language focused on memory safety.')
  assert.ok(found[0].score > found[1].score)
  const engine = index.asQueryEngine({ similarityTopK: 1 })
  const response = await engine.query({ query: 'sourdough starter fermentation' })
  assert.equal(response.toString(), 'mock answer')
  assert.equal(response.sourceNodes[0].node.metadata.topic, 'food')
})

test('VectorStoreIndex.fromVectorStore reads an already-populated store (no docstore needed)', async () => {
  const vs = await make()
  await vs.add(await embeddedNodes([['a', 'quantum field theory of gluons', { k: 1 }], ['b', 'recipe for tomato soup', { k: 2 }]]))
  const index = await VectorStoreIndex.fromVectorStore(vs)
  const [top] = await index.asRetriever({ similarityTopK: 1 }).retrieve('gluons quantum field')
  assert.equal(top.node.id_, 'a')
  assert.equal(top.node.metadata.k, 1)
})

test('default mode similarities are cosine scores; identical text scores 1', async () => {
  const vs = await make()
  await vs.add(await embeddedNodes([['a', 'alpha beta gamma', {}], ['b', 'delta epsilon zeta', {}]]))
  const res = await vs.query(await q('alpha beta gamma'))
  assert.equal(res.ids[0], 'a')
  assert.ok(Math.abs(res.similarities[0] - 1) < 1e-6, String(res.similarities[0]))
  assert.ok(res.similarities[1] < res.similarities[0])
})

test('every FilterOperator maps to the right documents', async () => {
  const vs = await make()
  await vs.add(
    await embeddedNodes([
      ['n1', 'alpha one', { year: 2019, lang: 'en', tags: ['a', 'b'], title: 'Intro to Rust' }],
      ['n2', 'alpha two', { year: 2021, lang: 'de', tags: ['b', 'c'], title: 'Cooking basics' }],
      ['n3', 'alpha three', { year: 2023, lang: 'en', tags: ['c'], title: 'Rust async' }],
      ['n4', 'alpha four', { lang: 'fr', tags: [] }],
    ]),
  )
  const ids = async (filters, condition) =>
    (await vs.query({ ...(await q('alpha')), similarityTopK: 10, filters: { filters, ...(condition ? { condition } : {}) } })).ids.sort()
  const f = (key, operator, value) => ({ key, operator, value })
  assert.deepEqual(await ids([f('lang', '==', 'en')]), ['n1', 'n3'])
  assert.deepEqual(await ids([f('lang', '!=', 'en')]), ['n2', 'n4'])
  assert.deepEqual(await ids([f('year', '>', 2019)]), ['n2', 'n3'])
  assert.deepEqual(await ids([f('year', '>=', 2021)]), ['n2', 'n3'])
  assert.deepEqual(await ids([f('year', '<', 2021)]), ['n1'])
  assert.deepEqual(await ids([f('year', '<=', 2021)]), ['n1', 'n2'])
  assert.deepEqual(await ids([f('lang', 'in', ['de', 'fr'])]), ['n2', 'n4'])
  assert.deepEqual(await ids([f('lang', 'nin', ['de', 'fr'])]), ['n1', 'n3'])
  assert.deepEqual(await ids([f('tags', 'contains', 'c')]), ['n2', 'n3'])
  assert.deepEqual(await ids([f('tags', 'any', ['a', 'c'])]), ['n1', 'n2', 'n3'])
  assert.deepEqual(await ids([f('tags', 'all', ['b', 'c'])]), ['n2'])
  assert.deepEqual(await ids([f('title', 'text_match', 'Rust')]), ['n1', 'n3'])
  assert.deepEqual(await ids([f('year', 'is_empty')]), ['n4'])
  assert.deepEqual(await ids([f('tags', 'is_empty')]), ['n4'])
  assert.deepEqual(await ids([f('lang', '==', 'en'), f('year', '>', 2020)]), ['n3'])
  assert.deepEqual(await ids([f('lang', '==', 'de'), f('year', '>', 2022)], 'or'), ['n2', 'n3'])
  await assert.rejects(ids([f('lang', 'in', 'notarray')]), (e) => e instanceof WarmError)
})

test('docIds restricts to the listed node ids', async () => {
  const vs = await make()
  await vs.add(await embeddedNodes([['a', 'alpha one', {}], ['b', 'alpha two', {}], ['c', 'alpha three', {}]]))
  const res = await vs.query({ ...(await q('alpha')), docIds: ['b', 'c'] })
  assert.deepEqual(res.ids.sort(), ['b', 'c'])
})

test('hybrid mode fuses BM25 with cosine: a rare exact term outranks a closer paraphrase', async () => {
  const vs = await make({ lexicalWeight: 1, semanticWeight: 0.2 })
  await vs.add(await embeddedNodes([['rare', 'invoice ZX-4471 was refunded in March', {}], ['other', 'the customer asked about refunds and billing in general', {}]]))
  const res = await vs.query(await q('ZX-4471', { mode: VectorStoreQueryMode.HYBRID }))
  assert.equal(res.ids[0], 'rare')
  await assert.rejects(vs.query({ ...(await q('x')), queryStr: undefined, mode: VectorStoreQueryMode.HYBRID }), WarmError)
})

test('mmr mode returns a more diverse set than plain similarity', async () => {
  const vs = await make()
  await vs.add(
    await embeddedNodes([
      ['s1', 'solar panel efficiency and photovoltaic cell cost', {}],
      ['s2', 'solar panel efficiency and photovoltaic cell cost today', {}],
      ['s3', 'solar panel efficiency and photovoltaic cell cost trends', {}],
      ['w', 'wind turbine blade design for offshore farms', {}],
    ]),
  )
  const text = 'solar panel efficiency photovoltaic cell cost'
  const plain = await vs.query(await q(text, { similarityTopK: 2 }))
  assert.ok(plain.ids.every((id) => id.startsWith('s')))
  const mmr = await vs.query(await q(text, { similarityTopK: 2, mode: VectorStoreQueryMode.MMR, mmrThreshold: 0.1 }))
  assert.equal(mmr.ids.length, 2)
  assert.ok(mmr.ids.includes('w'))
})

test('unsupported modes fail loudly instead of degrading silently', async () => {
  const vs = await make()
  for (const mode of [VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.SVM, VectorStoreQueryMode.LINEAR_REGRESSION, VectorStoreQueryMode.LOGISTIC_REGRESSION]) {
    await assert.rejects(vs.query(await q('x', { mode })), /not supported/)
  }
})

test('delete(refDocId) removes every node of a source document; deleteNodes and clear work', async () => {
  const vs = await make()
  const mk = (id, text, ref) => {
    const n = new TextNode({ id_: id, text })
    n.relationships = { SOURCE: { nodeId: ref, nodeType: 'DOCUMENT', metadata: {} } }
    return n
  }
  const nodes = [mk('a1', 'alpha one', 'docA'), mk('a2', 'alpha two', 'docA'), mk('b1', 'alpha three', 'docB')]
  const vectors = await embedModel.getTextEmbeddings(nodes.map((n) => n.text))
  nodes.forEach((n, i) => (n.embedding = vectors[i]))
  await vs.add(nodes)
  await vs.delete('docA')
  assert.deepEqual((await vs.getNodes(['a1', 'a2', 'b1'])).map((n) => n.id_), ['b1'])
  await vs.deleteNodes(['b1'])
  assert.equal((await vs.getNodes(['b1'])).length, 0)
  await vs.add(nodes)
  await vs.clear()
  assert.equal((await vs.getNodes(['a1', 'a2', 'b1'])).length, 0)
})

test('index.deleteRefDoc deletes through the vector store', async () => {
  const vs = await make()
  const storageContext = await storageContextFromDefaults({ vectorStore: vs })
  const index = await VectorStoreIndex.fromDocuments([new Document({ id_: 'd1', text: 'ephemeral note about walruses' })], { storageContext })
  assert.equal((await index.asRetriever({ similarityTopK: 3 }).retrieve('walruses')).length, 1)
  await index.deleteRefDoc('d1')
  assert.equal((await index.asRetriever({ similarityTopK: 3 }).retrieve('walruses')).length, 0)
})

test('similarityTopK boundary against maxFetch: limit-1, limit, limit+1', async () => {
  const vs = await make({ maxFetch: 10 })
  await vs.add(await embeddedNodes(Array.from({ length: 12 }, (_, i) => [`n${i}`, `alpha beta number${i}`, {}])))
  for (const k of [9, 10]) assert.equal((await vs.query(await q('alpha beta', { similarityTopK: k }))).ids.length, k)
  await assert.rejects(vs.query(await q('alpha beta', { similarityTopK: 11 })), (e) => e instanceof WarmError && e.code === 'limit_exceeded')
})

test('nodes added without an embedding are embedded by the store embedder', async () => {
  const vs = await make()
  await vs.add([new TextNode({ id_: 'plain', text: 'unembedded node about zeppelins' })])
  const res = await vs.query(await q('zeppelins airship'))
  assert.equal(res.ids[0], 'plain')
})
