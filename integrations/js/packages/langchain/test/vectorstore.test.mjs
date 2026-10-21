// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'
import { Document } from '@langchain/core/documents'
import { HashingEmbedder, HmsError, HmsVectorStore } from '../dist/index.js'

const hashing = new HashingEmbedder(256)
/** A LangChain `Embeddings` over the deterministic hashing embedder, so tests need no model. */
const embeddings = {
  embedDocuments: (texts) => hashing.encodeDocuments(texts),
  embedQuery: (text) => hashing.encodeQuery(text),
}
const tmp = () => mkdtempSync(join(tmpdir(), 'hms-lc-'))
const make = (opts = {}) => HmsVectorStore.create(embeddings, { path: tmp(), dimensions: 1024, ...opts })
const docs = (n, extra = {}) =>
  Array.from({ length: n }, (_, i) => new Document({ pageContent: `alpha beta gamma number${i} topic${i % 3}`, metadata: { n: i, parity: i % 2, ...extra } }))

test('fromTexts then similaritySearch returns the best match first and honours k', async () => {
  const vs = await HmsVectorStore.fromTexts(
    ['the cat sat on the mat', 'dogs chase cats in the park', 'quantum chromodynamics of gluons'],
    [{ id: 1 }, { id: 2 }, { id: 3 }],
    embeddings,
    { path: tmp(), dimensions: 1024 },
  )
  const found = await vs.similaritySearch('gluons quantum', 2)
  assert.equal(found.length, 2)
  assert.equal(found[0].pageContent, 'quantum chromodynamics of gluons')
  assert.deepEqual(found[0].metadata, { id: 3 })
})

test('similaritySearchWithScore returns scores in descending order', async () => {
  const vs = await make()
  await vs.addDocuments(docs(8))
  const res = await vs.similaritySearchWithScore('alpha beta gamma number5 topic2', 5)
  assert.equal(res.length, 5)
  assert.equal(res[0][0].metadata.n, 5)
  const scores = res.map(([, s]) => s)
  assert.deepEqual([...scores].sort((a, b) => b - a), scores)
  assert.ok(scores.every(Number.isFinite))
})

test('addDocuments ids: explicit ids, document.id, generated; same id overwrites, new ids add', async () => {
  const vs = await make()
  assert.deepEqual(await vs.addDocuments([new Document({ pageContent: 'one', metadata: {} })], { ids: ['a'] }), ['a'])
  assert.deepEqual(await vs.addDocuments([new Document({ id: 'b', pageContent: 'two', metadata: {} })]), ['b'])
  const [gen] = await vs.addDocuments([new Document({ pageContent: 'three', metadata: {} })])
  assert.match(gen, /^[0-9a-f-]{36}$/)
  await vs.addDocuments([new Document({ pageContent: 'one replaced', metadata: { v: 2 } })], { ids: ['a'] })
  const got = await vs.getByIds(['a', 'b', 'missing', gen])
  assert.deepEqual(got.map((d) => d.id), ['a', 'b', gen])
  assert.equal(got[0].pageContent, 'one replaced')
  assert.deepEqual(got[0].metadata, { v: 2 })
  await assert.rejects(vs.addDocuments(docs(2), { ids: ['x'] }), RangeError)
  assert.deepEqual(await vs.addDocuments([]), [])
})

test('delete by ids, by filter, and deleteAll; exactly one selector is required', async () => {
  const vs = await make()
  const ids = await vs.addDocuments(docs(10))
  await vs.delete({ ids: [ids[0], 'nonexistent'] })
  assert.equal((await vs.getByIds([ids[0]])).length, 0)
  await vs.delete({ filter: { parity: 1 } })
  const left = await vs.getByIds(ids)
  assert.deepEqual(left.map((d) => d.metadata.n).sort((a, b) => a - b), [2, 4, 6, 8])
  await vs.delete({ deleteAll: true })
  assert.equal((await vs.getByIds(ids)).length, 0)
  await assert.rejects(vs.delete({}), TypeError)
  await assert.rejects(vs.delete({ ids: ['a'], deleteAll: true }), TypeError)
})

test('filters: engine equality, client comparison, $in, $or, nested key', async () => {
  const vs = await make()
  await vs.addDocuments(docs(20).map((d, i) => new Document({ ...d, metadata: { ...d.metadata, nested: { tier: i % 4 } } })))
  const eq = await vs.similaritySearch('alpha', 6, { parity: 1 })
  assert.equal(eq.length, 6)
  assert.ok(eq.every((d) => d.metadata.parity === 1))
  const range = await vs.similaritySearch('alpha', 20, { n: { $gte: 15 } })
  assert.deepEqual(range.map((d) => d.metadata.n).sort((a, b) => a - b), [15, 16, 17, 18, 19])
  const inn = await vs.similaritySearch('alpha', 20, { n: { $in: [1, 2, 3] }, parity: 1 })
  assert.deepEqual(inn.map((d) => d.metadata.n).sort(), [1, 3])
  const or = await vs.similaritySearch('alpha', 20, { $or: [{ n: 0 }, { n: 19 }] })
  assert.deepEqual(or.map((d) => d.metadata.n).sort((a, b) => a - b), [0, 19])
  const nested = await vs.similaritySearch('alpha', 20, { 'nested.tier': 3 })
  assert.equal(nested.length, 5)
})

test('k boundary: maxFetch-1, maxFetch, maxFetch+1', async () => {
  const vs = await make({ maxFetch: 10 })
  await vs.addDocuments(docs(12))
  assert.equal((await vs.similaritySearch('alpha', 9)).length, 9)
  assert.equal((await vs.similaritySearch('alpha', 10)).length, 10)
  await assert.rejects(vs.similaritySearch('alpha', 11), (e) => e instanceof HmsError && e.code === 'limit_exceeded')
})

test('a filter with client-side terms can return fewer than k when candidates run out', async () => {
  const vs = await make({ maxFetch: 10, overfetch: 2 })
  await vs.addDocuments(docs(30))
  // k=3, overfetch 2 => 6 candidates are inspected; only n>=28 qualify and none of them rank in the top 6.
  const res = await vs.similaritySearch('alpha beta gamma number0 topic0', 3, { n: { $gte: 28 } })
  assert.ok(res.length < 3)
})

test('addVectors and similaritySearchVectorWithScore rank by cosine', async () => {
  const vs = await make()
  const texts = ['red apple', 'green pear', 'blue plum']
  const vectors = await embeddings.embedDocuments(texts)
  await vs.addVectors(vectors, texts.map((t) => new Document({ pageContent: t, metadata: {} })), { ids: ['r', 'g', 'b'] })
  const res = await vs.similaritySearchVectorWithScore(await embeddings.embedQuery('green pear'), 2)
  assert.equal(res[0][0].id, 'g')
  assert.equal(res.length, 2)
  await assert.rejects(vs.addVectors([vectors[0]], docs(2)), RangeError)
})

test('maxMarginalRelevanceSearch prefers diverse results over near duplicates', async () => {
  const vs = await make()
  const mk = (text) => new Document({ pageContent: text, metadata: {} })
  await vs.addDocuments([
    mk('solar panel efficiency and photovoltaic cell cost'),
    mk('solar panel efficiency and photovoltaic cell cost today'),
    mk('solar panel efficiency and photovoltaic cell cost trends'),
    mk('wind turbine blade design for offshore farms'),
  ])
  const plain = await vs.similaritySearch('solar panel efficiency photovoltaic cell cost', 2)
  assert.ok(plain.every((d) => d.pageContent.startsWith('solar')))
  const diverse = await vs.maxMarginalRelevanceSearch('solar panel efficiency photovoltaic cell cost', { k: 2, fetchK: 4, lambda: 0.1 })
  assert.equal(diverse.length, 2)
  assert.ok(diverse.some((d) => d.pageContent.startsWith('wind')))
})

test('asRetriever invokes similarity and mmr retrieval with a filter', async () => {
  const vs = await make()
  await vs.addDocuments(docs(12))
  const retriever = vs.asRetriever({ k: 3, filter: { parity: 0 } })
  const out = await retriever.invoke('alpha beta gamma number4 topic1')
  assert.equal(out.length, 3)
  assert.ok(out.every((d) => d.metadata.parity === 0))
  assert.equal(out[0].metadata.n, 4)
  const mmr = vs.asRetriever({ k: 3, searchType: 'mmr', searchKwargs: { fetchK: 8, lambda: 0.5 } })
  assert.equal((await mmr.invoke('alpha')).length, 3)
})

test('collection scopes ids and queries inside one native store', async () => {
  const path = tmp()
  const a = await HmsVectorStore.create(embeddings, { path, dimensions: 1024, collection: 'a' })
  const b = new (a.constructor)(embeddings, { store: new a.store.constructor({ memory: a.store.memory, embedder: a.store.embedder, collection: 'b' }) })
  await a.addDocuments([new Document({ pageContent: 'in collection a', metadata: {} })], { ids: ['same'] })
  await b.addDocuments([new Document({ pageContent: 'in collection b', metadata: {} })], { ids: ['same'] })
  assert.equal((await a.getByIds(['same']))[0].pageContent, 'in collection a')
  assert.equal((await b.getByIds(['same']))[0].pageContent, 'in collection b')
  assert.deepEqual((await a.similaritySearch('collection', 5)).map((d) => d.pageContent), ['in collection a'])
})

test('data persists across a process restart', async () => {
  const path = tmp()
  const space = { model: 'hashing', revision: 'r1' }
  const dist = pathToFileURL(new URL('../dist/index.js', import.meta.url).pathname).href
  // The engine locks a path for the life of the process, so the writer is a short-lived child.
  execFileSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `
    import { Document } from '@langchain/core/documents'
    import { HmsVectorStore, HashingEmbedder } from ${JSON.stringify(dist)}
    const h = new HashingEmbedder(256)
    const emb = { embedDocuments: (t) => h.encodeDocuments(t), embedQuery: (t) => h.encodeQuery(t) }
    const vs = await HmsVectorStore.create(emb, { path: ${JSON.stringify(path)}, dimensions: 1024, embeddingSpace: ${JSON.stringify(space)} })
    await vs.addDocuments([0, 1, 2].map((i) => new Document({ pageContent: 'alpha beta gamma number' + i + ' topic' + (i % 3), metadata: {} })), { ids: ['x', 'y', 'z'] })
    await vs.store.close()
  `,
    ],
    { cwd: new URL('..', import.meta.url).pathname },
  )
  const vs = await HmsVectorStore.create(embeddings, { path, dimensions: 1024, embeddingSpace: space })
  assert.deepEqual((await vs.getByIds(['x', 'z'])).map((d) => d.id), ['x', 'z'])
  assert.deepEqual((await vs.similaritySearch('alpha beta gamma number1 topic1', 1)).map((d) => d.id), ['y'])
})

test('encryption at rest is switched on through hmsConfig and the store still searches', async () => {
  const vs = await make({ hmsConfig: { encryptionEnabled: true, encryptionPassphrase: 'correct horse battery staple' } })
  assert.equal(vs.store.memory.securityStatus().encryptionActive, true)
  await vs.addDocuments(docs(4))
  assert.equal((await vs.similaritySearch('alpha beta gamma number2 topic2', 1))[0].metadata.n, 2)
})

test('hybrid knobs pass through: titleWeight and bm25 parameters are accepted by the engine', async () => {
  const vs = await make({ titleWeight: 3, bm25K1: 1.5, bm25B: 0.5, fusion: 'rrf', lexicalWeight: 1, semanticWeight: 0.5 })
  await vs.addDocuments([
    new Document({ pageContent: 'Zebra migration\nplain body text about animals', metadata: {} }),
    new Document({ pageContent: 'Other title\nthe zebra appears once in this body', metadata: {} }),
  ])
  const res = await vs.similaritySearch('zebra', 2)
  assert.equal(res.length, 2)
  assert.ok(res[0].pageContent.startsWith('Zebra migration'))
})
