// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// Contract tests for MastraVector, following the operations and filter syntax Mastra's own
// vector-store suites exercise (createIndex/upsert/query/filters/update/delete/describe).
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { MastraError } from '@mastra/core/error'
import { MastraVector } from '@mastra/core/vector'
import { HolographicVector } from '../dist/index.js'

const DIM = 8
const unit = (i) => Array.from({ length: DIM }, (_, j) => (i === j ? 1 : 0))
const mix = (i, j, w = 0.3) => unit(i).map((v, k) => v + (k === j ? w : 0))

const make = (opts = {}) => new HolographicVector({ id: 'hms', dimension: DIM, path: mkdtempSync(join(tmpdir(), 'hms-mastra-')), hypervectorDimensions: 1024, ...opts })

async function seeded() {
  const v = make()
  await v.createIndex({ indexName: 'docs', dimension: DIM })
  const ids = await v.upsert({
    indexName: 'docs',
    vectors: [unit(0), unit(1), unit(2), unit(3)],
    metadata: [
      { text: 'alpha', category: 'a', price: 10, tags: ['x', 'y'], active: true, when: new Date('2025-01-01T00:00:00Z') },
      { text: 'beta', category: 'b', price: 20, tags: ['y'], active: false },
      { text: 'gamma', category: 'a', price: 30, tags: ['z', 'x'], nested: { k: 1 } },
      { category: 'c', price: 40 },
    ],
    ids: ['v0', 'v1', 'v2', 'v3'],
  })
  assert.deepEqual(ids, ['v0', 'v1', 'v2', 'v3'])
  return v
}
const ids = (r) => r.map((x) => x.id).sort()

test('is a MastraVector', () => assert.ok(make() instanceof MastraVector))

test('index lifecycle: create, list, describe, idempotent create, delete', async () => {
  const v = make()
  await v.createIndex({ indexName: 'one', dimension: DIM })
  await v.createIndex({ indexName: 'two', dimension: DIM, metric: 'cosine' })
  await v.createIndex({ indexName: 'one', dimension: DIM })
  assert.deepEqual(await v.listIndexes(), ['one', 'two'])
  assert.deepEqual(await v.describeIndex({ indexName: 'one' }), { dimension: DIM, count: 0, metric: 'cosine' })
  await v.upsert({ indexName: 'one', vectors: [unit(0), unit(1)] })
  assert.equal((await v.describeIndex({ indexName: 'one' })).count, 2)
  await v.deleteIndex({ indexName: 'one' })
  assert.deepEqual(await v.listIndexes(), ['two'])
  await assert.rejects(v.describeIndex({ indexName: 'one' }), MastraError)
  await v.deleteIndex({ indexName: 'one' })
})

test('rejects unsupported index parameters with MastraError', async () => {
  const v = make()
  await assert.rejects(v.createIndex({ indexName: 'x', dimension: DIM + 1 }), MastraError)
  await assert.rejects(v.createIndex({ indexName: 'x', dimension: DIM, metric: 'euclidean' }), MastraError)
  await assert.rejects(v.createIndex({ indexName: 'a/b', dimension: DIM }), MastraError)
  await assert.rejects(v.upsert({ indexName: 'missing', vectors: [unit(0)] }), MastraError)
})

test('upsert validation: dimension, NaN, lengths, sparse', async () => {
  const v = await seeded()
  await assert.rejects(v.upsert({ indexName: 'docs', vectors: [[1, 2]] }), MastraError)
  await assert.rejects(v.upsert({ indexName: 'docs', vectors: [unit(0).map(() => NaN)] }), MastraError)
  await assert.rejects(v.upsert({ indexName: 'docs', vectors: [unit(0)], ids: ['a', 'b'] }), MastraError)
  await assert.rejects(v.upsert({ indexName: 'docs', vectors: [unit(0)], sparseVectors: [{ indices: [0], values: [1] }] }), MastraError)
  await assert.rejects(v.upsert({ indexName: 'docs', vectors: [unit(0)], metadata: [{ _hms_x: 1 }] }), MastraError)
})

test('query ranks by cosine and reports cosine scores', async () => {
  const v = await seeded()
  const r = await v.query({ indexName: 'docs', queryVector: mix(1, 0), topK: 3 })
  assert.deepEqual(r.slice(0, 2).map((x) => x.id), ['v1', 'v0'])
  const q = mix(1, 0)
  const norm = Math.hypot(...q)
  assert.ok(Math.abs(r[0].score - 1 / norm) < 1e-6, 'score is the cosine')
  assert.ok(Math.abs(r[1].score - 0.3 / norm) < 1e-6)
  assert.equal(r[0].metadata.category, 'b')
  assert.equal(r[0].vector, undefined)
})

test('query includeVector and topK bounds', async () => {
  const v = await seeded()
  const [top] = await v.query({ indexName: 'docs', queryVector: unit(2), topK: 1, includeVector: true })
  assert.equal(top.id, 'v2')
  assert.deepEqual(top.vector, unit(2))
  assert.ok(Math.abs(top.score - 1) < 1e-6)
  await assert.rejects(v.query({ indexName: 'docs', queryVector: unit(0), topK: 0 }), MastraError)
  assert.equal((await v.query({ indexName: 'docs', queryVector: unit(0), topK: 100 })).length, 4)
})

test('filter operators', async () => {
  const v = await seeded()
  const q = async (filter) => ids(await v.query({ indexName: 'docs', queryVector: unit(0), topK: 10, filter }))
  assert.deepEqual(await q({ category: 'a' }), ['v0', 'v2'])
  assert.deepEqual(await q({ category: { $eq: 'a' } }), ['v0', 'v2'])
  assert.deepEqual(await q({ category: { $ne: 'a' } }), ['v1', 'v3'])
  assert.deepEqual(await q({ price: { $gt: 10, $lte: 30 } }), ['v1', 'v2'])
  assert.deepEqual(await q({ price: { $gte: 30 } }), ['v2', 'v3'])
  assert.deepEqual(await q({ price: { $lt: 20 } }), ['v0'])
  assert.deepEqual(await q({ category: { $in: ['b', 'c'] } }), ['v1', 'v3'])
  assert.deepEqual(await q({ category: { $nin: ['b', 'c'] } }), ['v0', 'v2'])
  assert.deepEqual(await q({ tags: { $all: ['x', 'y'] } }), ['v0'])
  assert.deepEqual(await q({ tags: { $in: ['z'] } }), [])
  assert.deepEqual(await q({ nested: { $exists: true } }), ['v2'])
  assert.deepEqual(await q({ $and: [{ category: 'a' }, { price: { $gt: 10 } }] }), ['v2'])
  assert.deepEqual(await q({ $or: [{ category: 'b' }, { price: 40 }] }), ['v1', 'v3'])
  assert.deepEqual(await q({ $nor: [{ category: 'a' }, { category: 'b' }] }), ['v3'])
  assert.deepEqual(await q({ price: { $not: { $gt: 20 } } }), ['v0', 'v1'])
  assert.deepEqual(await q({ text: { $regex: '^al' } }), ['v0'])
  assert.deepEqual(await q({ text: { $regex: 'ALPHA', $options: 'i' } }), ['v0'])
  assert.deepEqual(await q({ 'nested.k': 1 }), ['v2'])
  assert.deepEqual(await q({ active: true }), ['v0'])
  assert.deepEqual(await q({ when: { $gte: new Date('2024-06-01T00:00:00Z') } }), ['v0'])
})

test('metadata-only query uses the filter and returns every match up to topK', async () => {
  const v = await seeded()
  const r = await v.query({ indexName: 'docs', filter: { category: 'a' }, topK: 10 })
  assert.deepEqual(ids(r), ['v0', 'v2'])
  await assert.rejects(v.query({ indexName: 'docs', topK: 3 }), MastraError)
})

test('filters apply before ranking: a rare match past the 100 nearest is still found', async () => {
  const v = make()
  await v.createIndex({ indexName: 'big', dimension: DIM })
  const n = 150
  await v.upsert({
    indexName: 'big',
    vectors: Array.from({ length: n }, (_, i) => (i === n - 1 ? unit(5) : mix(0, 1 + (i % 3), 0.01 * (i % 7)))),
    metadata: Array.from({ length: n }, (_, i) => ({ rare: i === n - 1 })),
    ids: Array.from({ length: n }, (_, i) => `b${i}`),
  })
  const r = await v.query({ indexName: 'big', queryVector: unit(0), topK: 5, filter: { rare: true } })
  assert.deepEqual(r.map((x) => x.id), [`b${n - 1}`])
})

test('upsert replaces by id and keeps the count', async () => {
  const v = await seeded()
  await v.upsert({ indexName: 'docs', vectors: [unit(7)], metadata: [{ text: 'replaced', category: 'z' }], ids: ['v0'] })
  assert.equal((await v.describeIndex({ indexName: 'docs' })).count, 4)
  const [top] = await v.query({ indexName: 'docs', queryVector: unit(7), topK: 1 })
  assert.deepEqual([top.id, top.metadata.category], ['v0', 'z'])
})

test('upsert deleteFilter replaces a source document atomically per call', async () => {
  const v = await seeded()
  await v.upsert({ indexName: 'docs', vectors: [unit(4)], metadata: [{ category: 'a', text: 'fresh' }], ids: ['n1'], deleteFilter: { category: 'a' } })
  const all = await v.query({ indexName: 'docs', queryVector: unit(4), topK: 10 })
  assert.deepEqual(ids(all), ['n1', 'v1', 'v3'])
  assert.equal((await v.describeIndex({ indexName: 'docs' })).count, 3)
})

test('updateVector by id merges metadata and can replace the vector', async () => {
  const v = await seeded()
  await v.updateVector({ indexName: 'docs', id: 'v1', update: { metadata: { reviewed: true } } })
  const [a] = await v.query({ indexName: 'docs', queryVector: unit(1), topK: 1, includeVector: true })
  assert.deepEqual([a.id, a.metadata.reviewed, a.metadata.category], ['v1', true, 'b'])
  assert.deepEqual(a.vector, unit(1))
  await v.updateVector({ indexName: 'docs', id: 'v1', update: { vector: unit(6) } })
  const [b] = await v.query({ indexName: 'docs', queryVector: unit(6), topK: 1 })
  assert.deepEqual([b.id, b.metadata.reviewed], ['v1', true])
  await assert.rejects(v.updateVector({ indexName: 'docs', id: 'nope', update: { metadata: { a: 1 } } }), MastraError)
})

test('updateVector by filter updates every match', async () => {
  const v = await seeded()
  await v.updateVector({ indexName: 'docs', filter: { category: 'a' }, update: { metadata: { status: 'archived' } } })
  const r = await v.query({ indexName: 'docs', queryVector: unit(0), topK: 10, filter: { status: 'archived' } })
  assert.deepEqual(ids(r), ['v0', 'v2'])
})

test('deleteVector and deleteVectors by ids or filter update the count', async () => {
  const v = await seeded()
  await v.deleteVector({ indexName: 'docs', id: 'v0' })
  assert.equal((await v.describeIndex({ indexName: 'docs' })).count, 3)
  await v.deleteVectors({ indexName: 'docs', ids: ['v1', 'ghost'] })
  assert.equal((await v.describeIndex({ indexName: 'docs' })).count, 2)
  await v.deleteVectors({ indexName: 'docs', filter: { category: 'a' } })
  assert.deepEqual(ids(await v.query({ indexName: 'docs', queryVector: unit(0), topK: 10 })), ['v3'])
  assert.equal((await v.describeIndex({ indexName: 'docs' })).count, 1)
  await assert.rejects(v.deleteVectors({ indexName: 'docs' }), MastraError)
  await assert.rejects(v.deleteVectors({ indexName: 'docs', ids: ['v3'], filter: { a: 1 } }), MastraError)
})

test('indexes are isolated from each other', async () => {
  const v = await seeded()
  await v.createIndex({ indexName: 'other', dimension: DIM })
  await v.upsert({ indexName: 'other', vectors: [unit(0)], ids: ['v0'], metadata: [{ category: 'other' }] })
  assert.equal((await v.query({ indexName: 'docs', queryVector: unit(0), topK: 10 })).length, 4)
  const [o] = await v.query({ indexName: 'other', queryVector: unit(0), topK: 10 })
  assert.equal(o.metadata.category, 'other')
  await v.deleteIndex({ indexName: 'other' })
  assert.equal((await v.query({ indexName: 'docs', queryVector: unit(0), topK: 10 })).length, 4)
})
