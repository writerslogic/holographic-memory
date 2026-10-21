// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { HashingEmbedder, HmsError, HmsStore, parseFilter, splitFilter } from '../dist/index.js'

const store = (opts = {}) => new HmsStore({ path: mkdtempSync(join(tmpdir(), 'hms-js-')), dimensions: 1024, ...opts })
const recs = (n, meta = {}) =>
  Array.from({ length: n }, (_, i) => ({ id: `d${i}`, text: `alpha beta gamma number${i} topic${i % 3}`, metadata: { n: i, parity: i % 2, ...meta } }))

test('round trip: remember, recall, get, list, forget', async () => {
  const s = store()
  await s.remember(recs(5))
  const hits = await s.recall('alpha beta gamma number3 topic0', { k: 2 })
  assert.equal(hits[0].id, 'd3')
  assert.deepEqual(hits[0].metadata, { n: 3, parity: 1 })
  assert.equal(hits[0].text, 'alpha beta gamma number3 topic0')
  const got = await s.get(['d1', 'nope', 'd4'])
  assert.deepEqual(got.map((r) => r?.id), ['d1', undefined, 'd4'])
  const page = await s.list()
  assert.deepEqual(page.records.map((r) => r.id).sort(), ['d0', 'd1', 'd2', 'd3', 'd4'])
  assert.equal(page.truncated, false)
  assert.equal(await s.forget(['d0', 'ghost']), 1)
  assert.deepEqual(await s.get(['d0']), [undefined])
  await s.close()
})

test('same id replaces text and metadata', async () => {
  const s = store()
  await s.remember([{ id: 'x', text: 'first text', metadata: { v: 1 } }])
  await s.remember([{ id: 'x', text: 'second text', metadata: { v: 2 } }])
  const [got] = await s.get(['x'])
  assert.equal(got.text, 'second text')
  assert.deepEqual(got.metadata, { v: 2 })
  assert.equal((await s.list()).records.length, 1)
})

test('long documents are chunked and reassembled byte-exactly', async () => {
  const s = store({ chunkWords: 50, overlapWords: 10 })
  const text = Array.from({ length: 400 }, (_, i) => `wörd${i}`).join(' ')
  await s.remember([{ id: 'long', text }])
  const [got] = await s.get(['long'])
  assert.equal(got.text, text)
  const hits = await s.recall('wörd250', { k: 3 })
  assert.equal(hits.length, 1)
  assert.equal(hits[0].text, text)
})

test('filters: pushed equality and client-side residual', async () => {
  const s = store()
  await s.remember(recs(20))
  const odd = await s.recall('alpha', { k: 5, filter: { parity: 1 } })
  assert.ok(odd.length === 5 && odd.every((h) => h.metadata.parity === 1))
  const ge = await s.list({ n: { $gte: 15 } })
  assert.deepEqual(ge.records.map((r) => r.id).sort(), ['d15', 'd16', 'd17', 'd18', 'd19'])
  const mixed = await s.list({ $or: [{ n: 1 }, { n: 2 }], parity: 1 })
  assert.deepEqual(mixed.records.map((r) => r.id), ['d1'])
  const arr = await s.list({ tags: ['a', 'b'] })
  assert.equal(arr.records.length, 0)
})

test('collections isolate ids and queries inside one store', async () => {
  const plain = store()
  const a = new HmsStore({ memory: plain.memory, collection: 'A' })
  await plain.remember([{ id: 'same', text: 'plain doc' }])
  await a.remember([{ id: 'same', text: 'collection doc' }])
  assert.equal((await plain.get(['same']))[0].text, 'plain doc')
  assert.equal((await a.get(['same']))[0].text, 'collection doc')
  assert.deepEqual((await a.list()).records.map((r) => r.text), ['collection doc'])
  assert.equal(await a.forget(['same']), 1)
  assert.ok((await plain.get(['same']))[0])
})

test('reserved metadata, bad ids and bad vectors are rejected', async () => {
  const s = store()
  await assert.rejects(s.remember([{ id: 'a', text: 't', metadata: { _hms_x: 1 } }]), HmsError)
  await assert.rejects(s.remember([{ id: 'hms:a', text: 't' }]), HmsError)
  await assert.rejects(s.remember([{ id: 'é'.repeat(129), text: 't' }]), HmsError)
  await assert.rejects(s.remember([{ id: 'a', text: 't', vector: [1, 2] }]), HmsError)
})

for (const delta of [-1, 0, 1]) {
  test(`list truncation at maxFetch-1, maxFetch, maxFetch+1 (${delta})`, async () => {
    const s = store({ maxFetch: 10 })
    await s.remember(recs(10 + delta))
    const page = await s.list()
    assert.equal(page.records.length, Math.min(10, 10 + delta))
    assert.equal(page.truncated, 10 + delta >= 10)
    await assert.rejects(s.recall('alpha', { k: 11 }), (e) => e.code === 'limit_exceeded')
  })
}

test('forgetWhere pages past maxFetch', async () => {
  const s = store({ maxFetch: 10 })
  await s.remember([...recs(25, { kind: 'x' }), { id: 'keep', text: 'keep me', metadata: { kind: 'y' } }])
  assert.equal(await s.forgetWhere({ kind: 'x' }), 25)
  assert.deepEqual((await s.list()).records.map((r) => r.id), ['keep'])
})

test('custom vectors rank by cosine; fusion and weights are passed through', async () => {
  const e = new HashingEmbedder(384)
  const s = store({ embedder: e, lexicalWeight: 0, semanticWeight: 1 })
  const v = await e.encodeQuery('custom unrelated words')
  await s.remember([{ id: 'v', text: 'zzz', vector: v }, { id: 'w', text: 'other thing entirely' }])
  const hits = await s.recall('no lexical overlap qqq', { k: 1, vector: v })
  assert.equal(hits[0].id, 'v')
  assert.ok(hits[0].semanticScore > 0.99)
})

test('titleWeight and BM25 knobs reach the engine', async () => {
  const mk = (titleWeight) => store({ titleWeight })
  const docs = [
    { id: 'titled', text: 'rocket\nfiller words about cooking and gardening and more filler text here' },
    { id: 'body', text: 'cooking\nrocket rocket filler words about cooking and gardening and more' },
  ]
  const plain = mk(1)
  const heavy = mk(8)
  await plain.remember(docs)
  await heavy.remember(docs)
  const opts = { k: 2, vector: new HashingEmbedder(384).encodeQuery('x') }
  const a = await plain.recall('rocket', { k: 2 })
  const b = await heavy.recall('rocket', { k: 2 })
  const sc = (h, id) => h.find((x) => x.id === id).lexicalScore
  assert.ok(sc(b, 'titled') > sc(a, 'titled'))
  assert.ok(opts)
})

test('encryption config is accepted and the store still works', async () => {
  const s = store({ hmsConfig: { encryptionEnabled: true, encryptionPassphrase: 'correct horse battery staple' } })
  assert.equal(s.memory.securityStatus().encryptionActive, true)
  await s.remember(recs(3))
  assert.equal((await s.recall('alpha beta gamma number2 topic2', { k: 1 }))[0].id, 'd2')
})

test('keepVectors returns the stored embedding from get, list and recall', async () => {
  const e = new HashingEmbedder(384)
  const s = store({ embedder: e, keepVectors: true })
  const v = await e.encodeQuery('stored words')
  await s.remember([{ id: 'k', text: 'stored words', metadata: { a: 1 }, vector: v }])
  const [got] = await s.get(['k'])
  assert.deepEqual(got.vector, v)
  assert.deepEqual(got.metadata, { a: 1 })
  assert.deepEqual((await s.list()).records[0].vector, v)
  assert.deepEqual((await s.recall('stored', { k: 1 }))[0].vector, v)
  const plain = store()
  await plain.remember([{ id: 'p', text: 'plain text' }])
  assert.equal((await plain.get(['p']))[0].vector, undefined)
})

test('filter parsing', () => {
  const f = parseFilter({ a: 1, b: { $in: [1, 2] }, $not: { c: true } })
  assert.equal(f.type, 'and')
  assert.deepEqual(splitFilter(parseFilter({ a: 1, b: { $gt: 2 } })).pushed, { a: 1 })
  assert.deepEqual(splitFilter(parseFilter({ a: 1, $or: [{ b: 1 }, { c: 2 }] })).pushed, { a: 1 })
  assert.throws(() => parseFilter({ a: { $bogus: 1 } }))
})

test('operations after close fail with a typed error', async () => {
  const s = store()
  await s.close()
  await assert.rejects(s.recall('x'), (e) => e instanceof HmsError && e.code === 'closed')
})
