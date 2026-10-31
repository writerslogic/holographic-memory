// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { HashingEmbedder, WarmError, WarmStore, parseFilter, splitFilter } from '../dist/index.js'

const store = (opts = {}) => new WarmStore({ path: mkdtempSync(join(tmpdir(), 'warm-js-')), dimensions: 1024, ...opts })
const recs = (n, meta = {}) =>
  Array.from({ length: n }, (_, i) => ({ id: `d${i}`, text: `alpha beta gamma number${i} topic${i % 3}`, metadata: { n: i, parity: i % 2, ...meta } }))

for (const [field, value] of Object.entries({
  embeddingModel: 'different-model', embeddingRevision: 'different-revision', embeddingDimensions: 128,
  embeddingNormalization: 'none', embeddingMetric: 'dot',
})) {
  test(`conflicting ${field} is rejected before creating storage`, () => {
    const path = join(mkdtempSync(join(tmpdir(), 'warm-identity-')), 'store')
    assert.throws(() => new WarmStore({ path, warmConfig: { [field]: value } }), (error) => error.code === 'invalid_argument')
    assert.equal(existsSync(path), false)
  })
}

for (const [field, value] of Object.entries({ model: '', revision: '', dimensions: 65537, normalization: 'none', metric: 'dot' })) {
  test(`invalid embedder ${field} is rejected before creating storage`, () => {
    const embedder = new HashingEmbedder()
    embedder.space[field] = value
    const path = join(mkdtempSync(join(tmpdir(), 'warm-identity-')), 'store')
    assert.throws(() => new WarmStore({ path, embedder }), (error) => error.code === 'invalid_argument')
    assert.equal(existsSync(path), false)
  })
}

test('persistent identity follows the embedder and cannot be relabeled on reopen', async () => {
  const path = mkdtempSync(join(tmpdir(), 'warm-identity-'))
  execFileSync(process.execPath, ['--input-type=module', '-e', `
    import { WarmStore } from ${JSON.stringify(new URL('../dist/index.js', import.meta.url).href)}
    const store = new WarmStore({ path: ${JSON.stringify(path)}, dimensions: 1024,
      warmConfig: { embeddingModel: 'warm-hashing', embeddingRevision: 'uni-bi-v1', embeddingDimensions: 384 } })
    await store.remember([{ id: 'one', text: 'original alpha' }])
    await store.close()
  `])
  const schema = readFileSync(join(path, 'store.json'))
  const other = new HashingEmbedder(384, false)
  assert.throws(() => new WarmStore({ path, dimensions: 1024, embedder: other,
    warmConfig: { embeddingRevision: 'uni-bi-v1' } }), (error) => error.code === 'invalid_argument')
  assert.deepEqual(readFileSync(join(path, 'store.json')), schema)
  const reopened = store({ path })
  assert.deepEqual(reopened.memory.securityStatus().embeddingSpace, reopened.embedder.space)
  assert.equal((await reopened.recall('original alpha', { k: 1 }))[0].id, 'one')
  await reopened.close()
})

test('older native bindings fail before a partial framework workflow', () => {
  const embedder = new HashingEmbedder(384)
  const memory = { securityStatus: () => ({ embeddingSpace: embedder.space }) }
  assert.throws(() => new WarmStore({ memory, embedder }), (error) => error.code === 'backend' && /current native binding/.test(error.message))
})

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
  const a = new WarmStore({ memory: plain.memory, collection: 'A' })
  await plain.remember([{ id: 'same', text: 'plain doc' }])
  await a.remember([{ id: 'same', text: 'collection doc' }])
  assert.equal((await plain.get(['same']))[0].text, 'plain doc')
  assert.equal((await a.get(['same']))[0].text, 'collection doc')
  assert.deepEqual((await a.list()).records.map((r) => r.text), ['collection doc'])
  assert.equal(await a.forget(['same']), 1)
  assert.ok((await plain.get(['same']))[0])
})

test('collection id collisions cannot overwrite or delete legacy mapped records', async () => {
  const plain = store()
  const a = new WarmStore({ memory: plain.memory, collection: 'A' })
  await a.remember([{ id: 'x', text: 'owned by A' }])
  await assert.rejects(plain.remember([{ id: 'A/x', text: 'foreign' }]), /another collection/)
  await assert.rejects(plain.forget(['A/x']), /another collection/)
  assert.equal((await a.get(['x']))[0].text, 'owned by A')
  await plain.remember([{ id: 'A/y', text: 'owned by default' }])
  await assert.rejects(a.remember([{ id: 'y', text: 'foreign' }]), /another collection/)
  await assert.rejects(a.forget(['y']), /another collection/)
  assert.equal((await plain.get(['A/y']))[0].text, 'owned by default')
  await plain.memory.memorizeDocument({ id: 'legacy', text: 'native document' })
  await assert.rejects(plain.remember([{ id: 'safe', text: 'safe' }, { id: 'legacy', text: 'foreign' }]), /unscoped native document/)
  assert.equal((await plain.get(['safe']))[0], undefined)
  await assert.rejects(plain.forget(['legacy']), /unscoped native document/)
  assert.ok(await plain.memory.getDocumentInfo('legacy'))
})

test('colliding writes on stores sharing a native handle are serialized', async () => {
  const plain = store()
  const a = new WarmStore({ memory: plain.memory, collection: 'A' })
  const results = await Promise.allSettled([
    plain.remember([{ id: 'A/x', text: 'default' }]),
    a.remember([{ id: 'x', text: 'A' }]),
  ])
  assert.equal(results.filter((result) => result.status === 'fulfilled').length, 1)
  assert.equal(results.filter((result) => result.status === 'rejected').length, 1)
  assert.equal((await plain.get(['A/x']))[0].text, 'default')
})

test('reserved metadata, bad ids and bad vectors are rejected', async () => {
  const s = store()
  await assert.rejects(s.remember([{ id: 'a', text: 't', metadata: { _warm_x: 1 } }]), WarmError)
  await assert.rejects(s.remember([{ id: 'hms:a', text: 't' }]), WarmError)
  await assert.rejects(s.remember([{ id: 'é'.repeat(129), text: 't' }]), WarmError)
  await assert.rejects(s.remember([{ id: 'a', text: 't', vector: [1, 2] }]), WarmError)
})

for (const delta of [-1, 0, 1]) {
  test(`list truncation at maxFetch-1, maxFetch, maxFetch+1 (${delta})`, async () => {
    const s = store({ maxFetch: 10 })
    await s.remember(recs(10 + delta))
    const page = await s.list()
    assert.equal(page.records.length, Math.min(10, 10 + delta))
    assert.equal(page.truncated, 10 + delta > 10)
    await assert.rejects(s.recall('alpha', { k: 11 }), (e) => e.code === 'limit_exceeded')
  })
}

test('forgetWhere pages past maxFetch', async () => {
  const s = store({ maxFetch: 10 })
  await s.remember([...recs(25, { kind: 'x' }), { id: 'keep', text: 'keep me', metadata: { kind: 'y' } }])
  assert.equal(await s.forgetWhere({ kind: 'x' }), 25)
  assert.deepEqual((await s.list()).records.map((r) => r.id), ['keep'])
})

test('residual filters find and delete matches beyond the metadata page and query caps', async () => {
  const s = store({ maxFetch: 10 })
  await s.remember(recs(1001))
  const filter = { n: { $gte: 995 } }
  const expected = ['d995', 'd996', 'd997', 'd998', 'd999', 'd1000'].sort()
  assert.deepEqual((await s.list(filter)).records.map((record) => record.id).sort(), expected)
  assert.deepEqual((await s.recall('alpha', { k: 10, filter })).map((hit) => hit.id).sort(), expected)
  assert.equal(await s.forgetWhere(filter), expected.length)
  assert.deepEqual((await s.list(filter)).records, [])
  await s.close()
})

test('fused residual filter admission distinguishes limit minus one, limit and limit plus one', async () => {
  const s = store()
  await s.remember(recs(1001))
  for (const count of [999, 1000]) {
    assert.equal((await s.recall('alpha', { k: 1, filter: { n: { $lt: count } } })).length, 1)
  }
  await assert.rejects(s.recall('alpha', { k: 1, filter: { n: { $lt: 1001 } } }), (error) => error.code === 'filter_limit')
  await s.close()
})

for (const chunks of [999, 1000, 1001]) {
  test(`record assembly preserves its declared passage boundary (${chunks})`, async () => {
    const s = store({ chunkWords: 2, overlapWords: 1 })
    const text = Array.from({ length: chunks + 1 }, (_, i) => `word${i}`).join(' ')
    await s.remember([{ id: 'long', text }])
    assert.equal((await s.memory.getDocumentInfo('long')).chunkCount, chunks)
    if (chunks > 1000) {
      await assert.rejects(s.get(['long']), (error) => error.code === 'limit_exceeded')
    } else {
      assert.equal((await s.get(['long']))[0].text, text)
    }
    await s.close()
  })
}

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
  const s = store({ warmConfig: { encryptionEnabled: true, encryptionPassphrase: 'correct horse battery staple' } })
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
  await assert.rejects(s.recall('x'), (e) => e instanceof WarmError && e.code === 'closed')
})
