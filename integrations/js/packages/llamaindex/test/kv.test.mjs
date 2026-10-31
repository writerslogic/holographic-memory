// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { TextNode, Document } from '@llamaindex/core/schema'
import { SimpleKVStore } from '@llamaindex/core/storage/kv-store'
import { KVDocumentStore } from '@llamaindex/core/storage/doc-store'
import { KVIndexStore } from '@llamaindex/core/storage/index-store'
import { WarmError, WarmKVStore, WarmStore } from '../dist/index.js'

const store = (opts = {}) => new WarmStore({ path: mkdtempSync(join(tmpdir(), 'warm-li-kv-')), dimensions: 1024, ...opts })

test('BaseKVStore contract matches SimpleKVStore for the same operation sequence', async () => {
  const ops = async (kv) => {
    const out = []
    out.push(await kv.get('missing'))
    out.push(await kv.get('missing', 'other'))
    await kv.put('k1', { a: 1, nested: { b: [1, 2, 3] } })
    await kv.put('k2', { a: 2 }, 'other')
    await kv.put('k3', { a: 3 }, 'other')
    out.push(await kv.get('k1'))
    out.push(await kv.get('k1', 'other'))
    out.push(await kv.getAll())
    out.push(await kv.getAll('other'))
    await kv.put('k2', { a: 22 }, 'other')
    out.push(await kv.get('k2', 'other'))
    out.push(await kv.delete('k2', 'other'))
    out.push(await kv.delete('k2', 'other'))
    out.push(await kv.getAll('other'))
    out.push(await kv.getAll('empty'))
    return out
  }
  assert.deepEqual(await ops(new WarmKVStore(store())), await ops(new SimpleKVStore()))
})

test('keys with separators and unicode do not collide across collections', async () => {
  const kv = new WarmKVStore(store())
  await kv.put('a:b', { v: 1 }, 'c')
  await kv.put('b', { v: 2 }, 'c:a')
  await kv.put('ключ/ü', { v: 3 })
  assert.deepEqual(await kv.get('a:b', 'c'), { v: 1 })
  assert.deepEqual(await kv.get('b', 'c:a'), { v: 2 })
  assert.deepEqual(await kv.get('ключ/ü'), { v: 3 })
  assert.deepEqual(await kv.getAll('c'), { 'a:b': { v: 1 } })
})

test('KVDocumentStore on WarmKVStore behaves like KVDocumentStore on SimpleKVStore', async () => {
  const run = async (kv) => {
    const docStore = new KVDocumentStore(kv)
    const n1 = new TextNode({ id_: 'n1', text: 'first node' })
    const n2 = new TextNode({ id_: 'n2', text: 'second node' })
    for (const n of [n1, n2]) n.relationships = { SOURCE: { nodeId: 'doc1', nodeType: 'DOCUMENT', metadata: {} } }
    const out = []
    await docStore.addDocuments([n1, n2], true)
    out.push((await docStore.getDocument('n1')).text, await docStore.documentExists('n2'), await docStore.documentExists('nope'))
    out.push((await docStore.getRefDocInfo('doc1')).nodeIds.sort(), Object.keys(await docStore.docs()).sort())
    await docStore.setDocumentHash('n1', 'h1')
    out.push(await docStore.getDocumentHash('n1'), await docStore.getAllDocumentHashes())
    await assert.rejects(docStore.addDocuments([new Document({ id_: 'n1', text: 'dup' })], false))
    await docStore.addDocuments([new TextNode({ id_: 'n1', text: 'updated' })], true)
    out.push((await docStore.getDocument('n1')).text)
    await docStore.deleteRefDoc('doc1')
    out.push(await docStore.documentExists('n1'), Object.keys(await docStore.docs()))
    return out
  }
  assert.deepEqual(await run(new WarmKVStore(store())), await run(new SimpleKVStore()))
})

test('KVIndexStore on WarmKVStore round trips an index struct', async () => {
  const kv = new WarmKVStore(store())
  const indexStore = new KVIndexStore(kv)
  const { IndexDict } = await import('@llamaindex/core/data-structs')
  const struct = new IndexDict()
  struct.addNode(new TextNode({ id_: 'node-1', text: 'hello' }), 'vec-1')
  await indexStore.addIndexStruct(struct)
  const back = await indexStore.getIndexStruct(struct.indexId)
  assert.deepEqual(Object.keys(back.nodesDict), Object.keys(struct.nodesDict))
  assert.equal((await indexStore.getIndexStructs()).length, 1)
  await indexStore.deleteIndexStruct(struct.indexId)
  assert.equal((await indexStore.getIndexStructs()).length, 0)
})

test('getAll uses pagination to prove completeness at maxFetch and rejects larger collections', async () => {
  for (const [n, ok] of [[9, true], [10, true], [11, false]]) {
    const kv = new WarmKVStore(store({ maxFetch: 10 }))
    for (let i = 0; i < n; i++) await kv.put(`k${i}`, { i })
    for (let i = 0; i < 11; i++) await kv.put(`other${i}`, { i }, 'other')
    if (ok) assert.deepEqual(await kv.getAll(), Object.fromEntries(Array.from({ length: n }, (_, i) => [`k${i}`, { i }])))
    else await assert.rejects(kv.getAll(), (e) => e instanceof WarmError && e.code === 'limit_exceeded')
    await assert.rejects(kv.getAll('other'), (e) => e instanceof WarmError && e.code === 'limit_exceeded')
    // exact operations stay unbounded
    assert.deepEqual(await kv.get(`k${n - 1}`), { i: n - 1 })
  }
})

test('large values (over the metadata cap, several chunks) round trip', async () => {
  const kv = new WarmKVStore(new WarmStore({ path: mkdtempSync(join(tmpdir(), 'warm-li-kv-')), dimensions: 1024, chunkWords: 50, overlapWords: 5 }))
  const big = { text: Array.from({ length: 30000 }, (_, i) => `wörd${i}`).join(' ') }
  await kv.put('big', big)
  assert.deepEqual(await kv.get('big'), big)
})
