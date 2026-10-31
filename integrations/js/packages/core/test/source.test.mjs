// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { WarmStore } from '../dist/index.js'

const tmp = () => mkdtempSync(join(tmpdir(), 'warm-source-'))
const fixture = '\u2003\t  alpha\nβeta\t三\r\n café\u0085omega\u00a0\n'

for (const overlapWords of [0, 1, 3]) {
  test(`source bytes and citation spans survive restart with overlap ${overlapWords}`, async () => {
    const path = tmp()
    execFileSync(process.execPath, ['--input-type=module', '-e', `
      import { WarmStore } from ${JSON.stringify(new URL('../dist/index.js', import.meta.url).href)}
      const s = new WarmStore({ path: ${JSON.stringify(path)}, dimensions: 1024, chunkWords: 4, overlapWords: ${overlapWords} })
      await s.remember([{ id: 'source', text: ${JSON.stringify(fixture)}, metadata: { label: 'original' } }])
      await s.close()
    `])
    const s = new WarmStore({ path, dimensions: 1024, chunkWords: 4, overlapWords })
    const [record] = await s.get(['source'])
    assert.equal(record.text, fixture)
    assert.deepEqual(record.metadata, { label: 'original' })
    assert.equal((await s.list()).records[0].text, fixture)
    const [hit] = await s.recall('omega', { k: 1 })
    assert.equal(hit.text, fixture)
    const chunks = await s.memory.chunkDocument({ id: 'source', text: fixture, chunkWords: 4, overlapWords })
    const passage = chunks.find((p) => p.startByte === hit.span.start && p.endByte === hit.span.end)
    assert.ok(passage)
    assert.equal(Buffer.from(hit.text).subarray(hit.span.start, hit.span.end).toString('utf8'), passage.text)
    await s.close()
  })
}

test('large JSON and zero-overlap passage gaps preserve every byte', async () => {
  const text = JSON.stringify({ items: Array.from({ length: 6000 }, (_, i) => `café ${i}\t雪`) }, null, 2)
  assert.ok(Buffer.byteLength(text) > 65536)
  const s = new WarmStore({ path: tmp(), dimensions: 1024, chunkWords: 100, overlapWords: 0 })
  await s.remember([{ id: 'json', text }])
  for (const record of [(await s.get(['json']))[0], (await s.list()).records[0], (await s.recall('café', { k: 1 }))[0]]) {
    assert.equal(record.text, text)
    assert.deepEqual(JSON.parse(record.text), JSON.parse(text))
  }
  await s.close()
})

test('blank text is rejected without writing other records in the request', async () => {
  const s = new WarmStore({ path: tmp(), dimensions: 1024 })
  for (const text of ['', ' ', '\t\n', '\u2003\u0085']) {
    await assert.rejects(s.remember([{ id: 'good', text: 'valid' }, { id: 'blank', text }]), (e) => e.code === 'invalid_argument')
    assert.deepEqual((await s.list()).records, [])
  }
  await s.close()
})

test('source metadata obeys the 64 KiB boundary before writing a batch', async () => {
  const s = new WarmStore({ path: tmp(), dimensions: 1024 })
  await s.remember([{ id: 'source', text: 'alpha', metadata: { pad: '' } }])
  const { metadata } = await s.memory.getDocumentInfo('source')
  const available = 65536 - Buffer.byteLength(JSON.stringify(metadata))
  for (const delta of [-1, 0, 1]) {
    const records = [{ id: 'source', text: 'alpha', metadata: { pad: 'x'.repeat(available + delta) } }]
    if (delta > 0) {
      await assert.rejects(s.remember([{ id: 'tentative', text: 'alpha' }, ...records]), (e) => e.code === 'limit_exceeded')
      assert.deepEqual(await s.get(['tentative']), [undefined])
    } else {
      await s.remember(records)
      assert.equal((await s.get(['source']))[0].metadata.pad.length, available + delta)
    }
  }
  await s.close()
})

test('a source replacement during reranking cannot produce a stale citation', async () => {
  const s = new WarmStore({ path: tmp(), dimensions: 1024, rerank: async (_query, passages) => {
    await s.remember([{ id: 'source', text: 'replacement' }])
    return passages.map(() => 1)
  } })
  await s.remember([{ id: 'source', text: 'original alpha' }])
  await assert.rejects(s.recall('alpha', { k: 1 }), (e) => e.code === 'concurrent_mutation')
  await s.close()
})

for (const damage of ['legacy', 'length', 'digest', 'gap', 'span']) {
  test(`missing or damaged source evidence (${damage}) requires reingestion`, async () => {
    const s = new WarmStore({ path: tmp(), dimensions: 1024, chunkWords: 4, overlapWords: 0 })
    await s.remember([{ id: 'source', text: fixture }])
    const { metadata } = await s.memory.getDocumentInfo('source')
    switch (damage) {
      case 'legacy': delete metadata._warm_source; break
      case 'length': metadata._warm_source.bytes = 8 * 1024 * 1024 + 1; break
      case 'digest': metadata._warm_source.sha256 = '0'.repeat(64); break
      case 'gap': metadata._warm_source.gaps[0].text = 'x'; break
      case 'span': metadata._warm_source.gaps[0].startByte = -1; break
    }
    const input = { id: 'source', text: fixture, metadata, chunkWords: 4, overlapWords: 0, storeText: true }
    const chunks = await s.memory.chunkDocument(input)
    const embeddings = await s.embedder.encodeDocuments(chunks.map((c) => c.text))
    await s.memory.memorizeDocument({ ...input, embeddings })
    await assert.rejects(s.get(['source']), /reingest the authoritative source/)
    await assert.rejects(s.list(), /reingest the authoritative source/)
    await assert.rejects(s.recall('alpha', { k: 1 }), /reingest the authoritative source/)
    await s.remember([{ id: 'source', text: fixture }])
    assert.equal((await s.get(['source']))[0].text, fixture)
    await s.close()
  })
}
