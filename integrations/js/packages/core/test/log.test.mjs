// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { HmsStore, OrderedLog } from '../dist/index.js'

const tmp = () => mkdtempSync(join(tmpdir(), 'hms-log-'))
const open = (path = tmp()) => new HmsStore({ path, dimensions: 256 })
const entries = (n) => Array.from({ length: n }, (_, i) => ({ text: `old ${i}` }))

function interrupt(store, boundary) {
  const forget = store.forget.bind(store)
  let deleted = 0
  store.forget = async (ids) => {
    let count = 0
    for (const id of ids) {
      count += await forget([id])
      if (++deleted === boundary) throw new Error('interrupted after durable delete')
    }
    return count
  }
  return () => { store.forget = forget }
}

test('every individual clear deletion leaves a retryable prefix across the batch boundary', async () => {
  for (let boundary = 1; boundary <= 101; boundary++) {
    const path = tmp()
    const store = open(path)
    const log = new OrderedLog(store, 'session')
    await log.append(entries(101))
    const restore = interrupt(store, boundary)
    await assert.rejects(log.clear(), /interrupted after durable delete/)
    restore()
    const retry = new OrderedLog(store, 'session')
    assert.equal(await retry.size(), 101 - boundary)
    assert.deepEqual((await retry.tail()).map((e) => e.text), entries(101 - boundary).map((e) => e.text))
    await retry.clear()
    assert.deepEqual((await store.list()).records, [])
    await retry.append([{ text: 'new first' }])
    assert.deepEqual((await retry.tail()).map((e) => e.text), ['new first'])
    await store.close()
    rmSync(path, { recursive: true, force: true })
  }
})

for (const boundary of [1, 99, 100, 101]) {
  test(`interrupted clear resumes after a process restart at deletion ${boundary}`, async () => {
    const path = tmp()
    execFileSync(process.execPath, ['--input-type=module', '-e', `
      import { HmsStore, OrderedLog } from ${JSON.stringify(new URL('../dist/index.js', import.meta.url).href)}
      const store = new HmsStore({ path: ${JSON.stringify(path)}, dimensions: 256 })
      const log = new OrderedLog(store, 'session')
      await log.append(${JSON.stringify(entries(101))})
      const forget = store.forget.bind(store)
      let deleted = 0
      store.forget = async (ids) => {
        for (const id of ids) {
          await forget([id])
          if (++deleted === ${boundary}) process.exit(0)
        }
        return ids.length
      }
      await log.clear()
      process.exit(1)
    `])
    const store = open(path)
    const log = new OrderedLog(store, 'session')
    assert.equal(await log.size(), 101 - boundary)
    await log.clear()
    assert.deepEqual((await store.list()).records, [])
    await log.append([{ text: 'new first' }])
    assert.deepEqual((await log.tail()).map((e) => e.text), ['new first'])
    await store.close()
  })
}

test('competing wrappers serialize append, clear, read and pop across one native handle', async () => {
  const store = open()
  const sibling = new HmsStore({ memory: store.memory })
  const a = new OrderedLog(store, 'shared')
  const b = new OrderedLog(sibling, 'shared')
  await Promise.all([a.append([{ text: 'first' }]), b.append([{ text: 'second' }])])
  assert.deepEqual((await b.tail()).map((e) => e.text), ['first', 'second'])
  const clearing = a.clear()
  const appended = b.append([{ text: 'after clear' }])
  const reading = a.tail()
  await Promise.all([clearing, appended])
  assert.deepEqual((await reading).map((e) => e.text), ['after clear'])
  const popped = await Promise.all([a.pop(), b.pop()])
  assert.deepEqual(popped.map((e) => e?.text), ['after clear', undefined])
  await store.close()
})

test('clear also removes orphan entries left by the older ascending-delete implementation', async () => {
  const store = open()
  const log = new OrderedLog(store, 'legacy')
  await log.append(entries(3))
  await store.forget(['legacy#0000000000'])
  await new OrderedLog(store, 'legacy').clear()
  assert.deepEqual((await store.list()).records, [])
  await log.append([{ text: 'new first' }])
  assert.deepEqual((await log.tail()).map((e) => e.text), ['new first'])
  await store.close()
})

test('clear can remove historical entries without reconstructing unavailable source bytes', async () => {
  const store = open()
  await store.memory.memorizeDocument({ id: 'legacy#0000000000', text: 'old payload',
    metadata: { _hms_collection: '', hms_log: 'legacy', hms_seq: 0 } })
  await new OrderedLog(store, 'legacy').clear()
  assert.deepEqual((await store.list()).records, [])
  await store.close()
})
