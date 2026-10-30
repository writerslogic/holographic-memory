// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { historyRecovery } from '../../../test/history-contract.mjs'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'
import { Agent, MemorySession, run, setTracingDisabled, user, assistant } from '@openai/agents'
import { HmsSession, HmsStore, createMemoryTools, itemText } from '../dist/index.js'

setTracingDisabled(true)
const tmp = () => mkdtempSync(join(tmpdir(), 'hms-oa-'))
const store = (path = tmp()) => new HmsStore({ path, dimensions: 1024 })
/** JSON round trips drop `undefined` fields such as providerData; compare modulo that. */
const norm = (x) => JSON.parse(JSON.stringify(x))
const msg = (role, text) => norm(role === 'user' ? user(text) : assistant(text))

/** Same sequence of operations on the SDK's own MemorySession and ours must give identical results. */
test('Session contract matches MemorySession: getItems(limit), popItem, clearSession', async () => {
  for (const make of [() => new MemorySession({ sessionId: 'ref' }), () => new HmsSession({ store: store(), sessionId: 'ref' })]) {
    const s = make()
    assert.equal(await s.getSessionId(), 'ref')
    assert.deepEqual(await s.getItems(), [])
    assert.equal(await s.popItem(), undefined)
    const items = Array.from({ length: 7 }, (_, i) => msg(i % 2 ? 'assistant' : 'user', `item ${i}`))
    await s.addItems(items.slice(0, 3))
    await s.addItems([])
    await s.addItems(items.slice(3))
    assert.deepEqual(await s.getItems(), items)
    assert.deepEqual(await s.getItems(3), items.slice(4))
    assert.deepEqual(await s.getItems(0), [])
    assert.deepEqual(await s.getItems(50), items)
    assert.deepEqual(await s.popItem(), items[6])
    assert.deepEqual(await s.getItems(), items.slice(0, 6))
    await s.addItems([msg('user', 'after pop')])
    assert.equal((await s.getItems(1))[0].content[0].text, 'after pop')
    await s.clearSession()
    assert.deepEqual(await s.getItems(), [])
    await s.addItems([items[0]])
    assert.deepEqual(await s.getItems(), [items[0]])
  }
})

for (const n of [99, 100, 101, 205]) {
  test(`${n} items round trip in order across the 100-item read batch`, async () => {
    const s = new HmsSession({ store: store(), sessionId: `n${n}` })
    const items = Array.from({ length: n }, (_, i) => msg(i % 2 ? 'assistant' : 'user', `turn ${i}`))
    await s.addItems(items.slice(0, 60))
    await s.addItems(items.slice(60))
    assert.deepEqual(await s.getItems(), items)
    assert.deepEqual(await s.getItems(2), items.slice(-2))
    assert.deepEqual(await s.popItem(), items[n - 1])
    assert.equal((await s.getItems()).length, n - 1)
  })
}

test('sessions persist across a process restart and stay isolated', async () => {
  const path = tmp()
  const dist = pathToFileURL(new URL('../dist/index.js', import.meta.url).pathname).href
  // The engine locks a path for the life of the process, so the writer is a short-lived child.
  execFileSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `
    import { user } from '@openai/agents'
    import { HmsSession, HmsStore } from ${JSON.stringify(dist)}
    const s = new HmsStore({ path: ${JSON.stringify(path)}, dimensions: 1024 })
    await new HmsSession({ store: s, sessionId: 'a' }).addItems([user('alpha')])
    await new HmsSession({ store: s, sessionId: 'b' }).addItems([user('beta')])
    await s.close()
  `,
    ],
    { cwd: new URL('..', import.meta.url).pathname },
  )
  const s2 = store(path)
  const a = new HmsSession({ store: s2, sessionId: 'a' })
  assert.deepEqual((await a.getItems()).map(itemText), ['alpha'])
  await a.clearSession()
  assert.deepEqual((await new HmsSession({ store: s2, sessionId: 'b' }).getItems()).map(itemText), ['beta'])
  assert.equal((await a.getItems()).length, 0)
})

test('an item larger than the 64 KiB metadata cap round trips byte-exactly', async () => {
  const s = new HmsSession({ store: new HmsStore({ path: tmp(), dimensions: 1024, chunkWords: 50, overlapWords: 5 }), sessionId: 'big' })
  const big = Array.from({ length: 20000 }, (_, i) => `wörd${i}`).join(' ')
  await s.addItems([msg('user', big), msg('assistant', 'small')])
  const [a, b] = await s.getItems()
  assert.equal(a.content[0].text, big)
  assert.equal(b.content[0].text, 'small')
})

historyRecovery({
  name: 'OpenAI Agents', store,
  create: (store) => new HmsSession({ store, sessionId: 'recovery' }),
  append: (session, texts) => session.addItems(texts.map((text) => msg('user', text))),
  read: async (session) => (await session.getItems()).map(itemText),
  clear: (session) => session.clearSession(),
})

test('tool call items are stored and searchable by their name and result', async () => {
  const s = new HmsSession({ store: store(), sessionId: 't' })
  const call = { type: 'function_call', callId: 'c1', name: 'get_weather', arguments: '{"city":"Oslo"}', status: 'completed' }
  const result = { type: 'function_call_result', callId: 'c1', name: 'get_weather', status: 'completed', output: { type: 'text', text: 'rainy in Oslo' } }
  await s.addItems([user('weather please'), call, result, assistant('It is rainy.')])
  assert.deepEqual((await s.getItems()).slice(1, 3), [call, result])
  const hit = await s.search('get_weather Oslo', 1)
  assert.equal(hit.length, 1)
  assert.equal(hit[0].type, 'function_call')
})

/** A scripted model: answers from the history it receives, so memory is observable end to end. */
function scriptedModel(log) {
  return {
    async getResponse(request) {
      const items = Array.isArray(request.input) ? request.input : [{ role: 'user', content: request.input }]
      log.push(items.length)
      const text = items.map((i) => (typeof i.content === 'string' ? i.content : (i.content ?? []).map((c) => c.text ?? '').join(' '))).join(' | ')
      return {
        usage: { requests: 1, inputTokens: 1, outputTokens: 1, totalTokens: 2, inputTokensDetails: [], outputTokensDetails: [] },
        output: [{ type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: `saw: ${text}` }] }],
      }
    },
    async *getStreamedResponse() {
      throw new Error('not used')
    },
  }
}

test('a real Runner threads the session history through consecutive runs', async () => {
  const seen = []
  const agent = new Agent({ name: 'echo', instructions: 'echo', model: scriptedModel(seen) })
  const session = new HmsSession({ store: store(), sessionId: 'run' })
  const first = await run(agent, 'my name is Ada', { session })
  assert.match(first.finalOutput, /my name is Ada/)
  const second = await run(agent, 'what is my name?', { session })
  assert.match(second.finalOutput, /my name is Ada/)
  assert.match(second.finalOutput, /what is my name\?/)
  assert.deepEqual(seen, [1, 3])
  assert.equal((await session.getItems()).length, 4)
})

test('memory tools remember, recall (scoped, filtered) and forget through the agent tool surface', async () => {
  const s = store()
  const [remember, recall, forget] = createMemoryTools(s, { scope: { user: 'ada' } })
  assert.deepEqual([remember.name, recall.name, forget.name], ['memory_remember', 'memory_recall', 'memory_forget'])
  const ctx = undefined
  const call = (t, args) => t.invoke(ctx, JSON.stringify(args))
  const saved = await call(remember, { text: 'Ada prefers dark roast coffee', metadata: { topic: 'food' } })
  const id = /Remembered as (\S+)/.exec(saved)[1]
  await call(remember, { text: 'Ada lives in London', metadata: { topic: 'place' } })
  // another user's memory in the same store must be invisible
  const [, otherRecall] = createMemoryTools(s, { scope: { user: 'bob' } })
  assert.equal(await call(otherRecall, { query: 'coffee' }), 'No matching memories.')
  const found = await call(recall, { query: 'coffee roast', limit: 1 })
  assert.match(found, /dark roast/)
  assert.match(found, new RegExp(id))
  assert.match(await call(recall, { query: 'Ada', filter: { topic: 'place' } }), /London/)
  assert.equal(await call(forget, { id }), `Forgot ${id}`)
  assert.equal(await call(forget, { id }), `No memory ${id}`)
  assert.doesNotMatch(await call(recall, { query: 'coffee roast' }), /dark roast/)
  // bob cannot forget ada's note
  const [, , bobForget] = createMemoryTools(s, { scope: { user: 'bob' } })
  const [kept] = await s.list({ topic: 'place' }).then((p) => p.records)
  assert.equal(await call(bobForget, { id: kept.id }), `No memory ${kept.id}`)
})

test('an agent with memory tools calls them through a real Runner turn', async () => {
  const s = store()
  const tools = createMemoryTools(s)
  let turn = 0
  const model = {
    async getResponse() {
      turn += 1
      const usage = { requests: 1, inputTokens: 1, outputTokens: 1, totalTokens: 2, inputTokensDetails: [], outputTokensDetails: [] }
      if (turn === 1) {
        return { usage, output: [{ type: 'function_call', callId: 'c1', name: 'memory_remember', arguments: JSON.stringify({ text: 'the launch code word is pelican' }), status: 'completed' }] }
      }
      return { usage, output: [{ type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: 'saved' }] }] }
    },
    async *getStreamedResponse() {
      throw new Error('not used')
    },
  }
  const agent = new Agent({ name: 'mem', instructions: 'use memory', model, tools })
  const result = await run(agent, 'remember the code word')
  assert.equal(result.finalOutput, 'saved')
  const hits = await s.recall('code word pelican', { k: 1 })
  assert.equal(hits[0].text, 'the launch code word is pelican')
})
