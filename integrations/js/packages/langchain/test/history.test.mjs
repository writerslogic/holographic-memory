// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
import { HmsChatMessageHistory, HmsStore } from '../dist/index.js'

const tmp = () => mkdtempSync(join(tmpdir(), 'hms-lc-hist-'))
const store = (path = tmp()) => new HmsStore({ path, dimensions: 1024 })

const dist = pathToFileURL(new URL('../dist/index.js', import.meta.url).pathname).href
/** The engine holds a file lock until the handle is garbage collected, so reopening is tested from a fresh process. */
const child = (code) => execFileSync(process.execPath, ['--input-type=module', '-e', code], { encoding: 'utf8', cwd: new URL('..', import.meta.url).pathname })

test('messages keep their order and types, and survive a process restart', async () => {
  const path = tmp()
  // The engine locks a path for the life of the process, so the writer is a short-lived child.
  child(`
    import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
    import { HmsChatMessageHistory, HmsStore } from ${JSON.stringify(dist)}
    const s = new HmsStore({ path: ${JSON.stringify(path)}, dimensions: 1024 })
    const h = new HmsChatMessageHistory({ store: s, sessionId: 's1' })
    await h.addMessage(new SystemMessage('You are terse.'))
    await h.addUserMessage('What is the capital of France?')
    await h.addAIMessage('Paris.')
    await h.addMessages([new HumanMessage('And Spain?'), new AIMessage({ content: 'Madrid.', additional_kwargs: { tag: 1 } }), new ToolMessage({ content: '42', tool_call_id: 'c1' })])
    await s.close()
  `)
  const msgs = await new HmsChatMessageHistory({ store: store(path), sessionId: 's1' }).getMessages()
  assert.deepEqual(msgs.map((m) => [m.getType(), m.content, m.additional_kwargs.tag ?? null, m.tool_call_id ?? null]), [
    ['system', 'You are terse.', null, null],
    ['human', 'What is the capital of France?', null, null],
    ['ai', 'Paris.', null, null],
    ['human', 'And Spain?', null, null],
    ['ai', 'Madrid.', 1, null],
    ['tool', '42', null, 'c1'],
  ])
})

test('sessions are isolated and clear only removes one', async () => {
  const s = store()
  const a = new HmsChatMessageHistory({ store: s, sessionId: 'a' })
  const b = new HmsChatMessageHistory({ store: s, sessionId: 'b' })
  await a.addUserMessage('from a')
  await b.addUserMessage('from b')
  await a.clear()
  assert.deepEqual(await a.getMessages(), [])
  assert.deepEqual((await b.getMessages()).map((m) => m.content), ['from b'])
  await a.addUserMessage('after clear')
  assert.deepEqual((await a.getMessages()).map((m) => m.content), ['after clear'])
})

for (const n of [99, 100, 101, 250]) {
  test(`${n} messages come back complete and in order (batch boundary 100)`, async () => {
    const h = new HmsChatMessageHistory({ store: store(), sessionId: `long${n}` })
    const batch = Array.from({ length: n }, (_, i) => (i % 2 ? new AIMessage(`m${i}`) : new HumanMessage(`m${i}`)))
    await h.addMessages(batch.slice(0, 40))
    await h.addMessages(batch.slice(40))
    const msgs = await h.getMessages()
    assert.deepEqual(msgs.map((m) => m.content), batch.map((m) => m.content))
  })
}

test('a second instance on the same session appends after existing messages', async () => {
  const s = store()
  const one = new HmsChatMessageHistory({ store: s, sessionId: 'shared' })
  await one.addUserMessage('first')
  const two = new HmsChatMessageHistory({ store: s, sessionId: 'shared' })
  await two.addUserMessage('second')
  await one.addUserMessage('third')
  assert.deepEqual((await two.getMessages()).map((m) => m.content), ['first', 'second', 'third'])
})

test('large and multi-chunk messages round trip exactly', async () => {
  const h = new HmsChatMessageHistory({ store: new HmsStore({ path: tmp(), dimensions: 1024, chunkWords: 50, overlapWords: 5 }), sessionId: 'big' })
  const big = Array.from({ length: 20000 }, (_, i) => `wörd${i}`).join(' ')
  await h.addMessages([new HumanMessage(big), new AIMessage([{ type: 'text', text: 'block reply' }])])
  const [m1, m2] = await h.getMessages()
  assert.equal(m1.content, big)
  assert.deepEqual(m2.content, [{ type: 'text', text: 'block reply' }])
})

test('search finds the relevant message of this session only', async () => {
  const s = store()
  const a = new HmsChatMessageHistory({ store: s, sessionId: 'sa' })
  const b = new HmsChatMessageHistory({ store: s, sessionId: 'sb' })
  await a.addMessages([new HumanMessage('my favourite colour is teal'), new HumanMessage('I live in Lisbon')])
  await b.addMessages([new HumanMessage('teal is also my colour')])
  const res = await a.search('favourite colour teal', 1)
  assert.equal(res.length, 1)
  assert.equal(res[0].content, 'my favourite colour is teal')
})
