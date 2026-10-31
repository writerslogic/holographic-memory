// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { generateText, isStepCount, streamText, wrapLanguageModel } from 'ai'
import { MockEmbeddingModelV4, MockLanguageModelV4, MockRerankingModelV4 } from 'ai/test'
import { WarmStore } from '@writerslogic/warm-core'
import { aiSdkEmbedder, aiSdkReranker, memoryMiddleware, memoryTools } from '../dist/index.js'

const store = (opts = {}) => new WarmStore({ path: mkdtempSync(join(tmpdir(), 'warm-vai-')), dimensions: 1024, ...opts })
const usage = { inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined }, outputTokens: { total: 1, text: 1, reasoning: undefined } }
const stop = { unified: 'stop', raw: undefined }
const text = (t) => ({ content: [{ type: 'text', text: t }], finishReason: stop, usage, warnings: [] })
const call = (toolName, input, id) => ({
  content: [{ type: 'tool-call', toolCallId: id, toolName, input: JSON.stringify(input) }],
  finishReason: { unified: 'tool-calls', raw: undefined },
  usage,
  warnings: [],
})

test('middleware injects recalled memories into the last user message', async () => {
  const s = store()
  await s.remember([
    { id: 'm1', text: 'The user favorite editor is Neovim', metadata: { userId: 'u1' } },
    { id: 'm2', text: 'The user favorite editor is Emacs', metadata: { userId: 'u2' } },
  ])
  const inner = new MockLanguageModelV4({ doGenerate: async () => text('ok') })
  const model = wrapLanguageModel({ model: inner, middleware: memoryMiddleware(s, { k: 2, scope: { userId: 'u1' } }) })
  await generateText({ model, prompt: 'What is my favorite editor?' })
  const sent = inner.doGenerateCalls[0].prompt
  const last = sent.at(-1)
  assert.equal(last.role, 'user')
  const joined = last.content.map((p) => p.text).join('\n')
  assert.match(joined, /Neovim/)
  assert.doesNotMatch(joined, /Emacs/)
  assert.match(joined, /What is my favorite editor\?/)
})

test('middleware works for streaming and leaves the prompt alone when nothing matches', async () => {
  const s = store()
  const inner = new MockLanguageModelV4({
    doStream: async () => ({
      stream: new ReadableStream({
        start(c) {
          c.enqueue({ type: 'text-start', id: 't' })
          c.enqueue({ type: 'text-delta', id: 't', delta: 'hi' })
          c.enqueue({ type: 'text-end', id: 't' })
          c.enqueue({ type: 'finish', finishReason: stop, usage })
          c.close()
        },
      }),
    }),
  })
  const model = wrapLanguageModel({ model: inner, middleware: memoryMiddleware(s) })
  const r = streamText({ model, prompt: 'anything' })
  assert.equal(await r.text, 'hi')
  assert.equal(inner.doStreamCalls[0].prompt.at(-1).content.length, 1)
})

test('tools: remember, recall and forget through a real tool loop', async () => {
  const s = store()
  const steps = [
    call('remember', { text: 'The deploy key rotates on Fridays', metadata: { team: 'infra' } }, 'c1'),
    call('recall', { query: 'when does the deploy key rotate', filter: { team: 'infra' } }, 'c2'),
    text('It rotates on Fridays.'),
  ]
  let i = 0
  const model = new MockLanguageModelV4({ doGenerate: async () => steps[i++] })
  const result = await generateText({
    model,
    tools: memoryTools(s, { scope: { userId: 'u1' } }),
    stopWhen: isStepCount(5),
    prompt: 'Remember the deploy key rotation and then check it',
  })
  assert.equal(result.text, 'It rotates on Fridays.')
  const recalled = result.steps[1].toolResults[0].output
  assert.equal(recalled.memories.length, 1)
  assert.match(recalled.memories[0].text, /Fridays/)
  assert.deepEqual(recalled.memories[0].metadata, { team: 'infra', userId: 'u1' })
  const id = recalled.memories[0].id

  // Another user's scope cannot delete it; its owner can.
  const other = memoryTools(s, { scope: { userId: 'u2' } })
  assert.deepEqual(await other.forget.execute({ ids: [id] }, { toolCallId: 'x', messages: [] }), { deleted: 0 })
  const mine = memoryTools(s, { scope: { userId: 'u1' } })
  assert.deepEqual(await mine.forget.execute({ ids: [id] }, { toolCallId: 'y', messages: [] }), { deleted: 1 })
  assert.equal((await s.get([id]))[0], undefined)
})

test('invalid tool input is rejected by the schema', async () => {
  const s = store()
  const steps = [call('recall', { query: 'x', k: 0 }, 'bad'), text('done')]
  let i = 0
  const model = new MockLanguageModelV4({ doGenerate: async () => steps[i++] })
  const result = await generateText({ model, tools: memoryTools(s), stopWhen: isStepCount(3), prompt: 'go' })
  const part = result.steps[0].content.find((p) => p.type === 'tool-error')
  assert.ok(part, 'schema violation surfaces as a tool error, not a store call')
  assert.equal(Object.keys(memoryTools(s, { readOnly: true })).includes('forget'), false)
})

test('aiSdkEmbedder drives the store and aiSdkReranker reorders results', async () => {
  const dims = 8
  const vec = (v) => Array.from({ length: dims }, (_, i) => (v.includes('apple') ? (i === 0 ? 1 : 0) : i === 1 ? 1 : 0))
  const embedding = new MockEmbeddingModelV4({
    maxEmbeddingsPerCall: 100,
    supportsParallelCalls: true,
    doEmbed: async ({ values }) => ({ embeddings: values.map(vec), usage: { tokens: values.length }, warnings: [] }),
  })
  const embedder = aiSdkEmbedder(embedding, { model: 'mock-embed', revision: '1', dimensions: dims })
  const s = store({ embedder, lexicalWeight: 0, semanticWeight: 1 })
  await s.remember([{ id: 'a', text: 'apple pie recipe' }, { id: 'b', text: 'bicycle repair manual' }])
  const hits = await s.recall('apple', { k: 1 })
  assert.equal(hits[0].id, 'a')

  const reranking = new MockRerankingModelV4({
    doRerank: async ({ documents }) => ({
      ranking: documents.values.map((d, index) => ({ index, relevanceScore: d.includes('bicycle') ? 0.9 : 0.1 })),
      warnings: [],
    }),
  })
  const r = store({ embedder, rerank: aiSdkReranker(reranking), lexicalWeight: 0, semanticWeight: 1 })
  await r.remember([{ id: 'a', text: 'apple pie recipe' }, { id: 'b', text: 'bicycle repair manual' }])
  assert.equal((await r.recall('apple', { k: 2 }))[0].id, 'b')
})
