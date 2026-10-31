// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// A real Mastra Agent with Memory semantic recall backed by HolographicVector, driven by a scripted model.
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { Agent } from '@mastra/core/agent'
import { LibSQLStore } from '@mastra/libsql'
import { Memory } from '@mastra/memory'
import { HolographicVector } from '../dist/index.js'

const DIM = 64
const words = ['neovim', 'editor', 'favorite', 'deploy', 'friday', 'rotation', 'weather', 'sunny', 'coffee', 'tea', 'lunch', 'meeting']
// Bag-of-words toy embedding: deterministic, shared by documents and queries.
const embed = (text) => {
  const v = new Array(DIM).fill(0)
  for (const w of text.toLowerCase().match(/[a-z]+/g) ?? []) {
    const i = words.indexOf(w)
    v[i >= 0 ? i : 12 + (w.length % (DIM - 12))] += 1
  }
  return v.some((x) => x) ? v : (v[0] = 1, v)
}
const embedder = {
  specificationVersion: 'v2',
  provider: 'test',
  modelId: 'bow',
  maxEmbeddingsPerCall: 100,
  supportsParallelCalls: true,
  doEmbed: async ({ values }) => ({ embeddings: values.map(embed), usage: { tokens: 1 } }),
}

const seen = []
const model = {
  specificationVersion: 'v2',
  provider: 'test',
  modelId: 'scripted',
  supportedUrls: {},
  doGenerate: async (options) => {
    seen.push(options.prompt)
    return { content: [{ type: 'text', text: 'ok' }], finishReason: 'stop', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 }, warnings: [] }
  },
  doStream: async (options) => {
    seen.push(options.prompt)
    return {
      stream: new ReadableStream({
        start(c) {
          c.enqueue({ type: 'stream-start', warnings: [] })
          c.enqueue({ type: 'text-start', id: 't' })
          c.enqueue({ type: 'text-delta', id: 't', delta: 'ok' })
          c.enqueue({ type: 'text-end', id: 't' })
          c.enqueue({ type: 'finish', finishReason: 'stop', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } })
          c.close()
        },
      }),
    }
  },
}

test('semantic recall retrieves an old message through HolographicVector', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'warm-mastra-agent-'))
  const vector = new HolographicVector({ id: 'warm-vector', dimension: DIM, path: join(dir, 'warm'), hypervectorDimensions: 1024 })
  const memory = new Memory({
    storage: new LibSQLStore({ id: 'store', url: `file:${join(dir, 'mastra.db')}` }),
    vector,
    embedder,
    options: { lastMessages: 1, semanticRecall: { topK: 2, messageRange: 0 } },
  })
  const agent = new Agent({ id: 'a', name: 'a', instructions: 'be brief', model, memory })
  const thread = { thread: 'thread-1', resource: 'user-1' }

  await agent.generate('My favorite editor is neovim', { memory: thread })
  await agent.generate('The weather is sunny and I drink coffee', { memory: thread })
  await agent.generate('Lunch meeting is on Friday', { memory: thread })
  seen.length = 0
  await agent.generate('which editor is my favorite', { memory: thread })

  const prompt = JSON.stringify(seen[0])
  assert.match(prompt, /neovim/, 'the old message was recalled from HolographicVector into the prompt')
  assert.match(prompt, /which editor is my favorite/)

  const indexes = await vector.listIndexes()
  assert.ok(indexes.length >= 1, 'Memory created its message index in the HolographicVector store')
  const { count } = await vector.describeIndex({ indexName: indexes[0] })
  assert.ok(count >= 3)
})
