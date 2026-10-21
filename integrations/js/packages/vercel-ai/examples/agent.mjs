// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// Runs offline with a scripted model: node examples/agent.mjs
// Swap `model` for a real provider model (e.g. openai('...')) and nothing else changes.
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ToolLoopAgent, wrapLanguageModel } from 'ai'
import { MockLanguageModelV4 } from 'ai/test'
import { HmsStore } from '@holographic-memory/core'
import { memoryMiddleware, memoryTools } from '../dist/index.js'

const store = new HmsStore({ path: mkdtempSync(join(tmpdir(), 'hms-example-')), dimensions: 1024 })
const usage = { inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined }, outputTokens: { total: 1, text: 1, reasoning: undefined } }

const script = [
  { content: [{ type: 'tool-call', toolCallId: '1', toolName: 'remember', input: JSON.stringify({ text: 'My favorite editor is Neovim' }) }], finishReason: { unified: 'tool-calls' }, usage, warnings: [] },
  { content: [{ type: 'text', text: 'Saved.' }], finishReason: { unified: 'stop' }, usage, warnings: [] },
  { content: [{ type: 'text', text: 'Your favorite editor is Neovim.' }], finishReason: { unified: 'stop' }, usage, warnings: [] },
]
let step = 0
const base = new MockLanguageModelV4({ doGenerate: async () => script[step++] })

// Tools let the model decide what to save; the middleware recalls before every call.
const agent = new ToolLoopAgent({
  model: wrapLanguageModel({ model: base, middleware: memoryMiddleware(store, { k: 3 }) }),
  tools: memoryTools(store),
})

console.log((await agent.generate({ prompt: 'Remember that my favorite editor is Neovim' })).text)
console.log((await agent.generate({ prompt: 'What is my favorite editor?' })).text)
console.log('model saw:', JSON.stringify(base.doGenerateCalls.at(-1).prompt.at(-1).content.map((p) => p.text)))
await store.close()
