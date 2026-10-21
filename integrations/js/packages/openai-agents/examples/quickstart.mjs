// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Run: node examples/quickstart.mjs   (after `npm run build` in this package; no API key needed:
// a scripted model stands in for the LLM. Remove `model` to use the SDK default, which needs OPENAI_API_KEY.)
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Agent, run, setTracingDisabled } from '@openai/agents'
import { HmsSession, HmsStore, createMemoryTools } from '@holographic-memory/openai-agents'

setTracingDisabled(true)
const store = new HmsStore({ path: mkdtempSync(join(tmpdir(), 'hms-agents-example-')), dimensions: 4096, titleWeight: 1 })

// 1. Conversation memory: pass the session to run().
const session = new HmsSession({ store, sessionId: 'ada-chat' })

// 2. Long-term memory: tools the model can call, scoped per user.
const tools = createMemoryTools(store, { scope: { user: 'ada' } })

const usage = { requests: 1, inputTokens: 1, outputTokens: 1, totalTokens: 2, inputTokensDetails: [], outputTokensDetails: [] }
const say = (text) => ({ usage, output: [{ type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text }] }] })
const call = (name, args) => ({ usage, output: [{ type: 'function_call', callId: `c-${name}`, name, arguments: JSON.stringify(args), status: 'completed' }] })

// A scripted model: remembers a note on turn 1, then recalls it on turn 2 by reading the tool output.
const script = [
  () => call('memory_remember', { text: 'Ada prefers dark roast coffee' }),
  () => say('Noted.'),
  () => call('memory_recall', { query: 'coffee' }),
  (items) => say(`From memory: ${items.at(-1).output?.text ?? JSON.stringify(items.at(-1).output)}`),
]
let step = 0
const model = {
  async getResponse(request) {
    const items = Array.isArray(request.input) ? request.input : []
    return script[step++](items)
  },
  async *getStreamedResponse() {
    throw new Error('not used')
  },
}

const agent = new Agent({ name: 'assistant', instructions: 'Use memory tools to remember and recall facts.', model, tools })
console.log((await run(agent, 'Remember that I like dark roast coffee.', { session })).finalOutput)
console.log((await run(agent, 'What coffee do I like?', { session })).finalOutput)
console.log('session items:', (await session.getItems()).length)
console.log('search the session:', (await session.search('coffee', 1)).map((i) => i.type ?? i.role))
