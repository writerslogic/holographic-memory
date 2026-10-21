// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { randomUUID } from 'node:crypto'
import type { AgentInputItem, Session } from '@openai/agents'
import type { HmsStore } from '@holographic-memory/core'
import { OrderedLog, packPayload, unpackPayload } from './log.js'

export interface HmsSessionOptions {
  store: HmsStore
  /** Defaults to a random UUID. */
  sessionId?: string
}

function partText(part: unknown): string {
  if (typeof part === 'string') return part
  if (part && typeof part === 'object') {
    const p = part as Record<string, unknown>
    if (typeof p.text === 'string') return p.text
    if (typeof p.transcript === 'string') return p.transcript
  }
  return ''
}

/** Readable text of an item, used for the record text so history is searchable. */
export function itemText(item: AgentInputItem): string {
  const it = item as unknown as Record<string, unknown>
  switch (it.type) {
    case 'function_call':
      return `${String(it.name ?? '')} ${String(it.arguments ?? '')}`.trim()
    case 'function_call_result': {
      const out = it.output
      return typeof out === 'string' ? out : JSON.stringify(out ?? '')
    }
    default: {
      const c = it.content
      if (typeof c === 'string') return c
      if (Array.isArray(c)) return c.map(partText).filter(Boolean).join('\n')
      return ''
    }
  }
}

/**
 * OpenAI Agents SDK `Session` stored in Holographic Memory: pass it to `run(agent, input, { session })`.
 * History is ordered and unbounded; `getItems(limit)` returns the latest `limit` items oldest first,
 * the same contract as the SDK's `MemorySession`. Use one writer per session id.
 */
export class HmsSession implements Session {
  readonly sessionId: string
  private readonly store: HmsStore
  private readonly log: OrderedLog

  constructor(options: HmsSessionOptions) {
    this.store = options.store
    this.sessionId = options.sessionId ?? randomUUID()
    this.log = new OrderedLog(options.store, `session:${this.sessionId}`)
  }

  async getSessionId(): Promise<string> {
    return this.sessionId
  }

  async getItems(limit?: number): Promise<AgentInputItem[]> {
    if (limit !== undefined && limit <= 0) return []
    const entries = await this.log.tail(limit)
    return entries.map((e) => unpackPayload<AgentInputItem>(e))
  }

  async addItems(items: AgentInputItem[]): Promise<void> {
    await this.log.append(
      items.map((item) => packPayload(itemText(item), item, { session: this.sessionId, kind: String((item as { type?: string }).type ?? 'message') })),
    )
  }

  async popItem(): Promise<AgentInputItem | undefined> {
    const entry = await this.log.pop()
    return entry ? (unpackPayload<AgentInputItem>(entry)) : undefined
  }

  async clearSession(): Promise<void> {
    await this.log.clear()
  }

  /** Items of this session most relevant to `query` (hybrid BM25 and cosine), best first. */
  async search(query: string, k = 4): Promise<AgentInputItem[]> {
    const hits = await this.store.recall(query, { k, filter: { session: this.sessionId } })
    return hits.map((h) => unpackPayload<AgentInputItem>(h))
  }
}
