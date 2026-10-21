// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { BaseListChatMessageHistory } from '@langchain/core/chat_history'
import {
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
  type BaseMessage,
  type StoredMessage,
} from '@langchain/core/messages'
import type { HmsStore } from '@holographic-memory/core'
import { OrderedLog, packPayload, unpackPayload } from './log.js'

export interface HmsChatMessageHistoryInput {
  store: HmsStore
  sessionId: string
}

function plainText(message: BaseMessage): string {
  const c = message.content
  if (typeof c === 'string') return c
  return c.map((part) => (typeof part === 'string' ? part : 'text' in part && typeof part.text === 'string' ? part.text : '')).filter(Boolean).join('\n')
}

/**
 * Chat history for one session. Messages keep their order and are unbounded (see `OrderedLog`).
 * The full serialized message lives in metadata, the readable text in the record, so history is
 * also searchable with `search()`. Use a store dedicated to history, or one `collection` per
 * purpose. One writer per session.
 */
export class HmsChatMessageHistory extends BaseListChatMessageHistory {
  lc_namespace = ['holographic_memory', 'chat_history']
  readonly sessionId: string
  private readonly store: HmsStore
  private readonly log: OrderedLog

  constructor(input: HmsChatMessageHistoryInput) {
    super()
    this.store = input.store
    this.sessionId = input.sessionId
    this.log = new OrderedLog(input.store, `chat:${input.sessionId}`)
  }

  async getMessages(): Promise<BaseMessage[]> {
    const entries = await this.log.tail()
    return mapStoredMessagesToChatMessages(entries.map((e) => unpackPayload<StoredMessage>(e)))
  }

  async addMessage(message: BaseMessage): Promise<void> {
    await this.addMessages([message])
  }

  async addMessages(messages: BaseMessage[]): Promise<void> {
    const stored = mapChatMessagesToStoredMessages(messages)
    await this.log.append(
      messages.map((m, i) => packPayload(plainText(m), stored[i], { role: m.getType(), session: this.sessionId })),
    )
  }

  async clear(): Promise<void> {
    await this.log.clear()
  }

  /** Messages of this session most relevant to `query` (hybrid BM25 and cosine), best first. */
  async search(query: string, k = 4): Promise<BaseMessage[]> {
    const hits = await this.store.recall(query, { k, filter: { session: this.sessionId } })
    return mapStoredMessagesToChatMessages(hits.map((h) => unpackPayload<StoredMessage>(h)))
  }
}
