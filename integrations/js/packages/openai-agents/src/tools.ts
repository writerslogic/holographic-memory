// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { randomUUID } from 'node:crypto'
import { tool } from '@openai/agents'
import { z } from 'zod'
import type { WarmStore } from '@writerslogic/warm-core'

export interface MemoryToolOptions {
  /** Tool name prefix; default `memory_` gives memory_remember, memory_recall, memory_forget. */
  prefix?: string
  /** Default and maximum results of the recall tool (default 4 and 20). */
  k?: number
  maxK?: number
  /** Metadata attached to every remembered note and required by recall/forget (e.g. `{ user: 'ada' }`). */
  scope?: Record<string, string | number | boolean>
}

const scalar = z.union([z.string(), z.number(), z.boolean()])

/**
 * Three function tools for an `Agent`: remember a note, recall notes relevant to a query, and
 * forget a note by id. `scope` partitions memory per user or agent without separate stores.
 */
export function createMemoryTools(store: WarmStore, options: MemoryToolOptions = {}) {
  const prefix = options.prefix ?? 'memory_'
  const defaultK = options.k ?? 4
  const maxK = options.maxK ?? 20
  const scope = options.scope ?? {}

  const remember = tool({
    name: `${prefix}remember`,
    description: 'Save a fact or note to long-term memory so it can be recalled in later conversations. Returns the note id.',
    parameters: z.object({
      text: z.string().min(1).describe('The note to store, as a self-contained statement.'),
      metadata: z.record(z.string(), scalar).optional().describe('Optional flat key/value tags for later filtering.'),
    }),
    async execute({ text, metadata }) {
      const [id] = await store.remember([{ id: `note-${randomUUID()}`, text, metadata: { ...(metadata ?? {}), ...scope } }])
      return `Remembered as ${id}`
    },
  })

  const recall = tool({
    name: `${prefix}recall`,
    description: 'Search long-term memory for notes relevant to a query. Returns the best matches, most relevant first, with their ids.',
    parameters: z.object({
      query: z.string().min(1).describe('What to look for.'),
      limit: z.number().int().min(1).max(maxK).optional().describe(`Number of notes to return (default ${defaultK}).`),
      filter: z.record(z.string(), scalar).optional().describe('Optional exact-match filter on note tags.'),
    }),
    async execute({ query, limit, filter }) {
      const hits = await store.recall(query, { k: Math.min(limit ?? defaultK, maxK), filter: { ...(filter ?? {}), ...scope } })
      if (!hits.length) return 'No matching memories.'
      return hits.map((h, i) => `${i + 1}. [${h.id}] ${h.text}`).join('\n')
    },
  })

  const forget = tool({
    name: `${prefix}forget`,
    description: 'Delete a note from long-term memory by its id.',
    parameters: z.object({ id: z.string().min(1).describe('The note id returned by remember or recall.') }),
    async execute({ id }) {
      if (Object.keys(scope).length) {
        const [rec] = await store.get([id])
        if (!rec || !Object.entries(scope).every(([k, v]) => rec.metadata[k] === v)) return `No memory ${id}`
      }
      const removed = await store.forget([id])
      return removed ? `Forgot ${id}` : `No memory ${id}`
    },
  })

  return [remember, recall, forget] as const
}
