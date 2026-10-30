// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { embed, embedMany, rerank, tool, type EmbeddingModel, type LanguageModelMiddleware, type RerankingModel, type Tool } from 'ai'
import { z } from 'zod'
import type { Embedder, FilterLike, Hit, HmsStore, Metadata, Reranker } from '@holographic-memory/core'

type Params = Parameters<NonNullable<LanguageModelMiddleware['transformParams']>>[0]['params']

const metadataSchema = z.record(z.string(), z.union([z.string(), z.number(), z.boolean()]))
const rememberSchema = z.object({
  text: z.string().min(1).describe('The information to remember, written as a self-contained statement'),
  id: z.string().min(1).optional().describe('Stable id; reusing an id replaces that memory'),
  metadata: metadataSchema.optional().describe('Flat key-value tags usable as recall filters'),
})
const forgetSchema = z.object({ ids: z.array(z.string().min(1)).min(1).max(100) })

function recallSchema(defaultK: number) {
  return z.object({
    query: z.string().min(1).describe('What to look for'),
    k: z.number().int().min(1).max(50).optional().describe(`Number of memories to return (default ${defaultK})`),
    filter: metadataSchema.optional().describe('Only memories whose metadata equals these values'),
  })
}

export type MemoryTools = Record<string,
  | Tool<z.infer<typeof rememberSchema>, { id: string; stored: boolean }>
  | Tool<z.infer<ReturnType<typeof recallSchema>>, { memories: ReturnType<typeof publicHit>[] }>
  | Tool<z.infer<typeof forgetSchema>, { deleted: number }>
>

export interface MemoryToolsOptions {
  /** Metadata merged into every remembered record and ANDed into every recall/forget filter, e.g. `{ userId }`. */
  scope?: Metadata
  /** Default number of memories returned by `recall` (default 4). */
  k?: number
  /** Tool names. Defaults: remember, recall, forget. */
  names?: { remember?: string; recall?: string; forget?: string }
  /** Omit `forget` from the tool set (default false). */
  readOnly?: boolean
}

function scoped(scope: Metadata | undefined, filter?: FilterLike): FilterLike {
  if (!scope || Object.keys(scope).length === 0) return filter
  if (!filter) return scope
  return { $and: [scope, filter as Record<string, unknown>] }
}

function publicHit(h: Hit) {
  return { id: h.id, text: h.text, metadata: h.metadata, score: h.score }
}

/**
 * Memory tools for `generateText`, `streamText` and `ToolLoopAgent`: spread the result into `tools`.
 * Inputs are validated by zod; `execute` returns plain JSON.
 */
export function memoryTools(store: HmsStore, options: MemoryToolsOptions = {}): MemoryTools {
  const names = { remember: 'remember', recall: 'recall', forget: 'forget', ...options.names }
  const defaultK = options.k ?? 4
  const remember = tool({
    description: 'Save a piece of information to long-term memory so it can be recalled in later conversations.',
    inputSchema: rememberSchema,
    execute: async ({ text, id, metadata }) => {
      const rid = id ?? `mem-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
      await store.remember([{ id: rid, text, metadata: { ...metadata, ...options.scope } }])
      return { id: rid, stored: true }
    },
  })
  const recall = tool({
    description: 'Search long-term memory for information relevant to a query. Returns the best matching memories.',
    inputSchema: recallSchema(defaultK),
    execute: async ({ query, k, filter }) => ({
      memories: (await store.recall(query, { k: k ?? defaultK, filter: scoped(options.scope, filter) })).map(publicHit),
    }),
  })
  const forget = tool({
    description: 'Delete memories by id.',
    inputSchema: forgetSchema,
    execute: async ({ ids }) => {
      if (options.scope) {
        const owned = (await store.get(ids)).filter((r): r is NonNullable<typeof r> => r !== undefined)
          .filter((r) => Object.entries(options.scope as Metadata).every(([k, v]) => r.metadata[k] === v))
        return { deleted: await store.forget(owned.map((r) => r.id)) }
      }
      return { deleted: await store.forget(ids) }
    },
  })
  return {
    [names.remember]: remember,
    [names.recall]: recall,
    ...(options.readOnly ? {} : { [names.forget]: forget }),
  }
}

export interface MemoryMiddlewareOptions {
  k?: number
  filter?: FilterLike
  scope?: Metadata
  /** Keep only hits whose cosine similarity is at least this (needs a semantic embedder). */
  minSemanticScore?: number
  /** Cap on characters of recalled text injected into the prompt (default 4000). */
  maxChars?: number
  /** Build the injected text. Default lists each memory on its own line under a short header. */
  format?: (hits: Hit[]) => string
  /** Query text from the prompt. Default: the text of the last user message. */
  queryFrom?: (params: Params) => string | undefined
}

function lastUserText(params: Params): string | undefined {
  for (let i = params.prompt.length - 1; i >= 0; i--) {
    const m = params.prompt[i]
    if (m?.role !== 'user') continue
    const text = m.content.flatMap((p) => (p.type === 'text' ? [p.text] : [])).join('\n').trim()
    return text || undefined
  }
  return undefined
}

function defaultFormat(hits: Hit[]): string {
  return 'Relevant long-term memories:\n' + hits.map((h) => `- ${h.text}`).join('\n')
}

/**
 * Language model middleware that recalls memories for the last user message and appends them to it.
 * Use with `wrapLanguageModel({ model, middleware: memoryMiddleware(store) })`. The model call gets the
 * recalled text; stored conversation history is untouched.
 */
export function memoryMiddleware(store: HmsStore, options: MemoryMiddlewareOptions = {}): LanguageModelMiddleware {
  const maxChars = options.maxChars ?? 4000
  const format = options.format ?? defaultFormat
  return {
    specificationVersion: 'v4',
    transformParams: async ({ params }) => {
      const query = (options.queryFrom ?? lastUserText)(params)
      if (!query) return params
      const recallOptions: { k: number; filter: FilterLike; minSemanticScore?: number } = {
        k: options.k ?? 4,
        filter: scoped(options.scope, options.filter),
      }
      if (options.minSemanticScore !== undefined) recallOptions.minSemanticScore = options.minSemanticScore
      const hits = await store.recall(query, recallOptions)
      if (hits.length === 0) return params
      const text = format(hits).slice(0, maxChars)
      const prompt = [...params.prompt]
      for (let i = prompt.length - 1; i >= 0; i--) {
        const m = prompt[i]
        if (m?.role === 'user') {
          prompt[i] = { ...m, content: [...m.content, { type: 'text', text }] }
          return { ...params, prompt }
        }
      }
      return params
    },
  }
}

/** Use an AI SDK embedding model as the store's encoder. `dimensions` must match the model's output. */
export function aiSdkEmbedder(
  model: EmbeddingModel,
  space: { model: string; revision: string; dimensions: number },
): Embedder {
  return {
    space: { ...space, normalization: 'l2', metric: 'cosine' },
    async encodeDocuments(texts) {
      if (texts.length === 0) return []
      const { embeddings } = await embedMany({ model, values: texts })
      return embeddings
    },
    async encodeQuery(text) {
      return (await embed({ model, value: text })).embedding
    },
  }
}

/** Use an AI SDK reranking model as the store's `rerank` hook. */
export function aiSdkReranker(model: RerankingModel): Reranker {
  return async (query, passages) => {
    const { ranking } = await rerank({ model, query, documents: passages })
    const scores = new Array<number>(passages.length).fill(Number.NEGATIVE_INFINITY)
    for (const r of ranking) scores[r.originalIndex] = r.score
    return scores
  }
}
