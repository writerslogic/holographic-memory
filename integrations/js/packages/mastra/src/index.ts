// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { randomUUID } from 'node:crypto'
import { ErrorCategory, ErrorDomain, MastraError } from '@mastra/core/error'
import {
  MastraVector,
  validateTopK,
  validateUpsert,
  type CreateIndexParams,
  type DeleteIndexParams,
  type DeleteVectorParams,
  type DeleteVectorsParams,
  type DescribeIndexParams,
  type IndexStats,
  type QueryResult,
  type QueryVectorParams,
  type UpdateVectorParams,
  type UpsertVectorParams,
  type VectorFilter,
} from '@mastra/core/vector'
import { WarmError, WarmStore, parseFilter, type Embedder, type FilterLike, type WarmStoreOptions, type Metadata, type StoredRecord } from '@writerslogic/warm-core'
import type { WarmMemorySystem } from 'writerslogic-warm'

const REGISTRY = '__warm_indexes__'
const STORE_NAME = 'HOLOGRAPHIC'

export interface HolographicVectorOptions
  extends Pick<WarmStoreOptions, 'path' | 'warmConfig' | 'maxFetch' | 'overfetch' | 'dimensions'> {
  id: string
  /** Vector length of every index in this store. Fixed when the store is created. Default 1536. */
  dimension?: number
  /** An existing native memory created for this dimension (embedding model `mastra-supplied`). */
  memory?: WarmMemorySystem
  /** Hypervector dimensions of a created store (default 4096). */
  hypervectorDimensions?: number
}

function suppliedEmbedder(dimensions: number): Embedder {
  const refuse = async (): Promise<never> => {
    throw new WarmError('Mastra supplies vectors; text is never embedded by this store', 'invalid_argument')
  }
  return {
    space: { model: 'mastra-supplied', revision: '1', dimensions, normalization: 'l2', metric: 'cosine' },
    encodeDocuments: refuse,
    encodeQuery: refuse,
  }
}

function normalizeDates(value: unknown): unknown {
  if (value instanceof Date) return value.toISOString()
  if (Array.isArray(value)) return value.map(normalizeDates)
  if (value !== null && typeof value === 'object' && !(value instanceof RegExp)) {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, normalizeDates(v)]))
  }
  return value
}

/**
 * Mastra vector store over the WARM native engine.
 *
 * An index is a collection inside one engine store; all indexes share the store's vector dimension.
 * Scores are cosine similarity. Filters use Mastra's MongoDB-style syntax.
 */
export class HolographicVector extends MastraVector<VectorFilter> {
  readonly dimension: number
  private readonly memory: WarmMemorySystem
  private readonly options: HolographicVectorOptions
  private readonly embedder: Embedder
  private readonly registry: WarmStore
  private readonly stores = new Map<string, WarmStore>()
  private lock: Promise<unknown> = Promise.resolve()

  constructor(options: HolographicVectorOptions) {
    super({ id: options.id })
    this.options = options
    this.dimension = options.dimension ?? 1536
    this.embedder = suppliedEmbedder(this.dimension)
    const base = new WarmStore({
      embedder: this.embedder,
      dimensions: options.hypervectorDimensions ?? options.dimensions,
      ...(options.memory ? { memory: options.memory } : {}),
      ...(options.path ? { path: options.path } : {}),
      ...(options.warmConfig ? { warmConfig: options.warmConfig } : {}),
      ...(options.maxFetch ? { maxFetch: options.maxFetch } : {}),
      collection: REGISTRY,
    })
    this.memory = base.memory
    this.registry = base
  }

  private fail(id: string, cause: unknown, category = ErrorCategory.THIRD_PARTY, details: Record<string, string | number> = {}): never {
    if (cause instanceof MastraError) throw cause
    const code = cause instanceof WarmError && cause.code === 'invalid_argument' ? ErrorCategory.USER : category
    throw new MastraError(
      { id: `${STORE_NAME}_VECTOR_${id}_FAILED` as Uppercase<string>, domain: ErrorDomain.MASTRA_VECTOR, category: code, details },
      cause,
    )
  }

  private user(id: string, text: string, details: Record<string, string | number> = {}): never {
    throw new MastraError({ id: `${STORE_NAME}_VECTOR_${id}_FAILED` as Uppercase<string>, domain: ErrorDomain.MASTRA_VECTOR, category: ErrorCategory.USER, text, details })
  }

  private exclusive<T>(op: () => Promise<T>): Promise<T> {
    const run = this.lock.then(op)
    this.lock = run.then(() => undefined, () => undefined)
    return run
  }

  private store(indexName: string): WarmStore {
    let s = this.stores.get(indexName)
    if (!s) {
      s = new WarmStore({
        memory: this.memory,
        embedder: this.embedder,
        collection: indexName,
        keepVectors: true,
        lexicalWeight: 0,
        semanticWeight: 1,
        ...(this.options.maxFetch ? { maxFetch: this.options.maxFetch } : {}),
        ...(this.options.overfetch ? { overfetch: this.options.overfetch } : {}),
      })
      this.stores.set(indexName, s)
    }
    return s
  }

  private unitVector(): number[] {
    return [1, ...new Array<number>(this.dimension - 1).fill(0)]
  }

  private async describe(indexName: string): Promise<{ dimension: number; metric: 'cosine'; count: number } | undefined> {
    const [rec] = await this.registry.get([indexName])
    if (!rec) return undefined
    return rec.metadata as { dimension: number; metric: 'cosine'; count: number }
  }

  private async requireIndex(indexName: string): Promise<NonNullable<Awaited<ReturnType<HolographicVector['describe']>>>> {
    const d = await this.describe(indexName)
    if (!d) this.user('INDEX_NOT_FOUND', `Index ${indexName} does not exist; call createIndex first`, { indexName })
    return d as NonNullable<typeof d>
  }

  private async setCount(indexName: string, delta: number): Promise<void> {
    const d = await this.requireIndex(indexName)
    await this.registry.remember([{ id: indexName, text: indexName, metadata: { ...d, count: Math.max(0, d.count + delta) }, vector: this.unitVector() }])
  }

  private translate(filter: unknown): FilterLike {
    if (filter === undefined || filter === null) return undefined
    try {
      return parseFilter(normalizeDates(filter) as Record<string, unknown>)
    } catch (cause) {
      return this.user('INVALID_FILTER', (cause as Error).message)
    }
  }

  async createIndex({ indexName, dimension, metric = 'cosine' }: CreateIndexParams): Promise<void> {
    if (!indexName || indexName.includes('/') || indexName === REGISTRY) this.user('INVALID_INDEX_NAME', `Invalid index name ${JSON.stringify(indexName)}`)
    if (metric !== 'cosine') this.user('UNSUPPORTED_METRIC', `Only the cosine metric is supported, not ${metric}`, { metric })
    if (!Number.isInteger(dimension) || dimension !== this.dimension) {
      this.user('DIMENSION_MISMATCH', `This store holds ${this.dimension}-dimensional vectors; got ${dimension}`, { dimension: String(dimension) })
    }
    await this.exclusive(async () => {
      const existing = await this.describe(indexName)
      if (existing) return
      await this.registry.remember([{ id: indexName, text: indexName, metadata: { dimension, metric, count: 0 }, vector: this.unitVector() }])
    })
  }

  async listIndexes(): Promise<string[]> {
    return (await this.registry.list()).records.map((r) => r.id).sort()
  }

  async describeIndex({ indexName }: DescribeIndexParams): Promise<IndexStats> {
    const d = await this.requireIndex(indexName)
    return { dimension: d.dimension, count: d.count, metric: d.metric }
  }

  async deleteIndex({ indexName }: DeleteIndexParams): Promise<void> {
    await this.exclusive(async () => {
      if (!(await this.describe(indexName))) return
      await this.store(indexName).forgetWhere()
      await this.registry.forget([indexName])
      this.stores.delete(indexName)
    })
  }

  async upsert({ indexName, vectors, metadata, ids, sparseVectors, deleteFilter }: UpsertVectorParams): Promise<string[]> {
    if (sparseVectors?.length) this.user('SPARSE_UNSUPPORTED', 'Sparse vectors are not supported')
    validateUpsert(STORE_NAME, vectors, metadata, ids, true)
    await this.requireIndex(indexName)
    for (const v of vectors) {
      if (v.length !== this.dimension) this.user('DIMENSION_MISMATCH', `Expected ${this.dimension}-dimensional vectors, got ${v.length}`)
    }
    const outIds = ids ?? vectors.map(() => randomUUID())
    if (new Set(outIds).size !== outIds.length) this.user('DUPLICATE_IDS', 'ids must be unique within one upsert')
    const store = this.store(indexName)
    const filter = deleteFilter === undefined ? undefined : this.translate(deleteFilter)
    return this.exclusive(async () => {
      try {
        let delta = 0
        if (filter) delta -= await store.forgetWhere(filter)
        const existing = await store.get(outIds)
        delta += existing.filter((r) => r === undefined).length
        await store.remember(
          vectors.map((vector, i) => {
            const md = (normalizeDates(metadata?.[i] ?? {}) as Metadata)
            const text = typeof md['text'] === 'string' && md['text'] ? (md['text'] as string) : (outIds[i] as string)
            return { id: outIds[i] as string, text, metadata: md, vector }
          }),
        )
        await this.setCount(indexName, delta)
        return outIds
      } catch (cause) {
        return this.fail('UPSERT', cause, ErrorCategory.THIRD_PARTY, { indexName })
      }
    })
  }

  async query({ indexName, queryVector, topK = 10, filter, includeVector = false }: QueryVectorParams<VectorFilter>): Promise<QueryResult[]> {
    validateTopK(STORE_NAME, topK)
    if (!queryVector && !filter) this.user('QUERY_REQUIRES_VECTOR_OR_FILTER', 'Provide queryVector, filter, or both')
    await this.requireIndex(indexName)
    const store = this.store(indexName)
    const f = this.translate(filter)
    try {
      if (!queryVector) {
        const page = await store.list(f, topK)
        return page.records.map((r) => this.result(r, 0, includeVector))
      }
      if (queryVector.length !== this.dimension) this.user('DIMENSION_MISMATCH', `Expected a ${this.dimension}-dimensional query vector, got ${queryVector.length}`)
      const hits = await store.recall('', { k: topK, filter: f, vector: queryVector, minSemanticScore: -1 })
      return hits.map((h) => this.result(h, h.semanticScore ?? h.score, includeVector))
    } catch (cause) {
      return this.fail('QUERY', cause, ErrorCategory.THIRD_PARTY, { indexName })
    }
  }

  private result(r: StoredRecord, score: number, includeVector: boolean): QueryResult {
    const out: QueryResult = { id: r.id, score, metadata: r.metadata }
    if (includeVector && r.vector) out.vector = r.vector
    return out
  }

  async updateVector(params: UpdateVectorParams<VectorFilter>): Promise<void> {
    const { indexName, update } = params
    if (update.vector && update.vector.length !== this.dimension) this.user('DIMENSION_MISMATCH', `Expected ${this.dimension}-dimensional vector, got ${update.vector.length}`)
    if (!update.vector && !update.metadata) this.user('EMPTY_UPDATE', 'update needs a vector, metadata, or both')
    await this.requireIndex(indexName)
    const store = this.store(indexName)
    await this.exclusive(async () => {
      try {
        let targets: StoredRecord[]
        if ('id' in params && params.id !== undefined) {
          const [rec] = await store.get([params.id])
          if (!rec) this.user('NOT_FOUND', `No vector with id ${params.id} in ${indexName}`, { indexName })
          targets = [rec as StoredRecord]
        } else {
          targets = (await store.list(this.translate((params as { filter: unknown }).filter))).records
        }
        for (let i = 0; i < targets.length; i += 100) {
          await store.remember(
            targets.slice(i, i + 100).map((rec) => {
              const md = { ...rec.metadata, ...(normalizeDates(update.metadata ?? {}) as Metadata) }
              const vector = update.vector ?? rec.vector
              if (!vector) this.user('VECTOR_UNAVAILABLE', `Vector for ${rec.id} was not stored`)
              const text = typeof md['text'] === 'string' && md['text'] ? (md['text'] as string) : rec.id
              return { id: rec.id, text, metadata: md, vector: vector as number[] }
            }),
          )
        }
      } catch (cause) {
        this.fail('UPDATE_VECTOR', cause, ErrorCategory.THIRD_PARTY, { indexName })
      }
    })
  }

  async deleteVector({ indexName, id }: DeleteVectorParams): Promise<void> {
    await this.deleteVectors({ indexName, ids: [id] })
  }

  async deleteVectors({ indexName, ids, filter }: DeleteVectorsParams<VectorFilter>): Promise<void> {
    if ((ids === undefined) === (filter === undefined)) this.user('DELETE_REQUIRES_IDS_OR_FILTER', 'Provide exactly one of ids or filter')
    await this.requireIndex(indexName)
    const store = this.store(indexName)
    await this.exclusive(async () => {
      try {
        const removed = ids !== undefined ? await store.forget(ids) : await store.forgetWhere(this.translate(filter))
        if (removed) await this.setCount(indexName, -removed)
      } catch (cause) {
        this.fail('DELETE_VECTORS', cause, ErrorCategory.THIRD_PARTY, { indexName })
      }
    })
  }

  /** Flush pending writes to disk. */
  async flush(): Promise<void> {
    await this.registry.flush()
  }
}

export { WarmError }
