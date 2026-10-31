// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { randomUUID } from 'node:crypto'
import { Document, type DocumentInterface } from '@langchain/core/documents'
import type { EmbeddingsInterface } from '@langchain/core/embeddings'
import { VectorStore, type MaxMarginalRelevanceSearchOptions } from '@langchain/core/vectorstores'
import { maximalMarginalRelevance } from '@langchain/core/utils/math'
import type { Callbacks } from '@langchain/core/callbacks/manager'
import { WarmStore, type Embedder, type FilterLike, type Hit, type WarmStoreOptions } from '@writerslogic/warm-core'

/**
 * Filter accepted by every search method: a MongoDB-style object (`{ lang: 'en', year: { $gte: 2020 } }`)
 * or a core `Filter`. Top-level equalities are evaluated by the engine before ranking; everything
 * else is evaluated after an over-fetch, so it can return fewer than `k` documents.
 */
export type WarmFilter = Exclude<FilterLike, null | undefined>

export interface WarmVectorStoreOptions extends Omit<WarmStoreOptions, 'embedder'> {
  /**
   * Identity of the embedding space. The engine refuses to reopen a store with a different model
   * or revision, so vectors from different models are never mixed. Default model
   * `langchain-embeddings`, revision `unversioned`: set both whenever you persist to `path`.
   */
  embeddingSpace?: { model: string; revision: string }
}

export interface AddOptions {
  /** One id per document; defaults to `document.id`, then a random UUID. */
  ids?: string[]
}

export interface DeleteParams {
  ids?: string[]
  filter?: WarmFilter
  deleteAll?: boolean
}

function asEmbedder(embeddings: EmbeddingsInterface, dimensions: number, space: { model: string; revision: string }): Embedder {
  return {
    space: { model: space.model, revision: space.revision, dimensions, normalization: 'l2', metric: 'cosine' },
    encodeDocuments: (texts) => embeddings.embedDocuments(texts),
    encodeQuery: (text) => embeddings.embedQuery(text),
  }
}

function toDocument(hit: { id: string; text: string; metadata: Record<string, unknown> }): Document {
  return new Document({ id: hit.id, pageContent: hit.text, metadata: hit.metadata })
}

/**
 * LangChain.js `VectorStore` over WARM.
 *
 * Search is hybrid: `similaritySearch` and `similaritySearchWithScore` send the query text to the
 * engine (BM25 fused with cosine over the vector from `embeddings`). The returned score is the
 * fused rank score, higher is better, not a cosine. `similaritySearchVectorWithScore` has no text,
 * so it ranks by cosine only.
 */
export class WarmVectorStore extends VectorStore {
  declare FilterType: WarmFilter
  readonly store: WarmStore

  static lc_name(): string {
    return 'WarmVectorStore'
  }

  _vectorstoreType(): string {
    return 'writerslogic_warm'
  }

  /** Wrap an existing `WarmStore` whose embedder produces vectors in the same space as `embeddings`. */
  constructor(embeddings: EmbeddingsInterface, args: { store: WarmStore }) {
    super(embeddings, args)
    this.store = args.store
  }

  /** Build the store, probing the embedding dimension from `embeddings`. */
  static async create(embeddings: EmbeddingsInterface, options: WarmVectorStoreOptions = {}): Promise<WarmVectorStore> {
    const { embeddingSpace, ...rest } = options
    const probe = await embeddings.embedQuery('dimension probe')
    const space = embeddingSpace ?? { model: 'langchain-embeddings', revision: 'unversioned' }
    const store = new WarmStore({ ...rest, embedder: asEmbedder(embeddings, probe.length, space) })
    return new WarmVectorStore(embeddings, { store })
  }

  static override async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    options: WarmVectorStoreOptions & AddOptions = {},
  ): Promise<WarmVectorStore> {
    const { ids, ...rest } = options
    const docs = texts.map(
      (pageContent, i) =>
        new Document({ pageContent, metadata: Array.isArray(metadatas) ? (metadatas[i] as Record<string, unknown>) : (metadatas as Record<string, unknown>) }),
    )
    return WarmVectorStore.fromDocuments(docs, embeddings, ids ? { ...rest, ids } : rest)
  }

  static override async fromDocuments(
    docs: DocumentInterface[],
    embeddings: EmbeddingsInterface,
    options: WarmVectorStoreOptions & AddOptions = {},
  ): Promise<WarmVectorStore> {
    const { ids, ...rest } = options
    const vs = await WarmVectorStore.create(embeddings, rest)
    await vs.addDocuments(docs, ids ? { ids } : {})
    return vs
  }

  private records(documents: DocumentInterface[], options: AddOptions | undefined, vectors?: number[][]) {
    const ids = options?.ids
    if (ids && ids.length !== documents.length) throw new RangeError('ids must have one entry per document')
    return documents.map((d, i) => {
      const vector = vectors?.[i]
      return {
        id: ids?.[i] ?? d.id ?? randomUUID(),
        text: d.pageContent,
        metadata: { ...d.metadata },
        ...(vector ? { vector } : {}),
      }
    })
  }

  async addDocuments(documents: DocumentInterface[], options?: AddOptions): Promise<string[]> {
    if (!documents.length) return []
    return this.store.remember(this.records(documents, options))
  }

  async addVectors(vectors: number[][], documents: DocumentInterface[], options?: AddOptions): Promise<string[]> {
    if (vectors.length !== documents.length) throw new RangeError('vectors and documents must have the same length')
    if (!documents.length) return []
    return this.store.remember(this.records(documents, options, vectors))
  }

  /** Remove by `ids`, by `filter` (engine-evaluated equalities page through the whole match set), or `deleteAll`. */
  async delete(params: DeleteParams = {}): Promise<void> {
    const given = [params.ids !== undefined, params.filter !== undefined, params.deleteAll === true].filter(Boolean).length
    if (given !== 1) throw new TypeError('delete takes exactly one of ids, filter or deleteAll')
    if (params.ids) await this.store.forget(params.ids)
    else await this.store.forgetWhere(params.filter)
  }

  /** Stored documents by id; ids that do not exist are skipped. */
  async getByIds(ids: string[]): Promise<Document[]> {
    const recs = await this.store.get(ids)
    return recs.flatMap((r) => (r ? [toDocument(r)] : []))
  }

  async similaritySearchVectorWithScore(query: number[], k: number, filter?: WarmFilter): Promise<[DocumentInterface, number][]> {
    const hits = await this.store.recall('', { k, vector: query, ...(filter !== undefined ? { filter } : {}) })
    return hits.map((h) => [toDocument(h), h.score])
  }

  async similaritySearchWithScore(query: string, k = 4, filter?: WarmFilter, _callbacks?: Callbacks): Promise<[DocumentInterface, number][]> {
    const hits = await this.store.recall(query, { k, ...(filter !== undefined ? { filter } : {}) })
    return hits.map((h) => [toDocument(h), h.score])
  }

  async similaritySearch(query: string, k = 4, filter?: WarmFilter, callbacks?: Callbacks): Promise<DocumentInterface[]> {
    return (await this.similaritySearchWithScore(query, k, filter, callbacks)).map(([d]) => d)
  }

  /**
   * Maximal marginal relevance over the top `fetchK` hits. The engine returns text, not vectors,
   * so hit texts are re-embedded with the store's embedder; the vectors are of the whole stored
   * text, not of the best passage.
   */
  async maxMarginalRelevanceSearch(
    query: string,
    options: MaxMarginalRelevanceSearchOptions<WarmFilter>,
    _callbacks?: Callbacks,
  ): Promise<DocumentInterface[]> {
    const { k, fetchK = 20, lambda = 0.5, filter } = options
    const hits: Hit[] = await this.store.recall(query, { k: Math.max(k, fetchK), ...(filter !== undefined ? { filter } : {}) })
    if (!hits.length) return []
    const queryEmbedding = Array.from(await this.store.embedder.encodeQuery(query))
    const docEmbeddings = (await this.store.embedder.encodeDocuments(hits.map((h) => h.text))).map((v) => Array.from(v))
    return maximalMarginalRelevance(queryEmbedding, docEmbeddings, lambda, k).map((i) => toDocument(hits[i] as Hit))
  }
}
