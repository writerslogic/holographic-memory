// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import type { BaseEmbedding } from '@llamaindex/core/embeddings'
import { MetadataMode, type BaseNode, type TextNode } from '@llamaindex/core/schema'
import {
  BaseVectorStore,
  VectorStoreQueryMode,
  metadataDictToNode,
  nodeToMetadata,
  type VectorStoreQuery,
  type VectorStoreQueryResult,
} from '@llamaindex/core/vector-store'
import { HashingEmbedder, WarmError, WarmStore, type Embedder, type Filter, type Hit, type WarmStoreOptions } from '@writerslogic/warm-core'
import { toFilter } from './filters.js'

export interface WarmVectorStoreOptions extends Omit<WarmStoreOptions, 'embedder'> {
  /** Embedding model of the index. Used for hybrid queries without a query embedding and for MMR. */
  embedModel?: BaseEmbedding
  /** Identity of the embedding space; set both whenever you persist to `path` (default `llamaindex-embeddings` / `unversioned`). */
  embeddingSpace?: { model: string; revision: string }
  /**
   * Vector length of the embeddings you will add when no `embedModel` is given (default 384). The
   * store then uses a dependency-free hashing embedder only for text it must embed itself.
   */
  embeddingDimensions?: number
  /** Candidates fetched per result in MMR mode (default 4, the LlamaIndex default). */
  mmrPrefetchFactor?: number
}

const NODE_ID_KEY = 'node_id'

function asEmbedder(model: BaseEmbedding, dimensions: number, space: { model: string; revision: string }): Embedder {
  return {
    space: { model: space.model, revision: space.revision, dimensions, normalization: 'l2', metric: 'cosine' },
    encodeDocuments: (texts) => model.getTextEmbeddings(texts),
    encodeQuery: (text) => model.getTextEmbedding(text),
  }
}

/**
 * LlamaIndex.TS vector store (`BaseVectorStore`) over WARM. It stores node text, so
 * `VectorStoreIndex.fromVectorStore(store)` works without a docstore.
 *
 * Query modes: `default` ranks by cosine of `queryEmbedding` (similarities are cosine scores);
 * `hybrid` and `semantic_hybrid` fuse BM25 over `queryStr` with that cosine (similarities are the
 * engine's fused rank score; `alpha` is ignored, set `lexicalWeight`/`semanticWeight`/`fusion` on
 * the store); `mmr` re-ranks the top `similarityTopK * mmrPrefetchFactor` for diversity using
 * `mmrThreshold` as lambda. `sparse`, `svm` and the regression modes throw.
 */
export class WarmVectorStore extends BaseVectorStore<WarmStore> {
  storesText = true
  readonly store: WarmStore
  private readonly mmrPrefetchFactor: number

  constructor(args: { store: WarmStore; embedModel?: BaseEmbedding; mmrPrefetchFactor?: number }) {
    super(args.embedModel ? { embeddingModel: args.embedModel } : {})
    this.store = args.store
    this.mmrPrefetchFactor = args.mmrPrefetchFactor ?? 4
  }

  /** Build the store, probing the embedding dimension from `embedModel` (or `embeddingDimensions` hashing dimensions without one). */
  static async create(options: WarmVectorStoreOptions = {}): Promise<WarmVectorStore> {
    const { embedModel, embeddingSpace, embeddingDimensions, mmrPrefetchFactor, ...rest } = options
    let embedder: Embedder | undefined
    if (embedModel) {
      const probe = await embedModel.getTextEmbedding('dimension probe')
      embedder = asEmbedder(embedModel, probe.length, embeddingSpace ?? { model: 'llamaindex-embeddings', revision: 'unversioned' })
    }
    else if (embeddingDimensions !== undefined) embedder = new HashingEmbedder(embeddingDimensions)
    const store = new WarmStore({ ...rest, ...(embedder ? { embedder } : {}) })
    return new WarmVectorStore({ store, ...(embedModel ? { embedModel } : {}), ...(mmrPrefetchFactor !== undefined ? { mmrPrefetchFactor } : {}) })
  }

  client(): WarmStore {
    return this.store
  }

  async add(nodes: BaseNode[]): Promise<string[]> {
    if (!nodes.length) return []
    return this.store.remember(
      nodes.map((node) => {
        const embedding = node.embedding
        const metadata = nodeToMetadata(node, true)
        metadata[NODE_ID_KEY] = node.id_
        return {
          id: node.id_,
          text: node.getContent(MetadataMode.NONE),
          metadata,
          ...(embedding ? { vector: embedding } : {}),
        }
      }),
    )
  }

  /** Delete every node whose `ref_doc_id` is `refDocId` (LlamaIndex's per-source-document delete). */
  async delete(refDocId: string): Promise<void> {
    await this.store.forgetWhere({ ref_doc_id: refDocId })
  }

  async deleteNodes(nodeIds: string[]): Promise<void> {
    await this.store.forget(nodeIds)
  }

  /** Remove everything in this store's collection. */
  async clear(): Promise<void> {
    await this.store.forgetWhere()
  }

  async getNodes(nodeIds: string[]): Promise<BaseNode[]> {
    const recs = await this.store.get(nodeIds)
    return recs.flatMap((r) => (r ? [this.node(r)] : []))
  }

  private node(rec: { id: string; text: string; metadata: Record<string, unknown> }): BaseNode {
    const { [NODE_ID_KEY]: _n, ...metadata } = rec.metadata
    const node = metadataDictToNode(metadata, { fallback: { text: rec.text, id_: rec.id } })
    // `add` stores the text in the record and blanks it inside `_node_content`; restore it.
    ;(node as TextNode).text = rec.text
    return node
  }

  async query(query: VectorStoreQuery): Promise<VectorStoreQueryResult> {
    const k = query.similarityTopK
    const filters: Filter[] = []
    const user = toFilter(query.filters)
    if (user) filters.push(user)
    if (query.docIds?.length) filters.push({ type: 'cmp', key: NODE_ID_KEY, op: 'in', value: query.docIds })
    const filter: Filter | undefined = filters.length === 0 ? undefined : filters.length === 1 ? filters[0] : { type: 'and', terms: filters }
    const base = filter ? { filter } : {}
    const vector = query.queryEmbedding

    let hits: Hit[]
    let cosine = false
    switch (query.mode) {
      case VectorStoreQueryMode.DEFAULT:
        if (!vector) throw new WarmError('default mode needs queryEmbedding', 'invalid_argument')
        hits = await this.store.recall('', { k, vector, ...base })
        cosine = true
        break
      case VectorStoreQueryMode.HYBRID:
      case VectorStoreQueryMode.SEMANTIC_HYBRID:
        if (!query.queryStr) throw new WarmError('hybrid mode needs queryStr', 'invalid_argument')
        hits = await this.store.recall(query.queryStr, { k, ...(vector ? { vector } : {}), ...base })
        break
      case VectorStoreQueryMode.MMR:
        hits = await this.mmr(query, k, base)
        cosine = true
        break
      default:
        throw new WarmError(`query mode "${query.mode}" is not supported; use default, hybrid, semantic_hybrid or mmr`, 'invalid_argument')
    }
    return {
      nodes: hits.map((h) => this.node(h)),
      similarities: hits.map((h) => (cosine ? (h.semanticScore ?? h.score) : h.score)),
      ids: hits.map((h) => h.id),
    }
  }

  private async mmr(query: VectorStoreQuery, k: number, base: { filter?: Filter }): Promise<Hit[]> {
    const vector = query.queryEmbedding
    if (!vector) throw new WarmError('mmr mode needs queryEmbedding', 'invalid_argument')
    const embedder = this.store.embedder
    if (embedder.space.dimensions !== vector.length) {
      throw new WarmError('mmr re-embeds hit text; build the store with WarmVectorStore.create({ embedModel }) so the embedder matches queryEmbedding', 'invalid_argument')
    }
    const pool = await this.store.recall('', { k: Math.max(k, k * this.mmrPrefetchFactor), vector, ...base })
    if (pool.length <= 1) return pool
    const docs = (await embedder.encodeDocuments(pool.map((h) => h.text))).map((v) => Array.from(v))
    const lambda = query.mmrThreshold ?? 0.5
    const relevance = docs.map((d) => cos(vector, d))
    const chosen: number[] = []
    const rest = new Set(pool.keys())
    while (chosen.length < Math.min(k, pool.length)) {
      let best = -1
      let bestScore = -Infinity
      for (const i of rest) {
        const redundancy = chosen.length ? Math.max(...chosen.map((j) => cos(docs[i] as number[], docs[j] as number[]))) : 0
        const score = lambda * (relevance[i] as number) - (1 - lambda) * redundancy
        if (score > bestScore) {
          bestScore = score
          best = i
        }
      }
      chosen.push(best)
      rest.delete(best)
    }
    return chosen.map((i) => pool[i] as Hit)
  }
}

function cos(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number
    const y = b[i] as number
    dot += x * y
    na += x * x
    nb += y * y
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0
}
