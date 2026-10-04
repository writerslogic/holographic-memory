import { HolographicMemorySystem, DocumentInput, DocumentResult, SearchOptions } from './index'

export interface Embedder {
  space: { model: string; revision: string; dimensions: number; normalization: 'l2'; metric: 'cosine' }
  encodeDocuments(texts: string[]): Promise<Array<ArrayLike<number>>>
  encodeQuery(text: string): Promise<ArrayLike<number>>
  dispose?(): Promise<unknown>
}
export type Reranker = ((query: string, passages: string[]) => Promise<number[]>) & { dispose?: () => Promise<unknown> }
export function createLocalEmbedder(options: {
  /** The returned space revision includes a fingerprint of pooling, prefixes, and dtype. */
  modelPath: string; modelId: string; revision: string; queryPrefix?: string; documentPrefix?: string; dtype?: string
}): Promise<Embedder>
export function createLocalReranker(options: { modelPath: string; dtype?: string }): Promise<Reranker>
export class DocumentMemory {
  constructor(memory: HolographicMemorySystem, options?: { embedder?: Embedder; rerank?: Reranker; maxPending?: number })
  memorize(input: DocumentInput): Promise<number>
  ingest(documents: Iterable<DocumentInput> | AsyncIterable<DocumentInput>): Promise<number>
  search(text: string, options?: SearchOptions): Promise<DocumentResult[]>
  delete(id: string): Promise<boolean>
  flush(): Promise<void>
  maintainIndices(): ReturnType<HolographicMemorySystem['maintainIndices']>
}
