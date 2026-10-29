// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { createRequire } from 'node:module'
import type { DocumentInput, DocumentResult, HmsConfigJs, HolographicMemorySystem, ListedDocument, SearchOptions } from 'holographic-memory'
import { HashingEmbedder, type Embedder, type Reranker } from './embedders.js'
import { HmsError } from './errors.js'
import { matches, parseFilter, splitFilter, type FilterLike, type Metadata } from './filters.js'

export const RESERVED_PREFIX = '_hms_'
const COLLECTION_KEY = '_hms_collection'
const VECTOR_KEY = '_hms_vector'
const MAX_ID_BYTES = 256
const NATIVE_MAX = 1000

export interface HmsRecord {
  id: string
  text: string
  metadata?: Metadata
  /** Embedding for the whole record; used for every chunk of it. Omit to embed with the store's embedder. */
  vector?: ArrayLike<number>
}

export interface StoredRecord {
  id: string
  text: string
  metadata: Metadata
  /** The stored embedding; present only when the store was created with `keepVectors: true`. */
  vector?: number[]
}

export interface Hit extends StoredRecord {
  /** Score of the best passage: fused lexical and cosine rank score, or the reranker score. */
  score: number
  lexicalScore: number
  semanticScore?: number
  /** Byte span of the best passage inside `text` of the stored record. */
  span: { start: number; end: number }
}

export interface Page {
  records: StoredRecord[]
  /** True when more records may match than one query can return (see `maxFetch`). */
  truncated: boolean
}

export interface RecallOptions {
  k?: number
  filter?: FilterLike
  /** Precomputed query embedding; skips the embedder. */
  vector?: ArrayLike<number>
  /** Drop hits below this cosine similarity (passed to the engine as minSemanticScore). */
  minSemanticScore?: number
}

export interface HmsStoreOptions {
  /** An existing native memory. Its embedding space must match `embedder` (default: the hashing embedder). */
  memory?: HolographicMemorySystem
  /** Storage directory for a store this class creates. Omit for an in-memory store. */
  path?: string
  /** Hypervector dimensions of a store this class creates (default 4096). */
  dimensions?: number
  /** Encoder. Default: `HashingEmbedder(384)`, lexical and dependency free. */
  embedder?: Embedder
  /** Optional reranker applied to the top candidates; needs stored text (always on here). */
  rerank?: Reranker
  /** Scope records inside one store. Ids are namespaced and every query restricted to it. */
  collection?: string
  /** Counts the first line of each document this many times in BM25 statistics. */
  titleWeight?: number
  bm25K1?: number
  bm25B?: number
  lexicalWeight?: number
  semanticWeight?: number
  /** `blend` (default) or `rrf`. */
  fusion?: 'blend' | 'rrf'
  chunkWords?: number
  overlapWords?: number
  /** Legacy configuration accepted for compatibility; residual filters now select metadata first. */
  overfetch?: number
  /** Ceiling on passages fetched per query, at most 1000 (the engine limit; default 1000). */
  maxFetch?: number
  /**
   * Keep each record's embedding (the first passage's) in its metadata so `get`, `list` and `recall` can
   * return it and a record can be re-saved without recomputing it. Costs about 20 bytes per dimension.
   */
  keepVectors?: boolean
  /** Extra native config for a store this class creates, e.g. `{ encryptionEnabled: true, encryptionPassphraseEnv: 'HMS_PASSPHRASE' }`. */
  hmsConfig?: HmsConfigJs
}

function bytes(s: string): number {
  return Buffer.byteLength(s, 'utf8')
}

function sameSpace(a: Embedder['space'], b: Record<string, unknown> | undefined): boolean {
  return !!b && (['model', 'revision', 'dimensions', 'normalization', 'metric'] as const).every((k) => a[k] === b[k])
}

function loadNative(): typeof import('holographic-memory') {
  try {
    return createRequire(import.meta.url)('holographic-memory')
  } catch (cause) {
    throw new HmsError('Install the holographic-memory native package to create a store from a path', 'backend', { cause })
  }
}

function requireMetadata(memory: HolographicMemorySystem): void {
  if (typeof memory.listDocuments !== 'function' || typeof memory.getDocumentInfo !== 'function') {
    throw new HmsError('install a current native binding with metadata paging and document info support', 'backend')
  }
}

export class HmsStore {
  readonly memory: HolographicMemorySystem
  readonly embedder: Embedder
  readonly collection: string
  private readonly opts: HmsStoreOptions
  private readonly maxFetch: number
  private readonly probe: number[]
  private writes: Promise<unknown> = Promise.resolve()
  private closed = false

  constructor(opts: HmsStoreOptions = {}) {
    this.opts = opts
    const overfetch = opts.overfetch ?? 5
    this.maxFetch = opts.maxFetch ?? NATIVE_MAX
    if (!Number.isInteger(overfetch) || overfetch < 1) throw new HmsError('overfetch must be an integer >= 1', 'invalid_argument')
    if (!Number.isInteger(this.maxFetch) || this.maxFetch < 1 || this.maxFetch > NATIVE_MAX) {
      throw new HmsError(`maxFetch must be 1..${NATIVE_MAX}`, 'invalid_argument')
    }
    if (opts.collection !== undefined && (opts.collection === '' || opts.collection.includes('/'))) {
      throw new HmsError("collection must be non-empty and contain no '/'", 'invalid_argument')
    }
    this.collection = opts.collection ?? ''
    if (opts.memory) {
      requireMetadata(opts.memory)
      this.memory = opts.memory
      const space = (opts.memory.securityStatus() as { embeddingSpace?: Record<string, unknown> }).embeddingSpace
      if (opts.embedder) this.embedder = opts.embedder
      else {
        const dims = Number(space?.dimensions)
        this.embedder = new HashingEmbedder(dims)
      }
      if (!sameSpace(this.embedder.space, space)) {
        throw new HmsError('The embedder does not match the memory store embedding space; pass the matching embedder', 'invalid_argument')
      }
    } else {
      this.embedder = opts.embedder ?? new HashingEmbedder(384)
      const { HolographicMemorySystem: Native } = loadNative()
      requireMetadata(Native.prototype)
      const s = this.embedder.space
      this.memory = new Native(opts.dimensions ?? 4096, opts.path ?? null, {
        embeddingModel: s.model,
        embeddingRevision: s.revision,
        embeddingDimensions: s.dimensions,
        ...opts.hmsConfig,
      })
    }
    this.probe = [1, ...new Array<number>(this.embedder.space.dimensions - 1).fill(0)]
  }

  private assertOpen(): void {
    if (this.closed) throw new HmsError('store is closed', 'closed')
  }

  private serialized<T>(op: () => Promise<T>): Promise<T> {
    const run = this.writes.then(op)
    this.writes = run.then(() => undefined, () => undefined)
    return run
  }

  private sid(id: string): string {
    if (typeof id !== 'string' || id === '') throw new HmsError('record id must be a non-empty string', 'invalid_argument')
    const sid = this.collection ? `${this.collection}/${id}` : id
    if (sid.startsWith('hms:')) throw new HmsError("ids may not start with 'hms:'", 'invalid_argument')
    if (bytes(sid) > MAX_ID_BYTES) throw new HmsError(`id exceeds ${MAX_ID_BYTES} bytes`, 'invalid_argument')
    return sid
  }

  private userId(sid: string): string {
    return this.collection ? sid.slice(this.collection.length + 1) : sid
  }

  private scope(pushed: Record<string, unknown>): Record<string, unknown> {
    return { ...pushed, [COLLECTION_KEY]: this.collection }
  }

  private clean(metadata: unknown): Metadata {
    const { [COLLECTION_KEY]: _c, [VECTOR_KEY]: _v, ...rest } = (metadata ?? {}) as Metadata
    return rest
  }

  private searchOptions(extra: SearchOptions): SearchOptions {
    const o = this.opts
    const base: SearchOptions = {}
    if (o.lexicalWeight !== undefined) base.lexicalWeight = o.lexicalWeight
    if (o.semanticWeight !== undefined) base.semanticWeight = o.semanticWeight
    if (o.fusion) base.fusion = o.fusion
    if (o.bm25K1 !== undefined) base.bm25K1 = o.bm25K1
    if (o.bm25B !== undefined) base.bm25B = o.bm25B
    return { ...base, ...extra }
  }

  private async native<T>(op: () => Promise<T>): Promise<T> {
    try {
      return await op()
    } catch (cause) {
      if (cause instanceof HmsError) throw cause
      throw new HmsError(`holographic-memory: ${(cause as Error).message}`, 'backend', { cause })
    }
  }

  /** Store records, replacing any with the same id. Returns the ids in order. */
  async remember(records: HmsRecord[]): Promise<string[]> {
    this.assertOpen()
    const prepared = records.map((r) => {
      const metadata = r.metadata ?? {}
      if (metadata === null || typeof metadata !== 'object' || Array.isArray(metadata)) throw new HmsError('metadata must be an object', 'invalid_argument')
      for (const key of Object.keys(metadata)) {
        if (key.startsWith(RESERVED_PREFIX)) throw new HmsError(`metadata key ${JSON.stringify(key)} uses the reserved prefix ${RESERVED_PREFIX}`, 'invalid_argument')
      }
      if (typeof r.text !== 'string') throw new HmsError('record text must be a string', 'invalid_argument')
      if (r.vector && r.vector.length !== this.embedder.space.dimensions) {
        throw new HmsError(`vector length ${r.vector.length} != embedding dimensions ${this.embedder.space.dimensions}`, 'invalid_argument')
      }
      const input: DocumentInput = {
        id: this.sid(r.id),
        text: r.text,
        metadata: { ...metadata, [COLLECTION_KEY]: this.collection },
        storeText: true,
      }
      if (this.opts.titleWeight !== undefined) input.titleWeight = this.opts.titleWeight
      if (this.opts.chunkWords !== undefined) input.chunkWords = this.opts.chunkWords
      if (this.opts.overlapWords !== undefined) input.overlapWords = this.opts.overlapWords
      return { record: r, input }
    })
    return this.serialized(() =>
      this.native(async () => {
        const chunked = await Promise.all(prepared.map(async (p) => ({ ...p, chunks: await this.memory.chunkDocument(p.input) })))
        const need = chunked.flatMap((c) => (c.record.vector ? [] : c.chunks.map((ch) => ch.text)))
        const vectors = need.length ? await this.embedder.encodeDocuments(need) : []
        if (vectors.length !== need.length) throw new HmsError('embedder returned the wrong number of vectors', 'backend')
        let next = 0
        for (const c of chunked) {
          const embeddings = c.chunks.map(() => Array.from(c.record.vector ?? (vectors[next++] as ArrayLike<number>)))
          const metadata = this.opts.keepVectors && embeddings[0] ? { ...c.input.metadata, [VECTOR_KEY]: embeddings[0] } : c.input.metadata
          await this.memory.memorizeDocument({ ...c.input, metadata, embeddings })
        }
        return records.map((r) => r.id)
      }),
    )
  }

  private async search(text: string, options: SearchOptions): Promise<DocumentResult[]> {
    return this.native(() => this.memory.searchDocuments(text, options))
  }

  private async *metadata(filter?: FilterLike): AsyncGenerator<ListedDocument> {
    const parsed = parseFilter(filter)
    let cursor: string | undefined
    for (;;) {
      const page = await this.native(() => this.memory.listDocuments(cursor, 1000))
      for (const record of page) {
        const md = record.metadata as Metadata
        if (md?.[COLLECTION_KEY] !== this.collection) continue
        if (parsed && !matches(parsed, this.clean(md))) continue
        yield record
      }
      if (page.length < 1000) return
      const next = page[page.length - 1]?.id
      if (next === undefined || (cursor !== undefined && next <= cursor)) {
        throw new HmsError('native metadata cursor did not advance', 'backend')
      }
      cursor = next
    }
  }

  /** Top-`k` records by fused BM25 and cosine rank (best passage per record). */
  async recall(query: string, options: RecallOptions = {}): Promise<Hit[]> {
    this.assertOpen()
    const k = options.k ?? 4
    if (!Number.isInteger(k) || k < 1) throw new HmsError('k must be an integer >= 1', 'invalid_argument')
    if (k > this.maxFetch) throw new HmsError(`k=${k} exceeds maxFetch=${this.maxFetch}`, 'limit_exceeded')
    const { pushed, residual } = splitFilter(parseFilter(options.filter))
    let eligible: string[] | undefined
    if (residual) {
      eligible = []
      for await (const record of this.metadata(options.filter)) {
        if (eligible.length === NATIVE_MAX) {
          throw new HmsError('fused residual filtering supports at most 1000 matching records; narrow the filter', 'filter_limit')
        }
        eligible.push(record.id)
      }
      if (!eligible.length) return []
    }
    // Hits are passages: several can belong to one record, so ask for more than `k`.
    const nativeK = Math.min(this.maxFetch, k * 2)
    const embedding = Array.from(options.vector ?? (await this.embedder.encodeQuery(query)))
    const so: SearchOptions = this.searchOptions({
      k: nativeK,
      candidateLimit: Math.min(NATIVE_MAX, Math.max(nativeK, 100)),
      filter: this.scope(pushed),
      embedding,
      documentIds: eligible,
    })
    if (options.minSemanticScore !== undefined) so.minSemanticScore = options.minSemanticScore
    let raw = await this.search(query, so)
    const desired = eligible ? Math.min(k, eligible.length) : k
    while (new Set(raw.map((hit) => hit.documentId)).size < desired && raw.length === so.k) {
      if (so.k === this.maxFetch) {
        throw new HmsError('the passage budget cannot prove a complete record result; increase maxFetch', 'limit_exceeded')
      }
      so.k = Math.min(this.maxFetch, (so.k as number) * 2)
      so.candidateLimit = Math.min(NATIVE_MAX, Math.max(so.k, 100))
      raw = await this.search(query, so)
    }
    if (this.opts.rerank && raw.length) {
      if (raw.some((r) => r.text == null)) throw new HmsError('reranking needs stored text', 'backend')
      const scores = await this.opts.rerank(query, raw.map((r) => r.text as string))
      if (scores.length !== raw.length || !scores.every(Number.isFinite)) throw new HmsError('reranker must return one finite score per passage', 'backend')
      raw = raw.map((r, i) => ({ ...r, score: scores[i] as number })).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    }
    const seen = new Set<string>()
    const best: DocumentResult[] = []
    for (const r of raw) {
      if (seen.has(r.documentId)) continue
      seen.add(r.documentId)
      const md = this.clean(r.metadata)
      if (residual && !matches(residual, md)) continue
      best.push(r)
      if (best.length === k) break
    }
    const full = await this.assemble(best.map((r) => r.documentId))
    return best.flatMap((r) => {
      const rec = full.get(r.documentId)
      if (!rec) return []
      const hit: Hit = {
        ...rec,
        score: r.score,
        lexicalScore: r.lexicalScore,
        span: { start: r.startByte, end: r.endByte },
      }
      if (r.semanticScore !== undefined) hit.semanticScore = r.semanticScore
      return [hit]
    })
  }

  /** Reassemble stored records from their passages using byte offsets. */
  private async assemble(sids: string[]): Promise<Map<string, StoredRecord>> {
    const out = new Map<string, StoredRecord>()
    let cursor = 0
    const worker = async (): Promise<void> => {
      for (let i = cursor++; i < sids.length; i = cursor++) {
        const sid = sids[i] as string
        const info = await this.native(() => this.memory.getDocumentInfo(sid))
        if (!info || (info.metadata as Metadata)?.[COLLECTION_KEY] !== this.collection) continue
        if (info.chunkCount > NATIVE_MAX) {
          throw new HmsError('record exceeds the native passage read limit; split it before storing', 'limit_exceeded')
        }
        const rows = await this.search('', {
          k: NATIVE_MAX,
          candidateLimit: NATIVE_MAX,
          documentIds: [sid],
          embedding: this.probe,
          minSemanticScore: -1,
          filter: this.scope({}),
        })
        const mine = rows.filter((r) => r.documentId === sid)
        if (!mine.length) continue
        if (mine.length !== info.chunkCount || mine.some((row) => row.version !== info.version)) {
          throw new HmsError('record changed while reading its passages; retry the operation', 'concurrent_mutation')
        }
        const parts = mine.filter((r) => r.text != null).sort((a, b) => a.startByte - b.startByte)
        const total = Math.max(...parts.map((p) => p.endByte))
        const buf = Buffer.alloc(total)
        for (const p of parts) buf.write(p.text as string, p.startByte, 'utf8')
        const raw = (mine[0]?.metadata ?? {}) as Metadata
        const rec: StoredRecord = { id: this.userId(sid), text: buf.toString('utf8'), metadata: this.clean(raw) }
        if (Array.isArray(raw[VECTOR_KEY])) rec.vector = raw[VECTOR_KEY] as number[]
        out.set(sid, rec)
      }
    }
    await Promise.all(Array.from({ length: Math.min(8, sids.length) }, worker))
    return out
  }

  /** Records for `ids` in order; `undefined` where an id is absent. */
  async get(ids: string[]): Promise<(StoredRecord | undefined)[]> {
    this.assertOpen()
    const sids = ids.map((id) => this.sid(id))
    const found = await this.assemble([...new Set(sids)])
    return sids.map((s) => found.get(s))
  }

  /** Up to `limit` matching records in id order (default and ceiling: `maxFetch`). */
  async list(filter?: FilterLike, limit?: number): Promise<Page> {
    this.assertOpen()
    if (limit !== undefined && (!Number.isInteger(limit) || limit < 1)) throw new HmsError('limit must be a positive integer', 'invalid_argument')
    const want = Math.min(limit ?? this.maxFetch, this.maxFetch)
    const sids: string[] = []
    for await (const record of this.metadata(filter)) {
      sids.push(record.id)
      if (sids.length > want) break
    }
    const full = await this.assemble(sids.slice(0, want))
    const records = sids.slice(0, want).flatMap((s) => (full.has(s) ? [full.get(s) as StoredRecord] : []))
    const truncated = sids.length > want
    return { records, truncated }
  }

  /** Delete by id. Returns how many existed. */
  async forget(ids: string[]): Promise<number> {
    this.assertOpen()
    const sids = ids.map((id) => this.sid(id))
    return this.serialized(() =>
      this.native(async () => {
        let removed = 0
        for (const sid of sids) if (await this.memory.deleteDocument(sid)) removed++
        return removed
      }),
    )
  }

  /**
   * Delete every matching record while paging metadata without retaining document text.
   */
  async forgetWhere(filter?: FilterLike): Promise<number> {
    let total = 0
    for await (const record of this.metadata(filter)) total += await this.forget([this.userId(record.id)])
    return total
  }

  /** Build the engine's optional indexes over stored embeddings. */
  async maintain(): Promise<unknown> {
    this.assertOpen()
    return this.serialized(() => this.native(() => this.memory.maintainIndices()))
  }

  async flush(): Promise<void> {
    this.assertOpen()
    await this.serialized(() => this.native(() => this.memory.flush()))
  }

  /** Flush and refuse further use. The native handle is released when garbage collected. */
  async close(): Promise<void> {
    if (this.closed) return
    await this.flush()
    this.closed = true
  }
}
