// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { createHash } from 'node:crypto'

/**
 * The encoder knob. `space` identifies the embedding space; the engine refuses a store reopened
 * with a different one. This is the same shape as `Embedder` in `holographic-memory/semantic`, so
 * `createLocalEmbedder(...)` results can be passed directly.
 */
export interface Embedder {
  space: { model: string; revision: string; dimensions: number; normalization: 'l2'; metric: 'cosine' }
  encodeDocuments(texts: string[]): Promise<ArrayLike<number>[]>
  encodeQuery(text: string): Promise<ArrayLike<number>>
  dispose?(): Promise<unknown>
}

export type Reranker = (query: string, passages: string[]) => Promise<number[]>

const TOKEN = /[\p{L}\p{N}_]+/gu

/** Signed feature hashing of lowercase word unigrams and bigrams, L2 normalized. Lexical, not semantic. */
export class HashingEmbedder implements Embedder {
  readonly space: Embedder['space']
  constructor(readonly dimensions = 384, private readonly bigrams = true) {
    if (!Number.isInteger(dimensions) || dimensions < 8) throw new RangeError('dimensions must be an integer >= 8')
    this.space = { model: 'hms-hashing', revision: bigrams ? 'uni-bi-v1' : 'uni-v1', dimensions, normalization: 'l2', metric: 'cosine' }
  }

  private encode(text: string): number[] {
    const words = text.toLowerCase().match(TOKEN) ?? []
    const feats = [...words]
    if (this.bigrams) for (let i = 0; i + 1 < words.length; i++) feats.push(`${words[i]} ${words[i + 1]}`)
    const vec = new Array<number>(this.dimensions).fill(0)
    for (const f of feats) {
      const d = createHash('sha256').update(f).digest()
      const idx = Number(d.readBigUInt64LE(0) % BigInt(this.dimensions))
      vec[idx] = (vec[idx] ?? 0) + ((d[8] ?? 0) & 1 ? 1 : -1)
    }
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0))
    if (norm === 0) {
      vec[0] = 1 // the engine rejects zero vectors; empty text maps to a fixed axis
      return vec
    }
    return vec.map((v) => v / norm)
  }

  async encodeDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.encode(t))
  }

  async encodeQuery(text: string): Promise<number[]> {
    return this.encode(text)
  }
}
