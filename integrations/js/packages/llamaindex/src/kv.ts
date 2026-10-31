// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { BaseKVStore } from '@llamaindex/core/storage/kv-store'
import type { StoredValue } from '@llamaindex/core/schema'
import { WarmError, type WarmStore } from '@writerslogic/warm-core'

const DEFAULT_COLLECTION = 'data'

/**
 * LlamaIndex `BaseKVStore` over WARM, for `KVDocumentStore`, `KVIndexStore` and
 * `storageContextFromDefaults({ docStore, indexStore })`. Values are JSON text; every record
 * carries a constant vector, so nothing is embedded. Use a store (or `collection`) dedicated to
 * key-value data.
 *
 * Limits: `get`, `put` and `delete` are exact and unbounded. `getAll` has to enumerate a
 * collection through metadata pages, retaining at most `maxFetch` records (default 1000), so it
 * throws `limit_exceeded` instead of silently returning a partial map when a collection is larger.
 * `KVDocumentStore.docs()` and `getAllDocumentHashes()` use `getAll`.
 */
export class WarmKVStore extends BaseKVStore {
  private readonly axis: number[]

  constructor(readonly store: WarmStore) {
    super()
    this.axis = [1, ...new Array<number>(store.embedder.space.dimensions - 1).fill(0)]
  }

  private id(key: string, collection: string): string {
    return `kv:${encodeURIComponent(collection)}:${encodeURIComponent(key)}`
  }

  async put(key: string, val: StoredValue, collection: string = DEFAULT_COLLECTION): Promise<void> {
    await this.store.remember([
      { id: this.id(key, collection), text: JSON.stringify(val), metadata: { kv_collection: collection, kv_key: key }, vector: this.axis },
    ])
  }

  async get(key: string, collection: string = DEFAULT_COLLECTION): Promise<StoredValue> {
    const [rec] = await this.store.get([this.id(key, collection)])
    return rec ? (JSON.parse(rec.text) as StoredValue) : (null as unknown as StoredValue)
  }

  async getAll(collection: string = DEFAULT_COLLECTION): Promise<Record<string, StoredValue>> {
    const page = await this.store.list({ kv_collection: collection })
    if (page.truncated) {
      throw new WarmError(`collection "${collection}" exceeds maxFetch; raise maxFetch or split the collection`, 'limit_exceeded')
    }
    const out: Record<string, StoredValue> = {}
    for (const r of page.records) out[String(r.metadata.kv_key)] = JSON.parse(r.text) as StoredValue
    return out
  }

  async delete(key: string, collection: string = DEFAULT_COLLECTION): Promise<boolean> {
    return (await this.store.forget([this.id(key, collection)])) > 0
  }
}
