// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import type { WarmStore } from './store.js'
import type { Metadata } from './filters.js'

const SEQ_DIGITS = 10
const BATCH = 100
const queues = new WeakMap<WarmStore['memory'], Map<string, Promise<unknown>>>()

export interface LogEntry {
  seq: number
  text: string
  metadata: Metadata
}

/**
 * An append-only, gap-free ordered log inside an `WarmStore`.
 *
 * Entry `n` of log `name` is the record
 * `name#<n zero padded>`. The length is found by galloping and binary search over `get`
 * (about 2 log2(n) lookups), every entry is fetched by exact id, and nothing depends on the
 * per-query result cap. Operations sharing a native handle, collection and name are serialized.
 * Direct native writes and separate native handles must not mutate the same log concurrently.
 */
export class OrderedLog {
  private length: number | undefined

  constructor(
    private readonly store: WarmStore,
    readonly name: string,
  ) {
    if (!name) throw new RangeError('log name must be non-empty')
  }

  private id(seq: number): string {
    return `${this.name}#${String(seq).padStart(SEQ_DIGITS, '0')}`
  }

  private serialized<T>(op: () => Promise<T>): Promise<T> {
    let logs = queues.get(this.store.memory)
    if (!logs) {
      logs = new Map()
      queues.set(this.store.memory, logs)
    }
    const key = JSON.stringify([this.store.collection, this.name])
    const run = (logs.get(key) ?? Promise.resolve()).then(op)
    const settled = run.then(() => undefined, () => undefined)
    logs.set(key, settled)
    void settled.then(() => { if (logs.get(key) === settled) logs.delete(key) })
    return run
  }

  private async exists(seq: number): Promise<boolean> {
    return this.store.has(this.id(seq))
  }

  /** Number of entries. Re-verified against the store on every call, so other writers are noticed. */
  async size(): Promise<number> {
    return this.serialized(() => this.sizeNow())
  }

  private async sizeNow(): Promise<number> {
    let lo = this.length ?? 0
    // Invariant: entries 0..lo-1 exist (checked below), entry `hi` does not.
    if (lo > 0 && !(await this.exists(lo - 1))) lo = 0
    let step = 1
    let hi = lo
    while (await this.exists(hi)) {
      lo = hi + 1
      hi = lo + step - 1
      step *= 2
    }
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2)
      if (await this.exists(mid)) lo = mid + 1
      else hi = mid
    }
    this.length = lo
    return lo
  }

  async append(entries: { text: string; metadata?: Metadata }[]): Promise<void> {
    return this.serialized(() => this.appendNow(entries))
  }

  private async appendNow(entries: { text: string; metadata?: Metadata }[]): Promise<void> {
    if (!entries.length) return
    const start = await this.sizeNow()
    if (!Number.isSafeInteger(start + entries.length)) throw new RangeError('log sequence exceeds safe integer range')
    await this.store.remember(
      entries.map((e, i) => ({
        id: this.id(start + i),
        text: e.text,
        metadata: { ...(e.metadata ?? {}), warm_log: this.name, warm_seq: start + i },
      })),
    )
    this.length = start + entries.length
  }

  /** Entries `[from, to)` in order. */
  async read(from: number, to: number): Promise<LogEntry[]> {
    if (!Number.isSafeInteger(from) || !Number.isSafeInteger(to) || from < 0 || to < from) throw new RangeError('log range must be non-negative safe integers with from <= to')
    return this.serialized(() => this.readNow(from, to))
  }

  private async readNow(from: number, to: number): Promise<LogEntry[]> {
    const out: LogEntry[] = []
    for (let s = from; s < to; s += BATCH) {
      const seqs = Array.from({ length: Math.min(BATCH, to - s) }, (_, i) => s + i)
      const recs = await this.store.get(seqs.map((q) => this.id(q)))
      recs.forEach((r, i) => {
        if (r) out.push({ seq: seqs[i] as number, text: r.text, metadata: stripLog(r.metadata) })
      })
    }
    return out
  }

  /** The last `limit` entries (all when omitted) in chronological order. */
  async tail(limit?: number): Promise<LogEntry[]> {
    if (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 0)) throw new RangeError('log limit must be a non-negative safe integer')
    return this.serialized(async () => {
      const n = await this.sizeNow()
      const from = limit === undefined ? 0 : Math.max(0, n - limit)
      return this.readNow(from, n)
    })
  }

  /** Remove and return the last entry. */
  async pop(): Promise<LogEntry | undefined> {
    return this.serialized(() => this.popNow())
  }

  private async popNow(): Promise<LogEntry | undefined> {
    const n = await this.sizeNow()
    if (n === 0) return undefined
    const [entry] = await this.readNow(n - 1, n)
    await this.store.forget([this.id(n - 1)])
    this.length = n - 1
    return entry
  }

  async clear(): Promise<void> {
    return this.serialized(async () => {
      const n = await this.sizeNow()
      // Every individual durable delete must leave a contiguous prefix, including within a batch.
      for (let end = n; end > 0; end -= BATCH) {
        await this.store.forget(Array.from({ length: Math.min(BATCH, end) }, (_, i) => this.id(end - 1 - i)))
      }
      // Earlier versions could leave holes before the retained tail after an interrupted clear.
      await this.store.forgetWhere({ warm_log: this.name })
      this.length = 0
    })
  }
}

/** Metadata is capped at 64 KiB by the engine; payloads up to this size stay in metadata, larger ones move to the record text. */
const INLINE_PAYLOAD_BYTES = 32 * 1024

/**
 * Serialize `payload` for a log entry. Small payloads go in metadata (`searchText` stays the
 * readable record text). Larger ones, which would exceed the metadata cap, become the record
 * text itself, so they are still stored and returned byte-exactly, but lexical search then sees
 * the JSON.
 */
export function packPayload(searchText: string, payload: unknown, extra: Metadata = {}): { text: string; metadata: Metadata } {
  const json = JSON.stringify(payload)
  if (Buffer.byteLength(json, 'utf8') <= INLINE_PAYLOAD_BYTES) return { text: searchText, metadata: { ...extra, payload: json } }
  return { text: json, metadata: { ...extra, payload_in_text: true } }
}

export function unpackPayload<T>(entry: { text: string; metadata: Metadata }): T {
  return JSON.parse(entry.metadata.payload_in_text === true ? entry.text : (entry.metadata.payload as string)) as T
}

function stripLog(metadata: Metadata): Metadata {
  const { warm_log: _l, warm_seq: _s, ...rest } = metadata
  return rest
}
